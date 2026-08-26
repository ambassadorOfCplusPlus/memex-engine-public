// Compute attention over the four zones with ggml, and check the answer against exact
// attention over the same positions.
//
// Everything measured about the zoned cache until now was volume: 3.22 GB re-read per
// token at 32k becomes 0.85 GB, and the bandwidth ceiling moves from 6.2 to 23.7 tokens a
// second. None of that is worth anything if the answer changes, and nothing checked. The
// reconstruction error of the basis is not the answer either: what reaches the residual
// stream is a softmax-weighted sum, so a large error on a position the query ignores costs
// nothing while a small error on the position it attends to costs everything.
//
// So this builds the real thing, with one detail that decides correctness: the softmax is
// global. Attention normalises over every position at once, so scores from the exact zones
// and scores from the compressed tail must be concatenated before the exponent. A softmax
// per zone would let a window position and a tail position stop competing, which is a
// different function and a wrong one.
//
// The tail is never decompressed. For a key k with latent c = W k and an orthonormal basis
// so that k ~ W^T c:
//
//     q_h . k[g] = q_h . (W[:,g]^T c) = (W[:,g] q_h) . c
//
// so the query is projected once per head into the latent space and multiplied by the
// latents directly. Values come back the same way: accumulate sum_j p_j c_j in the latent
// space, then lift once with W[:,g]^T, at a cost independent of context length.
//
// Worth stating plainly what this saves and what it does not, because an earlier probe
// overstated it. With G kv-heads the exact score costs G*head_dim multiply-adds per
// position and the latent score costs G*rank - equal at rank 128 of d_kv 512. The
// arithmetic does not shrink at all; only the reading does, by d_kv/rank. Decode attention
// is bandwidth-bound so that is the win, but it is a bandwidth win, not a compute win.
//
// Shapes follow llama.cpp's own attention: keys, values and queries are three-dimensional
// with heads in the last dimension, and ggml_mul_mat broadcasts the four kv-heads across
// the sixteen query heads by itself. That is not a stylistic choice. Expressing grouped
// attention instead as one two-dimensional slice per head group - a strided view of K with
// four query columns - produced an answer wrong by nine orders of magnitude, with the
// first head of each group behaving differently from the other three. The supported
// formulation is the one the rest of the codebase uses.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include "memex/kv_zones.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 31) & 1;
    const int32_t e = int32_t((x >> 23) & 0xFF) - 127 + 15;
    const uint32_t man = x & 0x7FFFFF;
    if (e <= 0) return uint16_t(sign << 15);
    if (e >= 31) return uint16_t((sign << 15) | 0x7C00);
    return uint16_t((sign << 15) | (uint32_t(e) << 10) | (man >> 13));
}

float h2f(uint16_t h) {
    const uint32_t sign = uint32_t(h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1F;
    const uint32_t man = h & 0x3FF;
    uint32_t out;
    if (e == 0) {
        out = sign;
    } else if (e == 31) {
        out = sign | 0x7F800000 | (man << 13);
    } else {
        out = sign | ((e + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// Real keys or values for one layer. Only records whose unit width is d_kv are taken: the
// graph names several nodes Kcur, and a name-matching tracer wrote the keys four times,
// including a pre-norm copy whose head norm is 0.97 against 40.1 for the stored ones.
bool read_kv_rows(const std::string& path, int layer, int d, bool values,
                  std::vector<float>* out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    const int want = -layer - (values ? 40001 : 30001);
    int32_t hdr[3];
    while (fread(hdr, sizeof(int32_t), 3, f) == 3) {
        const size_t n = size_t(hdr[1]) * size_t(hdr[2]);
        if (hdr[0] != want || hdr[1] != d || n % size_t(d) != 0) {
            // 64-bit seek is mandatory: traces run to gigabytes and MSVC's long is 32-bit,
            // so this skip would silently wrap and land inside a record instead of failing.
#ifdef _WIN32
            if (_fseeki64(f, (__int64)(4 * n), SEEK_CUR) != 0) break;
#else
            if (fseeko(f, (off_t)(4 * n), SEEK_CUR) != 0) break;
#endif
            continue;
        }
        const size_t base = out->size();
        out->resize(base + n);
        if (fread(out->data() + base, sizeof(float), n, f) != n) {
            out->resize(base);
            break;
        }
    }
    fclose(f);
    return !out->empty();
}

struct Shape {
    int n_kv_heads = 4;
    // Read off the model, not guessed: qwen3moe.attention.head_count is 32, not 16. The
    // earlier figure was wrong, which makes the grouped-query ratio 8 rather than 4. It does
    // not invalidate the comparisons - both sides always used the same shape - but it does
    // change the arithmetic-to-bytes ratio, so the numbers were re-measured after the fix.
    int n_q_heads = 32;
    int head_dim = 128;
    int d_kv() const { return n_kv_heads * head_dim; }
    int ratio() const { return n_q_heads / n_kv_heads; }
};

// Zone lengths are rounded up before they reach a tensor, and the padding is masked out of
// the softmax. This is not defensive tidiness: the fork's F16 matmul gives wrong results
// when the reduction length is not a multiple of four. Measured on this program, with a
// full-rank basis so the answer had to be exact - context 2780 agreed with a plain host
// computation to 0.001%, while 2777, 2778 and 2779 were off by more than 100%, and the
// output tensor came back holding the attention probabilities rather than their weighted
// sum. Nothing asserts, so an engine that let a cache reach an odd length would return
// confident nonsense.
//
// llama.cpp has the same requirement and answers it the same way, padding the cache to a
// multiple of 32 and masking the remainder; 32 is used here for the same reason.
constexpr int kPad = 32;
int pad_up(int n) { return (n + kPad - 1) / kPad * kPad; }

// Host-side reshapes into the layout the graph wants. Keys per head as [head_dim, n_pad,
// n_kv_heads]; values transposed within each head as [n_pad, head_dim, n_kv_heads], so the
// weighted sum is a plain matrix-vector product instead of a per-token ggml_cont of the
// whole cache - which was measured at 245.8 ms against 23.3 ms at 4k. Padded entries are
// zero, which the mask makes irrelevant.
void split_heads(const std::vector<uint16_t>& rows, const std::vector<uint16_t>& vrows,
                 int n_pos, int n_pad, const Shape& s,
                 std::vector<uint16_t>* k3, std::vector<uint16_t>* v3) {
    const int d = s.d_kv();
    const int hd = s.head_dim;
    k3->assign(size_t(n_pad) * size_t(d), 0);
    v3->assign(size_t(n_pad) * size_t(d), 0);
    for (int g = 0; g < s.n_kv_heads; ++g) {
        for (int p = 0; p < n_pos; ++p) {
            for (int j = 0; j < hd; ++j) {
                const size_t src = size_t(p) * size_t(d) + size_t(g) * size_t(hd) + size_t(j);
                (*k3)[size_t(g) * size_t(n_pad) * size_t(hd) +
                      size_t(p) * size_t(hd) + size_t(j)] = rows[src];
                (*v3)[size_t(g) * size_t(hd) * size_t(n_pad) +
                      size_t(j) * size_t(n_pad) + size_t(p)] = vrows[src];
            }
        }
    }
}

// A random Hadamard rotation, per head. The point is quantisation, not attention: a
// rotation spreads outliers over all coordinates, and block quantisation is limited by the
// largest value in a block, so flattening the distribution buys accuracy for free. This is
// what QuaRot does to reach near-lossless four-bit inference, and what the fork's own
// -khad/-vhad flags hint at.
//
// It costs nothing in the scores because a dot product is invariant under a shared
// rotation: (Rq)·(Rk) = q·k exactly. Values need one inverse rotation per token, since the
// weighted sum comes out rotated: sum_j p_j (R v_j) = R (sum_j p_j v_j).
//
// Sylvester construction times a fixed sign diagonal, so it is orthonormal, deterministic
// and needs no storage or calibration - only head_dim being a power of two, which it is.
struct Rotation {
    int n = 0;
    std::vector<float> m;          // [n, n], row-major, already normalised

    void build(int dim) {
        n = dim;
        std::vector<int8_t> sign;
        sign.resize(size_t(n));
        uint32_t s = 0x9E3779B9u;
        for (int i = 0; i < n; ++i) {
            s = s * 1664525u + 1013904223u;
            sign[size_t(i)] = (s >> 16) & 1 ? 1 : -1;
        }
        const float scale = 1.0f / std::sqrt(float(n));
        m.assign(size_t(n) * size_t(n), 0.0f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // Sylvester entry is the parity of the shared bits of i and j.
                int bits = i & j, par = 0;
                while (bits) {
                    par ^= bits & 1;
                    bits >>= 1;
                }
                m[size_t(i) * size_t(n) + size_t(j)] =
                    (par ? -scale : scale) * float(sign[size_t(j)]);
            }
        }
    }

    // y = R x, or y = R^T x when transposed.
    void apply(const float* x, float* y, bool transposed) const {
        for (int i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < n; ++j) {
                acc += transposed ? m[size_t(j) * size_t(n) + size_t(i)] * x[j]
                                  : m[size_t(i) * size_t(n) + size_t(j)] * x[j];
            }
            y[i] = acc;
        }
    }
};

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* out = nullptr;     // [head_dim, 1, n_q_heads]
    ggml_tensor* q = nullptr;       // [head_dim, 1, n_q_heads] f32
    // Second query, rotated, for scoring against a tail whose keys were rotated before
    // quantisation. Null when the tail is not rotated.
    ggml_tensor* q_rot = nullptr;
    std::size_t read_bytes = 0;

    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        alloc = nullptr; buf = nullptr; ctx = nullptr;
    }
};

bool finish(Graph* g, ggml_backend_buffer_type_t buft) {
    ggml_set_output(g->out);
    ggml_build_forward_expand(g->gf, g->out);
    g->alloc = ggml_gallocr_new(buft);
    return ggml_gallocr_reserve(g->alloc, g->gf) &&
           ggml_gallocr_alloc_graph(g->alloc, g->gf);
}

ggml_context* make_ctx(size_t n_tensors) {
    ggml_init_params ip = {ggml_tensor_overhead() * n_tensors +
                           ggml_graph_overhead_custom(n_tensors, false),
                           nullptr, true};
    return ggml_init(ip);
}

// Exact attention over every position: the reference the zoned form has to reproduce.
bool build_exact(Graph* g, ggml_backend_buffer_type_t buft, const Shape& s,
                 const std::vector<uint16_t>& k3, const std::vector<uint16_t>& v3,
                 int n_pos) {
    const size_t n_tensors = 256;
    g->ctx = make_ctx(n_tensors);
    if (!g->ctx) return false;

    const int n_pad = pad_up(n_pos);
    ggml_tensor* K = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, s.head_dim, n_pad,
                                        s.n_kv_heads);
    ggml_tensor* V = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, n_pad, s.head_dim,
                                        s.n_kv_heads);
    ggml_tensor* M = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, n_pad, 1);
    g->q = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, s.head_dim, 1, s.n_q_heads);
    g->buf = ggml_backend_alloc_ctx_tensors_from_buft(g->ctx, buft);
    if (!g->buf) return false;
    ggml_backend_tensor_set(K, k3.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(V, v3.data(), 0, ggml_nbytes(V));
    std::vector<float> mask;
    mask.assign(size_t(n_pad), 0.0f);
    for (int i = n_pos; i < n_pad; ++i) mask[size_t(i)] = -INFINITY;
    ggml_backend_tensor_set(M, mask.data(), 0, ggml_nbytes(M));
    // Counted on the real positions, not the padded ones: the padding is an artefact of
    // the kernel and would flatter or penalise nothing consistently.
    g->read_bytes = size_t(n_pos) * size_t(s.d_kv()) * 2 * 2;

    g->gf = ggml_new_graph_custom(g->ctx, n_tensors, false);
    const float scale = 1.0f / std::sqrt(float(s.head_dim));
    ggml_tensor* kq = ggml_mul_mat(g->ctx, K, g->q);          // [n_pad, 1, n_q_heads]
    ggml_tensor* p = ggml_soft_max_ext(g->ctx, kq, M, scale, 0.0f);
    g->out = ggml_mul_mat(g->ctx, V, p);                      // [head_dim, 1, n_q_heads]
    return finish(g, buft);
}

// The zoned form: exact zones and compressed tail under one softmax.
bool build_zoned(Graph* g, ggml_backend_buffer_type_t buft, const Shape& s,
                 const std::vector<uint16_t>& k3, const std::vector<uint16_t>& v3,
                 int n_ex,
                 const std::vector<uint16_t>& ck, const std::vector<uint16_t>& cvt,
                 int n_tail, int rank_k, int rank_v,
                 const std::vector<uint16_t>& wk3, const std::vector<uint16_t>& wv3) {
    const size_t n_tensors = 256;
    g->ctx = make_ctx(n_tensors);
    if (!g->ctx) return false;

    const int ex_pad = pad_up(n_ex);
    const int tl_pad = n_tail > 0 ? pad_up(n_tail) : 0;
    ggml_tensor* K = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, s.head_dim, ex_pad,
                                        s.n_kv_heads);
    ggml_tensor* V = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, ex_pad, s.head_dim,
                                        s.n_kv_heads);
    ggml_tensor* M = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, ex_pad + tl_pad, 1);
    // The latents are shared by every head - one per position, not one per head - so they
    // stay two-dimensional and mul_mat broadcasts all sixteen query heads against a single
    // pass over them. That is where the reading is saved.
    ggml_tensor* Ck = nullptr;
    ggml_tensor* Cv = nullptr;
    ggml_tensor* Wk = nullptr;
    ggml_tensor* Wv = nullptr;
    if (n_tail > 0) {
        Ck = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, rank_k, tl_pad);
        Cv = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, tl_pad, rank_v);
        Wk = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, s.head_dim, rank_k, s.n_kv_heads);
        Wv = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, rank_v, s.head_dim, s.n_kv_heads);
    }
    g->q = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, s.head_dim, 1, s.n_q_heads);
    g->buf = ggml_backend_alloc_ctx_tensors_from_buft(g->ctx, buft);
    if (!g->buf) return false;
    ggml_backend_tensor_set(K, k3.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(V, v3.data(), 0, ggml_nbytes(V));
    g->read_bytes = size_t(n_ex) * size_t(s.d_kv()) * 2 * 2;
    if (n_tail > 0) {
        ggml_backend_tensor_set(Ck, ck.data(), 0, ggml_nbytes(Ck));
        ggml_backend_tensor_set(Cv, cvt.data(), 0, ggml_nbytes(Cv));
        ggml_backend_tensor_set(Wk, wk3.data(), 0, ggml_nbytes(Wk));
        ggml_backend_tensor_set(Wv, wv3.data(), 0, ggml_nbytes(Wv));
        // The bases are read once per token rather than once per position, so they belong
        // with the weights and not with the context traffic.
        g->read_bytes += size_t(n_tail) * size_t(rank_k + rank_v) * 2;
    }
    // Real positions in both zones pass, padding in either is refused.
    std::vector<float> mask;
    mask.assign(size_t(ex_pad + tl_pad), -INFINITY);
    for (int i = 0; i < n_ex; ++i) mask[size_t(i)] = 0.0f;
    for (int i = 0; i < n_tail; ++i) mask[size_t(ex_pad + i)] = 0.0f;
    ggml_backend_tensor_set(M, mask.data(), 0, ggml_nbytes(M));

    g->gf = ggml_new_graph_custom(g->ctx, n_tensors, false);
    const float scale = 1.0f / std::sqrt(float(s.head_dim));
    ggml_tensor* sc_ex = ggml_mul_mat(g->ctx, K, g->q);         // [ex_pad, 1, n_q_heads]
    ggml_tensor* all = sc_ex;
    if (n_tail > 0) {
        // (W[:, g] q_h), one [rank_k, head_dim] product per head, paid once per token
        // whatever the context length.
        ggml_tensor* lat_q = ggml_mul_mat(g->ctx, Wk, g->q);    // [rank_k, 1, n_q_heads]
        ggml_tensor* sc_tl = ggml_mul_mat(g->ctx, Ck, lat_q);   // [tl_pad, 1, n_q_heads]
        all = ggml_concat(g->ctx, sc_ex, sc_tl, 0);
    }
    ggml_tensor* p = ggml_soft_max_ext(g->ctx, all, M, scale, 0.0f);

    // The slices are copied because a strided view cannot be the reduction operand, and
    // the copy is what the padded layout costs: one pass over the score array per zone.
    ggml_tensor* p_ex = ggml_cont(g->ctx,
        ggml_view_3d(g->ctx, p, ex_pad, 1, s.n_q_heads, p->nb[1], p->nb[2], 0));
    ggml_tensor* out = ggml_mul_mat(g->ctx, V, p_ex);           // [head_dim, 1, n_q_heads]
    if (n_tail > 0) {
        ggml_tensor* p_tl = ggml_cont(g->ctx,
            ggml_view_3d(g->ctx, p, tl_pad, 1, s.n_q_heads, p->nb[1], p->nb[2],
                         size_t(ex_pad) * sizeof(float)));
        // Accumulate in the latent space and lift once. This is where the structure pays:
        // the sum runs over the tail at width rank_v, and the lift back to head_dim does
        // not depend on how long the tail is.
        ggml_tensor* lat = ggml_mul_mat(g->ctx, Cv, p_tl);      // [rank_v, 1, n_q_heads]
        ggml_tensor* up = ggml_mul_mat(g->ctx, Wv, lat);        // [head_dim, 1, n_q_heads]
        out = ggml_add(g->ctx, out, up);
    }
    g->out = out;
    return finish(g, buft);
}

// One representation for the whole context, so that two ways of shrinking it can be
// compared per byte instead of in principle. Low-rank latents and low-precision full-rank
// values cost almost the same per position here - 512 bytes against 576 - which makes the
// comparison a fair one and the answer worth having.
bool build_uniform(Graph* g, ggml_backend_buffer_type_t buft, const Shape& s,
                   const std::vector<float>& k3f, const std::vector<float>& v3f,
                   int n_pos, ggml_type type) {
    const size_t n_tensors = 256;
    g->ctx = make_ctx(n_tensors);
    if (!g->ctx) return false;
    const int n_pad = pad_up(n_pos);

    ggml_tensor* K = ggml_new_tensor_3d(g->ctx, type, s.head_dim, n_pad, s.n_kv_heads);
    ggml_tensor* V = ggml_new_tensor_3d(g->ctx, type, n_pad, s.head_dim, s.n_kv_heads);
    ggml_tensor* M = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, n_pad, 1);
    g->q = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, s.head_dim, 1, s.n_q_heads);
    g->buf = ggml_backend_alloc_ctx_tensors_from_buft(g->ctx, buft);
    if (!g->buf) return false;

    std::vector<uint8_t> kb, vb;
    kb.resize(ggml_nbytes(K));
    vb.resize(ggml_nbytes(V));
    ggml_quantize_init(type);
    // Keys quantise along the head dimension and values along positions, which is how a
    // quantised KV cache is laid out: each block of 32 shares one scale either way.
    ggml_quantize_chunk(type, k3f.data(), kb.data(), 0,
                        int64_t(n_pad) * s.n_kv_heads, s.head_dim, nullptr, nullptr);
    ggml_quantize_chunk(type, v3f.data(), vb.data(), 0,
                        int64_t(s.head_dim) * s.n_kv_heads, n_pad, nullptr, nullptr);
    ggml_backend_tensor_set(K, kb.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(V, vb.data(), 0, ggml_nbytes(V));
    std::vector<float> mask;
    mask.assign(size_t(n_pad), 0.0f);
    for (int i = n_pos; i < n_pad; ++i) mask[size_t(i)] = -INFINITY;
    ggml_backend_tensor_set(M, mask.data(), 0, ggml_nbytes(M));
    g->read_bytes = size_t(n_pos) * ggml_row_size(type, s.d_kv()) * 2;

    g->gf = ggml_new_graph_custom(g->ctx, n_tensors, false);
    const float scale = 1.0f / std::sqrt(float(s.head_dim));
    ggml_tensor* kq = ggml_mul_mat(g->ctx, K, g->q);
    ggml_tensor* p = ggml_soft_max_ext(g->ctx, kq, M, scale, 0.0f);
    g->out = ggml_mul_mat(g->ctx, V, p);
    return finish(g, buft);
}

// The whole context as rank-r latents, attended to inside the compressed space. Same shape
// of computation as the tail of the zoned cache, applied to every position, so that its
// quality can be read off without the exact zones masking it.
bool build_latent(Graph* g, ggml_backend_buffer_type_t buft, const Shape& s,
                  const std::vector<uint16_t>& ck, const std::vector<uint16_t>& cvt,
                  int n_pos, int rank_k, int rank_v,
                  const std::vector<uint16_t>& wk3, const std::vector<uint16_t>& wv3) {
    const size_t n_tensors = 256;
    g->ctx = make_ctx(n_tensors);
    if (!g->ctx) return false;
    const int n_pad = pad_up(n_pos);

    ggml_tensor* Ck = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, rank_k, n_pad);
    ggml_tensor* Cv = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, n_pad, rank_v);
    ggml_tensor* Wk = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, s.head_dim, rank_k,
                                         s.n_kv_heads);
    ggml_tensor* Wv = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, rank_v, s.head_dim,
                                         s.n_kv_heads);
    ggml_tensor* M = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, n_pad, 1);
    g->q = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, s.head_dim, 1, s.n_q_heads);
    g->buf = ggml_backend_alloc_ctx_tensors_from_buft(g->ctx, buft);
    if (!g->buf) return false;
    ggml_backend_tensor_set(Ck, ck.data(), 0, ggml_nbytes(Ck));
    ggml_backend_tensor_set(Cv, cvt.data(), 0, ggml_nbytes(Cv));
    ggml_backend_tensor_set(Wk, wk3.data(), 0, ggml_nbytes(Wk));
    ggml_backend_tensor_set(Wv, wv3.data(), 0, ggml_nbytes(Wv));
    std::vector<float> mask;
    mask.assign(size_t(n_pad), 0.0f);
    for (int i = n_pos; i < n_pad; ++i) mask[size_t(i)] = -INFINITY;
    ggml_backend_tensor_set(M, mask.data(), 0, ggml_nbytes(M));
    g->read_bytes = size_t(n_pos) * size_t(rank_k + rank_v) * 2;

    g->gf = ggml_new_graph_custom(g->ctx, n_tensors, false);
    const float scale = 1.0f / std::sqrt(float(s.head_dim));
    ggml_tensor* lat_q = ggml_mul_mat(g->ctx, Wk, g->q);
    ggml_tensor* sc = ggml_mul_mat(g->ctx, Ck, lat_q);
    ggml_tensor* p = ggml_soft_max_ext(g->ctx, sc, M, scale, 0.0f);
    ggml_tensor* lat = ggml_mul_mat(g->ctx, Cv, p);
    g->out = ggml_mul_mat(g->ctx, Wv, lat);
    return finish(g, buft);
}

// The zoned cache as it should ship: exact zones in fp16, tail in full rank at eight bits,
// one global softmax over both. No basis and no lifting - the tail values are whole, so the
// weighted sum over them lands directly in head space.
//
// This is also the check on the Q8_0 blocks the zone code writes by hand. If a scale or a
// lane were laid out differently from ggml's own format the answer would be visibly wrong,
// so agreement with exact attention is the verification.
bool build_zoned_q8(Graph* g, ggml_backend_buffer_type_t buft, const Shape& s,
                    const std::vector<uint16_t>& k3, const std::vector<uint16_t>& v3,
                    int n_ex, const std::vector<uint8_t>& tk, const std::vector<uint8_t>& tv,
                    int n_tail, bool rotated) {
    const size_t n_tensors = 256;
    g->ctx = make_ctx(n_tensors);
    if (!g->ctx) return false;
    const int ex_pad = pad_up(n_ex);
    const int tl_pad = n_tail > 0 ? pad_up(n_tail) : 0;

    ggml_tensor* K = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, s.head_dim, ex_pad,
                                        s.n_kv_heads);
    ggml_tensor* V = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F16, ex_pad, s.head_dim,
                                        s.n_kv_heads);
    ggml_tensor* Tk = nullptr;
    ggml_tensor* Tv = nullptr;
    if (n_tail > 0) {
        Tk = ggml_new_tensor_3d(g->ctx, GGML_TYPE_Q8_0, s.head_dim, tl_pad, s.n_kv_heads);
        Tv = ggml_new_tensor_3d(g->ctx, GGML_TYPE_Q8_0, tl_pad, s.head_dim, s.n_kv_heads);
    }
    ggml_tensor* M = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, ex_pad + tl_pad, 1);
    g->q = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, s.head_dim, 1, s.n_q_heads);
    if (rotated && n_tail > 0) {
        g->q_rot = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, s.head_dim, 1, s.n_q_heads);
    }
    g->buf = ggml_backend_alloc_ctx_tensors_from_buft(g->ctx, buft);
    if (!g->buf) return false;
    ggml_backend_tensor_set(K, k3.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(V, v3.data(), 0, ggml_nbytes(V));
    g->read_bytes = size_t(n_ex) * size_t(s.d_kv()) * 2 * 2;
    if (n_tail > 0) {
        if (tk.size() < size_t(ggml_nbytes(Tk)) || tv.size() < size_t(ggml_nbytes(Tv))) {
            return false;
        }
        ggml_backend_tensor_set(Tk, tk.data(), 0, ggml_nbytes(Tk));
        ggml_backend_tensor_set(Tv, tv.data(), 0, ggml_nbytes(Tv));
        g->read_bytes += size_t(n_tail) * size_t(ggml_row_size(GGML_TYPE_Q8_0,
                                                               s.d_kv())) * 2;
    }
    std::vector<float> mask;
    mask.assign(size_t(ex_pad + tl_pad), -INFINITY);
    for (int i = 0; i < n_ex; ++i) mask[size_t(i)] = 0.0f;
    for (int i = 0; i < n_tail; ++i) mask[size_t(ex_pad + i)] = 0.0f;
    ggml_backend_tensor_set(M, mask.data(), 0, ggml_nbytes(M));

    g->gf = ggml_new_graph_custom(g->ctx, n_tensors, false);
    const float scale = 1.0f / std::sqrt(float(s.head_dim));
    ggml_tensor* sc_ex = ggml_mul_mat(g->ctx, K, g->q);
    ggml_tensor* all = sc_ex;
    if (n_tail > 0) {
        // The tail keys were rotated before quantisation, so they must be scored against
        // the rotated query. Values were not rotated - measured as worthless there - so
        // nothing has to be rotated back afterwards.
        all = ggml_concat(g->ctx, sc_ex,
                          ggml_mul_mat(g->ctx, Tk, g->q_rot ? g->q_rot : g->q), 0);
    }
    ggml_tensor* p = ggml_soft_max_ext(g->ctx, all, M, scale, 0.0f);
    ggml_tensor* p_ex = ggml_cont(g->ctx,
        ggml_view_3d(g->ctx, p, ex_pad, 1, s.n_q_heads, p->nb[1], p->nb[2], 0));
    ggml_tensor* out = ggml_mul_mat(g->ctx, V, p_ex);
    if (n_tail > 0) {
        ggml_tensor* p_tl = ggml_cont(g->ctx,
            ggml_view_3d(g->ctx, p, tl_pad, 1, s.n_q_heads, p->nb[1], p->nb[2],
                         size_t(ex_pad) * sizeof(float)));
        out = ggml_add(g->ctx, out, ggml_mul_mat(g->ctx, Tv, p_tl));
    }
    g->out = out;
    return finish(g, buft);
}

// The same zoned formula computed plainly on the host, as a second opinion. When the graph
// disagreed with exact attention by nine orders of magnitude even with a full-rank basis
// that reconstructs exactly, rereading the graph could not settle whether the fault was in
// the algebra or in the ggml expression of it. This decides that.
void host_zoned(const Shape& s, const uint16_t* ex_k, const uint16_t* ex_v, int n_ex,
                const uint16_t* ck, const uint16_t* cv, int n_tail,
                int rank_k, int rank_v, const uint16_t* wk, const uint16_t* wv,
                const std::vector<float>& q, std::vector<float>* out) {
    const int d = s.d_kv();
    const int hd = s.head_dim;
    const float sc_f = 1.0f / std::sqrt(float(hd));
    out->assign(size_t(hd) * size_t(s.n_q_heads), 0.0f);
    std::vector<float> latq, sc, latv;
    latq.resize(size_t(rank_k));
    latv.resize(size_t(rank_v));
    sc.resize(size_t(n_ex + n_tail));
    for (int h = 0; h < s.n_q_heads; ++h) {
        const size_t hoff = size_t(h / s.ratio()) * size_t(hd);
        const float* qh = q.data() + size_t(h) * size_t(hd);
        for (int i = 0; i < rank_k; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < hd; ++j) {
                acc += h2f(wk[size_t(i) * size_t(d) + hoff + size_t(j)]) * qh[j];
            }
            latq[size_t(i)] = acc;
        }
        for (int p = 0; p < n_ex; ++p) {
            float acc = 0.0f;
            for (int j = 0; j < hd; ++j) {
                acc += h2f(ex_k[size_t(p) * size_t(d) + hoff + size_t(j)]) * qh[j];
            }
            sc[size_t(p)] = acc * sc_f;
        }
        for (int t = 0; t < n_tail; ++t) {
            float acc = 0.0f;
            for (int i = 0; i < rank_k; ++i) {
                acc += h2f(ck[size_t(t) * size_t(rank_k) + size_t(i)]) * latq[size_t(i)];
            }
            sc[size_t(n_ex + t)] = acc * sc_f;
        }
        float mx = -1e30f;
        for (float x : sc) mx = std::max(mx, x);
        double sum = 0.0;
        for (float& x : sc) {
            x = std::exp(x - mx);
            sum += x;
        }
        for (float& x : sc) x = float(x / sum);
        float* oh = out->data() + size_t(h) * size_t(hd);
        for (int p = 0; p < n_ex; ++p) {
            const float w = sc[size_t(p)];
            for (int j = 0; j < hd; ++j) {
                oh[j] += w * h2f(ex_v[size_t(p) * size_t(d) + hoff + size_t(j)]);
            }
        }
        std::fill(latv.begin(), latv.end(), 0.0f);
        for (int t = 0; t < n_tail; ++t) {
            const float w = sc[size_t(n_ex + t)];
            for (int i = 0; i < rank_v; ++i) {
                latv[size_t(i)] += w * h2f(cv[size_t(t) * size_t(rank_v) + size_t(i)]);
            }
        }
        for (int i = 0; i < rank_v; ++i) {
            const float li = latv[size_t(i)];
            for (int j = 0; j < hd; ++j) {
                oh[j] += li * h2f(wv[size_t(i) * size_t(d) + hoff + size_t(j)]);
            }
        }
    }
}

double time_it(ggml_backend_t be, Graph* g, int iters) {
    ggml_backend_graph_compute(be, g->gf);
    ggml_backend_synchronize(be);
    // Median of single runs rather than a mean over a loop: one stray process turned an
    // identical measurement from 1.6 ms into 102 ms earlier in this project.
    std::vector<double> t;
    t.reserve(size_t(iters));
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        ggml_backend_graph_compute(be, g->gf);
        ggml_backend_synchronize(be);
        t.push_back(ms_since(t0));
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Relative L2 difference between two output vectors, which is what the residual stream
// would see, plus the largest single-component difference so that a small norm error
// hiding one badly wrong coordinate is still visible.
//
// A zero reference is reported as a negative relative error rather than as zero. The
// earlier version returned 0.0 in that case, and when the reference graph turned out to
// produce nothing at all, every comparison against it read as a perfect match.
void compare(const std::vector<float>& a, const std::vector<float>& b,
             double* rel, double* worst) {
    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double dx = double(a[i]) - double(b[i]);
        num += dx * dx;
        den += double(b[i]) * double(b[i]);
        mx = std::max(mx, std::abs(dx));
    }
    *rel = den > 0.0 ? std::sqrt(num / den) : -1.0;
    *worst = mx;
}

double norm_of(const std::vector<float>& x) {
    double a = 0.0;
    for (float v : x) a += double(v) * double(v);
    return std::sqrt(a);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Shape s;
    memex::KvConfig cfg;
    int n_ctx = 2777;
    int iters = 9;
    int n_queries = 16;
    int threads = 4;
    std::string basis = "D:/MemeX/blob/kv_mix_r128.bin";
    std::string trace = "D:/MemeX/results/tr_kv_ru.bin";
    int layer = 20;
    bool on_gpu = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) cfg.window = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--notebook") && i + 1 < argc) cfg.notebook_cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rank-k") && i + 1 < argc) cfg.rank_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rank-v") && i + 1 < argc) cfg.rank_v = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--basis") && i + 1 < argc) basis = argv[++i];
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace = argv[++i];
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--queries") && i + 1 < argc) n_queries = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        // Only to isolate ggml's grouped-query broadcast: with as many query heads as
        // kv-heads there is no broadcast left to get wrong.
        else if (!strcmp(argv[i], "--qheads") && i + 1 < argc) s.n_q_heads = atoi(argv[++i]);
        // What the tail holds. Eight-bit full rank is what the measurements below argue
        // for; the low-rank form is kept because it is what produced that argument.
        else if (!strcmp(argv[i], "--tail") && i + 1 < argc) {
            cfg.tail_form = strcmp(argv[++i], "q8") == 0 ? memex::TailForm::Q8_0
                                                         : memex::TailForm::LowRank;
        }
        else if (!strcmp(argv[i], "--gpu")) on_gpu = true;
    }
    const int d = s.d_kv();

    std::vector<float> rk, rv;
    if (!read_kv_rows(trace, layer, d, false, &rk) ||
        !read_kv_rows(trace, layer, d, true, &rv)) {
        printf("нет K или V для слоя %d в %s\n", layer, trace.c_str());
        return 1;
    }
    const int n_real = int(std::min(rk.size(), rv.size()) / size_t(d));
    printf("слой %d: настоящих позиций %d, контекст %d\n", layer, n_real, n_ctx);

    const bool tail_q8 = cfg.tail_form == memex::TailForm::Q8_0;
    if (tail_q8 && cfg.tail_cap == 0) {
        cfg.tail_cap = n_ctx;      // nothing longer than the context can reach the tail
    }
    memex::KvZones kv(cfg);
    kv.set_project_threads(threads);
    std::string err;
    // The quantised tail keeps full rank, so there is no basis to load and none to get
    // wrong. Requiring the file anyway would make the honest configuration the awkward one.
    if (!tail_q8 && !kv.load_basis(basis.c_str(), &err)) {
        printf("базис не загружен: %s\n", err.c_str());
        return 1;
    }
    printf("хвост: %s\n", tail_q8 ? "Q8_0, полный ранг"
                                  : "низкий ранг (опровергнут, см. ARCHITECTURE 8.23)");

    // Fill the zones with the real vectors, and keep the same rows for the reference so
    // both graphs answer over identical positions.
    std::vector<uint16_t> full_k, full_v;
    full_k.resize(size_t(n_ctx) * size_t(d));
    full_v.resize(size_t(n_ctx) * size_t(d));
    std::vector<uint16_t> row_k, row_v;
    row_k.resize(size_t(d));
    row_v.resize(size_t(d));
    for (int t = 0; t < n_ctx; ++t) {
        const size_t off = size_t(t % n_real) * size_t(d);
        for (int j = 0; j < d; ++j) {
            row_k[size_t(j)] = f2h(rk[off + size_t(j)]);
            row_v[size_t(j)] = f2h(rv[off + size_t(j)]);
            full_k[size_t(t) * size_t(d) + size_t(j)] = row_k[size_t(j)];
            full_v[size_t(t) * size_t(d) + size_t(j)] = row_v[size_t(j)];
        }
        // Importance stands in for what the model would supply: heavy-tailed, so the
        // notebook has something to choose between. Uniform importance would make it
        // meaningless.
        const float imp = std::exp(std::sin(float(t) * 12.9898f) * 2.0f);
        kv.append(row_k.data(), row_v.data(), imp);
    }
    kv.flush();

    std::vector<uint16_t> ex_k, ex_v;
    const int n_ex = kv.copy_exact(&ex_k, &ex_v);
    const auto tail = kv.span(memex::Zone::Tail);
    const int n_tail = tail.n_pos;
    printf("зоны: точных %d (стоки %d, блокнот %d, окно %d), хвост %d — %.1f%% сжато\n",
           n_ex, kv.span(memex::Zone::Sink).n_pos, kv.span(memex::Zone::Notebook).n_pos,
           kv.span(memex::Zone::Window).n_pos, n_tail,
           100.0 * n_tail / std::max(1, n_ex + n_tail));
    if (n_ex + n_tail != n_ctx) {
        printf("ВНИМАНИЕ: зоны держат %d позиций из %d — сравнение было бы неверным\n",
               n_ex + n_tail, n_ctx);
        return 1;
    }

    std::vector<uint16_t> fk3, fv3, ek3, ev3;
    split_heads(full_k, full_v, n_ctx, pad_up(n_ctx), s, &fk3, &fv3);
    split_heads(ex_k, ex_v, n_ex, pad_up(n_ex), s, &ek3, &ev3);

    // Tail latents: K needs no reshape, since a row is already one position's latent and
    // the reduction there runs over the rank, which is already a multiple of the padding.
    // V and the value basis do, because lifting the latent sums over the rank while the
    // weighted sum runs over positions - so it is V whose length has to be rounded up.
    const int tl_pad = n_tail > 0 ? pad_up(n_tail) : 0;
    std::vector<uint16_t> ck, cvt, wk3, wv3;
    const uint16_t* tv = nullptr;
    if (!tail_q8) {
        // Only meaningful for the low-rank tail: with the quantised one these spans hold
        // blocks, not latents, and reading them as latents runs off the end of the buffer.
        ck.assign(size_t(tl_pad) * size_t(cfg.rank_k), 0);
        std::copy((const uint16_t*)tail.k,
                  (const uint16_t*)tail.k + size_t(n_tail) * size_t(cfg.rank_k),
                  ck.begin());
        tv = (const uint16_t*)tail.v;
        cvt.assign(size_t(tl_pad) * size_t(cfg.rank_v), 0);
        for (int p = 0; p < n_tail; ++p) {
            for (int i = 0; i < cfg.rank_v; ++i) {
                cvt[size_t(i) * size_t(tl_pad) + size_t(p)] =
                    tv[size_t(p) * size_t(cfg.rank_v) + size_t(i)];
            }
        }
    }
    const uint16_t* wk = tail_q8 ? nullptr : kv.projection_up_k();
    const uint16_t* wv = tail_q8 ? nullptr : kv.projection_up_v();
    wk3.resize(size_t(cfg.rank_k) * size_t(d));
    wv3.resize(size_t(cfg.rank_v) * size_t(d));
    for (int g = 0; g < s.n_kv_heads && !tail_q8; ++g) {
        for (int j = 0; j < s.head_dim; ++j) {
            const size_t col = size_t(g) * size_t(s.head_dim) + size_t(j);
            for (int i = 0; i < cfg.rank_k; ++i) {
                wk3[size_t(g) * size_t(cfg.rank_k) * size_t(s.head_dim) +
                    size_t(i) * size_t(s.head_dim) + size_t(j)] =
                    wk[size_t(i) * size_t(d) + col];
            }
            for (int i = 0; i < cfg.rank_v; ++i) {
                wv3[size_t(g) * size_t(cfg.rank_v) * size_t(s.head_dim) +
                    size_t(j) * size_t(cfg.rank_v) + size_t(i)] =
                    wv[size_t(i) * size_t(d) + col];
            }
        }
    }

    ggml_backend_t be = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    if (on_gpu) {
#ifdef GGML_USE_VULKAN
        be = ggml_backend_vk_init(0);
        buft = ggml_backend_vk_buffer_type(0);
#endif
        if (!be) {
            printf("Vulkan недоступен, считаю на CPU\n");
            on_gpu = false;
        }
    }
    if (!be) {
        be = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(be, threads);
        buft = ggml_backend_cpu_buffer_type();
    }

    Graph ge, gz;
    if (!build_exact(&ge, buft, s, fk3, fv3, n_ctx)) {
        printf("эталонный граф не собрался\n");
        return 1;
    }
    // Compact the tail out of its preallocated buffers into tensors sized to what is
    // actually there. Keys are whole rows per position, so a prefix of rows; values are one
    // row per dimension, so a prefix of blocks within each row. Both are contiguous runs,
    // which is the point of the layout.
    std::vector<uint8_t> tkq, tvq;
    if (tail_q8 && n_tail > 0) {
        const int cap = kv.config().tail_cap;
        const int tl_pad = pad_up(n_tail);
        const int k_row = memex::KvZones::q8_row_bytes(s.head_dim);
        const int v_src = memex::KvZones::q8_row_bytes(cap);
        const int v_dst = memex::KvZones::q8_row_bytes(tl_pad);
        tkq.assign(size_t(s.n_kv_heads) * size_t(tl_pad) * size_t(k_row), 0);
        tvq.assign(size_t(s.n_kv_heads) * size_t(s.head_dim) * size_t(v_dst), 0);
        for (int g = 0; g < s.n_kv_heads; ++g) {
            std::memcpy(tkq.data() + size_t(g) * size_t(tl_pad) * size_t(k_row),
                        kv.tail_k_q() + size_t(g) * size_t(cap) * size_t(k_row),
                        size_t(tl_pad) * size_t(k_row));
            for (int j = 0; j < s.head_dim; ++j) {
                const size_t row = size_t(g) * size_t(s.head_dim) + size_t(j);
                std::memcpy(tvq.data() + row * size_t(v_dst),
                            kv.tail_v_q() + row * size_t(v_src), size_t(v_dst));
            }
        }
    }

    const bool built = tail_q8
        ? build_zoned_q8(&gz, buft, s, ek3, ev3, n_ex, tkq, tvq, n_tail,
                         kv.key_rotation() != nullptr)
        : build_zoned(&gz, buft, s, ek3, ev3, n_ex, ck, cvt, n_tail,
                      cfg.rank_k, cfg.rank_v, wk3, wv3);
    if (!built) {
        printf("зонный граф не собрался\n");
        return 1;
    }

    // Queries taken from the recorded keys. They are not the model's own queries - the
    // tracer does not keep those - but they are directions the key population actually
    // contains, which is what makes attention peaked rather than flat. A gaussian query
    // would spread the weight evenly and flatter the compressed tail.
    const int n_out = s.head_dim * s.n_q_heads;
    std::vector<float> q, oe, oz, oh;
    q.resize(size_t(n_out));
    oe.resize(size_t(n_out));
    oz.resize(size_t(n_out));

    auto set_query = [&](int qi, double qs) {
        for (int h = 0; h < s.n_q_heads; ++h) {
            const size_t src = size_t((qi * s.n_q_heads + h) % n_real) * size_t(d) +
                               size_t(h / s.ratio()) * size_t(s.head_dim);
            for (int j = 0; j < s.head_dim; ++j) {
                q[size_t(h) * size_t(s.head_dim) + size_t(j)] =
                    float(qs) * rk[src + size_t(j)];
            }
        }
    };
    std::vector<float> q_rotated;
    q_rotated.resize(size_t(n_out));
    auto run_both = [&]() {
        ggml_backend_tensor_set(ge.q, q.data(), 0, ggml_nbytes(ge.q));
        ggml_backend_tensor_set(gz.q, q.data(), 0, ggml_nbytes(gz.q));
        if (gz.q_rot) {
            const float* r = kv.key_rotation();
            for (int h = 0; h < s.n_q_heads; ++h) {
                const float* qh = q.data() + size_t(h) * size_t(s.head_dim);
                float* out = q_rotated.data() + size_t(h) * size_t(s.head_dim);
                for (int i = 0; i < s.head_dim; ++i) {
                    const float* row = r + size_t(i) * size_t(s.head_dim);
                    float acc = 0.0f;
                    for (int j = 0; j < s.head_dim; ++j) {
                        acc += row[j] * qh[j];
                    }
                    out[i] = acc;
                }
            }
            ggml_backend_tensor_set(gz.q_rot, q_rotated.data(), 0,
                                    ggml_nbytes(gz.q_rot));
        }
        ggml_backend_graph_compute(be, ge.gf);
        ggml_backend_synchronize(be);
        ggml_backend_tensor_get(ge.out, oe.data(), 0, ggml_nbytes(ge.out));
        ggml_backend_graph_compute(be, gz.gf);
        ggml_backend_synchronize(be);
        ggml_backend_tensor_get(gz.out, oz.data(), 0, ggml_nbytes(gz.out));
    };

    // Three routes for one query: exact attention in ggml, the zoned formula on the host,
    // and the zoned graph. Two of the three agreeing localises a fault without guesswork,
    // and the norms are printed because a comparison against a zero reference reads as a
    // perfect match - which is exactly how the first version of this hid a broken graph.
    set_query(0, 1.0);
    run_both();
    // The host reference implements the low-rank formula, so it applies only to that form.
    // The quantised tail is checked instead by the comparison against exact attention:
    // hand-written Q8_0 blocks disagreeing with ggml's own layout would show up there at
    // once, and a few per cent is not a number a malformed scale produces.
    if (!tail_q8) {
    host_zoned(s, ex_k.data(), ex_v.data(), n_ex, ck.data(), tv, n_tail,
               cfg.rank_k, cfg.rank_v, wk, wv, q, &oh);
    {
        // Fourth route: exact attention on the host, which is the same function the
        // reference graph computes with no compression anywhere. With a full-rank basis
        // all four must agree, so whichever pair agrees identifies the broken one.
        std::vector<float> ohx;
        host_zoned(s, full_k.data(), full_v.data(), n_ctx, nullptr, nullptr, 0,
                   cfg.rank_k, cfg.rank_v, wk, wv, q, &ohx);
        double r1 = 0.0, r2 = 0.0, r3 = 0.0, w = 0.0;
        compare(oh, ohx, &r1, &w);
        compare(oz, ohx, &r2, &w);
        compare(oe, ohx, &r3, &w);
        printf("\nчетыре пути: |хост точно| %.4f, |граф точно| %.4f, "
               "|хост зонами| %.4f, |граф зонами| %.4f\n",
               norm_of(ohx), norm_of(oe), norm_of(oh), norm_of(oz));
        printf("  граф точно  против хоста точно: %.3f%%\n", 100.0 * r3);
        printf("  хост зонами против хоста точно: %.3f%%\n", 100.0 * r1);
        printf("  граф зонами против хоста точно: %.3f%%\n", 100.0 * r2);
        printf("  голова 0, первые 6 компонент:\n");
        printf("    хост точно:");
        for (int j = 0; j < 6; ++j) printf(" %+8.5f", ohx[size_t(j)]);
        printf("\n    граф точно:");
        for (int j = 0; j < 6; ++j) printf(" %+8.5f", oe[size_t(j)]);
        printf("\n");
        // Where the graph's value came from: if it matches the mean of the value rows the
        // scores never reached the softmax; if it matches one v row the distribution
        // collapsed; if it matches neither, the shapes are wrong.
        std::vector<float> mean_v;
        mean_v.assign(size_t(s.head_dim), 0.0f);
        for (int p = 0; p < n_ctx; ++p) {
            for (int j = 0; j < s.head_dim; ++j) {
                mean_v[size_t(j)] += h2f(full_v[size_t(p) * size_t(d) + size_t(j)]) / n_ctx;
            }
        }
        printf("    среднее V :");
        for (int j = 0; j < 6; ++j) printf(" %+8.5f", mean_v[size_t(j)]);
        printf("\n");
    }
    }

    // What the tail should hold. The zoned structure decides which positions stay exact;
    // this decides what happens to the rest, and the candidates cost nearly the same per
    // position - 512 bytes as rank-128 latents against 576 as four-bit full rank - so the
    // comparison is per byte rather than in principle. Needs the basis, so it runs in the
    // low-rank configuration, which is where the question came from.
    if (!tail_q8) {
        std::vector<float> fk3f, fv3f;
        fk3f.resize(fk3.size());
        fv3f.resize(fv3.size());
        for (size_t i = 0; i < fk3.size(); ++i) fk3f[i] = h2f(fk3[i]);
        for (size_t i = 0; i < fv3.size(); ++i) fv3f[i] = h2f(fv3[i]);

        // Every position as latents, projected here so the comparison covers the whole
        // context rather than only whatever the policy happened to compress.
        const int n_pad = pad_up(n_ctx);
        std::vector<uint16_t> ack, acvt;
        ack.assign(size_t(n_pad) * size_t(cfg.rank_k), 0);
        acvt.assign(size_t(n_pad) * size_t(cfg.rank_v), 0);
        for (int p = 0; p < n_ctx; ++p) {
            for (int i = 0; i < cfg.rank_k; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < d; ++j) {
                    acc += h2f(wk[size_t(i) * size_t(d) + size_t(j)]) *
                           h2f(full_k[size_t(p) * size_t(d) + size_t(j)]);
                }
                ack[size_t(p) * size_t(cfg.rank_k) + size_t(i)] = f2h(acc);
            }
            for (int i = 0; i < cfg.rank_v; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < d; ++j) {
                    acc += h2f(wv[size_t(i) * size_t(d) + size_t(j)]) *
                           h2f(full_v[size_t(p) * size_t(d) + size_t(j)]);
                }
                acvt[size_t(i) * size_t(n_pad) + size_t(p)] = f2h(acc);
            }
        }

        // Rotated copies of the same vectors, per head. A rotation cannot change what
        // attention computes; it changes only how well a block quantiser can represent the
        // numbers, which is exactly the quantity in question here.
        Rotation rot;
        rot.build(s.head_dim);
        std::vector<float> rk3f, rv3f;
        rk3f.resize(fk3f.size());
        rv3f.resize(fv3f.size());
        {
            std::vector<float> tmp;
            tmp.resize(size_t(s.head_dim));
            const int n_pad_r = pad_up(n_ctx);
            for (int g = 0; g < s.n_kv_heads; ++g) {
                for (int p = 0; p < n_pad_r; ++p) {
                    const size_t off = size_t(g) * size_t(n_pad_r) * size_t(s.head_dim) +
                                       size_t(p) * size_t(s.head_dim);
                    rot.apply(fk3f.data() + off, tmp.data(), false);
                    std::copy(tmp.begin(), tmp.end(), rk3f.begin() + long(off));
                }
                // Values are stored with positions in the fast dimension, so a rotation
                // acts across rows rather than along them - gather, rotate, scatter.
                for (int p = 0; p < n_pad_r; ++p) {
                    for (int j = 0; j < s.head_dim; ++j) {
                        tmp[size_t(j)] = fv3f[size_t(g) * size_t(s.head_dim) *
                                              size_t(n_pad_r) +
                                              size_t(j) * size_t(n_pad_r) + size_t(p)];
                    }
                    std::vector<float> out;
                    out.resize(size_t(s.head_dim));
                    rot.apply(tmp.data(), out.data(), false);
                    for (int j = 0; j < s.head_dim; ++j) {
                        rv3f[size_t(g) * size_t(s.head_dim) * size_t(n_pad_r) +
                             size_t(j) * size_t(n_pad_r) + size_t(p)] = out[size_t(j)];
                    }
                }
            }
        }

        // Only block-of-32 types. The k-quants use superblocks of 256, and the padded
        // context length is not a multiple of that, so including them would compare a
        // different padding rather than a different format.
        struct Arm { const char* name; ggml_type type; bool rotated; Graph g; };
        Arm arms[12] = {
            {"fp16 полный ранг", GGML_TYPE_F16, false, {}},
            {"Q8_0", GGML_TYPE_Q8_0, false, {}},
            {"Q8_0 + Адамар", GGML_TYPE_Q8_0, true, {}},
            {"Q8_0 + Адамар K", GGML_TYPE_Q8_0, false, {}},

            {"Q5_1", GGML_TYPE_Q5_1, false, {}},
            {"Q5_1 + Адамар", GGML_TYPE_Q5_1, true, {}},
            {"Q5_0 + Адамар", GGML_TYPE_Q5_0, true, {}},
            {"Q4_1", GGML_TYPE_Q4_1, false, {}},
            {"Q4_1 + Адамар", GGML_TYPE_Q4_1, true, {}},
            {"Q4_0", GGML_TYPE_Q4_0, false, {}},
            {"Q4_0 + Адамар", GGML_TYPE_Q4_0, true, {}},
            {"низкий ранг", GGML_TYPE_F16, false, {}},
        };
        const int n_arms = 12;
        // One arm alive at a time. With all seven built at once the errors reproduced to
        // the digit while the timings jumped by up to 700x between runs, and which arm was
        // slow changed each time - the same work at 0.50 ms in one run and 428 ms in the
        // next. Whatever the mechanism (address alignment or page mapping luck), a
        // measurement that unstable is not a measurement, so the reference outputs are
        // collected first and each candidate is then built, compared, timed and freed.
        bool ok = build_uniform(&arms[0].g, buft, s, fk3f, fv3f, n_ctx, GGML_TYPE_F16);
        if (ok) {
            printf("\nчто хранить в хвосте (весь контекст одним представлением,"
                   " ранг %d):\n", cfg.rank_k);
            printf("%20s %10s %12s %10s\n", "", "байт/поз", "ош. L2", "мс/слой");
            // Reference outputs for every query, taken once while the fp16 arm is the only
            // graph alive.
            std::vector<std::vector<float>> refs;
            for (int qi = 0; qi < n_queries; ++qi) {
                set_query(qi, 1.0);
                ggml_backend_tensor_set(arms[0].g.q, q.data(), 0, ggml_nbytes(arms[0].g.q));
                ggml_backend_graph_compute(be, arms[0].g.gf);
                ggml_backend_synchronize(be);
                std::vector<float> r;
                r.resize(size_t(n_out));
                ggml_backend_tensor_get(arms[0].g.out, r.data(), 0,
                                        ggml_nbytes(arms[0].g.out));
                refs.push_back(r);
            }
            const double ref_bytes = double(arms[0].g.read_bytes) / n_ctx;
            const double ref_ms = time_it(be, &arms[0].g, iters);
            printf("%20s %10.0f %11.2f%% %9.2f\n", arms[0].name, ref_bytes, 0.0, ref_ms);
            arms[0].g.free_all();

            std::vector<float> got, qrot, tmp;
            got.resize(size_t(n_out));
            qrot.resize(size_t(n_out));
            tmp.resize(size_t(s.head_dim));
            for (int a = 1; a < n_arms; ++a) {
                const bool built = a < n_arms - 1
                    ? build_uniform(&arms[a].g, buft, s,
                                    (arms[a].rotated || a == 3) ? rk3f : fk3f,
                                    arms[a].rotated ? rv3f : fv3f, n_ctx, arms[a].type)
                    : build_latent(&arms[a].g, buft, s, ack, acvt, n_ctx,
                                   cfg.rank_k, cfg.rank_v, wk3, wv3);
                if (!built) {
                    printf("%20s — не собрался\n", arms[a].name);
                    continue;
                }
                double rel_sum = 0.0;
                for (int qi = 0; qi < n_queries; ++qi) {
                    set_query(qi, 1.0);
                    const float* qsrc = q.data();
                    if (arms[a].rotated || a == 3) {
                        // The query goes in rotated so the scores come out unchanged.
                        for (int h = 0; h < s.n_q_heads; ++h) {
                            rot.apply(q.data() + size_t(h) * size_t(s.head_dim),
                                      qrot.data() + size_t(h) * size_t(s.head_dim), false);
                        }
                        qsrc = qrot.data();
                    }
                    ggml_backend_tensor_set(arms[a].g.q, qsrc, 0,
                                            ggml_nbytes(arms[a].g.q));
                    ggml_backend_graph_compute(be, arms[a].g.gf);
                    ggml_backend_synchronize(be);
                    ggml_backend_tensor_get(arms[a].g.out, got.data(), 0,
                                            ggml_nbytes(arms[a].g.out));
                    if (arms[a].rotated) {
                        // The weighted sum came out rotated; one inverse per head puts it
                        // back, at a cost that does not grow with context.
                        for (int h = 0; h < s.n_q_heads; ++h) {
                            float* o = got.data() + size_t(h) * size_t(s.head_dim);
                            rot.apply(o, tmp.data(), true);
                            std::copy(tmp.begin(), tmp.end(), o);
                        }
                    }
                    double rel = 0.0, w = 0.0;
                    compare(got, refs[size_t(qi)], &rel, &w);
                    rel_sum += rel;
                }
                const double ms = time_it(be, &arms[a].g, iters);
                printf("%20s %10.0f %11.2f%% %9.2f\n", arms[a].name,
                       double(arms[a].g.read_bytes) / n_ctx,
                       100.0 * rel_sum / n_queries, ms);
                arms[a].g.free_all();
            }
        } else {
            printf("\nсравнение представлений хвоста не собралось\n");
            arms[0].g.free_all();
        }
    }

    printf("\n%8s %10s %10s %12s\n", "масштаб", "ош. L2", "макс |d|", "макс. вес");
    for (double qs : {0.5, 1.0, 2.0}) {
        double rel_sum = 0.0, worst_all = 0.0, peak_sum = 0.0;
        for (int qi = 0; qi < n_queries; ++qi) {
            set_query(qi, qs);
            run_both();
            double rel = 0.0, worst = 0.0;
            compare(oz, oe, &rel, &worst);
            rel_sum += rel;
            worst_all = std::max(worst_all, worst);

            // Peakedness of the reference distribution for head 0, computed on the host:
            // the same error means different things at different temperatures, and without
            // this the reader cannot tell which regime the number came from.
            std::vector<double> sc;
            sc.resize(size_t(n_ctx));
            double mx = -1e30, sum = 0.0, top = 0.0;
            const float sc_f = 1.0f / std::sqrt(float(s.head_dim));
            for (int p = 0; p < n_ctx; ++p) {
                double acc = 0.0;
                for (int j = 0; j < s.head_dim; ++j) {
                    acc += double(q[size_t(j)]) *
                           double(h2f(full_k[size_t(p) * size_t(d) + size_t(j)]));
                }
                sc[size_t(p)] = acc * sc_f;
                mx = std::max(mx, sc[size_t(p)]);
            }
            for (int p = 0; p < n_ctx; ++p) {
                const double e = std::exp(sc[size_t(p)] - mx);
                sum += e;
                top = std::max(top, e);
            }
            peak_sum += top / sum;
        }
        printf("%8.1f %9.3f%% %10.4f %11.2f%%\n", qs, 100.0 * rel_sum / n_queries,
               worst_all, 100.0 * peak_sum / n_queries);
    }

    const double t_ex = time_it(be, &ge, iters);
    const double t_zo = time_it(be, &gz, iters);
    printf("\nодин слой, %s, %d потоков:\n", on_gpu ? "Vulkan" : "CPU", threads);
    printf("  точное по всем %d : %7.2f мс, читает %7.2f МБ\n",
           n_ctx, t_ex, ge.read_bytes / 1e6);
    printf("  по зонам         : %7.2f мс, читает %7.2f МБ  (в %.2f раза меньше байт, "
           "в %.2f раза быстрее)\n",
           t_zo, gz.read_bytes / 1e6,
           double(ge.read_bytes) / double(gz.read_bytes), t_ex / t_zo);
    printf("\nна 48 слоёв: точное %.1f мс -> %.2f ток/с, зоны %.1f мс -> %.2f ток/с\n",
           48 * t_ex, 1000.0 / (48 * t_ex), 48 * t_zo, 1000.0 / (48 * t_zo));

    ge.free_all();
    gz.free_all();
    ggml_backend_free(be);
    return 0;
}
