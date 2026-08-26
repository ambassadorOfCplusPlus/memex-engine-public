// A layered correctness harness for our own inference engine.
//
// Until now the engine had exactly one check: run memex-fwd, look at the twenty-four
// generated tokens, and see whether they match the reference's. That is one bit of
// information with no location attached. It has already cost us twice. Once it fired on a
// divergence at token twelve that turned out to be a genuine near-tie in the logits - a
// top-2 gap of 0.026 out of 25 - so there was nothing to fix and a day went into looking.
// And it cannot fire at all on the failures we are about to risk: an error that grows with
// context length, or one that only appears past the first cache block, will keep producing
// the same twenty-four tokens right up to the point where it does not.
//
// We are about to change how weights are loaded and to split every layer's expert tensor
// into two precision groups. Both changes are exactly the kind that breaks a few coordinates
// in one layer and leaves the argmax alone. So the checks are built first, and they are
// built to answer "where" rather than "whether":
//
//   1  tensor agreement per layer - the reference's own named intermediates against ours,
//      so the first layer that disagrees is named
//   2  the whole final logit vector - max difference, relative L2, and the rank order of the
//      top fifty, printed next to the reference's top-2 gap
//   3  token agreement with the tie question answered in advance: a disagreement is
//      classified against the reference's own margin at that position, so "numerically
//      different" and "wrong" stop being the same report
//   4  the cache and context path past its first block, including a length that is not a
//      multiple of four and positions past 256
//   5  determinism, which is nearly free and is the likeliest way a loader rewrite fails
//
// Two things about the harness itself, both paid for in earlier debugging.
//
// A comparison must never be able to pass by accident. An entirely broken reference graph
// once read as a perfect match, because the difference between two zero tensors is zero. So
// compare() refuses a shape mismatch, refuses a zero-norm operand on either side, refuses a
// non-finite value, and prints both norms next to every number it does report. There is no
// path through it that turns "nothing was computed" into "0% error".
//
// And no timing. This deliberately does not include <chrono>: it is a correctness harness,
// its numbers are errors and counts, and a measurement pipeline sharing the machine should
// not have to compete with it.
//
// NOTE ON DUPLICATION. The engine graph below is a copy of the one in memex-fwd.cpp, because
// that file has no header to include and is being edited in parallel. The copy is faithful
// to the default (non-expert-reduced) path. When the loader work lands, the right move is to
// lift the graph builder into examples/memex-fwd/memex-fwd.hpp and have both include it -
// until then, a change to memex-fwd's graph must be mirrored here or this harness is testing
// yesterday's engine.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

namespace {

// ---------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------

// The result of one comparison, including both operands' norms. The norms are not decoration:
// the failure this guards against is a reference that computed nothing, and the only way to
// see that in a relative error is to be told what it was relative to.
struct Cmp {
    bool ok = false;             // false means the comparison was refused, not that it failed
    const char* why = "";        // why it was refused
    std::size_t n_ours = 0;
    std::size_t n_ref = 0;
    double rel_l2 = -1.0;        // ||ours - ref|| / ||ref||
    double max_abs = 0.0;
    double norm_ours = 0.0;
    double norm_ref = 0.0;
};

Cmp compare(const std::vector<float>& ours, const std::vector<float>& ref) {
    Cmp r;
    r.n_ours = ours.size();
    r.n_ref = ref.size();
    if (ours.empty() || ref.empty()) {
        r.why = "пустой операнд";
        return r;
    }
    // Refused, not truncated and not broadcast. Comparing the overlap of two different
    // shapes once put our first token against the reference's last and reported a 209%
    // divergence that did not exist; this project has been bitten by that four times.
    if (ours.size() != ref.size()) {
        r.why = "размеры не совпадают";
        return r;
    }
    double num = 0.0, den = 0.0, sq_ours = 0.0, mx = 0.0;
    for (std::size_t i = 0; i < ours.size(); ++i) {
        const double a = double(ours[i]);
        const double b = double(ref[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            r.why = "NaN или бесконечность в данных";
            return r;
        }
        const double d = a - b;
        num += d * d;
        den += b * b;
        sq_ours += a * a;
        mx = std::max(mx, std::abs(d));
    }
    r.norm_ours = std::sqrt(sq_ours);
    r.norm_ref = std::sqrt(den);
    r.max_abs = mx;
    // Both zero-norm cases are refusals. A zero reference makes the relative error
    // meaningless; a zero candidate against a live reference would be reported as 100% and
    // read as a normal number, when in fact our side produced nothing at all.
    if (!(den > 0.0)) {
        r.why = "норма эталона равна нулю — сравнивать не с чем";
        return r;
    }
    if (!(sq_ours > 0.0)) {
        r.why = "норма нашего тензора равна нулю — мы ничего не посчитали";
        return r;
    }
    r.rel_l2 = std::sqrt(num / den);
    r.ok = true;
    return r;
}

void print_cmp(const char* label, const Cmp& c) {
    if (!c.ok) {
        printf("  %-26s ОТКАЗ: %s (наш %zu знач., эталон %zu знач., "
               "|наш| %.6g, |эталон| %.6g)\n",
               label, c.why, c.n_ours, c.n_ref, c.norm_ours, c.norm_ref);
        return;
    }
    printf("  %-26s n %7zu  отн.L2 %9.5f%%  макс|d| %11.6f  |наш| %11.4f  |эталон| %11.4f\n",
           label, c.n_ours, 100.0 * c.rel_l2, c.max_abs, c.norm_ours, c.norm_ref);
}

// ---------------------------------------------------------------------------------------
// Hyperparameters, read from the file rather than assumed
// ---------------------------------------------------------------------------------------

struct HParams {
    std::string arch;
    int n_layer = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff_exp = 0;
    int n_vocab = 0;
    int n_ctx_train = 0;
    float rms_eps = 1e-6f;
    float rope_base = 10000.0f;
    int rope_type = 0;

    int d_kv() const { return n_head_kv * head_dim; }
    int d_q() const { return n_head * head_dim; }
};

int key_u32(gguf_context* g, const std::string& k, int fallback) {
    const int id = gguf_find_key(g, k.c_str());
    return id < 0 ? fallback : int(gguf_get_val_u32(g, id));
}

float key_f32(gguf_context* g, const std::string& k, float fallback) {
    const int id = gguf_find_key(g, k.c_str());
    return id < 0 ? fallback : gguf_get_val_f32(g, id);
}

bool read_hparams(const char* path, HParams* h) {
    gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g) return false;
    const int aid = gguf_find_key(g, "general.architecture");
    h->arch = aid < 0 ? "" : gguf_get_val_str(g, aid);
    const std::string a = h->arch;
    h->n_layer = key_u32(g, a + ".block_count", 0);
    h->n_embd = key_u32(g, a + ".embedding_length", 0);
    h->n_head = key_u32(g, a + ".attention.head_count", 0);
    h->n_head_kv = key_u32(g, a + ".attention.head_count_kv", h->n_head);
    // Qwen3 states the head dimension explicitly, and it is not n_embd / n_head for it.
    h->head_dim = key_u32(g, a + ".attention.key_length",
                          h->n_head ? h->n_embd / h->n_head : 0);
    h->n_expert = key_u32(g, a + ".expert_count", 0);
    h->n_expert_used = key_u32(g, a + ".expert_used_count", 0);
    h->n_ff_exp = key_u32(g, a + ".expert_feed_forward_length",
                          key_u32(g, a + ".feed_forward_length", 0));
    h->n_ctx_train = key_u32(g, a + ".context_length", 4096);
    h->rms_eps = key_f32(g, a + ".attention.layer_norm_rms_epsilon", 1e-6f);
    h->rope_base = key_f32(g, a + ".rope.freq_base", 10000.0f);
    gguf_free(g);
    return h->n_layer > 0 && h->n_embd > 0;
}

// Rounded up to a whole number of 32-position blocks. Not tidiness: the fork's F16 matmul is
// silently wrong when the reduction length is not a multiple of four, and both the score and
// the value product reduce over positions. The fix in the engine is to pad to 32 and mask the
// remainder with -inf, which is why level 4 can use a prompt length that is not a multiple
// of four at all - and why that length is a meaningful regression case.
int pad32(int n) { return (n + 31) / 32 * 32; }

// ---------------------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------------------

ggml_tensor* norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

struct Weights {
    ggml_tensor* tok_embd = nullptr;
    ggml_tensor* out_norm = nullptr;
    ggml_tensor* out = nullptr;
    struct Layer {
        ggml_tensor* attn_norm = nullptr;
        ggml_tensor* wq = nullptr;
        ggml_tensor* wk = nullptr;
        ggml_tensor* wv = nullptr;
        ggml_tensor* wo = nullptr;
        ggml_tensor* q_norm = nullptr;
        ggml_tensor* k_norm = nullptr;
        ggml_tensor* ffn_norm = nullptr;
        ggml_tensor* router = nullptr;
        ggml_tensor* up = nullptr;
        ggml_tensor* gate = nullptr;
        ggml_tensor* down = nullptr;
    };
    std::vector<Layer> layers;
};

ggml_tensor* need(llama_model* m, const std::string& name, bool* ok) {
    ggml_tensor* t = llama_get_model_tensor(m, name.c_str());
    if (!t) {
        printf("нет тензора: %s\n", name.c_str());
        *ok = false;
    }
    return t;
}

bool collect(llama_model* m, const HParams& h, Weights* w) {
    bool ok = true;
    w->tok_embd = need(m, "token_embd.weight", &ok);
    w->out_norm = need(m, "output_norm.weight", &ok);
    // Some conversions tie the output head to the embedding table instead of storing it.
    w->out = llama_get_model_tensor(m, "output.weight");
    if (!w->out) {
        printf("output.weight отсутствует, беру token_embd (связанные веса)\n");
        w->out = w->tok_embd;
    }
    w->layers.resize(std::size_t(h.n_layer));
    for (int il = 0; il < h.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        Weights::Layer& L = w->layers[std::size_t(il)];
        L.attn_norm = need(m, p + "attn_norm.weight", &ok);
        L.wq = need(m, p + "attn_q.weight", &ok);
        L.wk = need(m, p + "attn_k.weight", &ok);
        L.wv = need(m, p + "attn_v.weight", &ok);
        L.wo = need(m, p + "attn_output.weight", &ok);
        L.q_norm = need(m, p + "attn_q_norm.weight", &ok);
        L.k_norm = need(m, p + "attn_k_norm.weight", &ok);
        L.ffn_norm = need(m, p + "ffn_norm.weight", &ok);
        L.router = need(m, p + "ffn_gate_inp.weight", &ok);
        L.up = need(m, p + "ffn_up_exps.weight", &ok);
        L.gate = need(m, p + "ffn_gate_exps.weight", &ok);
        L.down = need(m, p + "ffn_down_exps.weight", &ok);
    }
    return ok;
}

// ---------------------------------------------------------------------------------------
// Our cache
// ---------------------------------------------------------------------------------------

// Keys as [head_dim, n_ctx, n_kv_heads] and values transposed within each head as
// [n_ctx, head_dim, n_kv_heads], both in half precision - which is what the reference holds
// and what halves the bytes attention re-reads. Written once per position and never moved.
struct Cache {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> k, v;
    int n_ctx = 0;

    bool init(ggml_backend_buffer_type_t buft, const HParams& h, int ctx_len) {
        n_ctx = ctx_len;
        ggml_init_params ip = {ggml_tensor_overhead() * std::size_t(h.n_layer) * 2 + 4096,
                               nullptr, true};
        ctx = ggml_init(ip);
        if (!ctx) return false;
        k.resize(std::size_t(h.n_layer));
        v.resize(std::size_t(h.n_layer));
        for (int il = 0; il < h.n_layer; ++il) {
            k[std::size_t(il)] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, h.head_dim, n_ctx,
                                                    h.n_head_kv);
            v[std::size_t(il)] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_ctx, h.head_dim,
                                                    h.n_head_kv);
            if (!k[std::size_t(il)] || !v[std::size_t(il)]) {
                printf("кэш: тензор слоя %d не создался\n", il);
                return false;
            }
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) printf("кэш: буфер на %d позиций не выделился\n", n_ctx);
        return buf != nullptr;
    }

    std::size_t bytes(const HParams& h) const {
        return std::size_t(h.n_layer) * 2 * std::size_t(n_ctx) * std::size_t(h.d_kv()) * 2;
    }

    void free_all() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
        k.clear();
        v.clear();
    }
};

// One position's keys for one layer, as raw bytes across every kv head. Read back as bytes
// rather than floats on purpose: the questions level 4 asks of the cache are "did this
// change" and "are these two the same", and bytes answer both exactly.
bool read_k_position(const Cache& kv, const HParams& h, int il, int pos,
                     std::vector<uint8_t>* out) {
    if (il < 0 || std::size_t(il) >= kv.k.size()) {
        printf("кэш: нет слоя %d\n", il);
        return false;
    }
    ggml_tensor* t = kv.k[std::size_t(il)];
    if (!t || pos < 0 || pos >= kv.n_ctx) {
        printf("кэш: позиция %d вне кэша на %d позиций\n", pos, kv.n_ctx);
        return false;
    }
    const std::size_t row = std::size_t(h.head_dim) * ggml_element_size(t);
    out->assign(row * std::size_t(h.n_head_kv), 0);
    for (int hh = 0; hh < h.n_head_kv; ++hh) {
        const std::size_t off = std::size_t(hh) * t->nb[2] + std::size_t(pos) * t->nb[1];
        ggml_backend_tensor_get(t, out->data() + std::size_t(hh) * row, off, row);
    }
    return true;
}

bool all_zero(const std::vector<uint8_t>& b) {
    for (uint8_t x : b) {
        if (x) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// Our engine's graph
// ---------------------------------------------------------------------------------------

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_backend_buffer_t inputs = nullptr;
    ggml_tensor* tokens = nullptr;     // [n_tokens] i32
    ggml_tensor* positions = nullptr;  // [n_tokens] i32
    ggml_tensor* mask = nullptr;       // [n_kv, n_tokens] f32
    ggml_tensor* logits = nullptr;
    // Our intermediates, under the names the reference gives its own, so a comparison can be
    // asked for by name instead of by hand-counted node index.
    std::vector<std::pair<std::string, ggml_tensor*>> probes;

    // Where this step writes its key and value. The cache offset is the only thing in a
    // decode graph that depends on the position, so recording the destination views lets the
    // graph be built once and re-aimed per step instead of rebuilt.
    struct CacheWrite {
        ggml_tensor* cpy = nullptr;    // the GGML_OP_CPY node, itself a view of the cache
        ggml_tensor* dst = nullptr;    // its destination view, the node's src[1]
        ggml_tensor* cache = nullptr;  // the cache tensor both are views of
        std::size_t step = 0;          // bytes from one position to the next
        int64_t n_pos = 0;             // positions the cache holds, along its own axis
        int64_t n_write = 0;           // positions this step writes
    };
    std::vector<CacheWrite> writes;

    ggml_tensor* probe(const std::string& name) const {
        for (const auto& p : probes) {
            if (p.first == name) return p.second;
        }
        return nullptr;
    }

    // Re-aims every cache write at `n_past`. Allocation resolved each view to a raw pointer
    // (view_src->data + view_offs) and ggml_cpy's own result node is a second view of the
    // same cache carrying the same offset, so both the offset and the resolved pointer have
    // to move on both tensors - the same four fields the reference patches in
    // llama_context::update_cache_copies() to reuse its graphs.
    bool aim_cache_writes(int n_past) {
        if (writes.empty()) return false;
        for (const CacheWrite& c : writes) {
            // A silent mismatch here would write keys into whatever the pointer happened to
            // reach, so it is checked rather than assumed.
            if (!c.cpy || !c.dst || !c.cache || c.cpy->op != GGML_OP_CPY ||
                c.cpy->src[1] != c.dst || c.cpy->view_src != c.cache ||
                c.dst->view_src != c.cache || !c.cache->data ||
                n_past < 0 || n_past + c.n_write > c.n_pos) {
                return false;
            }
            const std::size_t offs = std::size_t(n_past) * c.step;
            c.dst->view_offs = offs;
            c.dst->data = (char*)c.cache->data + offs;
            c.cpy->view_offs = offs;
            c.cpy->data = c.dst->data;
        }
        return true;
    }

    // The byte offset the first recorded write is currently aimed at. Level 4 reads this to
    // confirm the offset actually advances: the bug it exists for was a decode graph that was
    // built once and never re-aimed, so every step rewrote position zero and the output
    // collapsed into a repeated ".?" after two tokens.
    std::size_t write_offset() const {
        return writes.empty() ? 0 : writes.front().dst->view_offs;
    }

    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (inputs) ggml_backend_buffer_free(inputs);
        if (ctx) ggml_free(ctx);
        alloc = nullptr;
        inputs = nullptr;
        ctx = nullptr;
        probes.clear();
        writes.clear();
    }
};

// The engine, one architecture only: qwen3moe. A copy of memex-fwd's graph with the two arms
// unified, so there is one place where the arithmetic can be wrong instead of two.
//
//   kv == nullptr  attention runs over the prompt itself under a causal mask and nothing is
//                  cached - the plain prefill arm, whose reduction length is n_tokens and so
//                  must be a multiple of four
//   kv != nullptr  keys and values are written into the cache at n_past and attention reads
//                  n_kv cached positions - the arm generation uses, where n_kv is padded to
//                  a multiple of 32 and the remainder masked off, so n_tokens is free
bool build_engine(Graph* g, ggml_backend_buffer_type_t buft, const HParams& h,
                  const Weights& w, Cache* kv, int n_tokens, int n_past, int n_kv,
                  bool want_probes) {
    if (n_tokens <= 0 || n_kv <= 0) {
        printf("граф: бессмысленные размеры n_tokens %d, n_kv %d\n", n_tokens, n_kv);
        return false;
    }
    const std::size_t n_nodes = std::size_t(h.n_layer) * 80 + 512;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 512) +
                               ggml_graph_overhead_custom(n_nodes, false),
                           nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) {
        printf("граф: ggml_init не выделил метаданные\n");
        return false;
    }
    ggml_context* c = g->ctx;

    // Inputs live in their own buffer so they can be rewritten between runs.
    g->tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_kv, n_tokens);
    if (!g->tokens || !g->positions || !g->mask) {
        printf("граф: входные тензоры не создались\n");
        return false;
    }
    g->inputs = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!g->inputs) {
        printf("граф: буфер входов не выделился\n");
        return false;
    }

    g->gf = ggml_new_graph_custom(c, n_nodes, false);
    if (!g->gf) {
        printf("граф: ggml_new_graph_custom вернул пустоту\n");
        return false;
    }
    const int hd = h.head_dim;
    const float kq_scale = 1.0f / std::sqrt(float(hd));
    // The reference calls ggml_rope_multi, and for a text-only model the section widths are
    // all zero. Calling ggml_rope_ext instead looks equivalent and is not: it left a 5%
    // difference in the logits while the argmax still matched, which is exactly the kind of
    // near-miss that would have been written off as accumulation order.
    int sections[GGML_MROPE_SECTIONS] = {0};

    ggml_tensor* cur = ggml_get_rows(c, w.tok_embd, g->tokens);
    for (int il = 0; il < h.n_layer; ++il) {
        const Weights::Layer& L = w.layers[std::size_t(il)];
        ggml_tensor* inpSA = cur;
        const std::string sil = "-" + std::to_string(il);

        ggml_tensor* x = norm(c, cur, L.attn_norm, h.rms_eps);
        if (want_probes) g->probes.push_back({"attn_norm" + sil, x});

        ggml_tensor* q = ggml_mul_mat(c, L.wq, x);
        ggml_tensor* k = ggml_mul_mat(c, L.wk, x);
        ggml_tensor* v = ggml_mul_mat(c, L.wv, x);

        // Per-head norm before the rotation, in that order. Reversing them changes the
        // answer, because the rotation is not scale-invariant per pair.
        q = ggml_reshape_3d(c, q, hd, h.n_head, n_tokens);
        q = norm(c, q, L.q_norm, h.rms_eps);
        q = ggml_rope_multi(c, q, g->positions, nullptr, hd, sections, h.rope_type,
                            h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        if (want_probes) g->probes.push_back({"Qcur_roped" + sil, q});

        k = ggml_reshape_3d(c, k, hd, h.n_head_kv, n_tokens);
        k = norm(c, k, L.k_norm, h.rms_eps);
        k = ggml_rope_multi(c, k, g->positions, nullptr, hd, sections, h.rope_type,
                            h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        // Kcur_roped, exactly. The fork renamed this node from Kcur because a prefix match
        // had been silently capturing Kcur_normed instead - four times over - and a
        // pre-rotation key compares against a rotated one at about the right magnitude, so
        // nothing looked broken.
        if (want_probes) g->probes.push_back({"Kcur_roped" + sil, k});

        // Heads into the last dimension for both sides, so mul_mat broadcasts the kv heads
        // across the query heads by itself. Doing this with strided two-dimensional views
        // instead was measured to give a wrong answer.
        ggml_tensor* Q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
        ggml_tensor* K = nullptr;
        ggml_tensor* V = nullptr;

        if (!kv) {
            // Keys and values in half precision, which is what a cache holds and what the
            // reference attends against. Keeping them in f32 was measured as the entire
            // source of divergence: 0.146% per layer, compounding to 4.9974% on the logits
            // after 48 of them, with the argmax matching all the way.
            K = ggml_cast(c, ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3)), GGML_TYPE_F16);
            V = ggml_cast(c,
                          ggml_cont(c, ggml_permute(
                                           c, ggml_reshape_3d(c, v, hd, h.n_head_kv, n_tokens),
                                           1, 2, 0, 3)),
                          GGML_TYPE_F16);
        } else {
            // Into the cache at n_past, then read back from position zero: the store and the
            // load are separate views of the same tensor, which is what makes a decode step
            // cost one position of writing and n_kv of reading.
            ggml_tensor* kt = kv->k[std::size_t(il)];
            ggml_tensor* vt = kv->v[std::size_t(il)];
            ggml_tensor* Kc = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
            ggml_tensor* Vc = ggml_cont(c, ggml_permute(
                c, ggml_reshape_3d(c, v, hd, h.n_head_kv, n_tokens), 1, 2, 0, 3));
            ggml_tensor* kdst = ggml_view_3d(c, kt, hd, n_tokens, h.n_head_kv, kt->nb[1],
                                             kt->nb[2], std::size_t(n_past) * kt->nb[1]);
            ggml_tensor* vdst = ggml_view_3d(c, vt, n_tokens, hd, h.n_head_kv, vt->nb[1],
                                             vt->nb[2],
                                             std::size_t(n_past) * ggml_element_size(vt));
            ggml_tensor* kcpy = ggml_cpy(c, Kc, kdst);
            ggml_tensor* vcpy = ggml_cpy(c, Vc, vdst);
            ggml_build_forward_expand(g->gf, kcpy);
            ggml_build_forward_expand(g->gf, vcpy);
            // Recorded so a decode graph can be re-aimed instead of rebuilt. The position
            // counts differ between K and V because V is stored transposed: a position is a
            // row of K but a column of V.
            g->writes.push_back({kcpy, kdst, kt, kt->nb[1], kt->ne[1], n_tokens});
            g->writes.push_back({vcpy, vdst, vt, ggml_element_size(vt), vt->ne[0], n_tokens});

            K = ggml_view_3d(c, kt, hd, n_kv, h.n_head_kv, kt->nb[1], kt->nb[2], 0);
            V = ggml_view_3d(c, vt, n_kv, hd, h.n_head_kv, vt->nb[1], vt->nb[2], 0);
        }

        ggml_tensor* kq = ggml_mul_mat(c, K, Q);
        ggml_tensor* p = ggml_soft_max_ext(c, kq, g->mask, kq_scale, 0.0f);
        ggml_tensor* kqv = ggml_mul_mat(c, V, p);
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), h.d_q(), n_tokens);
        ggml_tensor* attn_out = ggml_mul_mat(c, L.wo, kqv);
        // The reference's "attn_out" is this node, before the residual is added - it renames
        // kqv_wo to attn_out and only then adds. Keeping the same convention means one probe
        // needs no arithmetic on the reference side at all.
        if (want_probes) g->probes.push_back({"attn_out" + sil, attn_out});
        cur = ggml_add(c, attn_out, inpSA);

        ggml_tensor* ffn_inp = cur;
        if (want_probes) g->probes.push_back({"attn_resid" + sil, ffn_inp});
        x = norm(c, ffn_inp, L.ffn_norm, h.rms_eps);
        if (want_probes) g->probes.push_back({"ffn_inp_normed" + sil, x});

        // Routing. The fork's own path is reproduced exactly: softmax over all experts,
        // top-k, then renormalise the chosen weights - Qwen3 normalises, and skipping that
        // scales every expert output by the same wrong factor.
        ggml_tensor* logits_e = ggml_mul_mat(c, L.router, x);
        ggml_tensor* probs = ggml_soft_max(c, logits_e);
        ggml_tensor* sel = ggml_top_k(c, probs, h.n_expert_used);
        ggml_tensor* weights = ggml_get_rows(
            c, ggml_reshape_3d(c, probs, 1, h.n_expert, n_tokens), sel);
        weights = ggml_reshape_2d(c, weights, h.n_expert_used, n_tokens);
        weights = ggml_div(c, weights, ggml_sum_rows(c, weights));
        weights = ggml_reshape_3d(c, weights, 1, h.n_expert_used, n_tokens);

        ggml_tensor* xe = ggml_reshape_3d(c, x, h.n_embd, 1, n_tokens);
        ggml_tensor* up = ggml_mul_mat_id(c, L.up, xe, sel);
        ggml_tensor* gate = ggml_silu(c, ggml_mul_mat_id(c, L.gate, xe, sel));
        ggml_tensor* out = ggml_mul_mat_id(c, L.down, ggml_mul(c, up, gate), sel);
        out = ggml_mul(c, out, weights);
        // Sum the chosen experts' contributions: they arrive as n_expert_used slices.
        ggml_tensor* moe = ggml_view_2d(c, out, h.n_embd, n_tokens, out->nb[2], 0);
        for (int e = 1; e < h.n_expert_used; ++e) {
            moe = ggml_add(c, moe, ggml_view_2d(c, out, h.n_embd, n_tokens, out->nb[2],
                                                std::size_t(e) * out->nb[1]));
        }
        if (want_probes) g->probes.push_back({"moe_out" + sil, moe});
        cur = ggml_add(c, moe, ffn_inp);
        if (want_probes) g->probes.push_back({"l_out" + sil, cur});
    }

    // Only the last token's row goes through the output head. The head is 151936 x 2048 -
    // 243 MB even at six bits - so running it over a whole prefill would cost more than the
    // prefill. The reference prunes its last layer for the same reason, which is why its
    // captured tensors for the final layer arrive with one row; level 1 handles that
    // explicitly rather than by truncating.
    if (n_tokens > 1) {
        cur = ggml_cont(c, ggml_view_2d(c, cur, h.n_embd, 1, cur->nb[1],
                                        std::size_t(n_tokens - 1) * cur->nb[1]));
    }
    cur = norm(c, cur, w.out_norm, h.rms_eps);
    if (want_probes) g->probes.push_back({"result_norm", cur});
    g->logits = ggml_mul_mat(c, w.out, cur);
    ggml_set_output(g->logits);
    for (auto& pr : g->probes) {
        ggml_set_output(pr.second);  // keep them alive so they can be read back
    }
    ggml_build_forward_expand(g->gf, g->logits);
    g->alloc = ggml_gallocr_new(buft);
    if (!g->alloc) {
        printf("граф: ggml_gallocr_new не удался\n");
        return false;
    }
    if (!ggml_gallocr_reserve(g->alloc, g->gf)) {
        printf("граф: резервирование не удалось\n");
        return false;
    }
    if (!ggml_gallocr_alloc_graph(g->alloc, g->gf)) {
        printf("граф: размещение не удалось\n");
        return false;
    }
    return true;
}

bool read_tensor(ggml_tensor* t, std::vector<float>* out) {
    if (!t) return false;
    if (t->type != GGML_TYPE_F32) {
        printf("  чтение %s: тип %s, а не f32\n", t->name, ggml_type_name(t->type));
        return false;
    }
    // ggml_backend_tensor_get reads nbytes of contiguous memory. On a strided view that
    // silently returns the wrong values at the right magnitude, which is the worst kind of
    // wrong for a harness whose whole job is to be trusted - so it is refused. Every probe we
    // register happens to be an op result rather than a view, and this is what keeps it true
    // if the engine's graph changes shape underneath us.
    if (!ggml_is_contiguous(t)) {
        printf("  чтение %s: тензор не непрерывный, читать его как плоский массив нельзя\n",
               t->name);
        return false;
    }
    out->assign(std::size_t(ggml_nelements(t)), 0.0f);
    ggml_backend_tensor_get(t, out->data(), 0, ggml_nbytes(t));
    return true;
}

// ---------------------------------------------------------------------------------------
// Catching the reference's own intermediates
// ---------------------------------------------------------------------------------------

// Splits "l_out-31" into the base name "l_out" and the layer 31. Names without a trailing
// "-<digits>" - result_norm is the one that matters - come back with il = -1.
//
// Exact base matching, never a prefix. A prefix match on "Kcur" also catches Kcur_normed and
// Kcur_scales, and that is precisely the bug the fork renamed a node to fix: a tracer had
// been recording the pre-rotation key four times over, at plausible magnitudes, so the wrong
// tensor never announced itself.
void split_ref_name(const char* name, std::string* base, int* il) {
    const std::string n = name ? name : "";
    *base = n;
    *il = -1;
    const std::size_t dash = n.rfind('-');
    if (dash == std::string::npos || dash + 1 >= n.size()) return;
    for (std::size_t i = dash + 1; i < n.size(); ++i) {
        if (n[i] < '0' || n[i] > '9') return;
    }
    *base = n.substr(0, dash);
    *il = std::atoi(n.c_str() + dash + 1);
}

// The reference's qwen3moe graph names these, and only these, are worth 160 KB a piece to
// keep. Read off the fork's own source (build_qwen3moe -> build_std_attention and
// llm_build_std_moe_ffn), and re-checkable at any time with --list-ref-names, because a
// whitelist that has quietly stopped matching reports "no divergence" just as loudly as a
// correct engine does.
bool ref_name_wanted(const std::string& base) {
    static const char* kWanted[] = {
        "attn_norm",       // post-norm attention input
        "Qcur_roped",      // queries after the per-head norm and the rotation
        "Kcur_roped",      // keys likewise - the exact name, see split_ref_name
        "attn_out",        // wo output, before the residual add
        "ffn_inp_normed",  // the MoE block's normed input
        "routed_out",      // the MoE block's output, before the residual add
        "ffn_moe_out",     // what the qwen3vl variant calls the same thing
        "l_out",           // the layer's output residual stream
        "result_norm",     // the final norm, one row
    };
    for (const char* w : kWanted) {
        if (base == w) return true;
    }
    return false;
}

struct Probe {
    // Off except during the one reference decode whose intermediates level 1 wants. The
    // callback is installed for the whole life of the context, and the generation levels run
    // hundreds of decodes through it; capturing those would cost gigabytes and answer nothing.
    bool enabled = false;
    bool list_only = false;
    std::vector<std::pair<std::string, std::vector<float>>> all;
    std::vector<std::string> listed;

    const std::vector<float>* get(const std::string& n) const {
        for (const auto& p : all) {
            if (p.first == n) return &p.second;
        }
        return nullptr;
    }

    void clear() { all.clear(); }
};

int probe_cb(ggml_tensor* t, bool ask, void* user_data) {
    Probe* p = (Probe*)user_data;
    if (!p || !p->enabled) return 0;

    std::string base;
    int il = 0;
    split_ref_name(t->name, &base, &il);

    if (p->list_only) {
        if (!ask) return 1;
        for (const std::string& s : p->listed) {
            if (s == base) return 0;
        }
        p->listed.push_back(base);
        printf("  %-32s %-8s ne %lld,%lld,%lld,%lld\n", t->name, ggml_type_name(t->type),
               (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
               (long long)t->ne[3]);
        return 0;
    }

    const bool useful = ref_name_wanted(base) && t->type == GGML_TYPE_F32;
    if (ask) return useful ? 1 : 0;
    if (!useful) return 1;

    const std::string full = t->name;
    if (p->get(full)) return 1;  // first occurrence wins; a name is written once per graph
    std::vector<float> d;
    d.assign(std::size_t(ggml_nelements(t)), 0.0f);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(d.data(), t->data, ggml_nbytes(t));
    } else {
        ggml_backend_tensor_get(t, d.data(), 0, ggml_nbytes(t));
    }
    p->all.push_back({full, std::move(d)});
    return 1;
}

// The reference has no name for the post-attention residual stream: qwen3moe adds it and
// hands the sum straight into the MoE block without a callback. It is recoverable exactly,
// because l_out = routed_out + ffn_inp is a plain elementwise add, so ffn_inp = l_out -
// routed_out to within the rounding of one addition.
//
// Recovering it from within the same layer, rather than as attn_out + l_out of the previous
// layer, is deliberate: an operand missing from one layer then cannot silently shift the
// whole comparison by a layer, which would make every number look wrong and the location
// look right.
bool ref_attn_resid(const Probe& p, int il, std::vector<float>* out, std::string* why) {
    const std::string sil = "-" + std::to_string(il);
    const std::vector<float>* l_out = p.get("l_out" + sil);
    const std::vector<float>* routed = p.get("routed_out" + sil);
    if (!routed) routed = p.get("ffn_moe_out" + sil);
    if (!l_out) {
        *why = "эталон не выдал l_out" + sil;
        return false;
    }
    if (!routed) {
        *why = "эталон не выдал ни routed_out" + sil + ", ни ffn_moe_out" + sil;
        return false;
    }
    if (l_out->size() != routed->size()) {
        *why = "у эталона l_out и routed_out разного размера";
        return false;
    }
    out->assign(l_out->size(), 0.0f);
    for (std::size_t i = 0; i < l_out->size(); ++i) {
        (*out)[i] = (*l_out)[i] - (*routed)[i];
    }
    return true;
}

// The reference prunes its final layer to the tokens it will emit logits for, so every
// captured tensor of layer n_layer-1 arrives as one row against our n_tokens. That is the one
// size difference that is legitimate, and this is the only place allowed to act on it: it
// selects our last row and says so in the label. Truncating instead once put our first token
// against the reference's last and reported a 209% divergence that did not exist.
std::vector<float> last_row(const std::vector<float>& v, std::size_t width) {
    if (width == 0 || v.size() < width || v.size() % width != 0) return std::vector<float>();
    return std::vector<float>(v.end() - std::ptrdiff_t(width), v.end());
}

// ---------------------------------------------------------------------------------------
// Logit-level helpers
// ---------------------------------------------------------------------------------------

int argmax_of(const std::vector<float>& v) {
    if (v.empty()) return -1;
    return int(std::max_element(v.begin(), v.end()) - v.begin());
}

// How much the winning logit won by. A greedy sequence can diverge from the reference on a
// near-tie without anything being wrong, and after the fact that is indistinguishable from a
// real fault - so the margin is recorded as it is produced.
double top2_gap(const std::vector<float>& v) {
    double best = -INFINITY, second = -INFINITY;
    for (float x : v) {
        const double d = double(x);
        if (d > best) {
            second = best;
            best = d;
        } else if (d > second) {
            second = d;
        }
    }
    return best - second;
}

// Spearman rank correlation between the reference's ordering of its own top-k tokens and
// ours, restricted to those k. The interesting question is whether the two engines would
// offer a sampler the same candidates in the same order, not whether they agree about the
// hundred and fifty thousand tokens no sampler will ever reach.
double top_k_rank_corr(const std::vector<float>& ours, const std::vector<float>& ref, int k,
                       bool* ok) {
    *ok = false;
    if (ours.size() != ref.size() || ours.empty()) return 0.0;
    const int n = int(ours.size());
    if (k < 3 || k > n) return 0.0;
    std::vector<int> ids;
    ids.assign(std::size_t(n), 0);
    for (int i = 0; i < n; ++i) ids[std::size_t(i)] = i;
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) {
        if (ref[std::size_t(a)] != ref[std::size_t(b)]) {
            return ref[std::size_t(a)] > ref[std::size_t(b)];
        }
        return a < b;  // a stable tie-break, so the ranks are well defined
    });
    std::vector<int> top(ids.begin(), ids.begin() + k);
    std::vector<int> mine = top;
    std::sort(mine.begin(), mine.end(), [&](int a, int b) {
        if (ours[std::size_t(a)] != ours[std::size_t(b)]) {
            return ours[std::size_t(a)] > ours[std::size_t(b)];
        }
        return a < b;
    });
    double sum_d2 = 0.0;
    for (int i = 0; i < k; ++i) {
        int r = -1;
        for (int j = 0; j < k; ++j) {
            if (mine[std::size_t(j)] == top[std::size_t(i)]) {
                r = j;
                break;
            }
        }
        if (r < 0) return 0.0;  // cannot happen: mine is a permutation of top
        const double d = double(i) - double(r);
        sum_d2 += d * d;
    }
    *ok = true;
    return 1.0 - 6.0 * sum_d2 / (double(k) * (double(k) * double(k) - 1.0));
}

// Relative L2 over the whole logit vector answers a question no sampler ever asks, and it
// reads far more alarming than the situation is. Two reasons, both measurable here rather
// than asserted:
//   - softmax is invariant to an additive shift, so any constant offset between the two
//     vectors lands in the numerator while changing no probability at all;
//   - the denominator is a sum over the whole vocabulary, ~150k entries of which sit near
//     the floor at |logit| ~ 3, so the metric is an average relative error over tokens no
//     sampler will reach, not over the handful that decide the token.
// These are the numbers that bound the decision instead. flip_margin is the one that
// matters: argmax cannot move while 2*max|d| over the candidate set stays under the
// reference's own top-2 gap, so flip_margin < 1 is a proof of token agreement, not evidence
// of it.
struct LogitCmp {
    bool ok = false;
    const char* why = "";
    double rel_full = -1.0;      // ||d|| / ||ref||, the number that looks bad
    double rel_centred = -1.0;   // the same after each side has its own mean removed
    double rel_shifted = -1.0;   // the same after each side has its own max removed
    double rel_top[3] = {-1.0, -1.0, -1.0};  // restricted to the reference's top k
    int k_used[3] = {1, 10, 100};
    double max_abs_top = 0.0;    // largest |d| inside the reference's top k_used[2]
    double gap = 0.0;            // the reference's own top-2 gap
    double flip_margin = -1.0;   // 2*max_abs_top / gap; < 1 proves argmax cannot flip
};

LogitCmp compare_logits(const std::vector<float>& ours, const std::vector<float>& ref) {
    LogitCmp r;
    if (ours.empty() || ref.empty()) {
        r.why = "пустой операнд";
        return r;
    }
    if (ours.size() != ref.size()) {
        r.why = "размеры не совпадают";
        return r;
    }
    const std::size_t n = ref.size();
    double num = 0.0, den = 0.0, sum_a = 0.0, sum_b = 0.0;
    double max_a = -INFINITY, max_b = -INFINITY;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = double(ours[i]);
        const double b = double(ref[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            r.why = "NaN или бесконечность в данных";
            return r;
        }
        const double d = a - b;
        num += d * d;
        den += b * b;
        sum_a += a;
        sum_b += b;
        max_a = std::max(max_a, a);
        max_b = std::max(max_b, b);
    }
    if (!(den > 0.0)) {
        r.why = "норма эталона равна нулю — сравнивать не с чем";
        return r;
    }
    r.rel_full = std::sqrt(num / den);

    const double mean_a = sum_a / double(n);
    const double mean_b = sum_b / double(n);
    double num_c = 0.0, den_c = 0.0, num_s = 0.0, den_s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ca = double(ours[i]) - mean_a;
        const double cb = double(ref[i]) - mean_b;
        num_c += (ca - cb) * (ca - cb);
        den_c += cb * cb;
        const double sa = double(ours[i]) - max_a;
        const double sb = double(ref[i]) - max_b;
        num_s += (sa - sb) * (sa - sb);
        den_s += sb * sb;
    }
    // A zero centred/shifted reference norm would mean a constant reference vector, which
    // for logits cannot happen; refuse rather than divide, so it can never print as 0%.
    if (den_c > 0.0) r.rel_centred = std::sqrt(num_c / den_c);
    if (den_s > 0.0) r.rel_shifted = std::sqrt(num_s / den_s);

    // The reference's own ordering picks the candidate set: those are the entries a sampler
    // can reach, and the only ones whose error can change the emitted token.
    const int kmax = int(std::min<std::size_t>(n, 100));
    r.k_used[0] = 1;
    r.k_used[1] = int(std::min(10, kmax));
    r.k_used[2] = kmax;
    std::vector<int> ids;
    ids.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) ids[i] = int(i);
    std::partial_sort(ids.begin(), ids.begin() + kmax, ids.end(), [&](int a, int b) {
        if (ref[std::size_t(a)] != ref[std::size_t(b)]) {
            return ref[std::size_t(a)] > ref[std::size_t(b)];
        }
        return a < b;  // stable tie-break, so the set is well defined
    });
    for (int s = 0; s < 3; ++s) {
        const int k = r.k_used[s];
        double tn = 0.0, td = 0.0;
        for (int i = 0; i < k; ++i) {
            const std::size_t id = std::size_t(ids[std::size_t(i)]);
            const double d = double(ours[id]) - double(ref[id]);
            tn += d * d;
            td += double(ref[id]) * double(ref[id]);
            if (std::abs(d) > r.max_abs_top) r.max_abs_top = std::abs(d);
        }
        if (td > 0.0) r.rel_top[s] = std::sqrt(tn / td);
    }

    r.gap = top2_gap(ref);
    if (r.gap > 0.0) r.flip_margin = 2.0 * r.max_abs_top / r.gap;
    r.ok = true;
    return r;
}

void print_logit_cmp(const char* label, const LogitCmp& c) {
    if (!c.ok) {
        printf("  %-26s ОТКАЗ: %s\n", label, c.why);
        return;
    }
    printf("  %-26s весь словарь %.5f%%; без среднего %.5f%%; без максимума %.5f%%\n",
           label, 100.0 * c.rel_full, 100.0 * c.rel_centred, 100.0 * c.rel_shifted);
    printf("  %-26s top-%d %.5f%%; top-%d %.5f%%; top-%d %.5f%%\n", "", c.k_used[0],
           100.0 * c.rel_top[0], c.k_used[1], 100.0 * c.rel_top[1], c.k_used[2],
           100.0 * c.rel_top[2]);
    printf("  %-26s макс|d| в top-%d %.6f, отрыв top-2 %.6f, запас по argmax %.4f (<1 — "
           "argmax перевернуться не может)\n", "", c.k_used[2], c.max_abs_top, c.gap,
           c.flip_margin);
}

std::string piece_of(llama_model* m, llama_token t) {
    char buf[128] = {0};
    const int len = llama_token_to_piece(m, t, buf, sizeof(buf) - 1, 0, true);
    if (len <= 0) return std::string("<?>");
    std::string s(buf, std::size_t(len));
    // Newlines and tabs in a table column are worse than useless.
    for (char& ch : s) {
        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
    }
    return s;
}

// ---------------------------------------------------------------------------------------
// The engine, wrapped so the levels read as intent rather than as plumbing
// ---------------------------------------------------------------------------------------

struct Engine {
    ggml_backend_t be = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    const HParams* h = nullptr;
    const Weights* w = nullptr;

    // Fills a step's inputs. The mask is -inf everywhere and zero only where a query at
    // position past+i may see key j, which is both the causal mask and the mask over the
    // padding that pad32 added.
    void set_inputs(Graph& gr, const llama_token* tk, int nt, int past, int n_kv) const {
        std::vector<int32_t> ps;
        ps.assign(std::size_t(nt), 0);
        for (int i = 0; i < nt; ++i) ps[std::size_t(i)] = past + i;
        std::vector<float> mk;
        mk.assign(std::size_t(n_kv) * std::size_t(nt), -INFINITY);
        for (int i = 0; i < nt; ++i) {
            const int last = past + i;
            for (int j = 0; j <= last && j < n_kv; ++j) {
                mk[std::size_t(i) * std::size_t(n_kv) + std::size_t(j)] = 0.0f;
            }
        }
        ggml_backend_tensor_set(gr.tokens, tk, 0, sizeof(int32_t) * std::size_t(nt));
        ggml_backend_tensor_set(gr.positions, ps.data(), 0, sizeof(int32_t) * std::size_t(nt));
        ggml_backend_tensor_set(gr.mask, mk.data(), 0, ggml_nbytes(gr.mask));
    }

    void run(Graph& gr) const {
        ggml_backend_graph_compute(be, gr.gf);
        ggml_backend_synchronize(be);
    }

    // The last position's logits, which after the output-head pruning is the only row there
    // is.
    bool logits_of(Graph& gr, std::vector<float>* out) const {
        if (!gr.logits) return false;
        const std::size_t rows = std::size_t(gr.logits->ne[1]);
        if (std::size_t(gr.logits->ne[0]) != std::size_t(h->n_vocab)) {
            printf("логиты нашего графа имеют ширину %lld, а словарь %d\n",
                   (long long)gr.logits->ne[0], h->n_vocab);
            return false;
        }
        out->assign(std::size_t(h->n_vocab), 0.0f);
        ggml_backend_tensor_get(gr.logits, out->data(),
                                (rows - 1) * std::size_t(h->n_vocab) * sizeof(float),
                                sizeof(float) * std::size_t(h->n_vocab));
        return true;
    }
};

// ---------------------------------------------------------------------------------------
// Result bookkeeping
// ---------------------------------------------------------------------------------------

struct LevelResult {
    bool selected = false;
    bool ran = false;
    bool pass = false;
    std::string key;  // the one number that says what happened
};

const char* verdict_of(const LevelResult& r) {
    if (!r.selected) return "-";
    if (!r.ran) return "НЕ ЗАПУЩЕН";
    return r.pass ? "ПРОШЁЛ" : "ПРОВАЛ";
}

}  // namespace

// ===========================================================================================
// main
// ===========================================================================================

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::string model_path = "D:/Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf";
    std::string prompt = "The quick brown fox jumps over the lazy dog. "
                         "Write a function that reverses a linked list in place.";
    std::string levels_arg = "1,2,3,4,5";
    int threads = 4;
    int max_tokens = 64;
    int n_gen = 16;
    double tie_threshold = 0.05;
    // Measured, not guessed. Baseline: Qwen3-Coder-30B-A3B (48 layers, 128/8 experts) on a
    // 22-token prompt, a run levels 3, 4 and 5 independently certified equivalent - 16/16
    // and 8/8 token agreement over three cache cases, zero real disagreements, bit-identical
    // reruns. The comparable last-row column topped out at 4.15% at layer 47, the all-rows
    // column at 1.76% at layer 46. 6% is 1.45x the observed floor: enough headroom that
    // rounding drift and a somewhat longer prompt do not trip it, tight enough that a real
    // fault - which shows as a step change, not a 40% overshoot - still does. The floor
    // grows about as sqrt(context length), so re-measure for prompts far longer than this.
    double l1_tol = 0.06;
    // Same baseline, same reasoning: full-vocabulary logit rel L2 measured 7.35% on a run
    // whose argmax matched exactly. That metric is dominated by ~150k near-floor logits and
    // is not decision-relevant (see compare_logits), so the gate is generous and the
    // decision-relevant numbers printed beside it are what to read.
    double l2_tol = 0.30;
    double rho_tol = 0.98;
    // argmax cannot move while 2*max|d| over the candidate set stays below the reference's
    // own top-2 gap. That is a proof, not a threshold, so the default sits just under 1.
    double flip_tol = 0.90;
    int top_k_corr = 50;
    int stress_len = 252;   // 252 + 8 generated positions crosses 256
    int stress_gen = 8;
    bool verbose = false;
    bool l1_nocache = false;
    bool list_ref_names = false;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!strcmp(a, "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(a, "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(a, "--levels") && i + 1 < argc) levels_arg = argv[++i];
        else if (!strcmp(a, "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--gen") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(a, "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(a, "--tie-threshold") && i + 1 < argc) tie_threshold = atof(argv[++i]);
        else if (!strcmp(a, "--l1-tol") && i + 1 < argc) l1_tol = atof(argv[++i]);
        else if (!strcmp(a, "--l2-tol") && i + 1 < argc) l2_tol = atof(argv[++i]);
        else if (!strcmp(a, "--rho-tol") && i + 1 < argc) rho_tol = atof(argv[++i]);
        else if (!strcmp(a, "--flip-tol") && i + 1 < argc) flip_tol = atof(argv[++i]);
        else if (!strcmp(a, "--top-k") && i + 1 < argc) top_k_corr = atoi(argv[++i]);
        else if (!strcmp(a, "--stress-len") && i + 1 < argc) stress_len = atoi(argv[++i]);
        else if (!strcmp(a, "--stress-gen") && i + 1 < argc) stress_gen = atoi(argv[++i]);
        else if (!strcmp(a, "--l1-nocache")) l1_nocache = true;
        else if (!strcmp(a, "--list-ref-names")) list_ref_names = true;
        else if (!strcmp(a, "--verbose") || !strcmp(a, "-v")) verbose = true;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            printf("memex-test — послойная проверка нашего движка против эталона\n"
                   "  -m <модель>            путь к GGUF\n"
                   "  -p <промпт>            текст промпта\n"
                   "  --levels 1,2,3,4,5     какие уровни запускать (по умолчанию все)\n"
                   "  --gen N                сколько позиций сверять на уровне 3\n"
                   "  --max-tokens N         предел длины промпта в токенах\n"
                   "  --tie-threshold F      отрыв top-2, ниже которого расхождение — ничья\n"
                   "  --l1-tol F             допуск послойной отн. L2 (доля, не проценты)\n"
                   "  --l2-tol F             допуск отн. L2 на логитах\n"
                   "  --rho-tol F            минимальная ранговая корреляция top-k\n"
                   "  --flip-tol F           предел запаса по argmax (2*макс|d| / отрыв)\n"
                   "  --top-k N              сколько токенов в ранговой корреляции\n"
                   "  --stress-len N         длина промпта для уровня 4\n"
                   "  --stress-gen N         сколько позиций генерировать на уровне 4\n"
                   "  --l1-nocache           уровень 1 через бескэшевый префилл\n"
                   "  --list-ref-names       выписать имена узлов эталона и выйти\n"
                   "  -t N                   потоков\n"
                   "  --verbose              подробности по каждому слою и позиции\n"
                   "код возврата: 0 — все выбранные уровни прошли, иначе 1\n");
            return 0;
        } else {
            printf("неизвестный аргумент: %s (см. --help)\n", a);
            return 1;
        }
    }

    LevelResult lv[6];
    if (levels_arg == "all") levels_arg = "1,2,3,4,5";
    {
        std::size_t pos = 0;
        while (pos < levels_arg.size()) {
            std::size_t comma = levels_arg.find(',', pos);
            if (comma == std::string::npos) comma = levels_arg.size();
            const std::string tok = levels_arg.substr(pos, comma - pos);
            const int n = std::atoi(tok.c_str());
            if (n < 1 || n > 5) {
                printf("--levels: «%s» не уровень, ожидается от 1 до 5\n", tok.c_str());
                return 1;
            }
            lv[n].selected = true;
            pos = comma + 1;
        }
    }
    if (list_ref_names) {
        // Only the reference is needed, and only its names.
        lv[1].selected = lv[2].selected = lv[3].selected = lv[4].selected = false;
        lv[5].selected = false;
    }
    if (threads < 1) threads = 1;
    if (n_gen < 1) n_gen = 1;
    if (stress_gen < 2) stress_gen = 2;
    if (top_k_corr < 3) top_k_corr = 3;

    HParams h;
    if (!read_hparams(model_path.c_str(), &h)) {
        printf("не читаются гиперпараметры из %s\n", model_path.c_str());
        return 1;
    }
    printf("архитектура %s: слоёв %d, n_embd %d, голов %d/%d, head_dim %d,\n"
           "  экспертов %d из них %d, ширина эксперта %d, rms_eps %g, rope_base %g\n",
           h.arch.c_str(), h.n_layer, h.n_embd, h.n_head, h.n_head_kv, h.head_dim,
           h.n_expert, h.n_expert_used, h.n_ff_exp, h.rms_eps, h.rope_base);

    // Refused early and explicitly. Our engine's graph is written for one architecture: it
    // needs the fused expert tensors and the per-head q/k norms, and a dense Qwen3 has
    // neither. Failing here with a sentence beats failing later with a missing tensor.
    if (h.arch != "qwen3moe") {
        printf("движок собран только под qwen3moe, а здесь «%s» — отказываюсь.\n"
               "  (это же причина, по которой мелкие модели на диске не годятся даже для\n"
               "   дымового прогона: 0.6B qwen3 — плотная, экспертных тензоров у неё нет)\n",
               h.arch.c_str());
        return 1;
    }
    if (h.n_expert_used <= 0 || h.n_expert <= 0) {
        printf("expert_count %d / expert_used_count %d — не похоже на MoE\n", h.n_expert,
               h.n_expert_used);
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.repack_tensors = false;  // the harness compares arithmetic, not layouts
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        printf("модель не загрузилась: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }
    h.n_vocab = llama_n_vocab(model);
    h.rope_type = int(llama_rope_type(model));
    printf("словарь %d, тип rope %d\n", h.n_vocab, h.rope_type);

    // Tokenise through the fork's tokeniser: it is correct and we have no reason to own it.
    std::vector<llama_token> toks;
    toks.assign(prompt.size() + 8, 0);
    int n = llama_tokenize(model, prompt.c_str(), int(prompt.size()), toks.data(),
                           int(toks.size()), true, false);
    if (n < 0) {
        toks.assign(std::size_t(-n), 0);
        n = llama_tokenize(model, prompt.c_str(), int(prompt.size()), toks.data(),
                           int(toks.size()), true, false);
    }
    if (n <= 0) {
        printf("промпт не токенизировался\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }
    n = std::min(n, max_tokens);
    // The no-cache arm reduces over n_tokens directly, and the fork's F16 matmul is silently
    // wrong when the reduction length is not a multiple of four. The cache arm pads to 32 and
    // masks, so it has no such restriction - which is exactly why level 4 uses a length that
    // is not a multiple of four as a regression case.
    if (l1_nocache && n % 4 != 0) {
        printf("бескэшевый арм: длина промпта %d не кратна 4, обрезаю до %d "
               "(F16-matmul форка на такой длине тихо неверен)\n", n, n / 4 * 4);
        n = n / 4 * 4;
    }
    if (n <= 0) {
        printf("после обрезки от промпта ничего не осталось\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }
    toks.resize(std::size_t(n));
    printf("токенов в промпте: %d\n", n);

    Weights w;
    if (!collect(model, h, &w)) {
        printf("не все тензоры на месте — архитектура не та, что ожидалась\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    Engine eng;
    eng.be = ggml_backend_cpu_init();
    if (!eng.be) {
        printf("CPU-бэкенд не инициализировался\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }
    ggml_backend_cpu_set_n_threads(eng.be, threads);
    eng.buft = ggml_backend_cpu_buffer_type();
    eng.h = &h;
    eng.w = &w;

    // The reference context, needed by every level that has something to compare against.
    // Level 5 does not: it asks whether we agree with ourselves, which is why --levels 5 is
    // the one mode that loads no second copy of anything.
    const bool need_ref = list_ref_names || lv[1].selected || lv[2].selected ||
                          lv[3].selected || lv[4].selected;
    int ctx_need = std::max(n + n_gen, stress_len + 1 + stress_gen) + 64;
    if (ctx_need < 512) ctx_need = 512;
    // n_ubatch has to cover the longest single prefill, and this is not about speed. If the
    // reference splits a prompt across two micro-batches, its final layer is pruned to the
    // output tokens of whichever micro-batch it is in, and the first occurrence of each name -
    // which is what the probe keeps - then belongs to the *first* micro-batch, not the one
    // holding the last token. Level 1 would compare our whole prefill against the reference's
    // opening fragment and report a divergence that is entirely an artefact.
    const int max_prefill = std::max(n, stress_len + 1);
    const int ubatch = std::max(max_prefill, 512);
    if (ctx_need < ubatch) ctx_need = ubatch;
    Probe probe;
    llama_context* lctx = nullptr;
    if (need_ref) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = uint32_t(ctx_need);
        cp.n_batch = uint32_t(ubatch);
        cp.n_ubatch = uint32_t(ubatch);
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.flash_attn = false;  // the plain path, which is what our graph mirrors
        cp.cb_eval = probe_cb;
        cp.cb_eval_user_data = &probe;
        lctx = llama_init_from_model(model, cp);
        if (!lctx) {
            printf("эталонный контекст не создался (n_ctx %d)\n", ctx_need);
            ggml_backend_free(eng.be);
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        printf("эталонный контекст: n_ctx %d, n_batch/n_ubatch %d, потоков %d\n", ctx_need,
               ubatch, threads);
    }

    std::vector<float> ref_lg, our_lg;
    ref_lg.assign(std::size_t(h.n_vocab), 0.0f);

    // A small closure over the reference: decode a batch and hand back the last position's
    // logits. Every failure it can have is reported, because a silent failure here would
    // leave a stale logit vector in place and every level would compare against it.
    auto ref_decode = [&](const llama_token* tk, int nt, int past,
                          std::vector<float>* out) -> bool {
        if (!lctx) {
            printf("  эталон не создан, а его просят посчитать\n");
            return false;
        }
        if (llama_decode(lctx, llama_batch_get_one((llama_token*)tk, nt, past, 0))) {
            printf("  llama_decode не прошёл (%d токенов с позиции %d)\n", nt, past);
            return false;
        }
        const float* p = llama_get_logits(lctx);
        if (!p) {
            printf("  эталонные логиты недоступны\n");
            return false;
        }
        out->assign(std::size_t(h.n_vocab), 0.0f);
        std::memcpy(out->data(), p, sizeof(float) * std::size_t(h.n_vocab));
        return true;
    };

    // -----------------------------------------------------------------------------------
    // --list-ref-names: what the reference actually calls its nodes
    // -----------------------------------------------------------------------------------
    if (list_ref_names) {
        printf("\nимена узлов эталонного графа (по одному на базовое имя):\n");
        probe.enabled = true;
        probe.list_only = true;
        llama_kv_cache_clear(lctx);
        const bool ok = ref_decode(toks.data(), n, 0, &ref_lg);
        probe.enabled = false;
        probe.list_only = false;
        printf("\nиз них в белом списке уровня 1:\n");
        for (const std::string& s : probe.listed) {
            if (ref_name_wanted(s)) printf("  %s\n", s.c_str());
        }
        llama_free(lctx);
        ggml_backend_free(eng.be);
        llama_free_model(model);
        llama_backend_free();
        return ok ? 0 : 1;
    }

    // -----------------------------------------------------------------------------------
    // Levels 1 and 2 share one prefill on each side
    // -----------------------------------------------------------------------------------
    if (lv[1].selected || lv[2].selected) {
        printf("\n=== эталонный префилл на %d токенах ===\n", n);
        probe.enabled = lv[1].selected;
        probe.clear();
        llama_kv_cache_clear(lctx);
        const bool ref_ok = ref_decode(toks.data(), n, 0, &ref_lg);
        probe.enabled = false;
        if (!ref_ok) {
            printf("эталон не посчитал префилл — уровни 1 и 2 запустить нельзя\n");
        } else {
            bool capture_ok = true;
            if (lv[1].selected) {
                printf("эталон отдал %zu именованных тензоров\n", probe.all.size());
                // The capture is only usable if the reference saw the whole prompt in one
                // micro-batch. Checked rather than assumed, because the failure mode is a
                // comparison against a fragment, which looks exactly like a real divergence.
                const std::vector<float>* l0 = probe.get("l_out-0");
                const std::size_t want = std::size_t(h.n_embd) * std::size_t(n);
                if (!l0) {
                    printf("эталон не выдал l_out-0 — белый список имён устарел? "
                           "проверьте --list-ref-names\n");
                    capture_ok = false;
                } else if (l0->size() != want) {
                    printf("эталонный l_out-0 содержит %zu значений вместо %zu — промпт "
                           "разбился на несколько микропакетов, сравнивать нельзя\n",
                           l0->size(), want);
                    capture_ok = false;
                }
            }

            // Our side. Level 1 defaults to the cache arm: it is the arm generation uses, it
            // has no multiple-of-four restriction, and running the same builder for the
            // layer probes and for the generation levels means one place where the
            // arithmetic can be wrong instead of two.
            Cache kv;
            Graph pre;
            bool our_ok = true;
            const int n_kv_max = pad32(n);
            if (!l1_nocache) {
                if (!kv.init(eng.buft, h, n_kv_max)) {
                    printf("кэш не выделился\n");
                    our_ok = false;
                }
            }
            if (our_ok && !build_engine(&pre, eng.buft, h, w, l1_nocache ? nullptr : &kv, n, 0,
                                        l1_nocache ? n : n_kv_max, /*want_probes=*/true)) {
                printf("наш граф не собрался\n");
                our_ok = false;
            }
            if (our_ok) {
                eng.set_inputs(pre, toks.data(), n, 0, l1_nocache ? n : n_kv_max);
                eng.run(pre);
                if (!eng.logits_of(pre, &our_lg)) {
                    printf("наши логиты не прочитались\n");
                    our_ok = false;
                }
            }

            // ---------------- Level 1 ----------------
            if (lv[1].selected) {
                lv[1].ran = our_ok && ref_ok && capture_ok;
                printf("\n=== уровень 1: послойное согласие тензоров ===\n");
                printf("арм: %s\n", l1_nocache ? "бескэшевый префилл"
                                               : "через наш KV-кэш (тот же, что в генерации)");
                if (!lv[1].ran) {
                    lv[1].key = "не запущен";
                } else {
                    bool level_ok = true;
                    int refused = 0;
                    double worst = -1.0;
                    int worst_layer = -1;
                    double worst_jump = -1.0;
                    int jump_layer = -1;
                    double prev_l_out = -1.0;
                    int missing_names = 0;
                    double worst_lastrow = -1.0;
                    int worst_lastrow_layer = -1;
                    double prev_lastrow = -1.0;
                    double worst_lastrow_jump = -1.0;
                    int lastrow_jump_layer = -1;
                    // Every layer, two headline numbers: the residual stream after attention
                    // and after the FFN. The point is not the numbers themselves - two
                    // implementations of the same arithmetic will differ by rounding and the
                    // difference will drift with depth - it is the shape of the curve. Steady
                    // growth is accumulation. A step change at one layer is a bug in that
                    // layer, and that is the whole reason this level exists.
                    //
                    // The fourth column exists because the first three cannot be read across
                    // the last layer. The reference prunes layer n_layer-1 to the rows it
                    // emits logits for (build_std_attention: `if (inp_out_ids) cur =
                    // ggml_get_rows(...)`), so there the first three columns are one row -
                    // the deepest position, which carries the most accumulated error - while
                    // every layer above them is an RMS over all n_tokens rows, most of which
                    // are early positions attending over almost nothing. Comparing 1.76% at
                    // layer 46 against 4.15% at layer 47 is comparing two different
                    // populations, and that alone reads as a step change at the last layer.
                    // The last-row column is the same row at every depth, so a step in it is
                    // a step in the arithmetic. That is the column the verdict is taken from.
                    printf("  %-6s %-12s %-12s %-12s %-12s %s\n", "слой", "после attn",
                           "после FFN", "прирост FFN", "посл.строка", "примечание");
                    for (int il = 0; il < h.n_layer; ++il) {
                        const std::string sil = "-" + std::to_string(il);
                        const bool is_last = il == h.n_layer - 1;
                        // The note below is Cyrillic: ~145 UTF-8 bytes. snprintf truncated it
                        // safely but mid-character, so the column printed garbage at the end.
                        char note[192] = {0};

                        // Post-attention residual, reference reconstructed inside the layer.
                        std::vector<float> ref_resid;
                        std::string why;
                        Cmp c_attn;
                        if (!ref_attn_resid(probe, il, &ref_resid, &why)) {
                            printf("  %-6d ОТКАЗ: %s\n", il, why.c_str());
                            ++missing_names;
                            level_ok = false;
                            continue;
                        }
                        std::vector<float> ours_resid;
                        if (!read_tensor(pre.probe("attn_resid" + sil), &ours_resid)) {
                            printf("  %-6d ОТКАЗ: у нас нет attn_resid%s\n", il, sil.c_str());
                            level_ok = false;
                            continue;
                        }
                        std::vector<float> ours_l_out;
                        const std::vector<float>* ref_l_out = probe.get("l_out" + sil);
                        if (!read_tensor(pre.probe("l_out" + sil), &ours_l_out) || !ref_l_out) {
                            printf("  %-6d ОТКАЗ: нет l_out%s с одной из сторон\n", il,
                                   sil.c_str());
                            level_ok = false;
                            continue;
                        }
                        // Both sides' last row, taken before any pruning-driven trim, so the
                        // comparable column is the same row at every depth.
                        const Cmp c_last = compare(last_row(ours_l_out, std::size_t(h.n_embd)),
                                                   last_row(*ref_l_out, std::size_t(h.n_embd)));

                        // The one legitimate size difference, named out loud.
                        if (is_last && ours_resid.size() != ref_resid.size()) {
                            ours_resid = last_row(ours_resid, ref_resid.size());
                            ours_l_out = last_row(ours_l_out, ref_l_out->size());
                            snprintf(note, sizeof(note) - 1,
                                     "последний слой эталона обрезан до выходных токенов, "
                                     "беру нашу последнюю строку");
                        }
                        c_attn = compare(ours_resid, ref_resid);
                        const Cmp c_lout = compare(ours_l_out, *ref_l_out);

                        if (!c_attn.ok || !c_lout.ok) {
                            printf("  %-6d ОТКАЗ  ", il);
                            if (!c_attn.ok) printf("после attn: %s; ", c_attn.why);
                            if (!c_lout.ok) printf("после FFN: %s; ", c_lout.why);
                            printf("|наш| %.6g/%.6g, |эталон| %.6g/%.6g\n", c_attn.norm_ours,
                                   c_lout.norm_ours, c_attn.norm_ref, c_lout.norm_ref);
                            ++refused;
                            level_ok = false;
                            continue;
                        }
                        const double jump = c_lout.rel_l2 - c_attn.rel_l2;
                        if (c_last.ok) {
                            printf("  %-6d %11.5f%% %11.5f%% %11.5f%% %11.5f%%  %s\n", il,
                                   100.0 * c_attn.rel_l2, 100.0 * c_lout.rel_l2,
                                   100.0 * jump, 100.0 * c_last.rel_l2, note);
                        } else {
                            // Never a silent zero: if the row could not be taken, say so.
                            printf("  %-6d %11.5f%% %11.5f%% %11.5f%% %11s  %s\n", il,
                                   100.0 * c_attn.rel_l2, 100.0 * c_lout.rel_l2,
                                   100.0 * jump, "ОТКАЗ", note);
                            level_ok = false;
                        }
                        if (c_lout.rel_l2 > worst) {
                            worst = c_lout.rel_l2;
                            worst_layer = il;
                        }
                        if (prev_l_out >= 0.0) {
                            const double step = c_lout.rel_l2 - prev_l_out;
                            if (step > worst_jump) {
                                worst_jump = step;
                                jump_layer = il;
                            }
                        }
                        prev_l_out = c_lout.rel_l2;
                        if (c_last.ok) {
                            if (c_last.rel_l2 > worst_lastrow) {
                                worst_lastrow = c_last.rel_l2;
                                worst_lastrow_layer = il;
                            }
                            if (prev_lastrow >= 0.0) {
                                const double step = c_last.rel_l2 - prev_lastrow;
                                if (step > worst_lastrow_jump) {
                                    worst_lastrow_jump = step;
                                    lastrow_jump_layer = il;
                                }
                            }
                            prev_lastrow = c_last.rel_l2;
                        }
                        // The verdict is taken from the comparable column. The first three
                        // are diagnostics: at the last layer they change population, and
                        // gating on them is what made this level fail on a healthy engine.
                        if (c_last.ok && c_last.rel_l2 > l1_tol) level_ok = false;
                        if (!is_last && (c_attn.rel_l2 > l1_tol || c_lout.rel_l2 > l1_tol)) {
                            level_ok = false;
                        }

                        if (verbose) {
                            static const char* extras[] = {"attn_norm", "Qcur_roped",
                                                           "Kcur_roped", "attn_out",
                                                           "ffn_inp_normed"};
                            for (const char* base : extras) {
                                const std::string full = std::string(base) + sil;
                                std::vector<float> mine;
                                const std::vector<float>* theirs = probe.get(full);
                                if (!theirs) {
                                    printf("    %-24s эталон такого узла не выдал\n",
                                           full.c_str());
                                    continue;
                                }
                                if (!read_tensor(pre.probe(full), &mine)) {
                                    printf("    %-24s у нас такого узла нет\n", full.c_str());
                                    continue;
                                }
                                if (is_last && mine.size() != theirs->size() &&
                                    theirs->size() > 0 && mine.size() % theirs->size() == 0) {
                                    mine = last_row(mine, theirs->size());
                                }
                                print_cmp(full.c_str(), compare(mine, *theirs));
                            }
                            const std::string moe = "moe_out" + sil;
                            const std::vector<float>* rmoe = probe.get("routed_out" + sil);
                            if (!rmoe) rmoe = probe.get("ffn_moe_out" + sil);
                            std::vector<float> mymoe;
                            if (rmoe && read_tensor(pre.probe(moe), &mymoe)) {
                                if (is_last && mymoe.size() != rmoe->size() &&
                                    rmoe->size() > 0 && mymoe.size() % rmoe->size() == 0) {
                                    mymoe = last_row(mymoe, rmoe->size());
                                }
                                print_cmp(moe.c_str(), compare(mymoe, *rmoe));
                            } else {
                                printf("    %-24s сравнить не с чем\n", moe.c_str());
                            }
                        }
                    }

                    // The final norm, which is the last thing before the output head and so
                    // the last place a layer-level fault can still be localised.
                    {
                        std::vector<float> mine;
                        const std::vector<float>* theirs = probe.get("result_norm");
                        if (!theirs) {
                            printf("  result_norm: эталон такого узла не выдал\n");
                            ++missing_names;
                            level_ok = false;
                        } else if (!read_tensor(pre.probe("result_norm"), &mine)) {
                            printf("  result_norm: у нас такого узла нет\n");
                            level_ok = false;
                        } else {
                            const Cmp c = compare(mine, *theirs);
                            print_cmp("result_norm", c);
                            if (!c.ok || c.rel_l2 > l1_tol) level_ok = false;
                        }
                    }

                    char key[320] = {0};
                    if (refused || missing_names) {
                        snprintf(key, sizeof(key) - 1,
                                 "отказов %d, ненайденных имён %d — сравнение недостоверно",
                                 refused, missing_names);
                    } else {
                        snprintf(key, sizeof(key) - 1,
                                 "посл.строка: худший слой %d: %.5f%%, крупнейший скачок на "
                                 "слое %d: %+.5f%% (допуск %.3f%%); все строки: худший слой "
                                 "%d: %.5f%%, скачок на слое %d: %+.5f%%",
                                 worst_lastrow_layer, 100.0 * worst_lastrow,
                                 lastrow_jump_layer, 100.0 * worst_lastrow_jump,
                                 100.0 * l1_tol, worst_layer, 100.0 * worst, jump_layer,
                                 100.0 * worst_jump);
                    }
                    lv[1].key = key;
                    lv[1].pass = level_ok;
                    printf("  итог: %s — %s\n", level_ok ? "ПРОШЁЛ" : "ПРОВАЛ", key);
                    if (worst_layer >= 0 && jump_layer >= 0) {
                        printf("  где искать: расхождение растёт плавно — накопление;\n"
                               "              скачок на одном слое — ошибка в этом слое\n");
                        // The tolerance is no longer a guess: see the note at its default.
                        // What is still worth saying out loud is that the floor scales with
                        // context length, so a much longer prompt legitimately drifts further.
                        printf("  допуск %.3f%% измерен по прогону, признанному эквивалентным\n"
                               "  уровнями 3-5 (посл.строка дала %.5f%% на слое %d). Пол\n"
                               "  растёт примерно как sqrt(длины контекста) — на промпте\n"
                               "  много длиннее этого пересними базовую линию\n",
                               100.0 * l1_tol, 100.0 * worst_lastrow, worst_lastrow_layer);
                    }
                }
            }

            // ---------------- Level 2 ----------------
            if (lv[2].selected) {
                printf("\n=== уровень 2: согласие логитов ===\n");
                lv[2].ran = our_ok && ref_ok && !our_lg.empty();
                if (!lv[2].ran) {
                    lv[2].key = "не запущен";
                } else {
                    const Cmp c = compare(our_lg, ref_lg);
                    print_cmp("логиты", c);
                    // The same vectors again, measured the way a sampler would care about.
                    const LogitCmp lc = compare_logits(our_lg, ref_lg);
                    print_logit_cmp("логиты, по существу", lc);
                    bool rho_ok = false;
                    const int k = std::min(top_k_corr, h.n_vocab);
                    const double rho = top_k_rank_corr(our_lg, ref_lg, k, &rho_ok);
                    const double gap = top2_gap(ref_lg);
                    const int a_ref = argmax_of(ref_lg);
                    const int a_our = argmax_of(our_lg);
                    printf("  ранговая корреляция top-%d: %s\n", k,
                           rho_ok ? (std::to_string(rho)).c_str() : "не вычислена");
                    printf("  отрыв top-2 у эталона: %.6f (порог ничьей %.4f)\n", gap,
                           tie_threshold);
                    printf("  лучший токен: эталон %d «%s», наш %d «%s» — %s\n", a_ref,
                           piece_of(model, a_ref).c_str(), a_our,
                           piece_of(model, a_our).c_str(),
                           a_ref == a_our ? "совпал" : "РАСХОДЯТСЯ");
                    // An argmax disagreement is only a failure when the reference itself was
                    // decisive. That distinction is the entire reason this harness exists:
                    // the divergence at token twelve that cost a day was a gap of 0.026.
                    const bool argmax_ok = a_ref == a_our || gap < tie_threshold;
                    if (a_ref != a_our && gap < tie_threshold) {
                        printf("  расхождение при отрыве %.6f < %.4f — это ничья, "
                               "не ошибка\n", gap, tie_threshold);
                    }
                    // The verdict now rests on the decision-relevant numbers. The
                    // full-vocabulary rel L2 is still gated, but loosely: it is an average
                    // relative error over 150k tokens no sampler reaches, and holding the
                    // whole level hostage to it is what made this level fail on a run whose
                    // argmax matched exactly and whose top-50 ordering agreed at rho 0.991.
                    const bool flip_ok = lc.ok && lc.flip_margin >= 0.0 &&
                                         lc.flip_margin <= flip_tol;
                    lv[2].pass = c.ok && lc.ok && c.rel_l2 <= l2_tol && rho_ok &&
                                 rho >= rho_tol && argmax_ok && flip_ok;
                    char key[288] = {0};
                    if (!c.ok) {
                        snprintf(key, sizeof(key) - 1, "сравнение отклонено: %s", c.why);
                    } else if (!lc.ok) {
                        snprintf(key, sizeof(key) - 1, "сравнение по существу отклонено: %s",
                                 lc.why);
                    } else {
                        snprintf(key, sizeof(key) - 1,
                                 "запас по argmax %.4f, top-%d отн.L2 %.5f%%, без среднего "
                                 "%.5f%%, rho %.5f, отрыв top-2 %.4f (весь словарь %.5f%%)",
                                 lc.flip_margin, lc.k_used[2], 100.0 * lc.rel_top[2],
                                 100.0 * lc.rel_centred, rho, gap, 100.0 * c.rel_l2);
                    }
                    lv[2].key = key;
                    printf("  итог: %s — %s\n", lv[2].pass ? "ПРОШЁЛ" : "ПРОВАЛ", key);
                }
            }

            pre.free_all();
            kv.free_all();
            // The capture is a few hundred megabytes at a 64-token prompt and nothing after
            // level 2 reads it. Levels 3 and 4 want that memory for cache and compute
            // buffers instead.
            probe.all.clear();
            probe.all.shrink_to_fit();
        }
    }

    // -----------------------------------------------------------------------------------
    // Levels 3 and 4: token agreement, with the tie question answered as it goes
    // -----------------------------------------------------------------------------------

    // One case: `prompt` of length `np`, then `npos` positions judged against the reference
    // under teacher forcing - our engine is fed the reference's token at every step, so every
    // position is judged in an identical context.
    //
    // Free-running both sides instead yields one number, the length of the common prefix, and
    // nothing after the first disagreement means anything, because the two engines are by
    // then answering different questions. Teacher forcing gives npos independent comparisons
    // for the same work, and it still contains the free-running answer: up to the first
    // disagreement the two schemes are the same sequence, so the position of that first
    // disagreement *is* the common prefix length.
    struct CaseOut {
        bool ran = false;
        bool pass = false;
        std::string key;
    };

    auto run_case = [&](const char* label, const std::vector<llama_token>& pr, int npos,
                        bool check_cache) -> CaseOut {
        CaseOut r;
        const int np = int(pr.size());
        printf("\n--- случай «%s»: промпт %d токенов%s, позиций %d ---\n", label, np,
               np % 4 ? " (НЕ кратно 4)" : "", npos);
        if (np <= 0 || npos <= 0) {
            r.key = "бессмысленные размеры";
            return r;
        }
        if (np + npos > ctx_need) {
            printf("  промпт и генерация (%d) не влезают в эталонный n_ctx %d\n", np + npos,
                   ctx_need);
            r.key = "не хватает n_ctx у эталона";
            return r;
        }
        const int n_kv_max = pad32(np + npos);
        Cache kv;
        if (!kv.init(eng.buft, h, n_kv_max)) {
            r.key = "кэш не выделился";
            return r;
        }
        printf("  наш кэш: %d позиций, %.1f МБ; %d позиций дополнения маскируются -inf\n",
               n_kv_max, double(kv.bytes(h)) / 1e6, n_kv_max - (np + npos));

        Graph pre, dec;
        bool ok = build_engine(&pre, eng.buft, h, w, &kv, np, 0, n_kv_max, false);
        if (ok) ok = build_engine(&dec, eng.buft, h, w, &kv, 1, np, n_kv_max, false);
        if (!ok) {
            printf("  графы не собрались\n");
            r.key = "графы не собрались";
            dec.free_all();
            pre.free_all();
            kv.free_all();
            return r;
        }

        probe.enabled = false;  // hundreds of decodes go through the callback below
        llama_kv_cache_clear(lctx);

        int n_agree = 0, n_tie = 0, n_real = 0, first_bad = -1;
        double worst_rel = 0.0, worst_tie_gap = 0.0;
        int worst_pos = -1;
        // The decision-relevant companions to worst_rel. worst_flip is the headline: it is a
        // bound on whether argmax could have moved, not a proxy for it.
        double worst_flip = 0.0, worst_top = 0.0, worst_centred = 0.0;
        int worst_flip_pos = -1, worst_top_pos = -1;
        double first_flip = -1.0, last_flip = -1.0;
        bool logit_metrics_ok = true;
        bool refused = false;
        std::vector<llama_token> ref_seq;

        // Step 0: both sides see only the prompt.
        std::vector<float> rl, ol;
        if (!ref_decode(pr.data(), np, 0, &rl)) {
            refused = true;
        } else {
            eng.set_inputs(pre, pr.data(), np, 0, n_kv_max);
            eng.run(pre);
            if (!eng.logits_of(pre, &ol)) refused = true;
        }

        // Snapshots for the structural cache checks. Position 0 is snapshotted because the
        // bug this level exists for was a decode graph that was never re-aimed: every step
        // rewrote position 0, and the output collapsed into a repeated ".?" after two tokens.
        std::vector<uint8_t> pos0_before, poslast_before;
        if (!refused && check_cache) {
            if (!read_k_position(kv, h, 0, 0, &pos0_before) ||
                !read_k_position(kv, h, 0, np - 1, &poslast_before)) {
                refused = true;
            } else if (all_zero(pos0_before)) {
                printf("  кэш: позиция 0 слоя 0 пуста после префилла — префилл ничего не "
                       "записал\n");
                refused = true;
            }
        }

        std::vector<std::vector<uint8_t>> written;
        std::size_t prev_offset = 0;
        bool offsets_ok = true;

        for (int i = 0; i < npos && !refused; ++i) {
            const int rt = argmax_of(rl);
            const int ot = argmax_of(ol);
            const double gap = top2_gap(rl);
            const Cmp c = compare(ol, rl);
            if (!c.ok) {
                printf("  позиция %d: ОТКАЗ сравнения логитов: %s "
                       "(|наш| %.6g, |эталон| %.6g)\n", i, c.why, c.norm_ours, c.norm_ref);
                refused = true;
                break;
            }
            if (c.rel_l2 > worst_rel) {
                worst_rel = c.rel_l2;
                worst_pos = i;
            }
            const LogitCmp lc = compare_logits(ol, rl);
            if (!lc.ok) {
                printf("  позиция %d: ОТКАЗ сравнения по существу: %s\n", i, lc.why);
                logit_metrics_ok = false;
            } else {
                if (lc.flip_margin > worst_flip) {
                    worst_flip = lc.flip_margin;
                    worst_flip_pos = i;
                }
                if (lc.rel_top[2] > worst_top) {
                    worst_top = lc.rel_top[2];
                    worst_top_pos = i;
                }
                worst_centred = std::max(worst_centred, lc.rel_centred);
                if (first_flip < 0.0) first_flip = lc.flip_margin;
                last_flip = lc.flip_margin;
            }
            if (rt == ot) {
                ++n_agree;
            } else {
                if (first_bad < 0) first_bad = i;
                const bool tie = gap < tie_threshold;
                if (tie) {
                    ++n_tie;
                    worst_tie_gap = std::max(worst_tie_gap, gap);
                } else {
                    ++n_real;
                }
                printf("  позиция %d: эталон %d «%s», наш %d «%s»; отрыв top-2 у эталона "
                       "%.6f -> %s (отн.L2 логитов %.5f%%)\n", i, rt,
                       piece_of(model, rt).c_str(), ot, piece_of(model, ot).c_str(), gap,
                       tie ? "НИЧЬЯ" : "НАСТОЯЩЕЕ РАСХОЖДЕНИЕ", 100.0 * c.rel_l2);
            }
            if (verbose) {
                printf("    позиция %2d: отн.L2 %9.5f%%, макс|d| %9.6f, отрыв %.6f, "
                       "|наш| %10.3f, |эталон| %10.3f\n", i, 100.0 * c.rel_l2, c.max_abs, gap,
                       c.norm_ours, c.norm_ref);
                if (lc.ok) {
                    printf("                 без среднего %9.5f%%, без максимума %9.5f%%, "
                           "top-%d %9.5f%%, top-%d %9.5f%%, top-%d %9.5f%%, "
                           "запас по argmax %.4f\n",
                           100.0 * lc.rel_centred, 100.0 * lc.rel_shifted, lc.k_used[0],
                           100.0 * lc.rel_top[0], lc.k_used[1], 100.0 * lc.rel_top[1],
                           lc.k_used[2], 100.0 * lc.rel_top[2], lc.flip_margin);
                }
            }
            ref_seq.push_back(llama_token(rt));
            if (i + 1 >= npos) break;

            // Teacher forcing: both sides are advanced with the reference's token.
            const llama_token fed = llama_token(rt);
            if (!ref_decode(&fed, 1, np + i, &rl)) {
                refused = true;
                break;
            }
            if (!dec.aim_cache_writes(np + i)) {
                printf("  не удалось перенаправить запись в кэш на позицию %d\n", np + i);
                refused = true;
                break;
            }
            const std::size_t off = dec.write_offset();
            if (i > 0 && off <= prev_offset) {
                printf("  смещение записи в кэш не выросло: было %zu, стало %zu — "
                       "это ровно та ошибка, из-за которой вывод сворачивался в «.?»\n",
                       prev_offset, off);
                offsets_ok = false;
            }
            prev_offset = off;
            eng.set_inputs(dec, &fed, 1, np + i, n_kv_max);
            eng.run(dec);
            if (!eng.logits_of(dec, &ol)) {
                refused = true;
                break;
            }
            if (check_cache) {
                std::vector<uint8_t> row;
                if (!read_k_position(kv, h, 0, np + i, &row)) {
                    refused = true;
                    break;
                }
                if (all_zero(row)) {
                    printf("  кэш: позиция %d слоя 0 пуста после шага — запись не дошла\n",
                           np + i);
                    offsets_ok = false;
                }
                written.push_back(row);
            }
        }

        bool cache_ok = offsets_ok;
        if (!refused && check_cache) {
            // Position 0 and the last prompt position must be untouched by generation.
            std::vector<uint8_t> pos0_after, poslast_after;
            if (!read_k_position(kv, h, 0, 0, &pos0_after) ||
                !read_k_position(kv, h, 0, np - 1, &poslast_after)) {
                refused = true;
            } else {
                if (pos0_after != pos0_before) {
                    printf("  кэш: позицию 0 перезаписали во время генерации — "
                           "это та самая ошибка с «.?»\n");
                    cache_ok = false;
                }
                if (poslast_after != poslast_before) {
                    printf("  кэш: последнюю позицию промпта (%d) перезаписали\n", np - 1);
                    cache_ok = false;
                }
            }
            // Distinct positions must hold distinct keys. Identical bytes at two positions
            // means either a repeated write to one slot or a stride computed wrong.
            int dup = 0;
            for (std::size_t a = 0; a + 1 < written.size(); ++a) {
                if (written[a] == written[a + 1]) ++dup;
            }
            if (dup) {
                printf("  кэш: %d пар соседних записанных позиций побайтово совпадают — "
                       "запись идёт не туда, куда думает граф\n", dup);
                cache_ok = false;
            }
            printf("  кэш: проверено %zu записанных позиций, смещение дошло до %zu байт\n",
                   written.size(), prev_offset);
        }

        const int judged = n_agree + n_tie + n_real;
        char key[320] = {0};
        if (refused) {
            snprintf(key, sizeof(key) - 1, "прогон прерван на позиции %d из %d", judged, npos);
            r.key = key;
            r.ran = false;
        } else {
            // A run whose only disagreements are ties passes, and says so.
            r.pass = n_real == 0 && cache_ok && logit_metrics_ok;
            r.ran = true;
            snprintf(key, sizeof(key) - 1,
                     "согласие %d/%d, ничьих %d, настоящих расхождений %d; общий префикс %d; "
                     "худший запас по argmax %.4f на позиции %d; худшая top-%d отн.L2 %.5f%% "
                     "(весь словарь %.5f%%)",
                     n_agree, judged, n_tie, n_real, first_bad < 0 ? judged : first_bad,
                     worst_flip, worst_flip_pos, 100, 100.0 * worst_top, 100.0 * worst_rel);
            r.key = key;
            printf("  согласие: %d из %d (%.1f%%)\n", n_agree, judged,
                   judged ? 100.0 * double(n_agree) / double(judged) : 0.0);
            printf("  общий префикс до первого расхождения: %d\n",
                   first_bad < 0 ? judged : first_bad);
            if (n_real == 0 && n_tie > 0) {
                printf("  все %d расхождений — ничьи (худший отрыв %.6f < %.4f): ПРОШЁЛ\n",
                       n_tie, worst_tie_gap, tie_threshold);
            } else if (n_real == 0) {
                printf("  расхождений нет\n");
            } else {
                printf("  настоящих расхождений: %d — это не ничьи\n", n_real);
            }
            // The headline used to be the full-vocabulary rel L2, and it read as a crisis:
            // 26.7% next to 8/8 token agreement. It is not a crisis, it is the wrong metric.
            // ||d||/||ref|| over 151936 logits is dominated by the ~150k entries near the
            // floor (|logit| ~ 3 against a top logit an order of magnitude larger), so it
            // reports an average relative error over tokens no sampler reaches; and softmax
            // is invariant to an additive shift, so any constant offset inflates it while
            // changing nothing. The numbers below are the ones that bound the decision, and
            // the last of them is a proof: while it stays under 1 argmax cannot move.
            printf("  запас по argmax: худший %.4f на позиции %d (<1 — argmax перевернуться "
                   "не мог)\n", worst_flip, worst_flip_pos);
            printf("  отн.L2 логитов: top-100 худшая %.5f%% на позиции %d; без среднего "
                   "худшая %.5f%%; по всему словарю %.5f%% на позиции %d\n",
                   100.0 * worst_top, worst_top_pos, 100.0 * worst_centred, 100.0 * worst_rel,
                   worst_pos);
            // Flat across positions means a per-step numerical difference whose size is set
            // by the prompt length; growing means the cache is degrading as it is reused.
            if (first_flip >= 0.0) {
                printf("  запас по argmax: первая позиция %.4f, последняя %.4f — %s\n",
                       first_flip, last_flip,
                       last_flip > 2.0 * first_flip
                           ? "растёт с позицией: смотреть на переиспользование кэша"
                           : "плоско по позициям: пошаговая численная разница, не кэш");
            }
            if (check_cache) {
                printf("  структурная проверка кэша: %s\n", cache_ok ? "ПРОШЛА" : "ПРОВАЛ");
            }
            printf("  итог: %s\n", r.pass ? "ПРОШЁЛ" : "ПРОВАЛ");
        }

        dec.free_all();
        pre.free_all();
        kv.free_all();
        return r;
    };

    if (lv[3].selected) {
        printf("\n=== уровень 3: согласие токенов с учётом ничьих ===\n");
        if (!lctx) {
            lv[3].key = "эталон не создан";
        } else {
            const CaseOut c = run_case("базовый промпт", toks, n_gen, /*check_cache=*/false);
            lv[3].ran = c.ran;
            lv[3].pass = c.pass;
            lv[3].key = c.key;
        }
    }

    if (lv[4].selected) {
        printf("\n=== уровень 4: кэш и длина контекста ===\n");
        if (!lctx) {
            lv[4].key = "эталон не создан";
        } else {
            // Case A: the plain generation path, with the cache checked structurally. This is
            // the case the ".?" collapse would have failed instantly.
            const CaseOut a = run_case("кэш за первым блоком", toks, n_gen, true);

            // Cases B and C need a long prompt. Cycling the prompt's own tokens keeps the
            // reference and our engine on identical input, which is all the comparison needs;
            // that the text is repetitive only makes near-ties more likely, and near-ties are
            // exactly what level 3's classification already handles.
            auto cycled = [&](int want) {
                std::vector<llama_token> out;
                out.assign(std::size_t(want), 0);
                for (int i = 0; i < want; ++i) {
                    out[std::size_t(i)] = toks[std::size_t(i % n)];
                }
                return out;
            };
            const int lenB = std::max(8, stress_len);
            const int lenC = lenB + 1;  // guaranteed not a multiple of 4 whenever lenB is
            printf("\n  (случаи ниже переваливают за 256 позиций: %d + %d и %d + %d)\n", lenB,
                   stress_gen, lenC, stress_gen);
            const CaseOut b = run_case("переход через 256 позиций", cycled(lenB), stress_gen,
                                       true);
            const CaseOut c = run_case("длина промпта не кратна 4", cycled(lenC), stress_gen,
                                       true);

            lv[4].ran = a.ran && b.ran && c.ran;
            lv[4].pass = a.pass && b.pass && c.pass;
            char key[256] = {0};
            snprintf(key, sizeof(key) - 1, "кэш: %s | 256: %s | не кратно 4: %s",
                     a.ran ? (a.pass ? "ПРОШЁЛ" : "ПРОВАЛ") : "не запущен",
                     b.ran ? (b.pass ? "ПРОШЁЛ" : "ПРОВАЛ") : "не запущен",
                     c.ran ? (c.pass ? "ПРОШЁЛ" : "ПРОВАЛ") : "не запущен");
            lv[4].key = key;
        }
    }

    // -----------------------------------------------------------------------------------
    // Level 5: determinism
    // -----------------------------------------------------------------------------------
    if (lv[5].selected) {
        printf("\n=== уровень 5: детерминизм ===\n");
        // Bit-identical, not "close". This is the cheapest level and it is the one most
        // likely to catch what the coming loader rewrite gets wrong: a buffer that is read
        // before it is written gives a different answer the second time round, and every
        // other level would report that as a small numerical difference and pass it.
        const int n_kv_max = pad32(n + 2);
        Cache kv1;
        Graph g1;
        bool ok = kv1.init(eng.buft, h, n_kv_max) &&
                  build_engine(&g1, eng.buft, h, w, &kv1, n, 0, n_kv_max, false);
        std::vector<float> a, b, c;
        int diff_same = -1, diff_fresh = -1, diff_dec = -1;
        if (ok) {
            eng.set_inputs(g1, toks.data(), n, 0, n_kv_max);
            eng.run(g1);
            ok = eng.logits_of(g1, &a);
        }
        if (ok) {
            // (a) the same graph, recomputed.
            eng.run(g1);
            ok = eng.logits_of(g1, &b);
            if (ok) {
                diff_same = 0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++diff_same;
                }
                printf("  тот же граф дважды: различающихся значений %d из %zu\n", diff_same,
                       a.size());
            }
        }
        Cache kv2;
        Graph g2;
        if (ok) {
            // (b) a freshly built graph over a freshly allocated cache. This is the arm that
            // sees an uninitialised buffer: a second allocation lands on different memory,
            // and whatever was there before is different too.
            ok = kv2.init(eng.buft, h, n_kv_max) &&
                 build_engine(&g2, eng.buft, h, w, &kv2, n, 0, n_kv_max, false);
            if (ok) {
                eng.set_inputs(g2, toks.data(), n, 0, n_kv_max);
                eng.run(g2);
                ok = eng.logits_of(g2, &c);
            }
            if (ok) {
                diff_fresh = 0;
                for (std::size_t i = 0; i < a.size(); ++i) {
                    if (std::memcmp(&a[i], &c[i], sizeof(float)) != 0) ++diff_fresh;
                }
                printf("  свежий граф и свежий кэш: различающихся значений %d из %zu\n",
                       diff_fresh, a.size());
            }
        }
        Graph dec;
        if (ok) {
            // (c) one decode step, run twice at the same position. Re-running a step is
            // idempotent - it rewrites the same key and value into the same slot - so any
            // difference is state that should not exist.
            ok = build_engine(&dec, eng.buft, h, w, &kv1, 1, n, n_kv_max, false);
            std::vector<float> d1, d2;
            const llama_token fed = toks[std::size_t(n - 1)];
            if (ok) ok = dec.aim_cache_writes(n);
            if (ok) {
                eng.set_inputs(dec, &fed, 1, n, n_kv_max);
                eng.run(dec);
                ok = eng.logits_of(dec, &d1);
            }
            if (ok) {
                eng.run(dec);
                ok = eng.logits_of(dec, &d2);
            }
            if (ok) {
                diff_dec = 0;
                for (std::size_t i = 0; i < d1.size(); ++i) {
                    if (std::memcmp(&d1[i], &d2[i], sizeof(float)) != 0) ++diff_dec;
                }
                printf("  шаг декода дважды: различающихся значений %d из %zu\n", diff_dec,
                       d1.size());
            }
        }
        // A zero-norm logit vector would make all three comparisons trivially "identical", so
        // it is checked rather than assumed - the same trap compare() exists to close.
        double norm_a = 0.0;
        for (float x : a) norm_a += double(x) * double(x);
        if (!(norm_a > 0.0)) {
            printf("  логиты нулевые — «побитово одинаково» здесь ничего не значит\n");
            ok = false;
        }
        lv[5].ran = ok && diff_same >= 0 && diff_fresh >= 0 && diff_dec >= 0;
        lv[5].pass = lv[5].ran && diff_same == 0 && diff_fresh == 0 && diff_dec == 0;
        char key[192] = {0};
        if (!lv[5].ran) {
            snprintf(key, sizeof(key) - 1, "прогон не завершился");
        } else {
            snprintf(key, sizeof(key) - 1,
                     "различий: тот же граф %d, свежий граф %d, шаг декода %d (|логиты| %.4f)",
                     diff_same, diff_fresh, diff_dec, std::sqrt(norm_a));
        }
        lv[5].key = key;
        printf("  итог: %s — %s\n", lv[5].pass ? "ПРОШЁЛ" : "ПРОВАЛ", key);
        dec.free_all();
        g2.free_all();
        g1.free_all();
        kv2.free_all();
        kv1.free_all();
    }

    // -----------------------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------------------
    printf("\n=== итоги ===\n");
    printf("%-8s %-12s %s\n", "уровень", "вердикт", "ключевое число");
    static const char* names[6] = {"", "1 тензоры", "2 логиты", "3 токены", "4 кэш",
                                   "5 детерм."};
    bool any_fail = false;
    bool any_selected = false;
    for (int i = 1; i <= 5; ++i) {
        if (!lv[i].selected) continue;
        any_selected = true;
        printf("%-8s %-12s %s\n", names[i], verdict_of(lv[i]),
               lv[i].key.empty() ? "-" : lv[i].key.c_str());
        if (!lv[i].ran || !lv[i].pass) any_fail = true;
    }
    if (!any_selected) {
        printf("ни один уровень не выбран\n");
        any_fail = true;
    }
    printf("\n%s\n", any_fail ? "ЕСТЬ ПРОВАЛЫ — код возврата 1"
                              : "все выбранные уровни прошли — код возврата 0");

    if (lctx) llama_free(lctx);
    ggml_backend_free(eng.be);
    llama_free_model(model);
    llama_backend_free();
    return any_fail ? 1 : 0;
}
