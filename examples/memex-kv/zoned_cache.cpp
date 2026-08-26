#include "zoned_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "ggml-backend.h"

namespace memex {

// Round to nearest even, with subnormals, and both halves exact inverses of each other on
// the values half precision can hold.
//
// This was truncation and an underflow-to-zero, and both were wrong in ways that only a
// real model showed. Truncating the mantissa instead of rounding it is a *biased* error of
// up to one ulp - 4.9e-4 relative - always toward zero, so it does not cancel over a dot
// product of five hundred terms, it accumulates. And underflow-to-zero threw away every
// component below 6.1e-5, which sounds negligible until you count them: 188 of the 10240
// values in one layer of a real Qwen3 cache are fp16 subnormals. Together they put 0.045%
// of error into an attention output that was supposed to be bit-identical, and that read as
// "this is what the zoned cache costs" when it was in fact the cost of this function.
//
// The lesson worth keeping: the arm of a comparison that is supposed to be exact has to be
// checked *for exactness*, not for being small. An f16 control that returns 0.045% instead
// of 1e-7 is a failure, and it is the only reason this was found.
uint16_t zc_f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint16_t s16 = uint16_t((x >> 31) << 15);
    const int32_t exp32 = int32_t((x >> 23) & 0xFF);
    const uint32_t man32 = x & 0x7FFFFF;
    if (exp32 == 0xFF) {
        // A NaN has to stay a NaN. Dropping the payload would turn it into an infinity,
        // which compares and propagates quite differently.
        return uint16_t(s16 | 0x7C00 | (man32 ? 0x0200 : 0));
    }
    int32_t e = exp32 - 127 + 15;
    if (e >= 31) return uint16_t(s16 | 0x7C00);        // overflows half precision
    if (e <= 0) {
        // Half-precision subnormal: value = man16 * 2^-24, so the significand with its
        // implicit leading one shifts down by 14 - e. Below -10 there is nothing left even
        // to round up to.
        if (e < -10) return s16;
        const uint32_t m = man32 | 0x800000u;
        const int shift = 14 - e;
        uint32_t half = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) half++;
        return uint16_t(s16 | half);                   // may carry into the exponent, correctly
    }
    uint32_t man16 = man32 >> 13;
    const uint32_t rem = man32 & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (man16 & 1u))) {
        man16++;
        if (man16 == 0x400u) {                         // the mantissa carried out
            man16 = 0;
            if (++e >= 31) return uint16_t(s16 | 0x7C00);
        }
    }
    return uint16_t(s16 | (uint32_t(e) << 10) | man16);
}

float zc_h2f(uint16_t h) {
    const uint32_t sign = uint32_t(h >> 15) << 31;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t out;
    if (e == 0) {
        if (man == 0) {
            out = sign;
        } else {
            // A half-precision subnormal is a normal single, so it is renormalised rather
            // than flushed: value = man * 2^-24, and 113 is the exponent field of 2^-14.
            e = 113;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                e--;
            }
            man &= 0x3FFu;
            out = sign | (e << 23) | (man << 13);
        }
    } else if (e == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((e + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

bool ZonedCacheParams::validate(std::string* err) const {
    auto fail = [&](const char* m) {
        if (err) *err = m;
        return false;
    };
    if (n_layers <= 0) return fail("n_layers must be positive");
    if (n_kv_heads <= 0 || n_q_heads <= 0) return fail("head counts must be positive");
    // Grouped-query attention broadcasts kv-heads across query heads inside
    // ggml_mul_mat, and it can only do that when the ratio is exact. A non-integer ratio
    // would not error there; it would silently pair the wrong head with the wrong key.
    if (n_q_heads % n_kv_heads != 0) {
        return fail("n_q_heads must be a whole multiple of n_kv_heads");
    }
    if (head_dim <= 0 || head_dim % kZonePad != 0) {
        return fail("head_dim must be a positive multiple of 32 (Q8_0 blocks, and the "
                    "F16 matmul is wrong when the reduction is not a multiple of 4)");
    }
    if (tail_q8 && rotate_keys && (head_dim & (head_dim - 1)) != 0) {
        return fail("rotate_keys needs head_dim to be a power of two (Sylvester)");
    }
    if (n_sinks < 0 || notebook_cap < 0 || window <= 0) {
        return fail("zone sizes must be non-negative and the window positive");
    }
    if (n_ctx <= 0) return fail("n_ctx must be positive");
    if (exact_slots() > n_ctx) {
        // Not fatal in principle, but it means the tail can never be reached, so every
        // number the example prints about compression would be about nothing.
        return fail("sinks + notebook + window exceeds n_ctx: nothing would be "
                    "compressed and the measurement would be empty");
    }
    return true;
}

ZonedCache::ZonedCache(const ZonedCacheParams& p) : p_(p) {
    ex_pad_ = zone_pad_up(p_.exact_slots());
    // The tail is allocated for the whole context: every position that is not exact can
    // end up here, and a Q8_0 value row cannot be grown because a block of 32 spans 32
    // positions, so growing it later would move every row. llama.cpp preallocates its
    // cache and masks the remainder for the same reason.
    tl_pad_ = zone_pad_up(p_.n_ctx);
    const int d = p_.d_kv();
    layers_.resize(size_t(p_.n_layers));
    for (Layer& L : layers_) {
        L.ex_k.assign(size_t(ex_pad_) * size_t(d), 0);
        L.ex_v.assign(size_t(ex_pad_) * size_t(d), 0);
        L.mask.assign(size_t(ex_pad_ + tl_pad_), -INFINITY);
        L.slot_used.assign(size_t(ex_pad_), 0);
        L.note_free.resize(size_t(p_.notebook_cap));
        for (int i = 0; i < p_.notebook_cap; ++i) L.note_free[size_t(i)] = i;
        std::reverse(L.note_free.begin(), L.note_free.end());   // fill low slots first
        if (p_.tail_q8) {
            KvConfig kc;
            kc.n_kv_heads = p_.n_kv_heads;
            kc.head_dim   = p_.head_dim;
            kc.tail_form  = TailForm::Q8_0;
            // Driven as a pure tail encoder: this module owns the zone map, KvZones owns
            // the Q8_0 bytes and the rotation. With no sinks, no window and no notebook
            // of its own, every position handed to it is compressed at once.
            kc.n_sinks = 0;
            kc.window = 0;
            kc.notebook_cap = 0;
            kc.elastic_notebook = false;
            kc.rotate_tail_keys = p_.rotate_keys;
            // Exactly tl_pad_, so the buffers KvZones allocates are the tensors' own
            // size and upload is a memcpy rather than a per-row compaction.
            kc.tail_cap = tl_pad_;
            L.q8.reset(new KvZones(kc));
        } else {
            L.tl_k.assign(size_t(tl_pad_) * size_t(d), 0);
            L.tl_v.assign(size_t(tl_pad_) * size_t(d), 0);
        }
    }
    // resize, not a constructor argument: `std::vector<uint16_t> row(size_t(d));` is the
    // most vexing parse and compiles as a function declaration.
    row_k_.resize(size_t(d));
    row_v_.resize(size_t(d));
    ev_k_.resize(size_t(d));
    ev_v_.resize(size_t(d));
}

// Keys as [head_dim, slot, kv_head], values as [slot, head_dim, kv_head]. Value blocks
// run along positions and key blocks along head_dim; getting that backwards makes the
// quantisation grain wrong, and in F16 it makes the reduction axis wrong, which the
// matmul will happily compute anyway.
void ZonedCache::write_exact(Layer& L, int slot, const uint16_t* k, const uint16_t* v,
                             bool mark) {
    const int hd = p_.head_dim;
    // Null rows are the live protocol: the engine's graph is about to store this slot
    // itself, so only the bookkeeping happens here.
    if (k && v) {
        for (int g = 0; g < p_.n_kv_heads; ++g) {
            const size_t src = size_t(g) * size_t(hd);
            uint16_t* kd = L.ex_k.data() + size_t(g) * size_t(ex_pad_) * size_t(hd) +
                           size_t(slot) * size_t(hd);
            for (int j = 0; j < hd; ++j) {
                kd[j] = k[src + size_t(j)];
                L.ex_v[size_t(g) * size_t(hd) * size_t(ex_pad_) +
                       size_t(j) * size_t(ex_pad_) + size_t(slot)] = v[src + size_t(j)];
            }
        }
    }
    if (mark) {
        L.slot_used[size_t(slot)] = 1;
        L.mask[size_t(slot)] = 0.0f;
    }
}

// The inverse gather, used once per eviction. The split layout is right for the graph and
// wrong for moving a single position between zones; paying an O(d_kv) gather here is
// cheaper than keeping a second copy of the window, which is the largest exact zone.
void ZonedCache::read_exact(const Layer& L, int slot, uint16_t* k, uint16_t* v) const {
    const int hd = p_.head_dim;
    for (int g = 0; g < p_.n_kv_heads; ++g) {
        const uint16_t* ks = L.ex_k.data() + size_t(g) * size_t(ex_pad_) * size_t(hd) +
                             size_t(slot) * size_t(hd);
        for (int j = 0; j < hd; ++j) {
            k[size_t(g) * size_t(hd) + size_t(j)] = ks[j];
            v[size_t(g) * size_t(hd) + size_t(j)] =
                L.ex_v[size_t(g) * size_t(hd) * size_t(ex_pad_) +
                       size_t(j) * size_t(ex_pad_) + size_t(slot)];
        }
    }
}

void ZonedCache::write_tail_f16(Layer& L, int slot, const uint16_t* k,
                                const uint16_t* v) {
    const int hd = p_.head_dim;
    for (int g = 0; g < p_.n_kv_heads; ++g) {
        const size_t src = size_t(g) * size_t(hd);
        uint16_t* kd = L.tl_k.data() + size_t(g) * size_t(tl_pad_) * size_t(hd) +
                       size_t(slot) * size_t(hd);
        for (int j = 0; j < hd; ++j) {
            kd[j] = k[src + size_t(j)];
            L.tl_v[size_t(g) * size_t(hd) * size_t(tl_pad_) +
                   size_t(j) * size_t(tl_pad_) + size_t(slot)] = v[src + size_t(j)];
        }
    }
    L.mask[size_t(ex_pad_ + slot)] = 0.0f;
}

void ZonedCache::append(int il, const float* k, const float* v, bool keep_exact) {
    const int d = p_.d_kv();
    for (int j = 0; j < d; ++j) {
        row_k_[size_t(j)] = zc_f2h(k[j]);
        row_v_[size_t(j)] = zc_f2h(v[j]);
    }
    append_row(il, row_k_.data(), row_v_.data(), keep_exact, nullptr);
}

int ZonedCache::append_live(int il, bool keep_exact, int* note_slot) {
    return append_row(il, nullptr, nullptr, keep_exact, note_slot);
}

void ZonedCache::set_live_row(int il, int slot, const float* k, const float* v) {
    Layer& L = layers_[size_t(il)];
    if (slot < 0 || slot >= ex_pad_ || !k || !v) return;
    const int d = p_.d_kv();
    for (int j = 0; j < d; ++j) {
        row_k_[size_t(j)] = zc_f2h(k[j]);
        row_v_[size_t(j)] = zc_f2h(v[j]);
    }
    // Data only. append_live already marked the slot, and marking it again would drop a
    // keep flag the engine asked for on this very position.
    write_exact(L, slot, row_k_.data(), row_v_.data(), /*mark=*/false);
}

ggml_tensor* ZonedCache::exact_k(int il) const {
    return layers_[size_t(il)].K;
}

ggml_tensor* ZonedCache::exact_v(int il) const {
    return layers_[size_t(il)].V;
}

int ZonedCache::append_row(int il, const uint16_t* k, const uint16_t* v, bool keep_exact,
                           int* note_slot) {
    Layer& L = layers_[size_t(il)];
    if (note_slot) *note_slot = -1;
    if (L.n_pos >= p_.n_ctx) {
        // Refuse rather than overrun. A cache that silently forgets the past looks
        // exactly like one that works, so the loss is counted and reported.
        L.dropped++;
        return -1;
    }
    L.n_pos++;

    if (L.n_sink < p_.n_sinks) {
        const int ss = sink_slot(L.n_sink);
        write_exact(L, ss, k, v);
        L.n_sink++;
        return ss;
    }

    // The window is a ring: the slot about to be reused holds the position that is
    // leaving, so the eviction is read-then-write with no shuffling of anything else.
    const int slot = win_slot(L.win_head);
    const bool evicting = L.slot_used[size_t(slot)] != 0;
    if (evicting) {
        read_exact(L, slot, ev_k_.data(), ev_v_.data());
    }
    // Whether the leaving position was flagged. Tracked per window slot rather than per
    // position so that no position->slot map has to be maintained.
    const bool ev_keep = evicting && L.slot_used[size_t(slot)] == 2;
    write_exact(L, slot, k, v);
    L.slot_used[size_t(slot)] = keep_exact ? 2 : 1;
    L.win_head = (L.win_head + 1) % p_.window;
    if (!evicting) {
        L.n_win++;
        return slot;
    }

    if (ev_keep && !L.note_free.empty()) {
        const int ns = note_slot_of(L.note_free.back());
        L.note_free.pop_back();
        write_exact(L, ns, ev_k_.data(), ev_v_.data());
        L.n_note++;
        if (note_slot) *note_slot = ns;
        return slot;
    }
    // Not flagged, or the notebook is full: compress. The notebook filling up is an
    // ordinary outcome, not an error - the position degrades rather than disappearing.
    if (p_.tail_q8) {
        if (L.q8->tail_positions() + 1 > tl_pad_) {
            L.dropped++;               // honest loss; the mask will not admit it
            return slot;
        }
        // The importance argument is unused by a KvZones configured as a pure tail, and
        // is passed as zero to say so rather than to imply a policy that is not running.
        L.q8->append(ev_k_.data(), ev_v_.data(), 0.0f);
    } else {
        if (L.n_tail >= tl_pad_) {
            L.dropped++;
            return slot;
        }
        write_tail_f16(L, L.n_tail, ev_k_.data(), ev_v_.data());
        L.n_tail++;
    }
    return slot;
}

void ZonedCache::flush() {
    for (Layer& L : layers_) {
        if (!p_.tail_q8) continue;
        L.q8->flush();
        const int n = L.q8->tail_positions();
        for (int i = L.n_tail; i < n; ++i) L.mask[size_t(ex_pad_ + i)] = 0.0f;
        L.n_tail = n;
    }
}

bool ZonedCache::create_tensors(ggml_context* ctx, std::string* err) {
    std::string verr;
    if (!p_.validate(&verr)) {
        if (err) *err = verr;
        return false;
    }
    const int hd = p_.head_dim;
    const ggml_type tt = p_.tail_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    int il = 0;
    for (Layer& L : layers_) {
        // Three dimensions, heads last, is not a style choice. Expressing grouped
        // attention as one two-dimensional strided slice per head group produced an
        // answer wrong by nine orders of magnitude in memex-attn, with the first head of
        // each group behaving differently from the other three. ggml_mul_mat broadcasts
        // kv-heads across query heads only in this form.
        L.K  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hd, ex_pad_, p_.n_kv_heads);
        L.V  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, ex_pad_, hd, p_.n_kv_heads);
        L.Tk = ggml_new_tensor_3d(ctx, tt, hd, tl_pad_, p_.n_kv_heads);
        L.Tv = ggml_new_tensor_3d(ctx, tt, tl_pad_, hd, p_.n_kv_heads);
        L.M  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ex_pad_ + tl_pad_, 1);
        L.R  = nullptr;
        if (p_.tail_q8 && p_.rotate_keys && L.q8->key_rotation()) {
            L.R = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, hd);
        }
        if (!L.K || !L.V || !L.Tk || !L.Tv || !L.M) {
            if (err) *err = "ggml context ran out of tensor slots";
            return false;
        }
        char nm[64];
        snprintf(nm, sizeof(nm), "zkv-%d-exact-k", il);  ggml_set_name(L.K, nm);
        snprintf(nm, sizeof(nm), "zkv-%d-exact-v", il);  ggml_set_name(L.V, nm);
        snprintf(nm, sizeof(nm), "zkv-%d-tail-k", il);   ggml_set_name(L.Tk, nm);
        snprintf(nm, sizeof(nm), "zkv-%d-tail-v", il);   ggml_set_name(L.Tv, nm);
        snprintf(nm, sizeof(nm), "zkv-%d-mask", il);     ggml_set_name(L.M, nm);
        if (L.R) {
            snprintf(nm, sizeof(nm), "zkv-%d-rot", il);  ggml_set_name(L.R, nm);
        }
        il++;
    }
    return true;
}

bool ZonedCache::upload_dynamic(int il, std::string* err) {
    return upload_impl(il, err, /*exact=*/false);
}

// One slot out of the two exact-zone layouts. Keys are one contiguous head_dim run per
// kv-head; values are one element per head_dim per kv-head, because a value row runs along
// slots. The value side is therefore d_kv single-element writes - which is why this is for
// notebook admissions, of which a run has at most notebook_cap, and not for the window.
bool ZonedCache::upload_slot(int il, int slot, std::string* err) {
    Layer& L = layers_[size_t(il)];
    if (!L.K || !L.V || !L.K->buffer || !L.V->buffer) {
        if (err) *err = "upload_slot before the exact zones were allocated";
        return false;
    }
    if (slot < 0 || slot >= ex_pad_) {
        if (err) *err = "upload_slot given a slot outside the exact zones";
        return false;
    }
    const int hd = p_.head_dim;
    const size_t es = sizeof(uint16_t);
    for (int g = 0; g < p_.n_kv_heads; ++g) {
        const uint16_t* ks = L.ex_k.data() + size_t(g) * size_t(ex_pad_) * size_t(hd) +
                             size_t(slot) * size_t(hd);
        ggml_backend_tensor_set(L.K, ks,
                                size_t(g) * size_t(L.K->nb[2]) + size_t(slot) * size_t(L.K->nb[1]),
                                size_t(hd) * es);
        for (int j = 0; j < hd; ++j) {
            const uint16_t* vs = L.ex_v.data() + size_t(g) * size_t(hd) * size_t(ex_pad_) +
                                 size_t(j) * size_t(ex_pad_) + size_t(slot);
            ggml_backend_tensor_set(L.V, vs,
                                    size_t(g) * size_t(L.V->nb[2]) +
                                        size_t(j) * size_t(L.V->nb[1]) + size_t(slot) * es,
                                    es);
        }
    }
    return true;
}

bool ZonedCache::upload(int il, std::string* err) {
    return upload_impl(il, err, /*exact=*/true);
}

bool ZonedCache::upload_impl(int il, std::string* err, bool exact) {
    Layer& L = layers_[size_t(il)];
    ggml_tensor* need[5] = {L.K, L.V, L.Tk, L.Tv, L.M};
    for (ggml_tensor* t : need) {
        if (!t) {
            if (err) *err = "upload before create_tensors";
            return false;
        }
        if (!t->buffer || !t->data) {
            // The failure this whole example is shaped around. A tensor with no buffer
            // swallows the write and reads back as zeros, and a comparison against zeros
            // reports a perfect match.
            if (err) {
                *err = std::string("tensor '") + ggml_get_name(t) +
                       "' has no backend buffer: allocate the context before upload";
            }
            return false;
        }
    }
    if (exact) {
        ggml_backend_tensor_set(L.K, L.ex_k.data(), 0, ggml_nbytes(L.K));
        ggml_backend_tensor_set(L.V, L.ex_v.data(), 0, ggml_nbytes(L.V));
    }
    if (p_.tail_q8) {
        // tail_cap was set to tl_pad_, so KvZones' buffers are exactly the tensors' size
        // and this is a straight copy. Keys are [q8_row(head_dim), cap, kv_head], values
        // [q8_row(cap), head_dim, kv_head] - the layouts the tensors were declared with.
        const size_t nk = size_t(ggml_nbytes(L.Tk));
        const size_t nv = size_t(ggml_nbytes(L.Tv));
        const size_t have_k = size_t(p_.n_kv_heads) * size_t(tl_pad_) *
                              size_t(KvZones::q8_row_bytes(p_.head_dim));
        const size_t have_v = size_t(p_.n_kv_heads) * size_t(p_.head_dim) *
                              size_t(KvZones::q8_row_bytes(tl_pad_));
        if (have_k != nk || have_v != nv) {
            if (err) *err = "Q8_0 tail buffer size disagrees with the tensor";
            return false;
        }
        ggml_backend_tensor_set(L.Tk, L.q8->tail_k_q(), 0, nk);
        ggml_backend_tensor_set(L.Tv, L.q8->tail_v_q(), 0, nv);
    } else {
        ggml_backend_tensor_set(L.Tk, L.tl_k.data(), 0, ggml_nbytes(L.Tk));
        ggml_backend_tensor_set(L.Tv, L.tl_v.data(), 0, ggml_nbytes(L.Tv));
    }
    ggml_backend_tensor_set(L.M, L.mask.data(), 0, ggml_nbytes(L.M));
    if (L.R) {
        if (!L.R->buffer) {
            if (err) *err = "rotation tensor has no backend buffer";
            return false;
        }
        ggml_backend_tensor_set(L.R, L.q8->key_rotation(), 0, ggml_nbytes(L.R));
    }
    return true;
}

ZonedAttn ZonedCache::build_attn(ggml_context* ctx, int il, ggml_tensor* q) {
    Layer& L = layers_[size_t(il)];
    ZonedAttn a;
    if (!L.K || !q) return a;

    const float scale = 1.0f / std::sqrt(float(p_.head_dim));
    // The graph shape does not depend on how full the cache is: every zone is present at
    // its allocated size and occupancy lives entirely in the mask. That costs some
    // arithmetic on empty slots and buys an engine one graph it can reuse for every
    // token, which is the trade llama.cpp also makes.
    ggml_tensor* sc_ex = ggml_mul_mat(ctx, L.K, q);      // [ex_pad, 1, n_q_heads]

    // The tail keys were rotated before quantisation, so they must be scored against the
    // rotated query. R is stored row-major and ggml_mul_mat pairs rows with rows, so
    // mul_mat(R, q) is exactly Rq. Values were not rotated - measured worthless there,
    // because value blocks run along positions - so nothing is rotated back afterwards.
    ggml_tensor* q_tail = L.R ? ggml_mul_mat(ctx, L.R, q) : q;
    ggml_tensor* sc_tl = ggml_mul_mat(ctx, L.Tk, q_tail); // [tl_pad, 1, n_q_heads]

    a.scores = ggml_concat(ctx, sc_ex, sc_tl, 0);
    // ONE softmax. This single call is the correctness claim of the whole module: the
    // exact zones and the compressed tail compete for the same probability mass. Two
    // calls here - one per zone - would compute a different function that still looks
    // like attention. ZonedAttn hands `probs` back so a test can sum it and check that
    // it is one distribution rather than four.
    a.probs = ggml_soft_max_ext(ctx, a.scores, L.M, scale, 0.0f);
    a.n_slots = ex_pad_ + tl_pad_;
    a.exact_slots = ex_pad_;

    // The slices are copied because a view strided along dimension 0 cannot be a
    // reduction operand, and one pass over the score array per zone is what the
    // concatenated layout costs.
    ggml_tensor* p_ex = ggml_cont(ctx,
        ggml_view_3d(ctx, a.probs, ex_pad_, 1, p_.n_q_heads,
                     a.probs->nb[1], a.probs->nb[2], 0));
    ggml_tensor* p_tl = ggml_cont(ctx,
        ggml_view_3d(ctx, a.probs, tl_pad_, 1, p_.n_q_heads,
                     a.probs->nb[1], a.probs->nb[2], size_t(ex_pad_) * sizeof(float)));
    a.out_exact = ggml_mul_mat(ctx, L.V, p_ex);          // [head_dim, 1, n_q_heads]
    a.out_tail  = ggml_mul_mat(ctx, L.Tv, p_tl);
    a.out = ggml_add(ctx, a.out_exact, a.out_tail);
    return a;
}

ZoneOccupancy ZonedCache::occupancy(int il) const {
    const Layer& L = layers_[size_t(il)];
    ZoneOccupancy o;
    o.sinks = L.n_sink;
    o.notebook = L.n_note;
    o.window = L.n_win;
    o.tail = L.n_tail;
    o.dropped = L.dropped;
    return o;
}

std::size_t ZonedCache::read_bytes(int il) const {
    const Layer& L = layers_[size_t(il)];
    const std::size_t d = std::size_t(p_.d_kv());
    const std::size_t exact = std::size_t(L.n_sink + L.n_note + L.n_win) * d * 2 * 2;
    const std::size_t tail = p_.tail_q8
        ? std::size_t(L.n_tail) * std::size_t(ggml_row_size(GGML_TYPE_Q8_0, int64_t(d))) * 2
        : std::size_t(L.n_tail) * d * 2 * 2;
    return exact + tail;
}

std::size_t ZonedCache::read_bytes_if_exact(int il) const {
    const Layer& L = layers_[size_t(il)];
    return std::size_t(L.n_sink + L.n_note + L.n_win + L.n_tail) *
           std::size_t(p_.d_kv()) * 2 * 2;
}

}  // namespace memex
