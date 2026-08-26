#include "resident_set.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace memex {

namespace {
// Written out rather than reached for as __builtin_ctzll / _BitScanForward64: this is two
// words per layer and the portable form costs nothing measurable, while the intrinsics differ
// between MSVC and gcc and this file is built by both.
int lowest_bit(uint64_t m) {
    int n = 0;
    while (!(m & 1ull)) { m >>= 1; ++n; }
    return n;
}
int bit_count(uint64_t m) {
    int n = 0;
    while (m) { m &= m - 1; ++n; }
    return n;
}
}  // namespace

// ---------------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------------

bool ResidentParams::validate(std::string* err) const {
    auto fail = [&](const char* what) {
        if (err) *err = what;
        return false;
    };
    if (n_layers <= 0) return fail("слоёв должно быть больше нуля");
    if (n_experts <= 0) return fail("экспертов должно быть больше нуля");
    // The window holds ids as int16_t: 128 experts per layer against a limit of 32767 is not
    // a constraint anybody will meet, but a silent wrap would corrupt the counts rather than
    // fail, so it is checked.
    if (n_experts > 32767) return fail("больше 32767 экспертов окно хранить не умеет");
    if (n_used <= 0 || n_used > n_experts) {
        return fail("выбираемых экспертов должно быть от одного до общего числа");
    }
    if (n_used > kMaxExpertsUsed) {
        return fail("выбирается больше экспертов, чем помещается в ResidentSplit "
                    "(kMaxExpertsUsed)");
    }
    if (capacity < 0 || capacity > n_experts) {
        return fail("ёмкость должна быть от нуля до общего числа экспертов");
    }
    if (window <= 0) return fail("окно должно быть положительным");
    if (period <= 0) return fail("период обновления должен быть положительным");
    if (budget < 0) return fail("бюджет подкачек не может быть отрицательным");
    return true;
}

ResidentStats ResidentStats::since(const ResidentStats& base) const {
    ResidentStats d;
    d.tokens       = tokens       - base.tokens;
    d.picks        = picks        - base.picks;
    d.hits         = hits         - base.hits;
    d.promotions   = promotions   - base.promotions;
    d.evictions    = evictions    - base.evictions;
    d.refreshes    = refreshes    - base.refreshes;
    d.budget_bound = budget_bound - base.budget_bound;
    d.activations  = activations  - base.activations;
    d.land_tokens  = land_tokens  - base.land_tokens;
    d.room_bound   = room_bound   - base.room_bound;
    return d;
}

// ---------------------------------------------------------------------------------------
// The set
// ---------------------------------------------------------------------------------------

ResidentSet::ResidentSet(const ResidentParams& p) : p_(p) {
    const std::size_t words = std::size_t((p_.n_experts + 63) / 64);
    layers_.resize(std::size_t(std::max(p_.n_layers, 0)));
    for (Layer& L : layers_) {
        L.bits.assign(words, 0ull);
        L.pend.assign(words, 0ull);
        L.ring.assign(std::size_t(p_.window) * std::size_t(p_.n_used), int16_t(-1));
        L.count.assign(std::size_t(p_.n_experts), 0);
        L.req_tok.assign(std::size_t(p_.n_experts), 0);
    }
    rank_.assign(std::size_t(p_.n_experts), 0);
    want_.assign(std::size_t(std::max(p_.capacity, 1)), 0);
    want_bits_.assign(words, 0ull);
    promote_.assign(std::size_t(std::max(p_.capacity, 1)), 0);
    victims_.assign(std::size_t(std::max(p_.capacity, 1)), 0);
    seen_.assign(std::size_t(p_.n_experts), 0u);
}

int ResidentSet::n_resident(int layer) const {
    return layers_[std::size_t(layer)].n_res;
}

int ResidentSet::n_pending(int layer) const {
    return layers_[std::size_t(layer)].n_pend;
}

void ResidentSet::activate(int layer, int expert) {
    if (layer < 0 || layer >= int(layers_.size())) return;
    if (expert < 0 || expert >= p_.n_experts) return;
    Layer& L = layers_[std::size_t(layer)];
    const unsigned w = unsigned(expert) >> 6, b = unsigned(expert) & 63;
    if (!((L.pend[w] >> b) & 1u)) return;   // not pending: already active, or never queued
    L.pend[w] &= ~(1ull << b);
    --L.n_pend;
    if (!((L.bits[w] >> b) & 1u)) {
        L.bits[w] |= 1ull << b;
        ++L.n_res;
    }
    ++st_.activations;
    // st_.tokens is the token counter end_token() advances, so this is in tokens, which is the
    // unit the refresh period is in and therefore the only unit that answers the question.
    const int64_t d = int64_t(st_.tokens) - int64_t(L.req_tok[std::size_t(expert)]);
    st_.land_tokens += uint64_t(d > 0 ? d : 0);
    ++revision_;
}

int ResidentSet::window_distinct(int layer) const {
    return layers_[std::size_t(layer)].distinct;
}

void ResidentSet::write_mask(int layer, float* dst) const {
    const uint64_t* b = bits_of(layer);
    std::memset(dst, 0, sizeof(float) * std::size_t(p_.n_experts));
    for (int e = 0; e < p_.n_experts; ++e) {
        if ((b[unsigned(e) >> 6] >> (unsigned(e) & 63)) & 1u) dst[e] = 1.0f;
    }
}

void ResidentSet::split(int layer, const int32_t* ids, const float* weights, int n,
                        ResidentSplit* out) const {
    out->n_resident = 0;
    out->n_other = 0;
    const int lim = std::min(n, kMaxExpertsUsed);
    for (int i = 0; i < lim; ++i) {
        ExpertPick pk;
        pk.slot = i;
        pk.id = int(ids[i]);
        pk.weight = weights ? weights[i] : 0.0f;
        // A slot the router did not fill belongs to neither half, but it has to land
        // somewhere or the two halves would stop covering every slot; it goes with the
        // non-resident ones, where its -1 makes mul_mat_id skip it exactly as it already
        // does on the unsplit path.
        if (pk.id >= 0 && pk.id < p_.n_experts && is_resident(layer, pk.id)) {
            out->resident[out->n_resident++] = pk;
        } else {
            out->other[out->n_other++] = pk;
        }
    }
}

void ResidentSet::push(Layer& L, int expert) {
    const int cap = int(L.ring.size());
    if (L.fill == cap) {
        const int old = int(L.ring[std::size_t(L.head)]);
        if (old >= 0 && --L.count[std::size_t(old)] == 0) --L.distinct;
    } else {
        ++L.fill;
    }
    L.ring[std::size_t(L.head)] = int16_t(expert);
    if (L.count[std::size_t(expert)]++ == 0) ++L.distinct;
    L.head = L.head + 1 == cap ? 0 : L.head + 1;
}

void ResidentSet::observe(int layer, const int32_t* ids, int n) {
    Layer& L = layers_[std::size_t(layer)];
    for (int i = 0; i < n; ++i) {
        const int e = int(ids[i]);
        if (e < 0 || e >= p_.n_experts) continue;
        ++st_.picks;
        if (is_resident(layer, e)) ++st_.hits;
        push(L, e);
    }
}

void ResidentSet::end_token() {
    ++st_.tokens;
    if (!on()) return;
    if (st_.tokens % uint64_t(p_.period) == 0) refresh();
}

// Frequency over the window, best first. Ties break on the lower id so two runs of the same
// prompt produce the same set - reproducibility is the only reason anything in this engine
// can be trusted, and a policy that broke ties on address or iteration order would make an
// A/B against a previous build meaningless.
int ResidentSet::choose_lfu(const Layer& L, int cap) {
    const int32_t* cnt = L.count.data();
    for (int e = 0; e < p_.n_experts; ++e) rank_[std::size_t(e)] = int16_t(e);
    auto better = [cnt](int16_t a, int16_t b) {
        const int32_t ca = cnt[a], cb = cnt[b];
        return ca != cb ? ca > cb : a < b;
    };
    std::partial_sort(rank_.begin(), rank_.begin() + cap, rank_.end(), better);
    for (int i = 0; i < cap; ++i) want_[std::size_t(i)] = rank_[std::size_t(i)];
    return cap;
}

// The most recently touched distinct experts, newest first. This is the control arm: the
// simulation found its hit rate indistinguishable from LFU's and its churn an order of
// magnitude worse, up to 208 MB of promotions per token against a link that carries 177 MB in
// a token's time. It is here so that claim stays reproducible.
int ResidentSet::choose_lru(const Layer& L, int cap) {
    ++stamp_;
    int n = 0;
    const int size = int(L.ring.size());
    for (int i = 0; i < L.fill && n < cap; ++i) {
        // Backwards from the newest entry, which sits one before head.
        int idx = L.head - 1 - i;
        while (idx < 0) idx += size;
        const int e = int(L.ring[std::size_t(idx)]);
        if (e < 0) continue;
        if (seen_[std::size_t(e)] == stamp_) continue;
        seen_[std::size_t(e)] = stamp_;
        want_[std::size_t(n++)] = int16_t(e);
    }
    return n;
}

void ResidentSet::refresh_layer(Layer& L) {
    // Never more than the window has actually seen. Filling the rest of the capacity with
    // experts of count zero would look like a warm set and would charge a promotion for every
    // one of them, so early promotions per token would be reported far above what the policy
    // really asks the link for.
    const int cap = std::min(p_.capacity, L.distinct);
    const int n_want = cap <= 0 ? 0 : (p_.lfu ? choose_lfu(L, cap) : choose_lru(L, cap));

    std::fill(want_bits_.begin(), want_bits_.end(), 0ull);
    for (int i = 0; i < n_want; ++i) {
        const unsigned e = unsigned(want_[std::size_t(i)]);
        want_bits_[e >> 6] |= 1ull << (e & 63);
    }

    if (deferred_) { refresh_layer_deferred(L, n_want); return; }

    // want \ resident, in the policy's own preference order, so a clip keeps the best.
    promote_.clear();
    for (int i = 0; i < n_want; ++i) {
        const unsigned e = unsigned(want_[std::size_t(i)]);
        if (!((L.bits[e >> 6] >> (e & 63)) & 1u)) promote_.push_back(int16_t(e));
    }
    int changed = int(promote_.size());
    const bool clipped = p_.budget > 0 && int(promote_.size()) > p_.budget;
    if (clipped) {
        promote_.resize(std::size_t(p_.budget));
        // Evict only as many as we promote, choosing the members the window likes least.
        // By count in both policies, which is what the simulation did: recency decides what
        // comes in under LRU, but the cheapest thing to throw out is still the least used.
        victims_.clear();
        const std::size_t words = L.bits.size();
        for (std::size_t wi = 0; wi < words; ++wi) {
            uint64_t m = L.bits[wi] & ~want_bits_[wi];
            while (m) {
                const int e = int(wi) * 64 + lowest_bit(m);
                m &= m - 1;
                victims_.push_back(int16_t(e));
            }
        }
        const int32_t* cnt = L.count.data();
        auto worse = [cnt](int16_t a, int16_t b) {
            const int32_t ca = cnt[a], cb = cnt[b];
            return ca != cb ? ca < cb : a < b;
        };
        const std::size_t take = std::min(victims_.size(), promote_.size());
        std::partial_sort(victims_.begin(), victims_.begin() + take, victims_.end(), worse);
        victims_.resize(take);
        for (int16_t v : victims_) {
            const unsigned e = unsigned(v);
            L.bits[e >> 6] &= ~(1ull << (e & 63));
            --L.n_res;
            ++st_.evictions;
        }
        for (int16_t nn : promote_) {
            const unsigned e = unsigned(nn);
            L.bits[e >> 6] |= 1ull << (e & 63);
            ++L.n_res;
        }
        changed = int(promote_.size()) + int(victims_.size());
        ++st_.budget_bound;
    } else {
        // The set becomes exactly what the window wants - which is the simulated policy to
        // the letter, including the case where `want` is a strict subset of the current set
        // and the refresh therefore only evicts. That case can only arise while the window
        // holds fewer distinct experts than the capacity, and keeping the surplus members
        // instead would be a small free improvement in hit rate. It is deliberately not
        // taken: the 67-80% this component is justified by was measured on `resident = want`,
        // and a policy that differs from the one that was measured is a policy with no number
        // behind it.
        int gone = 0;
        for (std::size_t wi = 0; wi < L.bits.size(); ++wi) {
            gone += bit_count(L.bits[wi] & ~want_bits_[wi]);
            L.bits[wi] = want_bits_[wi];
        }
        st_.evictions += uint64_t(gone);
        L.n_res = n_want;
        changed = int(promote_.size()) + gone;
    }
    st_.promotions += uint64_t(promote_.size());
    ++st_.refreshes;
    // Only when something moved. The engine rewrites the graph's residency mask off this
    // counter, so a refresh that changed nothing costs no upload.
    if (changed > 0) ++revision_;
}

// The same policy, with the promotion made asynchronous.
//
// Three rules, and each one exists to stop a specific failure:
//
//  1. A promotion sets the PENDING bit, never the resident one. Until the uploader confirms,
//     the expert is computed on the CPU exactly as a non-resident one is, so a layer can never
//     block on a transfer. That is the whole mechanism.
//
//  2. The capacity is charged against resident + pending, not resident. A pending expert
//     already owns its device slot - sync_slots maps it from is_claimed() - so counting only
//     the resident ones would let a refresh promise the device more slots than it has.
//     `room_bound` counts the refreshes where that ceiling actually bound.
//
//  3. Eviction takes victims from the RESIDENT set only; a pending transfer is protected until
//     it lands. The alternative - cancelling a promotion the window has changed its mind about
//     - would throw away bytes already crossing the link and free a slot whose upload is
//     already recorded into a command buffer, which is the one case where the slot map and the
//     device could disagree about what a slot holds. It costs at most a slot for a token or
//     two, and `tokens_to_land` is reported so that cost is visible rather than assumed.
//
// So a promotion still in flight when the next refresh runs is neither re-queued (it is
// already claimed, so it is not in want \ claimed) nor evicted. It simply stays pending.
void ResidentSet::refresh_layer_deferred(Layer& L, int n_want) {
    // Evict from the resident set only.
    int gone = 0;
    for (std::size_t wi = 0; wi < L.bits.size(); ++wi) {
        const uint64_t m = L.bits[wi] & ~want_bits_[wi];
        if (!m) continue;
        gone += bit_count(m);
        L.bits[wi] &= ~m;
    }
    L.n_res -= gone;
    st_.evictions += uint64_t(gone);

    // want \ (resident | pending), in the policy's own preference order so a clip keeps the
    // best - the same ordering rule the immediate path uses.
    promote_.clear();
    for (int i = 0; i < n_want; ++i) {
        const unsigned e = unsigned(want_[std::size_t(i)]);
        const uint64_t claimed = L.bits[e >> 6] | L.pend[e >> 6];
        if (!((claimed >> (e & 63)) & 1u)) promote_.push_back(int16_t(e));
    }
    int lim = int(promote_.size());
    if (p_.budget > 0 && lim > p_.budget) { lim = p_.budget; ++st_.budget_bound; }
    const int room = p_.capacity - (L.n_res + L.n_pend);
    if (lim > room) { lim = room > 0 ? room : 0; ++st_.room_bound; }

    for (int i = 0; i < lim; ++i) {
        const unsigned e = unsigned(promote_[std::size_t(i)]);
        L.pend[e >> 6] |= 1ull << (e & 63);
        ++L.n_pend;
        L.req_tok[std::size_t(e)] = int32_t(st_.tokens);
    }
    st_.promotions += uint64_t(lim);
    ++st_.refreshes;
    // An eviction changes the mask; a promotion changes only the slot map, but the engine
    // drives BOTH off this counter, so a refresh that only promoted still has to move it or
    // the uploads would never be queued.
    if (gone > 0 || lim > 0) ++revision_;
}

void ResidentSet::refresh() {
    for (Layer& L : layers_) refresh_layer(L);
}

// ---------------------------------------------------------------------------------------
// The graph side
// ---------------------------------------------------------------------------------------

namespace {

// Shared body. `flags` is [1, n_used, n_tokens] f32 and `ids` is [n_used, n_tokens] i32 with
// the argsort row stride ggml_top_k leaves behind, so both are addressed through nb rather
// than indexed as arrays. dst comes from ggml_dup_tensor and is contiguous.
void mask_ids(ggml_tensor* dst, const ggml_tensor* ids, const ggml_tensor* flags,
              int ith, bool want_resident) {
    if (ith != 0) return;   // eight integers a row; threading it would cost more than it saves
    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
        const char* src = (const char*)ids->data + i1 * ids->nb[1];
        const char* flg = (const char*)flags->data + i1 * flags->nb[2];
        int32_t* out = (int32_t*)((char*)dst->data + i1 * dst->nb[1]);
        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
            const int32_t id = *(const int32_t*)(src + i0 * ids->nb[0]);
            const float f = *(const float*)(flg + i0 * flags->nb[1]);
            const bool res = f > 0.5f;
            out[i0] = (res == want_resident) ? id : -1;
        }
    }
}

void mask_ids_resident(ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b,
                       int ith, int /*nth*/, void* /*userdata*/) {
    mask_ids(dst, a, b, ith, true);
}

void mask_ids_other(ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b,
                    int ith, int /*nth*/, void* /*userdata*/) {
    mask_ids(dst, a, b, ith, false);
}

void copy_ids(ggml_tensor* dst, const ggml_tensor* a, int ith, int /*nth*/,
              void* /*userdata*/) {
    if (ith != 0) return;
    for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
        const char* src = (const char*)a->data + i1 * a->nb[1];
        int32_t* out = (int32_t*)((char*)dst->data + i1 * dst->nb[1]);
        for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
            out[i0] = *(const int32_t*)(src + i0 * a->nb[0]);
        }
    }
}

}  // namespace

ggml_tensor* resident_split_ids(ggml_context* ctx, ggml_tensor* ids, ggml_tensor* flags,
                               bool resident) {
    return ggml_map_custom2(ctx, ids, flags,
                            resident ? mask_ids_resident : mask_ids_other,
                            /*n_tasks=*/1, /*userdata=*/nullptr);
}

ggml_tensor* resident_ids_copy(ggml_context* ctx, ggml_tensor* ids) {
    return ggml_map_custom1(ctx, ids, copy_ids, /*n_tasks=*/1, /*userdata=*/nullptr);
}

}  // namespace memex
