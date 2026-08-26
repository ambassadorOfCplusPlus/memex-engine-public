// A live set of experts held in video memory, and the policy that keeps it live.
//
// The design. Each layer selects 8 of 128 experts for the token. If some subset of the 128
// is already resident in video memory, the GPU can compute the selected experts that are
// resident while the CPU computes the ones that are not, and the two partial results are
// summed at the end of the layer. The win is exactly proportional to the hit rate: generation
// on this machine is 100% memory-bound, so a selection served from video memory is a
// selection whose bytes never cross the memory bus at all.
//
// What is here and what is not. This is the bookkeeping and the policy: the resident set, the
// per-token split of the 8 selections into resident and non-resident, the refresh, and the
// accounting. The GPU execution is a separate later step. That split of work is deliberate -
// this half is verifiable without any Vulkan code existing, because with the feature on and
// no GPU path the engine computes *both* halves on the CPU and sums them, which must produce
// the same tokens as not splitting at all. See resident_split_ids() below.
//
// The policy is not guesswork. examples/../vram_residency.py simulated it over real router
// traces from this model and settled four things:
//
//   - 29 resident experts per layer out of 128 - which is what 3.6 GB of usable video memory
//     holds at 2.5 MB per four-bit expert - gives a 67-80% hit rate depending on the text.
//   - LFU over a sliding window beats LRU decisively, and *not* on hit rate, which is the
//     same for both. It beats it on churn. Unbounded LRU wanted up to 208 MB of promotions
//     per token against a PCIe link that carries 177 MB in a token's time; LFU wanted 19-50
//     MB. So the policy has to be frequency-over-a-window, and `lru` exists here only as the
//     control that established that.
//   - A window of 64 tokens and a refresh every 3 tokens produced those numbers.
//   - A promotion budget of 8 per refresh almost never binds for LFU - 624 promotions against
//     626 unbounded - so it is in as a cheap safety rail against a pathological stretch of
//     input, not as a tuning knob. `budget_bound` in the stats counts the refreshes where it
//     actually clipped, so the claim stays checkable rather than inherited.
//
// And one thing that is the reason the set is online rather than precomputed: after the input
// switches language the hit rate recovers from 20% to 66% within 100 tokens. The earlier,
// closed, *static* version of this idea died precisely here - a set frozen from a calibration
// run carried no information off-distribution (24.1% of selections against 25.0% for a random
// set). Adaptation is not a refinement of the idea, it is the idea.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace memex {

// Hard cap on the router's top-k. This model uses 8. The cap exists so ResidentSplit is a
// plain struct that lives on the stack: the split runs 48 times per token and must not
// allocate.
constexpr int kMaxExpertsUsed = 32;

struct ResidentParams {
    int n_layers  = 0;
    int n_experts = 0;      // 128 for Qwen3-Coder-30B-A3B
    int n_used    = 0;      // 8; no shared/always-on experts in this model
    // Resident experts per layer. 0 means the whole mechanism is off, which is the default:
    // it changes the shape of the compute graph, so it stays behind a flag until the GPU half
    // exists, exactly as the zoned KV cache does.
    int capacity  = 0;
    int window    = 64;     // tokens of selection history the policy ranks over
    int period    = 3;      // refresh the set every N tokens
    int budget    = 8;      // most promotions one refresh may ask for; 0 = unbounded
    bool lfu      = true;   // false selects the LRU control

    // Refuses rather than repairs. Each of these has a way of becoming a wrong answer or a
    // silently disabled optimisation instead of an error further down.
    bool validate(std::string* err) const;
};

// One of the router's picks, and - the load-bearing field - the slot it occupied in the
// router's top-k output.
//
// Why the slot has to be carried. This engine applies the routing weights *after* the expert
// outputs are gathered: build_step computes out = mul_mat_id(down, act, ids) and only then
// out = ggml_mul(out, weights), where `weights` is indexed by slot, not by expert id. So a
// half that has lost track of which slot a pick came from cannot be multiplied by the right
// weight, and the two halves cannot be summed. Getting that order wrong - applying a weight
// to a half before summing, or renormalising the weights within a half so each half sums to
// one - produces a plausible-looking wrong answer with no shape error anywhere.
struct ExpertPick {
    int   slot   = 0;      // 0..n_used-1, the router's own ordering (descending weight)
    int   id     = -1;     // expert id, or -1 for a slot the router did not fill
    float weight = 0.0f;   // carried for the caller's benefit; NOT applied here
};

// The per-token split of one layer's selections. Both halves keep their slots, so the sum of
// the two partial results is the unsplit result.
struct ResidentSplit {
    ExpertPick resident[kMaxExpertsUsed];
    ExpertPick other[kMaxExpertsUsed];
    int n_resident = 0;
    int n_other    = 0;
    int n() const { return n_resident + n_other; }
};

struct ResidentStats {
    uint64_t tokens       = 0;   // tokens whose selections were observed
    uint64_t picks        = 0;   // expert selections seen: tokens * n_layers * n_used
    uint64_t hits         = 0;   // of those, the ones already resident when they were made
    uint64_t promotions   = 0;   // experts moved into the set (one PCIe transfer each)
    uint64_t evictions    = 0;
    uint64_t refreshes    = 0;
    uint64_t budget_bound = 0;   // refreshes where the promotion budget actually clipped
    // Deferred activation. `promotions` above is what the policy ASKED for; these two are what
    // actually happened, and the gap between them is the whole point of the mechanism.
    uint64_t activations  = 0;   // promotions the uploader confirmed had landed
    uint64_t land_tokens  = 0;   // sum over activations of (token landed - token requested)
    uint64_t room_bound   = 0;   // refreshes where pending transfers held every free slot

    double hit_rate() const {
        return picks ? double(hits) / double(picks) : 0.0;
    }
    double promotions_per_token() const {
        return tokens ? double(promotions) / double(tokens) : 0.0;
    }
    // Tokens a promotion takes to land, on average. This is the number that says whether the
    // refresh period is right: if it exceeds the period, the next refresh runs before the last
    // one's transfers have arrived and the period is wrong.
    double tokens_to_land() const {
        return activations ? double(land_tokens) / double(activations) : 0.0;
    }
    // Difference of two snapshots, so a caller can report the decode phase separately from
    // the prompt it warmed up on. The two are very different numbers and averaging them
    // together flatters neither honestly.
    ResidentStats since(const ResidentStats& base) const;
};

// One instance covers every layer, because that is how the engine holds it and because the
// capacity and the policy are identical across layers - only the contents differ.
//
// Lifecycle, per generated token:
//
//   1. write_mask(il, dst) for each layer -> the graph's residency input. Written BEFORE the
//      graph runs, so the set the token is served from is one chosen without seeing it.
//   2. the graph runs and splits by that mask.
//   3. observe(il, ids, n) for each layer, with the ids the graph actually dispatched. This
//      is both the hit accounting and the window update.
//   4. end_token(), which runs the refresh when the period says so.
//
// Steps 3 and 4 are in that order and step 4 is once per token, not once per layer: the
// period counts tokens, and the whole model refreshes together because a PCIe transfer budget
// is a property of the link, not of a layer.
class ResidentSet {
  public:
    explicit ResidentSet(const ResidentParams& p);

    const ResidentParams& params() const { return p_; }
    bool on() const { return p_.capacity > 0; }

    // O(1), and this is why the state is a bitset per layer rather than the sorted array of
    // resident ids the design sketch assumed. 128 experts is two 64-bit words: the test is a
    // shift and an AND on one cache line, with no branch and nothing to mispredict. A binary
    // search over 29 sorted ids is five dependent comparisons on unpredictable branches, and
    // it runs 8 * 48 = 384 times per token. The bitset also makes the two set operations the
    // refresh needs - want \ resident, and resident \ want - word-wise instead of a merge of
    // two sorted lists, and it makes write_mask a straight bit-to-float expansion.
    bool is_resident(int layer, int expert) const {
        const uint64_t* b = bits_of(layer);
        return (b[unsigned(expert) >> 6] >> (unsigned(expert) & 63)) & 1u;
    }

    // ---- deferred activation -------------------------------------------------------------
    //
    // A promoted expert is PENDING, not resident. While pending it is computed on the CPU
    // exactly as a non-resident expert is - is_resident, split and write_mask all say no - so a
    // promotion can never stall a layer waiting for its bytes. It becomes resident only when
    // the uploader confirms the bytes are in device memory, via activate().
    //
    // Pending still OCCUPIES its device slot: the slot map is built from is_claimed(), which is
    // resident-or-pending, and the refresh counts both against the capacity. That is what stops
    // a refresh over-committing video memory to transfers that have not landed.
    void set_deferred(bool on) { deferred_ = on; }
    bool deferred() const { return deferred_; }

    bool is_pending(int layer, int expert) const {
        const uint64_t* b = layers_[std::size_t(layer)].pend.data();
        return (b[unsigned(expert) >> 6] >> (unsigned(expert) & 63)) & 1u;
    }
    // Resident OR pending: everything that owns a device slot. This, not is_resident, is what
    // the slot map must be built from. With deferred off the two are identical.
    bool is_claimed(int layer, int expert) const {
        const Layer& L = layers_[std::size_t(layer)];
        const unsigned w = unsigned(expert) >> 6, b = unsigned(expert) & 63;
        return ((L.bits[w] | L.pend[w]) >> b) & 1u;
    }

    // The uploader's confirmation that (layer, expert)'s bytes are in device memory. Moves the
    // bit from pending to resident and bumps the revision, so the next token's mask includes
    // it. Silently ignores an expert that is not pending, which is the normal case for a
    // confirmation that arrives for a slot the policy has since changed its mind about.
    void activate(int layer, int expert);

    int n_resident(int layer) const;
    int n_pending(int layer) const;

    // The residency flag for every expert of one layer, in the form the compute graph wants:
    // n_experts floats, 1.0 resident and 0.0 not. The graph gathers this by the router's own
    // ids, so a dense vector is what it needs and a list of ids is not.
    void write_mask(int layer, float* dst) const;

    // Bumped by every refresh that changed anything. The engine writes the mask into the
    // graph's input buffer only when this moves, which with a period of 3 is once every three
    // tokens instead of every token.
    uint64_t revision() const { return revision_; }

    // The per-token split. `weights` may be null when the caller has not got them; the
    // weights are carried, never applied - see ExpertPick.
    void split(int layer, const int32_t* ids, const float* weights, int n,
               ResidentSplit* out) const;

    // Account and remember one layer's selections for one token. ids may contain -1 for
    // slots the router did not fill (expert reduction does that), and those are skipped
    // rather than counted as misses.
    void observe(int layer, const int32_t* ids, int n);

    // Once per token, after every layer has been observed.
    void end_token();

    const ResidentStats& stats() const { return st_; }

    // Distinct experts the window currently holds, for one layer. Printed rather than used:
    // early in a run it is smaller than the capacity, and a set that is smaller than its
    // capacity is the difference between "the policy has not warmed up" and "the policy is
    // broken".
    int window_distinct(int layer) const;

  private:
    struct Layer {
        std::vector<uint64_t> bits;     // (n_experts + 63) / 64 words: RESIDENT
        std::vector<uint64_t> pend;     // same shape: promoted, bytes not confirmed yet
        std::vector<int16_t>  ring;     // window * n_used, the sliding window itself
        std::vector<int32_t>  count;    // n_experts, kept current with the ring
        std::vector<int32_t>  req_tok;  // n_experts, token index a pending promotion started
        int head     = 0;               // next ring slot to overwrite
        int fill     = 0;               // slots written so far, up to ring.size()
        int distinct = 0;               // experts with a non-zero count
        int n_res    = 0;               // popcount of bits, kept rather than recomputed
        int n_pend   = 0;               // popcount of pend, likewise
    };

    const uint64_t* bits_of(int layer) const {
        return layers_[std::size_t(layer)].bits.data();
    }

    void push(Layer& L, int expert);
    void refresh();
    void refresh_layer(Layer& L);
    // Fills want_ with the capacity the policy wants, best first, and want_bits_ with the
    // same as a bitset. Returns how many it chose, which early in a run is fewer than the
    // capacity because the window has not seen that many distinct experts yet.
    int choose_lfu(const Layer& L, int cap);
    int choose_lru(const Layer& L, int cap);

    void refresh_layer_deferred(Layer& L, int n_want);

    ResidentParams p_;
    std::vector<Layer> layers_;
    ResidentStats st_;
    uint64_t revision_ = 0;
    // Off until the engine has an uploader that can confirm. With it off the set behaves
    // exactly as it did before this existed - which is what keeps --gpu-experts-selftest and
    // the CPU-only bookkeeping arm unchanged.
    bool deferred_ = false;

    // Scratch, so a refresh allocates nothing. It runs 48 times every three tokens.
    std::vector<int16_t>  rank_;        // n_experts, permuted by the ranking
    std::vector<int16_t>  want_;        // capacity, best first
    std::vector<uint64_t> want_bits_;
    std::vector<int16_t>  promote_;
    std::vector<int16_t>  victims_;
    std::vector<uint32_t> seen_;        // n_experts, stamped for the LRU walk
    uint32_t stamp_ = 0;
};

// ---------------------------------------------------------------------------------------
// The graph side
// ---------------------------------------------------------------------------------------

// dst[slot] = ids[slot] where that expert is resident, and -1 everywhere else. `resident`
// false gives the complement, so the two calls together partition the router's top-k with no
// slot lost and none counted twice.
//
// -1 rather than a separate shorter list, because ggml_mul_mat_id already treats an id
// outside [0, n_experts) as "this slot is not used": it skips the row entirely - so the bytes
// genuinely are not read, which is what makes the byte accounting a count rather than a model
// - and memsets that output slice to zero.
//
// Zero is what makes the whole thing bit-exact, and only if the two halves are added PER SLOT,
// before the routing weights and before the sum over slots. Then every slot of the sum is
// either x + 0.0 or 0.0 + x, which is x exactly, and the weighting and the eight-way fold that
// follow are literally the unsplit path's own operations in the unsplit path's own order.
// Folding each half separately and adding the two totals instead reassociates eight f32
// additions; that was measured at 2.5e-8 relative on the next layer's input, which this model
// amplified to 1.6% by layer 47 and 3-6% on the logits. The tokens survived it, but "the same
// answer to within amplified rounding" is a much weaker claim than "the same bits", and the
// stronger one costs one elementwise add per layer.
//
// `ids` is expected to be what ggml_top_k returns, which is a strided view of an argsort
// result rather than a contiguous block; the strides are honoured. `flags` is the residency
// mask already gathered by those same ids - shape [1, n_used, n_tokens] f32, which is what
// ggml_get_rows over a [1, n_experts, n_tokens] mask produces.
ggml_tensor* resident_split_ids(ggml_context* ctx, ggml_tensor* ids, ggml_tensor* flags,
                               bool resident);

// A contiguous I32 copy of `ids`. ggml_top_k hands back a view whose rows are n_experts apart,
// so the host cannot read a token's selections as one block; this exists so the policy can be
// warmed from the prompt's own routing, which is the difference between a reported hit rate
// that means something and one measured over a window that spent a third of the run empty.
ggml_tensor* resident_ids_copy(ggml_context* ctx, ggml_tensor* ids);

}  // namespace memex
