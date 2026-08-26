// A KV cache zoned by token position, packaged so an engine can adopt it without
// knowing the zone layout.
//
// The idea. Attention re-reads the whole cache for every generated token, so past
// context - not weights - is what caps decode speed once the context is long. But the
// positions are not equally worth their bytes, and this splits them by position into
// four zones:
//
//   sinks    - the first few positions. Attention leans on them whatever they contain,
//              so they stay exact for ever and are never evicted.
//   notebook - old positions the engine asked to keep exactly: the facts a later
//              question will need. Bounded, and the engine names them, because only the
//              model knows which position a future query will want.
//   tail     - the long middle, compressed. Full rank at eight bits, which is the
//              representation the earlier probes argued for; see below.
//   window   - the most recent positions, exact, because local attention is where
//              approximation hurts most.
//
// The one thing that decides correctness: **the softmax is global**. Scores from all
// four zones are concatenated and normalised once. A softmax per zone lets a window
// position and a tail position stop competing for probability mass, which is a
// different function and a wrong one - and it is a wrong one that still produces
// plausible-looking vectors, so it has to be checked rather than intended. build_attn
// returns the softmax node itself for exactly that reason: the test sums its output and
// insists on one, not one-per-zone.
//
// Why this is not already in the fork. -ctk-first / -ctk-last look like the same idea
// but their N counts *layers* (src/llama.cpp:1487-1502 indexes the layer loop), so
// within a layer the cache type is uniform and there is one ordinary softmax over one
// ordinary cache. Nothing in this codebase zones by position.
//
// What is reused rather than rebuilt. memex::KvZones (Desktop/MemeX/cpp) already writes
// Q8_0 blocks by hand in exactly the layout a ggml graph wants, and already builds the
// Hadamard rotation for the tail keys. That encoder is measured and is driven here
// unchanged, configured as a pure tail (no sinks, no window, no notebook of its own).
// The positional zone map lives here instead of there on purpose: KvZones evicts by an
// importance score with an elastic notebook, which is a richer policy than this module
// wants to impose. Positional zoning with an engine-supplied keep flag is the smaller
// contract, and the smaller contract is the point of packaging this separately.
//
// What the tail should hold, and what was refuted. Rank-r latents were the original
// plan and they lose: at 512 bytes per position rank 128 leaves 78.96% error on the
// attention output where eight-bit full rank at 1088 leaves 3.09% and four-bit at 576
// leaves 31.72%. The cause is structural - grouped-query attention already compressed
// the cache fourfold (d_kv 512 against n_embd 2048), so there is little spectrum left
// for a basis to exploit. Hence no basis here at all: the tail is full-rank Q8_0, and
// the only alternative offered is an F16 tail, which exists as a control rather than as
// a shipping option (see tail_q8 below).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"

#include "memex/kv_zones.hpp"

namespace memex {

// Zone lengths reach a tensor rounded up to this, and the padding is masked with -inf.
// Not tidiness: the fork's F16 matmul is silently wrong when the reduction length is not
// a multiple of four - measured in memex-attn, where context 2780 agreed with a host
// computation to 0.001% while 2777, 2778 and 2779 were off by more than 100% and the
// output tensor came back holding the attention probabilities instead of their weighted
// sum. Nothing asserts, so an unpadded cache returns confident nonsense. llama.cpp pads
// to 32 for the same reason, and Q8_0 needs a multiple of 32 along the block axis anyway.
constexpr int kZonePad = 32;
inline int zone_pad_up(int n) { return (n + kZonePad - 1) / kZonePad * kZonePad; }

struct ZonedCacheParams {
    int n_layers   = 1;
    int n_q_heads  = 8;
    int n_kv_heads = 2;
    int head_dim   = 128;      // power of two, or the Hadamard rotation is unavailable
    int n_ctx      = 1024;     // hard cap on positions per layer

    int n_sinks      = 4;
    int notebook_cap = 32;
    int window       = 256;

    // Eight-bit full rank, or F16. The F16 tail is a control and is treated as one: it
    // makes the zoned graph algebraically identical to exact attention, so its error
    // isolates "is the zoned expression right" from "what does the quantiser cost".
    bool tail_q8     = true;
    // Rotate tail keys by a random Hadamard before quantising. Free in the scores,
    // because (Rq).(Rk) = q.k exactly, and it halves the error because block
    // quantisation is limited by the largest value in a block and a rotation spreads
    // outliers evenly. Keys only, and that is measured, not assumed: rotating values as
    // well gave 1.58% against 1.57%, because value blocks run along *positions* while
    // the rotation mixes *dimensions*, so it cannot flatten anything inside a block.
    //
    // Ignored when tail_q8 is false. On an F16 tail a rotation cannot help - there is no
    // block scale to flatten - and it would add a round trip of rounding, which would
    // spoil the one arm whose error is supposed to be zero.
    bool rotate_keys = true;

    int d_kv() const { return n_kv_heads * head_dim; }
    int ratio() const { return n_kv_heads ? n_q_heads / n_kv_heads : 0; }
    // Slots the exact zones occupy in a tensor, before padding. Fixed at construction:
    // each zone owns its own slot range for ever, so a position lands with one O(d_kv)
    // write and no zone ever has to be shuffled.
    int exact_slots() const { return n_sinks + notebook_cap + window; }

    // Refuses rather than repairs. Every one of these has a way of turning into a wrong
    // answer instead of an error further down.
    bool validate(std::string* err) const;
};

// What build_attn hands back. `out` is all an engine needs; the other two exist so a
// test can look at the scores and at the *single* probability distribution rather than
// take the claim on trust.
struct ZonedAttn {
    ggml_tensor* out    = nullptr;   // [head_dim, 1, n_q_heads] f32
    ggml_tensor* scores = nullptr;   // [n_slots, 1, n_q_heads] f32, pre-softmax
    ggml_tensor* probs  = nullptr;   // [n_slots, 1, n_q_heads] f32, ONE softmax
    // The two partial weighted sums, before they are added. They are here for one
    // reason: with them and with `probs` a test can reconstruct, exactly, the answer a
    // *per-zone* softmax would have produced - divide each partial sum by its own zone's
    // probability mass - and print the error that mistake would cost. The maximum
    // subtraction cancels in a ratio, so the reconstruction is not an approximation of
    // the wrong function, it is the wrong function. That turns "the softmax is global"
    // from an intention in a comment into a number on the screen.
    ggml_tensor* out_exact = nullptr;
    ggml_tensor* out_tail  = nullptr;
    int n_slots         = 0;         // exact slots (padded) + tail slots (padded)
    int exact_slots     = 0;         // where the tail's share of `probs` begins
};

struct ZoneOccupancy {
    int sinks = 0, notebook = 0, window = 0, tail = 0;
    int dropped = 0;                 // positions the tail had no room for
    int total() const { return sinks + notebook + window + tail; }
};

// The cache. One instance covers every layer, because that is how an engine holds it and
// because the zone map is identical across layers - only the contents differ.
//
// Lifecycle, which is deliberately the same dance an engine already performs for its own
// KV cache:
//
//   1. ZonedCache c(params);
//   2. c.create_tensors(ctx, &err);            // ctx with no_alloc = true
//   3. ...engine creates its own tensors in the same ctx...
//   4. ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
//   5. c.append(il, k, v, keep) per position;  c.flush();
//   6. c.upload(il);                           // host mirror -> backend tensors
//   7. ZonedAttn a = c.build_attn(ctx, il, q); // one softmax over four zones
//
// Steps 5 and 6 are shown in that order because this test fills the cache in one go. The
// interface does not require it: append may be called between graph computations and
// upload again, and an engine writing straight into a mapped buffer can skip the mirror
// entirely - which is why the slot map is fixed and every write is O(d_kv).
class ZonedCache {
  public:
    explicit ZonedCache(const ZonedCacheParams& p);

    const ZonedCacheParams& params() const { return p_; }

    // One position of one layer. Rows are d_kv floats in the engine's own
    // [kv_head][head_dim] order - the order llama.cpp's Kcur/Vcur already have.
    //
    // keep_exact is the whole notebook policy: when this position later falls out of the
    // window it goes to the notebook if the engine asked and there is room, and to the
    // compressed tail otherwise. A boolean rather than an importance score because the
    // cache is in no position to rank; the model is. KvZones' importance-ranked elastic
    // notebook is the same idea with a richer signal, and an engine that has such a
    // signal can drive this by thresholding it.
    void append(int il, const float* k, const float* v, bool keep_exact);

    // Settle the tail's partially filled blocks. Value blocks of 32 run along positions,
    // so the last block is incomplete until 32 positions have arrived; until this is
    // called, those positions are staged rather than present. Call before upload.
    void flush();

    // Copy the host mirror into the backend tensors. Fails loudly if the tensors were
    // never allocated - a tensor with a null buffer would otherwise take the write into
    // nowhere and read back as zeros, which is the exact failure mode this whole example
    // is built to make impossible.
    bool upload(int il, std::string* err);

    bool create_tensors(ggml_context* ctx, std::string* err);

    // q is [head_dim, 1, n_q_heads] f32, in the engine's head order. The rotation the
    // tail keys were stored under is applied to q *inside the graph* - one head_dim x
    // head_dim product per token, independent of context length - so the caller never
    // learns that the tail is rotated. Getting that rotation wrong on the query side is
    // the one way this optimisation goes bad silently, so it is not left to the caller.
    ZonedAttn build_attn(ggml_context* ctx, int il, ggml_tensor* q);

    // ---- The engine-driven live slot ----------------------------------------------------
    //
    // append() wants the arriving row, and an engine that computes k and v inside its own
    // compute graph has not got it yet: the row does not exist until the graph has run, and
    // the graph needs the cache to already hold the position it is about to answer for. That
    // circle is why an engine cannot drive append() during decode, and it is broken by
    // splitting the call in two rather than by moving the zone map into the engine.
    //
    //   1. append_live(il, keep) does everything append() does *except* store the arriving
    //      row's data: it evicts whatever the slot held - to the notebook or to the
    //      compressed tail, by the same policy - marks the slot occupied in the mask and
    //      advances the window ring. It returns the slot the engine's graph must write, or
    //      -1 if the position was dropped, which is the same honest loss append() counts.
    //   2. the engine's graph copies its own k and v straight into exact_k()/exact_v() at
    //      that slot, in the same graph that then calls build_attn - so the store is ordered
    //      before the load exactly as it is in an ordinary llama.cpp decode graph, and no
    //      host round trip happens inside a layer.
    //   3. set_live_row(il, slot, k, v) mirrors the same row on the host afterwards. Without
    //      it the mirror would still hold the position that was evicted in step 1, and the
    //      *next* eviction of that slot would compress the wrong vector - silently, because
    //      the shapes are right and the numbers are plausible. That is the one way this
    //      protocol goes wrong, so it is written down here rather than left to be noticed.
    //
    // The three exist alongside append() rather than replacing it: the test driver fills the
    // cache from the host and must keep working unchanged, and a prefill - where the rows
    // are already known - is still one append() per position.
    // `note_slot`, when given, comes back holding the notebook slot this call wrote, or -1.
    // It exists because that write is the one thing under this protocol that the engine's
    // graph does *not* do and upload_dynamic does not carry: an evicted position promoted to
    // the notebook lands in the host mirror only, and a mask that admits a slot the backend
    // never received reads zeros - confidently. The engine must hand the slot back to
    // upload_slot. Ignoring the out-parameter is safe only while nothing is ever flagged.
    int append_live(int il, bool keep_exact, int* note_slot = nullptr);
    void set_live_row(int il, int slot, const float* k, const float* v);
    // Mask and tail only. Under the live protocol the exact zones belong to the engine's
    // graph, so copying the host mirror over them would undo the store the graph just made.
    bool upload_dynamic(int il, std::string* err);
    // One exact slot, host mirror to backend. The counterpart to append_live's note_slot,
    // and cheap enough to be called on every notebook admission because there are at most
    // notebook_cap of those in a whole run.
    bool upload_slot(int il, int slot, std::string* err);
    // The exact zones' tensors, so an engine can build its own destination views into them.
    // Keys are [head_dim, exact_pad, kv_head] and values [exact_pad, head_dim, kv_head] -
    // the same two layouts llama.cpp's own cache uses, which is what makes the engine's
    // existing copy nodes reusable against them.
    ggml_tensor* exact_k(int il) const;
    ggml_tensor* exact_v(int il) const;
    int exact_pad() const { return ex_pad_; }
    int tail_pad() const { return tl_pad_; }

    ZoneOccupancy occupancy(int il) const;
    // Bytes attention reads for one token at one layer, counted on occupied positions
    // only: the padding is a kernel artefact and counting it would flatter nothing
    // consistently.
    std::size_t read_bytes(int il) const;
    // The same count if every position were kept exact in F16. The denominator of every
    // ratio this example prints.
    std::size_t read_bytes_if_exact(int il) const;

  private:
    struct Layer {
        // Exact zones, in the layout the graph wants and nothing else: keys as
        // [head_dim, slot, kv_head] so a row is one head of one position, values as
        // [slot, head_dim, kv_head] so the weighted sum is a plain matrix-vector product
        // instead of a per-token ggml_cont of the whole cache (measured at 245.8 ms
        // against 23.3 ms at 4k in memex-attn).
        std::vector<uint16_t> ex_k, ex_v;
        // The F16 tail, when tail_q8 is off. Same two layouts.
        std::vector<uint16_t> tl_k, tl_v;
        // The Q8_0 tail. KvZones owns the encoding; this owns nothing but the handle.
        std::unique_ptr<KvZones> q8;
        std::vector<float> mask;          // [exact_pad + tail_pad], 0 or -inf
        std::vector<uint8_t> slot_used;   // exact slots only, for the mask and the count

        ggml_tensor* K = nullptr;
        ggml_tensor* V = nullptr;
        ggml_tensor* Tk = nullptr;
        ggml_tensor* Tv = nullptr;
        ggml_tensor* M = nullptr;
        ggml_tensor* R = nullptr;         // [head_dim, head_dim] f32, row-major

        int n_pos = 0;                    // positions appended
        int n_sink = 0;
        int n_note = 0;
        int n_win = 0;
        int n_tail = 0;
        int dropped = 0;
        int win_head = 0;                 // next window ring slot to overwrite
        std::vector<int> note_free;       // notebook slots not yet taken
    };

    // Slot ranges. Fixed for the lifetime of the cache, which is what makes every write
    // O(d_kv) and lets an engine keep one graph shape for every token.
    int sink_slot(int i) const { return i; }
    // Named ..._of because append_row now has a local out-parameter called note_slot, and a
    // member function shadowed by a parameter is the kind of thing that compiles.
    int note_slot_of(int i) const { return p_.n_sinks + i; }
    int win_slot(int i) const { return p_.n_sinks + p_.notebook_cap + i; }

    // `mark` false writes the data and leaves the mask and the keep flag alone, which is
    // what set_live_row needs: append_live has already marked the slot, and marking it a
    // second time would reset a keep flag the engine asked for.
    void write_exact(Layer& L, int slot, const uint16_t* k, const uint16_t* v,
                     bool mark = true);
    // The body append() and append_live() share. Null k and v mean "do the bookkeeping and
    // the eviction, but leave the arriving row's data to the caller". Returns the exact slot
    // the arriving position occupies, or -1 if it was dropped.
    int append_row(int il, const uint16_t* k, const uint16_t* v, bool keep_exact,
                   int* note_slot);
    bool upload_impl(int il, std::string* err, bool exact);
    void read_exact(const Layer& L, int slot, uint16_t* k, uint16_t* v) const;
    void write_tail_f16(Layer& L, int slot, const uint16_t* k, const uint16_t* v);

    ZonedCacheParams p_;
    int ex_pad_ = 0;
    int tl_pad_ = 0;
    std::vector<Layer> layers_;
    // Scratch, so append allocates nothing: the arriving row and the evicted one.
    std::vector<uint16_t> row_k_, row_v_;
    std::vector<uint16_t> ev_k_, ev_v_;
};

// fp16 helpers, shared with the test. The cache is half precision because that is what
// the model produces and what attention reads; float would double the very traffic the
// zones exist to reduce.
uint16_t zc_f2h(float f);
float zc_h2f(uint16_t h);

}  // namespace memex
