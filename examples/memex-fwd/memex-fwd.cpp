// MemeX computes the forward pass itself, and checks its logits against llama.cpp's.
//
// The inversion. Until now every piece we built - the expert blob, per-expert dispatch, the
// zoned KV cache, the router, concurrent two-backend execution - lived in its own example
// and was measured on its own. They could not be assembled because the graph belonged to
// the fork, and the fork's graph is built for every architecture at once, with a scheduler
// that runs devices one after another. So the pieces stayed measured and unused.
//
// This turns the dependency around. llama.cpp keeps the jobs it is good at and we are not:
// parsing GGUF with its dozens of quantisation formats, the tokeniser with all its edge
// cases, and above all ggml itself - the hand-written kernels are the entire reason to be
// in this fork. We take the model through the public API (llama_model_load_from_file,
// llama_get_model_tensor, llama_tokenize) and never call llama_decode. The graph is ours,
// for one architecture only, which is why it is a few hundred lines instead of ten thousand.
//
// The whole thing rests on one check: run the fork's own decode over the same tokens and
// compare the logits. While they agree, internals can be swapped one at a time - our
// attention, our dispatch, our scheduler - and any divergence points at the piece that just
// changed. That discipline is what the attention work needed and did not have until a
// four-way comparison caught a graph that was returning probabilities instead of an output.
//
// This first stage deliberately uses ordinary ggml operations and the fused expert tensors,
// exactly as the reference does. It is not meant to be faster - it is meant to be a correct
// starting point that is entirely ours.
//
// SECOND STAGE. The graph is now at parity with the fork's own llama-cli on the same file, so
// the harness has to become something a person other than its author can run: sampling, a
// chat template, speculative decoding, and every optimisation that cannot change the answer
// turned on without being asked for. "Cannot change the answer" is the whole rule, and it is
// narrow on purpose - repacking rewrites a weight layout and is bit-exact, speculation is
// provably distribution-preserving, and expert reduction is neither, which is why it stays
// behind a flag it has always been behind. The greedy default is unchanged to the token,
// because the only reason any of this can be trusted is that the old comparison still runs.
// EDITING THIS FILE WITH A SCRIPT: read this first.
//
// Most of the recent growth here arrived through generated patches - a python script doing
// anchored string replacement - and that method has two failure modes which have both
// already happened, in this file, and which behave very differently from a typo.
//
//   1. An anchor that slips duplicates a line. A second `struct Graph {` got in this way.
//      The compiler does catch it, but the error lands at the end of the translation unit
//      rather than at the duplication, and it costs a full build to find out.
//
//   2. An escape that survives into the output. Twenty-two printf strings ended up holding
//      a literal backslash followed by n instead of a newline, because the replacement text
//      was a python RAW string where "\\n" stays two characters - and, separately, because
//      this shell's heredocs collapse doubled backslashes on the way through. THE COMPILER
//      CANNOT SEE THIS. "...\\n" is a valid C string. The only symptom is output that runs
//      together, which reads as a formatting slip rather than as a corrupted patch, and the
//      next person to edit nearby will copy it forward.
//
// Both are the same underlying mistake: generating code with a tool whose escaping rules
// differ from the target language's. Neither is caught by review at the diff level, because
// a diff renders both of them as exactly what you meant to write.
//
// So: run check_source.py in this directory after any scripted edit and before any build.
// It checks brace balance, escaped newlines and duplicated adjacent lines, and it takes a
// second where a build takes fifteen minutes.
//
//     python examples/memex-fwd/check_source.py
//
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

// The memory guard needs the real figure, not a guess: setting repack_tensors makes the
// loader force mmap off, so the model stops being a file-backed mapping and becomes resident
// private memory. On a 32 GB machine that is the difference between running and thrashing.
#if defined(_WIN32)
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

// FOURTH STAGE. A subset of each layer's experts held resident in video memory, so that a GPU
// can compute the selected experts that are resident while the CPU computes the ones that are
// not. This header is the CPU half of that: the resident set, the per-token split, the refresh
// policy and the accounting. It is unconditional - unlike the zoned cache it needs nothing
// outside this directory - and off at run time, because --resident changes the shape of the
// compute graph and the unsplit path is what every reproducibility check runs through.
//
// The reason this half is worth building before the GPU question is settled: with the feature
// on and no GPU, the engine computes *both* halves on the CPU and sums them, so the split can
// be proved to be pure bookkeeping - identical tokens - before any Vulkan code exists. The
// alternative is to write the driver first and then have no way to tell a broken split from a
// broken kernel.
#include "resident_set.hpp"

// FIFTH STAGE, and the one the four before it were for. The resident half executes on the
// Vulkan device WHILE the CPU computes the non-resident half, and the two are joined at the
// end of the layer. Concurrency is the whole point and it is not something the fork can
// express: its scheduler puts a barrier in front of every split that has inputs, and a
// sequential handoff has been measured and loses - whole-layer offload runs at 9.16 tok/s
// against 13.41 for the CPU on its own.
//
// It is also not something ggml's asynchronous backend interface can express on this backend,
// which is the finding that shaped the implementation: ggml_backend_graph_compute_async is a
// direct call to iface.graph_compute (ggml-backend.cpp:327-329), the Vulkan backend's
// synchronize / set_tensor_async / get_tensor_async / event entries are all NULL under a
// "// TODO: enable async and synchronize" (ggml-vulkan.cpp:10700-10721), and
// ggml_backend_vk_graph_compute waits on a fence for its own last submit before returning
// (ggml-vulkan.cpp:9674-9678). So the submit blocks, unconditionally, and the concurrency is
// built out of a host thread that owns the backend. See gpu_experts.hpp.
#ifdef MEMEX_FWD_GPU_EXPERTS
#include "gpu_experts.hpp"
#else
namespace memex { class GpuExperts; }
#endif

// SIXTH STAGE, and the largest single lever left in the byte budget. Of the 1714 MB a
// generated token reads, 802 MB is STATIC - attention, the output head and the routers, the
// same bytes on every token - and static bytes need no residency policy at all: no
// prediction, no eviction, no churn, and a return of 1.00 bytes read per byte resident, which
// is the highest ratio anything in the model has.
//
// The static set is not one lever though, it is three, and they differ by how often the
// processor and the card have to meet. A rendezvous costs 177 us whatever it carries, so:
//
//     the head    1 crossing per TOKEN   243.4 MB   pays immediately
//     attention  48 crossings per token  510.4 MB   pays, but needs the KV cache on the card
//     the router 48 crossings per token   48.0 MB   loses alone, free once attention is there
//
// This stage is the head: it is the last node in the graph, so it is the one piece of the
// static set with a single boundary in the whole token and nothing left waiting behind it.
#ifdef MEMEX_FWD_GPU_EXPERTS
#include "gpu_static.hpp"
#else
namespace memex { class GpuStatic; }
#endif

// THIRD STAGE. The zoned KV cache from examples/memex-kv, adopted as an option that is off
// by default. Off by default is not timidity: the exact path is the only reason anything in
// this file can be trusted, so it stays the thing every reproducibility check runs through,
// bit for bit, and the zoned path has to earn its way past a gate that proves the argmax
// cannot move before it is worth switching on. The gate is --zoned-check and the proof it
// prints is the argmax margin, not an error percentage.
//
// The one thing that needed inventing here rather than adopting. The module's append() wants
// the arriving key and value, and this engine computes them *inside* its own compute graph -
// so at position p the row does not exist until the graph has run, and the graph needs the
// cache to already contain position p. That circle is broken by the module's live-slot
// protocol (append_live / set_live_row): the cache does the eviction and the bookkeeping
// before the graph runs and hands back a slot, the graph stores into that slot with an
// ordinary ggml_cpy - ordered before the attention read exactly as llama.cpp's own store is
// - and the host mirrors the same row afterwards so the *next* eviction of that slot
// compresses the right vector. Skipping the third step is silent: right shapes, plausible
// numbers, wrong past.
#ifdef MEMEX_FWD_ZONED
#include "zoned_cache.hpp"
using ZonedKvCache = memex::ZonedCache;
#else
// Declared and never defined. build_step's signature can then name the pointer in both
// builds while every use of it stays inside #ifdef, instead of the signature itself forking.
struct ZonedKvCache;
#endif

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Physical memory, in bytes. `ok` false means we could not find out - which is treated as a
// reason to be careful rather than as a zero, because a zero here would silently disable the
// one optimisation worth 38%.
struct PhysMem {
    bool ok = false;
    uint64_t total = 0;
    uint64_t avail = 0;
};

PhysMem phys_mem() {
    PhysMem m;
#if defined(_WIN32)
    MEMORYSTATUSEX s;
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) {
        m.ok = true;
        m.total = s.ullTotalPhys;
        m.avail = s.ullAvailPhys;
    }
#endif
    return m;
}

uint64_t file_size_bytes(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
#if defined(_WIN32)
    const long long sz = _ftelli64(f);
#else
    const long long sz = (long long)ftello(f);
#endif
    fclose(f);
    return sz > 0 ? uint64_t(sz) : 0;
}

double gb(uint64_t b) { return double(b) / 1e9; }

// What a layer does where its attention would go. Scalar hyperparameters cannot express
// this, and that is not a style preference: gemma4 alternates two DIFFERENT attention
// geometries inside one model - 8 kv heads of 256 on the windowed layers against 2 of 512
// on the full ones, each with its own rope base - and qwen35moe alternates attention with a
// recurrence that has no KV cache at all. A single n_head_kv is not merely inconvenient for
// those two, it is arithmetically false, and the shapes it produces are plausible enough to
// run and produce confident nonsense.
enum class LayerKind {
    ATTN,       // causal attention over the whole prefix
    ATTN_SWA,   // the same, except a query may only see the last n_swa keys
    DELTA_NET,  // gated delta-net: a fixed-size recurrent state and no KV cache at all
};

// One layer's attention geometry. Everything here is per-layer because at least one of the
// three architectures varies it per layer; anything that no architecture varies stays a
// scalar on HParams, which is what keeps this a widening rather than a rewrite.
struct LayerGeom {
    LayerKind kind = LayerKind::ATTN;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;      // key length. Equal to the value length in every model we accept,
                           // which is checked at load rather than assumed - the cache stores
                           // one head width and a mismatch would silently truncate V.
    int n_rot = 0;         // the rotated prefix of head_dim; the tail is left un-rotated.
                           // Equal to head_dim for qwen3moe and gemma4, 64 of 256 for
                           // qwen35moe - partial rope is the norm now, not the exception.
    float rope_base = 10000.0f;
    // Per-dimension divisors of the rope angle, or null. Gemma 4 ships one of these for its
    // full-attention layers and it is not decorative: the entries are 1.0 for the first 64
    // pairs and 1e30 for the remaining 192, which drives theta to zero and leaves dimensions
    // 128..511 unrotated. Dropping it does not fail, it just rotates what must not move.
    ggml_tensor* rope_freqs = nullptr;

    bool has_kv() const { return kind != LayerKind::DELTA_NET; }
    int d_kv() const { return n_head_kv * head_dim; }
    int d_q() const { return n_head * head_dim; }
};

// Hyperparameters, read from the file rather than assumed. Every one of these has bitten
// somebody who hardcoded it: the rope base differs between Qwen3 releases, the expert
// intermediate width is not the dense one, and the head count is not the kv head count.
//
// The scalars below are the MODEL-WIDE values - what the file states at the top level, what
// gets printed, and what a uniform model's every layer is initialised to. Anything a graph
// builder needs per layer it must take from L(il) and never from these, because for two of
// the three architectures the two disagree.
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
    int n_ff = 0;          // the dense feed-forward width, which the draft model uses
    int n_ff_shexp = 0;    // the shared expert's width, where there is one
    int n_vocab = 0;
    int n_ctx_train = 0;
    float rms_eps = 1e-6f;
    float rope_base = 10000.0f;
    int rope_type = 0;
    int n_swa = 0;         // the sliding window, in positions; 0 where no layer is windowed

    // The softmax scale on the attention scores. It is the WHOLE scale, not a correction to
    // 1/sqrt(head_dim). Zero means "use 1/sqrt(head_dim)", which is what every architecture
    // but gemma4 wants; gemma4 states 1.0 and means it, and the difference between 1.0 and
    // 1/sqrt(512) degrades the output rather than breaking it, which is the worst way for a
    // constant to be wrong.
    float f_attn_scale = 0.0f;
    // softcap*tanh(x/softcap) on the final logits, gemma4 only. Zero disables it.
    float f_logit_softcap = 0.0f;
    // sqrt(n_embd) on the token embeddings, gemma only. Zero disables it.
    float f_embd_scale = 0.0f;
    // Applied to the router's normalised input, gemma4 only. ONE means "no scaling" here
    // rather than zero, because unlike the three above this one multiplies a value that is
    // always present: the reference does the same multiplication to ffn_gate_inp.scale at
    // load time (llm_scale_gate_inp_s), so a file read straight off disk is already wrong.
    float f_router_scale = 1.0f;
    // mrope section widths. All zero for a text-only model, which is what ggml_rope_multi
    // wants; qwen35moe states {11,11,10,0} and is not text-only in its rope even though it
    // is in everything else we ask of it.
    int rope_sections[4] = {0, 0, 0, 0};

    // Gated delta-net, qwen35moe only. Zero everywhere else.
    int ssm_d_conv = 0;    // convolution kernel, 4
    int ssm_d_inner = 0;   // total value width, 4096
    int ssm_d_state = 0;   // key and value head width, 128
    int ssm_dt_rank = 0;   // number of V heads, 32
    int ssm_n_group = 0;   // number of K heads, 16

    // Per-layer geometry. Always n_layer long after read_hparams; a uniform model fills it
    // with n_layer copies of the scalars above, so the qwen3moe path reads it and gets
    // exactly what it read from the scalars before.
    std::vector<LayerGeom> layers;

    const LayerGeom& L(int il) const { return layers[std::size_t(il)]; }
    LayerGeom& L(int il) { return layers[std::size_t(il)]; }

    // Layer zero's, and only ever safe where the model is known uniform - the printing, the
    // draft model, and the two optional modules that refuse a non-uniform model by name.
    int d_kv() const { return n_head_kv * head_dim; }
    int d_q() const { return n_head * head_dim; }

    // Is every layer the same shape? Asked rather than assumed by everything that was
    // written when one shape was all there was: the zoned cache and the resident set both
    // index a single geometry, and a mixed model has to be refused there rather than
    // reinterpreted into one.
    bool uniform_attn() const {
        for (const LayerGeom& l : layers) {
            if (l.kind != LayerKind::ATTN || l.n_head != n_head ||
                l.n_head_kv != n_head_kv || l.head_dim != head_dim ||
                l.n_rot != head_dim || l.rope_base != rope_base || l.rope_freqs) {
                return false;
            }
        }
        return !layers.empty();
    }

    // Halves of KV a single position occupies across every layer that has a cache. Summed
    // rather than multiplied out from d_kv(), because for gemma4 the per-layer widths differ
    // by a factor of two in each direction and for qwen35moe three quarters of the layers
    // contribute nothing at all.
    std::size_t kv_elems_per_pos() const {
        std::size_t n = 0;
        for (const LayerGeom& l : layers) {
            if (l.has_kv()) n += 2u * std::size_t(l.d_kv());
        }
        return n;
    }

    // The fixed recurrent state one delta-net layer carries, in f32 elements: the
    // convolution window plus the state matrix. Zero for a model with no such layers. It
    // does not grow with the context, which is the entire point of it.
    std::size_t delta_state_elems() const {
        if (ssm_dt_rank <= 0 || ssm_n_group <= 0 || ssm_d_conv <= 0) return 0;
        const std::size_t head_v  = std::size_t(ssm_d_inner) / std::size_t(ssm_dt_rank);
        const std::size_t key_dim = std::size_t(ssm_d_state) * std::size_t(ssm_n_group);
        const std::size_t conv_dim = key_dim * 2 + std::size_t(ssm_d_inner);
        return std::size_t(ssm_d_conv - 1) * conv_dim +
               head_v * head_v * std::size_t(ssm_dt_rank);
    }

    int n_delta_layers() const {
        int n = 0;
        for (const LayerGeom& l : layers) {
            if (l.kind == LayerKind::DELTA_NET) ++n;
        }
        return n;
    }

    bool any_swa() const {
        for (const LayerGeom& l : layers) {
            if (l.kind == LayerKind::ATTN_SWA) return true;
        }
        return false;
    }
};

// Is this type one of the fork's row-interleaved repack targets? Asked of a loaded tensor, not
// of a flag: what the loader was ASKED to do and what a given tensor actually is are different
// questions, and only the second one decides whether the Vulkan backend can read the bytes.
// Every such type is named with an _r<N> tail (iq4_xs_r8, q6_k_r4, q8_KV_r8, bf16_r16, ...),
// and no plain type is.
bool type_is_interleaved(ggml_type t) {
    const char* n = ggml_type_name(t);
    if (!n) return false;
    const size_t len = strlen(n);
    for (size_t k = 2; k <= 4 && k < len; ++k) {
        // "..._r4" / "..._r8" / "..._r16"
        if (n[len - k] == '_' && (n[len - k + 1] == 'r' || n[len - k + 1] == 'R')) {
            bool digits = true;
            for (size_t j = len - k + 2; j < len; ++j) {
                if (n[j] < '0' || n[j] > '9') { digits = false; break; }
            }
            if (digits && len - k + 2 < len) return true;
        }
    }
    return false;
}

// The stored type has to be checked, not assumed. gguf_get_val_u32 asserts inside ggml when the
// key holds anything else, so pointing the engine at a foreign model used to abort with
// "GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT32) failed" instead of reaching the
// architecture check twelve hundred lines later and refusing politely. Reading hparams happens
// before that check by necessity - the architecture name is itself one of the keys - so the
// tolerance belongs here rather than in the caller.
//
// Widths are accepted and converted rather than demanded: writers differ on whether a count is
// u32 or i32, and refusing a perfectly readable i32 would be pedantry that costs a user their run.
int key_u32(gguf_context* g, const std::string& k, int fallback) {
    const int id = gguf_find_key(g, k.c_str());
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32:  return int(gguf_get_val_u32(g, id));
        case GGUF_TYPE_INT32:   return int(gguf_get_val_i32(g, id));
        case GGUF_TYPE_UINT16:  return int(gguf_get_val_u16(g, id));
        case GGUF_TYPE_INT16:   return int(gguf_get_val_i16(g, id));
        case GGUF_TYPE_UINT8:   return int(gguf_get_val_u8(g, id));
        case GGUF_TYPE_INT8:    return int(gguf_get_val_i8(g, id));
        case GGUF_TYPE_UINT64:  return int(gguf_get_val_u64(g, id));
        case GGUF_TYPE_INT64:   return int(gguf_get_val_i64(g, id));
        default:                return fallback;
    }
}

// A key that may be a scalar OR an array of length n. Both gemma4 and qwen35moe state
// per-layer numbers this way, and which form a converter chose is not something the engine
// gets to depend on: unsloth's gemma4 writes head_count_kv as thirty values and
// sliding_window_pattern as thirty booleans, while an otherwise identical file from another
// converter may write one value and mean it for every layer. Returns the count actually
// filled, so the caller can tell "the file said nothing" from "the file said one thing".
//
// Widths are converted rather than demanded, for the same reason key_u32 converts them: a
// perfectly readable i8 array of booleans is not a reason to refuse a run.
int key_arr_u32(gguf_context* g, const std::string& k, std::vector<int>* out, int n) {
    out->clear();
    const int id = gguf_find_key(g, k.c_str());
    if (id < 0) return 0;
    const gguf_type t = gguf_get_kv_type(g, id);
    if (t != GGUF_TYPE_ARRAY) {
        // A scalar means the same value for every layer, which is the shape the caller wants
        // back either way - it has one loop, not two.
        const int v = key_u32(g, k, 0);
        out->assign(std::size_t(n), v);
        return n;
    }
    const gguf_type at = gguf_get_arr_type(g, id);
    const int have = int(gguf_get_arr_n(g, id));
    const void* d = gguf_get_arr_data(g, id);
    if (!d || have <= 0) return 0;
    out->reserve(std::size_t(have));
    for (int i = 0; i < have; ++i) {
        switch (at) {
            case GGUF_TYPE_UINT32: out->push_back(int(((const uint32_t*)d)[i])); break;
            case GGUF_TYPE_INT32:  out->push_back(int(((const int32_t*)d)[i])); break;
            case GGUF_TYPE_UINT16: out->push_back(int(((const uint16_t*)d)[i])); break;
            case GGUF_TYPE_INT16:  out->push_back(int(((const int16_t*)d)[i])); break;
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_BOOL:   out->push_back(int(((const uint8_t*)d)[i])); break;
            case GGUF_TYPE_INT8:   out->push_back(int(((const int8_t*)d)[i])); break;
            case GGUF_TYPE_UINT64: out->push_back(int(((const uint64_t*)d)[i])); break;
            case GGUF_TYPE_INT64:  out->push_back(int(((const int64_t*)d)[i])); break;
            default: out->clear(); return 0;
        }
    }
    // A short array is a file we do not understand, not a file we can pad. Say so by
    // returning what was there and letting the caller decide.
    return have;
}

float key_f32(gguf_context* g, const std::string& k, float fallback) {
    const int id = gguf_find_key(g, k.c_str());
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(g, id);
        case GGUF_TYPE_FLOAT64: return float(gguf_get_val_f64(g, id));
        case GGUF_TYPE_UINT32:  return float(gguf_get_val_u32(g, id));
        case GGUF_TYPE_INT32:   return float(gguf_get_val_i32(g, id));
        default:                return fallback;
    }
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
    // Qwen3 states the head dimension explicitly, which is not n_embd / n_head for it.
    h->head_dim = key_u32(g, a + ".attention.key_length",
                          h->n_head ? h->n_embd / h->n_head : 0);
    h->n_expert = key_u32(g, a + ".expert_count", 0);
    h->n_expert_used = key_u32(g, a + ".expert_used_count", 0);
    h->n_ff_exp = key_u32(g, a + ".expert_feed_forward_length",
                          key_u32(g, a + ".feed_forward_length", 0));
    h->n_ctx_train = key_u32(g, a + ".context_length", 4096);
    h->rms_eps = key_f32(g, a + ".attention.layer_norm_rms_epsilon", 1e-6f);
    h->rope_base = key_f32(g, a + ".rope.freq_base", 10000.0f);
    h->n_ff = key_u32(g, a + ".feed_forward_length", 0);
    h->n_ff_shexp = key_u32(g, a + ".expert_shared_feed_forward_length", 0);
    // Needed before the model is loaded, so the memory guard can tell the expert bytes from the
    // rest of the file. Overwritten with llama_n_vocab() after the load; the two agree.
    {
        const int tid = gguf_find_key(g, "tokenizer.ggml.tokens");
        h->n_vocab = (tid >= 0 && gguf_get_kv_type(g, tid) == GGUF_TYPE_ARRAY)
                         ? int(gguf_get_arr_n(g, tid)) : 0;
    }

    // ------------------------------------------------------------------------------------
    // Per-layer geometry.
    //
    // Filled uniformly first, from the scalars just read, and then overwritten where an
    // architecture says otherwise. Done in that order deliberately: a model this engine has
    // never seen still ends up with a complete, self-consistent table rather than a
    // half-filled one, so the refusal further on can be a refusal by name and not a crash.
    // ------------------------------------------------------------------------------------
    if (h->n_layer > 0) {
        // The rotated width. Absent for qwen3moe (where it equals head_dim), 64 of 256 for
        // qwen35moe. Read before the loop because every layer starts from it.
        const int n_rot = key_u32(g, a + ".rope.dimension_count", h->head_dim);
        h->layers.assign(std::size_t(h->n_layer), LayerGeom{});
        for (int il = 0; il < h->n_layer; ++il) {
            LayerGeom& l = h->L(il);
            l.kind      = LayerKind::ATTN;
            l.n_head    = h->n_head;
            l.n_head_kv = h->n_head_kv;
            l.head_dim  = h->head_dim;
            l.n_rot     = n_rot > 0 ? n_rot : h->head_dim;
            l.rope_base = h->rope_base;
        }

        // The value length, checked rather than assumed: the cache holds one head width per
        // layer, so a model whose V is narrower than its K would be silently truncated into
        // it. Both models we accept state the two and state them equal.
        const int v_len = key_u32(g, a + ".attention.value_length", h->head_dim);

        // ---- gemma4: two interleaved attention geometries, five windowed to one full.
        if (a == "gemma4") {
            std::vector<int> kv_arr, swa_arr;
            const int n_kv_arr  = key_arr_u32(g, a + ".attention.head_count_kv", &kv_arr,
                                              h->n_layer);
            const int n_swa_arr = key_arr_u32(g, a + ".attention.sliding_window_pattern",
                                              &swa_arr, h->n_layer);
            h->n_swa = key_u32(g, a + ".attention.sliding_window", 0);
            // The windowed layers carry their own head width AND their own rope base. Two
            // separate keys, and both matter: a windowed layer is 8 heads of 256 at base
            // 1e4, a full one is 2 heads of 512 at base 1e6.
            const int swa_k     = key_u32(g, a + ".attention.key_length_swa", h->head_dim);
            const int swa_v     = key_u32(g, a + ".attention.value_length_swa", swa_k);
            const int swa_rot   = key_u32(g, a + ".rope.dimension_count_swa", swa_k);
            const float swa_base = key_f32(g, a + ".rope.freq_base_swa", h->rope_base);
            if (swa_k != swa_v) {
                printf("gemma4: key_length_swa %d против value_length_swa %d - "
                       "движок хранит одну ширину головы на слой\n", swa_k, swa_v);
                gguf_free(g);
                return false;
            }
            for (int il = 0; il < h->n_layer; ++il) {
                LayerGeom& l = h->L(il);
                const bool windowed = n_swa_arr >= h->n_layer && swa_arr[std::size_t(il)] != 0;
                if (n_kv_arr >= h->n_layer) l.n_head_kv = kv_arr[std::size_t(il)];
                if (windowed) {
                    l.kind      = LayerKind::ATTN_SWA;
                    l.head_dim  = swa_k;
                    l.n_rot     = swa_rot;
                    l.rope_base = swa_base;
                }
            }
            // Gemma 4 states its attention scale as 1.0 and means it - there is no
            // 1/sqrt(head_dim) anywhere in its attention. The fork carries the same constant
            // with the same comment (llama-hparams.cpp: "Gemma4 uses self.scaling = 1.0").
            // Left as a named field rather than a branch in the graph builder because it is
            // a property of the file, and because 1.0 against 1/sqrt(512) degrades the
            // output instead of breaking it, which is the hardest kind of wrong to find.
            h->f_attn_scale    = 1.0f;
            h->f_logit_softcap = key_f32(g, a + ".final_logit_softcapping", 30.0f);
            h->f_embd_scale    = std::sqrt(float(h->n_embd));
            // The reference does not read this from the file either - it rescales the
            // tensor itself, once, right after loading (llama.cpp:4868 -> :3805).
            h->f_router_scale  = 1.0f / std::sqrt(float(h->n_embd));
        }

        // ---- qwen35moe: attention every fourth layer, a gated delta-net in between.
        if (a == "qwen35moe") {
            h->ssm_d_conv   = key_u32(g, a + ".ssm.conv_kernel", 0);
            h->ssm_d_inner  = key_u32(g, a + ".ssm.inner_size", 0);
            h->ssm_d_state  = key_u32(g, a + ".ssm.state_size", 0);
            h->ssm_dt_rank  = key_u32(g, a + ".ssm.time_step_rank", 0);
            h->ssm_n_group  = key_u32(g, a + ".ssm.group_count", 0);
            const int interval = key_u32(g, a + ".full_attention_interval", 4);
            std::vector<int> secs;
            if (key_arr_u32(g, a + ".rope.dimension_sections", &secs, 4) >= 4) {
                for (int i = 0; i < 4; ++i) h->rope_sections[i] = secs[std::size_t(i)];
            }
            for (int il = 0; il < h->n_layer; ++il) {
                // The fork's own rule, and note that it is (il+1), not il: layer 3 is the
                // first attention layer, not layer 0. Getting this off by one produces a
                // model whose every tensor lookup fails, which is at least loud.
                if (interval > 0 && ((il + 1) % interval) != 0) {
                    h->L(il).kind = LayerKind::DELTA_NET;
                }
            }
        }

        if (a != "gemma4" && v_len != h->head_dim) {
            printf("%s: key_length %d против value_length %d - движок хранит одну ширину "
                   "головы на слой\n", a.c_str(), h->head_dim, v_len);
            gguf_free(g);
            return false;
        }
    }

    gguf_free(g);
    return h->n_layer > 0 && h->n_embd > 0;
}

// The chat template as the file states it. Read straight out of the gguf rather than through
// llama_model_chat_template so it can be reported before the model is loaded, and so a file
// that simply has no template says so instead of producing malformed prompts.
std::string read_chat_template(const char* path) {
    gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g) return std::string();
    std::string out;
    const int id = gguf_find_key(g, "tokenizer.chat_template");
    if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING) {
        const char* s = gguf_get_val_str(g, id);
        if (s) out = s;
    }
    gguf_free(g);
    return out;
}

// Every tensor name the file holds. Only ever printed: when a lookup fails, "tensor not
// found" on its own has cost this project days, because the name that IS there is usually one
// character away from the name that was asked for.
std::vector<std::string> list_tensor_names(const char* path) {
    std::vector<std::string> names;
    gguf_init_params p = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g) return names;
    const int n = gguf_get_n_tensors(g);
    for (int i = 0; i < n; ++i) {
        const char* nm = gguf_get_tensor_name(g, i);
        if (nm) names.push_back(nm);
    }
    gguf_free(g);
    return names;
}

// Prints the names that ARE present, filtered to the layer the caller cares about so a
// 48-layer model does not dump six hundred lines.
void print_present_tensors(const char* path, const char* filter) {
    const std::vector<std::string> names = list_tensor_names(path);
    if (names.empty()) {
        printf("  в файле не удалось перечислить ни одного тензора\n");
        return;
    }
    printf("  в файле %zu тензоров; те, что содержат \"%s\":\n", names.size(), filter);
    int shown = 0;
    for (const std::string& n : names) {
        if (n.find(filter) == std::string::npos) continue;
        printf("    %s\n", n.c_str());
        if (++shown >= 40) {
            printf("    ... остальные не печатаю\n");
            break;
        }
    }
    if (shown == 0) {
        printf("    ни одного; первые двадцать имён вообще:\n");
        for (size_t i = 0; i < names.size() && i < 20; ++i) {
            printf("    %s\n", names[i].c_str());
        }
    }
}

// Rounded up to a whole number of 32-position blocks. Not tidiness: the fork's F16 matmul
// is silently wrong when the reduction length is not a multiple of four, and the value
// product reduces over positions.
int pad32(int n) { return (n + 31) / 32 * 32; }

// RMS norm followed by the learned scale, which is what every llama-family norm is.
ggml_tensor* norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

// The same thing as one op, which is what the reference's llm_build_norm actually calls
// (llama-build-context.cpp: LLM_NORM_RMS with a weight goes to ggml_fused_rms_norm).
//
// It matters that it is the same OP and not merely the same arithmetic. The fused kernel
// computes (scale*w[j])*x[j] and the pair above computes (scale*x[j])*w[j]; in f32 those
// differ in the last bit, which is harmless on its own and is exactly the kind of harmless
// that compounds through thirty layers into a divergence nobody can attribute afterwards.
// This project has already lost days to a 3-6% logit difference that turned out to be pure
// reassociation, so the two new architectures are built out of the reference's own ops
// wherever one exists.
//
// norm() above is deliberately left alone: the qwen3moe path is verified against the
// reference with it, and re-verifying that is a separate job from landing two architectures.
ggml_tensor* fnorm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_fused_rms_norm(c, x, w, eps);
}

// The output head, on whichever processor owns it.
//
// One function rather than a conditional at each site, because the two qwen3moe builders
// (the cacheless comparison prefill and the step) must not drift: the card has to compute the
// same quantity, from the same input, that ggml_mul_mat would have. `cur` is the already
// normalised hidden state in both, F32 and contiguous, and the returned tensor is
// [n_vocab, n_rows] F32 either way - so every consumer of g->logits is unchanged.
ggml_tensor* head_matmul(ggml_context* c, ggml_tensor* out_w, ggml_tensor* cur,
                         memex::GpuStatic* gstat) {
#ifdef MEMEX_FWD_GPU_EXPERTS
    if (gstat && gstat->on()) return gstat->head(c, cur);
#else
    (void)gstat;
#endif
    return ggml_mul_mat(c, out_w, cur);
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
    w->layers.resize(size_t(h.n_layer));
    for (int il = 0; il < h.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        Weights::Layer& L = w->layers[size_t(il)];
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

// The draft model: dense qwen3, which is the same attention block with a plain gated
// feed-forward in place of the expert dispatch. Kept as its own struct rather than as an
// optional arm of Weights, so nothing in the target path can accidentally read a null.
struct DenseWeights {
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
        ggml_tensor* up = nullptr;
        ggml_tensor* gate = nullptr;
        ggml_tensor* down = nullptr;
    };
    std::vector<Layer> layers;
};

bool collect_dense(llama_model* m, const HParams& h, DenseWeights* w) {
    bool ok = true;
    w->tok_embd = need(m, "token_embd.weight", &ok);
    w->out_norm = need(m, "output_norm.weight", &ok);
    w->out = llama_get_model_tensor(m, "output.weight");
    if (!w->out) {
        printf("черновик: output.weight отсутствует, беру token_embd (связанные веса)\n");
        w->out = w->tok_embd;
    }
    w->layers.resize(size_t(h.n_layer));
    for (int il = 0; il < h.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        DenseWeights::Layer& L = w->layers[size_t(il)];
        L.attn_norm = need(m, p + "attn_norm.weight", &ok);
        L.wq = need(m, p + "attn_q.weight", &ok);
        L.wk = need(m, p + "attn_k.weight", &ok);
        L.wv = need(m, p + "attn_v.weight", &ok);
        L.wo = need(m, p + "attn_output.weight", &ok);
        L.q_norm = need(m, p + "attn_q_norm.weight", &ok);
        L.k_norm = need(m, p + "attn_k_norm.weight", &ok);
        L.ffn_norm = need(m, p + "ffn_norm.weight", &ok);
        L.up = need(m, p + "ffn_up.weight", &ok);
        L.gate = need(m, p + "ffn_gate.weight", &ok);
        L.down = need(m, p + "ffn_down.weight", &ok);
    }
    return ok;
}

// Gemma 4 26B-A4B. Every layer carries BOTH a dense feed-forward and a routed one, and the
// two are combined through their own pair of post-norms rather than one replacing the other -
// so "gate/up/down" and "gate_up_exps/down_exps" are both present and both live.
struct Gemma4Weights {
    ggml_tensor* tok_embd = nullptr;
    ggml_tensor* out_norm = nullptr;
    ggml_tensor* out = nullptr;          // tied to tok_embd: the file has no output.weight
    struct Layer {
        ggml_tensor* attn_norm = nullptr;
        ggml_tensor* wq = nullptr;
        ggml_tensor* wk = nullptr;
        // Absent on the five full-attention layers, and that is the architecture rather than
        // a broken file: those layers use the K projection as their V. See collect_gemma4.
        ggml_tensor* wv = nullptr;
        ggml_tensor* wo = nullptr;
        ggml_tensor* q_norm = nullptr;
        ggml_tensor* k_norm = nullptr;
        ggml_tensor* post_attn_norm = nullptr;
        // The dense half
        ggml_tensor* ffn_norm = nullptr;
        ggml_tensor* gate = nullptr;
        ggml_tensor* up = nullptr;
        ggml_tensor* down = nullptr;
        // The routed half
        ggml_tensor* pre_ffw_norm_2 = nullptr;
        ggml_tensor* router = nullptr;
        ggml_tensor* router_scale = nullptr;    // rms weight for the ROUTER's own input
        ggml_tensor* gate_up_exps = nullptr;    // fused: [n_embd, 2*n_ff_exp, n_expert]
        ggml_tensor* down_exps = nullptr;
        ggml_tensor* down_scale = nullptr;      // [n_expert], folded into the routing weight
        // The join
        ggml_tensor* post_ffw_norm_1 = nullptr;
        ggml_tensor* post_ffw_norm_2 = nullptr;
        ggml_tensor* post_ffw_norm = nullptr;
        ggml_tensor* out_scale = nullptr;       // [1], multiplies the whole layer output
    };
    std::vector<Layer> layers;
};

// Qwen 3.6 35B-A3B. A hybrid: one layer in four is attention, the rest are a gated
// delta-net with a fixed-size recurrent state. Every layer, of either kind, has the same
// routed-plus-shared feed-forward hanging off it.
struct Qwen35Weights {
    ggml_tensor* tok_embd = nullptr;
    ggml_tensor* out_norm = nullptr;
    ggml_tensor* out = nullptr;
    struct Layer {
        ggml_tensor* attn_norm = nullptr;   // every layer
        // post_attention_norm under its own name, because that is not what it does: the file
        // has no ffn_norm, and the reference aliases layer.ffn_norm = layer.attn_post_norm
        // (llama-load-tensors.cpp). It is the feed-forward's PRE-norm. Nothing normalises the
        // attention output before its residual in this architecture.
        ggml_tensor* ffn_norm = nullptr;
        // Attention layers only. wq is DOUBLE width: per head it carries 256 of query then
        // 256 of output gate, interleaved, not two contiguous halves.
        ggml_tensor* wq = nullptr;
        ggml_tensor* wk = nullptr;
        ggml_tensor* wv = nullptr;
        ggml_tensor* wo = nullptr;
        ggml_tensor* q_norm = nullptr;
        ggml_tensor* k_norm = nullptr;
        // Delta-net layers only.
        ggml_tensor* wqkv = nullptr;        // [n_embd, 2*key_dim + value_dim], q|k|v in order
        ggml_tensor* wqkv_gate = nullptr;   // [n_embd, value_dim]
        ggml_tensor* ssm_conv1d = nullptr;  // [d_conv, conv_dim]
        ggml_tensor* ssm_dt = nullptr;      // [n_v_heads] bias
        ggml_tensor* ssm_a = nullptr;       // [n_v_heads], holds -exp(A_log): all negative
        ggml_tensor* ssm_beta = nullptr;    // [n_embd, n_v_heads]
        ggml_tensor* ssm_alpha = nullptr;   // [n_embd, n_v_heads]
        ggml_tensor* ssm_norm = nullptr;    // [head_v_dim], the gated output norm
        ggml_tensor* ssm_out = nullptr;     // [value_dim, n_embd]
        // Every layer.
        ggml_tensor* router = nullptr;
        ggml_tensor* gate_exps = nullptr;
        ggml_tensor* up_exps = nullptr;
        ggml_tensor* down_exps = nullptr;
        ggml_tensor* shexp_gate = nullptr;  // [n_embd] -> one sigmoid scalar per token
        ggml_tensor* gate_shexp = nullptr;
        ggml_tensor* up_shexp = nullptr;
        ggml_tensor* down_shexp = nullptr;
    };
    std::vector<Layer> layers;
};

// Gemma 4's tensors, plus the two per-layer facts that are stored as tensors rather than as
// metadata: whether the layer has its own V projection, and the rope divisors.
bool collect_gemma4(llama_model* m, HParams* h, Gemma4Weights* w) {
    bool ok = true;
    w->tok_embd = need(m, "token_embd.weight", &ok);
    w->out_norm = need(m, "output_norm.weight", &ok);
    w->out = llama_get_model_tensor(m, "output.weight");
    if (!w->out) {
        // Expected for this file rather than tolerated: gemma4 ships no output head.
        printf("gemma4: output.weight отсутствует, голова связана с token_embd\n");
        w->out = w->tok_embd;
    }
    // One model-level tensor, not one per layer - the name carries no %d. Its 256 entries are
    // 1.0 for the first 64 pairs and 1e30 for the remaining 192, and 1e30 drives theta to
    // zero: the full-attention layers therefore rotate only dimensions 0..127 of their 512.
    // Handing null instead does not fail, it rotates 384 dimensions that must not move.
    ggml_tensor* rope_freqs = llama_get_model_tensor(m, "rope_freqs.weight");
    w->layers.resize(size_t(h->n_layer));
    int n_shared_v = 0;
    for (int il = 0; il < h->n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        Gemma4Weights::Layer& L = w->layers[size_t(il)];
        LayerGeom& G = h->L(il);
        L.attn_norm = need(m, p + "attn_norm.weight", &ok);
        L.wq = need(m, p + "attn_q.weight", &ok);
        L.wk = need(m, p + "attn_k.weight", &ok);
        // Optional by architecture. The five layers without it are exactly the five
        // full-attention layers, and they take V from the K projection - the raw one, before
        // attn_k_norm and before the rotation, put through an unweighted rms_norm. Verified
        // against the file (layers 5, 11, 17, 23, 29) and against the reference's own branch
        // in build_gemma4.cpp, which does `Vcur = Kcur` at exactly that point.
        L.wv = llama_get_model_tensor(m, (p + "attn_v.weight").c_str());
        if (!L.wv) ++n_shared_v;
        L.wo = need(m, p + "attn_output.weight", &ok);
        L.q_norm = need(m, p + "attn_q_norm.weight", &ok);
        L.k_norm = need(m, p + "attn_k_norm.weight", &ok);
        L.post_attn_norm = need(m, p + "post_attention_norm.weight", &ok);
        L.ffn_norm = need(m, p + "ffn_norm.weight", &ok);
        L.gate = need(m, p + "ffn_gate.weight", &ok);
        L.up = need(m, p + "ffn_up.weight", &ok);
        L.down = need(m, p + "ffn_down.weight", &ok);
        L.pre_ffw_norm_2 = need(m, p + "pre_ffw_norm_2.weight", &ok);
        L.router = need(m, p + "ffn_gate_inp.weight", &ok);
        L.router_scale = need(m, p + "ffn_gate_inp.scale", &ok);
        L.gate_up_exps = need(m, p + "ffn_gate_up_exps.weight", &ok);
        L.down_exps = need(m, p + "ffn_down_exps.weight", &ok);
        L.down_scale = llama_get_model_tensor(m, (p + "ffn_down_exps.scale").c_str());
        L.post_ffw_norm_1 = need(m, p + "post_ffw_norm_1.weight", &ok);
        L.post_ffw_norm_2 = need(m, p + "post_ffw_norm_2.weight", &ok);
        L.post_ffw_norm = need(m, p + "post_ffw_norm.weight", &ok);
        L.out_scale = llama_get_model_tensor(m, (p + "layer_output_scale.weight").c_str());
        // The reference hands freq_factors only to the full-attention layers; the windowed
        // ones rotate their whole 256 at base 1e4 and get null.
        G.rope_freqs = G.kind == LayerKind::ATTN ? rope_freqs : nullptr;
    }
    if (!ok) return false;
    // The geometry the file states and the geometry its tensors actually have must agree, or
    // every view built from the first is a plausible misreading of the second. Checked once,
    // here, rather than discovered as a wrong answer.
    for (int il = 0; il < h->n_layer && ok; ++il) {
        const LayerGeom& G = h->L(il);
        const Gemma4Weights::Layer& L = w->layers[size_t(il)];
        const int64_t want_q = int64_t(G.d_q()), want_kv = int64_t(G.d_kv());
        if (L.wq->ne[1] != want_q || L.wk->ne[1] != want_kv || L.wo->ne[0] != want_q ||
            L.q_norm->ne[0] != G.head_dim || L.k_norm->ne[0] != G.head_dim ||
            (L.wv && L.wv->ne[1] != want_kv)) {
            printf("gemma4: слой %d — метаданные обещают %d голов по %d (kv %d), "
                   "а тензоры дают q %lld, k %lld, o %lld, q_norm %lld\n", il, G.n_head,
                   G.head_dim, G.n_head_kv, (long long)L.wq->ne[1], (long long)L.wk->ne[1],
                   (long long)L.wo->ne[0], (long long)L.q_norm->ne[0]);
            ok = false;
        }
    }
    int n_win = 0;
    for (int il = 0; il < h->n_layer; ++il) {
        if (h->L(il).kind == LayerKind::ATTN_SWA) ++n_win;
    }
    // Printed rather than trusted. The two counts have to agree - the layers without a V
    // projection are precisely the full-attention ones - and a file where they do not is a
    // file this builder is about to misread.
    printf("gemma4: слоёв %d — с окном %d, полных %d; V берётся из K на %d слоях; "
           "rope_freqs %s\n", h->n_layer, n_win, h->n_layer - n_win, n_shared_v,
           rope_freqs ? "есть" : "НЕТ (полные слои будут вращаться целиком — это ошибка)");
    if (n_shared_v && n_shared_v != h->n_layer - n_win) {
        printf("gemma4: %d слоёв без attn_v против %d полных слоёв — это не та раскладка, "
               "под которую написан граф\n", n_shared_v, h->n_layer - n_win);
        ok = false;
    }
    return ok;
}

// Qwen 3.6's tensors. Which set a layer has is decided by its kind, which came from the
// metadata - so a disagreement between the two shows up here as a missing tensor with a
// name, rather than later as a shape that happens to fit.
bool collect_qwen35(llama_model* m, const HParams& h, Qwen35Weights* w) {
    bool ok = true;
    w->tok_embd = need(m, "token_embd.weight", &ok);
    w->out_norm = need(m, "output_norm.weight", &ok);
    w->out = llama_get_model_tensor(m, "output.weight");
    if (!w->out) {
        printf("qwen35moe: output.weight отсутствует, беру token_embd (связанные веса)\n");
        w->out = w->tok_embd;
    }
    w->layers.resize(size_t(h.n_layer));
    for (int il = 0; il < h.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        Qwen35Weights::Layer& L = w->layers[size_t(il)];
        const LayerGeom& G = h.L(il);
        L.attn_norm = need(m, p + "attn_norm.weight", &ok);
        L.ffn_norm = need(m, p + "post_attention_norm.weight", &ok);
        if (G.kind == LayerKind::DELTA_NET) {
            L.wqkv = need(m, p + "attn_qkv.weight", &ok);
            L.wqkv_gate = need(m, p + "attn_gate.weight", &ok);
            L.ssm_conv1d = need(m, p + "ssm_conv1d.weight", &ok);
            L.ssm_dt = need(m, p + "ssm_dt.bias", &ok);
            L.ssm_a = need(m, p + "ssm_a", &ok);          // no .weight suffix in this arch
            L.ssm_beta = need(m, p + "ssm_beta.weight", &ok);
            L.ssm_alpha = need(m, p + "ssm_alpha.weight", &ok);
            L.ssm_norm = need(m, p + "ssm_norm.weight", &ok);
            L.ssm_out = need(m, p + "ssm_out.weight", &ok);
        } else {
            L.wq = need(m, p + "attn_q.weight", &ok);
            L.wk = need(m, p + "attn_k.weight", &ok);
            L.wv = need(m, p + "attn_v.weight", &ok);
            L.wo = need(m, p + "attn_output.weight", &ok);
            L.q_norm = need(m, p + "attn_q_norm.weight", &ok);
            L.k_norm = need(m, p + "attn_k_norm.weight", &ok);
        }
        L.router = need(m, p + "ffn_gate_inp.weight", &ok);
        L.gate_exps = need(m, p + "ffn_gate_exps.weight", &ok);
        L.up_exps = need(m, p + "ffn_up_exps.weight", &ok);
        L.down_exps = need(m, p + "ffn_down_exps.weight", &ok);
        L.shexp_gate = need(m, p + "ffn_gate_inp_shexp.weight", &ok);
        L.gate_shexp = need(m, p + "ffn_gate_shexp.weight", &ok);
        L.up_shexp = need(m, p + "ffn_up_shexp.weight", &ok);
        L.down_shexp = need(m, p + "ffn_down_shexp.weight", &ok);
    }
    if (!ok) return false;
    // The attention layers' Q projection is twice as wide as its head count implies, because
    // half of every head's rows are an output gate. Checked rather than assumed: if a future
    // file ever stopped fusing the gate, the views below would silently take the first half
    // of the queries and call it all of them.
    for (int il = 0; il < h.n_layer && ok; ++il) {
        const LayerGeom& G = h.L(il);
        if (G.kind == LayerKind::DELTA_NET) continue;
        const Qwen35Weights::Layer& L = w->layers[size_t(il)];
        if (L.wq->ne[1] != int64_t(2 * G.d_q())) {
            printf("qwen35moe: слой %d — attn_q даёт %lld строк, а ожидались %d "
                   "(запрос и выходной вентиль вперемежку по головам)\n", il,
                   (long long)L.wq->ne[1], 2 * G.d_q());
            ok = false;
        }
    }
    return ok;
}

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_backend_buffer_t inputs = nullptr;
    ggml_tensor* tokens = nullptr;      // [n_tokens] i32
    ggml_tensor* positions = nullptr;   // [n_tokens] i32
    ggml_tensor* mask = nullptr;        // [n_kv, n_tokens] f32
    // The windowed mask, built only when some layer is windowed. Two masks rather than one
    // per layer because gemma4 has exactly two attention regimes and thirty layers: the
    // fork builds the same pair once and hands each layer whichever one it wants.
    ggml_tensor* mask_swa = nullptr;    // [n_kv, n_tokens] f32, or null
    // Which state slot each token belongs to, for ggml_ssm_conv. One sequence is all this
    // engine has ever generated, so it is [1, n_tokens] of zeros - but the op takes it as an
    // input rather than a constant, and it has to live in the input buffer to be written.
    ggml_tensor* seq_ids = nullptr;     // [1, n_tokens] i32, or null
    // Per-slot relative threshold for expert reduction: zero for the slots that are always
    // kept, the requested threshold for the rest.
    ggml_tensor* slot_thresh = nullptr;
    ggml_tensor* logits = nullptr;
    // Our own intermediates under the fork's own names, so a comparison can be asked for by
    // name instead of by hand-counted node index.
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
    // The same thing for the zoned cache's exact zones, aimed at a *slot* rather than a
    // position: the zone map is fixed but the window is a ring, so the slot the arriving
    // position occupies is chosen by the cache and handed back by append_live, and it is not
    // the position. Kept in its own list because the two are aimed at different numbers on
    // the same step, and a single list would have to be told which is which.
    std::vector<CacheWrite> zwrites;
    // The key and value this step computed, per layer, kept alive as graph outputs so the
    // host can mirror them into the zoned cache after the graph has run. Kc is
    // [head_dim, 1, kv_head] and Vc is [1, head_dim, kv_head], both contiguous, so each is
    // exactly d_kv halves in [kv_head][head_dim] order - which is the order the module
    // documents its rows in, so no repacking happens on the way.
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> rows;
    // Per-layer (input to the layer, attention output before the output projection), kept
    // only when a comparison asked for them. Final logits tell you that two paths differ;
    // these tell you at which layer, which is the difference between "the tail is lossy" and
    // "the wiring is wrong". The same discipline memex-test's layer sweep uses.
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> dbg;

    // The read side of attention, per layer. Recorded for exactly the reason the writes are:
    // the number of cached positions a step attends over is NOT a build-time constant, it is
    // a per-step number, and baking the cache's allocated length into the views makes every
    // step read the whole allocation. The reference sizes its own KV views from the batch's
    // n_kv - the occupied cells rounded up to a padding - and not from the cache length; this
    // engine builds its graph once, so it records the four tensors whose extent that number
    // sets and moves them instead of rebuilding.
    struct KvRead {
        ggml_tensor* K = nullptr;      // [head_dim, n_kv, n_head_kv], a view of the cache
        ggml_tensor* V = nullptr;      // [n_kv, head_dim, n_head_kv], transposed in the cache
        ggml_tensor* kq = nullptr;     // [n_kv, n_tokens, n_head], the scores
        ggml_tensor* probs = nullptr;  // the softmax over them, same shape
        // The mask this layer's softmax reads. Recorded per layer rather than assumed to be
        // the graph's one mask, because a windowed layer reads the windowed one and the two
        // have to be re-aimed together or a step reads the cache at one length and its mask
        // at another - which is not an error, it is a wrong answer.
        ggml_tensor* mask = nullptr;
    };
    std::vector<KvRead> reads;
    int64_t n_kv_built = 0;            // the extent the graph was reserved and allocated at

    // The resident expert set's three graph-side pieces.
    //
    // `res_mask` is an input, rewritten between steps: [n_expert, n_layer] f32, 1.0 where the
    // expert is resident in that layer. A dense vector rather than a list of ids because the
    // graph has to look residency up by the router's own output, and that output does not
    // exist until the graph has run the router - so the lookup is a ggml_get_rows over this,
    // gathered by the top-k ids.
    //
    // `res_ids` and `oth_ids` are outputs, one pair per layer: the ids the resident half was
    // actually dispatched with and the ids the other half got, each with -1 in every slot the
    // other half owns. Read back rather than recomputed on the host, so the hit accounting
    // counts what the graph did rather than what a model of the graph would have done - and so
    // the two can be compared against the host's own bitset, which is the only check that the
    // mask arrived where it was meant to.
    //
    // `sel_ids` belongs to the unsplit path: a contiguous copy of the router's top-k, so the
    // policy can be warmed on the prompt during the prefill. Without it the first third of a
    // short run reports a hit rate measured against an empty window.
    ggml_tensor* res_mask = nullptr;
    std::vector<ggml_tensor*> res_ids;
    std::vector<ggml_tensor*> oth_ids;
    std::vector<ggml_tensor*> sel_ids;

    ggml_tensor* probe(const std::string& name) const {
        for (const auto& p : probes) {
            if (p.first == name) return p.second;
        }
        return nullptr;
    }

    // Re-aims every cache write at `n_past`. Allocation resolved each view to a raw pointer
    // (view_src->data + view_offs) and ggml_cpy's own result node is a second view of the
    // same cache carrying the same offset, so both the offset and the resolved pointer have
    // to move on both tensors - which is exactly the four fields the reference patches in
    // llama_context::update_cache_copies() to reuse its graphs.
    static bool aim_writes(const std::vector<CacheWrite>& ws, int at) {
        if (ws.empty()) return false;
        for (const CacheWrite& c : ws) {
            // A silent mismatch here would write keys into whatever the pointer happened to
            // reach, so it is checked rather than assumed.
            if (!c.cpy || !c.dst || !c.cache || c.cpy->op != GGML_OP_CPY ||
                c.cpy->src[1] != c.dst || c.cpy->view_src != c.cache ||
                c.dst->view_src != c.cache || !c.cache->data ||
                at < 0 || at + c.n_write > c.n_pos) {
                return false;
            }
            const std::size_t offs = std::size_t(at) * c.step;
            c.dst->view_offs = offs;
            c.dst->data      = (char*)c.cache->data + offs;
            c.cpy->view_offs = offs;
            c.cpy->data      = c.dst->data;
        }
        return true;
    }

    // True when the card owns the whole attention block, so this graph has no cache writes
    // of its own to re-aim: GpuStatic::set_step aims the device graphs' write views instead.
    // Kept here rather than at each call site because there are six of them and the failure
    // of forgetting one is a refusal that reads like a bug in the cache arithmetic.
    bool on_card = false;

    bool aim_cache_writes(int n_past) {
        if (on_card) return true;
        return aim_writes(writes, n_past);
    }
    // The slot append_live handed back, not a position. Same patch, different number, and
    // the distinction is the whole reason there are two lists.
    bool aim_zoned_writes(int slot) { return aim_writes(zwrites, slot); }

    // Re-aims the read side of attention at `n_kv` cached positions, which is what makes a
    // decode step cost the positions that are occupied rather than the positions that were
    // allocated. Every tensor whose extent that number sets is either a view of the cache -
    // where the strides belong to the cache and only the extent moves - or a contiguous
    // intermediate, where the extent moves and the strides follow from it.
    //
    // Nothing is reallocated and nothing needs to be: the graph was reserved and allocated at
    // n_kv_built, so every buffer is already large enough, and shrinking a tensor cannot make
    // two of them overlap. This is the read-side twin of aim_writes, and the reason the graph
    // can still be built once.
    //
    // n_kv must stay a multiple of 32. It is the reduction length of the V*probs matmul, and
    // this fork's F16 matmul is silently wrong when that length is not a multiple of four;
    // the padding positions are read and multiplied by the zero the -inf mask puts into
    // probs, so they cost bytes but never correctness. Everything is validated before
    // anything is patched, so a refusal leaves the graph exactly as it was - reading the
    // whole allocation, which is slow and right rather than fast and wrong.
    bool aim_kv_reads(int n_kv) {
        if (reads.empty()) return true;   // the zoned path reads through its own subgraph
        if (!mask || n_kv <= 0 || n_kv % 32 != 0 || int64_t(n_kv) > n_kv_built) return false;
        for (const KvRead& r : reads) {
            if (!r.K || !r.V || !r.kq || !r.probs || !r.mask) return false;
            // All four carry the extent that was aimed at last, and so does the mask. A
            // disagreement means something else moved them, and patching from here would
            // then read the cache at one length and the mask at another.
            if (r.K->ne[1] != r.mask->ne[0] || r.V->ne[0] != r.mask->ne[0] ||
                r.kq->ne[0] != r.mask->ne[0] || r.probs->ne[0] != r.mask->ne[0]) return false;
            // Both masks are aimed at the same extent, always. The window is expressed in
            // the mask's CONTENTS, not in its width - narrowing the windowed layers' reads
            // to n_swa instead would be a real saving and a different change, and doing it
            // by accident here would silently drop the sink positions a windowed layer
            // still attends to.
            if (r.mask != mask && r.mask != mask_swa) return false;
        }
        if (mask->ne[0] == int64_t(n_kv)) return true;
        // A contiguous tensor's strides are its extents, so they have to be recomputed and
        // not merely left alone: soft_max asserts its result is contiguous, and mul_mat reads
        // nb1 as the destination row stride.
        auto restride = [](ggml_tensor* t, int64_t n0) {
            t->ne[0] = n0;
            t->nb[1] = t->nb[0] * std::size_t(t->ne[0]);
            t->nb[2] = t->nb[1] * std::size_t(t->ne[1]);
            t->nb[3] = t->nb[2] * std::size_t(t->ne[2]);
        };
        restride(mask, n_kv);
        if (mask_swa) restride(mask_swa, n_kv);
        for (const KvRead& r : reads) {
            r.K->ne[1] = n_kv;   // strides are the cache's own, so these two do not move
            r.V->ne[0] = n_kv;
            restride(r.kq, n_kv);
            restride(r.probs, n_kv);
        }
        return true;
    }

    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (inputs) ggml_backend_buffer_free(inputs);
        if (ctx) ggml_free(ctx);
        alloc = nullptr; inputs = nullptr; ctx = nullptr;
    }
};

// Every per-step input a graph has, written in one place.
//
// It exists as a free function because three callers need identical bytes: the generator's
// decode step, its prefill, and the one-shot comparison in main. They used to fill the mask
// separately and the windowed mask would have been a fourth place to forget.
//
// The causal mask is -inf everywhere and zero only where a query at position past+i may see
// key j. The windowed mask is that AND the window: a query may see key j only while
// past+i - j < n_swa, which is the reference's own condition (llama.cpp:5590) and carries no
// attention sinks - gemma4 has none, and inventing them here would be a different model.
void set_graph_inputs(Graph& gr, const HParams& h, const llama_token* tk, int nt, int past,
                      std::vector<int32_t>* ps, std::vector<float>* mk) {
    const int nkv = int(gr.mask->ne[0]);
    ps->assign(std::size_t(nt), 0);
    for (int i = 0; i < nt; ++i) (*ps)[std::size_t(i)] = past + i;
    ggml_backend_tensor_set(gr.tokens, tk, 0, sizeof(int32_t) * std::size_t(nt));
    ggml_backend_tensor_set(gr.positions, ps->data(), 0, sizeof(int32_t) * std::size_t(nt));

    mk->assign(std::size_t(nkv) * std::size_t(nt), -INFINITY);
    for (int i = 0; i < nt; ++i) {
        for (int j = 0; j <= past + i && j < nkv; ++j) {
            (*mk)[std::size_t(i) * std::size_t(nkv) + std::size_t(j)] = 0.0f;
        }
    }
    ggml_backend_tensor_set(gr.mask, mk->data(), 0, ggml_nbytes(gr.mask));

    if (gr.mask_swa) {
        mk->assign(std::size_t(nkv) * std::size_t(nt), -INFINITY);
        for (int i = 0; i < nt; ++i) {
            const int pos = past + i;
            const int lo = h.n_swa > 0 ? std::max(0, pos - h.n_swa + 1) : 0;
            for (int j = lo; j <= pos && j < nkv; ++j) {
                (*mk)[std::size_t(i) * std::size_t(nkv) + std::size_t(j)] = 0.0f;
            }
        }
        ggml_backend_tensor_set(gr.mask_swa, mk->data(), 0, ggml_nbytes(gr.mask_swa));
    }

    if (gr.seq_ids) {
        // One sequence, always. Written rather than left uninitialised because ggml_ssm_conv
        // reads it as an index and asserts 0 <= sq[0] < n_kv - and the graph's input buffer
        // is not zeroed for us.
        std::vector<int32_t> z(std::size_t(nt), 0);
        ggml_backend_tensor_set(gr.seq_ids, z.data(), 0, ggml_nbytes(gr.seq_ids));
    }
}

// One prefill over the whole prompt. No cache: attention runs over the prompt itself under
// a causal mask, which is exactly what a prefill is, and it is enough to compare logits.
bool build(Graph* g, ggml_backend_buffer_type_t buft, const HParams& h, const Weights& w,
           int n_tokens, memex::GpuStatic* gstat = nullptr) {
    const size_t n_nodes = size_t(h.n_layer) * 64 + 256;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 512) +
                           ggml_graph_overhead_custom(n_nodes, false),
                           nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) return false;
    ggml_context* c = g->ctx;

    // Inputs live in their own buffer so they can be written between runs.
    g->tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_tokens, n_tokens);
    g->inputs = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!g->inputs) return false;

    g->gf = ggml_new_graph_custom(c, n_nodes, false);
    const int hd = h.head_dim;
    const float kq_scale = 1.0f / std::sqrt(float(hd));
    // The reference calls ggml_rope_multi, and for a text-only model the section widths are
    // all zero. Calling ggml_rope_ext instead looks equivalent and is not: it left a 5%
    // difference in the logits while the argmax still matched, which is exactly the kind of
    // near-miss that would have been written off as accumulation order.
    int sections[GGML_MROPE_SECTIONS] = {0};

    ggml_tensor* cur = ggml_get_rows(c, w.tok_embd, g->tokens);
    for (int il = 0; il < h.n_layer; ++il) {
        const Weights::Layer& L = w.layers[size_t(il)];
        ggml_tensor* inpSA = cur;

        ggml_tensor* x = norm(c, cur, L.attn_norm, h.rms_eps);
        g->probes.push_back({"attn_norm-" + std::to_string(il), x});

        ggml_tensor* q = ggml_mul_mat(c, L.wq, x);
        ggml_tensor* k = ggml_mul_mat(c, L.wk, x);
        ggml_tensor* v = ggml_mul_mat(c, L.wv, x);

        // Per-head norm before the rotation, in that order. Reversing them changes the
        // answer, because the rotation is not scale-invariant per pair.
        q = ggml_reshape_3d(c, q, hd, h.n_head, n_tokens);
        q = norm(c, q, L.q_norm, h.rms_eps);
        q = ggml_rope_multi(c, q, g->positions, nullptr, hd, sections, h.rope_type,
                            h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        k = ggml_reshape_3d(c, k, hd, h.n_head_kv, n_tokens);
        k = norm(c, k, L.k_norm, h.rms_eps);
        k = ggml_rope_multi(c, k, g->positions, nullptr, hd, sections, h.rope_type,
                            h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Heads into the last dimension for both sides, so mul_mat broadcasts the four
        // kv-heads across the query heads by itself. Doing this with strided
        // two-dimensional views instead was measured to give a wrong answer.
        ggml_tensor* Q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
        // Keys and values in half precision, which is what a cache holds and what the
        // reference attends against. Keeping them in f32 was measured as the entire source
        // of divergence: 0.146% per layer, compounding to 4.9974% on the logits after 48 of
        // them, with the argmax still matching all the way. So the difference was not a bug
        // in either side - our graph was simply more precise than the thing it was checked
        // against, and half precision is the right answer anyway because it halves the bytes
        // attention re-reads.
        ggml_tensor* K = ggml_cast(c,
            ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3)), GGML_TYPE_F16);
        ggml_tensor* V = ggml_cast(c, ggml_cont(c, ggml_permute(
            c, ggml_reshape_3d(c, v, hd, h.n_head_kv, n_tokens), 1, 2, 0, 3)),
            GGML_TYPE_F16);

        ggml_tensor* kq = ggml_mul_mat(c, K, Q);
        ggml_tensor* p = ggml_soft_max_ext(c, kq, g->mask, kq_scale, 0.0f);
        ggml_tensor* kqv = ggml_mul_mat(c, V, p);
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), h.d_q(), n_tokens);
        cur = ggml_add(c, ggml_mul_mat(c, L.wo, kqv), inpSA);

        ggml_tensor* ffn_inp = cur;
        g->probes.push_back({"attn_out_resid-" + std::to_string(il), ffn_inp});
        x = norm(c, ffn_inp, L.ffn_norm, h.rms_eps);
        g->probes.push_back({"ffn_inp_normed-" + std::to_string(il), x});

        // Routing. The fork's own path is reproduced exactly: softmax over all experts,
        // top-k, then renormalise the chosen weights - Qwen3 normalises, and skipping that
        // scales every expert output by the same wrong factor.
        ggml_tensor* logits_e = ggml_mul_mat(c, L.router, x);
        ggml_tensor* probs = ggml_soft_max(c, logits_e);
        ggml_tensor* sel = ggml_top_k(c, probs, h.n_expert_used);
        ggml_tensor* weights = ggml_get_rows(c,
            ggml_reshape_3d(c, probs, 1, h.n_expert, n_tokens), sel);
        weights = ggml_reshape_2d(c, weights, h.n_expert_used, n_tokens);
        ggml_tensor* wsum = ggml_sum_rows(c, weights);
        weights = ggml_div(c, weights, wsum);
        weights = ggml_reshape_3d(c, weights, 1, h.n_expert_used, n_tokens);

        ggml_tensor* xe = ggml_reshape_3d(c, x, h.n_embd, 1, n_tokens);
        ggml_tensor* up = ggml_mul_mat_id(c, L.up, xe, sel);
        ggml_tensor* gate = ggml_mul_mat_id(c, L.gate, xe, sel);
        gate = ggml_silu(c, gate);
        ggml_tensor* act = ggml_mul(c, up, gate);
        ggml_tensor* out = ggml_mul_mat_id(c, L.down, act, sel);
        out = ggml_mul(c, out, weights);
        // Sum the chosen experts' contributions: they arrive as n_expert_used slices.
        ggml_tensor* moe = ggml_view_2d(c, out, h.n_embd, n_tokens,
                                        out->nb[2], 0);
        for (int e = 1; e < h.n_expert_used; ++e) {
            moe = ggml_add(c, moe,
                           ggml_view_2d(c, out, h.n_embd, n_tokens, out->nb[2],
                                        size_t(e) * out->nb[1]));
        }
        g->probes.push_back({"ffn_moe_out-" + std::to_string(il), moe});
        cur = ggml_add(c, moe, ffn_inp);
        g->probes.push_back({"l_out-" + std::to_string(il), cur});
    }

    cur = norm(c, cur, w.out_norm, h.rms_eps);
    g->probes.push_back({"result_norm", cur});
    g->logits = head_matmul(c, w.out, cur, gstat);
    ggml_set_output(g->logits);
    for (auto& p : g->probes) {
        ggml_set_output(p.second);      // keep them alive so they can be read back
    }
    ggml_build_forward_expand(g->gf, g->logits);
    g->alloc = ggml_gallocr_new(buft);
    return ggml_gallocr_reserve(g->alloc, g->gf) &&
           ggml_gallocr_alloc_graph(g->alloc, g->gf);
}

// The cache, ours, preallocated for the whole context. Keys as [head_dim, n_ctx,
// n_kv_heads] and values transposed within each head as [n_ctx, head_dim, n_kv_heads], both
// in half precision - which is what the reference holds and what halves the bytes attention
// re-reads. Written once per position and never moved.
//
// Sized per layer rather than once for the model. That is not generality for its own sake:
// gemma4's windowed layers hold 8 heads of 256 and its full layers 2 of 512, so one shape
// for all thirty would over-allocate five of them and under-allocate twenty-five; and
// qwen35moe's thirty delta-net layers hold no keys at all, so a uniform cache would reserve
// three quarters of its bytes for tensors nothing ever writes. Those layers get a null here,
// which every reader has to check - and does, because a null dereferences loudly where a
// zero-sized tensor would quietly read nothing.
struct Cache {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> k, v;
    int n_ctx = 0;

    bool init(ggml_backend_buffer_type_t buft, const HParams& h, int ctx_len) {
        n_ctx = ctx_len;
        ggml_init_params ip = {ggml_tensor_overhead() * size_t(h.n_layer) * 2 + 4096,
                               nullptr, true};
        ctx = ggml_init(ip);
        if (!ctx) return false;
        k.assign(size_t(h.n_layer), nullptr);
        v.assign(size_t(h.n_layer), nullptr);
        for (int il = 0; il < h.n_layer; ++il) {
            const LayerGeom& L = h.L(il);
            if (!L.has_kv()) continue;   // a delta-net layer carries a state, not a cache
            k[size_t(il)] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, L.head_dim, n_ctx,
                                               L.n_head_kv);
            v[size_t(il)] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_ctx, L.head_dim,
                                               L.n_head_kv);
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        return buf != nullptr;
    }

    std::size_t bytes(const HParams& h) const {
        return size_t(n_ctx) * h.kv_elems_per_pos() * 2;
    }

    void free_all() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr;
    }
};

// The gated delta-net's carried state, which is what qwen35moe has instead of a KV cache on
// three quarters of its layers.
//
// The shape of the thing is the point: it is FIXED. A KV cache grows one position per token
// and a decode step re-reads all of it; this is a convolution window of three positions and
// a 128x128 matrix per head, 2.20 MB a layer, and it costs the same at position 1 and at
// position 100000. Nothing here is re-aimed per step for the same reason - there is no
// offset to move, the graph writes the same slot every time.
//
// It lives in its own backend buffer rather than in the graph allocator's, because the graph
// both reads it at the top of a layer and writes it at the bottom, and the value has to
// survive between computations. Which also means it must be cleared before a prefill: a
// second prompt run against a state left over from the first is not an error, it is a wrong
// answer that looks like a distracted model.
//
// FOR WHOEVER MOVES THESE LAYERS ONTO THE CARD - read this before you design the transfer,
// because the obvious design is the wrong one.
//
// The boundary is one handoff per layer. Anything finer loses: a crossing costs about 177 us
// of host-side coordination (thread wakeup, mutex, condition variable), measured separately
// from the GPU work. On the 48-layer model the card plan is priced against, a per-block cut
// is roughly 4 crossings x 48 x 177 us = 34 ms per token - more than the whole win - and a
// per-node cut is 170 ms, which is absurd. (48 is that plan's reference model; this file's
// two new architectures have 30 and 40 layers, and the conclusion is the same for both.)
// That is why the card runs the ENTIRE layer graph - norms, ropes, softmaxes and all, at
// about 3 us each on the device - rather than only the parts with bytes in them. Cheap in
// bytes is not cheap in place: the cost of a norm on the host is the boundary it creates,
// not the work it does.
//
// Which puts this buffer on the card, and it must STAY there. It is not a value to be
// mirrored to the host each step. Two facts make that unambiguous:
//
//   - The size is fixed and small: 2.20 MB a layer, 65.9 MB for all thirty. It fits on any
//     device that could run the model at all, so there is no reason to evict it.
//   - It is read AND written every token, in place, by ggml_delta_net and ggml_ssm_conv.
//     Shuttling it would add 65.9 MB of traffic in each direction per token to save nothing,
//     and - worse - would put a second boundary crossing inside every one of thirty layers,
//     which is exactly the per-block cut that the arithmetic above rules out.
//
// So: device-resident, updated in place, never copied host-ward except when a human asks to
// inspect it. `clear()` below is the one host-side write, and it happens once per prompt
// rather than once per token.
struct DeltaState {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> s;     // per layer, [state_dim, 1] f32, or null
    std::size_t state_dim = 0;

    bool init(ggml_backend_buffer_type_t buft, const HParams& h) {
        state_dim = h.delta_state_elems();
        s.assign(std::size_t(h.n_layer), nullptr);
        if (state_dim == 0 || h.n_delta_layers() == 0) return true;   // nothing to carry
        ggml_init_params ip = {ggml_tensor_overhead() * std::size_t(h.n_layer) + 4096,
                               nullptr, true};
        ctx = ggml_init(ip);
        if (!ctx) return false;
        for (int il = 0; il < h.n_layer; ++il) {
            if (h.L(il).kind != LayerKind::DELTA_NET) continue;
            // Two dimensions rather than one so that nb[1] == ne[0]*nb[0] exactly, which is
            // what makes the sub-views below genuinely contiguous rather than accidentally
            // so. ggml_reshape asserts contiguity and a one-row view of a wider tensor is
            // not the same thing.
            s[std::size_t(il)] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                    int64_t(state_dim), 1);
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        return buf != nullptr;
    }

    // Back to no history at all. Called before every prefill, and before any run that is not
    // a continuation of the one before it.
    void clear() {
        if (!buf || state_dim == 0) return;
        std::vector<float> z(state_dim, 0.0f);
        for (ggml_tensor* t : s) {
            if (t) ggml_backend_tensor_set(t, z.data(), 0, ggml_nbytes(t));
        }
    }

    std::size_t bytes() const {
        std::size_t n = 0;
        for (const ggml_tensor* t : s) {
            if (t) n += std::size_t(state_dim) * sizeof(float);
        }
        return n;
    }

    void free_all() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr;
        s.clear();
    }
};

// ---------------------------------------------------------------------------------------
// The zoned cache, as an option
// ---------------------------------------------------------------------------------------

// The flags, under the module's own names so there is one vocabulary rather than two. Off by
// default, and the default values are the module's own defaults except for the notebook: the
// notebook is positions the *engine* asked to keep exactly, and this engine has no signal to
// ask with yet, so reserving thirty-two exact slots nothing will ever occupy would spend
// bytes on an empty zone. It stays reachable through --notebook for whoever wires a signal.
struct ZonedOpt {
    bool on = false;         // --zoned
    bool check = false;      // --zoned-check: run both caches and print the argmax margin
    int sinks = 4;           // --sinks
    int window = 256;        // --window
    int notebook = 0;        // --notebook
    bool tail_q8 = true;     // --tail-form q8_0 | f16
    bool rotate_keys = true; // --rotate-keys on | off

    int exact_slots() const { return sinks + notebook + window; }
    const char* tail_name() const {
        return tail_q8 ? (rotate_keys ? "q8_0+H" : "q8_0") : "f16";
    }
};

// Bytes attention re-reads for one token, over the whole model, at a given context length.
// Modelled rather than measured, because the point of printing it is to answer a question
// about context lengths this run is not at: at 2048 positions with a 512 window there is
// barely a tail and zoning looks pointless, and someone reading only the current run's
// figure deletes the option. The model is not a guess - it is the same arithmetic
// ZonedCache::read_bytes does, over occupied positions only.
struct KvBytes {
    double exact = 0.0;
    double zoned = 0.0;
    int n_exact = 0;   // positions kept exact at this context length
    int n_tail = 0;    // positions compressed
    double ratio() const { return zoned > 0.0 ? exact / zoned : 0.0; }
};

KvBytes kv_read_bytes(const HParams& h, int n_kv, const ZonedOpt& z) {
    KvBytes b;
    const double d_kv = double(h.d_kv());
    const double layers = double(h.n_layer);
    // Two sides, K and V, half precision.
    b.exact = layers * 2.0 * double(n_kv) * d_kv * 2.0;
    b.n_exact = std::min(n_kv, z.exact_slots());
    b.n_tail = n_kv - b.n_exact;
    // Q8_0 is 34 bytes per block of 32, so a position costs d_kv/32*34 per side instead of
    // d_kv*2 - a shade under half. The block axis differs between K and V (head_dim for K,
    // positions for V) but the per-position cost is the same either way once a block is full.
    const double tail_per_pos = z.tail_q8
        ? double(h.n_head_kv) * std::ceil(double(h.head_dim) / 32.0) * 34.0
        : d_kv * 2.0;
    b.zoned = layers * 2.0 * (double(b.n_exact) * d_kv * 2.0 + double(b.n_tail) * tail_per_pos);
    return b;
}

void print_kv_zoning(const HParams& h, int n_kv, const ZonedOpt& z) {
    printf("  KV при зонировании (стоки %d, блокнот %d, окно %d, хвост %s):\n", z.sinks,
           z.notebook, z.window, z.tail_name());
    printf("    %8s %11s %11s %8s  %s\n", "контекст", "точный KV", "зонный KV", "экономия",
           "точных/сжатых позиций");
    // The current context first, then a few longer ones. The saving is a property of the
    // *tail*, and the tail is what a short context has none of, so one figure alone is
    // either flattering or damning by accident.
    int lens[6] = {n_kv, 2048, 4096, 8192, 16384, 32768};
    for (int i = 0; i < 6; ++i) {
        if (lens[i] <= 0) continue;
        if (i > 0 && lens[i] == n_kv) continue;
        const KvBytes b = kv_read_bytes(h, lens[i], z);
        printf("    %8d %8.1f МБ %8.1f МБ %7.2fx  %d / %d%s\n", lens[i], b.exact / 1e6,
               b.zoned / 1e6, b.ratio(), b.n_exact, b.n_tail,
               b.n_tail == 0 ? "  (хвоста нет — зонировать нечего)" : "");
    }
}

#ifdef MEMEX_FWD_ZONED
// The zoned cache with its own ggml context and backend buffer, so its tensors are allocated
// once and outlive every graph that reads them - the same arrangement Cache has, and for the
// same reason: a decode graph is built once and must find the same pointers on every step.
struct ZonedKV {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    memex::ZonedCache* c = nullptr;
    memex::ZonedCacheParams p;

    bool init(ggml_backend_buffer_type_t buft, const HParams& h, int ctx_len,
              const ZonedOpt& z) {
        p.n_layers    = h.n_layer;
        p.n_q_heads   = h.n_head;
        p.n_kv_heads  = h.n_head_kv;
        p.head_dim    = h.head_dim;
        p.n_ctx       = ctx_len;
        p.n_sinks     = z.sinks;
        p.notebook_cap = z.notebook;
        p.window      = z.window;
        p.tail_q8     = z.tail_q8;
        p.rotate_keys = z.rotate_keys;
        std::string err;
        if (!p.validate(&err)) {
            printf("зонный кэш: конфигурация отвергнута — %s\n", err.c_str());
            printf("  стоки %d + блокнот %d + окно %d = %d при контексте %d\n", z.sinks,
                   z.notebook, z.window, z.exact_slots(), ctx_len);
            printf("  зоны появляются только когда контекст длиннее точных зон: увеличьте "
                   "-c или уменьшите --window\n");
            return false;
        }
        c = new memex::ZonedCache(p);
        // no_alloc, then one allocation for every layer's six tensors at once.
        ggml_init_params ip = {ggml_tensor_overhead() * size_t(h.n_layer) * 8 + 8192,
                               nullptr, true};
        ctx = ggml_init(ip);
        if (!ctx) {
            printf("зонный кэш: контекст тензоров не создался\n");
            return false;
        }
        if (!c->create_tensors(ctx, &err)) {
            printf("зонный кэш: тензоры не создались — %s\n", err.c_str());
            return false;
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            printf("зонный кэш: буфер бэкенда не выделился\n");
            return false;
        }
        // Uploaded once here, in full: the mask starts as -inf everywhere and the tail is
        // empty, and an unuploaded mask would read as zeros - which is a mask that admits
        // every empty slot, the quietest possible way to be wrong.
        for (int il = 0; il < h.n_layer; ++il) {
            if (!c->upload(il, &err)) {
                printf("зонный кэш: первичная загрузка слоя %d не прошла — %s\n", il,
                       err.c_str());
                return false;
            }
        }
        return true;
    }

    std::size_t bytes() const { return buf ? ggml_backend_buffer_get_size(buf) : 0; }

    void free_all() {
        delete c;
        c = nullptr;
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr;
    }
};

// Fill the zoned cache from the exact cache's first `n` positions. The prefill runs on the
// exact path - build_attn scores one query and a prompt is many - so the zoned run starts
// from the same past the exact run has, position for position, and every difference after
// that is the zoning and nothing else.
//
// Rows are gathered out of the two cache layouts rather than recomputed: keys sit as
// [head_dim, n_ctx, kv_head] and values as [n_ctx, head_dim, kv_head], and a row of d_kv is
// one head of one position in [kv_head][head_dim] order, which is the order the module
// documents. Reading the whole layer at once and indexing on the host beats hd*n_kv_heads
// strided two-byte reads through the backend by three orders of magnitude of call overhead.
bool seed_zoned(memex::ZonedCache* zc, const Cache& kv, const HParams& h, int n) {
    if (!zc || n <= 0 || n > kv.n_ctx) {
        printf("затравка зонного кэша: %d позиций при кэше на %d — бессмысленно\n", n,
               kv.n_ctx);
        return false;
    }
    const int hd = h.head_dim;
    const int d = h.d_kv();
    std::vector<uint16_t> kb, vb;
    std::vector<float> kr, vr;
    kr.assign(size_t(d), 0.0f);
    vr.assign(size_t(d), 0.0f);
    for (int il = 0; il < h.n_layer; ++il) {
        ggml_tensor* kt = kv.k[size_t(il)];
        ggml_tensor* vt = kv.v[size_t(il)];
        kb.assign(size_t(ggml_nelements(kt)), 0);
        vb.assign(size_t(ggml_nelements(vt)), 0);
        ggml_backend_tensor_get(kt, kb.data(), 0, ggml_nbytes(kt));
        ggml_backend_tensor_get(vt, vb.data(), 0, ggml_nbytes(vt));
        for (int pos = 0; pos < n; ++pos) {
            for (int gg = 0; gg < h.n_head_kv; ++gg) {
                const size_t kbase = size_t(gg) * size_t(kv.n_ctx) * size_t(hd) +
                                     size_t(pos) * size_t(hd);
                const size_t vbase = size_t(gg) * size_t(hd) * size_t(kv.n_ctx);
                for (int j = 0; j < hd; ++j) {
                    kr[size_t(gg) * size_t(hd) + size_t(j)] =
                        ggml_fp16_to_fp32(kb[kbase + size_t(j)]);
                    vr[size_t(gg) * size_t(hd) + size_t(j)] = ggml_fp16_to_fp32(
                        vb[vbase + size_t(j) * size_t(kv.n_ctx) + size_t(pos)]);
                }
            }
            // keep_exact false throughout: the notebook is positions the engine asked to
            // keep, and this engine has no signal to ask with. Saying "keep everything"
            // would fill the notebook with the oldest prompt tokens for no reason.
            zc->append(il, kr.data(), vr.data(), /*keep_exact=*/false);
        }
    }
    zc->flush();
    std::string err;
    for (int il = 0; il < h.n_layer; ++il) {
        if (!zc->upload(il, &err)) {
            printf("затравка зонного кэша: загрузка слоя %d не прошла — %s\n", il,
                   err.c_str());
            return false;
        }
    }
    return true;
}

// After seeding, the zoned cache's exact zones must hold exactly what the exact cache holds,
// slot for position. Checked rather than assumed, because everything downstream of it looks
// plausible when it is wrong: the shapes are right, the softmax still normalises, and the
// only symptom is an error that reads as "the approximation costs this much".
//
// It also pins down which of the two possible faults you have. If this passes and the two
// attention outputs still differ, the difference is arithmetic - summation order, or a kernel
// rounding an operand - and not a mis-wired cache.
bool verify_seed(memex::ZonedCache* zc, const Cache& kv, const HParams& h, int n) {
    const int hd = h.head_dim;
    const int ex_pad = zc->exact_pad();
    const memex::ZonedCacheParams& zp = zc->params();
    // Once the window has wrapped, the positions that left are in the tail and no longer
    // comparable slot for slot. Say so rather than compare the wrong pairs and report a
    // difference that is the zoning working as designed.
    if (n - zp.n_sinks > zp.window) {
        printf("  затравка не сверяется побитово: окно уже провернулось "
               "(%d позиций против %d точных слотов), часть прошлого в хвосте\n", n,
               zp.n_sinks + zp.window);
        return true;
    }
    // The slot the zone map gives a position when the cache is filled from empty. Derived
    // from the map rather than assumed equal to the position, because a non-zero notebook
    // sits between the sinks and the window and shifts every window slot along.
    auto slot_of = [&](int pos) {
        return pos < zp.n_sinks
                   ? pos
                   : zp.n_sinks + zp.notebook_cap + ((pos - zp.n_sinks) % zp.window);
    };
    std::size_t bad = 0;
    double worst = 0.0;
    int bad_layer = -1, bad_pos = -1, bad_head = -1, bad_dim = -1;
    char which = '?';
    std::vector<uint16_t> kb, vb, zk, zv;
    for (int il = 0; il < h.n_layer && bad == 0; ++il) {
        ggml_tensor* kt = kv.k[size_t(il)];
        ggml_tensor* vt = kv.v[size_t(il)];
        ggml_tensor* zkt = zc->exact_k(il);
        ggml_tensor* zvt = zc->exact_v(il);
        if (!zkt || !zvt) return false;
        kb.assign(size_t(ggml_nelements(kt)), 0);
        vb.assign(size_t(ggml_nelements(vt)), 0);
        zk.assign(size_t(ggml_nelements(zkt)), 0);
        zv.assign(size_t(ggml_nelements(zvt)), 0);
        ggml_backend_tensor_get(kt, kb.data(), 0, ggml_nbytes(kt));
        ggml_backend_tensor_get(vt, vb.data(), 0, ggml_nbytes(vt));
        ggml_backend_tensor_get(zkt, zk.data(), 0, ggml_nbytes(zkt));
        ggml_backend_tensor_get(zvt, zv.data(), 0, ggml_nbytes(zvt));
        for (int pos = 0; pos < n; ++pos) {
            const int slot = slot_of(pos);
            if (slot >= ex_pad) continue;
            for (int gg = 0; gg < h.n_head_kv; ++gg) {
                for (int j = 0; j < hd; ++j) {
                    const uint16_t a = kb[size_t(gg) * size_t(kv.n_ctx) * size_t(hd) +
                                          size_t(pos) * size_t(hd) + size_t(j)];
                    const uint16_t b = zk[size_t(gg) * size_t(ex_pad) * size_t(hd) +
                                          size_t(slot) * size_t(hd) + size_t(j)];
                    if (a != b) {
                        const double d = std::abs(double(ggml_fp16_to_fp32(a)) -
                                                  double(ggml_fp16_to_fp32(b)));
                        if (d > worst) {
                            worst = d; bad_layer = il; bad_pos = pos; bad_head = gg;
                            bad_dim = j; which = 'K';
                        }
                        bad++;
                    }
                    const uint16_t av = vb[size_t(gg) * size_t(hd) * size_t(kv.n_ctx) +
                                           size_t(j) * size_t(kv.n_ctx) + size_t(pos)];
                    const uint16_t bv = zv[size_t(gg) * size_t(hd) * size_t(ex_pad) +
                                           size_t(j) * size_t(ex_pad) + size_t(slot)];
                    if (av != bv) {
                        const double d = std::abs(double(ggml_fp16_to_fp32(av)) -
                                                  double(ggml_fp16_to_fp32(bv)));
                        if (d > worst) {
                            worst = d; bad_layer = il; bad_pos = pos; bad_head = gg;
                            bad_dim = j; which = 'V';
                        }
                        bad++;
                    }
                }
            }
        }
    }
    if (bad == 0) {
        printf("  затравка сверена побитово: точные зоны совпадают с точным кэшем "
               "на всех %d позициях\n", n);
        return true;
    }
    printf("  ЗАТРАВКА РАСХОДИТСЯ: %zu элементов, худшее расхождение %.6g "
           "(%c, слой %d, позиция %d, kv-голова %d, измерение %d)\n", bad, worst, which,
           bad_layer, bad_pos, bad_head, bad_dim);
    return false;
}

// The host mirror of the row the graph just stored. Kc and Vc are f32 - they are what
// ggml_cpy converted on its way into the half-precision cache - so each is passed through
// f16 and back before it reaches the mirror. That round trip is not decoration: without it
// the mirror would hold the f32 value and the tensor the f16 one, the two would disagree by
// an f16 rounding, and the disagreement would surface only much later as the tail being
// quantised from a vector the graph never scored against.
bool mirror_zoned_row(memex::ZonedCache* zc, const Graph& g, int n_layer, int slot,
                      int d_kv, std::vector<float>* buf) {
    if (int(g.rows.size()) != n_layer) {
        printf("зеркало зонного кэша: записано %zu слоёв из %d\n", g.rows.size(), n_layer);
        return false;
    }
    buf->assign(size_t(d_kv) * 2, 0.0f);
    float* kf = buf->data();
    float* vf = buf->data() + size_t(d_kv);
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor* kt = g.rows[size_t(il)].first;
        ggml_tensor* vt = g.rows[size_t(il)].second;
        if (!kt || !vt || ggml_nelements(kt) != d_kv || ggml_nelements(vt) != d_kv ||
            kt->type != GGML_TYPE_F32 || vt->type != GGML_TYPE_F32) {
            printf("зеркало зонного кэша: слой %d выдал не ту форму строки\n", il);
            return false;
        }
        ggml_backend_tensor_get(kt, kf, 0, sizeof(float) * size_t(d_kv));
        ggml_backend_tensor_get(vt, vf, 0, sizeof(float) * size_t(d_kv));
        for (int j = 0; j < d_kv; ++j) {
            kf[j] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(kf[j]));
            vf[j] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(vf[j]));
        }
        zc->set_live_row(il, slot, kf, vf);
    }
    return true;
}

// One decode step's worth of zone bookkeeping, before the graph runs. Returns the slot the
// graph must store into, or -1.
//
// Order matters and is the whole protocol: every layer is asked for its slot first (they
// agree, because the zone map does not depend on the layer - and that is checked, because if
// it ever stopped agreeing the graph would store one layer's key into another layer's slot),
// then the tail's partial blocks are settled, then the mask and the tail go to the backend.
// The exact zones are deliberately not uploaded: the graph owns them now, and copying the
// mirror over them would undo the store the previous step made.
int zoned_begin_step(memex::ZonedCache* zc, int n_layer, bool keep_exact) {
    int slot = -1;
    std::string err;
    for (int il = 0; il < n_layer; ++il) {
        int note = -1;
        const int s = zc->append_live(il, keep_exact, &note);
        if (s < 0) {
            printf("зонный кэш: позиция не принята слоем %d — контекст исчерпан\n", il);
            return -1;
        }
        if (il == 0) {
            slot = s;
        } else if (s != slot) {
            printf("зонный кэш: слой %d дал слот %d, а слой 0 дал %d — карта зон "
                   "разошлась между слоями\n", il, s, slot);
            return -1;
        }
        // A promotion to the notebook is the one exact write the graph does not make, so it
        // is the one that has to be pushed to the backend by hand. Left out, the mask would
        // admit a slot holding zeros.
        if (note >= 0 && !zc->upload_slot(il, note, &err)) {
            printf("зонный кэш: слот блокнота %d слоя %d не загрузился — %s\n", note, il,
                   err.c_str());
            return -1;
        }
    }
    zc->flush();
    for (int il = 0; il < n_layer; ++il) {
        if (!zc->upload_dynamic(il, &err)) {
            printf("зонный кэш: загрузка маски и хвоста слоя %d не прошла — %s\n", il,
                   err.c_str());
            return -1;
        }
    }
    return slot;
}
#endif  // MEMEX_FWD_ZONED

// One step over `n_tokens` tokens starting at position `n_past`, attending over `n_kv`
// cached positions. The prefill is this with n_past 0, and a decode step is this with
// n_tokens 1 - the same graph builder, so there is one place where the arithmetic can be
// wrong instead of two.
//
// n_kv is rounded up to a multiple of 32 and the remainder masked off, because the fork's
// F16 matmul returns silently wrong results when the reduction length is not a multiple of
// four. The reference does the same thing for the same reason.
// `all_logits` keeps every position's row instead of only the last. Speculative verification
// is the only caller that needs it: it has to know what the target would have said at each of
// the drafted positions, and that is the whole point of verifying a batch.
// `zc` non-null replaces the read side of attention with the zoned cache's own subgraph, and
// replaces the store side too: the key and value go into the zoned window slot instead of
// into `kv`, so a zoned run never touches the exact cache and the two can be driven against
// each other in one process without either writing over the other's past. Decode only -
// build_attn scores one query - and refused rather than approximated if asked for more.
// `rs` non-null and switched on splits every layer's expert dispatch in two - the selections
// that are resident in video memory and the ones that are not - and adds the two partial
// results back together at the end of the layer. Both halves run on the CPU here, which is
// the point: it makes the split checkable against the unsplit graph before a GPU path exists.
// On a graph of more than one token (a prefill) the split is not built - the residency lookup
// gathers one row of the mask per token and the mask has one row per layer, not per token - so
// the prefill stays on the unsplit path and only exports its routing, to warm the policy.
bool build_step(Graph* g, ggml_backend_buffer_type_t buft, const HParams& h,
                const Weights& w, Cache& kv, int n_tokens, int n_past, int n_kv,
                int min_experts = -1, float expert_thresh = 1.0f, bool all_logits = false,
                ZonedKvCache* zc = nullptr, bool keep_dbg = false,
                const memex::ResidentSet* rs = nullptr,
                memex::GpuExperts* gx = nullptr,
                memex::GpuStatic* gstat = nullptr) {
#ifndef MEMEX_FWD_ZONED
    if (zc) {
        printf("зонный кэш: эта сборка собрана без него (нет дерева MemeX/cpp)\n");
        return false;
    }
#endif
    if (zc && n_tokens != 1) {
        printf("зонный кэш: шаг на %d токенов — build_attn считает один запрос, "
               "префилл обязан идти точным путём\n", n_tokens);
        return false;
    }
    (void)gx;
    // The split builds a second expert triple and a second eight-way sum per layer, so it
    // roughly doubles the feed-forward's node count; the warm-up copy is one node a layer.
    const bool rsplit = rs && rs->on() && n_tokens == 1;
    const bool rwarm  = rs && rs->on() && !rsplit;
    // The zoned subgraph is a dozen nodes a layer where the exact one is five, and running
    // out of graph nodes shows up as a truncated graph rather than as an error.
    const size_t n_nodes = size_t(h.n_layer) * (zc ? 96 : (rsplit ? 128 : 72)) + 256;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 512) +
                           ggml_graph_overhead_custom(n_nodes, false),
                           nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) return false;
    ggml_context* c = g->ctx;

    g->tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_kv, n_tokens);
    g->n_kv_built = n_kv;
    g->slot_thresh = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_expert_used, 1);
    // One row per layer, created before the input buffer is allocated because that is the
    // buffer it has to live in: it is rewritten between graph computations, once per refresh
    // rather than once per token.
    if (rsplit) {
        g->res_mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_expert, h.n_layer);
    }
    g->inputs = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!g->inputs) return false;
    {
        std::vector<float> st;
        st.assign(size_t(h.n_expert_used), 0.0f);
        for (int j = 0; j < h.n_expert_used; ++j) {
            st[size_t(j)] = j < min_experts ? 0.0f : expert_thresh;
        }
        ggml_backend_tensor_set(g->slot_thresh, st.data(), 0,
                               ggml_nbytes(g->slot_thresh));
    }

    g->gf = ggml_new_graph_custom(c, n_nodes, false);
    const int hd = h.head_dim;
    const float kq_scale = 1.0f / std::sqrt(float(hd));
    int sections[GGML_MROPE_SECTIONS] = {0};

    // The card takes the whole attention block, the residual, the FFN norm and the router -
    // everything above the expert dispatch - as ONE node per layer. Not because the norms and
    // the ropes are worth moving on their own (they move almost no bytes) but because leaving
    // them behind would put the boundary between them: about twenty crossings a layer instead
    // of one, and a crossing is priced in host coordination rather than in bytes. Cheap in
    // bytes is not cheap in place.
    //
    // Decode only. A prefill is a different graph shape and is compute-bound rather than
    // bandwidth-bound - measured monotonically worse on the card, 26.87 -> 12.34 -> 7.73 at
    // ngl 0/8/16 - so the prompt stays on the host and its cache is uploaded once afterwards.
    const bool card_layers =
#ifdef MEMEX_FWD_GPU_EXPERTS
        gstat && gstat->layers_on() && n_tokens == 1 && !zc;
#else
        false;
#endif

    g->on_card = card_layers;

    ggml_tensor* cur = ggml_get_rows(c, w.tok_embd, g->tokens);
    for (int il = 0; il < h.n_layer; ++il) {
        const Weights::Layer& L = w.layers[size_t(il)];
        // The three quantities the feed-forward needs, wherever attention was computed: the
        // attention output plus the residual, the FFN-normed hidden state, and the router's
        // pre-softmax logits. Everything below this point is identical on both paths, which
        // is the property that makes the card path checkable against the CPU one.
        ggml_tensor* ffn_inp  = nullptr;
        ggml_tensor* x        = nullptr;
        ggml_tensor* logits_e = nullptr;
#ifdef MEMEX_FWD_GPU_EXPERTS
        if (card_layers) {
            // [ffn_inp | ffn-normed hidden | router logits], three quantities in one readback
            // because each readback is a submit and a fence. The logits come back PRE-softmax
            // so the host's softmax and top-k are the same ops in the same order they were
            // before the card existed: the routing decision does not move.
            ggml_tensor* lay = gstat->layer(c, il, cur, g->mask);
            ffn_inp  = ggml_reshape_2d(c, ggml_view_1d(c, lay, h.n_embd, 0), h.n_embd, 1);
            x        = ggml_reshape_2d(c,
                ggml_view_1d(c, lay, h.n_embd, size_t(h.n_embd) * sizeof(float)),
                h.n_embd, 1);
            logits_e = ggml_reshape_2d(c,
                ggml_view_1d(c, lay, h.n_expert, size_t(2 * h.n_embd) * sizeof(float)),
                h.n_expert, 1);
        } else
#endif
        {
        ggml_tensor* inpSA = cur;
        x = norm(c, cur, L.attn_norm, h.rms_eps);

        ggml_tensor* q = ggml_mul_mat(c, L.wq, x);
        ggml_tensor* k = ggml_mul_mat(c, L.wk, x);
        ggml_tensor* v = ggml_mul_mat(c, L.wv, x);

        q = ggml_reshape_3d(c, q, hd, h.n_head, n_tokens);
        q = norm(c, q, L.q_norm, h.rms_eps);
        q = ggml_rope_multi(c, q, g->positions, nullptr, hd, sections, h.rope_type,
                            h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_reshape_3d(c, k, hd, h.n_head_kv, n_tokens);
        k = norm(c, k, L.k_norm, h.rms_eps);
        k = ggml_rope_multi(c, k, g->positions, nullptr, hd, sections, h.rope_type,
                            h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Into the cache at n_past, then read back from position zero: the store and the
        // load are separate views of the same tensor, which is what makes a decode step
        // cost one position of writing and n_kv of reading.
        ggml_tensor* Kc = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
        ggml_tensor* Vc = ggml_cont(c, ggml_permute(
            c, ggml_reshape_3d(c, v, hd, h.n_head_kv, n_tokens), 1, 2, 0, 3));
        if (!zc) {
            ggml_tensor* kdst = ggml_view_3d(c, kv.k[size_t(il)], hd, n_tokens, h.n_head_kv,
                                             kv.k[size_t(il)]->nb[1], kv.k[size_t(il)]->nb[2],
                                             size_t(n_past) * kv.k[size_t(il)]->nb[1]);
            ggml_tensor* vdst = ggml_view_3d(c, kv.v[size_t(il)], n_tokens, hd, h.n_head_kv,
                                             kv.v[size_t(il)]->nb[1], kv.v[size_t(il)]->nb[2],
                                             size_t(n_past) * ggml_element_size(kv.v[size_t(il)]));
            ggml_tensor* kcpy = ggml_cpy(c, Kc, kdst);
            ggml_tensor* vcpy = ggml_cpy(c, Vc, vdst);
            ggml_build_forward_expand(g->gf, kcpy);
            ggml_build_forward_expand(g->gf, vcpy);
            // Recorded so a decode graph can be re-aimed instead of rebuilt. The steps are the
            // same strides the offsets above were built from, and the position counts differ
            // because V is stored transposed: a position is a row of K but a column of V.
            g->writes.push_back({kcpy, kdst, kv.k[size_t(il)], kv.k[size_t(il)]->nb[1],
                                 kv.k[size_t(il)]->ne[1], n_tokens});
            g->writes.push_back({vcpy, vdst, kv.v[size_t(il)],
                                 ggml_element_size(kv.v[size_t(il)]),
                                 kv.v[size_t(il)]->ne[0], n_tokens});
        }

        // Heads last, three dimensions. A two-dimensional strided slice per head group looks
        // like the same thing and is not: grouped-query broadcasting inside ggml_mul_mat only
        // happens in this form, and the wrong form returns garbage rather than an error.
        ggml_tensor* Q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
        ggml_tensor* kqv = nullptr;
#ifdef MEMEX_FWD_ZONED
        if (zc) {
            // Store into the slot the cache handed back, then let build_attn read every zone.
            // The offset here is a placeholder: aim_zoned_writes moves it to the slot of the
            // step, exactly as aim_cache_writes moves the exact path's.
            ggml_tensor* zk = zc->exact_k(il);
            ggml_tensor* zv = zc->exact_v(il);
            if (!zk || !zv || !zk->data || !zv->data) {
                printf("зонный кэш: точные зоны слоя %d не размещены в буфере\n", il);
                return false;
            }
            ggml_tensor* zkdst = ggml_view_3d(c, zk, hd, 1, h.n_head_kv,
                                              zk->nb[1], zk->nb[2], 0);
            ggml_tensor* zvdst = ggml_view_3d(c, zv, 1, hd, h.n_head_kv,
                                              zv->nb[1], zv->nb[2], 0);
            ggml_tensor* zkcpy = ggml_cpy(c, Kc, zkdst);
            ggml_tensor* zvcpy = ggml_cpy(c, Vc, zvdst);
            ggml_build_forward_expand(g->gf, zkcpy);
            ggml_build_forward_expand(g->gf, zvcpy);
            g->zwrites.push_back({zkcpy, zkdst, zk, zk->nb[1], zk->ne[1], 1});
            g->zwrites.push_back({zvcpy, zvdst, zv, ggml_element_size(zv), zv->ne[0], 1});
            // The host has to mirror this row after the graph, or the next eviction of this
            // slot compresses the position that just left instead of this one. Marked as
            // outputs so the allocator does not hand their memory to a later node - which it
            // otherwise would, and the mirror would then be whatever that node left behind.
            ggml_set_output(Kc);
            ggml_set_output(Vc);
            g->rows.push_back({Kc, Vc});
            memex::ZonedAttn za = zc->build_attn(c, il, Q);
            if (!za.out || !za.probs) {
                printf("зонный кэш: build_attn не собрал подграф на слое %d\n", il);
                return false;
            }
            // One softmax, and that is the correctness claim of the whole module rather than
            // a detail: a softmax per zone stops the window and the tail competing for
            // probability mass, which is a different function that still looks like
            // attention. Checked here because the engine is where it would go unnoticed.
            if (za.probs->op != GGML_OP_SOFT_MAX) {
                printf("зонный кэш: слой %d вернул %s вместо softmax\n", il,
                       ggml_op_name(za.probs->op));
                return false;
            }
            kqv = za.out;
        } else
#endif
        {
            ggml_tensor* K = ggml_view_3d(c, kv.k[size_t(il)], hd, n_kv, h.n_head_kv,
                                          kv.k[size_t(il)]->nb[1], kv.k[size_t(il)]->nb[2], 0);
            ggml_tensor* V = ggml_view_3d(c, kv.v[size_t(il)], n_kv, hd, h.n_head_kv,
                                          kv.v[size_t(il)]->nb[1], kv.v[size_t(il)]->nb[2], 0);
            ggml_tensor* kq = ggml_mul_mat(c, K, Q);
            ggml_tensor* p = ggml_soft_max_ext(c, kq, g->mask, kq_scale, 0.0f);
            kqv = ggml_mul_mat(c, V, p);
            // So the step can be narrowed to the occupied positions. Built at n_kv, which is
            // the cache's whole length for a graph that is reused, and aimed lower per step.
            g->reads.push_back({K, V, kq, p, g->mask});
        }
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), h.d_q(), n_tokens);
        if (keep_dbg) {
            ggml_set_output(inpSA);
            ggml_set_output(kqv);
            g->dbg.push_back({inpSA, kqv});
        }
        cur = ggml_add(c, ggml_mul_mat(c, L.wo, kqv), inpSA);

        ffn_inp = cur;
        x = norm(c, ffn_inp, L.ffn_norm, h.rms_eps);
        logits_e = ggml_mul_mat(c, L.router, x);
        }

        ggml_tensor* probs = ggml_soft_max(c, logits_e);
        // Expert reduction. The kernel already supports it: mul_mat_id skips an id of -1
        // and zeroes that output slice ("This is needed for SER", says the comment in
        // ggml.c), so a dropped expert is genuinely not read - and expert bytes are the
        // dominant per-token cost.
        //
        // The weights need care, and this is very likely why the graph-level call was left
        // commented out in the fork. ggml_get_rows does not guard a negative index, and the
        // renormalisation divides by the sum of the gathered weights - so gathering with the
        // thresholded ids would divide by a sum containing whatever lay before the buffer.
        // So the ids with -1 go only to mul_mat_id, while the weights are gathered with the
        // unthresholded ids and zeroed by the same comparison the kernel makes.
        const bool reduce = min_experts >= 0 && min_experts < h.n_expert_used;
        ggml_tensor* sel_full = ggml_top_k(c, probs, h.n_expert_used);
        ggml_tensor* sel = reduce
            ? ggml_top_k_thresh(c, probs, h.n_expert_used, min_experts, expert_thresh)
            : sel_full;
        ggml_tensor* weights = ggml_get_rows(c,
            ggml_reshape_3d(c, probs, 1, h.n_expert, n_tokens), sel_full);
        weights = ggml_reshape_2d(c, weights, h.n_expert_used, n_tokens);
        if (reduce) {
            // Threshold is relative to the best expert of the row, which is element zero
            // because the ids came back sorted. The per-slot threshold is zero for the first
            // min_experts, so those are always kept - the same rule the kernel applies.
            ggml_tensor* top = ggml_cont(c,
                ggml_view_2d(c, weights, 1, n_tokens, weights->nb[1], 0));
            ggml_tensor* thr = ggml_mul(c, ggml_repeat(c, top, weights), g->slot_thresh);
            ggml_tensor* keep = ggml_step(c, ggml_sub(c, weights, thr));
            weights = ggml_mul(c, weights, keep);
        }
        weights = ggml_div(c, weights, ggml_sum_rows(c, weights));
        weights = ggml_reshape_3d(c, weights, 1, h.n_expert_used, n_tokens);

        ggml_tensor* xe = ggml_reshape_3d(c, x, h.n_embd, 1, n_tokens);
        // The expert triple, producing one UNWEIGHTED output per slot: [n_embd, n_used,
        // n_tokens]. Factored out because the split needs it twice, with two id lists.
        auto experts = [&](ggml_tensor* ids) {
            ggml_tensor* up = ggml_mul_mat_id(c, L.up, xe, ids);
            ggml_tensor* gt = ggml_silu(c, ggml_mul_mat_id(c, L.gate, xe, ids));
            return ggml_mul_mat_id(c, L.down, ggml_mul(c, up, gt), ids);
        };
        // The routing weights, then the sum over slots. This is where the order matters: the
        // weights go on AFTER the expert outputs are gathered and are indexed by slot, which
        // is what forces the split to preserve slots. It is deliberately outside the split -
        // both halves come back here as one per-slot tensor - because that is what makes the
        // split bit-exact rather than merely close. See below.
        //
        // QUEUED: this is eight nodes where ggml_mul_multi_add is one, and the same
        // substitution is already made twice in this file (build_gemma4_step and
        // qwen35_ffn). Deliberately NOT taken yet - the two new architectures have to be
        // proven first, because a regression here with three things in flight would be very
        // hard to attribute. Three findings that the change rests on, all checked already:
        //
        //  - Association matches. The kernel (iqk_cpu_ops.cpp, iqk_mul_multi_add, the branch
        //    with no src[2]) writes y = x0*w0 for slot zero and then accumulates y += xj*wj
        //    from slot one upward: the same left-to-right order as the chain below, and the
        //    same per-slot product. The one thing left to confirm on the machine is whether
        //    the compiler contracts `y[k] += x0[k]*x1[0]` into an FMA, which would leave the
        //    product unrounded and make the two differ in the last bit.
        //
        //  - If it does differ, the FUSED one is right. The reference's cparams.fused_mmad
        //    defaults to true (llama.cpp:7728) and llm_build_moe_ffn takes that branch for
        //    qwen3moe, so the eight-node chain below is already the spelling that does NOT
        //    match what llama_decode computes. Taking the fused op moves this path toward
        //    the reference rather than away from it.
        //
        //  - The resident split stays safe, and by construction rather than by luck. The
        //    per-slot add of the two halves happens OUTSIDE this lambda (see the call site
        //    below: ggml_add(o_res, o_oth) is passed IN). So the fused op would receive one
        //    already-summed per-slot tensor, exactly as this chain does, and the rule that
        //    came out of the 3-6% logit incident - sum per slot, before the routing weights
        //    and before the fold - is preserved without needing a new argument. Verify with
        //    --decode-check and not only with the prefill comparison: this is the generation
        //    path, and the incident it guards against was invisible in the tokens.
        //
        auto weight_and_fold = [&](ggml_tensor* o) {
            o = ggml_mul(c, o, weights);
            ggml_tensor* s = ggml_view_2d(c, o, h.n_embd, n_tokens, o->nb[2], 0);
            for (int e = 1; e < h.n_expert_used; ++e) {
                s = ggml_add(c, s, ggml_view_2d(c, o, h.n_embd, n_tokens, o->nb[2],
                                                size_t(e) * o->nb[1]));
            }
            return s;
        };
        ggml_tensor* moe = nullptr;
        if (rsplit) {
            // Residency for each of this token's picks, gathered out of this layer's row of
            // the mask by the router's own ids. Gathered with sel_full rather than sel: with
            // expert reduction on, sel carries -1 in the dropped slots and ggml_get_rows does
            // not guard a negative index - it would read whatever lies before the buffer. The
            // two agree wherever sel is non-negative, and a dropped slot ends up -1 in both
            // halves either way, which is exactly what the unsplit path does with it.
            ggml_tensor* mrow = ggml_reshape_3d(c,
                ggml_view_1d(c, g->res_mask, h.n_expert,
                             size_t(il) * g->res_mask->nb[1]), 1, h.n_expert, 1);
            ggml_tensor* flags = ggml_get_rows(c, mrow, sel_full);
            ggml_tensor* ids_res = memex::resident_split_ids(c, sel, flags, /*resident=*/true);
            ggml_tensor* ids_oth = memex::resident_split_ids(c, sel, flags, /*resident=*/false);
            // Kept alive as outputs so the host can read the split back after the graph. Also
            // the only way to check the mask actually arrived: the host holds the same bitset
            // and can redo the split itself, and a disagreement is a failure that produces no
            // wrong shape and no NaN.
            ggml_set_output(ids_res);
            ggml_set_output(ids_oth);
            g->res_ids.push_back(ids_res);
            g->oth_ids.push_back(ids_oth);
            // The resident half is the one a GPU will take over. Computed on the CPU here,
            // and the two are added PER SLOT - before the weighting and before the fold -
            // rather than folded separately and added at the end.
            //
            // That choice is the difference between bit-exact and merely close, and it was
            // measured before it was believed. Folding each half over its own eight slots
            // reassociates the sum: the resident half adds its members in slot order with
            // exact zeros interleaved, the other half does the same, and (a+c)+(b+d) is not
            // (((a+b)+c)+d) in f32. That showed up as 2.5e-8 relative on layer 1's input,
            // which this model amplified to 0.11% by layer 16 and 1.6% by layer 47, for a
            // 3-6% relative L2 on the logits - with every token still identical. Adding per
            // slot instead is exact, because mul_mat_id zeroed the slice of every slot the
            // other half owns and x + 0.0 is x, so each slot's value arrives at the shared
            // weight-and-fold bit-identical to the unsplit path's, and the fold itself is
            // then literally the same eight additions in the same order.
            //
            // It is also the shape the GPU step wants: whatever computes the resident half
            // returns one vector per slot, and the sum is elementwise.
            //
            // Named locals rather than two arguments to ggml_add, because C++ leaves the
            // evaluation order of function arguments unspecified and that order is the order
            // the two halves' nodes enter the graph.
#ifdef MEMEX_FWD_GPU_EXPERTS
            if (gx && gx->on()) {
                // The fork returns a copy of the other half's id list, and the CPU experts
                // are dispatched with THAT rather than with ids_oth. The submit is therefore
                // an ancestor of every CPU expert node, so ggml's own topological order puts
                // it first - which is the only ordering guarantee available inside one graph,
                // and a good deal firmer than building the nodes in the hoped-for sequence.
                ggml_tensor* ids_oth_after = gx->fork(c, il, ids_oth, xe, ids_res);
                ggml_tensor* o_oth = experts(ids_oth_after);
                // Under --gpu-experts-check the CPU computes the resident half as well, so
                // the device's answer has something to be wrong against. It costs the entire
                // saving, which is what a check of this kind is for.
                ggml_tensor* o_res_cpu = gx->config().check ? experts(ids_res) : nullptr;
                // And the join takes the CPU half as its input, so it cannot be scheduled
                // before it either. It produces the resident half with an exact 0.0f in every
                // slot the device did not own - which is what keeps the add below a per-slot
                // x + 0.0f and therefore bit-identical in structure to the unsplit fold.
                ggml_tensor* o_res = gx->join(c, il, o_oth, o_res_cpu);
                moe = weight_and_fold(ggml_add(c, o_res, o_oth));
            } else
#endif
            {
                ggml_tensor* o_res = experts(ids_res);
                ggml_tensor* o_oth = experts(ids_oth);
                moe = weight_and_fold(ggml_add(c, o_res, o_oth));
            }
        } else {
            moe = weight_and_fold(experts(sel));
            if (rwarm) {
                // The prefill's routing, in a form the host can read as one block per token.
                // Nothing in the compute depends on it; it exists so the sliding window is
                // warm by the time the first token is generated.
                ggml_tensor* cp = memex::resident_ids_copy(c, sel_full);
                ggml_set_output(cp);
                g->sel_ids.push_back(cp);
                ggml_build_forward_expand(g->gf, cp);
            }
        }
        cur = ggml_add(c, moe, ffn_inp);
    }

    // Only the last token's row goes through the output head. The head is 151936 x 2048 -
    // 243 MB even at six bits - so running it over a whole prefill would cost more than the
    // prefill. The reference prunes its last layer for the same reason.
    if (n_tokens > 1 && !all_logits) {
        cur = ggml_view_2d(c, cur, h.n_embd, 1, cur->nb[1],
                           size_t(n_tokens - 1) * cur->nb[1]);
        cur = ggml_cont(c, cur);
    }
    cur = norm(c, cur, w.out_norm, h.rms_eps);
    g->logits = head_matmul(c, w.out, cur, gstat);
    ggml_set_output(g->logits);
    ggml_build_forward_expand(g->gf, g->logits);
    g->alloc = ggml_gallocr_new(buft);
    if (!g->alloc) {
        printf("граф: ggml_gallocr_new не удался\n");
        return false;
    }
    if (!ggml_gallocr_reserve(g->alloc, g->gf)) {
        printf("граф: резервирование не удалось (n_tokens %d, n_kv %d)\n", n_tokens, n_kv);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(g->alloc, g->gf)) {
        printf("граф: размещение не удалось (n_tokens %d, n_kv %d)\n", n_tokens, n_kv);
        return false;
    }
    return true;
}

// The draft model's step: dense qwen3. Attention is the same block as the target's - the same
// per-head norm before the same rotation, the same half-precision cache - and the only
// difference is that the feed-forward is one gated triple instead of a routed dispatch. It is
// written out rather than shared with build_step because the two differ in the FFN and in
// nothing else, and a shared builder with an `is_moe` flag would put the two architectures'
// arithmetic in one place where a change to either can reach the other.
bool build_dense_step(Graph* g, ggml_backend_buffer_type_t buft, const HParams& h,
                      const DenseWeights& w, Cache& kv, int n_tokens, int n_past, int n_kv,
                      bool all_logits) {
    if (n_tokens <= 0 || n_kv <= 0) {
        printf("черновик: бессмысленные размеры n_tokens %d, n_kv %d\n", n_tokens, n_kv);
        return false;
    }
    const size_t n_nodes = size_t(h.n_layer) * 48 + 256;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 512) +
                           ggml_graph_overhead_custom(n_nodes, false),
                           nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) {
        printf("черновик: ggml_init не выделил метаданные\n");
        return false;
    }
    ggml_context* c = g->ctx;

    g->tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_kv, n_tokens);
    g->n_kv_built = n_kv;
    g->inputs = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!g->inputs) {
        printf("черновик: буфер входов не выделился\n");
        return false;
    }

    g->gf = ggml_new_graph_custom(c, n_nodes, false);
    const int hd = h.head_dim;
    const float kq_scale = 1.0f / std::sqrt(float(hd));
    int sections[GGML_MROPE_SECTIONS] = {0};

    ggml_tensor* cur = ggml_get_rows(c, w.tok_embd, g->tokens);
    for (int il = 0; il < h.n_layer; ++il) {
        const DenseWeights::Layer& L = w.layers[size_t(il)];
        ggml_tensor* inpSA = cur;
        ggml_tensor* x = norm(c, cur, L.attn_norm, h.rms_eps);

        ggml_tensor* q = ggml_mul_mat(c, L.wq, x);
        ggml_tensor* k = ggml_mul_mat(c, L.wk, x);
        ggml_tensor* v = ggml_mul_mat(c, L.wv, x);

        q = ggml_reshape_3d(c, q, hd, h.n_head, n_tokens);
        q = norm(c, q, L.q_norm, h.rms_eps);
        q = ggml_rope_multi(c, q, g->positions, nullptr, hd, sections, h.rope_type,
                            h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_reshape_3d(c, k, hd, h.n_head_kv, n_tokens);
        k = norm(c, k, L.k_norm, h.rms_eps);
        k = ggml_rope_multi(c, k, g->positions, nullptr, hd, sections, h.rope_type,
                            h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        ggml_tensor* kt = kv.k[size_t(il)];
        ggml_tensor* vt = kv.v[size_t(il)];
        ggml_tensor* Kc = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
        ggml_tensor* Vc = ggml_cont(c, ggml_permute(
            c, ggml_reshape_3d(c, v, hd, h.n_head_kv, n_tokens), 1, 2, 0, 3));
        ggml_tensor* kdst = ggml_view_3d(c, kt, hd, n_tokens, h.n_head_kv, kt->nb[1],
                                         kt->nb[2], size_t(n_past) * kt->nb[1]);
        ggml_tensor* vdst = ggml_view_3d(c, vt, n_tokens, hd, h.n_head_kv, vt->nb[1],
                                         vt->nb[2], size_t(n_past) * ggml_element_size(vt));
        ggml_tensor* kcpy = ggml_cpy(c, Kc, kdst);
        ggml_tensor* vcpy = ggml_cpy(c, Vc, vdst);
        ggml_build_forward_expand(g->gf, kcpy);
        ggml_build_forward_expand(g->gf, vcpy);
        g->writes.push_back({kcpy, kdst, kt, kt->nb[1], kt->ne[1], n_tokens});
        g->writes.push_back({vcpy, vdst, vt, ggml_element_size(vt), vt->ne[0], n_tokens});

        ggml_tensor* K = ggml_view_3d(c, kt, hd, n_kv, h.n_head_kv, kt->nb[1], kt->nb[2], 0);
        ggml_tensor* V = ggml_view_3d(c, vt, n_kv, hd, h.n_head_kv, vt->nb[1], vt->nb[2], 0);
        ggml_tensor* Q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));

        ggml_tensor* kq = ggml_mul_mat(c, K, Q);
        ggml_tensor* p = ggml_soft_max_ext(c, kq, g->mask, kq_scale, 0.0f);
        ggml_tensor* kqv = ggml_mul_mat(c, V, p);
        g->reads.push_back({K, V, kq, p, g->mask});
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), h.d_q(), n_tokens);
        cur = ggml_add(c, ggml_mul_mat(c, L.wo, kqv), inpSA);

        ggml_tensor* ffn_inp = cur;
        x = norm(c, ffn_inp, L.ffn_norm, h.rms_eps);
        // SwiGLU, with the silu on the gate arm. Putting it on the up arm instead is a
        // one-token edit that changes nothing structural and every number.
        ggml_tensor* up = ggml_mul_mat(c, L.up, x);
        ggml_tensor* gate = ggml_silu(c, ggml_mul_mat(c, L.gate, x));
        cur = ggml_add(c, ggml_mul_mat(c, L.down, ggml_mul(c, up, gate)), ffn_inp);
    }

    if (n_tokens > 1 && !all_logits) {
        cur = ggml_cont(c, ggml_view_2d(c, cur, h.n_embd, 1, cur->nb[1],
                                        size_t(n_tokens - 1) * cur->nb[1]));
    }
    cur = norm(c, cur, w.out_norm, h.rms_eps);
    g->logits = ggml_mul_mat(c, w.out, cur);
    ggml_set_output(g->logits);
    ggml_build_forward_expand(g->gf, g->logits);
    g->alloc = ggml_gallocr_new(buft);
    if (!g->alloc) {
        printf("черновик: ggml_gallocr_new не удался\n");
        return false;
    }
    if (!ggml_gallocr_reserve(g->alloc, g->gf)) {
        printf("черновик: резервирование графа не удалось\n");
        return false;
    }
    if (!ggml_gallocr_alloc_graph(g->alloc, g->gf)) {
        printf("черновик: размещение графа не удалось\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// Gemma 4 26B-A4B
// ---------------------------------------------------------------------------------------

// One graph, whatever the token count: a prefill is this with n_tokens = the prompt and
// n_past = 0, a decode step is this with n_tokens = 1 and the writes re-aimed. Written out
// separately from build_step rather than sharing it behind a flag, for the reason the file
// already gives for build_dense_step: the two architectures' arithmetic would then live in
// one place where a change to either can reach the other, and there is no way to notice.
//
// The block, taken from the reference's build_gemma4() and not from a description of it:
//
//   attn_out = attn_norm -> attention -> post_attention_norm -> + residual
//   dense    = ffn_down( gelu(ffn_gate.h) * ffn_up.h ),   h = rms(attn_out)*ffn_norm
//   routed   = MoE( rms(attn_out)*pre_ffw_norm_2 ),  router reads attn_out, not h
//   cur      = rms(dense)*post_ffw_norm_1 + rms(routed)*post_ffw_norm_2
//   cur      = rms(cur)*post_ffw_norm + attn_out
//   cur      = cur * layer_output_scale
//
// Note what the router reads. Its input is the post-attention residual stream, normalised
// with a scale tensor of its own (ffn_gate_inp.scale) - NOT the activation the experts see.
// Feeding it the expert input instead produces a routing that is wrong in a way no shape
// check can catch and no output obviously betrays.
// STATE, 27 Aug 2026 - read examples/memex-fwd/gemma4_state.md before editing this function.
//
// The attention block below is VERIFIED: against the reference run with --ref-fa, layer 0 gives
// eleven consecutive tensors at exactly 0.0000% relative L2 with max |d| 0.00000 - attn_norm,
// Qcur, Kcur, Vcur, both norms, both ropes, kq, the masked softmax, kqv_out and attn_out. So the
// windowed geometry, the 1.0 attention scale, the unweighted V rms_norm, the V-from-K sharing on
// the five full layers, the per-layer rope base and the KV cache layout are all byte-for-byte
// the reference's, and none of them is a candidate for anything.
//
// --ref-fa is not a detail. With flash attention OFF the fork's own gemma4 V cache is stored
// through ggml_transpose + a flat ggml_cpy, and gemma4 is the one architecture that hands that
// code a 3-D V - so the reference itself is wrong in that arm, and comparing against it reported
// 414% on the logits and 217% on kqv_out. Same binary, same prompt, flash attention on: 10.54%
// and a matching argmax. Do not read a disagreement here as ours without checking that flag.
//
// What is still wrong is downstream of the attention, in this layer's feed-forward:
// ffn_moe_combined-0 sits at 17.11% while ffn_norm_2-0 above it is exact, and the two RMS values
// agree to six figures so it is not a missing scale. The probes that split it four ways
// (ffn_norm_1 for the dense half, ffn_moe_weighted for the routed half under the reference's own
// name) are in the code and have never been run.
bool build_gemma4_step(Graph* g, ggml_backend_buffer_type_t buft, const HParams& h,
                       const Gemma4Weights& w, Cache& kv, int n_tokens, int n_past,
                       int n_kv, bool all_logits, bool keep_probes) {
    if (n_tokens <= 0 || n_kv <= 0 || n_past < 0) {
        printf("gemma4: бессмысленные размеры n_tokens %d, n_past %d, n_kv %d\n",
               n_tokens, n_past, n_kv);
        return false;
    }
    // Attention is a dozen nodes, the dense feed-forward four, the routed half a dozen plus
    // the eight-way fold, and the join four. Ninety-six a layer is roughly double what it
    // needs, which is the right side to be wrong on: running out shows up as a truncated
    // graph rather than as an error.
    const size_t n_nodes = size_t(h.n_layer) * 96 + 256;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 512) +
                           ggml_graph_overhead_custom(n_nodes, false), nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) return false;
    ggml_context* c = g->ctx;

    g->tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_kv, n_tokens);
    if (h.any_swa()) {
        g->mask_swa = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_kv, n_tokens);
    }
    g->n_kv_built = n_kv;
    g->inputs = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!g->inputs) return false;
    g->gf = ggml_new_graph_custom(c, n_nodes, false);

    const float eps = h.rms_eps;
    ggml_tensor* cur = ggml_get_rows(c, w.tok_embd, g->tokens);
    // sqrt(n_embd) on the embeddings. Gemma has always done this and it is not folded into
    // the table by the converter, so leaving it out scales every activation in the model by
    // 1/53 and still generates fluent text.
    if (h.f_embd_scale > 0.0f) cur = ggml_scale(c, cur, h.f_embd_scale);

    for (int il = 0; il < h.n_layer; ++il) {
        const Gemma4Weights::Layer& L = w.layers[size_t(il)];
        const LayerGeom& G = h.L(il);
        const int hd = G.head_dim;
        // 1.0, not 1/sqrt(hd). See HParams::f_attn_scale - this is the single constant in
        // gemma4 most likely to be silently wrong, because the wrong value still produces
        // grammatical output.
        const float kq_scale = h.f_attn_scale != 0.0f ? h.f_attn_scale
                                                      : 1.0f / std::sqrt(float(hd));
        ggml_tensor* inpSA = cur;
        // Stage-by-stage probes, layers 0 and 1 only. Names are the reference's, so the
        // comparison finds them; the point is to say WHICH stage of the attention block a
        // divergence enters at, which no per-layer output can.
        const bool inner = keep_probes && il <= 1;
        const std::string sil = std::to_string(il);
        ggml_tensor* x = fnorm(c, cur, L.attn_norm, eps);
        if (inner) g->probes.push_back({"attn_norm-" + sil, x});

        ggml_tensor* q = ggml_mul_mat(c, L.wq, x);
        ggml_tensor* k = ggml_mul_mat(c, L.wk, x);
        // V's source: its own projection where there is one, and otherwise the RAW output of
        // the K projection - the node before attn_k_norm and before the rotation, which is
        // what `Vcur = Kcur` in the reference picks up at that point in its graph. Taking the
        // normed or roped K instead would be a different model that still runs.
        ggml_tensor* v = L.wv ? ggml_mul_mat(c, L.wv, x) : k;
        if (inner) {
            // The raw projections, before either norm. With attn_norm above them and
            // Qcur_normed below, a disagreement is attributable to the weight or to the
            // norm rather than to "somewhere in the attention block".
            g->probes.push_back({"Qcur-" + sil, q});
            g->probes.push_back({"Kcur-" + sil, k});
            if (L.wv) g->probes.push_back({"Vcur-" + sil, v});
        }

        q = ggml_reshape_3d(c, q, hd, G.n_head, n_tokens);
        q = fnorm(c, q, L.q_norm, eps);
        if (inner) g->probes.push_back({"Qcur_normed-" + sil, q});
        q = ggml_rope_ext(c, q, g->positions, G.rope_freqs, G.n_rot, h.rope_type,
                          h.n_ctx_train, G.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        if (inner) g->probes.push_back({"Qcur_roped-" + sil, q});

        // V is rms-normed per head with NO weight, on every gemma4 layer - the ones that have
        // their own V projection included. There is no v_norm tensor in the file; the norm is
        // unweighted and it is still there.
        v = ggml_rms_norm(c, ggml_reshape_3d(c, v, hd, G.n_head_kv, n_tokens), eps);

        k = ggml_reshape_3d(c, k, hd, G.n_head_kv, n_tokens);
        k = fnorm(c, k, L.k_norm, eps);
        if (inner) g->probes.push_back({"Kcur_normed-" + sil, k});
        k = ggml_rope_ext(c, k, g->positions, G.rope_freqs, G.n_rot, h.rope_type,
                          h.n_ctx_train, G.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        if (inner) g->probes.push_back({"Kcur_roped-" + sil, k});

        ggml_tensor* Kc = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
        ggml_tensor* Vc = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));
        {
            ggml_tensor* kc = kv.k[size_t(il)];
            ggml_tensor* vc = kv.v[size_t(il)];
            ggml_tensor* kdst = ggml_view_3d(c, kc, hd, n_tokens, G.n_head_kv,
                                             kc->nb[1], kc->nb[2],
                                             size_t(n_past) * kc->nb[1]);
            ggml_tensor* vdst = ggml_view_3d(c, vc, n_tokens, hd, G.n_head_kv,
                                             vc->nb[1], vc->nb[2],
                                             size_t(n_past) * ggml_element_size(vc));
            ggml_tensor* kcpy = ggml_cpy(c, Kc, kdst);
            ggml_tensor* vcpy = ggml_cpy(c, Vc, vdst);
            ggml_build_forward_expand(g->gf, kcpy);
            ggml_build_forward_expand(g->gf, vcpy);
            g->writes.push_back({kcpy, kdst, kc, kc->nb[1], kc->ne[1], n_tokens});
            g->writes.push_back({vcpy, vdst, vc, ggml_element_size(vc), vc->ne[0], n_tokens});
        }

        ggml_tensor* Q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
        // The reference calls this one "q" (llm_build_kqv), and it is a bare permute there
        // against our permute-and-cont. Same numbers in the same order if the permute is the
        // one we think it is, so a disagreement here would be the head layout itself.
        if (inner) g->probes.push_back({"q-" + sil, Q});
        ggml_tensor* K = ggml_view_3d(c, kv.k[size_t(il)], hd, n_kv, G.n_head_kv,
                                      kv.k[size_t(il)]->nb[1], kv.k[size_t(il)]->nb[2], 0);
        ggml_tensor* V = ggml_view_3d(c, kv.v[size_t(il)], n_kv, hd, G.n_head_kv,
                                      kv.v[size_t(il)]->nb[1], kv.v[size_t(il)]->nb[2], 0);
        // Which mask. The window lives in the mask's contents, not in the cache's extent -
        // the windowed layers still read the whole occupied cache and throw away what falls
        // outside. Narrowing the read is a real saving and a different change.
        ggml_tensor* msk = (G.kind == LayerKind::ATTN_SWA && g->mask_swa) ? g->mask_swa
                                                                         : g->mask;
        ggml_tensor* kq = ggml_mul_mat(c, K, Q);
        ggml_tensor* pr = ggml_soft_max_ext(c, kq, msk, kq_scale, 0.0f);
        // The three stages of the attention core, under the reference's names. Everything
        // above them was bit-identical and kqv_out was 218% off, so the fault is in here and
        // these split it three ways: kq is Q against the cached K, kq_soft_max_ext adds the
        // mask and the scale, kqv_merged_cont adds the cached V.
        if (inner) {
            g->probes.push_back({"kq-" + sil, kq});
            g->probes.push_back({"kq_soft_max_ext-" + sil, pr});
        }
        ggml_tensor* kqv = ggml_mul_mat(c, V, pr);
        // Before the permute, so the V product and the head-merge are separate lines.
        if (inner) g->probes.push_back({"kqv-" + sil, kqv});
        g->reads.push_back({K, V, kq, pr, msk});
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), G.d_q(), n_tokens);
        if (inner) g->probes.push_back({"kqv_merged_cont-" + sil, kqv});
        if (keep_probes) {
            ggml_set_output(inpSA);
            ggml_set_output(kqv);
            g->dbg.push_back({inpSA, kqv});
        }

        // Post-attention norm BEFORE the residual, which is the gemma shape and the reverse
        // of what every llama-derived block does.
        ggml_tensor* kqv_out = ggml_mul_mat(c, L.wo, kqv);
        if (inner) g->probes.push_back({"kqv_out-" + sil, kqv_out});
        ggml_tensor* attn = fnorm(c, kqv_out, L.post_attn_norm, eps);
        if (inner) g->probes.push_back({"sa_normed-" + sil, attn});
        ggml_tensor* attn_out = ggml_add(c, attn, inpSA);
        // The reference gives the name "attn_out" to two different quantities, and which one
        // depends on the layer. A layer that has its own wv - the 25 windowed ones - goes
        // through build_std_attention, which does cb(cur, "attn_out", il) and only THEN adds
        // the residual (llama-build-context.cpp:3696). A layer without wv - the five full
        // ones - is built inline in build_gemma4, which adds the residual first and names the
        // sum (build_gemma4.cpp:1021).
        //
        // Probing our post-residual value under that name on every layer therefore compared
        // two different things on 25 layers of 30, and reported 100-500% relative L2 for it.
        // That reads exactly like a broken attention block, which is the expensive kind of
        // wrong: it points the search at the graph when the fault is in the comparison.
        if (keep_probes) {
            g->probes.push_back({"attn_out-" + std::to_string(il),
                                 L.wv ? attn : attn_out});
        }

        // ---- the dense half. gelu(gate) * up, which is what LLM_FFN_GELU + LLM_FFN_PAR
        // resolves to; ggml_fused_up_gate is the reference's own op for it.
        ggml_tensor* hd_in = fnorm(c, attn_out, L.ffn_norm, eps);
        if (keep_probes) g->probes.push_back({"ffn_norm_1-" + sil, hd_in});
        ggml_tensor* dense = ggml_mul_mat(c, L.down,
            ggml_fused_up_gate(c, L.up, L.gate, hd_in, GGML_UNARY_OP_GELU));

        // ---- the routed half.
        ggml_tensor* moe_in = fnorm(c, attn_out, L.pre_ffw_norm_2, eps);
        if (keep_probes) g->probes.push_back({"ffn_norm_2-" + std::to_string(il), moe_in});
        // ffn_gate_inp.scale as it sits in the FILE is not the tensor the reference's graph
        // sees. llm_scale_gate_inp_s (llama.cpp:3805) walks every gemma4 layer at LOAD time
        // and multiplies that vector in place by 1/sqrt(n_embd) - 1/53.066 here. Reading the
        // weight straight out of the gguf, as everything else in this file legitimately
        // does, leaves the router logits 53x too large; softmax then collapses onto the top
        // expert, top-k picks the same eight (scaling is monotone) and the renormalised
        // weights come out near one-hot instead of a mixture. Nothing about the shapes,
        // the tensor names or the generated text betrays it.
        //
        // The scale is applied to the normed activation rather than to the weight because
        // the weight is mmapped and shared; multiplying x by s before the matmul is the same
        // product as the reference's pre-scaled w, and it is the same magnitude going into
        // mul_mat, which matters on the iqk path where the activation vector is quantised.
        ggml_tensor* rnormed = fnorm(c, attn_out, L.router_scale, eps);
        if (h.f_router_scale != 1.0f) rnormed = ggml_scale(c, rnormed, h.f_router_scale);
        ggml_tensor* rlogits = ggml_mul_mat(c, L.router, rnormed);
        ggml_tensor* probs = ggml_soft_max(c, rlogits);
        ggml_tensor* sel = ggml_top_k(c, probs, h.n_expert_used);
        ggml_tensor* weights = ggml_get_rows(c,
            ggml_reshape_3d(c, probs, 1, h.n_expert, n_tokens), sel);
        weights = ggml_reshape_2d(c, weights, h.n_expert_used, n_tokens);
        weights = ggml_div(c, weights, ggml_sum_rows(c, weights));
        weights = ggml_reshape_3d(c, weights, 1, h.n_expert_used, n_tokens);

        ggml_tensor* xe = ggml_reshape_3d(c, moe_in, h.n_embd, 1, n_tokens);
        // The experts, from ONE fused tensor. Its rows are [0, n_ff_exp) gate and
        // [n_ff_exp, 2*n_ff_exp) up, and the result is (up.x) * GELU(gate.x) - verified down
        // to the kernel's own argument order (ggml.c passes base+nb02/2 as Aup and base as
        // Agate; iqk_mul_mat.cpp applies the unary to Agate's product). Calling the fused op
        // rather than slicing two views is what keeps this bit-identical to the reference.
        //
        // NOTE for whoever unblocks --resident and --gpu-experts here: those need one id list
        // dispatched twice, and this single op cannot be split that way. The split is blocked
        // on verification, not on being impossible - build the two halves as two
        // ggml_moe_up_gate calls over the same fused tensor with the two id lists, keep the
        // per-slot add BEFORE the weighting exactly as build_step does, and check it against
        // the unsplit path. The row ranges above are the only thing that needed research.
        ggml_tensor* par = ggml_moe_up_gate(c, L.gate_up_exps, nullptr, xe, sel,
                                            GGML_UNARY_OP_GELU);
        ggml_tensor* eo = ggml_mul_mat_id(c, L.down_exps, par, sel);
        // Weight by the router, fold the eight slots, and apply the per-expert scale - all
        // in ONE node.
        //
        // Written this way for two reasons that happen to agree. The first is fidelity: the
        // reference's cparams.fused_mmad defaults to TRUE (llama.cpp:7728), so the decode we
        // compare against takes exactly this op, with the scale handed in through src[2] and
        // the chosen ids through src[3]. The second is cost. The obvious spelling - multiply
        // by the weights, then add eight slices - is eight nodes, and folding the scale in by
        // hand (repeat_4d, get_rows, mul) is three more. At 27 us of launch per node and
        // thirty layers that spelling would cost about 9 ms per token to compute something
        // bit-identical.
        //
        // Bit-identical is not an approximation here. The kernel forms s = w[j]*scale[id[j]]
        // once per slot and accumulates y += e[j][k]*s from slot zero upward, which is the
        // same value and the same left-to-right association as the long spelling.
        ggml_tensor* routed = ggml_mul_multi_add(c, eo, weights);
        if (L.down_scale) {
            // Assigned after construction, as the reference does. Safe for the graph walk:
            // ggml_build_forward_expand visits every src, and `sel` is already an ancestor
            // through eo, so nothing is scheduled out of order by this.
            routed->src[2] = L.down_scale;
            routed->src[3] = sel;
        }
        // The reference's own name for this, so it can be compared at all: llm_build_moe_ffn
        // names the mul_multi_add result "ffn_moe_weighted" and only then hands it the scale
        // through src[2], so the captured value is the routed half WITH ffn_down_exps.scale
        // folded in - exactly what ours is. "ffn_moe_out" was our invention and matched
        // nothing, which is why the routed half has never actually been checked.
        if (keep_probes) g->probes.push_back({"ffn_moe_weighted-" + sil, routed});

        // ---- the join. Each half gets its own rms and its own weight, then they are added:
        // one op in the reference, and one op here, because splitting it into two norms and
        // an add would reassociate nothing but would compute the two scales in a different
        // order from the kernel that is being compared against.
        ggml_tensor* comb = ggml_fused_rms_rms_add(c, dense, L.post_ffw_norm_1,
                                                   routed, L.post_ffw_norm_2, eps);
        if (keep_probes) g->probes.push_back({"ffn_moe_combined-" + std::to_string(il), comb});
        comb = fnorm(c, comb, L.post_ffw_norm, eps);
        cur = ggml_add(c, comb, attn_out);
        // A single learned scalar per layer, 0.07 at layer 0 and 0.20 at layer 29. Not a
        // normalisation constant that could be folded away - dropping it changes the residual
        // stream by more than an order of magnitude.
        if (L.out_scale) cur = ggml_mul(c, cur, L.out_scale);
        if (keep_probes) g->probes.push_back({"l_out-" + std::to_string(il), cur});
    }

    if (n_tokens > 1 && !all_logits) {
        cur = ggml_cont(c, ggml_view_2d(c, cur, h.n_embd, 1, cur->nb[1],
                                        size_t(n_tokens - 1) * cur->nb[1]));
    }
    cur = fnorm(c, cur, w.out_norm, eps);
    if (keep_probes) g->probes.push_back({"result_norm", cur});
    g->logits = ggml_mul_mat(c, w.out, cur);
    // 30*tanh(x/30). Stated in the file and applied by the reference; without it the top of
    // the distribution is uncapped and every temperature and top-p threshold means something
    // slightly different from what it means in llama-cli.
    if (h.f_logit_softcap > 0.0f) {
        g->logits = ggml_softcap(c, g->logits, 1.0f / h.f_logit_softcap, h.f_logit_softcap);
    }
    ggml_set_output(g->logits);
    for (auto& pr : g->probes) ggml_set_output(pr.second);
    ggml_build_forward_expand(g->gf, g->logits);
    g->alloc = ggml_gallocr_new(buft);
    if (!g->alloc) {
        printf("gemma4: ggml_gallocr_new не удался\n");
        return false;
    }
    if (!ggml_gallocr_reserve(g->alloc, g->gf) || !ggml_gallocr_alloc_graph(g->alloc, g->gf)) {
        printf("gemma4: граф не разместился (n_tokens %d, n_kv %d)\n", n_tokens, n_kv);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// Qwen 3.6 35B-A3B
// ---------------------------------------------------------------------------------------

// The routed-plus-shared feed-forward that hangs off every one of the forty layers, of both
// kinds. Factored out because the two layer kinds differ only in what feeds it.
//
//   h      = rms(attn_out) * post_attention_norm
//   routed = MoE(h) + attn_out          <- the residual joins the ROUTED half, before shared
//   shared = sigmoid(shexp_gate.h) * SharedFFN(h)
//   out    = routed + shared
//
// The order of those last two lines is the reference's and is kept: adding the residual to
// the routed half first and the shared expert afterwards is a different association from
// adding both to the residual at the end, and this project has already paid for treating
// that kind of difference as cosmetic.
static ggml_tensor* qwen35_ffn(ggml_context* c, const HParams& h,
                               const Qwen35Weights::Layer& L, ggml_tensor* attn_out,
                               int n_tokens, ggml_tensor** normed_out) {
    const float eps = h.rms_eps;
    ggml_tensor* x = fnorm(c, attn_out, L.ffn_norm, eps);
    if (normed_out) *normed_out = x;

    ggml_tensor* rlogits = ggml_mul_mat(c, L.router, x);
    ggml_tensor* probs = ggml_soft_max(c, rlogits);
    ggml_tensor* sel = ggml_top_k(c, probs, h.n_expert_used);
    ggml_tensor* weights = ggml_get_rows(c,
        ggml_reshape_3d(c, probs, 1, h.n_expert, n_tokens), sel);
    weights = ggml_reshape_2d(c, weights, h.n_expert_used, n_tokens);
    weights = ggml_div(c, weights, ggml_sum_rows(c, weights));
    weights = ggml_reshape_3d(c, weights, 1, h.n_expert_used, n_tokens);

    ggml_tensor* xe = ggml_reshape_3d(c, x, h.n_embd, 1, n_tokens);
    ggml_tensor* up = ggml_mul_mat_id(c, L.up_exps, xe, sel);
    ggml_tensor* gt = ggml_mul_mat_id(c, L.gate_exps, xe, sel);
    // silu(gate) * up in one node, which is the op the reference resolves LLM_FFN_SILU to.
    ggml_tensor* par = ggml_fused_mul_unary(c, gt, up, GGML_UNARY_OP_SILU);
    ggml_tensor* eo = ggml_mul_mat_id(c, L.down_exps, par, sel);
    // Weight and fold in one node - see the note in build_gemma4_step. Eight slots means
    // eight nodes saved a layer, and this model has forty layers.
    ggml_tensor* routed = ggml_mul_multi_add(c, eo, weights);
    routed = ggml_add(c, routed, attn_out);

    // The shared expert. Always active on every layer and every token - static traffic, not
    // expert traffic - and gated by a single sigmoid scalar per token.
    ggml_tensor* shared = ggml_mul_mat(c, L.down_shexp,
        ggml_fused_up_gate(c, L.up_shexp, L.gate_shexp, x, GGML_UNARY_OP_SILU));
    ggml_tensor* sg = ggml_mul_mat(c, L.shexp_gate, x);      // [1, n_tokens]
    if (n_tokens == 1) {
        // The reference takes the fused form only for a single token, and the branch is on
        // the same quantity, so a decode step and a prefill row go through the same kernel
        // here as they do there.
        shared = ggml_fused_mul_unary(c, sg, shared, GGML_UNARY_OP_SIGMOID);
    } else {
        shared = ggml_mul(c, shared, ggml_sigmoid(c, sg));
    }
    return ggml_add(c, routed, shared);
}

// One layer of gated delta-net: a linear-attention recurrence with a fixed-size state and no
// KV cache. Thirty of qwen35moe's forty layers are this.
//
// Two ggml ops carry almost all of it - ggml_ssm_conv and ggml_delta_net - and using them
// rather than unrolling the recurrence is the single most important decision in this
// function. It buys arithmetic identity with the reference, which is what makes a comparison
// mean anything; and it buys node count, because the alternative is a scan whose length is
// the token count. A hand-built recurrence would be correct and unusable.
//
// The recurrence, per V head, per token, with d = 128 (from ggml.c and the iqk kernel that
// actually runs):
//     decay = exp(min(g,50));  b = sigmoid(beta)
//     score = <k, q> / sqrt(d)                 (q and k arrive L2-normalised)
//     v'    = S.k ;  v_new = b*v - b*decay*v'
//     out   = (S.q)*decay/sqrt(d) + v_new*score
//     S     = decay*S + v_new (x) k ,  clamped to +-1e6
static ggml_tensor* qwen35_delta_layer(ggml_context* c, ggml_cgraph* gf, const HParams& h,
                                       const Qwen35Weights::Layer& L, ggml_tensor* st,
                                       ggml_tensor* seq_ids, ggml_tensor* inp, int n_tokens) {
    const float eps = h.rms_eps;
    const int Sk = h.ssm_d_state;                     // 128, the key/query head width
    const int Hk = h.ssm_n_group;                     // 16 K heads
    const int Hv = h.ssm_dt_rank;                     // 32 V heads
    const int Sv = h.ssm_d_inner / Hv;                // 128, the value head width
    const int key_dim = Sk * Hk;                      // 2048
    const int val_dim = Sv * Hv;                      // 4096
    const int conv_dim = key_dim * 2 + val_dim;       // 8192
    const int dconv = h.ssm_d_conv;                   // 4
    const int conv_state_dim = (dconv - 1) * conv_dim;
    const int ssm_state_dim = Sv * Sv * Hv;
    const size_t esz = sizeof(float);

    ggml_tensor* x = fnorm(c, inp, L.attn_norm, eps);

    // q | k | v in one projection, and the output gate in another.
    ggml_tensor* qkv = ggml_mul_mat(c, L.wqkv, x);            // [conv_dim, n_tokens]
    ggml_tensor* z = ggml_mul_mat(c, L.wqkv_gate, x);         // [val_dim, n_tokens]

    // beta and the decay. ssm_a holds -exp(A_log) - every entry in the file is negative -
    // so `gate` comes out negative and exp(gate) lands in (0,1) as a decay must.
    ggml_tensor* beta = ggml_reshape_4d(c, ggml_mul_mat(c, L.ssm_beta, x), Hv, 1, n_tokens, 1);
    ggml_tensor* alpha = ggml_reshape_3d(c, ggml_mul_mat(c, L.ssm_alpha, x), Hv, n_tokens, 1);
    ggml_tensor* gate = ggml_mul(c, ggml_softplus(c, ggml_add(c, alpha, L.ssm_dt)), L.ssm_a);

    // The state: a convolution window of dconv-1 positions, then the recurrent matrices.
    // Both are views of one slot so that a single pair of copies at the end writes it back.
    ggml_tensor* conv_state = ggml_reshape_3d(c,
        ggml_view_2d(c, st, conv_state_dim, 1, st->nb[1], 0), dconv - 1, conv_dim, 1);
    ggml_tensor* state = ggml_reshape_4d(c,
        ggml_view_2d(c, st, ssm_state_dim, 1, st->nb[1], size_t(conv_state_dim) * esz),
        Sv, Sv, Hv, 1);

    // The causal depthwise convolution over q|k|v, and the new window, in one op. Its output
    // is the convolved values first and the rolling window after them.
    ggml_tensor* conv_raw = ggml_ssm_conv(c, conv_state, qkv, L.ssm_conv1d, seq_ids, nullptr);
    ggml_tensor* y = ggml_silu(c, ggml_view_2d(c, conv_raw, conv_dim, n_tokens,
                                               size_t(conv_dim) * esz, 0));
    const size_t rowq = size_t(conv_dim) * esz;
    ggml_tensor* q = ggml_view_4d(c, y, Sk, Hk, n_tokens, 1,
                                  size_t(Sk) * esz, rowq, rowq * n_tokens, 0);
    ggml_tensor* k = ggml_view_4d(c, y, Sk, Hk, n_tokens, 1,
                                  size_t(Sk) * esz, rowq, rowq * n_tokens,
                                  size_t(key_dim) * esz);
    ggml_tensor* v = ggml_view_4d(c, y, Sv, Hv, n_tokens, 1,
                                  size_t(Sv) * esz, rowq, rowq * n_tokens,
                                  size_t(2 * key_dim) * esz);

    // The order of l2_norm and permute is not interchangeable, and the reference switches on
    // exactly this quantity. With more than one token the permute has to come first, because
    // l2_norm's result is what carries the contiguity ggml_delta_net asserts; with a single
    // token the permuted view is contiguous anyway - every axis it moves has extent one - so
    // normalising first and permuting after is both legal and one fewer copy.
    if (n_tokens > 1) {
        q = ggml_l2_norm(c, ggml_permute(c, q, 0, 2, 1, 3), eps);
        k = ggml_l2_norm(c, ggml_permute(c, k, 0, 2, 1, 3), eps);
    } else {
        q = ggml_permute(c, ggml_l2_norm(c, q, eps), 0, 2, 1, 3);
        k = ggml_permute(c, ggml_l2_norm(c, k, eps), 0, 2, 1, 3);
    }

    // Into the layout the op reads. These permutes are pure bookkeeping - no kernel runs -
    // and the layouts they produce are the ones the iqk kernel indexes: g and beta as
    // [token][head] with head contiguous, v strided by the op's own nb1/nb2/nb3.
    ggml_tensor* vp = ggml_permute(c, v, 0, 2, 1, 3);
    ggml_tensor* gp = ggml_permute(c, gate, 2, 0, 3, 1);
    ggml_tensor* bp = ggml_permute(c, beta, 2, 0, 1, 3);
    ggml_tensor* state_flat = ggml_reshape_4d(c, state, Sv, Sv * Hv, 1, 1);

    ggml_tensor* res = ggml_delta_net(c, q, k, vp, gp, bp, state_flat, nullptr);
    // How the 32 V heads map onto the 16 K heads. Type 1 is head % 16 - the block mapping -
    // and type 0 would be head / 2. The reference picks 1 whenever beta and alpha are stored
    // as separate tensors, which is this file. Getting it wrong pairs every V head with the
    // wrong K head and still produces text.
    res->op_params[0] = 1;

    const size_t out_elems = size_t(Sv) * size_t(Hv) * size_t(n_tokens);
    ggml_tensor* out = ggml_view_4d(c, res, Sv, Hv, n_tokens, 1, size_t(Sv) * esz,
                                    size_t(Sv * Hv) * esz, out_elems * esz, 0);
    ggml_tensor* new_state = ggml_reshape_4d(c,
        ggml_view_1d(c, res, ssm_state_dim, out_elems * esz), Sv, Sv, Hv, 1);

    // Write both halves of the state back. The reference expands the recurrent copy first,
    // and the order is load-bearing on its side (its fusion pass only matches the copy when
    // nothing but view no-ops stands between it and the op); ours follows it so the two
    // graphs schedule the same way.
    ggml_tensor* new_conv = ggml_cont(c,
        ggml_view_2d(c, conv_raw, dconv - 1, conv_dim, size_t(dconv) * esz,
                     (1 + size_t(conv_dim) * size_t(n_tokens)) * esz));
    ggml_tensor* ssm_cpy = ggml_cpy(c, ggml_reshape_2d(c, new_state, ssm_state_dim, 1),
        ggml_view_2d(c, st, ssm_state_dim, 1, st->nb[1], size_t(conv_state_dim) * esz));
    ggml_build_forward_expand(gf, ssm_cpy);
    ggml_tensor* conv_cpy = ggml_cpy(c, ggml_reshape_2d(c, new_conv, conv_state_dim, 1),
        ggml_view_2d(c, st, conv_state_dim, 1, st->nb[1], 0));
    ggml_build_forward_expand(gf, conv_cpy);

    // The gated output norm: rms over the head width, then the projection's own gate.
    ggml_tensor* o2 = ggml_reshape_2d(c, out, Sv, int64_t(Hv) * n_tokens);
    ggml_tensor* z2 = ggml_reshape_2d(c, z, Sv, int64_t(Hv) * n_tokens);
    ggml_tensor* on = ggml_fused_mul_unary(c, z2, fnorm(c, o2, L.ssm_norm, eps),
                                           GGML_UNARY_OP_SILU);
    ggml_tensor* proj = ggml_mul_mat(c, L.ssm_out,
                                     ggml_reshape_2d(c, on, val_dim, n_tokens));
    return ggml_add(c, proj, inp);
}

// One graph for the whole hybrid. Attention every fourth layer, delta-net in between, and
// the same feed-forward on both.
bool build_qwen35_step(Graph* g, ggml_backend_buffer_type_t buft, const HParams& h,
                       const Qwen35Weights& w, Cache& kv, DeltaState& ds, int n_tokens,
                       int n_past, int n_kv, bool all_logits, bool keep_probes) {
    if (n_tokens <= 0 || n_kv <= 0 || n_past < 0) {
        printf("qwen35moe: бессмысленные размеры n_tokens %d, n_past %d, n_kv %d\n",
               n_tokens, n_past, n_kv);
        return false;
    }
    if (h.n_delta_layers() > 0 && ds.state_dim == 0) {
        printf("qwen35moe: слои дельта-сети есть, а состояние не выделено\n");
        return false;
    }
    const size_t n_nodes = size_t(h.n_layer) * 96 + 256;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 512) +
                           ggml_graph_overhead_custom(n_nodes, false), nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) return false;
    ggml_context* c = g->ctx;

    g->tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    g->mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_kv, n_tokens);
    // ggml_ssm_conv takes the state slot per token as an input tensor, so it has to live in
    // the writable input buffer even though every entry of it is zero for us.
    if (h.n_delta_layers() > 0) {
        g->seq_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, 1, n_tokens);
    }
    g->n_kv_built = n_kv;
    g->inputs = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!g->inputs) return false;
    g->gf = ggml_new_graph_custom(c, n_nodes, false);

    const float eps = h.rms_eps;
    // {11,11,10,0}. Their doubled sum is 64, which is n_rot: qwen35moe rotates 64 of its 256
    // head dimensions and leaves the rest alone.
    int sections[GGML_MROPE_SECTIONS] = {0};
    for (int i = 0; i < 4 && i < GGML_MROPE_SECTIONS; ++i) sections[i] = h.rope_sections[i];

    ggml_tensor* cur = ggml_get_rows(c, w.tok_embd, g->tokens);
    for (int il = 0; il < h.n_layer; ++il) {
        const Qwen35Weights::Layer& L = w.layers[size_t(il)];
        const LayerGeom& G = h.L(il);
        ggml_tensor* attn_out = nullptr;

        if (G.kind == LayerKind::DELTA_NET) {
            if (!ds.s[size_t(il)] || !g->seq_ids) {
                printf("qwen35moe: слой %d — дельта-сеть без состояния или без карты "
                       "последовательностей\n", il);
                return false;
            }
            attn_out = qwen35_delta_layer(c, g->gf, h, L, ds.s[size_t(il)], g->seq_ids,
                                          cur, n_tokens);
            if (keep_probes) g->probes.push_back({"ssm_output-" + std::to_string(il), attn_out});
        } else {
            const int hd = G.head_dim;
            const float kq_scale = h.f_attn_scale != 0.0f ? h.f_attn_scale
                                                          : 1.0f / std::sqrt(float(hd));
            ggml_tensor* inpSA = cur;
            ggml_tensor* x = fnorm(c, cur, L.attn_norm, eps);

            // attn_q is double width and the halves are INTERLEAVED PER HEAD: head n owns
            // rows [2n*hd, 2n*hd+hd) as query and [2n*hd+hd, 2n*hd+2hd) as output gate. Two
            // contiguous halves would be the obvious reading and would be wrong - it would
            // take the first eight heads' queries and gates and call them sixteen queries.
            ggml_tensor* qaux = ggml_mul_mat(c, L.wq, x);
            const size_t row = size_t(hd) * sizeof(float);
            ggml_tensor* q = ggml_cont(c, ggml_view_3d(c, qaux, hd, G.n_head, n_tokens,
                                                       2 * row, qaux->nb[1], 0));
            ggml_tensor* agate = ggml_cont_2d(c,
                ggml_view_3d(c, qaux, hd, G.n_head, n_tokens, 2 * row, qaux->nb[1], row),
                G.d_q(), n_tokens);
            ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, x), hd,
                                             G.n_head_kv, n_tokens);
            ggml_tensor* v = ggml_mul_mat(c, L.wv, x);

            q = fnorm(c, q, L.q_norm, eps);
            k = fnorm(c, k, L.k_norm, eps);
            // ggml_rope_multi with real sections, not the all-zero text-only form: this rope
            // is interleaved-mrope and the section widths decide which dimension gets which
            // of the three angle streams.
            q = ggml_rope_multi(c, q, g->positions, nullptr, G.n_rot, sections, h.rope_type,
                                h.n_ctx_train, G.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            k = ggml_rope_multi(c, k, g->positions, nullptr, G.n_rot, sections, h.rope_type,
                                h.n_ctx_train, G.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

            ggml_tensor* Kc = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
            ggml_tensor* Vc = ggml_cont(c, ggml_permute(c,
                ggml_reshape_3d(c, v, hd, G.n_head_kv, n_tokens), 1, 2, 0, 3));
            {
                ggml_tensor* kc = kv.k[size_t(il)];
                ggml_tensor* vc = kv.v[size_t(il)];
                ggml_tensor* kdst = ggml_view_3d(c, kc, hd, n_tokens, G.n_head_kv,
                                                 kc->nb[1], kc->nb[2],
                                                 size_t(n_past) * kc->nb[1]);
                ggml_tensor* vdst = ggml_view_3d(c, vc, n_tokens, hd, G.n_head_kv,
                                                 vc->nb[1], vc->nb[2],
                                                 size_t(n_past) * ggml_element_size(vc));
                ggml_tensor* kcpy = ggml_cpy(c, Kc, kdst);
                ggml_tensor* vcpy = ggml_cpy(c, Vc, vdst);
                ggml_build_forward_expand(g->gf, kcpy);
                ggml_build_forward_expand(g->gf, vcpy);
                g->writes.push_back({kcpy, kdst, kc, kc->nb[1], kc->ne[1], n_tokens});
                g->writes.push_back({vcpy, vdst, vc, ggml_element_size(vc), vc->ne[0],
                                     n_tokens});
            }

            ggml_tensor* Q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
            ggml_tensor* K = ggml_view_3d(c, kv.k[size_t(il)], hd, n_kv, G.n_head_kv,
                                          kv.k[size_t(il)]->nb[1],
                                          kv.k[size_t(il)]->nb[2], 0);
            ggml_tensor* V = ggml_view_3d(c, kv.v[size_t(il)], n_kv, hd, G.n_head_kv,
                                          kv.v[size_t(il)]->nb[1],
                                          kv.v[size_t(il)]->nb[2], 0);
            ggml_tensor* kq = ggml_mul_mat(c, K, Q);
            ggml_tensor* pr = ggml_soft_max_ext(c, kq, g->mask, kq_scale, 0.0f);
            ggml_tensor* kqv = ggml_mul_mat(c, V, pr);
            g->reads.push_back({K, V, kq, pr, g->mask});
            kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), G.d_q(), n_tokens);
            if (keep_probes) {
                ggml_set_output(inpSA);
                ggml_set_output(kqv);
                g->dbg.push_back({inpSA, kqv});
            }
            // The output gate, applied to the attention result BEFORE the output
            // projection.
            //
            // Two nodes, not one, and deliberately. ggml_fused_mul_unary looks like the
            // right op and is not available here: its broadcast form accepts SIGMOID but
            // needs a->ne[0] == 1, and the gate is full width; its same-shape form is the
            // one that fits and it asserts the op is GELU, RELU or SILU (ggml.c, in
            // ggml_fused_mul_unary_impl). Handing it SIGMOID with matching shapes aborts.
            //
            // The reference reached the same wall from the other side: build_std_attention
            // has this exact fusion written out and guarded by `if (false && ...)`, with a
            // note that SIGMOID would have to be added to the fused op's supported list
            // first. So two nodes is also what llama_decode runs.
            kqv = ggml_mul(c, kqv, ggml_sigmoid(c, agate));
            ggml_tensor* proj = ggml_mul_mat(c, L.wo, kqv);
            attn_out = ggml_add(c, proj, inpSA);
            // Pre-residual, because that is what the reference calls "attn_out" here.
            // build_qwen35 passes add_input=true to build_std_attention, and that function
            // names the output projection "attn_out" and adds the residual afterwards
            // (llama-build-context.cpp:3696). Same trap as gemma4's windowed layers - see
            // the longer note in build_gemma4_step.
            if (keep_probes) {
                g->probes.push_back({"attn_out-" + std::to_string(il), proj});
            }
        }

        ggml_tensor* normed = nullptr;
        cur = qwen35_ffn(c, h, L, attn_out, n_tokens, keep_probes ? &normed : nullptr);
        if (keep_probes) {
            g->probes.push_back({"ffn_inp_normed-" + std::to_string(il), normed});
            g->probes.push_back({"l_out-" + std::to_string(il), cur});
        }
    }

    if (n_tokens > 1 && !all_logits) {
        cur = ggml_cont(c, ggml_view_2d(c, cur, h.n_embd, 1, cur->nb[1],
                                        size_t(n_tokens - 1) * cur->nb[1]));
    }
    cur = fnorm(c, cur, w.out_norm, eps);
    if (keep_probes) g->probes.push_back({"result_norm", cur});
    g->logits = ggml_mul_mat(c, w.out, cur);
    ggml_set_output(g->logits);
    for (auto& pr : g->probes) ggml_set_output(pr.second);
    ggml_build_forward_expand(g->gf, g->logits);
    g->alloc = ggml_gallocr_new(buft);
    if (!g->alloc) {
        printf("qwen35moe: ggml_gallocr_new не удался\n");
        return false;
    }
    if (!ggml_gallocr_reserve(g->alloc, g->gf) || !ggml_gallocr_alloc_graph(g->alloc, g->gf)) {
        printf("qwen35moe: граф не разместился (n_tokens %d, n_kv %d)\n", n_tokens, n_kv);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// A generation loop's machinery, in one place
// ---------------------------------------------------------------------------------------

// Our cache, the one-token decode graph that is built once and re-aimed per step, and - when
// speculation is on - a fixed-width verification graph that is also built once and re-aimed.
// The same struct serves the target and the draft: it holds either a Weights or a
// DenseWeights and dispatches on which, so the cache arithmetic that needed aim_cache_writes
// to get right exists once rather than three times.
struct Generator {
    ggml_backend_t be = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    const HParams* h = nullptr;
    // Exactly one of these four is set, and which one is the architecture. Kept as four
    // pointers rather than one tagged union because the compiler then checks that a builder
    // is only ever handed the weights it was written for - the failure this file cares about
    // is not a crash, it is a graph that reads plausible tensors under the wrong names.
    const Weights* w = nullptr;         // qwen3moe, the target
    const DenseWeights* dw = nullptr;   // qwen3, the draft
    const Gemma4Weights* g4 = nullptr;  // gemma4
    const Qwen35Weights* q35 = nullptr; // qwen35moe
    // The delta-net's carried state. Empty unless some layer needs it, and shared by the
    // decode graph and the prefill graph on purpose: they are two views of one running
    // recurrence, and giving them separate states would make a generation start over from
    // nothing at the first decoded token.
    DeltaState ds;
    int n_kv_max = 0;
    int min_experts = -1;
    float expert_thresh = 1.0f;
    Cache kv;
    Graph dec;                          // one token
    Graph ver;                          // draft_max + 1 tokens, every row of logits
    int ver_width = 0;
    bool dec_ready = false;
    bool ver_ready = false;
    std::vector<int32_t> ps;            // scratch, so a step allocates nothing
    std::vector<float> mk;
    bool aim_warned = false;            // said once, not once per token
    // Whether the graphs this generator builds keep their per-layer intermediates alive.
    // Off for a real generation - it pins a tensor per layer that the allocator would
    // otherwise reuse - and on for --decode-check, where comparing a layer at a time is the
    // entire point.
    bool want_probes = false;
    // The output head on the card, when it is on. Borrowed, not owned: main owns it, because
    // it has to exist before the first graph is built and outlive the last one.
    memex::GpuStatic* gstat = nullptr;

    bool dense() const { return dw != nullptr; }

    bool init(ggml_backend_t backend, ggml_backend_buffer_type_t bt, const HParams* hp,
              const Weights* tw, const DenseWeights* dwp, int n_kv, int min_e, float thr,
              const Gemma4Weights* g4p = nullptr, const Qwen35Weights* q35p = nullptr) {
        const int n_set = (tw ? 1 : 0) + (dwp ? 1 : 0) + (g4p ? 1 : 0) + (q35p ? 1 : 0);
        if (!backend || !bt || !hp || n_set != 1) {
            printf("генератор: некорректная инициализация "
                   "(ровно один из наборов весов обязателен, задано %d)\n", n_set);
            return false;
        }
        be = backend; buft = bt; h = hp; w = tw; dw = dwp; g4 = g4p; q35 = q35p;
        n_kv_max = n_kv; min_experts = min_e; expert_thresh = thr;
        if (n_kv_max <= 0) {
            printf("генератор: бессмысленный размер кэша %d\n", n_kv_max);
            return false;
        }
        if (!kv.init(buft, *h, n_kv_max)) {
            printf("генератор: кэш на %d позиций не выделился (%.1f МБ)\n", n_kv_max,
                   double(n_kv_max) * double(h->kv_elems_per_pos()) * 2.0 / 1e6);
            return false;
        }
        if (!ds.init(buft, *h)) {
            printf("генератор: состояние дельта-сети не выделилось (%.1f МБ)\n",
                   double(h->n_delta_layers()) * double(h->delta_state_elems()) * 4.0 / 1e6);
            return false;
        }
        // No history yet. Cheap, and the alternative is a first prompt conditioned on
        // whatever the allocator handed us.
        ds.clear();
        return true;
    }

    bool build_one(Graph* g, int n_tokens, int n_past, bool all_logits) {
        if (dw) {
            return build_dense_step(g, buft, *h, *dw, kv, n_tokens, n_past, n_kv_max,
                                    all_logits);
        }
        if (g4) {
            return build_gemma4_step(g, buft, *h, *g4, kv, n_tokens, n_past, n_kv_max,
                                     all_logits, want_probes);
        }
        if (q35) {
            return build_qwen35_step(g, buft, *h, *q35, kv, ds, n_tokens, n_past, n_kv_max,
                                     all_logits, want_probes);
        }
        return build_step(g, buft, *h, *w, kv, n_tokens, n_past, n_kv_max, min_experts,
                          expert_thresh, all_logits, /*zc=*/nullptr, /*keep_dbg=*/false,
                          /*rs=*/nullptr, /*gx=*/nullptr, gstat);
    }

    // The mask is -inf everywhere and zero only where a query at position past+i may see key
    // j: that is the causal mask and the mask over the padding pad32 added, in one.
    //
    // The read side is aimed at the occupied positions first, so a step over a cache
    // allocated for 16384 positions with 1500 of them in use moves 1504 positions' worth of
    // bytes and not 16384's. The mask then follows the extent the graph ended up at.
    void set_inputs(Graph& gr, const llama_token* tk, int nt, int past) {
        const int want = std::min(pad32(past + nt), n_kv_max);
        if (!gr.aim_kv_reads(want) && !aim_warned) {
            aim_warned = true;
            printf("не удалось сузить чтение кэша до %d позиций — шаги читают все %d "
                   "выделенных\n", want, n_kv_max);
        }
        set_graph_inputs(gr, *h, tk, nt, past, &ps, &mk);
#ifdef MEMEX_FWD_GPU_EXPERTS
        // The device graphs carry the same two numbers and have to be aimed by the same call,
        // in the same place: a step aimed on one side and not the other reads the cache at one
        // length and the mask at another, which is a wrong answer rather than an error.
        if (gr.on_card && gstat && !gstat->set_step(past, want) && !aim_warned) {
            aim_warned = true;
            printf("не удалось нацелить шаг на карте: past %d, n_kv %d\n", past, want);
        }
#endif
    }

    // One graph per prompt length, which is why it is built here and thrown away: a chat turn
    // brings a different number of new tokens every time, and the build costs milliseconds
    // against a prefill that costs seconds.
    bool prefill(const llama_token* tk, int nt, int past, std::vector<float>* out) {
        if (nt <= 0 || past < 0 || past + nt > n_kv_max) {
            printf("префилл: %d токенов на позиции %d не влезают в кэш на %d\n", nt, past,
                   n_kv_max);
            return false;
        }
        // A prefill from position zero is a new conversation, so the recurrence starts over.
        // A prefill at a non-zero position is a continuation - a chat turn appended to what
        // is already in the cache - and clearing there would silently drop everything the
        // delta-net layers remember while the KV cache kept everything the attention layers
        // do. That asymmetry would read as a model that forgets only some of the context.
        if (past == 0) ds.clear();
        Graph pre;
        if (!build_one(&pre, nt, past, /*all_logits=*/false)) {
            printf("префилл: граф на %d токенов не собрался\n", nt);
            return false;
        }
        set_inputs(pre, tk, nt, past);
        ggml_backend_graph_compute(be, pre.gf);
        ggml_backend_synchronize(be);
        out->assign(size_t(h->n_vocab), 0.0f);
        ggml_backend_tensor_get(pre.logits, out->data(), 0,
                                sizeof(float) * size_t(h->n_vocab));
        pre.free_all();
#ifdef MEMEX_FWD_GPU_EXPERTS
        // The prompt ran on the host, so the card's cache is empty - and an empty cache still
        // produces fluent text, because attention over zeros is attention over something. So
        // it is copied rather than assumed, once per prompt, and the failure is fatal.
        if (gstat && gstat->layers_on()) {
            std::string uerr;
            if (!gstat->upload_kv(kv.k.data(), kv.v.data(), h->n_layer, &uerr)) {
                printf("кэш промпта не уехал на карту: %s\n", uerr.c_str());
                return false;
            }
        }
#endif
        return true;
    }

    bool build_decode(int past_init) {
        if (dec_ready) return true;
        if (!build_one(&dec, 1, past_init, /*all_logits=*/false)) {
            printf("граф декода не собрался\n");
            return false;
        }
        dec_ready = true;
        return true;
    }

    bool step(llama_token tk, int past, std::vector<float>* out) {
        if (!dec_ready) {
            printf("шаг декода до сборки графа\n");
            return false;
        }
        if (!dec.aim_cache_writes(past)) {
            printf("не удалось перенаправить запись в кэш на позицию %d\n", past);
            return false;
        }
        set_inputs(dec, &tk, 1, past);
        ggml_backend_graph_compute(be, dec.gf);
        ggml_backend_synchronize(be);
        out->assign(size_t(h->n_vocab), 0.0f);
        ggml_backend_tensor_get(dec.logits, out->data(), 0,
                                sizeof(float) * size_t(h->n_vocab));
        return true;
    }

    bool build_verify(int width, int past_init) {
        if (ver_ready) return ver_width == width;
        if (width < 2) {
            printf("граф проверки шириной %d бессмысленен\n", width);
            return false;
        }
        if (!build_one(&ver, width, past_init, /*all_logits=*/true)) {
            printf("граф проверки на %d токенов не собрался\n", width);
            return false;
        }
        if (ver.logits->ne[0] != h->n_vocab || ver.logits->ne[1] != width) {
            printf("граф проверки выдал логиты %lldx%lld, а ожидались %dx%d\n",
                   (long long)ver.logits->ne[0], (long long)ver.logits->ne[1], h->n_vocab,
                   width);
            return false;
        }
        ver_width = width;
        ver_ready = true;
        return true;
    }

    // Every drafted position's logits in one pass. Positions after the first rejection are
    // computed against a context that will be thrown away, which is exactly why the caller
    // must stop reading at the first rejection - the numbers are there, they are just
    // conditioned on tokens that did not survive.
    bool verify(const llama_token* tk, int past, std::vector<float>* rows) {
        if (!ver_ready) {
            printf("проверка до сборки графа\n");
            return false;
        }
        if (past < 0 || past + ver_width > n_kv_max) {
            printf("проверка: %d токенов на позиции %d не влезают в кэш на %d\n", ver_width,
                   past, n_kv_max);
            return false;
        }
        if (!ver.aim_cache_writes(past)) {
            printf("не удалось перенаправить запись графа проверки на позицию %d\n", past);
            return false;
        }
        set_inputs(ver, tk, ver_width, past);
        ggml_backend_graph_compute(be, ver.gf);
        ggml_backend_synchronize(be);
        rows->assign(size_t(ver_width) * size_t(h->n_vocab), 0.0f);
        ggml_backend_tensor_get(ver.logits, rows->data(), 0,
                                sizeof(float) * rows->size());
        return true;
    }

    void free_all() {
        if (ver_ready) ver.free_all();
        if (dec_ready) dec.free_all();
        kv.free_all();
        ds.free_all();
        dec_ready = false;
        ver_ready = false;
    }
};

// Catches one named tensor out of the reference decode. Comparing final logits tells you
// that something differs; comparing a named intermediate tells you where, and the names are
// the fork's own, so the two sides can be lined up without counting nodes.
struct Probe {
    std::string want;
    std::vector<float> data;
    bool found = false;
    // Every node whose name we might want, captured in one pass. A model of this size takes
    // two minutes to load, so asking for one tensor per run turns a bisect into an hour;
    // catching all of them costs 160 KB each and answers in a single load.
    bool catch_all = false;
    std::vector<std::pair<std::string, std::vector<float>>> all;

    const std::vector<float>* get(const std::string& n) const {
        for (const auto& p : all) {
            if (p.first == n) return &p.second;
        }
        return nullptr;
    }
};

int probe_cb(ggml_tensor* t, bool ask, void* user_data) {
    Probe* p = (Probe*)user_data;
    // "list" prints what the reference actually names, which is not always what the graph
    // source suggests - some nodes are computed in place and never appear under their name.
    if (p->want == "list") {
        if (ask && p->data.size() < 120) {
            printf("  узел %2zu: %-28s %s ne %lld,%lld,%lld\n", p->data.size(), t->name,
                   ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1],
                   (long long)t->ne[2]);
            p->data.push_back(0.0f);
        }
        return 0;
    }
    if (p->catch_all) {
        const std::string n = t->name;
        // Only the nodes a comparison can use: layer outputs, the normed MoE input, and the
        // final norm. Names come from the reference itself, listed with --probe list.
        // Names the reference itself emits, listed with --probe list. The four new ones are
        // for the two architectures added since: gemma4 names the joined feed-forward
        // "ffn_moe_combined" and the expert input "ffn_norm_2", and qwen35moe names a
        // delta-net layer's output "ssm_output".
        //
        // "attn_out" is in the list but is NOT comparable on gemma4, and that is worth
        // knowing before it wastes an afternoon: the reference emits that name from two
        // different places in the block - before the residual on the twenty-five layers that
        // take build_std_attention, and after it on the five that do not. l_out is
        // unambiguous everywhere and is the one to trust.
        // Inside-the-attention names, and only for layers 0 and 1. When a divergence starts
        // at layer 0 there is nothing upstream to bisect against, so the bisection has to
        // happen INSIDE the block, and these are the reference's own names for its stages:
        // attn_norm -> Qcur_normed/Kcur_normed -> Qcur_roped/Kcur_roped -> kqv_out ->
        // sa_normed. Two layers rather than thirty because Qcur alone is 4096 wide and the
        // 1100-token arm would otherwise hold about five gigabytes of captured tensors.
        const bool head_two = n.size() > 2 && (n.compare(n.size() - 2, 2, "-0") == 0 ||
                                               n.compare(n.size() - 2, 2, "-1") == 0);
        const bool inner = head_two && (n.rfind("attn_norm-", 0) == 0 ||
                                        n.rfind("q-", 0) == 0 ||
                                        n.rfind("kq-", 0) == 0 ||
                                        n.rfind("kq_soft_max_ext-", 0) == 0 ||
                                        n.rfind("kqv-", 0) == 0 ||
                                        n.rfind("kqv_merged_cont-", 0) == 0 ||
                                        n.rfind("Qcur-", 0) == 0 ||
                                        n.rfind("Kcur-", 0) == 0 ||
                                        n.rfind("Vcur-", 0) == 0 ||
                                        n.rfind("Qcur_normed-", 0) == 0 ||
                                        n.rfind("Kcur_normed-", 0) == 0 ||
                                        n.rfind("Qcur_roped-", 0) == 0 ||
                                        n.rfind("Kcur_roped-", 0) == 0 ||
                                        n.rfind("kqv_out-", 0) == 0 ||
                                        n.rfind("sa_normed-", 0) == 0);
        const bool useful = inner ||
                            n.rfind("l_out-", 0) == 0 ||
                            n.rfind("ffn_inp_normed-", 0) == 0 ||
                            n.rfind("attn_out-", 0) == 0 ||
                            n.rfind("ffn_moe_out-", 0) == 0 ||
                            n.rfind("ffn_moe_weighted-", 0) == 0 ||
                            n.rfind("ffn_norm_1-", 0) == 0 ||
                            n.rfind("ffn_moe_combined-", 0) == 0 ||
                            n.rfind("ffn_norm_2-", 0) == 0 ||
                            n.rfind("ssm_output-", 0) == 0 ||
                            n == "result_norm";
        // Contiguity is not a detail here, it is the difference between a comparison and a
        // fabrication. The capture below is a flat memcpy of ggml_nbytes, so a PERMUTED VIEW
        // - which shares its source's memory and differs only in strides - is copied in the
        // SOURCE's order and then compared, element by element, against our permuted values.
        // The result is a large disagreement with no bug behind it. The reference names one
        // such node "q" (llm_build_kqv permutes without a cont), so this is not hypothetical.
        // Skipping is right rather than transposing on the fly: a node we cannot read
        // faithfully should produce no line, not a line that has to be discounted.
        const bool readable = ggml_is_contiguous(t);
        if (ask) {
            return useful && readable && t->type == GGML_TYPE_F32 ? 1 : 0;
        }
        if (!useful || !readable || t->type != GGML_TYPE_F32 || p->get(n)) {
            return 1;
        }
        std::vector<float> d;
        d.resize(size_t(ggml_nelements(t)));
        if (ggml_backend_buffer_is_host(t->buffer)) {
            std::memcpy(d.data(), t->data, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_get(t, d.data(), 0, ggml_nbytes(t));
        }
        p->all.push_back({n, std::move(d)});
        return 1;
    }
    const bool match = p->want.size() && std::string(t->name) == p->want;
    if (ask) {
        return match ? 1 : 0;
    }
    if (!match || p->found || t->type != GGML_TYPE_F32) {
        return 1;
    }
    p->data.resize(ggml_nelements(t));
    if (ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(p->data.data(), t->data, ggml_nbytes(t));
    } else {
        ggml_backend_tensor_get(t, p->data.data(), 0, ggml_nbytes(t));
    }
    p->found = true;
    return 1;
}

void report(const char* what, const std::vector<float>& ours,
            const std::vector<float>& ref) {
    // Refuses a size mismatch instead of comparing a prefix. The reference prunes its last
    // layer to the output tokens only, so l_out-47 arrives as one token against our twenty;
    // comparing the overlap put our first token against their last and reported 209% - a
    // divergence that did not exist. This project has now been bitten by that pruning four
    // times, and every time the symptom looked like a real fault.
    if (ours.size() != ref.size()) {
        printf("  %-22s размеры не совпадают: наш %zu, эталон %zu — не сравниваю\n",
               what, ours.size(), ref.size());
        return;
    }
    const size_t n = ours.size();
    double num = 0.0, den = 0.0, worst = 0.0, mine = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = double(ours[i]) - double(ref[i]);
        num += d * d;
        den += double(ref[i]) * double(ref[i]);
        mine += double(ours[i]) * double(ours[i]);
        worst = std::max(worst, std::abs(d));
    }
    // Both RMS values, always, next to the relative error. METHODS 11 asked for this
    // after a comparison returned "identical" for two zero tensors; it earns its keep
    // again on any disagreement, because a ratio near 1.0 with a large L2 means the
    // two answers are the same size and point elsewhere, while a ratio far from 1.0
    // names a missing scale or a missing normalisation on one side. Those are
    // different bugs and the L2 alone does not tell them apart.
    const double rms_o = n ? std::sqrt(mine / double(n)) : 0.0;
    const double rms_r = n ? std::sqrt(den / double(n)) : 0.0;
    printf("  %-22s %8zu, L2 %8.4f%%, max |d| %.5f, rms %.5f / %.5f\n",
           what, n, den > 0.0 ? 100.0 * std::sqrt(num / den) : -1.0, worst,
           rms_o, rms_r);
}

// ---------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------

// Ours, over our own logits. Not a choice of taste: llama's sampler chain hangs off a
// llama_context, and the generation path here deliberately never creates one - the reference
// context that exists for the logit comparison is torn down before generation matters and is
// absent entirely under --no-ref. So the fifty lines below are the price of not calling
// llama_decode, and they are the same fifty lines llama's own samplers are.
//
// The default is plain greedy and it takes its own branch on purpose. Every change in this
// project is guarded by comparing twenty-four generated tokens against the previous binary,
// and that comparison is worth something only while the default path still reaches
// std::max_element over untouched logits - not a temperature-1 softmax that happens to have
// the same argmax.
struct Sampling {
    float temp = 0.0f;             // 0 or below is greedy
    int   top_k = 0;               // 0 means the whole vocabulary
    float top_p = 1.0f;
    float min_p = 0.0f;
    float repeat_penalty = 1.0f;
    int   repeat_last_n = 64;
    uint64_t seed = 1234;          // fixed, never clock-derived: a default run must repeat

    bool plain_argmax() const { return temp <= 0.0f && repeat_penalty == 1.0f; }
    bool stochastic() const { return temp > 0.0f; }
};

struct Sampler {
    Sampling s;
    std::mt19937_64 rng;
    std::vector<float> work;       // penalised logits
    std::vector<int> idx;          // candidate ids, sorted when a nucleus rule needs it

    explicit Sampler(const Sampling& in) : s(in), rng(in.seed) {}

    // llama.cpp's own rule: divide a positive logit and multiply a negative one, so the
    // penalty always moves a token down instead of flipping its sign around zero.
    void penalise(const std::vector<llama_token>& hist) {
        if (s.repeat_penalty == 1.0f || s.repeat_last_n <= 0 || hist.empty()) return;
        const size_t take = std::min(size_t(s.repeat_last_n), hist.size());
        for (size_t i = hist.size() - take; i < hist.size(); ++i) {
            const long long t = (long long)hist[i];
            if (t < 0 || size_t(t) >= work.size()) continue;
            float& l = work[size_t(t)];
            l = l > 0.0f ? l / s.repeat_penalty : l * s.repeat_penalty;
        }
    }

    // The distribution a draw would come from, over the whole vocabulary, zero outside the
    // surviving candidate set. Produced explicitly rather than folded into the draw because
    // speculative acceptance needs both models' distributions in hand, and one function
    // producing both is the only way the acceptance test cannot be comparing two differently
    // transformed things.
    //
    // Order is llama.cpp's: penalties, temperature, top-k, top-p, min-p. It matters - min-p
    // is relative to the best surviving candidate, so moving it before top-p changes the set.
    bool dist(const std::vector<float>& logits, const std::vector<llama_token>& hist,
              std::vector<float>* out) {
        if (logits.empty()) {
            printf("сэмплер: пустой вектор логитов\n");
            return false;
        }
        work = logits;
        penalise(hist);

        const float lmax = *std::max_element(work.begin(), work.end());
        if (!std::isfinite(lmax)) {
            printf("сэмплер: логиты не конечны (max %g)\n", double(lmax));
            return false;
        }
        const float inv_t = 1.0f / (s.temp > 0.0f ? s.temp : 1.0f);
        out->assign(work.size(), 0.0f);
        double sum = 0.0;
        // Candidates are the ids whose exponential did not underflow to zero. That is an
        // exact statement about float arithmetic, not a heuristic cutoff: a token whose
        // probability is exactly zero cannot be drawn and cannot enter a nucleus.
        idx.clear();
        for (size_t i = 0; i < work.size(); ++i) {
            const float e = std::exp((work[i] - lmax) * inv_t);
            (*out)[i] = e;
            if (e > 0.0f) {
                sum += double(e);
                idx.push_back(int(i));
            }
        }
        if (!(sum > 0.0) || idx.empty()) {
            printf("сэмплер: вся вероятностная масса обнулилась\n");
            return false;
        }
        for (int i : idx) (*out)[size_t(i)] = float(double((*out)[size_t(i)]) / sum);

        const bool need_order = (s.top_k > 0 && size_t(s.top_k) < idx.size()) || s.top_p < 1.0f;
        if (need_order) {
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                if ((*out)[size_t(a)] != (*out)[size_t(b)]) {
                    return (*out)[size_t(a)] > (*out)[size_t(b)];
                }
                return a < b;   // a stable tie-break, so the nucleus is well defined
            });
        }
        size_t keep = idx.size();
        if (s.top_k > 0 && size_t(s.top_k) < keep) keep = size_t(s.top_k);
        if (s.top_p < 1.0f) {
            double c = 0.0;
            size_t k = 0;
            while (k < keep) {
                c += double((*out)[size_t(idx[k])]);
                ++k;
                if (c >= double(s.top_p)) break;
            }
            keep = std::max<size_t>(1, k);
        }
        for (size_t i = keep; i < idx.size(); ++i) (*out)[size_t(idx[i])] = 0.0f;
        idx.resize(keep);

        if (s.min_p > 0.0f) {
            float best = 0.0f;
            for (int i : idx) best = std::max(best, (*out)[size_t(i)]);
            const float floor_p = s.min_p * best;
            size_t w = 0;
            for (size_t i = 0; i < idx.size(); ++i) {
                if ((*out)[size_t(idx[i])] >= floor_p) {
                    idx[w++] = idx[i];
                } else {
                    (*out)[size_t(idx[i])] = 0.0f;
                }
            }
            // Never empty: min_p is relative to the best candidate, so the best always
            // survives its own threshold - but an all-zero set would be a silent hang, so it
            // is asserted rather than assumed.
            if (w == 0) {
                printf("сэмплер: min-p %.4f не оставил ни одного кандидата\n",
                       double(s.min_p));
                return false;
            }
            idx.resize(w);
        }

        double renorm = 0.0;
        for (int i : idx) renorm += double((*out)[size_t(i)]);
        if (!(renorm > 0.0)) {
            printf("сэмплер: после отсечения сумма вероятностей нулевая\n");
            return false;
        }
        for (int i : idx) (*out)[size_t(i)] = float(double((*out)[size_t(i)]) / renorm);
        return true;
    }

    // Inverse-CDF draw over the surviving candidates, in the order `idx` currently holds -
    // which is the order dist() left them in, so a given seed and a given logit vector always
    // produce the same token.
    llama_token draw(const std::vector<float>& probs) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const double r = u(rng);
        double c = 0.0;
        for (int i : idx) {
            c += double(probs[size_t(i)]);
            if (r < c) return llama_token(i);
        }
        return llama_token(idx.back());   // only reachable through rounding of the last cell
    }

    llama_token pick(const std::vector<float>& logits, const std::vector<llama_token>& hist,
                     bool* ok) {
        *ok = true;
        if (s.plain_argmax()) {
            return llama_token(std::max_element(logits.begin(), logits.end()) -
                               logits.begin());
        }
        work = logits;
        penalise(hist);
        if (!s.stochastic()) {
            return llama_token(std::max_element(work.begin(), work.end()) - work.begin());
        }
        std::vector<float> p;
        if (!dist(logits, hist, &p)) {
            *ok = false;
            return 0;
        }
        return draw(p);
    }
};

// ---------------------------------------------------------------------------------------
// The byte budget
// ---------------------------------------------------------------------------------------

// What one generated token actually has to read out of memory. On a mixture-of-experts model
// this is emphatically not the file size: a token touches every dense weight but only
// n_expert_used of n_expert experts, which is the entire reason a 24.5 GB file decodes at the
// speed of a 1.8 GB one.
//
// This line has been the single most useful diagnostic in the project. 73.78 ms/token against
// 24.8 GB/s of memory bandwidth reconciled with the modelled figure to within a percent, and
// that is what established the workload is bandwidth-bound rather than compute-bound - which
// is why every optimisation since has been about bytes and not about flops.
struct Budget {
    double dense = 0.0;    // attention, norms, router, embedding row, output head
    double expert = 0.0;   // only the experts a token selects
    double kv = 0.0;       // the cache attention re-reads
    double total() const { return dense + expert + kv; }
};

Budget byte_budget(const Weights& w, const HParams& h, int n_kv, double experts_used) {
    Budget b;
    auto add = [](double* dst, ggml_tensor* t) {
        if (t) *dst += double(ggml_nbytes(t));
    };
    add(&b.dense, w.out_norm);
    add(&b.dense, w.out);
    for (const Weights::Layer& L : w.layers) {
        add(&b.dense, L.attn_norm);
        add(&b.dense, L.wq);
        add(&b.dense, L.wk);
        add(&b.dense, L.wv);
        add(&b.dense, L.wo);
        add(&b.dense, L.q_norm);
        add(&b.dense, L.k_norm);
        add(&b.dense, L.ffn_norm);
        add(&b.dense, L.router);
        double per_layer = 0.0;
        add(&per_layer, L.up);
        add(&per_layer, L.gate);
        add(&per_layer, L.down);
        // The fraction of the expert blob a single token reaches. mul_mat_id genuinely does
        // not read the rows it was not given an id for, which is what makes this a byte count
        // and not an approximation.
        b.expert += h.n_expert > 0 ? per_layer * experts_used / double(h.n_expert)
                                   : per_layer;
    }
    // One embedding row, not the table: get_rows reads exactly the rows it is asked for.
    if (w.tok_embd && h.n_vocab > 0) {
        b.dense += double(ggml_nbytes(w.tok_embd)) / double(h.n_vocab);
    }
    // Summed per layer rather than multiplied out, because for two of the three supported
    // architectures the per-layer width is not the model-wide one - and for one of them
    // three quarters of the layers hold no keys at all.
    b.kv = double(n_kv) * double(h.kv_elems_per_pos()) * 2.0;
    return b;
}

// Bytes one expert of one layer occupies: its slice of the three fused expert tensors. This is
// exactly what one promotion has to push over PCIe and what one hit keeps off the memory bus.
// Read from the file rather than taken as the simulation's 2.5 MB, which was arithmetic on a
// four-bit model and would be wrong by a factor on any other quantisation.
double expert_bytes_one(const Weights& w, const HParams& h) {
    if (w.layers.empty() || h.n_expert <= 0) return 0.0;
    const Weights::Layer& L = w.layers.front();
    double per_layer = 0.0;
    if (L.up) per_layer += double(ggml_nbytes(L.up));
    if (L.gate) per_layer += double(ggml_nbytes(L.gate));
    if (L.down) per_layer += double(ggml_nbytes(L.down));
    return per_layer / double(h.n_expert);
}

// ---------------------------------------------------------------------------------------
// Resident experts in video memory, as an option
// ---------------------------------------------------------------------------------------

// Off by default, and the defaults for everything else are the values the simulation over real
// router traces settled on: 29 of 128 per layer is what 3.6 GB of usable video memory holds,
// and 64/3/8 is the window, period and promotion budget that gave 67-80% hits at 19-50 MB of
// promotions per token. See resident_set.hpp.
struct ResidentOpt {
    int capacity = 0;      // --resident, experts per layer; 0 = off
    int window   = 64;     // --resident-window, tokens
    int period   = 3;      // --resident-period, tokens
    int budget   = 8;      // --resident-budget, promotions per refresh
    bool lfu     = true;   // --resident-policy lfu | lru

    bool on() const { return capacity > 0; }
    const char* policy_name() const { return lfu ? "lfu" : "lru"; }
};

// ---------------------------------------------------------------------------------------
// The GPU half, as an option
// ---------------------------------------------------------------------------------------

// The static head on the card. A separate option from --gpu-experts and, for now, mutually
// exclusive with it: ggml_vk_get_device caches its devices (ggml-vulkan.cpp:3067-3073), so two
// ggml_backend_vk_init(0) calls share one vk_device - one queue, one command pool, one staging
// buffer - and GpuExperts' worker thread uploads whenever it has a promotion pending, which
// includes the moment the head is running. Refused rather than raced.
struct GpuStaticOpt {
    bool on       = false;   // --gpu-static
    bool verify   = false;   // --gpu-static-verify, byte compare after the upload
    bool selftest = false;   // --gpu-static-selftest, no model needed
    int  rows     = 8;       // --gpu-static-rows, widest block of logit rows per dispatch
    int  reserve_mib = 384;  // --gpu-static-reserve
    // --gpu-static-layers: the other two thirds of the static half. Attention is 510.4 MB and
    // the routers another 48.0, against the head's 243.4 - but the head is one crossing per
    // token and this is one per LAYER, so it is a different bet and it gets its own flag.
    bool layers   = false;
};

// Off by default, and it stays off until it is MEASURED faster - not until it looks right.
// The measurement is not taken here: a timing taken while anything else is running on this
// machine has already produced a 17% spread where a clean window gives 2.7%, and a number
// with that much slack in it would be used to argue either way.
struct GpuExpertOpt {
    bool on     = false;   // --gpu-experts
    bool check  = false;   // --gpu-experts-check
    bool selftest = false; // --gpu-experts-selftest, no model needed
    int  reserve_mib = 384;// --gpu-experts-reserve, headroom left on the device heap
};

// What the run measured. Two snapshots rather than one, because the prompt and the generated
// tokens are very different regimes - the prompt starts from an empty window - and averaging
// them together would report a hit rate nobody would ever see.
struct ResidentReport {
    memex::ResidentStats warm;      // the prefill
    memex::ResidentStats gen;       // the generated tokens
    int checked      = 0;           // steps compared against the unsplit graph
    int argmax_same  = 0;
    double worst_rel = 0.0;         // worst relative L2 on the logits, split vs unsplit
    uint64_t picks_checked  = 0;    // slots where the graph's split was compared to the host's
    uint64_t split_disagree = 0;    // ...and disagreed. Must be zero.
    int set_size     = 0;           // resident experts in layer 0 when the run ended
    int win_distinct = 0;           // distinct experts the window held, layer 0
    int pending_end  = 0;           // promotions still in flight in layer 0 at the end
};

// The PCIe 3.0 x4 link this machine's card sits on, measured. A promotion is one expert's
// three tensors crossing it, so the churn is a real cost and not bookkeeping - which is the
// whole reason the policy is LFU and not LRU.
constexpr double kPcieGbs = 3.94;

// Printed as what it WOULD save, because the GPU path does not exist yet. The zoned KV cache
// already sets that precedent in print_kv_zoning: a saving that is modelled rather than
// realised has to say so on the same line, or it gets quoted later as a measurement.
void print_resident(const ResidentOpt& r, const ResidentReport& rep, const HParams& h,
                    double expert_bytes_per_token, double bytes_per_expert,
                    double bandwidth_gbs) {
    printf("  резидентные эксперты (%d из %d на слой, %.0f%%; окно %d, период %d, "
           "бюджет %d, политика %s):\n", r.capacity, h.n_expert,
           100.0 * double(r.capacity) / double(std::max(h.n_expert, 1)), r.window, r.period,
           r.budget, r.policy_name());
    if (!r.on()) {
        printf("    ВЫКЛЮЧЕНЫ (--resident 0) — включите --resident N, чтобы получить эти "
               "числа на своём тексте\n");
        return;
    }
    const memex::ResidentStats& g = rep.gen;
    printf("    один эксперт %.2f МБ, эксперты на токен %.1f МБ (все %d слоёв)\n",
           bytes_per_expert / 1e6, expert_bytes_per_token / 1e6, h.n_layer);
    printf("    прогрев на промпте: %llu токенов, попаданий %.1f%% "
           "(окно стартует пустым, поэтому эта цифра — не оценка политики)\n",
           (unsigned long long)rep.warm.tokens, 100.0 * rep.warm.hit_rate());
    printf("    на генерации: %llu токенов, попаданий %.1f%% (%llu из %llu выборов)\n",
           (unsigned long long)g.tokens, 100.0 * g.hit_rate(),
           (unsigned long long)g.hits, (unsigned long long)g.picks);
    printf("    набор слоя 0: %d резидентных, окно знает %d различных\n", rep.set_size,
           rep.win_distinct);
    const double promo_per_tok = g.promotions_per_token();
    const double promo_bytes = promo_per_tok * bytes_per_expert;
    const double promo_ms = 1000.0 * promo_bytes / (kPcieGbs * 1e9);
    printf("    подкачек %.1f на токен = %.1f МБ по PCIe при %.2f ГБ/с = %.1f мс/токен "
           "(обновлений %llu, бюджет ограничил %llu из них)\n", promo_per_tok,
           promo_bytes / 1e6, kPcieGbs, promo_ms, (unsigned long long)g.refreshes,
           (unsigned long long)g.budget_bound);
    // Requested against activated, separately. They are different quantities the moment
    // activation is deferred, and averaging or reporting only one of them would hide exactly
    // the thing this mechanism exists to make visible.
    if (g.activations > 0 || rep.pending_end > 0) {
        printf("    отложенная активация: запрошено %llu, ПОДТВЕРЖДЕНО %llu, в полёте на "
               "конец прогона %d (слой 0)\n",
               (unsigned long long)g.promotions, (unsigned long long)g.activations,
               rep.pending_end);
        printf("      подкачка доходит за %.2f токена в среднем; период обновления %d — %s\n",
               g.tokens_to_land(), r.period,
               g.tokens_to_land() <= double(r.period)
                   ? "успевает, период выбран верно"
                   : "НЕ УСПЕВАЕТ: следующее обновление начинается раньше, чем доходит "
                     "предыдущее — период мал");
        if (g.room_bound > 0) {
            printf("      обновлений, где свободных слотов не осталось из-за незавершённых "
                   "подкачек: %llu\n", (unsigned long long)g.room_bound);
        }
    }
    const double from_vram = expert_bytes_per_token * g.hit_rate();
    const double from_ram = expert_bytes_per_token - from_vram;
    printf("    ИЗ ВИДЕОПАМЯТИ вместо ОЗУ было бы %.1f МБ/токен, в ОЗУ осталось бы "
           "%.1f МБ/токен\n", from_vram / 1e6, from_ram / 1e6);
    printf("      это %.2f мс/токен, которые CPU при %.1f ГБ/с не читал бы; PCIe на "
           "подкачки просит %.1f мс — %s\n",
           1000.0 * from_vram / (bandwidth_gbs * 1e9), bandwidth_gbs, promo_ms,
           promo_ms < 1000.0 * from_vram / (bandwidth_gbs * 1e9)
               ? "меньше, то есть подкачка сама себя оплачивает"
               : "БОЛЬШЕ, то есть подкачка съедает выигрыш");
    printf("    (ГРАФИЧЕСКОГО ПУТИ ПОКА НЕТ — строки выше это то, чего он стоил бы; "
           "обе половины считает CPU, поэтому сейчас это чистая бухгалтерия)\n");
    if (rep.picks_checked > 0) {
        printf("    расщепление против набора на хосте: %llu слотов сверено, расхождений "
               "%llu%s\n", (unsigned long long)rep.picks_checked,
               (unsigned long long)rep.split_disagree,
               rep.split_disagree == 0 ? " — маска дошла до графа" : " — МАСКА НЕ ТА");
    }
    if (rep.checked > 0) {
        // Bit-exactness is the claim, not "close": the halves are added per slot before the
        // weighting, so every slot arrives at the fold with the unsplit path's own bits. A
        // non-zero L2 here is therefore a fault, not rounding, and it is said as one.
        printf("    расщеплённый граф против нерасщеплённого: %d шагов, argmax совпал "
               "%d/%d, худшая отн. L2 логитов %.9f%%%s\n", rep.checked, rep.argmax_same,
               rep.checked, 100.0 * rep.worst_rel,
               (rep.argmax_same == rep.checked && rep.worst_rel == 0.0)
                   ? " — сумма половин ПОБИТОВО равна нерасщеплённому результату"
                   : (rep.argmax_same == rep.checked
                          ? " — токены те же, но биты РАСХОДЯТСЯ: половины складываются не "
                            "послотно"
                          : " — СУММА ПОЛОВИН РАСХОДИТСЯ"));
    }
}

// `n_kv` is the positions a step actually reads - the occupied ones rounded up to a multiple
// of 32, which is what aim_kv_reads narrows the graph to. `n_kv_occupied` and `n_kv_alloc` are
// printed beside it rather than folded into it: the budget used to be computed from the
// allocation, so at -c 16384 with 1464 positions in use it reported eleven times the cache
// bytes a token moves, and every roofline comparison made against this line inherited that.
// Printing all three makes the difference visible instead of silently corrected.
void print_budget(const Budget& b, const HParams& h, int n_kv, int n_kv_occupied,
                  int n_kv_alloc, double experts_used,
                  double bandwidth_gbs, double measured_ms_per_token,
                  const ZonedOpt& zopt, const ResidentOpt* ropt = nullptr,
                  const ResidentReport* rrep = nullptr,
                  double bytes_per_expert = 0.0) {
    printf("\nбайтовый бюджет на токен:\n");
    printf("  плотные веса и голова     %8.1f МБ\n", b.dense / 1e6);
    printf("  эксперты (%.1f из %d)     %8.1f МБ\n", experts_used, h.n_expert,
           b.expert / 1e6);
    printf("  KV на %5d позиций       %8.1f МБ\n", n_kv, b.kv / 1e6);
    printf("      занято %d, читается %d (округление до 32), выделено %d (-c)%s\n",
           n_kv_occupied, n_kv, n_kv_alloc,
           n_kv_alloc > n_kv
               ? "  — читается занятое, не выделенное"
               : "");
    printf("  итого                     %8.3f ГБ\n", b.total() / 1e9);
    // Printed whether or not --zoned was asked for, and at more than one context length on
    // purpose. At 2048 positions with a 512 window there is barely a tail and the option
    // looks like it does nothing; the row that shows what it does is the one further down,
    // and without it somebody reasonably deletes the option.
    print_kv_zoning(h, n_kv, zopt);
    if (!zopt.on) {
        printf("    (зонирование ВЫКЛЮЧЕНО — строки выше это то, чего --zoned стоил бы)\n");
    }
    // The same discipline for the resident expert set, and for the same reason: printed
    // whether or not it was asked for, because a saving nobody can see is a saving somebody
    // deletes. b.expert is already the bytes one token reads out of the expert blob at this
    // n_expert_used, which is exactly the quantity the hit rate scales.
    if (ropt) {
        static const ResidentReport empty_report;
        print_resident(*ropt, rrep ? *rrep : empty_report, h, b.expert, bytes_per_expert,
                       bandwidth_gbs);
    }
    const double floor_ms = 1000.0 * b.total() / (bandwidth_gbs * 1e9);
    printf("  при %.1f ГБ/с это %.2f мс/токен, то есть потолок %.2f ток/с\n", bandwidth_gbs,
           floor_ms, 1000.0 / floor_ms);
    if (measured_ms_per_token > 0.0) {
        const double implied = measured_ms_per_token * bandwidth_gbs * 1e9 / 1000.0;
        printf("  измерено %.2f мс/токен -> подразумевает %.3f ГБ при %.1f ГБ/с, "
               "то есть %.0f%% от модельного бюджета\n",
               measured_ms_per_token, implied / 1e9, bandwidth_gbs,
               100.0 * implied / b.total());
    }
}

// ---------------------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------------------

struct ChatMsg {
    std::string role;
    std::string content;
};

// Applies the file's own chat template through the fork's applier. That applier is not a
// jinja interpreter - it matches the template text against a list of known ones and returns
// -1 for anything else - so a refusal is a real answer about this file and gets reported,
// rather than papered over with a guessed format. A malformed prompt does not fail loudly; it
// produces plausible, wrong text, which is the worst failure mode available here.
bool apply_chat(const std::string& tmpl, const std::vector<ChatMsg>& msgs, bool add_ass,
                std::string* out, std::string* why) {
    if (tmpl.empty()) {
        *why = "в GGUF нет ключа tokenizer.chat_template";
        return false;
    }
    std::vector<llama_chat_message> raw;
    raw.reserve(msgs.size());
    for (const ChatMsg& m : msgs) {
        llama_chat_message c;
        c.role = m.role.c_str();
        c.content = m.content.c_str();
        raw.push_back(c);
    }
    const int32_t need = llama_chat_apply_template(tmpl.c_str(), raw.data(), raw.size(),
                                                   add_ass, nullptr, 0);
    if (need < 0) {
        *why = "llama_chat_apply_template не распознал шаблон "
               "(это не jinja-интерпретатор, а список известных шаблонов)";
        return false;
    }
    std::vector<char> buf(size_t(need) + 1, '\0');
    const int32_t got = llama_chat_apply_template(tmpl.c_str(), raw.data(), raw.size(),
                                                  add_ass, buf.data(), int32_t(buf.size()));
    if (got < 0 || got > need) {
        *why = "llama_chat_apply_template вернул несогласованную длину";
        return false;
    }
    out->assign(buf.data(), size_t(got));
    return true;
}

// ---------------------------------------------------------------------------------------
// The generation loop
// ---------------------------------------------------------------------------------------

// How much the winning logit won by. A greedy sequence can diverge from the reference on a
// near-tie without anything being wrong, and after the fact that is indistinguishable from a
// real fault - so the margin is recorded as it is produced, per step.
double top2_gap(const std::vector<float>& v) {
    float best = -INFINITY, second = -INFINITY;
    for (float x : v) {
        if (x > best) { second = best; best = x; }
        else if (x > second) { second = x; }
    }
    return double(best) - double(second);
}

int argmax_of(const std::vector<float>& v) {
    return int(std::max_element(v.begin(), v.end()) - v.begin());
}

// The logit comparison, lifted verbatim in definition from examples/memex-test
// (compare_logits / LogitCmp there) rather than reinvented, because a second definition of
// the same metric is a second number to argue about. The one that decides anything is
// flip_margin: twice the largest absolute difference inside the reference's own top-100,
// over the reference's own top-2 gap. Under one, the argmax cannot move - that is a proof,
// not a threshold, and it is the reason this file can say "the token is safe" about an
// approximation rather than "the error looked small".
//
// Relative L2 over the whole vocabulary is reported too and should not be read as alarming:
// softmax is invariant to an additive shift, so any constant offset lands in the numerator
// while changing no probability, and the denominator averages over ~150k tokens no sampler
// will ever reach.
struct LogitCmp {
    bool ok = false;
    const char* why = "";
    std::size_t n_ours = 0, n_ref = 0;
    double rel_full = -1.0;
    double rel_top100 = -1.0;
    double max_abs_top = 0.0;
    double gap = 0.0;
    double flip_margin = -1.0;
    double norm_ours = 0.0, norm_ref = 0.0;
    int argmax_ours = -1, argmax_ref = -1;
};

LogitCmp compare_logits(const std::vector<float>& ours, const std::vector<float>& ref) {
    LogitCmp r;
    r.n_ours = ours.size();
    r.n_ref = ref.size();
    if (ours.empty() || ref.empty()) {
        r.why = "пустой операнд";
        return r;
    }
    // Refused, not truncated and not broadcast. A comparison against a wrong-shaped or
    // all-zero reference has read as a perfect match in this project before, so both are
    // refusals rather than numbers.
    if (ours.size() != ref.size()) {
        r.why = "размеры не совпадают";
        return r;
    }
    const std::size_t n = ref.size();
    double num = 0.0, den = 0.0, sq_ours = 0.0;
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
        sq_ours += a * a;
    }
    r.norm_ours = std::sqrt(sq_ours);
    r.norm_ref = std::sqrt(den);
    if (!(den > 0.0)) {
        r.why = "норма эталона равна нулю — сравнивать не с чем";
        return r;
    }
    if (!(sq_ours > 0.0)) {
        r.why = "норма нашего вектора равна нулю — мы ничего не посчитали";
        return r;
    }
    r.rel_full = std::sqrt(num / den);

    // The reference's own ordering picks the candidate set: those are the entries a sampler
    // can reach, and the only ones whose error can change the emitted token.
    const int kmax = int(std::min<std::size_t>(n, 100));
    std::vector<int> ids;
    ids.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) ids[i] = int(i);
    std::partial_sort(ids.begin(), ids.begin() + kmax, ids.end(), [&](int a, int b) {
        if (ref[std::size_t(a)] != ref[std::size_t(b)]) {
            return ref[std::size_t(a)] > ref[std::size_t(b)];
        }
        return a < b;   // a stable tie-break, so the set is well defined
    });
    double tn = 0.0, td = 0.0;
    for (int i = 0; i < kmax; ++i) {
        const std::size_t id = std::size_t(ids[std::size_t(i)]);
        const double d = double(ours[id]) - double(ref[id]);
        tn += d * d;
        td += double(ref[id]) * double(ref[id]);
        if (std::abs(d) > r.max_abs_top) r.max_abs_top = std::abs(d);
    }
    if (td > 0.0) r.rel_top100 = std::sqrt(tn / td);
    r.gap = top2_gap(ref);
    if (r.gap > 0.0) r.flip_margin = 2.0 * r.max_abs_top / r.gap;
    r.argmax_ours = argmax_of(ours);
    r.argmax_ref = argmax_of(ref);
    r.ok = true;
    return r;
}

struct GenStats {
    double prefill_ms = 0.0;
    int prefill_tokens = 0;
    double gen_ms = 0.0;
    int gen_tokens = 0;
    int spec_rounds = 0;
    int spec_proposed = 0;
    int spec_accepted = 0;
    // How many positions of the cache are valid on return. Not the same as the context
    // length: a run that stops on an end-of-generation token has emitted that token without
    // ever writing its key and value, and a chat turn that re-used the cache one position too
    // far would attend to a key that was never written.
    int cache_valid = 0;
    std::vector<double> gaps;
};

// Draws from an unnormalised non-negative vector over `ids`. Used for the rejection
// residual, which is the one place the sampler's own candidate list is the wrong support.
llama_token draw_residual(const std::vector<double>& r, const std::vector<int>& ids,
                          std::mt19937_64* rng, bool* ok) {
    double sum = 0.0;
    for (int i : ids) sum += r[size_t(i)];
    if (!(sum > 0.0)) {
        // Can happen only if p and q are identical over the whole support, in which case any
        // token of p is a correct draw - but it is a real branch, so it is named rather than
        // silently falling through to the last id.
        *ok = false;
        return 0;
    }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double t = u(*rng) * sum;
    double c = 0.0;
    for (int i : ids) {
        c += r[size_t(i)];
        if (t < c) { *ok = true; return llama_token(i); }
    }
    *ok = true;
    return llama_token(ids.back());
}

// One generation run over an existing cache. `ctx` holds every committed token; positions
// [0, n_cached) are already in the target's cache, so a chat turn re-uses what the previous
// turn left there instead of re-reading 24 GB of weights over a prompt it has already seen.
//
// Two loops, chosen by whether a draft model was given. They are separate rather than one
// loop with a branch because the plain loop is the thing every reproducibility check in this
// project runs through, and it has to stay exactly what it was: prefill, argmax, then N steps
// of one token each with the last step's logits discarded.
bool generate(Generator& tgt, Generator* dft, Sampler& tsamp, Sampler* dsamp,
              llama_model* model, std::vector<llama_token>* ctx, int n_cached, int n_want,
              int draft_max, bool stop_on_eog, bool stream, GenStats* st) {
    if (n_want <= 0) return true;
    if (!ctx || ctx->empty()) {
        printf("генерация: пустой контекст\n");
        return false;
    }
    const int n_new = int(ctx->size()) - n_cached;
    if (n_new <= 0) {
        printf("генерация: в кэше уже %d позиций при контексте %zu — нечего префиллить\n",
               n_cached, ctx->size());
        return false;
    }
    std::vector<float> lg;
    const auto t_p = Clock::now();
    if (!tgt.prefill(ctx->data() + n_cached, n_new, n_cached, &lg)) return false;
    st->prefill_ms += ms_since(t_p);
    st->prefill_tokens += n_new;

    int past = n_cached + n_new;
    if (!tgt.build_decode(past)) return false;

    // The margin is recorded against the token it belongs to rather than at the moment it was
    // computed: under speculation a token can be drawn one round and emitted the next, and a
    // gaps vector that is off by one turns "first divergence at step 12" into a lie.
    auto emit = [&](llama_token t, double gap) {
        ctx->push_back(t);
        st->gaps.push_back(gap);
        ++st->gen_tokens;
        if (stream) {
            char piece[128];
            const int len = llama_token_to_piece(model, t, piece, sizeof(piece) - 1, 0, true);
            if (len > 0) {
                fwrite(piece, 1, size_t(len), stdout);
            }
        }
    };

    bool ok = true;
    llama_token next = tsamp.pick(lg, *ctx, &ok);
    if (!ok) return false;
    double next_gap = top2_gap(lg);

    const bool spec = dft && dsamp && draft_max > 0;
    const auto t_g = Clock::now();

    if (!spec) {
        for (int i = 0; i < n_want; ++i) {
            emit(next, next_gap);
            if (stop_on_eog && llama_token_is_eog(model, next)) break;
            if (past >= tgt.n_kv_max) {
                printf("\nкэш кончился на позиции %d из %d\n", past, tgt.n_kv_max);
                break;
            }
            if (!tgt.step(next, past, &lg)) return false;
            ++past;
            next = tsamp.pick(lg, *ctx, &ok);
            if (!ok) return false;
            next_gap = top2_gap(lg);
        }
        st->gen_ms += ms_since(t_g);
        st->cache_valid = past;
        return true;
    }

    // Speculative decoding, the Leviathan/Chen scheme. It is admissible under a
    // quality-first rule because it is not an approximation: every emitted token is drawn
    // from the target's own distribution, and the draft only decides which tokens get
    // offered. What it can change is the tie-breaking - the target's logits at a position are
    // computed inside a batch of draft_max+1 rather than alone, and two matmul shapes do not
    // agree to the last bit - so a near-tie can land the other way. That is reported rather
    // than hidden.
    //
    // Invariant at the top of each round: ctx holds every committed token, the target's cache
    // is valid for all of them except the last, and `next` is the token already drawn for the
    // position the cache stops at.
    if (!tgt.build_verify(draft_max + 1, past)) return false;

    // The draft's own cache, brought up to the committed prefix. Its prompt read is the one
    // unavoidable extra cost of the scheme at the start of a run.
    std::vector<float> dlg;
    int dpast = 0;
    {
        const auto t_dp = Clock::now();
        if (!dft->prefill(ctx->data(), int(ctx->size()), 0, &dlg)) return false;
        st->prefill_ms += ms_since(t_dp);
        // The draft now holds the whole committed prefix, so the first round feeds it exactly
        // one token - `next`, which the target drew but the draft has not seen.
        dpast = int(ctx->size());
        if (!dft->build_decode(dpast)) return false;
    }

    std::vector<float> rows;                   // (draft_max+1) x n_vocab
    std::vector<llama_token> batch;
    std::vector<llama_token> drafted;
    std::vector<std::vector<float>> qd;        // the draft's distribution per proposal
    std::vector<float> p;
    std::vector<double> resid;

    while (st->gen_tokens < n_want) {
        const int L = int(ctx->size());         // `next` will sit at position L
        if (L + draft_max + 1 > tgt.n_kv_max) {
            // Not enough room to verify a full batch: finish the run one token at a time
            // rather than shrinking the graph, which would mean a second verification graph.
            emit(next, next_gap);
            if (stop_on_eog && llama_token_is_eog(model, next)) break;
            if (past >= tgt.n_kv_max) {
                printf("\nкэш кончился на позиции %d из %d\n", past, tgt.n_kv_max);
                break;
            }
            if (!tgt.step(next, past, &lg)) return false;
            ++past;
            next = tsamp.pick(lg, *ctx, &ok);
            if (!ok) return false;
            next_gap = top2_gap(lg);
            continue;
        }

        // The draft catches up to the committed prefix plus `next`, then proposes.
        std::vector<llama_token> feed(ctx->begin() + dpast, ctx->end());
        feed.push_back(next);
        if (dpast + int(feed.size()) > dft->n_kv_max) {
            printf("\nкэш черновика кончился: %d + %zu > %d\n", dpast, feed.size(),
                   dft->n_kv_max);
            break;
        }
        if (int(feed.size()) == 1) {
            if (!dft->step(feed[0], dpast, &dlg)) return false;
        } else {
            if (!dft->prefill(feed.data(), int(feed.size()), dpast, &dlg)) return false;
        }
        dpast += int(feed.size());

        drafted.clear();
        qd.clear();
        // The history the penalties see has to grow with the proposals, or the draft would
        // penalise a different set of tokens than the target does at the same position.
        std::vector<llama_token> hist = *ctx;
        hist.push_back(next);
        for (int j = 0; j < draft_max; ++j) {
            llama_token dtok;
            if (dsamp->s.stochastic()) {
                std::vector<float> q;
                if (!dsamp->dist(dlg, hist, &q)) return false;
                dtok = dsamp->draw(q);
                qd.push_back(std::move(q));
            } else {
                // Through the sampler, not argmax, so that a repetition penalty applies to
                // the draft on exactly the terms it applies to the target.
                dtok = dsamp->pick(dlg, hist, &ok);
                if (!ok) return false;
            }
            drafted.push_back(dtok);
            hist.push_back(dtok);
            if (j + 1 < draft_max) {
                if (!dft->step(dtok, dpast, &dlg)) return false;
                ++dpast;
            }
        }
        st->spec_rounds += 1;
        st->spec_proposed += draft_max;

        // One target pass over [next, d_1 .. d_k] at positions L .. L+k. Row i is the
        // distribution at position L+i+1.
        batch.clear();
        batch.push_back(next);
        for (llama_token d : drafted) batch.push_back(d);
        if (!tgt.verify(batch.data(), L, &rows)) return false;
        // `next` is now genuinely in the cache at L, and so are the proposals - the ones that
        // are rejected get overwritten by the next round at the same positions, which is why
        // nothing has to be undone.
        emit(next, next_gap);
        past = L + 1;
        const bool hit_eog_on_next = stop_on_eog && llama_token_is_eog(model, next);

        auto row = [&](int i) {
            return std::vector<float>(rows.begin() + size_t(i) * size_t(tgt.h->n_vocab),
                                      rows.begin() + size_t(i + 1) * size_t(tgt.h->n_vocab));
        };

        int accepted = 0;
        bool have_next = false;      // did this round leave a token drawn but unwritten?
        bool done = hit_eog_on_next;
        while (!done && accepted < draft_max && st->gen_tokens < n_want) {
            // Row j is the target's distribution at the position drafted[j] occupies, because
            // the batch was [next, d_1..d_k] at positions L..L+k and row i answers for
            // position L+i+1.
            const std::vector<float> pl = row(accepted);
            const double gap = top2_gap(pl);
            const llama_token d = drafted[size_t(accepted)];
            if (!tsamp.s.stochastic()) {
                const llama_token best = tsamp.pick(pl, *ctx, &ok);
                if (!ok) return false;
                if (d != best) {
                    // The target's own token belongs at this position; it is emitted at the
                    // top of the next round, which is where the invariant wants it.
                    next = best;
                    next_gap = gap;
                    have_next = true;
                    break;
                }
            } else {
                if (!tsamp.dist(pl, *ctx, &p)) return false;
                const std::vector<int> pidx = tsamp.idx;
                const std::vector<float>& q = qd[size_t(accepted)];
                const double pq = double(p[size_t(d)]);
                const double qq = double(q[size_t(d)]);
                std::uniform_real_distribution<double> u(0.0, 1.0);
                const bool take = qq <= 0.0 ? pq > 0.0 : u(tsamp.rng) < pq / qq;
                if (!take) {
                    // The residual max(0, p - q). This is the step that makes the scheme
                    // exact rather than merely close: the tokens the draft over-proposed lose
                    // mass here in precisely the amount its proposal gained them.
                    resid.assign(p.size(), 0.0);
                    std::vector<int> ids;
                    for (int i2 : pidx) {
                        const double v = double(p[size_t(i2)]) - double(q[size_t(i2)]);
                        if (v > 0.0) { resid[size_t(i2)] = v; ids.push_back(i2); }
                    }
                    bool rok = false;
                    llama_token t2 = draw_residual(resid, ids, &tsamp.rng, &rok);
                    if (!rok) {
                        // p and q agreed everywhere either had mass, so a draw from p is the
                        // correct answer and nothing is being approximated.
                        tsamp.idx = pidx;
                        t2 = tsamp.draw(p);
                    }
                    // Unlike the greedy case this token IS emitted here, because it is not
                    // the target's argmax and there is no cheaper way to place it. Its key
                    // and value are not in the cache, so the step below puts them there.
                    emit(t2, gap);
                    if (stop_on_eog && llama_token_is_eog(model, t2)) { done = true; break; }
                    if (past >= tgt.n_kv_max) { done = true; break; }
                    if (!tgt.step(t2, past, &lg)) return false;
                    ++past;
                    next = tsamp.pick(lg, *ctx, &ok);
                    if (!ok) return false;
                    next_gap = top2_gap(lg);
                    have_next = true;
                    break;
                }
            }
            emit(d, gap);
            ++accepted;
            ++st->spec_accepted;
            past = L + 1 + accepted;
            if (stop_on_eog && llama_token_is_eog(model, d)) { done = true; break; }
        }

        // What the draft's cache is valid for, against the committed sequence. It holds
        // `next` at L and d_1..d_{k-1} at L+1..L+k-1, so acceptance of `accepted` proposals
        // leaves L + 1 + min(accepted, k-1) positions agreeing with what was committed. Being
        // one too generous here would let the next round condition on a rejected token, and
        // nothing would say so.
        dpast = L + 1 + std::min(accepted, draft_max - 1);

        if (done || st->gen_tokens >= n_want) break;
        if (!have_next) {
            // Every proposal survived, so the last row is a token the target has already paid
            // for - the bonus that makes the scheme worth running at all.
            const std::vector<float> pl = row(draft_max);
            next = tsamp.pick(pl, *ctx, &ok);
            if (!ok) return false;
            next_gap = top2_gap(pl);
        }
    }
    st->gen_ms += ms_since(t_g);
    st->cache_valid = past;
    return true;
}

// ---------------------------------------------------------------------------------------
// The draft model, on its own
// ---------------------------------------------------------------------------------------

// Loads only the draft, runs our dense graph over the prompt, and compares the last token's
// logits against the fork's own decode of the same file. This exists for a practical reason:
// the target is 24.5 GB and the draft is 300 MB, so the dense arm can be verified on a
// machine that has no room to load the model it will be drafting for. An unverified second
// graph is precisely the thing this project has been burned by - a graph that returns
// probabilities instead of an output looks like a working draft model, just a bad one.
int draft_selftest(const std::string& path, const std::string& prompt, int threads) {
    HParams d;
    if (!read_hparams(path.c_str(), &d)) {
        printf("черновик: не читаются гиперпараметры из %s\n", path.c_str());
        return 1;
    }
    printf("черновик %s: архитектура %s, слоёв %d, n_embd %d, голов %d/%d, head_dim %d,\n"
           "  ширина ffn %d, rms_eps %g, rope_base %g\n", path.c_str(), d.arch.c_str(),
           d.n_layer, d.n_embd, d.n_head, d.n_head_kv, d.head_dim, d.n_ff, d.rms_eps,
           d.rope_base);
    if (d.arch != "qwen3") {
        printf("черновик: архитектура \"%s\" не поддерживается; нужен плотный qwen3\n",
               d.arch.c_str());
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.repack_tensors = false;   // 300 MB, mmap is free and the point here is correctness
    llama_model* m = llama_model_load_from_file(path.c_str(), mp);
    if (!m) {
        printf("черновик: модель не загрузилась\n");
        return 1;
    }
    d.n_vocab = llama_n_vocab(m);
    d.rope_type = int(llama_rope_type(m));
    printf("черновик: словарь %d, тип rope %d\n", d.n_vocab, d.rope_type);

    std::vector<llama_token> toks;
    toks.resize(prompt.size() + 8);
    int n = llama_tokenize(m, prompt.c_str(), int(prompt.size()), toks.data(),
                           int(toks.size()), true, false);
    if (n < 0) {
        toks.resize(size_t(-n));
        n = llama_tokenize(m, prompt.c_str(), int(prompt.size()), toks.data(),
                           int(toks.size()), true, false);
    }
    if (n <= 0) {
        printf("черновик: промпт не токенизировался\n");
        return 1;
    }
    toks.resize(size_t(n));
    printf("черновик: токенов в промпте %d\n", n);

    DenseWeights dw;
    if (!collect_dense(m, d, &dw)) {
        printf("черновик: не все тензоры на месте\n");
        print_present_tensors(path.c_str(), "blk.0.");
        return 1;
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) {
        printf("черновик: backend не создался\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    const int n_kv = pad32(n + 8);
    Cache kv;
    if (!kv.init(buft, d, n_kv)) {
        printf("черновик: кэш на %d позиций не выделился\n", n_kv);
        return 1;
    }
    Graph g;
    if (!build_dense_step(&g, buft, d, dw, kv, n, 0, n_kv, /*all_logits=*/false)) {
        printf("черновик: граф не собрался\n");
        return 1;
    }
    std::vector<int32_t> ps(size_t(n), 0);
    for (int i = 0; i < n; ++i) ps[size_t(i)] = i;
    std::vector<float> mk(size_t(n_kv) * size_t(n), -INFINITY);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i && j < n_kv; ++j) {
            mk[size_t(i) * size_t(n_kv) + size_t(j)] = 0.0f;
        }
    }
    ggml_backend_tensor_set(g.tokens, toks.data(), 0, ggml_nbytes(g.tokens));
    ggml_backend_tensor_set(g.positions, ps.data(), 0, ggml_nbytes(g.positions));
    ggml_backend_tensor_set(g.mask, mk.data(), 0, ggml_nbytes(g.mask));
    ggml_backend_graph_compute(be, g.gf);
    ggml_backend_synchronize(be);

    std::vector<float> ours(size_t(d.n_vocab), 0.0f);
    ggml_backend_tensor_get(g.logits, ours.data(), 0, sizeof(float) * size_t(d.n_vocab));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn = false;
    llama_context* lctx = llama_init_from_model(m, cp);
    if (!lctx) {
        printf("черновик: эталонный контекст не создался\n");
        return 1;
    }
    if (llama_decode(lctx, llama_batch_get_one(toks.data(), n, 0, 0))) {
        printf("черновик: llama_decode не прошёл\n");
        return 1;
    }
    const float* rl = llama_get_logits(lctx);
    if (!rl) {
        printf("черновик: эталонные логиты недоступны\n");
        return 1;
    }
    std::vector<float> ref(rl, rl + size_t(d.n_vocab));

    printf("\nсверка логитов черновика на последнем токене:\n");
    report("draft_logits", ours, ref);
    auto amax = [](const std::vector<float>& v) {
        return int(std::max_element(v.begin(), v.end()) - v.begin());
    };
    const int a_o = amax(ours), a_r = amax(ref);
    char bo[64] = {0}, br[64] = {0};
    llama_token_to_piece(m, a_o, bo, sizeof(bo) - 1, 0, true);
    llama_token_to_piece(m, a_r, br, sizeof(br) - 1, 0, true);
    printf("  лучший токен: эталон %d '%s', наш %d '%s' — %s\n", a_r, br, a_o, bo,
           a_r == a_o ? "совпал" : "РАСХОДЯТСЯ");

    g.free_all();
    kv.free_all();
    ggml_backend_free(be);
    llama_free(lctx);
    llama_free_model(m);
    llama_backend_free();
    return a_r == a_o ? 0 : 2;
}

// The draft, loaded for real use alongside the target.
struct Draft {
    llama_model* model = nullptr;
    HParams h;
    DenseWeights w;
    Generator gen;
};

bool load_draft(const std::string& path, llama_model* target_model, const HParams& target,
                int n_kv, ggml_backend_t be, ggml_backend_buffer_type_t buft, Draft* d) {
    if (!read_hparams(path.c_str(), &d->h)) {
        printf("черновик: не читаются гиперпараметры из %s\n", path.c_str());
        if (file_size_bytes(path.c_str()) == 0) {
            printf("  файл не открывается или пуст\n");
        }
        return false;
    }
    if (d->h.arch != "qwen3") {
        printf("черновик: архитектура \"%s\" не поддерживается\n", d->h.arch.c_str());
        printf("  поддерживается только qwen3 (плотный) в роли черновика\n");
        return false;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    // Not repacked. The draft is read draft_max times per round against the target's once, so
    // it is the more bandwidth-sensitive of the two - but it is also a few hundred megabytes,
    // which fits in cache pressure terms, and forcing mmap off for it would take private
    // memory away from the target, which is where the guard is already tight.
    mp.repack_tensors = false;
    d->model = llama_model_load_from_file(path.c_str(), mp);
    if (!d->model) {
        printf("черновик: модель не загрузилась (%s)\n", path.c_str());
        return false;
    }
    d->h.n_vocab = llama_n_vocab(d->model);
    d->h.rope_type = int(llama_rope_type(d->model));
    // Speculation requires the two models to agree about what a token id means. A mismatched
    // vocabulary does not fail: it produces a draft that is rejected nearly always, which
    // looks exactly like a bad draft model and costs a day to find.
    if (d->h.n_vocab != target.n_vocab) {
        printf("черновик: словарь %d против %d у целевой модели — токенизаторы разные, "
               "спекуляция была бы бессмысленной\n", d->h.n_vocab, target.n_vocab);
        return false;
    }
    // The vocabulary sizes agreeing is necessary but not sufficient - two 151936-entry
    // vocabularies can still disagree about which id is which - so the two ids whose meaning
    // the loop actually depends on are checked by value.
    if (llama_token_eos(d->model) != llama_token_eos(target_model) ||
        llama_token_bos(d->model) != llama_token_bos(target_model)) {
        printf("черновик: bos/eos не совпадают (черновик %d/%d, цель %d/%d) — "
               "токенизаторы разные\n", llama_token_bos(d->model), llama_token_eos(d->model),
               llama_token_bos(target_model), llama_token_eos(target_model));
        return false;
    }
    printf("черновик %s: qwen3, слоёв %d, n_embd %d, голов %d/%d, head_dim %d, ffn %d\n",
           path.c_str(), d->h.n_layer, d->h.n_embd, d->h.n_head, d->h.n_head_kv,
           d->h.head_dim, d->h.n_ff);
    if (!collect_dense(d->model, d->h, &d->w)) {
        printf("черновик: не все тензоры на месте\n");
        print_present_tensors(path.c_str(), "blk.0.");
        return false;
    }
    if (!d->gen.init(be, buft, &d->h, nullptr, &d->w, n_kv, -1, 1.0f)) return false;
    printf("черновик: кэш %d позиций, %.1f МБ\n", n_kv, d->gen.kv.bytes(d->h) / 1e6);
    return true;
}

// ---------------------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------------------

std::string detokenise(llama_model* model, const llama_token* t, int n) {
    std::string out;
    char piece[256];
    for (int i = 0; i < n; ++i) {
        const int len = llama_token_to_piece(model, t[i], piece, sizeof(piece) - 1, 0, true);
        if (len > 0) out.append(piece, size_t(len));
    }
    return out;
}

// How many leading tokens two sequences share. The chat loop re-formats and re-tokenises the
// whole conversation every turn, because a chat template is not guaranteed to be a pure
// append - so what may be re-used from the cache is measured rather than assumed.
int common_prefix(const std::vector<llama_token>& a, const std::vector<llama_token>& b) {
    int i = 0;
    while (i < int(a.size()) && i < int(b.size()) && a[size_t(i)] == b[size_t(i)]) ++i;
    return i;
}

int run_chat(llama_model* model, const std::string& model_path, const HParams& h,
             const Weights& w, ggml_backend_t be, ggml_backend_buffer_type_t buft, int n_ctx,
             const Sampling& samp, const std::string& system_prompt,
             const std::string& first_user, int n_predict, int min_experts,
             float expert_thresh, Draft* draft, int draft_max, double bandwidth_gbs,
             const ZonedOpt& zopt) {
    // The template, from the file. llama_model_chat_template reads the same
    // tokenizer.chat_template key; read_chat_template is the fallback for the case where the
    // model was loaded in a way that did not keep the kv around.
    const char* tp = llama_model_chat_template(model, nullptr);
    std::string tmpl = tp ? tp : read_chat_template(model_path.c_str());
    bool raw = false;
    if (tmpl.empty()) {
        printf("в GGUF нет tokenizer.chat_template — перехожу на сырые промпты "
               "без разметки роли\n");
        raw = true;
    } else {
        std::vector<ChatMsg> probe_msgs;
        probe_msgs.push_back({"user", "x"});
        std::string probe_out, why;
        if (!apply_chat(tmpl, probe_msgs, true, &probe_out, &why)) {
            printf("шаблон чата не применяется: %s\n", why.c_str());
            printf("перехожу на сырые промпты без разметки роли — ответы будут хуже, "
                   "но по крайней мере вход не будет искажён\n");
            raw = true;
        } else {
            printf("шаблон чата из GGUF применён, пробный промпт %zu символов\n",
                   probe_out.size());
        }
    }

    // No card here. run_chat takes neither the module nor the placement lambda, and it has no
    // caller (see the byte-budget note in main), so wiring the card into it would be untested
    // code reached by nothing. The card path lives in main's harness, which is what measures.
    const int n_kv_max = pad32(n_ctx);
    Generator gen;
    if (!gen.init(be, buft, &h, &w, nullptr, n_kv_max, min_experts, expert_thresh)) return 1;
    printf("чат: кэш %d позиций, %.1f МБ; предел ответа %d токенов\n", n_kv_max,
           gen.kv.bytes(h) / 1e6, n_predict);

    Sampling dsampling = samp;
    // A different stream, not the same seed. Rejection sampling is exact only while the
    // acceptance coin is independent of the draw it is judging, and two generators seeded
    // alike produce the same sequence, which would correlate the two.
    dsampling.seed = samp.seed ^ 0x9E3779B97F4A7C15ull;
    Sampler tsamp(samp);
    Sampler dsamp(dsampling);
    Generator* dptr = draft ? &draft->gen : nullptr;
    Sampler* dsptr = draft ? &dsamp : nullptr;

    std::vector<ChatMsg> msgs;
    if (!system_prompt.empty()) msgs.push_back({"system", system_prompt});

    std::vector<llama_token> cached;    // exactly what the cache holds, position for position
    int n_cached = 0;
    std::string raw_accum;
    bool first = true;
    std::string line;

    if (samp.plain_argmax()) {
        printf("сэмплирование: жадное (--temp 0)\n");
    } else {
        printf("сэмплирование: temp %.2f, top-k %d, top-p %.2f, min-p %.2f, "
               "штраф за повтор %.2f по последним %d, seed %llu\n", double(samp.temp),
               samp.top_k, double(samp.top_p), double(samp.min_p),
               double(samp.repeat_penalty), samp.repeat_last_n,
               (unsigned long long)samp.seed);
    }
    printf("пустая строка — выход.\n");

    while (true) {
        std::string user;
        if (first && !first_user.empty()) {
            user = first_user;
            printf("\n> %s\n", user.c_str());
        } else {
            printf("\n> ");
            if (!std::getline(std::cin, line)) break;
            user = line;
        }
        first = false;
        if (user.empty()) break;

        std::string formatted;
        if (raw) {
            raw_accum += user;
            formatted = raw_accum;
        } else {
            msgs.push_back({"user", user});
            std::string why;
            if (!apply_chat(tmpl, msgs, /*add_ass=*/true, &formatted, &why)) {
                printf("шаблон чата перестал применяться: %s — прекращаю\n", why.c_str());
                break;
            }
        }

        std::vector<llama_token> seq;
        seq.resize(formatted.size() + 8);
        // add_special only on the first turn: a template that already emits the special
        // opening tokens plus a tokeniser that adds its own would produce two of them, and
        // for a Qwen tokeniser (which adds no BOS) this is a no-op either way.
        int nt = llama_tokenize(model, formatted.c_str(), int(formatted.size()), seq.data(),
                                int(seq.size()), n_cached == 0, true);
        if (nt < 0) {
            seq.resize(size_t(-nt));
            nt = llama_tokenize(model, formatted.c_str(), int(formatted.size()), seq.data(),
                                int(seq.size()), n_cached == 0, true);
        }
        if (nt <= 0) {
            printf("промпт не токенизировался (%zu символов)\n", formatted.size());
            break;
        }
        seq.resize(size_t(nt));
        if (nt + n_predict > n_kv_max) {
            printf("не хватает контекста: %d токенов промпта плюс %d ответа против %d "
                   "в кэше (--ctx больше)\n", nt, n_predict, n_kv_max);
            break;
        }

        // Re-use of the cache is measured against what is actually in it, and capped by what
        // the previous turn certified as written.
        const int lcp = std::min(common_prefix(cached, seq), n_cached);
        const int reuse = std::min(lcp, nt - 1);   // at least one token must be prefilled

        GenStats st;
        if (!generate(gen, dptr, tsamp, dsptr, model, &seq, reuse, n_predict, draft_max,
                      /*stop_on_eog=*/true, /*stream=*/true, &st)) {
            printf("\nгенерация не удалась\n");
            break;
        }
        printf("\n");
        const std::string answer = detokenise(model, seq.data() + nt, st.gen_tokens);
        if (!raw) {
            msgs.push_back({"assistant", answer});
        } else {
            raw_accum += answer;
        }
        cached = seq;
        n_cached = std::min(st.cache_valid, int(seq.size()));

        printf("[префилл %d токенов за %.0f мс (%.2f ток/с), из кэша повторно %d; "
               "генерация %d токенов за %.0f мс (%.2f ток/с)]\n",
               st.prefill_tokens, st.prefill_ms,
               st.prefill_ms > 0.0 ? 1000.0 * st.prefill_tokens / st.prefill_ms : 0.0, reuse,
               st.gen_tokens, st.gen_ms,
               st.gen_ms > 0.0 ? 1000.0 * st.gen_tokens / st.gen_ms : 0.0);
        if (st.spec_rounds > 0) {
            printf("[спекуляция: %d раундов, принято %d из %d предложенных, "
                   "%.2f токена за раунд]\n", st.spec_rounds, st.spec_accepted,
                   st.spec_proposed,
                   double(st.spec_accepted + st.spec_rounds) / double(st.spec_rounds));
        }
        if (st.gen_tokens > 0) {
            const int occ = nt + st.gen_tokens / 2;
            const int read = std::min(pad32(occ), n_kv_max);
            const Budget b = byte_budget(w, h, read, double(h.n_expert_used));
            print_budget(b, h, read, occ, n_kv_max, double(h.n_expert_used), bandwidth_gbs,
                         st.gen_ms / st.gen_tokens, zopt);
        }
    }

    gen.free_all();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string model_path = "D:/Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf";
    std::string prompt = "The quick brown fox jumps over the lazy dog. "
                         "Write a function that reverses a linked list in place.";
    int threads = 4;
    int max_tokens = 64;
    std::string probe_name;
    int n_gen = 0;
    int decode_check = 0;
    int min_experts = -1;
    float expert_thresh = 1.0f;
    // Three states, not two: "the user said nothing" has to be distinguishable from "the user
    // said yes", because the memory guard may override the first and must never override the
    // second. Silently doing the opposite of what was asked for is worth 38% here.
    enum RepackWish { REPACK_AUTO, REPACK_FORCE_ON, REPACK_FORCE_OFF };
    RepackWish repack_wish = REPACK_AUTO;
    // WHICH tensors repacking may touch. Not a hardcoded direction: which half of the model
    // wants the interleaved layout depends on which half runs on the CPU, and that is exactly
    // what we are trying to measure. Storage size and per-token traffic are different
    // quantities - the experts are 14.6 GB of a 15.3 GB file but only 8 of 128 per layer are
    // read, so they are 53% of the traffic; attention is small in the file and read in full
    // every token. So both boundaries have to be reachable without a rebuild.
    //
    //   all      everything repackable is repacked, mmap off
    //   experts  only *_exps.* repacked; attention/head/router stay PLAIN and uploadable
    //   static   everything but *_exps.* repacked; experts stay PLAIN and uploadable
    //   none     nothing repacked (same as --no-repack)
    // Anything else is taken literally as a comma-separated substring list for "only".
    //
    // Default `experts`: 53% of per-token traffic is expert bytes and they run on the CPU, so
    // that is where the interleaved layout pays; attention, the head and the router are 47% of
    // the traffic but only 0.8 GB of the file, which makes them the half worth putting on the
    // card - and to be uploadable they have to stay in their stored type.
    std::string repack_sel = "experts";
    // mmap under a filtered repack. auto: keep the mapping when the repacked share is small
    // (there is then something left for it to hold), drop it when the repacked share is most
    // of the file - keeping it would only add a full copy of the model to the same total.
    enum MmapWish { MMAP_AUTO, MMAP_ON, MMAP_OFF };
    MmapWish mmap_wish = MMAP_AUTO;
    Sampling samp;
    bool chat = false;
    std::string system_prompt;
    int n_ctx_req = 0;             // 0 means "size it from the prompt and --gen"
    // The reference's micro-batch. 512 is llama.cpp's default and is what every number
    // in this file has been measured against, so it stays the default. It is exposed
    // because a prompt longer than it makes llama_decode run several micro-batches, and
    // that is a second explanation for any disagreement at length - one that looks
    // exactly like a windowing fault, because both only appear once the prompt is long.
    // Setting this to the prompt length collapses the reference to a single micro-batch
    // and tells the two apart in one run.
    int ref_ubatch = 512;
    // Whether the REFERENCE runs flash attention. Ours never does, and false was the obvious
    // choice: the plain path is the one our graph mirrors node for node. It is also, for
    // gemma4, the path in which the reference stores its V cache incorrectly - see the note
    // at the flag's use - so this exists to compare against the arm the fork actually works
    // in rather than the arm we happen to imitate.
    bool ref_fa = false;
    int n_predict = 256;           // chat mode's own budget; --gen drives the harness path
    bool want_ref = true;
    std::string draft_path;
    int draft_max = 0;
    bool draft_check = false;
    double bandwidth_gbs = 24.8;   // measured on this machine, see the byte-budget comment
    ZonedOpt zopt;
    ResidentOpt ropt;
    GpuExpertOpt gopt;
    GpuStaticOpt sopt;
    bool bad_arg = false;

    auto want_val = [&](int i, const char* what) {
        if (i + 1 >= argc) {
            printf("флагу %s нужен аргумент\n", what);
            bad_arg = true;
            return false;
        }
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            printf(
"llama-memex-fwd — наш собственный движок вывода.\n"
"Целевые архитектуры: qwen3moe, gemma4, qwen35moe; qwen3 — только как черновик.\n"
"Граф свой, llama.cpp используется только для GGUF, токенизатора и ядер ggml;\n"
"llama_decode не вызывается никогда, кроме эталонного сравнения.\n"
"\n"
"модель и запуск\n"
"  -m ПУТЬ              файл модели GGUF (по умолчанию %s)\n"
"  -p ТЕКСТ             промпт\n"
"  -f, --file ПУТЬ      промпт из файла (нужен для длинного контекста)\n"
"  -t N                 потоков (%d)\n"
"  -c, --ctx N          сколько позиций держать в кэше (по умолчанию по промпту)\n"
"  --tokens N           обрезать промпт до N токенов (%d)\n"
"  --gen N              сгенерировать N токенов и сверить их с эталоном (%d)\n"
"  --decode-check N     сверять КАЖДЫЙ шаг декода с llama_decode, N шагов\n"
"                       (единственная проверка, которая ловит ошибку в переносимом\n"
"                        состоянии дельта-сети: она деградирует плавно и читается\n"
"                        как «модель чуть хуже», а не как поломка)\n"
"  -n, --n-predict N    предел генерации в режиме --chat (%d)\n"
"  --no-ref             не создавать эталонный llama_context и не сверять логиты\n"
"\n"
"перепаковка тензоров — включена по умолчанию, +38%%, побитово точна\n"
"  --repack             включить несмотря на оценку памяти\n"
"  --no-repack          выключить (модель останется отображённой через mmap)\n"
"  -rtr                 то же, что --repack (старое имя, сохранено)\n"
"  --repack-only=ЧТО    какие тензоры перепаковывать (по умолчанию experts). ЧТО:\n"
"                         all      всё; mmap отключается\n"
"                         experts  только *_exps.* — внимание/голова/роутер остаются\n"
"                                  в исходном типе и годятся для выгрузки на карту\n"
"                         static   всё, КРОМЕ *_exps.* — эксперты остаются исходными\n"
"                         none     ничего (то же, что --no-repack)\n"
"                         иное     список подстрок имён через запятую\n"
"  --repack-skip=СПИСОК подстроки имён, которые НЕ перепаковывать\n"
"  --mmap / --no-mmap   принудительно оставить/убрать отображение файла\n"
"\n"
"сэмплирование — по умолчанию строго жадное, как было до его появления\n"
"  --temp F             температура (%.2f; 0 или меньше — жадный выбор)\n"
"  --top-k N            (%d; 0 — весь словарь)\n"
"  --top-p F            (%.2f)\n"
"  --min-p F            (%.2f)\n"
"  --repeat-penalty F   (%.2f)\n"
"  --repeat-last-n N    (%d)\n"
"  --seed N             (%llu)\n"
"\n"
"чат\n"
"  --chat               интерактивный цикл, строки читаются со stdin\n"
"  --system ТЕКСТ       системное сообщение\n"
"\n"
"спекулятивное декодирование — сохраняет распределение, поэтому допустимо\n"
"  -md ПУТЬ             модель-черновик (плотный qwen3)\n"
"  --draft-max N        сколько токенов набрасывать за раунд (%d; 0 — выключено)\n"
"  --draft-check        проверить граф черновика против llama_decode и выйти\n"
"\n"
"сокращение экспертов — НЕ безопасно для качества, поэтому только вручную\n"
"  --min-experts N      всегда держать первые N экспертов (%d; -1 — выключено)\n"
"  --expert-thresh F    относительный порог для остальных (%.2f)\n"
"\n"
"зонный KV-кэш — приближение, поэтому ВЫКЛЮЧЕН по умолчанию\n"
"  имена флагов те же, что у модуля examples/memex-kv: один словарь, а не два\n"
"  --zoned              включить зонный кэш на декоде (префилл всегда точный)\n"
"  --zoned-check        прогнать один и тот же промпт точным и зонным кэшем в одном\n"
"                       процессе и напечатать запас по argmax; <1 — доказательство\n"
"  --sinks N            позиций-стоков, вечно точных (%d)\n"
"  --window N           точное недавнее окно (%d)\n"
"  --notebook N         точных слотов под «блокнот» (%d; движку пока нечем их просить)\n"
"  --tail-form q8_0|f16 чем сжимать хвост (%s; f16 — контроль, сжатия нет)\n"
"  --rotate-keys on|off вращение Адамара для ключей хвоста (%s; вдвое меньше ошибки)\n"
"\n"
"резидентные эксперты в видеопамяти, ВЫКЛЮЧЕНЫ по умолчанию\n"
"  --resident N без --gpu-experts: обе половины считает CPU и складывает послотно,\n"
"  поэтому токены обязаны совпадать с выключенным флагом — это проверка расщепления\n"
"  --gpu-experts: резидентная половина уходит на Vulkan ОДНОВРЕМЕННО с CPU. Токены при\n"
"  этом совпадать НЕ обязаны: половины считают разные ядра, и они расходятся сильнее,\n"
"  чем на округление (см. --gpu-experts-selftest, там это измерено против эталонного\n"
"  декодера — точнее оказывается ядро устройства, а не CPU)\n"
"  --resident N         резидентных экспертов на слой (%d; 0 — выключено; у целевой 128)\n"
"  --resident-window N  окно наблюдения в токенах (%d)\n"
"  --resident-period N  обновлять набор каждые N токенов (%d)\n"
"  --resident-budget N  максимум подкачек за обновление (%d; 0 — без ограничения)\n"
"  --resident-policy P  lfu или lru (%s; lru — контроль, он проигрывает по подкачкам)\n"
"  --gpu-experts        считать резидентную половину на Vulkan ОДНОВРЕМЕННО с CPU\n"
"                       (ВЫКЛЮЧЕНО по умолчанию; --resident без него оставляет обе\n"
"                        половины на CPU. Без --resident N ёмкость выбирается по\n"
"                        свободной видеопамяти)\n"
"  --gpu-experts-check  дополнительно считать резидентную половину и на CPU и сверять\n"
"                       послотно (съедает весь выигрыш; для этого и существует)\n"
"  --gpu-experts-selftest   проверить весь механизм на синтетических весах, без модели\n"
"  --gpu-experts-reserve N  запас видеопамяти в МиБ, который не занимать (%d)\n"
"\n"
"staticheskie vesa v videopamjati, VYKLJUCHENY po umolchaniju\n"
"  --gpu-static         schitat vyhodnuju golovu (output.weight) na Vulkan. Eto 243.4 MB\n"
"                       iz 1714 MB, kotorye tokjen chitaet, i edinstvennaja chast statiki\n"
"                       s ODNIM peresecheniem granicy za tokjen, a ne odnim na sloj\n"
"  --gpu-static-verify  posle zalivki sverit kazhdyj bajt golovy s modelju\n"
"  --gpu-static-layers  krome golovy - vsjo vnimanie (510.4 MB), marshrutizatory (48.0)\n"
"                       i KV-kesh na kartu. Odin razrez na sloj: karta schitaet blok\n"
"                       vnimanija, ostatok i normu FFN, host - ekspertov\n"
"  --gpu-static-selftest    proverit ves mehanizm na sinteticheskoj golove, bez modeli\n"
"  --gpu-static-rows N  shirina bloka logitov za odin dispatch (%d)\n"
"  --gpu-static-reserve N   zapas videopamjati v MiB, kotoryj ne zanimat (%d)\n"
"\n"
"диагностика\n"
"  --probe ИМЯ          сверить промежуточный тензор; \"all\" — все, \"list\" — перечислить\n"
"  --bandwidth F        ГБ/с для байтового бюджета (%.1f)\n"
"  -h, --help           это сообщение\n",
                model_path.c_str(), threads, max_tokens, n_gen, n_predict,
                double(samp.temp), samp.top_k, double(samp.top_p), double(samp.min_p),
                double(samp.repeat_penalty), samp.repeat_last_n,
                (unsigned long long)samp.seed, draft_max, min_experts,
                double(expert_thresh), zopt.sinks, zopt.window, zopt.notebook,
                zopt.tail_q8 ? "q8_0" : "f16", zopt.rotate_keys ? "on" : "off",
                ropt.capacity, ropt.window, ropt.period, ropt.budget, ropt.policy_name(),
                gopt.reserve_mib,
                sopt.rows, sopt.reserve_mib,
                bandwidth_gbs);
            return 0;
        }
        else if (!strcmp(a, "-m")) { if (want_val(i, a)) model_path = argv[++i]; }
        else if (!strcmp(a, "-p")) { if (want_val(i, a)) prompt = argv[++i]; }
        // Reading the prompt from a file is not a convenience: the zoned cache only does anything
        // once the context outgrows the exact zones, which needs several hundred tokens, and a
        // prompt that long does not go on a command line. Passing one inline hit the Windows
        // argument limit before it hit an interesting context length.
        else if (!strcmp(a, "-f") || !strcmp(a, "--file")) {
            if (want_val(i, a)) {
                const char* path = argv[++i];
                std::ifstream in(path, std::ios::binary);
                if (!in) { printf("не открылся файл промпта: %s\n", path); return 1; }
                std::ostringstream ss;
                ss << in.rdbuf();
                prompt = ss.str();
                if (prompt.empty()) { printf("файл промпта пуст: %s\n", path); return 1; }
            }
        }
        else if (!strcmp(a, "-t")) { if (want_val(i, a)) threads = atoi(argv[++i]); }
        else if (!strcmp(a, "-c") || !strcmp(a, "--ctx")) { if (want_val(i, a)) n_ctx_req = atoi(argv[++i]); }
        else if (!strcmp(a, "--ref-ubatch")) { if (want_val(i, a)) ref_ubatch = atoi(argv[++i]); }
        else if (!strcmp(a, "--ref-fa")) ref_fa = true;
        else if (!strcmp(a, "--no-ref-fa")) ref_fa = false;
        else if (!strcmp(a, "--tokens")) { if (want_val(i, a)) max_tokens = atoi(argv[++i]); }
        else if (!strcmp(a, "--probe")) { if (want_val(i, a)) probe_name = argv[++i]; }
        else if (!strcmp(a, "--gen")) { if (want_val(i, a)) n_gen = atoi(argv[++i]); }
        else if (!strcmp(a, "--decode-check")) {
            if (want_val(i, a)) decode_check = atoi(argv[++i]);
        }
        else if (!strcmp(a, "-n") || !strcmp(a, "--n-predict")) { if (want_val(i, a)) n_predict = atoi(argv[++i]); }
        else if (!strcmp(a, "--min-experts")) { if (want_val(i, a)) min_experts = atoi(argv[++i]); }
        else if (!strcmp(a, "--expert-thresh")) { if (want_val(i, a)) expert_thresh = float(atof(argv[++i])); }
        else if (!strcmp(a, "-rtr") || !strcmp(a, "--run-time-repack") || !strcmp(a, "--repack")) repack_wish = REPACK_FORCE_ON;
        else if (!strcmp(a, "--no-repack")) repack_wish = REPACK_FORCE_OFF;
        else if (!strncmp(a, "--repack-only=", 14)) repack_sel = a + 14;
        else if (!strcmp(a, "--repack-only")) { if (want_val(i, a)) repack_sel = argv[++i]; }
        else if (!strncmp(a, "--repack-skip=", 14)) repack_sel = std::string("skip:") + (a + 14);
        else if (!strcmp(a, "--mmap")) mmap_wish = MMAP_ON;
        else if (!strcmp(a, "--no-mmap")) mmap_wish = MMAP_OFF;
        else if (!strcmp(a, "--temp")) { if (want_val(i, a)) samp.temp = float(atof(argv[++i])); }
        else if (!strcmp(a, "--top-k")) { if (want_val(i, a)) samp.top_k = atoi(argv[++i]); }
        else if (!strcmp(a, "--top-p")) { if (want_val(i, a)) samp.top_p = float(atof(argv[++i])); }
        else if (!strcmp(a, "--min-p")) { if (want_val(i, a)) samp.min_p = float(atof(argv[++i])); }
        else if (!strcmp(a, "--repeat-penalty")) { if (want_val(i, a)) samp.repeat_penalty = float(atof(argv[++i])); }
        else if (!strcmp(a, "--repeat-last-n")) { if (want_val(i, a)) samp.repeat_last_n = atoi(argv[++i]); }
        else if (!strcmp(a, "--seed")) { if (want_val(i, a)) samp.seed = (uint64_t)strtoull(argv[++i], nullptr, 10); }
        else if (!strcmp(a, "--chat")) chat = true;
        else if (!strcmp(a, "--system")) { if (want_val(i, a)) system_prompt = argv[++i]; }
        else if (!strcmp(a, "--no-ref")) want_ref = false;
        else if (!strcmp(a, "-md") || !strcmp(a, "--model-draft")) { if (want_val(i, a)) draft_path = argv[++i]; }
        else if (!strcmp(a, "--draft-max")) { if (want_val(i, a)) draft_max = atoi(argv[++i]); }
        else if (!strcmp(a, "--draft-check")) draft_check = true;
        else if (!strcmp(a, "--bandwidth")) { if (want_val(i, a)) bandwidth_gbs = atof(argv[++i]); }
        else if (!strcmp(a, "--zoned")) zopt.on = true;
        else if (!strcmp(a, "--zoned-check")) { zopt.on = true; zopt.check = true; }
        else if (!strcmp(a, "--sinks")) { if (want_val(i, a)) zopt.sinks = atoi(argv[++i]); }
        else if (!strcmp(a, "--window")) { if (want_val(i, a)) zopt.window = atoi(argv[++i]); }
        else if (!strcmp(a, "--notebook")) { if (want_val(i, a)) zopt.notebook = atoi(argv[++i]); }
        else if (!strcmp(a, "--tail-form")) {
            if (want_val(i, a)) {
                const char* f = argv[++i];
                if (!strcmp(f, "q8_0")) zopt.tail_q8 = true;
                else if (!strcmp(f, "f16")) zopt.tail_q8 = false;
                else {
                    printf("--tail-form принимает q8_0 или f16, получено \"%s\"\n", f);
                    bad_arg = true;
                }
            }
        }
        else if (!strcmp(a, "--resident")) { if (want_val(i, a)) ropt.capacity = atoi(argv[++i]); }
        else if (!strcmp(a, "--resident-window")) { if (want_val(i, a)) ropt.window = atoi(argv[++i]); }
        else if (!strcmp(a, "--resident-period")) { if (want_val(i, a)) ropt.period = atoi(argv[++i]); }
        else if (!strcmp(a, "--resident-budget")) { if (want_val(i, a)) ropt.budget = atoi(argv[++i]); }
        else if (!strcmp(a, "--gpu-experts")) { gopt.on = true; }
        else if (!strcmp(a, "--gpu-experts-check")) { gopt.on = true; gopt.check = true; }
        else if (!strcmp(a, "--gpu-experts-selftest")) { gopt.selftest = true; }
        else if (!strcmp(a, "--gpu-experts-reserve")) {
            if (want_val(i, a)) gopt.reserve_mib = atoi(argv[++i]);
        }
        else if (!strcmp(a, "--resident-policy")) {
            if (want_val(i, a)) {
                const char* f = argv[++i];
                if (!strcmp(f, "lfu")) ropt.lfu = true;
                else if (!strcmp(f, "lru")) ropt.lfu = false;
                else {
                    printf("--resident-policy принимает lfu или lru, получено \"%s\"\n", f);
                    bad_arg = true;
                }
            }
        }
        else if (!strcmp(a, "--rotate-keys")) {
            if (want_val(i, a)) {
                const char* f = argv[++i];
                if (!strcmp(f, "on")) zopt.rotate_keys = true;
                else if (!strcmp(f, "off")) zopt.rotate_keys = false;
                else {
                    printf("--rotate-keys принимает on или off, получено \"%s\"\n", f);
                    bad_arg = true;
                }
            }
        }
        else if (!strcmp(a, "--gpu-static")) { sopt.on = true; }
        else if (!strcmp(a, "--gpu-static-verify")) { sopt.on = true; sopt.verify = true; }
        else if (!strcmp(a, "--gpu-static-layers")) { sopt.on = true; sopt.layers = true; }
        // Answered here, and it does nothing but exit zero. It is the queue's startability
        // probe (bench/build_safe.ps1, Test-Startable): a binary that fails to load its DLLs
        // dies before printing anything, with -1073741511 or -1073741515, and every log then
        // reads like "the run produced no data" rather than "the run never happened". This
        // engine used to answer it with "unknown flag" and exit 1, so the probe called every
        // build broken and paid for a ten-minute clean rebuild to fix nothing.
        else if (!strcmp(a, "--version")) {
            printf("llama-memex-fwd (MemeX)\n");
            return 0;
        }
        else if (!strcmp(a, "--gpu-static-selftest")) { sopt.selftest = true; }
        else if (!strcmp(a, "--gpu-static-rows")) {
            if (want_val(i, a)) sopt.rows = atoi(argv[++i]);
        }
        else if (!strcmp(a, "--gpu-static-reserve")) {
            if (want_val(i, a)) sopt.reserve_mib = atoi(argv[++i]);
        }
        else {
            printf("неизвестный флаг: %s (--help перечислит все)\n", a);
            bad_arg = true;
        }
    }
    if (bad_arg) return 1;
    // Before anything is loaded, because it loads nothing: the self-test builds its own
    // synthetic experts and runs in half a gigabyte, which is what makes it usable on a
    // machine that has a 30B model resident in the next process.
    if (sopt.selftest) {
#ifdef MEMEX_FWD_GPU_EXPERTS
        return memex::gpu_static_selftest(threads);
#else
        printf("--gpu-static-selftest: eta sborka sobrana bez Vulkan\n");
        return 1;
#endif
    }
    if (gopt.selftest) {
#ifdef MEMEX_FWD_GPU_EXPERTS
        return memex::gpu_experts_selftest(threads);
#else
        printf("--gpu-experts-selftest: эта сборка собрана без Vulkan\n");
        return 1;
#endif
    }
#ifndef MEMEX_FWD_ZONED
    if (zopt.on) {
        printf("--zoned запрошен, но эта сборка собрана без зонного кэша: не нашлось "
               "MemeX/cpp/src/kv_zones.cpp или examples/memex-kv/zoned_cache.cpp\n");
        return 1;
    }
#endif
    if (zopt.on) {
        if (zopt.sinks < 0 || zopt.notebook < 0 || zopt.window <= 0) {
            printf("--sinks и --notebook неотрицательны, --window положителен; "
                   "получено %d, %d, %d\n", zopt.sinks, zopt.notebook, zopt.window);
            return 1;
        }
        // Speculation verifies a batch of drafted positions in one graph, and build_attn
        // scores one query. Refused rather than silently falling back to the exact path,
        // which would look like the zoned cache running and be the opposite.
        if (draft_max > 0) {
            printf("--zoned вместе со спекуляцией пока не поддержан: проверка черновика "
                   "считает %d позиций одним графом, а build_attn считает одну\n",
                   draft_max + 1);
            return 1;
        }
        if (chat) {
            printf("--zoned вместе с --chat пока не поддержан: зонный путь живёт в ветке "
                   "--gen, где рядом можно поставить точный прогон и сравнить\n");
            return 1;
        }
        if (n_gen <= 0) {
            printf("--zoned без --gen N: зонный кэш работает на декоде, а декодировать "
                   "нечего\n");
            return 1;
        }
        if (zopt.notebook > 0) {
            // Said out loud rather than left to be discovered. The notebook is positions the
            // engine asked to keep exactly, and nothing in this engine asks yet - so the
            // slots are reserved, stay empty, and the code that pushes a promoted position to
            // the backend (upload_slot) never runs. Reserved-and-empty is not wrong, but it
            // is exact slots spent on nothing, and the path is therefore unexercised.
            printf("ВНИМАНИЕ: --notebook %d резервирует точные слоты, но движку пока нечем "
                   "помечать позиции (keep_exact всегда ложь), так что блокнот останется "
                   "пустым, а путь его загрузки — непройденным\n", zopt.notebook);
        }
    }
#ifndef MEMEX_FWD_GPU_EXPERTS
    if (sopt.on) {
        printf("--gpu-static: eta sborka sobrana bez Vulkan (nuzhno derevo build-vk, "
               "GGML_VULKAN=ON)\n");
        return 1;
    }
#endif
    // --gpu-static and --gpu-experts together: this is the whole design, not an exotic
    // combination. The card holds the static half (802 MB, every resident byte read on every
    // token) and the popular experts in whatever video memory is left, and the crossing the
    // static half already pays per layer is the crossing the expert split rides on.
    //
    // It was refused until now on a reading of the source that turns out to be wrong in the
    // part that mattered. Both modules do call ggml_backend_vk_init(0) and ggml does cache its
    // devices (ggml-vulkan.cpp:3067-3073), so they share one vk_device - but they get SEPARATE
    // ggml_backend_vk_contexts, and a command pool is per (context, queue) rather than per
    // device ("There's an instance of this for each (context,queue) pair", ggml-vulkan.cpp at
    // struct vk_command_pool). What is genuinely shared is protected where it is touched:
    // queue submits by the file-scope queue_mutex (ggml_vk_submit, :1432 and :1502), buffer
    // reads and writes and the sync_staging buffer behind them by device->mutex (:4725, :4816,
    // :4846, :4881), and pipeline compilation by the same (:1342, :1556).
    //
    // What is NOT protected is two contexts' prealloc buffers, and those are per context. So
    // the remaining exposure is the one thing measurement can see: a wrong answer under
    // --gpu-experts-check, or a device-lost. Both are loud. Refusing the combination outright
    // meant the design could never be measured at all, which is a worse failure than either.
    if (sopt.on && gopt.on && !gopt.check) {
        printf("--gpu-static i --gpu-experts vmeste: dva konteksta Vulkan na odnom "
               "ustrojstve. Ochered, bufery i kompiljacija konvejerov zashchishcheny "
               "sobstvennymi mutexami ggml; pervyj progon stoit gonjat s "
               "--gpu-experts-check\n");
    }
    if (sopt.on && (sopt.rows < 1 || sopt.rows > 64)) {
        printf("--gpu-static-rows %d vne diapazona 1..64\n", sopt.rows);
        return 1;
    }
#ifndef MEMEX_FWD_GPU_EXPERTS
    if (gopt.on) {
        printf("--gpu-experts: эта сборка собрана без Vulkan (нужно дерево build-vk, "
               "GGML_VULKAN=ON)\n");
        return 1;
    }
#endif
    if (gopt.on && ropt.capacity < 0) {
        printf("--resident %d бессмысленно\n", ropt.capacity);
        return 1;
    }
    if (ropt.on() || gopt.on) {
        // Refused rather than quietly ignored in each case, because every one of them is a
        // way of running with the flag on and the mechanism off - which reads as "it changed
        // nothing" and is worth days.
        if (n_gen <= 0) {
            printf("--resident %d без --gen N: расщепление живёт на декоде, а декодировать "
                   "нечего\n", ropt.capacity);
            return 1;
        }
        if (chat) {
            printf("--resident вместе с --chat пока не поддержан: расщеплённый путь живёт в "
                   "ветке --gen, где рядом стоит нерасщеплённый прогон и его можно "
                   "сравнить\n");
            return 1;
        }
        if (draft_max > 0) {
            printf("--resident вместе со спекуляцией пока не поддержан: проверка черновика "
                   "считает %d позиций одним графом, а маска резидентности хранит по строке "
                   "на слой, не на токен\n", draft_max + 1);
            return 1;
        }
        if (zopt.on) {
            printf("--resident вместе с --zoned пока не поддержан: обе опции меняют граф "
                   "декода, и совмещённый прогон не сказал бы, чья это разница\n");
            return 1;
        }
    }
    if (!draft_path.empty() && draft_max <= 0) draft_max = 3;
    if (draft_max > 0 && draft_path.empty()) {
        printf("--draft-max %d задан без -md: нечем набрасывать\n", draft_max);
        return 1;
    }
    if (bandwidth_gbs <= 0.0) {
        printf("--bandwidth должен быть положительным, получено %g\n", bandwidth_gbs);
        return 1;
    }
    // Speculation verifies several drafted positions in one graph, which is a MULTI-token
    // graph and therefore the host attention block - so it writes the host cache while the
    // card's stays where the last single-token step left it. Copying the cache across per
    // draft round is 200 MB a round and would cost more than the speculation is worth, and
    // speculation is closed by measurement anyway (best arm 13.78 against a 14.04 baseline).
    // Refused rather than left to produce a plausible wrong answer.
    if (sopt.layers && (!draft_path.empty() || draft_max > 0)) {
        printf("--gpu-static-layers vmeste so spekuljaciej nelzja: graf proverki chernovika "
               "mnogotokennyj, on pishet kesh hosta, a shag chitaet kesh karty\n");
        return 1;
    }
    // ggml reads GGML_VK_SUBMIT_* ONCE, when the device is created (ggml-vulkan.cpp:3905),
    // and something before us already creates it: stderr carries "Vulkan0: using device
    // Vulkan0 - 3824 MiB free" from the model loader. So setting these inside GpuStatic::init
    // is too late - it runs after the load - and the default divisor of 40 would stand.
    //
    // Why 1. The backend picks submit points by mul_mat_bytes >= total_mat_mul_bytes/divisor,
    // doubling the threshold for each of the first three submits, so a layer graph with seven
    // matmuls submits five or six times at a measured 24.1 us each: 120-145 us a layer, 6-7 ms
    // a token, against a stage whose whole budget is about 20. A divisor of 1 makes it one.
    //
    // Safe only because these graphs are small. Upstream submits early to bound how long one
    // submission runs against the two-second kernel timeout on Windows - the failure mode there
    // is a device-lost, not a slow run - and a 10 MB layer cannot approach it. An environment
    // that already carries a value wins, so a sweep over the divisor can still say so.
    if (sopt.layers) {
#ifdef _WIN32
        if (!getenv("GGML_VK_SUBMIT_DIVISOR")) _putenv_s("GGML_VK_SUBMIT_DIVISOR", "1");
        if (!getenv("GGML_VK_SUBMIT_TAIL"))    _putenv_s("GGML_VK_SUBMIT_TAIL", "0");
#else
        if (!getenv("GGML_VK_SUBMIT_DIVISOR")) setenv("GGML_VK_SUBMIT_DIVISOR", "1", 0);
        if (!getenv("GGML_VK_SUBMIT_TAIL"))    setenv("GGML_VK_SUBMIT_TAIL", "0", 0);
#endif
    }
    if (sopt.layers && zopt.on) {
        printf("--gpu-static-layers vmeste s --zoned nelzja: zonnyj kesh stroit svoj "
               "podgraf vnimanija na hoste\n");
        return 1;
    }

    // The draft graph on its own, checked against the fork's own decode of the same file.
    // This exists because the target is 24.5 GB and the draft is 300: the dense arm can be
    // verified on a machine that has no room to load the model it will be drafting for, and
    // an unverified second graph is exactly the thing this project has been burned by.
    if (draft_check) {
        if (draft_path.empty()) {
            printf("--draft-check без -md: нечего проверять\n");
            return 1;
        }
        return draft_selftest(draft_path, prompt, threads);
    }

    HParams h;
    if (!read_hparams(model_path.c_str(), &h)) {
        printf("не читаются гиперпараметры из %s\n", model_path.c_str());
        if (file_size_bytes(model_path.c_str()) == 0) {
            printf("файл не открывается или пуст — проверьте путь\n");
        } else {
            printf("файл открывается, но GGUF не разбирается — не тот формат?\n");
        }
        return 1;
    }
    printf("архитектура %s: слоёв %d, n_embd %d, голов %d/%d, head_dim %d,\n"
           "  экспертов %d из них %d, ширина эксперта %d, rms_eps %g, rope_base %g\n",
           h.arch.c_str(), h.n_layer, h.n_embd, h.n_head, h.n_head_kv, h.head_dim,
           h.n_expert, h.n_expert_used, h.n_ff_exp, h.rms_eps, h.rope_base);

    // Which architecture, decided once. Everything downstream branches on these three rather
    // than on the string, so a typo is a compile error rather than a silently skipped path.
    const bool arch_q3  = h.arch == "qwen3moe";
    const bool arch_g4  = h.arch == "gemma4";
    const bool arch_q35 = h.arch == "qwen35moe";

    // The graphs below are written for named architectures. Running one against another does
    // not fail - it produces confident nonsense, because the tensor names happen to overlap
    // and the shapes happen to be plausible. So anything else is refused by name.
    if (!arch_q3 && !arch_g4 && !arch_q35) {
        printf("архитектура \"%s\" не поддерживается этим движком.\n", h.arch.c_str());
        printf("  поддерживаются как целевые: qwen3moe, gemma4, qwen35moe\n");
        printf("  и qwen3 — но только как модель-черновик через -md\n");
        printf("  граф написан под конкретные архитектуры; на другой он выдаст не ошибку, "
               "а уверенную чепуху\n");
        return 1;
    }
    if (h.n_expert <= 0 || h.n_expert_used <= 0 || h.n_expert_used > h.n_expert) {
        printf("%s с %d экспертами и %d используемыми — так не бывает, "
               "метаданные GGUF не читаются\n", h.arch.c_str(), h.n_expert,
               h.n_expert_used);
        return 1;
    }
    // The per-layer table, printed as runs rather than as thirty or forty lines. It is the
    // one place the geometry this engine believes in becomes visible, and every one of the
    // three architectures has at least one thing here that a scalar header would have hidden.
    {
        printf("геометрия по слоям:\n");
        auto kind_name = [](LayerKind k) {
            return k == LayerKind::DELTA_NET ? "дельта-сеть"
                 : k == LayerKind::ATTN_SWA  ? "внимание+окно" : "внимание";
        };
        int run_start = 0;
        for (int il = 1; il <= h.n_layer; ++il) {
            const bool same = il < h.n_layer &&
                h.L(il).kind == h.L(run_start).kind &&
                h.L(il).n_head_kv == h.L(run_start).n_head_kv &&
                h.L(il).head_dim == h.L(run_start).head_dim &&
                h.L(il).n_rot == h.L(run_start).n_rot &&
                h.L(il).rope_base == h.L(run_start).rope_base;
            if (same) continue;
            const LayerGeom& G = h.L(run_start);
            if (G.kind == LayerKind::DELTA_NET) {
                printf("  слои %2d..%-2d  %-14s состояние %zu f32 (%.2f МБ)\n", run_start,
                       il - 1, kind_name(G.kind), h.delta_state_elems(),
                       double(h.delta_state_elems()) * 4.0 / 1e6);
            } else {
                printf("  слои %2d..%-2d  %-14s головы %d/%d по %d, rope %d из %d @ %g%s\n",
                       run_start, il - 1, kind_name(G.kind), G.n_head, G.n_head_kv,
                       G.head_dim, G.n_rot, G.head_dim, G.rope_base,
                       G.rope_freqs ? ", freq_factors" : "");
            }
            run_start = il;
        }
        if (h.n_swa > 0) printf("  окно %d позиций\n", h.n_swa);
        if (h.f_attn_scale != 0.0f) {
            printf("  масштаб внимания %g (а НЕ 1/sqrt(head_dim)=%g)\n", h.f_attn_scale,
                   1.0f / std::sqrt(float(h.head_dim)));
        }
        if (h.f_logit_softcap > 0.0f) {
            printf("  логиты ограничены: %g*tanh(x/%g)\n", h.f_logit_softcap,
                   h.f_logit_softcap);
        }
        if (h.f_embd_scale > 0.0f) printf("  вложения масштабируются на %g\n", h.f_embd_scale);
        if (h.f_router_scale != 1.0f) {
            printf("  vhod marshrutizatora umnozhaetsja na %g -- etalon delaet to zhe samoe\n", h.f_router_scale);
            printf("  s samim ffn_gate_inp.scale pri zagruzke (llm_scale_gate_inp_s)\n");
        }
    }

    // The optional modules were all written against one geometry, and every one of them
    // indexes it as a single shape. Refused by name here rather than left to produce a
    // plausible wrong answer - which is what they would do, because a head count that is
    // wrong by a factor of four still slices a buffer successfully.
    if (!arch_q3) {
        if (zopt.on) {
            printf("--zoned пока только для qwen3moe: модуль зон держит одну геометрию "
                   "головы на модель, а у \"%s\" она своя на каждом слое\n", h.arch.c_str());
            return 1;
        }
        if (ropt.on() || gopt.on) {
            // Gemma's expert pair is one fused tensor and the split needs two dispatches of
            // it; qwen35moe's shared experts are static traffic that the resident set does
            // not model. Both are tractable, neither is verified, and an unverified split is
            // exactly what this project has been burned by.
            printf("--resident/--gpu-experts пока только для qwen3moe.\n");
            if (arch_g4) {
                printf("  у gemma4 gate и up лежат в одном тензоре ffn_gate_up_exps, и "
                       "расщепление требует двух прогонов одного тензора с двумя списками "
                       "id — как это сделать, написано в build_gemma4_step\n");
            }
            return 1;
        }
        if (!draft_path.empty()) {
            printf("спекуляция пока только для qwen3moe: черновик должен делить словарь и "
                   "геометрию с целевой моделью\n");
            return 1;
        }
        if (n_gen > 0) {
            printf("--gen пока только для qwen3moe: эта ветка несёт зонный кэш, резидентный "
                   "набор и половину на карте, и все три написаны под одну геометрию.\n");
            printf("  для \"%s\" есть --decode-check N: он и генерирует, и сверяет каждый "
                   "шаг с llama_decode — что для дельта-сети и есть настоящая проверка\n",
                   h.arch.c_str());
            return 1;
        }
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    // Repacking is a model-load option, not a graph one: the loader rewrites each host
    // tensor into its interleaved _R4/_R8 sibling in place and changes nothing but the
    // ggml_type, so the tensors llama_get_model_tensor hands us are already repacked and
    // every ne/nb we build views from still holds. mul_mat and mul_mat_id both reach the
    // interleaved types through iqk_mul_mat, which is the whole point of them.
    //
    // ggml_get_rows does not, and that is the one thing that could have made this unsafe.
    // It takes the generic quantised path, one row at a time through to_float - but a
    // to_float for an _R4 type dequantises the four interleaved rows of a group at once, so
    // it would neither address nor fill a single row correctly, silently. Our only get_rows
    // over a weight is the embedding lookup, and iqk_repack_tensor's own forbidden list
    // excludes token_embd, so it stays un-repacked. That exclusion is load-bearing here.
    //
    // On by default, because it is worth 38% and cannot change an answer. The catch is that
    // the loader forces use_mmap off whenever this is set (src/llama-model-loader.cpp:588 -
    // the repack writes into the weights, so they cannot be a read-only mapping), which turns
    // a file-backed mapping into the whole model as resident private memory. A 24.5 GB model
    // does not fit that way on a 32 GB machine, so the decision is made against the actual
    // figures rather than hoped for.
    const uint64_t file_bytes = file_size_bytes(model_path.c_str());
    const PhysMem mem = phys_mem();
    // The overhead the resident model has to share the machine with, counted rather than
    // guessed: our own F16 cache at the context we will ask for, the reference context's KV
    // if we are going to create one, and half a gigabyte for graph compute buffers, the
    // 152k-float logit vectors and the process itself. The 3% on the file covers the
    // repacked layout, which is the same size to within block padding.
    const int ctx_guess = n_ctx_req > 0 ? n_ctx_req : 2048;
    const double our_kv = double(ctx_guess) * double(h.kv_elems_per_pos()) * 2.0 +
                          double(h.n_delta_layers()) * double(h.delta_state_elems()) * 4.0;
    const double ref_kv = want_ref ? 2048.0 * double(h.kv_elems_per_pos()) * 2.0 : 0.0;
    const double overhead = our_kv + ref_kv + 0.5e9;

    // ------------------------------------------------------------------------------------
    // WHICH tensors get repacked, and what it costs in resident memory.
    //
    // The invariant that decides everything here: a repacked tensor is private, writable memory
    // by necessity - the mapping is read-only (CreateFileMappingA(PAGE_READONLY)), so a rewritten
    // tensor has to leave it. mmap therefore saves exactly the bytes that KEEP their stored
    // layout, no more. Which is why the direction matters so much:
    //
    //   --repack-only=static  : experts (14.6 GB of 15.3) stay mapped   -> ~0.8 GB resident
    //   --repack-only=experts : experts are rewritten                   -> ~14.6 GB resident
    //   --repack-only=all     : everything is rewritten, mmap off       -> ~15.3 GB resident
    //
    // and equally which half can be uploaded to the card: whatever is NOT repacked is still the
    // stored type, which is the only thing the Vulkan backend implements (it has no _R* kernels
    // for any op).
    //
    // Element share rather than a guess: this guard exists because guesses were wrong before.
    const double exp_elems  = double(h.n_layer) * double(h.n_expert) * 3.0 *
                              double(h.n_embd)  * double(h.n_ff_exp);
    // Summed per layer: gemma4's two attention geometries differ by a factor of two in each
    // direction, and qwen35moe's delta-net layers have a projection stack of an entirely
    // different shape. A model-wide d_q() here used to be right by accident and is now the
    // wrong number for two architectures out of three.
    double attn_elems = 0.0;
    for (int il = 0; il < h.n_layer; ++il) {
        const LayerGeom& G = h.L(il);
        if (G.kind == LayerKind::DELTA_NET) {
            // qkv, its gate, and the output projection. The small ssm_* tensors are noise
            // against these and are left out rather than guessed at.
            const double key_dim = double(h.ssm_d_state) * double(h.ssm_n_group);
            const double val_dim = double(h.ssm_d_inner);
            attn_elems += double(h.n_embd) * (2.0 * key_dim + 2.0 * val_dim) +
                          val_dim * double(h.n_embd);
        } else {
            attn_elems += 2.0 * double(h.n_embd) * double(G.d_q()) +
                          2.0 * double(h.n_embd) * double(G.d_kv());
        }
    }
    const double emb_elems  = 2.0 * double(h.n_vocab) * double(h.n_embd);
    const double all_elems  = exp_elems + attn_elems + emb_elems;
    // token_embd is on iqk's own forbidden list (get_rows cannot read an interleaved type), so
    // it is never rewritten and never leaves the mapping; only the output head of the pair is.
    const double static_frac =
        all_elems > 0.0 ? (attn_elems + 0.5 * emb_elems) / all_elems : 1.0;
    const double expert_frac = all_elems > 0.0 ? exp_elems / all_elems : 0.0;

    // Resolve the selection into the two substring lists the loader takes.
    std::string sel = repack_sel;
    if (sel == "auto") sel = "experts";
    std::string rp_exclude, rp_only;
    double      rewritten_frac = 1.0;   // share of the file that leaves the mapping
    bool        sel_none = false;
    if (sel == "all") {
        rewritten_frac = 1.0;
    } else if (sel == "none") {
        sel_none = true;
        rewritten_frac = 0.0;
    } else if (sel == "experts") {
        rp_only = "_exps.";
        rewritten_frac = expert_frac;
    } else if (sel == "static") {
        rp_exclude = "_exps.";
        rewritten_frac = static_frac;
    } else if (sel.compare(0, 5, "skip:") == 0) {
        rp_exclude = sel.substr(5);
        rewritten_frac = 1.0;           // unknown; assume the worst
    } else {
        rp_only = sel;
        rewritten_frac = 1.0;           // unknown; assume the worst
    }
    if (sel_none) repack_wish = REPACK_FORCE_OFF;

    const double need_all = double(file_bytes) * 1.03 + overhead;
    // Only the rewritten part is resident; the rest stays file-backed and is reclaimable.
    const double need     = double(file_bytes) * 1.03 * rewritten_frac + overhead;
    bool repack = false;
    std::string why;
    if (repack_wish == REPACK_FORCE_OFF) {
        why = "выключена флагом --no-repack; модель останется отображённой через mmap";
    } else if (repack_wish == REPACK_FORCE_ON) {
        repack = true;
        why = "включена флагом --repack безусловно; mmap отключается";
        if (mem.ok && double(mem.avail) < need) {
            printf("ВНИМАНИЕ: доступно %.1f ГБ, а перепакованной модели нужно ~%.1f ГБ — "
                   "запрошено флагом, но машина, вероятно, начнёт свопить\n",
                   gb(mem.avail), need / 1e9);
        }
    } else if (file_bytes == 0) {
        why = "выключена: размер файла модели узнать не удалось, "
              "а без него нельзя обещать, что он поместится";
    } else if (!mem.ok) {
        why = "выключена: объём физической памяти узнать не удалось "
              "(GlobalMemoryStatusEx), поэтому берём безопасный вариант";
    } else if (double(mem.avail) < need && sel == "all" &&
               double(mem.avail) >= double(file_bytes) * 1.03 * static_frac + overhead) {
        // Всё сразу не влезает, а перепаковка без экспертов влезает. Это строго лучше, чем не
        // перепаковывать ничего: mmap остаётся, а внимание и голова всё-таки перепакованы.
        sel = "static";
        rp_exclude = "_exps.";
        rp_only.clear();
        rewritten_frac = static_frac;
        repack = true;
        char buf[320];
        snprintf(buf, sizeof(buf),
                 "включена ЧАСТИЧНО (--repack-only=static): целиком нужно ~%.1f ГБ, а доступно "
                 "%.1f ГБ из %.1f; эксперты остаются как есть (нужно ~%.1f ГБ), mmap сохраняется",
                 need_all / 1e9, gb(mem.avail), gb(mem.total),
                 (double(file_bytes) * 1.03 * static_frac + overhead) / 1e9);
        why = buf;
    } else if (double(mem.avail) < need) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "выключена: файл %.1f ГБ, с запасом нужно ~%.1f ГБ, "
                 "а доступно физической памяти %.1f ГБ из %.1f — остаёмся на mmap",
                 gb(file_bytes), need / 1e9, gb(mem.avail), gb(mem.total));
        why = buf;
    } else {
        repack = true;
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "включена: нужно ~%.1f ГБ, доступно %.1f ГБ из %.1f",
                 need / 1e9, gb(mem.avail), gb(mem.total));
        why = buf;
    }
    if (!repack) { rp_exclude.clear(); rp_only.clear(); }
    mp.repack_tensors = repack;
    mp.repack_exclude = rp_exclude.empty() ? nullptr : rp_exclude.c_str();
    mp.repack_only    = rp_only.empty()    ? nullptr : rp_only.c_str();

    // mmap. The loader keeps the mapping whenever a filter is set, but keeping it is only
    // worth something when there is a lot left in it: a repacked tensor has to be rewritten,
    // and the mapping is read-only, so those bytes become private memory whatever we do. When
    // the repacked share is most of the file, keeping the mapping only adds a second full read
    // of it on top of the same private total, so auto drops it.
    const bool filtered = !rp_exclude.empty() || !rp_only.empty();
    bool want_mmap = true;
    std::string mmap_why;
    if (mmap_wish == MMAP_ON) {
        mmap_why = "оставлено флагом --mmap";
    } else if (mmap_wish == MMAP_OFF) {
        want_mmap = false;
        mmap_why = "убрано флагом --no-mmap";
    } else if (!repack) {
        mmap_why = "перепаковки нет — отображение бесплатно";
    } else if (!filtered) {
        want_mmap = false;
        mmap_why = "перепаковывается всё — загрузчик всё равно снимет отображение";
    } else if (rewritten_frac > 0.5) {
        want_mmap = false;
        mmap_why = "перепаковывается большая часть файла: под отображением почти ничего "
                   "не останется, а копирование из него было бы лишним полным чтением";
    } else {
        mmap_why = "перепаковывается меньшая часть файла — остальное остаётся под "
                   "отображением и не занимает приватную память";
    }
    mp.use_mmap = want_mmap;
    // One line, always, naming the mode that actually ran. It is a 34% difference; a silent
    // fallback would show up as a mysteriously slow run weeks later.
    printf("перепаковка тензоров: %s (--repack-only=%s)\n",
           repack ? "ВКЛЮЧЕНА" : "выключена", repack ? sel.c_str() : "none");
    printf("  %s\n", why.c_str());
    if (repack) {
        if (!rp_only.empty())    printf("  перепаковываются ТОЛЬКО имена, содержащие [%s]\n", rp_only.c_str());
        if (!rp_exclude.empty()) printf("  НЕ перепаковываются имена, содержащие [%s]\n", rp_exclude.c_str());
        printf("  приватной памяти под перепакованное: ~%.0f%% файла (~%.1f ГБ) — "
               "перепакованный тензор обязан быть приватным, "
               "отображение экономит ровно то, что осталось в исходном виде\n",
               100.0 * rewritten_frac, double(file_bytes) * rewritten_frac / 1e9);
        if (!rp_only.empty()) {
            printf("  для карты остаются исходными: внимание, голова, роутер\n");
        } else if (!rp_exclude.empty()) {
            printf("  для карты остаются исходными: эксперты\n");
        }
    }
    printf("  mmap: %s — %s\n", want_mmap ? "ВКЛЮЧЁН" : "выключен", mmap_why.c_str());
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        printf("модель не загрузилась: %s\n", model_path.c_str());
        printf("  файл %.2f ГБ, доступно физической памяти %.1f ГБ; "
               "перепаковка была %s\n", gb(file_bytes),
               mem.ok ? gb(mem.avail) : -1.0, repack ? "включена" : "выключена");
        if (repack) {
            printf("  если это нехватка памяти — попробуйте --no-repack\n");
        }
        return 1;
    }
    h.n_vocab = llama_n_vocab(model);
    h.rope_type = int(llama_rope_type(model));
    printf("словарь %d, тип rope %d\n", h.n_vocab, h.rope_type);

    // Tokenise through the fork's tokeniser: it is correct and we have no reason to own it.
    std::vector<llama_token> toks;
    toks.resize(prompt.size() + 8);
    int n = llama_tokenize(model, prompt.c_str(), int(prompt.size()), toks.data(),
                           int(toks.size()), true, false);
    if (n < 0) {
        toks.resize(size_t(-n));
        n = llama_tokenize(model, prompt.c_str(), int(prompt.size()), toks.data(),
                           int(toks.size()), true, false);
    }
    if (n <= 0) {
        printf("промпт не токенизировался\n");
        return 1;
    }
    // Trimmed to a multiple of four. The fork's F16 matmul returns wrong results when the
    // reduction length is not, and the value product reduces over positions - measured, and
    // it fails silently. A real decode path pads and masks instead; here truncating keeps
    // the comparison honest because the reference sees the same tokens.
    n = std::min(n, max_tokens);
    n = n / 4 * 4;
    toks.resize(size_t(n));
    printf("токенов в промпте: %d\n", n);

    // Reference logits, from the fork's own decode.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_ubatch = ref_ubatch > 0 ? ref_ubatch : 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    // The plain path is what our graph mirrors node for node, so false is the default and it
    // is what every comparison in this file has used. For gemma4 it is ALSO the path in which
    // the reference is itself wrong, and that took a whole bisection to see:
    //
    //   flash_attn off  =>  kv_self.v_trans is TRUE, and llm_build_kv_store takes the
    //   transposed branch: `v_cur = ggml_transpose(ctx, v_cur)` followed by a ggml_cpy into a
    //   [n_tokens, n_embd_v_gqa] strided view. That is correct for every other architecture,
    //   because every other architecture hands it a 2-D V of [n_embd_v_gqa, n_tokens].
    //   gemma4 does not: its unweighted V rms_norm needs the per-head shape, so V arrives
    //   3-D as [n_embd_head_v, n_head_kv, n_tokens]. ggml_transpose swaps ne0 and ne1 ONLY,
    //   giving [n_head_kv, n_embd_head_v, n_tokens] - and ggml_cpy between mismatched shapes
    //   is a FLAT copy (ggml_compute_forward_dup's dst-counter loop), so the element order
    //   the store writes is not the order the read view expects. The cache ends up permuted.
    //
    //   flash_attn on   =>  v_trans is FALSE, the store is a flat view_1d, and the FA read is
    //   [n_embd_head_v, n_kv, n_head_kv] with matching strides. Store and read agree.
    //
    // Which is why our own numbers looked like a bug in our attention: attn_norm, Qcur, Kcur,
    // Vcur, both norms, both ropes, kq and even kq_soft_max_ext all came out BIT-identical at
    // layer 0, and the first divergence was kqv - the one node that reads the V cache.
    // METHODS 16 in one line: before hunting a fault in your own code, check the reference is
    // right in the configuration you are comparing against.
    cp.flash_attn = ref_fa;
    if (ref_fa) {
        printf("etalon schitaet vnimanie slitoj operaciej (--ref-fa): ego V-kesh ne transponirovan\n");
    }
    if (n > int(cp.n_ubatch)) {
        // Said out loud, because it is invisible otherwise and it changes what the
        // reference computes over: several micro-batches instead of one.
        printf("эталон разобьёт %d токенов на микропачки по %d\n", n, int(cp.n_ubatch));
    }
    Probe probe;
    probe.want = probe_name;
    probe.catch_all = probe_name == "all";
    if (!probe_name.empty()) {
        cp.cb_eval = probe_cb;
        cp.cb_eval_user_data = &probe;
    }
    llama_context* lctx = llama_init_from_model(model, cp);
    if (!lctx) {
        printf("контекст не создался\n");
        return 1;
    }
    const auto t_ref = Clock::now();
    if (llama_decode(lctx, llama_batch_get_one(toks.data(), n, 0, 0))) {
        printf("llama_decode не прошёл\n");
        return 1;
    }
    const double ref_ms = ms_since(t_ref);
    std::vector<float> ref;
    ref.assign(size_t(h.n_vocab), 0.0f);
    const float* rl = llama_get_logits(lctx);
    if (!rl) {
        printf("эталонные логиты недоступны\n");
        return 1;
    }
    std::memcpy(ref.data(), rl, sizeof(float) * size_t(h.n_vocab));

    // One of these three is filled; which one is the architecture. Declared together so the
    // graph dispatch below can be a straight three-way branch rather than a cast.
    Weights w;
    Gemma4Weights w4;
    Qwen35Weights w35;
    {
        bool got = false;
        if (arch_q3)  got = collect(model, h, &w);
        if (arch_g4)  got = collect_gemma4(model, &h, &w4);
        if (arch_q35) got = collect_qwen35(model, h, &w35);
        if (!got) {
            printf("не все тензоры на месте — архитектура не та, что ожидалась\n");
            // The names that ARE there, because "tensor not found" on its own has cost this
            // project days: the name present is usually one character from the name asked
            // for.
            print_present_tensors(model_path.c_str(), "blk.0.");
            return 1;
        }
    }

    // What the CPU half will actually run on. Printed from the loaded tensors, group by group,
    // because "repacking is on" is a request and this is the outcome: with the experts excluded
    // the win is partial by construction, so the partial has to be visible rather than assumed.
    //
    // Only for qwen3moe. The group list names that architecture's tensors, and printing an
    // empty table for the others would be worse than printing nothing - it would read as
    // "nothing got repacked".
    if (arch_q3) {
        struct Group { const char* label; ggml_tensor* t; };
        const Weights::Layer& L0 = w.layers[0];
        const Group gs[] = {
            {"blk.*.attn_q.weight     ", L0.wq},
            {"blk.*.attn_k.weight     ", L0.wk},
            {"blk.*.attn_v.weight     ", L0.wv},
            {"blk.*.attn_output.weight", L0.wo},
            {"blk.*.ffn_gate_inp.w    ", L0.router},
            {"blk.*.ffn_up_exps.w     ", L0.up},
            {"blk.*.ffn_gate_exps.w   ", L0.gate},
            {"blk.*.ffn_down_exps.w   ", L0.down},
            {"token_embd.weight       ", w.tok_embd},
            {"output.weight           ", w.out},
        };
        printf("\nраскладка весов после загрузки (что реально будет считать CPU):\n");
        for (const Group& gr : gs) {
            if (!gr.t) continue;
            printf("  %s  %-12s (тип %3d)  %s\n", gr.label, ggml_type_name(gr.t->type),
                   int(gr.t->type),
                   type_is_interleaved(gr.t->type)
                       ? "перепакован — быстрые ядра iqk, для Vulkan НЕПРИГОДЕН"
                       : "как в файле — обычные ядра, для Vulkan пригоден");
        }
        // Not every layer is layer 0: check them all, cheaply, and say so if they disagree.
        bool uniform = true;
        for (int il = 1; il < h.n_layer && uniform; ++il) {
            const Weights::Layer& L = w.layers[size_t(il)];
            uniform = L.wq->type == L0.wq->type && L.up->type == L0.up->type &&
                      L.gate->type == L0.gate->type && L.down->type == L0.down->type;
        }
        if (!uniform) printf("  ВНИМАНИЕ: слои различаются по типам, строка выше — только слой 0\n");
    }

    // The static head on the card, before any graph is built: every graph that uses it
    // captures the pointer at build time.
    memex::GpuStatic* gsp = nullptr;
#ifdef MEMEX_FWD_GPU_EXPERTS
    std::unique_ptr<memex::GpuStatic> gstat;
    if (sopt.on) {
        if (!arch_q3) {
            printf("--gpu-static: poka tolko qwen3moe - drugie arhitektury stroit golovu "
                   "svoim putjom (fnorm, softcap), i podmena tam ne proverena\n");
            return 1;
        }
        memex::GpuStaticConfig sc;
        sc.n_embd  = h.n_embd;
        sc.n_vocab = h.n_vocab;
        sc.max_rows = sopt.rows;
        sc.reserve = std::size_t(sopt.reserve_mib) << 20;
        sc.verify  = sopt.verify;
        sc.layers  = sopt.layers;
        sc.n_layer = h.n_layer;
        sc.n_head  = h.n_head;
        sc.n_head_kv = h.n_head_kv;
        sc.head_dim  = h.head_dim;
        sc.n_expert  = h.n_expert;
        sc.n_ctx_train = h.n_ctx_train;
        sc.rope_type = h.rope_type;
        sc.rope_base = h.rope_base;
        sc.rms_eps   = h.rms_eps;
        gstat.reset(new memex::GpuStatic());
        std::string serr;
        if (!gstat->init(sc, w.out, &serr)) {
            printf("--gpu-static: %s\n", serr.c_str());
            return 1;
        }
        gsp = gstat.get();
        printf("\nstaticheskaja golova na karte: %s\n", gsp->device_name().c_str());
        printf("  output.weight %s, %.1f MiB v videopamjati, blok do %d strok\n",
               gsp->head_type_name(), double(gsp->vram_bytes()) / 1048576.0, sopt.rows);
        for (const memex::GpuStaticBuffer& b : gsp->buffers()) {
            printf("  bufer: %-24s %8.2f MiB (nabivki %.2f MiB)  %s\n", b.what.c_str(),
                   double(b.bytes) / 1048576.0, double(b.padding) / 1048576.0,
                   b.over_bar ? "> 256 MiB - ne BAR"
                              : "<= 256 MiB - MOG SEST V BAR");
        }
    }
#else
    (void)sopt;
#endif

#ifdef MEMEX_FWD_GPU_EXPERTS
    // The nine attention tensors per layer, in the module's own struct. Named fields rather
    // than nine parallel arrays because nine parallel arrays is how a wq gets uploaded where a
    // wk belongs, with no shape error anywhere to catch it. Filled here, before any graph, and
    // handed to init_layers once the run's cache length is known.
    std::vector<memex::GpuStaticLayer> slayers;
    if (gsp && gsp->config().layers) {
        slayers.resize(std::size_t(h.n_layer));
        for (int il = 0; il < h.n_layer; ++il) {
            const Weights::Layer& L = w.layers[std::size_t(il)];
            memex::GpuStaticLayer& S = slayers[std::size_t(il)];
            S.attn_norm = L.attn_norm; S.wq = L.wq; S.wk = L.wk; S.wv = L.wv; S.wo = L.wo;
            S.q_norm = L.q_norm; S.k_norm = L.k_norm; S.ffn_norm = L.ffn_norm;
            S.router = L.router;
        }
    }
    // Placing the attention weights needs the cache length, which every mode decides for
    // itself, so this is a lambda called from each of them rather than a line here.
    auto place_layers = [&](int n_kv_max) -> bool {
        if (!gsp || !gsp->config().layers || gsp->layers_on()) return true;
        std::string lerr;
        const auto t_l = Clock::now();
        if (!gsp->init_layers(slayers.data(), n_kv_max, &lerr)) {
            printf("--gpu-static-layers: %s\n", lerr.c_str());
            return false;
        }
        printf("\nvnimanie, marshrutizatory i KV-kesh na karte: %d sloev, kesh %d pozicij, "
               "vsego %.1f MiB v videopamjati, za %.0f ms\n", h.n_layer, n_kv_max,
               double(gsp->vram_bytes()) / 1048576.0, ms_since(t_l));
        for (const memex::GpuStaticBuffer& b : gsp->buffers()) {
            printf("  bufer: %-28s %8.2f MiB (nabivki %.2f MiB)  %s\n", b.what.c_str(),
                   double(b.bytes) / 1048576.0, double(b.padding) / 1048576.0,
                   b.over_bar ? "> 256 MiB - ne BAR" : "<= 256 MiB - MOG SEST V BAR");
        }
        return true;
    };
#else
    auto place_layers = [&](int) { return true; };
#endif

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    Graph g;
    // The cache and the recurrent state the comparison prefill runs against.
    //
    // qwen3moe goes through build(), which attends over the prompt itself and needs no
    // cache at all - it is the original verified path and is left exactly as it was. The
    // other two go through their own step builder with n_past = 0, which is the same
    // computation with the keys parked in a cache on the way past. That is deliberate:
    // it means the graph the comparison validates is the SAME graph the generation uses,
    // rather than a second one written for the comparison and never run again.
    Cache cmp_kv;
    DeltaState cmp_ds;
    // Which row of g.logits holds the last token. build() keeps every row; the step
    // builders prune to the last one, because their output head is 262144 wide and running
    // it over a whole prompt costs more than the prompt.
    int logit_row = n - 1;
    std::vector<int32_t> pos;
    std::vector<float> mask;

    if (arch_q3) {
        if (!build(&g, buft, h, w, n, gsp)) {
            printf("наш граф не собрался\n");
            return 1;
        }
        pos.resize(size_t(n));
        for (int i = 0; i < n; ++i) pos[size_t(i)] = i;
        mask.assign(size_t(n) * size_t(n), 0.0f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j > i) mask[size_t(i) * size_t(n) + size_t(j)] = -INFINITY;
            }
        }
        ggml_backend_tensor_set(g.tokens, toks.data(), 0, ggml_nbytes(g.tokens));
        ggml_backend_tensor_set(g.positions, pos.data(), 0, ggml_nbytes(g.positions));
        ggml_backend_tensor_set(g.mask, mask.data(), 0, ggml_nbytes(g.mask));
    } else {
        const int nkv = pad32(n);
        if (!cmp_kv.init(buft, h, nkv)) {
            printf("кэш для сверки не выделился (%d позиций, %.1f МБ)\n", nkv,
                   double(nkv) * double(h.kv_elems_per_pos()) * 2.0 / 1e6);
            return 1;
        }
        if (!cmp_ds.init(buft, h)) {
            printf("состояние дельта-сети для сверки не выделилось\n");
            return 1;
        }
        // From no history. The reference's own decode starts from an empty state too, and
        // a comparison against a recurrence that began somewhere else is not a comparison.
        cmp_ds.clear();
        const bool built = arch_g4
            ? build_gemma4_step(&g, buft, h, w4, cmp_kv, n, 0, nkv, /*all_logits=*/false,
                                /*keep_probes=*/true)
            : build_qwen35_step(&g, buft, h, w35, cmp_kv, cmp_ds, n, 0, nkv,
                                /*all_logits=*/false, /*keep_probes=*/true);
        if (!built) {
            printf("наш граф не собрался\n");
            return 1;
        }
        set_graph_inputs(g, h, toks.data(), n, 0, &pos, &mask);
        logit_row = 0;
    }

    const auto t_our = Clock::now();
    ggml_backend_graph_compute(be, g.gf);
    ggml_backend_synchronize(be);
    const double our_ms = ms_since(t_our);

    std::vector<float> ours;
    ours.assign(size_t(h.n_vocab), 0.0f);
    // The last token's row: logits come out as [n_vocab, n_rows].
    ggml_backend_tensor_get(g.logits, ours.data(),
                            size_t(logit_row) * size_t(h.n_vocab) * sizeof(float),
                            sizeof(float) * size_t(h.n_vocab));

    double num = 0.0, den = 0.0, worst = 0.0;
    for (int i = 0; i < h.n_vocab; ++i) {
        const double d = double(ours[size_t(i)]) - double(ref[size_t(i)]);
        num += d * d;
        den += double(ref[size_t(i)]) * double(ref[size_t(i)]);
        worst = std::max(worst, std::abs(d));
    }
    auto argmax = [&](const std::vector<float>& v) {
        return int(std::max_element(v.begin(), v.end()) - v.begin());
    };
    const int a_ref = argmax(ref), a_our = argmax(ours);
    char buf_r[64] = {0}, buf_o[64] = {0};
    llama_token_to_piece(model, a_ref, buf_r, sizeof(buf_r) - 1, 0, true);
    llama_token_to_piece(model, a_our, buf_o, sizeof(buf_o) - 1, 0, true);

    if (probe_name == "all") {
        printf("\nсверка по слоям (наш путь против эталонного):\n");
        for (const auto& mine : g.probes) {
            const std::vector<float>* ref_t = probe.get(mine.first);
            if (!ref_t) continue;
            std::vector<float> got;
            got.resize(size_t(ggml_nelements(mine.second)));
            ggml_backend_tensor_get(mine.second, got.data(), 0,
                                    ggml_nbytes(mine.second));
            report(mine.first.c_str(), got, *ref_t);
        }
    } else if (!probe_name.empty()) {
        printf("\nсверка промежуточного тензора %s:\n", probe_name.c_str());
        ggml_tensor* mine = g.probe(probe_name);
        if (!mine) {
            printf("  у нас нет узла с таким именем\n");
        } else if (!probe.found) {
            printf("  эталон такого узла не выдал\n");
        } else {
            std::vector<float> got;
            got.resize(size_t(ggml_nelements(mine)));
            ggml_backend_tensor_get(mine, got.data(), 0, ggml_nbytes(mine));
            report(probe_name.c_str(), got, probe.data);
        }
    }

    printf("\nсверка логитов на последнем токене:\n");
    printf("  относительная ошибка L2: %.4f%%\n",
           den > 0.0 ? 100.0 * std::sqrt(num / den) : -1.0);
    printf("  наибольшее расхождение : %.5f\n", worst);
    printf("  лучший токен: эталон %d '%s', наш %d '%s' — %s\n",
           a_ref, buf_r, a_our, buf_o, a_ref == a_our ? "совпал" : "РАСХОДЯТСЯ");
    printf("\nвремя префилла: эталон %.0f мс, наш %.0f мс (%d токенов, %d потоков)\n",
           ref_ms, our_ms, n, threads);

    // ------------------------------------------------------------------------------------
    // Step-by-step decode against the reference.
    //
    // This is the check that matters, and the prefill comparison above is the cheap one.
    //
    // A prefill validates one graph over a fixed input. It cannot see a fault in state that
    // is CARRIED between steps, because a prefill carries nothing - it starts from zero and
    // ends. qwen35moe's thirty delta-net layers carry a 128x128 matrix per head from one
    // token to the next, and a slightly wrong carry degrades smoothly: the model stays
    // grammatical, stays on topic, and reads as "a bit worse", which is indistinguishable
    // from quantisation noise and from nothing at all. This project has already been burned
    // once by exactly that failure shape, when a float-reassociation bug moved the logits
    // 3-6% while every generated token still matched.
    //
    // So: our step and the reference's step, one token at a time, compared at each position.
    // With --probe all it compares every layer of every step, which is the difference
    // between "they diverge by step 6" and "layer 17 of step 6 is where it starts".
    if (decode_check > 0) {
        if (!want_ref) {
            printf("--decode-check вместе с --no-ref: эта проверка не делает ничего, "
                   "кроме сверки с эталоном\n");
            return 1;
        }
        if (n_gen > 0) {
            // Both walk the same reference context forward from position n. Running them one
            // after the other would feed the second a context that already holds the first
            // one's tokens, and every number it printed would be measured against the wrong
            // history - silently, because the shapes are all still right.
            printf("--decode-check и --gen вместе нельзя: оба продвигают один и тот же "
                   "эталонный контекст с позиции %d\n", n);
            return 1;
        }
        printf("\nпошаговая сверка декода: %d шагов\n", decode_check);
        const int n_kv_max = pad32(n + decode_check + 1);
        if (n_kv_max > 2048) {
            // The reference context above was created with n_ctx = 2048. Going past it does
            // not fail loudly - it evicts - so it is refused here instead.
            printf("--decode-check %d при промпте %d выходит за 2048 позиций эталонного "
                   "контекста\n", decode_check, n);
            return 1;
        }
        if (!place_layers(n_kv_max)) return 1;
        Generator gen;
        gen.want_probes = (probe_name == "all");
        gen.gstat = gsp;
        if (!gen.init(be, buft, &h, arch_q3 ? &w : nullptr, nullptr, n_kv_max, min_experts,
                      expert_thresh, arch_g4 ? &w4 : nullptr,
                      arch_q35 ? &w35 : nullptr)) {
            return 1;
        }
        printf("  кэш %d позиций, %.1f МБ", n_kv_max, gen.kv.bytes(h) / 1e6);
        if (h.n_delta_layers() > 0) {
            printf("; состояние дельта-сети %.1f МБ на %d слоёв", gen.ds.bytes() / 1e6,
                   h.n_delta_layers());
        }
        printf("\n");

        // The prompt again, through the generator's own cache this time. Its last-token
        // logits must match the reference's the same way the graph above did; if they do
        // not, the fault is in the cached path rather than in anything carried, and the
        // per-step numbers below would be measuring the wrong thing.
        std::vector<float> pre;
        if (!gen.prefill(toks.data(), n, 0, &pre)) return 1;
        report("префилл через кэш", pre, ref);

        if (!gen.build_decode(n)) return 1;
        // Greedy, because the check is about arithmetic and not about sampling: both sides
        // must be fed the same token at every position or the comparison decays into two
        // different conversations after the first disagreement.
        llama_token tok = llama_token(argmax_of(pre));
        std::vector<float> mine, theirs;
        int n_agree = 0, n_steps = 0;
        double worst_step_l2 = 0.0;
        int worst_step = -1;
        std::vector<llama_token> produced;

        for (int i = 0; i < decode_check; ++i) {
            const int past = n + i;
            if (!gen.step(tok, past, &mine)) return 1;
            // The reference's cb_eval fills `probe` as a side effect of llama_decode, and it
            // keeps the FIRST value it sees under each name - so a step's layers have to be
            // cleared before that step's decode or every step would be compared against the
            // prefill's.
            if (gen.want_probes) probe.all.clear();
            if (llama_decode(lctx, llama_batch_get_one(&tok, 1, past, 0))) {
                printf("  шаг %d: llama_decode не прошёл\n", i);
                return 1;
            }
            const float* rstep = llama_get_logits(lctx);
            if (!rstep) {
                printf("  шаг %d: эталонные логиты недоступны\n", i);
                return 1;
            }
            theirs.assign(rstep, rstep + h.n_vocab);

            double snum = 0.0, sden = 0.0;
            for (int k = 0; k < h.n_vocab; ++k) {
                const double d = double(mine[size_t(k)]) - double(theirs[size_t(k)]);
                snum += d * d;
                sden += double(theirs[size_t(k)]) * double(theirs[size_t(k)]);
            }
            const double l2 = sden > 0.0 ? 100.0 * std::sqrt(snum / sden) : -1.0;
            if (l2 > worst_step_l2) { worst_step_l2 = l2; worst_step = i; }
            const int am_mine = argmax_of(mine), am_theirs = argmax_of(theirs);
            const bool same = am_mine == am_theirs;
            if (same) ++n_agree;
            ++n_steps;
            char pm[64] = {0};
            llama_token_to_piece(model, am_mine, pm, sizeof(pm) - 1, 0, true);
            printf("  шаг %2d (позиция %4d): L2 %7.4f%%  токен наш %6d '%s'%s\n", i, past,
                   l2, am_mine, pm, same ? "" : "  — РАСХОДИТСЯ С ЭТАЛОНОМ");
            if (gen.want_probes) {
                for (const auto& mineP : gen.dec.probes) {
                    const std::vector<float>* ref_t = probe.get(mineP.first);
                    if (!ref_t) continue;
                    std::vector<float> got(size_t(ggml_nelements(mineP.second)));
                    ggml_backend_tensor_get(mineP.second, got.data(), 0,
                                            ggml_nbytes(mineP.second));
                    report(("    " + mineP.first).c_str(), got, *ref_t);
                }
            }
            produced.push_back(am_mine);
            // Both sides advance on OUR token, deliberately. Feeding each side its own pick
            // would let them drift apart into two different contexts and every later number
            // would describe two different computations rather than one.
            tok = am_mine;
            if (tok == llama_token_eos(model)) {
                printf("  (eos на шаге %d)\n", i);
                break;
            }
        }
        printf("  итог: %d из %d шагов дали тот же токен; худший L2 %.4f%% на шаге %d\n",
               n_agree, n_steps, worst_step_l2, worst_step);
        printf("  наш текст: %s\n",
               detokenise(model, produced.data(), int(produced.size())).c_str());
        if (n_agree != n_steps) {
            printf("  РАСХОЖДЕНИЕ: токены разошлись — граф считает не то, что эталон\n");
        } else if (worst_step_l2 > 1.0) {
            // Same tokens is necessary and not sufficient, and this is the line that says so.
            printf("  ВНИМАНИЕ: токены совпали, но логиты разошлись на %.4f%% — "
                   "совпадение токенов не является доказательством\n", worst_step_l2);
        }
        gen.free_all();
    }

    // Generation with our own cache, and the honest acceptance test: the same tokens, or
    // not. Logit L2 is a diagnostic - two implementations of the same arithmetic will differ
    // by rounding and the number will drift with depth. Whether the text is identical is the
    // question that actually matters.
    if (n_gen > 0) {
        // -c now reaches this path as well as chat's. It used to be sized from the prompt
        // alone, which is fine for a comparison and useless for the zoned cache: zones only
        // exist when the context is longer than the exact zones, so a run that wants to see
        // a tail has to be able to ask for one.
        const int n_kv_max = std::max(pad32(n + n_gen), pad32(n_ctx_req));
        if (!place_layers(n_kv_max)) return 1;
        Cache kv;
        if (!kv.init(buft, h, n_kv_max)) {
            printf("кэш не выделился\n");
            return 1;
        }
        printf("\nгенерация: наш кэш %d позиций, %.1f МБ\n", n_kv_max,
               kv.bytes(h) / 1e6);

        // The resident expert set. Constructed before the prefill graph because the prefill
        // graph exports its routing when the set exists, which is how the sliding window is
        // warm by the time the first token is generated.
        // The GPU half is built BEFORE the resident set, because when --gpu-experts is given
        // without --resident N it is the free video memory that decides the capacity, and the
        // resident set has to be constructed with the number that actually got allocated
        // rather than with the number that was asked for. A set larger than the slot map
        // would route picks to slots that do not exist, and the only symptom would be a
        // fraction of every layer silently contributing zero.
        // A raw pointer alongside the owner, so the build in this tree without Vulkan can
        // still name it: memex::GpuExperts is an incomplete type there, and a unique_ptr to
        // an incomplete type cannot be destroyed.
        memex::GpuExperts* gxp = nullptr;
#ifdef MEMEX_FWD_GPU_EXPERTS
        std::unique_ptr<memex::GpuExperts> gx;
        if (gopt.on) {
            std::vector<ggml_tensor*> gu, gg, gd;
            gu.reserve(size_t(h.n_layer)); gg.reserve(size_t(h.n_layer));
            gd.reserve(size_t(h.n_layer));
            for (int il = 0; il < h.n_layer; ++il) {
                gu.push_back(w.layers[size_t(il)].up);
                gg.push_back(w.layers[size_t(il)].gate);
                gd.push_back(w.layers[size_t(il)].down);
            }
            memex::GpuExpertsConfig gc;
            gc.n_layers  = h.n_layer;
            gc.n_experts = h.n_expert;
            gc.n_used    = h.n_expert_used;
            gc.n_embd    = h.n_embd;
            gc.capacity  = ropt.capacity;   // 0 means "as many as the device will hold"
            gc.reserve   = size_t(std::max(0, gopt.reserve_mib)) * 1024u * 1024u;
            gc.check     = gopt.check;
            // A promoted expert is not resident until its bytes are confirmed. See the
            // deferred-activation block in resident_set.hpp for why.
            gc.deferred  = true;
            // Where the uploader takes its bytes from. Decided on what the RAM tensors ACTUALLY
            // are, never on what the load was asked to do - those are different questions and
            // the second one has misled this project more than once.
            //
            //  - experts left in their stored type (option 3): the RAM copy IS the plain bytes.
            //    Upload straight from it: no second file handle, no seek per promotion, and the
            //    pages are the mapping's, so they are page cache rather than 15.5 GB of
            //    anonymous memory sitting in front of the Vulkan staging allocation.
            //  - experts repacked: RAM holds iq4_xs_r8, which no Vulkan op implements. Fall
            //    back to reading the pre-repack bytes out of the GGUF.
            bool experts_plain = true;
            for (int il = 0; il < h.n_layer && experts_plain; ++il) {
                ggml_tensor* tt[3] = {gu[size_t(il)], gg[size_t(il)], gd[size_t(il)]};
                for (ggml_tensor* x : tt) {
                    if (x == nullptr || type_is_interleaved(x->type) || x->data == nullptr ||
                        x->buffer == nullptr || !ggml_backend_buffer_is_host(x->buffer)) {
                        experts_plain = false;
                        break;
                    }
                }
            }
            gc.model_path = experts_plain ? std::string() : model_path;
            printf("\nрезидентные эксперты на GPU: инициализация\n");
            gx.reset(new memex::GpuExperts());
            std::string gerr;
            if (!gx->init(gc, gu.data(), gg.data(), gd.data(), &gerr)) {
                printf("--gpu-experts: %s\n", gerr.c_str());
                return 1;
            }
            gxp = gx.get();
            ropt.capacity = gx->capacity();
            printf("  устройство: %s\n", gx->device_name().c_str());
            // The single fact that decides whether repacking and the card can coexist.
            printf("  тип в видеопамяти: %s   (в RAM для CPU-половины: %s)\n",
                   gx->uploaded_type_name(), gx->host_type_name());
            printf("  источник промоушенов: %s\n",
                   gx->uploads_from_file()
                       ? "GGUF напрямую — неперепакованные байты (RAM перепакована)"
                       : "тензоры в RAM напрямую — эксперты не перепакованы, байты уже те самые");
            printf("  промежуточный буфер подкачки: %d промоушен(ов), память %s%s\n",
                   gx->stage_slots(),
                   gx->stage_pinned() ? "закреплённая (pinned)" : "обычная",
                   gx->stage_pinned()
                       ? " — подкачка идёт одним submit на пакет"
                       : " — ПАКЕТИРОВАНИЕ ВЫКЛЮЧЕНО: закрепить не удалось, каждая матрица "
                         "снова стоит submit+fence и требует общий sync_staging");
            printf("  %d экспертов на слой, %.2f МиБ каждый, %.2f ГиБ в видеопамяти\n",
                   gx->capacity(), double(gx->bytes_per_expert()) / 1048576.0,
                   double(gx->vram_bytes()) / 1073741824.0);
            // Where each group actually landed, which is the single most misleading thing in
            // this project's GPU work. The claim is not "device-local" - every report says
            // that, including for a buffer the driver put in system RAM behind the BAR window
            // - it is that the buffer is bigger than the BAR heap, and ggml's own memory type
            // search (ggml-vulkan.cpp:1580-1594) rejects a type whose heap is smaller than
            // the buffer. A buffer over 256 MiB therefore CANNOT be typed onto that heap.
            for (const memex::GpuExpertsBuffer& b : gx->buffers()) {
                printf("  буфер: слои %2d..%-2d  %8.2f МиБ  %s\n", b.first_layer,
                       b.first_layer + b.n_layers - 1, double(b.bytes) / 1048576.0,
                       b.over_bar ? "> 256 МиБ — только DEVICE_LOCAL, не BAR"
                                  : "<= 256 МиБ — МОГ СЕСТЬ В BAR-КУЧУ");
            }
            printf("  входы, идентификаторы и выход половины лежат в маленьких буферах и "
                   "потому host-visible: запись в них — memcpy, без submit\n");
            if (gopt.check) {
                printf("  --gpu-experts-check: резидентная половина считается ДВАЖДЫ, на "
                       "устройстве и на CPU, и сверяется послотно\n");
            }
        }
#endif

        std::unique_ptr<memex::ResidentSet> rset;
        ResidentReport rrep;
        if (ropt.on()) {
            memex::ResidentParams rp;
            rp.n_layers  = h.n_layer;
            rp.n_experts = h.n_expert;
            rp.n_used    = h.n_expert_used;
            rp.capacity  = ropt.capacity;
            rp.window    = ropt.window;
            rp.period    = ropt.period;
            rp.budget    = ropt.budget;
            rp.lfu       = ropt.lfu;
            std::string rerr;
            if (!rp.validate(&rerr)) {
                printf("резидентные эксперты: конфигурация отвергнута — %s\n", rerr.c_str());
                return 1;
            }
            rset.reset(new memex::ResidentSet(rp));
            printf("\nрезидентные эксперты: %d из %d на слой, окно %d токенов, обновление "
                   "каждые %d, бюджет %d, политика %s\n", ropt.capacity, h.n_expert,
                   ropt.window, ropt.period, ropt.budget, ropt.policy_name());
            printf("  один эксперт %.2f МБ, весь резидентный набор %.2f ГБ на %d слоёв\n",
                   expert_bytes_one(w, h) / 1e6,
                   expert_bytes_one(w, h) * double(ropt.capacity) * double(h.n_layer) / 1e9,
                   h.n_layer);
        }

        Graph pre;
        if (!build_step(&pre, buft, h, w, kv, n, 0, n_kv_max, min_experts, expert_thresh,
                        /*all_logits=*/false, /*zc=*/nullptr, /*keep_dbg=*/false,
                        rset.get(), /*gx=*/nullptr, gsp)) {
            printf("граф префилла не собрался\n");
            return 1;
        }
        // The decode graph is built once and re-aimed per step. The position a token writes
        // its key and value to used to be baked into the destination view, so the graph had
        // to be rebuilt every step; building it once and leaving the offset alone made every
        // step overwrite position zero, and the output collapsed into a repeated ".?" after
        // two tokens. The offset is not actually a build-time constant though - the
        // reference reuses its graphs too, and patches the write offset of the recorded cpy
        // nodes before each run (llama_context::update_cache_copies). Everything else a
        // decode step varies - the token, the position, the mask - was already an input
        // tensor, and n_kv is fixed at n_kv_max, so nothing else in the graph moves.
        std::vector<float> lg;
        lg.assign(size_t(h.n_vocab), 0.0f);
        std::vector<int32_t> ps;
        std::vector<float> mk;

        bool aim_warned = false;
        auto set_inputs = [&](Graph& gr, const llama_token* tk, int nt, int past) {
            // Narrow the step to the positions that are actually occupied, rounded up to 32
            // for the F16 matmul's sake. This is where the context-length dependence used to
            // come from: the graph is built at n_kv_max and, left alone, every step read the
            // whole allocation and threw away all of it above `past + nt` through the mask.
            const int want = std::min(pad32(past + nt), n_kv_max);
            if (!gr.aim_kv_reads(want) && !aim_warned) {
                aim_warned = true;
                printf("не удалось сузить чтение кэша до %d позиций — шаги читают все %d "
                       "выделенных\n", want, n_kv_max);
            }
            // Whatever the read side ended up at, the mask has to agree with it: it is the
            // mask's own row length, not n_kv_max, that the softmax steps through.
            const int nkv = int(gr.mask->ne[0]);
            ps.resize(size_t(nt));
            for (int i = 0; i < nt; ++i) ps[size_t(i)] = past + i;
            mk.assign(size_t(nkv) * size_t(nt), -INFINITY);
            for (int i = 0; i < nt; ++i) {
                for (int j = 0; j <= past + i && j < nkv; ++j) {
                    mk[size_t(i) * size_t(nkv) + size_t(j)] = 0.0f;
                }
            }
            ggml_backend_tensor_set(gr.tokens, tk, 0, sizeof(int32_t) * size_t(nt));
            ggml_backend_tensor_set(gr.positions, ps.data(), 0,
                                    sizeof(int32_t) * size_t(nt));
            ggml_backend_tensor_set(gr.mask, mk.data(), 0, ggml_nbytes(gr.mask));
#ifdef MEMEX_FWD_GPU_EXPERTS
            // The device graphs carry the same two numbers and are aimed by the same call, in
            // the same place. A step aimed on one side only reads the cache at one length and
            // the mask at another, which is a wrong answer rather than an error.
            if (gr.on_card && gsp && !gsp->set_step(past, want) && !aim_warned) {
                aim_warned = true;
                printf("не удалось нацелить шаг на карте: past %d, n_kv %d\n", past, want);
            }
#endif
        };
        auto argmax_of = [&](const std::vector<float>& v) {
            return llama_token(std::max_element(v.begin(), v.end()) - v.begin());
        };
        // How much the winning logit won by. A greedy sequence can diverge from the
        // reference on a near-tie without anything being wrong, and that is impossible to
        // tell apart from a real fault after the fact - so the margin is recorded as it is
        // produced, per step, and printed at the first disagreement.
        auto top2_gap = [&](const std::vector<float>& v) {
            float best = -INFINITY, second = -INFINITY;
            for (float x : v) {
                if (x > best) { second = best; best = x; }
                else if (x > second) { second = x; }
            }
            return double(best) - double(second);
        };
        std::vector<double> gaps;

        std::vector<llama_token> ours_seq;
        set_inputs(pre, toks.data(), n, 0);
        ggml_backend_graph_compute(be, pre.gf);
        ggml_backend_synchronize(be);
        ggml_backend_tensor_get(pre.logits, lg.data(), 0, sizeof(float) * size_t(h.n_vocab));
#ifdef MEMEX_FWD_GPU_EXPERTS
        // The prompt ran on the host, so the card's cache is empty - and an empty cache still
        // produces fluent text, because attention over zeros is attention over something. So
        // it is copied rather than assumed, once per prompt, and the failure is fatal.
        if (gsp && gsp->layers_on()) {
            std::string uerr;
            const auto t_kv = Clock::now();
            if (!gsp->upload_kv(kv.k.data(), kv.v.data(), h.n_layer, &uerr)) {
                printf("кэш промпта не уехал на карту: %s\n", uerr.c_str());
                return 1;
            }
            printf("  кэш промпта уехал на карту за %.0f мс (%.1f МБ)\n", ms_since(t_kv),
                   double(kv.bytes(h)) / 1e6);
        }
#endif

        // Scratch for the resident set, allocated once: the residency mask the graph reads and
        // the two id lists it writes back. Sized here so no step allocates.
        std::vector<float> rmask;
        std::vector<int32_t> rsel, rres, roth, rmerged;
        if (rset) {
            rmask.assign(size_t(h.n_expert) * size_t(h.n_layer), 0.0f);
            rres.assign(size_t(h.n_expert_used), 0);
            roth.assign(size_t(h.n_expert_used), 0);
            rmerged.assign(size_t(h.n_expert_used), 0);
        }

        // Warm the policy on the prompt's own routing. Without this the window is empty for
        // the first `window` generated tokens and the reported hit rate describes the warm-up
        // rather than the policy - which on a 24-token run is the whole run. The order matters:
        // the window is per layer but the refresh period counts tokens, so the walk is token
        // outer, layer inner.
        if (rset && !pre.sel_ids.empty()) {
            if (int(pre.sel_ids.size()) != h.n_layer) {
                printf("прогрев резидентного набора: граф префилла вернул %zu слоёв "
                       "маршрутизации вместо %d\n", pre.sel_ids.size(), h.n_layer);
                return 1;
            }
            rsel.assign(size_t(h.n_layer) * size_t(h.n_expert_used) * size_t(n), 0);
            for (int il = 0; il < h.n_layer; ++il) {
                ggml_tensor* t = pre.sel_ids[size_t(il)];
                if (ggml_nelements(t) != int64_t(h.n_expert_used) * int64_t(n)) {
                    printf("прогрев: слой %d вернул %lld идентификаторов вместо %d\n", il,
                           (long long)ggml_nelements(t), h.n_expert_used * n);
                    return 1;
                }
                ggml_backend_tensor_get(t,
                    rsel.data() + size_t(il) * size_t(h.n_expert_used) * size_t(n), 0,
                    ggml_nbytes(t));
            }
            for (int t = 0; t < n; ++t) {
                for (int il = 0; il < h.n_layer; ++il) {
                    rset->observe(il, rsel.data() +
                                          size_t(il) * size_t(h.n_expert_used) * size_t(n) +
                                          size_t(t) * size_t(h.n_expert_used),
                                  h.n_expert_used);
                }
                rset->end_token();
            }
            rrep.warm = rset->stats();
            printf("  прогрев на промпте: %d токенов, попаданий %.1f%%, "
                   "в наборе слоя 0 теперь %d из %d\n", n, 100.0 * rrep.warm.hit_rate(),
                   rset->n_resident(0), ropt.capacity);
        }

        llama_token next = argmax_of(lg);
        gaps.push_back(top2_gap(lg));

        // The exact decode graph. Skipped entirely when the zoned cache is free-running:
        // there is nothing for it to do then, and building it anyway would reserve a second
        // compute buffer to say so.
        Graph dec;
        bool dec_built = false;
        double build_ms = 0.0;
        if (!zopt.on || zopt.check) {
            const auto t_b = Clock::now();
            if (!build_step(&dec, buft, h, w, kv, 1, n, n_kv_max, min_experts, expert_thresh,
                            /*all_logits=*/false, /*zc=*/nullptr,
                            /*keep_dbg=*/zopt.check || rset != nullptr,
                            /*rs=*/nullptr, /*gx=*/nullptr, gsp)) {
                printf("граф декода не собрался\n");
                return 1;
            }
            build_ms = ms_since(t_b);
            dec_built = true;
        }

        // The split decode graph. The exact one above stays built and is what the split is
        // checked against, step for step: --resident is bookkeeping at this stage, so the sum
        // of the two halves must reproduce the unsplit result, and that is checkable here
        // without a line of Vulkan. Both graphs write the same key and value to the same
        // position of the same cache, so running one after the other on a step is harmless -
        // the second write is bit-identical to the first.
        Graph rdec;
        bool rdec_built = false;
        if (rset) {
            const auto t_rb = Clock::now();
            if (!build_step(&rdec, buft, h, w, kv, 1, n, n_kv_max, min_experts, expert_thresh,
                            /*all_logits=*/false, /*zc=*/nullptr, /*keep_dbg=*/true,
                            rset.get(), gxp, gsp)) {
                printf("расщеплённый граф декода не собрался\n");
                return 1;
            }
            if (int(rdec.res_ids.size()) != h.n_layer ||
                int(rdec.oth_ids.size()) != h.n_layer || !rdec.res_mask) {
                printf("расщеплённый граф декода собрался без маски или без списков "
                       "идентификаторов (%zu/%zu слоёв)\n", rdec.res_ids.size(),
                       rdec.oth_ids.size());
                return 1;
            }
            rdec_built = true;
            printf("  расщеплённый граф декода собран один раз за %.1f мс, узлов %d "
                   "(нерасщеплённый %d)\n", ms_since(t_rb), ggml_graph_n_nodes(rdec.gf),
                   dec_built ? ggml_graph_n_nodes(dec.gf) : -1);
#ifdef MEMEX_FWD_GPU_EXPERTS
            // The first fill of video memory, before the clock starts. Every slot is empty at
            // this point, so the first sync queues the whole resident set - a couple of
            // gigabytes over PCIe - and charging that to the first generated token would put
            // seconds into one step and make every per-token figure a fiction. Afterwards a
            // refresh moves only what the policy promoted.
            if (gxp) {
                const auto t_fill = Clock::now();
                gxp->sync_slots(*rset);
                gxp->drain();
                const std::string gfail = gxp->failure();
                if (!gfail.empty()) {
                    printf("резидентные эксперты на GPU: поток устройства упал — %s\n",
                           gfail.c_str());
                    return 1;
                }
                printf("  первичная заливка видеопамяти: %llu экспертов, %.2f ГБ, %.0f мс\n",
                       (unsigned long long)gxp->stats().promotions,
                       double(gxp->stats().promo_bytes) / 1e9, ms_since(t_fill));
                // Everything the first fill uploaded is already resident in the set (the
                // prompt's refreshes ran before deferred activation was switched on), so these
                // confirmations are no-ops. Drained anyway, so the queue starts empty and the
                // landing latencies below are measured only over the generated tokens.
                std::vector<uint32_t> fill_landed;
                gxp->take_landed(&fill_landed);
                // From here on a promotion is pending until the uploader says otherwise. Turned
                // on only now, and not at construction, so the prompt warm-up and its reported
                // hit rate are exactly what they were before this existed - there is no
                // uploader during the warm-up, and a set that could never activate would warm
                // to nothing.
                rset->set_deferred(true);
                printf("  отложенная активация: ВКЛЮЧЕНА — промоушен считается резидентным "
                       "только после подтверждения байтов в видеопамяти; до тех пор эксперт "
                       "считается на CPU и слой его не ждёт\n");
            }
#endif
        }

        const auto t_gen = Clock::now();
        if (zopt.on) {
#ifdef MEMEX_FWD_ZONED
            // ---- the zoned arm -------------------------------------------------------
            //
            // Two things are deliberately *not* zoned. The prefill is exact, because
            // build_attn scores one query and a prompt is many - so both arms start from the
            // same past, position for position, and every difference after that is the
            // zoning and nothing else. And the exact cache stays allocated and untouched by
            // this arm, which is what lets the gate run both in one process without either
            // writing over the other's history.
            ZonedKV zkv;
            if (!zkv.init(buft, h, n_kv_max, zopt)) return 1;
            printf("\nзонный кэш: %d позиций, тензоры %.1f МБ (точный кэш рядом %.1f МБ)\n",
                   n_kv_max, double(zkv.bytes()) / 1e6, kv.bytes(h) / 1e6);
            if (!seed_zoned(zkv.c, kv, h, n)) return 1;
            if (zopt.check) verify_seed(zkv.c, kv, h, n);
            {
                const memex::ZoneOccupancy o = zkv.c->occupancy(0);
                printf("  после префилла (слой 0): стоки %d, блокнот %d, окно %d, "
                       "хвост %d, потеряно %d\n", o.sinks, o.notebook, o.window, o.tail,
                       o.dropped);
            }
            Graph zdec;
            const auto t_zb = Clock::now();
            if (!build_step(&zdec, buft, h, w, kv, 1, n, n_kv_max, min_experts,
                            expert_thresh, /*all_logits=*/false, zkv.c,
                            /*keep_dbg=*/zopt.check, /*rs=*/nullptr, /*gx=*/nullptr, gsp)) {
                printf("зонный граф декода не собрался\n");
                return 1;
            }
            printf("  зонный граф декода собран один раз за %.1f мс, узлов %d\n",
                   ms_since(t_zb), ggml_graph_n_nodes(zdec.gf));

            std::vector<float> zlg;
            zlg.assign(size_t(h.n_vocab), 0.0f);
            std::vector<float> zrow;
            // Gate accumulators. Counted rather than sampled, and refusals counted
            // separately from failures: a comparison that could not be made is not a
            // comparison that passed.
            int cmp_n = 0, cmp_refused = 0, tok_same = 0, margin_proved = 0;
            int top_n = 0, margin_n = 0;
            int first_flip = -1;
            int dbg_done = 0;
            double worst_full = 0.0, sum_full = 0.0;
            double worst_top = 0.0, sum_top = 0.0;
            double worst_margin = -1.0, sum_margin = 0.0;
            std::vector<llama_token> zoned_seq;

            for (int i = 0; i < n_gen; ++i) {
                const int pos = n + i;
                if (zopt.check) {
                    // The exact arm first, so its token is the one both arms are fed. That
                    // teacher-forcing is on purpose: once the two sequences part, a
                    // per-position logit comparison is between different contexts and means
                    // nothing, and the interesting question - would the zoned cache have
                    // chosen the same token at this position - is answered by its own argmax
                    // rather than by where the sequences happened to diverge.
                    if (!dec.aim_cache_writes(pos)) {
                        printf("не удалось перенаправить запись точного кэша на шаге %d\n", i);
                        return 1;
                    }
                    set_inputs(dec, &next, 1, pos);
                    ggml_backend_graph_compute(be, dec.gf);
                    ggml_backend_synchronize(be);
                    ggml_backend_tensor_get(dec.logits, lg.data(), 0,
                                            sizeof(float) * size_t(h.n_vocab));
                }
                const int slot = zoned_begin_step(zkv.c, h.n_layer, /*keep_exact=*/false);
                if (slot < 0) return 1;
                if (!zdec.aim_zoned_writes(slot)) {
                    printf("не удалось перенаправить запись зонного кэша в слот %d "
                           "на шаге %d\n", slot, i);
                    return 1;
                }
                set_inputs(zdec, &next, 1, pos);
                ggml_backend_graph_compute(be, zdec.gf);
                ggml_backend_synchronize(be);
                ggml_backend_tensor_get(zdec.logits, zlg.data(), 0,
                                        sizeof(float) * size_t(h.n_vocab));
                // The mirror, after the graph and before the next step's eviction. Skipping
                // it is silent: the shapes stay right and the past goes quietly wrong.
                if (!mirror_zoned_row(zkv.c, zdec, h.n_layer, slot, h.d_kv(), &zrow)) {
                    return 1;
                }
                ours_seq.push_back(next);
                zoned_seq.push_back(llama_token(argmax_of(zlg)));
                if (zopt.check) {
                    const LogitCmp c = compare_logits(zlg, lg);
                    if (!c.ok) {
                        cmp_refused++;
                        printf("  шаг %d: сравнение ОТКАЗАНО — %s (|зонный| %.6g, "
                               "|точный| %.6g)\n", i, c.why, c.norm_ours, c.norm_ref);
                    } else {
                        cmp_n++;
                        worst_full = std::max(worst_full, c.rel_full);
                        sum_full += c.rel_full;
                        // A negative rel_top100 or flip_margin means the quantity was not
                        // computable (a zero top-100 norm, a zero top-2 gap). Counted as
                        // "not proved" rather than folded into an average, where it would
                        // pull the mean down and read as a better result.
                        if (c.rel_top100 >= 0.0) {
                            worst_top = std::max(worst_top, c.rel_top100);
                            sum_top += c.rel_top100;
                            top_n++;
                        }
                        if (c.flip_margin >= 0.0) {
                            worst_margin = std::max(worst_margin, c.flip_margin);
                            sum_margin += c.flip_margin;
                            margin_n++;
                            if (c.flip_margin < 1.0) margin_proved++;
                        }
                        if (c.argmax_ours == c.argmax_ref) {
                            tok_same++;
                        } else if (first_flip < 0) {
                            first_flip = i;
                        }
                        // Per step, not only in aggregate. Where the error starts is the
                        // question a summary cannot answer, and it is the question that
                        // separates "the tail is lossy" from "the wiring is wrong": the
                        // zones fill in a known order, so the first step with a non-trivial
                        // error names the zone that broke.
                        const memex::ZoneOccupancy oc = zkv.c->occupancy(0);
                        printf("  шаг %2d поз %4d: top-100 %8.5f%%, запас %8.4f, "
                               "зоны с/б/о/х %d/%d/%d/%d%s\n", i, pos,
                               100.0 * c.rel_top100, c.flip_margin, oc.sinks, oc.notebook,
                               oc.window, oc.tail,
                               c.argmax_ours == c.argmax_ref ? "" : "  ARGMAX РАЗОШЁЛСЯ");
                        // On the first step only, walk the layers. The final logits say the
                        // two paths differ; this says where, and it is the only way to tell
                        // an approximation in attention apart from a mis-wiring that happens
                        // to be plausible. Reported as the *jump*: how much attention added
                        // on top of the error its own input already carried.
                        // Step zero and the first step that is not exact. The second is the
                        // one that matters: if layer zero's *input* still matches and its
                        // attention output does not, the cache contents diverged on the
                        // previous step, which points at a write rather than at the graph.
                        const bool first_bad = dbg_done < 2 && c.rel_top100 > 1e-9;
                        if ((i == 0 || first_bad) && dbg_done < 2 &&
                            dec.dbg.size() == zdec.dbg.size() && !dec.dbg.empty()) {
                            dbg_done += first_bad ? 2 : 1;
                            printf("    (разбор по слоям на шаге %d)\n", i);
                            printf("    по слоям (вход слоя -> выход внимания), "
                                   "отн. L2 в %%:\n");
                            std::vector<float> ae, be_, ao, bo;
                            for (size_t li = 0; li < dec.dbg.size(); ++li) {
                                auto grab = [](ggml_tensor* t, std::vector<float>* out) {
                                    out->assign(size_t(ggml_nelements(t)), 0.0f);
                                    ggml_backend_tensor_get(t, out->data(), 0,
                                                            ggml_nbytes(t));
                                };
                                grab(dec.dbg[li].first, &ae);
                                grab(zdec.dbg[li].first, &be_);
                                grab(dec.dbg[li].second, &ao);
                                grab(zdec.dbg[li].second, &bo);
                                const LogitCmp ci = compare_logits(be_, ae);
                                const LogitCmp co = compare_logits(bo, ao);
                                if (li < 4 || li + 2 >= dec.dbg.size() ||
                                    (co.ok && co.rel_full > 1e-9 &&
                                     (!ci.ok || ci.rel_full <= 1e-9))) {
                                    printf("      слой %2zu: вход %10.6f -> внимание "
                                           "%10.6f%s\n", li,
                                           ci.ok ? 100.0 * ci.rel_full : -1.0,
                                           co.ok ? 100.0 * co.rel_full : -1.0,
                                           (ci.ok && co.ok && co.rel_full > 4.0 * ci.rel_full
                                            && co.rel_full > 1e-5)
                                               ? "   <- внимание добавило ошибку" : "");
                                }
                            }
                        }
                    }
                    next = argmax_of(lg);
                    gaps.push_back(top2_gap(lg));
                } else {
                    next = argmax_of(zlg);
                    gaps.push_back(top2_gap(zlg));
                }
            }

            {
                const memex::ZoneOccupancy o = zkv.c->occupancy(0);
                const std::size_t rb = zkv.c->read_bytes(0);
                const std::size_t rbe = zkv.c->read_bytes_if_exact(0);
                printf("\nзонный KV на текущем контексте (измерено, не смоделировано):\n");
                printf("  занятость зон слоя 0: стоки %d, блокнот %d, окно %d, хвост %d, "
                       "потеряно %d (всего %d)\n", o.sinks, o.notebook, o.window, o.tail,
                       o.dropped, o.total());
                printf("  читает %.1f МБ/токен против %.1f МБ/токен полностью точным "
                       "кэшем — %.2fx\n", double(rb) * h.n_layer / 1e6,
                       double(rbe) * h.n_layer / 1e6,
                       rb ? double(rbe) / double(rb) : 0.0);
                if (o.tail == 0) {
                    printf("  хвоста нет: точные зоны (%d слотов) покрывают весь контекст "
                           "(%d) — экономии здесь взяться неоткуда, смотрите таблицу "
                           "в байтовом бюджете ниже\n", zopt.exact_slots(), o.total());
                }
            }

            if (zopt.check) {
                printf("\nворота корректности: зонный кэш против точного, тот же промпт, "
                       "те же токены на тех же позициях\n");
                if (cmp_n == 0) {
                    printf("  НИ ОДНОГО сравнения не состоялось (%d отказов) — "
                           "ворота ничего не доказали\n", cmp_refused);
                } else {
                    printf("  сравнений %d, отказов %d\n", cmp_n, cmp_refused);
                    printf("  согласие по токену: %d из %d (%.1f%%); первое расхождение "
                           "на шаге %d\n", tok_same, cmp_n,
                           100.0 * double(tok_same) / double(cmp_n), first_flip);
                    printf("  отн. L2 логитов: худшая %.5f%%, средняя %.5f%% "
                           "(весь словарь)\n", 100.0 * worst_full,
                           100.0 * sum_full / double(cmp_n));
                    if (top_n > 0) {
                        printf("  отн. L2 логитов: худшая %.5f%%, средняя %.5f%% (top-100 — "
                               "то, до чего может дотянуться сэмплер)\n", 100.0 * worst_top,
                               100.0 * sum_top / double(top_n));
                    }
                    printf("  ЗАПАС ПО ARGMAX: худший %.4f, средний %.4f; "
                           "доказано (<1) на %d из %d шагов\n", worst_margin,
                           margin_n > 0 ? sum_margin / double(margin_n) : -1.0,
                           margin_proved, cmp_n);
                    printf("    <1 — это доказательство, а не порог: argmax не может "
                           "сдвинуться, пока 2*макс|d| по top-100 меньше отрыва top-2\n");
                    if (margin_proved == cmp_n && cmp_refused == 0) {
                        printf("  ворота ПРОЙДЕНЫ: на каждом шаге argmax сдвинуться не мог\n");
                    } else {
                        printf("  ворота НЕ пройдены: %d шагов без доказательства%s\n",
                               cmp_n - margin_proved,
                               tok_same == cmp_n
                                   ? " (хотя токены всё равно совпали — это свидетельство, "
                                     "а не доказательство)"
                                   : "");
                    }
                }
                std::string ztxt;
                char zp[128];
                for (llama_token t : zoned_seq) {
                    const int len = llama_token_to_piece(model, t, zp, sizeof(zp) - 1, 0,
                                                         true);
                    if (len > 0) ztxt.append(zp, size_t(len));
                }
                printf("  зонный выбор на каждой позиции: %s\n", ztxt.c_str());
            }
            zdec.free_all();
            zkv.free_all();
#endif
        } else if (rdec_built) {
            // ---- the split arm -------------------------------------------------------
            //
            // Every step: write the mask (only when a refresh moved it), run the split graph,
            // read the two id lists back, check them against the host's own bitset, account,
            // and let the policy refresh. The mask is written BEFORE the graph, so the set a
            // token is served from is one chosen without having seen it - the same discipline
            // the simulation used, and the reason its hit rates are not oracles.
            const memex::ResidentStats warm_base = rset->stats();
            uint64_t rev_written = UINT64_MAX;
            std::vector<float> ulg;
            ulg.assign(size_t(h.n_vocab), 0.0f);
            std::vector<uint32_t> landed;
            for (int i = 0; i < n_gen; ++i) {
                ours_seq.push_back(next);
#ifdef MEMEX_FWD_GPU_EXPERTS
                // Before the mask is written, not after: a promotion that landed during the
                // previous token becomes resident here, and the mask this token is served from
                // then includes it. Doing it after would serve the token from a set one refresh
                // stale and make the landing latency read one token longer than it is.
                if (gxp) {
                    gxp->take_landed(&landed);
                    for (uint32_t t : landed) {
                        rset->activate(int(t >> 16), int(t & 0xffff));
                    }
                }
#endif
                if (rev_written != rset->revision()) {
                    for (int il = 0; il < h.n_layer; ++il) {
                        rset->write_mask(il, rmask.data() + size_t(il) * size_t(h.n_expert));
                    }
                    ggml_backend_tensor_set(rdec.res_mask, rmask.data(), 0,
                                            ggml_nbytes(rdec.res_mask));
#ifdef MEMEX_FWD_GPU_EXPERTS
                    // The same refresh, on the video memory side: the slot map is rebuilt
                    // from the same bitset the mask came from and the slots whose occupant
                    // changed are queued for upload. Queued rather than uploaded: the worker
                    // gets ahead of the graph while the graph is still in the attention of
                    // layer 0, and flushes what a layer owes immediately before that layer's
                    // own dispatch. Doing it here, before the compute, is what makes the set
                    // the token is served from one chosen without having seen the token.
                    if (gxp) gxp->sync_slots(*rset);
#endif
                    rev_written = rset->revision();
                }
                if (!rdec.aim_cache_writes(n + i)) {
                    printf("не удалось перенаправить запись в кэш на шаге %d\n", i);
                    return 1;
                }
                set_inputs(rdec, &next, 1, n + i);
                ggml_backend_graph_compute(be, rdec.gf);
                ggml_backend_synchronize(be);
                ggml_backend_tensor_get(rdec.logits, lg.data(), 0,
                                        sizeof(float) * size_t(h.n_vocab));

#ifdef MEMEX_FWD_GPU_EXPERTS
                // Once, on the first step: read a few slots back out of video memory and
                // compare them byte for byte against the model tensor they were copied from.
                // Nothing else in this engine can catch a slot map that is off by one or a
                // stacking that transposed an axis - both produce the wrong expert with the
                // right shape, and the logits would only say "different", which is what they
                // say anyway when a GPU kernel rounds differently from a CPU one.
                if (gxp) {
                    const std::string gfail = gxp->failure();
                    if (!gfail.empty()) {
                        printf("резидентные эксперты на GPU: поток устройства упал на шаге "
                               "%d — %s\n", i, gfail.c_str());
                        return 1;
                    }
                }
                if (gxp && i == 0) {
                    int probed = 0, bad = 0;
                    std::string verr;
                    for (int il = 0; il < h.n_layer && probed < 8; il += 7) {
                        for (int s = 0; s < gxp->capacity() && probed < 8; s += 5) {
                            std::string e;
                            if (!gxp->verify_slot(il, s, &e)) {
                                if (e == "слот пуст") continue;
                                ++bad;
                                if (verr.empty()) verr = e;
                            }
                            ++probed;
                        }
                    }
                    printf("  видеопамять против модели: сверено слотов %d, расхождений %d%s%s\n",
                           probed, bad, bad ? " — " : "", bad ? verr.c_str() : "");
                    if (bad) return 1;
                }
#endif

                // The split the graph actually performed, layer by layer.
                for (int il = 0; il < h.n_layer; ++il) {
                    ggml_backend_tensor_get(rdec.res_ids[size_t(il)], rres.data(), 0,
                                            sizeof(int32_t) * size_t(h.n_expert_used));
                    ggml_backend_tensor_get(rdec.oth_ids[size_t(il)], roth.data(), 0,
                                            sizeof(int32_t) * size_t(h.n_expert_used));
                    // The two halves must partition the slots: exactly one of them holds a
                    // non-negative id in each slot, unless expert reduction dropped it.
                    for (int s = 0; s < h.n_expert_used; ++s) {
                        rmerged[size_t(s)] = rres[size_t(s)] >= 0 ? rres[size_t(s)]
                                                                  : roth[size_t(s)];
                    }
                    // The same split redone on the host, straight out of the bitset. The two
                    // are computed by different routes - the host tests a bit, the graph
                    // gathers a float by the router's id out of a tensor it was handed - so a
                    // disagreement means the mask did not arrive where it was meant to, which
                    // is the one fault in this plumbing that shows no wrong shape and no NaN.
                    // weights null on purpose: the renormalised routing weights never leave
                    // the graph, where the same tensor multiplies both halves after their
                    // expert outputs are gathered. Reading them back to carry them through the
                    // host split would be an invitation to apply them twice.
                    memex::ResidentSplit sp;
                    rset->split(il, rmerged.data(), /*weights=*/nullptr, h.n_expert_used, &sp);
                    int j = 0;
                    for (int s = 0; s < h.n_expert_used; ++s) {
                        rrep.picks_checked++;
                        const bool graph_says = rres[size_t(s)] >= 0;
                        const bool host_says = j < sp.n_resident && sp.resident[j].slot == s;
                        if (graph_says != host_says ||
                            (graph_says && sp.resident[j].id != rres[size_t(s)])) {
                            rrep.split_disagree++;
                        }
                        if (host_says) ++j;
                    }
                    rset->observe(il, rmerged.data(), h.n_expert_used);
                }
                rset->end_token();

                // The proof that the split is bookkeeping: the unsplit graph on the same
                // token at the same position, and its logits. Only the first few steps,
                // because it doubles the work of a step and four of them is already a
                // statement about the arithmetic rather than about one lucky token.
                if (dec_built && i < 4) {
                    if (!dec.aim_cache_writes(n + i)) {
                        printf("не удалось перенаправить нерасщеплённый граф на шаге %d\n", i);
                        return 1;
                    }
                    set_inputs(dec, &next, 1, n + i);
                    ggml_backend_graph_compute(be, dec.gf);
                    ggml_backend_synchronize(be);
                    ggml_backend_tensor_get(dec.logits, ulg.data(), 0,
                                            sizeof(float) * size_t(h.n_vocab));
                    const LogitCmp c = compare_logits(lg, ulg);
                    // Counted only when the comparison was actually possible: a comparison
                    // that was refused is not a comparison that passed, and folding it into
                    // the denominator would read as a failure while folding it into the
                    // numerator would read as a proof.
                    if (c.ok) {
                        rrep.checked++;
                        rrep.worst_rel = std::max(rrep.worst_rel, c.rel_full);
                        if (c.argmax_ours == c.argmax_ref) rrep.argmax_same++;
                    }
                    printf("  шаг %2d: расщеплённый против нерасщеплённого, отн. L2 %.9f%%, "
                           "argmax %s\n", i, c.ok ? 100.0 * c.rel_full : -1.0,
                           c.ok && c.argmax_ours == c.argmax_ref ? "совпал" : "РАСХОДИТСЯ");
                    // On the first step, walk the layers. The logits alone cannot tell a
                    // reassociated sum apart from a mis-wired split: both show up as a
                    // percentage. This can. The split changes nothing but the ORDER of eight
                    // f32 additions, so layer 1's input - which carries layer 0's expert sum
                    // and nothing else - must differ at the scale of f32 rounding, around
                    // 1e-7 relative. If it does and the series then grows smoothly layer by
                    // layer, the percentage on the logits is this model amplifying a rounding
                    // difference, which is a property of the model and not of the split. If
                    // layer 1's input is already off by a percent, the split is wrong, and no
                    // amount of matching tokens would make it right.
                    if (i == 0 && dec.dbg.size() == rdec.dbg.size() && !dec.dbg.empty()) {
                        printf("    по слоям, отн. L2 входа слоя (расщеплённый против "
                               "нерасщеплённого):\n");
                        std::vector<float> ax, bx;
                        auto grab = [](ggml_tensor* t, std::vector<float>* out) {
                            out->assign(size_t(ggml_nelements(t)), 0.0f);
                            ggml_backend_tensor_get(t, out->data(), 0, ggml_nbytes(t));
                        };
                        for (size_t li = 0; li < dec.dbg.size(); ++li) {
                            if (li > 5 && li % 8 != 0 && li + 2 < dec.dbg.size()) continue;
                            grab(dec.dbg[li].first, &ax);
                            grab(rdec.dbg[li].first, &bx);
                            const LogitCmp ci = compare_logits(bx, ax);
                            printf("      слой %2zu: %12.9f%%\n", li,
                                   ci.ok ? 100.0 * ci.rel_full : -1.0);
                        }
                    }
                }
                next = argmax_of(lg);
                gaps.push_back(top2_gap(lg));
            }
            rrep.gen = rset->stats().since(warm_base);
            rrep.set_size = rset->n_resident(0);
            rrep.win_distinct = rset->window_distinct(0);
            rrep.pending_end = rset->n_pending(0);
        } else {
            for (int i = 0; i < n_gen; ++i) {
                ours_seq.push_back(next);
                if (!dec.aim_cache_writes(n + i)) {
                    printf("не удалось перенаправить запись в кэш на шаге %d\n", i);
                    return 1;
                }
                set_inputs(dec, &next, 1, n + i);
                ggml_backend_graph_compute(be, dec.gf);
                ggml_backend_synchronize(be);
                ggml_backend_tensor_get(dec.logits, lg.data(), 0,
                                        sizeof(float) * size_t(h.n_vocab));
                next = argmax_of(lg);
                gaps.push_back(top2_gap(lg));
            }
        }
        const double gen_ms = ms_since(t_gen);
        if (dec_built) printf("граф декода собран один раз за %.1f мс\n", build_ms);

        // The reference, greedily, from the same prompt.
        llama_kv_cache_clear(lctx);
        std::vector<llama_token> ref_seq;
        if (llama_decode(lctx, llama_batch_get_one(toks.data(), n, 0, 0))) {
            printf("эталон не декодировал промпт\n");
            return 1;
        }
        std::vector<float> rlg;
        rlg.assign(size_t(h.n_vocab), 0.0f);
        std::memcpy(rlg.data(), llama_get_logits(lctx), sizeof(float) * size_t(h.n_vocab));
        llama_token rnext = argmax_of(rlg);
        const auto t_ref_gen = Clock::now();
        for (int i = 0; i < n_gen; ++i) {
            ref_seq.push_back(rnext);
            if (llama_decode(lctx, llama_batch_get_one(&rnext, 1, n + i, 0))) break;
            std::memcpy(rlg.data(), llama_get_logits(lctx),
                        sizeof(float) * size_t(h.n_vocab));
            rnext = argmax_of(rlg);
        }
        const double ref_gen_ms = ms_since(t_ref_gen);

        int same = 0;
        while (same < int(std::min(ours_seq.size(), ref_seq.size())) &&
               ours_seq[size_t(same)] == ref_seq[size_t(same)]) {
            ++same;
        }
        std::string ours_txt, ref_txt;
        char piece[128];
        for (llama_token t : ours_seq) {
            const int len = llama_token_to_piece(model, t, piece, sizeof(piece) - 1, 0, true);
            if (len > 0) ours_txt.append(piece, size_t(len));
        }
        for (llama_token t : ref_seq) {
            const int len = llama_token_to_piece(model, t, piece, sizeof(piece) - 1, 0, true);
            if (len > 0) ref_txt.append(piece, size_t(len));
        }
        printf("совпало подряд с начала: %d из %d токенов\n", same, n_gen);
        if (same < n_gen && size_t(same) < gaps.size()) {
            printf("на первом расхождении (шаг %d) наш отрыв top-2: %.5f\n", same,
                   gaps[size_t(same)]);
        }
        printf("наш   : %s\n", ours_txt.c_str());
        printf("эталон: %s\n", ref_txt.c_str());
        // The ids as well as the text: an A/B against an earlier build has to compare the
        // sequence, and two different token sequences can detokenise to the same string.
        printf("наши id:");
        for (llama_token t : ours_seq) printf(" %d", t);
        printf("\n");
        printf("\nскорость генерации: наш %.2f ток/с, эталон %.2f ток/с\n",
               1000.0 * n_gen / gen_ms, 1000.0 * n_gen / ref_gen_ms);
#ifdef MEMEX_FWD_GPU_EXPERTS
        if (gsp) {
            const memex::GpuStaticStats& ss = gsp->stats();
            const double calls = double(ss.calls > 0 ? ss.calls : 1);
            printf("staticheskaja golova na karte: %llu uzlov, %llu strok, %llu dispatchej\n",
                   (unsigned long long)ss.calls, (unsigned long long)ss.rows,
                   (unsigned long long)ss.blocks);
            printf("  na odin uzel: vsego %.3f ms = podjom %.3f + ustrojstvo %.3f + "
                   "zabor %.3f\n",
                   ss.ms_total / calls, ss.ms_upload / calls, ss.ms_device / calls,
                   ss.ms_readback / calls);
            printf("  zabrano s karty %.2f MB za progon\n",
                   double(ss.readback_bytes) / 1e6);
            if (ss.layer_calls > 0) {
                const double lc = double(ss.layer_calls);
                const double tk = double(n_gen > 0 ? n_gen : 1);
                printf("vnimanie i marshrutizator na karte: %llu peresechenij, %.1f na "
                       "tokjen\n", (unsigned long long)ss.layer_calls, lc / tk);
                printf("  na odno peresechenie: vsego %.3f ms = podjom %.3f + ustrojstvo "
                       "%.3f + zabor %.3f\n", ss.layer_ms_total / lc,
                       ss.layer_ms_upload / lc, ss.layer_ms_device / lc,
                       ss.layer_ms_readback / lc);
                printf("  na tokjen %.2f ms; zaborov cherez otobrazhenie %llu, cherez "
                       "zabor s zaborom %llu\n", ss.layer_ms_total / tk,
                       (unsigned long long)ss.layer_readback_mapped,
                       (unsigned long long)ss.layer_readback_fenced);
                printf("  kesh prompta uehal na kartu %llu raz za %.0f ms\n",
                       (unsigned long long)ss.kv_uploads, ss.kv_upload_ms);
            }
            if (!gsp->failure().empty()) {
                printf("  USTROJSTVO OTKAZALO: %s - logity nizhe nedejstvitelny\n",
                       gsp->failure().c_str());
            }
        }
#endif
        // One machine-readable line, ASCII, always printed.
        //
        // Not decoration: the line above is the only place the speed appears and it is in
        // Cyrillic, so a harness has to match it through whatever code page the redirect
        // happened to use - which is a way for an A/B to lose replicates silently, and this
        // project has lost two evenings to exactly that class of fault. ref_tok_s is on the
        // line on purpose and is the CONTROL: llama_decode does not know the card exists, so
        // if it moves between two arms of the same A/B then the machine moved, not the arm.
        {
            double head_ms = 0.0;
#ifdef MEMEX_FWD_GPU_EXPERTS
            if (gsp && gsp->stats().calls > 0) {
                head_ms = gsp->stats().ms_total / double(gsp->stats().calls);
            }
#endif
            printf("STATIC_AB our_tok_s %.4f ref_tok_s %.4f gen_ms %.1f n_gen %d "
                   "head_ms %.4f static %d\n",
                   1000.0 * n_gen / gen_ms, 1000.0 * n_gen / ref_gen_ms, gen_ms, n_gen,
                   head_ms, gsp ? 1 : 0);
        }

        // The byte budget, here rather than only in run_chat - which has no caller, so until
        // now the single most useful diagnostic in this project was unreachable from the
        // command line. It carries the zoned-versus-exact KV rows, which is the point: the
        // saving is a property of the tail, and a run short enough to have no tail has to say
        // so out loud instead of looking like the option is worthless.
        {
            // The positions the last step actually read, not the ones -c allocated: the
            // prompt plus everything generated, rounded up to 32 the way aim_kv_reads does.
            const int n_kv_occ  = std::min(n + n_gen, n_kv_max);
            const int n_kv_read = std::min(pad32(n_kv_occ), n_kv_max);
            const Budget bb = byte_budget(w, h, n_kv_read, double(h.n_expert_used));
            // The measured-milliseconds line is suppressed under --zoned-check: that run
            // computes two decode graphs per token, so the implied bandwidth would describe
            // a workload nobody is proposing to run.
            // Same suppression under --resident: the split arm computes the unsplit graph
            // alongside itself on the first steps, so the implied bandwidth would describe a
            // workload nobody is proposing to run.
            print_budget(bb, h, n_kv_read, n_kv_occ, n_kv_max, double(h.n_expert_used),
                         bandwidth_gbs,
                         (zopt.check || rdec_built) ? -1.0 : gen_ms / n_gen, zopt, &ropt,
                         &rrep, expert_bytes_one(w, h));
        }

#ifdef MEMEX_FWD_GPU_EXPERTS
        if (gxp) {
            const memex::GpuExpertsStats& gs = gxp->stats();
            printf("\nрезидентные эксперты на GPU\n");
            printf("  устройство            : %s\n", gxp->device_name().c_str());
            printf("  слоёв на устройстве   : %llu, из них без единой резидентной "
                   "выборки %llu\n",
                   (unsigned long long)gs.layers, (unsigned long long)gs.layers_empty);
            printf("  экспертов запущено    : %llu (%.2f на слой)\n",
                   (unsigned long long)gs.experts,
                   gs.layers ? double(gs.experts) / double(gs.layers) : 0.0);
            printf("  подкачек по PCIe      : %llu, %.2f ГБ\n",
                   (unsigned long long)gs.promotions, double(gs.promo_bytes) / 1e9);
            printf("  выборок без слота     : %llu%s\n", (unsigned long long)gs.no_slot,
                   gs.no_slot ? "  — ОШИБКА: часть слоя дала ноль" : "");
            // The fence budget, which is what the card path actually costs. Before batching a
            // promotion was three submit+fence pairs and every layer paid a second fence for
            // the readback; now a whole flush is one fence and a mapped readback is none.
            {
                const uint64_t up_f  = gs.batch_fences + gs.upload_fences_unbatched * 3ull;
                const uint64_t rb_f  = gs.readback_fenced;
                const uint64_t cmp_f = gs.layers;   // graph_compute always fences its last submit
                const uint64_t tot   = up_f + rb_f + cmp_f;
                printf("  барьеры (fence)       : подкачка %llu (пакетов %llu, вне пакета %llu"
                       " промоушенов), чтение %llu из %llu, вычисление %llu; всего %llu "
                       "= %.2f на слой\n",
                       (unsigned long long)up_f, (unsigned long long)gs.batch_fences,
                       (unsigned long long)gs.upload_fences_unbatched,
                       (unsigned long long)rb_f,
                       (unsigned long long)(gs.readback_fenced + gs.readback_mapped),
                       (unsigned long long)cmp_f, (unsigned long long)tot,
                       gs.layers ? double(tot) / double(gs.layers) : 0.0);
            }
            printf("  где что лежит         : веса экспертов — Vulkan (%.2f ГиБ, "
                   "%zu буфер(а/ов) > 256 МиБ); вход, идентификаторы, выход половины — "
                   "Vulkan (host-visible); всё остальное — CPU\n",
                   double(gxp->vram_bytes()) / 1073741824.0, gxp->buffers().size());
            if (gxp->config().check) {
                printf("  сверка с CPU-половиной: слотов %llu, ненулевых там где устройство "
                       "не считало %llu, нулевых там где считало %llu\n",
                       (unsigned long long)gs.checked, (unsigned long long)gs.zero_bad,
                       (unsigned long long)gs.owned_bad);
                printf("    два первых нуля — это и есть доказательство послотности: сумма "
                       "половин остаётся x + 0.0f в каждом слоте\n");
                printf("  худшая отн. L2 на занятом слоте: %.3e (макс. |разность| %.3e)\n",
                       gs.worst_rel, gs.worst_abs);
                printf("    это НЕ ноль и не должно им быть: шейдер складывает в другом "
                       "порядке, чем ядро CPU. Ноль здесь означал бы, что устройство ничего "
                       "не считало\n");
            }
        }
#endif

        if (rdec_built) rdec.free_all();
        if (dec_built) dec.free_all();
        pre.free_all();
        kv.free_all();
    }

    g.free_all();
    ggml_backend_free(be);
    llama_free(lctx);
    llama_free_model(model);
    llama_backend_free();
    return a_ref == a_our ? 0 : 2;
}


