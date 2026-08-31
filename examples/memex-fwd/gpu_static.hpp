// The static half of a token's traffic, held in video memory and read there instead of over
// the memory bus. "Static" means literally that: the same bytes on every token, with no
// residency policy, no eviction, no prediction and no slot map. That absence is the point -
// it is what makes this the cheapest large lever the engine has.
//
// WHAT THE ARITHMETIC SAYS, because that is why this file exists rather than a bigger one.
//
// Per generated token on Qwen3-Coder-30B-A3B-mx1, measured off the file:
//
//     vnimanie (48 sloev)        510.4 MB   30%   identical every token
//     golova output.weight       243.4 MB   14%   identical every token
//     marshrutizator gate_inp     48.0 MB    3%   identical every token
//     -----------------------------------------
//     statika                    801.8 MB   47%
//     eksperty (8 iz 128)        918.0 MB   53%   different every token
//
// Every static byte is read on every token, so its return is 1.00 bytes of read per byte of
// residency - the highest ratio anything in the model has. A whole offloaded layer, by
// contrast, holds 128 experts and reads 8: 0.09.
//
// BUT THE STATIC SET IS NOT ONE LEVER, IT IS THREE, AND THEY DIFFER BY HOW OFTEN THE
// PROCESSOR AND THE CARD HAVE TO MEET.
//
// A CPU/GPU rendezvous costs 177 us of host-side coordination, measured independently of any
// device work, and that cost is per crossing and indifferent to size. So each piece of the
// static set has to be judged on crossings, not on bytes:
//
//     golova     1 crossing per TOKEN    reads 243.4 MB   -> 9.8 ms saved, 0.18 ms paid
//     vnimanie  48 crossings per token   reads 510.4 MB   -> 16.7 ms saved, 8.5 ms paid,
//                                                            and needs the KV cache on the
//                                                            card, without which the
//                                                            crossings triple
//     router    48 crossings per token   reads  48.0 MB   ->  1.6 ms saved, 8.5 ms paid:
//                                                            a loss on its own, free once
//                                                            attention is already there
//
// The head is therefore the piece that pays first and pays most per line of code: it sits at
// the very end of the graph, after the last layer, so there is exactly one boundary in the
// whole token and nothing on the host is left waiting behind it. This file implements that
// piece. The attention stage needs the KV cache in video memory and a per-layer subgraph, and
// belongs on top of the residency this class already provides.
//
// WHY THERE IS NO THREAD IN HERE, WHICH IS THE FIRST DIFFERENCE FROM GpuExperts.
//
// GpuExperts needs a host worker thread because its device work is meant to run WHILE the CPU
// computes the other half of the same layer, and the Vulkan backend in this tree implements
// none of ggml's asynchronous interface (ggml-vulkan.cpp:10700-10721 leave synchronize,
// set/get_tensor_async and the five event entries NULL; ggml_backend_graph_compute_async is a
// straight call to iface.graph_compute at ggml-backend.cpp:327-329, and
// ggml_backend_vk_graph_compute fences its own last submit at ggml-vulkan.cpp:9674-9678). So
// concurrency there had to be built out of a thread.
//
// The head has nothing to overlap with. It is the last node in the graph: every layer has
// finished, and the only thing the host does after it is read the logits. A thread would add
// a condition-variable rendezvous - the 177 us item - to buy concurrency with an idle CPU.
// So the device call is made synchronously, from inside the graph node, on the one ggml
// thread that runs it (n_tasks = 1, ith == 0). One thread at a time touches the backend,
// which is all Vulkan requires.
//
// The corollary is a restriction and it is enforced rather than documented: this class and
// GpuExperts cannot both be on in one process. ggml_vk_get_device caches its devices
// (ggml-vulkan.cpp:3067-3073), so two ggml_backend_vk_init(0) calls share one vk_device -
// one command pool, one queue, one staging buffer - and GpuExperts' worker uploads whenever
// it has a promotion pending, which includes the moment the head is running.
//
// THE BAR TRAP, WHICH IS WHY THE BUFFER IS BIGGER THAN THE THING IN IT.
//
// ggml picks a memory type with find_properties (ggml-vulkan.cpp:1580-1594), which accepts a
// type only if heap.size >= the buffer size, and ggml_vk_create_buffer_device
// (ggml-vulkan.cpp:1698) asks first for DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT. On an
// RX 6500 XT that is heap 2, the 256 MiB BAR window, of which about 216 MiB is actually
// free. A buffer that FITS the window is put there; once the window is committed the driver
// backs the rest with system memory, and shader reads collapse from 131 GB/s to 3.1 GB/s
// while every report still calls the memory device-local. A buffer LARGER than 256 MiB
// cannot be typed onto that heap at all, so it falls through to plain DEVICE_LOCAL on heap 0.
//
// output.weight is 243.4 MiB. That is UNDER the ceiling, so on its own it would land in the
// BAR window and the head would read its weights at 3.1 GB/s - about 80 ms, twelve times
// worse than leaving it on the CPU, and it would measure as a working feature. So the buffer
// is deliberately padded past the ceiling. The padding is dead video memory and it is named
// as such in the placement report; it is what the attention weights and the routers will
// occupy when the second stage lands.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace memex {

// Odna geometrija sloja vnimanija. U Gemma 4 ih DVE na tridcat sloev - dvadcat pjat okonnyh s
// golovami 16/8 po 256 i pjat polnyh s 16/2 po 512 - i osnovanie povorota s masshtabom softmax
// menjajutsja vmeste s nimi. U qwen3moe ona odna, poetomu etogo tut ranshe ne bylo.
struct GpuStaticGeom {
    int   n_head    = 0;
    int   n_head_kv = 0;
    int   head_dim  = 0;
    int   n_rot     = 0;    // rope dimensions; equals head_dim unless the model says otherwise
    int   rope_type = 0;
    float rope_base = 0.0f;
    // The WHOLE softmax scale, not a correction to 1/sqrt(head_dim). Zero means "use
    // 1/sqrt(head_dim)", which is what every architecture but gemma4 wants.
    float attn_scale = 0.0f;
    // Sliding window in positions; 0 means the layer sees the whole cache. Twenty-five of
    // Gemma four thirty layers are windowed at 1024.
    int   n_swa     = 0;
};

struct GpuStaticConfig {
    int n_embd  = 0;
    int n_vocab = 0;
    // The widest block of logit rows computed in one device graph. Generation always asks for
    // one; speculative verification asks for draft_max + 1; the cacheless prefill comparison
    // asks for the whole prompt, which is served in blocks of this width rather than by
    // building a graph that wide - the output alone is 594 KiB per row.
    int max_rows = 8;
    // Headroom left on the large device-local heap for ggml's own scratch and the desktop.
    std::size_t reserve = 384u * 1024u * 1024u;
    // Read every uploaded byte back and compare it against the model tensor it came from.
    // Costs a full readback of the head at init; it is the only check that the bytes in video
    // memory are the bytes the model has, and nothing else in the engine can see that.
    bool verify = false;

    // ---- the layer path: attention and the router on the device --------------------------
    //
    // 510.4 MB of the 802 MB static half, against the head's 243.4. It is the bigger prize
    // and the harder one, for a reason that has nothing to do with bytes: the head sits after
    // the last layer and costs ONE crossing per token, while attention costs one per LAYER.
    // At a measured 177 us of host-side coordination per crossing that is 8.5 ms per token
    // before any weight is read, so the whole design question is whether the crossing is
    // cheap enough - and the head measurement says the fixed part of a device call is the
    // item that decides it, not the bandwidth.
    //
    // Decode only, deliberately. Attention over a prompt is a different graph shape (n_tokens
    // rows, a square mask) and prefill is not where the tokens per second are lost; the layer
    // path is built for n_tokens == 1 and the engine keeps the exact CPU graph for everything
    // else. That also means the KV cache the card owns is written one position at a time.
    bool layers = false;
    int   n_layer    = 0;
    int   n_head     = 0;
    int   n_head_kv  = 0;
    int   head_dim   = 0;
    int   n_expert   = 0;
    int   n_kv_max   = 0;   // cache positions to allocate on the device
    int   n_ctx_train = 0;
    int   rope_type  = 0;
    float rope_base  = 0.0f;
    float rms_eps    = 0.0f;

    // Per-layer geometry. EMPTY means "every layer is the shape given by the scalars above",
    // which is the qwen3moe case and keeps that path byte-for-byte what it was. When it is
    // filled it must be n_layer long and it wins over the scalars.
    std::vector<GpuStaticGeom> geom;

    // Build gemma4's block instead of qwen3moe's. Not derivable from the geometry: the shapes
    // could be identical and the block still different, and it is the block that decides which
    // nodes exist and in what order.
    bool gemma_block = false;

    // Vygruzhat li golovu. Dlja gemma4 - NET: ejo postroitel schitaet golovu svoim putjom, s
    // ogranicheniem logitov i privjazkoj k embeddingu, i head() u nejo ne vyzyvaetsja nikogda.
    // Bez etogo flaga na kartu ujdjot 748 MiB mjortvogo gruza iz 3,8 GB - pjataja chast pamjati
    // pod tenzor, kotoryj nikto ne prochtjot.
    bool head = true;

    // The shape of layer il, whichever way it was given.
    GpuStaticGeom at(int il) const {
        if (!geom.empty()) return geom[std::size_t(il)];
        GpuStaticGeom g;
        g.n_head = n_head; g.n_head_kv = n_head_kv; g.head_dim = head_dim;
        g.n_rot = head_dim; g.rope_type = rope_type; g.rope_base = rope_base;
        return g;
    }
};

// The model's own attention tensors for one layer, still on the host. Read once at init and
// never modified. Grouped into a struct rather than passed as nine parallel arrays because
// nine parallel arrays is how a wq gets uploaded where a wk belongs, with no shape error
// anywhere to catch it.
struct GpuStaticLayer {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* wq = nullptr;
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* q_norm = nullptr;
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* ffn_norm = nullptr;
    ggml_tensor* router = nullptr;
    // OPTIONAL. gemma4 full-attention layers only; null everywhere else. See layer_slots.
    ggml_tensor* rope_freqs = nullptr;
    // OPTIONAL, gemma4 only. Its block is not qwen3moe's: a norm sits between wo and the
    // residual, the router reads the ATTENTION OUTPUT through its own rms weight rather than the
    // feed-forward norm, and the routed half has a second pre-norm of its own.
    ggml_tensor* post_attn_norm  = nullptr;
    ggml_tensor* gate_inp_s      = nullptr;
    ggml_tensor* pre_ffw_norm_2  = nullptr;
};

struct GpuStaticStats {
    uint64_t calls    = 0;   // head nodes executed, i.e. graph computations of the host graph
    uint64_t rows     = 0;   // logit rows produced
    uint64_t blocks   = 0;   // device graph_compute calls (one per block of max_rows)
    uint64_t readback_bytes = 0;
    double   ms_total = 0.0; // wall time inside the node, host-measured
    double   ms_upload = 0.0;
    double   ms_device = 0.0;
    double   ms_readback = 0.0;
    // The layer path, counted separately because it is a different shape of cost: many small
    // crossings rather than one large one, which is the whole question about it.
    uint64_t layer_calls = 0;
    double   layer_ms_total = 0.0;
    double   layer_ms_upload = 0.0;
    double   layer_ms_device = 0.0;
    double   layer_ms_readback = 0.0;
    // Readbacks served by a memcpy through a host-visible mapping cost no fence; the fenced
    // ones cost a submit each. Which one a layer gets is a property of where the graph
    // allocator put its output buffer, so it is counted rather than assumed.
    uint64_t layer_readback_mapped = 0;
    uint64_t layer_readback_fenced = 0;
    uint64_t kv_uploads   = 0;   // whole-cache uploads: one per prompt, not per token
    double   kv_upload_ms = 0.0;
};

// One allocated device buffer and the single fact that decides whether it is fast.
struct GpuStaticBuffer {
    std::string what;
    std::size_t bytes    = 0;
    std::size_t padding  = 0;
    bool        over_bar = false;   // > the BAR heap, so it cannot be typed onto it
};

class GpuStatic {
  public:
    GpuStatic() = default;
    ~GpuStatic();
    GpuStatic(const GpuStatic&) = delete;
    GpuStatic& operator=(const GpuStatic&) = delete;

    // `out` is the model's own output.weight, still on the host, read once here and never
    // modified. It must be a type the Vulkan backend implements MUL_MAT for and it must not
    // have been repacked: the _R* interleaved forms appear zero times in ggml-vulkan.cpp for
    // any operation, so an interleaved head is refused by name rather than discovered as a
    // wrong answer. The engine's repack selection defaults to experts only, which is exactly
    // why this works without reading the GGUF back off disk the way GpuExperts has to.
    // `layers` is read only when cfg.layers is set and must then hold cfg.n_layer entries.
    bool init(const GpuStaticConfig& cfg, ggml_tensor* out, std::string* err);

    // The layer half of the residency, and it is a SECOND call rather than an argument to the
    // first for a reason that is not arbitrary: the cache length is not known when the head is
    // placed. n_kv_max comes from the mode - pad32(n_ctx) in chat, pad32(prompt + gen) in the
    // harness - and every one of those is decided after the model has loaded, while the head
    // has to exist before the first graph is built because every graph captures it.
    //
    // `layers` must hold cfg.n_layer entries and its tensors must be host-readable and not
    // repacked. Between the two calls layers_on() is false and the engine keeps the CPU
    // attention block, which is the right answer to "not placed yet" rather than a half state.
    bool init_layers(const GpuStaticLayer* layers, int n_kv_max, std::string* err);

    void shutdown();

    bool on() const { return be_ != nullptr; }
    const GpuStaticConfig& config() const { return cfg_; }
    const GpuStaticStats&  stats()  const { return st_; }
    const std::vector<GpuStaticBuffer>& buffers() const { return bufs_info_; }
    const std::string& device_name() const { return dev_name_; }
    const char* head_type_name() const;
    std::size_t vram_bytes() const { return vram_bytes_; }

    // The graph side, and it is one node.
    //
    // Returns a tensor holding [n_vocab, x->ne[1]] of F32 logits, computed on the device from
    // `x` - which must be the already-normalised hidden state, F32 and contiguous, exactly
    // what ggml_mul_mat(out, x) would have been given. Substituting this for that call is the
    // whole integration: every consumer of the logits tensor sees the same shape, type and
    // meaning, so nothing downstream changes.
    //
    // The shape has to come from somewhere, because ggml_map_custom2's destination is a
    // duplicate of its FIRST source and the first source of a matmul-shaped op is the wrong
    // shape. So a proto tensor of the output's shape is created here and passed as src0; its
    // contents are never read. It is 594 KiB per row of graph arena, and it buys the property
    // that the patch to the engine's graph builder is one line.
    ggml_tensor* head(ggml_context* c, ggml_tensor* x);

    // ---- the layer path -----------------------------------------------------------------
    //
    // One node per layer, and exactly one: the whole attention block plus the residual, the
    // FFN norm and the router matmul, computed on the device from the layer's input.
    //
    // Returns a tensor of [n_embd + n_embd + n_expert, 1] F32, three quantities the host then
    // views apart:
    //
    //     [0 .. n_embd)                 ffn_inp - attention output plus the residual, which
    //                                   is what the layer's final add needs
    //     [n_embd .. 2*n_embd)          the FFN-normed hidden state, which is what the expert
    //                                   dispatch multiplies
    //     [2*n_embd .. +n_expert)       the router's logits, PRE-softmax
    //
    // Three in one tensor rather than three tensors because each readback is a submit and a
    // fence: three would triple the item this design is trying to afford. The router's logits
    // come back raw so the host's softmax and top-k are the same ops in the same order they
    // were before the card existed - the routing decision must not move.
    //
    // `mask` is the host graph's own attention mask, passed as a source so the dependency is a
    // data dependency; its first n_kv entries are uploaded once per token, on the first layer.
    ggml_tensor* layer(ggml_context* c, int il, ggml_tensor* cur, ggml_tensor* mask);

    // Called once per token, from the host, BEFORE the graph that reads it: the write position
    // and the number of cached positions attention covers are per-step numbers, and baking
    // either into the built graph is what makes every step read the whole allocation. This is
    // the device-side twin of Graph::aim_cache_writes and Graph::aim_kv_reads and it patches
    // extents in place for the same reason - the graphs were reserved at n_kv_max, so nothing
    // needs reallocating and shrinking a tensor cannot make two of them overlap.
    //
    // Returns false and changes nothing if the numbers are not usable, in which case the
    // previous aim still stands: reading the whole allocation is slow and right.
    bool set_step(int n_past, int n_kv);

    // The prompt is computed on the host - a prefill is a different graph shape and is
    // compute-bound rather than bandwidth-bound, so moving it buys nothing - which leaves the
    // card's cache empty at the first generated token. This copies the host cache into it,
    // tensor for tensor: same shape, same F16 type, no repacking on the way. Once per prompt.
    //
    // It is the one piece of state that has to cross in that direction, and getting it wrong
    // is invisible: an empty device cache still produces fluent text, because attention over
    // zeros is attention over something. So the shapes are compared rather than assumed.
    bool upload_kv(ggml_tensor* const* k, ggml_tensor* const* v, int n_layer,
                   std::string* err);

    int  n_kv_max() const { return cfg_.n_kv_max; }
    bool layers_on() const { return on() && cfg_.layers && !lg_.empty(); }

    // Every byte of the head in video memory, against the host tensor it was copied from.
    // Waits for nothing - there is no worker - but it is still only valid between graph
    // computations, because it borrows the readback staging.
    bool verify_head(std::string* err);

    // Non-empty when a device call threw. The run is then wrong and has to say so rather
    // than carry on with a graph node that quietly produced zeros.
    std::string failure() const { return fail_msg_; }

  private:
    static void head_op(ggml_tensor* dst, const ggml_tensor* proto, const ggml_tensor* x,
                        int ith, int nth, void* ud);
    void do_head(ggml_tensor* dst, const ggml_tensor* x);
    // One block of at most max_rows rows: upload, dispatch, read back. Returns false and
    // fills fail_msg_ on a device error.
    bool block(const float* x, int n_rows, float* dst);

    bool alloc_head(ggml_tensor* out, std::string* err);
    bool build_graphs(std::string* err);

    static void layer_op(ggml_tensor* dst, const ggml_tensor* proto, const ggml_tensor* cur,
                         const ggml_tensor* mask, int ith, int nth, void* ud);
    void do_layer(int il, ggml_tensor* dst, const ggml_tensor* cur, const ggml_tensor* mask);
    bool alloc_layers(const GpuStaticLayer* src, std::string* err);
    bool build_layer_graphs(std::string* err);

    // One layer's device graph and the four tensors a step has to re-aim in it. The names are
    // the ones Graph::KvRead uses for the same quantities, because they are the same quantities
    // and a second vocabulary for them would be one more thing to get wrong.
    struct LayerGraph {
        ggml_cgraph*   gf = nullptr;
        ggml_gallocr_t ga = nullptr;
        ggml_tensor*   out = nullptr;    // the concatenated [2*n_embd + n_expert, 1] result
        // The three pieces separately. Used when MEMEX_SPLIT_OUT is on and `out` is null: the two
        // ggml_concat nodes that packed them existed only because a separate readback per output
        // used to cost a separate round trip, which folding the readback into the graph removed.
        ggml_tensor*   o_res = nullptr;   // the residual stream, [n_embd, 1]
        ggml_tensor*   o_xf  = nullptr;   // its normed copy, [n_embd, 1]
        ggml_tensor*   o_rl  = nullptr;   // the router logits, [n_expert, 1]
        // gemma4 only: the routed half's own pre-norm of the residual stream. Its dense half uses
        // ffn_norm (o_xf) and its routed half uses pre_ffw_norm_2, and they are different weights
        // over the same input - so the card returns both rather than making the host redo one.
        ggml_tensor*   o_xm  = nullptr;
        ggml_tensor*   kdst = nullptr;   // the write view, aimed at n_past
        ggml_tensor*   vdst = nullptr;
        ggml_tensor*   kcpy = nullptr;   // and the copy node that carries the same offset
        ggml_tensor*   vcpy = nullptr;
        ggml_tensor*   K = nullptr;      // the read views, aimed at n_kv
        ggml_tensor*   V = nullptr;
        ggml_tensor*   kq = nullptr;
        ggml_tensor*   probs = nullptr;
        const float*   mapped = nullptr; // the output's host mapping, when the driver gave one
        bool           mapped_probed = false;
    };

    GpuStaticConfig cfg_;
    GpuStaticStats  st_;
    std::string     dev_name_;
    std::string     fail_msg_;

    ggml_backend_t             be_   = nullptr;
    ggml_backend_buffer_type_t buft_ = nullptr;

    ggml_tensor* src_out_ = nullptr;   // the host tensor, for verify_head

    // The weights. One context, one buffer, deliberately larger than the BAR ceiling.
    ggml_context*         ctx_w_ = nullptr;
    ggml_backend_buffer_t buf_w_ = nullptr;
    ggml_tensor*          d_out_ = nullptr;
    ggml_tensor*          d_pad_ = nullptr;   // the reason the buffer clears the ceiling
    std::size_t           vram_bytes_ = 0;
    std::vector<GpuStaticBuffer> bufs_info_;

    // The input side: one device tensor max_rows wide, written per block. Small, so it lands
    // in the BAR window - which is where a host-written input WANTS to be, because
    // ggml-vulkan.cpp:4659-4664 takes the plain memcpy branch exactly when the buffer is
    // HOST_VISIBLE.
    ggml_context*         ctx_in_ = nullptr;
    ggml_backend_buffer_t buf_in_ = nullptr;
    ggml_tensor*          t_x_    = nullptr;

    // One graph per block width 1..max_rows. Built once; a width is used as it is asked for.
    ggml_context*                ctx_g_ = nullptr;
    std::vector<ggml_cgraph*>    gf_;
    std::vector<ggml_gallocr_t>  ga_;
    std::vector<ggml_tensor*>    n_out_;

    // Pinned host staging for the readback, so ggml_vk_host_get finds the destination and the
    // device copies straight into it instead of through the backend's own staging buffer.
    // NOT the mapped pointer: ggml_vk_buffer_read (ggml-vulkan.cpp:4805-4831) deliberately
    // takes the device-copy path on a non-UMA device even for host-visible memory, because
    // reading back over the BAR aperture is slower than a hardware copy, and 594 KiB a token
    // is far past where that trade turns.
    ggml_backend_buffer_t buf_rb_ = nullptr;
    float*                rb_     = nullptr;
    std::vector<float>    rb_fallback_;

    // ---- the layer path -----------------------------------------------------------------
    //
    // The weights live in their own context and buffer, separate from the head's, because the
    // grouping is what keeps each buffer clear of the 256 MiB BAR window and the two groups
    // have very different sizes: attention is 511 MiB across forty-eight layers, the head plus
    // the routers is 291 MiB. Two buffers, both over the ceiling, no padding needed for either.
    std::vector<ggml_context*>         ctxs_l_;
    std::vector<ggml_backend_buffer_t> bufs_l_;
    std::vector<GpuStaticLayer> lw_;      // the device tensors, same field names as the host's

    // The KV cache, on the device, in the SAME buffers as the attention weights. Keeping it
    // apart would leave 201 MB at a 2048 context - under the 256 MiB BAR window, hence backed
    // by system memory at 3.1 GB/s while every report still called it device-local.
    std::vector<ggml_tensor*> kv_k_, kv_v_;
    ggml_tensor*          kv_pad_ = nullptr;   // clears the BAR ceiling at short contexts

    // Inputs the layer path writes per step: the layer's hidden state, the position, and the
    // attention mask. Small, so they land in the BAR window, which is exactly where a
    // host-written input wants to be.
    ggml_context*         ctx_lin_ = nullptr;
    ggml_backend_buffer_t buf_lin_ = nullptr;
    ggml_tensor*          t_lx_   = nullptr;   // [n_embd, 1] F32
    ggml_tensor*          t_pos_  = nullptr;   // [1] I32
    ggml_tensor*          t_mask_ = nullptr;   // [n_kv_max, 1] F32

    // A node's userdata has to carry both the object and the layer, and ggml_map_custom takes
    // one void*. One of these per layer, addresses stable for the object's lifetime.
    struct Site { GpuStatic* self = nullptr; int il = 0; };

    ggml_context*             ctx_lg_ = nullptr;
    std::vector<LayerGraph>   lg_;
    std::vector<Site>         sites_;

    int  step_past_ = -1;   // what set_step last aimed the writes at
    int  step_nkv_  = 0;    // and the reads
    bool step_mask_sent_ = false;   // the mask is uploaded once per token, on the first layer
    // ...unless it changes. Gemma 4 has TWO masks - twenty-five layers see a 1024-position window
    // and five see the whole cache - so the once-per-token rule is wrong there. Comparing the
    // source pointer gives the same one upload for qwen3moe and the correct two for gemma4.
    const void* step_mask_src_ = nullptr;
};

// A whole-module check that needs no model file: a synthetic Q6_K head of a realistic width,
// uploaded through the same path the engine uses, then the same matmul computed on the CPU
// backend and on the device and compared row by row. It exists because the 30 B model needs a
// quiet machine and seventeen gigabytes, and because everything this class can get wrong - the
// padding that keeps the buffer out of the BAR window, the block loop over rows, the readback
// offsets - is wrong in a way that leaves plausible logits behind.
//
// Returns 0 on success. Prints what it checked either way.
int gpu_static_selftest(int threads);

}  // namespace memex
