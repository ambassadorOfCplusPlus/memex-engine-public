// The resident half of a layer's experts, computed on the Vulkan device while the CPU
// computes the other half, joined at the end of the layer.
//
// WHY THERE IS A THREAD IN HERE, WHICH IS THE FIRST THING TO ASK.
//
// ggml has an asynchronous backend interface - ggml_backend_graph_compute_async,
// ggml_backend_synchronize, ggml_backend_tensor_set_async / _get_async, and the event
// quartet. In this tree the Vulkan backend implements none of it:
//
//   ggml/src/ggml-vulkan.cpp:10700   // TODO: enable async and synchronize
//   ggml/src/ggml-vulkan.cpp:10705-10708  set/get/cpy_tensor_async  = NULL  (commented out)
//   ggml/src/ggml-vulkan.cpp:10708   synchronize                   = NULL  (commented out)
//   ggml/src/ggml-vulkan.cpp:10717-10721  the five event entries    = NULL
//
// and ggml_backend_graph_compute_async is, verbatim,
//
//   ggml/src/ggml-backend.cpp:327-329
//       return backend->iface.graph_compute(backend, cgraph);
//
// so it lands in ggml_backend_vk_graph_compute, which is itself synchronous by construction:
//
//   ggml/src/ggml-vulkan.cpp:9674-9678
//       // always wait for the GPU work to be done for the last submit
//       if (tensor_idx == subctx->exit_tensor_idx) { use_fence = true; }
//       ...
//       if (use_fence) { ggml_vk_wait_for_fence(ctx); }
//
// There is therefore no way to hand the Vulkan backend a graph and come back later. The
// call blocks until the device is idle, and ggml_backend_synchronize on this backend is a
// no-op that returns immediately (ggml-backend.cpp:295-298, iface.synchronize == NULL).
// Concurrency has to be built one level up, out of a host thread that owns the backend and
// makes that blocking call on its own time. That is what this class is.
//
// The shape per layer is then exactly the one the design asks for:
//
//   1. fork()  - a node in the CPU graph that hands the layer's activation and the resident
//                half's ids to the worker and returns immediately;
//   2. the CPU computes the non-resident half, at full thread count, in the same graph;
//   3. join()  - a node that waits for the worker and produces the resident half's output
//                with an exact zero in every slot the worker did not own.
//
// The two are ordered by data dependency rather than by hope: fork() returns the id list the
// CPU half is dispatched with, so no CPU expert node can be scheduled before the submit, and
// join() takes the CPU half's output as an input, so it cannot be scheduled before it.
//
// THE ZEROS ARE THE CORRECTNESS ARGUMENT AND THEY ARE NOT FREE HERE.
//
// The CPU-only split relies on ggml_mul_mat_id treating an id of -1 as "skip this slot and
// memset its output to zero". The Vulkan shader does not do that. mul_mat_vec_base.comp:64
// reads
//
//     const uint expert_id = data_ids[expert_idx];
//
// into an *unsigned*, with no guard, and then multiplies it by the batch stride. An id of -1
// is 0xffffffff there: not a skip, an out-of-bounds read at a wrapped offset. So the resident
// half is dispatched with a COMPACTED id list - only the slots the device actually owns, k of
// them, k <= n_used - and the worker scatters the k results back into their original slots
// and writes an exact 0.0f into the rest. That restores the property the sum depends on:
// every slot of ggml_add(o_res, o_oth) is either x + 0.0f or 0.0f + x, both of which are x
// bit for bit, so the routing weights and the eight-way fold that follow are the unsplit
// path's own operations in the unsplit path's own order. Folding each half separately and
// adding the totals instead reassociates the sum; that was measured at 1.6% by layer 47.
//
// WHAT LIVES IN VIDEO MEMORY AND WHY IT IS PACKED THE WAY IT IS.
//
// Per layer, three tensors of `capacity` experts each: up, gate and down, sliced out of the
// model's own fused [n_embd, n_ff, n_expert] tensors and stacked in SLOT order, not expert
// order. The slot map is maintained here from the ResidentSet's bitset, and a slot whose
// occupant changed is uploaded before the next graph that reads it.
//
// The packing constraint is the device's, and it is brutal on this card. ggml picks a memory
// type with find_properties (ggml-vulkan.cpp:1580-1594), which accepts a type only if
// heap.size >= the buffer size, and ggml_vk_create_buffer_device (ggml-vulkan.cpp:1698) asks
// first for DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT. On an RX 6500 XT that is heap 2,
// the 256 MiB BAR window; once it is committed the driver backs further allocations there
// with system memory and shader reads collapse to 3.1 GB/s while every report still calls the
// memory device-local. A buffer LARGER than 256 MiB cannot be typed onto that heap at all -
// find_properties rejects it on the heap.size test - so it falls through to plain
// DEVICE_LOCAL on heap 0. Hence: every expert buffer here is deliberately larger than
// 256 MiB, and the layer-to-buffer grouping exists to guarantee that rather than to be tidy.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace memex {

class ResidentSet;

struct GpuExpertsConfig {
    int n_layers  = 0;
    int n_experts = 0;
    int n_used    = 0;      // the router's top-k; also the widest compacted id list
    int n_embd    = 0;
    // Resident experts per layer to hold in video memory. 0 asks init() to choose the largest
    // number that fits the free device heap minus `reserve`, which is what --gpu-experts
    // without --resident does.
    int capacity  = 0;
    std::size_t reserve = 384u * 1024u * 1024u;
    // Build a second, CPU-computed copy of the resident half and compare it against the
    // device's, per layer per step. Costs the whole saving; exists to prove the plumbing.
    bool check = false;
    // Path to the GGUF the model was loaded from. When set, promotions read the PRE-REPACK
    // bytes straight out of this file instead of out of the in-RAM expert tensors, which
    // decouples the two layouts: RAM may stay interleaved _R8 for the CPU half (worth +34%)
    // while video memory gets the plain IQ4_XS the Vulkan shaders actually implement. Empty
    // falls back to reading the RAM tensors, which is only valid without repacking.
    std::string model_path;
    // Deferred activation: the ResidentSet treats a promoted expert as pending rather than
    // resident until this class confirms its bytes landed. Two things change here when it is
    // set - the slot map is built from is_claimed() rather than is_resident(), and a layer's
    // dispatch no longer flushes what that layer owes the link, because nothing the dispatch
    // reads can be mid-upload. With it clear the old behaviour is kept exactly.
    bool deferred = false;
};

struct GpuExpertsStats {
    uint64_t layers        = 0;   // layer-steps that reached the device
    uint64_t layers_empty  = 0;   // layer-steps with no resident pick: no dispatch at all
    uint64_t experts       = 0;   // compacted ids actually dispatched
    uint64_t promotions    = 0;   // expert slices uploaded over PCIe
    uint64_t promo_bytes   = 0;
    uint64_t no_slot       = 0;   // resident by the bitset, absent from the slot map: a bug
    // Fence accounting, which is what the card path actually costs. One batch_end is one
    // submit+fence covering however many promotions it held; upload_fences_unbatched counts
    // promotions that had to fall back to a fence per matrix.
    uint64_t batch_fences  = 0;
    uint64_t upload_fences_unbatched = 0;
    uint64_t readback_mapped = 0;   // readbacks served by memcpy through the mapping: 0 fences
    uint64_t readback_fenced = 0;   // readbacks that still cost a submit+fence
    // THE JOIN. This is the quantity that says whether the two halves are parallel at all,
    // and until now nothing counted it.
    //
    // The whole scheme rests on one claim: the device computes the resident half WHILE the
    // CPU computes the other, so the layer costs max(cpu, device) rather than cpu + device.
    // Every other number here - hit rate, promotions, fences - is about how much work the
    // device took. None of them says whether the CPU got to do anything meanwhile.
    //
    // ms_join_wait is that number, and it is a hard one: the join op runs with n_tasks = 1,
    // so while thread 0 sits in cv_done_.wait the other ggml threads are spinning on their
    // barrier with nothing to do. Time here is the ENTIRE pool stopped, not one thread.
    // If it is near zero the overlap works and the cost is elsewhere; if it is most of the
    // token, max() has been a sum all along.
    uint64_t join_waits      = 0;   // times the CPU reached the join
    uint64_t join_ready      = 0;   // ...and the device half was already there: a free join
    double   ms_join_wait    = 0.0; // wall time the ggml pool spent STOPPED at the join
    double   ms_cpu_half     = 0.0; // fork -> join: the work the CPU had to cover the device with
    double   ms_job          = 0.0; // device side: the worker's own dispatch, dequeue to done
    // THE OTHER THING THE WORKER DOES. It is one thread, and it serves two masters: the
    // per-layer dispatch the CPU is waiting for, and the promotions the refresh queued. A
    // job that arrives while a promotion is in flight waits for that promotion's fence, and
    // that wait is invisible in ms_job because ms_job starts after the dequeue.
    uint64_t fork_busy       = 0;   // forks that found the worker already inside a Vulkan call
    double   ms_promote      = 0.0; // worker time spent on promotions rather than dispatches
    // WHAT THE 1.30 ms OF A PROMOTION IS MADE OF. Three of the four terms were priced rather
    // than measured, and one of the three was priced wrong: the staging read was charged at
    // 0.10 ms by dividing 2.51 MB by the machine's AGGREGATE 24.8 GB/s, which is neither one
    // core's share nor a copy's two-sided traffic. 0.45 ms of 1.30 had no owner at all, and it
    // is the only term that was never measured (rule 76 - a per-token figure whose parts come
    // from different derivations is not an accounting).
    //
    // Cost of the instrument: six steady_clock::now() per promotion at ~25 ns, about 1 us per
    // token against 8.98 ms. The check that this is really free is that period 3 is the arm
    // the budget sweep already measured, so promo_ms_tok must come back at 8.99 (rule 69).
    double   ms_read         = 0.0; // read_plain: mmap or GGUF -> pinned staging
    double   ms_record       = 0.0; // recording the copy: batch_set_tensor or tensor_set
    double   ms_fence        = 0.0; // batch_end: submit and wait on the fence
    uint64_t n_read          = 0;   // read_plain calls: three per promotion
    uint64_t read_bytes      = 0;   // bytes those calls moved
    uint64_t n_fence_calls   = 0;   // batch_end calls that had anything recorded
    // --gpu-experts-check
    uint64_t checked       = 0;   // slots compared against the CPU's own resident half
    uint64_t zero_bad      = 0;   // slots the device does not own that came back non-zero
    uint64_t owned_bad     = 0;   // slots the device owns that the CPU left at zero
    double   worst_rel     = 0.0; // worst relative difference on an owned slot
    double   worst_abs     = 0.0;
};

// One allocated device buffer, and the single fact that decides whether it is fast.
struct GpuExpertsBuffer {
    int         first_layer = 0;
    int         n_layers    = 0;
    std::size_t bytes       = 0;
    bool        over_bar    = false;   // > the BAR heap, so it cannot be typed onto it
};

class GpuExperts {
  public:
    GpuExperts() = default;
    ~GpuExperts();
    GpuExperts(const GpuExperts&) = delete;
    GpuExperts& operator=(const GpuExperts&) = delete;

    // `up`, `gate` and `down` are the model's own fused expert tensors, one per layer, still
    // on the host. They are read on every promotion and never modified.
    bool init(const GpuExpertsConfig& cfg, ggml_tensor* const* up, ggml_tensor* const* gate,
              ggml_tensor* const* down, std::string* err);
    void shutdown();

    bool on() const { return be_ != nullptr; }
    int  capacity() const { return cfg_.capacity; }
    const GpuExpertsConfig& config() const { return cfg_; }
    const GpuExpertsStats& stats() const { return st_; }
    const std::vector<GpuExpertsBuffer>& buffers() const { return bufs_info_; }
    std::size_t bytes_per_expert() const { return bpe_; }
    std::size_t vram_bytes() const { return vram_bytes_; }
    const std::string& device_name() const { return dev_name_; }
    // The type of the bytes that actually reach video memory, and where they came from. The
    // engine refuses an unsupported type outright, so a successful run is the proof; this is
    // for the operator to see WHICH type made it.
    const char* uploaded_type_name() const;
    bool        uploads_from_file() const { return plain_on_; }
    // The type the CPU half is computing with, for the contrast.
    const char* host_type_name() const;
    // What the promotion staging ring actually got. Pinned is the whole point: an unpinned
    // source forces ggml to route each matrix through the device's shared sync_staging buffer,
    // which is both a fence per matrix and one more host-visible allocation to fail.
    int  stage_slots()  const { return stage_slots_; }
    bool stage_pinned() const { return stage_pinned_; }
    // How many queued promotions the idle worker folds behind ONE submit and ONE fence. This
    // is the batch size the fence accounting is a function of: at 1 every promotion pays its
    // own submit and its own fence, which is what 1902 batches for 1902 promotions meant.
    // Overridable by MEMEX_PROMO_DRAIN so both arms live in one binary - and printed, because
    // a setting that did not apply is indistinguishable from one that did not help (rule 68).
    int  promo_drain()  const { return promo_drain_; }
    // Whether a batch is closed early as soon as a dispatch is waiting. The fence saving is
    // kept for whatever was already recorded; what is given up is the tail of the batch.
    bool promo_yield()  const { return promo_yield_; }
    bool readback_mapped() const { return !out_mapped_.empty() && out_mapped_.back() != nullptr; }

    // Recompute the slot map from the resident set and queue the uploads the change implies.
    // Call whenever ResidentSet::revision() moves, and BEFORE the graph that reads the mask
    // of that revision: an upload for layer il is flushed by the worker immediately before
    // layer il's own dispatch, so the ordering is what makes a promoted expert's weights the
    // ones the device actually multiplies.
    void sync_slots(const ResidentSet& rs);

    // Promotions whose bytes are confirmed in device memory, as (layer << 16) | expert. Moved
    // out, so each is reported once. "Confirmed" means the fence that consumed the staging slot
    // has passed - batch_end waits on it - or, on the unbatched fallback, that the synchronous
    // ggml_backend_tensor_set returned. Call once per token from the thread that owns the
    // ResidentSet and hand each entry to ResidentSet::activate.
    void take_landed(std::vector<uint32_t>* out);

    // The slot map, for checking rather than for use. Three questions, and together they are
    // the whole safety property of deferred activation:
    //   slot_of(il, e)     which slot holds expert e, -1 if none
    //   expert_at(il, s)   which expert slot s is supposed to hold, -1 if free
    //   slot_dirty(il, s)  whether that slot's bytes are still owed to the link
    // A slot that is dirty must belong to an expert the residency mask does NOT claim. That is
    // what makes "the device slot cannot hold bytes that no longer belong to the expert the
    // mask claims" checkable instead of argued.
    int  slot_of(int il, int expert);
    int  expert_at(int il, int slot);
    bool slot_dirty(int il, int slot);

    // ---- the graph side -----------------------------------------------------------------
    //
    // fork returns a tensor with the same shape, type and contents as `ids_oth`. Using its
    // return value - rather than ids_oth itself - as the CPU half's id list is what forces
    // the submit to be scheduled before the CPU experts.
    ggml_tensor* fork(ggml_context* c, int il, ggml_tensor* ids_oth, ggml_tensor* xe,
                      ggml_tensor* ids_res);
    // join returns the resident half's output, [n_embd, n_used, 1], with an exact zero in
    // every slot the device did not own. `o_res_cpu` is only read under --gpu-experts-check
    // and may be null otherwise.
    ggml_tensor* join(ggml_context* c, int il, ggml_tensor* o_oth, ggml_tensor* o_res_cpu);

    // Reads back one slot's three weight slices from video memory and compares them against
    // the host tensors they were copied from. The one check that the stacking, the slot map
    // and the upload agree; a mismatch here is invisible to every other test in the engine.
    // Waits for the worker to go idle first, so it is safe to call between steps and only
    // between steps.
    bool verify_slot(int il, int slot, std::string* err);

    // Blocks until nothing is queued and the worker is not inside a Vulkan call.
    void drain();

    // Non-empty when the worker thread died; the run is then wrong and has to say so.
    std::string failure();

    // Every memory heap the device exposes, with VK_EXT_memory_budget's usage where it is
    // available. Printed around the allocation, because "device-local" in any other report
    // is exactly the word that hides a buffer the driver quietly moved to system memory.
    static void print_heaps(const char* when);

    // Where a ggml Vulkan buffer of this many bytes is REQUIRED to land, worked out the way
    // ggml works it out: a real VkBuffer of that size with ggml's own usage flags, its memory
    // requirements, and then find_properties' rule (ggml-vulkan.cpp:1585-1594) - first type
    // that is allowed by memoryTypeBits, has every requested property, and sits on a heap at
    // least as large as the buffer. Deterministic, and not the same thing as heapUsage, which
    // is what has misled this project before.
    static void print_placement(const char* what, std::size_t bytes);

    // EVERY QUEUE FAMILY THE DEVICE OFFERS, and which one a promotion is submitted on.
    //
    // The question this answers is whether the prefetch can be made asynchronous at all. Our
    // promotions occupy the worker thread for milliseconds per token, and the worker is the
    // thread the CPU joins on; a transfer-only queue family, backed by a DMA engine, copies
    // WHILE the compute queue runs instead of taking turns with it. Whether this card has one
    // is a fact about the hardware, and this project has been wrong four times in one day on
    // exactly that class of assumption - the flagship case being "mul_mat_id is unavailable",
    // which was false and cost half a day. So it is enumerated and printed, not assumed.
    //
    // The second half of the line is ggml's own choice, reproduced from
    // ggml_vk_find_queue_family_index (ggml-vulkan.cpp:1585): the compute family is the first
    // with COMPUTE preferring one without GRAPHICS, and the transfer family is the first with
    // TRANSFER preferring one without COMPUTE or GRAPHICS and different from the compute
    // family. It matters because ctx->transfer_cmd_pool is built on device->transfer_queue
    // (ggml-vulkan.cpp:4162), so our batched uploads already go wherever that rule points.
    static void print_queues();

  private:
    struct Site { GpuExperts* self = nullptr; int il = 0; };

    static void fork_op(ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b,
                        const ggml_tensor* c, int ith, int nth, void* ud);
    static void join_op(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void* ud);
    static void join_check_op(ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b,
                              int ith, int nth, void* ud);

    void do_fork(int il, const ggml_tensor* ids_res, const ggml_tensor* xe);
    void do_join(int il, ggml_tensor* dst, const ggml_tensor* o_res_cpu);

    void worker();
    void worker_loop();
    void flush_layer(int il);
    // False means the run is over: the source read failed, failed_ is set, and nothing the
    // open batch recorded may be confirmed. Callers must stop draining rather than carry on
    // into the next slot, which would only fail again with the batch already closed.
    bool upload(int il, int slot, int expert);
    void compute(int il, int k);

    GpuExpertsConfig cfg_;
    GpuExpertsStats  st_;
    std::string      dev_name_;

    ggml_backend_t             be_   = nullptr;
    ggml_backend_buffer_type_t buft_ = nullptr;

    bool alloc_weights(std::string* err);
    void free_weights();

    // the resident weights: one ggml context and one buffer per group of layers, because the
    // group boundaries are chosen here rather than left to ggml's max-size split - a split
    // that lands a 100 MiB remainder in the BAR heap is the failure this whole arrangement
    // exists to avoid, and ggml has no reason to know that.
    std::vector<ggml_context*>          ctxs_w_;
    std::vector<ggml_backend_buffer_t>  bufs_;
    std::vector<GpuExpertsBuffer>       bufs_info_;
    std::vector<ggml_tensor*>           up_, gate_, down_;
    std::vector<ggml_tensor*>           src_up_, src_gate_, src_down_;
    std::size_t bpe_        = 0;   // bytes one expert occupies, all three matrices
    std::size_t vram_bytes_ = 0;

    // ---- the pre-repack source of truth for everything that reaches the device ----------
    // One of these per (layer, kind) when cfg_.model_path is set. It describes the tensor as
    // it is STORED, not as it sits in RAM: `type` is the on-disk quantisation (IQ4_XS), which
    // is what the device tensors are allocated with and what the shaders can read, whereas
    // the RAM tensor of the same weight may be iq4_xs_r8 after load-time repacking.
    struct PlainSrc {
        ggml_type   type = GGML_TYPE_COUNT;
        int64_t     ne0  = 0;
        int64_t     ne1  = 0;
        std::size_t slab = 0;   // bytes of one expert, i.e. the stored nb[2]
        long long   off  = 0;   // absolute byte offset of the tensor's data in the file
    };
    // Indexed [layer*3 + kind], kind 0=up 1=gate 2=down. Empty when reading from RAM.
    std::vector<PlainSrc> plain_;
    bool  plain_on_ = false;
    FILE* gguf_f_   = nullptr;
    // seek+read is a two-step on one file handle, so it cannot be reentered: the worker
    // thread promotes while the main thread may be verifying a slot.
    std::mutex           io_mu_;

    // ---- promotion staging, pinned so uploads can be batched behind one fence -----------
    // A ring of whole promotions (three matrices each). Pinned via the Vulkan host buffer type
    // so ggml_backend_vk_batch_set_tensor records a straight device-side copy; with ordinary
    // heap memory it would have to route through the single shared device staging buffer, and
    // several batched copies through one staging buffer would alias.
    //
    // A slot cannot be reused until the fence that consumed it has passed, so the ring depth
    // is also the maximum number of promotions one batch may hold.
    ggml_backend_buffer_t buf_stage_ = nullptr;
    char*                 stage_     = nullptr;
    std::vector<char>     stage_fallback_;
    int                   stage_slots_ = 1;
    bool                  stage_pinned_ = false;
    // The ring depth is the ceiling on a batch, so this defaults to it: draining further would
    // make upload() close and reopen the batch mid-drain, reintroducing exactly the fence per
    // ring-full that the drain exists to remove, while making an arriving dispatch wait longer.
    int                   promo_drain_ = 1;
    bool                  promo_yield_ = true;
    // Scratch for the drain, owned by the worker thread: filled under mu_, uploaded outside it.
    std::vector<int>      dr_il_, dr_slot_, dr_expert_;
    bool                  batching_    = false;   // a batch is open on this thread
    int                   batch_used_  = 0;       // promotions recorded into the open batch
    // (layer << 16) | expert. batch_landed_ is worker-thread-only and holds what the OPEN batch
    // has recorded but not yet fenced; batch_end moves it into landed_ under mu_, which is
    // where take_landed drains it. Separating them is the whole confirmation guarantee: a
    // recorded copy is not a landed one until its fence passes.
    std::vector<uint32_t> batch_landed_;
    std::vector<uint32_t> landed_;

    void batch_begin();
    void batch_end();

    // Per compacted width k: the output tensor's mapped host address when the driver put its
    // buffer in host-visible coherent memory, in which case the per-layer readback is a memcpy
    // rather than a submit+fence. One entry per k because each width has its own allocator and
    // therefore its own buffer. Resolved on first use, after the allocators have run.
    std::vector<float*>  out_mapped_;
    std::vector<uint8_t> out_mapped_probed_;

    bool open_plain_source(std::string* err);
    void close_plain_source();
    // Reads expert `expert` of (il, kind) into `dst`, which must hold slab bytes.
    bool read_plain(int il, int kind, int expert, char* dst, std::string* err);
    // Descriptor for (il, kind): the stored one when plain_on_, else the RAM tensor's.
    PlainSrc desc(int il, int kind) const;

    // the compute graphs, one per compacted width k = 1..n_used, built once
    ggml_context*                ctx_g_ = nullptr;
    std::vector<ggml_cgraph*>    gf_;
    std::vector<ggml_gallocr_t>  ga_;
    std::vector<ggml_tensor*>    n_up_, n_gate_, n_down_, n_out_, n_ids_;
    ggml_context*                ctx_in_ = nullptr;
    ggml_backend_buffer_t        buf_in_ = nullptr;
    ggml_tensor*                 t_x_    = nullptr;

    // pinned host staging for the read back, so ggml_vk_host_get finds it and the device
    // copies straight into it instead of through its own staging buffer
    ggml_backend_buffer_t        buf_rb_ = nullptr;
    float*                       rb_     = nullptr;
    std::vector<float>           rb_fallback_;

    // the slot map, host side, touched by the main thread under mu_
    std::vector<int16_t> slot_of_;     // n_layers * n_experts, -1 when not resident
    std::vector<int16_t> expert_at_;   // n_layers * capacity,  -1 when free
    std::vector<uint8_t> dirty_;       // n_layers * capacity,  1 when the slot needs an upload
    std::vector<int32_t> pending_;     // per layer, how many dirty slots are left
    int64_t              pending_total_ = 0;

    // the rendezvous
    std::thread             th_;
    std::mutex              mu_;
    std::condition_variable cv_job_, cv_done_, cv_idle_;
    bool  quit_       = false;
    bool  job_active_ = false;
    bool  job_done_   = false;
    bool  busy_       = false;   // the worker is inside a Vulkan call right now
    // Set when the worker died. Everything it owed is then reported as zeros rather than
    // waited for: a Vulkan exception on that thread would otherwise be a hang on this one,
    // and a hang is the one failure mode that says nothing at all about its cause.
    bool  failed_     = false;
    std::string fail_msg_;
    int   job_il_     = -1;
    int   job_k_      = 0;
    // When do_fork handed this job over. The CPU half runs between this instant and the
    // moment do_join is entered, so it is what the overlap had to work with.
    std::chrono::steady_clock::time_point fork_t_{};
    std::vector<int32_t> job_slots_;   // n_used, slot index per compacted entry
    std::vector<int32_t> job_at_;      // n_used, original router slot per compacted entry
    std::vector<float>   job_x_;       // n_embd
    std::vector<float>   res_;         // n_embd * n_used, already scattered and zero-filled
    std::vector<Site>    sites_;
};

// A whole-module check that needs no model file. It builds forty-eight layers of synthetic
// IQ4_XS expert tensors, warms a real ResidentSet on synthetic routing, packs the resident
// experts into video memory through the same path the engine uses, and then - per layer -
// computes the same feed-forward three ways: unsplit on the CPU, split with both halves on
// the CPU, and split with the resident half on the device. It exists because the 30B model
// needs seventeen gigabytes of free memory and a machine that is not otherwise busy, and
// because everything this class can get wrong - the slot map, the stacking, the compaction of
// the id list, the scatter back into slots, the exact zeros - is wrong in a way that survives
// a run whose tokens still look right.
//
// Returns 0 on success. Prints what it checked either way.
int gpu_experts_selftest(int threads);

}  // namespace memex
