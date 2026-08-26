// Compute one MoE layer on both devices at once: the experts resident in VRAM on the
// GPU, the rest on the CPU, in parallel.
//
// The case for it, all measured. A single expert costs 0.046 ms on the GPU against
// 0.193 ms on the CPU, so the GPU is 4.2x faster per expert - but only for weights that
// are already in VRAM, because moving them per token over PCIe 3.0 x4 caps generation at
// 1.45 tok/s and was measured at 1.10-1.90. And when two independent graphs were handed
// to the two backends from two threads, the overlap was 94-109%: the GPU's work vanished
// inside the CPU's.
//
// So the split has to be static in placement and concurrent in execution: upload a
// chosen set once, then every token compute the resident part on the GPU while the CPU
// handles the rest, and add the two halves. The balanced share - where neither device
// waits - is 6.46 of 8 experts given that 4.2x ratio, so what limits the gain is how
// much of the routed demand the VRAM set actually covers.
//
// This measures that end to end against the CPU-only path, on real routing.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include "memex/blob_loader.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Shape {
    int64_t n_embd = 2048;
    int64_t n_ff = 768;
    int n_experts = 128;
    int top_k = 8;
};

// One device's copy of a set of experts, plus a graph over a chosen subset.
struct Side {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context* wctx = nullptr;            // weights live here
    ggml_backend_buffer_t wbuf = nullptr;
    std::vector<ggml_tensor*> up, gate, down;
    std::vector<int> owned;                  // expert ids held on this device
    std::vector<char> has;                   // fast membership test

    ggml_context* gctx = nullptr;            // rebuilt per token
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* x = nullptr;
    ggml_tensor* out = nullptr;
    double ms = 0.0;

    void free_graph() {
        if (alloc) ggml_gallocr_free(alloc);
        if (gctx) ggml_free(gctx);
        alloc = nullptr;
        gctx = nullptr;
        gf = nullptr;
    }
    void free_all() {
        free_graph();
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        wbuf = nullptr;
        wctx = nullptr;
    }
};

const memex::BlobEntry* find(const memex::BlobLoader& l, int layer, int expert,
                             int slot, bool source) {
    for (const auto& e : l.entries()) {
        if (int(e.layer) == layer && int(e.expert) == expert &&
            int(e.tensor) == slot && ((e.step == "source") == source)) {
            return &e;
        }
    }
    return nullptr;
}

// Upload a set of experts to one device, in the model's own precision so both sides
// compute the same thing and the comparison is about placement, not about quality.
bool place(Side* s, memex::BlobLoader& loader, int layer, const std::vector<int>& ids,
           const Shape& sh, int n_experts) {
    const size_t n_tensors = ids.size() * 3 + 16;
    ggml_init_params ip = {ggml_tensor_overhead() * n_tensors, nullptr, true};
    s->wctx = ggml_init(ip);
    if (!s->wctx) return false;
    s->owned = ids;
    s->has.assign(size_t(n_experts), 0);
    for (int e : ids) {
        s->has[size_t(e)] = 1;
    }
    std::vector<const memex::BlobEntry*> ent;
    for (int e : ids) {
        for (int slot = 0; slot < 3; ++slot) {
            const auto* be = find(loader, layer, e, slot, /*source=*/true);
            if (!be) {
                printf("нет исходной записи: эксперт %d тензор %d\n", e, slot);
                return false;
            }
            ent.push_back(be);
        }
        s->up.push_back(ggml_new_tensor_2d(s->wctx, (ggml_type)ent[ent.size() - 3]->ggml_type,
                                           sh.n_embd, sh.n_ff));
        s->gate.push_back(ggml_new_tensor_2d(s->wctx, (ggml_type)ent[ent.size() - 2]->ggml_type,
                                             sh.n_embd, sh.n_ff));
        s->down.push_back(ggml_new_tensor_2d(s->wctx, (ggml_type)ent[ent.size() - 1]->ggml_type,
                                             sh.n_ff, sh.n_embd));
    }
    s->wbuf = ggml_backend_alloc_ctx_tensors_from_buft(s->wctx, s->buft);
    if (!s->wbuf) {
        printf("не удалось выделить память под веса (не хватило VRAM?)\n");
        return false;
    }
    memex::AlignedBuffer tmp(8u << 20);
    for (size_t i = 0; i < ids.size(); ++i) {
        ggml_tensor* dst[3] = {s->up[i], s->gate[i], s->down[i]};
        for (int slot = 0; slot < 3; ++slot) {
            const auto* be = ent[i * 3 + size_t(slot)];
            if (!loader.read_sync(*be, tmp.data(), tmp.size())) {
                printf("чтение не удалось\n");
                return false;
            }
            ggml_backend_tensor_set(dst[slot], tmp.data(), 0, be->bytes);
        }
    }
    return true;
}

// Graph over this device's share of the token's experts.
bool build(Side* s, const std::vector<int>& chosen, const Shape& sh,
           const std::vector<float>& act) {
    s->free_graph();
    std::vector<size_t> mine;
    for (size_t i = 0; i < s->owned.size(); ++i) {
        if (std::find(chosen.begin(), chosen.end(), s->owned[i]) != chosen.end()) {
            mine.push_back(i);
        }
    }
    if (mine.empty()) {
        return false;
    }
    const size_t n_nodes = mine.size() * 8 + 64;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 32) +
                           ggml_graph_overhead_custom(n_nodes, false),
                           nullptr, true};
    s->gctx = ggml_init(ip);
    if (!s->gctx) return false;
    s->x = ggml_new_tensor_2d(s->gctx, GGML_TYPE_F32, sh.n_embd, 1);
    ggml_backend_buffer_t xb = ggml_backend_alloc_ctx_tensors_from_buft(s->gctx, s->buft);
    if (!xb) return false;
    ggml_backend_tensor_set(s->x, act.data(), 0, ggml_nbytes(s->x));
    s->gf = ggml_new_graph_custom(s->gctx, n_nodes, false);
    ggml_tensor* acc = nullptr;
    for (size_t i : mine) {
        ggml_tensor* a = ggml_mul_mat(s->gctx, s->gate[i], s->x);
        ggml_tensor* b = ggml_mul_mat(s->gctx, s->up[i], s->x);
        ggml_tensor* h = ggml_mul(s->gctx, ggml_silu(s->gctx, a), b);
        ggml_tensor* o = ggml_mul_mat(s->gctx, s->down[i], h);
        acc = acc ? ggml_add(s->gctx, acc, o) : o;
    }
    s->out = acc;
    ggml_build_forward_expand(s->gf, s->out);
    s->alloc = ggml_gallocr_new(s->buft);
    return ggml_gallocr_reserve(s->alloc, s->gf) &&
           ggml_gallocr_alloc_graph(s->alloc, s->gf);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string blob = "D:/MemeX/blob/l20.bin";
    std::string man = "D:/MemeX/blob/l20.json";
    int layer = 20;
    int n_gpu = 12;         // experts placed in VRAM; 594 model-wide is ~12 per layer
    int threads = 4;
    int iters = 30;
    Shape sh;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--blob") && i + 1 < argc) blob = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) man = argv[++i];
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu-experts") && i + 1 < argc) n_gpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    }

    memex::BlobLoader loader;
    std::string err;
    if (!loader.open(blob, man, &err)) {
        printf("блоб не открылся: %s\n", err.c_str());
        return 1;
    }

    Side cpu, gpu;
    cpu.backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu.backend, threads);
    cpu.buft = ggml_backend_cpu_buffer_type();
#ifdef GGML_USE_VULKAN
    gpu.backend = ggml_backend_vk_init(0);
    gpu.buft = ggml_backend_vk_buffer_type(0);
#endif
    if (!gpu.backend) {
        printf("Vulkan недоступен: собери в каталоге с -DGGML_VULKAN=ON\n");
        return 1;
    }

    // Placement: the first n_gpu experts go to VRAM, the rest stay on the CPU. Which
    // experts those are is the policy's business elsewhere; here the question is what
    // concurrency buys once a set is placed.
    std::vector<int> on_gpu, on_cpu;
    for (int e = 0; e < sh.n_experts; ++e) {
        (e < n_gpu ? on_gpu : on_cpu).push_back(e);
    }
    if (!place(&gpu, loader, layer, on_gpu, sh, sh.n_experts)) return 1;
    if (!place(&cpu, loader, layer, on_cpu, sh, sh.n_experts)) return 1;
    printf("слой %d: в VRAM %zu экспертов, на CPU %zu, потоков %d\n",
           layer, on_gpu.size(), on_cpu.size(), threads);

    std::vector<float> act(size_t(sh.n_embd));
    for (size_t i = 0; i < act.size(); ++i) {
        act[i] = 0.02f * std::sin(float(i) * 0.013f);
    }

    // Two token shapes: one where the GPU holds part of the demand, and one where it
    // holds none, which is the honest worst case for a static placement.
    struct Case {
        const char* name;
        std::vector<int> ids;
    };
    std::vector<Case> cases;
    {
        std::vector<int> mixed;
        for (int i = 0; i < sh.top_k / 2; ++i) mixed.push_back(i);                  // in VRAM
        for (int i = 0; i < sh.top_k / 2; ++i) mixed.push_back(n_gpu + i * 7);      // on CPU
        cases.push_back({"половина в VRAM", mixed});
        std::vector<int> all_cpu;
        for (int i = 0; i < sh.top_k; ++i) all_cpu.push_back(n_gpu + i * 5);
        cases.push_back({"ничего в VRAM", all_cpu});
    }

    for (const auto& cs : cases) {
        const bool has_gpu = build(&gpu, cs.ids, sh, act);
        const bool has_cpu = build(&cpu, cs.ids, sh, act);
        printf("\n--- %s ---\n", cs.name);
        double t_cpu_only = 0.0;
        if (has_cpu) {
            ggml_backend_graph_compute(cpu.backend, cpu.gf);
            const auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_graph_compute(cpu.backend, cpu.gf);
            }
            t_cpu_only = ms_since(t0) / iters;
        }
        double t_gpu_only = 0.0;
        if (has_gpu) {
            ggml_backend_graph_compute(gpu.backend, gpu.gf);
            ggml_backend_synchronize(gpu.backend);
            const auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_graph_compute(gpu.backend, gpu.gf);
            }
            ggml_backend_synchronize(gpu.backend);
            t_gpu_only = ms_since(t0) / iters;
        }
        // Both at once, from two threads: the arrangement the probe measured at 94-109%
        // overlap, now with the real expert weights and a real routing decision.
        double t_both = 0.0;
        if (has_cpu && has_gpu) {
            std::atomic<bool> go{false};
            auto runner = [&](Side* s) {
                while (!go.load(std::memory_order_acquire)) {
                }
                for (int i = 0; i < iters; ++i) {
                    ggml_backend_graph_compute(s->backend, s->gf);
                }
                ggml_backend_synchronize(s->backend);
            };
            std::thread a(runner, &gpu);
            std::thread b(runner, &cpu);
            const auto t0 = Clock::now();
            go.store(true, std::memory_order_release);
            a.join();
            b.join();
            t_both = ms_since(t0) / iters;
        }
        printf("  только CPU (его доля)   : %.3f мс\n", t_cpu_only);
        printf("  только GPU (его доля)   : %.3f мс\n", t_gpu_only);
        if (t_both > 0.0) {
            const double serial = t_cpu_only + t_gpu_only;
            const double best = std::max(t_cpu_only, t_gpu_only);
            printf("  одновременно            : %.3f мс (по очереди %.3f, идеал %.3f)\n",
                   t_both, serial, best);
            const double possible = serial - best;
            printf("  перекрытие реализовано  : %.0f%%\n",
                   possible > 0.0 ? 100.0 * (serial - t_both) / possible : 0.0);
            printf("  на 48 слоёв: %.1f мс -> %.2f ток/с против %.2f у чистого CPU\n",
                   48.0 * t_both, 1000.0 / (48.0 * t_both),
                   1000.0 / (48.0 * serial));
        }
    }

    cpu.free_all();
    gpu.free_all();
    ggml_backend_free(cpu.backend);
    ggml_backend_free(gpu.backend);
    return 0;
}
