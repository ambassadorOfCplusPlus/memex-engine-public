// Measure the three numbers the MemeX scheduler is being designed around, none of
// which can be taken from a datasheet.
//
// The design under test: hot experts stay resident in VRAM and are computed by the
// GPU, cold experts are computed by the CPU from RAM, and the two run at the same
// time. Weights never cross the bus - only activations do, 8 KB per layer. A
// replay of the model's own routing trace says this should roughly double
// throughput, but the whole projection rests on assumptions this program checks:
//
//   1. crossing cost - what one activation hop actually costs. The model assumed
//      30 us. An indirect estimate from llama.cpp's own hybrid split implied 67 ms
//      per hop, which would make the design worthless; that number is almost
//      certainly an artefact of its scheduler rather than the bus, and this
//      settles which it is.
//   2. resident-weight throughput - whether the GPU really reads VRAM at a speed
//      that makes its share of the experts nearly free.
//   3. overlap - whether CPU and GPU work truly proceeds in parallel. If wall time
//      for both together lands near max(cpu, gpu) the design holds; if it lands
//      near their sum, there is nothing to win and the scheduler must be built
//      differently.
//
// Shapes follow Qwen3-30B-A3B: hidden 2048, expert FFN 768, three matrices per
// expert, eight experts per layer, 48 layers.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Shapes {
    int hidden = 2048;
    int ff = 768;
    int experts_per_layer = 8;
    int layers = 48;
    // Chosen at run time, because it decides the whole design: a type the card handles
    // badly means the resident experts must be stored in something bulkier, and residency
    // falls in proportion.
    ggml_type wtype = GGML_TYPE_Q6_K;
};

// One "expert workload": three matrix-vector products of the shape an expert has.
// Built as a standalone graph so it can be handed to any backend.
struct Workload {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t weights = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* x = nullptr;
    ggml_tensor* out = nullptr;
    std::vector<uint8_t> scratch;

    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (weights) ggml_backend_buffer_free(weights);
        if (ctx) ggml_free(ctx);
        alloc = nullptr;
        weights = nullptr;
        ctx = nullptr;
    }
};

// n_experts three-matrix blocks, all reading distinct weights, summed at the end.
// Weights live in `buft`, which is what makes the difference between "resident in
// VRAM" and "read from RAM".
bool build_workload(Workload* w, ggml_backend_t backend,
                    ggml_backend_buffer_type_t buft, const Shapes& sh,
                    int n_experts, int n_tokens) {
    // Each expert contributes three weight tensors plus five intermediates, and
    // the graph needs a node per intermediate; the defaults (2048 nodes) are far
    // too small once the working set is sized to defeat the CPU cache.
    const size_t n_nodes = (size_t)n_experts * 8 + 64;
    // three weights and five intermediates per expert, all needing a descriptor:
    // undercounting here aborts inside ggml with the pool exhausted
    const size_t n_tensors = (size_t)n_experts * 12 + 128;
    const size_t overhead = ggml_tensor_overhead() * n_tensors +
                            ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params ip = {overhead, nullptr, true};   // no_alloc: tensors only
    w->ctx = ggml_init(ip);
    if (!w->ctx) return false;

    std::vector<ggml_tensor*> up(n_experts), gate(n_experts), down(n_experts);
    for (int e = 0; e < n_experts; ++e) {
        up[e] = ggml_new_tensor_2d(w->ctx, sh.wtype, sh.hidden, sh.ff);
        gate[e] = ggml_new_tensor_2d(w->ctx, sh.wtype, sh.hidden, sh.ff);
        down[e] = ggml_new_tensor_2d(w->ctx, sh.wtype, sh.ff, sh.hidden);
    }
    w->x = ggml_new_tensor_2d(w->ctx, GGML_TYPE_F32, sh.hidden, n_tokens);
    ggml_set_name(w->x, "x");

    // Weights and the input live in the backend's own buffer; the input is the
    // only thing that ever gets written from the host.
    w->weights = ggml_backend_alloc_ctx_tensors_from_buft(w->ctx, buft);
    if (!w->weights) return false;

    // Random-ish bytes: values do not matter for timing, but zeroed pages could
    // let an allocator or driver skip real work.
    const size_t need = ggml_nbytes(up[0]);
    w->scratch.resize(need);
    for (size_t i = 0; i < need; ++i) w->scratch[i] = uint8_t(i * 37 + 11);
    for (int e = 0; e < n_experts; ++e) {
        ggml_backend_tensor_set(up[e], w->scratch.data(), 0, ggml_nbytes(up[e]));
        ggml_backend_tensor_set(gate[e], w->scratch.data(), 0, ggml_nbytes(gate[e]));
        ggml_backend_tensor_set(down[e], w->scratch.data(), 0, ggml_nbytes(down[e]));
    }
    std::vector<float> xv((size_t)sh.hidden * n_tokens, 0.01f);
    ggml_backend_tensor_set(w->x, xv.data(), 0, ggml_nbytes(w->x));

    // Graph: for each expert, silu(x*gate) * (x*up) -> down, then accumulate.
    ggml_init_params gp = {overhead, nullptr, true};
    ggml_context* gctx = w->ctx;   // same context is fine, tensors are pre-alloc'd
    (void)gp;
    w->graph = ggml_new_graph_custom(gctx, n_nodes, false);
    ggml_tensor* acc = nullptr;
    for (int e = 0; e < n_experts; ++e) {
        ggml_tensor* a = ggml_mul_mat(gctx, gate[e], w->x);
        ggml_tensor* b = ggml_mul_mat(gctx, up[e], w->x);
        ggml_tensor* h = ggml_mul(gctx, ggml_silu(gctx, a), b);
        ggml_tensor* o = ggml_mul_mat(gctx, down[e], h);
        acc = acc ? ggml_add(gctx, acc, o) : o;
    }
    w->out = acc;
    ggml_build_forward_expand(w->graph, w->out);

    w->alloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_reserve(w->alloc, w->graph)) return false;
    if (!ggml_gallocr_alloc_graph(w->alloc, w->graph)) return false;
    GGML_UNUSED(backend);
    return true;
}

double time_graph(ggml_backend_t backend, Workload* w, int iters) {
    ggml_backend_graph_compute(backend, w->graph);   // warm up
    ggml_backend_synchronize(backend);
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, w->graph);
    }
    ggml_backend_synchronize(backend);
    return ms_since(t0) / iters;
}

}  // namespace

int main(int argc, char** argv) {
    // ggml aborts on a failed assert, and an abort discards buffered stdout - so
    // an unbuffered stream is the difference between seeing the failure and seeing
    // an empty screen.
    setvbuf(stdout, nullptr, _IONBF, 0);
    Shapes sh;
    int iters = 20;
    int gpu_experts = 5;      // 64% of 8, the measured resident share
    int cpu_experts = 3;
    int n_tokens = 1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu-experts") && i + 1 < argc) gpu_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu-experts") && i + 1 < argc) cpu_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) n_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wtype") && i + 1 < argc) {
            const char* t = argv[++i];
            if (!strcmp(t, "q6_k")) sh.wtype = GGML_TYPE_Q6_K;
            else if (!strcmp(t, "iq4_xs")) sh.wtype = GGML_TYPE_IQ4_XS;
            else if (!strcmp(t, "q4_k")) sh.wtype = GGML_TYPE_Q4_K;
            else if (!strcmp(t, "f16")) sh.wtype = GGML_TYPE_F16;
        }
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
#ifdef GGML_USE_VULKAN
    ggml_backend_t gpu = ggml_backend_vk_init(0);
#else
    ggml_backend_t gpu = nullptr;
#endif
    if (!gpu) {
        printf("Vulkan-бэкенд недоступен — собери с -DGGML_VULKAN=ON\n");
        return 1;
    }
#ifdef GGML_USE_VULKAN
    {
        char desc[256] = {0};
        ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
        size_t freeb = 0, total = 0;
        ggml_backend_vk_get_device_memory(0, &freeb, &total);
        printf("устройство: %s, свободно %.0f МБ из %.0f\n", desc,
               freeb / 1e6, total / 1e6);
    }
#endif

    const double expert_mb = double(3 * sh.hidden * sh.ff) * 0.82 / 1e6;
    printf("эксперт ~%.2f МБ; на GPU %d, на CPU %d, токенов %d, повторов %d\n\n",
           expert_mb, gpu_experts, cpu_experts, n_tokens, iters);

    // 1. crossing cost: write the activation, run a trivial graph, read it back
    {
        Workload tiny;
        if (!build_workload(&tiny, gpu, ggml_backend_vk_buffer_type(0), sh, 1, n_tokens)) {
            printf("не удалось собрать пробный граф\n");
            return 1;
        }
        std::vector<float> host((size_t)sh.hidden * n_tokens, 0.5f);
        std::vector<float> back(ggml_nelements(tiny.out));
        ggml_backend_graph_compute(gpu, tiny.graph);
        ggml_backend_synchronize(gpu);
        auto t0 = Clock::now();
        const int n = 200;
        for (int i = 0; i < n; ++i) {
            ggml_backend_tensor_set(tiny.x, host.data(), 0, ggml_nbytes(tiny.x));
            ggml_backend_graph_compute(gpu, tiny.graph);
            ggml_backend_tensor_get(tiny.out, back.data(), 0,
                                    ggml_nbytes(tiny.out) < back.size() * 4
                                        ? ggml_nbytes(tiny.out)
                                        : back.size() * 4);
        }
        ggml_backend_synchronize(gpu);
        const double per = ms_since(t0) / n;
        printf("1) переход туда-обратно + один эксперт на GPU: %.3f мс\n", per);
        printf("   (в модели планировщика заложено 0.030 мс на переход)\n");
        tiny.free_all();
    }

    // 2. resident-weight throughput on each side, measured separately
    Workload wg, wc;
    if (!build_workload(&wg, gpu, ggml_backend_vk_buffer_type(0), sh, gpu_experts, n_tokens)) {
        printf("GPU-граф не собрался (не хватило VRAM?)\n");
        return 1;
    }
    if (!build_workload(&wc, cpu, ggml_backend_cpu_buffer_type(), sh, cpu_experts, n_tokens)) {
        printf("CPU-граф не собрался\n");
        return 1;
    }
    const double gpu_ms = time_graph(gpu, &wg, iters);
    const double cpu_ms = time_graph(cpu, &wc, iters);
    printf("2) %d экспертов на GPU из VRAM: %.2f мс  (%.1f ГБ/с)\n",
           gpu_experts, gpu_ms, gpu_experts * expert_mb / gpu_ms);
    printf("   %d экспертов на CPU из RAM : %.2f мс  (%.1f ГБ/с)\n",
           cpu_experts, cpu_ms, cpu_experts * expert_mb / cpu_ms);

    // 3. the decisive one: both at once, from two threads
    {
        std::atomic<bool> go{false};
        double t_gpu = 0.0, t_cpu = 0.0;
        auto runner = [&](ggml_backend_t be, Workload* w, double* out) {
            while (!go.load(std::memory_order_acquire)) {
            }
            auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) {
                ggml_backend_graph_compute(be, w->graph);
            }
            ggml_backend_synchronize(be);
            *out = ms_since(t0) / iters;
        };
        std::thread a(runner, gpu, &wg, &t_gpu);
        std::thread b(runner, cpu, &wc, &t_cpu);
        auto t0 = Clock::now();
        go.store(true, std::memory_order_release);
        a.join();
        b.join();
        const double wall = ms_since(t0) / iters;
        const double serial = gpu_ms + cpu_ms;
        const double best = gpu_ms > cpu_ms ? gpu_ms : cpu_ms;
        printf("3) одновременно: %.2f мс на итерацию\n", wall);
        printf("   по очереди было бы %.2f, идеальное перекрытие дало бы %.2f\n",
               serial, best);
        const double gained = serial - wall;
        const double possible = serial - best;
        printf("   перекрытие реализовано на %.0f%% "
               "(GPU в этом режиме %.2f мс, CPU %.2f мс)\n",
               possible > 0 ? 100.0 * gained / possible : 0.0, t_gpu, t_cpu);
    }

    // Project onto the real model: the per-expert costs are what transfers, the
    // expert counts are the model's. Everything that is not expert weights -
    // attention, shared FFN, router, kernel overhead - is backed out of the
    // measured all-CPU generation rate rather than guessed.
    const double gpu_per_expert = gpu_ms / gpu_experts;
    const double cpu_per_expert = cpu_ms / cpu_experts;
    const double measured_tok_s = 8.85;
    const double all_cpu_ms = sh.layers * sh.experts_per_layer * cpu_per_expert;
    const double dense_ms = 1000.0 / measured_tok_s - all_cpu_ms;
    const double cross_ms = sh.layers * 0.20 * 0.5;  // half the hops pipelined away
    printf("\n=== pereschet na model: %d sloev, %d ekspertov na sloi ===\n",
           sh.layers, sh.experts_per_layer);
    printf("  odin ekspert: GPU %.3f ms, CPU %.3f ms -> GPU bystree v %.2f raza\n",
           gpu_per_expert, cpu_per_expert, cpu_per_expert / gpu_per_expert);
    printf("  vse na CPU: eksperty %.0f ms + ostalnoe %.0f ms = %.2f tok/s\n",
           all_cpu_ms, dense_ms, measured_tok_s);
    // The best split balances the workers instead of following popularity: give the
    // GPU exactly what it can finish while the CPU handles the rest.
    const double bal = sh.experts_per_layer * cpu_per_expert /
                       (cpu_per_expert + gpu_per_expert);
    printf("  sbalansirovannaya dolya GPU: %.2f iz %d (%.0f%%)\n",
           bal, sh.experts_per_layer, 100.0 * bal / sh.experts_per_layer);
    const double shares[3] = {0.64, 0.71, bal / sh.experts_per_layer};
    const char* what[3] = {"izmereno: smeshannyi tekst", "izmereno: proza",
                           "balans rabotnikov"};
    for (int i = 0; i < 3; ++i) {
        const double on_gpu = sh.experts_per_layer * shares[i];
        const double on_cpu = sh.experts_per_layer - on_gpu;
        const double t = dense_ms + cross_ms +
                sh.layers * std::max(on_gpu * gpu_per_expert,
                                     on_cpu * cpu_per_expert);
        printf("  dolya v VRAM %3.0f%% (%s): %.0f ms -> %.2f tok/s, x%.2f\n",
               100.0 * shares[i], what[i], t, 1000.0 / t,
               (1000.0 / t) / measured_tok_s);
    }
    wg.free_all();
    wc.free_all();
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return 0;
}
