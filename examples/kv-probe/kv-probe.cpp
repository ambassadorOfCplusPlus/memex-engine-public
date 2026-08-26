// Measure what the MemeX context memory actually costs on this machine, on both
// devices, at real context lengths.
//
// Why this is the next thing to measure. With experts split between GPU and CPU the
// projected token spends 27 ms on experts and 39 ms on everything else, so
// "everything else" became the larger half - and inside it the KV cache is the part
// that grows with context. For this model each context token costs 98 KB of KV
// reads across all layers (4 KV heads x 128 dims x 2 tensors x 2 bytes x 48
// layers), so 4k of context is already ~400 MB re-read on every generated token,
// and 32k would be 3.2 GB. At 20 GB/s that is 160 ms per token from attention
// alone: the context, not the weights, becomes the wall.
//
// The MemeX cache answers that with zones: [sinks | notebook | compressed tail |
// exact window]. Only the exact window and the notebook keep full-rank keys and
// values; the tail is kept as rank-r latents. Attention against the tail is then
// computed in the compressed space directly - the query is projected once,
// (W_up^T q), and multiplied by the latents - so the per-token read shrinks by
// d_kv/r instead of requiring any decompression.
//
// This program times both forms as ggml graphs, so the saving is measured rather
// than derived: full-rank attention against a real KV buffer, versus rank-r
// attention against latents of the same context length, on CPU and on Vulkan.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct Model {
    int n_head = 16;        // query heads
    int n_kv_head = 4;      // Qwen3-30B-A3B
    int head_dim = 128;
    int layers = 48;
};

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* out = nullptr;
    size_t kv_bytes = 0;

    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        alloc = nullptr; buf = nullptr; ctx = nullptr;
    }
};

// Full-rank attention for one layer: scores = K^T q, then V * softmax(scores).
// K and V are [d_kv, n_ctx] f16, which is exactly how a KV cache is laid out.
bool build_full(Graph* g, ggml_backend_buffer_type_t buft, const Model& m,
                int n_ctx, int reps) {
    const int d_kv = m.n_kv_head * m.head_dim;
    const size_t n_tensors = (size_t)reps * 8 + 64;
    ggml_init_params ip = {ggml_tensor_overhead() * n_tensors +
                           ggml_graph_overhead_custom(n_tensors, false),
                           nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) return false;

    std::vector<ggml_tensor*> K(reps), V(reps), Q(reps);
    for (int i = 0; i < reps; ++i) {
        K[i] = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, d_kv, n_ctx);
        // stored transposed on purpose: a per-token ggml_cont of the whole
        // cache would cost more than the attention itself
        V[i] = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, n_ctx, d_kv);
        Q[i] = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, d_kv, 1);
    }
    g->buf = ggml_backend_alloc_ctx_tensors_from_buft(g->ctx, buft);
    if (!g->buf) return false;
    g->kv_bytes = (size_t)reps * (ggml_nbytes(K[0]) + ggml_nbytes(V[0]));

    std::vector<uint8_t> junk(ggml_nbytes(K[0]));
    for (size_t i = 0; i < junk.size(); ++i) junk[i] = uint8_t(i * 31 + 7);
    std::vector<float> q((size_t)d_kv, 0.02f);
    for (int i = 0; i < reps; ++i) {
        ggml_backend_tensor_set(K[i], junk.data(), 0, ggml_nbytes(K[i]));
        ggml_backend_tensor_set(V[i], junk.data(), 0, ggml_nbytes(V[i]));
        ggml_backend_tensor_set(Q[i], q.data(), 0, ggml_nbytes(Q[i]));
    }

    g->gf = ggml_new_graph_custom(g->ctx, n_tensors, false);
    ggml_tensor* acc = nullptr;
    for (int i = 0; i < reps; ++i) {
        ggml_tensor* scores = ggml_mul_mat(g->ctx, K[i], Q[i]);      // [n_ctx, 1]
        ggml_tensor* p = ggml_soft_max(g->ctx, scores);
        ggml_tensor* o = ggml_mul_mat(g->ctx, V[i], p);          // [d_kv, 1]
        acc = acc ? ggml_add(g->ctx, acc, o) : o;
    }
    g->out = acc;
    ggml_build_forward_expand(g->gf, g->out);
    g->alloc = ggml_gallocr_new(buft);
    return ggml_gallocr_reserve(g->alloc, g->gf) &&
           ggml_gallocr_alloc_graph(g->alloc, g->gf);
}

// Compressed tail: the same attention, but keys and values are rank-r latents and
// the query has already been folded into that space, so nothing is decompressed.
// The output is lifted back once per layer by a [r, d_kv] up-projection, whose cost
// does not grow with context.
bool build_rank(Graph* g, ggml_backend_buffer_type_t buft, const Model& m,
                int n_ctx, int rank, int reps) {
    const int d_kv = m.n_kv_head * m.head_dim;
    const size_t n_tensors = (size_t)reps * 10 + 64;
    ggml_init_params ip = {ggml_tensor_overhead() * n_tensors +
                           ggml_graph_overhead_custom(n_tensors, false),
                           nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) return false;

    std::vector<ggml_tensor*> C(reps), Cv(reps), Qr(reps), Wup(reps);
    for (int i = 0; i < reps; ++i) {
        C[i] = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, rank, n_ctx);
        Cv[i] = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, n_ctx, rank);
        Qr[i] = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, rank, 1);
        Wup[i] = ggml_new_tensor_2d(g->ctx, GGML_TYPE_F16, rank, d_kv);
    }
    g->buf = ggml_backend_alloc_ctx_tensors_from_buft(g->ctx, buft);
    if (!g->buf) return false;
    g->kv_bytes = (size_t)reps * (ggml_nbytes(C[0]) + ggml_nbytes(Cv[0]));

    std::vector<uint8_t> junk(ggml_nbytes(C[0]));
    for (size_t i = 0; i < junk.size(); ++i) junk[i] = uint8_t(i * 17 + 3);
    std::vector<float> q((size_t)rank, 0.02f);
    for (int i = 0; i < reps; ++i) {
        ggml_backend_tensor_set(C[i], junk.data(), 0, ggml_nbytes(C[i]));
        ggml_backend_tensor_set(Cv[i], junk.data(), 0, ggml_nbytes(Cv[i]));
        ggml_backend_tensor_set(Qr[i], q.data(), 0, ggml_nbytes(Qr[i]));
        ggml_backend_tensor_set(Wup[i], junk.data(), 0, ggml_nbytes(Wup[i]));
    }

    g->gf = ggml_new_graph_custom(g->ctx, n_tensors, false);
    ggml_tensor* acc = nullptr;
    for (int i = 0; i < reps; ++i) {
        ggml_tensor* scores = ggml_mul_mat(g->ctx, C[i], Qr[i]);        // [n_ctx, 1]
        ggml_tensor* p = ggml_soft_max(g->ctx, scores);
        ggml_tensor* ctx_v = ggml_mul_mat(g->ctx, Cv[i], p);        // [rank, 1]
        // Wup is [rank, d_kv]: its first dimension already matches the latent, so
        // transposing it would ask for a d_kv-wide input and abort.
        ggml_tensor* o = ggml_mul_mat(g->ctx, Wup[i], ctx_v);
        acc = acc ? ggml_add(g->ctx, acc, o) : o;
    }
    g->out = acc;
    ggml_build_forward_expand(g->gf, g->out);
    g->alloc = ggml_gallocr_new(buft);
    return ggml_gallocr_reserve(g->alloc, g->gf) &&
           ggml_gallocr_alloc_graph(g->alloc, g->gf);
}

double time_it(ggml_backend_t be, Graph* g, int iters) {
    ggml_backend_graph_compute(be, g->gf);
    ggml_backend_synchronize(be);
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, g->gf);
    ggml_backend_synchronize(be);
    return ms_since(t0) / iters;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Model m;
    int iters = 8;
    int reps = 8;             // layers built at once; the rest is scaled linearly
    int rank = 128;
    std::vector<int> ctxs = {4096, 16384, 32768};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = atoi(argv[++i]);
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
#ifdef GGML_USE_VULKAN
    ggml_backend_t gpu = ggml_backend_vk_init(0);
#else
    ggml_backend_t gpu = nullptr;
#endif

    const int d_kv = m.n_kv_head * m.head_dim;
    printf("KV: %d kv-golov x %d = %d na sloi; %d sloev; rank %d (szhatie x%.1f)\n",
           m.n_kv_head, m.head_dim, d_kv, m.layers, rank, double(d_kv) / rank);
    printf("stroim %d sloev za raz, ostalnoe masshtabiruem\n\n", reps);
    printf("%8s %10s %12s %12s %12s %12s\n", "ctx", "KV vsego",
           "CPU polnyi", "CPU rank", "GPU polnyi", "GPU rank");

    for (int n_ctx : ctxs) {
        const double kv_gb = double(m.layers) * 2.0 * d_kv * n_ctx * 2.0 / 1e9;
        double res[4] = {0, 0, 0, 0};
        const double scale = double(m.layers) / reps;   // reps -> all layers
        {
            Graph g;
            if (build_full(&g, ggml_backend_cpu_buffer_type(), m, n_ctx, reps))
                res[0] = time_it(cpu, &g, iters) * scale;
            g.free_all();
        }
        {
            Graph g;
            if (build_rank(&g, ggml_backend_cpu_buffer_type(), m, n_ctx, rank, reps))
                res[1] = time_it(cpu, &g, iters) * scale;
            g.free_all();
        }
        if (gpu) {
            Graph g;
            if (build_full(&g, ggml_backend_vk_buffer_type(0), m, n_ctx, reps))
                res[2] = time_it(gpu, &g, iters) * scale;
            g.free_all();
        }
        if (gpu) {
            Graph g;
            if (build_rank(&g, ggml_backend_vk_buffer_type(0), m, n_ctx, rank, reps))
                res[3] = time_it(gpu, &g, iters) * scale;
            g.free_all();
        }
        printf("%8d %8.2f GB %9.1f ms %9.1f ms %9.1f ms %9.1f ms\n",
               n_ctx, kv_gb, res[0], res[1], res[2], res[3]);
    }

    printf("\nvremya - na odin sgenerirovannyi token, vse %d sloev\n", m.layers);
    printf("polnyi = KV kak est; rank = szhatyi hvost v prostranstve rangov\n");
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return 0;
}
