// What one more node in a Vulkan graph costs.
//
// THE QUESTION. The card plan is priced in dispatches: fifteen device-side nodes per layer over
// forty-eight layers is 720 dispatches per token, and whether that is 3.6 ms or 22 ms decides
// whether moving weights to the card is the first job or whether fusing the graph is.
//
// The number this project had - 31 us - is the LATENCY of one kernel, launch to completion. That
// is a round trip: submit, the GPU picks the work up, it runs, a fence signals, the host wakes.
// It is the right number for a graph submitted one node at a time and the wrong number for a
// graph submitted whole, because dispatches inside one command buffer pipeline: the host records
// them while the GPU is still executing earlier ones, and the recording is what the marginal node
// actually costs. Multiplying a round trip by the node count is therefore an upper bound that can
// be several times too high, and the plan is decided by the difference.
//
// SO MEASURE THE SLOPE, NOT A POINT. Time a graph of N identical tiny nodes for N over a sweep,
// and fit. The slope is the marginal cost of a node. The intercept is what a submit-and-fence
// costs no matter how little is in it, and that has an independent value to check against - the
// 177 us rendezvous this project measured elsewhere. If the intercept comes out far from it, one
// of the two numbers is wrong and that is worth knowing before either is trusted again.
//
// WHAT IS BEING TIMED, EXACTLY. ggml_backend_graph_compute on the Vulkan backend, which is what
// the module would call. That includes the host-side cost of building the command buffer - the
// pipeline lookup and descriptor writes ggml_vk_build_graph does per node - because the plan pays
// that too. It is not a raw vkCmdDispatch microbenchmark and is not meant to be; it is the cost
// of the thing we would actually do.
//
// THE KERNELS ARE DELIBERATELY TINY. 1024 floats, 4 KiB, chained so nothing can be eliminated as
// dead and nothing can be fused into its neighbour. At that size the arithmetic and the bandwidth
// are both far below the launch cost, which is the only way the slope means what it says. A chain
// rather than independent nodes because independent ones could in principle be reordered or
// merged, and because the real graph is a chain.
//
// ONE THING TO KNOW WHEN READING THE RESULT. ggml's Vulkan backend does not put an unbounded graph
// into one command buffer: ggml-vulkan.cpp:10353 submits every 100 nodes, and again when fewer
// than 20% of the nodes remain (the "almost ready" fence). So a sweep that crosses 100 nodes
// crosses a submit boundary, and the fit is reported over the sub-100 range as well as over the
// whole sweep. The sub-100 slope is the one the plan needs, since a per-layer graph is fifteen
// nodes and not a hundred.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

namespace {

using Clock = std::chrono::steady_clock;

double us_since(const Clock::time_point& t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}

// One measurement: a chain of `n_nodes` adds over `n_elem` floats, computed `reps` times.
// Returns microseconds per graph_compute call.
double time_graph(ggml_backend_t be, ggml_backend_buffer_type_t buft, int n_nodes, int n_elem,
                  int reps, int* built_nodes) {
    // Two persistent inputs in their own buffer, so the graph allocator only has to place the
    // intermediates. Both are written once; nothing is transferred inside the timed loop.
    ggml_init_params ip_in = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ggml_context* c_in = ggml_init(ip_in);
    ggml_tensor* x = ggml_new_tensor_1d(c_in, GGML_TYPE_F32, n_elem);
    ggml_tensor* k = ggml_new_tensor_1d(c_in, GGML_TYPE_F32, n_elem);
    ggml_set_name(x, "x");
    ggml_set_name(k, "k");
    ggml_backend_buffer_t b_in = ggml_backend_alloc_ctx_tensors_from_buft(c_in, buft);
    if (!b_in) { ggml_free(c_in); return -1.0; }
    {
        std::vector<float> h(std::size_t(n_elem));
        for (int i = 0; i < n_elem; ++i) h[std::size_t(i)] = 1.0f / float(i + 1);
        ggml_backend_tensor_set(x, h.data(), 0, sizeof(float) * std::size_t(n_elem));
        for (int i = 0; i < n_elem; ++i) h[std::size_t(i)] = 1e-6f;
        ggml_backend_tensor_set(k, h.data(), 0, sizeof(float) * std::size_t(n_elem));
    }

    // The chain. Each node depends on the previous one, so the backend cannot reorder them and
    // no node is dead. ggml_add on f32 has no fusion partner in ggml_vk_can_fuse (which pairs
    // only RMS_NORM with MUL), so the node count is the dispatch count.
    ggml_init_params ip = {ggml_tensor_overhead() * std::size_t(n_nodes + 8) + ggml_graph_overhead(),
                           nullptr, true};
    ggml_context* c = ggml_init(ip);
    if (!c) { ggml_backend_buffer_free(b_in); ggml_free(c_in); return -1.0; }
    ggml_cgraph* gf = ggml_new_graph(c);
    ggml_tensor* cur = x;
    for (int i = 0; i < n_nodes; ++i) {
        cur = ggml_add(c, cur, k);
    }
    ggml_build_forward_expand(gf, cur);
    if (built_nodes) *built_nodes = ggml_graph_n_nodes(gf);

    ggml_gallocr_t ga = ggml_gallocr_new(buft);
    if (!ga || !ggml_gallocr_reserve(ga, gf) || !ggml_gallocr_alloc_graph(ga, gf)) {
        if (ga) ggml_gallocr_free(ga);
        ggml_free(c);
        ggml_backend_buffer_free(b_in);
        ggml_free(c_in);
        return -1.0;
    }

    // Warm up: the first call compiles pipelines and allocates descriptor pools, which is a
    // one-off of a completely different magnitude and would otherwise land entirely in the
    // intercept.
    for (int i = 0; i < 8; ++i) ggml_backend_graph_compute(be, gf);

    const auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(be, gf);
    const double total = us_since(t0);

    ggml_gallocr_free(ga);
    ggml_free(c);
    ggml_backend_buffer_free(b_in);
    ggml_free(c_in);
    return total / double(reps);
}

// Ordinary least squares of y on x. Returns slope and intercept.
void fit(const std::vector<double>& xs, const std::vector<double>& ys, double* slope,
         double* intercept) {
    const double n = double(xs.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i];
    }
    const double den = n * sxx - sx * sx;
    *slope     = den != 0.0 ? (n * sxy - sx * sy) / den : 0.0;
    *intercept = den != 0.0 ? (sy - *slope * sx) / n : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    int  n_elem = 1024;          // 4 KiB per tensor
    int  reps   = 300;
    int  replicates = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--elems" && i + 1 < argc) n_elem = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--replicates" && i + 1 < argc) replicates = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            printf("llama-memex-vkdisp — сколько стоит ещё один узел в графе Vulkan.\n"
                   "  --elems N       элементов f32 в тензоре (%d, это %.1f КиБ)\n"
                   "  --reps N        вызовов graph_compute на точку (%d)\n"
                   "  --replicates N  повторов всего свипа (%d)\n",
                   n_elem, double(n_elem) * 4.0 / 1024.0, reps, replicates);
            return 0;
        }
    }

    ggml_backend_t be = ggml_backend_vk_init(0);
    if (!be) { printf("ggml_backend_vk_init(0) не удался — нет устройства Vulkan\n"); return 1; }
    ggml_backend_buffer_type_t buft = ggml_backend_vk_buffer_type(0);
    if (!buft) { printf("ggml_backend_vk_buffer_type(0) не удался\n"); ggml_backend_free(be); return 1; }
    {
        char d[256] = {0};
        ggml_backend_vk_get_device_description(0, d, sizeof(d));
        printf("устройство: %s\n", d);
    }
    printf("ядро: цепочка ggml_add по %d f32 (%.1f КиБ на тензор) — намеренно меньше того,\n"
           "  что упирается в пропускную способность, чтобы измерялся запуск, а не байты\n",
           n_elem, double(n_elem) * 4.0 / 1024.0);
    printf("на точку: %d вызовов ggml_backend_graph_compute, повторов свипа %d\n\n",
           reps, replicates);

    const std::vector<int> ns = {1, 2, 4, 8, 16, 32, 64, 128};

    // [replicate][point]
    std::vector<std::vector<double>> runs(std::size_t(replicates));
    for (int r = 0; r < replicates; ++r) {
        runs[std::size_t(r)].reserve(ns.size());
        for (int n : ns) {
            int built = 0;
            const double us = time_graph(be, buft, n, n_elem, reps, &built);
            if (us < 0.0) {
                printf("не удалось построить или разместить граф на %d узлов\n", n);
                ggml_backend_free(be);
                return 1;
            }
            if (built != n) {
                // Says so rather than silently fitting against the wrong x. If ggml ever fuses
                // or drops one of these, the node count is not the dispatch count and the whole
                // measurement means something else.
                printf("ВНИМАНИЕ: просили %d узлов, граф содержит %d — ось X не та\n", n, built);
            }
            runs[std::size_t(r)].push_back(us);
        }
    }

    // Per point: median across replicates, and the spread, which is what says whether a number
    // is a result at all. The noise floor on this machine is 4.2%.
    printf("  %6s %12s %12s %12s %10s\n", "узлов", "медиана,мкс", "мин,мкс", "макс,мкс", "разброс");
    std::vector<double> xs, ys, xs_lo, ys_lo;
    double worst_spread = 0.0;
    for (std::size_t i = 0; i < ns.size(); ++i) {
        std::vector<double> v;
        for (int r = 0; r < replicates; ++r) v.push_back(runs[std::size_t(r)][i]);
        std::sort(v.begin(), v.end());
        const double med = v[v.size() / 2];
        const double lo = v.front(), hi = v.back();
        const double spread = med > 0.0 ? (hi - lo) / med : 0.0;
        worst_spread = std::max(worst_spread, spread);
        printf("  %6d %12.2f %12.2f %12.2f %9.1f%%\n", ns[i], med, lo, hi, 100.0 * spread);
        xs.push_back(double(ns[i]));
        ys.push_back(med);
        // ggml submits every 100 nodes (ggml-vulkan.cpp:10353), so 128 sits past a submit
        // boundary. The plan's graph is fifteen nodes, so the sub-100 fit is the relevant one.
        if (ns[i] < 100) { xs_lo.push_back(double(ns[i])); ys_lo.push_back(med); }
    }

    double s_all = 0, b_all = 0, s_lo = 0, b_lo = 0;
    fit(xs, ys, &s_all, &b_all);
    fit(xs_lo, ys_lo, &s_lo, &b_lo);

    printf("\n  подгонка по всему свипу (1..128):  наклон %.2f мкс/узел, свободный член %.1f мкс\n",
           s_all, b_all);
    printf("  подгонка ниже границы submit (1..64): наклон %.2f мкс/узел, свободный член %.1f мкс\n",
           s_lo, b_lo);
    printf("  худший разброс между повторами: %.1f%% (шумовой порог проекта 4.2%%)%s\n",
           100.0 * worst_spread,
           worst_spread > 0.05 ? "  — ВЫШЕ ПОРОГА, число не считается" : "");

    // What the answer means for the plan, said in the plan's own units rather than left to be
    // multiplied later.
    const double per_layer_nodes = 15.0, layers = 48.0;
    printf("\n  что это значит для карты: %d узлов на слой x %d слоёв = %.0f диспатчей на токен\n",
           int(per_layer_nodes), int(layers), per_layer_nodes * layers);
    printf("    по наклону ниже границы submit: %.1f мс на токен только на запуски\n",
           s_lo * per_layer_nodes * layers / 1000.0);
    printf("    для сравнения, по 31 мкс за круговую задержку: %.1f мс\n",
           31.0 * per_layer_nodes * layers / 1000.0);
    printf("  свободный член %.1f мкс — это цена одного submit+fence независимо от содержимого;\n"
           "    независимо измеренная встреча (rendezvous) в этом проекте стоила 177 мкс, и\n"
           "    расхождение между ними означало бы, что одно из двух измерений неверно\n", b_lo);

    ggml_backend_free(be);
    return 0;
}
