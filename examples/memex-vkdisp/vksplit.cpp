// Where the 27 us per graph node actually goes: host recording, device execution, or submits.
//
// WHAT CAME BEFORE. vkdisp.cpp measured the marginal cost of one more node in a Vulkan graph and
// got 27.2 us/node for a dependent chain and 26.8 us/node for independent nodes - a ratio of 1.02,
// which says no part of it is a data dependency being waited on. The conclusion drawn there was
// "the launch itself is the cost". That conclusion is right about what it excludes and wrong about
// what it includes, and this file is about the difference, because the two candidates it lumps
// together have opposite fixes:
//
//   (a) HOST recording  - pipeline lookup, descriptor set write, vkCmdDispatch. CPU work. If this
//                         dominates, record our own command buffer once and re-submit it.
//   (b) DEVICE execution - the GPU picking the dispatch up, setting up workgroups, running. If this
//                         dominates, no better recording helps and only a persistent kernel does.
//
// WHAT READING ggml-vulkan.cpp TURNED UP, AND IT IS NEITHER. There is a third term, and on the
// graph vkdisp measured it is almost the whole of the 27 us. ggml_backend_vk_graph_compute decides
// when to submit like this (ggml-vulkan.cpp:10353-10370):
//
//     uint64_t mul_mat_bytes_per_submit = std::min(uint64_t(100*1000*1000), total_mat_mul_bytes / 40u);
//     ...
//     bool submit = (submitted_nodes >= nodes_per_submit) ||       // nodes_per_submit == 100
//                   (mul_mat_bytes >= mul_mat_bytes_per_submit) ||
//                   (i + fused == last_node) || (almost_ready && ...);
//
// total_mat_mul_bytes is the sum of ggml_nbytes(src[0]) over the graph's MUL_MAT and MUL_MAT_ID
// nodes. vkdisp's graph is a chain of ggml_add and contains no matmul at all, so that total is
// zero, so mul_mat_bytes_per_submit is zero, so `mul_mat_bytes >= mul_mat_bytes_per_submit` reads
// `0 >= 0` and is TRUE AT EVERY NODE. Every node ends the command buffer and calls
// ggml_vk_compute_forward, which calls ggml_vk_submit. vkdisp did not measure a graph of N nodes
// submitted once; it measured N graphs of one node each. The 27 us is a vkQueueSubmit, and the
// small intercept is small precisely because the submit is already inside the slope rather than
// being the thing the slope is added to.
//
// That also explains the 1.02 ratio without any appeal to launch cost: independent nodes cannot
// overlap when each one is its own submission with a fence-ordered predecessor, so of course the
// chain and the fan-out cost the same. The measurement was sound; it was answering a different
// question than the one asked of it.
//
// HOW TO PROVE THAT RATHER THAN ASSERT IT. The integer division is a knife edge. total/40u with
// total < 40 gives zero and submits every node; with total >= 40 gives at least one, and since an
// ADD node contributes nothing to mul_mat_bytes the accumulator sits at zero forever after and
// nothing submits until the 100-node or last-node rule fires. So prepend to the same chain of adds
// a single mul_mat whose src[0] is 32 bytes, and separately one whose src[0] is 64 bytes. Two
// graphs identical but for one row of an eight-element matrix, on either side of a threshold in
// ggml's arithmetic. If the reading above is right they differ by ~27 us per node. If it is wrong
// they differ by nothing, and the 27 us really is launch cost after all. There is no third outcome,
// which is what makes it worth running rather than reasoning about.
//
// FOUR PROLOGUES, THEN:
//
//     none   no matmul at all               total = 0     -> per_submit 0  -> submit every node
//     tiny   mul_mat, src[0] = 8x1 f32 = 32 B  total = 32 -> per_submit 0  -> submit every node
//     small  mul_mat, src[0] = 8x2 f32 = 64 B  total = 64 -> per_submit 1  -> batched
//     big    mul_mat, src[0] = 8x32768  = 1 MiB           -> per_submit 26214 -> batched
//
// tiny and small run the same shader with the same k; only the byte count differs, so nothing but
// the threshold can separate them. big is there so that the batched result cannot be an artefact
// of one particular prologue size.
//
// WHAT THE BATCHED SLOPE THEN IS. With the adds in one command buffer, the per-node cost is host
// recording plus device execution, pipelined against each other - which is the number the card plan
// actually needs, and the one that has to be split further. Two more sweeps do that:
//
//   * WORK SWEEP (measurement 2). Hold N and grow the bytes per node by ~4000x. Device execution
//     must grow with the bytes once they matter; host recording cannot, since recording a dispatch
//     over 4 MiB costs the same as over 1 KiB. A flat region followed by a ramp locates the fixed
//     cost and says where it stops mattering.
//
//   * DEVICE TIMESTAMPS (measurement 4, run separately). ggml has a timestamp path already:
//     GGML_VK_PERF_LOGGER=1 makes graph_compute write a vk::QueryType::eTimestamp either side of
//     every node and report the differences. In batched mode consecutive timestamps sit in the same
//     command buffer, so the difference is device time for that dispatch and its preceding barrier
//     and nothing else. Timestamps on eAllCommands can themselves act as a sync point, so this is
//     an upper bound on device cost - which is the helpful direction: a small upper bound settles
//     the question, a large one would not have.
//
// ONE MORE THING THE CODE SAYS, worth knowing whichever way the numbers fall. ggml_vk_sync_buffers
// (ggml-vulkan.cpp:1728) issues a full pipeline barrier - all shader and transfer reads and writes,
// both sides - and every op wrapper calls it immediately before its dispatch, unconditionally, with
// no reference to whether the tensors overlap. So in ggml's Vulkan backend no two dispatches ever
// overlap on the device, whatever the data dependencies are. That is a ceiling on what fusing can
// win and a floor under what our own command buffer could win, and it is not visible in any timing
// vkdisp took, because one-submit-per-node hid it.
//
// MEASUREMENT 3, one big dispatch against many small ones, is the direct bound on fusing: it is the
// whole overhead of splitting work N ways, with no model of where the overhead lives.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
// windows.h defines min and max as macros, which then eat every std::max( in this file.
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/resource.h>
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

namespace {

using Clock = std::chrono::steady_clock;

double us_since(const Clock::time_point& t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}

// Processor time this process has burned, in microseconds. This is the second, independent way of
// asking where the per-node cost lives, and it does not share a single failure mode with the device
// timestamps.
//
// Host-side recording - the pipeline lookup, the descriptor writes, vkCmdDispatch - is CPU work and
// shows up here. Device execution does not: while the GPU runs, this process is inside
// vkWaitForFences and burning nothing, or at worst spinning, and either way that cost does not
// scale with the node count because batched mode waits the same two or three times whatever N is.
// So the SLOPE of processor time in N is host recording and nothing else, while the slope of wall
// time is host recording plus device execution. The difference is the device side.
//
// Note std::clock() would not do here: MSVC returns wall time from it, not processor time, which
// would make the two columns identical and the comparison vacuous.
double proc_cpu_us() {
#ifdef _WIN32
    FILETIME cr, ex, kern, user;
    if (!GetProcessTimes(GetCurrentProcess(), &cr, &ex, &kern, &user)) return 0.0;
    auto to_us = [](const FILETIME& f) {
        return (double((uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime)) / 10.0;  // 100 ns ticks
    };
    return to_us(kern) + to_us(user);
#else
    rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
    return double(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1e6 +
           double(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
#endif
}

// The prologue decides, through ggml's own submit heuristic, whether the adds that follow are
// submitted one at a time or all at once. See the header comment: it is the whole experiment.
enum prologue_kind {
    PRO_NONE = 0,   // no matmul; total_mat_mul_bytes = 0
    PRO_TINY,       // 32 B of src[0]: 32/40 == 0, still submits every node
    PRO_SMALL,      // 64 B of src[0]: 64/40 == 1, batches
    PRO_BIG,        // 1 MiB of src[0]: batches, and by a wide margin
};

const char* prologue_name(prologue_kind p) {
    switch (p) {
        case PRO_NONE:  return "none  (нет matmul,      total=0 B)";
        case PRO_TINY:  return "tiny  (matmul src0 8x1, total=32 B)";
        case PRO_SMALL: return "small (matmul src0 8x2, total=64 B)";
        case PRO_BIG:   return "big   (matmul src0 8x32768, total=1 MiB)";
    }
    return "?";
}

// Rows of the prologue's src[0]. k is held at 8 for all of them so tiny and small run the same
// shader and differ only in the byte count the heuristic sees.
int prologue_rows(prologue_kind p) {
    switch (p) {
        case PRO_NONE:  return 0;
        case PRO_TINY:  return 1;
        case PRO_SMALL: return 2;
        case PRO_BIG:   return 32768;
    }
    return 0;
}

constexpr int PROLOGUE_K = 8;

// Everything a timed graph needs, built once and reused across repetitions. Building is not timed:
// the plan would build once and compute many times, and in any case ggml_gallocr_reserve is a
// different order of magnitude and would swamp the slope.
struct bench_graph {
    ggml_context*         c_in = nullptr;
    ggml_context*         c    = nullptr;
    ggml_backend_buffer_t b_in = nullptr;
    ggml_gallocr_t        ga   = nullptr;
    ggml_cgraph*          gf   = nullptr;
    int                   n_built = 0;   // nodes ggml actually put in the graph
    bool                  ok   = false;

    ~bench_graph() {
        if (ga)   ggml_gallocr_free(ga);
        if (c)    ggml_free(c);
        if (b_in) ggml_backend_buffer_free(b_in);
        if (c_in) ggml_free(c_in);
    }
};

// n_add nodes of n_elem f32 each, chained so none is dead and none can be reordered, optionally
// behind a mul_mat prologue whose only purpose is the byte count it contributes.
// `chain` false makes the adds independent instead - kept because it is the one thing that could
// distinguish launch from dependency, and it is worth re-asking once the submits are out of the way.
bool build(bench_graph& g, ggml_backend_buffer_type_t buft, int n_add, int n_elem,
           prologue_kind pro, bool chain) {
    const int rows = prologue_rows(pro);

    ggml_init_params ip_in = {ggml_tensor_overhead() * 8 + 1024, nullptr, true};
    g.c_in = ggml_init(ip_in);
    if (!g.c_in) return false;

    ggml_tensor* x = ggml_new_tensor_1d(g.c_in, GGML_TYPE_F32, n_elem);
    ggml_tensor* k = ggml_new_tensor_1d(g.c_in, GGML_TYPE_F32, n_elem);
    ggml_set_name(x, "x");
    ggml_set_name(k, "k");

    ggml_tensor* w = nullptr;   // prologue weights, [PROLOGUE_K, rows]
    ggml_tensor* v = nullptr;   // prologue activations, [PROLOGUE_K, 1]
    if (rows > 0) {
        w = ggml_new_tensor_2d(g.c_in, GGML_TYPE_F32, PROLOGUE_K, rows);
        v = ggml_new_tensor_2d(g.c_in, GGML_TYPE_F32, PROLOGUE_K, 1);
        ggml_set_name(w, "w");
        ggml_set_name(v, "v");
    }

    g.b_in = ggml_backend_alloc_ctx_tensors_from_buft(g.c_in, buft);
    if (!g.b_in) return false;

    {
        std::vector<float> h;
        h.resize(std::size_t(n_elem));
        for (int i = 0; i < n_elem; ++i) h[std::size_t(i)] = 1.0f / float(i + 1);
        ggml_backend_tensor_set(x, h.data(), 0, sizeof(float) * std::size_t(n_elem));
        for (int i = 0; i < n_elem; ++i) h[std::size_t(i)] = 1e-6f;
        ggml_backend_tensor_set(k, h.data(), 0, sizeof(float) * std::size_t(n_elem));
        if (w) {
            std::vector<float> hw(std::size_t(PROLOGUE_K) * std::size_t(rows), 0.5f);
            ggml_backend_tensor_set(w, hw.data(), 0, sizeof(float) * hw.size());
            std::vector<float> hv(std::size_t(PROLOGUE_K), 0.25f);
            ggml_backend_tensor_set(v, hv.data(), 0, sizeof(float) * hv.size());
        }
    }

    ggml_init_params ip = {
        ggml_tensor_overhead() * std::size_t(n_add + 16) + ggml_graph_overhead(), nullptr, true};
    g.c = ggml_init(ip);
    if (!g.c) return false;
    g.gf = ggml_new_graph(g.c);

    // The prologue goes in first so that by the time the adds are walked, ggml has already seen the
    // matmul bytes in its dry-run pass - though in fact the dry-run pass totals the whole graph
    // before the recording pass starts, so the order does not matter to the heuristic. It matters
    // for reading the result: the prologue's own cost is then a constant in the intercept.
    if (w) {
        ggml_build_forward_expand(g.gf, ggml_mul_mat(g.c, w, v));
    }

    if (chain) {
        ggml_tensor* cur = x;
        for (int i = 0; i < n_add; ++i) cur = ggml_add(g.c, cur, k);
        ggml_build_forward_expand(g.gf, cur);
    } else {
        for (int i = 0; i < n_add; ++i) ggml_build_forward_expand(g.gf, ggml_add(g.c, x, k));
    }

    g.n_built = ggml_graph_n_nodes(g.gf);

    g.ga = ggml_gallocr_new(buft);
    if (!g.ga || !ggml_gallocr_reserve(g.ga, g.gf) || !ggml_gallocr_alloc_graph(g.ga, g.gf)) {
        return false;
    }
    g.ok = true;
    return true;
}

// Microseconds per ggml_backend_graph_compute. reps is chosen by the caller to hold the wall time
// per point roughly constant: at the small end a graph is ~60 us and a fixed repetition count puts
// the operating system's scheduling jitter on the same order as the thing being measured.
double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct timing {
    double wall = -1.0;   // microseconds per graph_compute
    double cpu  = -1.0;   // processor microseconds per graph_compute
    bool ok() const { return wall >= 0.0; }
};

timing time_graph(ggml_backend_t be, bench_graph& g, int reps) {
    timing t;
    if (!g.ok) return t;
    for (int i = 0; i < 8; ++i) ggml_backend_graph_compute(be, g.gf);   // pipelines, descriptor pools
    const double c0 = proc_cpu_us();
    const auto   t0 = Clock::now();
    for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(be, g.gf);
    t.wall = us_since(t0) / double(reps);
    t.cpu  = (proc_cpu_us() - c0) / double(reps);
    return t;
}

// Build once, time `inner` times, keep the median.
//
// The inner median is not decoration. Without it the first three-replicate run of this file put a
// 4989 us point next to a 130 us one at the same node count - a single stall, from the desktop
// compositor sharing the GPU or from a driver allocation, landing in a min/max and reporting a
// 3713% spread on a measurement whose noise floor is 4.2%. One stall in five samples moves a median
// not at all and moves a mean by a factor of eight, and the thing being measured here is a few
// microseconds sitting on top of a submit that takes tens. Building once rather than per sample
// also keeps buffer allocation and pipeline warm-up out of the timed region entirely.
timing measure(ggml_backend_t be, ggml_backend_buffer_type_t buft, int n_add, int n_elem,
               prologue_kind pro, bool chain, int reps, int* n_built, int inner = 5) {
    bench_graph g;
    if (!build(g, buft, n_add, n_elem, pro, chain)) return timing{};
    if (n_built) *n_built = g.n_built;
    std::vector<double> w, c;
    for (int i = 0; i < inner; ++i) {
        const timing t = time_graph(be, g, reps);
        if (!t.ok()) return timing{};
        w.push_back(t.wall);
        c.push_back(t.cpu);
    }
    timing out;
    out.wall = median_of(w);
    out.cpu  = median_of(c);
    return out;
}

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

// A point is a median over replicates plus the spread that says whether it is a result at all.
// The noise floor on this machine is 4.2%; a single replicate reports 0.0% and means nothing.
struct point {
    double med = 0, lo = 0, hi = 0, spread = 0;
};

point summarise(const std::vector<double>& v) {
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    point p;
    p.med = s[s.size() / 2];
    p.lo  = s.front();
    p.hi  = s.back();
    p.spread = p.med > 0.0 ? (p.hi - p.lo) / p.med : 0.0;
    return p;
}

// Repetitions for a point, scaled so each takes about the same wall time whatever it costs.
int reps_for(int base, double approx_us) {
    if (approx_us <= 0.0) return base;
    const int r = int(1.5e5 / approx_us);           // aim for ~0.15 s per sample, five samples a point
    return std::max(16, std::min(30000, std::max(base / 8, r)));
}

}  // namespace

int main(int argc, char** argv) {
    int  n_elem     = 1024;      // 4 KiB per tensor, same as vkdisp
    int  replicates = 3;
    bool do_perflog = false;
    int  perflog_nodes = 32;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--elems" && i + 1 < argc) n_elem = std::atoi(argv[++i]);
        else if (a == "--replicates" && i + 1 < argc) replicates = std::atoi(argv[++i]);
        else if (a == "--perflog") do_perflog = true;
        else if (a == "--perflog-nodes" && i + 1 < argc) perflog_nodes = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            printf("llama-memex-vksplit — куда уходят 27 мкс на узел графа\n"
                   "  --elems N          элементов f32 в тензоре (%d)\n"
                   "  --replicates N     повторов каждого свипа (%d)\n"
                   "  --perflog          один прогон под GGML_VK_PERF_LOGGER=1: печатает\n"
                   "                     штампы времени устройства и больше ничего\n"
                   "  --perflog-nodes N  сколько узлов в этом прогоне (%d)\n",
                   n_elem, replicates, perflog_nodes);
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

    // ---------------------------------------------------------------------------------------
    // Measurement 4 lives here rather than in its own binary because it needs exactly the same
    // graph. It is a separate invocation because GGML_VK_PERF_LOGGER is read once at instance
    // init, and because the logger prints on every graph_compute - so this mode computes a
    // countable number of times and nothing else.
    // ---------------------------------------------------------------------------------------
    if (do_perflog) {
        const bool on = getenv("GGML_VK_PERF_LOGGER") != nullptr;
        printf("режим штампов устройства: GGML_VK_PERF_LOGGER %s\n",
               on ? "включён" : "НЕ ВКЛЮЧЁН — числа ниже пусты");
        // Two node counts, not one, and the wall time as well as the timestamps. The timestamps
        // alone are not enough: writeTimestamp on eAllCommands is itself a sync point and inflates
        // the device timeline by an unknown per-node amount. But that amount also shows up in the
        // wall clock of the same run, and the wall clock without timestamps is measured by
        // measurement 2. So:
        //
        //     штамп       = наклон_стены_со_штампами - наклон_стены_без_штампов
        //     устройство  = штамп_устройства - штамп
        //     хост        = наклон_стены_без_штампов - устройство
        //
        // Two node counts give a slope, which removes the constants - the extra submit, the fence
        // wait and getQueryPoolResults that the perf-logger path adds once per graph.
        //
        // The subtraction assumes recording and execution do not overlap in batched mode, which is
        // very nearly true: with one submit at the matmul, one at the almost_ready fence around 80%
        // through and one at the last node, only the last fifth of the graph is recorded while the
        // GPU works. That biases the host share up by at most about a fifth.
        // A sweep in N rather than a single point, because the per-ADD figure the logger reports is
        // contaminated at exactly one node and the contamination is separable.
        //
        // In batched mode the graph is not one command buffer but three: the matmul prologue is
        // submitted on its own, then the body, then whatever the almost_ready fence cuts off. Two of
        // the N timestamp differences therefore span a command-buffer boundary and contain the host
        // gap between one submission and the next, which has nothing to do with executing an ADD.
        // The logger averages over all N, so what it prints is
        //
        //     среднее(N) = устройство + разрыв / N
        //
        // and a fit against 1/N returns both terms: the intercept is the device time for one ADD,
        // the slope is the host gap at a submit boundary. One point cannot separate them; four can,
        // and the fit failing to be a straight line would itself say the model is wrong.
        const int perf_ns[4] = {8, 16, 32, perflog_nodes};
        for (int m = 0; m < 2; ++m) {
            const prologue_kind pro = m == 0 ? PRO_SMALL : PRO_TINY;
            const char* label = m == 0 ? "ПАКЕТНО" : "ПО-ОДНОМУ";
            printf("\n=== %s, ADD по %d f32 ===\n", label, n_elem);
            fflush(stdout);
            for (int j = 0; j < 4; ++j) {
                bench_graph g;
                if (!build(g, buft, perf_ns[j], n_elem, pro, true)) {
                    printf("  не удалось построить граф на %d узлов\n", perf_ns[j]);
                    continue;
                }
                fprintf(stderr, "--- РАЗОГРЕВ %s N=%d ---\n", label, perf_ns[j]);
                for (int i = 0; i < 8; ++i) ggml_backend_graph_compute(be, g.gf);
                fflush(stderr);
                fprintf(stderr, "--- ИЗМЕРЕНИЕ %s N=%d ---\n", label, perf_ns[j]);
                const int reps = 20;
                const auto t0 = Clock::now();
                for (int i = 0; i < reps; ++i) ggml_backend_graph_compute(be, g.gf);
                const double wall = us_since(t0) / double(reps);
                fflush(stderr);
                printf("  %s N=%3d: стена %.2f мкс на graph_compute\n", label, perf_ns[j], wall);
                fflush(stdout);
            }
        }
        printf("\n  штампы устройства — на stderr, блоками; медиана блоков после строки\n"
               "  «ИЗМЕРЕНИЕ ... N=...» — это среднее(N) выше\n");
        ggml_backend_free(be);
        return 0;
    }

    printf("повторов каждого свипа: %d (порог шума 4.2%%, одна реплика — не результат)\n", replicates);

    // ===========================================================================================
    // MEASUREMENT 1. The submit threshold. Same chain of adds, four prologues, two of which sit
    // either side of ggml's `total_mat_mul_bytes / 40u` integer division.
    // ===========================================================================================
    printf("\n"
           "===========================================================================\n"
           "ИЗМЕРЕНИЕ 1 — порог submit. Одна и та же цепочка ADD по %d f32 (%.1f КиБ),\n"
           "  меняется только пролог, и вместе с ним — решает ли ggml делать submit\n"
           "  после каждого узла или копить их в один командный буфер.\n"
           "===========================================================================\n",
           n_elem, double(n_elem) * 4.0 / 1024.0);

    const std::vector<int> ns = {1, 2, 4, 8, 16, 32, 64};
    const prologue_kind pros[4] = {PRO_NONE, PRO_TINY, PRO_SMALL, PRO_BIG};
    double slope_of[4] = {0, 0, 0, 0};
    double inter_of[4] = {0, 0, 0, 0};
    double cpu_slope_of[4] = {0, 0, 0, 0};
    double worst_of[4] = {0, 0, 0, 0};

    for (int pi = 0; pi < 4; ++pi) {
        const prologue_kind pro = pros[pi];
        printf("\n--- пролог %s ---\n", prologue_name(pro));

        // Not `runs(std::size_t(replicates))`: that is the most vexing parse and declares a
        // function taking a size_t. vkdisp.cpp hit the same thing and its comment says so.
        std::vector<std::vector<double>> runs, cpus;
        runs.resize(std::size_t(replicates));
        cpus.resize(std::size_t(replicates));
        for (int r = 0; r < replicates; ++r) {
            for (int n : ns) {
                int built = 0;
                // A first cheap pass to find out roughly what the point costs, then the real one
                // sized from it. The probe pass is 24 computes and costs a few milliseconds.
                const timing probe = measure(be, buft, n, n_elem, pro, true, 24, &built);
                if (!probe.ok()) {
                    printf("не удалось построить или разместить граф на %d узлов\n", n);
                    ggml_backend_free(be);
                    return 1;
                }
                const int expect = n + (prologue_rows(pro) > 0 ? 1 : 0);
                if (built != expect) {
                    printf("ВНИМАНИЕ: ожидали %d узлов, граф содержит %d — ось X не та\n",
                           expect, built);
                }
                const timing t = measure(be, buft, n, n_elem, pro, true, reps_for(300, probe.wall), nullptr);
                runs[std::size_t(r)].push_back(t.wall);
                cpus[std::size_t(r)].push_back(t.cpu);
            }
        }

        printf("  %6s %12s %12s %12s %10s %12s\n", "узлов", "стена,мкс", "мин,мкс", "макс,мкс",
               "разброс", "процессор,мкс");
        std::vector<double> xs, ys, ycpu;
        double worst = 0.0;
        for (std::size_t i = 0; i < ns.size(); ++i) {
            std::vector<double> v, vc;
            for (int r = 0; r < replicates; ++r) {
                v.push_back(runs[std::size_t(r)][i]);
                vc.push_back(cpus[std::size_t(r)][i]);
            }
            const point p = summarise(v);
            const point pc = summarise(vc);
            worst = std::max(worst, p.spread);
            printf("  %6d %12.2f %12.2f %12.2f %9.1f%% %12.2f\n", ns[i], p.med, p.lo, p.hi,
                   100.0 * p.spread, pc.med);
            xs.push_back(double(ns[i]));
            ys.push_back(p.med);
            ycpu.push_back(pc.med);
        }
        double dummy = 0;
        fit(xs, ys, &slope_of[pi], &inter_of[pi]);
        fit(xs, ycpu, &cpu_slope_of[pi], &dummy);
        printf("  наклон стены %.2f мкс/узел (своб. член %.1f), наклон процессора %.2f мкс/узел\n"
               "  худший разброс %.1f%%%s\n",
               slope_of[pi], inter_of[pi], cpu_slope_of[pi],
               100.0 * worst, worst > 0.042 ? "  — ВЫШЕ ПОРОГА" : "");
        worst_of[pi] = worst;
    }

    const double slope_solo    = 0.5 * (slope_of[0] + slope_of[1]);   // none, tiny
    const double slope_batched = 0.5 * (slope_of[2] + slope_of[3]);   // small, big
    const double submit_cost   = slope_solo - slope_batched;

    printf("\n  --- что показал порог ---\n");
    printf("  submit на узел (none, tiny): %.2f и %.2f мкс/узел, среднее %.2f\n",
           slope_of[0], slope_of[1], slope_solo);
    printf("  пакетно      (small, big) : %.2f и %.2f мкс/узел, среднее %.2f\n",
           slope_of[2], slope_of[3], slope_batched);
    printf("  разница                    : %.2f мкс/узел\n", submit_cost);

    // The second, independent question, asked of the same runs: of the batched per-node cost, how
    // much is the CPU recording the dispatch and how much is the GPU running it? Processor time
    // does not accumulate while the process waits on a fence, and batched mode waits the same two
    // or three times whatever N is, so the slope of processor time in N is host recording alone.
    const double cpu_solo    = 0.5 * (cpu_slope_of[0] + cpu_slope_of[1]);
    const double cpu_batched = 0.5 * (cpu_slope_of[2] + cpu_slope_of[3]);
    printf("\n  --- хост против устройства, по процессорному времени ---\n");
    printf("  пакетно: стена %.2f мкс/узел, из них процессор %.2f — значит устройство ~%.2f\n",
           slope_batched, cpu_batched, slope_batched - cpu_batched);
    printf("  submit на узел: стена %.2f мкс/узел, из них процессор %.2f\n", slope_solo, cpu_solo);
    if (slope_batched > 0.0) {
        const double share = 100.0 * cpu_batched / slope_batched;
        printf("  доля хоста в пакетном узле: %.0f%% — %s\n", share,
               share > 60.0 ? "запись команды на хосте, свой командный буфер имеет смысл"
                            : (share < 35.0
                                   ? "исполнение на устройстве; лучшая запись команд не поможет"
                                   : "ни то ни другое не главное; смотрите абсолютную величину"));
    }
    if (slope_batched > 0 && slope_solo / slope_batched > 1.5) {
        printf("  tiny и small отличаются одной строкой матрицы 8x1 против 8x2 — ничем, кроме\n"
               "  того, на какой стороне деления total/40u они оказались. Раз наклоны у них\n"
               "  разные, 27 мкс из vkdisp — это vkQueueSubmit, а не запуск ядра.\n");
    } else {
        printf("  наклоны совпали. Порог submit ни при чём, и 27 мкс — действительно цена\n"
               "  запуска, как и предполагалось. Тогда читайте измерение 2 как решающее.\n");
    }

    // ===========================================================================================
    // MEASUREMENT 1b. Chain against fan-out, now that submits are out of the way. vkdisp asked this
    // and got 1.02x, but with one submit per node the answer could only ever have been 1.00x.
    // Note before reading it: ggml_vk_sync_buffers puts a full pipeline barrier before every
    // dispatch unconditionally, so the expected answer is still 1.00x - and if it is, that is a
    // measured fact about the backend rather than an artefact of the submit rule.
    // ===========================================================================================
    printf("\n"
           "===========================================================================\n"
           "ИЗМЕРЕНИЕ 1b — цепочка против веера, уже в пакетном режиме\n"
           "===========================================================================\n");
    {
        const int n = 32;
        std::vector<double> vc, vw;
        for (int r = 0; r < replicates; ++r) {
            const timing pc = measure(be, buft, n, n_elem, PRO_SMALL, true, 24, nullptr);
            vc.push_back(measure(be, buft, n, n_elem, PRO_SMALL, true, reps_for(300, pc.wall), nullptr).wall);
            const timing pw = measure(be, buft, n, n_elem, PRO_SMALL, false, 24, nullptr);
            vw.push_back(measure(be, buft, n, n_elem, PRO_SMALL, false, reps_for(300, pw.wall), nullptr).wall);
        }
        const point pc = summarise(vc), pw = summarise(vw);
        printf("  цепочка   %d узлов: %.2f мкс  (разброс %.1f%%)\n", n, pc.med, 100.0 * pc.spread);
        printf("  веер      %d узлов: %.2f мкс  (разброс %.1f%%)\n", n, pw.med, 100.0 * pw.spread);
        printf("  отношение          : %.3fx — %s\n", pw.med > 0 ? pc.med / pw.med : 0.0,
               (pw.med > 0 && pc.med / pw.med > 1.15)
                   ? "независимые узлы дешевле: что-то всё-таки перекрывается"
                   : "не перекрывается ничего — барьер в ggml_vk_sync_buffers стоит между "
                     "каждой парой диспатчей");
    }

    // ===========================================================================================
    // MEASUREMENT 2. Work per node. Host recording cannot grow with the bytes; device execution
    // must. So sweep the bytes and watch which happens.
    // ===========================================================================================
    printf("\n"
           "===========================================================================\n"
           "ИЗМЕРЕНИЕ 2 — работа на узел. N фиксировано, размер тензора растёт в 4096 раз.\n"
           "  Запись команды на хосте не может расти вместе с байтами; исполнение на\n"
           "  устройстве обязано. Плоский участок — это постоянная часть; где начинается\n"
           "  подъём, там она перестаёт иметь значение.\n"
           "===========================================================================\n");
    {
        const std::vector<int> elems = {256, 1024, 4096, 16384, 65536, 262144, 1048576};
        const int n_lo = 8, n_hi = 40;   // slope from two points kills the intercept exactly
        printf("  %10s %8s %13s %13s %13s %11s %9s\n",
               "элементов", "КиБ", "пакетно,мкс", "процессор,мкс", "по одному,мкс", "байт/мкс",
               "разброс");
        for (int e : elems) {
            std::vector<double> sb, ss, sbc;
            for (int r = 0; r < replicates; ++r) {
                for (int m = 0; m < 2; ++m) {
                    const prologue_kind pro = m == 0 ? PRO_SMALL : PRO_NONE;
                    const timing p_lo = measure(be, buft, n_lo, e, pro, true, 8, nullptr);
                    const timing p_hi = measure(be, buft, n_hi, e, pro, true, 8, nullptr);
                    if (!p_lo.ok() || !p_hi.ok()) { printf("  %10d  не удалось разместить\n", e); goto next_elem; }
                    const timing t_lo = measure(be, buft, n_lo, e, pro, true, reps_for(200, p_lo.wall), nullptr);
                    const timing t_hi = measure(be, buft, n_hi, e, pro, true, reps_for(200, p_hi.wall), nullptr);
                    const double per  = (t_hi.wall - t_lo.wall) / double(n_hi - n_lo);
                    const double perc = (t_hi.cpu  - t_lo.cpu ) / double(n_hi - n_lo);
                    if (m == 0) { sb.push_back(per); sbc.push_back(perc); } else ss.push_back(per);
                }
            }
            {
                const point pb = summarise(sb), ps = summarise(ss), pbc = summarise(sbc);
                // Three tensors touched per add: two read, one written.
                const double bytes = 3.0 * 4.0 * double(e);
                printf("  %10d %8.1f %13.2f %13.2f %13.2f %11.0f %8.1f%%\n",
                       e, double(e) * 4.0 / 1024.0, pb.med, pbc.med, ps.med,
                       pb.med > 0 ? bytes / pb.med : 0.0, 100.0 * pb.spread);
            }
            next_elem:;
        }
        printf("  «процессор» — тот же наклон, но по процессорному времени. Оно не может расти\n"
               "  вместе с байтами: запись vkCmdDispatch стоит одинаково над 1 КиБ и над 4 МиБ.\n"
               "  Значит всё, что растёт в колонке «пакетно» и не растёт здесь, — устройство.\n");
    }

    // ===========================================================================================
    // MEASUREMENT 3. One dispatch against N. The direct bound on what fusing can win, with no
    // model of where the overhead lives.
    // ===========================================================================================
    printf("\n"
           "===========================================================================\n"
           "ИЗМЕРЕНИЕ 3 — один большой диспатч против N маленьких на ту же работу\n"
           "===========================================================================\n");
    {
        const int n = 32;
        const std::vector<int> per_node = {1024, 16384, 262144};
        printf("  %12s %14s %14s %12s\n", "на узел", "32 узла,мкс", "1 узел,мкс", "выигрыш");
        for (int e : per_node) {
            std::vector<double> vs, vb;
            for (int r = 0; r < replicates; ++r) {
                const timing ps = measure(be, buft, n, e, PRO_SMALL, true, 8, nullptr);
                vs.push_back(measure(be, buft, n, e, PRO_SMALL, true, reps_for(200, ps.wall), nullptr).wall);
                const timing pb = measure(be, buft, 1, e * n, PRO_SMALL, true, 8, nullptr);
                vb.push_back(measure(be, buft, 1, e * n, PRO_SMALL, true, reps_for(200, pb.wall), nullptr).wall);
            }
            const point ps = summarise(vs), pb = summarise(vb);
            printf("  %12d %14.2f %14.2f %11.2fx  (разброс %.1f%% / %.1f%%)\n",
                   e, ps.med, pb.med, pb.med > 0 ? ps.med / pb.med : 0.0,
                   100.0 * ps.spread, 100.0 * pb.spread);
        }
        printf("  оба варианта делают ровно одинаковое число операций и трогают одинаковое\n"
               "  число байт; вся разница — накладные расходы на разбиение на 32 части\n");
    }

    // ===========================================================================================
    // MEASUREMENT 5. The chain of adds is a laboratory graph. A real one is mostly matmuls, and for
    // a graph that IS matmuls the submit rule reads differently and, it turns out, badly at exactly
    // the size the card plan uses.
    //
    // With N identical matmuls of B bytes each, total_mat_mul_bytes = N*B and the threshold is
    // N*B/40. Each matmul adds B to the accumulator, so a submit happens every ceil(N/40) matmuls -
    // and ceil(N/40) is ONE for every N up to forty. A fifteen-node layer graph submits after every
    // matmul in it. Only once the graph holds more than forty matmuls does the rule start batching,
    // and the doubling for the first three submits pushes the useful threshold higher still.
    //
    // So the same 720 nodes cost very different amounts depending on whether they are handed to
    // ggml as forty-eight graphs of fifteen nodes or as one graph of 720. This measures which.
    // ===========================================================================================
    printf("\n"
           "===========================================================================\n"
           "ИЗМЕРЕНИЕ 5 — граф из одних matmul: цена узла как функция размера ГРАФА.\n"
           "  Правило ggml даёт submit каждые ceil(N/40) матумножений, а ceil(N/40)\n"
           "  равно единице для любого N до сорока. Значит граф на слой (15 узлов)\n"
           "  делает submit после каждого matmul, а граф на весь токен — нет.\n"
           "===========================================================================\n");
    {
        const int d = 256;                       // src0 = 256x256 f32 = 256 КиБ
        const std::vector<int> nm = {8, 15, 30, 45, 60, 120, 240};
        printf("  %8s %14s %14s %16s\n", "matmul", "всего,мкс", "мкс/узел", "submit каждые");
        for (int n : nm) {
            std::vector<double> v;
            for (int r = 0; r < replicates; ++r) {
                // Built here rather than through build(): the chain is matmuls, not adds.
                ggml_init_params ip_in = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
                ggml_context* c_in = ggml_init(ip_in);
                ggml_tensor* w = ggml_new_tensor_2d(c_in, GGML_TYPE_F32, d, d);
                ggml_tensor* x = ggml_new_tensor_2d(c_in, GGML_TYPE_F32, d, 1);
                ggml_backend_buffer_t b_in = ggml_backend_alloc_ctx_tensors_from_buft(c_in, buft);
                if (!b_in) { ggml_free(c_in); printf("  %8d  нет памяти\n", n); break; }
                {
                    std::vector<float> h(std::size_t(d) * std::size_t(d), 0.01f);
                    ggml_backend_tensor_set(w, h.data(), 0, sizeof(float) * h.size());
                    std::vector<float> hx(std::size_t(d), 0.5f);
                    ggml_backend_tensor_set(x, hx.data(), 0, sizeof(float) * hx.size());
                }
                ggml_init_params ip = {
                    ggml_tensor_overhead() * std::size_t(n + 8) + ggml_graph_overhead(), nullptr, true};
                ggml_context* c = ggml_init(ip);
                ggml_cgraph* gf = ggml_new_graph(c);
                ggml_tensor* cur = x;
                for (int i = 0; i < n; ++i) cur = ggml_mul_mat(c, w, cur);
                ggml_build_forward_expand(gf, cur);
                ggml_gallocr_t ga = ggml_gallocr_new(buft);
                if (!ga || !ggml_gallocr_reserve(ga, gf) || !ggml_gallocr_alloc_graph(ga, gf)) {
                    if (ga) ggml_gallocr_free(ga);
                    ggml_free(c); ggml_backend_buffer_free(b_in); ggml_free(c_in);
                    printf("  %8d  не удалось разместить\n", n);
                    break;
                }
                for (int i = 0; i < 8; ++i) ggml_backend_graph_compute(be, gf);
                std::vector<double> inner;
                for (int s = 0; s < 5; ++s) {
                    const int reps = std::max(8, std::min(2000, 40000 / std::max(n, 1)));
                    const auto t0 = Clock::now();
                    for (int q = 0; q < reps; ++q) ggml_backend_graph_compute(be, gf);
                    inner.push_back(us_since(t0) / double(reps));
                }
                v.push_back(median_of(inner));
                ggml_gallocr_free(ga);
                ggml_free(c);
                ggml_backend_buffer_free(b_in);
                ggml_free(c_in);
            }
            if (v.empty()) continue;
            const point p = summarise(v);
            printf("  %8d %14.1f %14.2f %16d   (разброс %.1f%%)\n",
                   n, p.med, p.med / double(n), (n + 39) / 40, 100.0 * p.spread);
        }
        printf("  «submit каждые» — предсказание по правилу ggml, не измерение. Если цена\n"
               "  узла падает там же, где это число становится больше единицы, правило\n"
               "  прочитано верно.\n");
    }

    // ===========================================================================================
    // What it means for the plan, in the plan's own units.
    // ===========================================================================================
    printf("\n"
           "===========================================================================\n"
           "ЧТО ЭТО ЗНАЧИТ ДЛЯ КАРТЫ: 15 узлов на слой x 48 слоёв = 720 узлов на токен\n"
           "===========================================================================\n");
    printf("  по vkdisp (27.14 мкс/узел, submit на каждый узел)     : %.1f мс/токен\n",
           27.14 * 720.0 / 1000.0);
    printf("  по наклону при submit на узел, измерено здесь (%.2f)  : %.1f мс/токен\n",
           slope_solo, slope_solo * 720.0 / 1000.0);
    printf("  по пакетному наклону (%.2f)                           : %.1f мс/токен\n",
           slope_batched, slope_batched * 720.0 / 1000.0);
    printf("  разница — %.1f мс/токен, и она уходит целиком в submit, который граф\n"
           "  из настоящей модели делает реже, потому что в нём есть matmul\n",
           (slope_solo - slope_batched) * 720.0 / 1000.0);
    printf("  если ужать 15 узлов до 8, пакетный наклон даёт %.1f мс/токен\n",
           slope_batched * 8.0 * 48.0 / 1000.0);

    double worst_all = 0.0;
    for (int i = 0; i < 4; ++i) worst_all = std::max(worst_all, worst_of[i]);
    printf("\n  худший разброс по измерению 1: %.1f%% (порог 4.2%%)%s\n",
           100.0 * worst_all, worst_all > 0.042 ? "  — часть точек выше порога" : "");
    if (replicates < 3) {
        printf("  ВНИМАНИЕ: реплик %d. Меньше трёх — это не результат, а одно наблюдение.\n",
               replicates);
    }

    ggml_backend_free(be);
    return 0;
}
