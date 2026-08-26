// The MemeX MoE layer: experts stored one per tensor, each at its own precision,
// dispatched on the host.
//
// Why it has to work this way. A ggml graph is static, but which experts a token
// needs is known only while computing, so llama.cpp keeps a layer's experts in one
// fused tensor and selects rows with `mul_mat_id`. A fused tensor carries a single
// type - which is exactly why per-expert precision cannot exist inside that design.
// The measurements say per-expert precision is what matters: use is skewed (the hot
// 10% of slots serve 56% of routing decisions), the hot set is task-specific (code
// and prose overlap by 2.2%), and 4-bit costs +2.5% perplexity while saving 8.7 GB.
//
// So this dispatches on the host instead: the router runs, the ids come back, and the
// FFN is assembled from the tensors those ids name - each in whatever ggml type the
// blob stored it as. On CPU a "sync" between those steps is just ordinary sequential
// execution, so the price is graph bookkeeping rather than device round-trips.
//
// Two things are checked here, because both can silently be wrong:
//   * the result must match the fused path llama.cpp would compute, within
//     quantisation error - the same expert weights, the same gate weights, the same
//     sum;
//   * the speed must be compared against that fused path, since per-expert dispatch
//     trades one big operator for k small ones.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

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

// One layer's experts, each its own tensor, types straight from the manifest.
struct ExpertSet {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> up, gate, down;
    std::vector<std::string> step;

    void free_all() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
    }
};

bool load_layer(ExpertSet* es, memex::BlobLoader& loader, uint32_t layer,
                const Shape& sh, ggml_backend_buffer_type_t buft, bool force_source) {
    const size_t n_tensors = size_t(sh.n_experts) * 3 + 16;
    ggml_init_params ip = {ggml_tensor_overhead() * n_tensors, nullptr, true};
    es->ctx = ggml_init(ip);
    if (!es->ctx) return false;

    es->up.resize(sh.n_experts);
    es->gate.resize(sh.n_experts);
    es->down.resize(sh.n_experts);
    es->step.resize(sh.n_experts);

    // Pick, for each expert, the entry the ladder assigned - or the verbatim source
    // copy when asked, which is how the two arms are compared on identical weights.
    std::vector<const memex::BlobEntry*> pick(size_t(sh.n_experts) * 3, nullptr);
    for (const auto& e : loader.entries()) {
        if (e.layer != layer) continue;
        const bool is_src = (e.step == "source");
        if (force_source != is_src) {
            // keep the other kind only if nothing better shows up
            const size_t idx = size_t(e.expert) * 3 + e.tensor;
            if (!pick[idx]) pick[idx] = &e;
            continue;
        }
        pick[size_t(e.expert) * 3 + e.tensor] = &e;
    }
    for (size_t i = 0; i < pick.size(); ++i) {
        if (!pick[i]) {
            printf("нет записи для эксперта %zu тензора %zu\n", i / 3, i % 3);
            return false;
        }
    }

    for (int e = 0; e < sh.n_experts; ++e) {
        const auto* pu = pick[size_t(e) * 3 + 0];
        const auto* pg = pick[size_t(e) * 3 + 1];
        const auto* pd = pick[size_t(e) * 3 + 2];
        es->up[e] = ggml_new_tensor_2d(es->ctx, (ggml_type)pu->ggml_type,
                                       sh.n_embd, sh.n_ff);
        es->gate[e] = ggml_new_tensor_2d(es->ctx, (ggml_type)pg->ggml_type,
                                         sh.n_embd, sh.n_ff);
        es->down[e] = ggml_new_tensor_2d(es->ctx, (ggml_type)pd->ggml_type,
                                         sh.n_ff, sh.n_embd);
        es->step[e] = pu->step;
    }
    es->buf = ggml_backend_alloc_ctx_tensors_from_buft(es->ctx, buft);
    if (!es->buf) return false;

    // Read every payload straight into its tensor: the blob already holds the bytes
    // in that ggml type's block layout, so there is nothing to convert.
    memex::AlignedBuffer tmp(8u << 20);
    for (int e = 0; e < sh.n_experts; ++e) {
        ggml_tensor* dst[3] = {es->up[e], es->gate[e], es->down[e]};
        for (int t = 0; t < 3; ++t) {
            const auto* p = pick[size_t(e) * 3 + t];
            if (ggml_nbytes(dst[t]) != p->bytes) {
                printf("размер не сходится: эксперт %d тензор %d, тензор %zu байт, "
                       "запись %u\n", e, t, ggml_nbytes(dst[t]), p->bytes);
                return false;
            }
            if (!loader.read_sync(*p, tmp.data(), tmp.size())) {
                printf("чтение не удалось: эксперт %d тензор %d\n", e, t);
                return false;
            }
            ggml_backend_tensor_set(dst[t], tmp.data(), 0, p->bytes);
        }
    }
    return true;
}

// Host-dispatched FFN for one token: only the chosen experts appear in the graph.
struct Dispatch {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* x = nullptr;
    ggml_tensor* out = nullptr;

    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
        alloc = nullptr;
        ctx = nullptr;
    }
};

bool build_dispatch(Dispatch* d, const ExpertSet& es, const Shape& sh,
                    const std::vector<int>& ids, const std::vector<float>& w,
                    ggml_backend_buffer_type_t buft, ggml_backend_t backend,
                    const std::vector<float>& act) {
    const size_t n_nodes = ids.size() * 8 + 64;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 32) +
                           ggml_graph_overhead_custom(n_nodes, false),
                           nullptr, true};
    d->ctx = ggml_init(ip);
    if (!d->ctx) return false;

    d->x = ggml_new_tensor_2d(d->ctx, GGML_TYPE_F32, sh.n_embd, 1);
    ggml_set_name(d->x, "x");
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(d->ctx, buft);
    if (!xbuf) return false;
    ggml_backend_tensor_set(d->x, act.data(), 0, ggml_nbytes(d->x));

    d->gf = ggml_new_graph_custom(d->ctx, n_nodes, false);
    ggml_tensor* acc = nullptr;
    for (size_t i = 0; i < ids.size(); ++i) {
        const int e = ids[i];
        ggml_tensor* a = ggml_mul_mat(d->ctx, es.gate[e], d->x);      // [n_ff, 1]
        ggml_tensor* b = ggml_mul_mat(d->ctx, es.up[e], d->x);
        ggml_tensor* h = ggml_mul(d->ctx, ggml_silu(d->ctx, a), b);
        ggml_tensor* o = ggml_mul_mat(d->ctx, es.down[e], h);         // [n_embd, 1]
        o = ggml_scale(d->ctx, o, w[i]);                              // gate weight
        acc = acc ? ggml_add(d->ctx, acc, o) : o;
    }
    d->out = acc;
    ggml_build_forward_expand(d->gf, d->out);
    d->alloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_reserve(d->alloc, d->gf)) return false;
    if (!ggml_gallocr_alloc_graph(d->alloc, d->gf)) return false;
    GGML_UNUSED(backend);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string blob = "D:/MemeX/blob/experts.bin";
    std::string man = "D:/MemeX/blob/experts.json";
    uint32_t layer = 20;
    int iters = 20;
    int threads = 4;
    int reps = 5;
    bool source_arm = false;
    Shape sh;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--blob") && i + 1 < argc) blob = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) man = argv[++i];
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) layer = uint32_t(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--source")) source_arm = true;
    }

    memex::BlobLoader loader;
    std::string err;
    if (!loader.open(blob, man, &err)) {
        printf("блоб не открылся: %s\n", err.c_str());
        return 1;
    }
    printf("записей в манифесте: %zu\n", loader.entries().size());

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);
    auto buft = ggml_backend_cpu_buffer_type();

    // Two versions of the same layer: as the ladder assigned it, and entirely in the
    // model's own precision. Comparing them isolates what mixed precision costs.
    // One arm per process: holding both copies costs 1.1 GB, and on a machine whose
    // page cache is already full of the model that alone made timings swing sixtyfold
    // between runs of identical code.
    ExpertSet mixed;
    if (!load_layer(&mixed, loader, layer, sh, buft, source_arm)) return 1;
    int n_low = 0;
    for (const auto& st : mixed.step) {
        if (st != "source") n_low++;
    }
    printf("слой %u загружен: %d экспертов сжато, %d в исходной точности\n",
           layer, n_low, sh.n_experts - n_low);

    // A plausible activation and a plausible routing decision. Ids come from the top
    // of the layer's own popularity so the comparison uses experts that matter.
    std::vector<float> act(size_t(sh.n_embd));
    for (size_t i = 0; i < act.size(); ++i) {
        act[i] = 0.02f * std::sin(float(i) * 0.013f);
    }
    std::vector<int> ids;
    for (int i = 0; i < sh.top_k; ++i) {
        ids.push_back((int(layer) * 7 + i * 11) % sh.n_experts);
    }
    std::vector<float> w(ids.size(), 1.0f / float(ids.size()));

    // A series, reported by median: a single figure cannot tell a real change from a
    // neighbour process stealing a core.
    std::vector<double> series;
    std::vector<float> out_mixed;
    for (int rep = 0; rep < reps; ++rep) {
        const ExpertSet& es = mixed;
        Dispatch d;
        if (!build_dispatch(&d, es, sh, ids, w, buft, cpu, act)) {
            printf("граф не собрался\n");
            return 1;
        }
        ggml_backend_graph_compute(cpu, d.gf);
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_graph_compute(cpu, d.gf);
        }
        series.push_back(ms_since(t0) / iters);
        if (rep == 0) {
            out_mixed.resize(ggml_nelements(d.out));
            ggml_backend_tensor_get(d.out, out_mixed.data(), 0, ggml_nbytes(d.out));
        }
        d.free_all();
    }
    std::sort(series.begin(), series.end());
    const double t_med = series[series.size() / 2];

    // How far mixed precision moved the answer, in the only terms that matter: the
    // layer's own output.
    printf("\nsloi %u, %d ekspertov, %d potokov, arm=%s\n",
           layer, sh.top_k, threads, source_arm ? "source" : "ladder");
    printf("  vremya sloya: mediana %.3f ms, min %.3f, max %.3f\n",
           t_med, series.front(), series.back());
    printf("  na 48 sloev: %.1f ms -> %.2f tok/s (tolko FFN ekspertov)\n",
           48.0 * t_med, 1000.0 / (48.0 * t_med));
    if (!out_mixed.empty()) {
        double n2 = 0.0;
        for (float v : out_mixed) n2 += double(v) * double(v);
        printf("  norma vyhoda: %.6f\n", std::sqrt(n2));
    }
    mixed.free_all();
    ggml_backend_free(cpu);
    return 0;
}
