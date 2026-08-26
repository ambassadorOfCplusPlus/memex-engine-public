// The whole engine on one path: residency policy, asynchronous precision moves,
// mixed-precision compute, and routing that knows what is resident.
//
// Every part of this was verified alone before it was joined, and the joining is the
// point: a policy that looks right against a cost model can still ask for transfers
// the disk will not deliver, and routing that prefers resident experts changes which
// experts the policy then wants. Only the closed loop shows that.
//
// What runs here, and the measurement that put it there:
//
//   * per-expert precision from the blob. Mixed precision computed a layer in 1.245 ms
//     against 1.324 for the model's own weights, and it is the reason the ladder exists.
//   * asynchronous moves in both directions, staged then swapped, so an expert is
//     always readable. A synchronous demotion used to stall a token by 15-60 ms.
//   * a retained coarse copy (EPLB's redundant expert, applied to precision), which
//     turns a demotion into a pointer swap.
//   * a statistics window with periodic rebalance instead of per-token greed, which is
//     what stopped 117k wasted demotions.
//   * cache-conditional routing: a bonus to resident experts before top-k. Offline it
//     cut misses by 36% for +0.73% perplexity, and here it closes the loop - residency
//     steers routing, routing steers residency.
//   * prefill priming: the prompt's own routing arranges memory before the first
//     generated token, which is free because prefill computes it anyway.
//
// Input is a recorded routing trace with the full router distribution, so the
// experiment replays what the model really asked for rather than a synthetic pattern.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "memex/blob_loader.hpp"
#include "memex/layer_runtime.hpp"

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

// Router distributions for one layer, read from a trace produced with
// MOE_TRACE_PROBS=1. Tags: >=0 ids, -(l+1) chosen weights, -(l+1)-10000 distribution.
bool load_probs(const std::string& path, int layer, int n_experts,
                std::vector<std::vector<float>>* out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("нет трассы: %s\n", path.c_str());
        return false;
    }
    const int want_tag = -layer - 10001;
    int32_t hdr[3];
    while (fread(hdr, sizeof(int32_t), 3, f) == 3) {
        const int tag = hdr[0], nu = hdr[1], nt = hdr[2];
        const size_t n = size_t(nu) * size_t(nt);
        // The graph exposes the distribution twice, as [n_expert, tokens] and as a
        // flat view; taking only the first form avoids counting every token twice.
        if (tag != want_tag || nu != n_experts) {
            // 64-bit seek is mandatory: traces run to gigabytes and MSVC's long is 32-bit,
            // so this skip would silently wrap and land inside a record instead of failing.
#ifdef _WIN32
            if (_fseeki64(f, (__int64)(4 * n), SEEK_CUR) != 0) break;
#else
            if (fseeko(f, (off_t)(4 * n), SEEK_CUR) != 0) break;
#endif
            continue;
        }
        std::vector<float> buf(n);
        if (fread(buf.data(), sizeof(float), n, f) != n) {
            break;
        }
        for (int t = 0; t < nt; ++t) {
            out->emplace_back(buf.begin() + size_t(t) * n_experts,
                              buf.begin() + size_t(t + 1) * n_experts);
        }
    }
    fclose(f);
    return !out->empty();
}

// Top-k with a bonus for experts already held at full precision. The bonus is a
// fraction of the token's own strongest score, so it only ever resolves near-ties.
std::vector<int> choose_experts(const std::vector<float>& probs,
                                const memex::LayerRuntime& rt, int top_k,
                                float bonus) {
    const int n = int(probs.size());
    std::vector<float> score(probs);
    if (bonus > 0.0f) {
        const float mx = *std::max_element(probs.begin(), probs.end());
        for (int e = 0; e < n; ++e) {
            if (rt.expert(e).step == "source") {
                score[e] += bonus * mx;
            }
        }
    }
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return score[a] > score[b]; });
    idx.resize(top_k);
    return idx;
}

// One layer's FFN over the experts the router chose, each in whatever ggml type it
// currently holds. Tensors are wrapped around the runtime's buffers, so the graph
// always computes with the precision resident at this instant.
struct Compute {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_backend_buffer_t xbuf = nullptr;

    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (xbuf) ggml_backend_buffer_free(xbuf);
        if (ctx) ggml_free(ctx);
        alloc = nullptr;
        xbuf = nullptr;
        ctx = nullptr;
    }
};

double run_layer(const memex::LayerRuntime& rt, const std::vector<int>& ids,
                 const Shape& sh, ggml_backend_t cpu,
                 ggml_backend_buffer_type_t buft, const std::vector<float>& act) {
    Compute c;
    const size_t n_nodes = ids.size() * 8 + 64;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 64) +
                           ggml_graph_overhead_custom(n_nodes, false),
                           nullptr, true};
    c.ctx = ggml_init(ip);
    if (!c.ctx) {
        return -1.0;
    }
    ggml_tensor* x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, sh.n_embd, 1);
    c.xbuf = ggml_backend_alloc_ctx_tensors_from_buft(c.ctx, buft);
    if (!c.xbuf) {
        c.free_all();
        return -1.0;
    }
    ggml_backend_tensor_set(x, act.data(), 0, ggml_nbytes(x));

    c.gf = ggml_new_graph_custom(c.ctx, n_nodes, false);
    ggml_tensor* acc = nullptr;
    // Pin the buffers for the whole compute. Without this a promotion finishing on a
    // loader thread frees the copy this graph is reading from - which it did, as a
    // segfault, the first time the pieces ran together.
    std::vector<memex::LayerRuntime::Held> held;
    held.reserve(ids.size());
    for (int e : ids) {
        held.push_back(rt.hold(e));
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto& hv = held[i];
        if (!hv.ok) {
            continue;
        }
        // Wrap the resident bytes without copying: the runtime owns them, and the type
        // is whatever the ladder currently assigns this expert.
        ggml_tensor* up = ggml_new_tensor_2d(c.ctx, (ggml_type)hv.ggml_type[0],
                                             sh.n_embd, sh.n_ff);
        ggml_tensor* gate = ggml_new_tensor_2d(c.ctx, (ggml_type)hv.ggml_type[1],
                                               sh.n_embd, sh.n_ff);
        ggml_tensor* down = ggml_new_tensor_2d(c.ctx, (ggml_type)hv.ggml_type[2],
                                               sh.n_ff, sh.n_embd);
        up->data = hv.buf[0]->data();
        gate->data = hv.buf[1]->data();
        down->data = hv.buf[2]->data();
        ggml_tensor* a = ggml_mul_mat(c.ctx, gate, x);
        ggml_tensor* b = ggml_mul_mat(c.ctx, up, x);
        ggml_tensor* h = ggml_mul(c.ctx, ggml_silu(c.ctx, a), b);
        ggml_tensor* o = ggml_mul_mat(c.ctx, down, h);
        acc = acc ? ggml_add(c.ctx, acc, o) : o;
    }
    if (!acc) {
        c.free_all();
        return -1.0;
    }
    ggml_build_forward_expand(c.gf, acc);
    c.alloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_reserve(c.alloc, c.gf) ||
        !ggml_gallocr_alloc_graph(c.alloc, c.gf)) {
        c.free_all();
        return -1.0;
    }
    const auto t0 = Clock::now();
    ggml_backend_graph_compute(cpu, c.gf);
    const double ms = ms_since(t0);
    c.free_all();
    return ms;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string blob = "D:/MemeX/blob/experts.bin";
    std::string man = "D:/MemeX/blob/experts.json";
    std::string trace = "D:/MemeX/results/tr_p_code.bin";
    int layer = 20;
    int budget = 32;
    int threads = 4;
    int prefill = 128;          // prompt tokens used to arrange memory up front
    int max_tokens = 400;
    float bonus = 0.10f;        // measured: -36% misses for +0.73% perplexity
    int loaders = 2;
    Shape sh;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--blob") && i + 1 < argc) blob = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) man = argv[++i];
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace = argv[++i];
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) budget = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefill") && i + 1 < argc) prefill = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bonus") && i + 1 < argc) bonus = float(atof(argv[++i]));
        else if (!strcmp(argv[i], "--loaders") && i + 1 < argc) loaders = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-prefill")) prefill = 0;
    }

    memex::BlobLoader loader;
    std::string err;
    if (!loader.open(blob, man, &err)) {
        printf("блоб не открылся: %s\n", err.c_str());
        return 1;
    }
    loader.start_workers(loaders);

    std::vector<std::vector<float>> probs;
    if (!load_probs(trace, layer, sh.n_experts, &probs)) {
        printf("нет распределений роутера для слоя %d\n", layer);
        return 1;
    }
    printf("слой %d: токенов в трассе %zu, бонус %.2f, бюджет %d, загрузчиков %d\n",
           layer, probs.size(), bonus, budget, loaders);

    memex::LayerRuntime rt(&loader, uint32_t(layer), sh.n_experts);
    rt.set_full_budget(budget);
    const auto t_prime = Clock::now();
    if (!rt.prime(&err)) {
        printf("не удалось загрузить слой: %s\n", err.c_str());
        return 1;
    }
    printf("слой загружен за %.1f с, резидентно %.2f ГБ\n",
           ms_since(t_prime) / 1000.0, rt.resident_bytes() / 1e9);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);
    auto buft = ggml_backend_cpu_buffer_type();
    std::vector<float> act(size_t(sh.n_embd));
    for (size_t i = 0; i < act.size(); ++i) {
        act[i] = 0.02f * std::sin(float(i) * 0.013f);
    }

    // Prefill: the prompt's own routing, used only to arrange memory. No compute here,
    // which is the point - the information is a by-product of a pass that happens anyway.
    const int n_pre = std::min<int>(prefill, int(probs.size()));
    if (n_pre > 0) {
        std::vector<std::vector<int>> pre;
        pre.reserve(n_pre);
        for (int t = 0; t < n_pre; ++t) {
            pre.push_back(choose_experts(probs[t], rt, sh.top_k, 0.0f));
        }   // unbiased on purpose: the prompt should say what it needs, not what is here
        const auto t0 = Clock::now();
        rt.prime_from_prefill(pre);
        printf("прогрев по %d токенам промпта: %.0f мс, повышений %llu\n",
               n_pre, ms_since(t0),
               (unsigned long long)rt.stats().promotions_done);
    }

    // Generation: routing sees residency, residency follows routing.
    std::vector<double> times;
    int served_full = 0, served_total = 0;
    const int n_gen = std::min<int>(max_tokens, int(probs.size()) - n_pre);
    for (int t = 0; t < n_gen; ++t) {
        const auto& p = probs[size_t(n_pre + t)];
        // Two selections from the same distribution. The biased one is what gets
        // computed; the unbiased one is what the policy is told about. Feeding the
        // biased choice back into the statistics locks the cache onto whatever it
        // already holds - measured as 94.2% "hits" with zero promotions, because the
        // bonus made the resident set look popular by construction.
        const std::vector<int> ids = choose_experts(p, rt, sh.top_k, bonus);
        const std::vector<int> unbiased = choose_experts(p, rt, sh.top_k, 0.0f);
        const int at_source = rt.on_token(unbiased);
        // Report what the *computed* experts were served at, since that is the
        // quality that actually happened.
        int computed_full = 0;
        for (int e : ids) {
            if (rt.expert(e).step == "source") {
                computed_full++;
            }
        }
        served_full += computed_full;
        served_total += int(ids.size());
        (void)at_source;
        const double ms = run_layer(rt, ids, sh, cpu, buft, act);
        if (ms > 0.0) {
            times.push_back(ms);
        }
    }
    std::sort(times.begin(), times.end());

    const auto& st = rt.stats();
    printf("\n=== итог по %d сгенерированным токенам ===\n", n_gen);
    printf("  обслужено полной точностью: %.1f%% обращений\n",
           served_total ? 100.0 * served_full / served_total : 0.0);
    printf("  время слоя: медиана %.3f мс, мин %.3f, макс %.3f\n",
           times.empty() ? 0.0 : times[times.size() / 2],
           times.empty() ? 0.0 : times.front(), times.empty() ? 0.0 : times.back());
    printf("  повышений начато %llu, завершено %llu, отброшено %llu\n",
           (unsigned long long)st.promotions_started,
           (unsigned long long)st.promotions_done,
           (unsigned long long)st.promotions_dropped);
    printf("  понижений %llu, из них мгновенных (по сохранённой копии) %llu\n",
           (unsigned long long)st.demotions,
           (unsigned long long)st.instant_demotions);
    printf("  перебалансировок %llu, резидентно %.2f ГБ\n",
           (unsigned long long)st.rebalances, rt.resident_bytes() / 1e9);
    if (!times.empty()) {
        const double med = times[times.size() / 2];
        printf("  пересчёт на 48 слоёв: %.1f мс -> %.2f ток/с (только эксперты)\n",
               48.0 * med, 1000.0 / (48.0 * med));
    }

    loader.stop_workers();
    ggml_backend_free(cpu);
    return 0;
}
