// memex-kv - the positional zoned KV cache as a self-contained module, with its own
// unit test in the same binary.
//
// The module is zoned_cache.hpp/.cpp; this file is the proof that it computes attention.
// Nothing here needs the 30B model: K, V and the queries are synthesised from a fixed
// seed, exact attention over the same positions is the reference, and the zoned result is
// compared against it. That is deliberate. The zoned cache is a change to *what attention
// reads*, and whether that changes the answer is a question about arithmetic, not about
// any particular model - so it should be answerable in a second on a busy machine rather
// than behind a 17 GB load.
//
// What this test is built to catch, in order of how badly the project has been bitten:
//
//  1. A comparison against nothing. An earlier probe in this project had a reference
//     graph that produced no output at all, and every comparison against it read as a
//     perfect match, because the difference of two zeros is zero. So every comparison
//     here goes through one guard that refuses a zero-norm or wrong-shaped operand
//     instead of dividing, prints both operands' norms next to the number, and makes the
//     process exit non-zero. The guard is itself tested, at the top of main, by being fed
//     a zero reference and a wrong-shaped one and being required to refuse both.
//
//  2. A per-zone softmax. Concatenating the zones' scores and normalising once is the
//     whole correctness claim of the module, and normalising per zone computes a
//     different function that still looks like attention. This does not assert that the
//     softmax is global. It counts the GGML_OP_SOFT_MAX nodes in the built graph, it sums
//     the probabilities over all four zones and requires one, and it reconstructs exactly
//     what a per-zone softmax would have answered - from the same probabilities, so there
//     is no second implementation to get wrong - and prints the error that mistake costs.
//     If that number were small the test would be worthless; it is not small.
//
//  3. Flattering queries. A gaussian query spreads attention almost evenly over the
//     context, which makes any compression of the tail look excellent, because with a
//     near-uniform distribution the weighted sum is a mean and quantisation error
//     averages out. The synthetic keys here are drawn around a small number of topic
//     directions and the queries are drawn from the same set, so the distribution is
//     peaked - and the peak weight is printed so the reader can see which regime the
//     error came from.
//
// The four traps that this project has already paid for, and where they are answered:
// the F16 matmul is wrong when the reduction length is not a multiple of four, so every
// zone length is padded to 32 and the padding masked with -inf (zoned_cache.hpp,
// kZonePad); grouped-query views must be three-dimensional with heads last, or mul_mat's
// broadcast silently pairs the wrong head (create_tensors); value blocks run along
// positions and key blocks along head_dim (write_exact); the Hadamard rotation applies to
// keys only, and the query is rotated to match inside the graph (build_attn).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "zoned_cache.hpp"

namespace {

using memex::zc_f2h;
using memex::zc_h2f;
using memex::zone_pad_up;

// ---------------------------------------------------------------------------------------
// The comparison guard. Same shape and the same refusal messages as memex-test's, because
// it is the same lesson: a comparison that cannot be made must be refused, not answered
// with a number that looks fine.
// ---------------------------------------------------------------------------------------

struct Cmp {
    bool ok = false;             // false means the comparison was refused
    const char* why = "";
    std::size_t n_ours = 0;
    std::size_t n_ref = 0;
    double rel_l2 = -1.0;        // ||ours - ref|| / ||ref||
    double max_abs = 0.0;
    double norm_ours = 0.0;
    double norm_ref = 0.0;
};

Cmp compare(const std::vector<float>& ours, const std::vector<float>& ref) {
    Cmp r;
    r.n_ours = ours.size();
    r.n_ref = ref.size();
    if (ours.empty() || ref.empty()) {
        r.why = "пустой операнд";
        return r;
    }
    // Refused, not truncated and not broadcast. Comparing the overlap of two different
    // shapes once put our first token against the reference's last and reported a 209%
    // divergence that did not exist.
    if (ours.size() != ref.size()) {
        r.why = "размеры не совпадают";
        return r;
    }
    double num = 0.0, den = 0.0, sq_ours = 0.0, mx = 0.0;
    for (std::size_t i = 0; i < ours.size(); ++i) {
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
        mx = std::max(mx, std::abs(d));
    }
    r.norm_ours = std::sqrt(sq_ours);
    r.norm_ref = std::sqrt(den);
    r.max_abs = mx;
    // Both zero-norm cases are refusals. A zero reference makes the relative error
    // meaningless and would print as a perfect match; a zero candidate against a live
    // reference would print as 100% and read as an ordinary bad number, when in fact our
    // side produced nothing at all.
    if (!(den > 0.0)) {
        r.why = "норма эталона равна нулю — сравнивать не с чем";
        return r;
    }
    if (!(sq_ours > 0.0)) {
        r.why = "норма нашего результата равна нулю — мы ничего не посчитали";
        return r;
    }
    r.rel_l2 = std::sqrt(num / den);
    r.ok = true;
    return r;
}

void print_cmp(const char* label, const Cmp& c) {
    if (!c.ok) {
        printf("  %-30s ОТКАЗ: %s (наш %zu знач., эталон %zu знач., "
               "|наш| %.6g, |эталон| %.6g)\n",
               label, c.why, c.n_ours, c.n_ref, c.norm_ours, c.norm_ref);
        return;
    }
    printf("  %-30s n %7zu  отн.L2 %9.5f%%  макс|d| %10.6f  |наш| %10.4f  "
           "|эталон| %10.4f\n",
           label, c.n_ours, 100.0 * c.rel_l2, c.max_abs, c.norm_ours, c.norm_ref);
}

// The guard checking itself. Without this the guard is just more code that could be
// wrong in the same direction as the thing it guards, and the failure it exists to
// prevent is exactly the one that produces no symptom.
bool guard_selftest() {
    printf("проверка самой защиты (каждый случай обязан быть ОТКАЗАН):\n");
    std::vector<float> live;
    live.assign(8, 0.0f);
    for (int i = 0; i < 8; ++i) live[size_t(i)] = float(i + 1);
    std::vector<float> zeros;
    zeros.assign(8, 0.0f);
    std::vector<float> shorter;
    shorter.assign(6, 1.0f);
    std::vector<float> nans;
    nans.assign(8, std::nanf(""));

    struct Case { const char* name; const std::vector<float>* a; const std::vector<float>* b; };
    const Case bad[5] = {
        {"эталон из нулей",      &live,   &zeros},
        {"наш результат нулевой", &zeros, &live},
        {"размеры не совпадают",  &live,  &shorter},
        {"NaN в эталоне",         &live,  &nans},
        {"пустой операнд",        &live,  nullptr},
    };
    bool all_refused = true;
    std::vector<float> empty;
    for (const Case& c : bad) {
        const Cmp r = compare(*c.a, c.b ? *c.b : empty);
        print_cmp(c.name, r);
        if (r.ok) {
            printf("    ПРОВАЛ: этот случай прошёл, хотя должен был быть отказан — "
                   "любой отчёт этой программы теперь ничего не стоит\n");
            all_refused = false;
        }
        // The point of the guard is that it never *reports* zero error on nothing.
        if (r.ok && r.rel_l2 == 0.0) {
            printf("    ПРОВАЛ: отчёт 0%% на пустом сравнении\n");
            all_refused = false;
        }
    }
    // And the positive control, so a guard that refuses everything is not mistaken for a
    // working one.
    const Cmp same = compare(live, live);
    print_cmp("контроль: сам с собой", same);
    if (!same.ok || same.rel_l2 != 0.0) {
        printf("    ПРОВАЛ: защита отказала в законном сравнении\n");
        all_refused = false;
    }
    printf("  защита %s\n\n", all_refused ? "работает" : "СЛОМАНА");
    return all_refused;
}

// ---------------------------------------------------------------------------------------
// Synthetic context. Fixed seed, so the whole run is reproducible from the command line.
// ---------------------------------------------------------------------------------------

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return uint32_t(s >> 32);
    }
    float unit() { return float(next()) * (1.0f / 4294967296.0f); }
    // Box-Muller, one value per call. The second value is thrown away on purpose: keeping
    // it would make the stream depend on how many times the caller happened to ask, so
    // the same seed would stop meaning the same data.
    float normal() {
        float u1 = unit();
        if (u1 < 1e-7f) u1 = 1e-7f;
        const float u2 = unit();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
    }
};

void normalise(float* x, int n) {
    double a = 0.0;
    for (int i = 0; i < n; ++i) a += double(x[i]) * double(x[i]);
    a = a > 0.0 ? 1.0 / std::sqrt(a) : 0.0;
    for (int i = 0; i < n; ++i) x[i] = float(double(x[i]) * a);
}

struct Synth {
    int n_pos = 0;
    int n_queries = 0;
    std::vector<float> k, v;        // [n_pos, d_kv], raw
    std::vector<uint16_t> kh, vh;   // the same rows in fp16, which is what the cache and
                                    // the reference graph actually hold
    std::vector<uint8_t> keep;      // the engine's notebook flag, per position
    std::vector<float> q;           // [n_queries, head_dim * n_q_heads]
};

// Keys clustered around a few directions per kv-head, queries drawn from the same
// directions. This is the difference between a test that means something and one that
// does not: with gaussian queries the reference distribution is nearly uniform, the
// weighted sum degenerates into a mean, and quantisation error in the tail averages away.
void synthesise(const memex::ZonedCacheParams& p, int n_pos, int n_queries, int n_topics,
                double peak, uint64_t seed, Synth* out) {
    Rng rng(seed);
    const int hd = p.head_dim;
    const int d = p.d_kv();
    out->n_pos = n_pos;
    out->n_queries = n_queries;

    // Topic directions, per kv-head, unit norm.
    std::vector<float> topic;
    topic.assign(size_t(n_topics) * size_t(p.n_kv_heads) * size_t(hd), 0.0f);
    for (int t = 0; t < n_topics; ++t) {
        for (int g = 0; g < p.n_kv_heads; ++g) {
            float* dst = topic.data() +
                         (size_t(t) * size_t(p.n_kv_heads) + size_t(g)) * size_t(hd);
            for (int j = 0; j < hd; ++j) dst[j] = rng.normal();
            normalise(dst, hd);
        }
    }
    // Value directions, correlated with the topic, so that getting the *weights* right
    // matters. Values independent of the key would make any error in the distribution
    // invisible in the output.
    std::vector<float> vtopic;
    vtopic.assign(topic.size(), 0.0f);
    for (int t = 0; t < n_topics; ++t) {
        for (int g = 0; g < p.n_kv_heads; ++g) {
            float* dst = vtopic.data() +
                         (size_t(t) * size_t(p.n_kv_heads) + size_t(g)) * size_t(hd);
            for (int j = 0; j < hd; ++j) dst[j] = rng.normal();
            normalise(dst, hd);
        }
    }

    out->k.assign(size_t(n_pos) * size_t(d), 0.0f);
    out->v.assign(size_t(n_pos) * size_t(d), 0.0f);
    out->keep.assign(size_t(n_pos), 0);
    for (int pos = 0; pos < n_pos; ++pos) {
        const int t = int(rng.next() % uint32_t(n_topics));
        // A per-position lognormal gain on the topic component. Without it every key
        // sharing a topic scores identically, the softmax comes out uniform over the
        // ~n_pos/n_topics matching positions, and the weighted sum degenerates into a
        // mean where quantisation error cancels. The first version of this test had a
        // peak weight of 1.63% and reported 0.37% output error for exactly that reason -
        // a number about a regime real attention is never in. The gain gives the scores
        // a spread, so a few positions carry the answer and their representation has to
        // be right rather than merely unbiased.
        const float gp = std::exp(0.45f * rng.normal());
        for (int g = 0; g < p.n_kv_heads; ++g) {
            const float* kt = topic.data() +
                              (size_t(t) * size_t(p.n_kv_heads) + size_t(g)) * size_t(hd);
            const float* vt = vtopic.data() +
                              (size_t(t) * size_t(p.n_kv_heads) + size_t(g)) * size_t(hd);
            float* kd = out->k.data() + size_t(pos) * size_t(d) + size_t(g) * size_t(hd);
            float* vd = out->v.data() + size_t(pos) * size_t(d) + size_t(g) * size_t(hd);
            for (int j = 0; j < hd; ++j) {
                kd[j] = gp * kt[j] + 0.35f * rng.normal() / std::sqrt(float(hd));
                vd[j] = vt[j] + 0.50f * rng.normal() / std::sqrt(float(hd));
            }
        }
        // One position in sixteen is flagged for the notebook. Sparse, because that is
        // what a real policy produces: most of the past is not worth keeping exactly.
        out->keep[size_t(pos)] = (rng.next() % 16u) == 0u ? 1 : 0;
    }

    // fp16 copies. Both the reference and the cache read these, so the F16 rounding is
    // not counted as zoning error - which it is not.
    out->kh.assign(out->k.size(), 0);
    out->vh.assign(out->v.size(), 0);
    for (size_t i = 0; i < out->k.size(); ++i) {
        out->kh[i] = zc_f2h(out->k[i]);
        out->vh[i] = zc_f2h(out->v[i]);
    }

    // Queries. Each query head gets a topic direction of its own kv-head, scaled so the
    // logits reach roughly `peak` before the softmax: q.k / sqrt(head_dim) with unit-norm
    // k means the gain has to carry the sqrt.
    const double gain = peak * std::sqrt(double(hd));
    out->q.assign(size_t(n_queries) * size_t(hd) * size_t(p.n_q_heads), 0.0f);
    for (int qi = 0; qi < n_queries; ++qi) {
        for (int h = 0; h < p.n_q_heads; ++h) {
            const int g = h / p.ratio();
            const int t = int(rng.next() % uint32_t(n_topics));
            const float* kt = topic.data() +
                              (size_t(t) * size_t(p.n_kv_heads) + size_t(g)) * size_t(hd);
            float* dst = out->q.data() +
                         size_t(qi) * size_t(hd) * size_t(p.n_q_heads) +
                         size_t(h) * size_t(hd);
            for (int j = 0; j < hd; ++j) dst[j] = kt[j] + 0.30f * rng.normal() /
                                                  std::sqrt(float(hd));
            normalise(dst, hd);
            for (int j = 0; j < hd; ++j) dst[j] = float(double(dst[j]) * gain);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Exact attention: the reference, twice. Once on the host and once as a ggml graph, so
// that the reference itself is checked before anything is compared against it.
// ---------------------------------------------------------------------------------------

void host_exact(const memex::ZonedCacheParams& p, const std::vector<uint16_t>& kh,
                const std::vector<uint16_t>& vh, int n_pos, const float* q,
                std::vector<float>* out, double* peak_weight) {
    const int hd = p.head_dim;
    const int d = p.d_kv();
    const float sc = 1.0f / std::sqrt(float(hd));
    out->assign(size_t(hd) * size_t(p.n_q_heads), 0.0f);
    std::vector<double> s;
    s.resize(size_t(n_pos));
    double peak = 0.0;
    for (int h = 0; h < p.n_q_heads; ++h) {
        const size_t hoff = size_t(h / p.ratio()) * size_t(hd);
        const float* qh = q + size_t(h) * size_t(hd);
        double mx = -1e300;
        for (int pos = 0; pos < n_pos; ++pos) {
            double acc = 0.0;
            for (int j = 0; j < hd; ++j) {
                acc += double(zc_h2f(kh[size_t(pos) * size_t(d) + hoff + size_t(j)])) *
                       double(qh[j]);
            }
            s[size_t(pos)] = acc * double(sc);
            mx = std::max(mx, s[size_t(pos)]);
        }
        double sum = 0.0, top = 0.0;
        for (int pos = 0; pos < n_pos; ++pos) {
            const double e = std::exp(s[size_t(pos)] - mx);
            s[size_t(pos)] = e;
            sum += e;
            top = std::max(top, e);
        }
        float* oh = out->data() + size_t(h) * size_t(hd);
        for (int pos = 0; pos < n_pos; ++pos) {
            const double w = s[size_t(pos)] / sum;
            for (int j = 0; j < hd; ++j) {
                oh[j] += float(w * double(zc_h2f(
                    vh[size_t(pos) * size_t(d) + hoff + size_t(j)])));
            }
        }
        peak = std::max(peak, top / sum);
    }
    if (peak_weight) *peak_weight = peak;
}

struct Arena {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;

    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        alloc = nullptr;
        buf = nullptr;
        ctx = nullptr;
        gf = nullptr;
    }
};

constexpr size_t kNodes = 512;

ggml_context* make_ctx() {
    ggml_init_params ip = {ggml_tensor_overhead() * kNodes +
                           ggml_graph_overhead_custom(kNodes, false),
                           nullptr, true};
    return ggml_init(ip);
}

int count_softmax(const ggml_cgraph* gf) {
    int n = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->op == GGML_OP_SOFT_MAX) n++;
    }
    return n;
}

// Whole context, F16, one ordinary softmax. Padded to 32 and masked for the same reason
// the zoned graph is: the fork's F16 matmul is wrong when the reduction length is not a
// multiple of four, and it does not complain.
bool build_exact_graph(Arena* a, ggml_backend_buffer_type_t buft,
                       const memex::ZonedCacheParams& p,
                       const std::vector<uint16_t>& kh, const std::vector<uint16_t>& vh,
                       int n_pos, ggml_tensor** q_out, ggml_tensor** out_out) {
    a->ctx = make_ctx();
    if (!a->ctx) return false;
    const int hd = p.head_dim;
    const int d = p.d_kv();
    const int n_pad = zone_pad_up(n_pos);

    ggml_tensor* K = ggml_new_tensor_3d(a->ctx, GGML_TYPE_F16, hd, n_pad, p.n_kv_heads);
    ggml_tensor* V = ggml_new_tensor_3d(a->ctx, GGML_TYPE_F16, n_pad, hd, p.n_kv_heads);
    ggml_tensor* M = ggml_new_tensor_2d(a->ctx, GGML_TYPE_F32, n_pad, 1);
    ggml_tensor* q = ggml_new_tensor_3d(a->ctx, GGML_TYPE_F32, hd, 1, p.n_q_heads);
    a->buf = ggml_backend_alloc_ctx_tensors_from_buft(a->ctx, buft);
    if (!a->buf) return false;

    std::vector<uint16_t> k3, v3;
    k3.assign(size_t(n_pad) * size_t(d), 0);
    v3.assign(size_t(n_pad) * size_t(d), 0);
    for (int g = 0; g < p.n_kv_heads; ++g) {
        for (int pos = 0; pos < n_pos; ++pos) {
            for (int j = 0; j < hd; ++j) {
                const size_t src = size_t(pos) * size_t(d) + size_t(g) * size_t(hd) +
                                   size_t(j);
                k3[size_t(g) * size_t(n_pad) * size_t(hd) + size_t(pos) * size_t(hd) +
                   size_t(j)] = kh[src];
                v3[size_t(g) * size_t(hd) * size_t(n_pad) + size_t(j) * size_t(n_pad) +
                   size_t(pos)] = vh[src];
            }
        }
    }
    ggml_backend_tensor_set(K, k3.data(), 0, ggml_nbytes(K));
    ggml_backend_tensor_set(V, v3.data(), 0, ggml_nbytes(V));
    std::vector<float> mask;
    mask.assign(size_t(n_pad), 0.0f);
    for (int i = n_pos; i < n_pad; ++i) mask[size_t(i)] = -INFINITY;
    ggml_backend_tensor_set(M, mask.data(), 0, ggml_nbytes(M));

    a->gf = ggml_new_graph_custom(a->ctx, kNodes, false);
    const float scale = 1.0f / std::sqrt(float(hd));
    ggml_tensor* kq = ggml_mul_mat(a->ctx, K, q);
    ggml_tensor* pr = ggml_soft_max_ext(a->ctx, kq, M, scale, 0.0f);
    ggml_tensor* out = ggml_mul_mat(a->ctx, V, pr);
    ggml_set_output(out);
    ggml_build_forward_expand(a->gf, out);
    a->alloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_reserve(a->alloc, a->gf) ||
        !ggml_gallocr_alloc_graph(a->alloc, a->gf)) {
        return false;
    }
    *q_out = q;
    *out_out = out;
    return true;
}

// ---------------------------------------------------------------------------------------
// One zoned configuration, built, filled, run and measured.
// ---------------------------------------------------------------------------------------

struct Row {
    memex::ZonedCacheParams p;
    memex::ZoneOccupancy occ;
    std::size_t bytes = 0;
    std::size_t bytes_exact = 0;
    double rel = 0.0;            // mean relative L2 against exact attention
    double worst = 0.0;
    double rel_perzone = -1.0;   // what a softmax per zone would have cost instead
    double mass_lo = 0.0;        // smallest total probability mass over query heads
    double mass_hi = 0.0;        // largest
    double tail_mass = 0.0;      // mean share of the probability the tail carries
    // Carried into the table so that no error figure this program prints is ever
    // separated from the norms of the two things it was computed from. A row whose
    // |ours| or |ref| is zero cannot have an honest 0% error, and the reader can see it.
    double norm_ours = 0.0;
    double norm_ref = 0.0;
    int n_softmax = -1;
    // The last query's output, kept so that main can turn the guard loose on a real
    // attention result rather than only on the toy vectors of the self-test. A guard that
    // has only ever been shown hand-made operands has not been shown the thing it guards.
    std::vector<float> last;
    bool ok = false;
    std::string why;
};

// Deliberate sabotage, off by default. The point is not to test the cache; it is to test
// the test. A harness that has only ever been run against working code has never
// demonstrated that it can fail, and "all comparisons passed" from such a harness is not
// evidence of anything. With --sabotage the program inverts its own verdict: it requires
// the error to be large and fails if the mistake went unnoticed.
//
// The sabotage is head mis-pairing: query head h is fed the query of head h + ratio,
// which belongs to a different kv-group. That is not an arbitrary perturbation, it is
// this project's second trap - expressing grouped attention as a two-dimensional strided
// view once produced exactly this, with the first head of each group behaving differently
// from the other three. A weaker sabotage was tried first, scaling the query by 1.02, and
// it moved the output by only 0.65%: with a peaked softmax a small change of temperature
// barely changes the answer, so it would have set the bar for "noticed" absurdly low.
bool g_sabotage_heads = false;

// qs scales the queries, which is the softmax temperature in disguise: the same cache and
// the same reference at a different peakedness. The same error means different things at
// different temperatures, so no single number is reported without one.
bool run_config(ggml_backend_t be, ggml_backend_buffer_type_t buft,
                const memex::ZonedCacheParams& p, const Synth& syn, double qs,
                const std::vector<std::vector<float>>& refs, bool verbose, Row* row) {
    row->p = p;
    std::string err;
    if (!p.validate(&err)) {
        row->why = err;
        return false;
    }
    memex::ZonedCache cache(p);
    Arena a;
    a.ctx = make_ctx();
    if (!a.ctx) {
        row->why = "ggml_init не дал контекст";
        return false;
    }
    if (!cache.create_tensors(a.ctx, &err)) {
        row->why = err;
        a.free_all();
        return false;
    }
    ggml_tensor* q = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, p.head_dim, 1, p.n_q_heads);
    a.buf = ggml_backend_alloc_ctx_tensors_from_buft(a.ctx, buft);
    if (!a.buf) {
        row->why = "буфер бэкенда не выделился";
        a.free_all();
        return false;
    }

    const int d = p.d_kv();
    for (int pos = 0; pos < syn.n_pos; ++pos) {
        cache.append(0, syn.k.data() + size_t(pos) * size_t(d),
                     syn.v.data() + size_t(pos) * size_t(d),
                     syn.keep[size_t(pos)] != 0);
    }
    cache.flush();
    if (!cache.upload(0, &err)) {
        row->why = err;
        a.free_all();
        return false;
    }
    row->occ = cache.occupancy(0);
    row->bytes = cache.read_bytes(0);
    row->bytes_exact = cache.read_bytes_if_exact(0);
    if (row->occ.total() != syn.n_pos || row->occ.dropped != 0) {
        // The zones must between them hold every position, or the comparison is against
        // a different context and the error would be meaningless rather than large.
        char buf[192];
        snprintf(buf, sizeof(buf),
                 "зоны держат %d позиций из %d (потеряно %d)",
                 row->occ.total(), syn.n_pos, row->occ.dropped);
        row->why = buf;
        a.free_all();
        return false;
    }

    a.gf = ggml_new_graph_custom(a.ctx, kNodes, false);
    memex::ZonedAttn at = cache.build_attn(a.ctx, 0, q);
    if (!at.out || !at.probs || !at.out_exact || !at.out_tail) {
        row->why = "зонный подграф не собрался";
        a.free_all();
        return false;
    }
    ggml_set_output(at.out);
    ggml_set_output(at.probs);
    ggml_set_output(at.out_exact);
    ggml_set_output(at.out_tail);
    ggml_build_forward_expand(a.gf, at.out);
    a.alloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_reserve(a.alloc, a.gf) ||
        !ggml_gallocr_alloc_graph(a.alloc, a.gf)) {
        row->why = "граф не разместился";
        a.free_all();
        return false;
    }
    // Counted, not assumed. One softmax node in the whole subgraph is what "global"
    // means structurally; the probability mass check below is what it means numerically.
    row->n_softmax = count_softmax(a.gf);

    const int n_out = p.head_dim * p.n_q_heads;
    std::vector<float> got, oex, otl, prob, wrong, qbuf;
    qbuf.resize(size_t(n_out));
    got.resize(size_t(n_out));
    oex.resize(size_t(n_out));
    otl.resize(size_t(n_out));
    wrong.resize(size_t(n_out));
    prob.resize(size_t(at.n_slots) * size_t(p.n_q_heads));

    double rel_sum = 0.0, worst = 0.0, tail_mass_sum = 0.0, perzone_sum = 0.0;
    int perzone_n = 0;
    row->mass_lo = 1e300;
    row->mass_hi = -1e300;
    for (int qi = 0; qi < syn.n_queries; ++qi) {
        const float* qsrc = syn.q.data() + size_t(qi) * size_t(n_out);
        for (int h = 0; h < p.n_q_heads; ++h) {
            const int sh = g_sabotage_heads ? (h + p.ratio()) % p.n_q_heads : h;
            for (int j = 0; j < p.head_dim; ++j) {
                qbuf[size_t(h) * size_t(p.head_dim) + size_t(j)] =
                    float(qs * double(qsrc[size_t(sh) * size_t(p.head_dim) + size_t(j)]));
            }
        }
        ggml_backend_tensor_set(q, qbuf.data(), 0, ggml_nbytes(q));
        ggml_backend_graph_compute(be, a.gf);
        ggml_backend_synchronize(be);
        ggml_backend_tensor_get(at.out, got.data(), 0, ggml_nbytes(at.out));
        ggml_backend_tensor_get(at.out_exact, oex.data(), 0, ggml_nbytes(at.out_exact));
        ggml_backend_tensor_get(at.out_tail, otl.data(), 0, ggml_nbytes(at.out_tail));
        ggml_backend_tensor_get(at.probs, prob.data(), 0, ggml_nbytes(at.probs));

        const Cmp c = compare(got, refs[size_t(qi)]);
        if (!c.ok) {
            row->why = c.why;
            if (verbose) print_cmp("зоны против точного", c);
            a.free_all();
            return false;
        }
        rel_sum += c.rel_l2;
        worst = std::max(worst, c.max_abs);
        row->norm_ours += c.norm_ours / syn.n_queries;
        row->norm_ref += c.norm_ref / syn.n_queries;

        // One distribution over all four zones, or four distributions? The masses answer
        // it: a single softmax puts the whole unit of probability across the concatenated
        // slots, so the total is one and each zone's share is less than one. A per-zone
        // softmax would give every zone a full unit and this total would be two.
        for (int h = 0; h < p.n_q_heads; ++h) {
            const float* ph = prob.data() + size_t(h) * size_t(at.n_slots);
            double s_ex = 0.0, s_tl = 0.0;
            for (int i = 0; i < at.exact_slots; ++i) s_ex += double(ph[i]);
            for (int i = at.exact_slots; i < at.n_slots; ++i) s_tl += double(ph[i]);
            const double tot = s_ex + s_tl;
            row->mass_lo = std::min(row->mass_lo, tot);
            row->mass_hi = std::max(row->mass_hi, tot);
            tail_mass_sum += s_tl / (tot > 0.0 ? tot : 1.0);
            // Reconstruct, exactly, the answer a per-zone softmax would have produced.
            // Renormalising each zone's share of the same probabilities *is* the per-zone
            // softmax: the shared maximum subtraction cancels in the ratio. So this is
            // not an approximation of the mistake, it is the mistake.
            if (s_ex > 1e-30 && s_tl > 1e-30) {
                float* w = wrong.data() + size_t(h) * size_t(p.head_dim);
                const float* ae = oex.data() + size_t(h) * size_t(p.head_dim);
                const float* atl = otl.data() + size_t(h) * size_t(p.head_dim);
                for (int j = 0; j < p.head_dim; ++j) {
                    w[j] = float(double(ae[j]) / s_ex + double(atl[j]) / s_tl);
                }
            } else {
                float* w = wrong.data() + size_t(h) * size_t(p.head_dim);
                const float* g = got.data() + size_t(h) * size_t(p.head_dim);
                for (int j = 0; j < p.head_dim; ++j) w[j] = g[j];
            }
        }
        const Cmp cw = compare(wrong, refs[size_t(qi)]);
        if (cw.ok) {
            perzone_sum += cw.rel_l2;
            perzone_n++;
        }
        if (verbose) {
            char lbl[64];
            snprintf(lbl, sizeof(lbl), "запрос %d: зоны/точно", qi);
            print_cmp(lbl, c);
            snprintf(lbl, sizeof(lbl), "запрос %d: по зонам softmax", qi);
            print_cmp(lbl, cw);
        }
    }
    row->last = got;
    row->rel = rel_sum / syn.n_queries;
    row->worst = worst;
    row->tail_mass = tail_mass_sum / (syn.n_queries * p.n_q_heads);
    row->rel_perzone = perzone_n ? perzone_sum / perzone_n : -1.0;
    row->ok = true;
    a.free_all();
    return true;
}

// Exact attention for every query at one temperature, computed twice - as a ggml graph
// and on the host - and refused unless the two agree. A reference that is not checked
// against a second implementation of itself is just an opinion, and this project has
// already had one broken reference grade everything else as perfect.
struct RefSet {
    std::vector<std::vector<float>> out;
    double peak = 0.0;      // largest single attention weight anywhere
    bool ok = false;
};

RefSet make_refs(ggml_backend_t be, const memex::ZonedCacheParams& p, const Synth& syn,
                 ggml_cgraph* gf, ggml_tensor* exq, ggml_tensor* exout, double qs,
                 bool verbose, bool quiet) {
    RefSet rs;
    rs.ok = true;
    const int n_out = p.head_dim * p.n_q_heads;
    std::vector<float> qbuf;
    qbuf.resize(size_t(n_out));
    for (int qi = 0; qi < syn.n_queries; ++qi) {
        const float* qsrc = syn.q.data() + size_t(qi) * size_t(n_out);
        for (int j = 0; j < n_out; ++j) qbuf[size_t(j)] = float(qs * double(qsrc[j]));
        ggml_backend_tensor_set(exq, qbuf.data(), 0, ggml_nbytes(exq));
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        std::vector<float> g;
        g.resize(size_t(n_out));
        ggml_backend_tensor_get(exout, g.data(), 0, ggml_nbytes(exout));

        std::vector<float> h;
        double pw = 0.0;
        host_exact(p, syn.kh, syn.vh, syn.n_pos, qbuf.data(), &h, &pw);
        rs.peak = std::max(rs.peak, pw);
        const Cmp c = compare(g, h);
        if ((qi == 0 && !quiet) || verbose || !c.ok || c.rel_l2 > 1e-3) {
            char lbl[64];
            snprintf(lbl, sizeof(lbl), "масштаб %.2f, запрос %d", qs, qi);
            print_cmp(lbl, c);
        }
        if (!c.ok || c.rel_l2 > 1e-3) {
            printf("    ПРОВАЛ: эталон не согласен сам с собой — сравнивать не с чем\n");
            rs.ok = false;
        }
        rs.out.push_back(g);
    }
    return rs;
}

const char* tail_name(const memex::ZonedCacheParams& p) {
    if (!p.tail_q8) return "f16";
    return p.rotate_keys ? "q8_0+H" : "q8_0";
}

void print_row(const Row& r) {
    if (!r.ok) {
        printf("%6d %7d %7d %8s   ОТКАЗ: %s\n", r.p.n_sinks, r.p.notebook_cap,
               r.p.window, tail_name(r.p), r.why.c_str());
        return;
    }
    printf("%6d %7d %7d %8s %6d %6d %9.1f %7.2fx %10.4f%% %10.2f%% %7.3f "
           "%9.4f %9.4f\n",
           r.p.n_sinks, r.occ.notebook, r.p.window, tail_name(r.p),
           r.occ.sinks + r.occ.notebook + r.occ.window, r.occ.tail,
           double(r.bytes) / 1024.0,
           double(r.bytes_exact) / double(r.bytes ? r.bytes : 1),
           100.0 * r.rel, 100.0 * r.rel_perzone, r.tail_mass,
           r.norm_ours, r.norm_ref);
}

void print_head() {
    printf("%6s %7s %7s %8s %6s %6s %9s %8s %11s %11s %7s %9s %9s\n",
           "стоки", "блокнот", "окно", "хвост", "точно", "хвост",
           "КБ/ток", "экономия", "ош.L2", "по зонам", "вес хв.",
           "|наш|", "|эталон|");
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    memex::ZonedCacheParams p;
    p.n_layers = 1;
    p.n_q_heads = 8;
    p.n_kv_heads = 2;
    p.head_dim = 128;
    p.n_ctx = 1024;
    p.n_sinks = 4;
    p.notebook_cap = 32;
    p.window = 256;
    p.tail_q8 = true;
    p.rotate_keys = true;

    uint64_t seed = 20260821;
    int n_queries = 8;
    int n_topics = 8;
    double peak = 4.0;
    int threads = 4;
    bool verbose = false;
    bool sweep = true;
    bool sabotage = false;

    auto on_off = [](const char* s) { return strcmp(s, "on") == 0 || strcmp(s, "1") == 0; };
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto has = [&](const char* f) { return strcmp(a, f) == 0 && i + 1 < argc; };
        if (has("--sinks")) p.n_sinks = atoi(argv[++i]);
        else if (has("--notebook")) p.notebook_cap = atoi(argv[++i]);
        else if (has("--window")) p.window = atoi(argv[++i]);
        else if (has("--ctx")) p.n_ctx = atoi(argv[++i]);
        else if (has("--heads")) p.n_q_heads = atoi(argv[++i]);
        else if (has("--kv-heads")) p.n_kv_heads = atoi(argv[++i]);
        else if (has("--head-dim")) p.head_dim = atoi(argv[++i]);
        else if (has("--tail-form")) p.tail_q8 = strcmp(argv[++i], "f16") != 0;
        else if (has("--rotate-keys")) p.rotate_keys = on_off(argv[++i]);
        else if (has("--seed")) seed = strtoull(argv[++i], nullptr, 10);
        else if (has("--queries")) n_queries = atoi(argv[++i]);
        else if (has("--topics")) n_topics = atoi(argv[++i]);
        else if (has("--peak")) peak = atof(argv[++i]);
        else if (has("--threads")) threads = atoi(argv[++i]);
        else if (strcmp(a, "--verbose") == 0) verbose = true;
        else if (strcmp(a, "--no-sweep") == 0) sweep = false;
        else if (strcmp(a, "--sabotage") == 0) { sabotage = true; sweep = false; }
        else if (strcmp(a, "--help") == 0) {
            printf("memex-kv [--ctx N] [--sinks N] [--window N] [--notebook N]\n"
                   "         [--heads N] [--kv-heads N] [--head-dim N]\n"
                   "         [--tail-form q8_0|f16] [--rotate-keys on|off]\n"
                   "         [--seed N] [--queries N] [--topics N] [--peak X]\n"
                   "         [--threads N] [--verbose] [--no-sweep] [--sabotage]\n"
                   "--sabotage портит запрос зонного графа и требует, чтобы тест это "
                   "заметил: проверка того, что проверка вообще способна провалиться\n");
            return 0;
        } else {
            printf("неизвестный аргумент: %s (см. --help)\n", a);
            return 2;
        }
    }

    int failures = 0;

    // Before anything is measured, the thing that decides whether a measurement can be
    // trusted at all.
    if (!guard_selftest()) {
        printf("защита сравнений сломана — дальше идти незачем\n");
        return 1;
    }

    std::string err;
    if (!p.validate(&err)) {
        printf("конфигурация отклонена: %s\n", err.c_str());
        return 2;
    }

    printf("синтетика: контекст %d, голов запроса %d, kv-голов %d, head_dim %d, "
           "тем %d, seed %llu\n",
           p.n_ctx, p.n_q_heads, p.n_kv_heads, p.head_dim, n_topics,
           (unsigned long long)seed);

    Synth syn;
    synthesise(p, p.n_ctx, n_queries, n_topics, peak, seed, &syn);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

    // The reference, and then the reference checked against a second implementation of
    // itself. Skipping this step is how a broken reference gets to grade everything else.
    Arena ex;
    ggml_tensor* exq = nullptr;
    ggml_tensor* exout = nullptr;
    if (!build_exact_graph(&ex, buft, p, syn.kh, syn.vh, p.n_ctx, &exq, &exout)) {
        printf("эталонный граф не собрался\n");
        ex.free_all();
        ggml_backend_free(be);
        return 1;
    }
    printf("эталонный граф: узлов %d, из них softmax %d\n",
           ex.gf->n_nodes, count_softmax(ex.gf));

    printf("\nэталон против самого себя (ggml против хоста, одни и те же fp16 строки):\n");
    RefSet ref1 = make_refs(be, p, syn, ex.gf, exq, exout, 1.0, verbose, false);
    printf("  максимальный вес в эталонном распределении: %.2f%%  "
           "(плоское распределение сделало бы любое сжатие безупречным)\n",
           100.0 * ref1.peak);
    if (ref1.peak < 0.02) {
        // Not a failure - the numbers below are still true - but they would be true about
        // a regime attention is never in, so say so instead of letting the reader assume.
        printf("    ВНИМАНИЕ: распределение почти плоское, ошибки ниже занижены; "
               "поднимите --peak или уменьшите --topics\n");
    }
    if (!ref1.ok) {
        ex.free_all();
        ggml_backend_free(be);
        return 1;
    }
    const std::vector<std::vector<float>>& refs = ref1.out;

    if (sabotage) {
        g_sabotage_heads = true;
        printf("\nРЕЖИМ ДИВЕРСИИ: головы запроса переставлены между kv-группами (сдвиг "
               "на %d), эталон не тронут. Ошибка ОБЯЗАНА быть большой; маленькая "
               "означает, что тест ничего не проверяет.\n", p.ratio());
    }

    // The requested configuration, in full detail.
    printf("\nзапрошенная конфигурация\n");
    print_head();
    Row base;
    const bool base_ok = run_config(be, buft, p, syn, 1.0, refs, verbose, &base);
    print_row(base);
    if (sabotage) {
        // Inverted verdict. A perturbed query is a different question, so a cache that is
        // working must give a visibly different answer; if it does not, the comparison
        // below is not looking at what it claims to.
        const bool noticed = !base_ok || base.rel > 0.10;
        printf("  диверсия %s: ошибка %.4f%%\n",
               noticed ? "замечена, тест способен провалиться"
                       : "НЕ ЗАМЕЧЕНА — тест ничего не проверяет",
               100.0 * base.rel);
        if (!noticed) failures++;
        ex.free_all();
        ggml_backend_free(be);
        printf("\n%s (провалов %d)\n", failures ? "ЕСТЬ ПРОВАЛЫ" : "всё сошлось",
               failures);
        return failures ? 1 : 0;
    }
    if (!base_ok) {
        failures++;
    } else {
        // Structural: one softmax node, not one per zone.
        printf("  softmax-узлов в подграфе: %d %s\n", base.n_softmax,
               base.n_softmax == 1 ? "(один, глобальный)"
                                   : "(ДОЛЖЕН БЫТЬ РОВНО ОДИН)");
        if (base.n_softmax != 1) failures++;
        // Numerical: one unit of probability shared by all four zones.
        printf("  полная вероятностная масса по головам: [%.6f, %.6f] "
               "(по-зонный softmax дал бы 2.0)\n", base.mass_lo, base.mass_hi);
        if (!(std::abs(base.mass_lo - 1.0) < 1e-4 && std::abs(base.mass_hi - 1.0) < 1e-4)) {
            printf("    ПРОВАЛ: масса не единица — softmax не глобальный\n");
            failures++;
        }
        // And the cost of getting it wrong, so the check is known to be sensitive.
        printf("  цена ошибки: softmax по зонам дал бы %.2f%% против %.3f%% — "
               "проверка чувствительна\n",
               100.0 * base.rel_perzone, 100.0 * base.rel);
        if (!(base.rel_perzone > 10.0 * base.rel + 1e-6)) {
            printf("    ПРОВАЛ: по-зонный softmax почти не отличается — тогда этот тест "
                   "не отличил бы и настоящую ошибку\n");
            failures++;
        }
        printf("  читает %zu Б/ток против %zu Б/ток при полностью точном кэше "
               "(%.2f раза меньше)\n",
               base.bytes, base.bytes_exact,
               double(base.bytes_exact) / double(base.bytes ? base.bytes : 1));

        // The guard, on the real thing. The self-test at the top of main proves it refuses
        // hand-made operands; this proves it refuses the actual attention output when the
        // reference is missing or the wrong shape, which is the failure that happened for
        // real in this project and produced no symptom at all.
        printf("  защита на настоящем выходе (оба случая обязаны быть ОТКАЗАНЫ):\n");
        std::vector<float> zero_ref;
        zero_ref.assign(base.last.size(), 0.0f);
        std::vector<float> short_ref;
        short_ref.assign(base.last.size() - 1, 1.0f);
        const Cmp cz = compare(base.last, zero_ref);
        const Cmp cs = compare(base.last, short_ref);
        print_cmp("против нулевого эталона", cz);
        print_cmp("против эталона не той формы", cs);
        if (cz.ok || cs.ok) {
            printf("    ПРОВАЛ: защита пропустила пустое сравнение на реальных данных\n");
            failures++;
        }
    }

    if (sweep) {
        // The trade, made visible. Sinks and window are the two knobs that decide how
        // much of the context stays exact, so the table is error against bytes over both.
        printf("\nразмен: ошибка против байтов (хвост %s)\n", tail_name(p));
        print_head();
        const int sinks[4] = {0, 1, 4, 16};
        const int windows[4] = {32, 64, 256, 512};
        for (int si = 0; si < 4; ++si) {
            for (int wi = 0; wi < 4; ++wi) {
                memex::ZonedCacheParams c = p;
                c.n_sinks = sinks[si];
                c.window = windows[wi];
                if (c.exact_slots() > c.n_ctx) continue;
                Row r;
                if (!run_config(be, buft, c, syn, 1.0, refs, false, &r)) failures++;
                print_row(r);
                if (r.ok && r.n_softmax != 1) {
                    printf("    ПРОВАЛ: softmax-узлов %d\n", r.n_softmax);
                    failures++;
                }
            }
        }

        // What the tail holds, at one zone layout, so the two questions do not mix. The
        // F16 arm is the control: it makes the zoned graph algebraically identical to
        // exact attention, so anything but a near-zero error there is a bug in the zoned
        // expression rather than a cost of compression.
        printf("\nчто хранить в хвосте (зоны те же)\n");
        print_head();
        // Said plainly, because the numbers below invite the wrong conclusion: this test
        // cannot show what the Hadamard rotation is for. The rotation flattens outlier
        // coordinates so a block scale is not wasted on them, and these synthetic keys are
        // gaussian around a direction - they have no outliers. On the model's real vectors
        // the rotation halved the output error, 3.09% to 1.57% (memex-attn). Here it will
        // move nothing, and that is a fact about the data, not about the rotation.
        for (int arm = 0; arm < 3; ++arm) {
            memex::ZonedCacheParams c = p;
            c.tail_q8 = arm != 2;
            c.rotate_keys = arm == 0;
            Row r;
            if (!run_config(be, buft, c, syn, 1.0, refs, false, &r)) failures++;
            print_row(r);
            // The F16 tail differs from exact attention only in the order the same fp16
            // numbers are accumulated, so anything above a few parts per million is a
            // fault in the zoned expression, not a cost of anything.
            if (arm == 2 && r.ok) {
                // Printed in exponential form because the table's four decimal places of
                // per cent cannot tell 0 from 1e-7, and the difference matters here: the
                // former would mean the comparison is degenerate, the latter that f32
                // accumulates the same fp16 numbers in a different order. It is the
                // latter - the exact graph sums 1024 positions in one row, the zoned one
                // sums the zones separately and adds.
                printf("    контроль f16: ошибка %.3e (не ноль — это порядок сложения "
                       "f32, а не сжатие; ноль означал бы, что сравнение вырождено)\n",
                       r.rel);
            }
            if (arm == 2 && r.ok && r.rel > 1e-5) {
                printf("    ПРОВАЛ: f16-хвост обязан быть точным до округления, "
                       "а даёт %.4f%% — значит зонное выражение неверно, "
                       "а не квантование дорого\n", 100.0 * r.rel);
                failures++;
            }
        }

        // The same cache at three temperatures. Without this the error above is a single
        // number from an unstated regime, and the regime is most of the answer: a flat
        // distribution turns the weighted sum into a mean and averages the quantiser's
        // error away, while a peaked one puts the whole answer on a handful of positions
        // whose representation then has to be right. The reference is recomputed at each
        // scale, and re-checked against the host, rather than reused.
        printf("\nтемпература: та же зонная конфигурация, разная заострённость\n");
        printf("%8s %10s %11s %11s %8s\n",
               "масштаб", "макс.вес", "ош.L2", "по зонам", "вес хв.");
        for (double qs : {0.25, 0.5, 1.0, 2.0}) {
            RefSet rr = make_refs(be, p, syn, ex.gf, exq, exout, qs, false, true);
            if (!rr.ok) {
                failures++;
                continue;
            }
            Row r;
            if (!run_config(be, buft, p, syn, qs, rr.out, false, &r)) {
                failures++;
                printf("%8.2f   ОТКАЗ: %s\n", qs, r.why.c_str());
                continue;
            }
            printf("%8.2f %9.2f%% %10.4f%% %10.2f%% %8.3f\n",
                   qs, 100.0 * rr.peak, 100.0 * r.rel, 100.0 * r.rel_perzone,
                   r.tail_mass);
        }
    }

    ex.free_all();
    ggml_backend_free(be);
    printf("\n%s (провалов %d)\n", failures ? "ЕСТЬ ПРОВАЛЫ" : "всё сошлось", failures);
    return failures ? 1 : 0;
}
