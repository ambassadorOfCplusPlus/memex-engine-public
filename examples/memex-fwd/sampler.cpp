// Refactor step 1: the logit-comparison helpers moved verbatim out of memex-fwd.cpp. The
// bodies are byte-for-byte the anonymous-namespace originals; only the linkage changed
// (internal -> external in namespace memex). Sampling/Sampler live inline in the header.

#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace memex {

double top2_gap(const std::vector<float>& v) {
    float best = -INFINITY, second = -INFINITY;
    for (float x : v) {
        if (x > best) { second = best; best = x; }
        else if (x > second) { second = x; }
    }
    return double(best) - double(second);
}

int argmax_of(const std::vector<float>& v) {
    return int(std::max_element(v.begin(), v.end()) - v.begin());
}

LogitCmp compare_logits(const std::vector<float>& ours, const std::vector<float>& ref) {
    LogitCmp r;
    r.n_ours = ours.size();
    r.n_ref = ref.size();
    if (ours.empty() || ref.empty()) {
        r.why = "пустой операнд";
        return r;
    }
    // Refused, not truncated and not broadcast. A comparison against a wrong-shaped or
    // all-zero reference has read as a perfect match in this project before, so both are
    // refusals rather than numbers.
    if (ours.size() != ref.size()) {
        r.why = "размеры не совпадают";
        return r;
    }
    const std::size_t n = ref.size();
    double num = 0.0, den = 0.0, sq_ours = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
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
    }
    r.norm_ours = std::sqrt(sq_ours);
    r.norm_ref = std::sqrt(den);
    if (!(den > 0.0)) {
        r.why = "норма эталона равна нулю — сравнивать не с чем";
        return r;
    }
    if (!(sq_ours > 0.0)) {
        r.why = "норма нашего вектора равна нулю — мы ничего не посчитали";
        return r;
    }
    r.rel_full = std::sqrt(num / den);

    // The reference's own ordering picks the candidate set: those are the entries a sampler
    // can reach, and the only ones whose error can change the emitted token.
    const int kmax = int(std::min<std::size_t>(n, 100));
    std::vector<int> ids;
    ids.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) ids[i] = int(i);
    std::partial_sort(ids.begin(), ids.begin() + kmax, ids.end(), [&](int a, int b) {
        if (ref[std::size_t(a)] != ref[std::size_t(b)]) {
            return ref[std::size_t(a)] > ref[std::size_t(b)];
        }
        return a < b;   // a stable tie-break, so the set is well defined
    });
    double tn = 0.0, td = 0.0;
    for (int i = 0; i < kmax; ++i) {
        const std::size_t id = std::size_t(ids[std::size_t(i)]);
        const double d = double(ours[id]) - double(ref[id]);
        tn += d * d;
        td += double(ref[id]) * double(ref[id]);
        if (std::abs(d) > r.max_abs_top) r.max_abs_top = std::abs(d);
    }
    if (td > 0.0) r.rel_top100 = std::sqrt(tn / td);
    r.gap = top2_gap(ref);
    if (r.gap > 0.0) r.flip_margin = 2.0 * r.max_abs_top / r.gap;
    r.argmax_ours = argmax_of(ours);
    r.argmax_ref = argmax_of(ref);
    r.ok = true;
    return r;
}

} // namespace memex
