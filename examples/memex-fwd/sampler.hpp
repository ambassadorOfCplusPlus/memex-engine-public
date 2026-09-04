#pragma once

// Refactor step 1: the token sampler and the logit-comparison helpers, factored out of
// memex-fwd.cpp. These depend only on a float logit vector and the vocabulary size - never on
// HParams, Graph or Weights - so they compile into memex_core.
//
// A note on the token type. The originals named llama's `llama_token`, which is exactly
// `typedef int32_t llama_token`. memex_core links ggml only and must not pull llama, so the
// underlying int32_t is written here instead: the type, the width and every value are
// identical, and a std::vector<llama_token> passed by a caller is the very same type as the
// std::vector<int32_t> below. Nothing about the arithmetic or the tokens chosen changes.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace memex {

// Ours, over our own logits. Not a choice of taste: llama's sampler chain hangs off a
// llama_context, and the generation path here deliberately never creates one - the reference
// context that exists for the logit comparison is torn down before generation matters and is
// absent entirely under --no-ref. So the fifty lines below are the price of not calling
// llama_decode, and they are the same fifty lines llama's own samplers are.
//
// The default is plain greedy and it takes its own branch on purpose. Every change in this
// project is guarded by comparing twenty-four generated tokens against the previous binary,
// and that comparison is worth something only while the default path still reaches
// std::max_element over untouched logits - not a temperature-1 softmax that happens to have
// the same argmax.
struct Sampling {
    float temp = 0.0f;             // 0 or below is greedy
    int   top_k = 0;               // 0 means the whole vocabulary
    float top_p = 1.0f;
    float min_p = 0.0f;
    float repeat_penalty = 1.0f;
    int   repeat_last_n = 64;
    uint64_t seed = 1234;          // fixed, never clock-derived: a default run must repeat

    bool plain_argmax() const { return temp <= 0.0f && repeat_penalty == 1.0f; }
    bool stochastic() const { return temp > 0.0f; }
};

struct Sampler {
    Sampling s;
    std::mt19937_64 rng;
    std::vector<float> work;       // penalised logits
    std::vector<int> idx;          // candidate ids, sorted when a nucleus rule needs it

    explicit Sampler(const Sampling& in) : s(in), rng(in.seed) {}

    // llama.cpp's own rule: divide a positive logit and multiply a negative one, so the
    // penalty always moves a token down instead of flipping its sign around zero.
    void penalise(const std::vector<int32_t>& hist) {
        if (s.repeat_penalty == 1.0f || s.repeat_last_n <= 0 || hist.empty()) return;
        const size_t take = std::min(size_t(s.repeat_last_n), hist.size());
        for (size_t i = hist.size() - take; i < hist.size(); ++i) {
            const long long t = (long long)hist[i];
            if (t < 0 || size_t(t) >= work.size()) continue;
            float& l = work[size_t(t)];
            l = l > 0.0f ? l / s.repeat_penalty : l * s.repeat_penalty;
        }
    }

    // The distribution a draw would come from, over the whole vocabulary, zero outside the
    // surviving candidate set. Produced explicitly rather than folded into the draw because
    // speculative acceptance needs both models' distributions in hand, and one function
    // producing both is the only way the acceptance test cannot be comparing two differently
    // transformed things.
    //
    // Order is llama.cpp's: penalties, temperature, top-k, top-p, min-p. It matters - min-p
    // is relative to the best surviving candidate, so moving it before top-p changes the set.
    bool dist(const std::vector<float>& logits, const std::vector<int32_t>& hist,
              std::vector<float>* out) {
        if (logits.empty()) {
            printf("сэмплер: пустой вектор логитов\n");
            return false;
        }
        work = logits;
        penalise(hist);

        const float lmax = *std::max_element(work.begin(), work.end());
        if (!std::isfinite(lmax)) {
            printf("сэмплер: логиты не конечны (max %g)\n", double(lmax));
            return false;
        }
        const float inv_t = 1.0f / (s.temp > 0.0f ? s.temp : 1.0f);
        out->assign(work.size(), 0.0f);
        double sum = 0.0;
        // Candidates are the ids whose exponential did not underflow to zero. That is an
        // exact statement about float arithmetic, not a heuristic cutoff: a token whose
        // probability is exactly zero cannot be drawn and cannot enter a nucleus.
        idx.clear();
        for (size_t i = 0; i < work.size(); ++i) {
            const float e = std::exp((work[i] - lmax) * inv_t);
            (*out)[i] = e;
            if (e > 0.0f) {
                sum += double(e);
                idx.push_back(int(i));
            }
        }
        if (!(sum > 0.0) || idx.empty()) {
            printf("сэмплер: вся вероятностная масса обнулилась\n");
            return false;
        }
        for (int i : idx) (*out)[size_t(i)] = float(double((*out)[size_t(i)]) / sum);

        const bool need_order = (s.top_k > 0 && size_t(s.top_k) < idx.size()) || s.top_p < 1.0f;
        if (need_order) {
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                if ((*out)[size_t(a)] != (*out)[size_t(b)]) {
                    return (*out)[size_t(a)] > (*out)[size_t(b)];
                }
                return a < b;   // a stable tie-break, so the nucleus is well defined
            });
        }
        size_t keep = idx.size();
        if (s.top_k > 0 && size_t(s.top_k) < keep) keep = size_t(s.top_k);
        if (s.top_p < 1.0f) {
            double c = 0.0;
            size_t k = 0;
            while (k < keep) {
                c += double((*out)[size_t(idx[k])]);
                ++k;
                if (c >= double(s.top_p)) break;
            }
            keep = std::max<size_t>(1, k);
        }
        for (size_t i = keep; i < idx.size(); ++i) (*out)[size_t(idx[i])] = 0.0f;
        idx.resize(keep);

        if (s.min_p > 0.0f) {
            float best = 0.0f;
            for (int i : idx) best = std::max(best, (*out)[size_t(i)]);
            const float floor_p = s.min_p * best;
            size_t w = 0;
            for (size_t i = 0; i < idx.size(); ++i) {
                if ((*out)[size_t(idx[i])] >= floor_p) {
                    idx[w++] = idx[i];
                } else {
                    (*out)[size_t(idx[i])] = 0.0f;
                }
            }
            // Never empty: min_p is relative to the best candidate, so the best always
            // survives its own threshold - but an all-zero set would be a silent hang, so it
            // is asserted rather than assumed.
            if (w == 0) {
                printf("сэмплер: min-p %.4f не оставил ни одного кандидата\n",
                       double(s.min_p));
                return false;
            }
            idx.resize(w);
        }

        double renorm = 0.0;
        for (int i : idx) renorm += double((*out)[size_t(i)]);
        if (!(renorm > 0.0)) {
            printf("сэмплер: после отсечения сумма вероятностей нулевая\n");
            return false;
        }
        for (int i : idx) (*out)[size_t(i)] = float(double((*out)[size_t(i)]) / renorm);
        return true;
    }

    // Inverse-CDF draw over the surviving candidates, in the order `idx` currently holds -
    // which is the order dist() left them in, so a given seed and a given logit vector always
    // produce the same token.
    int32_t draw(const std::vector<float>& probs) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const double r = u(rng);
        double c = 0.0;
        for (int i : idx) {
            c += double(probs[size_t(i)]);
            if (r < c) return int32_t(i);
        }
        return int32_t(idx.back());   // only reachable through rounding of the last cell
    }

    int32_t pick(const std::vector<float>& logits, const std::vector<int32_t>& hist,
                 bool* ok) {
        *ok = true;
        if (s.plain_argmax()) {
            return int32_t(std::max_element(logits.begin(), logits.end()) -
                           logits.begin());
        }
        work = logits;
        penalise(hist);
        if (!s.stochastic()) {
            return int32_t(std::max_element(work.begin(), work.end()) - work.begin());
        }
        std::vector<float> p;
        if (!dist(logits, hist, &p)) {
            *ok = false;
            return 0;
        }
        return draw(p);
    }
};

// How much the winning logit won by. A greedy sequence can diverge from the reference on a
// near-tie without anything being wrong, and after the fact that is indistinguishable from a
// real fault - so the margin is recorded as it is produced, per step.
double top2_gap(const std::vector<float>& v);

int argmax_of(const std::vector<float>& v);

// The logit comparison, lifted verbatim in definition from examples/memex-test
// (compare_logits / LogitCmp there) rather than reinvented, because a second definition of
// the same metric is a second number to argue about. The one that decides anything is
// flip_margin: twice the largest absolute difference inside the reference's own top-100,
// over the reference's own top-2 gap. Under one, the argmax cannot move - that is a proof,
// not a threshold, and it is the reason this file can say "the token is safe" about an
// approximation rather than "the error looked small".
//
// Relative L2 over the whole vocabulary is reported too and should not be read as alarming:
// softmax is invariant to an additive shift, so any constant offset lands in the numerator
// while changing no probability, and the denominator averages over ~150k tokens no sampler
// will ever reach.
struct LogitCmp {
    bool ok = false;
    const char* why = "";
    std::size_t n_ours = 0, n_ref = 0;
    double rel_full = -1.0;
    double rel_top100 = -1.0;
    double max_abs_top = 0.0;
    double gap = 0.0;
    double flip_margin = -1.0;
    double norm_ours = 0.0, norm_ref = 0.0;
    int argmax_ours = -1, argmax_ref = -1;
};

LogitCmp compare_logits(const std::vector<float>& ours, const std::vector<float>& ref);

} // namespace memex
