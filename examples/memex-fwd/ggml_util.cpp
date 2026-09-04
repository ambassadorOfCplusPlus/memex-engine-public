// Refactor step 1: ggml-graph helpers moved verbatim out of memex-fwd.cpp. Behaviour is
// unchanged - the bodies are byte-for-byte what the anonymous-namespace definitions held;
// only their linkage changed (internal -> external in namespace memex) so the executable can
// call them through the header.

#include "ggml_util.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace memex {

ggml_tensor* norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

ggml_tensor* fnorm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_fused_rms_norm(c, x, w, eps);
}

void report_first_nonfinite(const char* what,
                            const std::vector<std::pair<std::string, ggml_tensor*>>& probes) {
    if (probes.empty()) return;
    auto is_premask_score = [](const std::string& nm) {
        return nm.rfind("kq-", 0) == 0 || nm.rfind("q-", 0) == 0;
    };
    std::vector<float> pv;
    std::string first_any, first_must;
    std::size_t idx_any = 0, idx_must = 0, n_any = 0, n_must = 0;
    float val_any = 0.0f, val_must = 0.0f;
    for (const auto& pr : probes) {
        const std::size_t np = std::size_t(ggml_nelements(pr.second));
        pv.resize(np);
        ggml_backend_tensor_get(pr.second, pv.data(), 0, np * sizeof(float));
        for (std::size_t q = 0; q < np; ++q) {
            if (std::isfinite(pv[q])) continue;
            if (first_any.empty()) {
                first_any = pr.first; idx_any = q; n_any = np; val_any = pv[q];
            }
            if (first_must.empty() && !is_premask_score(pr.first)) {
                first_must = pr.first; idx_must = q; n_must = np; val_must = pv[q];
            }
            break;
        }
        if (!first_must.empty()) break;
    }
    if (first_any.empty()) {
        printf("  %s: vse %zu zondov finitny\n", what, probes.size());
        return;
    }
    printf("  %s: pervyj nefinitnyj VOOBSHCHE - %s (element %zu iz %zu, %g)\n",
           what, first_any.c_str(), idx_any, n_any, val_any);
    if (first_must.empty()) {
        printf("  %s: sredi velichin, OBJAZANNYH byt konechnymi, nefinitnyh net - NaN vyshe "
               "sidit v syrom kq do maski i eto normalno\n", what);
    } else {
        printf("  %s: pervyj nefinitnyj sredi OBJAZATELNYH - %s (element %zu iz %zu, %g)\n",
               what, first_must.c_str(), idx_must, n_must, val_must);
    }
}

} // namespace memex
