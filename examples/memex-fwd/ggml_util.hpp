#pragma once

// Small ggml-graph helpers, factored out of memex-fwd.cpp in refactor step 1. These depend
// on nothing but ggml (and the ggml backend, for the tensor read in the probe): no HParams,
// no Graph, no Weights. They compile into memex_core, whose only public link dependency is
// ggml, so nothing here may pull llama or our own structures.

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <utility>
#include <vector>

namespace memex {

// RMS norm followed by the learned scale, which is what every llama-family norm is.
ggml_tensor* norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps);

// The same thing as one op, which is what the reference's llm_build_norm actually calls
// (llama-build-context.cpp: LLM_NORM_RMS with a weight goes to ggml_fused_rms_norm).
//
// It matters that it is the same OP and not merely the same arithmetic. The fused kernel
// computes (scale*w[j])*x[j] and the pair above computes (scale*x[j])*w[j]; in f32 those
// differ in the last bit, which is harmless on its own and is exactly the kind of harmless
// that compounds through thirty layers into a divergence nobody can attribute afterwards.
ggml_tensor* fnorm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps);

// First non-finite probe in a graph, by name - and TWO of them, because one of the two is
// routinely a false alarm. `kq-<il>` is the raw Q.K product over the WHOLE cache extent,
// including positions the prefill never wrote: that memory is not zeroed, so a NaN there is
// normal and the mask kills it in the softmax that follows. So the scan reports the first
// non-finite overall AND the first among the quantities that MUST be finite.
void report_first_nonfinite(const char* what,
                            const std::vector<std::pair<std::string, ggml_tensor*>>& probes);

} // namespace memex
