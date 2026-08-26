// Dump every expert-routing decision the model makes, so the two optimisations
// the project is considering can be costed from data instead of from hope.
//
// Both hinge on the same unknown. Pinning experts in fast memory only pays in
// proportion to how concentrated expert use is. Sharing one expert set across
// several tokens only pays if the union of experts over those tokens is much
// smaller than the sum - and in a speculative verify pass that union is exactly
// what determines how many bytes of weights get read for the whole batch.
//
// Neither number can be guessed: they are properties of the trained router. So
// this hooks the graph callback, keeps only the `ffn_moe_topk` nodes (the chosen
// expert ids, int32, [n_expert_used, n_tokens]) and writes them to a file that
// the analysis script reads.
//
// Record layout, little-endian, repeated per captured tensor:
//   int32 layer, int32 n_expert_used, int32 n_tokens, then the ids row-major.
//
// With MOE_TRACE_WEIGHTS set, the normalised router weights are captured too, in the
// same layout but with a negated layer field so a reader can tell them apart. They
// answer the question adaptive-K depends on: how much of the routing mass the last
// experts of a top-k selection actually carry. If the tail is nearly weightless,
// dropping it cuts read traffic proportionally for almost nothing.

#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct trace_data {
    FILE * out = nullptr;
    bool want_weights = false;
    std::vector<uint8_t> buf;
    long long tensors = 0;
    long long ids = 0;
};

// The graph callback fires twice per node: once to ask whether we want the node,
// once with its data. Returning true on the first call requests the data.
static int moe_trace_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (trace_data *) user_data;

    const char * name = t->name;
    const bool is_ids = strncmp(name, "ffn_moe_topk", 12) == 0;
    // The normalised weights are the ones the FFN actually scales by. Qwen3 does not
    // take the softmax branch, so the node to capture is the normalised one; matching
    // the softmax name found nothing at all.
    const bool is_w = d->want_weights &&
                      strncmp(name, "ffn_moe_weights_norm", 20) == 0;
    // The full pre-selection distribution over all experts, needed to ask what a
    // *different* expert would have cost: cache-conditional routing swaps a marginal
    // pick for a resident one, and only these values say how much that costs.
    const bool is_p = getenv("MOE_TRACE_PROBS") != nullptr &&
                      strncmp(name, "ffn_moe_probs", 13) == 0;
    // The normalised hidden state that feeds the MoE block. With this and the router
    // weight, our own code can compute the routing decision and be checked against the
    // ids the model actually produced - the difference between "the mechanism runs" and
    // "the mechanism is right".
    //
    // Which node that is, is selectable: MOE_TRACE_ACT_NAME overrides the matched name
    // prefix and defaults to the input, so old captures reproduce exactly. The reason to
    // make it selectable is that the input answers only half the question. Activation
    // sparsity in the intermediate `silu(gate(x))*up(x)` - node "ffn_moe_gate_par",
    // [n_ff_exp, n_expert_used, n_tokens] - is what decides whether rows of `down` can go
    // unread, and a gated product may concentrate where a normalised hidden state does not.
    // The record writer below needs no change for it: n_used = ne[0] and n_tok = ne[1]*ne[2]
    // give one row per (token, selected expert) in top-k order, matching the ids already
    // stored in these traces.
    static const char * const act_name = getenv("MOE_TRACE_ACT_NAME") != nullptr
                                             ? getenv("MOE_TRACE_ACT_NAME")
                                             : "ffn_inp_normed";
    static const size_t act_len = strlen(act_name);
    const bool is_a = getenv("MOE_TRACE_ACT") != nullptr &&
                      strncmp(name, act_name, act_len) == 0;
    // Keys and values as the model produces them, before they enter the cache. The
    // compressed tail needs a basis fitted to *these* vectors: a random basis measures
    // the same bytes while destroying the content, so without them the zoned cache
    // cannot be checked for quality at all.
    // Exactly the keys the cache stores, and nothing that merely resembles them. The
    // graph calls cb() on several nodes named Kcur - the raw view out of the fused QKV
    // matmul, the output of k_norm, and the output of RoPE - so matching the name alone
    // recorded the layer's keys four times over.
    //
    // That was not waste but a measurement error. Covariance weights a vector by the
    // square of its norm, and the four differ in norm: the pre-norm copy has head norm
    // 0.97, the post-norm one 40.1. Fitting a basis to the mixture fits it to whichever
    // dominates, and the first attempt reported 13% reconstruction error on keys the
    // cache never holds. Worse, the pre-norm keys make attention almost uniform - head
    // norm 0.97 over sqrt(128) gives scores around 0.09 - so a quality measurement taken
    // on them sits in the flattest, most forgiving regime there is.
    //
    // The fork therefore names the post-RoPE keys distinctly, which nothing else reads.
    const bool is_kv = getenv("MOE_TRACE_KV") != nullptr &&
                       (strncmp(name, "Kcur_roped", 10) == 0 ||
                        strncmp(name, "Vcur", 4) == 0);
    const bool is_v = is_kv && name[0] == 'V';
    if (ask) {
        return (is_ids || is_w || is_p || is_a || is_kv) ? 1 : 0;
    }
    if (!is_ids && !is_w && !is_p && !is_a && !is_kv) {
        return 1;
    }
    if (is_ids && t->type != GGML_TYPE_I32) {
        return 1;
    }
    if ((is_w || is_p || is_a) && t->type != GGML_TYPE_F32) {
        return 1;
    }
    if (is_kv && t->type != GGML_TYPE_F32) {
        return 1;      // only the pre-cache F32 form is useful for fitting a basis
    }

    // the layer index is appended to the node name as "ffn_moe_topk-<il>"
    int layer = -1;
    if (const char * dash = strrchr(name, '-')) {
        layer = atoi(dash + 1);
    }

    // top_k hands back a view whose rows are strided over the full expert count,
    // so copying ggml_nbytes would store mostly padding. Walk the rows and keep
    // only the ids that were actually selected.
    const size_t n_bytes = ggml_nbytes(t);
    const uint8_t * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const uint8_t *) t->data;
    } else {
        d->buf.resize(n_bytes);
        ggml_backend_tensor_get(t, d->buf.data(), 0, n_bytes);
        data = d->buf.data();
    }

    // weights come as [1, n_expert_used, n_tokens]; ids as [n_expert_used, n_tokens]
    int64_t n_used = t->ne[0];
    int64_t n_tok  = (is_w || is_p || is_a || is_kv)
                         ? (t->ne[1] * t->ne[2] * t->ne[3] / (t->ne[3] ? 1 : 1))
                         : t->ne[1];
    int64_t stride = t->nb[1];
    // The post-RoPE keys arrive as [head_dim, n_head_kv, tokens], so a row is one head of
    // one position rather than a position. Since the tensor is contiguous, a whole
    // position is contiguous too - dump it at full width and every reader downstream sees
    // the same layout as the values, with no regrouping to get wrong.
    if (is_kv && t->ne[2] > 1 && ggml_is_contiguous(t)) {
        n_used = t->ne[0] * t->ne[1];
        n_tok  = t->ne[2] * t->ne[3];
        stride = t->nb[2];
    }
    // layer tags: >=0 ids, -(l+1) chosen weights, -(l+1)-10000 full distribution,
    // -(l+1)-20000 the activation that produced them
    // -(l+1)-30000 keys, -(l+1)-40000 values
    int32_t tag = (int32_t) layer;
    if (is_kv) {
        tag = is_v ? -(int32_t) layer - 40001 : -(int32_t) layer - 30001;
    } else if (is_a) {
        tag = -(int32_t) layer - 20001;
    } else if (is_p) {
        tag = -(int32_t) layer - 10001;
    } else if (is_w) {
        tag = -(int32_t) layer - 1;
    }
    const int32_t hdr[3] = {tag, (int32_t) n_used, (int32_t) n_tok};
    fwrite(hdr, sizeof(int32_t), 3, d->out);
    for (int64_t i = 0; i < n_tok; ++i) {
        fwrite(data + i * stride, 4, (size_t) n_used, d->out);
    }
    d->tensors++;
    d->ids += n_used * n_tok;
    return 1;
}

static bool run_prompt(llama_context * ctx, const gpt_params & params) {
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));
    std::vector<llama_token> tokens = ::common_tokenize(ctx, params.prompt, add_bos);
    if (tokens.empty()) {
        fprintf(stderr, "%s: пустой промпт\n", __func__);
        return false;
    }
    fprintf(stderr, "%s: токенов в промпте: %zu\n", __func__, tokens.size());
    // One decode over the whole prompt: consecutive tokens land in one batch,
    // which is the same shape a speculative verify pass has.
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
        fprintf(stderr, "%s: llama_decode не прошёл\n", __func__);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    const char * out_path = getenv("MOE_TRACE_OUT");
    if (out_path == nullptr) {
        out_path = "moe_trace.bin";
    }

    trace_data d;
    d.want_weights = getenv("MOE_TRACE_WEIGHTS") != nullptr;
    d.out = fopen(out_path, "wb");
    if (d.out == nullptr) {
        fprintf(stderr, "не открывается файл трассы: %s\n", out_path);
        return 1;
    }

    if (getenv("MOE_TRACE_ACT") != nullptr) {
        const char * an = getenv("MOE_TRACE_ACT_NAME");
        fprintf(stderr, "трасса активаций: узел с префиксом \"%s\"\n",
                an != nullptr ? an : "ffn_inp_normed");
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = moe_trace_cb;
    params.cb_eval_user_data = &d;
    params.warmup = false;               // a warmup pass would pollute the trace

    llama_init_result init = llama_init_from_gpt_params(params);
    if (init.model == nullptr || init.context == nullptr) {
        fprintf(stderr, "модель не загрузилась\n");
        return 1;
    }

    const bool ok = run_prompt(init.context, params);

    fclose(d.out);
    fprintf(stderr, "трасса: %lld тензоров, %lld выборов экспертов -> %s\n",
            d.tensors, d.ids, out_path);

    llama_free(init.context);
    llama_free_model(init.model);
    llama_backend_free();
    return ok ? 0 : 1;
}
