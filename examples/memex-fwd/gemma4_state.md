# Gemma 4 in our engine: what is proven, what is ruled out, what is left

Written 27 Aug 2026 on handover. Everything here is measured on
`gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf`, 12-token prompt, `--no-repack`, t=8, per-layer comparison
against the reference's own node names. Numbers are relative L2 in percent.

## The headline, and it reverses the earlier reading

**Our gemma4 attention block is byte-for-byte identical to the reference. The 147.95% figure on
`attn_out-0` was measured against a reference that is itself wrong in the configuration we chose.**

The comparison used to set `cparams.flash_attn = false`, on the reasonable grounds that the plain
path is what our graph mirrors node for node. That is also the path in which the fork's gemma4 V
cache is written incorrectly:

- `cache.v_trans = !recurrent && !cparams.flash_attn && !hybrid` (llama.cpp:1192), so
  flash_attn off means v_trans TRUE.
- With v_trans true, `llm_build_kv_store` does `v_cur = ggml_transpose(ctx, v_cur)` and copies
  into a `[n_tokens, n_embd_v_gqa]` strided view. Correct for every architecture that hands it a
  2-D V of `[n_embd_v_gqa, n_tokens]` - which is all of them except one.
- gemma4 is the exception: its unweighted V rms_norm needs the per-head shape, so V arrives
  **3-D** as `[n_embd_head_v, n_head_kv, n_tokens]` (llama-build-context.cpp:3580, and the same
  in build_gemma4's own branch for the five layers without attn_v). `ggml_transpose` swaps ne0
  and ne1 ONLY, giving `[n_head_kv, n_embd_head_v, n_tokens]`.
- `ggml_cpy` between differently-shaped tensors is a flat linear-index copy - the dst-counter
  loop in `ggml_compute_forward_dup_f32` walks source and destination each in its own index
  order and pairs them by position. So the order the store writes is not the order the read view
  expects, and the V cache comes out permuted.
- With flash_attn TRUE, v_trans is false, the store is a flat `view_1d`, and the FA read is
  `[n_embd_head_v, n_kv, n_head_kv]` with matching strides. Store and read agree.

`--ref-fa` selects the reference's working arm. The A/B is one flag on one binary:

    arm                        prefill logits   best token   decode, 6 steps
    flash_attn off             L2 414.33%       DIFFER       0 of 6
    flash_attn on (--ref-fa)   L2  10.54%       MATCH        3 of 6

Against the working arm, layer 0 gives **eleven consecutive tensors at exactly 0.0000%**, max |d|
0.00000, with both RMS values equal to five decimals:

    attn_norm-0  Qcur-0  Kcur-0  Vcur-0  Qcur_normed-0  Qcur_roped-0
    Kcur_normed-0  Kcur_roped-0  kqv_out-0  attn_out-0  ffn_norm_2-0

`kqv_out-0` and `attn_out-0` read 217.66% and 147.95% against the broken arm and 0.0000% against
the working one. On the broken arm, `kq-0` and `kq_soft_max_ext-0` were also exactly 0.0000%,
which is what localised the fault to the single node that reads the V cache.

## Ruled out, and by exact zeros rather than by argument

- **The attention scale.** `f_attn_scale = 1.0`, set for gemma4 where the hparams are read and
  used as the KQ scale in `build_gemma4_step`, matching `llama-hparams.cpp:859`.
  `kq_soft_max_ext-0` is bit-identical, which tests the scale directly - it is the softmax's own
  argument.
- **The per-layer geometry.** Layer 0 gets ATTN_SWA, head_dim 256, n_rot 256, rope base 1e4,
  8 KV heads. Confirmed by `Qcur-0`/`Kcur-0` being bit-equal, which they could not be under the
  model-level 512 / 2 / 1e6. Pattern polarity checks out against the head_count_kv array.
- **q_norm and k_norm, their weights AND their placement relative to rope.** `Qcur_normed-0`,
  `Kcur_normed-0`, `Qcur_roped-0`, `Kcur_roped-0` are all exact. Gemma norms before rope and so
  do we.
- **The unweighted rms_norm on V, and V-from-K sharing.** `kqv_out-0` exact means the V that
  reached the attention product was the reference's V.
- **The KV cache layout, the GQA head mapping and the windowed mask.** `kq-0` exact.
- **The probe anchoring.** This one was a real fault and is fixed. The reference gives the name
  `attn_out` to two different quantities: pre-residual on the 25 layers that go through
  `build_std_attention` (it does `cb(cur,"attn_out",il)` and adds the residual on the next line,
  llama-build-context.cpp:3696) and post-residual on the five built inline in `build_gemma4`
  (build_gemma4.cpp:1021). Probing our post-residual value under that name compared two different
  things on 25 layers of 30 and reported 100-500% of pure artefact. Our probe now switches on
  `L.wv`. See METHODS 61.
- **The tied output head and the 30.0 logit softcap** - both downstream, and the prefill logits
  now agree to 10.54% with a matching argmax.

## What is actually left

One fault, and it is small and localised. Against the working reference arm the first non-zero
line at layer 0 is

    ffn_moe_combined-0    L2 17.11%   max |d| 92.59   rms 20.65637 / 20.65505

with `ffn_norm_2-0` (the MoE's own input) exact above it. Same RMS to six figures, so this is not
a missing scale. The fault is inside layer 0's feed-forward, and there are four places it can be:

1. the dense half - `ffn_norm`, then `ggml_fused_up_gate(up, gate, x, GELU)`, then `down`;
2. the router - note the fix below, which is already applied;
3. the experts - the fused `ffn_gate_up_exps`, the GELU, `down_exps`, and the
   `ffn_down_exps.scale` folded through `ggml_mul_multi_add`'s src[2]/src[3];
4. the `ggml_fused_rms_rms_add` that joins the two halves with `post_ffw_norm_1/_2`.

Two probes that split those four ways are **already in the source and never got a run**:
`ffn_norm_1-<il>` (the dense half's input) and `ffn_moe_weighted-<il>` (the routed half's output,
under the reference's own name). Worth knowing why the routed half had never been checked: our
probe called it `ffn_moe_out`, a name the reference never emits, so `report` skipped it silently.
One run of `arch_verify_run.ps1 -Steps short` prints both.

Downstream the error stays bounded - prefill logits 10.54%, decode steps 7.3 to 36.1% - and the
text is sensible (" Tokyo. The capital of France"), which is what one arithmetic difference in one
sub-block looks like rather than a wiring fault.

## The one engine bug found and fixed

`ffn_gate_inp.scale` as it sits in the file is **not** the tensor the reference's graph sees.
`llm_scale_gate_inp_s` (llama.cpp:3805, called from :4868) walks every gemma4 layer at LOAD time
and multiplies that vector in place by `1/sqrt(n_embd)` = 1/53.066. Reading the weight straight
out of the gguf - which is what every other tensor in this engine legitimately does - leaves the
router logits 53x too large; softmax collapses onto the top expert, top-k picks the same eight
(scaling is monotone) and the renormalised weights come out near one-hot instead of a mixture.
Nothing about the shapes, the tensor names or the generated text betrays it.

Now applied to the router's normalised input in `build_gemma4_step` (`h.f_router_scale`) rather
than to the weight, because the weight is mmapped and shared.

Generalise this before adding any further architecture: **a weight can be transformed between the
file and the graph.** Our engine reads tensors by name out of the same `llama_model` the reference
uses, which makes it natural to assume the values match - and they do, until the reference edits
one in place after loading. Grep every mention of the new arch's `LLM_ARCH_*` outside its graph
builder. For gemma4 and qwen35moe, `llm_scale_gate_inp_s` is the only such edit.

Whether the fix helped cannot be read off the numbers above: there is no before/after pair for it
against the working reference arm, because it was applied before `--ref-fa` existed. That is one
cheap arm for whoever returns to this.

## Not verified at all

Qwen 3.6 35B. The graph builds and the binary loads it, but no comparison has ever completed - the
short arm died on an unquoted prompt yesterday and on a missing ggml.dll three times today, and
the queue was busy for the rest. Note that qwen35moe is **not** exposed to the V-cache fault
above: `llm_build_mul_mat_qkv_gated` returns V flat, so the transposed store gets the 2-D tensor
it was written for. Its comparison with `flash_attn = false` should be valid as-is.

The 1100-token window-crossing arm has also never run against a valid reference. Its earlier
numbers were all taken against the broken arm and say nothing about the window.
