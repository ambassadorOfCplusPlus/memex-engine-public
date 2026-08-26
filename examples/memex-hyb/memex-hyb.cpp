// The design this project has been circling: hot experts resident on the card, cold experts
// on the processor, both computing at the same time, and the resident set refreshed in the
// background rather than on the critical path.
//
// Everything here rests on measurements that had to be taken first, and two of them changed
// the shape of the thing:
//
//   - `mul_mat_id` is unavailable on this card. Not because of the weight type - IQ4_XS is in
//     the supported list - but because the fused-expert kernel needs more shared memory than
//     the RX 6500 XT's 32 KB, so the pipelines are never created, `supports_op` returns false,
//     and ggml falls back to the CPU by dragging the weights back over PCIe. That is what made
//     `-ngl 6` cost 492 ms per token: 1.86 GB over a 3.94 GB/s link, which is exactly what it
//     measured. So the resident experts are computed as plain per-expert matrix-vector
//     products, which the card does handle - 0.083 ms per expert against 0.128 on the CPU.
//
//   - The card is faster by 1.54x, not by four. The earlier figure came from a build with
//     GGML_FMA off, which handicapped the processor it was compared against. With both built
//     the same way, VRAM reaches 37.1 GB/s on this workload and RAM 24.8.
//
// From those two: sequential offload is nearly pointless (5 experts on the card and 3 on the
// processor, one after the other, is 6% better than eight on the processor), and the whole
// gain has to come from overlap. Hence the explicit structure below - launch the card
// asynchronously, compute the processor's share meanwhile, synchronise once per layer - which
// also has to be explicit because ggml's own scheduler will not overlap CPU with GPU: its
// parallel path is gated on the processor having at most one graph split, and it puts a
// barrier before every split that has inputs.
//
// Correctness is checked the way the rest of this engine is checked: the same tokens as
// llama.cpp's own decode, or it is wrong. With zero resident experts this path must reproduce
// the reference exactly, which makes the split itself testable before the card is involved.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#include "llama.h"

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int pad32(int n) { return (n + 31) / 32 * 32; }

struct HParams {
    std::string arch;
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_vocab = 0, n_ctx_train = 0;
    float rms_eps = 1e-6f, rope_base = 10000.0f;
    int rope_type = 0;
    int d_kv() const { return n_head_kv * head_dim; }
    int d_q() const { return n_head * head_dim; }
};

int key_u32(gguf_context* g, const std::string& k, int fb) {
    const int id = gguf_find_key(g, k.c_str());
    return id < 0 ? fb : int(gguf_get_val_u32(g, id));
}
float key_f32(gguf_context* g, const std::string& k, float fb) {
    const int id = gguf_find_key(g, k.c_str());
    return id < 0 ? fb : gguf_get_val_f32(g, id);
}

bool read_hparams(const char* path, HParams* h) {
    gguf_init_params p = {true, nullptr};
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g) return false;
    const int aid = gguf_find_key(g, "general.architecture");
    h->arch = aid < 0 ? "" : gguf_get_val_str(g, aid);
    const std::string a = h->arch;
    h->n_layer = key_u32(g, a + ".block_count", 0);
    h->n_embd = key_u32(g, a + ".embedding_length", 0);
    h->n_head = key_u32(g, a + ".attention.head_count", 0);
    h->n_head_kv = key_u32(g, a + ".attention.head_count_kv", h->n_head);
    h->head_dim = key_u32(g, a + ".attention.key_length",
                          h->n_head ? h->n_embd / h->n_head : 0);
    h->n_expert = key_u32(g, a + ".expert_count", 0);
    h->n_expert_used = key_u32(g, a + ".expert_used_count", 0);
    h->n_ff_exp = key_u32(g, a + ".expert_feed_forward_length",
                          key_u32(g, a + ".feed_forward_length", 0));
    h->n_ctx_train = key_u32(g, a + ".context_length", 4096);
    h->rms_eps = key_f32(g, a + ".attention.layer_norm_rms_epsilon", 1e-6f);
    h->rope_base = key_f32(g, a + ".rope.freq_base", 10000.0f);
    gguf_free(g);
    return h->n_layer > 0;
}

ggml_tensor* norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

struct Weights {
    ggml_tensor* tok_embd = nullptr;
    ggml_tensor* out_norm = nullptr;
    ggml_tensor* out = nullptr;
    struct Layer {
        ggml_tensor *attn_norm, *wq, *wk, *wv, *wo, *q_norm, *k_norm;
        ggml_tensor *ffn_norm, *router, *up, *gate, *down;
    };
    std::vector<Layer> layers;
};

ggml_tensor* need(llama_model* m, const std::string& n, bool* ok) {
    ggml_tensor* t = llama_get_model_tensor(m, n.c_str());
    if (!t) { printf("нет тензора: %s\n", n.c_str()); *ok = false; }
    return t;
}

bool collect(llama_model* m, const HParams& h, Weights* w) {
    bool ok = true;
    w->tok_embd = need(m, "token_embd.weight", &ok);
    w->out_norm = need(m, "output_norm.weight", &ok);
    w->out = llama_get_model_tensor(m, "output.weight");
    if (!w->out) w->out = w->tok_embd;
    w->layers.resize(size_t(h.n_layer));
    for (int il = 0; il < h.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        Weights::Layer& L = w->layers[size_t(il)];
        L.attn_norm = need(m, p + "attn_norm.weight", &ok);
        L.wq = need(m, p + "attn_q.weight", &ok);
        L.wk = need(m, p + "attn_k.weight", &ok);
        L.wv = need(m, p + "attn_v.weight", &ok);
        L.wo = need(m, p + "attn_output.weight", &ok);
        L.q_norm = need(m, p + "attn_q_norm.weight", &ok);
        L.k_norm = need(m, p + "attn_k_norm.weight", &ok);
        L.ffn_norm = need(m, p + "ffn_norm.weight", &ok);
        L.router = need(m, p + "ffn_gate_inp.weight", &ok);
        L.up = need(m, p + "ffn_up_exps.weight", &ok);
        L.gate = need(m, p + "ffn_gate_exps.weight", &ok);
        L.down = need(m, p + "ffn_down_exps.weight", &ok);
    }
    return ok;
}

struct Cache {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> k, v;
    int n_ctx = 0;
    bool init(ggml_backend_buffer_type_t buft, const HParams& h, int len) {
        n_ctx = len;
        ggml_init_params ip = {ggml_tensor_overhead() * size_t(h.n_layer) * 2 + 4096,
                               nullptr, true};
        ctx = ggml_init(ip);
        if (!ctx) return false;
        k.resize(size_t(h.n_layer));
        v.resize(size_t(h.n_layer));
        for (int il = 0; il < h.n_layer; ++il) {
            k[size_t(il)] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, h.head_dim, n_ctx,
                                               h.n_head_kv);
            v[size_t(il)] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_ctx, h.head_dim,
                                               h.n_head_kv);
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        return buf != nullptr;
    }
    void free_all() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr;
    }
};

// A view of one expert out of the fused [n_embd, n_ff, n_expert] tensor. Valid for quantised
// types because a slice along the last dimension keeps whole rows intact.
ggml_tensor* expert_view(ggml_context* c, ggml_tensor* fused, int e) {
    return ggml_view_2d(c, fused, fused->ne[0], fused->ne[1], fused->nb[1],
                        size_t(e) * fused->nb[2]);
}

// Experts that live on the card. Stacked per layer so that a slot is a slice, and the copy
// from host memory is one contiguous block per expert per matrix.
struct Resident {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> up, gate, down;      // [n_embd, n_ff, slots] per layer
    std::vector<std::vector<int>> slot_of;         // [layer][expert] -> slot, or -1
    std::vector<std::vector<int>> expert_of;       // [layer][slot] -> expert, or -1
    int slots = 0;
    std::size_t bytes = 0;

    bool init(ggml_backend_buffer_type_t buft, const HParams& h, const Weights& w,
              int n_slots) {
        slots = n_slots;
        if (slots <= 0) return true;
        ggml_init_params ip = {ggml_tensor_overhead() * size_t(h.n_layer) * 3 + 8192,
                               nullptr, true};
        ctx = ggml_init(ip);
        if (!ctx) return false;
        up.resize(size_t(h.n_layer));
        gate.resize(size_t(h.n_layer));
        down.resize(size_t(h.n_layer));
        for (int il = 0; il < h.n_layer; ++il) {
            const Weights::Layer& L = w.layers[size_t(il)];
            up[size_t(il)] = ggml_new_tensor_3d(ctx, L.up->type, L.up->ne[0], L.up->ne[1],
                                                slots);
            gate[size_t(il)] = ggml_new_tensor_3d(ctx, L.gate->type, L.gate->ne[0],
                                                  L.gate->ne[1], slots);
            down[size_t(il)] = ggml_new_tensor_3d(ctx, L.down->type, L.down->ne[0],
                                                  L.down->ne[1], slots);
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) return false;
        bytes = ggml_backend_buffer_get_size(buf);
        slot_of.assign(size_t(h.n_layer), std::vector<int>(size_t(h.n_expert), -1));
        expert_of.assign(size_t(h.n_layer), std::vector<int>(size_t(slots), -1));
        return true;
    }

    // A move that has been issued but whose bytes may not have landed yet.
    struct Pending { int il, slot, expert; };
    std::vector<Pending> pending;

    // Issue the copy and nothing more. The slot keeps serving its previous occupant until
    // `activate` is called, which is the staged swap this project settled on earlier: a slot
    // must never be half-written while a graph reads it, and waiting for the copy on the
    // critical path is exactly what made the first version of this three times slower than
    // plain CPU.
    void place_async(ggml_backend_t be, const Weights& w, int il, int slot, int e) {
        const Weights::Layer& L = w.layers[size_t(il)];
        const size_t sz_u = ggml_nbytes(L.up) / size_t(L.up->ne[2]);
        const size_t sz_g = ggml_nbytes(L.gate) / size_t(L.gate->ne[2]);
        const size_t sz_d = ggml_nbytes(L.down) / size_t(L.down->ne[2]);
        ggml_backend_tensor_set_async(be, up[size_t(il)],
                                      (const char*)L.up->data + size_t(e) * sz_u,
                                      size_t(slot) * sz_u, sz_u);
        ggml_backend_tensor_set_async(be, gate[size_t(il)],
                                      (const char*)L.gate->data + size_t(e) * sz_g,
                                      size_t(slot) * sz_g, sz_g);
        ggml_backend_tensor_set_async(be, down[size_t(il)],
                                      (const char*)L.down->data + size_t(e) * sz_d,
                                      size_t(slot) * sz_d, sz_d);
        pending.push_back({il, slot, e});
    }

    // The bytes have landed; the slots may now be routed to.
    void activate() {
        for (const Pending& p : pending) {
            const int prev = expert_of[size_t(p.il)][size_t(p.slot)];
            if (prev >= 0) slot_of[size_t(p.il)][size_t(prev)] = -1;
            expert_of[size_t(p.il)][size_t(p.slot)] = p.expert;
            slot_of[size_t(p.il)][size_t(p.expert)] = p.slot;
        }
        pending.clear();
    }

    void place(const Weights& w, int il, int slot, int e) {
        const Weights::Layer& L = w.layers[size_t(il)];
        const int prev = expert_of[size_t(il)][size_t(slot)];
        if (prev >= 0) slot_of[size_t(il)][size_t(prev)] = -1;
        const size_t sz_u = ggml_nbytes(L.up) / size_t(L.up->ne[2]);
        const size_t sz_g = ggml_nbytes(L.gate) / size_t(L.gate->ne[2]);
        const size_t sz_d = ggml_nbytes(L.down) / size_t(L.down->ne[2]);
        ggml_backend_tensor_set(up[size_t(il)], (const char*)L.up->data + size_t(e) * sz_u,
                                size_t(slot) * sz_u, sz_u);
        ggml_backend_tensor_set(gate[size_t(il)],
                                (const char*)L.gate->data + size_t(e) * sz_g,
                                size_t(slot) * sz_g, sz_g);
        ggml_backend_tensor_set(down[size_t(il)],
                                (const char*)L.down->data + size_t(e) * sz_d,
                                size_t(slot) * sz_d, sz_d);
        expert_of[size_t(il)][size_t(slot)] = e;
        slot_of[size_t(il)][size_t(e)] = slot;
    }

    void free_all() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr;
    }
};

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_backend_buffer_t inputs = nullptr;
    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (inputs) ggml_backend_buffer_free(inputs);
        if (ctx) ggml_free(ctx);
        alloc = nullptr; inputs = nullptr; ctx = nullptr;
    }
};

// Everything of a layer up to and including the routing decision. Ends there because the
// decision is what the host needs in order to split the experts between devices, and reading
// it from a CPU graph costs nothing.
struct AttnGraph : Graph {
    ggml_tensor* inp = nullptr;        // [n_embd, 1] f32, the layer input
    ggml_tensor* positions = nullptr;
    ggml_tensor* mask = nullptr;
    ggml_tensor* ffn_inp = nullptr;    // out: attention output plus residual
    ggml_tensor* x_normed = nullptr;   // out: what the experts consume
    ggml_tensor* probs = nullptr;      // out: the full router distribution
};

bool build_attn(AttnGraph* g, ggml_backend_buffer_type_t buft, const HParams& h,
                const Weights& w, Cache& kv, int il, int n_past, int n_kv) {
    const size_t n_nodes = 128;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 64) +
                           ggml_graph_overhead_custom(n_nodes, false), nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) return false;
    ggml_context* c = g->ctx;
    const int hd = h.head_dim;
    const Weights::Layer& L = w.layers[size_t(il)];

    g->inp = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_embd, 1);
    g->positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    g->mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_kv, 1);
    g->inputs = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!g->inputs) return false;

    g->gf = ggml_new_graph_custom(c, n_nodes, false);
    int sections[GGML_MROPE_SECTIONS] = {0};
    ggml_tensor* x = norm(c, g->inp, L.attn_norm, h.rms_eps);
    ggml_tensor* q = ggml_mul_mat(c, L.wq, x);
    ggml_tensor* k = ggml_mul_mat(c, L.wk, x);
    ggml_tensor* v = ggml_mul_mat(c, L.wv, x);
    q = ggml_reshape_3d(c, q, hd, h.n_head, 1);
    q = norm(c, q, L.q_norm, h.rms_eps);
    q = ggml_rope_multi(c, q, g->positions, nullptr, hd, sections, h.rope_type,
                        h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_reshape_3d(c, k, hd, h.n_head_kv, 1);
    k = norm(c, k, L.k_norm, h.rms_eps);
    k = ggml_rope_multi(c, k, g->positions, nullptr, hd, sections, h.rope_type,
                        h.n_ctx_train, h.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    ggml_tensor* Kc = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
    ggml_tensor* Vc = ggml_cont(c, ggml_permute(
        c, ggml_reshape_3d(c, v, hd, h.n_head_kv, 1), 1, 2, 0, 3));
    ggml_tensor* kdst = ggml_view_3d(c, kv.k[size_t(il)], hd, 1, h.n_head_kv,
                                     kv.k[size_t(il)]->nb[1], kv.k[size_t(il)]->nb[2],
                                     size_t(n_past) * kv.k[size_t(il)]->nb[1]);
    ggml_tensor* vdst = ggml_view_3d(c, kv.v[size_t(il)], 1, hd, h.n_head_kv,
                                     kv.v[size_t(il)]->nb[1], kv.v[size_t(il)]->nb[2],
                                     size_t(n_past) * ggml_element_size(kv.v[size_t(il)]));
    ggml_build_forward_expand(g->gf, ggml_cpy(c, Kc, kdst));
    ggml_build_forward_expand(g->gf, ggml_cpy(c, Vc, vdst));

    ggml_tensor* K = ggml_view_3d(c, kv.k[size_t(il)], hd, n_kv, h.n_head_kv,
                                  kv.k[size_t(il)]->nb[1], kv.k[size_t(il)]->nb[2], 0);
    ggml_tensor* V = ggml_view_3d(c, kv.v[size_t(il)], n_kv, hd, h.n_head_kv,
                                  kv.v[size_t(il)]->nb[1], kv.v[size_t(il)]->nb[2], 0);
    ggml_tensor* Q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
    ggml_tensor* kq = ggml_mul_mat(c, K, Q);
    ggml_tensor* p = ggml_soft_max_ext(c, kq, g->mask, 1.0f / std::sqrt(float(hd)), 0.0f);
    ggml_tensor* kqv = ggml_mul_mat(c, V, p);
    kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), h.d_q(), 1);
    g->ffn_inp = ggml_add(c, ggml_mul_mat(c, L.wo, kqv), g->inp);
    g->x_normed = norm(c, g->ffn_inp, L.ffn_norm, h.rms_eps);

    // The whole distribution comes out, and the choice is made on the host. Our own routing
    // was verified against the model's earlier - 512 of 512 exact - and doing it here is what
    // makes a residency bonus, a per-layer expert count, or a similarity redirect a few lines
    // of ordinary code instead of new graph operations.
    g->probs = ggml_soft_max(c, ggml_mul_mat(c, L.router, g->x_normed));

    ggml_set_output(g->ffn_inp);
    ggml_set_output(g->x_normed);
    ggml_set_output(g->probs);
    ggml_build_forward_expand(g->gf, g->ffn_inp);
    ggml_build_forward_expand(g->gf, g->x_normed);
    ggml_build_forward_expand(g->gf, g->probs);
    g->alloc = ggml_gallocr_new(buft);
    return ggml_gallocr_reserve(g->alloc, g->gf) &&
           ggml_gallocr_alloc_graph(g->alloc, g->gf);
}

// The expert side, for one device and one explicit list of (weight tensor, expert, weight)
// triples. Built fresh whenever the list changes, which on the card is every token - the
// slots a token touches are not known until the router has spoken, and a slot index is baked
// into the view at build time.
struct MoeGraph : Graph {
    ggml_tensor* x = nullptr;        // [n_embd, 1] f32 in
    ggml_tensor* wts = nullptr;      // [1, n] f32 in, one weight per listed expert
    ggml_tensor* out = nullptr;      // [n_embd, 1] f32 out
};

// The cold branch, built once per layer and never rebuilt.
//
// The processor does support `mul_mat_id`, which the card does not, and that changes two
// things at once. It batches all eight experts into one call, so the kernel keeps its
// locality and its threading instead of being handed eight separate matrix-vector products.
// And it takes the expert numbers as an *input tensor*, so the graph is independent of which
// experts a token picked - which removes forty-eight graph rebuilds per token.
//
// Residency is expressed by writing -1 into the slots the card is handling: mul_mat_id skips
// those and zeroes their output slices, so their routing weights multiply zero and contribute
// nothing. No branch, no special case.
struct ColdGraph {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_backend_buffer_t inputs = nullptr;
    ggml_tensor* x = nullptr;      // [n_embd, 1] f32
    ggml_tensor* ids = nullptr;    // [n_expert_used, 1] i32, -1 where the card takes it
    ggml_tensor* wts = nullptr;    // [1, n_expert_used, 1] f32
    ggml_tensor* out = nullptr;

    bool init(ggml_backend_buffer_type_t buft, const HParams& h, const Weights::Layer& L) {
        const size_t n_nodes = 128;
        ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 32) +
                               ggml_graph_overhead_custom(n_nodes, false), nullptr, true};
        ctx = ggml_init(ip);
        if (!ctx) return false;
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, 1);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, h.n_expert_used, 1);
        wts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, h.n_expert_used, 1);
        inputs = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!inputs) return false;
        gf = ggml_new_graph_custom(ctx, n_nodes, false);
        ggml_tensor* x3 = ggml_reshape_3d(ctx, x, h.n_embd, 1, 1);
        ggml_tensor* up = ggml_mul_mat_id(ctx, L.up, x3, ids);
        ggml_tensor* gt = ggml_silu(ctx, ggml_mul_mat_id(ctx, L.gate, x3, ids));
        ggml_tensor* o = ggml_mul_mat_id(ctx, L.down, ggml_mul(ctx, up, gt), ids);
        o = ggml_mul(ctx, o, wts);
        ggml_tensor* acc = ggml_view_2d(ctx, o, h.n_embd, 1, o->nb[2], 0);
        for (int e = 1; e < h.n_expert_used; ++e) {
            acc = ggml_add(ctx, acc,
                           ggml_view_2d(ctx, o, h.n_embd, 1, o->nb[2],
                                        size_t(e) * o->nb[1]));
        }
        out = acc;
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);
        alloc = ggml_gallocr_new(buft);
        return alloc && ggml_gallocr_reserve(alloc, gf) &&
               ggml_gallocr_alloc_graph(alloc, gf);
    }
    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (inputs) ggml_backend_buffer_free(inputs);
        if (ctx) ggml_free(ctx);
        alloc = nullptr; inputs = nullptr; ctx = nullptr;
    }
};

// Inputs and the allocator, created once per layer per device and reused for every token.
//
// This exists because the obvious version - a fresh context, buffer and allocator per layer
// per token - cost 716 ms out of a 28-position run, which is 0.53 ms per layer, all of it
// driver bookkeeping. It made a design that saves 312 ms of processor work look like it loses
// three quarters of a second. The graph nodes themselves are cheap to rebuild; the buffers
// are not.
struct MoePool {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor* x = nullptr;
    ggml_tensor* wts = nullptr;

    bool init(ggml_backend_buffer_type_t buft, const HParams& h) {
        ggml_init_params ip = {ggml_tensor_overhead() * 8 + 1024, nullptr, true};
        ctx = ggml_init(ip);
        if (!ctx) return false;
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, 1);
        wts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, h.n_expert_used);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) return false;
        alloc = ggml_gallocr_new(buft);
        return alloc != nullptr;
    }
    void free_all() {
        if (alloc) ggml_gallocr_free(alloc);
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        alloc = nullptr; buf = nullptr; ctx = nullptr;
    }
};

// Builds the nodes only. Inputs and the allocator come from the pool, so nothing is allocated
// from the driver here.
bool build_moe_pooled(MoeGraph* g, MoePool& pool, ggml_tensor* up, ggml_tensor* gate,
                      ggml_tensor* down, const std::vector<int>& slots) {
    const size_t n_nodes = 32 + slots.size() * 12;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 16) +
                           ggml_graph_overhead_custom(n_nodes, false), nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) return false;
    ggml_context* c = g->ctx;
    g->x = pool.x;
    g->wts = pool.wts;
    g->gf = ggml_new_graph_custom(c, n_nodes, false);
    ggml_tensor* acc = nullptr;
    for (size_t i = 0; i < slots.size(); ++i) {
        const int s = slots[i];
        ggml_tensor* u = ggml_mul_mat(c, expert_view(c, up, s), pool.x);
        ggml_tensor* gt = ggml_silu(c, ggml_mul_mat(c, expert_view(c, gate, s), pool.x));
        ggml_tensor* o = ggml_mul_mat(c, expert_view(c, down, s), ggml_mul(c, u, gt));
        ggml_tensor* wv = ggml_view_2d(c, pool.wts, 1, 1, pool.wts->nb[1],
                                       i * pool.wts->nb[1]);
        acc = acc ? ggml_add(c, acc, ggml_mul(c, o, wv)) : ggml_mul(c, o, wv);
    }
    if (!acc) acc = ggml_scale(c, pool.x, 0.0f);
    g->out = acc;
    ggml_set_output(g->out);
    ggml_build_forward_expand(g->gf, g->out);
    return ggml_gallocr_alloc_graph(pool.alloc, g->gf);
}

// Only the node arena belongs to the graph now; the pool owns everything expensive.
void free_moe_nodes(MoeGraph* g) {
    if (g->ctx) ggml_free(g->ctx);
    g->ctx = nullptr;
    g->alloc = nullptr;
    g->inputs = nullptr;
}

bool build_moe(MoeGraph* g, ggml_backend_buffer_type_t buft, const HParams& h,
               ggml_tensor* up, ggml_tensor* gate, ggml_tensor* down,
               const std::vector<int>& slots) {
    const size_t n_nodes = 32 + slots.size() * 12;
    ggml_init_params ip = {ggml_tensor_overhead() * (n_nodes + 64) +
                           ggml_graph_overhead_custom(n_nodes, false), nullptr, true};
    g->ctx = ggml_init(ip);
    if (!g->ctx) return false;
    ggml_context* c = g->ctx;
    g->x = ggml_new_tensor_2d(c, GGML_TYPE_F32, h.n_embd, 1);
    g->wts = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, int(std::max<size_t>(slots.size(), 1)));
    g->inputs = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!g->inputs) return false;
    g->gf = ggml_new_graph_custom(c, n_nodes, false);

    ggml_tensor* acc = nullptr;
    for (size_t i = 0; i < slots.size(); ++i) {
        const int s = slots[i];
        ggml_tensor* u = ggml_mul_mat(c, expert_view(c, up, s), g->x);
        ggml_tensor* gt = ggml_silu(c, ggml_mul_mat(c, expert_view(c, gate, s), g->x));
        ggml_tensor* o = ggml_mul_mat(c, expert_view(c, down, s), ggml_mul(c, u, gt));
        // The routing weight is data, not a constant, so it arrives as a one-element view
        // and broadcasts along the model dimension.
        ggml_tensor* wv = ggml_view_2d(c, g->wts, 1, 1, g->wts->nb[1],
                                       i * g->wts->nb[1]);
        o = ggml_mul(c, o, wv);
        acc = acc ? ggml_add(c, acc, o) : o;
    }
    if (!acc) {
        // Nothing for this device this token: hand back zeros rather than special-casing the
        // caller, which would otherwise need a branch per layer per token.
        acc = ggml_scale(c, g->x, 0.0f);
    }
    g->out = acc;
    ggml_set_output(g->out);
    ggml_build_forward_expand(g->gf, g->out);
    g->alloc = ggml_gallocr_new(buft);
    return ggml_gallocr_reserve(g->alloc, g->gf) &&
           ggml_gallocr_alloc_graph(g->alloc, g->gf);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string model_path = "D:/Qwen3-Coder-30B-A3B-mx1.gguf";
    std::string prompt = "The quick brown fox jumps over the lazy dog. "
                         "Write a function that reverses a linked list in place.";
    int threads = 4, max_tokens = 32, n_gen = 24, vram_slots = 0, refresh = 3;
    int move_budget_arg = 2;
    float move_margin_arg = 1.5f;
    float bonus = 0.0f;
    bool check_cold_arg = false;
    bool use_gpu = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slots") && i + 1 < argc) vram_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--refresh") && i + 1 < argc) refresh = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-gpu")) use_gpu = false;
        else if (!strcmp(argv[i], "--check-cold")) check_cold_arg = true;
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) move_budget_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--margin") && i + 1 < argc) move_margin_arg = float(atof(argv[++i]));
        else if (!strcmp(argv[i], "--bonus") && i + 1 < argc) bonus = float(atof(argv[++i]));
    }

    HParams h;
    if (!read_hparams(model_path.c_str(), &h)) {
        printf("гиперпараметры не читаются\n");
        return 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { printf("модель не загрузилась\n"); return 1; }
    h.n_vocab = llama_n_vocab(model);
    h.rope_type = int(llama_rope_type(model));

    std::vector<llama_token> toks;
    toks.resize(prompt.size() + 8);
    int n = llama_tokenize(model, prompt.c_str(), int(prompt.size()), toks.data(),
                           int(toks.size()), true, false);
    if (n <= 0) { printf("промпт не токенизировался\n"); return 1; }
    n = std::min(n, max_tokens) / 4 * 4;
    toks.resize(size_t(n));

    Weights w;
    if (!collect(model, h, &w)) return 1;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    ggml_backend_t gpu = nullptr;
    ggml_backend_buffer_type_t gpu_buft = nullptr;
#ifdef GGML_USE_VULKAN
    if (use_gpu && vram_slots > 0) {
        gpu = ggml_backend_vk_init(0);
        if (gpu) gpu_buft = ggml_backend_vk_buffer_type(0);
    }
#endif
    if (vram_slots > 0 && !gpu) {
        printf("карта недоступна, считаю всё на процессоре\n");
        vram_slots = 0;
    }

    Resident res;
    if (!res.init(gpu ? gpu_buft : cpu_buft, h, w, vram_slots)) {
        printf("резидентные эксперты не выделились (слотов %d)\n", vram_slots);
        return 1;
    }
    if (vram_slots > 0) {
        printf("резидентных слотов %d на слой, VRAM %.0f МБ\n", vram_slots,
               res.bytes / 1e6);
    }

    // One pool per device, not per layer. The inputs are overwritten for every layer anyway,
    // and forty-eight separate Vulkan buffers ran the device out of memory at layer fourteen:
    // the driver's allocation granularity is far larger than the eight kilobytes asked for.
    // The cold branch: one graph per layer, built once and never rebuilt. The expert numbers
    // are an input, so a token's choice changes the data and not the graph.
    std::vector<ColdGraph> cold(size_t(h.n_layer));
    for (int il = 0; il < h.n_layer; ++il) {
        if (!cold[size_t(il)].init(cpu_buft, h, w.layers[size_t(il)])) {
            printf("холодный граф слоя %d не собрался\n", il);
            return 1;
        }
    }

    MoePool pool_cpu, pool_gpu;
    if (!pool_cpu.init(cpu_buft, h)) {
        printf("пул процессора не выделился\n");
        return 1;
    }
    if (gpu && !pool_gpu.init(gpu_buft, h)) {
        printf("пул карты не выделился\n");
        return 1;
    }

    const int n_kv_max = pad32(n + n_gen);
    Cache kv;
    if (!kv.init(cpu_buft, h, n_kv_max)) { printf("кэш не выделился\n"); return 1; }

    // The prompt is processed one position at a time. Slower than a batched prefill, but this
    // program exists to measure the generation path, and one code path is easier to trust
    // than two.
    std::vector<float> x, ffn_inp, moe_c, moe_g, wts_c, wts_g, lg;
    x.assign(size_t(h.n_embd), 0.0f);
    ffn_inp.assign(size_t(h.n_embd), 0.0f);
    moe_c.assign(size_t(h.n_embd), 0.0f);
    moe_g.assign(size_t(h.n_embd), 0.0f);
    lg.assign(size_t(h.n_vocab), 0.0f);
    std::vector<int32_t> ids, ids_cold;
    std::vector<float> wt, probs;
    ids.resize(size_t(h.n_expert_used));
    ids_cold.resize(size_t(h.n_expert_used));
    wt.resize(size_t(h.n_expert_used));
    probs.resize(size_t(h.n_expert));
    std::vector<int32_t> pos1(1);
    std::vector<float> mask1;

    // Output head, on its own small graph: it is 243 MB of reading per token and does not
    // belong inside the per-layer loop.
    Graph head;
    ggml_tensor *head_in = nullptr, *head_out = nullptr;
    {
        ggml_init_params ip = {ggml_tensor_overhead() * 32 +
                               ggml_graph_overhead_custom(16, false), nullptr, true};
        head.ctx = ggml_init(ip);
        head_in = ggml_new_tensor_2d(head.ctx, GGML_TYPE_F32, h.n_embd, 1);
        head.inputs = ggml_backend_alloc_ctx_tensors_from_buft(head.ctx, cpu_buft);
        head.gf = ggml_new_graph_custom(head.ctx, 16, false);
        head_out = ggml_mul_mat(head.ctx, w.out,
                                norm(head.ctx, head_in, w.out_norm, h.rms_eps));
        ggml_set_output(head_out);
        ggml_build_forward_expand(head.gf, head_out);
        head.alloc = ggml_gallocr_new(cpu_buft);
        if (!ggml_gallocr_reserve(head.alloc, head.gf) ||
            !ggml_gallocr_alloc_graph(head.alloc, head.gf)) {
            printf("граф головы не собрался\n");
            return 1;
        }
    }

    // Statistics for the refresh: how often each expert of each layer was chosen.
    std::vector<std::vector<int>> hits(size_t(h.n_layer),
                                       std::vector<int>(size_t(h.n_expert), 0));
    std::vector<std::vector<int>> stat_hits(size_t(h.n_layer),
                                            std::vector<int>(size_t(h.n_expert), 0));
    long long gpu_hits = 0, total_hits = 0;
    bool check_cold = check_cold_arg;
    int gap_left = 4;
    double cold_worst = 0.0;
    int cold_worst_n = 0;
    double t_gpu_wait = 0.0, t_cpu_moe = 0.0, t_attn = 0.0, t_refresh = 0.0;
    // Graph construction and teardown, timed separately. On the card these involve driver
    // allocations, and a design that rebuilds per layer per token does 576 of them during a
    // short generation - which is exactly the kind of cost that hides outside the obvious
    // timers and makes a working idea look broken.
    double t_gpu_graph = 0.0, t_cpu_graph = 0.0;

    auto one_position = [&](llama_token tok, int past, bool want_logits) -> llama_token {
        // Embedding row for this token, straight out of the table.
        {
            const size_t row = size_t(tok) * size_t(h.n_embd);
            if (w.tok_embd->type == GGML_TYPE_F32) {
                std::memcpy(x.data(), (const char*)w.tok_embd->data + row * sizeof(float),
                            sizeof(float) * size_t(h.n_embd));
            } else {
                // Dequantise one row through ggml's own converter rather than by hand.
                const ggml_type_traits_t tt = ggml_internal_get_type_traits(w.tok_embd->type);
                const size_t rb = ggml_nbytes(w.tok_embd) / size_t(w.tok_embd->ne[1]);
                tt.to_float((const char*)w.tok_embd->data + size_t(tok) * rb, x.data(),
                            h.n_embd);
            }
        }
        pos1[0] = past;
        mask1.assign(size_t(n_kv_max), -INFINITY);
        for (int j = 0; j <= past; ++j) mask1[size_t(j)] = 0.0f;

        for (int il = 0; il < h.n_layer; ++il) {
            const auto t0 = Clock::now();
            AttnGraph ag;
            if (!build_attn(&ag, cpu_buft, h, w, kv, il, past, n_kv_max)) {
                printf("граф внимания слоя %d не собрался\n", il);
                exit(1);
            }
            ggml_backend_tensor_set(ag.inp, x.data(), 0, ggml_nbytes(ag.inp));
            ggml_backend_tensor_set(ag.positions, pos1.data(), 0, sizeof(int32_t));
            ggml_backend_tensor_set(ag.mask, mask1.data(), 0, ggml_nbytes(ag.mask));
            ggml_backend_graph_compute(cpu, ag.gf);
            ggml_backend_tensor_get(ag.ffn_inp, ffn_inp.data(), 0, ggml_nbytes(ag.ffn_inp));
            ggml_backend_tensor_get(ag.x_normed, x.data(), 0, ggml_nbytes(ag.x_normed));
            ggml_backend_tensor_get(ag.probs, probs.data(), 0, ggml_nbytes(ag.probs));
            ag.free_all();
            t_attn += ms_since(t0);

            // Routing, on the host. A resident expert gets a bonus before the comparison, so
            // a marginal pick is nudged towards weights that are already on the card - the
            // cache-conditional routing this project measured earlier at 67.3% to 79.1% hits
            // for +0.73% perplexity. The *weights* used afterwards are the unbiased
            // probabilities: the bonus decides who is chosen, never how much they count, or
            // the output would be scaled by an arbitrary constant.
            {
                std::vector<int> order;
                order.resize(size_t(h.n_expert));
                for (int e = 0; e < h.n_expert; ++e) order[size_t(e)] = e;
                const bool biased = bonus > 0.0f && vram_slots > 0;
                std::partial_sort(order.begin(), order.begin() + h.n_expert_used,
                                  order.end(), [&](int a, int b) {
                    float pa = probs[size_t(a)], pb = probs[size_t(b)];
                    if (biased) {
                        if (res.slot_of[size_t(il)][size_t(a)] >= 0) pa += bonus;
                        if (res.slot_of[size_t(il)][size_t(b)] >= 0) pb += bonus;
                    }
                    return pa > pb;
                });
                float sum = 0.0f;
                for (int j = 0; j < h.n_expert_used; ++j) {
                    ids[size_t(j)] = order[size_t(j)];
                    wt[size_t(j)] = probs[size_t(order[size_t(j)])];
                    sum += wt[size_t(j)];
                }
                if (sum > 0.0f) {
                    for (int j = 0; j < h.n_expert_used; ++j) wt[size_t(j)] /= sum;
                }
                // Statistics are gathered from the *unbiased* choice. Counting the biased one
                // instead makes the resident set popular by construction and it never changes
                // again - the refresh cost fell from 194 ms to 7 ms when that happened, which
                // looked like an improvement and was actually the policy going deaf. This
                // project has made that mistake once before, in the expert-residency driver.
                if (biased) {
                    std::vector<int> plain;
                    plain.resize(size_t(h.n_expert));
                    for (int e = 0; e < h.n_expert; ++e) plain[size_t(e)] = e;
                    std::partial_sort(plain.begin(), plain.begin() + h.n_expert_used,
                                      plain.end(), [&](int a, int b) {
                        return probs[size_t(a)] > probs[size_t(b)];
                    });
                    for (int j = 0; j < h.n_expert_used; ++j) {
                        stat_hits[size_t(il)][size_t(plain[size_t(j)])]++;
                    }
                } else {
                    for (int j = 0; j < h.n_expert_used; ++j) {
                        stat_hits[size_t(il)][size_t(ids[size_t(j)])]++;
                    }
                }
            }

            // Split the chosen experts by where their weights already are.
            std::vector<int> slots_g;
            wts_g.clear();
            ids_cold.assign(size_t(h.n_expert_used), -1);
            for (int j = 0; j < h.n_expert_used; ++j) {
                const int e = ids[size_t(j)];
                if (e < 0 || e >= h.n_expert) continue;
                hits[size_t(il)][size_t(e)]++;
                total_hits++;
                const int s = vram_slots > 0 ? res.slot_of[size_t(il)][size_t(e)] : -1;
                if (s >= 0) {
                    slots_g.push_back(s);
                    wts_g.push_back(wt[size_t(j)]);
                    gpu_hits++;
                } else {
                    ids_cold[size_t(j)] = e;   // the rest keep -1 and are skipped
                }
            }

            // The card is launched first and asynchronously, so its work overlaps the
            // processor's share. Without this the split is worth about six per cent.
            MoeGraph mg;
            bool have_gpu_work = false;
            if (gpu && !slots_g.empty()) {
                const auto tg = Clock::now();
                const bool built_g = build_moe_pooled(&mg, pool_gpu,
                                                      res.up[size_t(il)],
                                                      res.gate[size_t(il)],
                                                      res.down[size_t(il)], slots_g);
                t_gpu_graph += ms_since(tg);
                if (built_g) {
                    // Asynchronous writes, queued on the same stream as the compute. The
                    // synchronous form waits for a staging copy and a submit each time; at
                    // 1344 calls over a short run that was 450 ms of pure waiting, hidden
                    // outside every timer in this program.
                    ggml_backend_tensor_set_async(gpu, mg.x, x.data(), 0,
                                                  ggml_nbytes(mg.x));
                    ggml_backend_tensor_set_async(gpu, mg.wts, wts_g.data(), 0,
                                                  sizeof(float) * wts_g.size());
                    ggml_backend_graph_compute_async(gpu, mg.gf);
                    have_gpu_work = true;
                }
            }

            const auto t1 = Clock::now();
            ColdGraph& cg = cold[size_t(il)];
            ggml_backend_tensor_set(cg.x, x.data(), 0, ggml_nbytes(cg.x));
            ggml_backend_tensor_set(cg.ids, ids_cold.data(), 0, ggml_nbytes(cg.ids));
            ggml_backend_tensor_set(cg.wts, wt.data(), 0, ggml_nbytes(cg.wts));
            ggml_backend_graph_compute(cpu, cg.gf);
            ggml_backend_tensor_get(cg.out, moe_c.data(), 0, ggml_nbytes(cg.out));
            t_cpu_moe += ms_since(t1);

            // Two implementations of the same sum, compared against each other. The batched
            // form became suspect when correctness turned out to *improve* as the card took
            // more experts off it - one of 12 tokens matched with eight experts on the
            // processor, twelve of 12 with six. That ordering points at the batched call, and
            // guessing between them is what this settles.
            if (check_cold) {
                std::vector<int> ec;
                std::vector<float> wc;
                for (int j = 0; j < h.n_expert_used; ++j) {
                    if (ids_cold[size_t(j)] >= 0) {
                        ec.push_back(ids_cold[size_t(j)]);
                        wc.push_back(wt[size_t(j)]);
                    }
                }
                MoeGraph ref;
                if (build_moe_pooled(&ref, pool_cpu, w.layers[size_t(il)].up,
                                     w.layers[size_t(il)].gate, w.layers[size_t(il)].down,
                                     ec)) {
                    ggml_backend_tensor_set(ref.x, x.data(), 0, ggml_nbytes(ref.x));
                    if (!wc.empty()) {
                        ggml_backend_tensor_set(ref.wts, wc.data(), 0,
                                                sizeof(float) * wc.size());
                    }
                    ggml_backend_graph_compute(cpu, ref.gf);
                    std::vector<float> alt;
                    alt.resize(size_t(h.n_embd));
                    ggml_backend_tensor_get(ref.out, alt.data(), 0, ggml_nbytes(ref.out));
                    double num = 0.0, den = 0.0;
                    for (int i = 0; i < h.n_embd; ++i) {
                        const double d = double(alt[size_t(i)]) - double(moe_c[size_t(i)]);
                        num += d * d;
                        den += double(alt[size_t(i)]) * double(alt[size_t(i)]);
                    }
                    printf("  [сверка] экспертов на процессоре %zu, батчем против поштучно: "
                           "%.4f%%\n", ec.size(),
                           den > 0.0 ? 100.0 * std::sqrt(num / den) : -1.0);
                    free_moe_nodes(&ref);
                }
                check_cold = false;
            }

            std::fill(moe_g.begin(), moe_g.end(), 0.0f);
            if (have_gpu_work) {
                const auto t2 = Clock::now();
                ggml_backend_synchronize(gpu);
                ggml_backend_tensor_get(mg.out, moe_g.data(), 0, ggml_nbytes(mg.out));
                t_gpu_wait += ms_since(t2);
                const auto t3 = Clock::now();
                free_moe_nodes(&mg);
                t_gpu_graph += ms_since(t3);
            }

            for (int i = 0; i < h.n_embd; ++i) {
                x[size_t(i)] = ffn_inp[size_t(i)] + moe_c[size_t(i)] + moe_g[size_t(i)];
            }
        }

        if (!want_logits) return -1;
        ggml_backend_tensor_set(head_in, x.data(), 0, ggml_nbytes(head_in));
        ggml_backend_graph_compute(cpu, head.gf);
        ggml_backend_tensor_get(head_out, lg.data(), 0, sizeof(float) * size_t(h.n_vocab));
        const llama_token best = llama_token(std::max_element(lg.begin(), lg.end()) -
                                             lg.begin());
        // How close the decision was. Greedy sampling turns a near-tie into a coin flip
        // between implementations that differ only in rounding, and this engine sums each
        // layer's expert shares on the host rather than inside one graph - so its rounding is
        // genuinely different from the reference's. Without this number, "the text diverged"
        // cannot be told apart from "the arithmetic is wrong".
        if (gap_left > 0) {
            float second = -1e30f;
            for (int i = 0; i < h.n_vocab; ++i) {
                if (i != int(best) && lg[size_t(i)] > second) second = lg[size_t(i)];
            }
            printf("  [зазор] лучший %.5f, второй %.5f, разница %.5f\n",
                   lg[size_t(best)], second, lg[size_t(best)] - second);
            --gap_left;
        }
        return best;
    };

    // Refresh: every `refresh` tokens the resident set is rebuilt from the statistics so far.
    // Copies go to the card between positions, off the layer loop, which is what keeps them
    // from stalling generation.
    // Bounded, hysteretic, and staged. The unbounded version of this replaced almost the
    // whole resident set on its first call - 768 experts, 1.9 GB over a 3.94 GB/s link, 490
    // ms - and turned a design that wins 32 ms per token into one that loses 63.
    //
    // So: at most `budget` moves per layer, and a candidate must beat the weakest resident
    // by a margin before it displaces anything. The margin is what stops two experts of
    // similar popularity from trading places forever, which is the same thrashing the
    // precision ladder had to be taught to avoid.
    const int move_budget = move_budget_arg;
    const float move_margin = move_margin_arg;
    long long moves = 0;
    auto do_refresh = [&]() {
        if (vram_slots <= 0 || !gpu) return;
        const auto t0 = Clock::now();
        // Whatever was issued last time has had a whole refresh interval to land.
        ggml_backend_synchronize(gpu);
        res.activate();
        for (int il = 0; il < h.n_layer; ++il) {
            const std::vector<int>& hl = stat_hits[size_t(il)];
            // Weakest residents first: those are the ones worth giving up.
            // resize, never a constructor argument: this is the eighth time the most vexing
            // parse has been written in this project by accident
            std::vector<int> slots;
            slots.resize(size_t(vram_slots));
            for (int s = 0; s < vram_slots; ++s) slots[size_t(s)] = s;
            std::sort(slots.begin(), slots.end(), [&](int a, int b) {
                const int ea = res.expert_of[size_t(il)][size_t(a)];
                const int eb = res.expert_of[size_t(il)][size_t(b)];
                return (ea < 0 ? -1 : hl[size_t(ea)]) < (eb < 0 ? -1 : hl[size_t(eb)]);
            });
            // Best non-resident candidates.
            std::vector<int> cand;
            cand.reserve(size_t(h.n_expert));
            for (int e = 0; e < h.n_expert; ++e) {
                if (res.slot_of[size_t(il)][size_t(e)] < 0) cand.push_back(e);
            }
            std::partial_sort(cand.begin(),
                              cand.begin() + std::min<size_t>(cand.size(),
                                                              size_t(move_budget)),
                              cand.end(),
                              [&](int a, int b) { return hl[size_t(a)] > hl[size_t(b)]; });
            int done = 0;
            for (size_t ci = 0; ci < cand.size() && done < move_budget; ++ci) {
                const int want = cand[ci];
                const int slot = slots[size_t(done)];
                const int held = res.expert_of[size_t(il)][size_t(slot)];
                const int held_hits = held < 0 ? -1 : hl[size_t(held)];
                if (double(hl[size_t(want)]) <= double(move_margin) * double(held_hits)) {
                    break;      // nothing here is worth the transfer
                }
                res.place_async(gpu, w, il, slot, want);
                moves++;
                done++;
            }
        }
        t_refresh += ms_since(t0);
    };

    printf("модель %s, слоёв %d, экспертов %d из них %d\n",
           h.arch.c_str(), h.n_layer, h.n_expert, h.n_expert_used);
    printf("промпт %d токенов, генерация %d, слотов в VRAM %d, обновление каждые %d\n",
           n, n_gen, vram_slots, refresh);

    // The prompt runs with an empty resident set. That costs nothing to arrange and gives the
    // statistics their first real data; choosing experts by index instead would put the wrong
    // ones on the card and then pay to replace all of them.
    for (int i = 0; i < n - 1; ++i) one_position(toks[size_t(i)], i, false);
    llama_token next = one_position(toks[size_t(n - 1)], n - 1, true);

    // Priming: one unbounded placement from what the prompt itself routed to. It happens
    // before generation, so it is on nobody's critical path - and the prompt's own routes were
    // measured earlier to reach 91% of what the whole text would have selected.
    if (vram_slots > 0 && gpu) {
        const auto t_prime = Clock::now();
        for (int il = 0; il < h.n_layer; ++il) {
            std::vector<int> order;
            order.resize(size_t(h.n_expert));
            for (int e = 0; e < h.n_expert; ++e) order[size_t(e)] = e;
            std::partial_sort(order.begin(), order.begin() + vram_slots, order.end(),
                              [&](int a, int b) {
                                  return stat_hits[size_t(il)][size_t(a)] >
                                         stat_hits[size_t(il)][size_t(b)];
                              });
            for (int s = 0; s < vram_slots; ++s) res.place(w, il, s, order[size_t(s)]);
        }
        ggml_backend_synchronize(gpu);
        printf("прогрев набора по промпту: %.0f мс\n", ms_since(t_prime));
    }

    std::vector<llama_token> ours;
    const auto t_gen = Clock::now();
    for (int i = 0; i < n_gen; ++i) {
        ours.push_back(next);
        if (refresh > 0 && i > 0 && i % refresh == 0) do_refresh();
        next = one_position(next, n + i, true);
    }
    const double gen_ms = ms_since(t_gen);

    // The reference, greedily, from the same prompt.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn = false;
    llama_context* lctx = llama_init_from_model(model, cp);
    std::vector<llama_token> ref;
    if (lctx && !llama_decode(lctx, llama_batch_get_one(toks.data(), n, 0, 0))) {
        const float* rl = llama_get_logits(lctx);
        llama_token r = llama_token(std::max_element(rl, rl + h.n_vocab) - rl);
        for (int i = 0; i < n_gen; ++i) {
            ref.push_back(r);
            if (llama_decode(lctx, llama_batch_get_one(&r, 1, n + i, 0))) break;
            rl = llama_get_logits(lctx);
            r = llama_token(std::max_element(rl, rl + h.n_vocab) - rl);
        }
    }
    int same = 0;
    while (same < int(std::min(ours.size(), ref.size())) &&
           ours[size_t(same)] == ref[size_t(same)]) ++same;

    std::string txt;
    char piece[128];
    for (llama_token t : ours) {
        const int len = llama_token_to_piece(model, t, piece, sizeof(piece) - 1, 0, true);
        if (len > 0) txt.append(piece, size_t(len));
    }
    std::string rtxt;
    for (llama_token t : ref) {
        const int len = llama_token_to_piece(model, t, piece, sizeof(piece) - 1, 0, true);
        if (len > 0) rtxt.append(piece, size_t(len));
    }
    printf("\nсовпало с эталоном подряд: %d из %d\n", same, n_gen);
    printf("наш текст: %s\n", txt.c_str());
    printf("эталон   : %s\n", rtxt.c_str());
    printf("\nскорость: %.2f ток/с (%.1f мс на токен)\n",
           1000.0 * n_gen / gen_ms, gen_ms / n_gen);
    printf("попаданий в резидентных: %lld из %lld (%.1f%%)\n", gpu_hits, total_hits,
           total_hits ? 100.0 * double(gpu_hits) / double(total_hits) : 0.0);
    printf("время: внимание %.0f мс, холодные эксперты %.0f мс, ожидание карты %.0f мс, "
           "обновление резидентных %.0f мс\n", t_attn, t_cpu_moe, t_gpu_wait, t_refresh);
    printf("сборка и снос графов карты: %.0f мс\n", t_gpu_graph);

    for (auto& c : cold) c.free_all();
    head.free_all();
    kv.free_all();
    res.free_all();
    if (lctx) llama_free(lctx);
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    llama_free_model(model);
    llama_backend_free();
    return same == n_gen ? 0 : 2;
}
