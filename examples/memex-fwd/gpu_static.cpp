#include "gpu_static.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "ggml-alloc.h"
#include "ggml-vulkan.h"
// Only for the two heap reports, which are static members there and are the one thing in this
// arrangement that cannot be checked any other way: "device-local" in every other report is
// exactly the word that hides a buffer the driver quietly moved to system memory.
#include "gpu_experts.hpp"

// MEMEX_STATIC_TRUNC, read once. See the truncation block in the layer graph builder for what it
// is for; in short, it is the only instrument on this backend that can attribute device cost,
// because per-node timestamps cannot (METHODS 80).
static int memex_static_trunc() {
    static const int v = getenv("MEMEX_STATIC_TRUNC") ? atoi(getenv("MEMEX_STATIC_TRUNC")) : 0;
    return v;
}

namespace memex {

namespace {

constexpr std::size_t kBarHeapCeiling = 256u * 1024u * 1024u;
// How far past the ceiling to land. find_properties compares heap.size against the buffer
// size, so anything strictly greater than the BAR heap cannot be typed onto it; a margin is
// kept anyway because the comparison is against the heap's declared size and a driver that
// reported 256 MiB + epsilon would otherwise put us back in the window.
constexpr std::size_t kOverBarMargin = 8u * 1024u * 1024u;
// Uploads go in pieces this big. ggml_vk_buffer_write_2d_async takes the staging path for a
// non-pinned source and calls ggml_vk_ensure_sync_staging_buffer with the WHOLE copy size
// (ggml-vulkan.cpp:4687-4691), so one 243 MiB set_tensor would ask the driver for a 243 MiB
// host-visible buffer - which is both a large allocation that can fail and, if it succeeds,
// exactly the kind of thing that competes for the BAR window this class works to stay out of.
constexpr std::size_t kUploadChunk = 16u * 1024u * 1024u;

inline double ms_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

// The interleaved repacked forms are the ones with no Vulkan implementation for any operation,
// so an interleaved head has to be refused by name here rather than found later as a wrong
// answer. ggml_backend_supports_op would also refuse it, but this message names the cause.
bool type_is_interleaved(ggml_type t) {
    const char* n = ggml_type_name(t);
    if (!n) return false;
    const std::size_t len = std::strlen(n);
    // _r4 / _r8 / _rN at the end of the name.
    for (std::size_t i = 0; i + 2 < len + 1; ++i) {
        if (n[i] == '_' && (n[i + 1] == 'r' || n[i + 1] == 'R') && i + 2 < len &&
            n[i + 2] >= '0' && n[i + 2] <= '9') {
            return true;
        }
    }
    return false;
}

inline uint32_t xrand(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

inline float xuni(uint32_t& s, float scale) {
    return scale * (float(xrand(s) & 0xffffu) / 32768.0f - 1.0f);
}

// One submit per device graph. Set before the device is created, because ggml reads these
// once at device init (ggml-vulkan.cpp:3905). Not overwritten if the environment already
// carries a value: a sweep over the divisor has to be able to say so.
void prefer_one_submit_per_graph() {
#ifdef _WIN32
    if (!getenv("GGML_VK_SUBMIT_DIVISOR")) _putenv_s("GGML_VK_SUBMIT_DIVISOR", "1");
    if (!getenv("GGML_VK_SUBMIT_TAIL"))    _putenv_s("GGML_VK_SUBMIT_TAIL", "0");
#else
    if (!getenv("GGML_VK_SUBMIT_DIVISOR")) setenv("GGML_VK_SUBMIT_DIVISOR", "1", 0);
    if (!getenv("GGML_VK_SUBMIT_TAIL"))    setenv("GGML_VK_SUBMIT_TAIL", "0", 0);
#endif
}

// The tensors of one layer, always in this order, so that a group's byte estimate and its actual
// allocation cannot drift apart. Parallel arrays is how a wq gets uploaded where a wk belongs,
// with no shape error anywhere to catch it.
//
// SLOT 9 IS OPTIONAL and every user must treat it so. It is gemma4's rope_freqs: per-dimension
// divisors of the rope angle, 1.0 for the first 64 pairs and 1e30 for the remaining 192, which
// drives theta to zero and leaves dimensions 128..511 unrotated. Only gemma4's full-attention
// layers carry one; qwen3moe has none anywhere. Dropping it does not fail and does not change a
// shape - it rotates what must not move, which is the worst way for an input to be missing.
void layer_slots(const GpuStaticLayer& L, ggml_tensor** out) {
    out[0] = L.attn_norm; out[1] = L.wq; out[2] = L.wk; out[3] = L.wv; out[4] = L.wo;
    out[5] = L.q_norm;    out[6] = L.k_norm; out[7] = L.ffn_norm; out[8] = L.router;
    out[9]  = L.rope_freqs;      // may be null
    out[10] = L.post_attn_norm;  // gemma4 only
    out[11] = L.gate_inp_s;      // gemma4 only
    out[12] = L.pre_ffw_norm_2;  // gemma4 only
}

void layer_unslot(GpuStaticLayer& L, ggml_tensor* const* in) {
    L.attn_norm = in[0]; L.wq = in[1]; L.wk = in[2]; L.wv = in[3]; L.wo = in[4];
    L.q_norm    = in[5]; L.k_norm = in[6]; L.ffn_norm = in[7]; L.router = in[8];
    L.rope_freqs     = in[9];
    L.post_attn_norm = in[10];
    L.gate_inp_s     = in[11];
    L.pre_ffw_norm_2 = in[12];
}

}  // namespace

GpuStatic::~GpuStatic() { shutdown(); }

const char* GpuStatic::head_type_name() const {
    return d_out_ ? ggml_type_name(d_out_->type) : "-";
}

// ---------------------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------------------

bool GpuStatic::alloc_head(ggml_tensor* out, std::string* err) {
    const std::size_t need = ggml_nbytes(out);
    const std::size_t max_buf = ggml_backend_buft_get_max_size(buft_);
    if (need > max_buf) {
        char b[256];
        snprintf(b, sizeof(b),
                 "golova %.1f MiB ne vlezaet v odin bufer bekenda (%.1f MiB)",
                 double(need) / 1048576.0, double(max_buf) / 1048576.0);
        *err = b;
        return false;
    }
    // The whole reason this function is not three lines. A buffer that FITS the BAR window is
    // put in it, and once the window is committed the driver backs further allocations there
    // with system memory: shader reads fall to 3.1 GB/s while heapUsage still says
    // device-local. So if what we actually need is under the ceiling, ask for more.
    std::size_t pad = 0;
    if (need <= kBarHeapCeiling) {
        pad = kBarHeapCeiling + kOverBarMargin - need;
        // ggml pads each tensor to the buffer type's alignment, so the sum of the two tensors
        // is at least need + pad and the ceiling is cleared with room to spare either way.
        if (need + pad > max_buf) {
            *err = "bufer, dostatochno bolshoj chtoby ne sest v BAR-kuchu, prevyshaet potolok "
                   "bekenda - golovu na kartu na etom ustrojstve polozhit nelzja";
            return false;
        }
    }

    ggml_init_params ip = {ggml_tensor_overhead() * 4 + 4096, nullptr, true};
    ctx_w_ = ggml_init(ip);
    if (!ctx_w_) { *err = "ggml_init dlja vesov ne udalsja"; return false; }

    d_out_ = ggml_new_tensor_2d(ctx_w_, out->type, out->ne[0], out->ne[1]);
    ggml_set_name(d_out_, "vk.output.weight");
    if (ggml_nbytes(d_out_) != need) {
        *err = "razmer golovy v videopamjati ne sovpal s razmerom v modeli";
        ggml_free(ctx_w_); ctx_w_ = nullptr;
        return false;
    }
    if (pad > 0) {
        d_pad_ = ggml_new_tensor_1d(ctx_w_, GGML_TYPE_F32, int64_t(pad / sizeof(float)));
        ggml_set_name(d_pad_, "vk.bar.padding");
    }

    buf_w_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w_, buft_);
    if (!buf_w_) {
        char b[200];
        snprintf(b, sizeof(b), "ne vydelilos %.1f MiB pod golovu",
                 double(need + pad) / 1048576.0);
        *err = b;
        ggml_free(ctx_w_); ctx_w_ = nullptr;
        d_out_ = nullptr; d_pad_ = nullptr;
        return false;
    }
    const std::size_t sz = ggml_backend_buffer_get_size(buf_w_);
    vram_bytes_ += sz;
    GpuStaticBuffer bi;
    bi.what = "golova output.weight";
    bi.bytes = sz;
    bi.padding = pad;
    bi.over_bar = sz > kBarHeapCeiling;
    bufs_info_.push_back(bi);
    if (!bi.over_bar) {
        // Never reached with the padding above, and checked anyway: this is the one failure
        // that measures as a working feature.
        *err = "bufer golovy <= 256 MiB - on sjadet v BAR-kuchu i chtenija shejdera upadut v "
               "sorok raz";
        return false;
    }

    // In pieces, so the driver is never asked for a staging buffer the size of the head.
    const char* src = (const char*)out->data;
    for (std::size_t off = 0; off < need; off += kUploadChunk) {
        const std::size_t n = std::min(kUploadChunk, need - off);
        ggml_backend_tensor_set(d_out_, src + off, off, n);
    }
    return true;
}

bool GpuStatic::build_graphs(std::string* err) {
    const int mr = cfg_.max_rows;
    {
        ggml_init_params ip = {ggml_tensor_overhead() * 4 + 4096, nullptr, true};
        ctx_in_ = ggml_init(ip);
        if (!ctx_in_) { *err = "ggml_init dlja vhoda ne udalsja"; return false; }
        t_x_ = ggml_new_tensor_2d(ctx_in_, GGML_TYPE_F32, cfg_.n_embd, mr);
        ggml_set_name(t_x_, "vk.head.x");
        buf_in_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_in_, buft_);
        if (!buf_in_) { *err = "vhodnoj bufer ne vydelilsja"; return false; }
    }
    {
        const std::size_t nodes = 8;
        ggml_init_params ip = {
            (ggml_tensor_overhead() * (nodes + 8) + ggml_graph_overhead_custom(nodes, false))
                * std::size_t(mr) + 4096,
            nullptr, true};
        ctx_g_ = ggml_init(ip);
        if (!ctx_g_) { *err = "ggml_init dlja grafov ne udalsja"; return false; }
        gf_.assign(std::size_t(mr) + 1u, nullptr);
        ga_.assign(std::size_t(mr) + 1u, nullptr);
        n_out_.assign(std::size_t(mr) + 1u, nullptr);
    }
    // Width one eagerly, because it is the width generation uses and because the op support
    // check has to happen at init rather than on the first token. The rest are built when a
    // width is first asked for: each one's output is 594 KiB per row of video memory, and a
    // run that never verifies a draft never needs them.
    return true;
}

// ---------------------------------------------------------------------------------------
// init / shutdown
// ---------------------------------------------------------------------------------------

bool GpuStatic::init(const GpuStaticConfig& cfg, ggml_tensor* out, std::string* err) {
    cfg_ = cfg;
    if (cfg_.n_embd <= 0 || cfg_.n_vocab <= 0 || cfg_.max_rows <= 0) {
        *err = "bessmyslennaja konfiguracija";
        return false;
    }
    if (!out) { *err = "output.weight otsutstvuet"; return false; }
    if (out->ne[0] != cfg_.n_embd || out->ne[1] != cfg_.n_vocab) {
        char b[200];
        snprintf(b, sizeof(b),
                 "output.weight [%lld,%lld] ne sovpal s n_embd %d / n_vocab %d",
                 (long long)out->ne[0], (long long)out->ne[1], cfg_.n_embd, cfg_.n_vocab);
        *err = b;
        return false;
    }
    if (!out->data || !out->buffer || !ggml_backend_buffer_is_host(out->buffer)) {
        *err = "output.weight nedostupen hostu - zalivat neotkuda";
        return false;
    }
    if (type_is_interleaved(out->type)) {
        char b[280];
        snprintf(b, sizeof(b),
                 "golova perepakovana v %s, a formy _R* Vulkan ne chitaet ni odnoj operaciej; "
                 "iskljuchite golovu iz perepakovki (po umolchaniju perepakovyvajutsja tolko "
                 "eksperty)", ggml_type_name(out->type));
        *err = b;
        return false;
    }
    src_out_ = out;

    if (cfg_.layers) {
        if (cfg_.n_layer <= 0 || cfg_.n_head <= 0 || cfg_.n_head_kv <= 0 ||
            cfg_.head_dim <= 0 || cfg_.n_expert <= 0) {
            *err = "sloi na kartu: geometrija ne zapolnena";
            return false;
        }
        // Before the device exists, because ggml reads the submit knobs once at device init.
        prefer_one_submit_per_graph();
    }

    be_ = ggml_backend_vk_init(0);
    if (!be_) { *err = "ggml_backend_vk_init(0) ne udalsja"; return false; }
    buft_ = ggml_backend_vk_buffer_type(0);
    if (!buft_) { *err = "ggml_backend_vk_buffer_type(0) ne udalsja"; shutdown(); return false; }
    {
        char d[256] = {0};
        ggml_backend_vk_get_device_description(0, d, sizeof(d));
        dev_name_ = d;
    }

    GpuExperts::print_heaps("do razmeshchenija staticheskoj golovy");
    if (!alloc_head(out, err)) { shutdown(); return false; }
    if (!build_graphs(err)) { shutdown(); return false; }

    // Pinned host memory for the readback: ggml_vk_buffer_read_2d_async looks the destination
    // up in the pinned registry (ggml_vk_host_get) and copies straight into it when it finds
    // it, and pays one more hop through the backend's shared staging buffer when it does not.
    {
        const std::size_t need =
            std::size_t(cfg_.n_vocab) * std::size_t(cfg_.max_rows) * sizeof(float);
        ggml_backend_buffer_type_t hb = ggml_backend_vk_host_buffer_type();
        if (hb) buf_rb_ = ggml_backend_buft_alloc_buffer(hb, need);
        if (buf_rb_) rb_ = (float*)ggml_backend_buffer_get_base(buf_rb_);
        if (!rb_) {
            rb_fallback_.assign(need / sizeof(float), 0.0f);
            rb_ = rb_fallback_.data();
        }
    }

    // One dispatch of width one, thrown away. Three things happen on a first dispatch that
    // must not be charged to the first generated token: the Q6_K mul_mat_vec pipeline is
    // compiled, this width's graph is built, and its allocator runs. All three are one-off and
    // all three would otherwise land inside the interval the engine reports as generation.
    // Doing it here also moves the ggml_backend_supports_op check to init, which is where a
    // refusal belongs - a head in a type the backend has no pipeline for aborts inside the
    // backend rather than returning an error, so it has to be caught before any token.
    {
        std::vector<float> zx(std::size_t(cfg_.n_embd), 0.0f);
        std::vector<float> zo(std::size_t(cfg_.n_vocab), 0.0f);
        if (!block(zx.data(), 1, zo.data())) {
            *err = fail_msg_;
            shutdown();
            return false;
        }
        st_ = GpuStaticStats();
    }
    GpuExperts::print_heaps("posle razmeshchenija staticheskoj golovy");
    printf("  --- kuda bufer OBJAZAN byl lech (pravilo find_properties) ---\n");
    for (const GpuStaticBuffer& b : bufs_info_) {
        GpuExperts::print_placement(b.what.c_str(), b.bytes);
    }
    if (buf_in_) {
        GpuExperts::print_placement("vhod golovy",
                                    ggml_backend_buffer_get_size(buf_in_));
    }

    if (cfg_.verify) {
        std::string verr;
        if (!verify_head(&verr)) {
            *err = "proverka bajtov golovy v videopamjati ne proshla: " + verr;
            shutdown();
            return false;
        }
        printf("  bajty golovy v videopamjati sovpadajut s modelju: %.1f MiB provereno\n",
               double(ggml_nbytes(d_out_)) / 1048576.0);
    }
    return true;
}

void GpuStatic::shutdown() {
    for (LayerGraph& G : lg_) {
        if (G.ga) ggml_gallocr_free(G.ga);
    }
    lg_.clear();
    sites_.clear();
    if (ctx_lg_) { ggml_free(ctx_lg_); ctx_lg_ = nullptr; }
    if (buf_lin_) { ggml_backend_buffer_free(buf_lin_); buf_lin_ = nullptr; }
    if (ctx_lin_) { ggml_free(ctx_lin_); ctx_lin_ = nullptr; }
    t_lx_ = nullptr; t_pos_ = nullptr; t_mask_ = nullptr;
    for (ggml_backend_buffer_t b : bufs_l_) {
        if (b) ggml_backend_buffer_free(b);
    }
    bufs_l_.clear();
    for (ggml_context* cx : ctxs_l_) {
        if (cx) ggml_free(cx);
    }
    ctxs_l_.clear();
    lw_.clear();
    kv_k_.clear();
    kv_v_.clear();
    kv_pad_ = nullptr;
    step_past_ = -1;
    step_nkv_ = 0;
    step_mask_sent_ = false;
    step_mask_src_  = nullptr;

    for (ggml_gallocr_t g : ga_) {
        if (g) ggml_gallocr_free(g);
    }
    ga_.clear();
    gf_.clear();
    n_out_.clear();
    if (ctx_g_) { ggml_free(ctx_g_); ctx_g_ = nullptr; }
    if (buf_in_) { ggml_backend_buffer_free(buf_in_); buf_in_ = nullptr; }
    if (ctx_in_) { ggml_free(ctx_in_); ctx_in_ = nullptr; }
    t_x_ = nullptr;
    if (buf_rb_) { ggml_backend_buffer_free(buf_rb_); buf_rb_ = nullptr; }
    rb_ = nullptr;
    rb_fallback_.clear();
    if (buf_w_) { ggml_backend_buffer_free(buf_w_); buf_w_ = nullptr; }
    if (ctx_w_) { ggml_free(ctx_w_); ctx_w_ = nullptr; }
    d_out_ = nullptr; d_pad_ = nullptr;
    vram_bytes_ = 0;
    bufs_info_.clear();
    if (be_) { ggml_backend_free(be_); be_ = nullptr; }
}

// ---------------------------------------------------------------------------------------
// The device side
// ---------------------------------------------------------------------------------------

bool GpuStatic::block(const float* x, int n_rows, float* dst) {
    const std::size_t w = std::size_t(n_rows);
    if (n_rows <= 0 || n_rows > cfg_.max_rows) {
        fail_msg_ = "blok logitov shire, chem grafy, kotorye byli postroeny";
        return false;
    }
    // Built on first use, not at init: an unused width would otherwise cost 594 KiB per row
    // of video memory for nothing.
    if (!gf_[w]) {
        ggml_tensor* xv = ggml_view_2d(ctx_g_, t_x_, cfg_.n_embd, n_rows, t_x_->nb[1], 0);
        ggml_tensor* o  = ggml_mul_mat(ctx_g_, d_out_, xv);
        ggml_set_output(o);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx_g_, 8, false);
        ggml_build_forward_expand(gf, o);
        ggml_gallocr_t ga = ggml_gallocr_new(buft_);
        if (!ga || !ggml_gallocr_reserve(ga, gf) || !ggml_gallocr_alloc_graph(ga, gf)) {
            if (ga) ggml_gallocr_free(ga);
            fail_msg_ = "graf golovy ne razmestilsja";
            return false;
        }
        // Every node, asked of the backend that is going to run it. This is the guard against
        // the quantisation trap: IQ4_KS, IQ4_K and every _R* type appear zero times in
        // ggml-vulkan.cpp, so a head in one of them would reach ggml_vk_mul_mat with no
        // pipeline and abort inside the backend instead of being refused here.
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (ggml_backend_supports_op(be_, node)) continue;
            char b[240];
            snprintf(b, sizeof(b),
                     "bekend Vulkan ne podderzhivaet %s nad %s - golova dolzhna byt tipa, "
                     "kotoryj bekend chitaet (Q6_K, Q8_0, IQ4_XS, Q4_0, F16)",
                     ggml_op_name(node->op),
                     node->src[0] ? ggml_type_name(node->src[0]->type) : "?");
            ggml_gallocr_free(ga);
            fail_msg_ = b;
            return false;
        }
        gf_[w] = gf;
        ga_[w] = ga;
        n_out_[w] = o;
    }

    const std::size_t xbytes = std::size_t(cfg_.n_embd) * w * sizeof(float);
    const std::size_t obytes = std::size_t(cfg_.n_vocab) * w * sizeof(float);

    auto t0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(t_x_, x, 0, xbytes);
    auto t1 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(be_, gf_[w]);
    auto t2 = std::chrono::steady_clock::now();
    // Not the mapped pointer. ggml_vk_buffer_read (ggml-vulkan.cpp:4805-4831) deliberately
    // takes the hardware copy path on a non-UMA device even when the buffer is host-visible,
    // because reading back through the BAR aperture is slower than a device-to-host copy - and
    // 594 KiB a row is far past the size where that trade turns. rb_ being pinned is what
    // keeps this a single copy.
    ggml_backend_tensor_get(n_out_[w], rb_, 0, obytes);
    auto t3 = std::chrono::steady_clock::now();
    std::memcpy(dst, rb_, obytes);

    st_.ms_upload   += std::chrono::duration<double, std::milli>(t1 - t0).count();
    st_.ms_device   += std::chrono::duration<double, std::milli>(t2 - t1).count();
    st_.ms_readback += std::chrono::duration<double, std::milli>(t3 - t2).count();
    st_.readback_bytes += obytes;
    ++st_.blocks;
    return true;
}

void GpuStatic::do_head(ggml_tensor* dst, const ggml_tensor* x) {
    const int64_t rows = x->ne[1];
    const std::size_t vocab = std::size_t(cfg_.n_vocab);
    auto t0 = std::chrono::steady_clock::now();
    ++st_.calls;

    if (!fail_msg_.empty()) {
        std::memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
        x->ne[0] != cfg_.n_embd || x->ne[2] != 1 || x->ne[3] != 1) {
        fail_msg_ = "vhod golovy ne nepreryvnyj F32 [n_embd, n_rows]";
        std::memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }
    if (dst->ne[0] != cfg_.n_vocab || dst->ne[1] != rows || !ggml_is_contiguous(dst)) {
        fail_msg_ = "vyhod golovy ne nepreryvnyj [n_vocab, n_rows]";
        std::memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }

    const float* xs = (const float*)x->data;
    float* os = (float*)dst->data;
    try {
        for (int64_t r = 0; r < rows; r += cfg_.max_rows) {
            const int n = int(std::min<int64_t>(cfg_.max_rows, rows - r));
            if (!block(xs + std::size_t(r) * std::size_t(cfg_.n_embd), n,
                       os + std::size_t(r) * vocab)) {
                break;
            }
        }
    } catch (const std::exception& e) {
        fail_msg_ = std::string("iskljuchenie na ustrojstve: ") + e.what();
    } catch (...) {
        fail_msg_ = "neizvestnoe iskljuchenie na ustrojstve";
    }
    if (!fail_msg_.empty()) std::memset(dst->data, 0, ggml_nbytes(dst));
    st_.rows += uint64_t(rows);
    st_.ms_total += ms_since(t0);
}

void GpuStatic::head_op(ggml_tensor* dst, const ggml_tensor* /*proto*/, const ggml_tensor* x,
                        int ith, int /*nth*/, void* ud) {
    if (ith != 0) return;
    ((GpuStatic*)ud)->do_head(dst, x);
}

ggml_tensor* GpuStatic::head(ggml_context* c, ggml_tensor* x) {
    // ggml_map_custom2's destination is a duplicate of its FIRST source, so the shape has to
    // arrive as a tensor. proto is never read; it exists to say [n_vocab, n_rows].
    ggml_tensor* proto = ggml_new_tensor_2d(c, GGML_TYPE_F32, cfg_.n_vocab, x->ne[1]);
    return ggml_map_custom2(c, proto, x, head_op, /*n_tasks=*/1, this);
}

// ---------------------------------------------------------------------------------------
// The one check nothing else can make
// ---------------------------------------------------------------------------------------

bool GpuStatic::verify_head(std::string* err) {
    if (!on() || !d_out_ || !src_out_) { *err = "vykljucheno"; return false; }
    const std::size_t total = ggml_nbytes(d_out_);
    const std::size_t chunk = kUploadChunk;
    std::vector<char> tmp;
    tmp.resize(chunk);
    const char* ref = (const char*)src_out_->data;
    for (std::size_t off = 0; off < total; off += chunk) {
        const std::size_t n = std::min(chunk, total - off);
        ggml_backend_tensor_get(d_out_, tmp.data(), off, n);
        if (std::memcmp(tmp.data(), ref + off, n) != 0) {
            char b[200];
            snprintf(b, sizeof(b),
                     "bajty golovy rashodjatsja v kuske so smeshchenija %.1f MiB",
                     double(off) / 1048576.0);
            *err = b;
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// The layer path: attention, the residual, the FFN norm and the router, on the device
//
// 510.4 MB of attention weights plus 48.0 MB of routers against the head's 243.4 - the bigger
// two thirds of the static half. What makes it a different piece of engineering from the head
// is not the bytes, it is the CUTS: the head is one boundary per token and this is one per
// LAYER, so everything here is arranged around making a crossing cheap rather than around
// making a matmul fast.
//
// WHAT THE HEAD MEASUREMENT SAYS ABOUT THAT, because it corrects the plan this file was
// written against. Per crossing, measured on the real model:
//
//     podjom       0.011 ms      the hidden state, 8 KiB, into a host-visible buffer
//     ustrojstvo   4.833 ms      the matmul itself: 243 MB at about 50 GB/s
//     zabor        0.852 ms      594 KiB per row, and the prefill blocks are eight rows
//     -----------------------------------
//     vsego        6.202 ms
//
// The 6.343 ms head cost was read as "the handoff costs more than reading the tensor". It does
// not: 78% of it is the device kernel, and the fixed part of a crossing is well under a
// millisecond. That is the number this stage lives or dies by, and it is far friendlier than
// the 177 us x 48 = 8.5 ms the plan budgeted - that figure is the CONDVAR rendezvous
// GpuExperts pays to hand work to a worker thread, and this path has no worker and no condvar.
//
// WHAT IS ON THE DEVICE AND WHY IT IS EVERYTHING, NOT ONLY THE MATMULS. Of the ~20 nodes in a
// decode layer, only seven read a weight. The other thirteen - two norms, two ropes, the
// softmax, the permutes and the copies into the cache - move almost nothing. They are here
// anyway, because leaving them on the host would put the boundary between them: about twenty
// crossings a layer instead of one. Cheap in bytes is not cheap in place.
//
// SUBMITS, WHICH ARE THE OTHER HALF OF THE CROSSING'S COST. ggml's Vulkan backend picks submit
// points by mul_mat_bytes >= total_mat_mul_bytes / divisor, doubling the threshold for the
// first three, so a graph with seven matmuls submits five or six times - each one a queue
// submit at a measured 24.1 us. On a per-layer graph that is 120-145 us of pure submit, 6-7 ms
// a token, which is a third of what this stage is trying to save. So the divisor is set to 1
// here: one submit for the whole layer graph. That is safe precisely because the graph is
// small - the reason upstream submits early is to bound how long one submission runs against
// the two-second kernel timeout on Windows, and a 10 MB layer cannot approach it.
//
// THE KV CACHE COMES TOO, AND IT IS NOT OPTIONAL. Attention on the card with the cache on the
// host would put the KQ product on one side and the projections on the other: three crossings
// a layer instead of one, which is more than the whole stage is worth. So the cache is
// allocated here, written here one position per step, and read here - and the prefill's
// cache, which is computed on the host, is uploaded once before the first generated token.


// The layer half of the residency, deferred to its own call for one reason that is not
// arbitrary: the cache length is not known when the head is placed. n_kv_max comes from the
// mode - pad32(n_ctx) in chat, pad32(prompt + gen) in the harness, pad32(prompt + steps + 1)
// under --decode-check - and every one of those is decided after the model has loaded, while
// the head has to exist before the first graph is built because every graph captures it.
//
// So: init() takes the head and sets the submit knobs (which ggml reads once, at device
// init); this takes the attention weights, the routers and the cache, at the length the run
// actually asked for. Between the two calls layers_on() is false and build_step keeps the CPU
// attention block, which is the correct answer to "not placed yet" rather than a half state.
bool GpuStatic::init_layers(const GpuStaticLayer* layers, int n_kv_max, std::string* err) {
    if (!on()) { *err = "ustrojstvo ne podnjato"; return false; }
    if (!cfg_.layers) { *err = "sloi na kartu ne zaprosheny"; return false; }
    if (!layers) { *err = "sloi na kartu bez tenzorov sloev"; return false; }
    if (!lg_.empty()) { *err = "sloi na kartu uzhe razmeshcheny"; return false; }
    if (n_kv_max <= 0 || n_kv_max % 32 != 0) {
        *err = "sloi na kartu: dlina kesha objazana byt polozhitelnoj i kratnoj 32 - eto "
               "dlina svjortki F16-umnozhenija, i pri nekratnoj ono molcha neverno";
        return false;
    }
    cfg_.n_kv_max = n_kv_max;
    // rb_ is the pinned readback staging, allocated for a block of logits. The layer output is
    // far smaller, but "far smaller" is an argument and this is a bounds check.
    if (!rb_ || std::size_t(cfg_.n_vocab) * std::size_t(cfg_.max_rows) <
                std::size_t(2 * cfg_.n_embd + cfg_.n_expert)) {
        *err = "bufer zabora menshe vyhoda sloja";
        return false;
    }
    if (!alloc_layers(layers, err)) return false;
    if (!build_layer_graphs(err)) return false;
    // The same for every layer graph, and for the same three reasons: the pipelines for the
    // attention types are compiled here rather than inside the first generated token, the
    // allocator has already run, and a backend that has no pipeline for one of these types
    // aborts rather than returning an error - so it must abort at init, not at token one.
    if (cfg_.layers) {
        std::vector<float> zx(std::size_t(cfg_.n_embd), 0.0f);
        // A one-position step against a 32-position window: the smallest aim that is legal.
        if (!set_step(0, 32)) {
            *err = "sloi na kartu: probnyj shag ne nacelilsja";
            shutdown();
            return false;
        }
        std::vector<float> zmask(32, 0.0f);
        ggml_backend_tensor_set(t_mask_, zmask.data(), 0, 32 * sizeof(float));
        step_mask_sent_ = true;
        for (int il = 0; il < cfg_.n_layer; ++il) {
            try {
                ggml_backend_tensor_set(t_lx_, zx.data(), 0,
                                        std::size_t(cfg_.n_embd) * sizeof(float));
                ggml_backend_graph_compute(be_, lg_[std::size_t(il)].gf);
            } catch (const std::exception& e) {
                *err = std::string("probnyj prohod sloja ") + std::to_string(il) + ": " +
                       e.what();
                shutdown();
                return false;
            }
        }
        step_past_ = -1;
        step_nkv_  = 0;
        st_ = GpuStaticStats();
    }
    for (const GpuStaticBuffer& b : bufs_info_) {
        GpuExperts::print_placement(b.what.c_str(), b.bytes);
    }
    return true;
}

bool GpuStatic::alloc_layers(const GpuStaticLayer* src, std::string* err) {
    const int nl = cfg_.n_layer;
    // hd i nkv_heads zdes tolko dlja proverok i pechati; sam kesh vydeljaetsja po
    // POSLOJNOJ geometrii nizhe - u gemma4 golovy 16/2 po 512 i 16/8 po 256 v odnoj modeli,
    // i odin razmer na vse sloi vydelil by chetvert nuzhnogo libо vchetvero bolshe.
    const int hd = cfg_.head_dim;
    const int nkv_heads = cfg_.n_head_kv;
    const std::size_t align = ggml_backend_buft_get_alignment(buft_);
    const std::size_t max_buf = ggml_backend_buft_get_max_size(buft_);
    auto padded = [&](std::size_t n) { return (n + align - 1) / align * align; };

    // Per-layer bytes: nine weights plus the two halves of that layer's cache. Weights and
    // cache share a buffer deliberately - keeping them apart would leave the cache at 201 MB
    // for a 2048 context, which is UNDER the 256 MiB BAR window and would therefore be backed
    // by system memory at 3.1 GB/s while every report still called it device-local.
    std::vector<std::size_t> per_layer(std::size_t(nl), 0);
    std::size_t total = 0;
    const std::size_t kv_one = padded(std::size_t(ggml_type_size(GGML_TYPE_F16)) *
                                      std::size_t(hd) * std::size_t(cfg_.n_kv_max) *
                                      std::size_t(nkv_heads));
    for (int il = 0; il < nl; ++il) {
        ggml_tensor* s[13];
        layer_slots(src[std::size_t(il)], s);
        std::size_t b = 0;
        for (int i = 0; i < 13; ++i) {
            // Slots 9..12 are optional by design; slot 3 (wv) is optional because five of Gemma's
            // layers ship no attn_v and take V from the raw K projection instead.
            if (!s[i] && (i >= 9 || i == 3)) continue;
            if (!s[i]) {
                char m[160];
                snprintf(m, sizeof(m), "sloj %d: ne hvataet tenzora vnimanija #%d", il, i);
                *err = m;
                return false;
            }
            if (type_is_interleaved(s[i]->type)) {
                char m[240];
                snprintf(m, sizeof(m),
                         "sloj %d: tenzor vnimanija perepakovan v %s, a formy _R* Vulkan ne "
                         "chitaet ni odnoj operaciej", il, ggml_type_name(s[i]->type));
                *err = m;
                return false;
            }
            if (!s[i]->data) {
                *err = "tenzor vnimanija nedostupen hostu - zalivat neotkuda";
                return false;
            }
            b += padded(ggml_nbytes(s[i]));
        }
        b += 2 * kv_one;
        per_layer[std::size_t(il)] = b;
        total += b;
    }

    // Grouping, and the whole rule is: every buffer must come out STRICTLY larger than the BAR
    // window. find_properties accepts a memory type only if heap.size >= the buffer size, so a
    // buffer that fits the 256 MiB window is put in it, and once the window is committed the
    // driver backs the rest from system memory - a 40x penalty that measures as a working
    // feature. A buffer larger than the window cannot be typed onto that heap at all.
    const std::size_t min_group = kBarHeapCeiling + kOverBarMargin;
    std::size_t n_groups = 1;
    if (max_buf > 0 && total > max_buf) n_groups = (total + max_buf - 1) / max_buf;
    const std::size_t cap_groups = total / min_group;
    if (cap_groups == 0) {
        n_groups = 1;   // one buffer, padded past the ceiling below
    } else if (n_groups > cap_groups) {
        char m[280];
        snprintf(m, sizeof(m),
                 "vnimanie i kesh %.1f MiB ne razlozhit: predel bekenda %.1f MiB trebuet %llu "
                 "buferov, a bolshe %llu iz nih vyjdut <= 256 MiB i sjadut v BAR-kuchu",
                 double(total) / 1048576.0, double(max_buf) / 1048576.0,
                 (unsigned long long)n_groups, (unsigned long long)cap_groups);
        *err = m;
        return false;
    }

    // Layers per group, balanced, so the last group is never the small one - a 100 MiB
    // remainder in the BAR heap is the exact failure this arrangement exists to avoid, and
    // ggml's own max-size split has no reason to know that.
    std::vector<int> first(n_groups + 1, 0);
    for (std::size_t gi = 0; gi <= n_groups; ++gi) {
        first[gi] = int((std::size_t(nl) * gi) / n_groups);
    }
    first[n_groups] = nl;

    ctxs_l_.assign(n_groups, nullptr);
    bufs_l_.assign(n_groups, nullptr);
    lw_.assign(std::size_t(nl), GpuStaticLayer());
    kv_k_.assign(std::size_t(nl), nullptr);
    kv_v_.assign(std::size_t(nl), nullptr);

    for (std::size_t gi = 0; gi < n_groups; ++gi) {
        const int l0 = first[gi], l1 = first[gi + 1];
        std::size_t gbytes = 0;
        for (int il = l0; il < l1; ++il) gbytes += per_layer[std::size_t(il)];
        std::size_t pad = 0;
        if (gbytes <= min_group) pad = min_group - gbytes + align;

        const std::size_t n_t = std::size_t(l1 - l0) * 11 + 4;
        ggml_init_params ip = {ggml_tensor_overhead() * n_t + 4096, nullptr, true};
        ggml_context* cx = ggml_init(ip);
        if (!cx) { *err = "ggml_init dlja vesov sloev ne udalsja"; return false; }
        ctxs_l_[gi] = cx;

        for (int il = l0; il < l1; ++il) {
            ggml_tensor* ss[13];
            ggml_tensor* dd[13] = {nullptr};
            layer_slots(src[std::size_t(il)], ss);
            for (int i = 0; i < 13; ++i) {
                if (!ss[i]) { dd[i] = nullptr; continue; }   // optional slot
                dd[i] = ggml_new_tensor(cx, ss[i]->type, GGML_MAX_DIMS, ss[i]->ne);
                char nm[64];
                snprintf(nm, sizeof(nm), "vk.blk%d.w%d", il, i);
                ggml_set_name(dd[i], nm);
                if (ggml_nbytes(dd[i]) != ggml_nbytes(ss[i])) {
                    *err = "razmer tenzora vnimanija v videopamjati ne sovpal s modelju";
                    return false;
                }
            }
            layer_unslot(lw_[std::size_t(il)], dd);
            // The cache, in exactly the host cache's layout: keys [head_dim, n_kv, heads] and
            // values transposed within each head. Same shape and same type is what lets the
            // prefill's cache be uploaded tensor for tensor rather than repacked on the way.
            // POSLOJNO. U gemma4 golovy 16/2 po 512 i 16/8 po 256 zhivut v odnoj modeli, tak chto
            // odin razmer na vse sloi vydelil by libo chetvert nuzhnogo, libo vchetvero bolshe -
            // i pervoe ne dalo by nikakoj oshibki formy, prosto nevernyj otvet.
            const GpuStaticGeom gk = cfg_.at(il);
            kv_k_[std::size_t(il)] = ggml_new_tensor_3d(cx, GGML_TYPE_F16, gk.head_dim,
                                                        cfg_.n_kv_max, gk.n_head_kv);
            kv_v_[std::size_t(il)] = ggml_new_tensor_3d(cx, GGML_TYPE_F16, cfg_.n_kv_max,
                                                        gk.head_dim, gk.n_head_kv);
        }
        if (pad > 0) {
            ggml_tensor* p = ggml_new_tensor_1d(cx, GGML_TYPE_F32,
                                                int64_t(pad / sizeof(float)));
            ggml_set_name(p, "vk.bar.padding.layers");
            if (!kv_pad_) kv_pad_ = p;
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(cx, buft_);
        if (!buf) {
            char m[200];
            snprintf(m, sizeof(m), "ne vydelilos %.1f MiB pod sloi %d..%d",
                     double(gbytes + pad) / 1048576.0, l0, l1 - 1);
            *err = m;
            return false;
        }
        bufs_l_[gi] = buf;
        const std::size_t sz = ggml_backend_buffer_get_size(buf);
        vram_bytes_ += sz;
        GpuStaticBuffer bi;
        {
            char nm[64];
            snprintf(nm, sizeof(nm), "vnimanie+kesh sloi %d..%d", l0, l1 - 1);
            bi.what = nm;
        }
        bi.bytes = sz;
        bi.padding = pad;
        bi.over_bar = sz > kBarHeapCeiling;
        bufs_info_.push_back(bi);
        if (!bi.over_bar) {
            *err = "bufer sloev <= 256 MiB - on sjadet v BAR-kuchu i chtenija shejdera "
                   "upadut v sorok raz";
            return false;
        }

        // In pieces, so the driver is never asked for a staging buffer the size of a tensor.
        for (int il = l0; il < l1; ++il) {
            ggml_tensor* ss[13];
            ggml_tensor* dd[13];
            layer_slots(src[std::size_t(il)], ss);
            layer_slots(lw_[std::size_t(il)], dd);
            for (int i = 0; i < 13; ++i) {
                if (!ss[i] || !dd[i]) continue;   // optional slot, see layer_slots
                const std::size_t need = ggml_nbytes(ss[i]);
                const char* sp = (const char*)ss[i]->data;
                for (std::size_t off = 0; off < need; off += kUploadChunk) {
                    const std::size_t n = std::min(kUploadChunk, need - off);
                    ggml_backend_tensor_set(dd[i], sp + off, off, n);
                }
            }
        }
    }
    return true;
}

bool GpuStatic::build_layer_graphs(std::string* err) {
    const int nl = cfg_.n_layer;
    const int ne = cfg_.n_embd;
    // GEOMETRIJA TEPER POSLOJNAJA. Ranshe hd/nh/nkvh brались odin raz na vsju model, potomu chto
    // u qwen3moe ona odna. U gemma4 ih dve: dvadcat pjat okonnyh sloev s golovami 16/8 po 256 i
    // pjat polnyh s 16/2 po 512, s raznym osnovaniem povorota i masshtabom softmax. cfg_.at(il)
    // otdajot skaljary, kogda vektor pust, tak chto put qwen3moe ne izmenilsja ni na bajt.
    int sections[GGML_MROPE_SECTIONS] = {0};

    // The per-step inputs. Small, so they land in the BAR window, which is exactly where a
    // host-written input wants to be: ggml_vk_buffer_write takes the plain memcpy branch when
    // the buffer is HOST_VISIBLE and pays a staging hop when it is not.
    {
        ggml_init_params ip = {ggml_tensor_overhead() * 8 + 4096, nullptr, true};
        ctx_lin_ = ggml_init(ip);
        if (!ctx_lin_) { *err = "ggml_init dlja vhodov sloja ne udalsja"; return false; }
        t_lx_   = ggml_new_tensor_2d(ctx_lin_, GGML_TYPE_F32, ne, 1);
        t_pos_  = ggml_new_tensor_1d(ctx_lin_, GGML_TYPE_I32, 1);
        t_mask_ = ggml_new_tensor_2d(ctx_lin_, GGML_TYPE_F32, cfg_.n_kv_max, 1);
        ggml_set_name(t_lx_, "vk.layer.x");
        ggml_set_name(t_pos_, "vk.layer.pos");
        ggml_set_name(t_mask_, "vk.layer.mask");
        buf_lin_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_lin_, buft_);
        if (!buf_lin_) { *err = "vhodnoj bufer sloja ne vydelilsja"; return false; }
    }

    const std::size_t nodes = 48;
    ggml_init_params ip = {
        (ggml_tensor_overhead() * (nodes + 24) + ggml_graph_overhead_custom(nodes, false))
            * std::size_t(nl) + 8192,
        nullptr, true};
    ctx_lg_ = ggml_init(ip);
    if (!ctx_lg_) { *err = "ggml_init dlja grafov sloev ne udalsja"; return false; }
    ggml_context* c = ctx_lg_;
    lg_.assign(std::size_t(nl), LayerGraph());
    sites_.assign(std::size_t(nl), Site());

    for (int il = 0; il < nl; ++il) {
        const GpuStaticGeom gm = cfg_.at(il);
        const int   hd   = gm.head_dim;
        const int   nh   = gm.n_head;
        const int   nkvh = gm.n_head_kv;
        const int   dq   = nh * hd;
        // Ves masshtab, a ne popravka k 1/sqrt(hd): gemma4 zajavljaet edinicu i imenno edinicu
        // imeet v vidu, a raznica mezhdu 1.0 i 1/sqrt(512) portit vyhod, a ne lomaet ego - to est
        // hudshim iz vozmozhnyh sposobov.
        const float kq_scale = gm.attn_scale > 0.0f ? gm.attn_scale : 1.0f / std::sqrt(float(hd));
        LayerGraph& G = lg_[std::size_t(il)];
        GpuStaticLayer& L = lw_[std::size_t(il)];
        sites_[std::size_t(il)].self = this;
        sites_[std::size_t(il)].il = il;

        // Node for node the host's own qwen3moe decode block, in the host's own order, with
        // two families of node removed. Both removals are counted rather than assumed, because
        // the dispatch is what this stage pays: measured 38 graph nodes, 2 submits, and 0.683
        // ms of device time against 0.11 ms of weight bandwidth - so about 20 us per dispatch,
        // and a node is worth roughly a fifth of a millisecond a token across 48 layers.
        //
        // REMOVAL ONE: the four rms_norm + mul pairs become ggml_fused_rms_norm. That is the
        // op the REFERENCE's llm_build_norm actually calls, so this moves the card toward
        // llama_decode rather than away from it. It is not bit-identical to the pair: the
        // fused kernel computes (scale*w[j])*x[j] and the pair computes (scale*x[j])*w[j],
        // which differ in the last bit. Four nodes a layer, 192 a token.
        //
        // REMOVAL TWO: the four ggml_cont after ggml_permute are pure reshapes HERE and only
        // here, because this graph is built for exactly one token. q is [hd, n_head, 1] and
        // permute(0,2,1,3) asks for [hd, 1, n_head]; the swapped dimension has extent one, so
        // element (i,0,j) of the result sits at the same offset j*hd+i as element (i,j,0) of
        // the source. The copy moves bytes to where they already are. The host path cannot do
        // this - it must work for n_tokens > 1, where the permutation is real - which is why
        // the two paths differ here and why this is a place to be explicit rather than clever.
        // Bit-identical, and four of the most byte-moving nodes in the layer.
        ggml_tensor* cur = t_lx_;
        ggml_tensor* x = ggml_fused_rms_norm(c, cur, L.attn_norm, cfg_.rms_eps);

        ggml_tensor* q = ggml_mul_mat(c, L.wq, x);
        ggml_tensor* k = ggml_mul_mat(c, L.wk, x);
        // V's source: its own projection where there is one, and otherwise the RAW output of the
        // K projection - the node BEFORE k_norm and BEFORE the rotation. Five of Gemma's thirty
        // layers (5, 11, 17, 23, 29 - the full-attention ones) ship no attn_v at all, and the
        // reference's `Vcur = Kcur` picks up exactly that node. Taking the normed or roped K
        // instead would be a different model that still runs.
        ggml_tensor* v = L.wv ? ggml_mul_mat(c, L.wv, x) : k;

        q = ggml_reshape_3d(c, q, hd, nh, 1);
        q = ggml_fused_rms_norm(c, q, L.q_norm, cfg_.rms_eps);
        // MEMEX_NO_ROPE=1 drops both rope nodes from this graph. The OUTPUT IS THEN WRONG and
        // that is the point: device time does not depend on the values, so an arm whose
        // arithmetic is deliberately broken still measures the cost of the nodes that were
        // removed (the same licence rule 73 gives, and the same one the node-count sweeps used).
        //
        // Why it is worth an arm of its own. The per-node device timings say ROPE costs 117 us a
        // call, twice a layer, 235 us of the 476 us a crossing takes - on 16 KB of data, i.e.
        // 0.14 GB/s. Barriers were refuted as the explanation (narrowing them changed nothing)
        // and so were submits (one instead of two changed nothing), so either those timings are
        // honest and rope really is the largest single term in attention, or the timestamp is
        // charging rope for the drain of the projection before it. Removing the node cannot be
        // misattributed: whatever the crossing loses is what the node cost.
        static const int no_rope = getenv("MEMEX_NO_ROPE") ? atoi(getenv("MEMEX_NO_ROPE")) : 0;
        static bool no_rope_said = false;
        if (!no_rope_said) { no_rope_said = true;
            fprintf(stderr, "NO_ROPE %d\n", no_rope); fflush(stderr); }
        if (!no_rope) {
            q = ggml_rope_multi(c, q, t_pos_, L.rope_freqs, gm.n_rot, sections, gm.rope_type,
                                cfg_.n_ctx_train, gm.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        }
        k = ggml_reshape_3d(c, k, hd, nkvh, 1);
        k = ggml_fused_rms_norm(c, k, L.k_norm, cfg_.rms_eps);
        if (!no_rope) {
            k = ggml_rope_multi(c, k, t_pos_, L.rope_freqs, gm.n_rot, sections, gm.rope_type,
                                cfg_.n_ctx_train, gm.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        }

        // [hd, n_head_kv, 1] -> [hd, 1, n_head_kv], same bytes at the same offsets.
        ggml_tensor* Kc = ggml_reshape_3d(c, k, hd, 1, nkvh);
        // v is [n_head_kv*hd, 1]; the host writes it as permute(reshape(v,hd,nkvh,1),1,2,0,3),
        // which for one token is [1, hd, n_head_kv] over the same contiguous bytes.
        ggml_tensor* Vc = ggml_reshape_3d(c, v, 1, hd, nkvh);
        ggml_tensor* kcache = kv_k_[std::size_t(il)];
        ggml_tensor* vcache = kv_v_[std::size_t(il)];
        ggml_tensor* kdst = ggml_view_3d(c, kcache, hd, 1, nkvh,
                                         kcache->nb[1], kcache->nb[2], 0);
        ggml_tensor* vdst = ggml_view_3d(c, vcache, 1, hd, nkvh,
                                         vcache->nb[1], vcache->nb[2], 0);
        ggml_tensor* kcpy = ggml_cpy(c, Kc, kdst);
        ggml_tensor* vcpy = ggml_cpy(c, Vc, vdst);

        ggml_tensor* Q = ggml_reshape_3d(c, q, hd, 1, nh);
        ggml_tensor* K = ggml_view_3d(c, kcache, hd, cfg_.n_kv_max, nkvh,
                                      kcache->nb[1], kcache->nb[2], 0);
        ggml_tensor* V = ggml_view_3d(c, vcache, cfg_.n_kv_max, hd, nkvh,
                                      vcache->nb[1], vcache->nb[2], 0);
        ggml_tensor* kq = ggml_mul_mat(c, K, Q);
        ggml_tensor* p = ggml_soft_max_ext(c, kq, t_mask_, kq_scale, 0.0f);
        ggml_tensor* kqv = ggml_mul_mat(c, V, p);
        // [hd, 1, n_head] -> [n_head*hd, 1]: the host's cont_2d(permute(...)) over the same
        // contiguous bytes, for one token.
        kqv = ggml_reshape_2d(c, kqv, dq, 1);

        // DVA BLOKA, a ne odin s raznymi razmerami. U gemma4 mezhdu wo i ostatkom stoit norma,
        // kotoroj u qwen3moe net, a marshrutizator chitaet VYHOD VNIMANIJA cherez svoj sobstvennyj
        // ves ffn_gate_inp_s, a ne vyhod ffn_norm. Formy pri etom mogli by sovpadat - reshaet
        // poriadok uzlov, poetomu vetka po flagu, a ne po geometrii.
        //
        // Uzel v uzel povtorjaet build_gemma4_step, kotoryj proveren protiv etalona (vse zondy
        // 0,0000%). Ljuboe rashozhdenie zdes - eto rashozhdenie s NIM, i iskat ego nado sravneniem
        // dvuh, a ne razmyshleniem.
        ggml_tensor* ffn_inp = nullptr;   // ostatochnyj potok posle vnimanija
        ggml_tensor* xf      = nullptr;   // vhod plotnoj poloviny (ffn_norm)
        ggml_tensor* xm      = nullptr;   // vhod marshrutiziruemoj poloviny (pre_ffw_norm_2)
        ggml_tensor* rl      = nullptr;   // logity marshrutizatora, syrye
        if (cfg_.gemma_block) {
            ggml_tensor* kqv_out = ggml_mul_mat(c, L.wo, kqv);
            ggml_tensor* attn    = ggml_fused_rms_norm(c, kqv_out, L.post_attn_norm, cfg_.rms_eps);
            ffn_inp = ggml_add(c, attn, cur);
            xf      = ggml_fused_rms_norm(c, ffn_inp, L.ffn_norm, cfg_.rms_eps);
            xm      = ggml_fused_rms_norm(c, ffn_inp, L.pre_ffw_norm_2, cfg_.rms_eps);
            // Marshrutizator ot vyhoda vnimanija, cherez svoj ves. Masshtab 1/sqrt(n_embd) uzhe
            // vnutri etogo tenzora - etalon perestavljaet ego data na masshtabirovannuju kopiju
            // pri zagruzke, i primenit ego vtoroj raz znachit podelit logity na 53 eshchjo raz.
            ggml_tensor* tmp = ggml_fused_rms_norm(c, ffn_inp, L.gate_inp_s, cfg_.rms_eps);
            rl = ggml_mul_mat(c, L.router, tmp);
        } else {
            ffn_inp = ggml_add(c, ggml_mul_mat(c, L.wo, kqv), cur);
            xf      = ggml_fused_rms_norm(c, ffn_inp, L.ffn_norm, cfg_.rms_eps);
            rl      = ggml_mul_mat(c, L.router, xf);
        }

        // One readback rather than three. Each ggml_backend_tensor_get on a device buffer the
        // driver did not host-map is a submit and a fence; three of them per layer would treble
        // the very item this stage exists to keep small. The router's logits go back RAW: the
        // host's softmax and top-k stay the same ops in the same order they were before the
        // card existed, so the routing decision does not move.
        // MEMEX_SPLIT_OUT=1 (default) drops the two ggml_concat nodes. They existed only to pack
        // the three things the host needs - the residual stream, its normed copy and the router
        // logits - into ONE tensor, because a separate ggml_backend_tensor_get per output would
        // have been three submit-and-fence round trips. That reason is gone: readbacks are now
        // folded into the graph's own command buffer, so three copies cost three entries in one
        // command buffer and still one round trip.
        //
        // What it saves: two dispatches of nineteen, 14.3 us each, 1.4 ms of a 53.9 ms token.
        // Bit-identical - a concat moves bytes, it does not compute.
        static const int split_out = getenv("MEMEX_SPLIT_OUT")
                                         ? atoi(getenv("MEMEX_SPLIT_OUT")) : 1;
        static bool split_said = false;
        if (!split_said) { split_said = true;
            fprintf(stderr, "SPLIT_OUT %d\n", split_out); fflush(stderr); }
        ggml_tensor* out = nullptr;
        if (split_out || cfg_.gemma_block) {
            // gemma4 vsegda razdelno: u nejo CHETYRE vyhoda, i sklejka ih v odin tenzor potrebovala
            // by trjoh concat vmesto dvuh - to est tri lishnih dispatcha radi togo, chto slozhennoe
            // chtenie i tak dostajot odnim krugom.
            ggml_set_output(ffn_inp);
            ggml_set_output(xf);
            ggml_set_output(rl);
            if (xm) ggml_set_output(xm);
        } else {
            out = ggml_concat(c, ggml_concat(c, ffn_inp, xf, 0), rl, 0);
            ggml_set_output(out);
        }

        ggml_cgraph* gf = ggml_new_graph_custom(c, nodes, false);
        // The two cache writes go in first, exactly as the host graph orders them: the read
        // views below are views of the same buffer rather than descendants of the copies, so
        // nothing but insertion order puts the write before the read.
        // MEMEX_STATIC_TRUNC=N: build the graph only up to stage N. THE OUTPUT IS THEN WRONG,
        // deliberately, and that is what makes this an instrument: device time does not depend
        // on the values (rule 73), so the INCREMENT from stage N to stage N+1 is the true cost
        // of the nodes between them - including their pre-dispatch barrier and any submit they
        // trigger.
        //
        // Why not per-node timestamps. Tried, and refuted (METHODS 80): execution on the card is
        // serial, so a small node's timestamp span includes the drain of the big node before it,
        // and the logger duly reported ROPE as the most expensive op in attention while removing
        // it changed nothing. An increment cannot be misattributed - the barrier and the wait are
        // part of what was removed.
        //
        // Why stages rather than single nodes. Run-to-run noise on this box is about 1 ms per
        // token, i.e. ~20 us per crossing; one node at ~10 us is below it. Stages remove five to
        // ten nodes at a time, which puts the signal above the floor.
        //
        //   1 input norm only          2 + q,k,v projections      3 + q,k norms and ropes
        //   4 + KV cache writes        5 + kq                     6 + softmax
        //   7 + kqv                    8 + o_proj and residual    0 or >=9 full graph
        const int trunc = memex_static_trunc();
        if (trunc >= 1 && trunc <= 8) {
            ggml_tensor* stop = nullptr;
            switch (trunc) {
                case 1: stop = x;   break;
                case 2: stop = v;   break;   // q and k are ancestors of nothing yet; v pulls x
                case 3: stop = k;   break;   // k carries norm+rope; q's pair is a sibling
                case 4: stop = nullptr; break;
                case 5: stop = kq;  break;
                case 6: stop = p;   break;
                case 7: stop = kqv; break;
                case 8: stop = ffn_inp; break;
            }
            if (trunc == 2) { ggml_build_forward_expand(gf, q); ggml_build_forward_expand(gf, k); }
            if (trunc == 3) { ggml_build_forward_expand(gf, q); }
            if (trunc >= 4) { ggml_build_forward_expand(gf, kcpy); ggml_build_forward_expand(gf, vcpy); }
            if (stop) ggml_build_forward_expand(gf, stop);
            if (trunc == 4) { /* nothing beyond the writes */ }
            fprintf(stderr, "STATIC_TRUNC %d uzlov %d\n", trunc, ggml_graph_n_nodes(gf));
            fflush(stderr);
        } else {
            ggml_build_forward_expand(gf, kcpy);
            ggml_build_forward_expand(gf, vcpy);
            if (out) {
                ggml_build_forward_expand(gf, out);
            } else {
                ggml_build_forward_expand(gf, ffn_inp);
                ggml_build_forward_expand(gf, xf);
                ggml_build_forward_expand(gf, rl);
                if (xm) ggml_build_forward_expand(gf, xm);
            }
            static bool said = false;
            if (!said) { said = true; fprintf(stderr, "STATIC_TRUNC 0 uzlov %d\n", ggml_graph_n_nodes(gf)); fflush(stderr); }
        }

        ggml_gallocr_t ga = ggml_gallocr_new(buft_);
        if (!ga || !ggml_gallocr_reserve(ga, gf) || !ggml_gallocr_alloc_graph(ga, gf)) {
            if (ga) ggml_gallocr_free(ga);
            *err = "graf sloja ne razmestilsja";
            return false;
        }
        // Every node, asked of the backend that is going to run it. A type with no pipeline
        // aborts inside the backend rather than returning an error, so it has to be refused
        // here - at init, before any token.
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (ggml_backend_supports_op(be_, node)) continue;
            char b[256];
            snprintf(b, sizeof(b),
                     "bekend Vulkan ne podderzhivaet %s nad %s na sloe %d - vnimanie na kartu "
                     "polozhit nelzja", ggml_op_name(node->op),
                     node->src[0] ? ggml_type_name(node->src[0]->type) : "?", il);
            ggml_gallocr_free(ga);
            *err = b;
            return false;
        }
        G.gf = gf;  G.ga = ga;  G.out = out;
        G.o_res = ffn_inp; G.o_xf = xf; G.o_rl = rl; G.o_xm = xm;
        G.kdst = kdst; G.vdst = vdst; G.kcpy = kcpy; G.vcpy = vcpy;
        G.K = K; G.V = V; G.kq = kq; G.probs = p;
        G.mapped = nullptr;
        G.mapped_probed = false;
        // COUNTED, ONCE, BECAUSE THIS IS THE PRICE OF THE CROSSING - AND COUNTED TWICE OVER,
        // because only one of the two counts is priced at 7.2 us.
        //
        // The device spends 0.467 ms per crossing on a layer whose weights are 10.6 MB; at
        // this card's 131 GB/s the bytes are worth 0.081 ms. The rest is launch, and launch
        // is priced per DISPATCH: measured as a slope, 38 nodes -> 29 moved the device time
        // 0.683 -> 0.618, i.e. 7.2 us, with submits per graph unchanged at 2.00.
        //
        // ggml_graph_n_nodes is NOT the dispatch count. It counts RESHAPE and VIEW nodes, and
        // the Vulkan backend does not dispatch those: ggml_vk_is_empty (ggml-vulkan.cpp:10333)
        // returns true for NONE / RESHAPE / VIEW / PERMUTE / TRANSPOSE and graph_compute skips
        // the node entirely. They cost a loop iteration, not a launch. The nine nodes whose
        // removal produced the 7.2 us slope were all real - four rms_norm+mul pairs folded into
        // fused_rms_norm, four ggml_cont that are pure reshapes for a single token - so the
        // slope is a per-dispatch price and multiplying it by 29 overstates the launch item.
        //
        // Separately, and worth not confusing with this: the eight-node MoE tail in
        // memex-fwd.cpp's build_step is a HOST budget. Different nodes, different price, and
        // cutting it does not move this number at all.
        if (il == 0) {
            int total = ggml_graph_n_nodes(gf);
            int disp = 0;
            for (int i = 0; i < total; ++i) {
                ggml_tensor* n = ggml_graph_node(gf, i);
                const bool empty = ggml_is_empty(n) || n->op == GGML_OP_NONE ||
                                   n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW ||
                                   n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE;
                if (!empty) ++disp;
            }
            printf("  graf sloja: %d uzlov, iz nih %d dispatchej (%d reshape/view bekend "
                   "propuskaet); pri 7.2 us na dispatch eto %.3f ms zapuska iz kazhdogo "
                   "peresechenija\n",
                   total, disp, total - disp, 0.0072 * double(disp));
        }
    }
    return true;
}

bool GpuStatic::set_step(int n_past, int n_kv) {
    if (!cfg_.layers || lg_.empty()) return false;
    // The multiple of 32 is not decoration: n_kv is the reduction length of the V*probs
    // matmul, and this fork's F16 matmul is silently wrong when it is not a multiple of four.
    // The padding positions are read and multiplied by the zero the -inf mask puts into probs,
    // so they cost bytes and never correctness. Everything is validated before anything is
    // patched, so a refusal leaves the graphs reading the whole allocation - slow and right.
    if (n_kv <= 0 || n_kv % 32 != 0 || n_kv > cfg_.n_kv_max) return false;
    if (n_past < 0 || n_past >= n_kv) return false;

    auto restride = [](ggml_tensor* t, int64_t n0) {
        t->ne[0] = n0;
        t->nb[1] = t->nb[0] * std::size_t(t->ne[0]);
        t->nb[2] = t->nb[1] * std::size_t(t->ne[1]);
        t->nb[3] = t->nb[2] * std::size_t(t->ne[2]);
    };
    if (int(t_mask_->ne[0]) != n_kv) restride(t_mask_, n_kv);
    for (LayerGraph& G : lg_) {
        G.K->ne[1] = n_kv;    // strides belong to the cache, so only the extent moves
        G.V->ne[0] = n_kv;
        restride(G.kq, n_kv);
        restride(G.probs, n_kv);
    }
    if (step_past_ != n_past) {
        for (std::size_t il = 0; il < lg_.size(); ++il) {
            LayerGraph& G = lg_[il];
            ggml_tensor* kc = kv_k_[il];
            ggml_tensor* vc = kv_v_[il];
            const std::size_t ko = std::size_t(n_past) * kc->nb[1];
            const std::size_t vo = std::size_t(n_past) * ggml_element_size(vc);
            G.kdst->view_offs = ko;
            G.kdst->data = (char*)kc->data + ko;
            G.kcpy->view_offs = ko;
            G.kcpy->data = G.kdst->data;
            G.vdst->view_offs = vo;
            G.vdst->data = (char*)vc->data + vo;
            G.vcpy->view_offs = vo;
            G.vcpy->data = G.vdst->data;
        }
        const int32_t pos = int32_t(n_past);
        ggml_backend_tensor_set(t_pos_, &pos, 0, sizeof(int32_t));
    }
    step_past_ = n_past;
    step_nkv_  = n_kv;
    step_mask_sent_ = false;
    return true;
}

bool GpuStatic::upload_kv(ggml_tensor* const* k, ggml_tensor* const* v, int n_layer,
                          std::string* err) {
    if (!cfg_.layers || lg_.empty()) { *err = "sloi ne na karte"; return false; }
    if (n_layer != cfg_.n_layer) { *err = "chislo sloev ne sovpadaet"; return false; }
    auto t0 = std::chrono::steady_clock::now();
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor* sk = k[std::size_t(il)];
        ggml_tensor* sv = v[std::size_t(il)];
        ggml_tensor* dk = kv_k_[std::size_t(il)];
        ggml_tensor* dv = kv_v_[std::size_t(il)];
        if (!sk || !sv || !dk || !dv) { *err = "kesh sloja otsutstvuet"; return false; }
        if (ggml_nbytes(sk) != ggml_nbytes(dk) || ggml_nbytes(sv) != ggml_nbytes(dv) ||
            sk->type != dk->type || sv->type != dv->type) {
            char m[240];
            snprintf(m, sizeof(m),
                     "sloj %d: kesh hosta %.2f/%.2f MiB ne sovpal s keshem karty %.2f/%.2f "
                     "MiB - n_kv_max karty i dlina kesha dvizhka objazany sovpadat", il,
                     double(ggml_nbytes(sk)) / 1048576.0, double(ggml_nbytes(sv)) / 1048576.0,
                     double(ggml_nbytes(dk)) / 1048576.0, double(ggml_nbytes(dv)) / 1048576.0);
            *err = m;
            return false;
        }
        for (int kind = 0; kind < 2; ++kind) {
            ggml_tensor* s = kind == 0 ? sk : sv;
            ggml_tensor* d = kind == 0 ? dk : dv;
            const std::size_t total = ggml_nbytes(s);
            const char* sp = (const char*)s->data;
            for (std::size_t off = 0; off < total; off += kUploadChunk) {
                const std::size_t n = std::min(kUploadChunk, total - off);
                ggml_backend_tensor_set(d, sp + off, off, n);
            }
        }
    }
    ++st_.kv_uploads;
    st_.kv_upload_ms += ms_since(t0);
    return true;
}

void GpuStatic::do_layer(int il, ggml_tensor* dst, const ggml_tensor* cur,
                         const ggml_tensor* mask) {
    auto t0 = std::chrono::steady_clock::now();
    ++st_.layer_calls;
    // Tri velichiny u qwen3moe (ostatok, ego norma, logity) i CHETYRE u gemma4: u nejo dve raznye
    // pre-normy nad odnim i tem zhe ostatkom - ffn_norm dlja plotnoj poloviny i pre_ffw_norm_2 dlja
    // marshrutiziruemoj - i vernut obe deshevle, chem zastavit host schitat odnu zanovo.
    const std::size_t out_floats = cfg_.gemma_block
        ? std::size_t(3 * cfg_.n_embd + cfg_.n_expert)
        : std::size_t(2 * cfg_.n_embd + cfg_.n_expert);

    if (!fail_msg_.empty()) { std::memset(dst->data, 0, ggml_nbytes(dst)); return; }
    if (step_nkv_ <= 0) {
        fail_msg_ = "set_step ne byl vyzvan pered grafom - shag ne naceljon";
        std::memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }
    if (cur->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) ||
        cur->ne[0] != cfg_.n_embd || ggml_nelements(cur) != cfg_.n_embd) {
        fail_msg_ = "vhod sloja ne nepreryvnyj F32 [n_embd, 1]";
        std::memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }
    if (std::size_t(ggml_nelements(dst)) != out_floats || !ggml_is_contiguous(dst)) {
        fail_msg_ = "vyhod sloja ne [2*n_embd + n_expert, 1]";
        std::memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }

    LayerGraph& G = lg_[std::size_t(il)];
    try {
        auto ta = std::chrono::steady_clock::now();
        // The mask changes once per token, not once per layer. Uploading it on the first layer
        // and not on the other forty-seven removes 47/48 of that transfer, and at long context
        // the mask is the largest of the three inputs.
        // Otpravljaem masku, kogda ona SMENILAS, a ne odin raz za shag. Ranshe bylo odin raz, na
        // pervom sloe, i dlja qwen3moe eto verno - tam maska odna na vsju model. U gemma4 ih dve:
        // dvadcat pjat sloev so skolzjashchim oknom v 1024 pozicii i pjat polnyh, i host stroit dlja
        // nih raznye maski. Sravnenie ukazatelja dajot to zhe samoe dlja qwen3moe (odna otpravka na
        // shag) i pravilnoe dlja gemma4 (dve). Dispatchi idut posledovatelno, po odnomu na sloj,
        // tak chto otpravit nuzhnuju masku pered dispatchem sloja - dostatochno.
        if (step_mask_src_ != (mask ? mask->data : nullptr)) {
            if (!mask || mask->type != GGML_TYPE_F32 || mask->ne[0] < step_nkv_) {
                fail_msg_ = "maska vnimanija uzhe, chem naceleno pozicij";
                std::memset(dst->data, 0, ggml_nbytes(dst));
                return;
            }
            ggml_backend_tensor_set(t_mask_, mask->data, 0,
                                    std::size_t(step_nkv_) * sizeof(float));
            step_mask_src_ = mask->data;
            step_mask_sent_ = true;
        }
        ggml_backend_tensor_set(t_lx_, cur->data, 0,
                                std::size_t(cfg_.n_embd) * sizeof(float));
        // Fold the readback into the graph's own command buffer, so this layer costs ONE
        // round trip to the device instead of two. Measured on the expert path first: the
        // separate read there was 113 us a layer for 46 KB, i.e. two submit-and-fence round
        // trips where one would do, and folding it in took the expert dispatch from 21.60 to
        // 16.82 ms per token with the tokens still identical.
        //
        // Here the same term is BIGGER: 0.114 ms of fence on each of 49 crossings per token,
        // 5.59 ms of a 58.7 ms token, and the CPU thread is stopped for all of it because the
        // layer path runs inside ggml_map_custom on the thread executing the node.
        //
        // rb_ is pinned, which is the condition for folding; if arming refuses we take the old
        // path and are no worse off. MEMEX_FOLD_READBACK=0 forces the old path.
        static const int fold_rb = getenv("MEMEX_FOLD_READBACK")
                                       ? atoi(getenv("MEMEX_FOLD_READBACK")) : 1;
        static bool fold_said_st = false;
        if (!fold_said_st) { fold_said_st = true;
            fprintf(stderr, "FOLD_READBACK_STATIC %d\n", fold_rb); fflush(stderr); }
        // Under MEMEX_STATIC_TRUNC the output tensor is not in the graph, so gallocr never gave
        // it a buffer and both arming and reading it would abort on "tensor buffer not set".
        // The values are garbage in that mode by design; only the device time is being read.
        const bool truncated = memex_static_trunc() != 0;
        bool folded = false;
        if (fold_rb && !truncated) {
            if (G.out) {
                folded = ggml_backend_vk_arm_readback(be_, G.out, rb_, 0,
                                                      out_floats * sizeof(float));
            } else {
                // Three copies into the same pinned buffer at the offsets the concatenated
                // layout used, so the host side downstream is byte-for-byte what it was.
                const std::size_t ne = std::size_t(cfg_.n_embd);
                // Poriadok tot zhe, chto davala sklejka: ostatok, normy, logity. Host nizhe chitaet
                // rb_ po tem zhe smeshchenijam, poetomu menjat ih nelzja bez pravki obeih storon.
                bool ok = ggml_backend_vk_arm_readback(be_, G.o_res, rb_,      0, ne * sizeof(float));
                ok = ok && ggml_backend_vk_arm_readback(be_, G.o_xf, rb_ + ne, 0, ne * sizeof(float));
                std::size_t off = 2 * ne;
                if (G.o_xm) {
                    ok = ok && ggml_backend_vk_arm_readback(be_, G.o_xm, rb_ + off, 0, ne * sizeof(float));
                    off += ne;
                }
                ok = ok && ggml_backend_vk_arm_readback(be_, G.o_rl, rb_ + off, 0,
                                                        std::size_t(cfg_.n_expert) * sizeof(float));
                folded = ok;
            }
        }
        auto tb = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be_, G.gf);
        auto tc = std::chrono::steady_clock::now();
        // THE READBACK, AND THIS IS A CORRECTED MISTAKE RATHER THAN A CHOICE.
        //
        // The first version took the mapped pointer: the layer's output is 16.9 KB, so the
        // allocator puts it in the host-visible BAR window, and a memcpy through that mapping
        // costs no submit and no fence at all. It measured 0.758 ms per layer - 22 MB/s - and
        // that was HALF the entire per-layer cost, 36 ms of a token.
        //
        // The reason is that a host READ from the BAR aperture is uncached and uncombined:
        // every cache line is a separate PCIe transaction, and the direction matters. Writes
        // through the same mapping are write-combined and genuinely free, which is why the
        // inputs above are written exactly this way. Reads are not.
        //
        // So the readback goes the other way: ggml_vk_buffer_read
        // (ggml-vulkan.cpp:4805-4831) deliberately takes the hardware copy path on a non-UMA
        // device even for host-visible memory, and rb_ being Vulkan-pinned is what makes it a
        // single DMA into our own buffer rather than a hop through the backend's staging. It
        // costs one submit and one fence, about 59 us, and moves the bytes at DMA speed.
        //
        // The head path already had this right and said so in its comment. The layer path
        // reintroduced the shortcut because the output is small - and small is exactly where
        // the fixed 59 us looks expensive, which is what made the trade look different. It is
        // not different: 59 us beats 758 us by an order of magnitude.
        //
        // rb_ is shared with the head. Safe by construction rather than by luck: both run from
        // ggml_map_custom with n_tasks == 1 on the one thread that executes the node, and the
        // head is the last node of the graph while every layer is strictly before it.
        if (!folded && !truncated) {
            if (G.out) {
                ggml_backend_tensor_get(G.out, rb_, 0, out_floats * sizeof(float));
            } else {
                const std::size_t ne = std::size_t(cfg_.n_embd);
                ggml_backend_tensor_get(G.o_res, rb_,      0, ne * sizeof(float));
                ggml_backend_tensor_get(G.o_xf,  rb_ + ne, 0, ne * sizeof(float));
                std::size_t off = 2 * ne;
                if (G.o_xm) {
                    ggml_backend_tensor_get(G.o_xm, rb_ + off, 0, ne * sizeof(float));
                    off += ne;
                }
                ggml_backend_tensor_get(G.o_rl, rb_ + off, 0,
                                        std::size_t(cfg_.n_expert) * sizeof(float));
            }
        }
        if (!truncated) {
            std::memcpy(dst->data, rb_, out_floats * sizeof(float));
        }
        // NO KEEP-WARM POKE HERE, and the reason is worth keeping.
        //
        // It was built on a measured correlation: the round trip cost 310 us when the card had
        // been idle under 100 us before the submit and 486 us when it had been idle over 600, so
        // 75 us a crossing and about 7.6 ms of a 53.9 ms token looked like it was being paid for
        // letting the card fall asleep. A 256-byte fire-and-forget copy was supposed to stop that.
        //
        // Measured: **-0,7%** (16,0363 against 16,1471 tok/s), and the poke's own cost showed up
        // in the fence column, 0,002 -> 0,031 ms a crossing. The correlation was not causal.
        //
        // It was also UNSAFE. The poke took its command buffer from transfer_cmd_pool and never
        // waited for it, while ggml_vk_graph_cleanup resets that same pool at the end of every
        // graph_compute - resetting a command pool whose buffer is still executing is invalid use
        // of Vulkan, and the two timings overlap (poke 310-486 us against a layer's 450-600 us).
        // A 192-of-192 token check would never have caught it; it shows up as a driver fault under
        // load. Found in review, and the right answer to "refuted and unsafe" is to delete rather
        // than keep behind a flag.
        ++st_.layer_readback_fenced;
        auto td = std::chrono::steady_clock::now();
        st_.layer_ms_upload += std::chrono::duration<double, std::milli>(tb - ta).count();
        st_.layer_ms_device += std::chrono::duration<double, std::milli>(tc - tb).count();
        st_.layer_ms_readback += std::chrono::duration<double, std::milli>(td - tc).count();
    } catch (const std::exception& e) {
        fail_msg_ = std::string("iskljuchenie na ustrojstve v sloe: ") + e.what();
    } catch (...) {
        fail_msg_ = "neizvestnoe iskljuchenie na ustrojstve v sloe";
    }
    if (!fail_msg_.empty()) std::memset(dst->data, 0, ggml_nbytes(dst));
    st_.layer_ms_total += ms_since(t0);
}

void GpuStatic::layer_op(ggml_tensor* dst, const ggml_tensor* /*proto*/,
                         const ggml_tensor* cur, const ggml_tensor* mask,
                         int ith, int /*nth*/, void* ud) {
    if (ith != 0) return;
    Site* s = (Site*)ud;
    s->self->do_layer(s->il, dst, cur, mask);
}

ggml_tensor* GpuStatic::layer(ggml_context* c, int il, ggml_tensor* cur, ggml_tensor* mask) {
    // ggml_map_custom3's destination is a duplicate of its FIRST source, so the shape has to
    // arrive as a tensor. proto is never read; it exists to say [2*n_embd + n_expert, 1].
    ggml_tensor* proto = ggml_new_tensor_2d(c, GGML_TYPE_F32,
                                            2 * cfg_.n_embd + cfg_.n_expert, 1);
    return ggml_map_custom3(c, proto, cur, mask, layer_op, /*n_tasks=*/1,
                            &sites_[std::size_t(il)]);
}

// ---------------------------------------------------------------------------------------
// The self-test
//
// No model file, no seventeen gigabytes and no quiet machine: a synthetic head of a realistic
// width, uploaded through the same path the engine uses, then the same matmul computed on the
// CPU backend and on the device. Q6_K on purpose, because that is what output.weight is in
// every file this project measures, and 151936 columns on purpose, because the padding
// decision and the BAR ceiling both turn on the real size.
// ---------------------------------------------------------------------------------------

int gpu_static_selftest(int threads) {
    const int n_embd = 2048, n_vocab = 151936, max_rows = 8;
    const ggml_type qt = GGML_TYPE_Q6_K;

    printf("=== samoproverka staticheskoj golovy na GPU ===\n");
    printf("  n_embd %d, n_vocab %d, tip %s, blok do %d strok, potokov %d\n",
           n_embd, n_vocab, ggml_type_name(qt), max_rows, threads);

    // ---- the synthetic head on the host --------------------------------------------------
    ggml_init_params wip = {ggml_tensor_overhead() * 4 + 4096, nullptr, true};
    ggml_context* cw = ggml_init(wip);
    if (!cw) { printf("  ggml_init ne udalsja\n"); return 1; }
    ggml_tensor* wout = ggml_new_tensor_2d(cw, qt, n_embd, n_vocab);
    ggml_set_name(wout, "output.weight");
    ggml_backend_buffer_t wbuf =
        ggml_backend_alloc_ctx_tensors_from_buft(cw, ggml_backend_cpu_buffer_type());
    if (!wbuf) { printf("  golova na hoste ne vydelilas\n"); ggml_free(cw); return 1; }
    printf("  sinteticheskaja golova na hoste: %.1f MiB\n",
           double(ggml_nbytes(wout)) / 1048576.0);
    {
        uint32_t seed = 0x9e3779b9u;
        // Row by row, so the scratch is one row rather than 1.2 GB of floats.
        // resize rather than a sized constructor: `std::vector<float> row(std::size_t(n_embd))`
        // is a function declaration, not a variable - the most vexing parse, and MSVC reports
        // it as "index requires an array or pointer" three lines later.
        std::vector<float> row;
        row.resize(std::size_t(n_embd));
        char* dp = (char*)wout->data;
        const std::size_t rb = ggml_row_size(qt, n_embd);
        for (int64_t r = 0; r < n_vocab; ++r) {
            for (int i = 0; i < n_embd; ++i) row[std::size_t(i)] = xuni(seed, 0.08f);
            ggml_quantize_chunk(qt, row.data(), dp + std::size_t(r) * rb, 0, 1, n_embd,
                                nullptr, nullptr);
        }
    }

    // ---- the module ----------------------------------------------------------------------
    GpuStaticConfig gc;
    gc.n_embd = n_embd;
    gc.n_vocab = n_vocab;
    gc.max_rows = max_rows;
    gc.reserve = 256u * 1024u * 1024u;
    gc.verify = true;    // the byte compare is the half of this test nothing else can do
    GpuStatic gs;
    std::string gerr;
    if (!gs.init(gc, wout, &gerr)) {
        printf("  inicializacija ne udalas: %s\n", gerr.c_str());
        ggml_backend_buffer_free(wbuf);
        ggml_free(cw);
        return 1;
    }
    for (const GpuStaticBuffer& b : gs.buffers()) {
        printf("  bufer: %-24s %8.2f MiB (nabivki %.2f MiB)  %s\n", b.what.c_str(),
               double(b.bytes) / 1048576.0, double(b.padding) / 1048576.0,
               b.over_bar ? "> 256 MiB - ne BAR" : "<= 256 MiB - MOG SEST V BAR");
    }
    printf("  ustrojstvo: %s, v videopamjati %.1f MiB, tip %s\n",
           gs.device_name().c_str(), double(gs.vram_bytes()) / 1048576.0,
           gs.head_type_name());

    // ---- the same matmul on the CPU and on the device ------------------------------------
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { printf("  bekend CPU ne podnjalsja\n"); return 1; }
    ggml_backend_cpu_set_n_threads(cpu, threads);
    ggml_backend_buffer_type_t cbuft = ggml_backend_cpu_buffer_type();

    int widths[3] = {1, 3, max_rows};
    int bad = 0, checked = 0;
    double worst_rel = 0.0, worst_abs = 0.0;
    double cpu_ms_1 = 0.0, gpu_ms_1 = 0.0;

    for (int wi = 0; wi < 3; ++wi) {
        const int rows = widths[wi];
        // A graph whose input lives in its own buffer, so it can be written after the
        // allocator has run without the allocator having reused its bytes for an
        // intermediate.
        ggml_init_params ipi = {ggml_tensor_overhead() * 4 + 4096, nullptr, true};
        ggml_context* ci = ggml_init(ipi);
        ggml_tensor* xin = ggml_new_tensor_2d(ci, GGML_TYPE_F32, n_embd, rows);
        ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ci, cbuft);
        if (!xbuf) { printf("  vhod ne vydelilsja\n"); return 1; }
        {
            uint32_t seed = 0xdeadbeefu ^ uint32_t(rows);
            std::vector<float> xs(std::size_t(n_embd) * std::size_t(rows));
            for (std::size_t i = 0; i < xs.size(); ++i) xs[i] = xuni(seed, 1.0f);
            ggml_backend_tensor_set(xin, xs.data(), 0, xs.size() * sizeof(float));
        }

        const std::size_t nodes = 16;
        ggml_init_params ipg = {ggml_tensor_overhead() * (nodes + 8) +
                                ggml_graph_overhead_custom(nodes, false) * 2 + 4096,
                                nullptr, true};
        ggml_context* cg = ggml_init(ipg);

        // arm one: the reference, ggml's own CPU mul_mat, which is what the engine does today
        ggml_tensor* ycpu = ggml_mul_mat(cg, wout, xin);
        ggml_set_output(ycpu);
        ggml_cgraph* gcpu = ggml_new_graph_custom(cg, nodes, false);
        ggml_build_forward_expand(gcpu, ycpu);

        // arm two: the module's node, in a graph the CPU backend executes - which is exactly
        // how it runs in the engine, a custom op inside the host graph
        ggml_tensor* ygpu = gs.head(cg, xin);
        ggml_set_output(ygpu);
        ggml_cgraph* ggpu = ggml_new_graph_custom(cg, nodes, false);
        ggml_build_forward_expand(ggpu, ygpu);

        ggml_gallocr_t ga = ggml_gallocr_new(cbuft);
        if (!ga || !ggml_gallocr_reserve(ga, gcpu) || !ggml_gallocr_alloc_graph(ga, gcpu)) {
            printf("  graf CPU ne razmestilsja\n"); return 1;
        }
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(cpu, gcpu);
        const double cms = ms_since(t0);
        std::vector<float> ref(std::size_t(n_vocab) * std::size_t(rows));
        ggml_backend_tensor_get(ycpu, ref.data(), 0, ref.size() * sizeof(float));
        ggml_gallocr_free(ga);

        ggml_gallocr_t gb = ggml_gallocr_new(cbuft);
        if (!gb || !ggml_gallocr_reserve(gb, ggpu) || !ggml_gallocr_alloc_graph(gb, ggpu)) {
            printf("  graf s uzlom golovy ne razmestilsja\n"); return 1;
        }
        // Once to warm the pipeline compile, then the one that is timed.
        ggml_backend_graph_compute(cpu, ggpu);
        auto t1 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(cpu, ggpu);
        const double gms = ms_since(t1);
        std::vector<float> got(std::size_t(n_vocab) * std::size_t(rows));
        ggml_backend_tensor_get(ygpu, got.data(), 0, got.size() * sizeof(float));
        ggml_gallocr_free(gb);

        if (!gs.failure().empty()) {
            printf("  ustrojstvo otkazalo: %s\n", gs.failure().c_str());
            return 1;
        }

        for (int r = 0; r < rows; ++r) {
            double num = 0.0, den = 0.0, wabs = 0.0;
            int argmax_ref = 0, argmax_got = 0;
            for (int i = 0; i < n_vocab; ++i) {
                const std::size_t k = std::size_t(r) * std::size_t(n_vocab) + std::size_t(i);
                const double a = double(got[k]), b = double(ref[k]);
                num += (a - b) * (a - b);
                den += b * b;
                wabs = std::max(wabs, std::fabs(a - b));
                if (ref[k] > ref[std::size_t(r) * std::size_t(n_vocab) +
                                 std::size_t(argmax_ref)]) argmax_ref = i;
                if (got[k] > got[std::size_t(r) * std::size_t(n_vocab) +
                                 std::size_t(argmax_got)]) argmax_got = i;
            }
            const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
            worst_rel = std::max(worst_rel, rel);
            worst_abs = std::max(worst_abs, wabs);
            ++checked;
            // The card is the more accurate of the two - it does not quantise the activation
            // vector the way the CPU kernels do - so this is a plausibility bound, not an
            // equality. A wrong offset, a lost block or a stale buffer all land far outside it.
            if (rel > 5e-3 || argmax_ref != argmax_got) {
                if (!bad) {
                    printf("  RASHOZHDENIE: shirina %d stroka %d rel %.3e argmax %d/%d\n",
                           rows, r, rel, argmax_ref, argmax_got);
                }
                ++bad;
            }
        }
        printf("  shirina %d: CPU %7.2f ms, karta %7.2f ms (x%.2f), hudshij rel %.3e\n",
               rows, cms, gms, cms > 0.0 ? cms / std::max(gms, 1e-9) : 0.0, worst_rel);
        if (rows == 1) { cpu_ms_1 = cms; gpu_ms_1 = gms; }

        ggml_free(cg);
        ggml_backend_buffer_free(xbuf);
        ggml_free(ci);
    }

    const GpuStaticStats& s = gs.stats();
    printf("  uzlov %llu, strok %llu, blokov %llu; podjom %.2f ms, ustrojstvo %.2f ms, "
           "zabor %.2f ms\n",
           (unsigned long long)s.calls, (unsigned long long)s.rows,
           (unsigned long long)s.blocks, s.ms_upload, s.ms_device, s.ms_readback);
    printf("  odna stroka: CPU %.2f ms protiv karty %.2f ms => ekonomija %.2f ms na tokjen\n",
           cpu_ms_1, gpu_ms_1, cpu_ms_1 - gpu_ms_1);
    printf("  rashozhdenij %d iz %d strok, hudshee rel %.3e, hudshee abs %.3e\n",
           bad, checked, worst_rel, worst_abs);

    gs.shutdown();
    ggml_backend_free(cpu);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cw);
    printf(bad ? "=== samoproverka NE PROSHLA ===\n" : "=== samoproverka proshla ===\n");
    return bad ? 1 : 0;
}

}  // namespace memex
