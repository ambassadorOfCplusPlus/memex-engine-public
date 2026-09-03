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
//
// SLOTY 16..25 - qwen3next, gejted delta-set; 26..29 - ego zhe OBSHCHIJ EKSPERT. Vse
// neobjazatelnye: na sloe delta-seti pusty wq/wk/wv/wo/q_norm/k_norm, na sloe vnimanija -
// vsja ssm-polovina. Kakoj slot objazatelen na kakom sloe, reshaet slot_required nizhe, a ne
// nomer: pravilo "menshe devjati - objazatelen" verno tolko dlja qwen3moe i gemma4.
constexpr int kSlots = 30;

void layer_slots(const GpuStaticLayer& L, ggml_tensor** out) {
    out[0] = L.attn_norm; out[1] = L.wq; out[2] = L.wk; out[3] = L.wv; out[4] = L.wo;
    out[5] = L.q_norm;    out[6] = L.k_norm; out[7] = L.ffn_norm; out[8] = L.router;
    out[9]  = L.rope_freqs;      // may be null
    out[10] = L.post_attn_norm;  // gemma4 only
    out[11] = L.gate_inp_s;      // gemma4 only
    out[12] = L.pre_ffw_norm_2;  // gemma4 only
    out[13] = L.ffn_up;          // gemma4 dense half, only with --gpu-static-dense
    out[14] = L.ffn_gate;
    out[15] = L.ffn_down;
    out[16] = L.wqkv;            // qwen3next delta-net
    out[17] = L.wqkv_gate;
    out[18] = L.ssm_conv1d;
    out[19] = L.ssm_dt;
    out[20] = L.ssm_a;
    out[21] = L.ssm_ba;
    out[22] = L.ssm_beta;
    out[23] = L.ssm_alpha;
    out[24] = L.ssm_norm;
    out[25] = L.ssm_out;
    out[26] = L.shexp_gate;      // qwen3next shared expert, every layer
    out[27] = L.gate_shexp;
    out[28] = L.up_shexp;
    out[29] = L.down_shexp;
}

void layer_unslot(GpuStaticLayer& L, ggml_tensor* const* in) {
    L.attn_norm = in[0]; L.wq = in[1]; L.wk = in[2]; L.wv = in[3]; L.wo = in[4];
    L.q_norm    = in[5]; L.k_norm = in[6]; L.ffn_norm = in[7]; L.router = in[8];
    L.rope_freqs     = in[9];
    L.post_attn_norm = in[10];
    L.gate_inp_s     = in[11];
    L.pre_ffw_norm_2 = in[12];
    L.ffn_up         = in[13];
    L.ffn_gate       = in[14];
    L.ffn_down       = in[15];
    L.wqkv           = in[16];
    L.wqkv_gate      = in[17];
    L.ssm_conv1d     = in[18];
    L.ssm_dt         = in[19];
    L.ssm_a          = in[20];
    L.ssm_ba         = in[21];
    L.ssm_beta       = in[22];
    L.ssm_alpha      = in[23];
    L.ssm_norm       = in[24];
    L.ssm_out        = in[25];
    L.shexp_gate     = in[26];
    L.gate_shexp     = in[27];
    L.up_shexp       = in[28];
    L.down_shexp     = in[29];
}

// Objazatelen li slot na etom sloe. Do qwen3next pravilo bylo odno na vsju model - "nomer
// menshe devjati, krome wv" - i ono verno ROVNO dlja qwen3moe i gemma4. U qwen3next dva roda
// sloev s neperesekajushchimisja naborami vesov, i chestnoe pravilo objazano ih razlichat:
// inache libо sloj delta-seti otkazyvaet po otsutstvujushchemu wq, libo propadaet proverka na
// nedostajushchij ssm_out - a nedostajushchij ves eto ne padenie, eto nol v grafe.
bool slot_required(int i, bool qwen3next, bool delta) {
    if (!qwen3next) return i < 9 && i != 3;   // wv neobjazatelen: pjat sloev gemma4 bez nego
    switch (i) {
        case 0: case 7: case 8:                       // attn_norm, ffn_norm, marshrutizator
        case 26: case 27: case 28: case 29:           // obshchij ekspert - na KAZHDOM sloe
            return true;
        case 1: case 2: case 3: case 4: case 5: case 6:
            return !delta;                            // wq, wk, wv, wo, q_norm, k_norm
        case 16: case 17: case 18: case 19: case 20: case 24: case 25:
            return delta;                             // ssm-polovina
        default:
            return false;                             // ssm_ba libo beta/alpha - proverjaetsja
    }                                                 // otdelno, eto "odno iz dvuh"
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
    // Golova neobjazatelna: gemma4 beret sloi, no ne golovu (sm. cfg_.head).
    if (cfg_.head && !out) { *err = "output.weight otsutstvuet"; return false; }
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
    if (cfg_.head && !alloc_head(out, err)) { shutdown(); return false; }
    // Grafy GOLOVY - tolko kogda golova est. Bez etoj proverki oni stroilis by po nulevomu
    // tenzoru: padenie s narusheniem dostupa srazu posle pechati kuch, bez edinogo soobshchenija.
    if (cfg_.head && !build_graphs(err)) { shutdown(); return false; }

    // Pinned host memory for the readback: ggml_vk_buffer_read_2d_async looks the destination
    // up in the pinned registry (ggml_vk_host_get) and copies straight into it when it finds
    // it, and pays one more hop through the backend's shared staging buffer when it does not.
    {
        const std::size_t need_head =
            std::size_t(cfg_.n_vocab) * std::size_t(cfg_.max_rows) * sizeof(float);
        // Pol po sloju: pri --gpu-static-nohead golova ne vydeljaetsja, i n_vocab*max_rows
        // bolshe ne objazan pokryvat vyhod sloja - a chitaem my v tot zhe bufer.
        const std::size_t need_lay =
            std::size_t(3 * cfg_.n_embd + cfg_.n_expert)
            * std::size_t(cfg_.layer_width > 0 ? cfg_.layer_width : 1) * sizeof(float);
        const std::size_t need = std::max(need_head, need_lay) + 4096;
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
    if (cfg_.head) {
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

    if (cfg_.verify && cfg_.head) {
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
    t_lx_ = nullptr; t_pos_ = nullptr; t_mask_ = nullptr; t_mask_sw_ = nullptr;
    t_seq_ = nullptr;
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
    dstate_.clear();
    dstate_elems_ = 0;
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

// Kazhdyj vygruzhennyj tenzor SLOJA obratno i sravnit s modelju, slot za slotom. To zhe, chto
// verify_head delaet dlja golovy, i po toj zhe prichine: eto edinstvennaja proverka, chto v
// videopamjati lezhat imenno te bajty, kotorye u modeli, i nichto drugoe v dvizhke ejo ne vidit.
//
// Napisana potomu, chto arifmetika ukazala imenno sjuda: normirovka na karte dajot rms 11,22 pri
// etalonnyh 1,18, a normirovka po srednekvadratichnomu vozvrashchaet velichinu porjadka svoego
// VESA - znachit ves na karte primerno vdesjatero bolshe nastojashchego, to est v slot popal ne
// tot tenzor. Slot za slotom eto i nazovjot.
bool GpuStatic::verify_layers(const GpuStaticLayer* src, std::string* err) {
    if (!on() || lg_.empty()) { *err = "sloi ne na karte"; return false; }
    static const char* kNames[kSlots] = {"attn_norm","wq","wk","wv","wo","q_norm","k_norm",
                                     "ffn_norm","router","rope_freqs","post_attn_norm",
                                     "gate_inp_s","pre_ffw_norm_2",
                                     "ffn_up","ffn_gate","ffn_down",
                                     "wqkv","wqkv_gate","ssm_conv1d","ssm_dt","ssm_a",
                                     "ssm_ba","ssm_beta","ssm_alpha","ssm_norm","ssm_out",
                                     "shexp_gate","gate_shexp","up_shexp","down_shexp"};
    std::vector<char> tmp;
    int bad = 0, cmp_slots = 0;
    for (int il = 0; il < cfg_.n_layer; ++il) {
        ggml_tensor* ss[kSlots]; ggml_tensor* dd[kSlots];
        layer_slots(src[std::size_t(il)], ss);
        layer_slots(lw_[std::size_t(il)], dd);
        for (int i = 0; i < kSlots; ++i) {
            if (!ss[i] || !dd[i]) {
                if ((ss[i] == nullptr) != (dd[i] == nullptr)) {
                    printf("  sloj %2d slot %-15s: odna storona est, drugoj net" "\n", il, kNames[i]);
                    ++bad;
                }
                continue;
            }
            const std::size_t total = ggml_nbytes(dd[i]);
            if (total != ggml_nbytes(ss[i])) {
                printf("  sloj %2d slot %-15s: razmery ne sovpali, %zu protiv %zu" "\n",
                       il, kNames[i], total, ggml_nbytes(ss[i]));
                ++bad; continue;
            }
            tmp.resize(total);
            ggml_backend_tensor_get(dd[i], tmp.data(), 0, total);
            if (std::memcmp(tmp.data(), ss[i]->data, total) != 0) {
                printf("  sloj %2d slot %-15s: BAJTY RASHODJATSJA (%s protiv %s)" "\n",
                       il, kNames[i], ggml_get_name(dd[i]), ggml_get_name(ss[i]));
                ++bad;
            }
            ++cmp_slots;
        }
    }
    if (bad) { char b[96]; snprintf(b, sizeof(b), "rashozhdenij v slotah: %d", bad); *err = b; return false; }
    // What was COMPARED, not the size of the table walked. Four of the thirteen slots are
    // gemma4-only and are null on both sides for qwen3moe - skipped correctly, and then
    // counted in the pass line, which claimed a number it had not verified (rule 83).
    if (cmp_slots == 0) {
        *err = "ni odin slot ne sverjalsja - eto ne sovpadenie, a otsutstvie proverki";
        return false;
    }
    printf("  bajty vesov sloev v videopamjati sovpadajut s modelju: svereno %d slotov"
           " na %d slojah, propushcheno pustyh s oboih storon %d" "\n",
           cmp_slots, cfg_.n_layer, cfg_.n_layer * kSlots - cmp_slots);
    return true;
}

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
    // KOLCO reshaetsja ZDES: ot nego zavisjat i podschjot bajtov, i vydelenie, i reshenie o
    // golove. Klyuch, potomu chto put trebuet svoej sverki, a odnotokennyj bez kolca uzhe
    // proveren.
    //
    // Razmer kolca: okno plus shirina grafa plus 64 pozicii zapasa. Zapas nuzhen potomu, chto
    // nizhnjaja granica sreza okrugljaetsja vniz do 32, a n_kv - vverh do 32, i bez zapasa
    // ekstent vida ne pokryl by nuzhnyj diapazon.
    {
        const int Wl = cfg_.layer_width > 0 ? cfg_.layer_width : 1;
        // PO UMOLCHANIJU VKLJUCHENO. Na korotkom kontekste otklyuchaetsja SAMO (kolco ne
        // men'she kesha - znachit smysla net), poetomu riska tam net po postroeniju. Na
        // kontekste 5500 izmereno: kesh 1120 pozicij vmesto 5536, karta 2785,1 MiB vmesto
        // 3647,6, GOLOVA OSTAJOTSJA na meste, i 9,466 tok/s protiv 7,088 bez kolca i 0,581
        // do vsej etoj raboty. Token sverjen s chisto processornym etalonom na kontekste 1900,
        // gde perehod cherez granicu kolca dejstvitelno zadejstvovan.
        // MEMEX_SWA_RING=0 vozvrashchaet staroe povedenie.
        swa_ring_ = getenv("MEMEX_SWA_RING")
                        ? atoi(getenv("MEMEX_SWA_RING")) != 0 : true;
        ring_ = 0;
        if (swa_ring_) {
            int mx = 0;
            for (int il = 0; il < cfg_.n_layer; ++il) {
                const int nsw = cfg_.at(il).n_swa;
                if (nsw > mx) mx = nsw;
            }
            if (mx > 0) {
                ring_ = std::min(cfg_.n_kv_max, GGML_PAD(mx + Wl + 64, 32));
                // Kolco imeet smysl tolko esli ono MEN'SHE kesha: inache eto tot zhe kesh s
                // lishnim ostatkom po modulju.
                if (ring_ >= cfg_.n_kv_max) { ring_ = 0; swa_ring_ = false; }
            } else {
                swa_ring_ = false;
            }
        }
        fprintf(stderr, "SWA_RING %d razmer %d\n", swa_ring_ ? 1 : 0, ring_);
        fflush(stderr);
        if (swa_ring_) {
            printf("kolcevoj kesh okonnyh sloev: %d pozicij vmesto %d (okno %d)\n",
                   ring_, cfg_.n_kv_max, ring_ - (cfg_.layer_width > 0 ? cfg_.layer_width : 1) - 64);
        }
    }
    // rb_ is the pinned readback staging, allocated for a block of logits. The layer output is
    // far smaller, but "far smaller" is an argument and this is a bounds check.
    if (!rb_) { *err = "bufer zabora ne vydelen"; return false; }
    {
        const std::size_t need =
            (std::size_t(cfg_.out_embd_slots()) * std::size_t(cfg_.n_embd)
             + std::size_t(cfg_.n_expert)) * std::size_t(cfg_.layer_width > 0
                                                             ? cfg_.layer_width : 1);
        const std::size_t have =
            std::max(std::size_t(cfg_.n_vocab) * std::size_t(cfg_.max_rows),
                     std::size_t(3 * cfg_.n_embd + cfg_.n_expert)
                         * std::size_t(cfg_.layer_width > 0 ? cfg_.layer_width : 1));
        if (have < need) { *err = "bufer zabora menshe vyhoda sloja"; return false; }
    }
    // ODIN TOKEN, i eto ne ogranichenie realizacii, a to, chto umeet karta: shejder
    // ggml_delta_net na Vulkan sveren i podderzhan TOLKO dlja odnogo tokena i odnoj
    // posledovatelnosti (STATE.md, shag 3a), a poriadok l2_norm i perestanovki v sloe
    // delta-seti pri shirine > 1 drugoj. Otkaz vsluh vmesto tihogo neverja.
    if (cfg_.qwen3next_block && cfg_.layer_width != 1) {
        char m[220];
        snprintf(m, sizeof(m),
                 "qwen3next na karte: shirina sloja %d, a DELTA_NET na Vulkan sveren tolko na "
                 "odnom tokene - OTKAZ", cfg_.layer_width);
        *err = m;
        return false;
    }
    // NE DAT DRAJVERU PODLOZHIT SISTEMNUJU PAMJAT. Izmereno na kontekste 5500: kesh s golovoj
    // trebuet 3647,6 MiB iz 3824 svobodnyh, drajver nachinaet podkladyvat sistemnuju pamjat, i
    // peresechenie stoit 35,423 ms vmesto 1,115 - to est 0,581 tok/s vmesto 7,280.
    // Dvenadcatikratnaja poterja, i pri etom KAZHDYJ otchjot po-prezhnemu nazyvaet pamjat
    // device-local. Poetomu reshenie prinimaetsja ZDES, po chislam, a ne otdajotsja drajveru.
    //
    // Golova otdajotsja pervoj: ona samaja krupnaja snimaemaja veshch (748 MiB u gemma4) i ejo
    // cena izvestna - 25,5% skorosti na korotkom kontekste, chto neizmerimo men'she
    // dvenadcatikratnoj poteri na dlinnom.
    //
    // Zapas 224 MiB: grafam sloev nuzhno rabochee mesto, i ego ggml vydeljaet posle etoj
    // proverki. Bez zapasa my by vlezli rovno i upali by v podkachku na pervom zhe grafе.
    if (cfg_.head) {
        std::vector<std::size_t> per;
        std::size_t need = 0;
        std::string e2;
        if (layer_bytes(layers, &per, &need, &e2)) {
            std::size_t vk_free = 0, vk_total = 0;
            ggml_backend_vk_get_device_memory(0, &vk_free, &vk_total);
            const std::size_t margin = 224ull << 20;
            // PECHATAETSJA VSEGDA, a ne tolko pri srabatyvanii. Pervaja popytka etoj proverki
            // molcha ne srabotala, i po logu nelzja bylo skazat, chto imenno ona uvidela -
            // nol, ves objom ili chestnyj ostatok. Kanal objazan govorit, chto on izmeril.
            // SCHITAEM PO SVOEMU UCHJOTU, a ne po "svobodno" ot drajvera: on vozvrashchaet
            // free == total == 3824 MiB, to est ves objom ustrojstva, a ne ostatok.
            // Pervaja versija etoj proverki sravnivala s konstantoj i ne mogla srabotat
            // nikogda; vidno eto stalo tolko potomu, chto pechat postavlena BEZUSLOVNO.
            const std::size_t placed = vram_bytes_;
            const std::size_t budget = vk_total ? vk_total : (3824ull << 20);
            printf("sloi na kartu: razmeshcheno %.1f MiB, nuzhno eshchjo %.1f, bjudzhet "
                   "%.1f, zapas %.0f\n",
                   double(placed) / 1048576.0, double(need) / 1048576.0,
                   double(budget) / 1048576.0, double(margin) / 1048576.0);
            if (placed + need + margin > budget) {
                printf("sloi na kartu: %.1f + %.1f + zapas %.0f prevyshaet bjudzhet %.1f "
                       "MiB - GOLOVA SNIMAETSJA S KARTY\n",
                       double(placed) / 1048576.0, double(need) / 1048576.0,
                       double(margin) / 1048576.0, double(budget) / 1048576.0);
                printf("  prichina: pri perepolnenii drajver podkladyvaet sistemnuju pamjat, i "
                       "ona v sorok raz medlennee, prodolzhaja nazyvatsja device-local. Izmereno "
                       "na kontekste 5500: 0,581 tok/s s golovoj protiv 7,280 bez nejo.\n");
                free_head_only();
            }
        }
    }
    if (!alloc_layers(layers, err)) return false;
    if (!build_layer_graphs(err)) return false;
    // The same for every layer graph, and for the same three reasons: the pipelines for the
    // attention types are compiled here rather than inside the first generated token, the
    // allocator has already run, and a backend that has no pipeline for one of these types
    // aborts rather than returning an error - so it must abort at init, not at token one.
    if (cfg_.layers) {
        const int Wi = cfg_.layer_width > 0 ? cfg_.layer_width : 1;
        std::vector<float> zx(std::size_t(cfg_.n_embd) * std::size_t(Wi), 0.0f);
        // A one-position step against a 32-position window: the smallest aim that is legal.
        if (!set_step(0, 32)) {
            *err = "sloi na kartu: probnyj shag ne nacelilsja";
            shutdown();
            return false;
        }
        std::vector<float> zmask(std::size_t(32) * std::size_t(Wi), 0.0f);
        ggml_backend_tensor_set(t_mask_, zmask.data(), 0, zmask.size() * sizeof(float));
        step_mask_sent_ = true;
        for (int il = 0; il < cfg_.n_layer; ++il) {
            try {
                ggml_backend_tensor_set(t_lx_, zx.data(), 0, zx.size() * sizeof(float));
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
    // CHTO IMENNO LEZHIT NA KARTE, po rodam - a ne odna summa. Summa ne otlichaet "36 sloev
    // delta-seti s sostojaniem" ot "48 keshej, iz kotoryh 36 nikto ne prochtjot", a raznica
    // mezhdu nimi - tri chetverti pamjati ustrojstva.
    if (cfg_.qwen3next_block) {
        int n_delta = delta_layers();
        int n_attn  = cfg_.n_layer - n_delta;
        std::size_t kvb = 0;
        for (int il = 0; il < cfg_.n_layer; ++il) {
            if (kv_k_[std::size_t(il)]) kvb += ggml_nbytes(kv_k_[std::size_t(il)]);
            if (kv_v_[std::size_t(il)]) kvb += ggml_nbytes(kv_v_[std::size_t(il)]);
        }
        printf("qwen3next na karte: %d sloev delta-seti (sostojanie %.1f MiB, %.2f MiB na "
               "sloj) i %d sloev vnimanija (KV %.1f MiB na %d pozicij)\n",
               n_delta, double(std::size_t(n_delta) * dstate_elems_ * sizeof(float)) / 1048576.0,
               double(dstate_elems_ * sizeof(float)) / 1048576.0,
               n_attn, double(kvb) / 1048576.0, cfg_.n_kv_max);
    }
    for (const GpuStaticBuffer& b : bufs_info_) {
        GpuExperts::print_placement(b.what.c_str(), b.bytes);
    }
    return true;
}

void GpuStatic::free_head_only() {
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
    if (buf_w_) { ggml_backend_buffer_free(buf_w_); buf_w_ = nullptr; }
    if (ctx_w_) { ggml_free(ctx_w_); ctx_w_ = nullptr; }
    d_out_ = nullptr; d_pad_ = nullptr;
    // rb_ NE osvobozhdaetsja: ego ispolzuet i put sloev.
    cfg_.head = false;
    // Bufery golovy uhodjat iz otchjota o razmeshchenii - inache summa perestanet sxoditsja s
    // tem, chto realno lezhit na karte.
    for (std::size_t i = 0; i < bufs_info_.size();) {
        if (bufs_info_[i].what.find("golova") != std::string::npos) {
            vram_bytes_ -= std::min(vram_bytes_, bufs_info_[i].bytes);
            bufs_info_.erase(bufs_info_.begin() + long(i));
        } else {
            ++i;
        }
    }
}

bool GpuStatic::layer_bytes(const GpuStaticLayer* src, std::vector<std::size_t>* per,
                            std::size_t* total_out, std::string* err) const {
    const int nl = cfg_.n_layer;
    const std::size_t align = ggml_backend_buft_get_alignment(buft_);
    auto padded = [&](std::size_t n) { return (n + align - 1) / align * align; };
    per->assign(std::size_t(nl), 0);
    std::size_t total = 0;
    // Kesh - POSLOJNO: u gemma4 golovy 16/2 po 512 i 16/8 po 256 v odnoj modeli, i odin razmer
    // na vse sloi dal by libo chetvert nuzhnogo, libo vchetvero bolshe.
    for (int il = 0; il < nl; ++il) {
        const GpuStaticGeom gk = cfg_.at(il);
        const int rr = ring_of(il);
        // U sloja delta-seti KV net vovse - u nego perenosimoe sostojanie postojannogo razmera.
        // Odin razmer na vse sloi vydelil by 48 keshej tam, gde nuzhno 12, i 36 iz nih nikto
        // by ne prochjol: eto ne oshibka formy, eto tri chetverti pamjati karty vpustuju.
        const std::size_t kv_one = gk.delta ? 0
            : padded(std::size_t(ggml_type_size(GGML_TYPE_F16)) *
                     std::size_t(gk.head_dim) *
                     std::size_t(rr > 0 ? rr : cfg_.n_kv_max) *
                     std::size_t(gk.n_head_kv));
        ggml_tensor* s[kSlots];
        layer_slots(src[std::size_t(il)], s);
        std::size_t b = 0;
        for (int i = 0; i < kSlots; ++i) {
            if (!s[i] && !slot_required(i, cfg_.qwen3next_block, gk.delta)) continue;
            if (!s[i]) {
                char m[160];
                snprintf(m, sizeof(m), "sloj %d: ne hvataet tenzora #%d (%s)", il, i,
                         gk.delta ? "delta-set" : "vnimanie");
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
        // Odno iz dvuh: slitoe ssm_ba (qwen3next) libo razdelnye beta/alpha (qwen35moe).
        // repeat_type u ggml_delta_net vybiraetsja imenno po etomu priznaku, i propushchennaja
        // proverka zdes oznachala by nol v grafe vmesto vesa - to est rabotajushchij nevernyj
        // otvet, rovno lovushka 7.1.
        if (gk.delta) {
            const GpuStaticLayer& S = src[std::size_t(il)];
            if (!S.ssm_ba && !(S.ssm_beta && S.ssm_alpha)) {
                char m[160];
                snprintf(m, sizeof(m),
                         "sloj %d: net ni slitogo ssm_ba, ni pary ssm_beta/ssm_alpha", il);
                *err = m;
                return false;
            }
        }
        b += 2 * kv_one;
        if (gk.delta) b += padded(cfg_.delta_state_elems() * sizeof(float));
        (*per)[std::size_t(il)] = b;
        total += b;
    }
    *total_out = total;
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
    std::vector<std::size_t> per_layer;
    std::size_t total = 0;
    if (!layer_bytes(src, &per_layer, &total, err)) return false;

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
    dstate_.assign(std::size_t(nl), nullptr);
    dstate_elems_ = cfg_.qwen3next_block ? cfg_.delta_state_elems() : 0;

    for (std::size_t gi = 0; gi < n_groups; ++gi) {
        const int l0 = first[gi], l1 = first[gi + 1];
        std::size_t gbytes = 0;
        for (int il = l0; il < l1; ++il) gbytes += per_layer[std::size_t(il)];
        std::size_t pad = 0;
        if (gbytes <= min_group) pad = min_group - gbytes + align;

        // 16 tenzorov na sloj, a ne 11. Bylo devjat vesov plus dva kesha; stalo do trinadcati
        // vesov (tri sobstvennyh u gemma4 plus rope_freqs) plus te zhe dva kesha. Staryj raschjot
        // davai rovno 74400 bajt na gruppu v vosemnadcat sloev, a trebovalos 74464 - i otkaz byl
        // ne diagnostiruemyj: ggml_new_object vozvrashchaet nol posredi razmeshchenija, a padaet
        // potom i v drugom meste, s narusheniem dostupa i bez edinogo soobshchenija.
        const std::size_t n_t = std::size_t(l1 - l0) * (kSlots + 3) + 8;
        ggml_init_params ip = {ggml_tensor_overhead() * n_t + 4096, nullptr, true};
        ggml_context* cx = ggml_init(ip);
        if (!cx) { *err = "ggml_init dlja vesov sloev ne udalsja"; return false; }
        ctxs_l_[gi] = cx;

        for (int il = l0; il < l1; ++il) {
            ggml_tensor* ss[kSlots];
            ggml_tensor* dd[kSlots] = {nullptr};
            layer_slots(src[std::size_t(il)], ss);
            for (int i = 0; i < kSlots; ++i) {
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
            if (gk.delta) {
                // Ni K, ni V: u etogo sloja net kesha. Vmesto nih - okno svjortki i matrica
                // sostojanija odnim nepreryvnym tenzorom, tochno takim zhe po forme, kak u
                // hostovogo DeltaState, chtoby posle prefilla ego mozhno bylo podnjat kak est.
                dstate_[std::size_t(il)] = ggml_new_tensor_2d(
                    cx, GGML_TYPE_F32, int64_t(dstate_elems_), 1);
                char nm[64];
                snprintf(nm, sizeof(nm), "vk.blk%d.dstate", il);
                ggml_set_name(dstate_[std::size_t(il)], nm);
                continue;
            }
            const int rr = ring_of(il);
            const int64_t kvlen = rr > 0 ? rr : cfg_.n_kv_max;
            kv_k_[std::size_t(il)] = ggml_new_tensor_3d(cx, GGML_TYPE_F16, gk.head_dim,
                                                        kvlen, gk.n_head_kv);
            kv_v_[std::size_t(il)] = ggml_new_tensor_3d(cx, GGML_TYPE_F16, kvlen,
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
        // ZERO IT, for the same reason the host caches are zeroed: this buffer holds the
        // card's KV cache, the decode reads it over a PADDED extent, and the allocator does
        // not clear. Uninitialised NaN there survives the mask - NaN + (-INFINITY) is NaN -
        // and poisons the softmax row. The host side of this was found first, through
        // gemma4's kq_soft_max_ext-0; the card has its own cache and needed its own memset.
        // One clear of ~1.2 GiB at 131 GB/s is about 9 ms, once.
        ggml_backend_buffer_clear(buf, 0);
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
            ggml_tensor* ss[kSlots];
            ggml_tensor* dd[kSlots];
            layer_slots(src[std::size_t(il)], ss);
            layer_slots(lw_[std::size_t(il)], dd);
            for (int i = 0; i < kSlots; ++i) {
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

// Norma sloja: fused libo para rms_norm+mul, po klyuchu MEMEX_FUSED_NORM.
//
// ZACHEM KLYUCH. ggml_fused_rms_norm - operacija samogo ik_llama, a ne mainline ggml, i ejo
// realizacija na Vulkan mozhet byt verna tolko dlja odnogo tokena. Pri shirine sloja 2 i 4
// kartochnyj put rashoditsja s etalonom (K odinochnyh shagov) na 6..24% NA VSEH strokah, vklyuchaja
// nulevuju - a nulevaja stroka ne zavisit ni ot zapisi chuzhih pozicij v kesh, ni ot maski
// sosednih tokenov. Znachit vinovato chto-to OBSHCHEE dlja shiriny, i fused norma - pervyj
// kandidat: pri W == 1 tenzor [n_embd, 1], pri W > 1 - [n_embd, W], i shejder mozhet schitat
// normu po vsemu tenzoru vmesto stolbca.
//
// Klyuch, a ne zamena: pri MEMEX_FUSED_NORM=1 (po umolchaniju) put ostajotsja tem, chto byl
// verificirovan na shirine 1. Para daet drugie poslednie bity (fused schitaet (scale*w)*x, para
// (scale*x)*w), poetomu vkljuchat ejo bez nuzhdy nelzja.
static ggml_tensor* lnorm(ggml_context* c, ggml_tensor* t, ggml_tensor* w, float eps) {
    static const int fused = getenv("MEMEX_FUSED_NORM")
                                 ? atoi(getenv("MEMEX_FUSED_NORM")) : 1;
    static bool said = false;
    if (!said) { said = true;
        fprintf(stderr, "FUSED_NORM %d\n", fused); fflush(stderr); }
    if (fused) return ggml_fused_rms_norm(c, t, w, eps);
    return ggml_mul(c, ggml_rms_norm(c, t, eps), w);
}

bool GpuStatic::build_layer_graphs(std::string* err) {
    const int nl = cfg_.n_layer;
    const int ne = cfg_.n_embd;
    // Shirina: skolko tokenov graf schitaet za dispatch. Pri W == 1 nizhe rabotaet tot zhe kod,
    // chto rabotal do MTP - vetki po W stojat tolko tam, gde odnotokennyj put polzuetsja tem, chto
    // perestanovka s ekstentom 1 est te zhe bajty po tem zhe smeshchenijam.
    const int W = cfg_.layer_width > 0 ? cfg_.layer_width : 1;
    if (W < 1 || W > 16) { *err = "shirina sloja vne 1..16"; return false; }
    // Suzhenie chtenija u okonnyh sloev. Klyuch, a ne bezuslovno: ekstent vidov K i V
    // zavisit ot etogo resheniya, poetomu ono prinimaetsja odin raz na postroenii, i
    // staryj put ostajotsja dostupnym dlja A/B.
    // PO UMOLCHANIJU VKLJUCHENO. Izmereno: pri kontekste 1900 +3,6%, a pri 5500 - v 6,7 raza
    // (peresechenie 35,423 -> 4,653 ms, 0,581 -> 3,911 tok/s), potomu chto na dlinnom kontekste
    // kesh karty perestajot vlezat v videopamjat i drajver podkladyvaet sistemnuju; suzhennoe
    // chtenie pochti ne popadaet v podlozhennuju chast. Korrektnost sverena s CHISTO
    // processornym etalonom: token sovpadaet, L2 dazhe nizhe (7,05 protiv 7,28).
    // MEMEX_SWA_NARROW=0 vozvrashchaet staroe povedenie dlja A/B.
    swa_narrow_ = getenv("MEMEX_SWA_NARROW")
                      ? atoi(getenv("MEMEX_SWA_NARROW")) != 0 : true;
    fprintf(stderr, "SWA_NARROW %d\n", swa_narrow_ ? 1 : 0); fflush(stderr);
    // Samoidentifikacija (pravilo 68): graf objazan skazat, na kakuju shirinu on sobran. Bez
    // etoj stroki simptom "karta schitaet odin token tam, gde graf podajot chetyre" vygljadel
    // kak oshibka arifmetiki - a chislo uzlov 38 pri shirine 4 i pri shirine 1 sovpadalo.
    fprintf(stderr, "LAYER_WIDTH %d\n", W); fflush(stderr);
    // GEOMETRIJA TEPER POSLOJNAJA. Ranshe hd/nh/nkvh brались odin raz na vsju model, potomu chto
    // u qwen3moe ona odna. U gemma4 ih dve: dvadcat pjat okonnyh sloev s golovami 16/8 po 256 i
    // pjat polnyh s 16/2 po 512, s raznym osnovaniem povorota i masshtabom softmax. cfg_.at(il)
    // otdajot skaljary, kogda vektor pust, tak chto put qwen3moe ne izmenilsja ni na bajt.
    int sections[GGML_MROPE_SECTIONS] = {0};
    // SEKCII MROPE. Do qwen3next zdes stojal nol vo vseh chetyrjoh, i eto rabotalo rovno
    // potomu, chto u qwen3moe rope_type ne MROPE i vetka s sekcijami ne berjotsja vovse.
    // U qwen3next sekcii {11,11,10,0}, ih udvoennaja summa i est n_rot 64 iz head_dim 256, i
    // ggml_rope_multi s nuljami ne prosto dast drugoj otvet - on OTKAZHET utverzhdeniem
    // (ggml.c:21050 trebuet hotja by odnu polozhitelnuju sekciju).
    for (int i = 0; i < 4 && i < GGML_MROPE_SECTIONS; ++i) sections[i] = cfg_.rope_sections[i];
    // CHETYRE POZICII NA TOKEN pri mnogomernom rope: ggml_rope_multi trebuet
    // `a->ne[2] * 4 == b->ne[0]`. S odnoj poziciej na token postroenie grafa padaet na etom
    // utverzhdenii do pervogo tokena - imenno tak eto i bylo najdeno v processornom stroitele.
    const int pos_per_token =
        (cfg_.rope_type & GGML_ROPE_TYPE_MROPE) || (cfg_.rope_type & GGML_ROPE_TYPE_IMROPE)
            ? 4 : 1;

    // The per-step inputs. Small, so they land in the BAR window, which is exactly where a
    // host-written input wants to be: ggml_vk_buffer_write takes the plain memcpy branch when
    // the buffer is HOST_VISIBLE and pays a staging hop when it is not.
    // Potolok srezа okonnyh sloev. Nuzhen ZDES, do sozdanija maski okna: ggml_soft_max_ext
    // trebuet mask->ne[0] == a->ne[0] TOCHNO, i proverjaet eto pri postroenii uzla. Maska na
    // n_kv_max protiv vhoda na 1120 valila process na etom utverzhdenii.
    int swa_cap = 0;
    if (swa_ring_ && ring_ > 0) {
        swa_cap = ring_;          // pri kolce vidy chitajut VSJO kolco
    } else if (swa_narrow_) {
        for (int il = 0; il < nl; ++il) {
            const int nsw = cfg_.at(il).n_swa;
            if (nsw <= 0) continue;
            const int cap = std::min(cfg_.n_kv_max, GGML_PAD(nsw + W + 64, 32));
            if (cap > swa_cap) swa_cap = cap;
        }
    }
    if (swa_cap == 0) swa_cap = cfg_.n_kv_max;
    {
        ggml_init_params ip = {ggml_tensor_overhead() * 12 + 4096, nullptr, true};
        ctx_lin_ = ggml_init(ip);
        if (!ctx_lin_) { *err = "ggml_init dlja vhodov sloja ne udalsja"; return false; }
        t_lx_   = ggml_new_tensor_2d(ctx_lin_, GGML_TYPE_F32, ne, W);
        t_pos_  = ggml_new_tensor_1d(ctx_lin_, GGML_TYPE_I32, W * pos_per_token);
        if (cfg_.qwen3next_block) {
            // Karta posledovatelnostej dlja ggml_ssm_conv. Vsegda nol - odna posledovatelnost, -
            // no lezhat ona objazana v pisuemom bufere: chuzhoj musor v nej adresuet chuzhoj slot.
            t_seq_ = ggml_new_tensor_2d(ctx_lin_, GGML_TYPE_I32, 1, W);
            ggml_set_name(t_seq_, "vk.layer.seq");
        }
        // Strok u maski - vyravnennoe chislo, a ne rovno W. Trebovanie GGML_KQ_MASK_PAD
        // otnositsja k flash-attention, a ne k soft_max_ext (tam tolko mask->ne[1] >= a->ne[1]),
        // no shejder mozhet chitat masku blokami, i lishnie stroki nichego ne stojat: eto
        // n_kv_max*15 float odin raz na vsju zhizn processa. Zapolnjajutsja oni minus
        // beskonechnostju pri progreve, tak chto dazhe prochitannye ne dobavjat vesa.
        const int64_t mrows = W > 1 ? GGML_PAD(W, GGML_KQ_MASK_PAD) : 1;
        t_mask_ = ggml_new_tensor_2d(ctx_lin_, GGML_TYPE_F32, cfg_.n_kv_max, mrows);
        t_mask_sw_ = ggml_new_tensor_2d(ctx_lin_, GGML_TYPE_F32, swa_cap, mrows);
        ggml_set_name(t_mask_sw_, "vk.layer.mask.swa");
        ggml_set_name(t_lx_, "vk.layer.x");
        ggml_set_name(t_pos_, "vk.layer.pos");
        ggml_set_name(t_mask_, "vk.layer.mask");
        buf_lin_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_lin_, buft_);
        if (!buf_lin_) { *err = "vhodnoj bufer sloja ne vydelilsja"; return false; }
        if (t_seq_) {
            std::vector<int32_t> z(std::size_t(W), 0);
            ggml_backend_tensor_set(t_seq_, z.data(), 0, z.size() * sizeof(int32_t));
        }
    }

    // Zapas uzlov na sloj. U bloka gemma4 ih bolshe: norma mezhdu wo i ostatkom, dve pre-normy
    // polovin vmesto odnoj i otdelnaja norma vhoda marshrutizatora - okolo chetyrjoh sverh
    // qwen3moe, plus vidy na chetyre vyhoda vmesto trjoh. Staryj zapas konchalsja na 64 bajtah
    // ('needed 74464, available 74400'), i otkaz pri etom ne diagnostiruemyj: ggml_new_object
    // vozvrashchaet nol posredi postroenija, a padaet potom i v drugom meste.
    // +4 a layer for the dense half (up, gate, gelu*mul, down) when it is on the card.
    // Pri W > 1 chetyre "besplatnyh" reshape prevrashchajutsja v perestanovki, a odna iz nih
    // eshchjo i v cont - to est uzlov stanovitsja bolshe, i zapas objazan eto uchest. Otkaz
    // ggml_new_object ne diagnostiruem (vozvrashchaet nol posredi postroenija, padaet potom i v
    // drugom meste), poetomu luchshe pereplatit pamjatju na deskriptory.
    // U qwen3next sloj delta-seti - okolo tridcati uzlov (svjortka, dve l2-normy, softplus,
    // delta-set, dve zapisi sostojanija, gejtovannaja norma) plus obshchij ekspert (chetyre
    // umnozhenija) plus marshrutizator; sloj vnimanija - stolko zhe, no s dvojnoj wq, gejtom i
    // KV. Zapas s izbytkom: otkaz ggml_new_object NE diagnostiruem - on vozvrashchaet nol
    // posredi postroenija, a padaet potom, v drugom meste i bez soobshchenija.
    const std::size_t nodes = (cfg_.qwen3next_block ? 112
                                                    : (cfg_.gemma_block
                                                           ? (cfg_.dense_ffn ? 96 : 72) : 48))
                            + (W > 1 ? 16 : 0);
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

        // qwen3next - SVOJ blok celikom, a ne vetka vnutri etogo. Dva roda sloev s
        // neperesekajushchimisja naborami vesov, dvojnaja wq s gejtom, chastichnyj rope po
        // sekcijam i perenosimoe sostojanie vmesto kesha: obshchego s blokom nizhe u nego
        // ostajotsja rovno hvost (ffn_norm i marshrutizator), i vpletat ego vetkami znachilo by
        // pravit put, kotoryj uzhe sveren na trjoh modeljah.
        if (cfg_.qwen3next_block) {
            if (!build_next_layer(c, il, nodes, err)) return false;
            continue;
        }

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
        ggml_tensor* x = lnorm(c, cur, L.attn_norm, cfg_.rms_eps);

        ggml_tensor* q = ggml_mul_mat(c, L.wq, x);
        ggml_tensor* k = ggml_mul_mat(c, L.wk, x);
        // V's source: its own projection where there is one, and otherwise the RAW output of the
        // K projection - the node BEFORE k_norm and BEFORE the rotation. Five of Gemma's thirty
        // layers (5, 11, 17, 23, 29 - the full-attention ones) ship no attn_v at all, and the
        // reference's `Vcur = Kcur` picks up exactly that node. Taking the normed or roped K
        // instead would be a different model that still runs.
        ggml_tensor* v = L.wv ? ggml_mul_mat(c, L.wv, x) : k;
        if (cfg_.gemma_block) {
            // V NORMIRUETSJA po golove, BEZ vesa, na kazhdom sloe gemma4 - vkljuchaja te, u
            // kotoryh est svoja proekcija V. Tenzora v_norm v fajle net; norma bezvesovaja i ona
            // vsjo ravno tam. U qwen3moe ejo net vovse, poetomu v grafe karty ejo i ne bylo.
            //
            // Bez nejo karta pishet v kesh nenormirovannoe V, i vnimanie nevernо s PERVOGO shaga
            // dekoda - rovno tot priznak, chto nabljudalsja: L2 16,5% na pozicii 4 i odin
            // sovpavshij tokjen iz shesti.
            v = ggml_rms_norm(c, ggml_reshape_3d(c, v, hd, nkvh, W), cfg_.rms_eps);
        }

        q = ggml_reshape_3d(c, q, hd, nh, W);
        q = lnorm(c, q, L.q_norm, cfg_.rms_eps);
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
            // ROPE_EXT dlja gemma4, ROPE_MULTI dlja qwen3moe. Eto RAZNYE operacii, a ne odna s
            // raznymi parametrami: multi vrashchaet po sekcijam (mnogomernyj rope), ext - obychnyj.
            // Proverennyj build_gemma4_step zovjot imenno ext, i podmena odnogo drugim dajot
            // rabotajushchij graf s drugim otvetom - 1 sovpavshij tokjen iz 6 i L2 do 78%.
            q = cfg_.gemma_block
                ? ggml_rope_ext(c, q, t_pos_, L.rope_freqs, gm.n_rot, gm.rope_type,
                                cfg_.n_ctx_train, gm.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f)
                : ggml_rope_multi(c, q, t_pos_, L.rope_freqs, gm.n_rot, sections, gm.rope_type,
                                  cfg_.n_ctx_train, gm.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        }
        k = ggml_reshape_3d(c, k, hd, nkvh, W);
        k = lnorm(c, k, L.k_norm, cfg_.rms_eps);
        if (!no_rope) {
            k = cfg_.gemma_block
                ? ggml_rope_ext(c, k, t_pos_, L.rope_freqs, gm.n_rot, gm.rope_type,
                                cfg_.n_ctx_train, gm.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f)
                : ggml_rope_multi(c, k, t_pos_, L.rope_freqs, gm.n_rot, sections, gm.rope_type,
                                  cfg_.n_ctx_train, gm.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        }

        // Pri W == 1 eto te zhe bajty po tem zhe smeshchenijam, i togda reshape deshevle
        // perestanovki. Pri W > 1 perestanovka REALNA, i togda nado delat to zhe, chto delaet
        // processornyj stroitel - on i est etalon, protiv kotorogo eta shirina sverjaetsja.
        ggml_tensor* Kc = (W == 1)
            ? ggml_reshape_3d(c, k, hd, 1, nkvh)               // [hd, nkvh, 1] -> [hd, 1, nkvh]
            // ggml_cont OBJAZATELEN, i eto vzjato u etalona bukvalno: processornyj stroitel
            // pishet cont(permute(...)) vo vseh trjoh mestah. Bez nego Vulkan poluchaet
            // NEPLOTNYJ istochnik tam, gde etalon dajot plotnyj - i attn_out rashodilsja na 23%
            // uzhe na nulevom sloe.
            : ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));    // [hd, nkvh, W] -> [hd, W, nkvh]
        // v u gemma4 uzhe trjohmernyj (posle bezvesovoj normy), u qwen3moe - dvumernyj.
        ggml_tensor* v3 = (v->ne[2] == W && v->ne[1] == nkvh)
                              ? v : ggml_reshape_3d(c, v, hd, nkvh, W);
        ggml_tensor* Vc = (W == 1)
            ? ggml_reshape_3d(c, v, 1, hd, nkvh)               // [1, hd, nkvh]
            : ggml_cont(c, ggml_permute(c, v3, 1, 2, 0, 3));   // [W, hd, nkvh]
        ggml_tensor* kcache = kv_k_[std::size_t(il)];
        ggml_tensor* vcache = kv_v_[std::size_t(il)];
        ggml_tensor* kdst = ggml_view_3d(c, kcache, hd, W, nkvh,
                                         kcache->nb[1], kcache->nb[2], 0);
        ggml_tensor* vdst = ggml_view_3d(c, vcache, W, hd, nkvh,
                                         vcache->nb[1], vcache->nb[2], 0);
        ggml_tensor* kcpy = ggml_cpy(c, Kc, kdst);
        ggml_tensor* vcpy = ggml_cpy(c, Vc, vdst);

        // Skolko pozicij etot sloj voobshche mozhet potrebovat. U okonnogo sloja eto okno plus
        // W novyh pozicij, okruglennoe vverh do 32 (dlina svjortki f16-matmula dolzhna byt
        // kratna chetyrjom, a shag kesha my derzhim kratnym 32). U polnogo - ves kesh.
        const bool win = gm.n_swa > 0 && (swa_ring_ || swa_narrow_);
        const int kv_cap = win ? swa_cap : cfg_.n_kv_max;
        G.n_swa = win ? gm.n_swa : 0;
        G.ring  = (swa_ring_ && gm.n_swa > 0) ? ring_ : 0;

        ggml_tensor* Q = (W == 1) ? ggml_reshape_3d(c, q, hd, 1, nh)
                                  : ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3)); // [hd, W, nh]
        ggml_tensor* K = ggml_view_3d(c, kcache, hd, kv_cap, nkvh,
                                      kcache->nb[1], kcache->nb[2], 0);
        ggml_tensor* V = ggml_view_3d(c, vcache, kv_cap, hd, nkvh,
                                      vcache->nb[1], vcache->nb[2], 0);
        ggml_tensor* kq = ggml_mul_mat(c, K, Q);
        // MEMEX_NO_SOFTMAX=1 UBIRAET odin dispatch iz sloja, ostavljaja formu toj zhe.
        //
        // ZACHEM. Vsja moja ocenka "karta platit ~30 mks za dispatch" byla poluchena OSTATKOM:
        // iz 0,934 ms ustrojstva na peresechenie vychel bajty bolshih umnozhenij i zapusk po
        // 7,2 mks, a ostatok podelil na chislo melkih operacij. Ostatok - ne izmerenie. Zdes
        // odin dispatch ubiraetsja nasovsem, i esli cena dispatcha okolo 30 mks, token
        // dolzhen upast na 30 sloev x 30 mks = 0,9 ms. Esli ne upadjot - ocenka neverna, i
        // vsja arifmetika slijanij operacij vmeste s nej.
        //
        // Vyhod pri etom NEVEREN, i eto razreshaet pravilo 73: vremja ustrojstva ne zavisit ot
        // znachenij. Forma sohranena tochno - kq i probs odinakovy po forme, - poetomu nizhe
        // nichego ne menjaetsja i graf ostajotsja polnym grafom, a ne usechjonnym.
        static const int no_sm = getenv("MEMEX_NO_SOFTMAX")
                                     ? atoi(getenv("MEMEX_NO_SOFTMAX")) : 0;
        static bool sm_said = false;
        if (!sm_said) { sm_said = true;
            fprintf(stderr, "NO_SOFTMAX %d\n", no_sm); fflush(stderr); }
        // Maska - svoja u okonnogo sloja: u nejo drugaja dlina stroki.
        ggml_tensor* mtens = G.n_swa > 0 ? t_mask_sw_ : t_mask_;
        ggml_tensor* p = no_sm ? kq : ggml_soft_max_ext(c, kq, mtens, kq_scale, 0.0f);
        ggml_tensor* kqv = ggml_mul_mat(c, V, p);
        // TOCHNOST JADRA VNIMANIJA PRI SHIRINE > 1, i eto ne dogadka, a chtenie dispetchera.
        //
        // ggml_vk_mul_mat vybiraet jadro po dst->ne[1]:
        //     ne[1] == 1  -> mul_mat_vec_p021_f16_f32  (nash kq)   nakoplenie f32
        //     ne[1] == 1  -> mul_mat_vec_nc_f16_f32    (nash kqv)  nakoplenie f32
        //     inache      -> ggml_vk_mul_mat_q_f16     obshchij GEMM
        // Uslovie `dst->ne[1] <= 8 && src1->ne[2]*ne[3] == 1` ne spasaet: u Q tret'ja os - eto
        // golovy, ih 16. A obshchij GEMM po umolchaniju berjot f16acc-konvejer i perevodit src1
        // v f16 (a pri integer_dot_product - voobshche v Q8_1).
        //
        // Otsjuda i bralos rashozhdenie s etalonom na VSEH strokah pri shirine 2 i 4, kotoroe
        // vygljadelo kak oshibka perenosa: chetyre gipotezy o forme grafa byli oprovergnuty
        // bitovo imenno potomu, chto forma byla vernoj - menjalos JADRO.
        //
        // GGML_PREC_F32 perevodit vybor na f32acc (ggml_vk_get_mul_mat_mat_pipeline: prec !=
        // GGML_PREC_DEFAULT -> .f32acc), a dlja f16-vesa s f32-vhodom est otdelnyj
        // pipeline_matmul_f16_f32.f32acc - to est i vhod ostajotsja f32. Pri W == 1 nichego ne
        // stavim: tam rabotajut specializirovannye jadra, i put verificirovan kak est.
        if (W > 1) {
            ggml_mul_mat_set_prec(kq,  GGML_PREC_F32);
            ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
        }
        // Pri W == 1 eto host's cont_2d(permute(...)) po tem zhe nepreryvnym bajtam. Pri W > 1
        // nuzhen nastojashchij cont - rovno tot, chto stoit v processornom stroitele.
        kqv = (W == 1) ? ggml_reshape_2d(c, kqv, dq, 1)
                       : ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), dq, W);

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
            ggml_tensor* attn    = lnorm(c, kqv_out, L.post_attn_norm, cfg_.rms_eps);
            ffn_inp = ggml_add(c, attn, cur);
            xf      = lnorm(c, ffn_inp, L.ffn_norm, cfg_.rms_eps);
            xm      = lnorm(c, ffn_inp, L.pre_ffw_norm_2, cfg_.rms_eps);
            // Marshrutizator ot vyhoda vnimanija, cherez svoj ves. Masshtab 1/sqrt(n_embd) uzhe
            // vnutri etogo tenzora - etalon perestavljaet ego data na masshtabirovannuju kopiju
            // pri zagruzke, i primenit ego vtoroj raz znachit podelit logity na 53 eshchjo raz.
            ggml_tensor* tmp = lnorm(c, ffn_inp, L.gate_inp_s, cfg_.rms_eps);
            rl = ggml_mul_mat(c, L.router, tmp);
            // THE DENSE HALF, on the card, in the slot xf used to occupy.
            //
            // gemma4 runs a dense feed-forward on every token beside the routed one:
            // 3 x 2816 x 2112 per layer at q8_0 is 569 MB read from host RAM every token,
            // 22.9 ms of an 86.6 ms token. It is read UNCONDITIONALLY - no routing decides
            // it - which makes it the most predictable traffic in the model and a better
            // resident than any expert: an expert pays only when it is picked.
            //
            // It goes in xf's output slot rather than beside it, so the layout the host reads
            // back is unchanged: [ffn_inp, <xf or dense>, xm, router logits]. The host simply
            // stops computing the dense half when the card returns it.
            //
            // NOT ggml_fused_up_gate: GGML_OP_FUSED_UP_GATE has no Vulkan implementation
            // (the same fact that keeps the expert path on plain mul_mat_id), so building it
            // here would strand the layer on the CPU with its weights in video memory.
            if (cfg_.dense_ffn && L.ffn_up && L.ffn_gate && L.ffn_down) {
                ggml_tensor* du = ggml_mul_mat(c, L.ffn_up, xf);
                ggml_tensor* dg = ggml_mul_mat(c, L.ffn_gate, xf);
                ggml_tensor* da = ggml_mul(c, du, ggml_gelu(c, dg));
                xf = ggml_mul_mat(c, L.ffn_down, da);
            }
        } else {
            ffn_inp = ggml_add(c, ggml_mul_mat(c, L.wo, kqv), cur);
            xf      = lnorm(c, ffn_inp, L.ffn_norm, cfg_.rms_eps);
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
            if (!said) { said = true;
                fprintf(stderr, "STATIC_TRUNC 0 uzlov %d gemma_block %d geom %d rope %s\n",
                        ggml_graph_n_nodes(gf), cfg_.gemma_block ? 1 : 0,
                        (int)cfg_.geom.size(), cfg_.gemma_block ? "ext" : "multi");
                fflush(stderr); }
        }

        G.out = out;
        G.o_res = ffn_inp; G.o_xf = xf; G.o_rl = rl; G.o_xm = xm;
        G.kdst = kdst; G.vdst = vdst; G.kcpy = kcpy; G.vcpy = vcpy;
        G.K = K; G.V = V; G.kq = kq; G.probs = p;
        if (!finish_layer_graph(il, gf, G, err)) return false;
    }
    return true;
}

// Hvost postroenija odnogo sloja, obshchij dlja oboih blokov: razmestit graf i sprosit bekend
// o KAZHDOM uzle. Vopros ne formalnyj - tip bez konvejera na Vulkan ne vozvrashchaet oshibku,
// on padaet vnutri bekenda, tak chto otkaz objazan prozvuchat zdes, do pervogo tokena.
bool GpuStatic::finish_layer_graph(int il, ggml_cgraph* gf, LayerGraph& G, std::string* err) {
    ggml_gallocr_t ga = ggml_gallocr_new(buft_);
    if (!ga || !ggml_gallocr_reserve(ga, gf) || !ggml_gallocr_alloc_graph(ga, gf)) {
        if (ga) ggml_gallocr_free(ga);
        *err = "graf sloja ne razmestilsja";
        return false;
    }
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor* node = ggml_graph_node(gf, i);
        if (ggml_backend_supports_op(be_, node)) continue;
        char b[320];
        snprintf(b, sizeof(b),
                 "bekend Vulkan ne podderzhivaet %s nad %s na sloe %d (uzel %d iz %d) - sloj "
                 "na kartu polozhit nelzja, i tihogo otkata na CPU vnutri sloja net",
                 ggml_op_name(node->op),
                 node->src[0] ? ggml_type_name(node->src[0]->type) : "?", il, i,
                 ggml_graph_n_nodes(gf));
        ggml_gallocr_free(ga);
        *err = b;
        return false;
    }
    G.gf = gf;
    G.ga = ga;
    G.mapped = nullptr;
    G.mapped_probed = false;
    // COUNTED, ONCE PER ROD SLOJA, BECAUSE THIS IS THE PRICE OF THE CROSSING.
    //
    // ggml_graph_n_nodes is NOT the dispatch count. It counts RESHAPE and VIEW nodes, and the
    // Vulkan backend does not dispatch those: ggml_vk_is_empty returns true for
    // NONE / RESHAPE / VIEW / PERMUTE / TRANSPOSE and graph_compute skips the node entirely.
    // They cost a loop iteration, not a launch. Pri 7,2 mks na dispatch (izmereno naklonom:
    // 38 uzlov -> 29 sdvinulo vremja ustrojstva 0,683 -> 0,618) eto i est cena zapuska.
    //
    // Dva roda sloev u qwen3next pechatajutsja OTDELNO: u delta-seti i u vnimanija raznye uzly
    // i raznoe ih chislo, i odna stroka na oba nazvala by chislo, kotorogo net ni u odnogo.
    static bool said_attn = false, said_delta = false;
    bool& said = G.delta ? said_delta : said_attn;
    if (!said) {
        said = true;
        int total = ggml_graph_n_nodes(gf);
        int disp = 0;
        for (int i = 0; i < total; ++i) {
            ggml_tensor* n = ggml_graph_node(gf, i);
            const bool empty = ggml_is_empty(n) || n->op == GGML_OP_NONE ||
                               n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW ||
                               n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE;
            if (!empty) ++disp;
        }
        printf("  graf sloja (%s, sloj %d): %d uzlov, iz nih %d dispatchej (%d reshape/view "
               "bekend propuskaet); pri 7.2 us na dispatch eto %.3f ms zapuska iz kazhdogo "
               "peresechenija\n",
               G.delta ? "delta-set" : "vnimanie", il, total, disp, total - disp,
               0.0072 * double(disp));
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// qwen3next: sloj gejted delta-seti i sloj vnimanija, oba celikom na karte
// ---------------------------------------------------------------------------------------
//
// POCHEMU ETO OTDELNAJA FUNKCIJA, a ne vetka vnutri bloka vyshe. U etoj arhitektury DVA roda
// sloev s neperesekajushchimisja naborami vesov: 36 sloev delta-seti (wqkv, wqkv_gate, ssm_*)
// i 12 sloev vnimanija (wq dvojnoj shiriny s gejtom, wk, wv, wo, q_norm, k_norm). Obshchego u
// nih rovno tri veshchi - attn_norm na vhode, ffn_norm s marshrutizatorom na vyhode i OBSHCHIJ
// EKSPERT, kotoryj schitaetsja na kazhdom sloe i kazhdom tokene.
//
// ARIFMETICHESKIJ ETALON - qwen35_delta_layer i vetka vnimanija build_qwen35_step v
// memex-fwd.cpp. Zdes povtorjaetsja UZEL V UZEL: ljuboe rashozhdenie - eto rashozhdenie s NIMI,
// i iskat ego nado sravneniem dvuh, a ne razmyshleniem. Dva mesta, gde odna i ta zhe forma
// dajot dva raznyh otveta, uzhe stoili etomu proektu po dnju kazhdoe:
//
//   - repeat_type u ggml_delta_net: NOL dlja slitogo ssm_ba (qwen3next) i EDINICA dlja
//     razdelnyh beta/alpha. Model rabotaet i pri nevernom, i dajot pravdopodobnyj tekst.
//   - polovina wq: 256 zaprosa i 256 gejta CHEREDUJUTSJA PO GOLOVE, a ne lezhat dvumja
//     nepreryvnymi polovinami. Vtoroe chtenie beryot zaprosy pervyh vosmi golov i ih gejty i
//     nazyvaet ih shestnadcatju zaprosami.
//
// CHEGO ZDES NET, i eto namerenno: obshchij ekspert vozvrashchaetsja hostu UZHE umnozhennym na
// svoj sigmoid, no NE slozhennym s ostatkom. Slozhenie idjot na hoste v tom zhe porjadke, chto v
// qwen35_ffn (ostatok k marshrutiziruemoj polovine, potom obshchij ekspert): eto dva slozhenija
// po 2048 chisel, i vtoroj dispatch na kartu radi nih stoil by 48 krugovyh obmenov na token.
bool GpuStatic::build_next_layer(ggml_context* c, int il, std::size_t nodes, std::string* err) {
    const GpuStaticGeom gm = cfg_.at(il);
    const int W  = cfg_.layer_width > 0 ? cfg_.layer_width : 1;
    const float eps = cfg_.rms_eps;
    LayerGraph& G = lg_[std::size_t(il)];
    GpuStaticLayer& L = lw_[std::size_t(il)];
    G.delta = gm.delta;
    G.n_swa = 0;
    G.ring  = 0;

    ggml_cgraph* gf = ggml_new_graph_custom(c, nodes, false);
    ggml_tensor* cur = t_lx_;
    ggml_tensor* attn_out = nullptr;
    ggml_tensor* kcpy = nullptr;
    ggml_tensor* vcpy = nullptr;

    if (gm.delta) {
        const int Sk = cfg_.ssm_d_state;                  // 128, shirina golovy k i q
        const int Hk = cfg_.ssm_n_group;                  // 16 K-golov
        const int Hv = cfg_.ssm_dt_rank;                  // 32 V-golovy
        const int Sv = cfg_.ssm_d_inner / Hv;             // 128, shirina golovy v
        const int key_dim  = Sk * Hk;                     // 2048
        const int val_dim  = Sv * Hv;                     // 4096
        const int conv_dim = key_dim * 2 + val_dim;       // 8192
        const int dconv    = cfg_.ssm_d_conv;             // 4
        const int conv_state_dim = (dconv - 1) * conv_dim;
        const int ssm_state_dim  = Sv * Sv * Hv;
        const std::size_t esz = sizeof(float);
        ggml_tensor* st = dstate_[std::size_t(il)];
        if (!st || !t_seq_) { *err = "sloj delta-seti bez sostojanija na karte"; return false; }

        ggml_tensor* x = lnorm(c, cur, L.attn_norm, eps);
        ggml_tensor* qkv = ggml_mul_mat(c, L.wqkv, x);        // [conv_dim, W]
        ggml_tensor* z   = ggml_mul_mat(c, L.wqkv_gate, x);   // [val_dim, W]

        ggml_tensor* beta = nullptr;
        ggml_tensor* alpha = nullptr;
        if (L.ssm_ba) {
            // SLITYJ beta|alpha. Razdeljaetsja NE ves, a REZULTAT umnozhenija, i ne dvumja
            // polovinami: znachenija peremeshany po gruppam k-golov. Uzel v uzel po etalonu.
            ggml_tensor* mixed = ggml_mul_mat(c, L.ssm_ba, x);
            const int ba_dim = 2 * Hv / Hk;
            const int half   = Hv / Hk;
            ggml_tensor* r = ggml_reshape_4d(c, mixed, ba_dim, Hk, W, 1);
            ggml_tensor* bv = ggml_view_4d(c, r, half, Hk, W, 1,
                                           r->nb[1], r->nb[2], r->nb[3], 0);
            ggml_tensor* av = ggml_view_4d(c, r, half, Hk, W, 1,
                                           r->nb[1], r->nb[2], r->nb[3],
                                           std::size_t(half) * ggml_element_size(r));
            beta  = ggml_cont_4d(c, bv, Hv, 1, W, 1);
            alpha = ggml_cont_3d(c, av, Hv, W, 1);
        } else {
            beta  = ggml_reshape_4d(c, ggml_mul_mat(c, L.ssm_beta, x), Hv, 1, W, 1);
            alpha = ggml_reshape_3d(c, ggml_mul_mat(c, L.ssm_alpha, x), Hv, W, 1);
        }
        ggml_tensor* gate = ggml_mul(c, ggml_softplus(c, ggml_add(c, alpha, L.ssm_dt)), L.ssm_a);

        ggml_tensor* conv_state = ggml_reshape_3d(c,
            ggml_view_2d(c, st, conv_state_dim, 1, st->nb[1], 0), dconv - 1, conv_dim, 1);
        ggml_tensor* state = ggml_reshape_4d(c,
            ggml_view_2d(c, st, ssm_state_dim, 1, st->nb[1],
                         std::size_t(conv_state_dim) * esz), Sv, Sv, Hv, 1);

        ggml_tensor* conv_raw = ggml_ssm_conv(c, conv_state, qkv, L.ssm_conv1d, t_seq_, nullptr);
        ggml_tensor* y = ggml_silu(c, ggml_view_2d(c, conv_raw, conv_dim, W,
                                                   std::size_t(conv_dim) * esz, 0));
        const std::size_t rowq = std::size_t(conv_dim) * esz;
        ggml_tensor* q = ggml_view_4d(c, y, Sk, Hk, W, 1,
                                      std::size_t(Sk) * esz, rowq, rowq * std::size_t(W), 0);
        ggml_tensor* k = ggml_view_4d(c, y, Sk, Hk, W, 1,
                                      std::size_t(Sk) * esz, rowq, rowq * std::size_t(W),
                                      std::size_t(key_dim) * esz);
        ggml_tensor* v = ggml_view_4d(c, y, Sv, Hv, W, 1,
                                      std::size_t(Sv) * esz, rowq, rowq * std::size_t(W),
                                      std::size_t(2 * key_dim) * esz);
        // Odin token: perestavlennyj vid i tak nepreryven (u kazhdoj perestavljaemoj osi
        // ekstent edinica), tak chto normirovka pervoj i perestanovka vtoroj - zakonny i na
        // odnu kopiju deshevle. Pri W > 1 poriadok obratnyj, i etot put otkazan vyshe.
        q = ggml_permute(c, ggml_l2_norm(c, q, eps), 0, 2, 1, 3);
        k = ggml_permute(c, ggml_l2_norm(c, k, eps), 0, 2, 1, 3);
        ggml_tensor* vp = ggml_permute(c, v, 0, 2, 1, 3);
        ggml_tensor* gp = ggml_permute(c, gate, 2, 0, 3, 1);
        ggml_tensor* bp = ggml_permute(c, beta, 2, 0, 1, 3);
        ggml_tensor* state_flat = ggml_reshape_4d(c, state, Sv, Sv * Hv, 1, 1);

        ggml_tensor* res = ggml_delta_net(c, q, k, vp, gp, bp, state_flat, nullptr);
        // repeat_type: KAK beta i gate razmnozhajutsja po gruppam golov. NOL dlja slitoj
        // raskladki (qwen3next), EDINICA dlja razdelnoj. Lovushka 7.1: pri nevernom znachenii
        // model rabotaet, ne padaet i dajot pravdopodobnyj tekst, rashodjas s etalonom na
        // 14,81% uzhe na pervom sloe.
        res->op_params[0] = L.ssm_ba ? 0 : 1;

        const std::size_t out_elems = std::size_t(Sv) * std::size_t(Hv) * std::size_t(W);
        ggml_tensor* out_dn = ggml_view_4d(c, res, Sv, Hv, W, 1, std::size_t(Sv) * esz,
                                           std::size_t(Sv * Hv) * esz, out_elems * esz, 0);
        ggml_tensor* new_state = ggml_reshape_4d(c,
            ggml_view_1d(c, res, ssm_state_dim, out_elems * esz), Sv, Sv, Hv, 1);
        ggml_tensor* new_conv = ggml_cont(c,
            ggml_view_2d(c, conv_raw, dconv - 1, conv_dim, std::size_t(dconv) * esz,
                         (1 + std::size_t(conv_dim) * std::size_t(W)) * esz));
        // Obe poloviny sostojanija - obratno v tot zhe bufer na karte. Poriadok tot zhe, chto u
        // etalona: snachala rekurrentnaja kopija, potom okno svjortki.
        ggml_tensor* ssm_cpy = ggml_cpy(c, ggml_reshape_2d(c, new_state, ssm_state_dim, 1),
            ggml_view_2d(c, st, ssm_state_dim, 1, st->nb[1],
                         std::size_t(conv_state_dim) * esz));
        ggml_build_forward_expand(gf, ssm_cpy);
        ggml_tensor* conv_cpy = ggml_cpy(c, ggml_reshape_2d(c, new_conv, conv_state_dim, 1),
            ggml_view_2d(c, st, conv_state_dim, 1, st->nb[1], 0));
        ggml_build_forward_expand(gf, conv_cpy);

        ggml_tensor* o2 = ggml_reshape_2d(c, out_dn, Sv, int64_t(Hv) * W);
        ggml_tensor* z2 = ggml_reshape_2d(c, z, Sv, int64_t(Hv) * W);
        ggml_tensor* on = ggml_fused_mul_unary(c, z2, lnorm(c, o2, L.ssm_norm, eps),
                                               GGML_UNARY_OP_SILU);
        ggml_tensor* proj = ggml_mul_mat(c, L.ssm_out,
                                         ggml_reshape_2d(c, on, val_dim, W));
        attn_out = ggml_add(c, proj, cur);
    } else {
        const int hd   = gm.head_dim;
        const int nh   = gm.n_head;
        const int nkvh = gm.n_head_kv;
        const int dq   = nh * hd;
        const float kq_scale = gm.attn_scale > 0.0f ? gm.attn_scale
                                                    : 1.0f / std::sqrt(float(hd));
        int sections[GGML_MROPE_SECTIONS] = {0};
        for (int i = 0; i < 4 && i < GGML_MROPE_SECTIONS; ++i) {
            sections[i] = cfg_.rope_sections[i];
        }

        ggml_tensor* x = lnorm(c, cur, L.attn_norm, eps);
        // wq DVOJNOJ SHIRINY, i poloviny CHEREDUJUTSJA PO GOLOVE: golova n vladeet strokami
        // [2n*hd, 2n*hd+hd) kak zaprosom i [2n*hd+hd, 2n*hd+2hd) kak gejtom vyhoda. Dve
        // nepreryvnye poloviny - ochevidnoe chtenie i nevernoe: ono vzjalo by zaprosy i gejty
        // pervyh vosmi golov i nazvalo ih shestnadcatju zaprosami.
        ggml_tensor* qaux = ggml_mul_mat(c, L.wq, x);
        const std::size_t row = std::size_t(hd) * sizeof(float);
        ggml_tensor* q = ggml_cont(c, ggml_view_3d(c, qaux, hd, nh, W,
                                                   2 * row, qaux->nb[1], 0));
        ggml_tensor* agate = ggml_cont_2d(c,
            ggml_view_3d(c, qaux, hd, nh, W, 2 * row, qaux->nb[1], row), dq, W);
        ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, x), hd, nkvh, W);
        ggml_tensor* v = ggml_mul_mat(c, L.wv, x);

        q = lnorm(c, q, L.q_norm, eps);
        k = lnorm(c, k, L.k_norm, eps);
        // ggml_rope_multi s NASTOJASHCHIMI sekcijami, a ne s tekstovoj formoj iz nulej: eto
        // interleaved-mrope, i shiriny sekcij reshajut, kakoe izmerenie poluchit kakoj iz trjoh
        // potokov uglov. n_rot 64 iz head_dim 256 - povorachivaetsja chetvert golovy.
        q = ggml_rope_multi(c, q, t_pos_, L.rope_freqs, gm.n_rot, sections, gm.rope_type,
                            cfg_.n_ctx_train, gm.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_multi(c, k, t_pos_, L.rope_freqs, gm.n_rot, sections, gm.rope_type,
                            cfg_.n_ctx_train, gm.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        ggml_tensor* Kc = (W == 1)
            ? ggml_reshape_3d(c, k, hd, 1, nkvh)
            : ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
        ggml_tensor* Vc = (W == 1)
            ? ggml_reshape_3d(c, v, 1, hd, nkvh)
            : ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, hd, nkvh, W), 1, 2, 0, 3));
        ggml_tensor* kcache = kv_k_[std::size_t(il)];
        ggml_tensor* vcache = kv_v_[std::size_t(il)];
        if (!kcache || !vcache) { *err = "sloj vnimanija bez kesha na karte"; return false; }
        ggml_tensor* kdst = ggml_view_3d(c, kcache, hd, W, nkvh,
                                         kcache->nb[1], kcache->nb[2], 0);
        ggml_tensor* vdst = ggml_view_3d(c, vcache, W, hd, nkvh,
                                         vcache->nb[1], vcache->nb[2], 0);
        kcpy = ggml_cpy(c, Kc, kdst);
        vcpy = ggml_cpy(c, Vc, vdst);

        ggml_tensor* Q = (W == 1) ? ggml_reshape_3d(c, q, hd, 1, nh)
                                  : ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
        ggml_tensor* K = ggml_view_3d(c, kcache, hd, cfg_.n_kv_max, nkvh,
                                      kcache->nb[1], kcache->nb[2], 0);
        ggml_tensor* V = ggml_view_3d(c, vcache, cfg_.n_kv_max, hd, nkvh,
                                      vcache->nb[1], vcache->nb[2], 0);
        ggml_tensor* kq = ggml_mul_mat(c, K, Q);
        ggml_tensor* p = ggml_soft_max_ext(c, kq, t_mask_, kq_scale, 0.0f);
        ggml_tensor* kqv = ggml_mul_mat(c, V, p);
        if (W > 1) {
            ggml_mul_mat_set_prec(kq,  GGML_PREC_F32);
            ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
        }
        kqv = (W == 1) ? ggml_reshape_2d(c, kqv, dq, 1)
                       : ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), dq, W);
        // Gejt vyhoda - DO vyhodnoj proekcii. Dva uzla, a ne odin: ggml_fused_mul_unary s
        // SIGMOID trebuet a->ne[0] == 1, a gejt polnoj shiriny; ta zhe stena i u etalona, gde
        // eto slijanie napisano i zakryto proverkoj, kotoraja nikogda ne istinna.
        kqv = ggml_mul(c, kqv, ggml_sigmoid(c, agate));
        attn_out = ggml_add(c, ggml_mul_mat(c, L.wo, kqv), cur);
        G.kdst = kdst; G.vdst = vdst; G.kcpy = kcpy; G.vcpy = vcpy;
        G.K = K; G.V = V; G.kq = kq; G.probs = p;
    }

    // Hvost, obshchij dlja oboih rodov sloja.
    ggml_tensor* xf = lnorm(c, attn_out, L.ffn_norm, eps);
    ggml_tensor* rl = ggml_mul_mat(c, L.router, xf);
    // OBSHCHIJ EKSPERT. Ne ggml_fused_up_gate: pri kvantovannyh up/gate odnogo tipa ta funkcija
    // uhodit v GGML_OP_FUSED_UP_GATE, u kotorogo na Vulkan realizacii net vovse - to est sloj
    // ostalsja by na CPU s vesami v videopamjati. Zdes razvjornuto rovno v to, vo chto
    // razvorachivaet ejo sobstvennaja nekvantovannaja vetka: dva umnozhenija i fused_mul_unary.
    ggml_tensor* s_up = ggml_mul_mat(c, L.up_shexp, xf);
    ggml_tensor* s_gt = ggml_mul_mat(c, L.gate_shexp, xf);
    ggml_tensor* sh = ggml_mul_mat(c, L.down_shexp,
                                   ggml_fused_mul_unary(c, s_gt, s_up, GGML_UNARY_OP_SILU));
    ggml_tensor* s_g = ggml_mul_mat(c, L.shexp_gate, xf);      // [1, W]
    // Tozhe dva uzla vmesto slitogo: SIGMOID u fused_mul_unary na Vulkan ne podderzhan (tolko
    // GELU/SILU/RELU), a shirokoveshchatelnaja forma trebuet imenno ego.
    sh = ggml_mul(c, sh, ggml_sigmoid(c, s_g));

    ggml_set_output(attn_out);
    ggml_set_output(xf);
    ggml_set_output(sh);
    ggml_set_output(rl);
    if (kcpy) ggml_build_forward_expand(gf, kcpy);
    if (vcpy) ggml_build_forward_expand(gf, vcpy);
    ggml_build_forward_expand(gf, attn_out);
    ggml_build_forward_expand(gf, xf);
    ggml_build_forward_expand(gf, sh);
    ggml_build_forward_expand(gf, rl);

    G.out = nullptr;
    G.o_res = attn_out; G.o_xf = xf; G.o_sh = sh; G.o_rl = rl; G.o_xm = nullptr;
    return finish_layer_graph(il, gf, G, err);
}

bool GpuStatic::set_step(int n_past, int n_kv) {
    if (!cfg_.layers || lg_.empty()) return false;
    // The multiple of 32 is not decoration: n_kv is the reduction length of the V*probs
    // matmul, and this fork's F16 matmul is silently wrong when it is not a multiple of four.
    // The padding positions are read and multiplied by the zero the -inf mask puts into probs,
    // so they cost bytes and never correctness. Everything is validated before anything is
    // patched, so a refusal leaves the graphs reading the whole allocation - slow and right.
    const int W = cfg_.layer_width > 0 ? cfg_.layer_width : 1;
    if (n_kv <= 0 || n_kv % 32 != 0 || n_kv > cfg_.n_kv_max) return false;
    // W pozicij, a ne odna: prohod na chetyre tokena pishet chetyre pozicii, i poslednjaja iz
    // nih objazana byt vnutri okna vnimanija - inache karta pishet za predel kesha.
    if (n_past < 0 || n_past + W > n_kv) return false;

    auto restride = [](ggml_tensor* t, int64_t n0) {
        t->ne[0] = n0;
        t->nb[1] = t->nb[0] * std::size_t(t->ne[0]);
        t->nb[2] = t->nb[1] * std::size_t(t->ne[1]);
        t->nb[3] = t->nb[2] * std::size_t(t->ne[2]);
    };
    if (int(t_mask_->ne[0]) != n_kv) restride(t_mask_, n_kv);
    // SREZ OKONNYH SLOEV. Pozicija i smotrit na [i - n_swa + 1, i]; graf schitaet pozicii
    // n_past..n_past+W-1, znachit nuzhen diapazon [n_past - n_swa + 1, n_past + W - 1].
    // Nizhnjaja granica okrugljaetsja VNIZ do 32: dlina svjortki f16-matmula dolzhna byt
    // kratna chetyrjom, a shag v 32 my derzhim vezde. Lishnie do 31 pozicij snizu zakryty
    // maskoj - oni stojat bajtov, no ne stojat korrektnosti.
    //
    // Vse okonnye sloi u gemma4 imejut ODNO okno (1024), poetomu srez u nih obshchij, i odnoj
    // maski t_mask_sw_ hvataet na vseh. Esli kogda-nibud okna stanut raznymi, eto perestanet
    // byt verno - poetomu nizhe stoit proverka, a ne predpolozhenie.
    // PLAN SCHITAETSJA I PROVERJAETSJA DO LJUBOJ MUTACII.
    //
    // Etot metod dokumentiruet sebja kak "vozvrashchaet false i NICHEGO ne menjaet". Ranshe eto
    // bylo nepravdoj dlja kolcevogo puti: ohrana perehoda cherez granicu kolca stojala v
    // SEREDINE cikla perenacelivanija zapisej, i k momentu `return false` chast sloev uzhe byla
    // perenacelena, maska kolca uzhe perezalita, a t_pos_ i step_past_ ostavalis starymi - to
    // est nesoglasovannaja smes vmesto "nichego ne izmenilos". Najdeno revju.
    //
    // Teper: pervyj prohod tolko SCHITAET srez kazhdogo sloja i proverjaet vsjo, chto mozhet ne
    // sojtis; vtoroj prohod primenjaet uzhe proverennyj plan i ne mozhet otkazat.
    std::vector<int> plan_lo(lg_.size(), 0), plan_len(lg_.size(), n_kv);
    int sw_lo = -1, sw_len = 0, sw_win = 0;
    for (std::size_t pi = 0; pi < lg_.size(); ++pi) {
        const LayerGraph& G = lg_[pi];
        // U sloja delta-seti net ni kesha, ni maski, ni pozicij: celit nechego. Ranshe etot
        // obhod predpolagal vnimanie v KAZHDOM sloe, i u qwen3next tri chetverti sloev
        // razymenovali by nulevye vidy.
        if (G.delta) continue;
        int lo = 0, len = n_kv;
        if (G.ring > 0) {
            // Zapis shirinoj W ne dolzhna perehodit granicu kolca: odna kopija etogo ne
            // vyrazhaet. Proverjaem ZDES, do mutacij.
            if (W > 1 && (n_past % G.ring) + W > G.ring) return false;
            lo = 0;
            len = std::min(G.ring, n_kv > 0 ? GGML_PAD(n_past + W, 32) : G.ring);
            if (len > G.ring) len = G.ring;
            if (len < 32) len = std::min(32, G.ring);
            if (sw_lo < 0) { sw_lo = 0; sw_len = len; sw_win = G.n_swa; }
            else if (sw_len != len) return false;
        } else if (G.n_swa > 0) {
            lo = n_past - G.n_swa + 1;
            if (lo < 0) lo = 0;
            lo = (lo / 32) * 32;
            len = n_kv - lo;
            if (sw_lo < 0) { sw_lo = lo; sw_len = len; }
            else if (sw_lo != lo || sw_len != len) return false;
        }
        plan_lo[pi] = lo; plan_len[pi] = len;
    }

    // Ot etoj stroki i nizhe otkazov byt ne mozhet - vsjo proverено vyshe.
    for (std::size_t pi = 0; pi < lg_.size(); ++pi) {
        LayerGraph& G = lg_[pi];
        if (G.delta) continue;
        const int lo = plan_lo[pi], len = plan_len[pi];
        G.kv_lo = lo; G.kv_len = len;
        G.K->ne[1] = len;    // strides belong to the cache, so only the extent moves
        G.V->ne[0] = len;
        restride(G.kq, len);
        restride(G.probs, len);
    }
    if (t_mask_sw_ && sw_len > 0) {
        if (int(t_mask_sw_->ne[0]) != sw_len) restride(t_mask_sw_, sw_len);
        step_mask_sw_src_ = nullptr;   // srez smenilsja - maska okna ustarela
    } else if (t_mask_sw_ && int(t_mask_sw_->ne[0]) != n_kv) {
        restride(t_mask_sw_, n_kv);
        step_mask_sw_src_ = nullptr;
    }

    // MASKU KOLCA STROIT KARTA. Hostovaja maska indeksirovana POZICIJAMI, a v kolce dannye
    // lezhat po slotam p % ring - host ob etoj raskladke ne znaet i znat ne dolzhen.
    //
    // Kakaja pozicija lezhit v slote s: samaja svezhaja p <= B s p = s (mod ring), gde B -
    // novejshaja zapisannaja pozicija (n_past + W - 1, potomu chto zapisi grafa idut PERED
    // chteniem). Slot goden dlja stroki r (pozicija n_past + r), esli ego pozicija ne iz
    // budushchego i ne vypala iz okna. Otricatelnaja pozicija znachit, chto slot eshchjo ni razu
    // ne zapisan - takoe byvaet, poka kolco ne zapolnilos.
    if (swa_ring_ && ring_ > 0 && t_mask_sw_ && sw_len > 0 && sw_win > 0) {
        const int B = n_past + W - 1;
        std::vector<float> m(std::size_t(sw_len) * std::size_t(W), -INFINITY);
        for (int r = 0; r < W; ++r) {
            const int pos = n_past + r;
            for (int s = 0; s < sw_len; ++s) {
                int d = (B - s) % ring_;
                if (d < 0) d += ring_;
                const int ps = B - d;
                if (ps < 0 || ps > pos || ps < pos - sw_win + 1) continue;
                m[std::size_t(r) * std::size_t(sw_len) + std::size_t(s)] = 0.0f;
            }
        }
        ggml_backend_tensor_set(t_mask_sw_, m.data(), 0, m.size() * sizeof(float));
        // Pometka "uzhe zagruzhena etim shagom": do_layer ne dolzhen ejo perezapisyvat
        // hostovym srezom.
        step_mask_sw_src_ = (const void*)t_mask_sw_;
    }
    if (step_past_ != n_past) {
        for (std::size_t il = 0; il < lg_.size(); ++il) {
            LayerGraph& G = lg_[il];
            if (G.delta) continue;   // pisat nekuda: kesha u etogo sloja net
            ggml_tensor* kc = kv_k_[il];
            ggml_tensor* vc = kv_v_[il];
            // Slot zapisi: pri kolce eto pozicija po modulju, inache sama pozicija.
            const int wslot = G.ring > 0 ? (n_past % G.ring) : n_past;
            const std::size_t ko = std::size_t(wslot) * kc->nb[1];
            const std::size_t vo = std::size_t(wslot) * ggml_element_size(vc);
            G.kdst->view_offs = ko;
            G.kdst->data = (char*)kc->data + ko;
            G.kcpy->view_offs = ko;
            G.kcpy->data = G.kdst->data;
            G.vdst->view_offs = vo;
            G.vdst->data = (char*)vc->data + vo;
            G.vcpy->view_offs = vo;
            G.vcpy->data = G.vdst->data;
            // I VIDY CHTENIJA tozhe: u okonnogo sloja oni nachinajutsja ne s nulja.
            if (G.n_swa > 0) {
                const std::size_t kro = std::size_t(G.kv_lo) * kc->nb[1];
                const std::size_t vro = std::size_t(G.kv_lo) * ggml_element_size(vc);
                G.K->view_offs = kro;
                G.K->data = (char*)kc->data + kro;
                G.V->view_offs = vro;
                G.V->data = (char*)vc->data + vro;
            }
            // Perehod cherez granicu kolca proverjen v pervom prohode, do mutacij - zdes
            // otkazyvat nelzja: chast sloev uzhe perenacelena, i vyhod otsjuda ostavil by
            // nesoglasovannoe sostojanie. Imenno tak i bylo do ispravlenija.
        }
        // SKOLKO SEKCIJ POZICIJ - govorit sam tenzor, a ne otdelnyj flag. Pri mnogomernom rope
        // (MROPE, i qwen3next iz nih) na token prihoditsja CHETYRE pozicii: rope indeksiruet
        // pos[i], pos[n+i], pos[2n+i], pos[3n+i], i dlja teksta eto t, t, t, 0. Raskladka - ta
        // zhe, chto u processornogo stroitelja (set_graph_inputs), i vzjata ottuda bukvalno.
        const int pt = int(t_pos_->ne[0]) / W;
        int32_t pos[64] = {0};
        for (int j = 0; j < W; ++j) {
            const int32_t pv = int32_t(n_past + j);
            pos[j] = pv;
            if (pt == 4) {
                pos[W + j]     = pv;
                pos[2 * W + j] = pv;
                pos[3 * W + j] = 0;   // chetvjortaja sekcija u teksta pustaja
            }
        }
        ggml_backend_tensor_set(t_pos_, pos, 0,
                                std::size_t(W) * std::size_t(pt) * sizeof(int32_t));
    }
    step_past_ = n_past;
    step_nkv_  = n_kv;
    step_mask_sent_ = false;
    // I ISTOCHNIK TOZHE. Bez etogo sbrosa ukazatel na masku ostajotsja "uzhe otpravlennym"
    // navsegda: mezhdu shagami on ne menjaetsja, menjaetsja SODERZHIMOE - chislo pozicij rastjot
    // s kazhdym tokjenom. Karta togda schitaet vnimanie po maske pervogo shaga do konca progona.
    // Stoilo 7 sovpavshih tokenov iz 192 i bylo pojmano tolko potomu, chto obvjazka sverjaet
    // tokeny i objavljaet plecho nedejstvitelnym vmesto togo, chtoby otdat pravdopodobnye cifry.
    step_mask_src_  = nullptr;
    return true;
}

bool GpuStatic::upload_kv(ggml_tensor* const* k, ggml_tensor* const* v, int n_layer,
                          int n_valid, std::string* err) {
    if (!cfg_.layers || lg_.empty()) { *err = "sloi ne na karte"; return false; }
    if (n_layer != cfg_.n_layer) { *err = "chislo sloev ne sovpadaet"; return false; }
    auto t0 = std::chrono::steady_clock::now();
    for (int il = 0; il < n_layer; ++il) {
        // Sloj delta-seti kesha ne imeet ni na hoste, ni na karte. Ranshe eta funkcija
        // predpolagala vnimanie v kazhdom sloe i otkazala by na pervom zhe iz tridcati shesti.
        if (cfg_.at(il).delta) continue;
        ggml_tensor* sk = k[std::size_t(il)];
        ggml_tensor* sv = v[std::size_t(il)];
        ggml_tensor* dk = kv_k_[std::size_t(il)];
        ggml_tensor* dv = kv_v_[std::size_t(il)];
        if (!sk || !sv || !dk || !dv) { *err = "kesh sloja otsutstvuet"; return false; }

        // KOLCEVOJ SLOJ: perekladka pozicij v sloty. Hostovyj kesh indeksirovan pozicijami,
        // kolco - slotami p % ring, i tolko POSLEDNIE ring pozicij voobshche nuzhny okonnomu
        // sloju. Sobiraem ves tenzor sloja v hostovom bufere obychnym memcpy i otdajom odnim
        // vyzovom: pooperacionnaja zagruzka dala by desjatki tysjach vyzovov na sloj.
        //
        // Nepreryvnyj diapazon pozicij perehodit v nepreryvnyj diapazon slotov s odnim
        // perehodom cherez granicu kolca - to est ne bolee dvuh memcpy na kazhduju stroku.
        const int rr = ring_of(il);
        if (rr > 0) {
            const GpuStaticGeom gk = cfg_.at(il);
            const int hd = gk.head_dim, nh = gk.n_head_kv;
            const int L  = std::min(n_valid, rr);
            if (L <= 0) continue;                    // pustoj promt - nechego perekladyvat
            const int p0 = n_valid - L;
            const int s0 = p0 % rr;
            const int c1 = std::min(L, rr - s0);     // do granicy kolca
            const int c2 = L - c1;                   // posle perehoda
            // K: [hd, ring, nh]; nepreryvnyj blok na poziciju
            {
                std::vector<char> stg(std::size_t(ggml_nbytes(dk)), 0);
                for (int h = 0; h < nh; ++h) {
                    const char* sp = (const char*)sk->data + std::size_t(h) * sk->nb[2];
                    char* dp = stg.data() + std::size_t(h) * dk->nb[2];
                    std::memcpy(dp + std::size_t(s0) * dk->nb[1],
                                sp + std::size_t(p0) * sk->nb[1],
                                std::size_t(c1) * sk->nb[1]);
                    if (c2 > 0) {
                        std::memcpy(dp, sp + std::size_t(p0 + c1) * sk->nb[1],
                                    std::size_t(c2) * sk->nb[1]);
                    }
                }
                ggml_backend_tensor_set(dk, stg.data(), 0, stg.size());
            }
            // V: [ring, hd, nh]; pozicija - element, poetomu po dve kopii na kazhduju paru
            // (golova, izmerenie)
            {
                const std::size_t es = ggml_type_size(GGML_TYPE_F16);
                std::vector<char> stg(std::size_t(ggml_nbytes(dv)), 0);
                for (int h = 0; h < nh; ++h) {
                    for (int d = 0; d < hd; ++d) {
                        const char* sp = (const char*)sv->data + std::size_t(h) * sv->nb[2]
                                       + std::size_t(d) * sv->nb[1];
                        char* dp = stg.data() + std::size_t(h) * dv->nb[2]
                                 + std::size_t(d) * dv->nb[1];
                        std::memcpy(dp + std::size_t(s0) * es, sp + std::size_t(p0) * es,
                                    std::size_t(c1) * es);
                        if (c2 > 0) {
                            std::memcpy(dp, sp + std::size_t(p0 + c1) * es,
                                        std::size_t(c2) * es);
                        }
                    }
                }
                ggml_backend_tensor_set(dv, stg.data(), 0, stg.size());
            }
            continue;
        }

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

int GpuStatic::delta_layers() const {
    int n = 0;
    for (ggml_tensor* t : dstate_) {
        if (t) ++n;
    }
    return n;
}

// PERENOSIMOE SOSTOJANIE POSLE PREFILLA - na kartu. Prefill schitaetsja na hoste (drugaja
// forma grafa, i on uprjot v arifmetiku, a ne v polosu), poetomu k pervomu generiruemomu tokenu
// okno svjortki i matrica sostojanija 36 sloev lezhat v HOSTOVOM DeltaState, a graf karty chitaet
// svoj bufer. Odin raz na promt, 75,4 MiB.
//
// Bez etogo vyzova karta nachala by dekod s NULEVOGO sostojanija. Eto ne padenie i ne oshibka
// formy: poluchilsja by svjaznyj tekst, zabyvshij promt - rovno tot rod otkaza, iz-za kotorogo
// upload_kv sravnivaet razmery, a ne predpolagaet ih. Poetomu razmery sravnivajutsja i zdes.
bool GpuStatic::upload_delta(ggml_tensor* const* st, int n_layer, std::string* err) {
    if (!cfg_.layers || lg_.empty()) { *err = "sloi ne na karte"; return false; }
    if (n_layer != cfg_.n_layer) { *err = "chislo sloev ne sovpadaet"; return false; }
    if (dstate_elems_ == 0) { *err = "u etoj arhitektury net perenosimogo sostojanija"; return false; }
    auto t0 = std::chrono::steady_clock::now();
    int moved = 0;
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor* d = dstate_[std::size_t(il)];
        if (!d) continue;                      // sloj vnimanija - u nego kesh, a ne sostojanie
        ggml_tensor* s = st[std::size_t(il)];
        if (!s) {
            char m[160];
            snprintf(m, sizeof(m),
                     "sloj %d: na karte sostojanie delta-seti est, a u hosta ego net", il);
            *err = m;
            return false;
        }
        if (ggml_nbytes(s) != ggml_nbytes(d) || s->type != d->type) {
            char m[240];
            snprintf(m, sizeof(m),
                     "sloj %d: sostojanie hosta %.2f MiB (%s) ne sovpalo s sostojaniem karty "
                     "%.2f MiB (%s)", il, double(ggml_nbytes(s)) / 1048576.0,
                     ggml_type_name(s->type), double(ggml_nbytes(d)) / 1048576.0,
                     ggml_type_name(d->type));
            *err = m;
            return false;
        }
        if (!s->data) { *err = "sostojanie hosta nedostupno hostu"; return false; }
        const std::size_t total = ggml_nbytes(s);
        const char* sp = (const char*)s->data;
        for (std::size_t off = 0; off < total; off += kUploadChunk) {
            const std::size_t n = std::min(kUploadChunk, total - off);
            ggml_backend_tensor_set(d, sp + off, off, n);
        }
        ++moved;
    }
    if (moved == 0) { *err = "ni odin sloj delta-seti ne perenesjon - eto ne uspeh"; return false; }
    ++st_.kv_uploads;
    st_.kv_upload_ms += ms_since(t0);
    return true;
}

void GpuStatic::do_layer(int il, ggml_tensor* dst, const ggml_tensor* cur,
                         const ggml_tensor* mask) {
    auto t0 = std::chrono::steady_clock::now();
    ++st_.layer_calls;
    // Tri velichiny u qwen3moe (ostatok, ego norma, logity) i CHETYRE u gemma4 i qwen3next: u nejo dve raznye
    // pre-normy nad odnim i tem zhe ostatkom - ffn_norm dlja plotnoj poloviny i pre_ffw_norm_2 dlja
    // marshrutiziruemoj - i vernut obe deshevle, chem zastavit host schitat odnu zanovo.
    const int W = cfg_.layer_width > 0 ? cfg_.layer_width : 1;
    const std::size_t out_floats =
        (std::size_t(cfg_.out_embd_slots()) * std::size_t(cfg_.n_embd)
         + std::size_t(cfg_.n_expert)) * std::size_t(W);

    if (!fail_msg_.empty()) { std::memset(dst->data, 0, ggml_nbytes(dst)); return; }
    if (step_nkv_ <= 0) {
        fail_msg_ = "set_step ne byl vyzvan pered grafom - shag ne naceljon";
        std::memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }
    if (cur->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) ||
        cur->ne[0] != cfg_.n_embd ||
        ggml_nelements(cur) != int64_t(cfg_.n_embd) * int64_t(W)) {
        fail_msg_ = "vhod sloja ne nepreryvnyj F32 [n_embd, shirina]";
        std::memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }
    if (std::size_t(ggml_nelements(dst)) != out_floats || !ggml_is_contiguous(dst)) {
        fail_msg_ = "vyhod sloja ne [(2..3)*n_embd + n_expert, shirina]";
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
        // OKONNYJ SLOJ: svoj tenzor maski i svoj srez. Sravnenie po ukazatelju zdes tozhe
        // rabotaet - mezhdu smenami sreza (a on menjaetsja raz na shag) zagruzka odna.
        LayerGraph& Gm = lg_[std::size_t(il)];
        if (Gm.ring > 0) {
            // Maska kolca uzhe postroena i zagruzhena v set_step - hostovyj srez zdes ne
            // nuzhen i byl by NEVEREN: on indeksirovan pozicijami, a ne slotami.
        } else if (Gm.n_swa > 0 && t_mask_sw_) {
            if (step_mask_sw_src_ != (mask ? mask->data : nullptr)) {
                if (!mask || mask->type != GGML_TYPE_F32 ||
                    mask->ne[0] < int64_t(Gm.kv_lo + Gm.kv_len) || mask->ne[1] < W) {
                    fail_msg_ = "maska okna uzhe, chem srez sloja";
                    std::memset(dst->data, 0, ggml_nbytes(dst));
                    return;
                }
                for (int r = 0; r < W; ++r) {
                    ggml_backend_tensor_set(
                        t_mask_sw_,
                        (const char*)mask->data + std::size_t(r) * mask->nb[1]
                            + std::size_t(Gm.kv_lo) * sizeof(float),
                        std::size_t(r) * std::size_t(Gm.kv_len) * sizeof(float),
                        std::size_t(Gm.kv_len) * sizeof(float));
                }
                step_mask_sw_src_ = mask->data;
            }
        } else if (step_mask_src_ != (mask ? mask->data : nullptr)) {
            if (!mask || mask->type != GGML_TYPE_F32 || mask->ne[0] < step_nkv_ ||
                mask->ne[1] < W) {
                fail_msg_ = "maska vnimanija uzhe, chem naceleno pozicij ili strok";
                std::memset(dst->data, 0, ggml_nbytes(dst));
                return;
            }
            // Odnim memcpy tolko kogda stroka hosta rovno nasha. Pri W > 1 i bolee shirokoj
            // stroke ploskaja kopija vzjala by masku sosednego tokena so sdvigom - oshibka,
            // kotoraja dajot rabotajushchee vnimanie po chuzhim pozicijam.
            if (mask->ne[0] == step_nkv_) {
                ggml_backend_tensor_set(t_mask_, mask->data, 0,
                                        std::size_t(step_nkv_) * std::size_t(W) * sizeof(float));
            } else {
                for (int r = 0; r < W; ++r) {
                    ggml_backend_tensor_set(
                        t_mask_, (const char*)mask->data + std::size_t(r) * mask->nb[1],
                        std::size_t(r) * std::size_t(step_nkv_) * sizeof(float),
                        std::size_t(step_nkv_) * sizeof(float));
                }
            }
            step_mask_src_ = mask->data;
            step_mask_sent_ = true;
        }
        ggml_backend_tensor_set(t_lx_, cur->data, 0,
                                std::size_t(cfg_.n_embd) * std::size_t(W) * sizeof(float));
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
                // ne - eto uzhe n_embd*W: kazhdyj vyhod nesjot W stolbcov, i sklejka idjot
                // PO VYHODAM, a ne po stolbcam. Host nizhe rezhet rb_ po tem zhe smeshchenijam,
                // poetomu menjat ih nelzja bez pravki obeih storon.
                const std::size_t ne = std::size_t(cfg_.n_embd) * std::size_t(W);
                bool ok = ggml_backend_vk_arm_readback(be_, G.o_res, rb_,      0, ne * sizeof(float));
                ok = ok && ggml_backend_vk_arm_readback(be_, G.o_xf, rb_ + ne, 0, ne * sizeof(float));
                std::size_t off = 2 * ne;
                // TRETIJ VYHOD razmerom n_embd: u gemma4 eto vtoraja pre-norma ostatka, u
                // qwen3next - vyhod obshchego eksperta. Odno mesto v raskladke na oba, potomu
                // chto host rezhet vyhod po SMESHCHENIJAM, i dva nezavisimyh ih schjota - eto
                // rovno tot rod rashozhdenija, kotoryj ne dajot oshibki formy.
                ggml_tensor* third = G.o_xm ? G.o_xm : G.o_sh;
                if (third) {
                    ok = ok && ggml_backend_vk_arm_readback(be_, third, rb_ + off, 0, ne * sizeof(float));
                    off += ne;
                }
                ok = ok && ggml_backend_vk_arm_readback(be_, G.o_rl, rb_ + off, 0,
                                                        std::size_t(cfg_.n_expert)
                                                            * std::size_t(W) * sizeof(float));
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
                const std::size_t ne = std::size_t(cfg_.n_embd) * std::size_t(W);
                ggml_backend_tensor_get(G.o_res, rb_,      0, ne * sizeof(float));
                ggml_backend_tensor_get(G.o_xf,  rb_ + ne, 0, ne * sizeof(float));
                std::size_t off = 2 * ne;
                ggml_tensor* third = G.o_xm ? G.o_xm : G.o_sh;
                if (third) {
                    ggml_backend_tensor_get(third, rb_ + off, 0, ne * sizeof(float));
                    off += ne;
                }
                ggml_backend_tensor_get(G.o_rl, rb_ + off, 0,
                                        std::size_t(cfg_.n_expert)
                                            * std::size_t(W) * sizeof(float));
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
    // Tri velichiny u qwen3moe i CHETYRE u gemma4 - u nejo dve raznye pre-normy nad odnim
    // ostatkom. Razmer dolzhen sovpadat s tem, chto realno chitaetsja obratno (out_floats v
    // do_layer), inache vid vyhodit za predely istochnika i ggml.c:5427 lovit eto utverzhdeniem.
    // Odnomernyj po dline i shirine srazu: sklejka idjot PO VYHODAM (ostatok, normy, logity),
    // i kazhdyj iz nih nesjot W stolbcov, tak chto vyhody NE lezhat stolbcami odnogo [X, W]
    // tenzora. Graf dekoda rezhet eto ggml_view_2d po smeshcheniju - ono dolzhno sovpadat s tem,
    // po kakomu do_layer skladyvaet rb_.
    const int Wp = cfg_.layer_width > 0 ? cfg_.layer_width : 1;
    ggml_tensor* proto = ggml_new_tensor_2d(c, GGML_TYPE_F32,
                                            (cfg_.out_embd_slots() * cfg_.n_embd
                                                + cfg_.n_expert) * Wp, 1);
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
            // den == 0 means the REFERENCE row is all zeros - there is nothing to be
            // relative to. Returning 0.0 scored that as a perfect match and fed the
            // pass/fail directly (rule 11); the sibling helper in gpu_experts.cpp has
            // always returned 1.0 for the same case.
            const double rel = den > 0.0 ? std::sqrt(num / den) : 1.0;
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
