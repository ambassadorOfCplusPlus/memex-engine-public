#include "gpu_static.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ggml-alloc.h"
#include "ggml-vulkan.h"
// Only for the two heap reports, which are static members there and are the one thing in this
// arrangement that cannot be checked any other way: "device-local" in every other report is
// exactly the word that hides a buffer the driver quietly moved to system memory.
#include "gpu_experts.hpp"

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
