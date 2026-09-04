#include "expert_store.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>

#if defined(_WIN32)
// NOMINMAX objazatelen: windows.h objavljaet min i max MAKROSAMI, i togda std::min nizhe
// razbiraetsja kak std::(...) - oshibka C2589 v strochke, kotoraja k Windows otnoshenija ne
// imeet. Odin raz na etom uzhe stojali polchasa.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <malloc.h>   // _aligned_malloc / _aligned_free
#pragma comment(lib, "psapi.lib")
#endif

// Repak zhivjot v ggml/src/iqk/iqk_quantize.h, kotoryj ne lezhit v puti vkljuchenij primerov.
// Objavlenija zdes - te zhe samye, i oni pod extern "C", kak v samom zagolovke, tak chto
// nesovpadenie signatury bylo by oshibkoj komponovki, a ne tihoj oshibkoj vyzova.
extern "C" {
void iqk_repack_tensor(struct ggml_tensor* tensor);
int  iqk_repacked_type(const struct ggml_tensor* tensor);
}

namespace memex {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::size_t align_up(std::size_t v, std::size_t a) {
    return a ? ((v + a - 1) / a) * a : v;
}

// Opisatel sreza odnogo slota kak otdelnogo tenzora - dlja iqk_repack_tensor, kotoryj
// prinimaet imenno ggml_tensor. Vsjo pole obnuljaetsja: imja dolzhno byt pustym (spisok
// zapreshchennyh tenzorov u repaka sravnivaet imenno imja), view_src i buffer - nulevymi.
ggml_tensor slice_descr(ggml_type base, const ggml_tensor* t, void* data) {
    ggml_tensor d;
    std::memset(&d, 0, sizeof(d));
    d.type = base;
    d.ne[0] = t->ne[0];
    d.ne[1] = t->ne[1];
    d.ne[2] = 1;
    d.ne[3] = 1;
    d.nb[0] = t->nb[0];
    d.nb[1] = t->nb[1];
    d.nb[2] = t->nb[2];
    d.nb[3] = t->nb[2];
    d.data = data;
    return d;
}

}  // namespace

ExpertStoreStats ExpertStoreStats::since(const ExpertStoreStats& b) const {
    ExpertStoreStats d;
    d.tokens      = tokens      - b.tokens;
    d.picks       = picks       - b.picks;
    d.hits        = hits        - b.hits;
    d.sync_misses = sync_misses - b.sync_misses;
    d.refreshes   = refreshes   - b.refreshes;
    d.fills       = fills       - b.fills;
    d.fill_bytes  = fill_bytes  - b.fill_bytes;
    d.evictions   = evictions   - b.evictions;
    d.ms_sync     = ms_sync     - b.ms_sync;
    d.ms_fill     = ms_fill     - b.ms_fill;
    d.ms_repack   = ms_repack   - b.ms_repack;
    d.ms_map      = ms_map      - b.ms_map;
    d.pref_predicted = pref_predicted - b.pref_predicted;
    d.pref_issued    = pref_issued    - b.pref_issued;
    d.pref_used      = pref_used      - b.pref_used;
    d.pref_waits     = pref_waits     - b.pref_waits;
    d.ms_pref_wait   = ms_pref_wait   - b.ms_pref_wait;
    return d;
}

ProcMem proc_mem() {
    ProcMem m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    std::memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        m.ok     = true;
        m.rss    = uint64_t(pmc.WorkingSetSize);
        m.peak   = uint64_t(pmc.PeakWorkingSetSize);
        m.faults = uint64_t(pmc.PageFaultCount);
    }
#endif
    return m;
}

ExpertStore::~ExpertStore() {
    // Ostanovit I/O-potok do osvobozhdenija buferov: on pishet v sloty i v io_buf_pref_.
    if (worker_up_) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        worker_up_ = false;
    }
    for (Layer& L : layers_) {
        if (L.buf) ggml_backend_buffer_free(L.buf);
        L.buf = nullptr;
    }
    if (ctx_) ggml_free(ctx_);
    ctx_ = nullptr;
#if defined(_WIN32)
    if (fh_ && fh_ != INVALID_HANDLE_VALUE) CloseHandle((HANDLE)fh_);
    if (io_buf_) _aligned_free(io_buf_);
    if (fh_pref_ && fh_pref_ != INVALID_HANDLE_VALUE) CloseHandle((HANDLE)fh_pref_);
    if (io_buf_pref_) _aligned_free(io_buf_pref_);
#endif
    fh_ = nullptr;
    io_buf_ = nullptr;
    fh_pref_ = nullptr;
    io_buf_pref_ = nullptr;
}

namespace {
constexpr std::size_t kSector = 4096;   // NO_BUFFERING trebuet vyravnivanija po sektoru diska
}

// Otkryt gguf, uznat fajlovye smeshchenija tenzorov ekspertov po IMENAM src-tenzorov, otkryt
// fajl s FILE_FLAG_NO_BUFFERING, vydelit vyrovnennyj io_buf_. Vsjo eto - odin raz.
bool ExpertStore::open_direct(std::string* err) {
#if !defined(_WIN32)
    if (err) *err = "prjamoe chtenie realizovano tolko dlja Windows";
    return false;
#else
    // Smeshchenija iz gguf: data_offset (nachalo sekcii dannyh v fajle) + tensor_offset.
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    gguf_context* g = gguf_init_from_file(cfg_.gguf_path.c_str(), gp);
    if (!g) {
        if (err) *err = "gguf_init_from_file ne otkryl " + cfg_.gguf_path;
        return false;
    }
    const uint64_t data_off = (uint64_t)gguf_get_data_offset(g);
    std::size_t max_chunk = 0;
    bool okall = true;
    std::string why;
    for (int il = 0; il < cfg_.n_layer && okall; ++il) {
        Layer& L = layers_[std::size_t(il)];
        const ggml_tensor* three[3] = {L.src_gate, L.src_up, L.src_down};
        uint64_t* dstoff[3] = {&L.off_gate, &L.off_up, &L.off_down};
        for (int k = 0; k < 3; ++k) {
            const char* nm = ggml_get_name(three[k]);
            const int ti = gguf_find_tensor(g, nm);
            if (ti < 0) {
                okall = false;
                why = std::string("tenzor '") + (nm ? nm : "?") + "' ne najden v gguf";
                break;
            }
            *dstoff[k] = data_off + (uint64_t)gguf_get_tensor_offset(g, ti);
            const std::size_t nb2 = std::size_t(three[k]->nb[2]);
            if (nb2 > max_chunk) max_chunk = nb2;
        }
    }
    gguf_free(g);
    if (!okall) {
        if (err) *err = "prjamoe chtenie: " + why;
        return false;
    }

    HANDLE h = CreateFileA(cfg_.gguf_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        if (err) *err = "CreateFile(NO_BUFFERING) ne otkryl fajl, kod " +
                        std::to_string((unsigned long)GetLastError());
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz)) {
        CloseHandle(h);
        if (err) *err = "GetFileSizeEx ne dal razmer fajla";
        return false;
    }
    file_size_ = (uint64_t)sz.QuadPart;

    // Bufer pod odin kusok: dlina eksperta plus do dvuh sektorov na vyravnivanie kraev.
    io_cap_ = align_up(max_chunk, kSector) + 2 * kSector;
    io_buf_ = _aligned_malloc(io_cap_, kSector);
    if (!io_buf_) {
        CloseHandle(h);
        io_cap_ = 0;
        if (err) *err = "io-bufer pod prjamoe chtenie ne vydelilsja";
        return false;
    }
    fh_ = (void*)h;
    direct_ = true;
    return true;
#endif
}

// Prjamoe chtenie [foff, foff+len) v dst mimo strannichnogo kesha cherez UKAZANNYE hendl i
// bufer. Krajа vyravnivajutsja po sektoru; chitaem vyrovnennyj diapazon v buf, kopiruem
// nuzhnyj poddiapazon. Schjotchikov ne trogaet - eto delaet vyzyvajushchij (potok grafa i
// I/O-potok schitajut razdelno).
bool ExpertStore::read_range(void* fh, void* buf, std::size_t cap, uint64_t foff,
                             std::size_t len, void* dst) {
#if !defined(_WIN32)
    (void)fh; (void)buf; (void)cap; (void)foff; (void)len; (void)dst;
    return false;
#else
    const uint64_t astart = foff & ~(uint64_t)(kSector - 1);
    const uint64_t head   = foff - astart;
    uint64_t aend = align_up(std::size_t(foff + len), kSector);
    // Ne chitat za sektornyj hvost fajla: poslednij sektor mozhet byt nepolnym, i ReadFile
    // vernjot menshe bajt - eto normal'no, poka pokryt head+len.
    const uint64_t fend = align_up(std::size_t(file_size_), kSector);
    if (aend > fend) aend = fend;
    const std::size_t alen = std::size_t(aend - astart);
    if (alen > cap || head + len > alen) return false;

    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)astart;
    if (!SetFilePointerEx((HANDLE)fh, li, nullptr, FILE_BEGIN)) return false;
    DWORD got = 0;
    if (!ReadFile((HANDLE)fh, buf, (DWORD)alen, &got, nullptr)) return false;
    if ((std::size_t)got < head + len) return false;
    std::memcpy(dst, (const char*)buf + head, len);
    return true;
#endif
}

bool ExpertStore::read_chunk(uint64_t foff, std::size_t len, void* dst) {
    if (!read_range(fh_, io_buf_, io_cap_, foff, len, dst)) return false;
    direct_reads_++;
    return true;
}

std::size_t ExpertStore::bytes_per_expert(const ggml_tensor* gate, const ggml_tensor* up,
                                          const ggml_tensor* down) {
    if (!gate || !up || !down) return 0;
    return std::size_t(gate->nb[2]) + std::size_t(up->nb[2]) + std::size_t(down->nb[2]);
}

int ExpertStore::auto_capacity(uint64_t avail, uint64_t reserve, std::size_t bpe,
                               int n_layer, int spares, int n_expert) {
    if (bpe == 0 || n_layer <= 0) return 0;
    if (avail <= reserve) return 0;
    const double room = double(avail - reserve);
    const double slots = room / (double(bpe) * double(n_layer));
    int c = int(slots) - spares;
    if (c < 0) c = 0;
    if (c > n_expert) c = n_expert;
    return c;
}

bool ExpertStore::init(const ExpertStoreConfig& cfg,
                       const std::vector<const ggml_tensor*>& gate,
                       const std::vector<const ggml_tensor*>& up,
                       const std::vector<const ggml_tensor*>& down,
                       std::string* err) {
    cfg_ = cfg;
    if (cfg_.capacity <= 0) return true;    // vykljucheno - i eto ne oshibka
    if (int(gate.size()) != cfg_.n_layer || int(up.size()) != cfg_.n_layer ||
        int(down.size()) != cfg_.n_layer) {
        if (err) *err = "spisok tenzorov ekspertov ne po chislu sloev";
        return false;
    }
    // Ёmkost ne bolshe chisla ekspertov: pri C == n_expert rezidentno VSJO, zapasnye sloty
    // terjajut smysl, i promahu vzjatsja neotkuda.
    if (cfg_.capacity >= cfg_.n_expert) {
        cfg_.capacity = cfg_.n_expert;
        cfg_.spares   = 0;
    } else if (cfg_.spares < cfg_.n_used) {
        // Zapasnyh dolzhno hvatat na hudshij token: do n_used promahov v odnom sloe, i ni
        // odin iz nih ne imeet prava vytesnit slot, zapolnennyj na etom zhe tokene.
        cfg_.spares = cfg_.n_used;
    }
    if (cfg_.capacity + cfg_.spares > cfg_.n_expert) {
        cfg_.spares = cfg_.n_expert - cfg_.capacity;
    }
    n_slots_ = cfg_.capacity + cfg_.spares;

    per_expert_ = bytes_per_expert(gate[0], up[0], down[0]);
    if (per_expert_ == 0) {
        if (err) *err = "razmer eksperta poluchilsja nulevym";
        return false;
    }

    ggml_init_params ip = {};
    ip.mem_size   = ggml_tensor_overhead() * std::size_t(3 * cfg_.n_layer + 8) + 4096;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ctx_ = ggml_init(ip);
    if (!ctx_) {
        if (err) *err = "ggml_init dlja hranilishcha ne udalsja";
        return false;
    }

    layers_.resize(std::size_t(cfg_.n_layer));
    sites_.resize(std::size_t(cfg_.n_layer));
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    const std::size_t alg = ggml_backend_buft_get_alignment(buft);

    for (int il = 0; il < cfg_.n_layer; ++il) {
        Layer& L = layers_[std::size_t(il)];
        sites_[std::size_t(il)].self = this;
        sites_[std::size_t(il)].il   = il;
        const ggml_tensor* sg = gate[std::size_t(il)];
        const ggml_tensor* su = up[std::size_t(il)];
        const ggml_tensor* sd = down[std::size_t(il)];
        if (!sg || !su || !sd) {
            if (err) *err = "u sloja " + std::to_string(il) + " net tenzora ekspertov";
            return false;
        }
        // Nepreryvnost - uslovie kopirovanija po nb[2] i uslovie repaka. Pri mmap ona est;
        // proverjaetsja imenno potomu, chto obratnoe dalo by pravdopodobnyj musor.
        if (!ggml_is_contiguous(sg) || !ggml_is_contiguous(su) || !ggml_is_contiguous(sd)) {
            if (err) *err = "tenzory ekspertov sloja " + std::to_string(il) + " ne nepreryvny";
            return false;
        }
        if (sg->ne[2] != cfg_.n_expert || su->ne[2] != cfg_.n_expert ||
            sd->ne[2] != cfg_.n_expert) {
            if (err) *err = "tenzory ekspertov sloja " + std::to_string(il) +
                            " ne po chislu ekspertov";
            return false;
        }
        L.src_gate = sg; L.src_up = su; L.src_down = sd;
        L.gate = ggml_new_tensor_3d(ctx_, sg->type, sg->ne[0], sg->ne[1], n_slots_);
        L.up   = ggml_new_tensor_3d(ctx_, su->type, su->ne[0], su->ne[1], n_slots_);
        L.down = ggml_new_tensor_3d(ctx_, sd->type, sd->ne[0], sd->ne[1], n_slots_);
        if (!L.gate || !L.up || !L.down) {
            if (err) *err = "tenzory hranilishcha sloja " + std::to_string(il) +
                            " ne sozdalis";
            return false;
        }
        ggml_set_name(L.gate, ("es.gate." + std::to_string(il)).c_str());
        ggml_set_name(L.up,   ("es.up."   + std::to_string(il)).c_str());
        ggml_set_name(L.down, ("es.down." + std::to_string(il)).c_str());
        // Shag po ekspertu dolzhen sovpadat s istochnikom, inache kopija tri-kuska ljozhet
        // ne tuda i dast pravdopodobnyj musor.
        if (L.gate->nb[2] != sg->nb[2] || L.up->nb[2] != su->nb[2] ||
            L.down->nb[2] != sd->nb[2]) {
            if (err) *err = "shag po ekspertu v sloe " + std::to_string(il) +
                            " ne sovpal s modelju";
            return false;
        }

        const std::size_t nb_g = ggml_nbytes(L.gate);
        const std::size_t nb_u = ggml_nbytes(L.up);
        const std::size_t nb_d = ggml_nbytes(L.down);
        const std::size_t need = align_up(nb_g, alg) + align_up(nb_u, alg) + nb_d;
        L.buf = ggml_backend_buft_alloc_buffer(buft, need);
        if (!L.buf) {
            char b[192];
            snprintf(b, sizeof(b),
                     "pamjat pod sloj %d (%.2f GiB) ne vydelilas; vydeleno do etogo %.2f GiB",
                     il, double(need) / 1073741824.0, double(bytes_) / 1073741824.0);
            if (err) *err = b;
            return false;
        }
        char* base = (char*)ggml_backend_buffer_get_base(L.buf);
        std::size_t off = 0;
        ggml_backend_tensor_alloc(L.buf, L.gate, base + off); off += align_up(nb_g, alg);
        ggml_backend_tensor_alloc(L.buf, L.up,   base + off); off += align_up(nb_u, alg);
        ggml_backend_tensor_alloc(L.buf, L.down, base + off);
        bytes_ += need;

        L.score.assign(std::size_t(cfg_.n_expert), 0.0f);
        L.slot_of.assign(std::size_t(cfg_.n_expert), -1);
        L.want.assign(std::size_t(cfg_.n_expert), 0);
        L.seen.assign(std::size_t(cfg_.n_expert), 0);
        L.id_of.assign(std::size_t(n_slots_), -1);
        L.touched.assign(std::size_t(n_slots_), 0);
        L.filling.assign(std::size_t(n_slots_), 0);
        L.pref_tag.assign(std::size_t(n_slots_), 0);
        L.ring = 0;
    }

    // REPAK: reshaetsja odin raz i po SAMIM tipam, a ne po nazvaniju kvantizacii.
    repacked_ = false;
    if (cfg_.repack) {
        const Layer& L0 = layers_[0];
        const ggml_tensor* three[3] = {L0.gate, L0.up, L0.down};
        std::string bad;
        for (int k = 0; k < 3 && bad.empty(); ++k) {
            const ggml_tensor* t = three[k];
            ggml_tensor d = slice_descr(t->type, t, nullptr);
            d.ne[2] = t->ne[2];
            d.nb[3] = d.nb[2] * d.ne[2];
            const ggml_type nt = (ggml_type)iqk_repacked_type(&d);
            if (nt == t->type) {
                bad = std::string("tip ") + ggml_type_name(t->type) +
                      " repak ne podderzhivaet";
            } else if (ggml_row_size(nt, t->ne[0]) != ggml_row_size(t->type, t->ne[0])) {
                bad = std::string("u ") + ggml_type_name(nt) +
                      " drugoj razmer stroki, chem u " + ggml_type_name(t->type);
            }
        }
        if (bad.empty()) {
            repacked_ = true;
            repack_why_ = "vkljuchjon";
        } else {
            repack_why_ = "OTKAZAN: " + bad;
        }
    } else {
        repack_why_ = "ne prosili";
    }

    // PRJAMOE CHTENIE MIMO MMAP. Esli zadan put k fajlu, sloty zapolnjajutsja ReadFile'om s
    // NO_BUFFERING, a ne memcpy iz mmap. Bez etogo hranilishche - VTORAJA kopija ekspertov
    // (privatnaja + strannichnyj kesh mmap), i dve ne vlezajut v 26 GB (STATE, shag 4 A/B).
    // Otkaz vsluh: esli put zadan, no chtenie ne nastroilos - eto oshibka, a ne tihij otkat
    // na memcpy, potomu chto memcpy vernul by trjoshing, kotoryj my i ubiraem.
    if (!cfg_.gguf_path.empty()) {
        std::string derr;
        if (!open_direct(&derr)) {
            if (err) *err = derr;
            return false;
        }
    }
    token_ = 1;
    return true;
}

bool ExpertStore::load_prior(const char* path, std::string* err) {
    if (!on()) return true;
    FILE* f = fopen(path, "rb");
    if (!f) {
        if (err) *err = std::string("fajl zatravki ne otkrylsja: ") + path;
        return false;
    }
    int32_t hd[4] = {0, 0, 0, 0};
    if (fread(hd, sizeof(int32_t), 4, f) != 4) {
        fclose(f);
        if (err) *err = "zagolovok zatravki ne prochitalsja";
        return false;
    }
    if (hd[0] != 0x45585052) {
        fclose(f);
        if (err) *err = "eto ne fajl zatravki (net metki EXPR)";
        return false;
    }
    if (hd[1] != cfg_.n_layer || hd[2] != cfg_.n_expert) {
        fclose(f);
        char b[160];
        snprintf(b, sizeof(b), "zatravka na %d x %d, a model %d x %d", hd[1], hd[2],
                 cfg_.n_layer, cfg_.n_expert);
        if (err) *err = b;
        return false;
    }
    std::vector<float> row(std::size_t(cfg_.n_expert));
    for (int il = 0; il < cfg_.n_layer; ++il) {
        if (fread(row.data(), sizeof(float), row.size(), f) != row.size()) {
            fclose(f);
            if (err) *err = "telo zatravki koroche zagolovka";
            return false;
        }
        Layer& L = layers_[std::size_t(il)];
        for (int e = 0; e < cfg_.n_expert; ++e) L.score[std::size_t(e)] += row[std::size_t(e)];
    }
    fclose(f);
    return true;
}

void ExpertStore::observe_prefill(const int32_t* ids, int n_layer, int n_tokens, int n_used) {
    if (!on() || !ids) return;
    for (int il = 0; il < n_layer && il < cfg_.n_layer; ++il) {
        Layer& L = layers_[std::size_t(il)];
        const int32_t* p = ids + std::size_t(il) * std::size_t(n_tokens) * std::size_t(n_used);
        for (int t = 0; t < n_tokens; ++t) {
            for (int j = 0; j < n_used; ++j) {
                const int32_t e = p[std::size_t(t) * std::size_t(n_used) + std::size_t(j)];
                if (e < 0 || e >= cfg_.n_expert) continue;
                L.score[std::size_t(e)] += 1.0f;
                L.seen[std::size_t(e)] = 1;
            }
        }
    }
}

void ExpertStore::fill_slot(Layer& L, int slot, int id) {
    const int32_t old = L.id_of[std::size_t(slot)];
    if (old >= 0) {
        L.slot_of[std::size_t(old)] = -1;
        st_.evictions++;
    }
    const auto t0 = Clock::now();
    const ggml_tensor* src[3] = {L.src_gate, L.src_up, L.src_down};
    ggml_tensor* dst[3] = {L.gate, L.up, L.down};
    const uint64_t off[3] = {L.off_gate, L.off_up, L.off_down};
    for (int k = 0; k < 3; ++k) {
        const std::size_t nb2 = std::size_t(dst[k]->nb[2]);
        char* d = (char*)dst[k]->data + std::size_t(slot) * nb2;
        if (direct_) {
            // Prjamoe chtenie s diska: NE trogaet stranicy mmap etogo eksperta, tak chto oni
            // ostajutsja holodnymi i ne konkurirujut za pamjat s privatnym slotom.
            const uint64_t foff = off[k] + std::size_t(id) * nb2;
            if (!read_chunk(foff, nb2, d)) {
                static bool said = false;
                if (!said) {
                    said = true;
                    printf("hranilishche ekspertov: PRJAMOE CHTENIE ne udalos (sloj-tenzor %d, "
                           "ekspert %d, smeshchenie %llu, %zu bajt) - slot NE zapolnen\n",
                           k, id, (unsigned long long)foff, nb2);
                    fflush(stdout);
                }
            }
        } else {
            std::memcpy(d, (const char*)src[k]->data + std::size_t(id) * nb2, nb2);
        }
    }
    st_.ms_fill += ms_since(t0);
    st_.fills++;
    st_.fill_bytes += per_expert_;
    L.id_of[std::size_t(slot)] = id;
    L.slot_of[std::size_t(id)] = slot;
    L.touched[std::size_t(slot)] = token_;
    // Pri pervichnoj zalivke sloty NE perepakovyvajutsja po odnomu: prime() perepakuet ves
    // tenzor odnim vyzovom posle togo, kak vse sloty zapolneny. Bez etogo uslovija slot
    // perepakovalsja by DVAZHDY - po odnomu i potom v sostave tenzora, - a repak ne
    // idempotenten: vtoroj prohod prochital by uzhe pereplejotennye bajty kak ishodnye.
    if (repacked_ && !bulk_) {
        const auto t1 = Clock::now();
        for (int k = 0; k < 3; ++k) {
            ggml_tensor d = slice_descr(k == 2 ? L.src_down->type
                                               : (k == 0 ? L.src_gate->type : L.src_up->type),
                                        dst[k],
                                        (char*)dst[k]->data +
                                            std::size_t(slot) * std::size_t(dst[k]->nb[2]));
            iqk_repack_tensor(&d);
        }
        st_.ms_repack += ms_since(t1);
    }
}

// Zhertva pod sinhronnyj promah: pervyj po kolcu slot, kotoryj (a) pust, libo (b) zanjat
// nerezidentnym ekspertom I ne byl TRONUT na etom tokene. Vtoroe uslovie i est ta samaja
// zashchita, o kotoroj dlinnoe zamechanie v do_map: "tronut" znachit i "zapolnen", i
// "prochitan kak popadanie".
//
// Pochemu zapasnyh dolzhno byt ne menshe top-k. K momentu obrabotki m-go mesta sloja tronuto
// ne bolshe m-1 nerezidentnyh slotov, tak chto pri spares >= n_used zhertva est vsegda, i
// pri m = n_used tozhe. Poetomu init podnimaet spares do n_used, a ne doverjaet flagu.
int ExpertStore::pick_victim(Layer& L) {
    for (int k = 0; k < n_slots_; ++k) {
        const int s = (L.ring + k) % n_slots_;
        const int32_t occ = L.id_of[std::size_t(s)];
        if (occ < 0) { L.ring = (s + 1) % n_slots_; return s; }
        if (L.want[std::size_t(occ)]) continue;
        if (L.touched[std::size_t(s)] == token_) continue;
        L.ring = (s + 1) % n_slots_;
        return s;
    }
    return -1;
}

void ExpertStore::do_map(int il, ggml_tensor* dst, const ggml_tensor* sel) {
    const auto tm = Clock::now();
    Layer& L = layers_[std::size_t(il)];
    const int nj = int(sel->ne[0]);
    const int nr = int(sel->ne[1]);
    for (int r = 0; r < nr; ++r) {
        for (int j = 0; j < nj; ++j) {
            const int32_t id = *(const int32_t*)((const char*)sel->data +
                                                 std::size_t(r) * sel->nb[1] +
                                                 std::size_t(j) * sel->nb[0]);
            int32_t s = -1;
            if (id >= 0 && id < cfg_.n_expert) {
                st_.picks++;
                L.score[std::size_t(id)] += 1.0f;
                L.seen[std::size_t(id)] = 1;
                s = L.slot_of[std::size_t(id)];
                if (s >= 0) {
                    // PREDZAGRUZKA MOGLA NE USPET. Esli slot zapolnjaet I/O-potok (filling),
                    // zhdjom ego zdes - eto tot samyj "sinhronnyj promah, kotoryj predzagruzka
                    // ne uspela zakryt". Bajty ot ozhidanija te zhe, chto ot sinhronnogo
                    // chtenija, tak chto otvet ne menjaetsja; menjaetsja tolko, spryatalos li
                    // chtenie za schjotom K sloev ili net.
                    if (pref_on_) {
                        std::unique_lock<std::mutex> lk(mu_);
                        if (L.filling[std::size_t(s)]) {
                            const auto tw = Clock::now();
                            cv_.wait(lk, [&] { return L.filling[std::size_t(s)] == 0; });
                            st_.ms_pref_wait += ms_since(tw);
                            st_.pref_waits++;
                        }
                        if (L.pref_tag[std::size_t(s)] == token_) st_.pref_used++;
                    }
                    st_.hits++;
                    // POPADANIE TOZHE ZAKREPLJAET SLOT NA ETOT TOKEN, i eto ne
                    // optimizacija, a uslovie pravilnosti. Bez etoj stroki byl real'nyj
                    // defekt, i on vyglyadel imenno tak, kak zdes vsegda: pervyj token
                    // sovpadal s etalonom do znaka, a so vtorogo tekst rashodilsja.
                    //
                    // Mehanizm. Pust ekspert X lezhit v zapasnom slote 32 (ne rezidentnyj,
                    // znachit want[X] == 0). Na etom tokene mesto j=0 popalo v X: dst[0] = 32,
                    // i BOLSHE NICHEGO ne pomenjalos. Mesto j=3 - promah, pick_victim ishchet
                    // zhertvu, vidit slot 32 (zanjat, ne want, na etom tokene ne zapolnjalsja)
                    // i kladjot tuda eksperta Y. Teper dst[0] i dst[3] oba ukazyvajut na 32:
                    // mul_mat_id dvazhdy schitaet Y i ni razu X. Formy vernye, NaN net, tekst
                    // pravdopodobnyj - rovno tot rod otkaza, protiv kotorogo napisan ves etot
                    // proekt. Na PERVOM tokene defekta ne vidno voobshche, potomu chto vse
                    // zapasnye sloty eshchjo pusty i zhertvoj vsegda okazyvaetsja pustoj slot.
                    L.touched[std::size_t(s)] = token_;
                } else {
                    const auto t0 = Clock::now();
                    const int v = pick_victim(L);
                    if (v < 0) {
                        // Nekuda polozhit. Eto ne "propustim eksperta": takoj rezhim zdes
                        // zapreshchjon, potomu chto on menjaet otvet. Poetomu gromkij otkaz
                        // odin raz na sloj i -1 v ide - mul_mat_id obnulit mesto, i eto
                        // budet VIDNO, a ne prinjato za uskorenie.
                        static bool said = false;
                        if (!said) {
                            said = true;
                            printf("hranilishche ekspertov: v sloe %d NET slota pod promah "
                                   "(vsego %d, rezidentnyh %d, zapasnyh %d) - mesto obnuleno, "
                                   "OTVET IZMENJON\n", il, n_slots_, cfg_.capacity,
                                   cfg_.spares);
                            fflush(stdout);
                        }
                    } else {
                        fill_slot(L, v, id);
                        s = v;
                    }
                    st_.sync_misses++;
                    st_.ms_sync += ms_since(t0);
                }
            }
            *(int32_t*)((char*)dst->data + std::size_t(r) * dst->nb[1] +
                        std::size_t(j) * dst->nb[0]) = s;
        }
    }
    st_.ms_map += ms_since(tm);
}

void ExpertStore::map_op(ggml_tensor* dst, const ggml_tensor* a, int ith, int /*nth*/,
                         void* ud) {
    if (ith != 0) return;
    Site* s = (Site*)ud;
    s->self->do_map(s->il, dst, a);
}

ggml_tensor* ExpertStore::slots_node(ggml_context* c, int il, ggml_tensor* sel,
                                     ggml_cgraph* gf) {
    ggml_tensor* t = ggml_map_custom1(c, sel, map_op, /*n_tasks=*/1,
                                      &sites_[std::size_t(il)]);
    // Zavisimost po dannym, a ne nadezhda na porjadok: mul_mat_id nizhe beret imenno t, tak
    // chto ni odin uzel ekspertov ne mozhet byt poschitan do togo, kak promahi prochitany.
    if (gf) ggml_build_forward_expand(gf, t);
    return t;
}

bool ExpertStore::prime(std::string* err) {
    if (!on()) return true;
    bulk_ = true;
    std::vector<int> ord(std::size_t(cfg_.n_expert));
    for (int il = 0; il < cfg_.n_layer; ++il) {
        Layer& L = layers_[std::size_t(il)];
        std::iota(ord.begin(), ord.end(), 0);
        // Ustojchivyj poriadok pri ravnyh schjotchikah: inache dva progona s odnoj i toj zhe
        // zatravkoj dali by raznye nabory i raznoe chislo promahov.
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
            if (L.score[std::size_t(a)] != L.score[std::size_t(b)]) {
                return L.score[std::size_t(a)] > L.score[std::size_t(b)];
            }
            return a < b;
        });
        std::fill(L.want.begin(), L.want.end(), uint8_t(0));
        for (int k = 0; k < cfg_.capacity; ++k) {
            const int id = ord[std::size_t(k)];
            L.want[std::size_t(id)] = 1;
            fill_slot(L, k, id);
            L.touched[std::size_t(k)] = 0;   // zalivka byla do pervogo tokena
        }
    }
    // Repak celym tenzorom, a ne po slotam: eto te zhe bajty (gruppa iz chetyrjoh strok
    // nikogda ne perehodit granicu eksperta, potomu chto n_ff i n_embd delyatsja na 4), no
    // odin vyzov na tenzor vmesto trjoh na kazhdyj slot.
    if (repacked_) {
        const auto t0 = Clock::now();
        for (int il = 0; il < cfg_.n_layer; ++il) {
            Layer& L = layers_[std::size_t(il)];
            ggml_tensor* three[3] = {L.gate, L.up, L.down};
            for (int k = 0; k < 3; ++k) {
                const ggml_type before = three[k]->type;
                iqk_repack_tensor(three[k]);
                if (three[k]->type == before) {
                    if (err) {
                        *err = std::string("repak tenzora ") + ggml_get_name(three[k]) +
                               " ne srabotal, hotja byl razreshjon";
                    }
                    bulk_ = false;
                    return false;
                }
            }
        }
        st_.ms_repack += ms_since(t0);
    }
    bulk_ = false;
    primed_ = true;
    return true;
}

void ExpertStore::refresh() {
    std::vector<int> ord(std::size_t(cfg_.n_expert));
    std::vector<int> need;
    std::vector<int> free_slots;
    for (int il = 0; il < cfg_.n_layer; ++il) {
        Layer& L = layers_[std::size_t(il)];
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
            if (L.score[std::size_t(a)] != L.score[std::size_t(b)]) {
                return L.score[std::size_t(a)] > L.score[std::size_t(b)];
            }
            return a < b;
        });
        std::fill(L.want.begin(), L.want.end(), uint8_t(0));
        need.clear();
        for (int k = 0; k < cfg_.capacity; ++k) {
            const int id = ord[std::size_t(k)];
            L.want[std::size_t(id)] = 1;
            // Slot u nego uzhe est - hot rezidentnyj, hot zapasnoj. Perekladyvat ne nado:
            // pul slotov odin, a want tolko zashchishchaet ot vytesnenija. Perekladyvanie
            // stoilo by povtornogo chtenija tam, gde chitat nechego.
            if (L.slot_of[std::size_t(id)] < 0) need.push_back(id);
        }
        if (need.empty()) continue;
        free_slots.clear();
        for (int s = 0; s < n_slots_; ++s) {
            const int32_t occ = L.id_of[std::size_t(s)];
            if (occ < 0 || !L.want[std::size_t(occ)]) free_slots.push_back(s);
        }
        const std::size_t m = std::min(need.size(), free_slots.size());
        for (std::size_t k = 0; k < m; ++k) {
            fill_slot(L, free_slots[k], need[k]);
        }
    }
    st_.refreshes++;
}

void ExpertStore::end_token() {
    if (!on()) return;
    // DRENAZH PREDZAGRUZKI mezhdu tokenami: dozhdatsja, poka I/O-potok dopishet vse sloty
    // etogo tokena. Bez etogo refresh ili pick_victim sledujushchego tokena mogli by vzjat
    // slot, v kotoryj potok eshchjo pishet - gonka za bajty. Chtenija bystrye (2,7 ms) na
    // fone tokena, poetomu obychno eto mgnovenno; esli net - eto chestnaja cena, i ona vidna
    // v ms_pref_wait ne popadaet (zdes zhdjom hvost, kotoryj v etom tokene ne ponadobilsja).
    drain_prefetch();
    st_.tokens++;
    token_++;
    if (cfg_.period > 0 && cfg_.capacity < cfg_.n_expert &&
        (st_.tokens % uint64_t(cfg_.period)) == 0) {
        refresh();
    }
}

int ExpertStore::n_resident(int il) const {
    if (!on()) return 0;
    const Layer& L = layers_[std::size_t(il)];
    int n = 0;
    for (int e = 0; e < cfg_.n_expert; ++e) {
        if (L.want[std::size_t(e)] && L.slot_of[std::size_t(e)] >= 0) ++n;
    }
    return n;
}

int ExpertStore::n_seen(int il) const {
    if (!on()) return 0;
    const Layer& L = layers_[std::size_t(il)];
    int n = 0;
    for (int e = 0; e < cfg_.n_expert; ++e) n += L.seen[std::size_t(e)] ? 1 : 0;
    return n;
}

bool ExpertStore::verify_slot(int il, int slot, std::string* why) const {
    if (!on()) { if (why) *why = "hranilishche vykljucheno"; return false; }
    if (il < 0 || il >= cfg_.n_layer || slot < 0 || slot >= n_slots_) {
        if (why) *why = "sloj ili slot vne predelov";
        return false;
    }
    const Layer& L = layers_[std::size_t(il)];
    const int32_t id = L.id_of[std::size_t(slot)];
    if (id < 0) { if (why) *why = "slot pust"; return false; }
    if (repacked_) {
        if (why) {
            *why = "repak vkljuchjon: bajty slota po postroeniju drugie, sravnenie s "
                   "modelju NEVOZMOZHNO";
        }
        return false;
    }
    const ggml_tensor* src[3] = {L.src_gate, L.src_up, L.src_down};
    const ggml_tensor* dst[3] = {L.gate, L.up, L.down};
    const char* nm[3] = {"gate", "up", "down"};
    for (int k = 0; k < 3; ++k) {
        const std::size_t nb2 = std::size_t(dst[k]->nb[2]);
        const char* a = (const char*)dst[k]->data + std::size_t(slot) * nb2;
        const char* b = (const char*)src[k]->data + std::size_t(id) * nb2;
        if (std::memcmp(a, b, nb2) != 0) {
            if (why) {
                *why = std::string("sloj ") + std::to_string(il) + " slot " +
                       std::to_string(slot) + " (ekspert " + std::to_string(id) + "): " +
                       nm[k] + " rashoditsja s modelju";
            }
            return false;
        }
    }
    return true;
}

// ================= PREDZAGRUZKA PO R1 (shag 5) =================

bool ExpertStore::open_pref_handle(std::string* err) {
#if !defined(_WIN32)
    if (err) *err = "predzagruzka realizovana tolko dlja Windows";
    return false;
#else
    if (!direct_) {
        if (err) *err = "predzagruzka trebuet prjamogo chtenija (gguf_path u hranilishcha)";
        return false;
    }
    HANDLE h = CreateFileA(cfg_.gguf_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        if (err) *err = "predzagruzka: vtoroj CreateFile ne otkryl fajl, kod " +
                        std::to_string((unsigned long)GetLastError());
        return false;
    }
    io_cap_pref_ = io_cap_;   // ta zhe geometrija, chto u osnovnogo io-bufera
    io_buf_pref_ = _aligned_malloc(io_cap_pref_, kSector);
    if (!io_buf_pref_) {
        CloseHandle(h);
        io_cap_pref_ = 0;
        if (err) *err = "predzagruzka: io-bufer ne vydelilsja";
        return false;
    }
    fh_pref_ = (void*)h;
    return true;
#endif
}

bool ExpertStore::load_prefetch(const char* path, int budget, bool sync, std::string* err) {
    if (!on()) {
        if (err) *err = "predzagruzka bez hranilishcha (--expert-store) ne imeet smysla";
        return false;
    }
    if (!direct_) {
        if (err) *err = "predzagruzka trebuet prjamogo chtenija fajla (gguf_path)";
        return false;
    }
    FILE* f = fopen(path, "rb");
    if (!f) {
        if (err) *err = std::string("fajl popravok R1 ne otkrylsja: ") + path;
        return false;
    }
    int32_t hd[4] = {0, 0, 0, 0};
    if (fread(hd, sizeof(int32_t), 4, f) != 4) {
        fclose(f);
        if (err) *err = "zagolovok popravok R1 ne prochitalsja";
        return false;
    }
    const int n_layer = hd[0], n_in = hd[1], n_out = hd[2], k = hd[3];
    if (n_layer != cfg_.n_layer || n_out != cfg_.n_expert || n_in != cfg_.n_expert + 1) {
        fclose(f);
        char b[192];
        snprintf(b, sizeof(b),
                 "popravki R1 na %d sloev, n_in %d, n_out %d, a model %d sloev, n_expert %d "
                 "(zhdu n_in = n_expert+1)", n_layer, n_in, n_out, cfg_.n_layer, cfg_.n_expert);
        if (err) *err = b;
        return false;
    }
    if (k < 0 || k >= cfg_.n_layer) {
        fclose(f);
        if (err) *err = "K iz zagolovka popravok vne predelov [0, n_layer)";
        return false;
    }
    // Telo f16 -> f32. ggml_fp16_to_fp32 est v ggml.h.
    const std::size_t n = std::size_t(n_layer) * std::size_t(n_in) * std::size_t(n_out);
    std::vector<uint16_t> raw(n);
    if (fread(raw.data(), sizeof(uint16_t), n, f) != n) {
        fclose(f);
        if (err) *err = "telo popravok R1 koroche zagolovka";
        return false;
    }
    fclose(f);
    pref_corr_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        pref_corr_[i] = ggml_fp16_to_fp32((ggml_fp16_t)raw[i]);
    }
    pref_n_in_   = n_in;
    pref_n_out_  = n_out;
    pref_k_      = k;
    pref_budget_ = budget > 0 ? budget : 16;
    if (pref_budget_ > cfg_.n_expert) pref_budget_ = cfg_.n_expert;
    pref_sync_   = sync;
    pref_sites_.resize(std::size_t(cfg_.n_layer));
    for (int il = 0; il < cfg_.n_layer; ++il) {
        pref_sites_[std::size_t(il)].self = this;
        pref_sites_[std::size_t(il)].tgt  = il;
    }
    pref_score_.assign(std::size_t(cfg_.n_expert), 0.0f);
    pref_idx_.assign(std::size_t(cfg_.n_expert), 0);
    pref_on_ = true;

    if (!pref_sync_) {
        // Vtoroj fajlovyj hendl+bufer i I/O-potok - tolko dlja asinhronnogo rezhima.
        if (!open_pref_handle(err)) {
            pref_on_ = false;
            return false;
        }
        stop_ = false;
        worker_ = std::thread(&ExpertStore::worker_main, this);
        worker_up_ = true;
    }
    return true;
}

void ExpertStore::worker_main() {
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !jobs_.empty(); });
            if (jobs_.empty()) {
                if (stop_) return;
                continue;
            }
            j = jobs_.front();
            jobs_.pop_front();
        }
        Layer& L = layers_[std::size_t(j.il)];
        // Bajty slota j.slot pishem BEZ zamka: slot zashchishchjon flagom filling==1, tak chto
        // ni pick_victim, ni do_map na etot slot ne pretendujut, poka my ne snimem flag.
        read_slot(L, j.slot, j.id, fh_pref_, io_buf_pref_, io_cap_pref_, &pref_reads_,
                  &pref_read_bytes_);
        {
            std::unique_lock<std::mutex> lk(mu_);
            L.filling[std::size_t(j.slot)] = 0;
            --inflight_;
        }
        cv_.notify_all();
    }
}

void ExpertStore::drain_prefetch() {
    if (!pref_on_ || pref_sync_) return;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return jobs_.empty() && inflight_ == 0; });
}

// Chtenie treh kuskov (gate/up/down) eksperta id v slot cherez ukazannye hendl+bufer, potom -
// esli vkljuchjon repak i ne idjot bulk - poslotovyj repak, kak v fill_slot.
void ExpertStore::read_slot(Layer& L, int slot, int id, void* fh, void* buf, std::size_t cap,
                            std::atomic<uint64_t>* reads, std::atomic<uint64_t>* rbytes) {
    const ggml_tensor* src[3] = {L.src_gate, L.src_up, L.src_down};
    ggml_tensor* dst[3] = {L.gate, L.up, L.down};
    const uint64_t off[3] = {L.off_gate, L.off_up, L.off_down};
    for (int k = 0; k < 3; ++k) {
        const std::size_t nb2 = std::size_t(dst[k]->nb[2]);
        char* d = (char*)dst[k]->data + std::size_t(slot) * nb2;
        const uint64_t foff = off[k] + std::size_t(id) * nb2;
        if (read_range(fh, buf, cap, foff, nb2, d)) {
            if (reads) reads->fetch_add(1, std::memory_order_relaxed);
            if (rbytes) rbytes->fetch_add(nb2, std::memory_order_relaxed);
        } else {
            static bool said = false;
            if (!said) {
                said = true;
                printf("predzagruzka: PRJAMOE CHTENIE ne udalos (sloj-tenzor %d, ekspert %d) - "
                       "slot NE zapolnen, ostajotsja sinhronnyj promah kak straxovka\n", k, id);
                fflush(stdout);
            }
        }
    }
    if (repacked_ && !bulk_) {
        for (int k = 0; k < 3; ++k) {
            ggml_tensor d = slice_descr(k == 2 ? L.src_down->type
                                               : (k == 0 ? L.src_gate->type : L.src_up->type),
                                        dst[k],
                                        (char*)dst[k]->data +
                                            std::size_t(slot) * std::size_t(dst[k]->nb[2]));
            iqk_repack_tensor(&d);
        }
    }
}

// Postavit eksperta id sloja tgt v zapasnoj slot. Vyzyvaetsja tolko s potoka grafa (vnutri
// pref_map_op), tak chto vsja rabota s metadannymi slotov (id_of/slot_of/touched/pref_tag) -
// bez zamka; s I/O-potokom delitsja tolko flag filling i ochered (pod mu_).
void ExpertStore::prefetch_expert(Layer& L, int tgt, int id) {
    if (id < 0 || id >= cfg_.n_expert) return;
    if (L.slot_of[std::size_t(id)] >= 0) return;   // uzhe v slote (rezident, zapasnoj ili v ocheredi)
    const int v = pick_victim(L);
    if (v < 0) return;                              // net zhertvy - ostanetsja sinhronnym promahom
    st_.pref_issued++;
    if (pref_sync_) {
        // Sinhronno pryamo v uzle: fill_slot sam vytesnjaet, chitaet bajty i stavit metadannye
        // (touched, id_of, slot_of). Eto NE prjachet chtenie - potok grafa stoit, - no daot
        // chestnoe A/B protiv asinhronnogo rezhima.
        fill_slot(L, v, id);
        L.pref_tag[std::size_t(v)] = token_;
        return;
    }
    // Asinhronno: metadannye stavim SEJCHAS (chtoby do_map videl popadanie), a bajty otdajom
    // I/O-potoku. filling==1 zapreshchaet komu-libo tronut slot, poka potok pishet.
    const int32_t old = L.id_of[std::size_t(v)];
    if (old >= 0) {
        L.slot_of[std::size_t(old)] = -1;
        st_.evictions++;
    }
    L.id_of[std::size_t(v)]    = id;
    L.slot_of[std::size_t(id)] = v;
    L.touched[std::size_t(v)]  = token_;
    L.pref_tag[std::size_t(v)] = token_;
    {
        std::unique_lock<std::mutex> lk(mu_);
        L.filling[std::size_t(v)] = 1;
        jobs_.push_back(Job{tgt, v, id});
        ++inflight_;
    }
    cv_.notify_one();
}

void ExpertStore::do_prefetch(int tgt, const ggml_tensor* pred_logits) {
    if (!pref_on_ || tgt < 0 || tgt >= cfg_.n_layer) return;
    if (tgt < pref_k_) return;   // dlja etih sloev popravka tozhdestvennaja, a istochnika net
    Layer& L = layers_[std::size_t(tgt)];
    const int E = cfg_.n_expert;
    // Vhod - syrye logity R0 (router_tgt * x_l), [n_expert] f32.
    const float* r0 = (const float*)pred_logits->data;
    const float* Wc = pref_corr_.data() + std::size_t(tgt) * std::size_t(pref_n_in_) *
                                              std::size_t(pref_n_out_);
    // score[e] = sum_i r0[i]*Wc[i][e] + 1*Wc[n_out][e]. Wc razlozhen strokami po n_in.
    const int nin = pref_n_in_, nout = pref_n_out_;
    float* sc = pref_score_.data();
    const float* bias = Wc + std::size_t(nin - 1) * std::size_t(nout);
    for (int e = 0; e < nout; ++e) sc[e] = bias[e];
    for (int i = 0; i < E; ++i) {
        const float ri = r0[i];
        if (ri == 0.0f) continue;
        const float* row = Wc + std::size_t(i) * std::size_t(nout);
        for (int e = 0; e < nout; ++e) sc[e] += ri * row[e];
    }
    // top-B po score.
    int B = pref_budget_;
    if (B > E) B = E;
    int* idx = pref_idx_.data();
    for (int e = 0; e < E; ++e) idx[e] = e;
    std::nth_element(idx, idx + B, idx + E,
                     [&](int a, int b) { return sc[a] > sc[b]; });
    st_.pref_predicted += (uint64_t)B;
    for (int t = 0; t < B; ++t) prefetch_expert(L, tgt, idx[t]);
}

void ExpertStore::pref_map_op(ggml_tensor* /*dst*/, const ggml_tensor* a, int ith,
                              int /*nth*/, void* ud) {
    if (ith != 0) return;
    PrefSite* s = (PrefSite*)ud;
    s->self->do_prefetch(s->tgt, a);
}

ggml_tensor* ExpertStore::prefetch_node(ggml_context* c, ggml_cgraph* gf, int tgt,
                                        ggml_tensor* pred_logits) {
    if (!pref_on_ || tgt < 0 || tgt >= cfg_.n_layer) return nullptr;
    ggml_tensor* t = ggml_map_custom1(c, pred_logits, pref_map_op, /*n_tasks=*/1,
                                      &pref_sites_[std::size_t(tgt)]);
    // Pobochnoe dejstvie bez potrebitelja - javno v graf, inache gallocr ne dast emu bufera
    // i pervoe chtenie upadjot (ta zhe lovushka, chto u slots_node/sel).
    if (gf) ggml_build_forward_expand(gf, t);
    return t;
}

}  // namespace memex
