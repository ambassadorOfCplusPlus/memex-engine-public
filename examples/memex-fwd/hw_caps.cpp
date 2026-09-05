#include "hw_caps.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "expert_store.hpp"   // ExpertStore::auto_capacity - the same C the store would pick

#if defined(_WIN32)
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <winioctl.h>
#endif

#ifdef MEMEX_FWD_VULKAN
#  include <vulkan/vulkan.h>
#endif

namespace memex {

namespace {

// Historical defaults of the box this project was tuned on. Used ONLY as a fallback when the
// corresponding channel was not measured, so the reference machine behaves identically with
// or without the probe, and every other machine gets the measured number instead.
constexpr std::size_t kDefaultBarBytes  = 256ull  * 1024ull * 1024ull;
constexpr std::size_t kDefaultVramBytes = 3824ull * 1024ull * 1024ull;

HardwareCaps  g_caps;
bool          g_measured = false;

// One-directional memcpy bandwidth, single thread. Not a decision input - printed only - so
// its job is to be a real number, not a precise one. Best of a few passes to shed scheduler
// noise; the buffer is large enough to defeat the L3 (this box has 6 MiB).
double measure_memcpy_gbps() {
    const std::size_t n = std::size_t(128) * 1024 * 1024;  // 128 MiB
    std::vector<char> a, b;
    try {
        a.assign(n, 1);
        b.assign(n, 0);
    } catch (...) {
        return 0.0;
    }
    // Touch once so both are resident before timing.
    std::memcpy(b.data(), a.data(), n);
    double best = 0.0;
    for (int pass = 0; pass < 4; ++pass) {
        const auto t0 = std::chrono::steady_clock::now();
        std::memcpy(b.data(), a.data(), n);
        const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (sec > 0.0) {
            const double gbps = double(n) / sec / 1e9;
            if (gbps > best) best = gbps;
        }
    }
    return best;
}

#if defined(_WIN32)
// True/false into *ssd when the query answers; return false when it could not be asked (no
// admin volume handle, unsupported device). Opens the volume read-only with no access rights,
// which needs no elevation for the property query.
bool disk_is_ssd(char letter, bool* ssd) {
    char path[8];
    std::snprintf(path, sizeof(path), "\\\\.\\%c:", letter);
    HANDLE h = CreateFileA(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    STORAGE_PROPERTY_QUERY q{};
    q.PropertyId = StorageDeviceSeekPenaltyProperty;
    q.QueryType  = PropertyStandardQuery;
    DEVICE_SEEK_PENALTY_DESCRIPTOR d{};
    DWORD ret = 0;
    const BOOL ok = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q),
                                    &d, sizeof(d), &ret, nullptr);
    CloseHandle(h);
    if (!ok || ret < sizeof(d)) return false;
    *ssd = (d.IncursSeekPenalty == FALSE);
    return true;
}
#endif

void probe_disk(HardwareCaps::Disk* out, char letter) {
    out->letter = letter;
#if defined(_WIN32)
    bool ssd = false;
    if (disk_is_ssd(letter, &ssd)) {
        out->known = true;
        out->ssd   = ssd;
        // ESTIMATE by type. From this project's own random-read medians (HANDOFF §1): a SSD
        // past its cache ~0.46 GB/s, a mechanical disk ~0.07 GB/s. Conservative and marked.
        out->est_gbps = ssd ? 0.46 : 0.07;
    }
#else
    (void)letter;
#endif
}

#ifdef MEMEX_FWD_VULKAN
// Enumerate the physical device's memory heaps to fill vram/bar. Independent of ggml and of
// the driver's lying "free" number: it reads heap SIZES and the type flags, both of which are
// static properties of the device, not runtime accounting.
void probe_vulkan(HardwareCaps* c) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "memex-fwd";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (n == 0) { vkDestroyInstance(inst, nullptr); return; }
    std::vector<VkPhysicalDevice> devs;
    devs.resize(std::size_t(n));
    if (vkEnumeratePhysicalDevices(inst, &n, devs.data()) != VK_SUCCESS || n == 0) {
        vkDestroyInstance(inst, nullptr);
        return;
    }
    devs.resize(std::size_t(n));

    // Select a DISCRETE GPU rather than blindly devs[0]. On a laptop with an iGPU + discrete
    // card, or when lavapipe/llvmpipe is present, devs[0] can be the integrated or software
    // device, which hands back system RAM as device-local - plausible VRAM/BAR for the wrong
    // card. Prefer the first discrete GPU; if there is none, fall back to devs[0] but record
    // the real device type so the printer can say so honestly.
    uint32_t chosen = 0;
    bool found_discrete = false;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties pp{};
        vkGetPhysicalDeviceProperties(devs[i], &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            chosen = i;
            found_discrete = true;
            break;
        }
    }
    {
        VkPhysicalDeviceProperties pp{};
        vkGetPhysicalDeviceProperties(devs[chosen], &pp);
        c->vram_discrete    = found_discrete;
        c->vram_device_type = (int)pp.deviceType;
        std::snprintf(c->gpu_name, sizeof(c->gpu_name), "%s", pp.deviceName);
    }

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(devs[chosen], &mp);

    // Largest device-local heap = VRAM.
    uint64_t vram = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        const VkMemoryHeap& h = mp.memoryHeaps[i];
        if (!(h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
        if (uint64_t(h.size) > vram) vram = uint64_t(h.size);
    }
    if (vram > 0) { c->vram_measured = true; c->vram_bytes = vram; }

    // BAR = smallest heap referenced by a type that is BOTH device-local and host-visible.
    // That is the window buffers can fall into. On a ReBAR card that heap IS the whole VRAM.
    uint64_t bar = 0;
    bool     bar_found = false;
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
        const VkMemoryType& mt = mp.memoryTypes[t];
        const bool dl = (mt.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        const bool hv = (mt.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        if (!(dl && hv)) continue;
        const uint64_t sz = uint64_t(mp.memoryHeaps[mt.heapIndex].size);
        if (!bar_found || sz < bar) { bar = sz; bar_found = true; }
    }
    if (bar_found) {
        c->bar_measured = true;
        c->bar_bytes    = bar;
        c->rebar        = (vram > 0 && bar >= vram);
    }
    vkDestroyInstance(inst, nullptr);
}
#endif  // MEMEX_FWD_VULKAN

}  // namespace

const HardwareCaps& measure_hardware() {
    if (g_measured) return g_caps;

    HardwareCaps c;
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) {
        c.ram_ok          = true;
        c.ram_total_bytes = s.ullTotalPhys;
        c.ram_avail_bytes = s.ullAvailPhys;
    }
#endif
    const double bw = measure_memcpy_gbps();
    if (bw > 0.0) { c.ram_bw_measured = true; c.ram_bw_gbps = bw; }

#ifdef MEMEX_FWD_VULKAN
    probe_vulkan(&c);
#endif

    probe_disk(&c.disk_c, 'C');
    probe_disk(&c.disk_d, 'D');

    g_caps     = c;
    g_measured = true;
    return g_caps;
}

const HardwareCaps& hw_caps() { return g_caps; }

std::size_t hw_bar_bytes() {
    // The number the placement code wants is the size of the SLOW window it must keep buffers
    // out of. A genuine small BAR (bar < vram) is exactly that, and is returned as measured -
    // 256 MiB on the reference card, 512 on a card with a bigger window, and so on. But on a
    // ReBAR card the host-visible+device-local heap IS the whole of VRAM: there is no slow
    // window at all, and handing the grouping logic the whole-VRAM figure would make it demand
    // buffers larger than VRAM and refuse. So the ReBAR case falls back to the historical
    // ceiling, which the old hardcode also produced there and which the grouping trivially
    // satisfies - i.e. "dodge nothing", the correct outcome when nothing is slow.
    if (g_measured && g_caps.bar_measured && g_caps.bar_bytes > 0 && !g_caps.rebar)
        return std::size_t(g_caps.bar_bytes);
    return kDefaultBarBytes;
}

std::size_t hw_vram_bytes() {
    if (g_measured && g_caps.vram_measured && g_caps.vram_bytes > 0)
        return std::size_t(g_caps.vram_bytes);
    return kDefaultVramBytes;
}

void print_hardware_caps(const HardwareCaps& c) {
    printf("\n=== HardwareCaps (samozamer pri starte) ===\n");
    if (c.ram_ok) {
        printf("  OZU:   %.1f GiB vsego, %.1f GiB svobodno            [izmereno: "
               "GlobalMemoryStatusEx]\n",
               double(c.ram_total_bytes) / 1073741824.0,
               double(c.ram_avail_bytes) / 1073741824.0);
    } else {
        printf("  OZU:   NE UZNANO (GlobalMemoryStatusEx ne otvetil)\n");
    }
    if (c.ram_bw_measured) {
        printf("  polosa OZU: %.1f GB/s (memcpy 128 MiB, odnopotochno)   [izmereno]\n",
               c.ram_bw_gbps);
    } else {
        printf("  polosa OZU: NE IZMERENA\n");
    }
    if (c.vram_measured) {
        printf("  VRAM:  %.0f MiB (krupnejshaja device-local kucha)     [izmereno: Vulkan, "
               "NE po \"svobodno\" drajvera]\n",
               double(c.vram_bytes) / 1048576.0);
        const char* kind = c.vram_discrete ? "diskretnaja GPU"
                         : c.vram_device_type == 1 ? "vstroennaja GPU (iGPU!)"
                         : c.vram_device_type == 4 ? "programmnyj rasterizator (CPU/lavapipe!)"
                         : "NE diskretnaja GPU!";
        printf("  GPU:   %s  [%s]%s\n",
               c.gpu_name[0] ? c.gpu_name : "?", kind,
               c.vram_discrete ? "" : "  <- VRAM/BAR mogut byt sistemnoj OZU, ne kartoj");
    } else {
        printf("  VRAM:  NE UZNANO (Vulkan ne skompilirovan ili net ustrojstva) -> "
               "default %.0f MiB [ocenka]\n",
               double(kDefaultVramBytes) / 1048576.0);
    }
    if (c.bar_measured) {
        printf("  BAR:   %.0f MiB (device-local + host-visible kucha)%s  [izmereno: Vulkan]\n",
               double(c.bar_bytes) / 1048576.0,
               c.rebar ? " = ves VRAM (ReBAR, okna net)" : "  <- uzkoe okno, obhodim");
    } else {
        printf("  BAR:   NE UZNANO -> default %.0f MiB [ocenka]\n",
               double(kDefaultBarBytes) / 1048576.0);
    }
    auto disk_line = [](const char* nm, const HardwareCaps::Disk& d) {
        if (!d.known) { printf("  disk %s: tip NE UZNAN\n", nm); return; }
        printf("  disk %s: %s   ~%.2f GB/s [tip izmeren, polosa - OCENKA po tipu]\n",
               nm, d.ssd ? "SSD" : "HDD (mehanicheskij)", d.est_gbps);
    };
    disk_line("C:", c.disk_c);
    disk_line("D:", c.disk_d);
    printf("===========================================\n");
}

Strategy choose_strategy(const HardwareCaps& caps, const ModelGeometry& geo,
                         uint64_t reserve_bytes, int spares, bool print) {
    Strategy st;
    const std::size_t vram = hw_vram_bytes();
    // Working room the layer graphs need on the card beyond the resident set; the static
    // module keeps a 224 MiB margin, mirror it here so the fit test matches what it does.
    const std::size_t work_margin = std::size_t(224) << 20;

    // 1. Head / static set on the card?
    st.head_on_card = geo.static_set_bytes > 0 &&
                      (geo.static_set_bytes + work_margin <= vram);

    // 2. Resident expert count C, from FREE RAM, by the store's own formula.
    st.resident_experts = ExpertStore::auto_capacity(
        caps.ram_ok ? caps.ram_avail_bytes : 0, reserve_bytes,
        geo.bytes_per_expert, geo.n_layer, spares, geo.n_expert);

    // 3. Do all experts fit RAM? If not, a warm SSD tier earns its keep.
    st.experts_bytes = uint64_t(geo.bytes_per_expert) * uint64_t(geo.n_expert) *
                       uint64_t(geo.n_layer);
    if (caps.ram_ok && st.experts_bytes > caps.ram_avail_bytes) {
        st.tiers_needed = true;
        // Suggest a warm cap from whichever local disk looks like SSD, leaving headroom.
        // Purely advisory (we do not know free bytes here); a conservative flat suggestion.
        st.warm_cap_gib = 0.0;  // engine's --warm-cap stays the authority; only flag when set
    }

    st.feasible = st.resident_experts > 0 || caps.ram_ok == false;

    if (print) {
        printf("\n=== vybor strategii (model %s) ===\n", geo.name && geo.name[0] ? geo.name : "?");
        printf("  VRAM %.0f MiB, staticheskij nabor %.0f MiB, zapas %.0f -> golova/statika %s\n",
               double(vram) / 1048576.0, double(geo.static_set_bytes) / 1048576.0,
               double(work_margin) / 1048576.0,
               st.head_on_card ? "NA KARTU" : "na CPU (ne vlezaet ili net geometrii)");
        if (caps.ram_ok) {
            printf("  OZU svobodno %.1f GiB, zapas %.1f GiB, ekspert %.3f MiB x %d sloev "
                   "-> C = %d iz %d rezidentno\n",
                   double(caps.ram_avail_bytes) / 1073741824.0,
                   double(reserve_bytes) / 1073741824.0,
                   double(geo.bytes_per_expert) / 1048576.0, geo.n_layer,
                   st.resident_experts, geo.n_expert);
        } else {
            printf("  OZU: svobodno ne uznano -> C ne schitaetsja avtomaticheski, nuzhen flag\n");
        }
        printf("  vse eksperty %.1f GiB %s OZU (%.1f GiB) -> urovni %s\n",
               double(st.experts_bytes) / 1073741824.0,
               st.tiers_needed ? ">" : "<=",
               caps.ram_ok ? double(caps.ram_avail_bytes) / 1073741824.0 : 0.0,
               st.tiers_needed ? "NUZHNY (tjoplyj uroven na SSD/HDD)" : "ne nuzhny");
        if (!st.feasible) {
            printf("  OTKAZ: v svobodnuju pamjat ne vlezaet ni odin rezidentnyj ekspert - "
                   "model tak ne obsluzhit, osvobodite OZU ili vozmite men'shij kvant\n");
        }
        printf("  (avto-vybor; ljuboj javnyj flag pereopredeljaet)\n");
        printf("==================================\n");
    }
    return st;
}

}  // namespace memex
