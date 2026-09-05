#pragma once

// HardwareCaps: the engine measures the machine it is on at start-up and reports what it
// found, so the placement code stops carrying the numbers of one particular card.
//
// WHY THIS EXISTS. The mechanism was tuned on an RX 6500 XT / 32 GB box and the numbers of
// that box had leaked into the code as literals: the 256 MiB BAR window, a 3824 MiB video
// budget, and assorted "does it fit" thresholds. On any other card those are simply wrong -
// a ReBAR card has no 256 MiB window, a 12 GB card has no 3824 MiB budget. This module
// measures each channel once and hands the placement code a value instead of a constant.
//
// THE RULE THIS FILE OBEYS (project rule 0): every channel says whether it was MEASURED or
// ESTIMATED. A driver that cannot answer, a query not compiled in, a disk whose type is
// unknown - each returns its `*_measured`/`known` flag false rather than a plausible zero,
// because a zero here silently disables the very check it feeds. Disk BANDWIDTH is the one
// thing deliberately not measured (a real probe costs seconds at start-up); its type is
// measured and the bandwidth is a conservative estimate BY type, marked as such.

#include <cstddef>
#include <cstdint>

namespace memex {

struct HardwareCaps {
    // ---- system RAM ----
    bool     ram_ok            = false;  // GlobalMemoryStatusEx answered
    uint64_t ram_total_bytes   = 0;
    uint64_t ram_avail_bytes   = 0;      // free at the moment of the probe
    bool     ram_bw_measured   = false;  // memcpy sweep ran
    double   ram_bw_gbps       = 0.0;    // one-directional copy bandwidth, best of a few passes

    // ---- video memory (Vulkan heaps; the DRIVER'S free is not trusted) ----
    bool     vram_measured     = false;  // heaps enumerated
    uint64_t vram_bytes        = 0;      // size of the LARGEST device-local heap
    // Which physical device was probed for VRAM/BAR. Blindly taking devs[0] measures an iGPU or
    // a software rasteriser (lavapipe/llvmpipe) on a laptop with two GPUs and reports system RAM
    // as device-local: plausible numbers for the wrong card. We select a discrete GPU when one
    // exists and record here what we actually measured, honestly.
    bool     vram_discrete     = false;  // selected device is VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    int      vram_device_type  = -1;     // raw VkPhysicalDeviceType of the probed device
    char     gpu_name[256]     = {0};    // deviceName of the probed device

    // ---- BAR window ----
    // The heap that is BOTH device-local AND host-visible. On a small-BAR discrete card that
    // is a separate ~256 MiB heap and a buffer that fits it is read at ~1/40 the bandwidth
    // while every report still calls it device-local; on a ReBAR card the whole of VRAM is
    // host-visible and there is no penalty (bar_bytes >= vram_bytes, `rebar` set).
    bool     bar_measured      = false;
    uint64_t bar_bytes         = 0;
    bool     rebar             = false;  // bar_bytes >= vram_bytes: no slow window to dodge

    // ---- disks (type MEASURED, bandwidth ESTIMATED by type) ----
    struct Disk {
        bool   known     = false;   // seek-penalty query answered
        bool   ssd       = false;   // no seek penalty => solid state
        double est_gbps  = 0.0;     // conservative, BY TYPE, never measured
        char   letter    = 0;
    };
    Disk disk_c;
    Disk disk_d;
};

// Measure everything once and cache it; repeated calls return the cached result. The RAM
// bandwidth sweep runs only on the first call.
const HardwareCaps& measure_hardware();
// The cached result without forcing a measurement (all-false if never measured).
const HardwareCaps& hw_caps();
// Human-readable dump, every line tagged izmereno / ocenka.
void print_hardware_caps(const HardwareCaps& c);

// Values that used to be hardcoded, now served from the measurement. They return the MEASURED
// figure once measure_hardware() has run, and otherwise the reference card's historical
// default - so behaviour on the original box is identical whether or not the probe ran, and
// correct on other boxes once it has.
std::size_t hw_bar_bytes();    // default 256 MiB
std::size_t hw_vram_bytes();   // default 3824 MiB

// ---- strategy ----------------------------------------------------------------------------

// What the model asks of the machine, in the terms the decision needs. Filled by the harness
// from the GGUF geometry it already read.
struct ModelGeometry {
    int         n_layer         = 0;
    int         n_expert        = 0;   // per layer
    int         n_expert_used   = 0;   // top-k
    std::size_t bytes_per_expert = 0;  // gate+up+down of one expert, one layer
    std::size_t static_set_bytes = 0;  // head + resident attention/router that would go on card
    std::size_t head_bytes      = 0;   // just the output head
    uint64_t    file_bytes      = 0;   // whole model on disk
    const char* name            = "";
};

// The decision, and every field is accompanied in print by the number that produced it.
struct Strategy {
    bool        feasible        = true;   // false => model cannot be served on this machine
    bool        head_on_card    = false;  // head fits VRAM minus working room
    int         resident_experts = 0;     // C: how many experts per layer stay in RAM
    bool        tiers_needed    = false;  // experts do not fit RAM => a warm SSD tier helps
    double      warm_cap_gib    = 0.0;    // suggested warm-tier cap from free SSD (estimate)
    uint64_t    experts_bytes   = 0;      // all experts, all layers
};

// Decide (and, if print, explain) the strategy from measured hardware and model geometry.
// This is ADVISORY: it prints the choice the engine would make and why. The engine's own
// auto-paths (--expert-store-auto sizing C from free RAM, repack sizing from free RAM) use
// the same free-RAM arithmetic; an explicit flag always overrides. `reserve_bytes` and
// `spares` mirror the expert-store defaults so the printed C equals the one the store picks.
Strategy choose_strategy(const HardwareCaps& caps, const ModelGeometry& geo,
                         uint64_t reserve_bytes, int spares, bool print);

}  // namespace memex
