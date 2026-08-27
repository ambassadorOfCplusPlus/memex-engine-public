#include "gpu_experts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "ggml-alloc.h"
#include "ggml-vulkan.h"
#include "resident_set.hpp"

#if defined(_WIN32)
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#ifdef MEMEX_FWD_VULKAN
#  include <vulkan/vulkan.h>
#endif

namespace memex {

namespace {

constexpr std::size_t kBarHeapCeiling = 256u * 1024u * 1024u;

// Free bytes on the largest device-local heap, and the size of the smallest one - which on a
// discrete card is the BAR window and is the number that decides how the buffers have to be
// grouped. Returns false when the query is not compiled in, and the caller then falls back to
// a fixed budget and says so rather than guessing quietly.
bool device_heap_facts(std::size_t* free_large, std::size_t* smallest_device_heap) {
#ifdef MEMEX_FWD_VULKAN
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "memex-fwd";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return false;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (n == 0) { vkDestroyInstance(inst, nullptr); return false; }
    std::vector<VkPhysicalDevice> devs;
    devs.resize(std::size_t(n));
    vkEnumeratePhysicalDevices(inst, &n, devs.data());

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(devs[0], nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts;
    exts.resize(std::size_t(ne));
    vkEnumerateDeviceExtensionProperties(devs[0], nullptr, &ne, exts.data());
    bool has_budget = false;
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            has_budget = true;
        }
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 mp2{};
    mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mp2.pNext = has_budget ? (void*)&budget : nullptr;
    vkGetPhysicalDeviceMemoryProperties2(devs[0], &mp2);

    std::size_t best = 0, smallest = 0;
    for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; ++i) {
        const VkMemoryHeap& h = mp2.memoryProperties.memoryHeaps[i];
        if (!(h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
        const std::size_t avail = has_budget
            ? std::size_t(budget.heapBudget[i] > budget.heapUsage[i]
                          ? budget.heapBudget[i] - budget.heapUsage[i] : 0)
            : std::size_t(h.size);
        if (std::size_t(h.size) > kBarHeapCeiling && avail > best) best = avail;
        if (smallest == 0 || std::size_t(h.size) < smallest) smallest = std::size_t(h.size);
    }
    vkDestroyInstance(inst, nullptr);
    if (best == 0) return false;
    *free_large = best;
    *smallest_device_heap = smallest;
    return true;
#else
    (void)free_large; (void)smallest_device_heap;
    return false;
#endif
}

// One contiguous i32 read out of a tensor that may be a strided view.
inline int32_t id_at(const ggml_tensor* t, int64_t i0) {
    return *(const int32_t*)((const char*)t->data + std::size_t(i0) * t->nb[0]);
}

}  // namespace

void GpuExperts::print_heaps(const char* when) {
#ifdef MEMEX_FWD_VULKAN
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
    vkEnumeratePhysicalDevices(inst, &n, devs.data());

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(devs[0], nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts;
    exts.resize(std::size_t(ne));
    vkEnumerateDeviceExtensionProperties(devs[0], nullptr, &ne, exts.data());
    bool has_budget = false;
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            has_budget = true;
        }
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 mp2{};
    mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mp2.pNext = has_budget ? (void*)&budget : nullptr;
    vkGetPhysicalDeviceMemoryProperties2(devs[0], &mp2);

    printf("  --- кучи памяти устройства (%s) ---\n", when);
    for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; ++i) {
        const VkMemoryHeap& h = mp2.memoryProperties.memoryHeaps[i];
        const bool dl = (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        printf("  куча %u  %-13s размер %8.2f МиБ", i, dl ? "DEVICE_LOCAL" : "host",
               double(h.size) / 1048576.0);
        if (has_budget) {
            printf("   бюджет %8.2f МиБ   занято %8.2f МиБ",
                   double(budget.heapBudget[i]) / 1048576.0,
                   double(budget.heapUsage[i]) / 1048576.0);
        }
        printf("\n");
    }
    // The memory TYPES, once. ggml chooses one with find_properties (ggml-vulkan.cpp:1580),
    // which walks this list in order and takes the first type that has every requested flag
    // AND whose heap is at least as large as the buffer. ggml_vk_create_buffer_device asks
    // first for DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT and falls back to DEVICE_LOCAL
    // alone (ggml-vulkan.cpp:1698). Printing the table is what turns "the buffer is
    // device-local" - which every report says, including for memory the driver put behind a
    // BAR window and backed with system RAM - into a statement about which heap it can
    // actually be on.
    static bool types_shown = false;
    if (!types_shown) {
        types_shown = true;
        printf("  --- типы памяти (ggml берёт первый подходящий, чья куча не меньше буфера) ---\n");
        for (uint32_t i = 0; i < mp2.memoryProperties.memoryTypeCount; ++i) {
            const VkMemoryType& t = mp2.memoryProperties.memoryTypes[i];
            printf("  тип %2u -> куча %u  %s%s%s%s\n", i, t.heapIndex,
                   (t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
                   (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "",
                   (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "",
                   (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED " : "");
        }
    }
    vkDestroyInstance(inst, nullptr);
#else
    (void)when;
#endif
}

void GpuExperts::print_queues() {
#ifdef MEMEX_FWD_VULKAN
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "memex-fwd";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        printf("  очереди устройства: экземпляр Vulkan не создался — НЕ ИЗМЕРЕНО\n");
        return;
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (n == 0) {
        printf("  очереди устройства: ни одного физического устройства — НЕ ИЗМЕРЕНО\n");
        vkDestroyInstance(inst, nullptr);
        return;
    }
    // Not `std::vector<T> v(std::size_t(n))`: that is the most vexing parse - the compiler
    // reads it as a function declaration, and every later use then fails with an unrelated
    // message about indexing a non-array.
    std::vector<VkPhysicalDevice> devs;
    devs.resize(std::size_t(n));
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(devs[0], &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfp;
    qfp.resize(std::size_t(nq));
    vkGetPhysicalDeviceQueueFamilyProperties(devs[0], &nq, qfp.data());

    printf("  семейств очередей: %u\n", unsigned(nq));
    for (uint32_t i = 0; i < nq; ++i) {
        const VkQueueFlags f = qfp[std::size_t(i)].queueFlags;
        char fl[128];
        snprintf(fl, sizeof(fl), "%s%s%s%s%s",
                 (f & VK_QUEUE_GRAPHICS_BIT)       ? "GRAPHICS "  : "",
                 (f & VK_QUEUE_COMPUTE_BIT)        ? "COMPUTE "   : "",
                 (f & VK_QUEUE_TRANSFER_BIT)       ? "TRANSFER "  : "",
                 (f & VK_QUEUE_SPARSE_BINDING_BIT) ? "SPARSE "    : "",
                 (f & VK_QUEUE_PROTECTED_BIT)      ? "PROTECTED " : "");
        printf("    семейство %u: очередей %u, флаги 0x%02x = %s| метки времени %u бит, "
               "гранулярность передачи %ux%ux%u%s\n",
               unsigned(i), unsigned(qfp[std::size_t(i)].queueCount), unsigned(f), fl,
               unsigned(qfp[std::size_t(i)].timestampValidBits),
               unsigned(qfp[std::size_t(i)].minImageTransferGranularity.width),
               unsigned(qfp[std::size_t(i)].minImageTransferGranularity.height),
               unsigned(qfp[std::size_t(i)].minImageTransferGranularity.depth),
               ((f & VK_QUEUE_TRANSFER_BIT) &&
                !(f & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)))
                   ? "  <-- ТОЛЬКО ПЕРЕДАЧА (DMA)" : "");
    }

    // ggml's own selection rule, reproduced rather than guessed at
    // (ggml_vk_find_queue_family_index, ggml-vulkan.cpp:1585). Written as the same four
    // fallbacks in the same order, because the answer that matters is not "does a DMA family
    // exist" but "does OUR upload get submitted on it", and that is decided by this rule.
    auto find_family = [&](VkQueueFlags required, VkQueueFlags avoid, int compute_index,
                           uint32_t min_queues) -> int {
        for (uint32_t i = 0; i < nq; ++i) {
            if (qfp[i].queueCount >= min_queues && (compute_index < 0 || int(i) != compute_index)
                && (qfp[i].queueFlags & required) && !(qfp[i].queueFlags & avoid)) return int(i);
        }
        for (uint32_t i = 0; i < nq; ++i) {
            if (qfp[i].queueCount >= min_queues && (compute_index < 0 || int(i) != compute_index)
                && (qfp[i].queueFlags & required)) return int(i);
        }
        for (uint32_t i = 0; i < nq; ++i) {
            if (qfp[i].queueCount >= min_queues && (qfp[i].queueFlags & required)) return int(i);
        }
        for (uint32_t i = 0; i < nq; ++i) {
            if (qfp[i].queueFlags & required) return int(i);
        }
        return compute_index;
    };
    const int cq = find_family(VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT, -1, 1);
    const int tq = find_family(VK_QUEUE_TRANSFER_BIT,
                               VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT, cq, 1);
    const bool separate = (cq >= 0 && tq >= 0 && cq != tq);
    printf("    выбор ggml: счёт — семейство %d, передача — семейство %d — %s\n", cq, tq,
           separate
               ? "РАЗНЫЕ ОЧЕРЕДИ: подкачка идёт на своей, а не в порядке с диспатчами"
               : "ОДНА И ТА ЖЕ: transfer_queue.copyFrom(compute_queue), подкачка стоит в "
                 "той же очереди, что и счёт");
    vkDestroyInstance(inst, nullptr);
#else
    printf("  очереди устройства: собрано без заголовков Vulkan — НЕ ИЗМЕРЕНО\n");
#endif
}

void GpuExperts::print_placement(const char* what, std::size_t bytes) {
#ifdef MEMEX_FWD_VULKAN
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
    vkEnumeratePhysicalDevices(inst, &n, devs.data());

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(devs[0], &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfp;
    qfp.resize(std::size_t(nq));
    vkGetPhysicalDeviceQueueFamilyProperties(devs[0], &nq, qfp.data());
    uint32_t qf = 0;
    for (uint32_t i = 0; i < nq; ++i) {
        if (qfp[std::size_t(i)].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(devs[0], &dci, nullptr, &dev) != VK_SUCCESS) {
        vkDestroyInstance(inst, nullptr);
        return;
    }

    // ggml_vk_create_buffer's own usage flags (ggml-vulkan.cpp:1610-1617).
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(dev, &bci, nullptr, &buf) == VK_SUCCESS) {
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(dev, buf, &mr);
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(devs[0], &mp);

        auto pick = [&](VkMemoryPropertyFlags want) -> int {
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
                if (!(mr.memoryTypeBits & (1u << i))) continue;
                if ((mp.memoryTypes[i].propertyFlags & want) != want) continue;
                if (mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size < mr.size) continue;
                return int(i);
            }
            return -1;
        };
        const VkMemoryPropertyFlags rebar = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        int t = pick(rebar);
        const char* how = "DEVICE_LOCAL|HOST_VISIBLE (BAR)";
        if (t < 0) { t = pick(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); how = "DEVICE_LOCAL"; }
        if (t < 0) {
            printf("  %-34s %9.2f МиБ -> НЕ РАЗМЕЩАЕТСЯ\n", what, double(bytes) / 1048576.0);
        } else {
            const uint32_t hi = mp.memoryTypes[std::size_t(t)].heapIndex;
            printf("  %-34s %9.2f МиБ -> тип %u, куча %u (%.0f МиБ), %s%s\n", what,
                   double(bytes) / 1048576.0, unsigned(t), hi,
                   double(mp.memoryHeaps[hi].size) / 1048576.0, how,
                   (mp.memoryTypes[std::size_t(t)].propertyFlags &
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? ", запись хоста = memcpy"
                                                         : ", запись хоста = staging+fence");
        }
        vkDestroyBuffer(dev, buf, nullptr);
    }
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
#else
    (void)what; (void)bytes;
#endif
}

GpuExperts::~GpuExperts() { shutdown(); }

// ---------------------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------------------

void GpuExperts::free_weights() {
    for (ggml_backend_buffer_t b : bufs_) {
        if (b) ggml_backend_buffer_free(b);
    }
    bufs_.clear();
    for (ggml_context* c : ctxs_w_) {
        if (c) ggml_free(c);
    }
    ctxs_w_.clear();
    bufs_info_.clear();
    up_.clear(); gate_.clear(); down_.clear();
    vram_bytes_ = 0;
}

bool GpuExperts::alloc_weights(std::string* err) {
    const int nl  = cfg_.n_layers;
    const int cap = cfg_.capacity;
    const std::size_t per_layer = std::size_t(cap) * bpe_;
    const std::size_t max_buf   = ggml_backend_buft_get_max_size(buft_);

    if (per_layer == 0) { *err = "нулевая ёмкость"; return false; }

    // Layers per buffer. Two bounds, and only one of them is ggml's: no buffer may exceed the
    // backend's own ceiling, and no buffer may be small enough to fit the BAR heap - because
    // one that fits will be put there, and once that heap is committed the driver backs the
    // rest with system memory at a fortieth of the bandwidth while still calling it
    // device-local. So the groups are made as equal as possible and then the smallest one is
    // checked against 256 MiB rather than assumed to clear it.
    int per_buf = int(max_buf / per_layer);
    if (per_buf < 1) { *err = "слой не помещается в один буфер бэкенда"; return false; }
    if (per_buf > nl) per_buf = nl;
    const int n_groups = (nl + per_buf - 1) / per_buf;
    // Remainder spread one layer at a time rather than piled on the last group: an eight-way
    // split of forty-eight layers into sevens leaves a group of one, and a group of one layer
    // is a 57 MiB buffer, which is precisely the buffer that gets put in the BAR heap.
    const int base_layers = nl / n_groups;
    const int extra       = nl % n_groups;

    const std::size_t smallest_group_bytes = std::size_t(base_layers) * per_layer;
    if (smallest_group_bytes <= kBarHeapCeiling) {
        char buf[320];
        snprintf(buf, sizeof(buf),
                 "разбиение даёт буфер %.1f МиБ (<= 256 МиБ) — он сядет в BAR-кучу и "
                 "чтения шейдера упадут в сорок раз; уменьшите число групп или ёмкость",
                 double(smallest_group_bytes) / 1048576.0);
        *err = buf;
        return false;
    }

    up_.assign(std::size_t(nl), nullptr);
    gate_.assign(std::size_t(nl), nullptr);
    down_.assign(std::size_t(nl), nullptr);

    int first = 0;
    for (int g = 0; g < n_groups; ++g) {
        const int cnt = base_layers + (g < extra ? 1 : 0);
        if (cnt <= 0) break;
        ggml_init_params ip = {ggml_tensor_overhead() * std::size_t(cnt) * 3 + 4096, nullptr,
                               true};
        ggml_context* c = ggml_init(ip);
        if (!c) { *err = "ggml_init для группы весов не удался"; free_weights(); return false; }
        ctxs_w_.push_back(c);
        for (int i = 0; i < cnt; ++i) {
            const int il = first + i;
            // The STORED descriptors, not the RAM tensors': with repacking on, the RAM type is
            // iq4_xs_r8 and allocating that here is what the Vulkan backend rightly refuses.
            const PlainSrc su = desc(il, 0);
            const PlainSrc sg = desc(il, 1);
            const PlainSrc sd = desc(il, 2);
            up_[std::size_t(il)] = ggml_new_tensor_3d(c, su.type, su.ne0, su.ne1, cap);
            gate_[std::size_t(il)] = ggml_new_tensor_3d(c, sg.type, sg.ne0, sg.ne1, cap);
            down_[std::size_t(il)] = ggml_new_tensor_3d(c, sd.type, sd.ne0, sd.ne1, cap);
            char nm[64];
            snprintf(nm, sizeof(nm), "vk.up.%d", il);   ggml_set_name(up_[std::size_t(il)], nm);
            snprintf(nm, sizeof(nm), "vk.gate.%d", il); ggml_set_name(gate_[std::size_t(il)], nm);
            snprintf(nm, sizeof(nm), "vk.down.%d", il); ggml_set_name(down_[std::size_t(il)], nm);
            // The per-expert slice has to be the same number of bytes on both sides or the
            // upload writes a different expert than the one the slot map says is there, with
            // no shape error anywhere.
            if (up_[std::size_t(il)]->nb[2] != su.slab ||
                gate_[std::size_t(il)]->nb[2] != sg.slab ||
                down_[std::size_t(il)]->nb[2] != sd.slab) {
                *err = "шаг эксперта в видеопамяти не совпал с шагом в модели";
                free_weights();
                return false;
            }
        }
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, buft_);
        if (!b) {
            char buf[200];
            snprintf(buf, sizeof(buf),
                     "не выделилось %.1f МиБ на слои %d..%d",
                     double(std::size_t(cnt) * per_layer) / 1048576.0, first, first + cnt - 1);
            *err = buf;
            free_weights();
            return false;
        }
        const std::size_t sz = ggml_backend_buffer_get_size(b);
        bufs_.push_back(b);
        bufs_info_.push_back({first, cnt, sz, sz > kBarHeapCeiling});
        vram_bytes_ += sz;
        first += cnt;
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// The pre-repack source
//
// Load-time repacking (iqk_repack_tensor) rewrites IQ4_XS into the interleaved iq4_xs_r8 in
// place. That is what makes the CPU half fast, and it is exactly what the Vulkan backend
// cannot read: it implements no _R* type for any op. One in-memory copy cannot serve both,
// and choosing per expert is not open either, because which side owns a given expert changes
// every time the resident set is refreshed.
//
// What IS open: the GGUF on disk still holds the plain bytes. A promotion already reads 2.5 MB
// per expert, so taking it from the file instead of from the repacked tensor costs one seek
// and one read of pages the loader has already walked - and buys back the whole +34% that
// repacking is worth on the CPU side.
//
// Only the descriptors are kept, never the metadata context: shape, stored type, stored
// per-expert stride and absolute file offset are all the uploader needs.
// ---------------------------------------------------------------------------------------
bool GpuExperts::open_plain_source(std::string* err) {
    ggml_context*    meta = nullptr;
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx      = &meta;
    gguf_context* g = gguf_init_from_file(cfg_.model_path.c_str(), gp);
    if (g == nullptr || meta == nullptr) {
        *err = "не удалось открыть GGUF повторно для чтения неперепакованных байтов: " +
               cfg_.model_path;
        if (g) gguf_free(g);
        return false;
    }
    const size_t data_off = gguf_get_data_offset(g);

    plain_.assign(std::size_t(cfg_.n_layers) * 3u, PlainSrc());
    bool ok = true;
    for (int il = 0; il < cfg_.n_layers && ok; ++il) {
        ggml_tensor* ram[3] = {src_up_[std::size_t(il)], src_gate_[std::size_t(il)],
                               src_down_[std::size_t(il)]};
        for (int kind = 0; kind < 3; ++kind) {
            // The name is the join between the two views of the same weight. Repacking
            // rewrites bytes and the type, never the name, so the RAM tensor's name is still
            // the GGUF's name for it.
            const char* nm = ggml_get_name(ram[kind]);
            const int   tid = (nm && nm[0]) ? gguf_find_tensor(g, nm) : -1;
            if (tid < 0) {
                *err = std::string("тензор '") + (nm ? nm : "(без имени)") +
                       "' не найден в GGUF — неперепакованные байты взять неоткуда";
                ok = false;
                break;
            }
            ggml_tensor* mt = ggml_get_tensor(meta, nm);
            if (mt == nullptr) {
                *err = std::string("описатель '") + nm + "' отсутствует в метаданных GGUF";
                ok = false;
                break;
            }
            if (mt->ne[2] != cfg_.n_experts) {
                *err = std::string("'") + nm + "' в файле не сложен по третьей оси";
                ok = false;
                break;
            }
            PlainSrc p;
            p.type = mt->type;
            p.ne0  = mt->ne[0];
            p.ne1  = mt->ne[1];
            p.slab = std::size_t(mt->nb[2]);
            p.off  = (long long)(data_off + gguf_get_tensor_offset(g, tid));
            plain_[std::size_t(il) * 3u + std::size_t(kind)] = p;
        }
    }
    gguf_free(g);
    ggml_free(meta);
    if (!ok) { plain_.clear(); return false; }

    gguf_f_ = fopen(cfg_.model_path.c_str(), "rb");
    if (gguf_f_ == nullptr) {
        *err = "GGUF не открылся на чтение: " + cfg_.model_path;
        plain_.clear();
        return false;
    }
    plain_on_ = true;
    return true;
}

const char* GpuExperts::uploaded_type_name() const {
    if (up_.empty() || up_[0] == nullptr) return "(нет)";
    return ggml_type_name(up_[0]->type);
}

const char* GpuExperts::host_type_name() const {
    if (src_up_.empty() || src_up_[0] == nullptr) return "(нет)";
    return ggml_type_name(src_up_[0]->type);
}

void GpuExperts::close_plain_source() {
    if (gguf_f_) { fclose(gguf_f_); gguf_f_ = nullptr; }
    plain_.clear();
    plain_on_ = false;
}

GpuExperts::PlainSrc GpuExperts::desc(int il, int kind) const {
    if (plain_on_) return plain_[std::size_t(il) * 3u + std::size_t(kind)];
    ggml_tensor* t = kind == 0 ? src_up_[std::size_t(il)]
                  : kind == 1 ? src_gate_[std::size_t(il)]
                              : src_down_[std::size_t(il)];
    PlainSrc p;
    p.type = t->type;
    p.ne0  = t->ne[0];
    p.ne1  = t->ne[1];
    p.slab = std::size_t(t->nb[2]);
    p.off  = -1;   // in RAM, not in the file
    return p;
}

bool GpuExperts::read_plain(int il, int kind, int expert, char* dst, std::string* err) {
    const PlainSrc p = desc(il, kind);
    if (!plain_on_) {
        ggml_tensor* t = kind == 0 ? src_up_[std::size_t(il)]
                      : kind == 1 ? src_gate_[std::size_t(il)]
                                  : src_down_[std::size_t(il)];
        std::memcpy(dst, (const char*)t->data + std::size_t(expert) * p.slab, p.slab);
        return true;
    }
    const long long off = p.off + (long long)expert * (long long)p.slab;
    std::lock_guard<std::mutex> lk(io_mu_);
    // 64-bit seek is mandatory: the file is ~17 GB and a 32-bit offset would silently wrap,
    // promoting the wrong expert instead of failing.
#ifdef _WIN32
    const int seek_rc = _fseeki64(gguf_f_, (__int64)off, SEEK_SET);
#else
    const int seek_rc = fseeko(gguf_f_, (off_t)off, SEEK_SET);
#endif
    if (seek_rc != 0) {
        if (err) *err = "сдвиг в GGUF не удался";
        return false;
    }
    const std::size_t got = fread(dst, 1, p.slab, gguf_f_);
    if (got != p.slab) {
        if (err) *err = "короткое чтение слэба эксперта из GGUF";
        return false;
    }
    return true;
}

bool GpuExperts::init(const GpuExpertsConfig& cfg, ggml_tensor* const* up,
                      ggml_tensor* const* gate, ggml_tensor* const* down, std::string* err) {
    cfg_ = cfg;
    if (cfg_.n_layers <= 0 || cfg_.n_experts <= 0 || cfg_.n_used <= 0 || cfg_.n_embd <= 0) {
        *err = "бессмысленная конфигурация";
        return false;
    }
    src_up_.assign(up, up + cfg_.n_layers);
    src_gate_.assign(gate, gate + cfg_.n_layers);
    src_down_.assign(down, down + cfg_.n_layers);
    for (int il = 0; il < cfg_.n_layers; ++il) {
        ggml_tensor* t[3] = {src_up_[std::size_t(il)], src_gate_[std::size_t(il)],
                             src_down_[std::size_t(il)]};
        for (ggml_tensor* x : t) {
            if (!x) {
                *err = "тензор экспертов отсутствует";
                return false;
            }
        }
    }
    // Before anything is sized or allocated: if a GGUF path was given, the stored descriptors
    // become the authority on type, shape and stride. Everything below - the device tensors,
    // the stride equality check, bytes-per-expert - then talks about the PLAIN layout, and the
    // RAM tensors are free to be repacked.
    if (!cfg_.model_path.empty()) {
        if (!open_plain_source(err)) return false;
    }
    for (int il = 0; il < cfg_.n_layers; ++il) {
        ggml_tensor* t[3] = {src_up_[std::size_t(il)], src_gate_[std::size_t(il)],
                             src_down_[std::size_t(il)]};
        for (ggml_tensor* x : t) {
            // Only needed when promotions read RAM. With the file as the source the RAM copy
            // may be repacked, and after repacking it is no longer an mmap either.
            if (!plain_on_ &&
                (!x->data || !x->buffer || !ggml_backend_buffer_is_host(x->buffer))) {
                *err = "тензоры экспертов недоступны хосту — промоушен читать неоткуда";
                close_plain_source();
                return false;
            }
            if (x->ne[2] != cfg_.n_experts) {
                *err = "тензор экспертов не сложен по третьей оси";
                close_plain_source();
                return false;
            }
        }
    }
    bpe_ = desc(0, 0).slab + desc(0, 1).slab + desc(0, 2).slab;

    be_ = ggml_backend_vk_init(0);
    if (!be_) { *err = "ggml_backend_vk_init(0) не удался"; return false; }
    buft_ = ggml_backend_vk_buffer_type(0);
    if (!buft_) { *err = "ggml_backend_vk_buffer_type(0) не удался"; shutdown(); return false; }
    {
        char d[256] = {0};
        ggml_backend_vk_get_device_description(0, d, sizeof(d));
        dev_name_ = d;
    }

    // The capacity, if it was not asked for outright. The budget is the free bytes of the
    // large device-local heap - not of the BAR heap, which is device-local too and is exactly
    // the trap - minus the headroom ggml's own scratch and the desktop need.
    std::size_t heap_free = 0, smallest_heap = 0;
    const bool have_facts = device_heap_facts(&heap_free, &smallest_heap);
    if (!have_facts) {
        heap_free = 3ull * 1024 * 1024 * 1024;
        printf("резидентные эксперты на GPU: VK_EXT_memory_budget недоступен в этой сборке, "
               "бюджет принят равным %.2f ГиБ\n", double(heap_free) / 1073741824.0);
    }
    const std::size_t budget = heap_free > cfg_.reserve ? heap_free - cfg_.reserve : 0;
    const int fits = int(budget / (std::size_t(cfg_.n_layers) * bpe_));
    if (cfg_.capacity <= 0) {
        cfg_.capacity = std::min(fits, cfg_.n_experts);
    } else if (cfg_.capacity > fits) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "запрошено %d резидентных экспертов на слой, а в свободные %.2f ГиБ "
                 "видеопамяти помещается %d", cfg_.capacity,
                 double(heap_free) / 1073741824.0, fits);
        *err = buf;
        shutdown();
        return false;
    }
    if (cfg_.capacity <= 0) {
        *err = "в свободную видеопамять не помещается ни одного эксперта на слой";
        shutdown();
        return false;
    }

    print_heaps("до размещения резидентных экспертов");
    // Printed here, before any allocation, because it is a property of the device and not of
    // this run: whether the prefetch CAN be asynchronous rather than take turns with compute.
    print_queues();
    // A retry rather than a refusal: heapBudget is the driver's opinion at one instant and
    // the desktop can take a few tens of megabytes between the query and the allocation.
    std::string aerr;
    while (cfg_.capacity > 0) {
        if (alloc_weights(&aerr)) break;
        free_weights();
        const int next = cfg_.capacity - 1;
        if (next <= 0) { *err = aerr; shutdown(); return false; }
        printf("резидентные эксперты на GPU: %s — пробуем %d на слой\n", aerr.c_str(), next);
        cfg_.capacity = next;
    }
    print_heaps("после размещения резидентных экспертов");

    // ---- the compute graphs, one per compacted width ------------------------------------
    const int nu = cfg_.n_used;
    {
        ggml_init_params ip = {ggml_tensor_overhead() * std::size_t(nu + 2) + 4096, nullptr,
                               true};
        ctx_in_ = ggml_init(ip);
        if (!ctx_in_) { *err = "ggml_init для входов не удался"; shutdown(); return false; }
        t_x_ = ggml_new_tensor_3d(ctx_in_, GGML_TYPE_F32, cfg_.n_embd, 1, 1);
        ggml_set_name(t_x_, "vk.x");
        n_ids_.assign(std::size_t(nu + 1), nullptr);
        for (int k = 1; k <= nu; ++k) {
            n_ids_[std::size_t(k)] = ggml_new_tensor_2d(ctx_in_, GGML_TYPE_I32, k, 1);
            char nm[32];
            snprintf(nm, sizeof(nm), "vk.ids.%d", k);
            ggml_set_name(n_ids_[std::size_t(k)], nm);
        }
        buf_in_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_in_, buft_);
        if (!buf_in_) { *err = "входной буфер не выделился"; shutdown(); return false; }
    }
    {
        const std::size_t nodes = 32;
        ggml_init_params ip = {
            (ggml_tensor_overhead() * (nodes + 16) + ggml_graph_overhead_custom(nodes, false))
                * std::size_t(nu) + 4096,
            nullptr, true};
        ctx_g_ = ggml_init(ip);
        if (!ctx_g_) { *err = "ggml_init для графов не удался"; shutdown(); return false; }
        gf_.assign(std::size_t(nu + 1), nullptr);
        ga_.assign(std::size_t(nu + 1), nullptr);
        n_up_.assign(std::size_t(nu + 1), nullptr);
        n_gate_.assign(std::size_t(nu + 1), nullptr);
        n_down_.assign(std::size_t(nu + 1), nullptr);
        n_out_.assign(std::size_t(nu + 1), nullptr);
        for (int k = 1; k <= nu; ++k) {
            ggml_tensor* ids = n_ids_[std::size_t(k)];
            // Plain mul_mat_id, three of them, and never the fork's fused up-gate op:
            // GGML_OP_FUSED_UP_GATE does not exist in the Vulkan backend, so building it here
            // would put the MoE back on the CPU with its weights stranded in video memory and
            // dragged back over PCIe every token - measured at 1.68 tok/s against 9.16.
            ggml_tensor* u = ggml_mul_mat_id(ctx_g_, up_[0], t_x_, ids);
            ggml_tensor* g = ggml_mul_mat_id(ctx_g_, gate_[0], t_x_, ids);
            ggml_tensor* a = ggml_mul(ctx_g_, u, ggml_silu(ctx_g_, g));
            ggml_tensor* o = ggml_mul_mat_id(ctx_g_, down_[0], a, ids);
            ggml_set_output(o);
            ggml_cgraph* gf = ggml_new_graph_custom(ctx_g_, nodes, false);
            ggml_build_forward_expand(gf, o);
            ggml_gallocr_t ga = ggml_gallocr_new(buft_);
            if (!ga || !ggml_gallocr_reserve(ga, gf) || !ggml_gallocr_alloc_graph(ga, gf)) {
                *err = "граф резидентной половины не разместился";
                shutdown();
                return false;
            }
            // Every node, asked of the backend that is going to run it. This is the guard
            // against the quantisation trap: IQ4_KS, IQ4_K and every _R4 type appear zero
            // times in ggml-vulkan.cpp, so a model quantised with one of them would reach
            // ggml_vk_mul_mat_id with no pipeline and abort inside the backend rather than
            // refuse here. IQ4_XS and Q6_K are the two that work.
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_tensor* node = ggml_graph_node(gf, i);
                if (ggml_backend_supports_op(be_, node)) continue;
                char buf[220];
                snprintf(buf, sizeof(buf),
                         "бэкенд Vulkan не поддерживает %s над %s — резидентные эксперты "
                         "должны быть IQ4_XS или Q6_K",
                         ggml_op_name(node->op),
                         node->src[0] ? ggml_type_name(node->src[0]->type) : "?");
                *err = buf;
                ggml_gallocr_free(ga);
                shutdown();
                return false;
            }
            gf_[std::size_t(k)]     = gf;
            ga_[std::size_t(k)]     = ga;
            n_up_[std::size_t(k)]   = u;
            n_gate_[std::size_t(k)] = g;
            n_down_[std::size_t(k)] = o;
            n_out_[std::size_t(k)]  = o;
        }
    }

    // Pinned host memory for the read back. ggml_vk_buffer_read_async looks the destination
    // up in the pinned registry (ggml_vk_host_get) and copies straight into it when it finds
    // it; an ordinary heap pointer costs a second hop through the backend's staging buffer.
    {
        const std::size_t need = std::size_t(cfg_.n_embd) * std::size_t(nu) * sizeof(float);
        ggml_backend_buffer_type_t hb = ggml_backend_vk_host_buffer_type();
        if (hb) buf_rb_ = ggml_backend_buft_alloc_buffer(hb, need);
        if (buf_rb_) rb_ = (float*)ggml_backend_buffer_get_base(buf_rb_);
        if (!rb_) {
            rb_fallback_.assign(need / sizeof(float), 0.0f);
            rb_ = rb_fallback_.data();
        }
    }

    // Resolved lazily in compute(), once the allocator for that width has actually placed the
    // tensor: if the driver put the small output buffer in host-visible coherent memory - which
    // it does, a buffer this size lands in the BAR heap - the readback becomes a memcpy through
    // the mapping. ggml_backend_tensor_get would instead take ggml_vk_buffer_read's non-UMA
    // branch: a whole submit and fence, per layer, to move 64 KB.
    out_mapped_.assign(std::size_t(nu) + 1u, nullptr);
    out_mapped_probed_.assign(std::size_t(nu) + 1u, 0);

    // Pinned staging for promotions. Depth is capped so the pinned footprint stays small: a
    // deeper ring folds more promotions behind one fence but pins more memory, and the win is
    // already most of the way there by eight.
    // Descending, because pinned memory is scarce exactly when it is most wanted: with
    // repacking on, the model occupies ~15.5 GB of the host heap and the Vulkan budget for it
    // has only tens of MiB left. Taking the largest ring that fits, rather than insisting on
    // one size, keeps a tight machine on the batched path instead of dropping it to a fence
    // per matrix - and never starves a later allocation, because it stops as soon as one works.
    {
        ggml_backend_buffer_type_t hb = ggml_backend_vk_host_buffer_type();
        for (int want_slots : {8, 4, 2, 1}) {
            if (!hb || bpe_ == 0) break;
            buf_stage_ = ggml_backend_buft_alloc_buffer(hb, bpe_ * std::size_t(want_slots));
            if (buf_stage_) {
                stage_ = (char*)ggml_backend_buffer_get_base(buf_stage_);
                if (stage_) { stage_slots_ = want_slots; break; }
                ggml_backend_buffer_free(buf_stage_);
                buf_stage_ = nullptr;
            }
        }
        if (stage_) {
            stage_pinned_ = true;
        } else {
            // No pinned memory: correctness is unaffected, every upload just keeps its own
            // fence as before.
            stage_fallback_.assign(bpe_, 0);
            stage_        = stage_fallback_.data();
            stage_slots_  = 1;
            stage_pinned_ = false;
        }
    }

    // How many queued promotions one batch may hold. The ring depth is the hard ceiling - a
    // staging slot cannot be rewritten while a recorded copy still points into it - so taking
    // the ring size is taking the largest batch that costs exactly one fence. Draining past it
    // buys nothing: upload() would close and reopen the batch every stage_slots_ promotions,
    // which is the same fence count with a longer wait for any dispatch that arrives meanwhile.
    //
    // Unpinned staging leaves this at 1 on purpose. There is no batching to fold into in that
    // mode (batch_begin/batch_end are no-ops and every matrix keeps its own fence), so a drain
    // would only make an arriving dispatch queue behind several synchronous writes.
    {
        // DEFAULT ONE, i.e. the pre-drain behaviour, because the drain was MEASURED at -2.1%
        // on the token (three rounds, spreads 0.5% and 0.8%). The mechanism works exactly as
        // designed - 6.77 promotions per fence instead of 1.00, prefetch 8.96 -> 8.26 ms/token,
        // device layer 29.63 -> 27.36 ms - and the token still got slower, so it does not go in
        // the default path. It stays behind the switch rather than being deleted because it may
        // net positive combined with a smaller promotion budget, and rebuilding it to find out
        // would cost more than keeping it.
        promo_drain_ = 1;
        if (const char* e = getenv("MEMEX_PROMO_DRAIN")) {
            const int v = atoi(e);
            if (v >= 1) promo_drain_ = v;
        }
        if (promo_drain_ < 1) promo_drain_ = 1;
        // Close the batch early when a dispatch is waiting. On by default because the first
        // measurement of the drain showed the cost is the hold, not the fence.
        promo_yield_ = true;
        if (const char* e = getenv("MEMEX_PROMO_YIELD")) promo_yield_ = (atoi(e) != 0);
    }

    slot_of_.assign(std::size_t(cfg_.n_layers) * std::size_t(cfg_.n_experts), int16_t(-1));
    expert_at_.assign(std::size_t(cfg_.n_layers) * std::size_t(cfg_.capacity), int16_t(-1));
    dirty_.assign(std::size_t(cfg_.n_layers) * std::size_t(cfg_.capacity), 0);
    pending_.assign(std::size_t(cfg_.n_layers), 0);
    job_slots_.assign(std::size_t(nu), 0);
    job_at_.assign(std::size_t(nu), 0);
    job_x_.assign(std::size_t(cfg_.n_embd), 0.0f);
    res_.assign(std::size_t(cfg_.n_embd) * std::size_t(nu), 0.0f);
    sites_.resize(std::size_t(cfg_.n_layers));
    for (int il = 0; il < cfg_.n_layers; ++il) {
        sites_[std::size_t(il)].self = this;
        sites_[std::size_t(il)].il   = il;
    }

    // Third reading, after the graphs and the input tensors exist. The delta from the second
    // is where the small buffers went, and small buffers are supposed to go to the BAR heap:
    // the input and the id lists are written by the host once per layer, and a host-visible
    // destination turns that write into a memcpy instead of a staged copy with a fence
    // (ggml-vulkan.cpp:4659-4664 takes the memcpy branch exactly when the buffer is
    // HOST_VISIBLE). It is the one place in this design where landing in BAR is the goal.
    print_heaps("после размещения графов и входов");
    printf("  --- куда каждая группа ОБЯЗАНА была лечь (правило find_properties) ---\n");
    for (const GpuExpertsBuffer& b : bufs_info_) {
        char nm[64];
        snprintf(nm, sizeof(nm), "веса экспертов, слои %d..%d", b.first_layer,
                 b.first_layer + b.n_layers - 1);
        print_placement(nm, b.bytes);
    }
    if (buf_in_) print_placement("вход и списки идентификаторов",
                                 ggml_backend_buffer_get_size(buf_in_));
    print_placement("рабочая память графа (на ширину)",
                    std::size_t(cfg_.n_embd) * std::size_t(nu) * sizeof(float) * 2);

    th_ = std::thread(&GpuExperts::worker, this);
#if defined(_WIN32)
    // The join blocks one ggml thread and leaves the rest spinning on their barrier; on a
    // machine whose thread count is its core count that is every core busy while the only
    // thread with work to do is this one. Raising it is not a fix for oversubscription - use
    // one thread fewer - but it stops the scheduler from choosing the spinners.
    SetThreadPriority((HANDLE)th_.native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
    return true;
}

void GpuExperts::shutdown() {
    if (th_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_job_.notify_all();
        th_.join();
    }
    for (ggml_gallocr_t g : ga_) {
        if (g) ggml_gallocr_free(g);
    }
    ga_.clear();
    if (ctx_g_) { ggml_free(ctx_g_); ctx_g_ = nullptr; }
    if (buf_in_) { ggml_backend_buffer_free(buf_in_); buf_in_ = nullptr; }
    if (ctx_in_) { ggml_free(ctx_in_); ctx_in_ = nullptr; }
    if (buf_rb_) { ggml_backend_buffer_free(buf_rb_); buf_rb_ = nullptr; rb_ = nullptr; }
    if (buf_stage_) {
        ggml_backend_buffer_free(buf_stage_);
        buf_stage_ = nullptr;
        stage_ = nullptr;
        stage_pinned_ = false;
    }
    free_weights();
    // After the worker is joined: it is the only thread that reads the file handle.
    close_plain_source();
    if (be_) { ggml_backend_free(be_); be_ = nullptr; }
}

// ---------------------------------------------------------------------------------------
// The slot map
// ---------------------------------------------------------------------------------------

void GpuExperts::sync_slots(const ResidentSet& rs) {
    if (!on()) return;
    const int nl = cfg_.n_layers, ne = cfg_.n_experts, cap = cfg_.capacity;
    std::vector<int> freelist;
    freelist.reserve(std::size_t(cap));
    {
        std::lock_guard<std::mutex> lk(mu_);
        pending_total_ = 0;
        for (int il = 0; il < nl; ++il) {
            const std::size_t sb = std::size_t(il) * std::size_t(cap);
            const std::size_t eb = std::size_t(il) * std::size_t(ne);
            // Evict first, so a slot freed by this refresh is available to it.
            freelist.clear();
            for (int s = 0; s < cap; ++s) {
                const int16_t e = expert_at_[sb + std::size_t(s)];
                if (e < 0) { freelist.push_back(s); continue; }
                // is_claimed, not is_resident: under deferred activation a promoted expert is
                // pending rather than resident, and it already owns this slot. Evicting on
                // is_resident would free the slot of a transfer that is in flight, and then
                // hand it to someone else - the one way the slot map and the device could
                // disagree about what a slot holds. With deferred off the two are identical.
                if (!rs.is_claimed(il, e)) {
                    expert_at_[sb + std::size_t(s)] = -1;
                    slot_of_[eb + std::size_t(e)]   = -1;
                    dirty_[sb + std::size_t(s)]     = 0;
                    freelist.push_back(s);
                }
            }
            std::size_t next_free = 0;
            for (int e = 0; e < ne; ++e) {
                if (!rs.is_claimed(il, e)) continue;
                if (slot_of_[eb + std::size_t(e)] >= 0) continue;
                if (next_free >= freelist.size()) { ++st_.no_slot; continue; }
                const int s = freelist[next_free++];
                expert_at_[sb + std::size_t(s)] = int16_t(e);
                slot_of_[eb + std::size_t(e)]   = int16_t(s);
                dirty_[sb + std::size_t(s)]     = 1;
            }
            int p = 0;
            for (int s = 0; s < cap; ++s) p += dirty_[sb + std::size_t(s)] ? 1 : 0;
            pending_[std::size_t(il)] = p;
            pending_total_ += p;
        }
    }
    cv_job_.notify_all();
}

// ---------------------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------------------

void GpuExperts::worker() {
    try {
        worker_loop();
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu_);
        failed_     = true;
        fail_msg_   = e.what();
        job_active_ = false;
        job_done_   = true;
        busy_       = false;
        pending_total_ = 0;
        cv_done_.notify_all();
        cv_idle_.notify_all();
    } catch (...) {
        std::lock_guard<std::mutex> lk(mu_);
        failed_     = true;
        fail_msg_   = "неизвестное исключение на потоке устройства";
        job_active_ = false;
        job_done_   = true;
        busy_       = false;
        pending_total_ = 0;
        cv_done_.notify_all();
        cv_idle_.notify_all();
    }
}

std::string GpuExperts::failure() {
    std::lock_guard<std::mutex> lk(mu_);
    return failed_ ? fail_msg_ : std::string();
}

void GpuExperts::worker_loop() {
    for (;;) {
        int il = -1, k = 0;
        bool drained = false;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_job_.wait(lk, [&] { return quit_ || job_active_ || pending_total_ > 0; });
            if (quit_) return;
            if (job_active_) {
                il = job_il_;
                k  = job_k_;
            } else {
                // Idle: get ahead on the uploads this refresh asked for, lowest layer first,
                // and take up to promo_drain_ of them in one pass so that ONE submit and ONE
                // fence cover the lot.
                //
                // This used to take exactly one, on the reasoning that a job arriving
                // mid-prefetch should wait for one slice rather than for the whole refresh.
                // The counter refuted the reasoning: 1902 promotions produced 1902 batches, so
                // every promotion paid its own submit and its own fence, and the promotions
                // together held the worker - the very thread the CPU joins on - for 13.04 ms
                // per token while moving 17.3 MB, which is 1.33 GB/s over a 3.94 GB/s link.
                // A shorter wait per arrival is worth less than not spending the 8.6 ms of
                // excess in the first place.
                //
                // What does NOT change, and must not: how many promotions there are. The
                // budget lives in ResidentSet::refresh (resident_set.cpp:245,350) and clips
                // the request per layer per refresh before anything reaches this queue.
                // Draining changes the fence count, never the byte count.
                dr_il_.clear();
                dr_slot_.clear();
                dr_expert_.clear();
                const int want = promo_drain_;
                for (int i = 0; i < cfg_.n_layers && int(dr_slot_.size()) < want; ++i) {
                    if (pending_[std::size_t(i)] == 0) continue;
                    const std::size_t sb = std::size_t(i) * std::size_t(cfg_.capacity);
                    for (int s = 0; s < cfg_.capacity && int(dr_slot_.size()) < want; ++s) {
                        if (!dirty_[sb + std::size_t(s)]) continue;
                        dirty_[sb + std::size_t(s)] = 0;
                        --pending_[std::size_t(i)];
                        --pending_total_;
                        dr_il_.push_back(i);
                        dr_slot_.push_back(s);
                        dr_expert_.push_back(expert_at_[sb + std::size_t(s)]);
                    }
                }
                // pending_total_ said there was work and the map says there is not. Clearing
                // it rather than looping on it: a spin here would be a live hang with no
                // symptom but a hot core.
                if (dr_slot_.empty()) { pending_total_ = 0; continue; }
                drained = true;
            }
            busy_ = true;
        }

        if (drained) {
            // One batch for all of them: batch_begin, every upload, one batch_end. The three
            // matrices of one promotion were already bracketed together; now the promotions
            // are too.
            //
            // Deferred activation survives unchanged, and it is batch_end that makes it so.
            // upload() records into batch_landed_, which is worker-thread-local; batch_end
            // waits on the fence and only then moves the tags into landed_, where take_landed
            // picks them up and the host flips the mask. So "landed" now means the whole batch
            // landed, and no expert in the batch is claimed as resident one moment earlier
            // than the last byte of the batch. A slot being written meanwhile belongs to a
            // PENDING expert, which the mask does not name, so no dispatch can read it - the
            // same invariant as before, just held for longer.
            //
            // A partly-drained queue is consistent by construction: every slot taken here had
            // its dirty_ bit cleared and pending_ decremented under the lock, so the slots
            // still in the queue are exactly those not in this batch, and sync_slots keeps its
            // hands off all of them because is_claimed() covers pending as well as resident.
            //
            // AND IT YIELDS. The first measurement of the drain said the fence count is not
            // where the time is: folding 6.77 promotions behind one fence took the prefetch
            // from 8.94 to 8.27 ms/token (-7.5%) and pushed the join wait UP, 11.58 to 15.18
            // ms/token, because a dispatch arriving mid-batch now queues behind the whole
            // batch instead of behind one promotion. So the batch is closed as soon as a
            // dispatch is waiting: whatever was already recorded still costs one fence, and
            // the dispatch waits for the tail of the batch rather than for all of it.
            // MEMEX_PROMO_YIELD=0 turns the check off, which is the non-yielding arm.
            const auto t_pr = std::chrono::steady_clock::now();
            batch_begin();
            std::size_t j = 0;
            for (; j < dr_slot_.size(); ++j) {
                if (promo_yield_ && j > 0) {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (job_active_) break;
                }
                if (dr_expert_[j] < 0) continue;
                if (!upload(dr_il_[j], dr_slot_[j], dr_expert_[j])) { ++j; break; }
            }
            batch_end();
            // Anything not uploaded goes back on the queue exactly as it was: the dirty bit
            // set again and pending_ restored. Without this the slot would keep its new
            // occupant in the map while its bytes were never sent - the one way the map and
            // the device could disagree about what a slot holds.
            if (j < dr_slot_.size()) {
                std::lock_guard<std::mutex> lk(mu_);
                for (std::size_t r = j; r < dr_slot_.size(); ++r) {
                    const std::size_t sb = std::size_t(dr_il_[r]) * std::size_t(cfg_.capacity);
                    if (dirty_[sb + std::size_t(dr_slot_[r])]) continue;   // never twice
                    dirty_[sb + std::size_t(dr_slot_[r])] = 1;
                    ++pending_[std::size_t(dr_il_[r])];
                    ++pending_total_;
                }
            }
            const double pr_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_pr).count();
            std::lock_guard<std::mutex> lk(mu_);
            st_.ms_promote += pr_ms;
            busy_ = false;
            cv_idle_.notify_all();
            continue;
        }

        // A dispatch.
        //
        // Without deferred activation, anything this layer still owes the link has to land
        // first: the mask the host built these ids from already names the new occupant, so
        // computing before the upload would multiply by the expert that just left. That flush
        // is exactly what put a promotion back on the critical path - the prefetch was
        // asynchronous but its result was not.
        //
        // With deferred activation it is not needed and must not be done: a slot that is still
        // being uploaded belongs to a PENDING expert, which the mask does not claim, so no id
        // in this dispatch can name it. The layer therefore never waits for the link, and a
        // promotion costs at worst another token or two of the expert being computed on the
        // CPU - which is what it would have cost anyway had it not been promoted.
        if (!cfg_.deferred) flush_layer(il);
        // The device half's own wall time, measured on the thread that does it. Against
        // ms_cpu_half it says which side is the longer of the two, and therefore which one
        // max() is - and if ms_join_wait is large while this is small, the answer is neither
        // and the cost is the handover itself.
        const auto t_job = std::chrono::steady_clock::now();
        compute(il, k);
        const double job_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_job).count();
        {
            std::lock_guard<std::mutex> lk(mu_);
            st_.ms_job += job_ms;
            job_active_ = false;
            job_done_   = true;
            busy_       = false;
        }
        cv_done_.notify_one();
        cv_idle_.notify_all();
    }
}

void GpuExperts::flush_layer(int il) {
    // Everything this layer owes goes into one command buffer and costs one fence, instead of
    // three per promotion. It has to land before compute(il) reads the slots, and batch_end
    // waits on the fence, so the ordering the old per-write fences gave is preserved exactly.
    batch_begin();
    for (;;) {
        int slot = -1, expert = -1;
        bool done = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (pending_[std::size_t(il)] == 0) {
                done = true;
            } else {
                const std::size_t sb = std::size_t(il) * std::size_t(cfg_.capacity);
                for (int s = 0; s < cfg_.capacity; ++s) {
                    if (!dirty_[sb + std::size_t(s)]) continue;
                    dirty_[sb + std::size_t(s)] = 0;
                    --pending_[std::size_t(il)];
                    --pending_total_;
                    slot = s;
                    expert = expert_at_[sb + std::size_t(s)];
                    break;
                }
                if (slot < 0) { pending_[std::size_t(il)] = 0; done = true; }
            }
        }
        // Outside the lock: batch_end takes mu_ for its own accounting.
        if (done) { batch_end(); return; }
        if (expert >= 0 && !upload(il, slot, expert)) return;
    }
}

bool GpuExperts::upload(int il, int slot, int expert) {
    ggml_tensor* d[3] = {up_[std::size_t(il)], gate_[std::size_t(il)], down_[std::size_t(il)]};

    // A staging slot cannot be rewritten while a recorded copy still points into it, so a full
    // ring means closing the batch - one fence - and starting the next.
    if (batching_ && batch_used_ >= stage_slots_) {
        batch_end();
        batch_begin();
    }
    char* base = stage_ + std::size_t(batching_ ? batch_used_ : 0) * bpe_;

    std::size_t o = 0;
    bool all_batched = batching_;
    for (int i = 0; i < 3; ++i) {
        const std::size_t n = desc(il, i).slab;
        char* sp = base + o;
        o += n;
        // stage_ belongs to this thread: upload() is reached only from worker_loop and
        // flush_layer, both of which run on the worker.
        std::string rerr;
        if (!read_plain(il, i, expert, sp, &rerr)) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                failed_   = true;
                fail_msg_ = "промоушен не смог прочитать веса: " + rerr;
            }
            // Nothing this batch recorded may be confirmed: the run is over and the bytes of
            // this promotion are incomplete.
            batch_landed_.clear();
            batch_end();   // outside the lock; drops whatever was recorded, run is over anyway
            return false;
        }
        // Batched when it can be: one submit and one fence will cover this and its neighbours.
        // The fallback is the old behaviour - a submit and fence per matrix - and is correct,
        // just not cheap. A refused record closes the batch first, so a synchronous write is
        // never interleaved with copies already recorded into an open command buffer.
        bool recorded = false;
        if (batching_) {
            recorded = ggml_backend_vk_batch_set_tensor(be_, d[i], sp, std::size_t(slot) * n, n);
            if (!recorded) batch_end();
        }
        if (!recorded) {
            all_batched = false;
            ggml_backend_tensor_set(d[i], sp, std::size_t(slot) * n, n);
        }
    }
    const uint32_t tag = (uint32_t(il) << 16) | uint32_t(expert & 0xffff);
    if (all_batched) {
        ++batch_used_;
        // Recorded, not landed. It becomes landed when batch_end's fence passes.
        batch_landed_.push_back(tag);
    }
    std::lock_guard<std::mutex> lk(mu_);
    ++st_.promotions;
    st_.promo_bytes += bpe_;
    if (!all_batched) {
        ++st_.upload_fences_unbatched;
        // ggml_backend_tensor_set is synchronous on this backend, so by here the bytes are in
        // device memory already.
        landed_.push_back(tag);
    }
    return true;
}

void GpuExperts::take_landed(std::vector<uint32_t>* out) {
    out->clear();
    if (!on()) return;
    std::lock_guard<std::mutex> lk(mu_);
    out->swap(landed_);
}

int GpuExperts::slot_of(int il, int expert) {
    if (!on() || il < 0 || il >= cfg_.n_layers || expert < 0 || expert >= cfg_.n_experts) {
        return -1;
    }
    std::lock_guard<std::mutex> lk(mu_);
    return slot_of_[std::size_t(il) * std::size_t(cfg_.n_experts) + std::size_t(expert)];
}

int GpuExperts::expert_at(int il, int slot) {
    if (!on() || il < 0 || il >= cfg_.n_layers || slot < 0 || slot >= cfg_.capacity) return -1;
    std::lock_guard<std::mutex> lk(mu_);
    return expert_at_[std::size_t(il) * std::size_t(cfg_.capacity) + std::size_t(slot)];
}

bool GpuExperts::slot_dirty(int il, int slot) {
    if (!on() || il < 0 || il >= cfg_.n_layers || slot < 0 || slot >= cfg_.capacity) return false;
    std::lock_guard<std::mutex> lk(mu_);
    return dirty_[std::size_t(il) * std::size_t(cfg_.capacity) + std::size_t(slot)] != 0;
}

// One submit and one fence for everything recorded since batch_begin. Both are no-ops when the
// staging ring is not pinned, in which case uploads keep their own fences and stay correct.
void GpuExperts::batch_begin() {
    if (!stage_pinned_ || batching_) return;
    ggml_backend_vk_batch_begin(be_);
    batching_   = true;
    batch_used_ = 0;
}

void GpuExperts::batch_end() {
    if (!batching_) return;
    const bool had_work = batch_used_ > 0;
    ggml_backend_vk_batch_end(be_);
    batching_   = false;
    batch_used_ = 0;
    // batch_end waits on the fence, so everything this batch recorded is now in device memory.
    // That, and only that, is what a landing confirmation means.
    if (had_work || !batch_landed_.empty()) {
        std::lock_guard<std::mutex> lk(mu_);
        if (had_work) ++st_.batch_fences;
        landed_.insert(landed_.end(), batch_landed_.begin(), batch_landed_.end());
    }
    batch_landed_.clear();
}

void GpuExperts::compute(int il, int k) {
    const std::size_t ne = std::size_t(cfg_.n_embd);
    std::fill(res_.begin(), res_.begin() + ne * std::size_t(cfg_.n_used), 0.0f);
    if (k <= 0) return;

    ggml_backend_tensor_set(t_x_, job_x_.data(), 0, ne * sizeof(float));
    ggml_backend_tensor_set(n_ids_[std::size_t(k)], job_slots_.data(), 0,
                            std::size_t(k) * sizeof(int32_t));
    // Re-aiming rather than rebuilding, the same trick the decode graph's cache writes use:
    // every layer's three expert stacks have the same shape and type, so one built graph per
    // width serves all forty-eight of them and the only thing that moves is src[0].
    n_up_[std::size_t(k)]->src[0]   = up_[std::size_t(il)];
    n_gate_[std::size_t(k)]->src[0] = gate_[std::size_t(il)];
    n_down_[std::size_t(k)]->src[0] = down_[std::size_t(il)];

    ggml_backend_graph_compute(be_, gf_[std::size_t(k)]);

    // THE READBACK, AND IT WAS COSTING THE WHOLE SCHEME. Measured, not reasoned about.
    //
    // What stood here: if this width's output landed in host-visible coherent memory, take the
    // mapped pointer and memcpy, so the second fence per layer disappears. The counter beside
    // it counted FENCES, and by that measure it was a total success - 9188 readbacks, 0 fences.
    //
    // What it actually cost: a host READ from the BAR aperture is uncached and uncombined, so
    // every cache line is its own PCIe transaction. The same shortcut was measured directly in
    // GpuStatic's layer path at 16.9 KB per call: 0.758 ms, i.e. 22 MB/s. This output is
    // n_embd * k floats - about 46 KB at the average compacted width - which is roughly 2 ms
    // per layer, 91 ms per token over forty-eight. That is the entire difference between the
    // full scheme running at 5.74 tok/s and the static half alone running at 12.02.
    //
    // The direction is what decides it, not the size. Writes through the same mapping are
    // write-combined and nearly free, which is why the inputs above are written exactly that
    // way (8 KB at 0.003 ms). Reads are not. ggml_vk_buffer_read
    // (ggml-vulkan.cpp:4805-4831) already refuses the aperture on a non-UMA device for this
    // reason and takes the hardware copy path; rb_ being Vulkan-pinned makes that one DMA into
    // our own buffer rather than a hop through the backend's shared staging. It costs the
    // submit and fence this branch was written to avoid - about 59 us - against 2000.
    //
    // Kept as a counted quantity rather than deleted, because "fences per layer" is still worth
    // knowing; it is simply not the quantity that decides this.
    const std::size_t ki = std::size_t(k);
    ggml_backend_tensor_get(n_out_[ki], rb_, 0, ne * std::size_t(k) * sizeof(float));
    {
        std::lock_guard<std::mutex> lk(mu_);
        ++st_.readback_fenced;
    }

    // Scatter back into router slots. Everything else stays the exact 0.0f the fill left,
    // which is what makes ggml_add(o_res, o_oth) a per-slot x + 0.0f and therefore the
    // unsplit path's own arithmetic in the unsplit path's own order.
    for (int j = 0; j < k; ++j) {
        const int s = job_at_[std::size_t(j)];
        std::memcpy(res_.data() + ne * std::size_t(s), rb_ + ne * std::size_t(j),
                    ne * sizeof(float));
    }
}

void GpuExperts::drain() {
    if (!on()) return;
    std::unique_lock<std::mutex> lk(mu_);
    cv_idle_.wait(lk, [&] {
        return failed_ || (!job_active_ && !busy_ && pending_total_ == 0);
    });
}

// ---------------------------------------------------------------------------------------
// The graph side
// ---------------------------------------------------------------------------------------

void GpuExperts::do_fork(int il, const ggml_tensor* ids_res, const ggml_tensor* xe) {
    const int nu = cfg_.n_used;
    std::unique_lock<std::mutex> lk(mu_);
    int k = 0;
    const std::size_t eb = std::size_t(il) * std::size_t(cfg_.n_experts);
    for (int s = 0; s < nu; ++s) {
        const int32_t id = id_at(ids_res, s);
        if (id < 0) continue;              // this slot belongs to the CPU half
        const int16_t slot = slot_of_[eb + std::size_t(id)];
        if (slot < 0) { ++st_.no_slot; continue; }
        job_slots_[std::size_t(k)] = slot;
        job_at_[std::size_t(k)]    = s;
        ++k;
    }
    std::memcpy(job_x_.data(), xe->data, std::size_t(cfg_.n_embd) * sizeof(float));
    job_il_     = il;
    job_k_      = k;
    job_active_ = true;
    job_done_   = false;
    fork_t_     = std::chrono::steady_clock::now();
    ++st_.layers;
    // Did this job find the worker already inside a Vulkan call? If so it will not start
    // until that call's fence returns, and the CPU half is overlapping with nothing.
    if (busy_) ++st_.fork_busy;
    if (k == 0) ++st_.layers_empty;
    st_.experts += uint64_t(k);
    lk.unlock();
    cv_job_.notify_one();
}

void GpuExperts::do_join(int il, ggml_tensor* dst, const ggml_tensor* o_res_cpu) {
    (void)il;
    const std::size_t n = std::size_t(cfg_.n_embd) * std::size_t(cfg_.n_used);
    // Timed from BEFORE the lock, because contending for mu_ is waiting too. Everything
    // between here and the predicate coming true is the whole ggml pool stopped: this op has
    // n_tasks = 1, so the other threads are at their barrier.
    const auto t_arrive = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(mu_);
    const bool ready_on_arrival = job_done_ || failed_;
    cv_done_.wait(lk, [&] { return job_done_ || failed_; });
    {
        using ms_d = std::chrono::duration<double, std::milli>;
        const auto t_go = std::chrono::steady_clock::now();
        ++st_.join_waits;
        if (ready_on_arrival) ++st_.join_ready;
        st_.ms_join_wait += ms_d(t_go - t_arrive).count();
        st_.ms_cpu_half  += ms_d(t_arrive - fork_t_).count();
    }
    if (failed_) {
        std::memset(dst->data, 0, n * sizeof(float));
        return;
    }
    std::memcpy(dst->data, res_.data(), n * sizeof(float));

    if (cfg_.check && o_res_cpu && o_res_cpu->data) {
        const float* a = res_.data();
        const float* b = (const float*)o_res_cpu->data;
        for (int s = 0; s < cfg_.n_used; ++s) {
            const std::size_t off = std::size_t(s) * std::size_t(cfg_.n_embd);
            bool cpu_zero = true, gpu_zero = true;
            double num = 0.0, den = 0.0, worst = 0.0;
            for (int i = 0; i < cfg_.n_embd; ++i) {
                const float x = a[off + std::size_t(i)];
                const float y = b[off + std::size_t(i)];
                if (x != 0.0f) gpu_zero = false;
                if (y != 0.0f) cpu_zero = false;
                const double d = double(x) - double(y);
                num += d * d;
                den += double(y) * double(y);
                worst = std::max(worst, std::fabs(d));
            }
            ++st_.checked;
            // The CPU-only split owns exactly the slots the device owns, because both halves
            // come from the same mask. A slot the device did not fill must therefore be one
            // the CPU half left at zero too, and the other way round; either mismatch means
            // the compaction lost a slot, which is the one failure that produces no wrong
            // shape and no NaN.
            if (!gpu_zero && cpu_zero) ++st_.zero_bad;
            if (gpu_zero && !cpu_zero) ++st_.owned_bad;
            if (!cpu_zero) {
                st_.worst_abs = std::max(st_.worst_abs, worst);
                if (den > 0.0) {
                    st_.worst_rel = std::max(st_.worst_rel, std::sqrt(num / den));
                }
            }
        }
    }
}

void GpuExperts::fork_op(ggml_tensor* dst, const ggml_tensor* a, const ggml_tensor* b,
                         const ggml_tensor* c, int ith, int /*nth*/, void* ud) {
    if (ith != 0) return;
    Site* s = (Site*)ud;
    s->self->do_fork(s->il, c, b);
    // and then the identity that makes the ordering a data dependency rather than a wish:
    // the CPU half is dispatched with this copy of its own id list, so no expert node can be
    // scheduled ahead of the submit above.
    for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
        const char* src = (const char*)a->data + i1 * a->nb[1];
        int32_t* out = (int32_t*)((char*)dst->data + i1 * dst->nb[1]);
        for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
            out[i0] = *(const int32_t*)(src + i0 * a->nb[0]);
        }
    }
}

void GpuExperts::join_op(ggml_tensor* dst, const ggml_tensor* /*a*/, int ith, int /*nth*/,
                         void* ud) {
    if (ith != 0) return;
    Site* s = (Site*)ud;
    s->self->do_join(s->il, dst, nullptr);
}

void GpuExperts::join_check_op(ggml_tensor* dst, const ggml_tensor* /*a*/,
                               const ggml_tensor* b, int ith, int /*nth*/, void* ud) {
    if (ith != 0) return;
    Site* s = (Site*)ud;
    s->self->do_join(s->il, dst, b);
}

ggml_tensor* GpuExperts::fork(ggml_context* c, int il, ggml_tensor* ids_oth, ggml_tensor* xe,
                              ggml_tensor* ids_res) {
    return ggml_map_custom3(c, ids_oth, xe, ids_res, fork_op, /*n_tasks=*/1,
                            &sites_[std::size_t(il)]);
}

ggml_tensor* GpuExperts::join(ggml_context* c, int il, ggml_tensor* o_oth,
                              ggml_tensor* o_res_cpu) {
    if (cfg_.check && o_res_cpu) {
        return ggml_map_custom2(c, o_oth, o_res_cpu, join_check_op, /*n_tasks=*/1,
                                &sites_[std::size_t(il)]);
    }
    return ggml_map_custom1(c, o_oth, join_op, /*n_tasks=*/1, &sites_[std::size_t(il)]);
}

// ---------------------------------------------------------------------------------------
// The one check nothing else can make
// ---------------------------------------------------------------------------------------

bool GpuExperts::verify_slot(int il, int slot, std::string* err) {
    if (!on()) { *err = "выключено"; return false; }
    drain();
    int expert = -1;
    {
        std::lock_guard<std::mutex> lk(mu_);
        expert = expert_at_[std::size_t(il) * std::size_t(cfg_.capacity) + std::size_t(slot)];
    }
    if (expert < 0) { *err = "слот пуст"; return false; }
    ggml_tensor* d[3] = {up_[std::size_t(il)], gate_[std::size_t(il)], down_[std::size_t(il)]};
    const char* what[3] = {"up", "gate", "down"};
    std::vector<char> tmp, ref;
    for (int i = 0; i < 3; ++i) {
        const std::size_t n = desc(il, i).slab;
        tmp.assign(n, 0);
        ref.assign(n, 0);
        // Compared against the same bytes the uploader used, i.e. the stored plain layout -
        // comparing against a repacked RAM tensor would fail on every byte and prove nothing.
        if (!read_plain(il, i, expert, ref.data(), err)) return false;
        ggml_backend_tensor_get(d[i], tmp.data(), std::size_t(slot) * n, n);
        if (std::memcmp(tmp.data(), ref.data(), n) != 0) {
            char buf[200];
            snprintf(buf, sizeof(buf),
                     "слой %d слот %d (эксперт %d): %s в видеопамяти не совпал с моделью",
                     il, slot, expert, what[i]);
            *err = buf;
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// The self-test
// ---------------------------------------------------------------------------------------

namespace {

// Bit-for-bit, deliberately. These are the comparisons the whole per-slot argument rests on
// and "close" is not what is being claimed.
bool same_bits(const float* a, const float* b, std::size_t n) {
    return std::memcmp(a, b, n * sizeof(float)) == 0;
}

bool all_zero_bits(const float* a, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != 0.0f) return false;
    }
    return true;
}

double rel_l2(const float* a, const float* b, std::size_t n) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(b[i]) * double(b[i]);
    }
    return den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
}

uint32_t xrand(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

// Roughly normal, and that matters rather more than it looks. Uniform weights spread the
// IQ4_XS codebook indices evenly, including the two ends of kvalues_iq4nl at +-127, and the
// CPU kernel's int16 accumulation saturates there - so a self-test built on uniform noise
// measures the test's own pathology and reports it as the device disagreeing by 18%. Trained
// weights are concentrated near zero, the indices cluster in the middle of the codebook, and
// the same comparison lands at 1e-3. Twelve uniforms summed is close enough to a normal to
// reproduce that, and it costs nothing.
float xnorm(uint32_t& s, float sigma) {
    float acc = 0.0f;
    for (int i = 0; i < 12; ++i) acc += float(xrand(s) >> 8) / 16777216.0f;
    return (acc - 6.0f) * sigma;
}

}  // namespace

// One op at a time, CPU against Vulkan, on identical bytes. This exists because the whole-
// layer comparison cannot tell a kernel that rounds differently from a kernel that decodes a
// quantisation differently, and the two call for opposite decisions: the first is a cost you
// accept, the second means the type must not be used on the device at all.
namespace {

void op_probe(int threads) {
    const int n_embd = 2048, n_ff = 256, n_exp = 4, n_ids = 4;
    const ggml_type types[3] = {GGML_TYPE_F16, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS};

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);
    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (!vk) { printf("  проба по операциям: устройство не открылось\n"); return; }
    ggml_backend_buffer_type_t cbuft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t vbuft = ggml_backend_vk_buffer_type(0);

    printf("\n  --- проба по операциям: одна mul_mat_id, одинаковые байты ---\n");
    for (ggml_type qt : types) {
        // Host copy, quantised once, then the same bytes pushed to the device.
        ggml_init_params ip = {ggml_tensor_overhead() * 16 + 4096, nullptr, true};
        ggml_context* ch = ggml_init(ip);
        ggml_tensor* wh = ggml_new_tensor_3d(ch, qt, n_embd, n_ff, n_exp);
        ggml_tensor* xh_t = ggml_new_tensor_3d(ch, GGML_TYPE_F32, n_embd, 1, 1);
        ggml_tensor* ih = ggml_new_tensor_2d(ch, GGML_TYPE_I32, n_ids, 1);
        ggml_backend_buffer_t hb = ggml_backend_alloc_ctx_tensors_from_buft(ch, cbuft);

        uint32_t seed = 42u;
        std::vector<float> tmp;
        tmp.resize(std::size_t(n_embd) * std::size_t(n_ff) * std::size_t(n_exp));
        for (std::size_t i = 0; i < tmp.size(); ++i) {
            tmp[i] = xnorm(seed, 0.02f);
        }
        if (qt == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(tmp.data(), (ggml_fp16_t*)wh->data, int64_t(tmp.size()));
        } else {
            ggml_quantize_chunk(qt, tmp.data(), wh->data, 0,
                                int64_t(n_ff) * int64_t(n_exp), n_embd, nullptr, nullptr);
        }
        std::vector<float> xv;
        xv.resize(std::size_t(n_embd));
        for (int i = 0; i < n_embd; ++i) {
            xv[std::size_t(i)] = xnorm(seed, 1.0f);
        }
        std::vector<int32_t> iv;
        iv.resize(std::size_t(n_ids));
        for (int i = 0; i < n_ids; ++i) iv[std::size_t(i)] = i % n_exp;
        ggml_backend_tensor_set(xh_t, xv.data(), 0, xv.size() * sizeof(float));
        ggml_backend_tensor_set(ih, iv.data(), 0, iv.size() * sizeof(int32_t));

        // The device copy of the very same bytes.
        ggml_context* cd = ggml_init(ip);
        ggml_tensor* wd = ggml_new_tensor_3d(cd, qt, n_embd, n_ff, n_exp);
        ggml_tensor* xd = ggml_new_tensor_3d(cd, GGML_TYPE_F32, n_embd, 1, 1);
        ggml_tensor* idd = ggml_new_tensor_2d(cd, GGML_TYPE_I32, n_ids, 1);
        ggml_backend_buffer_t db = ggml_backend_alloc_ctx_tensors_from_buft(cd, vbuft);
        if (!hb || !db) { printf("  проба: буферы не выделились\n"); return; }
        ggml_backend_tensor_set(wd, wh->data, 0, ggml_nbytes(wh));
        ggml_backend_tensor_set(xd, xv.data(), 0, xv.size() * sizeof(float));
        ggml_backend_tensor_set(idd, iv.data(), 0, iv.size() * sizeof(int32_t));

        auto run = [&](ggml_context* c, ggml_tensor* w, ggml_tensor* x, ggml_tensor* ids,
                       ggml_backend_t be, ggml_backend_buffer_type_t bt,
                       std::vector<float>* out, bool with_silu) {
            ggml_init_params gp = {ggml_tensor_overhead() * 32 +
                                   ggml_graph_overhead_custom(32, false), nullptr, true};
            ggml_context* g = ggml_init(gp);
            ggml_tensor* r = ggml_mul_mat_id(g, w, x, ids);
            if (with_silu) r = ggml_silu(g, r);
            ggml_set_output(r);
            ggml_cgraph* gf = ggml_new_graph_custom(g, 32, false);
            ggml_build_forward_expand(gf, r);
            ggml_gallocr_t ga = ggml_gallocr_new(bt);
            ggml_gallocr_reserve(ga, gf);
            ggml_gallocr_alloc_graph(ga, gf);
            ggml_backend_graph_compute(be, gf);
            out->assign(std::size_t(ggml_nelements(r)), 0.0f);
            ggml_backend_tensor_get(r, out->data(), 0, ggml_nbytes(r));
            ggml_gallocr_free(ga);
            ggml_free(g);
            (void)c;
        };
        std::vector<float> a, b, a2, b2;
        run(ch, wh, xh_t, ih, cpu, cbuft, &a, false);
        run(cd, wd, xd, idd, vk, vbuft, &b, false);
        run(ch, wh, xh_t, ih, cpu, cbuft, &a2, true);
        run(cd, wd, xd, idd, vk, vbuft, &b2, true);

        // The arbiter. Neither kernel is the definition of the type: the definition is the
        // reference decoder in ggml-quants.c, so the dot product is redone here from the
        // dequantised weights in double precision and both kernels are measured against IT.
        // Without this a disagreement says only that they differ, and the decision - accept
        // the rounding, or refuse the type on the device - depends entirely on which one is
        // wrong.
        std::vector<float> ref;
        ref.resize(a.size());
        {
            ggml_type_traits_t tt = ggml_internal_get_type_traits(qt);
            std::vector<float> row;
            row.resize(std::size_t(n_embd));
            for (int j = 0; j < n_ids; ++j) {
                const int e = iv[std::size_t(j)];
                for (int r = 0; r < n_ff; ++r) {
                    const char* src = (const char*)wh->data + std::size_t(e) * wh->nb[2] +
                                      std::size_t(r) * wh->nb[1];
                    tt.to_float(src, row.data(), n_embd);
                    double acc = 0.0;
                    for (int i = 0; i < n_embd; ++i) {
                        acc += double(row[std::size_t(i)]) * double(xv[std::size_t(i)]);
                    }
                    ref[std::size_t(j) * std::size_t(n_ff) + std::size_t(r)] = float(acc);
                }
            }
        }
        // Noise or bias. A best-fit scale of 1.000 says the kernel is simply less precise; a
        // scale away from one says it decodes the type differently, which is a different
        // problem with a different owner.
        double sc_num = 0.0, sc_den = 0.0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            sc_num += double(a[i]) * double(ref[i]);
            sc_den += double(ref[i]) * double(ref[i]);
        }
        printf("  %-8s против эталонного декодера (двойная точность): CPU %.3e, Vulkan %.3e"
               "  (масштаб CPU %.6f)\n",
               ggml_type_name(qt), rel_l2(a.data(), ref.data(), ref.size()),
               rel_l2(b.data(), ref.data(), ref.size()),
               sc_den > 0.0 ? sc_num / sc_den : 0.0);
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            dot += double(a[i]) * double(b[i]);
            na += double(a[i]) * double(a[i]);
            nb += double(b[i]) * double(b[i]);
        }
        printf("  %-8s mul_mat_id: отн. L2 %.3e, косинус %.9f | +silu: отн. L2 %.3e\n",
               ggml_type_name(qt), rel_l2(b.data(), a.data(), a.size()),
               (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0,
               rel_l2(b2.data(), a2.data(), a2.size()));

        ggml_backend_buffer_free(db);
        ggml_backend_buffer_free(hb);
        ggml_free(cd);
        ggml_free(ch);
    }
    ggml_backend_free(vk);
    ggml_backend_free(cpu);
}

}  // namespace

int gpu_experts_selftest(int threads) {
    op_probe(threads);
    // Small enough to run beside anything, large enough that the packing is the real packing:
    // forty-eight layers at twelve resident experts of 0.80 MiB is 459 MiB, one buffer, over
    // the 256 MiB BAR ceiling - which is the constraint being tested. IQ4_XS wants rows that
    // are a multiple of 256, so 2048 and 256 rather than round numbers.
    const int n_layer = 48, n_expert = 16, n_used = 8, n_embd = 2048, n_ff = 256, cap = 12;
    const ggml_type qt = GGML_TYPE_IQ4_XS;

    printf("=== самопроверка резидентных экспертов на GPU ===\n");
    printf("  %d слоёв, %d экспертов, top-%d, n_embd %d, n_ff %d, резидентных %d, тип %s\n",
           n_layer, n_expert, n_used, n_embd, n_ff, cap, ggml_type_name(qt));

    // ---- synthetic weights on the host ---------------------------------------------------
    ggml_init_params wip = {ggml_tensor_overhead() * std::size_t(n_layer) * 3 + 8192, nullptr,
                            true};
    ggml_context* cw = ggml_init(wip);
    if (!cw) { printf("  ggml_init не удался\n"); return 1; }
    std::vector<ggml_tensor*> up, gate, down;
    up.resize(std::size_t(n_layer));
    gate.resize(std::size_t(n_layer));
    down.resize(std::size_t(n_layer));
    for (int il = 0; il < n_layer; ++il) {
        up[std::size_t(il)]   = ggml_new_tensor_3d(cw, qt, n_embd, n_ff, n_expert);
        gate[std::size_t(il)] = ggml_new_tensor_3d(cw, qt, n_embd, n_ff, n_expert);
        down[std::size_t(il)] = ggml_new_tensor_3d(cw, qt, n_ff, n_embd, n_expert);
    }
    ggml_backend_buffer_t wbuf =
        ggml_backend_alloc_ctx_tensors_from_buft(cw, ggml_backend_cpu_buffer_type());
    if (!wbuf) { printf("  веса не выделились\n"); ggml_free(cw); return 1; }
    printf("  синтетические веса на хосте: %.1f МиБ\n",
           double(ggml_backend_buffer_get_size(wbuf)) / 1048576.0);

    {
        uint32_t seed = 0x9e3779b9u;
        std::vector<float> tmp;
        auto fill = [&](ggml_tensor* t, int il) {
            const int64_t rows = t->ne[1] * t->ne[2];
            const int64_t n0   = t->ne[0];
            tmp.assign(std::size_t(rows * n0), 0.0f);
            for (int64_t i = 0; i < rows * n0; ++i) {
                // Different every layer and every expert, so a slot map that fetched the
                // wrong layer or the wrong expert cannot accidentally agree.
                tmp[std::size_t(i)] = xnorm(seed, 0.02f + 0.0005f * float(il));
            }
            ggml_quantize_chunk(t->type, tmp.data(), t->data, 0, rows, n0, nullptr, nullptr);
        };
        for (int il = 0; il < n_layer; ++il) {
            fill(up[std::size_t(il)], il);
            fill(gate[std::size_t(il)], il);
            fill(down[std::size_t(il)], il);
        }
    }

    // ---- a real resident set, warmed on synthetic routing --------------------------------
    ResidentParams rp;
    rp.n_layers = n_layer; rp.n_experts = n_expert; rp.n_used = n_used;
    rp.capacity = cap; rp.window = 64; rp.period = 3; rp.budget = 8; rp.lfu = true;
    std::string rerr;
    if (!rp.validate(&rerr)) { printf("  набор отвергнут: %s\n", rerr.c_str()); return 1; }
    ResidentSet rs(rp);
    {
        uint32_t seed = 12345u;
        std::vector<int32_t> ids;
        ids.resize(std::size_t(n_used));
        for (int t = 0; t < 96; ++t) {
            for (int il = 0; il < n_layer; ++il) {
                for (int j = 0; j < n_used; ++j) {
                    int e;
                    bool dup;
                    do {
                        e = int(xrand(seed) % uint32_t(n_expert));
                        dup = false;
                        for (int m = 0; m < j; ++m) if (ids[std::size_t(m)] == e) dup = true;
                    } while (dup);
                    ids[std::size_t(j)] = e;
                }
                rs.observe(il, ids.data(), n_used);
            }
            rs.end_token();
        }
    }
    printf("  набор прогрет: в слое 0 резидентны %d из %d\n", rs.n_resident(0), n_expert);

    // ---- the module ----------------------------------------------------------------------
    GpuExpertsConfig gc;
    gc.n_layers = n_layer; gc.n_experts = n_expert; gc.n_used = n_used;
    gc.n_embd = n_embd; gc.capacity = cap;
    // check on, so the self-test drives the same join node --gpu-experts-check drives in the
    // engine - ggml_map_custom2, with the CPU's own resident half as a second input - rather
    // than only the cheap one, and so the module's per-slot accounting is exercised and can be
    // cross-checked against this function's independent count of the same thing.
    gc.check = true;
    gc.reserve = 256u * 1024u * 1024u;
    GpuExperts gx;
    std::string gerr;
    if (!gx.init(gc, up.data(), gate.data(), down.data(), &gerr)) {
        printf("  инициализация не удалась: %s\n", gerr.c_str());
        ggml_backend_buffer_free(wbuf);
        ggml_free(cw);
        return 1;
    }
    for (const GpuExpertsBuffer& b : gx.buffers()) {
        printf("  буфер: слои %2d..%-2d  %8.2f МиБ  %s\n", b.first_layer,
               b.first_layer + b.n_layers - 1, double(b.bytes) / 1048576.0,
               b.over_bar ? "> 256 МиБ — не BAR" : "<= 256 МиБ — МОГ СЕСТЬ В BAR");
    }
    gx.sync_slots(rs);
    gx.drain();
    if (!gx.failure().empty()) {
        printf("  поток устройства упал: %s\n", gx.failure().c_str());
        return 1;
    }
    printf("  заливка: %llu экспертов, %.3f ГБ\n",
           (unsigned long long)gx.stats().promotions,
           double(gx.stats().promo_bytes) / 1e9);

    // Every slot of every layer, byte for byte against the model tensor it came from.
    {
        int probed = 0, bad = 0;
        std::string e;
        for (int il = 0; il < n_layer; ++il) {
            for (int s = 0; s < cap; ++s) {
                if (!gx.verify_slot(il, s, &e)) {
                    if (e == "слот пуст") continue;
                    if (!bad) printf("  РАСХОЖДЕНИЕ: %s\n", e.c_str());
                    ++bad;
                }
                ++probed;
            }
        }
        printf("  видеопамять против модели: %d слотов, расхождений %d\n", probed, bad);
        if (bad) return 1;
    }

    // ---- per layer: unsplit, split on the CPU, split with the device ---------------------
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);
    ggml_backend_buffer_type_t cbuft = ggml_backend_cpu_buffer_type();

    const std::size_t nslot = std::size_t(n_embd);
    const std::size_t nall  = nslot * std::size_t(n_used);
    std::vector<float> v_full(nall), v_rescpu(nall), v_oth(nall), v_resgpu(nall), v_sum(nall);
    std::vector<float> xh;
    xh.resize(std::size_t(n_embd));
    std::vector<int32_t> i_full, i_res, i_oth;
    i_full.resize(std::size_t(n_used));
    i_res.resize(std::size_t(n_used));
    i_oth.resize(std::size_t(n_used));

    int layers_ok = 0, split_exact = 0, zero_ok = 0, sum_exact = 0, slots = 0;
    int fail_cpu_split = 0, fail_zero = 0, fail_sum = 0, fail_partition = 0;
    double worst_dev = 0.0, worst_tot = 0.0, worst_cos = 1.0;
    double ref_cpu = 0.0, ref_gpu = 0.0;
    int ref_layers = 0;
    std::vector<double> rels;
    uint32_t seed = 777u;

    for (int il = 0; il < n_layer; ++il) {
        for (int i = 0; i < n_embd; ++i) {
            xh[std::size_t(i)] = xnorm(seed, 1.0f);
        }
        for (int j = 0; j < n_used; ++j) {
            int e; bool dup;
            do {
                e = int(xrand(seed) % uint32_t(n_expert));
                dup = false;
                for (int m = 0; m < j; ++m) if (i_full[std::size_t(m)] == e) dup = true;
            } while (dup);
            i_full[std::size_t(j)] = e;
        }
        // One slot dropped outright on some layers, the way expert reduction drops one: it
        // must end up -1 in BOTH halves and contribute an exact zero from each.
        if (il % 5 == 3) i_full[std::size_t(n_used - 1)] = -1;
        for (int j = 0; j < n_used; ++j) {
            const int32_t e = i_full[std::size_t(j)];
            const bool res = e >= 0 && rs.is_resident(il, e);
            i_res[std::size_t(j)] = res ? e : -1;
            i_oth[std::size_t(j)] = (e >= 0 && !res) ? e : -1;
        }

        ggml_init_params ip = {ggml_tensor_overhead() * 128 +
                               ggml_graph_overhead_custom(128, false), nullptr, true};
        ggml_context* c = ggml_init(ip);
        ggml_tensor* xin = ggml_new_tensor_3d(c, GGML_TYPE_F32, n_embd, 1, 1);
        ggml_tensor* tf  = ggml_new_tensor_2d(c, GGML_TYPE_I32, n_used, 1);
        ggml_tensor* tr  = ggml_new_tensor_2d(c, GGML_TYPE_I32, n_used, 1);
        ggml_tensor* to  = ggml_new_tensor_2d(c, GGML_TYPE_I32, n_used, 1);
        ggml_backend_buffer_t ib = ggml_backend_alloc_ctx_tensors_from_buft(c, cbuft);
        if (!ib) { printf("  входы слоя %d не выделились\n", il); return 1; }

        ggml_cgraph* gf = ggml_new_graph_custom(c, 128, false);
        auto experts = [&](ggml_tensor* ids) {
            ggml_tensor* u = ggml_mul_mat_id(c, up[std::size_t(il)], xin, ids);
            ggml_tensor* g = ggml_silu(c, ggml_mul_mat_id(c, gate[std::size_t(il)], xin, ids));
            return ggml_mul_mat_id(c, down[std::size_t(il)], ggml_mul(c, u, g), ids);
        };
        ggml_tensor* o_full   = experts(tf);
        ggml_tensor* o_rescpu = experts(tr);
        ggml_tensor* to2      = gx.fork(c, il, to, xin, tr);
        ggml_tensor* o_oth    = experts(to2);
        ggml_tensor* o_resgpu = gx.join(c, il, o_oth, o_rescpu);
        ggml_tensor* o_sum    = ggml_add(c, o_resgpu, o_oth);
        ggml_set_output(o_full); ggml_set_output(o_rescpu); ggml_set_output(o_oth);
        ggml_set_output(o_resgpu); ggml_set_output(o_sum);
        ggml_build_forward_expand(gf, o_full);
        ggml_build_forward_expand(gf, o_rescpu);
        ggml_build_forward_expand(gf, o_sum);

        ggml_gallocr_t ga = ggml_gallocr_new(cbuft);
        if (!ga || !ggml_gallocr_reserve(ga, gf) || !ggml_gallocr_alloc_graph(ga, gf)) {
            printf("  граф слоя %d не разместился\n", il);
            return 1;
        }
        ggml_backend_tensor_set(xin, xh.data(), 0, std::size_t(n_embd) * sizeof(float));
        ggml_backend_tensor_set(tf, i_full.data(), 0, std::size_t(n_used) * sizeof(int32_t));
        ggml_backend_tensor_set(tr, i_res.data(), 0, std::size_t(n_used) * sizeof(int32_t));
        ggml_backend_tensor_set(to, i_oth.data(), 0, std::size_t(n_used) * sizeof(int32_t));
        ggml_backend_graph_compute(cpu, gf);
        if (!gx.failure().empty()) {
            printf("  поток устройства упал на слое %d: %s\n", il, gx.failure().c_str());
            return 1;
        }
        ggml_backend_tensor_get(o_full,   v_full.data(),   0, nall * sizeof(float));
        ggml_backend_tensor_get(o_rescpu, v_rescpu.data(), 0, nall * sizeof(float));
        ggml_backend_tensor_get(o_oth,    v_oth.data(),    0, nall * sizeof(float));
        ggml_backend_tensor_get(o_resgpu, v_resgpu.data(), 0, nall * sizeof(float));
        ggml_backend_tensor_get(o_sum,    v_sum.data(),    0, nall * sizeof(float));

        for (int s = 0; s < n_used; ++s) {
            const std::size_t off = std::size_t(s) * nslot;
            const bool res = i_res[std::size_t(s)] >= 0;
            const bool oth = i_oth[std::size_t(s)] >= 0;
            ++slots;
            // 1. The CPU-only split, unchanged: exactly one half owns each live slot, and the
            //    slot the router dropped is owned by neither.
            const bool zr = all_zero_bits(v_rescpu.data() + off, nslot);
            const bool zo = all_zero_bits(v_oth.data() + off, nslot);
            if (res == zr || oth == zo) ++fail_partition;
            // 2. The device's half is exactly zero in every slot it does not own. This is the
            //    property the Vulkan shader does NOT give for free - it reads an id of -1 as
            //    0xffffffff and fetches garbage - and it is why the id list is compacted and
            //    the result scattered back rather than passed through with holes.
            if (all_zero_bits(v_resgpu.data() + off, nslot) == res) ++fail_zero;
            else ++zero_ok;
            // 3. And therefore the per-slot sum is bit-identical to whichever half owned the
            //    slot: x + 0.0f and 0.0f + x, never a reassociated pair of partial sums.
            const float* want = res ? v_resgpu.data() + off
                                    : (oth ? v_oth.data() + off : nullptr);
            if (want) {
                if (same_bits(v_sum.data() + off, want, nslot)) ++sum_exact;
                else ++fail_sum;
            } else if (!all_zero_bits(v_sum.data() + off, nslot)) {
                ++fail_sum;
            } else {
                ++sum_exact;
            }
            // 4. What the device actually computed, against the same half computed by the CPU
            //    kernel. Never zero and never expected to be: two kernels, two summation
            //    orders. A zero here would mean the device did not run.
            if (res) {
                const double r = rel_l2(v_resgpu.data() + off, v_rescpu.data() + off, nslot);
                worst_dev = std::max(worst_dev, r);
                rels.push_back(r);
                // Direction, separately from magnitude. A kernel that rounds differently
                // stays parallel to the one it is checked against; a kernel that read the
                // wrong expert, the wrong layer or the wrong slot does not, and the two are
                // indistinguishable in a relative L2 alone.
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (std::size_t i = 0; i < nslot; ++i) {
                    const double a = v_resgpu[off + i], b = v_rescpu[off + i];
                    dot += a * b; na += a * a; nb += b * b;
                }
                const double cs = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 0.0;
                worst_cos = std::min(worst_cos, cs);
            }
        }
        // 5. The CPU-only split against the unsplit path, bit for bit. Unchanged by any of
        //    this, and re-checked here because it is the claim everything else is measured
        //    against.
        {
            std::vector<float> cpu_sum(nall);
            for (std::size_t i = 0; i < nall; ++i) {
                cpu_sum[i] = v_rescpu[i] + v_oth[i];
            }
            if (same_bits(cpu_sum.data(), v_full.data(), nall)) ++split_exact;
            else ++fail_cpu_split;
        }
        worst_tot = std::max(worst_tot, rel_l2(v_sum.data(), v_full.data(), nall));

        // Every sixth layer, the arbiter: the same feed-forward recomputed from the reference
        // decoder in double precision. Neither kernel defines the answer, so "the device
        // disagrees with the CPU" is not a verdict until something says which of them moved.
        // On this model's own quantisation it is the CPU that moves: its mul_mat_id decodes
        // through an integer dot with a quantised activation, and against the reference that
        // is worth about 5e-2 on IQ4_XS and 5e-3 on Q6_K, while the Vulkan shader lands at
        // 1e-7. So the split is expected to come out MORE accurate with the device in it, not
        // less, and this is where that gets checked rather than asserted.
        if (il % 6 == 0) {
            std::vector<double> oref;
            oref.assign(nall, 0.0);
            std::vector<float> rw;
            std::vector<double> act;
            act.resize(std::size_t(n_ff));
            for (int s = 0; s < n_used; ++s) {
                const int e = i_full[std::size_t(s)];
                if (e < 0) continue;
                auto row_dot = [&](ggml_tensor* w, int r, const double* vec, int len) {
                    rw.assign(std::size_t(len), 0.0f);
                    ggml_type_traits_t tt = ggml_internal_get_type_traits(w->type);
                    tt.to_float((const char*)w->data + std::size_t(e) * w->nb[2] +
                                    std::size_t(r) * w->nb[1],
                                rw.data(), len);
                    double acc = 0.0;
                    for (int i = 0; i < len; ++i) acc += double(rw[std::size_t(i)]) * vec[i];
                    return acc;
                };
                std::vector<double> xd;
                xd.resize(std::size_t(n_embd));
                for (int i = 0; i < n_embd; ++i) xd[std::size_t(i)] = double(xh[std::size_t(i)]);
                for (int r = 0; r < n_ff; ++r) {
                    const double u = row_dot(up[std::size_t(il)], r, xd.data(), n_embd);
                    const double g = row_dot(gate[std::size_t(il)], r, xd.data(), n_embd);
                    act[std::size_t(r)] = u * (g / (1.0 + std::exp(-g)));
                }
                for (int i = 0; i < n_embd; ++i) {
                    oref[std::size_t(s) * nslot + std::size_t(i)] =
                        row_dot(down[std::size_t(il)], i, act.data(), n_ff);
                }
            }
            double n_cpu = 0.0, n_gpu = 0.0, den = 0.0;
            for (std::size_t i = 0; i < nall; ++i) {
                const double dc = double(v_full[i]) - oref[i];
                const double dg = double(v_sum[i]) - oref[i];
                n_cpu += dc * dc; n_gpu += dg * dg; den += oref[i] * oref[i];
            }
            if (den > 0.0) {
                ref_cpu = std::max(ref_cpu, std::sqrt(n_cpu / den));
                ref_gpu = std::max(ref_gpu, std::sqrt(n_gpu / den));
                ++ref_layers;
            }
        }
        ++layers_ok;
        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(ib);
        ggml_free(c);
    }

    // ---- deferred activation, model-free --------------------------------------------------
    //
    // Everything above ran with deferred activation OFF, which is the configuration that was
    // there before it existed and the one this self-test has always gated. This phase turns it
    // on and checks the one hazard the design must not have: a device slot holding bytes that
    // no longer belong to the expert the residency mask claims.
    //
    // Three invariants, checked once per simulated token, BEFORE anything would be dispatched:
    //
    //   A. resident and pending are disjoint. A bit cannot be both.
    //   B. every RESIDENT expert has a slot, that slot names it back, and that slot is not
    //      still owed to the link. This is the hazard, stated positively.
    //   C. resident + pending never exceeds the capacity. Pending occupies its slot, so
    //      counting only the resident ones would let a refresh over-commit video memory.
    //
    // Nothing is drained inside the loop: the point is to observe the set while transfers are
    // genuinely in flight. The byte-level comparison is done once at the end, after a drain and
    // a final round of confirmations, over every slot the mask then claims.
    int def_tokens = 0, def_bad_disjoint = 0, def_bad_claim = 0, def_overcommit = 0;
    int def_bad_bytes = 0, def_probed = 0, def_pend_max = 0;
    uint64_t def_req = 0, def_act = 0;
    double   def_land = 0.0;
    {
        const ResidentStats base = rs.stats();
        rs.set_deferred(true);
        std::vector<uint32_t> landed;
        std::vector<int32_t>  ids;
        ids.resize(std::size_t(n_used));
        uint32_t seed = 0xC0FFEEu;
        for (int t = 0; t < 90; ++t) {
            // 1. Confirmations first, exactly as the engine does them - before the mask that
            //    serves this token is written.
            gx.take_landed(&landed);
            for (uint32_t v : landed) rs.activate(int(v >> 16), int(v & 0xffff));

            // 2. The invariants.
            for (int il = 0; il < n_layer; ++il) {
                int claimed = 0, pend = 0;
                for (int e = 0; e < n_expert; ++e) {
                    const bool r = rs.is_resident(il, e);
                    const bool p = rs.is_pending(il, e);
                    if (r && p) ++def_bad_disjoint;                       // A
                    if (r || p) ++claimed;
                    if (p) ++pend;
                    if (!r) continue;
                    const int s = gx.slot_of(il, e);                       // B
                    if (s < 0 || gx.expert_at(il, s) != e || gx.slot_dirty(il, s)) {
                        ++def_bad_claim;
                    }
                }
                if (claimed > cap) ++def_overcommit;                       // C
                def_pend_max = std::max(def_pend_max, pend);
            }

            // 3. One token's routing, then the refresh and the slot-map sync the engine does.
            for (int il = 0; il < n_layer; ++il) {
                for (int j = 0; j < n_used; ++j) {
                    int e; bool dup;
                    do {
                        e = int(xrand(seed) % uint32_t(n_expert));
                        dup = false;
                        for (int m = 0; m < j; ++m) if (ids[std::size_t(m)] == e) dup = true;
                    } while (dup);
                    ids[std::size_t(j)] = e;
                }
                rs.observe(il, ids.data(), n_used);
            }
            rs.end_token();
            gx.sync_slots(rs);
            ++def_tokens;
        }
        // Settle: let the link finish, confirm what landed, then compare bytes for every slot
        // the mask now claims. This is the same comparison the fill above passed, but over a
        // set that has been churned by ninety refreshes rather than uploaded once.
        gx.drain();
        gx.take_landed(&landed);
        for (uint32_t v : landed) rs.activate(int(v >> 16), int(v & 0xffff));
        std::string ve;
        for (int il = 0; il < n_layer; ++il) {
            for (int e = 0; e < n_expert; ++e) {
                if (!rs.is_resident(il, e)) continue;
                const int s = gx.slot_of(il, e);
                if (s < 0 || gx.expert_at(il, s) != e) { ++def_bad_claim; continue; }
                ++def_probed;
                if (!gx.verify_slot(il, s, &ve)) ++def_bad_bytes;
            }
        }
        const ResidentStats d = rs.stats().since(base);
        def_req  = d.promotions;
        def_act  = d.activations;
        def_land = d.tokens_to_land();
    }

    printf("\n  слоёв проверено                                     : %d\n", layers_ok);
    printf("  слотов                                              : %d\n", slots);
    printf("  расщепление на CPU == нерасщеплённому, бит в бит     : %d/%d%s\n",
           split_exact, layers_ok, fail_cpu_split ? "  ПРОВАЛ" : "");
    printf("  половина устройства ровно ноль в чужих слотах        : %d/%d%s\n",
           zero_ok, slots, fail_zero ? "  ПРОВАЛ" : "");
    printf("  сумма послотно бит в бит равна владельцу слота       : %d/%d%s\n",
           sum_exact, slots, fail_sum ? "  ПРОВАЛ" : "");
    printf("  разбиение слотов между половинами                   : %s\n",
           fail_partition ? "ПРОВАЛ" : "верное");
    std::sort(rels.begin(), rels.end());
    const double med = rels.empty() ? 0.0 : rels[rels.size() / 2];
    const double p90 = rels.empty() ? 0.0 : rels[rels.size() * 9 / 10];
    printf("  устройство против той же половины на CPU, отн. L2    :\n");
    printf("      медиана %.3e, p90 %.3e, максимум %.3e по %zu слотам\n",
           med, p90, worst_dev, rels.size());
    printf("      худший косинус между половинами                  : %.9f\n", worst_cos);
    printf("  вся сумма против нерасщеплённого пути, отн. L2       : %.3e\n", worst_tot);
    printf("  против эталонного декодера в двойной точности (%d слоёв):\n", ref_layers);
    printf("      нерасщеплённый путь, целиком на CPU              : %.3e\n", ref_cpu);
    printf("      расщеплённый, резидентная половина на устройстве : %.3e\n", ref_gpu);
    printf("      это судья: mul_mat_id на устройстве точнее ядра CPU этой сборки,\n"
           "      поэтому расщепление обязано быть НЕ ХУЖЕ нерасщеплённого пути\n");
    printf("  экспертов запущено на устройстве                     : %llu за %llu слоёв\n",
           (unsigned long long)gx.stats().experts, (unsigned long long)gx.stats().layers);
    printf("  собственный счёт модуля (--gpu-experts-check)        : слотов %llu, "
           "ненулевых в чужих %llu, нулевых в своих %llu\n",
           (unsigned long long)gx.stats().checked, (unsigned long long)gx.stats().zero_bad,
           (unsigned long long)gx.stats().owned_bad);

    printf("  отложенная активация, %d токенов:\n", def_tokens);
    printf("      резидентный и в полёте — непересекающиеся             : %s\n",
           def_bad_disjoint ? "ПРОВАЛ" : "да");
    printf("      у каждого резидентного свой слот, и он не грязный     : %s\n",
           def_bad_claim ? "ПРОВАЛ" : "да");
    printf("      резидентных + в полёте никогда больше ёмкости         : %s (пик в полёте %d "
           "из %d)\n", def_overcommit ? "ПРОВАЛ" : "да", def_pend_max, cap);
    printf("      байты слотов после %d обновлений                      : %d слотов, "
           "расхождений %d\n", def_tokens, def_probed, def_bad_bytes);
    printf("      запрошено %llu, подтверждено %llu, доходит за %.2f токена (период %d)\n",
           (unsigned long long)def_req, (unsigned long long)def_act, def_land, rp.period);

    // What is gated and what is only reported.
    //
    // Gated: every bit-exact structural claim; the device having actually computed something
    // in the right direction (the cosine, which is what collapses if a slot, a layer or an
    // expert was fetched wrong, where a relative L2 alone would not distinguish that from
    // rounding); and the split being no further from the reference decoder than the unsplit
    // CPU path.
    //
    // Reported, not gated: the device-against-CPU difference. That is a property of two
    // kernels rather than of this code, and on IQ4_XS it is dominated by the CPU's own
    // integer dot with a quantised activation - the arbiter above says so. Its relative L2 is
    // also inflated by cancellation, because a dot of two uncorrelated vectors of length N is
    // sqrt(N) smaller than the scale of its own terms, so per-element noise shows up
    // forty-five-fold. Hence the median rather than the maximum even where it is printed.
    const bool ok = !fail_cpu_split && !fail_zero && !fail_sum && !fail_partition &&
                    med > 0.0 && worst_cos > 0.95 && gx.stats().no_slot == 0 &&
                    ref_layers > 0 && ref_gpu <= ref_cpu * 1.05 &&
                    gx.stats().checked > 0 && gx.stats().zero_bad == 0 &&
                    gx.stats().owned_bad == 0 &&
                    // deferred activation: the invariants are gated, the latency is reported
                    def_bad_disjoint == 0 && def_bad_claim == 0 && def_overcommit == 0 &&
                    def_bad_bytes == 0 && def_probed > 0 && def_act > 0;
    printf("\n  ИТОГ: %s\n", ok ? "ПРОЙДЕНО" : "ПРОВАЛ");

    gx.shutdown();
    ggml_backend_free(cpu);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cw);
    return ok ? 0 : 1;
}

}  // namespace memex
