// memex-vk: standalone probe of the Vulkan backend on this device.
//
// Answers three things, all by measurement, none by estimate:
//   1. which gathered/indexed matmul shapes the Vulkan backend actually accepts,
//      and how many vkCmdDispatch calls each one costs;
//   2. dispatch + fence + PCIe round-trip + VRAM-read timings at 48 layers;
//   3. how much device memory is free, how much the backend eats before any
//      model tensor is allocated, and the largest single allocation the driver
//      will hand out.
//
// Deliberately synthetic: no model is loaded, total VRAM touched stays a few
// hundred MB.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// timing helpers
// ---------------------------------------------------------------------------

static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

static void row(const char * label, double total_ms, int n) {
    printf("  %-58s %9.3f ms total   %9.1f us / iter\n", label, total_ms, total_ms * 1000.0 / n);
}

// ---------------------------------------------------------------------------
// part A / part 3: raw Vulkan device query, independent of ggml
// ---------------------------------------------------------------------------

struct raw_vk {
    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys     = VK_NULL_HANDLE;
    VkDevice         dev      = VK_NULL_HANDLE;
    uint32_t         queue_family = 0;
    bool             has_budget = false;
    VkPhysicalDeviceMemoryProperties memprops{};
};

static bool raw_vk_init(raw_vk & r) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "memex-vk";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    if (vkCreateInstance(&ici, nullptr, &r.instance) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed\n");
        return false;
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(r.instance, &n, nullptr);
    if (n == 0) { fprintf(stderr, "no vulkan devices\n"); return false; }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(r.instance, &n, devs.data());
    r.phys = devs[0];

    // does the device expose VK_EXT_memory_budget?
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne);
    vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &ne, exts.data());
    for (const auto & e : exts) {
        if (strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            r.has_budget = true;
        }
    }

    vkGetPhysicalDeviceMemoryProperties(r.phys, &r.memprops);

    // a logical device, needed only for the allocation probe
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nq, qfp.data());
    for (uint32_t i = 0; i < nq; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { r.queue_family = i; break; }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = r.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    std::vector<const char *> dev_exts;
    if (r.has_budget) dev_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t) dev_exts.size();
    dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();

    if (vkCreateDevice(r.phys, &dci, nullptr, &r.dev) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateDevice failed\n");
        return false;
    }
    return true;
}

static void raw_vk_print_budget(raw_vk & r, const char * when) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

    VkPhysicalDeviceMemoryProperties2 mp2{};
    mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mp2.pNext = r.has_budget ? (void *) &budget : nullptr;

    vkGetPhysicalDeviceMemoryProperties2(r.phys, &mp2);

    printf("  --- memory heaps (%s) ---\n", when);
    for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; i++) {
        const VkMemoryHeap & h = mp2.memoryProperties.memoryHeaps[i];
        const bool device_local = (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        printf("  heap %u  %-13s size %8.2f MiB", i, device_local ? "DEVICE_LOCAL" : "host", h.size / 1048576.0);
        if (r.has_budget) {
            printf("   budget %8.2f MiB   usage %8.2f MiB",
                   budget.heapBudget[i] / 1048576.0, budget.heapUsage[i] / 1048576.0);
        }
        printf("\n");
    }
}

// largest single vkAllocateMemory the driver will accept on the given heap
// heap usage in MiB, for spotting a silent spill of "device" buffers into host RAM
static void raw_vk_usage(raw_vk & r, double & vram_mib, double & host_mib) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 mp2{};
    mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mp2.pNext = r.has_budget ? (void *) &budget : nullptr;
    vkGetPhysicalDeviceMemoryProperties2(r.phys, &mp2);
    vram_mib = 0; host_mib = 0;
    for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; i++) {
        const double u = budget.heapUsage[i] / 1048576.0;
        if (mp2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram_mib += u;
        else host_mib += u;
    }
}

static size_t raw_vk_probe_max_alloc(raw_vk & r, uint32_t heap_index, size_t hi_bytes) {
    // find a memory type on that heap
    int type_index = -1;
    for (uint32_t i = 0; i < r.memprops.memoryTypeCount; i++) {
        if (r.memprops.memoryTypes[i].heapIndex == heap_index &&
            (r.memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type_index = (int) i;
            break;
        }
    }
    if (type_index < 0) return 0;

    size_t lo = 0;
    size_t hi = hi_bytes;
    const size_t step = 16ull * 1024 * 1024;

    while (hi - lo > step) {
        size_t mid = lo + (hi - lo) / 2;
        mid = (mid / step) * step;
        if (mid <= lo) break;

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mid;
        ai.memoryTypeIndex = (uint32_t) type_index;

        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkResult res = vkAllocateMemory(r.dev, &ai, nullptr, &mem);
        if (res == VK_SUCCESS) {
            vkFreeMemory(r.dev, mem, nullptr);
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// largest single VkBuffer the driver will create + back with memory
static size_t raw_vk_probe_max_buffer(raw_vk & r, size_t hi_bytes) {
    size_t lo = 0;
    size_t hi = hi_bytes;
    const size_t step = 16ull * 1024 * 1024;

    while (hi - lo > step) {
        size_t mid = lo + (hi - lo) / 2;
        mid = (mid / step) * step;
        if (mid <= lo) break;

        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = mid;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buf = VK_NULL_HANDLE;
        if (vkCreateBuffer(r.dev, &bci, nullptr, &buf) != VK_SUCCESS) {
            hi = mid;
            continue;
        }

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(r.dev, buf, &req);

        int type_index = -1;
        for (uint32_t i = 0; i < r.memprops.memoryTypeCount; i++) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (r.memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                !(r.memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                type_index = (int) i;
                break;
            }
        }
        if (type_index < 0) { vkDestroyBuffer(r.dev, buf, nullptr); hi = mid; continue; }

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = (uint32_t) type_index;

        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkResult res = vkAllocateMemory(r.dev, &ai, nullptr, &mem);
        if (res == VK_SUCCESS && vkBindBufferMemory(r.dev, buf, mem, 0) == VK_SUCCESS) {
            lo = mid;
        } else {
            hi = mid;
        }
        if (mem != VK_NULL_HANDLE) vkFreeMemory(r.dev, mem, nullptr);
        vkDestroyBuffer(r.dev, buf, nullptr);
    }
    return lo;
}

// ---------------------------------------------------------------------------
// part 1: which shapes does the backend accept
// ---------------------------------------------------------------------------

static struct ggml_context * meta_ctx() {
    struct ggml_init_params p = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_init(p);
}

static void probe_supports(ggml_backend_t backend) {
    const int64_t n_embd   = 5120;   // hidden size, DeepSeek/Qwen3-class
    const int64_t n_ff_exp = 1536;   // per-expert ff
    const int64_t n_res    = 32;     // resident experts stacked in one tensor
    const int64_t n_used   = 8;      // experts selected per token

    const ggml_type types[] = {
        GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL,
    };
    const char * type_names[] = { "F16", "Q4_0", "Q8_0", "Q4_K", "Q6_K", "IQ4_XS", "IQ4_NL" };
    const int n_types = (int) (sizeof(types) / sizeof(types[0]));

    printf("\n=== Q1: op/type support matrix (ggml_backend_supports_op on the live device) ===\n");
    printf("  %-10s %-12s %-12s %-16s %-16s\n", "type", "MUL_MAT_2D", "GET_ROWS", "MUL_MAT_ID(t=1)", "MUL_MAT_3D_batch");

    for (int t = 0; t < n_types; t++) {
        struct ggml_context * ctx = meta_ctx();

        // plain 2-D matmul, one expert
        struct ggml_tensor * w2  = ggml_new_tensor_2d(ctx, types[t], n_embd, n_ff_exp);
        struct ggml_tensor * x1  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
        struct ggml_tensor * mm2 = ggml_mul_mat(ctx, w2, x1);

        // get_rows on the stacked resident tensor
        struct ggml_tensor * wg  = ggml_new_tensor_2d(ctx, types[t], n_embd, n_ff_exp * n_res);
        struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ff_exp * n_used);
        struct ggml_tensor * gr  = ggml_get_rows(ctx, wg, idx);

        // mul_mat_id, single token, resident experts stacked on dim 2
        struct ggml_tensor * w3  = ggml_new_tensor_3d(ctx, types[t], n_embd, n_ff_exp, n_res);
        struct ggml_tensor * xb  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
        struct ggml_tensor * mmid = ggml_mul_mat_id(ctx, w3, xb, ids);

        // plain 3-D batched matmul: batch == resident experts, no gather
        struct ggml_tensor * w3b = ggml_new_tensor_3d(ctx, types[t], n_embd, n_ff_exp, n_used);
        struct ggml_tensor * x3b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_used);
        struct ggml_tensor * mm3 = ggml_mul_mat(ctx, w3b, x3b);

        printf("  %-10s %-12s %-12s %-16s %-16s\n",
               type_names[t],
               ggml_backend_supports_op(backend, mm2)  ? "yes" : "NO",
               ggml_backend_supports_op(backend, gr)   ? "yes" : "NO",
               ggml_backend_supports_op(backend, mmid) ? "yes" : "NO",
               ggml_backend_supports_op(backend, mm3)  ? "yes" : "NO");

        ggml_free(ctx);
    }

    // mul_mat_id at a few token counts, to see whether the shared-memory path matters
    printf("\n  MUL_MAT_ID(Q6_K) at varying token counts (t=1 uses mul_mat_vec_id, t>1 uses mul_mm_id):\n");
    for (int64_t nt : { (int64_t)1, (int64_t)2, (int64_t)8, (int64_t)64, (int64_t)512 }) {
        struct ggml_context * ctx = meta_ctx();
        struct ggml_tensor * w3  = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff_exp, n_res);
        struct ggml_tensor * xb  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, nt);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, nt);
        struct ggml_tensor * mmid = ggml_mul_mat_id(ctx, w3, xb, ids);
        printf("    n_tokens = %4lld : %s\n", (long long) nt,
               ggml_backend_supports_op(backend, mmid) ? "supported" : "NOT supported");
        ggml_free(ctx);
    }
}

// ---------------------------------------------------------------------------
// part 2: timings
// ---------------------------------------------------------------------------

static const int N_LAYERS = 48;

struct bench_ctx {
    struct ggml_context   * ctx = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    std::vector<struct ggml_tensor *> outs;
    std::vector<struct ggml_tensor *> weights;
    struct ggml_tensor * act = nullptr;   // 16 KiB activation vector
};

static void bench_all(ggml_backend_t backend) {
    // 48 distinct Q6_K weights of ~6.56 MiB each: 4096 x 2048 = 8,388,608 weights
    // -> 32768 Q6_K blocks (210 B per 256 weights) -> 6,881,280 B. That is the
    // per-layer resident-expert read the design needs.
    const int64_t W_K = 4096;
    const int64_t W_N = 2048;

    struct ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);
    const size_t GS = 256; // graph capacity

    std::vector<struct ggml_tensor *> W(N_LAYERS);
    std::vector<struct ggml_tensor *> A(N_LAYERS);
    for (int i = 0; i < N_LAYERS; i++) {
        W[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, W_K, W_N);
        A[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);   // 16 KiB
    }
    struct ggml_tensor * xvec = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W_K, 1);
    struct ggml_tensor * act  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096); // 16 KiB

    // --- build every graph BEFORE allocating, so the result tensors get device
    //     memory too. A node whose destination has no buffer is silently skipped
    //     by the Vulkan backend, which would make every timing below meaningless.

    // 1: one trivial op per graph, 48 graphs
    std::vector<struct ggml_cgraph *> g_triv(N_LAYERS);
    for (int i = 0; i < N_LAYERS; i++) {
        g_triv[i] = ggml_new_graph_custom(ctx, GS, false);
        ggml_build_forward_expand(g_triv[i], ggml_scale(ctx, A[i], 1.0f));
    }
    // 2: 48 trivial ops in one graph
    struct ggml_cgraph * g_triv_batch = ggml_new_graph_custom(ctx, GS, false);
    for (int i = 0; i < N_LAYERS; i++) {
        ggml_build_forward_expand(g_triv_batch, ggml_scale(ctx, A[i], 1.0f));
    }
    // 3: one 6.56 MiB matmul per graph, 48 graphs
    std::vector<struct ggml_cgraph *> g_mm(N_LAYERS);
    for (int i = 0; i < N_LAYERS; i++) {
        g_mm[i] = ggml_new_graph_custom(ctx, GS, false);
        ggml_build_forward_expand(g_mm[i], ggml_mul_mat(ctx, W[i], xvec));
    }
    // 3b: all 48 matmuls in one graph
    struct ggml_cgraph * g_mm_batch = ggml_new_graph_custom(ctx, GS, false);
    for (int i = 0; i < N_LAYERS; i++) {
        ggml_build_forward_expand(g_mm_batch, ggml_mul_mat(ctx, W[i], xvec));
    }
    // 6: three matmuls per layer, all in one graph
    struct ggml_cgraph * g_mm3 = ggml_new_graph_custom(ctx, GS * 4, false);
    for (int i = 0; i < N_LAYERS; i++) {
        for (int k = 0; k < 3; k++) {
            ggml_build_forward_expand(g_mm3, ggml_mul_mat(ctx, W[i], xvec));
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) { fprintf(stderr, "failed to allocate tensors on device\n"); return; }

    printf("\n  device buffer for the benchmark tensors: %.2f MiB\n",
           ggml_backend_buffer_get_size(buf) / 1048576.0);
    printf("  one Q6_K weight = %.3f MiB, x%d = %.2f MiB\n",
           ggml_nbytes(W[0]) / 1048576.0, N_LAYERS, ggml_nbytes(W[0]) * (double) N_LAYERS / 1048576.0);

    // sanity: confirm every node in a graph really has a device buffer
    {
        int missing = 0;
        for (int i = 0; i < g_mm_batch->n_nodes; i++) {
            if (g_mm_batch->nodes[i]->buffer == nullptr) missing++;
        }
        printf("  graph nodes without a device buffer (must be 0): %d\n", missing);
    }

    std::vector<float> host(4096, 0.001f);
    ggml_backend_tensor_set(xvec, host.data(), 0, 4096 * sizeof(float));
    for (int i = 0; i < N_LAYERS; i++) {
        ggml_backend_tensor_set(A[i], host.data(), 0, 16384);
    }
    // give the weights bytes so the dequantiser does not walk into denormals
    {
        std::vector<uint8_t> wjunk(ggml_nbytes(W[0]), 0x20);
        for (int i = 0; i < N_LAYERS; i++) {
            ggml_backend_tensor_set(W[i], wjunk.data(), 0, ggml_nbytes(W[i]));
        }
    }
    ggml_backend_synchronize(backend);

    const int REPS = 20;
    printf("\n=== Q2: dispatch and synchronisation, %d layers, %d reps each ===\n", N_LAYERS, REPS);

    // ---- 1. 48 sequential trivial dispatches, a fence after each ----------
    {
        for (int i = 0; i < N_LAYERS; i++) ggml_backend_graph_compute(backend, g_triv[i]);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < N_LAYERS; i++) ggml_backend_graph_compute(backend, g_triv[i]);
        }
        row("1. 48 trivial dispatches, one fence wait each", (now_ms() - t0) / REPS, N_LAYERS);
    }

    // ---- 2. 48 trivial dispatches, one submit + one fence ----------------
    {
        ggml_backend_graph_compute(backend, g_triv_batch);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) ggml_backend_graph_compute(backend, g_triv_batch);
        row("2. 48 trivial dispatches, batched, one fence at end", (now_ms() - t0) / REPS, N_LAYERS);
    }

    // ---- 3a. 48 realistic matmuls, one fence per layer --------------------
    {
        for (int i = 0; i < N_LAYERS; i++) ggml_backend_graph_compute(backend, g_mm[i]);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < N_LAYERS; i++) ggml_backend_graph_compute(backend, g_mm[i]);
        }
        double dt = (now_ms() - t0) / REPS;
        row("3a. 48 matmuls of 6.56 MiB, one fence wait each", dt, N_LAYERS);
        printf("      effective VRAM read bandwidth: %.1f GB/s\n",
               (ggml_nbytes(W[0]) * (double) N_LAYERS) / (dt * 1e-3) / 1e9);
    }

    // ---- 3b. same, batched ------------------------------------------------
    {
        ggml_backend_graph_compute(backend, g_mm_batch);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) ggml_backend_graph_compute(backend, g_mm_batch);
        double dt = (now_ms() - t0) / REPS;
        row("3b. 48 matmuls of 6.56 MiB, batched, one fence at end", dt, N_LAYERS);
        printf("      effective VRAM read bandwidth: %.1f GB/s\n",
               (ggml_nbytes(W[0]) * (double) N_LAYERS) / (dt * 1e-3) / 1e9);
    }

    // ---- 4. 16 KiB host->device->host round trip, 48x ---------------------
    {
        std::vector<float> up(4096, 1.0f);
        std::vector<float> down(4096, 0.0f);
        for (int i = 0; i < N_LAYERS; i++) {
            ggml_backend_tensor_set(act, up.data(), 0, 16384);
            ggml_backend_tensor_get(act, down.data(), 0, 16384);
        }
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < N_LAYERS; i++) {
                ggml_backend_tensor_set(act, up.data(), 0, 16384);
                ggml_backend_tensor_get(act, down.data(), 0, 16384);
            }
        }
        row("4. 48x 16 KiB round trip (set + get, blocking)", (now_ms() - t0) / REPS, N_LAYERS);

        t0 = now_ms();
        for (int r = 0; r < REPS; r++)
            for (int i = 0; i < N_LAYERS; i++) ggml_backend_tensor_set(act, up.data(), 0, 16384);
        row("4a.   ... host -> device only", (now_ms() - t0) / REPS, N_LAYERS);

        t0 = now_ms();
        for (int r = 0; r < REPS; r++)
            for (int i = 0; i < N_LAYERS; i++) ggml_backend_tensor_get(act, down.data(), 0, 16384);
        row("4b.   ... device -> host only", (now_ms() - t0) / REPS, N_LAYERS);
    }

    // ---- 5. the whole per-layer thing, as the design would run it ---------
    {
        std::vector<float> up(4096, 1.0f);
        std::vector<float> down(4096, 0.0f);
        for (int i = 0; i < N_LAYERS; i++) {
            ggml_backend_tensor_set(act, up.data(), 0, 16384);
            ggml_backend_graph_compute(backend, g_mm[i]);
            ggml_backend_tensor_get(act, down.data(), 0, 16384);
        }
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < N_LAYERS; i++) {
                ggml_backend_tensor_set(act, up.data(), 0, 16384);
                ggml_backend_graph_compute(backend, g_mm[i]);
                ggml_backend_tensor_get(act, down.data(), 0, 16384);
            }
        }
        row("5. FULL per-layer: 16K up + 6.56 MiB matmul + fence + 16K down", (now_ms() - t0) / REPS, N_LAYERS);
    }

    // ---- 5b. the same rendezvous with NO VRAM read at all: this is the fixed
    //          per-layer cost of CPU/GPU co-execution, the floor the design pays
    //          before it reads a single expert byte.
    {
        std::vector<float> up(4096, 1.0f);
        std::vector<float> down(4096, 0.0f);
        for (int i = 0; i < N_LAYERS; i++) {
            ggml_backend_tensor_set(act, up.data(), 0, 16384);
            ggml_backend_graph_compute(backend, g_triv[i]);
            ggml_backend_tensor_get(act, down.data(), 0, 16384);
        }
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < N_LAYERS; i++) {
                ggml_backend_tensor_set(act, up.data(), 0, 16384);
                ggml_backend_graph_compute(backend, g_triv[i]);
                ggml_backend_tensor_get(act, down.data(), 0, 16384);
            }
        }
        row("5b. RENDEZVOUS FLOOR: 16K up + trivial dispatch + fence + 16K down",
            (now_ms() - t0) / REPS, N_LAYERS);
    }

    // ---- 6. three matmuls per layer, batched ------------------------------
    {
        ggml_backend_graph_compute(backend, g_mm3);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) ggml_backend_graph_compute(backend, g_mm3);
        double dt = (now_ms() - t0) / REPS;
        row("6. 144 matmuls (3 per layer x 48), batched, one fence", dt, N_LAYERS);
        printf("      effective VRAM read bandwidth: %.1f GB/s\n",
               (ggml_nbytes(W[0]) * 3.0 * N_LAYERS) / (dt * 1e-3) / 1e9);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// the shape the design actually needs: mul_mat_id over a stacked resident-expert
// tensor, one node per matrix, ids naming the resident subset for this token.
// ---------------------------------------------------------------------------

static void bench_mmid(ggml_backend_t backend) {
    const int64_t n_embd   = 5120;
    const int64_t n_ff_exp = 1536;
    const int64_t n_res    = 3;   // resident experts stacked per layer (VRAM-limited here)
    const int64_t n_hit    = 3;   // of the 8 selected, the ones that are resident

    struct ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);
    const size_t GS = 512;

    std::vector<struct ggml_tensor *> E(N_LAYERS);
    std::vector<struct ggml_tensor *> C(N_LAYERS);   // control: the same bytes, plain 2-D
    for (int i = 0; i < N_LAYERS; i++) {
        E[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff_exp, n_res);
        C[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff_exp * n_hit);
    }
    struct ggml_tensor * xb  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
    struct ggml_tensor * xc  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_hit, 1);
    struct ggml_tensor * act = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4096);

    // one mul_mat_id per layer
    std::vector<struct ggml_cgraph *> g1(N_LAYERS);
    for (int i = 0; i < N_LAYERS; i++) {
        g1[i] = ggml_new_graph_custom(ctx, GS, false);
        ggml_build_forward_expand(g1[i], ggml_mul_mat_id(ctx, E[i], xb, ids));
    }
    // all 48 in one graph
    struct ggml_cgraph * g1_batch = ggml_new_graph_custom(ctx, GS, false);
    for (int i = 0; i < N_LAYERS; i++) {
        ggml_build_forward_expand(g1_batch, ggml_mul_mat_id(ctx, E[i], xb, ids));
    }
    // control: exactly the same number of Q6_K bytes, read by a plain mul_mat
    struct ggml_cgraph * gc_batch = ggml_new_graph_custom(ctx, GS, false);
    for (int i = 0; i < N_LAYERS; i++) {
        ggml_build_forward_expand(gc_batch, ggml_mul_mat(ctx, C[i], xc));
    }
    std::vector<struct ggml_cgraph *> gc(N_LAYERS);
    for (int i = 0; i < N_LAYERS; i++) {
        gc[i] = ggml_new_graph_custom(ctx, GS, false);
        ggml_build_forward_expand(gc[i], ggml_mul_mat(ctx, C[i], xc));
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        fprintf(stderr, "  mul_mat_id bench: could not allocate %.2f MiB on device\n",
                ggml_nbytes(E[0]) * (double) N_LAYERS / 1048576.0);
        ggml_free(ctx);
        return;
    }

    printf("\n=== Q1/Q2: mul_mat_id on a stacked resident-expert tensor ===\n");
    printf("  stacked tensor per layer : Q6_K [%lld, %lld, %lld] = %.2f MiB\n",
           (long long) n_embd, (long long) n_ff_exp, (long long) n_res, ggml_nbytes(E[0]) / 1048576.0);
    printf("  one expert slice         : %.3f MiB\n", ggml_nbytes(E[0]) / 1048576.0 / n_res);
    printf("  ids select %lld of %lld resident -> %.2f MiB read per node\n",
           (long long) n_hit, (long long) n_res, ggml_nbytes(E[0]) / 1048576.0 * n_hit / n_res);
    printf("  device buffer total      : %.2f MiB\n", ggml_backend_buffer_get_size(buf) / 1048576.0);

    // fill ids with the first n_hit expert indices, and give the weights bytes
    {
        std::vector<int32_t> hid(n_hit);
        for (int64_t k = 0; k < n_hit; k++) hid[k] = (int32_t) k;
        ggml_backend_tensor_set(ids, hid.data(), 0, n_hit * sizeof(int32_t));
        std::vector<float> xh(n_embd, 0.001f);
        ggml_backend_tensor_set(xb, xh.data(), 0, n_embd * sizeof(float));
        ggml_backend_tensor_set(xc, xh.data(), 0, n_embd * sizeof(float));
        std::vector<uint8_t> junk(ggml_nbytes(E[0]), 0x20);
        for (int i = 0; i < N_LAYERS; i++) ggml_backend_tensor_set(E[i], junk.data(), 0, ggml_nbytes(E[i]));
        for (int i = 0; i < N_LAYERS; i++) ggml_backend_tensor_set(C[i], junk.data(), 0, ggml_nbytes(C[i]));
        ggml_backend_synchronize(backend);
    }

    const int REPS = 20;
    const double bytes_read = ggml_nbytes(E[0]) * (double) n_hit / n_res * N_LAYERS;

    {
        for (int i = 0; i < N_LAYERS; i++) ggml_backend_graph_compute(backend, g1[i]);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++)
            for (int i = 0; i < N_LAYERS; i++) ggml_backend_graph_compute(backend, g1[i]);
        double dt = (now_ms() - t0) / REPS;
        row("7. 48 mul_mat_id (1 node/layer), one fence wait each", dt, N_LAYERS);
        printf("      effective VRAM read bandwidth: %.1f GB/s\n", bytes_read / (dt * 1e-3) / 1e9);
    }
    {
        ggml_backend_graph_compute(backend, g1_batch);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) ggml_backend_graph_compute(backend, g1_batch);
        double dt = (now_ms() - t0) / REPS;
        row("8. 48 mul_mat_id, batched, one fence at end", dt, N_LAYERS);
        printf("      effective VRAM read bandwidth: %.1f GB/s\n", bytes_read / (dt * 1e-3) / 1e9);
    }
    {
        std::vector<float> up(4096, 1.0f), down(4096, 0.0f);
        for (int i = 0; i < N_LAYERS; i++) {
            ggml_backend_tensor_set(act, up.data(), 0, 16384);
            ggml_backend_graph_compute(backend, g1[i]);
            ggml_backend_tensor_get(act, down.data(), 0, 16384);
        }
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < N_LAYERS; i++) {
                ggml_backend_tensor_set(act, up.data(), 0, 16384);
                ggml_backend_graph_compute(backend, g1[i]);
                ggml_backend_tensor_get(act, down.data(), 0, 16384);
            }
        }
        row("9. FULL per-layer: 16K up + mul_mat_id + fence + 16K down", (now_ms() - t0) / REPS, N_LAYERS);
    }
    // 10: the same, but the 16 KiB transfers ride along with the compute submit
    {
        std::vector<float> up(4096, 1.0f), down(4096, 0.0f);
        for (int i = 0; i < N_LAYERS; i++) {
            ggml_backend_tensor_set_async(backend, act, up.data(), 0, 16384);
            ggml_backend_graph_compute(backend, g1[i]);
            ggml_backend_tensor_get_async(backend, act, down.data(), 0, 16384);
        }
        ggml_backend_synchronize(backend);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < N_LAYERS; i++) {
                ggml_backend_tensor_set_async(backend, act, up.data(), 0, 16384);
                ggml_backend_graph_compute(backend, g1[i]);
                ggml_backend_tensor_get_async(backend, act, down.data(), 0, 16384);
            }
            ggml_backend_synchronize(backend);
        }
        row("10. same but async transfers (set_async/get_async)", (now_ms() - t0) / REPS, N_LAYERS);
    }
    // control: identical byte count, plain mul_mat instead of mul_mat_id
    {
        for (int i = 0; i < N_LAYERS; i++) ggml_backend_graph_compute(backend, gc[i]);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++)
            for (int i = 0; i < N_LAYERS; i++) ggml_backend_graph_compute(backend, gc[i]);
        double dt = (now_ms() - t0) / REPS;
        row("11. CONTROL same bytes as 7, plain mul_mat, fence each", dt, N_LAYERS);
        printf("      effective VRAM read bandwidth: %.1f GB/s\n", bytes_read / (dt * 1e-3) / 1e9);
    }
    {
        ggml_backend_graph_compute(backend, gc_batch);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) ggml_backend_graph_compute(backend, gc_batch);
        double dt = (now_ms() - t0) / REPS;
        row("12. CONTROL same bytes as 8, plain mul_mat, batched", dt, N_LAYERS);
        printf("      effective VRAM read bandwidth: %.1f GB/s\n", bytes_read / (dt * 1e-3) / 1e9);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// does the cost of one mul_mat_id depend on how large the resident stack is,
// or only on how many experts the ids actually name? The design has a large
// stack (~32 experts per layer) and a small hit count (~3), so this is the
// number that decides it.
// ---------------------------------------------------------------------------

static void bench_mmid_scan(ggml_backend_t backend, raw_vk & rv) {
    const int64_t n_embd   = 5120;
    const int64_t n_ff_exp = 1536;
    const double  slice_mib = (double) (n_embd * n_ff_exp / 256 * 210) / 1048576.0;

    printf("\n=== Q1/Q2: mul_mat_id cost vs resident stack size (Q6_K, 1 token) ===\n");
    printf("  one expert slice = %.3f MiB; per-node bytes read should be n_hit x that\n", slice_mib);
    printf("  mul_mat_id vs a plain mul_mat over views of the SAME bytes at the SAME offsets\n");
    printf("  %-8s %-8s %-8s %12s %12s %14s %12s %12s\n",
           "n_res", "n_hit", "layers", "mmid us/node", "mmid GB/s", "mmid us x48", "view us/nd", "view GB/s");
    printf("  plus: MiB the buffers really occupy in VRAM vs in host RAM\n");

    const int64_t cases[][2] = { {3,3}, {4,3}, {8,3}, {16,3}, {32,3}, {32,1}, {32,2}, {32,4}, {32,8} };
    const int n_cases = (int) (sizeof(cases) / sizeof(cases[0]));

    for (int c = 0; c < n_cases; c++) {
        const int64_t n_res = cases[c][0];
        const int64_t n_hit = cases[c][1];

        // keep the device buffer under ~2.2 GiB
        int layers = (int) (2200.0 / (n_res * slice_mib));
        if (layers > N_LAYERS) layers = N_LAYERS;
        if (layers < 4) layers = 4;

        struct ggml_init_params ip = {
            /*.mem_size   =*/ (size_t) 32 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx = ggml_init(ip);

        std::vector<struct ggml_tensor *> E(layers);
        for (int i = 0; i < layers; i++) {
            E[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff_exp, n_res);
        }
        struct ggml_tensor * xb  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
        struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_hit, 1);

        struct ggml_cgraph * g = ggml_new_graph_custom(ctx, 1024, false);
        for (int i = 0; i < layers; i++) {
            ggml_build_forward_expand(g, ggml_mul_mat_id(ctx, E[i], xb, ids));
        }
        // the control: n_hit plain mul_mats over 2-D views of E[i] at exactly the
        // offsets the ids name. Same bytes, same addresses, no gather.
        struct ggml_tensor * xv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
        struct ggml_cgraph * gv = ggml_new_graph_custom(ctx, 1024, false);
        for (int i = 0; i < layers; i++) {
            for (int64_t k = 0; k < n_hit; k++) {
                const int64_t e = (k * n_res) / n_hit;
                struct ggml_tensor * v = ggml_view_2d(ctx, E[i], n_embd, n_ff_exp,
                                                      E[i]->nb[1], (size_t) e * E[i]->nb[2]);
                ggml_build_forward_expand(gv, ggml_mul_mat(ctx, v, xv));
            }
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) {
            printf("  %-8lld %-8lld %-8d  allocation of %.0f MiB failed, skipped\n",
                   (long long) n_res, (long long) n_hit, layers, n_res * slice_mib * layers);
            ggml_free(ctx);
            continue;
        }

        // ids spread over the stack, as a real resident-set mapping would be
        std::vector<int32_t> hid(n_hit);
        for (int64_t k = 0; k < n_hit; k++) hid[k] = (int32_t) ((k * n_res) / n_hit);
        ggml_backend_tensor_set(ids, hid.data(), 0, n_hit * sizeof(int32_t));
        std::vector<float> xh(n_embd, 0.001f);
        ggml_backend_tensor_set(xb, xh.data(), 0, n_embd * sizeof(float));
        ggml_backend_tensor_set(xv, xh.data(), 0, n_embd * sizeof(float));
        {
            std::vector<uint8_t> junk(ggml_nbytes(E[0]), 0x20);
            for (int i = 0; i < layers; i++) ggml_backend_tensor_set(E[i], junk.data(), 0, ggml_nbytes(E[i]));
        }
        ggml_backend_synchronize(backend);

        double vram_mib = 0, host_mib = 0;
        raw_vk_usage(rv, vram_mib, host_mib);

        const int REPS = 20;
        const double bytes = slice_mib * 1048576.0 * n_hit * layers;

        ggml_backend_graph_compute(backend, g);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) ggml_backend_graph_compute(backend, g);
        double dt = (now_ms() - t0) / REPS;

        ggml_backend_graph_compute(backend, gv);
        double t1 = now_ms();
        for (int r = 0; r < REPS; r++) ggml_backend_graph_compute(backend, gv);
        double dtv = (now_ms() - t1) / REPS;

        const double per_node_us = dt * 1000.0 / layers;
        printf("  %-8lld %-8lld %-8d %12.1f %12.1f %14.0f %12.1f %12.1f\n",
               (long long) n_res, (long long) n_hit, layers,
               per_node_us, bytes / (dt * 1e-3) / 1e9, per_node_us * N_LAYERS,
               dtv * 1000.0 / layers, bytes / (dtv * 1e-3) / 1e9);
        printf("           requested %.0f MiB -> VRAM %.0f MiB, host %.0f MiB\n",
               n_res * slice_mib * layers, vram_mib, host_mib);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
}

// ---------------------------------------------------------------------------
// The design's real read pattern: one large resident pool in VRAM, and each
// layer reads a few 6 MiB expert windows out of it at whatever offsets the
// router picked. Measures the bandwidth that pattern actually achieves,
// with no stacking or stride artefacts.
// ---------------------------------------------------------------------------

static void bench_pool(ggml_backend_t backend, raw_vk & rv, int n_pools) {
    const int64_t n_embd    = 5120;
    const int64_t n_ff_exp  = 1536;
    // 160 windows x 6.152 MiB = 984 MiB, just under the backend's 1 GiB per-buffer cap
    const int64_t n_windows = 160 * n_pools;
    const int64_t n_hit     = 3;                    // windows read per layer

    struct ggml_init_params ip = {
        /*.mem_size   =*/ (size_t) 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);

    // n_pools flat pools of just under 1 GiB, exactly as a resident-expert
    // buffer of this size would have to be split
    std::vector<struct ggml_tensor *> pools(n_pools);
    for (int p = 0; p < n_pools; p++) {
        pools[p] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_ff_exp * 160);
    }
    struct ggml_tensor * xv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);

    const size_t window_bytes = ggml_nbytes(pools[0]) / 160;
    const size_t pool_bytes   = ggml_nbytes(pools[0]);
    const size_t total_bytes  = pool_bytes * n_pools;

    auto window_view = [&](int64_t w) {
        struct ggml_tensor * p = pools[w / 160];
        return ggml_view_2d(ctx, p, n_embd, n_ff_exp, p->nb[1], (size_t) (w % 160) * window_bytes);
    };

    // scattered: pick windows with a large odd stride so consecutive layers land
    // far apart, the way a router-driven resident set would
    struct ggml_cgraph * g_scatter = ggml_new_graph_custom(ctx, 1024, false);
    int64_t w = 0;
    for (int i = 0; i < N_LAYERS; i++) {
        for (int64_t k = 0; k < n_hit; k++) {
            w = (w + 97) % n_windows;
            ggml_build_forward_expand(g_scatter, ggml_mul_mat(ctx, window_view(w), xv));
        }
    }
    // sequential: the same number of windows, walked in order
    struct ggml_cgraph * g_seq = ggml_new_graph_custom(ctx, 1024, false);
    for (int i = 0; i < N_LAYERS; i++) {
        for (int64_t k = 0; k < n_hit; k++) {
            ggml_build_forward_expand(g_seq, ggml_mul_mat(ctx, window_view((i * n_hit + k) % n_windows), xv));
        }
    }
    // reference: one mul_mat over each whole pool, the best this card can do
    struct ggml_cgraph * g_all = ggml_new_graph_custom(ctx, 16, false);
    for (int p = 0; p < n_pools; p++) ggml_build_forward_expand(g_all, ggml_mul_mat(ctx, pools[p], xv));

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        printf("\n  bench_pool: could not allocate %.0f MiB, skipped\n", total_bytes / 1048576.0);
        ggml_free(ctx);
        return;
    }

    printf("\n=== Q2: bandwidth for the design's real read pattern, pool = %.2f GiB in %d buffer(s) ===\n",
           total_bytes / 1073741824.0, n_pools);
    printf("  %lld windows of %.3f MiB; per layer %lld windows = %.2f MiB\n",
           (long long) n_windows, window_bytes / 1048576.0, (long long) n_hit, window_bytes * n_hit / 1048576.0);

    {
        std::vector<float> xh(n_embd, 0.001f);
        ggml_backend_tensor_set(xv, xh.data(), 0, n_embd * sizeof(float));
        std::vector<uint8_t> junk(window_bytes, 0x20);
        for (int p = 0; p < n_pools; p++) {
            for (int64_t k = 0; k < 160; k++) {
                ggml_backend_tensor_set(pools[p], junk.data(), (size_t) k * window_bytes, window_bytes);
            }
        }
        ggml_backend_synchronize(backend);
    }
    raw_vk_print_budget(rv, "with the pool resident");

    const int REPS = 20;
    const double bytes = (double) window_bytes * n_hit * N_LAYERS;

    struct { const char * name; struct ggml_cgraph * g; double bytes; } runs[] = {
        { "13. 48 layers x 3 scattered 6.15 MiB windows, batched", g_scatter, bytes },
        { "14. 48 layers x 3 sequential 6.15 MiB windows, batched", g_seq,     bytes },
        { "15. reference: one mul_mat over every pool byte",        g_all,    (double) total_bytes },
    };
    for (auto & rr : runs) {
        ggml_backend_graph_compute(backend, rr.g);
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++) ggml_backend_graph_compute(backend, rr.g);
        double dt = (now_ms() - t0) / REPS;
        row(rr.name, dt, N_LAYERS);
        printf("      %.2f MiB read, effective VRAM read bandwidth: %.1f GB/s\n",
               rr.bytes / 1048576.0, rr.bytes / (dt * 1e-3) / 1e9);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// How much can the ggml Vulkan backend actually hold, with the desktop up?
// Allocates 256 MiB buffers until one fails, then measures whether reads at the
// ceiling still run at VRAM speed or have silently migrated to host memory.
// ---------------------------------------------------------------------------

static void bench_ceiling(ggml_backend_t backend, raw_vk & rv) {
    printf("\n=== Q3: usable VRAM ceiling through the ggml Vulkan backend ===\n");
    ggml_backend_buffer_type_t buft = ggml_backend_vk_buffer_type(0);
    printf("  buft max single buffer   : %.0f MiB\n", ggml_backend_buft_get_max_size(buft) / 1048576.0);

    const size_t chunk = 256ull * 1024 * 1024;
    std::vector<ggml_backend_buffer_t> bufs;
    size_t total = 0;
    for (int i = 0; i < 32; i++) {
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, chunk);
        if (b == nullptr) break;
        bufs.push_back(b);
        total += chunk;
    }
    printf("  allocated %zu x 256 MiB   = %.0f MiB (%.2f GiB) before failure\n",
           bufs.size(), total / 1048576.0, total / 1073741824.0);
    raw_vk_print_budget(rv, "at the allocation ceiling");

    for (auto b : bufs) ggml_backend_buffer_free(b);
    printf("  (all freed)\n");
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    (void) argc; (void) argv;

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    printf("=== memex-vk: Vulkan backend probe ===\n\n");

    raw_vk r;
    if (!raw_vk_init(r)) return 1;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(r.phys, &props);

    VkPhysicalDeviceMaintenance3Properties m3{};
    m3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &m3;
    vkGetPhysicalDeviceProperties2(r.phys, &p2);

    printf("=== Q3: device and limits ===\n");
    printf("  device                        : %s\n", props.deviceName);
    printf("  driverVersion                 : %u.%u.%u\n",
           VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion), VK_VERSION_PATCH(props.driverVersion));
    printf("  VK_EXT_memory_budget          : %s\n", r.has_budget ? "present" : "ABSENT (falling back to heap sizes)");
    printf("  maxComputeSharedMemorySize    : %u bytes\n", props.limits.maxComputeSharedMemorySize);
    printf("  maxStorageBufferRange         : %u bytes (%.2f MiB)\n",
           props.limits.maxStorageBufferRange, props.limits.maxStorageBufferRange / 1048576.0);
    printf("  maxMemoryAllocationSize       : %llu bytes (%.2f MiB)\n",
           (unsigned long long) m3.maxMemoryAllocationSize, m3.maxMemoryAllocationSize / 1048576.0);
    printf("  maxComputeWorkGroupInvocations: %u\n", props.limits.maxComputeWorkGroupInvocations);

    raw_vk_print_budget(r, "before any ggml init, desktop running");

    // find the big device-local heap
    uint32_t big_heap = 0;
    VkDeviceSize big_size = 0;
    for (uint32_t i = 0; i < r.memprops.memoryHeapCount; i++) {
        if ((r.memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            r.memprops.memoryHeaps[i].size > big_size) {
            big_size = r.memprops.memoryHeaps[i].size;
            big_heap = i;
        }
    }

    printf("\n  --- largest single allocation the driver accepts ---\n");
    size_t max_alloc = raw_vk_probe_max_alloc(r, big_heap, (size_t) big_size);
    printf("  vkAllocateMemory on heap %u    : %.2f MiB (%zu bytes)\n", big_heap, max_alloc / 1048576.0, max_alloc);
    size_t max_buf = raw_vk_probe_max_buffer(r, (size_t) big_size);
    printf("  VkBuffer create + bind        : %.2f MiB (%zu bytes)\n", max_buf / 1048576.0, max_buf);

    // ---- now bring up the ggml Vulkan backend ----------------------------
    printf("\n  --- ggml Vulkan backend baseline ---\n");
    ggml_backend_t backend = ggml_backend_vk_init(0);
    if (backend == nullptr) { fprintf(stderr, "ggml_backend_vk_init failed\n"); return 1; }

    raw_vk_print_budget(r, "after ggml_backend_vk_init, before any tensor");

    probe_supports(backend);
    bench_all(backend);
    bench_mmid(backend);
    bench_mmid_scan(backend, r);
    bench_pool(backend, r, 1);
    bench_pool(backend, r, 2);
    bench_pool(backend, r, 3);
    bench_ceiling(backend, r);

    raw_vk_print_budget(r, "after the benchmark tensors were freed");

    ggml_backend_free(backend);
    vkDestroyDevice(r.dev, nullptr);
    vkDestroyInstance(r.instance, nullptr);
    return 0;
}
