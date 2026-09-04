set(TARGET llama-memex-fwd)

# The zoned KV cache is compiled in from examples/memex-kv rather than installed as a
# library, the same arrangement memex-kv itself uses, and the same reason: the zone module
# lives outside the fork so it can be unit-tested without ggml. Guarded by EXISTS so a
# checkout without the MemeX tree still configures - the engine then builds without the
# --zoned option and says so when it is asked for, instead of failing to compile.
set(MEMEX_DIR "C:/Users/User11/Desktop/MemeX/cpp")
set(MEMEX_KV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../memex-kv")

# ============================================================================
# memex_core - static library of the mechanism modules that are ALREADY
# standalone translation units. This is refactor step 0: no code moves, no
# behaviour changes; the four files below simply link through a library target
# instead of being compiled straight into the executable. That is what lets
# llama-server (see docs/server_integration_map.md) link our mechanisms.
#
#   resident_set.cpp  - the resident expert set. Ours, local, needs nothing but
#                       ggml. Off by default at run time via --resident 0.
#   expert_store.cpp  - the resident expert STORE (step 4). Same story: ours,
#                       local, host RAM only, no Vulkan. Off via --expert-store 0.
# Both are unconditional. Verified: neither pulls a single type out of
# memex-fwd.cpp (the names HParams/Graph/Weights/DeltaState do not appear in
# them at all), so they compile in a library with zero header surgery.
# ============================================================================
set(MEMEX_CORE_SRC resident_set.cpp expert_store.cpp)

# The GPU half needs a ggml that has the Vulkan backend in it, which is a property of the
# build tree and not of this directory: build-vk has it and build does not, and the same
# source has to configure in both. gpu_experts.cpp / gpu_static.cpp reference the memex-fwd
# structures (Graph, HParams, DeltaState) only in prose comments, never as code, so they
# too move into the library untouched - guarded by GGML_VULKAN exactly as before. The run
# time gates (--gpu-experts / --gpu-static refuse in a tree without the Vulkan backend)
# live in memex-fwd.cpp and are unaffected.
if (GGML_VULKAN)
    list(APPEND MEMEX_CORE_SRC gpu_experts.cpp gpu_static.cpp)
endif()

add_library(memex_core STATIC ${MEMEX_CORE_SRC})
# PUBLIC ggml: propagates ggml's include directories (ggml.h, ggml-backend.h, and under
# Vulkan ggml-vulkan.h) to both this library's own compilation and to anything that links
# it. This is the same include surface the four files had when compiled into the exe.
target_link_libraries(memex_core PUBLIC ggml)
# PUBLIC so the executable, linking memex_core, sees the module headers by directory too.
target_include_directories(memex_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(memex_core PUBLIC cxx_std_17)

# Raw Vulkan is needed only by gpu_experts.cpp, and only under MEMEX_FWD_VULKAN (the per-heap
# free-bytes / heap-placement queries ggml does not expose). This mirrors what the executable
# target did before the split; the define is scoped to the library that actually uses it.
if (GGML_VULKAN)
    find_package(Vulkan)
    if (Vulkan_FOUND)
        target_link_libraries(memex_core PRIVATE Vulkan::Vulkan)
        target_compile_definitions(memex_core PRIVATE MEMEX_FWD_VULKAN)
    else()
        message(WARNING "memex-fwd: Vulkan headers not found; --gpu-experts will size the "
                        "resident set from a fixed budget instead of the device's heaps")
    endif()
endif()

# ============================================================================
# llama-memex-fwd - the harness executable. memex-fwd.cpp stays here (it is the
# 10k-line monolith the plan splits later); it now links memex_core instead of
# recompiling the four modules. The static half of the byte budget - the output
# head in video memory - is integrated in memex-fwd.cpp as one helper,
# head_matmul(), and one substituted call per graph builder, so the CPU path is
# byte-for-byte the code it replaced whenever the flag is off.
# ============================================================================
set(MEMEX_FWD_SRC memex-fwd.cpp)

if(EXISTS "${MEMEX_DIR}/src/kv_zones.cpp" AND EXISTS "${MEMEX_KV_DIR}/zoned_cache.cpp")
    add_executable(${TARGET} ${MEMEX_FWD_SRC}
                   "${MEMEX_KV_DIR}/zoned_cache.cpp"
                   "${MEMEX_DIR}/src/kv_zones.cpp")
    target_include_directories(${TARGET} PRIVATE "${MEMEX_DIR}/include" "${MEMEX_KV_DIR}")
    target_compile_definitions(${TARGET} PRIVATE MEMEX_FWD_ZONED)
else()
    add_executable(${TARGET} ${MEMEX_FWD_SRC})
endif()
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE memex_core common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_VULKAN)
    target_compile_definitions(${TARGET} PRIVATE MEMEX_FWD_GPU_EXPERTS)
    # Raw Vulkan, only for the two things ggml does not expose and that decide whether this
    # whole arrangement is fast or forty times slower: the per-heap free bytes (ggml's own
    # ggml_backend_vk_get_device_memory reports free == total, ggml-vulkan.cpp:10758) and
    # which heap a buffer could possibly have been placed on. Optional, exactly as in
    # examples/memex-vk: without it the capacity falls back to a printed constant.
    find_package(Vulkan)
    if (Vulkan_FOUND)
        target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
        target_compile_definitions(${TARGET} PRIVATE MEMEX_FWD_VULKAN)
    else()
        message(WARNING "memex-fwd: Vulkan headers not found; --gpu-experts will size the "
                        "resident set from a fixed budget instead of the device's heaps")
    endif()
endif()
