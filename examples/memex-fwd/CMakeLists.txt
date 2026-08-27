set(TARGET llama-memex-fwd)

# The zoned KV cache is compiled in from examples/memex-kv rather than installed as a
# library, the same arrangement memex-kv itself uses, and the same reason: the zone module
# lives outside the fork so it can be unit-tested without ggml. Guarded by EXISTS so a
# checkout without the MemeX tree still configures - the engine then builds without the
# --zoned option and says so when it is asked for, instead of failing to compile.
set(MEMEX_DIR "C:/Users/User11/Desktop/MemeX/cpp")
set(MEMEX_KV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../memex-kv")
# The resident expert set, by contrast, is unconditional: it is ours, it lives here, and it
# needs nothing but ggml. Off by default at run time through --resident 0, not at build time.
set(MEMEX_FWD_SRC memex-fwd.cpp resident_set.cpp)
# The GPU half needs a ggml that has the Vulkan backend in it, which is a property of the
# build tree and not of this directory: build-vk has it and build does not, and the same
# source has to configure in both. --gpu-experts refuses at run time in the tree without it
# rather than being quietly absent from --help.
# The static half of the byte budget - the output head in video memory - is the same story and
# the same gate: it needs a ggml with the Vulkan backend, so it compiles only in build-vk, and
# --gpu-static refuses at run time in the tree without it rather than being quietly absent from
# --help. The integration in memex-fwd.cpp is one helper function, head_matmul(), and one
# substituted call per qwen3moe graph builder, so the CPU path is byte-for-byte the code it
# replaced whenever the flag is off.
if (GGML_VULKAN)
    list(APPEND MEMEX_FWD_SRC gpu_experts.cpp gpu_static.cpp)
endif()

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
target_link_libraries(${TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
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
