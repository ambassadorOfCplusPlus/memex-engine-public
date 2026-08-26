set(TARGET llama-memex-kv)
# Same arrangement as memex-attn: the zone module lives outside the fork so it can be
# tested without ggml, and kv_zones.cpp is compiled in directly rather than installed.
# Guarded by EXISTS so a checkout without the MemeX tree still configures.
set(MEMEX_DIR "C:/Users/User11/Desktop/MemeX/cpp")
if(EXISTS "${MEMEX_DIR}/src/kv_zones.cpp")
    add_executable(${TARGET} memex-kv.cpp zoned_cache.cpp "${MEMEX_DIR}/src/kv_zones.cpp")
    target_include_directories(${TARGET} PRIVATE "${MEMEX_DIR}/include")
    # ggml only. No common, no llama: the unit test is synthetic on purpose, so nothing
    # here should be able to reach a model file even by accident.
    target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
