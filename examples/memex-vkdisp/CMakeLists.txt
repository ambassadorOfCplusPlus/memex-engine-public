# Only meaningful in a tree that has the Vulkan backend. In the CPU tree the targets are simply
# not created, rather than built into something that refuses at run time: they measure a
# property of the Vulkan backend and there is nothing for them to say without one.
if (GGML_VULKAN)
    set(TARGET llama-memex-vkdisp)
    add_executable(${TARGET} vkdisp.cpp)
    install(TARGETS ${TARGET} RUNTIME)
    target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    # vkdisp measures the marginal cost of a node; vksplit takes that cost apart into the submit,
    # the host-side recording and the device-side execution. Separate binaries because vkdisp is
    # the record of what was measured before and should keep producing the same number.
    set(TARGET llama-memex-vksplit)
    add_executable(${TARGET} vksplit.cpp)
    install(TARGETS ${TARGET} RUNTIME)
    target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
