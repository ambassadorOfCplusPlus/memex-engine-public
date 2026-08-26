set(TARGET llama-memex-vkdisp)

# Only meaningful in a tree that has the Vulkan backend. In the CPU tree the target is simply
# not created, rather than built into something that refuses at run time: it measures a
# property of the Vulkan backend and there is nothing for it to say without one.
if (GGML_VULKAN)
    add_executable(${TARGET} vkdisp.cpp)
    install(TARGETS ${TARGET} RUNTIME)
    target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
