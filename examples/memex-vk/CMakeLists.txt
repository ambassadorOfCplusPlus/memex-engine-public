set(TARGET llama-memex-vk)
add_executable(${TARGET} memex-vk.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# the probe talks to Vulkan directly for VK_EXT_memory_budget and the
# single-allocation ceiling, which ggml does not expose
find_package(Vulkan)
if (Vulkan_FOUND)
    target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
else()
    message(WARNING "memex-vk: Vulkan not found, target will not build")
endif()
