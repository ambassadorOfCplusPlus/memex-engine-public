set(TARGET llama-memex-moe)
# The engine lives in its own tree; compile its loader straight in so the example
# uses the same code path the runtime will.
set(MEMEX_DIR "C:/Users/User11/Desktop/MemeX/cpp")
if(EXISTS "${MEMEX_DIR}/src/blob_loader.cpp")
    add_executable(${TARGET} memex-moe.cpp "${MEMEX_DIR}/src/blob_loader.cpp")
    target_include_directories(${TARGET} PRIVATE "${MEMEX_DIR}/include")
    target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
else()
    message(STATUS "MemeX engine not found at ${MEMEX_DIR}, skipping ${TARGET}")
endif()
