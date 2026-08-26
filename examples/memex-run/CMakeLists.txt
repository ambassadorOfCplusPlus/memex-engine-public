set(TARGET llama-memex-run)
set(MEMEX_DIR "C:/Users/User11/Desktop/MemeX/cpp")
if(EXISTS "${MEMEX_DIR}/src/blob_loader.cpp")
    add_executable(${TARGET} memex-run.cpp
                   "${MEMEX_DIR}/src/blob_loader.cpp"
                   "${MEMEX_DIR}/src/layer_runtime.cpp")
    target_include_directories(${TARGET} PRIVATE "${MEMEX_DIR}/include")
    target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
else()
    message(STATUS "MemeX engine not found at ${MEMEX_DIR}, skipping ${TARGET}")
endif()
