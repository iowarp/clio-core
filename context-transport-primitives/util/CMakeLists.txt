# util: logging, timers, config parsing, singletons, GPU API helpers and
# the real-API interception layer.
# util/src/gpu_linker.cc is built into its own library (ctp::gpu_linker) only
# when CUDA is enabled; see cmake/ClioCtpLibraries.cmake.

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
