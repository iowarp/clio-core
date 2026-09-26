cmake_minimum_required(VERSION 3.10)
project(clio_cte_compressor)

#------------------------------------------------------------------------------
# Build compressor chimod
#------------------------------------------------------------------------------

# The codec is chosen by the CALLER. There is no predictor, no feature
# extraction and no model to link -- that whole layer was removed, to be
# brought back on a better footing later.
set(COMPRESSOR_SOURCES
    src/compressor_runtime.cc
    src/autogen/compressor_lib_exec.cc
)

add_clio_module_runtime(
  SOURCES
    ${COMPRESSOR_SOURCES}
  LINK_LIBRARIES
    ctp::compress
)

# CUDA DRIVER API. A GPU codec runs in its own CUDA context (cuCtxCreate) and
# returns results with cuMemcpyPeer, neither of which the CUDA runtime API can
# express. Without this the module still BUILDS -- the failure is at dlopen,
# where an undefined cuCtxCreate_v4 makes the ChiMod silently unloadable.
if(CLIO_CORE_ENABLE_CUDA)
    if(TARGET CUDA::cuda_driver)
        target_link_libraries(clio_cte_compressor_runtime PUBLIC CUDA::cuda_driver)
    else()
        find_library(CLIO_CUDA_DRIVER_LIB cuda
                     HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}
                     PATH_SUFFIXES stubs lib64/stubs targets/x86_64-linux/lib/stubs)
        if(CLIO_CUDA_DRIVER_LIB)
            message(STATUS "compressor: CUDA driver API at ${CLIO_CUDA_DRIVER_LIB}")
            target_link_libraries(clio_cte_compressor_runtime PUBLIC ${CLIO_CUDA_DRIVER_LIB})
        else()
            message(WARNING "compressor: libcuda not found; GPU codec contexts disabled")
        endif()
    endif()
endif()

add_clio_module_client(
  SOURCES
    src/compressor_client.cc
)

# Add core include directory for Context struct
target_include_directories(clio_cte_compressor_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cte_compressor_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
