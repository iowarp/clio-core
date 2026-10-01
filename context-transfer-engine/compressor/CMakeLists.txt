cmake_minimum_required(VERSION 3.10)
project(clio_cte_compressor)

#------------------------------------------------------------------------------
# Build compressor chimod
#------------------------------------------------------------------------------

# Determine which predictor sources to include
set(COMPRESSOR_SOURCES
    src/compressor_runtime.cc
    src/models/linreg_table_predictor.cc
    src/models/qtable_predictor.cc
    src/autogen/compressor_lib_exec.cc
)

# Add dense NN predictor if Eigen is available
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    list(APPEND COMPRESSOR_SOURCES src/models/dense_nn_predictor.cc)
    message(STATUS "Compressor: Dense NN predictor enabled (Eigen3 found)")
else()
    message(STATUS "Compressor: Dense NN predictor disabled (Eigen3 not found)")
endif()

# Add XGBoost predictor if available
find_package(xgboost QUIET)
if(xgboost_FOUND)
    list(APPEND COMPRESSOR_SOURCES src/models/xgboost_predictor.cc)
    message(STATUS "Compressor: XGBoost predictor enabled")
else()
    message(STATUS "Compressor: XGBoost predictor disabled")
endif()

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

# Link optional dependencies if available
if(Eigen3_FOUND)
target_link_libraries(clio_cte_compressor_runtime PRIVATE Eigen3::Eigen)
endif()
if(xgboost_FOUND)
    target_link_libraries(clio_cte_compressor_runtime PRIVATE xgboost::xgboost)
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
# Synthetic Data Generators
#------------------------------------------------------------------------------
add_subdirectory(generator)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
