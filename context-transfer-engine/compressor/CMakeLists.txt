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
    src/models/neuropress_bridge.cc
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
    ctp::compress_model
)

# Link optional dependencies if available
if(Eigen3_FOUND)
    target_link_libraries(clio_cte_compressor_runtime PRIVATE Eigen3::Eigen)
endif()
if(xgboost_FOUND)
    target_link_libraries(clio_cte_compressor_runtime PRIVATE xgboost::xgboost)
endif()

# EstCompressionStats (compressor_runtime.cc) needs CTP_ENABLE_CUDA/ROCM to
# see ctp::GpuApi::IsDevicePointer / ctp::ComputeDeviceStats, so it can
# detect a CUDA-IPC device buffer and compute NeuroPress's selection
# features on-device instead of segfaulting on a direct host read.
# ctp::compress_model (linked above) only propagates CTP_ENABLE_NEUROPRESS_GPU,
# not CTP_ENABLE_CUDA itself (that stays contained to avoid the g++/CUDA-
# builtins issue -- see context-runtime/src/CMakeLists.txt and modules/admin/
# CMakeLists.txt for the same pattern applied one level down).
if(CLIO_CORE_ENABLE_CUDA)
    target_link_libraries(clio_cte_compressor_runtime PRIVATE ctp::cuda_cxx)
elseif(CLIO_CORE_ENABLE_ROCM)
    target_link_libraries(clio_cte_compressor_runtime PRIVATE ctp::rocm_cxx)
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
