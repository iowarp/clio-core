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
    src/neuropress_chunk_diag.cc
    src/neuropress_telemetry.cc
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

# clio::cte::core_client: compressor_runtime.cc's Compress() calls into
# clio::cte::core::Client::AsyncPutBlob (compressor storing its own output
# via CTE core), whose device-buffer staging now lives out-of-line in
# core_client.cc -- must link the library that defines it.
add_clio_module_runtime(
  SOURCES
    ${COMPRESSOR_SOURCES}
  LINK_LIBRARIES
    ctp::compress
    ctp::compress_model
    clio::cte::core_client
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
if(CLIO_NEUROPRESS_PATH_TRACE)
  target_compile_definitions(clio_cte_compressor_runtime PRIVATE
    CLIO_NEUROPRESS_PATH_TRACE)
endif()

target_include_directories(clio_cte_compressor_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cte_compressor_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)

#------------------------------------------------------------------------------
# Synthetic Data Generators
#------------------------------------------------------------------------------
add_subdirectory(generator)


#------------------------------------------------------------------------------
# End-to-end path trace (issue #693)
#------------------------------------------------------------------------------
# Compile-time, off by default. When on, the compressor and the HDF5 VOL each
# print "[np-path]" lines at every hop a write/read takes through the stack, so
# one grep shows whether data was really intercepted and compressed rather than
# quietly passed to native HDF5. Undefined => the calls expand to nothing.
option(CLIO_NEUROPRESS_PATH_TRACE "Trace the NeuroPress write/read path" OFF)
if(CLIO_NEUROPRESS_PATH_TRACE)
  message(STATUS "NeuroPress path trace: ENABLED (-DCLIO_NEUROPRESS_PATH_TRACE)")
endif()

#------------------------------------------------------------------------------
# Examples
#------------------------------------------------------------------------------
# The NeuroPress examples are kept out of the repository (see .gitignore), so
# this is guarded: present in a working tree that has them, a no-op elsewhere.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/example/CMakeLists.txt)
  add_subdirectory(example)
endif()

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
