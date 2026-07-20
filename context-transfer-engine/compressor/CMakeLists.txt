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

# Link optional dependencies if available
if(Eigen3_FOUND)
    target_link_libraries(clio_cte_compressor_runtime PRIVATE Eigen3::Eigen)
endif()
if(xgboost_FOUND)
    target_link_libraries(clio_cte_compressor_runtime PRIVATE xgboost::xgboost)
endif()

# cuszp.h runs cuSZp in a dedicated CUDA context (driver API: cuCtxCreate/cuInit/
# cuCtxPushCurrent). Link the CUDA driver library so the dlopen'd module .so
# resolves those symbols at load time (else the ChiMod fails to load).
#
# Only when CUDA is actually enabled: a CPU-only build (CLIO_CORE_ENABLE_CUDA=OFF,
# e.g. the deploy-cpu container) does NOT compile the cuSZp path, so it must not
# link the CUDA driver — libcuda is absent there and the bare `-lcuda` fallback
# below fails the link ("cannot find -lcuda").
if(CLIO_CORE_ENABLE_CUDA)
    if(TARGET CUDA::cuda_driver)
        target_link_libraries(clio_cte_compressor_runtime PRIVATE CUDA::cuda_driver)
    else()
        target_link_libraries(clio_cte_compressor_runtime PRIVATE cuda)
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
# Synthetic Data Generators
#------------------------------------------------------------------------------
add_subdirectory(generator)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
