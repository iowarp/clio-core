cmake_minimum_required(VERSION 3.10)
project(wrp_cte_compressor)

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

add_chimod_runtime(
  SOURCES
    ${COMPRESSOR_SOURCES}
  LINK_LIBRARIES
    hshm::compress
)

# Link optional dependencies if available
if(Eigen3_FOUND)
    target_link_libraries(wrp_cte_compressor_runtime PRIVATE Eigen3::Eigen)
endif()
if(xgboost_FOUND)
    target_link_libraries(wrp_cte_compressor_runtime PRIVATE xgboost::xgboost)
endif()

add_chimod_client(
  SOURCES
    src/compressor_client.cc
)

# Add core include directory for Context struct
target_include_directories(wrp_cte_compressor_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(wrp_cte_compressor_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
