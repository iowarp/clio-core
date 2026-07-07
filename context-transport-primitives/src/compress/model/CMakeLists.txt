# ----------------------------------------------------------------------------
# Unified compression-metric prediction models (issue #693).
#
# One static library, clio_ctp_compress_model, holding every C++ stat model
# (Q-table, linear-regression table, XGBoost, NeuroPress NN). The pure-C++
# models have no external dependencies and always build; XGBoost is compiled
# in only when CLIO_CTP_ENABLE_XGBOOST is ON (the class otherwise degrades to
# a not-ready stub, so the library links either way).
# ----------------------------------------------------------------------------
project(clio_ctp)

set(CLIO_CTP_COMPRESS_MODEL_SRC
    qtable_predictor.cc
    linreg_table_predictor.cc
    xgboost_predictor.cc
    neuropress_nn_predictor.cc
)

add_library(clio_ctp_compress_model STATIC ${CLIO_CTP_COMPRESS_MODEL_SRC})

target_include_directories(clio_ctp_compress_model PUBLIC
    $<BUILD_INTERFACE:${CLIO_CTP_ROOT}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(clio_ctp_compress_model PROPERTIES POSITION_INDEPENDENT_CODE ON)
hshm_target_compile_definitions(clio_ctp_compress_model)

# Optional XGBoost backend. Default OFF so the model library builds everywhere;
# when enabled we locate the xgboost C API and flip CTP_ENABLE_XGBOOST on.
option(CLIO_CTP_ENABLE_XGBOOST "Build the XGBoost compression predictor" OFF)
if(CLIO_CTP_ENABLE_XGBOOST)
    find_path(XGBOOST_INCLUDE_DIR NAMES xgboost/c_api.h)
    find_library(XGBOOST_LIBRARY NAMES xgboost)
    if(XGBOOST_INCLUDE_DIR AND XGBOOST_LIBRARY)
        target_include_directories(clio_ctp_compress_model PRIVATE ${XGBOOST_INCLUDE_DIR})
        target_link_libraries(clio_ctp_compress_model PRIVATE ${XGBOOST_LIBRARY})
        target_compile_definitions(clio_ctp_compress_model PUBLIC CTP_ENABLE_XGBOOST=1)
        message(STATUS "clio_ctp_compress_model: XGBoost enabled (${XGBOOST_LIBRARY})")
    else()
        message(WARNING "CLIO_CTP_ENABLE_XGBOOST=ON but xgboost not found; "
                        "building the XGBoost predictor as a disabled stub")
        target_compile_definitions(clio_ctp_compress_model PUBLIC CTP_ENABLE_XGBOOST=0)
    endif()
else()
    target_compile_definitions(clio_ctp_compress_model PUBLIC CTP_ENABLE_XGBOOST=0)
endif()

if(NOT TARGET ctp::compress_model)
    add_library(ctp::compress_model ALIAS clio_ctp_compress_model)
endif()

# Installed via CLIO_CTP_LIBS in ../../CMakeLists.txt (single EXPORT set).
