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

# NeuroPress GPU backend: when CUDA is available, its inference and online-
# SGD training run as real CUDA kernels operating on device-resident
# weights, matching the original NeuroPress project's own design (src/nn/
# nn_gpu.cu) rather than falling back to the pure-C++ CPU port. Added
# directly into this target (not a separate library) so it doesn't need its
# own entry in the ClioCtp install(EXPORT) set -- the other predictors
# (Q-table/linreg/XGBoost) stay untouched either way.
if(CLIO_CORE_ENABLE_CUDA)
    list(APPEND CLIO_CTP_COMPRESS_MODEL_SRC neuropress_nn_gpu_kernels.cu)
    # Device-native entropy/MAD/second-derivative computation (issue #693
    # follow-up): lets a caller resolving a CUDA-IPC device buffer feed
    # NeuroPress's selection features without staging the buffer through
    # host memory first. Lives next to its header under compress/preprocess/
    # but is compiled into this target -- same rationale as the NeuroPress
    # GPU kernels above: no separate install(EXPORT) entry needed.
    list(APPEND CLIO_CTP_COMPRESS_MODEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../preprocess/data_stats_gpu_kernels.cu)
endif()

add_library(clio_ctp_compress_model STATIC ${CLIO_CTP_COMPRESS_MODEL_SRC})

if(CLIO_CORE_ENABLE_CUDA)
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}"
            CACHE STRING "CUDA architectures to compile for" FORCE)
    endif()
    set_source_files_properties(neuropress_nn_gpu_kernels.cu PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/../preprocess/data_stats_gpu_kernels.cu
        PROPERTIES LANGUAGE CUDA)
    set_target_properties(clio_ctp_compress_model PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Static
    )
    # Plain CUDA runtime only -- NOT ctp::cuda_cxx/clio_ctp_cuda, which
    # creates a real dependency cycle here (clio_ctp_cuda's own sources
    # already link clio_ctp_compress_model, and this target linking back to
    # clio_ctp_cuda closes the loop; CMake refuses cycles between a shared
    # and a static library). This .cu file only calls the plain CUDA
    # runtime API (cudaMalloc/cudaMemcpy/kernel launches), none of CTP's
    # own GPU utility headers, so it doesn't need that heavier target.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(clio_ctp_compress_model PRIVATE CUDA::cudart)
    target_compile_definitions(clio_ctp_compress_model PUBLIC CTP_ENABLE_NEUROPRESS_GPU=1)
    message(STATUS "clio_ctp_compress_model: NeuroPress GPU backend enabled (CUDA)")
else()
    target_compile_definitions(clio_ctp_compress_model PUBLIC CTP_ENABLE_NEUROPRESS_GPU=0)
endif()

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
