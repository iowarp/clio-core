project(hermes_shm)

# -----------------------------------------------------------------------------
# Build HSHM
# -----------------------------------------------------------------------------
set(HSHM_LIBS "")

# BUILD HSHM FOR HOST ONLY
set(SRC_FILES
    system_info.cc
    # memory_manager.cc  # NOTE: Deleted during hard refactoring
)

# Create library with system_info.cc
add_library(hermes_shm_host ${SRC_FILES})

#------------------------------------------------------------------------------
# Compression Predictor Libraries (Optional)
#------------------------------------------------------------------------------

# XGBoost Predictor Library
find_package(xgboost QUIET)
if(xgboost_FOUND)
    add_library(xgboost_predictor
        compression/xgboost_predictor.cc
    )
    target_include_directories(xgboost_predictor PUBLIC
        $<BUILD_INTERFACE:${HSHM_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(xgboost_predictor PUBLIC
        xgboost::xgboost
    )
    set_property(TARGET xgboost_predictor PROPERTY CXX_STANDARD 17)
    list(APPEND HSHM_LIBS xgboost_predictor)
    message(STATUS "Building XGBoost predictor library")
endif()

# Dense NN Predictor Library (requires Eigen, MiniDNN, and nlohmann_json)
find_package(Eigen3 QUIET)
find_package(nlohmann_json QUIET)
# MiniDNN is header-only, check if it exists
set(MINIDNN_INCLUDE_DIR "${HSHM_ROOT}/external/MiniDNN/include")
if(EXISTS "${MINIDNN_INCLUDE_DIR}/MiniDNN.h")
    set(MINIDNN_FOUND TRUE)
else()
    set(MINIDNN_FOUND FALSE)
endif()

if(Eigen3_FOUND AND MINIDNN_FOUND AND nlohmann_json_FOUND)
    add_library(dense_nn_predictor
        compression/dense_nn_predictor.cc
    )
    target_include_directories(dense_nn_predictor PUBLIC
        $<BUILD_INTERFACE:${HSHM_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${MINIDNN_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(dense_nn_predictor PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
    )
    set_property(TARGET dense_nn_predictor PROPERTY CXX_STANDARD 17)
    list(APPEND HSHM_LIBS dense_nn_predictor)
    message(STATUS "Building Dense NN predictor library with MiniDNN + Eigen")
elseif(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found - Dense NN predictor will not be built")
    message(STATUS "  Install with: conda install -c conda-forge eigen")
elseif(NOT MINIDNN_FOUND)
    message(STATUS "MiniDNN not found - Dense NN predictor will not be built")
    message(STATUS "  Clone to: ${HSHM_ROOT}/external/MiniDNN")
elseif(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found - Dense NN predictor will not be built")
endif()
target_link_libraries(hermes_shm_host PUBLIC
    ${REAL_TIME_FLAGS}
    configure
    thread_all
)
target_include_directories(hermes_shm_host PUBLIC
    $<BUILD_INTERFACE:${HSHM_ROOT}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
hshm_target_compile_definitions(hermes_shm_host)
list(APPEND HSHM_LIBS hermes_shm_host)
add_library(cxx INTERFACE)
target_link_libraries(cxx INTERFACE hermes_shm_host)
target_include_directories(cxx INTERFACE
    $<BUILD_INTERFACE:${HSHM_ROOT}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
hshm_target_compile_definitions(cxx)

# Create hshm::cxx alias for unified builds (when used as subdirectory)
if(NOT TARGET hshm::cxx)
  add_library(hshm::cxx ALIAS cxx)
endif()

list(APPEND HSHM_LIBS cxx)

# BUILD HSHM FOR CUDA ONLY
if(HSHM_ENABLE_CUDA)
    add_cuda_library(hermes_shm_cuda SHARED TRUE ${SRC_FILES})
    target_link_libraries(hermes_shm_cuda PUBLIC host_deps)
    target_compile_definitions(hermes_shm_cuda PUBLIC HSHM_ENABLE_CUDA=1)
    hshm_target_compile_definitions(hermes_shm_cuda)
    list(APPEND HSHM_LIBS hermes_shm_cuda)
    add_library(cudacxx INTERFACE)
    target_link_libraries(cudacxx INTERFACE hermes_shm_cuda)
    target_include_directories(cudacxx INTERFACE
        $<BUILD_INTERFACE:${HSHM_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(cudacxx INTERFACE HSHM_ENABLE_CUDA=1 HSHM_ENABLE_ROCM=0)
    hshm_target_compile_definitions(cudacxx)
    add_library(hshm::cuda_cxx ALIAS cudacxx)
    list(APPEND HSHM_LIBS cudacxx)
endif()

# BUILD HSHM FOR ROCM ONLY
if(HSHM_ENABLE_ROCM)
    add_rocm_gpu_library(hermes_shm_rocm_gpu STATIC TRUE ${SRC_FILES})
    target_link_libraries(hermes_shm_rocm_gpu PUBLIC host_deps)
    target_compile_definitions(hermes_shm_rocm_gpu PUBLIC HSHM_ENABLE_ROCM=1)
    hshm_target_compile_definitions(hermes_shm_rocm_gpu)
    list(APPEND HSHM_LIBS hermes_shm_rocm_gpu)
    add_library(rocmcxx_gpu INTERFACE)
    target_link_libraries(rocmcxx_gpu INTERFACE hermes_shm_rocm_gpu)
    target_include_directories(rocmcxx_gpu INTERFACE
        $<BUILD_INTERFACE:${HSHM_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(rocmcxx_gpu INTERFACE HSHM_ENABLE_ROCM=1 HSHM_ENABLE_CUDA=0)
    hshm_target_compile_definitions(rocmcxx_gpu)
    add_library(hshm::rocm_cxx ALIAS rocmcxx_gpu)
    list(APPEND HSHM_LIBS rocmcxx_gpu)

    add_rocm_host_library(hermes_shm_rocm_host TRUE ${SRC_FILES})
    target_link_libraries(hermes_shm_rocm_host PUBLIC host_deps)
    target_compile_definitions(hermes_shm_rocm_host PUBLIC HSHM_ENABLE_ROCM=1)
    hshm_target_compile_definitions(hermes_shm_rocm_host)
    list(APPEND HSHM_LIBS hermes_shm_rocm_host)
    add_library(rocmcxx_host INTERFACE)
    target_link_libraries(rocmcxx_host INTERFACE hermes_shm_rocm_host)
    target_include_directories(rocmcxx_host INTERFACE
        $<BUILD_INTERFACE:${HSHM_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(rocmcxx_host INTERFACE HSHM_ENABLE_ROCM=1 HSHM_ENABLE_CUDA=0)
    hshm_target_compile_definitions(rocmcxx_host)
    list(APPEND HSHM_LIBS rocmcxx_host)
endif()

# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
# Note: yaml-cpp and cereal are NOT exported - consuming projects must find them independently
# This ensures cross-platform compatibility regardless of how yaml-cpp was installed

set(_CEREAL_TARGET "")
if(TARGET cereal AND NOT TARGET cereal::cereal)
    # Only include if built as submodule (not a system/imported target)
    set(_CEREAL_TARGET cereal)
endif()

install(TARGETS
    ${HSHM_LIBS}
    configure
    serialize
    interceptor
    lightbeam
    thread_all
    mpi
    compress
    encrypt
    host_deps
    ${_CEREAL_TARGET}
    EXPORT
    ${HSHM_EXPORTED_TARGETS}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# -----------------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(hermes_shm_host)
endif()
