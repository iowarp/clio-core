project(hermes_shm)

# -----------------------------------------------------------------------------
# Build HSHM
# -----------------------------------------------------------------------------
set(HSHM_LIBS "")

# BUILD HSHM FOR HOST ONLY
if(WIN32)
    set(SRC_FILES
        system_info.cc
        malloc_allocator.cc
        socket_win.cc
    )
else()
    set(SRC_FILES
        system_info.cc
        malloc_allocator.cc
        socket_posix.cc
    )
endif()

# Create library with system_info.cc
add_library(hermes_shm_host ${SRC_FILES})

#------------------------------------------------------------------------------
# Note: Compression Predictor Libraries (XGBoost, Dense NN, Q-Table, LinReg)
# have been moved to the compressor chimod in context-transfer-engine/compressor
#------------------------------------------------------------------------------

target_link_libraries(hermes_shm_host PUBLIC
    ${REAL_TIME_FLAGS}
    configure
    thread_all
)
if(WIN32)
    target_link_libraries(hermes_shm_host PUBLIC ws2_32)
endif()
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
if(WRP_CORE_ENABLE_CUDA)
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
if(WRP_CORE_ENABLE_ROCM)
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

# Install compiled libraries (interface libraries are installed in parent CMakeLists.txt)
install(TARGETS
    ${HSHM_LIBS}
    ${_CEREAL_TARGET}
    EXPORT
    ${HSHM_EXPORTED_TARGETS}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

