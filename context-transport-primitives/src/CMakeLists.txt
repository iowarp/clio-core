project(hermes_shm)

# -----------------------------------------------------------------------------
# Build HSHM
# -----------------------------------------------------------------------------
set(HSHM_LIBS "")

# BUILD HSHM FOR HOST ONLY
set(SRC_FILES

    # system_info.cc   # NOTE(llogan): Commented out because ROCm is annoying and buggy
    memory_manager.cc)
add_library(hermes_shm_host ${SRC_FILES})
target_link_libraries(hermes_shm_host PUBLIC
    ${REAL_TIME_FLAGS}
    configure
    thread_all
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
    target_compile_definitions(hermes_shm_cuda PUBLIC HSHM_ENABLE_CUDA)
    hshm_target_compile_definitions(hermes_shm_cuda)
    list(APPEND HSHM_LIBS hermes_shm_cuda)
    add_library(cudacxx INTERFACE)
    target_link_libraries(cudacxx INTERFACE hermes_shm_cuda)
    target_include_directories(cudacxx INTERFACE
        $<BUILD_INTERFACE:${HSHM_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    hshm_target_compile_definitions(cudacxx)
    list(APPEND HSHM_LIBS cudacxx)
endif()

# BUILD HSHM FOR ROCM ONLY
if(HSHM_ENABLE_ROCM)
    add_rocm_gpu_library(hermes_shm_rocm_gpu STATIC TRUE ${SRC_FILES})
    target_link_libraries(hermes_shm_rocm_gpu PUBLIC host_deps)
    target_compile_definitions(hermes_shm_rocm_gpu PUBLIC HSHM_ENABLE_ROCM)
    hshm_target_compile_definitions(hermes_shm_rocm_gpu)
    list(APPEND HSHM_LIBS hermes_shm_rocm_gpu)
    add_library(rocmcxx_gpu INTERFACE)
    target_link_libraries(rocmcxx_gpu INTERFACE hermes_shm_rocm_gpu)
    target_include_directories(rocmcxx_gpu INTERFACE
        $<BUILD_INTERFACE:${HSHM_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    hshm_target_compile_definitions(rocmcxx_gpu)
    list(APPEND HSHM_LIBS rocmcxx_gpu)

    add_rocm_host_library(hermes_shm_rocm_host TRUE ${SRC_FILES})
    target_link_libraries(hermes_shm_rocm_host PUBLIC host_deps)
    hshm_target_compile_definitions(hermes_shm_rocm_host)
    list(APPEND HSHM_LIBS hermes_shm_rocm_host)
    add_library(rocmcxx_host INTERFACE)
    target_link_libraries(rocmcxx_host INTERFACE hermes_shm_rocm_host)
    target_include_directories(rocmcxx_host INTERFACE
        $<BUILD_INTERFACE:${HSHM_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    hshm_target_compile_definitions(rocmcxx_host)
    list(APPEND HSHM_LIBS rocmcxx_host)
endif()

# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
# Include yaml-cpp and cereal in export if they were built as submodules
set(_YAML_CPP_TARGET "")
if(TARGET yaml-cpp)
    set(_YAML_CPP_TARGET yaml-cpp)
endif()

set(_CEREAL_TARGET "")
if(TARGET cereal)
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
    ${_YAML_CPP_TARGET}
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
