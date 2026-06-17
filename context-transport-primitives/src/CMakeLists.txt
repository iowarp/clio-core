project(clio_ctp)

# -----------------------------------------------------------------------------
# Build CTP
# -----------------------------------------------------------------------------
set(CLIO_CTP_LIBS "")

# BUILD CTP FOR HOST ONLY
if(WIN32)
    set(SRC_FILES
        system_info.cc
        malloc_allocator.cc
        socket_win.cc
        # Windows-specific backends whose impl pulls in <windows.h>.
        # Headers stay Win32-clean; impls live here.
        iocp_io.cc
        event_manager_win.cc
    )
else()
    set(SRC_FILES
        system_info.cc
        malloc_allocator.cc
        socket_posix.cc
    )
endif()

# Create library with system_info.cc
add_library(clio_ctp_host ${SRC_FILES})
set_target_properties(clio_ctp_host PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

#------------------------------------------------------------------------------
# Note: Compression Predictor Libraries (XGBoost, Dense NN, Q-Table, LinReg)
# have been moved to the compressor chimod in context-transfer-engine/compressor
#------------------------------------------------------------------------------

target_link_libraries(clio_ctp_host PUBLIC
    ${REAL_TIME_FLAGS}
    configure
    thread_all
)
if(WIN32)
    target_link_libraries(clio_ctp_host PUBLIC ws2_32)
endif()
target_include_directories(clio_ctp_host PUBLIC
    $<BUILD_INTERFACE:${CLIO_CTP_ROOT}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
hshm_target_compile_definitions(clio_ctp_host)
if(CLIO_CORE_ENABLE_PCH)
    target_precompile_headers(clio_ctp_host PRIVATE
        <string> <vector> <memory> <unordered_map>
        <functional> <algorithm> <cstdint> <cstring> <iostream>
    )
endif()
list(APPEND CLIO_CTP_LIBS clio_ctp_host)
add_library(cxx INTERFACE)
target_link_libraries(cxx INTERFACE clio_ctp_host)
target_include_directories(cxx INTERFACE
    $<BUILD_INTERFACE:${CLIO_CTP_ROOT}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
hshm_target_compile_definitions(cxx)

# Create ctp::cxx alias for unified builds (when used as subdirectory)
if(NOT TARGET ctp::cxx)
  add_library(ctp::cxx ALIAS cxx)
endif()

list(APPEND CLIO_CTP_LIBS cxx)

# BUILD CTP FOR CUDA ONLY
if(CLIO_CORE_ENABLE_CUDA)
    add_cuda_library(clio_ctp_cuda SHARED TRUE ${SRC_FILES}
      INCLUDE_DIRS
        ${CLIO_CTP_ROOT}/include
        ${PROJECT_BINARY_DIR}/include
    )
    target_include_directories(clio_ctp_cuda PUBLIC
        $<BUILD_INTERFACE:${CLIO_CTP_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(clio_ctp_cuda PUBLIC host_deps)
    target_compile_definitions(clio_ctp_cuda PUBLIC CTP_ENABLE_CUDA=1)
    hshm_target_compile_definitions(clio_ctp_cuda)
    list(APPEND CLIO_CTP_LIBS clio_ctp_cuda)
    add_library(cudacxx INTERFACE)
    # cudadevrt: CUDA device runtime, required to device-link relocatable device
    # code (-rdc=true) / dynamic parallelism. Carried on ctp::cuda_cxx so every
    # GPU consumer gets it transitively — no per-target CUDA links needed.
    #
    # Resolve it to a full path rather than the bare name `cudadevrt`: bare
    # -lcudadevrt isn't on the default linker search path on Linux (only Windows
    # happened to find it). Prefer the imported CUDA::cudadevrt target when the
    # toolkit's CMake package defines it; CUDA 13's FindCUDAToolkit no longer
    # creates that target, so fall back to find_library() against the toolkit
    # lib dir, and only then to the bare name as a last resort.
    find_package(CUDAToolkit QUIET)
    if(TARGET CUDA::cudadevrt)
        target_link_libraries(cudacxx INTERFACE clio_ctp_cuda CUDA::cudadevrt)
    else()
        find_library(CLIO_CUDADEVRT_LIB cudadevrt
            HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_BIN_DIR}/../lib64
            PATH_SUFFIXES lib64 lib)
        if(CLIO_CUDADEVRT_LIB)
            target_link_libraries(cudacxx INTERFACE clio_ctp_cuda ${CLIO_CUDADEVRT_LIB})
        else()
            message(WARNING "cudadevrt not found via find_library; falling back to "
                            "bare -lcudadevrt (may fail to link on Linux)")
            target_link_libraries(cudacxx INTERFACE clio_ctp_cuda cudadevrt)
        endif()
    endif()
    target_include_directories(cudacxx INTERFACE
        $<BUILD_INTERFACE:${CLIO_CTP_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(cudacxx INTERFACE CTP_ENABLE_CUDA=1 CTP_ENABLE_ROCM=0)
    hshm_target_compile_definitions(cudacxx)
    add_library(ctp::cuda_cxx ALIAS cudacxx)
    list(APPEND CLIO_CTP_LIBS cudacxx)

    # GPU Linker library (CUDA driver API for runtime device code linking)
    add_library(clio_ctp_gpu_linker gpu_linker.cc)
    target_link_libraries(clio_ctp_gpu_linker PUBLIC clio_ctp_host cuda)
    target_compile_definitions(clio_ctp_gpu_linker PUBLIC CTP_ENABLE_CUDA=1)
    hshm_target_compile_definitions(clio_ctp_gpu_linker)
    if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
      target_include_directories(clio_ctp_gpu_linker PUBLIC
          ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    list(APPEND CLIO_CTP_LIBS clio_ctp_gpu_linker)
    add_library(ctp::gpu_linker ALIAS clio_ctp_gpu_linker)
endif()

# BUILD CTP FOR ROCM ONLY
if(CLIO_CORE_ENABLE_ROCM)
    add_rocm_gpu_library(clio_ctp_rocm_gpu SHARED TRUE ${SRC_FILES})
    target_link_libraries(clio_ctp_rocm_gpu PUBLIC host_deps)
    target_compile_definitions(clio_ctp_rocm_gpu PUBLIC CTP_ENABLE_ROCM=1)
    hshm_target_compile_definitions(clio_ctp_rocm_gpu)
    list(APPEND CLIO_CTP_LIBS clio_ctp_rocm_gpu)
    add_library(rocmcxx_gpu INTERFACE)
    target_link_libraries(rocmcxx_gpu INTERFACE clio_ctp_rocm_gpu)
    target_include_directories(rocmcxx_gpu INTERFACE
        $<BUILD_INTERFACE:${CLIO_CTP_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(rocmcxx_gpu INTERFACE CTP_ENABLE_ROCM=1 CTP_ENABLE_CUDA=0)
    hshm_target_compile_definitions(rocmcxx_gpu)
    add_library(ctp::rocm_cxx ALIAS rocmcxx_gpu)
    list(APPEND CLIO_CTP_LIBS rocmcxx_gpu)

    add_rocm_host_library(clio_ctp_rocm_host TRUE ${SRC_FILES})
    target_link_libraries(clio_ctp_rocm_host PUBLIC host_deps)
    target_compile_definitions(clio_ctp_rocm_host PUBLIC CTP_ENABLE_ROCM=1)
    hshm_target_compile_definitions(clio_ctp_rocm_host)
    list(APPEND CLIO_CTP_LIBS clio_ctp_rocm_host)
    add_library(rocmcxx_host INTERFACE)
    target_link_libraries(rocmcxx_host INTERFACE clio_ctp_rocm_host)
    target_include_directories(rocmcxx_host INTERFACE
        $<BUILD_INTERFACE:${CLIO_CTP_ROOT}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(rocmcxx_host INTERFACE CTP_ENABLE_ROCM=1 CTP_ENABLE_CUDA=0)
    hshm_target_compile_definitions(rocmcxx_host)
    list(APPEND CLIO_CTP_LIBS rocmcxx_host)
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
    ${CLIO_CTP_LIBS}
    ${_CEREAL_TARGET}
    EXPORT
    ${CLIO_CTP_EXPORTED_TARGETS}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

