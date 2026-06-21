# src/CMakeLists.txt
# Main CLIO Runtime library source code

cmake_minimum_required(VERSION 3.10)

# Get all source files, excluding GPU-specific files
file(GLOB_RECURSE CHIMAERA_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Collect GPU sources from modules (not picked up by the src/ glob above)
# Only match src/ subdirectories to avoid picking up test files.
file(GLOB_RECURSE CHIMAERA_MODULE_GPU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/../modules/*/src/*_gpu.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../modules/*/src/*_sycl.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../modules/*/src/*_hip.cc
)

# Separate GPU sources (_gpu.cc / _sycl.cc / _hip.cc files) from CPU
# sources. _gpu.cc and _hip.cc are compiled by NVCC/HIPCC under
# CLIO_CORE_ENABLE_CUDA/ROCM, and skipped under SYCL via their own
# `#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM` guard. _sycl.cc is the
# SYCL twin: gated by `#if CTP_ENABLE_SYCL` and compiled by
# add_sycl_library() when CLIO_CORE_ENABLE_SYCL is on.
set(CHIMAERA_GPU_SOURCES "")
set(CHIMAERA_CPU_SOURCES "")
foreach(src ${CHIMAERA_SOURCES})
  if(src MATCHES "_(gpu|sycl|hip)\\.cc$")
    list(APPEND CHIMAERA_GPU_SOURCES ${src})
  else()
    list(APPEND CHIMAERA_CPU_SOURCES ${src})
  endif()
endforeach()

# Module GPU sources are always GPU-only
foreach(src ${CHIMAERA_MODULE_GPU_SOURCES})
  list(APPEND CHIMAERA_GPU_SOURCES ${src})
endforeach()

# Collect GPU sources from external modules (e.g., CTE core) that register
# GPU containers in the orchestrator but live outside context-runtime/modules/.
file(GLOB_RECURSE CHIMAERA_EXTERNAL_GPU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../context-transfer-engine/core/src/*_gpu.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../context-transfer-engine/core/src/*_sycl.cc
)
foreach(src ${CHIMAERA_EXTERNAL_GPU_SOURCES})
  list(APPEND CHIMAERA_GPU_SOURCES ${src})
endforeach()

# Create the main CLIO Runtime library. Canonical name is `clio_run_cxx`
# (target + .so file); legacy aliases `chimaera_cxx` and `chimaera::cxx` are
# created below for backward compat. The `CHIMAERA_LIB_NAME` variable kept
# its historical name to minimise churn across the in-file uses.
set(CHIMAERA_LIB_NAME "clio_run_cxx")
add_library(${CHIMAERA_LIB_NAME} SHARED ${CHIMAERA_CPU_SOURCES})

# Build GPU work orchestrator library if CUDA/ROCm/SYCL enabled and GPU sources exist
if(CHIMAERA_GPU_SOURCES)
  if(CLIO_CORE_ENABLE_CUDA)
    add_cuda_library(${CHIMAERA_LIB_NAME}_gpu SHARED TRUE ${CHIMAERA_GPU_SOURCES}
      INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../modules/admin/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../modules/MOD_NAME/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../modules/bdev/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../context-transport-primitives/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../context-transfer-engine/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_include_directories(${CHIMAERA_LIB_NAME}_gpu PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/admin/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/MOD_NAME/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/bdev/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../context-transport-primitives/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../context-transfer-engine/core/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
    target_link_libraries(${CHIMAERA_LIB_NAME}_gpu PUBLIC ctp::cuda_cxx)
    # PRIVATE link: prevents CTP_ENABLE_CUDA from propagating to g++ compilations
    # of chimaera_cxx sources (transport-primitives headers use CUDA builtins
    # guarded by CTP_ENABLE_CUDA which are not available in g++)
    target_link_libraries(${CHIMAERA_LIB_NAME} PRIVATE ${CHIMAERA_LIB_NAME}_gpu)
  elseif(CLIO_CORE_ENABLE_SYCL)
    add_sycl_library(${CHIMAERA_LIB_NAME}_gpu SHARED TRUE ${CHIMAERA_GPU_SOURCES}
      INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../modules/admin/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../modules/MOD_NAME/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../modules/bdev/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../context-transport-primitives/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../context-transfer-engine/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_include_directories(${CHIMAERA_LIB_NAME}_gpu PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/admin/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/MOD_NAME/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/bdev/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../context-transport-primitives/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../context-transfer-engine/core/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
    target_link_libraries(${CHIMAERA_LIB_NAME}_gpu PUBLIC ctp::cxx)
    target_link_libraries(${CHIMAERA_LIB_NAME} PRIVATE ${CHIMAERA_LIB_NAME}_gpu)
    # Add CTP_ENABLE_SYCL=1 + sycl headers to the host library (NOT -fsycl,
    # which would force a device-pass parse of every host TU and trip
    # type-mismatches on host-only methods accessed via CHI_IPC). Without
    # -fsycl, host code uses sycl::malloc_host / sycl::queue / sycl::free
    # as plain runtime API calls. The chimaera_cxx_gpu library compiled by
    # add_sycl_library still gets full -fsycl + device-pass treatment.
    # SYCL_DISABLE_FSYCL_SYCLHPP_WARNING silences the "including <sycl/sycl.hpp>
    # without -fsycl" pragma warning that <sycl/sycl.hpp> emits in host TUs.
    target_compile_definitions(${CHIMAERA_LIB_NAME} PRIVATE
        CTP_ENABLE_SYCL=1 SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1)
    # CLIO_SYCL_INCLUDE_DIR is discovered in wrp_core_enable_sycl() from the
    # actual compiler location (works on Aurora oneAPI, not just /opt/intel).
    if(CLIO_SYCL_INCLUDE_DIR)
      target_include_directories(${CHIMAERA_LIB_NAME} PRIVATE ${CLIO_SYCL_INCLUDE_DIR})
    endif()
  elseif(CLIO_CORE_ENABLE_ROCM)
    add_rocm_gpu_library(${CHIMAERA_LIB_NAME}_gpu SHARED FALSE ${CHIMAERA_GPU_SOURCES})
    target_include_directories(${CHIMAERA_LIB_NAME}_gpu PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/admin/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/MOD_NAME/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/bdev/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../context-transport-primitives/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../context-transfer-engine/core/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
    # Link ctp::rocm_cxx so CTP_ENABLE_ROCM=1 propagates to chimaera_cxx
    # (PRIVATE link to ${CHIMAERA_LIB_NAME}_gpu carries those compile
    # definitions into chimaera_cxx's own TUs without exposing them to
    # external consumers — same model as the CUDA branch above).
    target_link_libraries(${CHIMAERA_LIB_NAME}_gpu PUBLIC ctp::rocm_cxx)
    target_link_libraries(${CHIMAERA_LIB_NAME} PRIVATE ${CHIMAERA_LIB_NAME}_gpu)
  endif()

  # Backward-compat alias for the GPU twin, matching the host-lib
  # `chimaera_cxx` alias added at the bottom of this file.  In-tree
  # consumers (context-runtime/test/unit, context-transfer-engine
  # gpu_vector adapter) still link to the legacy name; adding an ALIAS
  # here lets that work without renaming every CMakeLists.  Guarded
  # on the GPU target's existence because CHIMAERA_GPU_SOURCES can be
  # non-empty even when none of CUDA/SYCL/ROCm are enabled (CPU-only
  # builds in CI), in which case no GPU target was actually defined.
  if(TARGET ${CHIMAERA_LIB_NAME}_gpu AND NOT TARGET chimaera_cxx_gpu)
    add_library(chimaera_cxx_gpu ALIAS ${CHIMAERA_LIB_NAME}_gpu)
  endif()
endif()

# Set include directories as PUBLIC so they propagate to consumers
# Use CMAKE_CURRENT_SOURCE_DIR/.. to get runtime directory (this is runtime/src)
# Also include clio_ctp headers which are in context-transport-primitives/include
target_include_directories(${CHIMAERA_LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/admin/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../context-transport-primitives/include>
  $<INSTALL_INTERFACE:include>
)

# When CUDA is enabled, g++ source files get CTP_ENABLE_CUDA=1 transitively from
# the GPU library link. They need the CUDA toolkit include directory so that
# cuda_runtime_api.h (host-callable CUDA runtime functions) can be found.
if(CLIO_CORE_ENABLE_CUDA AND CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
  find_package(CUDAToolkit QUIET)
  target_include_directories(${CHIMAERA_LIB_NAME} PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  # Link cudart for host-callable CUDA runtime functions used in ipc_manager.cc etc.
  target_link_libraries(${CHIMAERA_LIB_NAME} PUBLIC CUDA::cudart)
endif()

# ROCm twin: g++ source files get CTP_ENABLE_ROCM=1 from the chimaera_cxx_gpu
# PRIVATE link, so they need the HIP host-API headers (hip/hip_runtime_api.h) to
# resolve hipMalloc / hipMemcpy / hipIpcMemHandle_t etc. Under HIP_PLATFORM=nvidia
# the HIP runtime forwards to cudart, so cudart needs to be linked too.
if(CLIO_CORE_ENABLE_ROCM)
  target_include_directories(${CHIMAERA_LIB_NAME} PUBLIC ${ROCM_ROOT}/include)
  target_compile_definitions(${CHIMAERA_LIB_NAME} PUBLIC CTP_ENABLE_ROCM=1)
  if(CLIO_ROCM_HIP_PLATFORM STREQUAL "nvidia")
    target_compile_definitions(${CHIMAERA_LIB_NAME} PUBLIC
      __HIP_PLATFORM_NVIDIA__=1
      __HIP_PLATFORM_NVCC__=1)
    find_package(CUDAToolkit QUIET)
    if(TARGET CUDA::cudart)
      target_link_libraries(${CHIMAERA_LIB_NAME} PUBLIC CUDA::cudart)
    endif()
    if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
      target_include_directories(${CHIMAERA_LIB_NAME} PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
  endif()
endif()

# Add runtime definition
# Removed CHIMAERA_RUNTIME compile definition - using runtime IsRuntime() checks instead

# Add CMAKE_INSTALL_PREFIX as a compile definition for ChiMod search paths
target_compile_definitions(${CHIMAERA_LIB_NAME} PRIVATE
  CHI_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
  # Tell clio_runtime/api.h to emit dllexport (instead of the default
  # dllimport that consumer DLLs see).
  CLIO_RUN_BUILDING_DLL=1
)

# Link libraries
# CLIO Runtime runtime links to hshm modular targets
# Clients only link to ctp::cxx
target_link_libraries(${CHIMAERA_LIB_NAME}
  PUBLIC
    ctp::cxx
    ctp::serialize  # For cereal
    Threads::Threads
    ctp::lightbeam  # For ZMQ transport
    # chimods_admin_client  # Admin client library for global task routing - TODO: Enable when admin chimod is built
)

# Set target properties
set_target_properties(${CHIMAERA_LIB_NAME} PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER "${CLIO_RUN_ROOT}/include/clio_runtime/clio_runtime.h"
  OUTPUT_NAME "clio_run_cxx"
  # On Windows, auto-export all non-static symbols so dependent modules
  # (clio_admin_client, etc.) can link against clio_run_cxx without each
  # symbol needing an explicit __declspec(dllexport).
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# Export target for parent scope. EXPORT_NAME is intentionally `cxx` so the
# installed alias is `chimaera::cxx` (the historical install-time name that
# external consumers — coeus-adapter, etc. — still link against).
set_target_properties(${CHIMAERA_LIB_NAME} PROPERTIES EXPORT_NAME cxx)

# Precompiled headers for faster builds. Skipped under SYCL because the
# SYCL flag set (`-fsycl`) is incompatible with clang's PCH bundler:
# clang-offload-bundler refuses 'pch' as an input file type.
if(NOT CLIO_CORE_ENABLE_SYCL)
  target_precompile_headers(${CHIMAERA_LIB_NAME} PRIVATE
      <string> <vector> <memory> <unordered_map>
      <functional> <algorithm> <cstdint> <cstring> <iostream>
      <sstream> <fstream> <chrono> <mutex> <thread>
  )
endif()

# Backward-compat target aliases (one ALIAS per legacy name pointing at the
# canonical `clio_run_cxx` target). The install-time form is `chimaera::cxx`
# (kept stable for external find_package consumers), but in-tree code should
# link via the canonical `clio::run::cxx`.  The `clio::run::cxx` waypoint is
# kept as a pre-`::`-rename alias.
add_library(${CLIO_RUN_NAMESPACE}::cxx ALIAS ${CHIMAERA_LIB_NAME})  # chimaera::cxx
if(NOT TARGET clio::run::cxx)
  add_library(clio::run::cxx ALIAS ${CHIMAERA_LIB_NAME})
endif()
if(NOT TARGET clio::run::cxx)
  add_library(clio::run::cxx ALIAS ${CHIMAERA_LIB_NAME})
endif()
if(NOT TARGET chimaera_cxx)
  add_library(chimaera_cxx ALIAS ${CHIMAERA_LIB_NAME})
endif()
