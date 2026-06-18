cmake_minimum_required(VERSION 3.10)

# Collect optional NIXL dependency
set(BDEV_OPTIONAL_LIBS "")
if(CLIO_CORE_ENABLE_NIXL)
  list(APPEND BDEV_OPTIONAL_LIBS ctp::nixl)
endif()

# Collect optional CUDA dependency for kHbm/kPinned bdev types
set(BDEV_GPU_LIBS "")
if(CLIO_CORE_ENABLE_CUDA)
  list(APPEND BDEV_GPU_LIBS ctp::cuda_cxx)
elseif(CLIO_CORE_ENABLE_ROCM)
  list(APPEND BDEV_GPU_LIBS ctp::rocm_cxx)
endif()

set(BDEV_S3_SOURCES "")
set(BDEV_S3_LIBS "")
if(CLIO_ENABLE_AMAZON_DRIVE)
  list(APPEND BDEV_S3_SOURCES src/transports/s3_bdev_transport.cc)
  list(APPEND BDEV_S3_LIBS aws-cpp-sdk-s3 aws-cpp-sdk-core)
endif()

# Client library — ctp::aio provides async I/O headers via its INTERFACE includes.
# LIB_NAME pins the library to `clio_bdev_client` (the new brand-aligned
# name); ALIAS keeps `chimaera_bdev_client` working as a CMake target for
# any in-tree or out-of-tree consumer that still references the old name.
add_clio_module_client(
  LIB_NAME clio_bdev
  SOURCES
    src/bdev_client.cc
  LINK_LIBRARIES
    clio_admin_client
    ctp::aio
    ${BDEV_OPTIONAL_LIBS}
)
add_library(chimaera_bdev_client ALIAS clio_bdev_client)

# Runtime library (see client block above for the LIB_NAME + ALIAS rationale).
add_clio_module_runtime(
  LIB_NAME clio_bdev
  SOURCES
    src/bdev_runtime.cc
    src/transports/block_allocator.cc
    src/transports/bdev_transport.cc
    src/transports/fs_bdev_transport.cc
    src/transports/mem_bdev_transport.cc
    ${BDEV_S3_SOURCES}
    src/autogen/bdev_lib_exec.cc
  LINK_LIBRARIES
    clio_admin_runtime
    ctp::aio
    ${BDEV_OPTIONAL_LIBS}
    ${BDEV_GPU_LIBS}
    ${BDEV_S3_LIBS}
)
add_library(chimaera_bdev_runtime ALIAS clio_bdev_runtime)

# SYCL HBM bdev support: a separate task. Adding -fsycl to
# clio_bdev_runtime conflicts with the runtime's precompiled
# headers (clang-offload-bundler rejects .pch). When that's resolved
# (likely needs PCH disabled for SYCL builds), enable here:
#
#   if(CLIO_CORE_ENABLE_SYCL AND TARGET clio_bdev_runtime)
#     set_target_properties(clio_bdev_runtime PROPERTIES
#       DISABLE_PRECOMPILE_HEADERS ON)
#     wrp_core_apply_sycl_flags(clio_bdev_runtime)
#   endif()
#
# Until then, the kHbm bdev path under SYCL is compiled out by the
# `#if CTP_ENABLE_SYCL && !(CUDA||ROCM)` guard in src/bdev_runtime.cc.

# Unit tests
if(CLIO_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()
