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

# Google Cloud Storage transport (kGcs). Talks to GCS over HTTPS via Poco; the
# root CMakeLists already auto-disables CLIO_ENABLE_GOOGLE_CLOUD if Poco is absent.
set(BDEV_GCS_SOURCES "")
set(BDEV_GCS_LIBS "")
if(CLIO_ENABLE_GOOGLE_CLOUD)
  list(APPEND BDEV_GCS_SOURCES src/transports/gcs_bdev_transport.cc)
  list(APPEND BDEV_GCS_LIBS Poco::Net Poco::NetSSL Poco::Crypto)
endif()

# Feature flags must reach the compiler as -D defines, not just gate the source
# list: the transport .cc files and the factory switch are guarded by
# #ifdef CLIO_ENABLE_AMAZON_DRIVE / #ifdef CLIO_ENABLE_GOOGLE_CLOUD. Propagated
# below via the runtime target's COMPILE_DEFINITIONS.
set(BDEV_CLOUD_DEFS "")
if(CLIO_ENABLE_AMAZON_DRIVE)
  list(APPEND BDEV_CLOUD_DEFS CLIO_ENABLE_AMAZON_DRIVE=1)
endif()
if(CLIO_ENABLE_GOOGLE_CLOUD)
  list(APPEND BDEV_CLOUD_DEFS CLIO_ENABLE_GOOGLE_CLOUD=1)
endif()

# Client library — ctp::aio provides async I/O headers via its INTERFACE includes.
# LIB_NAME pins the library to `clio_bdev_client` (the new brand-aligned
# name); ALIAS keeps `clio_bdev_client` working as a CMake target for
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
    ${BDEV_GCS_SOURCES}
    src/autogen/bdev_lib_exec.cc
  COMPILE_DEFINITIONS
    ${BDEV_CLOUD_DEFS}
  LINK_LIBRARIES
    clio_admin_runtime
    ctp::aio
    ${BDEV_OPTIONAL_LIBS}
    ${BDEV_GPU_LIBS}
    ${BDEV_S3_LIBS}
    ${BDEV_GCS_LIBS}
)

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
