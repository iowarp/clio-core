#------------------------------------------------------------------------------
# clio_enable_nvcomp_device(<target>)
#
# Turns on gpu_vector's IN-KERNEL nvcomp decode for one target: encoded pages in
# the kHbm tier are decompressed by the faulting warp itself, device pointer in
# and device pointer out, with no host round trip and no host staging.
#
# Without it, device_vector.h compiles a single-threaded scalar LZ4 decoder as
# the airtight fallback. That fallback is CORRECT but roughly two orders of
# magnitude slower than nvcomp's warp decoder, so a benchmark that silently
# lands on it reports "nvcomp is slow" while never calling nvcomp. Anything
# measuring the compressed fault path MUST call this.
#
# WHERE THE DEVICE API COMES FROM
#
# The distro nvcomp package (and the conda one) ship the HOST/manager API only:
# nvcomp/lz4.hpp, nvcompManagerFactory.hpp, libnvcomp.so. Verified on Ubuntu
# 24.04 with nvcomp 5.2.0.10 -- there is no nvcomp/device/ directory, no
# user_api.hpp and no libnvcomp_device_static.a anywhere in the package. The
# in-kernel decoder needs exactly those, and they exist only in NVIDIA's
# standalone redistributable archive.
#
# So this FETCHES that archive, the same way the build already fetches
# nlohmann_json and nanobind. It is pinned by version AND by SHA256 from
# NVIDIA's own redistrib manifest, and it lands in the build tree. It is NOT
# looked for in /opt or a home directory: a build that depends on someone
# having hand-unpacked a tarball is not reproducible, and it silently changes
# behaviour depending on whose machine it runs on.
#
# CLIO_NVCOMP_DEVICE_ROOT still exists, but purely as an override for an
# offline or air-gapped build that has the archive already. It is not required
# and nothing probes for it.
#------------------------------------------------------------------------------

set(CLIO_NVCOMP_DEVICE_ROOT "" CACHE PATH
    "Pre-unpacked nvcomp archive with lib/libnvcomp_device_static.a. Optional: \
leave empty and the archive is fetched.")
option(CLIO_NVCOMP_DEVICE_FETCH
       "Fetch the nvcomp redistributable providing the device API" ON)
option(CLIO_NVCOMP_DEVICE_REQUIRED
       "Fail configuration if in-kernel nvcomp decode is unavailable" OFF)

# 4.2.0.11 is the version the in-kernel decoder is written against: gpu_vector's
# chunk parser is byte-exact against this LZ4Manager's output (see
# ParseEncodedBlob). The cuda12 build links fine against a CUDA 13 toolkit --
# the device side is templates plus a relocatable object.
set(CLIO_NVCOMP_DEVICE_VERSION "4.2.0.11" CACHE STRING
    "nvcomp redistributable version providing the device API")
set(_clio_nvdev_url
    "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-${CLIO_NVCOMP_DEVICE_VERSION}_cuda12-archive.tar.xz")
# From NVIDIA's redistrib_4.2.0.11.json.
set(_clio_nvdev_sha256
    "0e235903b08f0173835b204e9fa90208660c33eb6a986ec178bb05ea891d5119")

# Resolved once per configure; empty means unavailable.
set(_clio_nvdev_root "")

function(_clio_nvcomp_device_resolve out_var)
  # 1. Explicit override (offline builds).
  if (CLIO_NVCOMP_DEVICE_ROOT AND
      EXISTS "${CLIO_NVCOMP_DEVICE_ROOT}/lib/libnvcomp_device_static.a")
    set(${out_var} "${CLIO_NVCOMP_DEVICE_ROOT}" PARENT_SCOPE)
    return()
  endif()

  # 2. Fetch it. Linux x86_64 only: that is the only place the in-kernel path
  #    is built, and the archive is per-platform.
  if (NOT CLIO_NVCOMP_DEVICE_FETCH OR NOT UNIX OR
      NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  include(FetchContent)
  FetchContent_Declare(clio_nvcomp_device
      URL      "${_clio_nvdev_url}"
      URL_HASH SHA256=${_clio_nvdev_sha256}
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  # Populate only -- the archive is prebuilt, there is nothing to add_subdirectory.
  FetchContent_GetProperties(clio_nvcomp_device)
  if (NOT clio_nvcomp_device_POPULATED)
    message(STATUS "Fetching nvcomp ${CLIO_NVCOMP_DEVICE_VERSION} "
                   "(device API for in-kernel decode)")
    FetchContent_Populate(clio_nvcomp_device)
  endif()
  if (EXISTS "${clio_nvcomp_device_SOURCE_DIR}/lib/libnvcomp_device_static.a")
    set(${out_var} "${clio_nvcomp_device_SOURCE_DIR}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

function(clio_enable_nvcomp_device target)
  get_target_property(_type ${target} TYPE)
  if (NOT _type STREQUAL "EXECUTABLE")
    message(FATAL_ERROR
        "clio_enable_nvcomp_device(${target}): EXECUTABLE targets only "
        "(nvcomp ships its device object non-PIC). See ggml-cuda-clio for the "
        "shared-library recipe.")
  endif()

  if (NOT _clio_nvdev_root)
    _clio_nvcomp_device_resolve(_clio_nvdev_root)
    set(_clio_nvdev_root "${_clio_nvdev_root}" PARENT_SCOPE)
  endif()

  if (NOT _clio_nvdev_root)
    set(_msg
        "${target}: in-kernel nvcomp decode OFF -- the nvcomp device API could "
        "not be obtained. The compressed fault path will use the scalar LZ4 "
        "fallback, which is ~100x slower and is NOT a measurement of nvcomp. "
        "Tests that need it SKIP rather than reporting a fallback result.")
    if (CLIO_NVCOMP_DEVICE_REQUIRED)
      message(FATAL_ERROR ${_msg})
    endif()
    message(WARNING ${_msg})
    return()
  endif()

  # -rdc: the nvcomp device decompressor is an external device function, so the
  # call must survive to a separate device link step.
  set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(${target} PRIVATE CLIO_GV_NVCOMP_DEVICE)
  target_include_directories(${target} BEFORE PRIVATE
      "${_clio_nvdev_root}/include")
  # Plain signature on purpose: add_cuda_executable() links these targets
  # without keywords, and CMake forbids mixing the two forms on one target.
  target_link_libraries(${target}
      "${_clio_nvdev_root}/lib/libnvcomp_device_static.a")
  # nvcomp ships these objects non-PIC, which a PIE executable cannot absorb
  # ("relocation R_X86_64_32 ... can not be used when making a PIE object").
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE OFF)
  target_link_options(${target} PRIVATE -no-pie)
  message(STATUS "${target}: in-kernel nvcomp decode ON (${_clio_nvdev_root})")
endfunction()
