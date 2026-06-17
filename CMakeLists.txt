cmake_minimum_required(VERSION 3.20)
project(iowarp-core VERSION 2.1.0 LANGUAGES C CXX)

#------------------------------------------------------------------------------
# MSVC-like front end detection
#------------------------------------------------------------------------------
# True for cl.exe (MSVC) and for compilers that drive an MSVC-compatible command
# line: clang-cl and Intel's icx.exe on Windows. These accept "/"-style flags
# (e.g. /Zi, /Od, /Fo<obj>) and MUST NOT be given GNU-style flags. Mixing GNU
# flags (-g -O0 -Wfatal-errors) onto icx.exe flips it into clang driver mode,
# where it ignores /Fo<obj> and writes <basename>.o into the build root --
# leaving CMake/ninja's expected .obj absent and breaking the DLL link
# (WINDOWS_EXPORT_ALL_SYMBOLS / bindexplib). See issue #555.
if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    set(CLIO_CORE_MSVC_LIKE ON)
else()
    set(CLIO_CORE_MSVC_LIKE OFF)
endif()

#------------------------------------------------------------------------------
# Project Options
#==============================================================================
# BUILD CONFIGURATION OPTIONS
#==============================================================================
# All build options are centralized here for easy configuration and discovery.
# Options are organized by component for clarity.
#==============================================================================

#------------------------------------------------------------------------------
# Component Enable/Disable
#------------------------------------------------------------------------------
option(CLIO_CORE_ENABLE_RUNTIME "Enable runtime component" ON)
option(CLIO_CORE_ENABLE_CTE "Enable context-transfer-engine component" ON)
option(CLIO_CORE_ENABLE_CAE "Enable context-assimilation-engine component" ON)
option(CLIO_CORE_ENABLE_CEE "Enable context-exploration-engine component" ON)

#------------------------------------------------------------------------------
# Global Master Switches
#------------------------------------------------------------------------------
option(CLIO_CORE_ENABLE_TESTS "Master switch: Enable testing for all components" OFF)
option(CLIO_CORE_ENABLE_BENCHMARKS "Master switch: Enable benchmarks for all components" ON)
option(CLIO_CORE_ENABLE_PYTHON "Enable Python bindings for all components" OFF)
option(CLIO_CORE_ENABLE_CMAKE_DOTENV "Enable reading environment variables from .env.cmake" OFF)
# Precompiled headers speed up builds but conflict with WINDOWS_EXPORT_ALL_SYMBOLS
# under the Ninja generator with non-MSVC Windows compilers (icx/clang-cl): the
# bindexplib pass (cmake -E __create_def) fails to read the cmake_pch object.
# Default ON (no change for existing builds); turn OFF for those toolchains.
option(CLIO_CORE_ENABLE_PCH "Enable precompiled headers for faster builds" ON)
option(CLIO_CORE_ENABLE_ASAN "Enable AddressSanitizer + LeakSanitizer for debugging" OFF)
option(CLIO_CORE_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer for debugging" OFF)
option(CLIO_CORE_ENABLE_MSAN "Enable MemorySanitizer for debugging (requires instrumented dependencies)" OFF)
option(CLIO_CORE_ENABLE_COVERAGE "Enable code coverage instrumentation for all components" OFF)
option(CLIO_CORE_ENABLE_DOXYGEN "Check how well the code is documented" OFF)
option(CLIO_CORE_ENABLE_RPATH "Enable RPATH configuration with relative ($ORIGIN) paths" ON)
option(CLIO_CORE_ENABLE_CPACK "Enable CPack for package generation" OFF)
option(CLIO_CORE_ENABLE_DEB_PACKAGE "Build Debian .deb package via CPack" OFF)
option(CLIO_CORE_ENABLE_RPM_PACKAGE "Build RPM .rpm package via CPack" OFF)
option(CLIO_CORE_ENABLE_APPIMAGE "Build portable AppImage via appimagetool" OFF)
option(CLIO_CORE_ENABLE_CONDA "Enable Conda environment configuration (adds Conda paths to include/link directories)" OFF)
option(CLIO_CORE_ENABLE_DOCKER_CI "Enable Docker-based integration tests (requires Docker)" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)
option(CLIO_CORE_STATIC_DEPS "Link external dependencies statically (for self-contained wheels)" OFF)

# Installation Prefix Configuration
# - For Conda builds: Set CMAKE_INSTALL_PREFIX via -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
# - For custom installs: Set CMAKE_INSTALL_PREFIX to desired root directory
# Libraries install to ${CMAKE_INSTALL_PREFIX}/lib
# Python bindings install to the detected Python's site-packages (venv, conda, or system)

#------------------------------------------------------------------------------
# Conda Environment Configuration (Explicit)
#------------------------------------------------------------------------------
# Conda support is enabled explicitly via CLIO_CORE_ENABLE_CONDA option.
# When enabled, the build system expects CONDA_PREFIX to be set.
if(CLIO_CORE_ENABLE_CONDA)
    if(NOT DEFINED ENV{CONDA_PREFIX})
        message(FATAL_ERROR "CLIO_CORE_ENABLE_CONDA is ON but CONDA_PREFIX environment variable is not set")
    endif()

    set(IN_CONDA_BUILD TRUE)
    message(STATUS "Conda environment enabled: $ENV{CONDA_PREFIX}")

    # Add conda prefix to CMAKE_PREFIX_PATH for dependency discovery
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")

    # Conda-specific library and include paths
    list(APPEND CMAKE_PREFIX_PATH
        "$ENV{CONDA_PREFIX}/lib/cmake"
        "$ENV{CONDA_PREFIX}/share/cmake"
    )
    include_directories(SYSTEM "$ENV{CONDA_PREFIX}/include")
    link_directories("$ENV{CONDA_PREFIX}/lib")

    message(STATUS "Conda build configuration:")
    message(STATUS "  Conda prefix:       $ENV{CONDA_PREFIX}")
    message(STATUS "  Install prefix:     ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  CMAKE_PREFIX_PATH:  ${CMAKE_PREFIX_PATH}")
else()
    set(IN_CONDA_BUILD FALSE)
    message(STATUS "Conda environment support disabled (use -DCLIO_CORE_ENABLE_CONDA=ON to enable)")
endif()

# Print build system summary
message(STATUS "Build system configuration:")
message(STATUS "  Conda build:        ${IN_CONDA_BUILD}")
message(STATUS "  Install prefix:     ${CMAKE_INSTALL_PREFIX}")

#------------------------------------------------------------------------------
# Transport and Network Features
#------------------------------------------------------------------------------
option(CLIO_CORE_ENABLE_MPI "Enable MPI support" OFF)
option(CLIO_CORE_ENABLE_ZMQ "Enable ZeroMQ transport" ON)
option(CLIO_CORE_ENABLE_LIBFABRIC "Enable Libfabric transport" OFF)
option(CLIO_CORE_ENABLE_THALLIUM "Build tests which depend on thallium" OFF)
option(CLIO_CORE_ENABLE_REDIS "Find hiredis and build the Redis comparison benchmark" OFF)

#------------------------------------------------------------------------------
# Data and Serialization Features
#------------------------------------------------------------------------------
option(CLIO_CORE_ENABLE_CEREAL "Enable serialization using cereal" ON)
option(CLIO_CTE_ENABLE_COMPRESS "Enable compression support (builds compressor chimod and tests)" OFF)
option(CLIO_CORE_ENABLE_ENCRYPT "Enable encryption" OFF)
option(CLIO_CORE_ENABLE_IO_URING "Enable io_uring async I/O backend (Linux 5.1+, requires liburing)" OFF)

# Set internal compression flags from CLIO_CTE_ENABLE_COMPRESS
if(CLIO_CTE_ENABLE_COMPRESS)
    set(CLIO_CORE_ENABLE_COMPRESS ON)
else()
    set(CLIO_CORE_ENABLE_COMPRESS OFF)
endif()

#------------------------------------------------------------------------------
# System and Runtime Features
#------------------------------------------------------------------------------
option(CLIO_CORE_ENABLE_ELF "Enable ELF support (required for CTE adapters)" OFF)
option(CLIO_CORE_ENABLE_OPENMP "Enable the use of OpenMP" OFF)
option(CLIO_CORE_ENABLE_CUDA "Enable CUDA support" OFF)
option(CLIO_CORE_ENABLE_GPU_RUNTIME "Enable GPU runtime (CDP-based parallelism)" OFF)
option(CLIO_CORE_ENABLE_ROCM "Enable ROCm support" OFF)
option(CLIO_CORE_ENABLE_NIXL "Enable NIXL (NVIDIA Inference Xfer Library) transport" OFF)
option(CLIO_CORE_ENABLE_NVSHMEM "Enable NVSHMEM GPU-to-GPU communication library" OFF)
option(CLIO_CORE_ENABLE_SYCL "Enable Intel GPU support via SYCL/oneAPI (icpx -fsycl)" OFF)
option(CLIO_CORE_ENABLE_JARVIS "Enable Jarvis CI infrastructure installation" OFF)

#------------------------------------------------------------------------------
# ClioCtp (context-transport-primitives) Options
#------------------------------------------------------------------------------
option(CLIO_CTP_ENABLE_PTHREADS "Support spawning pthreads" OFF)
option(CLIO_CTP_ENABLE_WINDOWS_THREADS "Support spawning windows threads" OFF)
option(CLIO_CTP_DEBUG_LOCK "Used for debugging locks" OFF)
option(CLIO_CTP_NO_COMPILE "Disable compiling / installing this library" OFF)
set(CLIO_CTP_LOG_LEVEL "1" CACHE STRING "Log level threshold (0=Debug, 1=Info, 2=Warning, 3=Error, 4=Fatal)")

#------------------------------------------------------------------------------
# CLIO Runtime (context-runtime) Options
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# CTE (context-transfer-engine) Options
#------------------------------------------------------------------------------
# CTE Adapter Options (require CLIO_CORE_ENABLE_ELF=ON)
option(CLIO_CTE_ENABLE_POSIX_ADAPTER "Enable POSIX adapter" ON)
option(CLIO_CTE_ENABLE_STDIO_ADAPTER "Enable STDIO adapter" OFF)
option(CLIO_CTE_ENABLE_MPIIO_ADAPTER "Enable MPI-IO adapter" OFF)
option(CLIO_CTE_ENABLE_VFD "Enable HDF5 VFD adapter" OFF)
option(CLIO_CTE_ENABLE_NVIDIA_GDS_ADAPTER "Enable NVIDIA GDS adapter" OFF)
option(CLIO_CTE_ENABLE_ADIOS2_ADAPTER "Enable ADIOS2 adapter" OFF)
option(CLIO_CTE_ENABLE_FUSE_ADAPTER "Enable FUSE3 adapter" OFF)
option(CLIO_CTE_ENABLE_HDF5_VOL "Enable HDF5 VOL connector (requires HDF5 >= 1.14)" OFF)

#------------------------------------------------------------------------------
# CAE (context-assimilation-engine) Options
#------------------------------------------------------------------------------
option(CAE_ENABLE_GLOBUS "Enable Globus transfer support in CAE" OFF)
# LLM labeling/summary + semantic-search features and their integration tests
# require a running Ollama service. OFF by default so default builds don't
# build or run them without Ollama.
option(CLIO_CAE_ENABLE_SUMMARY "Enable CAE LLM labeling/summary + semantic-search (requires a running Ollama service)" OFF)

#------------------------------------------------------------------------------
# External Examples and Applications
#------------------------------------------------------------------------------
option(CLIO_CORE_ENABLE_GRAY_SCOTT "Enable Gray-Scott ADIOS2 example (requires ADIOS2)" OFF)
option(CLIO_CORE_ENABLE_LLAMA "Enable llama.cpp LLM inference integration (llm-hooks)" OFF)
option(CLIO_CORE_ENABLE_LLAMA_SERVER "Add external/llama.cpp to the build (requires CLIO_CORE_ENABLE_LLAMA=ON)" OFF)

#==============================================================================
# END BUILD CONFIGURATION OPTIONS
#==============================================================================

# Hierarchical control logic:
# If CLIO_CORE_ENABLE_TESTS is OFF, force all component tests OFF
# If CLIO_CORE_ENABLE_BENCHMARKS is OFF, force all component benchmarks OFF
# Otherwise, use individual component values

# Apply master test switch: propagate to CMake's standard BUILD_TESTING variable
# so components that use if(BUILD_TESTING) are also controlled by this flag.
if(NOT CLIO_CORE_ENABLE_TESTS)
  set(BUILD_TESTING OFF CACHE BOOL "Build tests (controlled by CLIO_CORE_ENABLE_TESTS)" FORCE)
endif()

# Set CTP_ENABLE_* aliases for backward compatibility
# These allow component code to continue using CTP_ENABLE_* while we transition
# Mark them as cache variables to prevent options from overriding them
set(CLIO_CTP_ENABLE_MPI ${CLIO_CORE_ENABLE_MPI} CACHE BOOL "Enable MPI (controlled by CLIO_CORE_ENABLE_MPI)" FORCE)
set(CLIO_CTP_ENABLE_ZMQ ${CLIO_CORE_ENABLE_ZMQ} CACHE BOOL "Enable ZMQ (controlled by CLIO_CORE_ENABLE_ZMQ)" FORCE)
set(CLIO_CTP_ENABLE_LIBFABRIC ${CLIO_CORE_ENABLE_LIBFABRIC} CACHE BOOL "Enable Libfabric (controlled by CLIO_CORE_ENABLE_LIBFABRIC)" FORCE)
set(CLIO_CTP_ENABLE_THALLIUM ${CLIO_CORE_ENABLE_THALLIUM} CACHE BOOL "Enable Thallium (controlled by CLIO_CORE_ENABLE_THALLIUM)" FORCE)
set(CLIO_CTP_ENABLE_OPENMP ${CLIO_CORE_ENABLE_OPENMP} CACHE BOOL "Enable OpenMP (controlled by CLIO_CORE_ENABLE_OPENMP)" FORCE)
set(CLIO_CTP_ENABLE_CEREAL ${CLIO_CORE_ENABLE_CEREAL} CACHE BOOL "Enable Cereal (controlled by CLIO_CORE_ENABLE_CEREAL)" FORCE)
set(CLIO_CTP_ENABLE_DOXYGEN ${CLIO_CORE_ENABLE_DOXYGEN} CACHE BOOL "Enable Doxygen (controlled by CLIO_CORE_ENABLE_DOXYGEN)" FORCE)
set(CLIO_CTP_ENABLE_COMPRESS ${CLIO_CTE_ENABLE_COMPRESS} CACHE BOOL "Enable Compress (controlled by CLIO_CTE_ENABLE_COMPRESS)" FORCE)
set(CLIO_CTP_ENABLE_ENCRYPT ${CLIO_CORE_ENABLE_ENCRYPT} CACHE BOOL "Enable Encrypt (controlled by CLIO_CORE_ENABLE_ENCRYPT)" FORCE)
set(CLIO_CTP_ENABLE_ELF ${CLIO_CORE_ENABLE_ELF} CACHE BOOL "Enable ELF (controlled by CLIO_CORE_ENABLE_ELF)" FORCE)
set(CLIO_CTP_ENABLE_CUDA ${CLIO_CORE_ENABLE_CUDA} CACHE BOOL "Enable CUDA (controlled by CLIO_CORE_ENABLE_CUDA)" FORCE)
set(CLIO_CTP_ENABLE_ROCM ${CLIO_CORE_ENABLE_ROCM} CACHE BOOL "Enable ROCm (controlled by CLIO_CORE_ENABLE_ROCM)" FORCE)
set(CLIO_CTP_ENABLE_SYCL ${CLIO_CORE_ENABLE_SYCL} CACHE BOOL "Enable SYCL (controlled by CLIO_CORE_ENABLE_SYCL)" FORCE)
set(CLIO_CTP_ENABLE_IO_URING ${CLIO_CORE_ENABLE_IO_URING} CACHE BOOL "Enable io_uring (controlled by CLIO_CORE_ENABLE_IO_URING)" FORCE)
set(CLIO_CTP_ENABLE_NIXL ${CLIO_CORE_ENABLE_NIXL} CACHE BOOL "Enable NIXL (controlled by CLIO_CORE_ENABLE_NIXL)" FORCE)
set(CLIO_CTP_ENABLE_NVSHMEM ${CLIO_CORE_ENABLE_NVSHMEM} CACHE BOOL "Enable NVSHMEM (controlled by CLIO_CORE_ENABLE_NVSHMEM)" FORCE)
# Note: Coverage is handled globally via add_compile_options when CLIO_CORE_ENABLE_COVERAGE=ON
# No need for component-level CTP_ENABLE_COVERAGE

# Print final test/benchmark configuration
message(STATUS "Test/Benchmark Configuration:")
message(STATUS "  CLIO_CORE_ENABLE_TESTS: ${CLIO_CORE_ENABLE_TESTS}")
message(STATUS "  CLIO_CORE_ENABLE_BENCHMARKS: ${CLIO_CORE_ENABLE_BENCHMARKS}")

# Read environment variables from .env.cmake if enabled
if(CLIO_CORE_ENABLE_CMAKE_DOTENV)
  if(EXISTS "${CMAKE_SOURCE_DIR}/.env.cmake")
    include("${CMAKE_SOURCE_DIR}/.env.cmake")
  endif()
endif()

#------------------------------------------------------------------------------
# Load ClioCoreCommon utilities
#------------------------------------------------------------------------------
# Provides macros for dependency resolution and ChiMod build functions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(ClioCoreCommon)

#------------------------------------------------------------------------------
# External Library Discovery
#------------------------------------------------------------------------------
# Priority order for dependency discovery:
# 1. Conda environment (if IN_CONDA_BUILD is TRUE)
# 2. System packages via pkg-config
# 3. CMake find_package() in standard locations

# Static dependency linking (for self-contained pip wheels)
if(CLIO_CORE_STATIC_DEPS)
    message(STATUS "Static dependency linking enabled")
endif()

# Pkg-Config
if(NOT WIN32)
    find_package(PkgConfig REQUIRED)
else()
    find_package(PkgConfig QUIET)
endif()
if(PkgConfig_FOUND)
    message(STATUS "found pkg-config")
    if(IN_CONDA_BUILD)
        # Set pkg-config search path to prioritize conda environment
        set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
        message(STATUS "PKG_CONFIG_PATH prioritizes Conda: $ENV{PKG_CONFIG_PATH}")
    endif()
endif()

# Doxygen (conditional)
if(CLIO_CORE_ENABLE_DOXYGEN)
    find_package(Perl REQUIRED)
    find_package(Doxygen REQUIRED)
    message(STATUS "found doxygen at ${DOXYGEN_EXECUTABLE}")
endif()

# YAML-CPP - find from system or install.sh installation
find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
    message(STATUS "found yaml-cpp at ${yaml-cpp_DIR}")
    resolve_yaml_cpp_target()
endif()

# When static deps are enabled, override yaml-cpp target to use the .a file.
# find_package creates an imported target whose per-config properties
# (IMPORTED_LOCATION_RELEASE, IMPORTED_SONAME_RELEASE) take precedence
# over the generic IMPORTED_LOCATION.  We must override every config that
# was registered so the linker always picks the static archive.
# On Windows, vcpkg manages static vs shared via the triplet; skip the override.
# On macOS, Homebrew ships only .dylib (no .a). delocate then bundles the
# shared lib into the wheel — the wheel still self-contains, just via a
# bundled dylib instead of a static archive.
if(CLIO_CORE_STATIC_DEPS AND NOT WIN32 AND NOT APPLE)
    find_library(YAML_CPP_STATIC_LIB NAMES libyaml-cpp.a REQUIRED)
    if(YAML_CPP_STATIC_LIB)
        message(STATUS "Static yaml-cpp: ${YAML_CPP_STATIC_LIB}")
        # Override every registered configuration
        get_property(_yaml_configs TARGET yaml-cpp::yaml-cpp
                     PROPERTY IMPORTED_CONFIGURATIONS)
        foreach(_cfg ${_yaml_configs})
            set_property(TARGET yaml-cpp::yaml-cpp
                         PROPERTY IMPORTED_LOCATION_${_cfg} "${YAML_CPP_STATIC_LIB}")
            set_property(TARGET yaml-cpp::yaml-cpp
                         PROPERTY IMPORTED_SONAME_${_cfg} "")
        endforeach()
        # Also set the generic fallback
        set_target_properties(yaml-cpp::yaml-cpp PROPERTIES
            IMPORTED_LOCATION "${YAML_CPP_STATIC_LIB}")
    endif()
endif()

# Cereal - find from system or install.sh installation
find_package(cereal REQUIRED)
if(cereal_FOUND)
    message(STATUS "found cereal at ${cereal_DIR}")
endif()

# MessagePack C library (pure C, no Boost dependency)
# Prefer static linkage so pip wheels don't need libmsgpack-c.so at runtime.
# Try msgpack-c first (v6+, post C/C++ split), then msgpack (v3.x, pre-split).
find_package(msgpack-c CONFIG QUIET)
if(NOT msgpack-c_FOUND)
  find_package(msgpack REQUIRED CONFIG)
  message(STATUS "found msgpack (legacy v3.x) at ${msgpack_DIR}")
  # Create alias targets matching the msgpack-c v6+ naming convention
  if(TARGET msgpackc-static AND NOT TARGET msgpack-c-static)
    add_library(msgpack-c-static ALIAS msgpackc-static)
  endif()
  if(TARGET msgpackc AND NOT TARGET msgpack-c)
    add_library(msgpack-c ALIAS msgpackc)
  endif()
else()
  message(STATUS "found msgpack-c at ${msgpack-c_DIR}")
endif()
if(TARGET msgpack-c-static)
  set(MSGPACK_TARGET msgpack-c-static)
  message(STATUS "  using static msgpack-c")
else()
  set(MSGPACK_TARGET msgpack-c)
  message(STATUS "  using shared msgpack-c (static not available)")
endif()

# MPICH (only find when explicitly enabled)
if(CLIO_CORE_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C CXX)
    set(MPI_LIBS MPI::MPI_CXX)
    message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")
endif()

# ROCm (conditional). Only require the HIP CMake package when HIP-AMD is
# in use; the HIP-NVCC path goes through CMake's CUDA language and the
# HIP CMake package isn't shipped by ROCm builds that target NVIDIA.
if(CLIO_CORE_ENABLE_ROCM)
    if(NOT DEFINED CLIO_ROCM_HIP_PLATFORM)
        if(DEFINED ENV{HIP_PLATFORM})
            set(CLIO_ROCM_HIP_PLATFORM "$ENV{HIP_PLATFORM}" CACHE STRING
                "HIP platform (amd|nvidia)")
        else()
            set(CLIO_ROCM_HIP_PLATFORM "amd" CACHE STRING
                "HIP platform (amd|nvidia)")
        endif()
    endif()
    if(NOT CLIO_ROCM_HIP_PLATFORM STREQUAL "nvidia")
        find_package(HIP REQUIRED)
    endif()
    message(STATUS "found ROCm/HIP support (HIP_PLATFORM=${CLIO_ROCM_HIP_PLATFORM})")
endif()

# OpenMP (conditional)
if(CLIO_CORE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS C CXX)
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
    message(STATUS "found omp.h at ${OpenMP_CXX_INCLUDE_DIRS}")
endif()

# Thallium (conditional)
if(CLIO_CORE_ENABLE_THALLIUM)
    find_package(thallium CONFIG REQUIRED)
    if(thallium_FOUND)
        message(STATUS "found thallium at ${thallium_DIR}")
    endif()
    set(SERIALIZATION_LIBS thallium ${SERIALIZATION_LIBS})
    set(CLIO_CORE_ENABLE_CEREAL ON)
endif()

# hiredis (conditional) — Redis comparison benchmark only.
if(CLIO_CORE_ENABLE_REDIS)
    find_package(hiredis CONFIG QUIET)
    if(hiredis_FOUND)
        set(CLIO_REDIS_LIBS hiredis::hiredis)
        message(STATUS "found hiredis (CMake config) at ${hiredis_DIR}")
    else()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(HIREDIS REQUIRED IMPORTED_TARGET hiredis)
        set(CLIO_REDIS_LIBS PkgConfig::HIREDIS)
        message(STATUS "found hiredis (pkg-config) v${HIREDIS_VERSION}")
    endif()
endif()

# Update cereal libraries if enabled
if(CLIO_CORE_ENABLE_CEREAL)
    set(SERIALIZATION_LIBS cereal::cereal ${SERIALIZATION_LIBS})
endif()

# Add msgpack-c to serialization libs (always enabled)
set(SERIALIZATION_LIBS ${MSGPACK_TARGET} ${SERIALIZATION_LIBS})

# Thread support
find_package(Threads REQUIRED)

# Compression libraries (conditional)
if(CLIO_CTE_ENABLE_COMPRESS)
    message(STATUS "CLIO_CTE_ENABLE_COMPRESS is ON")

    # Libraries with pkg-config support
    pkg_check_modules(libzstd REQUIRED libzstd)
    message(STATUS "found zstd.h at ${libzstd_INCLUDE_DIRS}")

    pkg_check_modules(liblz4 REQUIRED liblz4)
    message(STATUS "found lz4.h at ${liblz4_INCLUDE_DIRS}")

    pkg_check_modules(zlib REQUIRED zlib)
    message(STATUS "found zlib.h at ${zlib_INCLUDE_DIRS}")

    pkg_check_modules(liblzma REQUIRED liblzma)
    message(STATUS "found liblzma.h at ${liblzma_INCLUDE_DIRS}")

    pkg_check_modules(libbrotlicommon REQUIRED libbrotlicommon libbrotlidec libbrotlienc)
    message(STATUS "found libbrotli.h at ${libbrotlicommon_INCLUDE_DIRS}")

    pkg_check_modules(blosc2 REQUIRED blosc2)
    message(STATUS "found blosc2.h at ${blosc2_INCLUDE_DIRS}")

    # bzip2
    find_path(bzip2_INCLUDE_DIR bzlib.h REQUIRED)
    find_library(bzip2_LIBRARY bz2 REQUIRED)
    if(bzip2_INCLUDE_DIR AND bzip2_LIBRARY)
        set(bzip2_INCLUDE_DIRS ${bzip2_INCLUDE_DIR})
        set(bzip2_LIBRARIES bz2)
        get_filename_component(bzip2_LIBRARY_DIRS "${bzip2_LIBRARY}" DIRECTORY)
        message(STATUS "found bz2.h at ${bzip2_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "bzip2 not found. Install with: apt-get install libbz2-dev")
    endif()

    # lzo
    find_path(lzo2_INCLUDE_DIR lzo/lzo1x.h REQUIRED)
    find_library(lzo2_LIBRARY lzo2 REQUIRED)
    if(lzo2_INCLUDE_DIR AND lzo2_LIBRARY)
        set(lzo2_INCLUDE_DIRS ${lzo2_INCLUDE_DIR})
        set(lzo2_LIBRARIES lzo2)
        get_filename_component(lzo2_LIBRARY_DIRS "${lzo2_LIBRARY}" DIRECTORY)
        get_filename_component(lzo2_dir "${lzo2_INCLUDE_DIRS}" DIRECTORY)
        message(STATUS "found lzo2.h at ${lzo2_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "lzo not found. Install with: apt-get install liblzo2-dev")
    endif()

    # snappy
    find_path(snappy_INCLUDE_DIR snappy.h REQUIRED)
    find_library(snappy_LIBRARY snappy REQUIRED)
    if(snappy_INCLUDE_DIR AND snappy_LIBRARY)
        set(snappy_INCLUDE_DIRS ${snappy_INCLUDE_DIR})
        set(snappy_LIBRARIES snappy)
        get_filename_component(snappy_LIBRARY_DIRS "${snappy_LIBRARY}" DIRECTORY)
        message(STATUS "found snappy.h at ${snappy_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "snappy not found. Install with: apt-get install libsnappy-dev")
    endif()

    # LibPressio - try find_package first (CMake-based), fall back to pkg-config
    # Make it optional since libpressio may not be installed
    set(CLIO_CTP_ENABLE_LIBPRESSIO OFF)
    find_package(LibPressio QUIET)
    if(LibPressio_FOUND)
        message(STATUS "found libpressio.h via CMake at ${LibPressio_INCLUDE_DIRS}")
        if(TARGET LibPressio::libpressio)
            set(libpressio_LIBRARIES LibPressio::libpressio)
        elseif(TARGET libpressio)
            set(libpressio_LIBRARIES libpressio)
        else()
            # Fallback - use the library directly
            set(libpressio_LIBRARIES liblibpressio)
        endif()
        set(libpressio_INCLUDE_DIRS ${LibPressio_INCLUDE_DIRS})
        set(libpressio_LIBRARY_DIRS "")
        set(CLIO_CTP_ENABLE_LIBPRESSIO ON)
    else()
        pkg_check_modules(libpressio QUIET libpressio)
        if(libpressio_FOUND)
            message(STATUS "found libpressio.h at ${libpressio_INCLUDE_DIRS}")
            set(CLIO_CTP_ENABLE_LIBPRESSIO ON)
        else()
            message(STATUS "libpressio not found - LibPressio wrapper will be disabled")
            set(libpressio_LIBRARIES "")
            set(libpressio_INCLUDE_DIRS "")
            set(libpressio_LIBRARY_DIRS "")
        endif()
    endif()

    # nvcomp (NVIDIA GPU compression) - optional, requires CUDA.
    # The redistributable ships nvcomp-config.cmake, so find_package works and
    # exposes the imported target nvcomp::nvcomp (which pulls in CUDA::cudart).
    set(CLIO_CTP_ENABLE_NVCOMP OFF)
    if(CLIO_CORE_ENABLE_CUDA)
        find_package(nvcomp QUIET)
        if(nvcomp_FOUND)
            message(STATUS "found nvcomp ${nvcomp_VERSION} (GPU compression) - target nvcomp::nvcomp")
            set(CLIO_CTP_ENABLE_NVCOMP ON)
        else()
            message(STATUS "nvcomp not found - GPU compression (nvcomp) will be disabled")
        endif()
    else()
        message(STATUS "CLIO_CORE_ENABLE_CUDA is OFF - GPU compression (nvcomp) disabled")
    endif()

    # zfp-SYCL (Intel-XPU GPU compression) - optional. A SYCL-enabled libzfp
    # (built with icpx from the zfp SYCL port) installed to a dedicated prefix
    # (default /opt/zfp-sycl) so it does not shadow the stock CPU libzfp that
    # ADIOS2/LibPressio link against. The SYCL kernels live inside that libzfp;
    # the ctp::SyclZfp wrapper only calls zfp's C API, so no SYCL compiler is
    # needed to build CTP -- we just link this libzfp and its headers.
    # NOTE: this is DISTINCT from CLIO_CTP_ENABLE_SYCL (the SYCL-language switch,
    # controlled by CLIO_CORE_ENABLE_SYCL, which triggers <sycl/sycl.hpp> + -fsycl
    # in macros.h). zfp-sycl needs neither, so it uses its own flag/define.
    set(CLIO_CTP_ENABLE_ZFP_SYCL OFF)
    set(CLIO_ZFP_SYCL_ROOT "/opt/zfp-sycl" CACHE PATH
        "Install prefix of the SYCL-enabled libzfp (for zfp-sycl compressor)")
    find_library(ZFP_SYCL_LIBRARY NAMES zfp
        HINTS ${CLIO_ZFP_SYCL_ROOT}/lib ${CLIO_ZFP_SYCL_ROOT}/lib64
        NO_DEFAULT_PATH)
    find_path(ZFP_SYCL_INCLUDE_DIR NAMES zfp.h
        HINTS ${CLIO_ZFP_SYCL_ROOT}/include
        NO_DEFAULT_PATH)
    if(ZFP_SYCL_LIBRARY AND ZFP_SYCL_INCLUDE_DIR)
        message(STATUS "found SYCL libzfp at ${ZFP_SYCL_LIBRARY} (zfp-sycl GPU compression)")
        set(CLIO_CTP_ENABLE_ZFP_SYCL ON)
    else()
        message(STATUS "SYCL libzfp not found under ${CLIO_ZFP_SYCL_ROOT} - zfp-sycl compression disabled")
    endif()

    set(COMPRESS_LIBS
        bz2
        ${lzo2_LIBRARIES}
        ${libzstd_LIBRARIES}
        ${liblz4_LIBRARIES}
        ${zlib_LIBRARIES}
        ${liblzma_LIBRARIES}
        ${libbrotlicommon_LIBRARIES}
        ${snappy_LIBRARIES}
        ${blosc2_LIBRARIES}
    )
    set(COMPRESS_INCLUDES
        ${bzip2_INCLUDE_DIRS}
        ${lzo2_INCLUDE_DIRS} ${lzo2_dir}
        ${libzstd_INCLUDE_DIRS}
        ${liblz4_INCLUDE_DIRS}
        ${zlib_INCLUDE_DIRS}
        ${liblzma_INCLUDE_DIRS}
        ${libbrotlicommon_INCLUDE_DIRS}
        ${snappy_INCLUDE_DIRS}
        ${blosc2_INCLUDE_DIRS}
    )
    set(COMPRESS_LIB_DIRS
        ${bzip2_LIBRARY_DIRS}
        ${lzo2_LIBRARY_DIRS}
        ${libzstd_LIBRARY_DIRS}
        ${liblz4_LIBRARY_DIRS}
        ${zlib_LIBRARY_DIRS}
        ${liblzma_LIBRARY_DIRS}
        ${libbrotlicommon_LIBRARY_DIRS}
        ${snappy_LIBRARY_DIRS}
        ${blosc2_LIBRARY_DIRS}
    )
    # Only add libpressio if found
    if(CLIO_CTP_ENABLE_LIBPRESSIO)
        list(APPEND COMPRESS_LIBS ${libpressio_LIBRARIES})
        list(APPEND COMPRESS_INCLUDES ${libpressio_INCLUDE_DIRS})
        list(APPEND COMPRESS_LIB_DIRS ${libpressio_LIBRARY_DIRS})
    endif()

    # Only add nvcomp if found (imported target carries its own includes)
    if(CLIO_CTP_ENABLE_NVCOMP)
        list(APPEND COMPRESS_LIBS nvcomp::nvcomp)
    endif()

    # Only add the SYCL libzfp if found
    if(CLIO_CTP_ENABLE_ZFP_SYCL)
        list(APPEND COMPRESS_LIBS ${ZFP_SYCL_LIBRARY})
        list(APPEND COMPRESS_INCLUDES ${ZFP_SYCL_INCLUDE_DIR})
    endif()

    # Make CTP_ENABLE_LIBPRESSIO available to subdirectories (CTP)
    set(CLIO_CTP_ENABLE_LIBPRESSIO ${CLIO_CTP_ENABLE_LIBPRESSIO} CACHE BOOL "Enable LibPressio wrapper (auto-detected)" FORCE)
    # Make CTP_ENABLE_NVCOMP available to subdirectories (CTP)
    set(CLIO_CTP_ENABLE_NVCOMP ${CLIO_CTP_ENABLE_NVCOMP} CACHE BOOL "Enable nvcomp GPU compression (auto-detected)" FORCE)
    # Make CLIO_CTP_ENABLE_ZFP_SYCL available to subdirectories (CTP). Distinct
    # from CLIO_CTP_ENABLE_SYCL (the SYCL-language switch) -- do not conflate.
    set(CLIO_CTP_ENABLE_ZFP_SYCL ${CLIO_CTP_ENABLE_ZFP_SYCL} CACHE BOOL "Enable zfp-sycl GPU compression (auto-detected)" FORCE)
endif()

# Encryption libraries (conditional)
if(CLIO_CORE_ENABLE_ENCRYPT)
    pkg_check_modules(libcrypto REQUIRED libcrypto)
    message(STATUS "found libcrypto.h at ${libcrypto_INCLUDE_DIRS}")

    set(ENCRYPT_LIBS ${libcrypto_LIBRARIES})
    set(ENCRYPT_INCLUDES ${libcrypto_INCLUDE_DIRS})
    set(ENCRYPT_LIB_DIRS ${libcrypto_LIBRARY_DIRS})
endif()

# Note: ELF library detection moved to after CTP_ENABLE_ELF is set (line ~520)

# ZeroMQ (conditional) - try system first, fall back to submodule
if(CLIO_CORE_ENABLE_ZMQ)
    set(_ZMQ_FOUND FALSE)
    # Try pkg-config first (Linux/macOS)
    if(PkgConfig_FOUND)
        pkg_check_modules(ZeroMQ QUIET libzmq)
        if(ZeroMQ_FOUND)
            message(STATUS "found zmq.h at ${ZeroMQ_INCLUDE_DIRS}")
            if(CLIO_CORE_STATIC_DEPS AND NOT WIN32 AND NOT APPLE)
                # Find static .a archives directly — pkg-config _STATIC_LIBRARIES
                # still returns -l flags that the linker resolves to shared libs.
                # Skipped on Windows (vcpkg manages static/shared via triplet)
                # and macOS (Homebrew ships .dylib only; delocate bundles it).
                find_library(ZMQ_STATIC_LIB NAMES libzmq.a REQUIRED)
                set(ZMQ_LIBS ${ZMQ_STATIC_LIB})
                # Pull in zmq's transitive static deps (libsodium, pthreads, etc.)
                foreach(_dep ${ZeroMQ_STATIC_LIBRARIES})
                    if(NOT _dep STREQUAL "zmq")
                        find_library(_dep_path NAMES lib${_dep}.a ${_dep})
                        if(_dep_path)
                            list(APPEND ZMQ_LIBS ${_dep_path})
                        else()
                            list(APPEND ZMQ_LIBS ${_dep})
                        endif()
                        unset(_dep_path CACHE)
                    endif()
                endforeach()
                set(ZMQ_LIB_DIRS "")
                message(STATUS "Static zmq libs: ${ZMQ_LIBS}")
            else()
                set(ZMQ_LIBS ${ZeroMQ_LIBRARIES})
                set(ZMQ_LIB_DIRS ${ZeroMQ_LIBRARY_DIRS})
            endif()
            set(ZMQ_INCLUDES ${ZeroMQ_INCLUDE_DIRS})
            set(_ZMQ_FOUND TRUE)
        endif()
    endif()
    # Fall back to CMake config mode (vcpkg on Windows)
    if(NOT _ZMQ_FOUND)
        find_package(ZeroMQ CONFIG QUIET)
        if(ZeroMQ_FOUND)
            message(STATUS "found ZeroMQ via CMake config")
            set(ZMQ_LIBS libzmq)
            set(ZMQ_INCLUDES "")
            set(ZMQ_LIB_DIRS "")
            set(_ZMQ_FOUND TRUE)
        endif()
    endif()
    if(NOT _ZMQ_FOUND)
        message(FATAL_ERROR "ZeroMQ not found. Please install ZeroMQ or run install.sh to build from submodules.")
    endif()
endif()

# Libfabric (conditional)
if(CLIO_CORE_ENABLE_LIBFABRIC)
    pkg_check_modules(Libfabric REQUIRED libfabric)
    message(STATUS "found rdma/fabric.h at ${Libfabric_INCLUDE_DIRS}")
    set(LIBFABRIC_LIBS ${Libfabric_LIBRARIES})
    set(LIBFABRIC_INCLUDES ${Libfabric_INCLUDE_DIRS})
    set(LIBFABRIC_LIB_DIRS ${Libfabric_LIBRARY_DIRS})
endif()

# HDF5 (optional - for CAE HDF5 assimilator and CTE HDF5 VOL connector)
if(CLIO_CTE_ENABLE_HDF5_VOL)
    # Try CONFIG mode first (for HDF5 installed to non-standard cmake/ directory)
    find_package(HDF5 COMPONENTS C QUIET CONFIG)
    if(NOT HDF5_FOUND)
        # Fallback to module mode (for system HDF5 installations)
        find_package(HDF5 COMPONENTS C QUIET)
    endif()
    if(HDF5_FOUND)
        # Ensure HDF5_LIBRARIES is set for both CONFIG and MODULE mode
        if(NOT HDF5_LIBRARIES)
            # CONFIG mode creates targets but may not set HDF5_LIBRARIES.
            # HDF5's own hdf5-config.cmake (used by vcpkg and HDF Group
            # binary installs) only exports namespaced hdf5::hdf5-{shared,
            # static} targets, so check those too (issue #518: the VOL
            # connector DLL linked without hdf5.lib on Windows/vcpkg).
            # Prefer shared libraries over static for better compatibility.
            if(TARGET HDF5::HDF5)
                set(HDF5_LIBRARIES HDF5::HDF5)
            elseif(TARGET hdf5::hdf5-shared)
                set(HDF5_LIBRARIES hdf5::hdf5-shared)
            elseif(TARGET hdf5-shared)
                set(HDF5_LIBRARIES hdf5-shared)
            elseif(TARGET hdf5::hdf5-static)
                set(HDF5_LIBRARIES hdf5::hdf5-static)
            elseif(TARGET hdf5-static)
                set(HDF5_LIBRARIES hdf5-static)
            endif()
        endif()
        if(NOT HDF5_LIBRARIES)
            message(FATAL_ERROR "HDF5 found but no linkable HDF5 target/"
                                "library was identified (HDF5_DIR=${HDF5_DIR})")
        endif()

        # hdf5-config.cmake sets HDF5_INCLUDE_DIR (singular); consumers in
        # this project use HDF5_INCLUDE_DIRS.
        if(NOT HDF5_INCLUDE_DIRS AND HDF5_INCLUDE_DIR)
            set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
        endif()

        message(STATUS "found HDF5 ${HDF5_VERSION} at '${HDF5_INCLUDE_DIRS}' "
                       "(libraries: ${HDF5_LIBRARIES})")

        # libhdf5.so (system package) depends on libcurl internally;
        # the linker needs it listed explicitly due to --as-needed.
        find_package(CURL QUIET)
        if(CURL_FOUND)
            list(APPEND HDF5_LIBRARIES CURL::libcurl)
        endif()
    else()
        message(FATAL_ERROR "HDF5 not found. Please install HDF5 or run install.sh to build from submodules.")
    endif()
else()
    message(STATUS "HDF5 support disabled")
endif()

# Poco and nlohmann_json (optional - for CAE Globus support)
find_package(Poco COMPONENTS Net NetSSL Crypto JSON QUIET)
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git
      GIT_TAG        v3.11.3
      GIT_SHALLOW    ON
  )
  FetchContent_MakeAvailable(nlohmann_json)
  set(nlohmann_json_FOUND TRUE)
endif()

if (CLIO_CTE_ENABLE_ADIOS2_ADAPTER OR CLIO_CORE_ENABLE_GRAY_SCOTT)
    # Find ADIOS2
    find_package(ADIOS2 REQUIRED)
    if(ADIOS2_FOUND)
        message(STATUS "found ADIOS2 at ${ADIOS2_DIR}")
    else()
        message(FATAL_ERROR "CLIO_CORE_ENABLE_GRAY_SCOTT is ON but ADIOS2 not found. Please install ADIOS2 or disable the option.")
    endif()
endif()

# ADIOS2 (optional - for Gray-Scott example)
if(CLIO_CORE_ENABLE_GRAY_SCOTT)
    # Gray-Scott requires MPI - find it if not already found
    if(NOT MPI_FOUND)
        find_package(MPI REQUIRED COMPONENTS C CXX)
        if(MPI_FOUND)
            message(STATUS "found MPI for Gray-Scott at ${MPI_CXX_INCLUDE_DIRS}")
        endif()
    endif()
else()
    message(STATUS "Gray-Scott example disabled")
endif()

#------------------------------------------------------------------------------
# Python Bindings Configuration
#------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_PYTHON)
  message(STATUS "Python bindings enabled - configuring nanobind")

  # Find Python executable using helper from ClioCoreCommon
  find_first_path_python()

  # Find Python for nanobind
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  # Fetch nanobind (no submodule needed)
  include(FetchContent)
  FetchContent_Declare(nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind.git
      GIT_TAG        v2.9.2
      GIT_SHALLOW    ON
  )
  FetchContent_MakeAvailable(nanobind)

  # Determine the real site-packages directory for the detected Python.
  # Python_SITEARCH is an absolute path (works for venv, conda, and system Python).
  # Fall back to a relative path under CMAKE_INSTALL_PREFIX if unavailable.
  if(Python_SITEARCH)
      set(PYTHON_SITE_PACKAGES "${Python_SITEARCH}" CACHE PATH
          "Python site-packages directory (absolute)")
  else()
      set(PYTHON_SITE_PACKAGES
          "${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages"
          CACHE PATH "Python site-packages directory (fallback)")
  endif()

  message(STATUS "Python bindings enabled:")
  message(STATUS "  Install prefix:     ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "  Site-packages:      ${PYTHON_SITE_PACKAGES}")
else()
  message(STATUS "Python bindings disabled")
endif()

#------------------------------------------------------------------------------
# Global Settings
#------------------------------------------------------------------------------
# C++ Standard - Applied to all subdirectories
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C++20 Coroutines support
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fcoroutines)
  message(STATUS "GCC detected - enabling coroutines with -fcoroutines")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  # NVHPC 25.5 crashes (ICE in llc) when compiling GCC 14 libstdc++ coroutines.
  # CLIO Runtime uses ucontext_t fibers via CHI_CO_AWAIT/CHI_CO_RETURN macros instead.
  # No __cpp_impl_coroutine needed - we do not include <coroutine> for NVHPC.
  message(STATUS "NVHPC detected - using ucontext_t fiber fallback for C++20 coroutines")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # Clang 16+ enables C++20 coroutines natively (no flags needed for CXX).
  # NVCC host compilation uses GCC which needs -fcoroutines for the builtins.
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-fcoroutines>)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "16")
    add_compile_definitions(__cpp_impl_coroutine=201902L)
    message(STATUS "Clang ${CMAKE_CXX_COMPILER_VERSION} detected - enabling coroutines with -D__cpp_impl_coroutine (libstdc++ compat)")
  else()
    message(STATUS "Clang ${CMAKE_CXX_COMPILER_VERSION} detected - C++20 coroutines enabled natively")
  endif()
endif()

# Export compile commands for IDE integration - Applied to all subdirectories
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Position Independent Code - Applied to all subdirectories
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ccache — accelerate rebuilds by caching compiled objects.
# NOT used for MSVC-like compilers (cl / clang-cl / icx-cl): as a compiler
# launcher under Ninja, ccache mishandles the cl-style /Fo<obj> output flag and
# the object is written as <base>.o into the build root instead of the expected
# path, which breaks WINDOWS_EXPORT_ALL_SYMBOLS / the DLL link. (The Visual
# Studio generator never applied the launcher, which is why MSVC CI was fine.)
# See issue #555.
option(CLIO_CORE_ENABLE_CCACHE "Use ccache as compiler launcher when available" ON)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND CLIO_CORE_ENABLE_CCACHE AND NOT CLIO_CORE_MSVC_LIKE)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
elseif(CCACHE_PROGRAM)
    message(STATUS "ccache found but not used (CLIO_CORE_ENABLE_CCACHE=${CLIO_CORE_ENABLE_CCACHE}, CLIO_CORE_MSVC_LIKE=${CLIO_CORE_MSVC_LIKE})")
else()
    message(STATUS "ccache not found — install ccache for faster rebuilds")
endif()

# RPATH configuration - Uses $ORIGIN-relative paths for portability
# This ensures binaries and libraries can find their dependencies regardless
# of where the installation is located (works for system installs, pip wheels,
# conda packages, and relocatable builds).
if(CLIO_CORE_ENABLE_RPATH)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    if(APPLE)
        set(CMAKE_MACOSX_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../lib")
        set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/bin")
    elseif(UNIX)
        set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
        set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/bin")
    endif()

    message(STATUS "RPATH: Using relative paths ($ORIGIN/../lib)")
    message(STATUS "  INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")
    message(STATUS "  BUILD_RPATH:   ${CMAKE_BUILD_RPATH}")
else()
    message(STATUS "RPATH DISABLED - libraries may require LD_LIBRARY_PATH")
endif()

#------------------------------------------------------------------------------
# Compiler Flags
#------------------------------------------------------------------------------
# Use CLIO_CORE_MSVC_LIKE (not just MSVC) so clang-cl / icx.exe on Windows take
# the MSVC-style flag path; GNU flags break icx's /Fo handling (see issue #555).
if(NOT CLIO_CORE_MSVC_LIKE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
        # Add compiler flags following Google C++ style guide
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
        # Disable warnings that are false positives or from external dependencies
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-variable -Wno-reorder")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic -Wno-sign-compare -Wno-missing-field-initializers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-unused-function -Wno-cast-function-type")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-self-move -Wno-format")
        # Suppress false positives in template-heavy shared memory code (Release -O3)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds")
        # -Wno-stringop-overflow/-overread are GCC-only; check before adding
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-Wno-stringop-overflow HAS_WNO_STRINGOP_OVERFLOW)
        if(HAS_WNO_STRINGOP_OVERFLOW)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
        endif()
        check_cxx_compiler_flag(-Wno-stringop-overread HAS_WNO_STRINGOP_OVERREAD)
        if(HAS_WNO_STRINGOP_OVERREAD)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overread")
        endif()
        # Suppress unused-result warnings (freopen/system in test code)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
    endif()

    # Debug configuration
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")

    # Release configuration
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

    # CUDA: allow constexpr __host__ functions in __host__ __device__ contexts
    if(CLIO_CORE_ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -diag-suppress=20011,20012,20013")
    endif()

    # AddressSanitizer + LeakSanitizer configuration
    if(CLIO_CORE_ENABLE_ASAN)
        message(STATUS "AddressSanitizer + LeakSanitizer (ASan/LSan) enabled for iowarp-core")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()

    # UndefinedBehaviorSanitizer configuration
    if(CLIO_CORE_ENABLE_UBSAN)
        message(STATUS "UndefinedBehaviorSanitizer (UBSan) enabled for iowarp-core")
        add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
        add_link_options(-fsanitize=undefined)
    endif()

    # MemorySanitizer configuration (incompatible with ASan; requires instrumented dependencies)
    if(CLIO_CORE_ENABLE_MSAN)
        if(CLIO_CORE_ENABLE_ASAN)
            message(FATAL_ERROR "CLIO_CORE_ENABLE_MSAN and CLIO_CORE_ENABLE_ASAN are mutually exclusive")
        endif()
        message(STATUS "MemorySanitizer (MSan) enabled for iowarp-core")
        message(WARNING "MSan requires all dependencies to be built with -fsanitize=memory; false-positives are expected from uninstrumented libs")
        add_compile_options(-fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=memory
                            -fsanitize-memory-track-origins=2
                            -fsanitize-ignorelist=${CMAKE_SOURCE_DIR}/CI/msan_ignorelist.txt)
        add_link_options(-fsanitize=memory)
    endif()

    # Code Coverage configuration
    # Exclude CUDA: NVCC passes --coverage through to device code, generating
    # __gcov_* symbols in device stubs that the host linker can't resolve.
    if(CLIO_CORE_ENABLE_COVERAGE)
        message(STATUS "Code coverage instrumentation enabled for iowarp-core")
        add_compile_options($<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:--coverage>
                            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fprofile-arcs>
                            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-ftest-coverage>)
        add_link_options(--coverage)
    endif()
else()
    # MSVC compiler flags
    # /FS enables synchronized PDB writes to avoid C1041 errors with parallel
    # builds. Scope it per-language: passed directly for C/CXX, but routed
    # through -Xcompiler for CUDA — nvcc treats a bare "/FS" as a second input
    # file ("nvcc fatal : A single input file is required ...").
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/FS>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FS>)
    # CUDA 13.x CCCL headers (cuda/std/*) require MSVC's conformant
    # preprocessor and #error out on the traditional one. Enable it for the
    # CUDA host compile only, so the existing CXX build is unaffected.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    # Prevent windows.h from defining min/max macros that conflict with
    # std::numeric_limits<>::max() and similar standard library usage.
    add_compile_definitions(NOMINMAX)
    # In the CUDA build, nvcc force-includes cuda_runtime.h (which pulls in
    # windows.h, and thus the legacy winsock.h) before the socket sources can
    # include winsock2.h, causing 'sockaddr' redefinition (C2011). Keep the
    # legacy winsock out of windows.h for the CUDA host compile.
    add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:WIN32_LEAN_AND_MEAN>)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
endif()

# Global include path for simple_test.h test framework
include_directories(${CMAKE_SOURCE_DIR}/context-runtime/test)

# When building for Python wheels on x86_64, restrict to baseline x86_64 ISA
# so the wheel runs on any x86_64 CPU. AlmaLinux 9 (manylinux_2_34) defaults
# to x86_64_v2 (SSE4.2/POPCNT) which auditwheel rejects for x86_64 wheels.
if(DEFINED SKBUILD AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=x86-64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64")
    message(STATUS "Wheel build: restricting to baseline x86_64 ISA (-march=x86-64)")
endif()

#------------------------------------------------------------------------------
# Binary Output Directories
#------------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# For multi-config generators (Visual Studio, Xcode), keep runtime artifacts
# (.exe/.dll) in a flat bin/ for every configuration: tests and the module
# loader resolve binaries at ${CMAKE_BINARY_DIR}/bin regardless of config.
# Archives (static libs and DLL import libs) must NOT be flattened: they are
# link-time-only inputs, and sharing one bin/<name>.lib across configurations
# lets a Release build clobber the Debug archive (and vice versa), which then
# fails the next Debug link with LNK2038 runtime-library mismatches (#528).
# Leaving the per-config archive variables unset makes multi-config
# generators append a per-configuration subdirectory (bin/Debug, bin/Release)
# to CMAKE_ARCHIVE_OUTPUT_DIRECTORY automatically.
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_TESTS)
    enable_testing()
    include(CTest)

    # -----------------------------------------------------------------------
    # Test cleanup fixture (pre + post)
    # Kills any orphaned CLIO Runtime server processes and removes leftover shared
    # state (IPC socket files, shared memory, memfd) plus build-dir test
    # artifacts.  The pre-test variant runs once before any other CTest test.
    # The post-test variant is registered as FIXTURES_CLEANUP and is invoked
    # after every test that requires the fixture, even if those tests fail
    # or crash, ensuring temp files do not persist between runs.
    # -----------------------------------------------------------------------
    set(_chimaera_cleanup_cmd
        "fuser -k 5555/tcp 5558/tcp 9413/tcp 9414/tcp 9416/tcp 2>/dev/null; \
         rm -f /tmp/chimaera_*.ipc 2>/dev/null; \
         rm -f /tmp/chimaera_server_timing.log 2>/dev/null; \
         rm -rf /tmp/chimaera_memfd 2>/dev/null; \
         rm -rf /tmp/chimaera_${USER} 2>/dev/null; \
         rm -f /dev/shm/sm_segment.* 2>/dev/null; \
         rm -f /dev/shm/test_buddy_bench_* 2>/dev/null; \
         rm -f \"${CMAKE_BINARY_DIR}\"/cte_*.dat* \"${CMAKE_BINARY_DIR}\"/cte_*.yaml \"${CMAKE_BINARY_DIR}\"/*stress*.bin* \"${CMAKE_BINARY_DIR}\"/*reorganize*.bin* \"${CMAKE_BINARY_DIR}\"/*reorganize*.yaml \"${CMAKE_BINARY_DIR}\"/temp_unregister_test.dat 2>/dev/null; \
         rm -rf \"${CMAKE_BINARY_DIR}\"/cee_test_data 2>/dev/null; \
         exit 0")

    if(UNIX)
      add_test(
          NAME cr_pre_test_cleanup
          COMMAND bash -c "${_chimaera_cleanup_cmd}"
      )
      set_tests_properties(cr_pre_test_cleanup PROPERTIES
          TIMEOUT 30
          FIXTURES_SETUP chimaera_test_cleanup_fixture
      )

      add_test(
          NAME cr_post_test_cleanup
          COMMAND bash -c "${_chimaera_cleanup_cmd}"
      )
      set_tests_properties(cr_post_test_cleanup PROPERTIES
          TIMEOUT 30
          FIXTURES_CLEANUP chimaera_test_cleanup_fixture
      )
    elseif(WIN32)
      # Windows equivalent: kill any lingering test binaries that might still
      # hold our TCP ports (9413/9414/9416). On Windows TIME_WAIT can hold a
      # listener-bound port for ~2 minutes after the process dies, so cleaning
      # up between back-to-back tests prevents "Address in use" failures.
      # We use PowerShell because cmd.exe lacks a one-liner equivalent.
      #
      # The kill is best-effort (-ErrorAction SilentlyContinue throughout) so a
      # stale rule never fails the whole suite. Build artifacts under
      # ${CMAKE_BINARY_DIR}/bin are scrubbed the same way the POSIX path scrubs
      # ${CMAKE_BINARY_DIR}/cte_*.dat etc.
      set(_chimaera_cleanup_cmd_win
          "$ErrorActionPreference='Continue'; \
           Get-Process | Where-Object { $_.Path -and $_.Path -like '${CMAKE_BINARY_DIR}/bin/*' } | Stop-Process -Force -ErrorAction SilentlyContinue; \
           Get-ChildItem '${CMAKE_BINARY_DIR}' -Filter 'cte_*.dat*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; \
           Get-ChildItem '${CMAKE_BINARY_DIR}' -Filter 'cte_*.yaml' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; \
           Get-ChildItem '${CMAKE_BINARY_DIR}' -Filter '*stress*.bin*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; \
           Get-ChildItem '${CMAKE_BINARY_DIR}' -Filter '*reorganize*.bin*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; \
           if (Test-Path '${CMAKE_BINARY_DIR}/cee_test_data') { Remove-Item -Recurse -Force '${CMAKE_BINARY_DIR}/cee_test_data' -ErrorAction SilentlyContinue }; \
           exit 0")

      add_test(
          NAME cr_pre_test_cleanup
          COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command "${_chimaera_cleanup_cmd_win}"
      )
      set_tests_properties(cr_pre_test_cleanup PROPERTIES
          TIMEOUT 30
          FIXTURES_SETUP chimaera_test_cleanup_fixture
      )

      add_test(
          NAME cr_post_test_cleanup
          COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command "${_chimaera_cleanup_cmd_win}"
      )
      set_tests_properties(cr_post_test_cleanup PROPERTIES
          TIMEOUT 30
          FIXTURES_CLEANUP chimaera_test_cleanup_fixture
      )
    endif()
endif()

#------------------------------------------------------------------------------
# Component Subdirectories
#------------------------------------------------------------------------------

# Catch2 testing framework.
#
# Prefer the in-tree external/Catch2 source over a system/conda
# find_package. On this build host the conda env ships
# libCatch2.so.3.14.0 built with a newer gcc (CXXABI_1.3.15); linking
# the unit-test executables against it with host gcc 11.4 fails at
# link time with `undefined reference to __cxa_call_terminate@
# CXXABI_1.3.15`. Building Catch2 from the bundled source with the
# same host compiler keeps the CXXABI consistent. Falls back to
# find_package only when the bundled source is absent.
if(CLIO_CORE_ENABLE_TESTS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/Catch2/CMakeLists.txt")
    message(STATUS "Using bundled Catch2 (external/Catch2) built with host compiler")
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
    add_subdirectory(external/Catch2 EXCLUDE_FROM_ALL)
  else()
    message(STATUS "Bundled Catch2 absent; finding system Catch2")
    find_package(Catch2 3 REQUIRED)
    if(Catch2_FOUND)
      message(STATUS "found Catch2 at ${Catch2_DIR}")
    endif()
  endif()
endif()


# yaml-cpp is already found by ClioCoreCommon (line 202)
if(NOT TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
  message(FATAL_ERROR "yaml-cpp was not found. Please run install.sh to install dependencies.")
endif()
message(STATUS "yaml-cpp already found by ClioCoreCommon")

# Context Transport Primitives (formerly context-transport-primitives) - Required by runtime
message(STATUS "Building context-transport-primitives component")

# Detect ELF library if enabled
if(CLIO_CORE_ENABLE_ELF)
    pkg_check_modules(libelf REQUIRED libelf)
    message(STATUS "found libelf.h at ${libelf_INCLUDE_DIRS}")

    set(ELF_LIBS ${libelf_LIBRARIES})
    set(ELF_INCLUDES ${libelf_INCLUDE_DIRS})
    set(ELF_LIB_DIRS ${libelf_LIBRARY_DIRS})
endif()

# BaM (Big Accelerator Memory) — GPU page cache with optional NVMe
option(CLIO_CORE_ENABLE_BAM "Enable BaM GPU page cache module" OFF)
option(CLIO_CORE_ENABLE_BAM_NVME "Enable NVMe backend for BaM (requires gpu-nvme-bridge)" OFF)
if(CLIO_CORE_ENABLE_BAM)
    message(STATUS "Building BaM GPU page cache module")
    set(BAM_ENABLE_NVME ${CLIO_CORE_ENABLE_BAM_NVME} CACHE BOOL "NVMe for BaM" FORCE)
    set(BAM_ENABLE_TESTS ${CLIO_CORE_ENABLE_TESTS} CACHE BOOL "BaM tests" FORCE)
    add_subdirectory(external/bam)
else()
    message(STATUS "BaM module disabled")
endif()

set(_WRP_ENABLED_COMPONENT_DIRS "")

add_subdirectory(context-transport-primitives)
list(APPEND _WRP_ENABLED_COMPONENT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/context-transport-primitives")
# Set CTP_ROOT for cross-component access
set(CLIO_CTP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/context-transport-primitives" CACHE PATH "Root directory for context-transport-primitives")

# Runtime component
if(CLIO_CORE_ENABLE_RUNTIME)
    message(STATUS "Building runtime component")
    add_subdirectory(context-runtime)
    list(APPEND _WRP_ENABLED_COMPONENT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/context-runtime")
    # Set CLIO_RUN_ROOT for cross-component access (e.g., CTE needs bdev includes)
    set(CLIO_RUN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/context-runtime" CACHE PATH "Root directory for runtime component")
else()
    message(STATUS "Runtime component disabled")
endif()

# Context Transfer Engine
if(CLIO_CORE_ENABLE_CTE)
    message(STATUS "Building context-transfer-engine component")
    add_subdirectory(context-transfer-engine)
    list(APPEND _WRP_ENABLED_COMPONENT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/context-transfer-engine")
else()
    message(STATUS "Context-transfer-engine component disabled")
endif()

# Context Assimilation Engine
if(CLIO_CORE_ENABLE_CAE)
    message(STATUS "Building context-assimilation-engine component")
    add_subdirectory(context-assimilation-engine)
    list(APPEND _WRP_ENABLED_COMPONENT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/context-assimilation-engine")
else()
    message(STATUS "Context-assimilation-engine component disabled")
endif()

# Context Exploration Engine
if(CLIO_CORE_ENABLE_CEE)
    message(STATUS "Building context-exploration-engine component")
    add_subdirectory(context-exploration-engine)
    list(APPEND _WRP_ENABLED_COMPONENT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/context-exploration-engine")
else()
    message(STATUS "Context-exploration-engine component disabled")
endif()

# Context Visualizer (pure Python web dashboard)
if(CLIO_CORE_ENABLE_PYTHON)
    message(STATUS "Building context-visualizer component (Python)")
    add_subdirectory(context-visualizer)
else()
    message(STATUS "Context-visualizer component disabled (requires CLIO_CORE_ENABLE_PYTHON)")
endif()

# Gray-Scott Example Application
if(CLIO_CORE_ENABLE_GRAY_SCOTT)
    message(STATUS "Building Gray-Scott ADIOS2 example")
    add_subdirectory(external/iowarp-gray-scott)
else()
    message(STATUS "Gray-Scott example disabled")
endif()

#------------------------------------------------------------------------------
# Serialize CLIO Runtime runtime tests to prevent port 9413 conflicts
#------------------------------------------------------------------------------
# All tests that start a CLIO Runtime server compete for the same default port
# (9413). Under parallel ctest execution (e.g. ctest -j4) this causes
# "Address already in use" failures. RESOURCE_LOCK ensures only one such
# test runs at a time. CTP tests (context-transport-primitives) use different
# ports or no server, so they are intentionally excluded here.
#
# get_directory_property(TESTS) only returns tests from the exact directory
# scope, not from nested subdirectories. We therefore recurse through the
# full SUBDIRECTORIES tree of each component to apply the lock.
if(CLIO_CORE_ENABLE_TESTS)
    # Build the LSAN_OPTIONS value when ASan or combined sanitize is enabled,
    # so that ctest (run standalone or via run_sanitizers.sh) always finds the
    # suppressions file and does not falsely fail on known architectural leaks.
    set(_lsan_opts "")
    if(CLIO_CORE_ENABLE_ASAN AND NOT CLIO_CORE_ENABLE_MSAN)
        set(_lsan_opts
            "detect_leaks=1:halt_on_error=0:print_stacktrace=1:symbolize=1:suppressions=${CMAKE_SOURCE_DIR}/CI/lsan_suppressions.txt")
    endif()

    function(_chimaera_lock_tests_recursive dir lsan_opts)
        get_directory_property(_tests DIRECTORY "${dir}" TESTS)
        if(_tests)
            set_tests_properties(${_tests}
                DIRECTORY "${dir}"
                PROPERTIES RESOURCE_LOCK "chimaera_runtime_port")
            if(lsan_opts)
                set_tests_properties(${_tests}
                    DIRECTORY "${dir}"
                    PROPERTIES ENVIRONMENT_MODIFICATION
                        "LSAN_OPTIONS=set:${lsan_opts}")
            endif()
        endif()
        get_directory_property(_subdirs DIRECTORY "${dir}" SUBDIRECTORIES)
        foreach(_sub ${_subdirs})
            _chimaera_lock_tests_recursive("${_sub}" "${lsan_opts}")
        endforeach()
    endfunction()

    foreach(_component_dir_pair
            "context-runtime:CLIO_CORE_ENABLE_RUNTIME"
            "context-transfer-engine:CLIO_CORE_ENABLE_CTE"
            "context-assimilation-engine:CLIO_CORE_ENABLE_CAE"
            "context-exploration-engine:CLIO_CORE_ENABLE_CEE")
        string(REPLACE ":" ";" _pair "${_component_dir_pair}")
        list(GET _pair 0 _component_dir)
        list(GET _pair 1 _component_flag)
        if(${_component_flag} AND IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_component_dir}")
            _chimaera_lock_tests_recursive(
                "${CMAKE_CURRENT_SOURCE_DIR}/${_component_dir}"
                "${_lsan_opts}")
        endif()
    endforeach()

    # -----------------------------------------------------------------------
    # Apply the cleanup fixture to every component test so that the
    # cr_post_test_cleanup hook runs after the whole suite, even when
    # individual tests fail. Excludes the cleanup fixtures themselves.
    # -----------------------------------------------------------------------
    function(_chimaera_require_cleanup_fixture_recursive dir)
        get_directory_property(_tests DIRECTORY "${dir}" TESTS)
        foreach(_t ${_tests})
            if(NOT _t STREQUAL "cr_pre_test_cleanup" AND
               NOT _t STREQUAL "cr_post_test_cleanup")
                set_property(TEST ${_t} DIRECTORY "${dir}" APPEND PROPERTY
                    FIXTURES_REQUIRED chimaera_test_cleanup_fixture)
            endif()
        endforeach()
        get_directory_property(_subdirs DIRECTORY "${dir}" SUBDIRECTORIES)
        foreach(_sub ${_subdirs})
            _chimaera_require_cleanup_fixture_recursive("${_sub}")
        endforeach()
    endfunction()

    foreach(_component_dir_pair
            "context-transport-primitives:CLIO_CORE_ENABLE_TESTS"
            "context-runtime:CLIO_CORE_ENABLE_RUNTIME"
            "context-transfer-engine:CLIO_CORE_ENABLE_CTE"
            "context-assimilation-engine:CLIO_CORE_ENABLE_CAE"
            "context-exploration-engine:CLIO_CORE_ENABLE_CEE")
        string(REPLACE ":" ";" _pair "${_component_dir_pair}")
        list(GET _pair 0 _component_dir)
        list(GET _pair 1 _component_flag)
        if(${_component_flag} AND IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_component_dir}")
            _chimaera_require_cleanup_fixture_recursive(
                "${CMAKE_CURRENT_SOURCE_DIR}/${_component_dir}")
        endif()
    endforeach()
endif()

# IOWarp KV Cache + Weight Offload libraries (clio_llm_kvcache, clio_llm_weights).
# These can be built independently of llama.cpp so they can be tested without
# the full llama.cpp integration (which causes a CUDA platform conflict when
# llama.cpp's project("llama.cpp" C CXX) resets the CUDA compiler state).
if(CLIO_CORE_ENABLE_LLAMA)
    if(NOT CLIO_CORE_ENABLE_CTE)
        message(FATAL_ERROR "CLIO_CORE_ENABLE_LLAMA requires CLIO_CORE_ENABLE_CTE=ON")
    endif()
    message(STATUS "Building llm-hooks KV cache + weight offload libraries")
    add_subdirectory(context-transfer-engine/llm-hooks)

    # llama.cpp full integration: add the llama.cpp subdirectory so that the
    # server patches and ggml backend are available for linking.
    # NOTE: external/llama.cpp calls project("llama.cpp" C CXX) which resets
    # CMake's CUDA platform variables.  Add it LAST so it cannot corrupt the
    # IOWarp CUDA targets already defined above.
    if(CLIO_CORE_ENABLE_LLAMA_SERVER)
        # GGML_CUDA controls llama.cpp's own GPU inference backend independently
        # from CLIO_CORE_ENABLE_CUDA (IOWarp's own CUDA targets).
        # Pass -DGGML_CUDA=ON on the command line to enable GPU inference.
        add_subdirectory(external/llama.cpp)
        message(STATUS "llama.cpp server integration enabled")

        # llama.cpp's project("llama.cpp" C CXX) call may partially reset CMake's
        # CUDA platform variables (_CMAKE_CUDA_WHOLE_FLAG etc.).  Re-enabling CUDA
        # at the top level forces CMake to re-run CMakeCUDAInformation.cmake and
        # restore all required device-link variables before generating build rules.
        if(CLIO_CORE_ENABLE_CUDA)
            enable_language(CUDA)
        endif()

        # Patch llama.cpp targets with IOWarp KV-cache definitions NOW — after
        # add_subdirectory(external/llama.cpp) so that server-context and llama
        # targets actually exist.  The if(TARGET ...) guards in llm-hooks/
        # CMakeLists.txt run before this point (ordering issue), so we do the
        # patching here instead.
        if(TARGET server-context)
            target_compile_definitions(server-context PRIVATE IOWARP_LLM_KVCACHE)
            target_link_libraries(server-context PRIVATE clio_llm_kvcache)
            message(STATUS "llm-hooks: patched server-context with IOWARP_LLM_KVCACHE")
        endif()
        if(TARGET llama)
            target_compile_definitions(llama PRIVATE IOWARP_LLM_KVCACHE)
            target_link_libraries(llama PRIVATE clio_llm_kvcache)
            message(STATUS "llm-hooks: patched llama with IOWARP_LLM_KVCACHE")
        endif()

        # IOWarp weight offloading via GpuVmm — requires CUDA and clio_llm_weights.
        # clio_llm_weights is only built when CLIO_CORE_ENABLE_CUDA=ON (it depends
        # on clio_cte_uvm which needs CUDA driver API).
        if(CLIO_CORE_ENABLE_CUDA AND TARGET clio_llm_weights)
            if(TARGET server-context)
                target_compile_definitions(server-context PRIVATE IOWARP_LLM_WEIGHTS)
                target_link_libraries(server-context PRIVATE clio_llm_weights)
                message(STATUS "llm-hooks: patched server-context with IOWARP_LLM_WEIGHTS")
            endif()
            if(TARGET llama)
                target_compile_definitions(llama PRIVATE IOWARP_LLM_WEIGHTS)
                target_link_libraries(llama PRIVATE clio_llm_weights)
                message(STATUS "llm-hooks: patched llama with IOWARP_LLM_WEIGHTS")
            endif()
        endif()
    else()
        message(STATUS "llama.cpp server integration skipped (set CLIO_CORE_ENABLE_LLAMA_SERVER=ON)")
    endif()
else()
    message(STATUS "llm-hooks disabled (set CLIO_CORE_ENABLE_LLAMA=ON to enable)")
endif()

#------------------------------------------------------------------------------
# Unified IOWarp Core Package Configuration
#------------------------------------------------------------------------------
# Generate and install the unified IowarpCore package configuration

include(CMakePackageConfigHelpers)

# Configure the package config file from template
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/ClioCoreConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/iowarp-coreConfig.cmake"
    INSTALL_DESTINATION lib/cmake/iowarp-core
)

# Generate the version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/iowarp-coreConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install the package config files. Both ClioCoreCommon (canonical) and
# the IowarpCoreCommon forwarder are installed so external CMake code
# that does `include(IowarpCoreCommon)` after find_package(iowarp-core)
# keeps working unchanged.
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/iowarp-coreConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/iowarp-coreConfigVersion.cmake"
        "${CMAKE_SOURCE_DIR}/cmake/ClioCoreCommon.cmake"
        "${CMAKE_SOURCE_DIR}/cmake/IowarpCoreCommon.cmake"
    DESTINATION
        lib/cmake/iowarp-core
)

# Install a `clio-core` alias package next to `iowarp-core` so downstream
# code can pick the canonical clio_* name:
#   find_package(clio-core CONFIG REQUIRED)   # canonical
#   find_package(iowarp-core CONFIG REQUIRED) # legacy (still works)
# clio-coreConfig.cmake is a thin shim that includes iowarp-coreConfig.cmake
# next door, so both names resolve to the same target set with zero
# duplication.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/clio-coreConfig.cmake"
"# Auto-generated shim by /CMakeLists.txt.  clio-core is the canonical
# name for the package historically called iowarp-core; this file simply
# forwards to iowarp-coreConfig.cmake installed in the sibling
# lib/cmake/iowarp-core/ directory.
get_filename_component(_clio_core_self_dir
    \"\${CMAKE_CURRENT_LIST_DIR}\" REALPATH)
get_filename_component(_clio_core_iowarp_dir
    \"\${_clio_core_self_dir}/../iowarp-core\" REALPATH)
if(NOT EXISTS \"\${_clio_core_iowarp_dir}/iowarp-coreConfig.cmake\")
    set(clio-core_FOUND FALSE)
    set(clio-core_NOT_FOUND_MESSAGE
        \"clio-coreConfig.cmake forwards to iowarp-coreConfig.cmake but it \"
        \"was not found at \${_clio_core_iowarp_dir}/iowarp-coreConfig.cmake\")
    return()
endif()
include(\"\${_clio_core_iowarp_dir}/iowarp-coreConfig.cmake\")
")
configure_file(
    "${CMAKE_CURRENT_BINARY_DIR}/iowarp-coreConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/clio-coreConfigVersion.cmake"
    COPYONLY)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/clio-coreConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/clio-coreConfigVersion.cmake"
    DESTINATION
        lib/cmake/clio-core
)

message(STATUS "")
message(STATUS "IOWarp Core unified package configuration:")
message(STATUS "  Package name: iowarp-core (canonical alias: clio-core)")
message(STATUS "  Install destination: lib/cmake/iowarp-core, lib/cmake/clio-core")
message(STATUS "  Components included:")
message(STATUS "    - ClioCtp (context-transport-primitives)")
message(STATUS "    - Chimaera (runtime)")
message(STATUS "    - Core ChiMods: admin, bdev")
if(CLIO_CORE_ENABLE_CTE)
    message(STATUS "    - CTE ChiMod: clio_cte_core")
endif()
if(CLIO_CORE_ENABLE_CAE)
    message(STATUS "    - CAE ChiMod: clio_cae_core")
endif()

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "IOWarp Core Configuration Summary:")
message(STATUS "  Build Shared Libraries:       ${BUILD_SHARED_LIBS}")
message(STATUS "  ")
message(STATUS "  Component Modules:")
message(STATUS "    Runtime:                    ${CLIO_CORE_ENABLE_RUNTIME}")
message(STATUS "    Context Transfer Engine:    ${CLIO_CORE_ENABLE_CTE}")
message(STATUS "    Context Assimilation:       ${CLIO_CORE_ENABLE_CAE}")
message(STATUS "    Context Exploration:        ${CLIO_CORE_ENABLE_CEE}")
message(STATUS "  ")
message(STATUS "  External Examples:")
message(STATUS "    Gray-Scott (ADIOS2):        ${CLIO_CORE_ENABLE_GRAY_SCOTT}")
message(STATUS "  ")
message(STATUS "  Global Options:")
message(STATUS "    Enable Tests:               ${CLIO_CORE_ENABLE_TESTS}")
message(STATUS "    Python Bindings:            ${CLIO_CORE_ENABLE_PYTHON}")
message(STATUS "    AddressSanitizer (ASan):    ${CLIO_CORE_ENABLE_ASAN}")
message(STATUS "    UndefinedBehaviorSan (UBSan): ${CLIO_CORE_ENABLE_UBSAN}")
message(STATUS "    MemorySanitizer (MSan):     ${CLIO_CORE_ENABLE_MSAN}")
message(STATUS "    Code Coverage:              ${CLIO_CORE_ENABLE_COVERAGE}")
message(STATUS "")


#------------------------------------------------------------------------------
# Pip Package Component
#------------------------------------------------------------------------------
# All COMPONENT pip_package installs land under iowarp_core/ inside the wheel
# via scikit-build-core's wheel.install-dir setting (see installers/pip/
# pyproject.toml).  That means the DESTINATIONs below are package-relative:
# "lib" -> iowarp_core/lib, "bin" -> iowarp_core/bin, etc.
#
# Exception: the .pth file uses DESTINATION ".." to land at site-packages
# root (one level above the iowarp_core/ package directory) so Python picks
# it up at startup.

# Shared libraries -> iowarp_core/lib/
set(PIP_LIB_TARGETS)
foreach(_t clio_ctp_host clio_run_cxx)
  if(TARGET ${_t})
    list(APPEND PIP_LIB_TARGETS ${_t})
  endif()
endforeach()
if(TARGET clio_admin_client)
  list(APPEND PIP_LIB_TARGETS clio_admin_client clio_admin_runtime)
endif()
if(TARGET clio_bdev_client)
  list(APPEND PIP_LIB_TARGETS clio_bdev_client clio_bdev_runtime)
endif()
# Template chimod (used by clio_run_thrpt_bench's "latency" test case via
# chimaera::MOD_NAME_client).  Without this the benchmark binary's NEEDED list
# references libchimaera_MOD_NAME_client.so but the wheel doesn't contain it,
# and the dynamic linker segfaults at load time on glibc/patchelf combinations
# instead of failing cleanly.  See #429.
if(TARGET chimaera_MOD_NAME_client)
  list(APPEND PIP_LIB_TARGETS chimaera_MOD_NAME_client chimaera_MOD_NAME_runtime)
endif()
if(CLIO_CORE_ENABLE_CTE AND TARGET clio_cte_core_client)
  list(APPEND PIP_LIB_TARGETS clio_cte_core_client clio_cte_core_runtime)
endif()
if(CLIO_CORE_ENABLE_CAE AND TARGET clio_cae_core_client)
  list(APPEND PIP_LIB_TARGETS clio_cae_core_client clio_cae_core_runtime)
endif()
if(CLIO_CORE_ENABLE_CEE AND TARGET clio_cee_api)
  list(APPEND PIP_LIB_TARGETS clio_cee_api)
endif()

if(PIP_LIB_TARGETS)
  install(TARGETS ${PIP_LIB_TARGETS}
    COMPONENT pip_package
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/chimaera)
  # Mirror the install into the default ("Unspecified") component so the
  # same shared libs ship in the .deb / .rpm built by cpack -G DEB/RPM
  # (those generators filter to CPACK_COMPONENTS_ALL=Unspecified). Without
  # this, libclio_admin_client.so / libclio_admin_runtime.so /
  # libclio_bdev_*.so / libchimaera_MOD_NAME_*.so are only in the pip
  # wheel — and the .rpm's clio_run binary then fails to start with
  #   error while loading shared libraries: libclio_admin_client.so
  # because its NEEDED entry (Fedora's ld doesn't strip --as-needed as
  # aggressively as Ubuntu's) can't be resolved. The .deb happened to
  # work because Ubuntu's linker stripped the direct NEEDED reference,
  # but the libs were still missing from the package — same root cause,
  # different surface symptom. Shipping the libs in both packages is
  # the correct fix.
  install(TARGETS ${PIP_LIB_TARGETS}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/chimaera)
endif()

# CLI binary -> iowarp_core/bin/
# Install the canonical `clio_run` binary plus a legacy `chimaera` symlink.
# Matches the parallel install in context-runtime/util/CMakeLists.txt for
# system installs; this branch handles the pip-wheel COMPONENT pip_package.
if(TARGET clio_run)
  install(TARGETS clio_run
    COMPONENT pip_package
    RUNTIME DESTINATION bin)
  install(CODE "
    if(WIN32)
      set(_pip_src \"\${CMAKE_INSTALL_PREFIX}/bin/clio_run.exe\")
      set(_pip_dst \"\${CMAKE_INSTALL_PREFIX}/bin/chimaera.exe\")
    else()
      set(_pip_src \"\${CMAKE_INSTALL_PREFIX}/bin/clio_run\")
      set(_pip_dst \"\${CMAKE_INSTALL_PREFIX}/bin/chimaera\")
    endif()
    if(EXISTS \"\${_pip_src}\" AND NOT EXISTS \"\${_pip_dst}\")
      if(WIN32)
        file(INSTALL \"\${_pip_src}\"
             DESTINATION \"\${CMAKE_INSTALL_PREFIX}/bin\" RENAME chimaera.exe)
      else()
        execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink clio_run \"\${_pip_dst}\")
      endif()
    endif()
  " COMPONENT pip_package)
  # Also install to PYTHON_SITE_PACKAGES/iowarp_core/bin/ so that
  # `sudo make install` keeps the binary in sync with the Python package,
  # matching how chimaera_runtime_ext.so is installed directly to site-packages.
  if(CLIO_CORE_ENABLE_PYTHON AND PYTHON_SITE_PACKAGES)
    install(TARGETS clio_run
      RUNTIME DESTINATION "${PYTHON_SITE_PACKAGES}/iowarp_core/bin")
    install(CODE "
      if(WIN32)
        set(_py_src \"${PYTHON_SITE_PACKAGES}/iowarp_core/bin/clio_run.exe\")
        set(_py_dst \"${PYTHON_SITE_PACKAGES}/iowarp_core/bin/chimaera.exe\")
      else()
        set(_py_src \"${PYTHON_SITE_PACKAGES}/iowarp_core/bin/clio_run\")
        set(_py_dst \"${PYTHON_SITE_PACKAGES}/iowarp_core/bin/chimaera\")
      endif()
      if(EXISTS \"\${_py_src}\" AND NOT EXISTS \"\${_py_dst}\")
        if(WIN32)
          file(INSTALL \"\${_py_src}\"
               DESTINATION \"${PYTHON_SITE_PACKAGES}/iowarp_core/bin\" RENAME chimaera.exe)
        else()
          execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink clio_run \"\${_py_dst}\")
        endif()
      endif()
    ")
  endif()
endif()

# Benchmark binaries -> iowarp_core/bin/
# Per-subdir install(TARGETS) calls in */benchmark/CMakeLists.txt land in
# the default component, which scikit-build-core filters out of the wheel.
# Re-install the user-facing benchmarks here under COMPONENT pip_package
# so they ship in the wheel.  if(TARGET ...) guards skip benchmarks that
# weren't built (e.g. MPI/Redis-only ones disabled in pip configurations).
if(CLIO_CORE_ENABLE_BENCHMARKS)
  set(PIP_BENCH_TARGETS
    clio_cte_bench
    clio_run_thrpt_bench)
  foreach(_bench ${PIP_BENCH_TARGETS})
    if(TARGET ${_bench})
      install(TARGETS ${_bench}
        COMPONENT pip_package
        RUNTIME DESTINATION bin)
    endif()
  endforeach()
endif()

# Belt-and-braces fallback install for context-runtime binaries into the
# default ("Unspecified") component picked up by `cpack -G DEB/RPM` (the
# component filter is set in cmake/packaging/CPackConfig.cmake to
# `Unspecified` only — pip_package files are filtered out).
#
# The per-subdir installs at context-runtime/util/CMakeLists.txt:40 and
# context-runtime/benchmark/CMakeLists.txt:17 are functionally identical
# to the working installs at context-transfer-engine/benchmark and
# context-assimilation-engine/core, but at v1.9.5 the resulting .deb
# shipped without clio_run / clio_run_thrpt_bench / chimaera while it
# DID ship clio_cte_bench / clio_cae / allocator_benchmark. Exact root
# cause is unidentified (the per-subdir rules look correct in cmake_install.cmake).
# Add explicit top-level installs as a safety net so any future drop-out
# can't ship a half-broken native package silently. PR #458's new
# test-deb / test-rpm jobs will catch a recurrence.
if(TARGET clio_run)
  install(TARGETS clio_run RUNTIME DESTINATION bin)
endif()
if(TARGET clio_run_thrpt_bench)
  install(TARGETS clio_run_thrpt_bench RUNTIME DESTINATION bin)
endif()
# `chimaera` is created as a symlink to `clio_run` by an install(CODE)
# in context-runtime/util/CMakeLists.txt — that runs after this point
# during cmake --install, so the symlink lands in the default
# component too. No extra rule needed here.

# Python extensions -> iowarp_core/ext/
if(CLIO_CORE_ENABLE_PYTHON)
  if(TARGET clio_cte_core_ext)
    install(TARGETS clio_cte_core_ext
      COMPONENT pip_package
      LIBRARY DESTINATION ext)
  endif()
  if(TARGET clio_cee)
    install(TARGETS clio_cee
      COMPONENT pip_package
      LIBRARY DESTINATION ext)
  endif()
  # Note: the context_visualizer Python package ships at the wheel root
  # (top-level "context_visualizer/") via scikit-build-core's wheel.packages
  # setting in pyproject.toml, not via CMake install.  That keeps the
  # `python -m context_visualizer` and `context-visualizer` entry-point
  # paths consistent with the package's own __main__.py.
endif()

# Default config -> iowarp_core/data/
install(FILES context-runtime/config/chimaera_default.yaml
  COMPONENT pip_package
  DESTINATION data)

# .pth file at site-packages root so "import clio_cee" works without
# explicitly importing iowarp_core first (triggers _setup() at startup).
# DESTINATION ".." escapes wheel.install-dir (= iowarp_core/) by one level
# so the file lands at platlib root.
install(FILES installers/pip/iowarp_core.pth
  COMPONENT pip_package
  DESTINATION ..)

# On Windows, bundle vcpkg runtime DLLs into bin/ so the installed .exe
# binaries and the .pyd extensions can find their dependencies. vcpkg's
# build-tree VCPKG_APPLOCAL_DEPS handles the .exe-adjacent copies during
# build, but the install tree only contains our own DLLs unless we copy
# the third-party ones ourselves. Without this step, every clio_cte_bench
# invocation fails with "libzmq-mt-*.dll: cannot open shared object file".
if(WIN32 AND DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
  set(_vcpkg_bin "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
  if(IS_DIRECTORY "${_vcpkg_bin}")
    file(GLOB _vcpkg_dlls "${_vcpkg_bin}/*.dll")
    if(_vcpkg_dlls)
      install(FILES ${_vcpkg_dlls}
        COMPONENT pip_package
        DESTINATION bin)
    endif()
  endif()
endif()

#------------------------------------------------------------------------------
# CPack Configuration
#------------------------------------------------------------------------------
# Delegates to cmake/packaging/CPackConfig.cmake for centralized
# package generation control (DEB, RPM, TGZ)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/packaging/CPackConfig.cmake)

#------------------------------------------------------------------------------
# AppImage Configuration
#------------------------------------------------------------------------------
# Only include AppImage support if explicitly enabled
if(CLIO_CORE_ENABLE_APPIMAGE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/packaging/AppImage.cmake)
endif()
