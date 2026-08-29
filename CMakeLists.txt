cmake_minimum_required(VERSION 3.20)
project(iowarp-core VERSION 2.2.1 LANGUAGES C CXX)

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
option(CLIO_CORE_ENABLE_LEAK_CHECK "Enable allocator allocation-size tracking (CTP_ALLOC_TRACK_SIZE) so tests can assert no runtime memory is leaked. Adds a small per-allocation atomic; keep OFF in production." OFF)
option(CLIO_CORE_ENABLE_DOXYGEN "Check how well the code is documented" OFF)
option(CLIO_CORE_ENABLE_RPATH "Enable RPATH configuration with relative ($ORIGIN) paths" ON)
option(CLIO_CORE_ENABLE_CPACK "Enable CPack for package generation" OFF)
option(CLIO_CORE_ENABLE_DEB_PACKAGE "Build Debian .deb package via CPack" OFF)
option(CLIO_CORE_ENABLE_RPM_PACKAGE "Build RPM .rpm package via CPack" OFF)
option(CLIO_CORE_ENABLE_APPIMAGE "Build portable AppImage via appimagetool" OFF)
option(CLIO_CORE_ENABLE_WINDOWS_PACKAGE "Build Windows ZIP archive + NSIS installer .exe via CPack" OFF)
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
option(CLIO_CORE_ENABLE_BOOST_COROUTINES "Use Boost.Context stackful coroutines for task execution instead of C++20 stackless coroutines (issue #620). Opt-in; reuses fiber stacks to avoid the per-task heap frame the stackless backend allocates." OFF)

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
# ONE SWITCH FOR THE WHOLE GPU BUILD, not a per-feature flag.
#
# clang compiles CUDA as C++, so it gets C++20 device coroutines; nvcc does
# not, and nothing that needs them can be built by it. That is a property of
# the TOOLCHAIN, so it belongs to the build as a whole -- picking the compiler
# in one place and letting every GPU component ask "is this a clang build?"
# rather than each carrying its own flag and its own way to be misconfigured.
#
# gpu_vector is the component that needs it: its device API is coroutines all
# the way down. With this OFF the vector is not built at all (see its
# CMakeLists) rather than failing halfway through a compile with "namespace
# clio::run::gpu has no member YCoroTask", which is what a CI job that did not
# know about this got.
option(CLIO_GPU_CLANG
       "Use clang as the CUDA compiler for ALL GPU code (enables device coroutines; required by gpu_vector)"
       OFF)
option(CLIO_CORE_ENABLE_GPU_RUNTIME "Enable GPU runtime (CDP-based parallelism)" OFF)
option(CLIO_CORE_ENABLE_ROCM "Enable ROCm support" OFF)
option(CLIO_CORE_ENABLE_NIXL "Enable NIXL (NVIDIA Inference Xfer Library) transport" OFF)
option(CLIO_CORE_ENABLE_NVSHMEM "Enable NVSHMEM GPU-to-GPU communication library" OFF)
option(CLIO_CORE_ENABLE_SYCL "Enable Intel GPU support via SYCL/oneAPI (icpx -fsycl)" OFF)
option(CLIO_CORE_ENABLE_JARVIS "Enable Jarvis CI infrastructure installation" OFF)
option(CLIO_ENABLE_AMAZON_DRIVE "Enable Amazon S3 intermediate storage" OFF)
option(CLIO_ENABLE_GOOGLE_CLOUD "Enable Google Cloud Storage intermediate storage" OFF)

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

# Link libhdf5 into components that READ or WRITE real .h5 files but are not the
# VOL connector -- today that is CAE (the HDF5 assimilator, and ExportData, which
# exports a CTE tag as an .h5).
#
# This option is what context-assimilation-engine/CMakeLists.txt has always
# derived CLIO_CAE_ENABLE_HDF5 from, but it was never actually DECLARED anywhere,
# so it silently evaluated as empty/false and CAE's entire HDF5 path -- including
# the pre-existing ExportData -- compiled out to "HDF5 support not compiled in".
# Declaring it makes that code reachable. It stays OFF by default: HDF5 is an
# optional system dependency and nothing in the default build needs it.
option(CLIO_CORE_ENABLE_HDF5 "Enable HDF5 support in CAE (assimilator + .h5 export)" OFF)

#------------------------------------------------------------------------------
# CAE (context-assimilation-engine) Options
#------------------------------------------------------------------------------
option(CAE_ENABLE_GLOBUS "Enable Globus transfer support in CAE" OFF)
option(CAE_ENABLE_S3 "Enable Amazon S3 (s3://) import in CAE" OFF)
option(CAE_ENABLE_GCS "Enable Google Cloud Storage (gs://) import in CAE" OFF)
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
# CTP_ENABLE_COMPRESS changes clio::cte::core::Context's LAYOUT (and with it
# every Put/Get task). It must therefore be a TREE-WIDE definition: when only
# the targets linking ctp::compress saw it, the compressor module compiled a
# 440-byte PutBlobTask against a core/client world using the 344-byte one —
# tasks silently misparse and puts vanish (the ABI-skew hazard documented in
# core_tasks.h; previously bit the llama.cpp weight work the same way).
if(CLIO_CTE_ENABLE_COMPRESS)
    add_compile_definitions(CTP_ENABLE_COMPRESS=1)
endif()
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

# BEFORE ClioCoreCommon, because that is where enable_language(CUDA) runs and
# the compiler cannot be chosen after the language is enabled.
if(CLIO_GPU_CLANG)
    if(NOT CLIO_CORE_ENABLE_CUDA)
        message(FATAL_ERROR "CLIO_GPU_CLANG requires CLIO_CORE_ENABLE_CUDA=ON")
    endif()
    if(NOT CMAKE_CUDA_COMPILER)
        # Newest first: device coroutines need a recent clang, and the probe
        # is the same one a developer would do by hand.
        find_program(CLIO_GPU_CLANGXX
                     NAMES clang++-22 clang++-21 clang++-20 clang++-19 clang++
                     DOC "clang++ used as the CUDA compiler")
        if(NOT CLIO_GPU_CLANGXX)
            message(FATAL_ERROR
                    "CLIO_GPU_CLANG=ON but no clang++ was found. Install one "
                    "(clang++-22 or newer is what this tree is tested with) "
                    "or pass -DCMAKE_CUDA_COMPILER=/path/to/clang++.")
        endif()
        set(CMAKE_CUDA_COMPILER "${CLIO_GPU_CLANGXX}" CACHE FILEPATH
            "CUDA compiler (clang, selected by CLIO_GPU_CLANG)" FORCE)
    elseif(NOT CMAKE_CUDA_COMPILER MATCHES "clang")
        message(FATAL_ERROR
                "CLIO_GPU_CLANG=ON but CMAKE_CUDA_COMPILER is "
                "'${CMAKE_CUDA_COMPILER}'. nvcc cannot compile device "
                "coroutines; pass a clang++ or turn CLIO_GPU_CLANG off.")
    endif()
    message(STATUS "GPU toolchain: clang (${CMAKE_CUDA_COMPILER}) -- device coroutines on")
endif()

include(ClioCoreCommon)
# Measured per-kernel register ceilings for device-coroutine kernels; a no-op
# unless CLIO_GPU_CLANG is on.
include(ClioCoroRegCap)

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

# YAML-CPP - find from system or CI/ci-deps.sh installation
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

# Cereal - find from system or CI/ci-deps.sh installation
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
        # THESE MUST BE TOP-LEVEL, not inside wrp_core_enable_rocm().
        #
        # macros.h includes <hip/hip_runtime_api.h> for every TU with
        # CTP_ENABLE_ROCM set, and that header guards its whole body on
        # __HIP_PLATFORM_AMD__. amdclang++ defines it implicitly for -x hip;
        # the plain C++ compiler does not. Setting it inside the macro looked
        # right and was not: add_compile_definitions/include_directories are
        # DIRECTORY-scoped, and the macro is called from context-runtime/ and
        # context-transport-primitives/ -- so context-transfer-engine and
        # everything else got neither, and failed with ~470 errors of the form
        # "hipGetLastError was not declared in this scope" the moment the
        # build got far enough to reach them.
        if(NOT DEFINED ROCM_ROOT)
            if(DEFINED ROCM_PATH)
                set(ROCM_ROOT "${ROCM_PATH}" CACHE PATH "ROCm root")
            elseif(DEFINED ENV{ROCM_PATH})
                set(ROCM_ROOT "$ENV{ROCM_PATH}" CACHE PATH "ROCm root")
            else()
                set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm root")
            endif()
        endif()
        if(EXISTS "${ROCM_ROOT}/include")
            include_directories(SYSTEM "${ROCM_ROOT}/include")
        endif()
        # The HIP toolchain emits a bare `-lamdhip64` for HIP-linked targets.
        # That resolves for an install at the default prefix and not for one
        # anywhere else, failing with "unable to find library -lamdhip64", so
        # give the linker the search path explicitly. RPATH too, otherwise the
        # binaries link and then cannot start.
        if(EXISTS "${ROCM_ROOT}/lib")
            link_directories("${ROCM_ROOT}/lib")
            list(APPEND CMAKE_BUILD_RPATH "${ROCM_ROOT}/lib")
            list(APPEND CMAKE_INSTALL_RPATH "${ROCM_ROOT}/lib")
        endif()
        add_compile_definitions(__HIP_PLATFORM_AMD__=1)
        # ...and link the HIP runtime everywhere, for exactly the reason the
        # SYCL block below says of libsycl: gpu_api.h's inline helpers call
        # hipPointerGetAttributes, hipMemcpy and friends, so ANY target that
        # merely includes it needs the library -- not just the ones compiled
        # as HIP. Without this the build fails late, at link, with
        # "undefined reference to hipPointerGetAttributes ... DSO missing from
        # command line" on whichever unrelated benchmark happened to include
        # gpu_api.h.
        if(TARGET hip::host)
            link_libraries(hip::host)
        endif()
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

# CUDA 13 moved libcu++ (cuda/std/*) under include/cccl. Anything including
# <cuda/std/...> -- kvhdf5 does -- otherwise fails with a bare
# "cuda/std/cstdint: No such file or directory". This has to be at top level:
# include_directories() only affects the directory that calls it and the
# subdirectories added after, so setting it inside a helper invoked from one
# component never reaches the others. Harmless on older toolkits, where the
# directory does not exist.
if(CLIO_CORE_ENABLE_CUDA)
    foreach(_cccl "${CUDAToolkit_INCLUDE_DIRS}/cccl"
                  "/usr/local/cuda/targets/x86_64-linux/include/cccl")
        if(EXISTS "${_cccl}")
            message(STATUS "libcu++ (CUDA 13 layout) at ${_cccl}")
            include_directories("${_cccl}")
            break()
        endif()
    endforeach()
endif()

# Compression libraries (conditional)
if(CLIO_CTE_ENABLE_COMPRESS)
    message(STATUS "CLIO_CTE_ENABLE_COMPRESS is ON")

    # EVERY codec is optional.
    #
    # These used to be REQUIRED, which made enabling compression an
    # all-or-nothing proposition: a box missing any single one of nine dev
    # packages could not configure at all, even to use a codec it did have.
    # The GPU-side codecs (nvcomp, cuSZ, ...) were already detected this way;
    # the lossless CPU ones now match. A codec that is not found is compiled
    # out via its CTP_ENABLE_<X> definition, and CompressionFactory returns
    # nullptr for it -- which every caller already handles, because that is
    # what the optional GPU codecs have always done.
    set(CLIO_CTP_ENABLE_ZSTD OFF)
    set(CLIO_CTP_ENABLE_LZ4 OFF)
    set(CLIO_CTP_ENABLE_ZLIB OFF)
    set(CLIO_CTP_ENABLE_LZMA OFF)
    set(CLIO_CTP_ENABLE_BROTLI OFF)
    set(CLIO_CTP_ENABLE_BLOSC2 OFF)
    set(CLIO_CTP_ENABLE_BZIP2 OFF)
    set(CLIO_CTP_ENABLE_LZO OFF)
    set(CLIO_CTP_ENABLE_SNAPPY OFF)

    # A .pc file is not proof the headers are installed: conda ships
    # libbrotlidec.pc from a RUNTIME-only package, so pkg-config reports
    # brotli found and the build then dies on a missing brotli/decode.h.
    # Every pkg-config codec therefore also has to locate its principal
    # header before being declared usable.
    macro(clio_check_codec_header out_var pkg header)
        find_path(${out_var} ${header} HINTS ${${pkg}_INCLUDE_DIRS})
    endmacro()

    # Libraries with pkg-config support
    pkg_check_modules(libzstd QUIET libzstd)
    clio_check_codec_header(_zstd_hdr libzstd zstd.h)
    if(libzstd_FOUND AND _zstd_hdr)
        set(CLIO_CTP_ENABLE_ZSTD ON)
        message(STATUS "found zstd.h at ${libzstd_INCLUDE_DIRS}")
    else()
        message(STATUS "zstd not found - zstd codec disabled (apt: libzstd-dev)")
    endif()

    pkg_check_modules(liblz4 QUIET liblz4)
    clio_check_codec_header(_lz4_hdr liblz4 lz4.h)
    if(liblz4_FOUND AND _lz4_hdr)
        set(CLIO_CTP_ENABLE_LZ4 ON)
        message(STATUS "found lz4.h at ${liblz4_INCLUDE_DIRS}")
    else()
        message(STATUS "lz4 not found - lz4 codec disabled (apt: liblz4-dev)")
    endif()

    pkg_check_modules(zlib QUIET zlib)
    clio_check_codec_header(_zlib_hdr zlib zlib.h)
    if(zlib_FOUND AND _zlib_hdr)
        set(CLIO_CTP_ENABLE_ZLIB ON)
        message(STATUS "found zlib.h at ${zlib_INCLUDE_DIRS}")
    else()
        message(STATUS "zlib not found - zlib codec disabled (apt: zlib1g-dev)")
    endif()

    pkg_check_modules(liblzma QUIET liblzma)
    clio_check_codec_header(_lzma_hdr liblzma lzma.h)
    if(liblzma_FOUND AND _lzma_hdr)
        set(CLIO_CTP_ENABLE_LZMA ON)
        message(STATUS "found liblzma.h at ${liblzma_INCLUDE_DIRS}")
    else()
        message(STATUS "lzma not found - lzma codec disabled (apt: liblzma-dev)")
    endif()

    pkg_check_modules(libbrotlicommon QUIET libbrotlicommon libbrotlidec libbrotlienc)
    clio_check_codec_header(_brotli_hdr libbrotlicommon brotli/decode.h)
    if(libbrotlicommon_FOUND AND _brotli_hdr)
        set(CLIO_CTP_ENABLE_BROTLI ON)
        message(STATUS "found libbrotli.h at ${libbrotlicommon_INCLUDE_DIRS}")
    else()
        message(STATUS "brotli not found - brotli codec disabled (apt: libbrotli-dev)")
    endif()

    pkg_check_modules(blosc2 QUIET blosc2)
    clio_check_codec_header(_blosc2_hdr blosc2 blosc2.h)
    if(blosc2_FOUND AND _blosc2_hdr)
        set(CLIO_CTP_ENABLE_BLOSC2 ON)
        message(STATUS "found blosc2.h at ${blosc2_INCLUDE_DIRS}")
    else()
        message(STATUS "blosc2 not found - blosc codec disabled (apt: libblosc2-dev)")
    endif()

    # The find_path/find_library codecs need the same prefixes pkg-config
    # already knows about. Without this bzip2 read as "not found" on a host
    # whose bzlib.h sits in a conda prefix, while zstd and lz4 right beside it
    # were found -- because those came through pkg-config, which had been told
    # where to look, and these had not.
    set(CLIO_CODEC_HINTS "$ENV{CONDA_PREFIX}" ${CMAKE_PREFIX_PATH})

    # bzip2
    find_path(bzip2_INCLUDE_DIR bzlib.h HINTS ${CLIO_CODEC_HINTS}
              PATH_SUFFIXES include)
    find_library(bzip2_LIBRARY bz2 HINTS ${CLIO_CODEC_HINTS}
                 PATH_SUFFIXES lib lib64)
    if(bzip2_INCLUDE_DIR AND bzip2_LIBRARY)
        set(CLIO_CTP_ENABLE_BZIP2 ON)
        set(bzip2_INCLUDE_DIRS ${bzip2_INCLUDE_DIR})
        set(bzip2_LIBRARIES bz2)
        get_filename_component(bzip2_LIBRARY_DIRS "${bzip2_LIBRARY}" DIRECTORY)
        message(STATUS "found bz2.h at ${bzip2_INCLUDE_DIRS}")
    else()
        message(STATUS "bzip2 not found - bzip2 codec disabled (apt: libbz2-dev)")
    endif()

    # lzo
    find_path(lzo2_INCLUDE_DIR lzo/lzo1x.h HINTS ${CLIO_CODEC_HINTS}
              PATH_SUFFIXES include)
    find_library(lzo2_LIBRARY lzo2 HINTS ${CLIO_CODEC_HINTS}
                 PATH_SUFFIXES lib lib64)
    if(lzo2_INCLUDE_DIR AND lzo2_LIBRARY)
        set(CLIO_CTP_ENABLE_LZO ON)
        set(lzo2_INCLUDE_DIRS ${lzo2_INCLUDE_DIR})
        set(lzo2_LIBRARIES lzo2)
        get_filename_component(lzo2_LIBRARY_DIRS "${lzo2_LIBRARY}" DIRECTORY)
        get_filename_component(lzo2_dir "${lzo2_INCLUDE_DIRS}" DIRECTORY)
        message(STATUS "found lzo2.h at ${lzo2_INCLUDE_DIRS}")
    else()
        message(STATUS "lzo not found - lzo codec disabled (apt: liblzo2-dev)")
    endif()

    # snappy
    find_path(snappy_INCLUDE_DIR snappy.h HINTS ${CLIO_CODEC_HINTS}
              PATH_SUFFIXES include)
    find_library(snappy_LIBRARY snappy HINTS ${CLIO_CODEC_HINTS}
                 PATH_SUFFIXES lib lib64)
    if(snappy_INCLUDE_DIR AND snappy_LIBRARY)
        set(CLIO_CTP_ENABLE_SNAPPY ON)
        set(snappy_INCLUDE_DIRS ${snappy_INCLUDE_DIR})
        set(snappy_LIBRARIES snappy)
        get_filename_component(snappy_LIBRARY_DIRS "${snappy_LIBRARY}" DIRECTORY)
        message(STATUS "found snappy.h at ${snappy_INCLUDE_DIRS}")
    else()
        message(STATUS "snappy not found - snappy codec disabled (apt: libsnappy-dev)")
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

    # cuSZ (https://github.com/szcompressor/cuSZ) - GPU error-bounded lossy
    # compressor for floating-point data. Optional, requires CUDA. Auto-detected
    # via its CMake config (target CUSZ::cusz) with a raw find_library fallback.
    set(CLIO_CTP_ENABLE_CUSZ OFF)
    if(CLIO_CORE_ENABLE_CUDA)
        find_package(CUSZ QUIET)
        if(CUSZ_FOUND)
            set(cusz_LIBRARIES CUSZ::cusz)
            set(cusz_INCLUDE_DIRS ${CUSZ_INCLUDE_DIRS})
            set(CLIO_CTP_ENABLE_CUSZ ON)
            message(STATUS "found cuSZ via CMake config (GPU lossy compression)")
        else()
            find_path(cusz_INCLUDE_DIR cusz.h PATH_SUFFIXES cusz)
            find_library(cusz_LIBRARY NAMES cusz pszcuda cuszapi)
            if(cusz_INCLUDE_DIR AND cusz_LIBRARY)
                set(cusz_LIBRARIES ${cusz_LIBRARY})
                set(cusz_INCLUDE_DIRS ${cusz_INCLUDE_DIR})
                set(CLIO_CTP_ENABLE_CUSZ ON)
                message(STATUS "found cusz.h at ${cusz_INCLUDE_DIRS}")
            else()
                message(STATUS "cuSZ not found - cuSZ GPU compression disabled")
            endif()
        endif()
    else()
        message(STATUS "CLIO_CORE_ENABLE_CUDA is OFF - cuSZ GPU compression disabled")
    endif()

    # ndzip (https://github.com/celerity/ndzip) - GPU high-throughput lossless
    # compressor for floating-point data. Optional, requires CUDA and an ndzip
    # built with NDZIP_WITH_CUDA=ON (provides the ndzip-cuda library). Auto-
    # detected via its CMake config (ndzip::ndzip-cuda) with a find_library
    # fallback.
    set(CLIO_CTP_ENABLE_NDZIP OFF)
    if(CLIO_CORE_ENABLE_CUDA)
        find_package(ndzip QUIET)
        if(ndzip_FOUND AND TARGET ndzip::ndzip-cuda)
            set(ndzip_LIBRARIES ndzip::ndzip ndzip::ndzip-cuda)
            set(ndzip_INCLUDE_DIRS "")
            set(CLIO_CTP_ENABLE_NDZIP ON)
            message(STATUS "found ndzip via CMake config (GPU lossless compression)")
        else()
            find_path(ndzip_INCLUDE_DIR ndzip/ndzip.hh)
            find_library(ndzip_LIBRARY NAMES ndzip)
            find_library(ndzip_cuda_LIBRARY NAMES ndzip-cuda)
            if(ndzip_INCLUDE_DIR AND ndzip_LIBRARY AND ndzip_cuda_LIBRARY)
                set(ndzip_LIBRARIES ${ndzip_cuda_LIBRARY} ${ndzip_LIBRARY})
                set(ndzip_INCLUDE_DIRS ${ndzip_INCLUDE_DIR})
                set(CLIO_CTP_ENABLE_NDZIP ON)
                message(STATUS "found ndzip.hh at ${ndzip_INCLUDE_DIRS}")
            else()
                message(STATUS "ndzip not found - ndzip GPU compression disabled")
            endif()
        endif()
    else()
        message(STATUS "CLIO_CORE_ENABLE_CUDA is OFF - ndzip GPU compression disabled")
    endif()

    # cuSZp (https://github.com/szcompressor/cuSZp) - GPU ultra-fast error-bounded
    # lossy compressor for floating-point data. Optional, requires CUDA. Auto-
    # detected via its CMake config (cuSZp::cuSZp) with a find_library fallback.
    set(CLIO_CTP_ENABLE_CUSZP OFF)
    if(CLIO_CORE_ENABLE_CUDA)
        find_package(cuSZp QUIET)
        if(cuSZp_FOUND AND TARGET cuSZp::cuSZp)
            set(cuszp_LIBRARIES cuSZp::cuSZp)
            set(cuszp_INCLUDE_DIRS "")
            set(CLIO_CTP_ENABLE_CUSZP ON)
            message(STATUS "found cuSZp via CMake config (GPU lossy compression)")
        else()
            find_path(cuszp_INCLUDE_DIR cuSZp.h)
            find_library(cuszp_LIBRARY NAMES cuSZp)
            if(cuszp_INCLUDE_DIR AND cuszp_LIBRARY)
                set(cuszp_LIBRARIES ${cuszp_LIBRARY})
                set(cuszp_INCLUDE_DIRS ${cuszp_INCLUDE_DIR})
                set(CLIO_CTP_ENABLE_CUSZP ON)
                message(STATUS "found cuSZp.h at ${cuszp_INCLUDE_DIRS}")
            else()
                message(STATUS "cuSZp not found - cuSZp GPU compression disabled")
            endif()
        endif()
    else()
        message(STATUS "CLIO_CORE_ENABLE_CUDA is OFF - cuSZp GPU compression disabled")
    endif()

    # Only the codecs actually found contribute libs/includes. An absent one
    # leaves its *_LIBRARIES empty anyway, but listing it unconditionally also
    # dragged in a bare `bz2` link flag for a library that may not exist.
    set(COMPRESS_LIBS "")
    set(COMPRESS_INCLUDES "")
    set(COMPRESS_LIB_DIRS "")
    macro(clio_add_codec enabled libs includes libdirs)
        if(${enabled})
            list(APPEND COMPRESS_LIBS ${libs})
            list(APPEND COMPRESS_INCLUDES ${includes})
            list(APPEND COMPRESS_LIB_DIRS ${libdirs})
        endif()
    endmacro()
    clio_add_codec(CLIO_CTP_ENABLE_BZIP2  "${bzip2_LIBRARIES}"          "${bzip2_INCLUDE_DIRS}"          "${bzip2_LIBRARY_DIRS}")
    clio_add_codec(CLIO_CTP_ENABLE_LZO    "${lzo2_LIBRARIES}"           "${lzo2_INCLUDE_DIRS};${lzo2_dir}" "${lzo2_LIBRARY_DIRS}")
    clio_add_codec(CLIO_CTP_ENABLE_ZSTD   "${libzstd_LIBRARIES}"        "${libzstd_INCLUDE_DIRS}"        "${libzstd_LIBRARY_DIRS}")
    clio_add_codec(CLIO_CTP_ENABLE_LZ4    "${liblz4_LIBRARIES}"         "${liblz4_INCLUDE_DIRS}"         "${liblz4_LIBRARY_DIRS}")
    clio_add_codec(CLIO_CTP_ENABLE_ZLIB   "${zlib_LIBRARIES}"           "${zlib_INCLUDE_DIRS}"           "${zlib_LIBRARY_DIRS}")
    clio_add_codec(CLIO_CTP_ENABLE_LZMA   "${liblzma_LIBRARIES}"        "${liblzma_INCLUDE_DIRS}"        "${liblzma_LIBRARY_DIRS}")
    clio_add_codec(CLIO_CTP_ENABLE_BROTLI "${libbrotlicommon_LIBRARIES}" "${libbrotlicommon_INCLUDE_DIRS}" "${libbrotlicommon_LIBRARY_DIRS}")
    clio_add_codec(CLIO_CTP_ENABLE_SNAPPY "${snappy_LIBRARIES}"         "${snappy_INCLUDE_DIRS}"         "${snappy_LIBRARY_DIRS}")
    clio_add_codec(CLIO_CTP_ENABLE_BLOSC2 "${blosc2_LIBRARIES}"         "${blosc2_INCLUDE_DIRS}"         "${blosc2_LIBRARY_DIRS}")
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

    # Only add cuSZ if found
    if(CLIO_CTP_ENABLE_CUSZ)
        list(APPEND COMPRESS_LIBS ${cusz_LIBRARIES})
        list(APPEND COMPRESS_INCLUDES ${cusz_INCLUDE_DIRS})
    endif()

    # Only add ndzip if found
    if(CLIO_CTP_ENABLE_NDZIP)
        list(APPEND COMPRESS_LIBS ${ndzip_LIBRARIES})
        list(APPEND COMPRESS_INCLUDES ${ndzip_INCLUDE_DIRS})
    endif()

    # Only add cuSZp if found
    if(CLIO_CTP_ENABLE_CUSZP)
        list(APPEND COMPRESS_LIBS ${cuszp_LIBRARIES})
        list(APPEND COMPRESS_INCLUDES ${cuszp_INCLUDE_DIRS})
    endif()

    # Make the per-codec switches available to subdirectories (CTP), same as
    # the GPU codecs below.
    foreach(_codec ZSTD LZ4 ZLIB LZMA BROTLI BLOSC2 BZIP2 LZO SNAPPY)
        set(CLIO_CTP_ENABLE_${_codec} ${CLIO_CTP_ENABLE_${_codec}}
            CACHE BOOL "Enable ${_codec} codec (auto-detected)" FORCE)
    endforeach()

    # Make CTP_ENABLE_LIBPRESSIO available to subdirectories (CTP)
    set(CLIO_CTP_ENABLE_LIBPRESSIO ${CLIO_CTP_ENABLE_LIBPRESSIO} CACHE BOOL "Enable LibPressio wrapper (auto-detected)" FORCE)
    # Make CTP_ENABLE_NVCOMP available to subdirectories (CTP)
    set(CLIO_CTP_ENABLE_NVCOMP ${CLIO_CTP_ENABLE_NVCOMP} CACHE BOOL "Enable nvcomp GPU compression (auto-detected)" FORCE)
    # Make CLIO_CTP_ENABLE_ZFP_SYCL available to subdirectories (CTP). Distinct
    # from CLIO_CTP_ENABLE_SYCL (the SYCL-language switch) -- do not conflate.
    set(CLIO_CTP_ENABLE_ZFP_SYCL ${CLIO_CTP_ENABLE_ZFP_SYCL} CACHE BOOL "Enable zfp-sycl GPU compression (auto-detected)" FORCE)
    # Make CLIO_CTP_ENABLE_CUSZ / CLIO_CTP_ENABLE_NDZIP available to subdirs (CTP)
    set(CLIO_CTP_ENABLE_CUSZ ${CLIO_CTP_ENABLE_CUSZ} CACHE BOOL "Enable cuSZ GPU compression (auto-detected)" FORCE)
    set(CLIO_CTP_ENABLE_NDZIP ${CLIO_CTP_ENABLE_NDZIP} CACHE BOOL "Enable ndzip GPU compression (auto-detected)" FORCE)
    set(CLIO_CTP_ENABLE_CUSZP ${CLIO_CTP_ENABLE_CUSZP} CACHE BOOL "Enable cuSZp GPU compression (auto-detected)" FORCE)
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
        message(FATAL_ERROR "ZeroMQ not found. Please install ZeroMQ or run CI/ci-deps.sh to build from submodules.")
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
if(CLIO_CTE_ENABLE_HDF5_VOL OR CLIO_CORE_ENABLE_HDF5)
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
        message(FATAL_ERROR "HDF5 not found. Please install HDF5 or run CI/ci-deps.sh to build from submodules.")
    endif()
else()
    message(STATUS "HDF5 support disabled")
endif()

# Poco and nlohmann_json (optional - for CAE Globus support and the Google
# Cloud Storage bdev adapter, which uses Poco::Net for HTTPS + Poco::JSON).
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

# The Google Cloud Storage adapter (kGcs bdev transport) talks to GCS over HTTPS
# via Poco. If the option is requested but Poco isn't available, warn and disable
# rather than fail the configure (mirrors the CAE Globus dependency check).
if(CLIO_ENABLE_GOOGLE_CLOUD AND NOT Poco_FOUND)
  message(WARNING "CLIO_ENABLE_GOOGLE_CLOUD is ON but the Poco library "
                  "(Net, NetSSL, Crypto, JSON components) was not found. "
                  "Google Cloud Storage support will be disabled. "
                  "Install Poco (libpoco-dev) to enable it.")
  set(CLIO_ENABLE_GOOGLE_CLOUD OFF)
endif()
if(CLIO_ENABLE_GOOGLE_CLOUD)
  message(STATUS "Google Cloud Storage bdev support: ENABLED (Poco found)")
endif()

# AWS SDK for C++ (optional - for the CAE S3 assimilator, s3://). Reuses the
# same dependency the bdev S3 transport links (aws-cpp-sdk-s3/-core). If the
# option is requested but the SDK isn't found, warn and disable rather than
# fail the configure (mirrors the CAE Globus dependency check).
if(CAE_ENABLE_S3)
  find_package(AWSSDK COMPONENTS s3 QUIET)
  if(AWSSDK_FOUND)
    message(STATUS "found AWS SDK for C++ (CAE S3 support: ENABLED)")
  else()
    message(WARNING "CAE_ENABLE_S3 is ON but the AWS SDK for C++ (component s3) "
                    "was not found. S3 import will be disabled. "
                    "Install aws-sdk-cpp to enable it.")
    set(CAE_ENABLE_S3 OFF)
  endif()
endif()

# google-cloud-cpp (optional - for the CAE GCS assimilator, gs://). A new
# optional dependency (the storage component). If requested but not found,
# warn and disable rather than fail the configure.
if(CAE_ENABLE_GCS)
  find_package(google_cloud_cpp_storage QUIET)
  if(google_cloud_cpp_storage_FOUND)
    message(STATUS "found google-cloud-cpp storage (CAE GCS support: ENABLED)")
  else()
    message(WARNING "CAE_ENABLE_GCS is ON but google-cloud-cpp (storage "
                    "component) was not found. GCS import will be disabled. "
                    "Install google-cloud-cpp to enable it.")
    set(CAE_ENABLE_GCS OFF)
  endif()
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

# Allocation-size tracking for leak-detection tests. Compiler-agnostic and
# applied to every target/subdirectory so allocator headers and the runtime
# observe the same definition. Off by default to avoid the per-allocation
# atomic add/sub in production builds.
if(CLIO_CORE_ENABLE_LEAK_CHECK)
    message(STATUS "Allocation leak tracking enabled (CTP_ALLOC_TRACK_SIZE)")
    add_compile_definitions(CTP_ALLOC_TRACK_SIZE)
endif()

# C++20 Coroutines support
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fcoroutines)
  message(STATUS "GCC detected - enabling coroutines with -fcoroutines")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  # NVHPC 25.5 crashes (ICE in llc) when compiling GCC 14 libstdc++ coroutines,
  # so it cannot use the C++20 stackless backend. Force the Boost.Context
  # stackful backend instead (issue #620). The C++ carries no NVHPC-specific
  # coroutine path anymore: the compiler decision is made here, in the build,
  # and the runtime never includes <coroutine> when the Boost backend is on.
  set(CLIO_CORE_ENABLE_BOOST_COROUTINES ON CACHE BOOL
      "Forced ON: NVHPC cannot compile C++20 coroutines (issue #620)" FORCE)
  message(STATUS "NVHPC detected - C++20 coroutines unsupported; forcing Boost.Context stackful backend")
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

# Boost.Context stackful coroutine backend (issue #620). Opt-in alternative to
# the C++20 stackless coroutines above. Selected globally (the coroutine return
# type / macros are header-driven, so the whole program must agree) and links
# Boost::context into every target via the runtime's public interface.
if(CLIO_CORE_ENABLE_BOOST_COROUTINES)
  find_package(Boost REQUIRED COMPONENTS context)
  add_compile_definitions(CLIO_ENABLE_BOOST_COROUTINES=1)
  # make_task_fiber() (boost::context::fiber) is header-instantiated in every
  # module's task methods, so every target must link Boost::context. This block
  # runs before any add_subdirectory(), so link_libraries() reaches them all.
  link_libraries(Boost::context)
  message(STATUS "Boost stackful coroutine backend ENABLED (Boost::context ${Boost_VERSION})")
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

    # CUDA dialect flags are per compiler. NOTE: the CUDA language is
    # enabled by the subprojects (wrp_core_enable_cuda), AFTER this file
    # runs, so CMAKE_CUDA_COMPILER_ID does not exist yet -- detect clang
    # from the compiler the user/preset selected instead.
    if(CLIO_CORE_ENABLE_CUDA)
        if(CMAKE_CUDA_COMPILER MATCHES "clang")
            # CMake 3.28's clang-CUDA identification probes with sm_52/30/20,
            # all removed from CUDA 13's ptxas, so identification fails with
            # "ptxas fatal: Value 'sm_52' is not defined". Newer CMake probes
            # with the SDK's own defaults. (pip install cmake gets a current
            # one without touching the system package.)
            if(CMAKE_VERSION VERSION_LESS "3.30")
                message(FATAL_ERROR
                    "clang as the CUDA compiler needs CMake >= 3.30 with a "
                    "CUDA 13 toolkit (this CMake ${CMAKE_VERSION} probes "
                    "compiler identity with GPU archs the toolkit dropped).")
            endif()
            # clang -x cuda. Differences from nvcc that matter here:
            #  - no --expt-relaxed-constexpr (constexpr functions are
            #    implicitly host+device already) and no -diag-suppress
            #    (the host-call-from-device sloppiness those numbers hide
            #    is a hard error under clang -- and fixing the call sites
            #    is the point);
            #  - the toolkit is usually newer than the last CUDA release
            #    this clang was taught about;
            #  - nvcc adds the toolkit's CCCL directory (cuda/std, thrust,
            #    cub) to the include path implicitly, clang does not;
            #  - device TUs in this tree type-pun through differently-typed
            #    fills (task archives, shared-header copies). nvcc's codegen
            #    never exploited strict aliasing on them; keep clang from
            #    being the first (measured: reordered shared-header reads in
            #    the llama bridge produced wild pointers).
            find_package(CUDAToolkit REQUIRED)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -fno-strict-aliasing")
            if(EXISTS "${CUDAToolkit_TARGET_DIR}/include/cccl")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${CUDAToolkit_TARGET_DIR}/include/cccl")
            endif()
        else()
            # nvcc: allow constexpr __host__ functions in __host__ __device__
            # contexts, and quiet the host-call-from-device warning family.
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -diag-suppress=20011,20012,20013")
        endif()
    endif()

    # C++20 device coroutines as the yieldable-kernel mechanism
    # (yield_coro.h). Opt-in, and clang-only: nvcc rejects co_await in
    # device code outright. The macro mechanism stays the default and the
    # two share one stack and host protocol, so this flips per build, not
    # per kernel.
    # KEPT AS AN ALIAS. CLIO_GPU_CLANG is the switch now -- one toolchain
    # decision for all GPU code -- but scripts and presets that pass the old
    # flag should keep working rather than silently building something else.
    option(CLIO_GPU_YIELD_CORO
           "Deprecated alias for CLIO_GPU_CLANG" OFF)
    if(CLIO_GPU_YIELD_CORO AND NOT CLIO_GPU_CLANG)
        message(FATAL_ERROR
                "CLIO_GPU_YIELD_CORO is now CLIO_GPU_CLANG, which also picks "
                "the compiler. Re-run with -DCLIO_GPU_CLANG=ON.")
    endif()
    if(CLIO_GPU_CLANG)
        add_compile_definitions(CLIO_YIELD_CORO)
        message(STATUS "Yieldable GPU kernels: C++20 device coroutines (clang)")
    endif()

    # SYCL has no second mechanism to choose between. The macro yield system
    # reaches its per-block state through `extern __shared__`, which SYCL does
    # not have, so coroutines are not the opt-in there -- they are the only
    # implementation. Not gated on CLIO_GPU_CLANG: a SYCL device compiler is
    # always clang (DPC++) or AdaptiveCpp, never nvcc, so the reason that
    # option exists does not apply.
    if(CLIO_CORE_ENABLE_SYCL)
        add_compile_definitions(CLIO_YIELD_CORO)
        message(STATUS "Yieldable GPU kernels: C++20 device coroutines (SYCL)")

        # CTP_ENABLE_SYCL GLOBALLY, and this is a correctness requirement
        # rather than convenience.
        #
        # CUDA can afford to scope CTP_ENABLE_CUDA to the targets that link
        # ctp::cuda_cxx, because host and device TUs are compiled by
        # different compilers and a host-only consumer genuinely wants the
        # non-GPU behaviour. Under SYCL every TU is built by the SAME
        # compiler, and CTP_ENABLE_SYCL selects between two bodies of the
        # SAME inline functions in gpu_api.h -- IsDeviceAccessiblePointer,
        # DeviceAwareMemcpy. Defining it unevenly is an ODR violation whose
        # symptom is that the linker keeps whichever body it saw first.
        #
        # Measured: mem_bdev_transport.cc did not get the define, so its
        # DeviceAwareMemcpy was a plain std::memcpy, and the CTE's writeback
        # of a page from device memory segfaulted the runtime worker in
        # MemBdevTransport::DirectWrite.
        #
        # -fsycl stays per-target: only TUs that actually contain kernels
        # need the device pass, and it is expensive.
        add_compile_definitions(CTP_ENABLE_SYCL=1)
        # <sycl/sycl.hpp> is pulled in by macros.h whenever CTP_ENABLE_SYCL
        # is set, so every TU now needs the header path -- the driver only
        # adds it under -fsycl, which most TUs do not get.
        if(CLIO_SYCL_INCLUDE_DIR)
            include_directories(SYSTEM ${CLIO_SYCL_INCLUDE_DIR})
        endif()
        # Silence the "<sycl/sycl.hpp> included without -fsycl" pragma in
        # exactly the TUs where that is deliberate.
        add_compile_definitions(SYCL_DISABLE_FSYCL_SYCLHPP_WARNING)

        # ...and link libsycl everywhere, for the same reason the define is
        # global. With CTP_ENABLE_SYCL set, gpu_api.h's inline helpers
        # (IsDevicePointer, DeviceAwareMemcpy, HasSyclGpuDevice) reference
        # sycl:: symbols, so any target that touches them needs the library
        # -- not just the ones compiled with -fsycl. Without this the build
        # fails late, at link, with "DSO missing from command line" on
        # whichever unrelated test happened to include gpu_api.h.
        if(CLIO_SYCL_INCLUDE_DIR)
            get_filename_component(_clio_sycl_root
                                   "${CLIO_SYCL_INCLUDE_DIR}" DIRECTORY)
            if(EXISTS "${_clio_sycl_root}/lib")
                link_directories("${_clio_sycl_root}/lib")
                # ...and put it on the RUNPATH, not just the LINK path.
                # Without this every binary in a SYCL build dies at startup
                # with "libsycl.so.9: cannot open shared object file" unless
                # the caller exported LD_LIBRARY_PATH by hand -- which made
                # cte_gpu_vector_smoke_sycl a test that passes or fails
                # depending on the developer's shell, and would have made
                # every SYCL bench ctest entry do the same.
                list(APPEND CMAKE_BUILD_RPATH "${_clio_sycl_root}/lib")
                list(APPEND CMAKE_INSTALL_RPATH "${_clio_sycl_root}/lib")
            endif()
        endif()
        link_libraries(sycl)
    endif()

    # AddressSanitizer + LeakSanitizer configuration
    if(CLIO_CORE_ENABLE_ASAN)
        message(STATUS "AddressSanitizer + LeakSanitizer (ASan/LSan) enabled for iowarp-core")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
        # issue #856: annotate Boost.Context stack switches. Without these ASan
        # keeps shadow state for the stack it believes is current, so the first
        # fault on a fiber stack aborts with "DEADLYSIGNAL / nested bug in the
        # same thread" and reports NOTHING — which makes ASan useless across
        # most of the runtime, where tasks run on fibers.
        add_compile_definitions(CLIO_ASAN_FIBERS)
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
        # Lets tests that must exit via _exit() (skipping a crash-prone global
        # teardown) still flush gcov counters explicitly via __gcov_dump(); the
        # symbol only exists in coverage builds, so it is gated on this define.
        add_compile_definitions(CLIO_COVERAGE_ENABLED)
    endif()
else()
    # MSVC compiler flags
    # /FS enables synchronized PDB writes to avoid C1041 errors with parallel
    # builds. Scope it per-language: passed directly for C/CXX, but routed
    # through -Xcompiler for CUDA — nvcc treats a bare "/FS" as a second input
    # file ("nvcc fatal : A single input file is required ...").
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/FS>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FS>)
    # Template/coroutine-heavy TUs (core_runtime.cc) exceed the COFF 65k
    # section limit at default settings (C1128); raise it everywhere.
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
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

    # CTest coverage exclusions (honoured by ctest_coverage(), which drives the
    # CDash coverage submission — unlike lcov it has no per-line exclusion, so
    # untestable whole files are dropped here by filename regex). Keep this in
    # sync with the lcov --remove list in CI/calculate_coverage.sh and the
    # codecov.yml ignore list.
    #   * fuse_cte_main.cc — FUSE process entry / mount glue (fuse_main + the
    #     apptainer --fusemount fd-injection path); not unit-testable.
    file(WRITE "${CMAKE_BINARY_DIR}/CTestCustom.cmake"
         "set(CTEST_CUSTOM_COVERAGE_EXCLUDE \${CTEST_CUSTOM_COVERAGE_EXCLUDE}\n"
         "    \".*/fuse_cte_main\\\\.cc\")\n")

    # -----------------------------------------------------------------------
    # Test cleanup fixture (pre + post)
    # Kills any orphaned CLIO Runtime server processes and removes leftover shared
    # state (IPC socket files, shared memory, memfd) plus build-dir test
    # artifacts.  The pre-test variant runs once before any other CTest test.
    # The post-test variant is registered as FIXTURES_CLEANUP and is invoked
    # after every test that requires the fixture, even if those tests fail
    # or crash, ensuring temp files do not persist between runs.
    # -----------------------------------------------------------------------
    # Freeing the runtime ports is the load-bearing half of this fixture, and it
    # was a silent no-op wherever `fuser` is absent -- which includes the
    # project's own dev container. `2>/dev/null` swallowed "command not found"
    # and the trailing `exit 0` made the fixture report success regardless, so
    # the next test to start an embedded runtime hit
    #   Failed to start main server on 127.0.0.1:9413 - Address already in use
    # and died in ~0.02 s. Reproduced at 4 failures in 5 consecutive runs of
    # `ctest -R "vfd|cte_hdf5_vol_io_"`, while the same tests pass 12/12 in
    # isolation -- i.e. it presents as a flaky test and is actually a
    # deterministic missing dependency.
    #
    # Try fuser, then ss, then lsof, and say so when none of them exists rather
    # than pretending the port is clear.
    #
    # The per-user IPC dir below is expanded by the SHELL, not by CMake. It was
    # `${USER}` -- a bare CMake variable, which CMake does not define -- so the
    # generated command was literally `rm -rf /tmp/clio_`, and the directory it
    # was meant to clear was never touched on any platform. The spelling here
    # mirrors SystemInfo::GetMemfdDir exactly, fallback included: CLIO_MEMFD_DIR
    # override, else /tmp/clio_$USER, else the literal "unknown" that the
    # runtime itself uses when USER is unset. Guessing a different fallback (the
    # real uid, say) would clean a directory the runtime never writes to.
    set(_clio_port_kill
        # 10620 is the HDF5 VOL I/O suite. Any test moved off the default port
        # has to be listed here too, or it silently loses the stray-listener
        # sweep that moving off the default was meant to make unnecessary.
        #
        # The ports this comment used to list as "still missing" are now here.
        # That gap was not cosmetic: 10500 is the GPU stress suite's port
        # (CLIO_PORT=10500 in its ctest ENVIRONMENT), so a runtime that
        # outlived its test kept the port and every later test that needed it
        # failed to bind. In a full run that presents as a CASCADE of dozens of
        # unrelated red tests, each of which passes when run on its own.
        "_clio_ports='5555 5558 9413 9414 9416 10500 10596 10597 10598 10599 10613 10615 10617 10620'; \
         if command -v fuser >/dev/null 2>&1; then \
           for _p in \$_clio_ports; do fuser -k \$_p/tcp >/dev/null 2>&1; done; \
         elif command -v ss >/dev/null 2>&1; then \
           for _p in \$_clio_ports; do \
             ss -tlnpH \"sport = :\$_p\" 2>/dev/null \
               | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u \
               | while read -r _pid; do [ -n \"\$_pid\" ] && kill -9 \"\$_pid\" 2>/dev/null; done; \
           done; \
         elif command -v lsof >/dev/null 2>&1; then \
           for _p in \$_clio_ports; do lsof -ti tcp:\$_p 2>/dev/null | xargs -r kill -9 2>/dev/null; done; \
         else \
           echo 'clio cleanup: no fuser/ss/lsof; cannot free runtime ports -- back-to-back embedded-runtime tests may fail to bind' >&2; \
         fi")

    # A port sweep only finds processes that still HOLD a port. A leaked daemon
    # keeps its SHM segments and poisons later tests just as effectively, so
    # also reap it by name.
    #
    # EXACT match, never a prefix. Every test executable here is named
    # clio_run_<something> and `comm` is truncated to 15 chars, so a
    # /^clio_run/ pattern also matches clio_run_shutdown_battletest (as
    # "clio_run_shutdo") and kills the test that is currently running -- that
    # mistake turned 7 green shutdown battletests red. `pkill -f clio_run` is
    # worse still: it matches the cleanup shell itself.
    # NOTE ON THE DOLLARS: CMake passes $ through this string LITERALLY, so
    # what is written here is what bash receives. awk's field refs must
    # therefore be $1/$2 (writing $$2 emits $$2, which awk reads as $($2)),
    # the shell variable must be $_pid (writing $$_pid emits bash's PID glued
    # to "_pid"), and bash's own PID must be $$. Both mistakes were made here
    # first and neither is visible in the CMake source -- they were found by
    # reading the GENERATED command with `ctest -N -V`.
    set(_clio_proc_kill
        "ps -eo pid,comm 2>/dev/null \
           | awk -v self=$$ '($2 == \"clio_run\" || $2 == \"clio_daemon\") && $1 != self {print $1}' \
           | while read -r _pid; do kill -9 \"$_pid\" 2>/dev/null; done; \
         true")
    set(_clio_cleanup_cmd
        "${_clio_port_kill}; ${_clio_proc_kill}; \
         rm -f /tmp/clio_*.ipc 2>/dev/null; \
         rm -f /tmp/clio_server_timing.log 2>/dev/null; \
         rm -rf /tmp/clio_memfd 2>/dev/null; \
         rm -rf \"\${CLIO_MEMFD_DIR:-/tmp/clio_\${USER:-unknown}}\" 2>/dev/null; \
         rm -f /dev/shm/sm_segment.* 2>/dev/null; \
         rm -f /dev/shm/test_buddy_bench_* 2>/dev/null; \
         rm -f /tmp/test_bdev_* /tmp/cae_error_test_storage.dat /tmp/safe_cons_* 2>/dev/null; \
         rm -f \"${CMAKE_BINARY_DIR}\"/cte_*.dat* \"${CMAKE_BINARY_DIR}\"/cte_*.yaml \"${CMAKE_BINARY_DIR}\"/*stress*.bin* \"${CMAKE_BINARY_DIR}\"/*reorganize*.bin* \"${CMAKE_BINARY_DIR}\"/*reorganize*.yaml \"${CMAKE_BINARY_DIR}\"/temp_unregister_test.dat 2>/dev/null; \
         rm -rf \"${CMAKE_BINARY_DIR}\"/cee_test_data 2>/dev/null; \
         exit 0")

    if(UNIX)
      add_test(
          NAME cr_pre_test_cleanup
          COMMAND bash -c "${_clio_cleanup_cmd}"
      )
      set_tests_properties(cr_pre_test_cleanup PROPERTIES
          TIMEOUT 30
          FIXTURES_SETUP clio_test_cleanup_fixture
      )

      add_test(
          NAME cr_post_test_cleanup
          COMMAND bash -c "${_clio_cleanup_cmd}"
      )
      set_tests_properties(cr_post_test_cleanup PROPERTIES
          TIMEOUT 30
          FIXTURES_CLEANUP clio_test_cleanup_fixture
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
      #
      # %TEMP% is scrubbed too, mirroring the POSIX path's /tmp cleanup. Several
      # tests write bdev backing files under temp_directory_path() and never
      # delete them, so each suite run leaked GBs onto the system drive:
      #   test_bdev_*                  10MB, some 100MB (bdev ChiMod tests)
      #   cae_error_test_storage.dat   1GB   (CAE error-handling test)
      #   safe_cons_*.bin/.alog        16MB  (safe-bdev consumer tests)
      # Left alone that eventually fills C:, and the failures it causes are
      # baffling: tests fail, or ctest dies mid-run, with no disk-space message
      # anywhere in the output.
      set(_clio_cleanup_cmd_win
          "$ErrorActionPreference='Continue'; \
           Get-Process | Where-Object { $_.Path -and $_.Path -like '${CMAKE_BINARY_DIR}/bin/*' } | Stop-Process -Force -ErrorAction SilentlyContinue; \
           Get-ChildItem (Join-Path $env:TEMP '*') -File -Include 'test_bdev_*','cae_error_test_storage.dat','safe_cons_*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; \
           Get-ChildItem '${CMAKE_BINARY_DIR}' -Filter 'cte_*.dat*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; \
           Get-ChildItem '${CMAKE_BINARY_DIR}' -Filter 'cte_*.yaml' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; \
           Get-ChildItem '${CMAKE_BINARY_DIR}' -Filter '*stress*.bin*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; \
           Get-ChildItem '${CMAKE_BINARY_DIR}' -Filter '*reorganize*.bin*' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; \
           if (Test-Path '${CMAKE_BINARY_DIR}/cee_test_data') { Remove-Item -Recurse -Force '${CMAKE_BINARY_DIR}/cee_test_data' -ErrorAction SilentlyContinue }; \
           exit 0")

      add_test(
          NAME cr_pre_test_cleanup
          COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command "${_clio_cleanup_cmd_win}"
      )
      set_tests_properties(cr_pre_test_cleanup PROPERTIES
          TIMEOUT 30
          FIXTURES_SETUP clio_test_cleanup_fixture
      )

      add_test(
          NAME cr_post_test_cleanup
          COMMAND powershell -NoProfile -ExecutionPolicy Bypass -Command "${_clio_cleanup_cmd_win}"
      )
      set_tests_properties(cr_post_test_cleanup PROPERTIES
          TIMEOUT 30
          FIXTURES_CLEANUP clio_test_cleanup_fixture
      )

      # Pre-allow every test binary through Windows Defender Firewall so ctest
      # doesn't pop a per-binary "Allow access?" prompt. Adding rules needs
      # Administrator, but the script never elevates on its own: an unanswerable
      # UAC dialog would block until this fixture hit its TIMEOUT. Unelevated it
      # just reports which binaries still need rules; apply them by running
      # scripts/windows/allow_firewall.ps1 once from an elevated prompt. It is
      # best-effort (always exits 0), and is a SETUP for the shared fixture so it
      # runs before any networked test.
      add_test(
          NAME cr_firewall_allow
          COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                  -File "${CMAKE_SOURCE_DIR}/scripts/windows/allow_firewall.ps1"
      )
      set_tests_properties(cr_firewall_allow PROPERTIES
          TIMEOUT 240
          FIXTURES_SETUP clio_test_cleanup_fixture
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
  message(FATAL_ERROR "yaml-cpp was not found. Please run CI/ci-deps.sh to install dependencies.")
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

    function(_clio_lock_tests_recursive dir lsan_opts)
        get_directory_property(_tests DIRECTORY "${dir}" TESTS)
        if(_tests)
            set_tests_properties(${_tests}
                DIRECTORY "${dir}"
                PROPERTIES RESOURCE_LOCK "clio_runtime_port")
            if(lsan_opts)
                set_tests_properties(${_tests}
                    DIRECTORY "${dir}"
                    PROPERTIES ENVIRONMENT_MODIFICATION
                        "LSAN_OPTIONS=set:${lsan_opts}")
            endif()
        endif()
        get_directory_property(_subdirs DIRECTORY "${dir}" SUBDIRECTORIES)
        foreach(_sub ${_subdirs})
            _clio_lock_tests_recursive("${_sub}" "${lsan_opts}")
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
            _clio_lock_tests_recursive(
                "${CMAKE_CURRENT_SOURCE_DIR}/${_component_dir}"
                "${_lsan_opts}")
        endif()
    endforeach()

    # -----------------------------------------------------------------------
    # Per-test DEFAULTS: what every test gets unless it opts out.
    #
    # Two things. The cleanup fixture, so cr_post_test_cleanup runs after the
    # suite even when individual tests fail. And isolation from process-global
    # CLIO state, because the default was the other way round and that is the
    # expensive half of this tree's debugging cost.
    #
    # The specific case: RuntimeManager replays ~/.clio/restart_log.bin on EVERY
    # startup (manager.cc), re-composing whatever is registered there through the
    # admin client. That file is HOME-scoped and shared by every test, every
    # local `clio_run compose start`, and every previous session. 61 test files
    # stand up a runtime; 7 isolated the WAL by hand. The other 54 inherited
    # whatever the machine happened to be carrying -- and a foreign entry
    # composes pool 512.0 with create-params that are not theirs.
    #
    # Isolating it per test here rather than per test file means a NEW test is
    # isolated by default and sharing is what you opt into, which is the way
    # round it should have been. Tests that genuinely want shared state can
    # still set the variable themselves.
    #
    # PREPENDED, not appended: ctest applies ENVIRONMENT entries in order and the
    # last assignment wins, so a test that sets the same variable keeps its own
    # value. (Tests that setenv() in C++ override the process env regardless.)
    #
    # Not stamped here, deliberately:
    #   CLIO_PORT -- would override the port in a test's own CLIO_SERVER_CONF
    #     yaml (env beats yaml in ConfigManager), pointing the client at a
    #     different port than the runtime it just started. Isolation that breaks
    #     working tests is not isolation.
    #   CLIO_REQUIRE_RUNTIME -- plenty of tests legitimately run with no runtime
    #     (the *_no_runtime binaries exist to prove the connectors survive that).
    #     It belongs on suites that mean to measure a live tier, not on all.
    # -----------------------------------------------------------------------
    function(_clio_apply_test_defaults_recursive dir)
        get_directory_property(_tests DIRECTORY "${dir}" TESTS)
        foreach(_t ${_tests})
            if(NOT _t STREQUAL "cr_pre_test_cleanup" AND
               NOT _t STREQUAL "cr_post_test_cleanup")
                set_property(TEST ${_t} DIRECTORY "${dir}" APPEND PROPERTY
                    FIXTURES_REQUIRED clio_test_cleanup_fixture)
                get_property(_env TEST ${_t} DIRECTORY "${dir}"
                             PROPERTY ENVIRONMENT)
                set_property(TEST ${_t} DIRECTORY "${dir}" PROPERTY ENVIRONMENT
                    "CLIO_RESTART_LOG=${CMAKE_BINARY_DIR}/test_isolation/${_t}.restart.bin"
                    ${_env})
            endif()
        endforeach()
        get_directory_property(_subdirs DIRECTORY "${dir}" SUBDIRECTORIES)
        foreach(_sub ${_subdirs})
            _clio_apply_test_defaults_recursive("${_sub}")
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
            _clio_apply_test_defaults_recursive(
                "${CMAKE_CURRENT_SOURCE_DIR}/${_component_dir}")
        endif()
    endforeach()

    # -----------------------------------------------------------------------
    # Per-test environment applied to every component test.
    #
    # 1. CLIO_TEST_MODE=1 makes the test hermetic: the runtime then ignores the
    #    developer's ~/.clio/clio.yaml (see ConfigManager::GetServerConfigPath)
    #    and defaults to a loopback bind. Without it, whatever happens to be in
    #    that file leaks into every test -- a real cluster hostfile makes each
    #    test come up as a peer and block on nodes that aren't running, and a
    #    `capacity: 0g` RAM bdev sizes itself at 80% of DRAM and falls back to
    #    the slow private-heap path. Both look like tests that time out locally
    #    but pass in CI, which has no such file. Tests needing a specific config
    #    set CLIO_SERVER_CONF (in ENVIRONMENT or via setenv), which still wins.
    #
    # 2. On Windows with icx, prepend the compiler's own runtime DLL directory.
    #    Those binaries import the Intel runtime (libmmdd.dll and friends) from
    #    there, and nothing copies it next to our executables, so the tests only
    #    run if oneAPI's setvars.bat ran in the launching shell AND succeeded.
    #    When it hasn't, the loader raises a MODAL "System Error" dialog: the
    #    test does not fail, it blocks until ctest's TIMEOUT expires (a 1500s
    #    hang for ctp_sub_alloc, with no output to explain it), and the dialog
    #    waits for a human that a dashboard run does not have. Deriving the
    #    directory from CMAKE_CXX_COMPILER keeps this correct for any oneAPI
    #    install location.
    #
    # APPEND so these compose with the LSAN_OPTIONS set above.
    # -----------------------------------------------------------------------
    # ENVIRONMENT_MODIFICATION needs CMake >= 3.22, above this project's
    # declared 3.20 floor, so gate on it explicitly.
    if(NOT CMAKE_VERSION VERSION_LESS 3.22)
        set(_clio_test_env_mods "CLIO_TEST_MODE=set:1")

        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            get_filename_component(_clio_icx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
            if(IS_DIRECTORY "${_clio_icx_bin}")
                list(APPEND _clio_test_env_mods
                     "PATH=path_list_prepend:${_clio_icx_bin}")
                message(STATUS
                    "Tests: prepending Intel runtime DLL dir to PATH: ${_clio_icx_bin}")
            endif()
        endif()

        function(_clio_append_test_env_recursive dir mods)
            get_directory_property(_tests DIRECTORY "${dir}" TESTS)
            foreach(_t ${_tests})
                set_property(TEST ${_t} DIRECTORY "${dir}" APPEND PROPERTY
                    ENVIRONMENT_MODIFICATION "${mods}")
            endforeach()
            get_directory_property(_subdirs DIRECTORY "${dir}" SUBDIRECTORIES)
            foreach(_sub ${_subdirs})
                _clio_append_test_env_recursive("${_sub}" "${mods}")
            endforeach()
        endfunction()

        # Same guard as the loops above: a component's source directory can
        # exist on disk while being disabled, in which case CMake never
        # processed it and get_directory_property() would error out.
        foreach(_component_dir_pair
                "context-transport-primitives:CLIO_CORE_ENABLE_TESTS"
                "context-runtime:CLIO_CORE_ENABLE_RUNTIME"
                "context-transfer-engine:CLIO_CORE_ENABLE_CTE"
                "context-assimilation-engine:CLIO_CORE_ENABLE_CAE"
                "context-exploration-engine:CLIO_CORE_ENABLE_CEE")
            string(REPLACE ":" ";" _pair "${_component_dir_pair}")
            list(GET _pair 0 _component_dir)
            list(GET _pair 1 _component_flag)
            if(${_component_flag} AND
               IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_component_dir}")
                _clio_append_test_env_recursive(
                    "${CMAKE_CURRENT_SOURCE_DIR}/${_component_dir}"
                    "${_clio_test_env_mods}")
            endif()
        endforeach()
    else()
        message(WARNING
            "CMake < 3.22: tests will read the per-user ~/.clio/clio.yaml and "
            "may hang or time out if it configures a cluster or a 0g capacity")
    endif()
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
message(STATUS "    - Clio (runtime)")
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
message(STATUS "    Leak tracking (alloc size): ${CLIO_CORE_ENABLE_LEAK_CHECK}")
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
# clio::run::MOD_NAME_client).  Without this the benchmark binary's NEEDED list
# references libclio_MOD_NAME_client.so but the wheel doesn't contain it,
# and the dynamic linker segfaults at load time on glibc/patchelf combinations
# instead of failing cleanly.  See #429.
if(TARGET clio_MOD_NAME_client)
  list(APPEND PIP_LIB_TARGETS clio_MOD_NAME_client clio_MOD_NAME_runtime)
endif()
if(CLIO_CORE_ENABLE_CTE AND TARGET clio_cte_core_client)
  list(APPEND PIP_LIB_TARGETS clio_cte_core_client clio_cte_core_runtime)
endif()
# Cache + replication chimods (#886) build ON by default and ship in the
# deb/rpm/AppImage via the module macros' default-component installs — but
# the wheel filters to COMPONENT pip_package, so without these entries
# `pip install iowarp-core` silently lacks both modules (#899).
if(TARGET clio_cte_cache_client)
  list(APPEND PIP_LIB_TARGETS clio_cte_cache_client clio_cte_cache_runtime)
endif()
if(TARGET clio_cte_replication_client)
  list(APPEND PIP_LIB_TARGETS clio_cte_replication_client clio_cte_replication_runtime)
endif()
# Filesystem chimod: NEEDED by the clio_cte_fuse adapter binary that ships
# in the wheel when CLIO_CTE_ENABLE_FUSE_ADAPTER is ON (#899).
if(TARGET clio_cte_filesystem_client)
  list(APPEND PIP_LIB_TARGETS clio_cte_filesystem_client clio_cte_filesystem_runtime)
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
    PUBLIC_HEADER DESTINATION include/clio_runtime)
  # Mirror the install into the default ("Unspecified") component so the
  # same shared libs ship in the .deb / .rpm built by cpack -G DEB/RPM
  # (those generators filter to CPACK_COMPONENTS_ALL=Unspecified). Without
  # this, libclio_admin_client.so / libclio_admin_runtime.so /
  # libclio_bdev_*.so / libclio_MOD_NAME_*.so are only in the pip
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
    PUBLIC_HEADER DESTINATION include/clio_runtime)
endif()

# CLI binary -> iowarp_core/bin/ (pip-wheel COMPONENT pip_package).
if(TARGET clio_run)
  install(TARGETS clio_run
    COMPONENT pip_package
    RUNTIME DESTINATION bin)
  # Also install to PYTHON_SITE_PACKAGES/iowarp_core/bin/ so that
  # `sudo make install` keeps the binary in sync with the Python package,
  # matching how clio_runtime_ext.so is installed directly to site-packages.
  if(CLIO_CORE_ENABLE_PYTHON AND PYTHON_SITE_PACKAGES)
    install(TARGETS clio_run
      RUNTIME DESTINATION "${PYTHON_SITE_PACKAGES}/iowarp_core/bin")
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
# shipped without clio_run / clio_run_thrpt_bench / clio_run while it
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
# `clio_run` is created as a symlink to `clio_run` by an install(CODE)
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
  # Note: the web dashboard is served by the runtime itself (issue #990);
  # its per-ChiMod assets install via clio_add_viz_assets() under both the
  # default and pip_package components.
endif()

# Default config -> iowarp_core/data/
install(FILES context-runtime/config/clio_default.yaml
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
      # Mirror into the default component so the native Windows package
      # (CPack ZIP/NSIS filters to Unspecified, like DEB/RPM) also ships
      # the third-party DLLs its exes need (#899).
      install(FILES ${_vcpkg_dlls}
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
