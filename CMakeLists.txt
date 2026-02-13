cmake_minimum_required(VERSION 3.20)
project(iowarp-core VERSION 1.0.0 LANGUAGES C CXX)

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
option(WRP_CORE_ENABLE_RUNTIME "Enable runtime component" ON)
option(WRP_CORE_ENABLE_CTE "Enable context-transfer-engine component" ON)
option(WRP_CORE_ENABLE_CAE "Enable context-assimilation-engine component" ON)
option(WRP_CORE_ENABLE_CEE "Enable context-exploration-engine component" ON)

#------------------------------------------------------------------------------
# Global Master Switches
#------------------------------------------------------------------------------
option(WRP_CORE_ENABLE_TESTS "Master switch: Enable testing for all components" OFF)
option(WRP_CORE_ENABLE_BENCHMARKS "Master switch: Enable benchmarks for all components" ON)
option(WRP_CORE_ENABLE_PYTHON "Enable Python bindings for all components" OFF)
option(WRP_CORE_ENABLE_CMAKE_DOTENV "Enable reading environment variables from .env.cmake" OFF)
option(WRP_CORE_ENABLE_ASAN "Enable AddressSanitizer for debugging" OFF)
option(WRP_CORE_ENABLE_COVERAGE "Enable code coverage instrumentation for all components" OFF)
option(WRP_CORE_ENABLE_DOXYGEN "Check how well the code is documented" OFF)
option(WRP_CORE_ENABLE_RPATH "Enable RPATH configuration with absolute paths" ON)
option(WRP_CORE_ENABLE_CPACK "Enable CPack for package generation" OFF)
option(WRP_CORE_ENABLE_CONDA "Enable Conda environment configuration (adds Conda paths to include/link directories)" OFF)
option(WRP_CORE_ENABLE_DOCKER_CI "Enable Docker-based integration tests (requires Docker)" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)

# Installation Prefix Configuration
# - For Conda builds: Set CMAKE_INSTALL_PREFIX via -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
# - For custom installs: Set CMAKE_INSTALL_PREFIX to desired root directory
# Libraries install to ${CMAKE_INSTALL_PREFIX}/lib
# Python bindings install to ${CMAKE_INSTALL_PREFIX}/lib/pythonX.Y/site-packages

#------------------------------------------------------------------------------
# Conda Environment Configuration (Explicit)
#------------------------------------------------------------------------------
# Conda support is enabled explicitly via WRP_CORE_ENABLE_CONDA option.
# When enabled, the build system expects CONDA_PREFIX to be set.
if(WRP_CORE_ENABLE_CONDA)
    if(NOT DEFINED ENV{CONDA_PREFIX})
        message(FATAL_ERROR "WRP_CORE_ENABLE_CONDA is ON but CONDA_PREFIX environment variable is not set")
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
    message(STATUS "Conda environment support disabled (use -DWRP_CORE_ENABLE_CONDA=ON to enable)")
endif()

# Print build system summary
message(STATUS "Build system configuration:")
message(STATUS "  Conda build:        ${IN_CONDA_BUILD}")
message(STATUS "  Install prefix:     ${CMAKE_INSTALL_PREFIX}")

#------------------------------------------------------------------------------
# Transport and Network Features
#------------------------------------------------------------------------------
option(WRP_CORE_ENABLE_MPI "Enable MPI support" OFF)
option(WRP_CORE_ENABLE_ZMQ "Enable ZeroMQ transport" ON)
option(WRP_CORE_ENABLE_LIBFABRIC "Enable Libfabric transport" OFF)
option(WRP_CORE_ENABLE_THALLIUM "Build tests which depend on thallium" OFF)

#------------------------------------------------------------------------------
# Data and Serialization Features
#------------------------------------------------------------------------------
option(WRP_CORE_ENABLE_CEREAL "Enable serialization using cereal" ON)
option(WRP_CTE_ENABLE_COMPRESS "Enable compression support (builds compressor chimod and tests)" OFF)
option(WRP_CORE_ENABLE_ENCRYPT "Enable encryption" OFF)

# Set internal compression flags from WRP_CTE_ENABLE_COMPRESS
if(WRP_CTE_ENABLE_COMPRESS)
    set(WRP_CORE_ENABLE_COMPRESS ON)
else()
    set(WRP_CORE_ENABLE_COMPRESS OFF)
endif()
option(WRP_CORE_ENABLE_HDF5 "Enable HDF5 support in CAE" ON)

#------------------------------------------------------------------------------
# System and Runtime Features
#------------------------------------------------------------------------------
option(WRP_CORE_ENABLE_ELF "Enable ELF support (required for CTE adapters)" OFF)
option(WRP_CORE_ENABLE_OPENMP "Enable the use of OpenMP" OFF)
option(WRP_CORE_ENABLE_CUDA "Enable CUDA support" OFF)
option(WRP_CORE_ENABLE_ROCM "Enable ROCm support" OFF)
option(WRP_CORE_ENABLE_JARVIS "Enable Jarvis CI infrastructure installation" OFF)

#------------------------------------------------------------------------------
# HermesShm (context-transport-primitives) Options
#------------------------------------------------------------------------------
option(HSHM_ENABLE_PTHREADS "Support spawning pthreads" OFF)
option(HSHM_ENABLE_WINDOWS_THREADS "Support spawning windows threads" OFF)
option(HSHM_DEBUG_LOCK "Used for debugging locks" OFF)
option(HSHM_NO_COMPILE "Disable compiling / installing this library" OFF)
set(HSHM_LOG_LEVEL "1" CACHE STRING "Log level threshold (0=Debug, 1=Info, 2=Warning, 3=Error, 4=Fatal)")

#------------------------------------------------------------------------------
# Chimaera (context-runtime) Options
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# CTE (context-transfer-engine) Options
#------------------------------------------------------------------------------
# CTE Adapter Options (require WRP_CORE_ENABLE_ELF=ON)
option(WRP_CTE_ENABLE_POSIX_ADAPTER "Enable POSIX adapter" ON)
option(WRP_CTE_ENABLE_STDIO_ADAPTER "Enable STDIO adapter" OFF)
option(WRP_CTE_ENABLE_MPIIO_ADAPTER "Enable MPI-IO adapter" OFF)
option(WRP_CTE_ENABLE_VFD "Enable HDF5 VFD adapter" OFF)
option(WRP_CTE_ENABLE_NVIDIA_GDS_ADAPTER "Enable NVIDIA GDS adapter" OFF)
option(WRP_CTE_ENABLE_ADIOS2_ADAPTER "Enable ADIOS2 adapter" OFF)

#------------------------------------------------------------------------------
# CAE (context-assimilation-engine) Options
#------------------------------------------------------------------------------
option(CAE_ENABLE_GLOBUS "Enable Globus transfer support in CAE" OFF)

#------------------------------------------------------------------------------
# External Examples and Applications
#------------------------------------------------------------------------------
option(WRP_CORE_ENABLE_GRAY_SCOTT "Enable Gray-Scott ADIOS2 example (requires ADIOS2)" OFF)

#==============================================================================
# END BUILD CONFIGURATION OPTIONS
#==============================================================================

# Hierarchical control logic:
# If WRP_CORE_ENABLE_TESTS is OFF, force all component tests OFF
# If WRP_CORE_ENABLE_BENCHMARKS is OFF, force all component benchmarks OFF
# Otherwise, use individual component values

# Apply master test switch (no-op, WRP_CORE_ENABLE_TESTS used directly)

# Set HSHM_ENABLE_* aliases for backward compatibility
# These allow component code to continue using HSHM_ENABLE_* while we transition
# Mark them as cache variables to prevent options from overriding them
set(HSHM_ENABLE_MPI ${WRP_CORE_ENABLE_MPI} CACHE BOOL "Enable MPI (controlled by WRP_CORE_ENABLE_MPI)" FORCE)
set(HSHM_ENABLE_ZMQ ${WRP_CORE_ENABLE_ZMQ} CACHE BOOL "Enable ZMQ (controlled by WRP_CORE_ENABLE_ZMQ)" FORCE)
set(HSHM_ENABLE_LIBFABRIC ${WRP_CORE_ENABLE_LIBFABRIC} CACHE BOOL "Enable Libfabric (controlled by WRP_CORE_ENABLE_LIBFABRIC)" FORCE)
set(HSHM_ENABLE_THALLIUM ${WRP_CORE_ENABLE_THALLIUM} CACHE BOOL "Enable Thallium (controlled by WRP_CORE_ENABLE_THALLIUM)" FORCE)
set(HSHM_ENABLE_OPENMP ${WRP_CORE_ENABLE_OPENMP} CACHE BOOL "Enable OpenMP (controlled by WRP_CORE_ENABLE_OPENMP)" FORCE)
set(HSHM_ENABLE_CEREAL ${WRP_CORE_ENABLE_CEREAL} CACHE BOOL "Enable Cereal (controlled by WRP_CORE_ENABLE_CEREAL)" FORCE)
set(HSHM_ENABLE_DOXYGEN ${WRP_CORE_ENABLE_DOXYGEN} CACHE BOOL "Enable Doxygen (controlled by WRP_CORE_ENABLE_DOXYGEN)" FORCE)
set(HSHM_ENABLE_COMPRESS ${WRP_CTE_ENABLE_COMPRESS} CACHE BOOL "Enable Compress (controlled by WRP_CTE_ENABLE_COMPRESS)" FORCE)
set(HSHM_ENABLE_ENCRYPT ${WRP_CORE_ENABLE_ENCRYPT} CACHE BOOL "Enable Encrypt (controlled by WRP_CORE_ENABLE_ENCRYPT)" FORCE)
set(HSHM_ENABLE_ELF ${WRP_CORE_ENABLE_ELF} CACHE BOOL "Enable ELF (controlled by WRP_CORE_ENABLE_ELF)" FORCE)
set(HSHM_ENABLE_CUDA ${WRP_CORE_ENABLE_CUDA} CACHE BOOL "Enable CUDA (controlled by WRP_CORE_ENABLE_CUDA)" FORCE)
set(HSHM_ENABLE_ROCM ${WRP_CORE_ENABLE_ROCM} CACHE BOOL "Enable ROCm (controlled by WRP_CORE_ENABLE_ROCM)" FORCE)
# Note: Coverage is handled globally via add_compile_options when WRP_CORE_ENABLE_COVERAGE=ON
# No need for component-level HSHM_ENABLE_COVERAGE

# Print final test/benchmark configuration
message(STATUS "Test/Benchmark Configuration:")
message(STATUS "  WRP_CORE_ENABLE_TESTS: ${WRP_CORE_ENABLE_TESTS}")
message(STATUS "  WRP_CORE_ENABLE_BENCHMARKS: ${WRP_CORE_ENABLE_BENCHMARKS}")

# Read environment variables from .env.cmake if enabled
if(WRP_CORE_ENABLE_CMAKE_DOTENV)
  if(EXISTS "${CMAKE_SOURCE_DIR}/.env.cmake")
    include("${CMAKE_SOURCE_DIR}/.env.cmake")
  endif()
endif()

#------------------------------------------------------------------------------
# Load IowarpCoreCommon utilities
#------------------------------------------------------------------------------
# Provides macros for dependency resolution and ChiMod build functions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(IowarpCoreCommon)

#------------------------------------------------------------------------------
# External Library Discovery
#------------------------------------------------------------------------------
# Priority order for dependency discovery:
# 1. Conda environment (if IN_CONDA_BUILD is TRUE)
# 2. System packages via pkg-config
# 3. CMake find_package() in standard locations

# Pkg-Config
find_package(PkgConfig REQUIRED)
if(PkgConfig_FOUND)
    message(STATUS "found pkg-config")
    if(IN_CONDA_BUILD)
        # Set pkg-config search path to prioritize conda environment
        set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
        message(STATUS "PKG_CONFIG_PATH prioritizes Conda: $ENV{PKG_CONFIG_PATH}")
    endif()
endif()

# Doxygen (conditional)
if(WRP_CORE_ENABLE_DOXYGEN)
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

# Cereal - find from system or install.sh installation
find_package(cereal REQUIRED)
if(cereal_FOUND)
    message(STATUS "found cereal at ${cereal_DIR}")
endif()

# MPICH (only find when explicitly enabled)
if(WRP_CORE_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C CXX)
    set(MPI_LIBS MPI::MPI_CXX)
    message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")
endif()

# ROCm (conditional)
if(WRP_CORE_ENABLE_ROCM)
    find_package(HIP REQUIRED)
    message(STATUS "found ROCm/HIP support")
endif()

# OpenMP (conditional)
if(WRP_CORE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS C CXX)
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
    message(STATUS "found omp.h at ${OpenMP_CXX_INCLUDE_DIRS}")
endif()

# Thallium (conditional)
if(WRP_CORE_ENABLE_THALLIUM)
    find_package(thallium CONFIG REQUIRED)
    if(thallium_FOUND)
        message(STATUS "found thallium at ${thallium_DIR}")
    endif()
    set(SERIALIZATION_LIBS thallium ${SERIALIZATION_LIBS})
    set(WRP_CORE_ENABLE_CEREAL ON)
endif()

# Update cereal libraries if enabled
if(WRP_CORE_ENABLE_CEREAL)
    set(SERIALIZATION_LIBS cereal::cereal ${SERIALIZATION_LIBS})
endif()

# Thread support
find_package(Threads REQUIRED)

# Compression libraries (conditional)
if(WRP_CTE_ENABLE_COMPRESS)
    message(STATUS "WRP_CTE_ENABLE_COMPRESS is ON")

    # Verify conda environment is enabled for compression libraries
    if(NOT WRP_CORE_ENABLE_CONDA OR NOT IN_CONDA_BUILD)
        message(FATAL_ERROR "WRP_CTE_ENABLE_COMPRESS requires WRP_CORE_ENABLE_CONDA=ON. Please enable conda or install compression libraries manually.")
    endif()

    # Libraries with pkg-config support - verify they're from conda
    pkg_check_modules(libzstd REQUIRED libzstd)
    if(NOT libzstd_INCLUDE_DIRS MATCHES "$ENV{CONDA_PREFIX}")
        message(FATAL_ERROR "libzstd not from conda: ${libzstd_INCLUDE_DIRS} (expected in $ENV{CONDA_PREFIX})")
    endif()
    message(STATUS "found zstd.h at ${libzstd_INCLUDE_DIRS}")

    pkg_check_modules(liblz4 REQUIRED liblz4)
    if(NOT liblz4_INCLUDE_DIRS MATCHES "$ENV{CONDA_PREFIX}")
        message(FATAL_ERROR "liblz4 not from conda: ${liblz4_INCLUDE_DIRS} (expected in $ENV{CONDA_PREFIX})")
    endif()
    message(STATUS "found lz4.h at ${liblz4_INCLUDE_DIRS}")

    pkg_check_modules(zlib REQUIRED zlib)
    if(NOT zlib_INCLUDE_DIRS MATCHES "$ENV{CONDA_PREFIX}")
        message(FATAL_ERROR "zlib not from conda: ${zlib_INCLUDE_DIRS} (expected in $ENV{CONDA_PREFIX})")
    endif()
    message(STATUS "found zlib.h at ${zlib_INCLUDE_DIRS}")

    pkg_check_modules(liblzma REQUIRED liblzma)
    if(NOT liblzma_INCLUDE_DIRS MATCHES "$ENV{CONDA_PREFIX}")
        message(FATAL_ERROR "liblzma not from conda: ${liblzma_INCLUDE_DIRS} (expected in $ENV{CONDA_PREFIX})")
    endif()
    message(STATUS "found liblzma.h at ${liblzma_INCLUDE_DIRS}")

    pkg_check_modules(libbrotlicommon REQUIRED libbrotlicommon libbrotlidec libbrotlienc)
    if(NOT libbrotlicommon_INCLUDE_DIRS MATCHES "$ENV{CONDA_PREFIX}")
        message(FATAL_ERROR "libbrotlicommon not from conda: ${libbrotlicommon_INCLUDE_DIRS} (expected in $ENV{CONDA_PREFIX})")
    endif()
    message(STATUS "found libbrotli.h at ${libbrotlicommon_INCLUDE_DIRS}")

    pkg_check_modules(blosc2 REQUIRED blosc2)
    if(NOT blosc2_INCLUDE_DIRS MATCHES "$ENV{CONDA_PREFIX}")
        message(FATAL_ERROR "blosc2 not from conda: ${blosc2_INCLUDE_DIRS} (expected in $ENV{CONDA_PREFIX})")
    endif()
    message(STATUS "found blosc2.h at ${blosc2_INCLUDE_DIRS}")

    # Libraries without pkg-config - use conda paths directly
    # bzip2
    find_path(bzip2_INCLUDE_DIR bzlib.h PATHS $ENV{CONDA_PREFIX}/include NO_DEFAULT_PATH REQUIRED)
    find_library(bzip2_LIBRARY bz2 PATHS $ENV{CONDA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
    if(bzip2_INCLUDE_DIR AND bzip2_LIBRARY)
        set(bzip2_INCLUDE_DIRS ${bzip2_INCLUDE_DIR})
        set(bzip2_LIBRARIES bz2)
        set(bzip2_LIBRARY_DIRS $ENV{CONDA_PREFIX}/lib)
        message(STATUS "found bz2.h at ${bzip2_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "bzip2 not found in conda environment. Install with: conda install bzip2")
    endif()

    # lzo
    find_path(lzo2_INCLUDE_DIR lzo/lzo1x.h PATHS $ENV{CONDA_PREFIX}/include NO_DEFAULT_PATH REQUIRED)
    find_library(lzo2_LIBRARY lzo2 PATHS $ENV{CONDA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
    if(lzo2_INCLUDE_DIR AND lzo2_LIBRARY)
        set(lzo2_INCLUDE_DIRS ${lzo2_INCLUDE_DIR})
        set(lzo2_LIBRARIES lzo2)
        set(lzo2_LIBRARY_DIRS $ENV{CONDA_PREFIX}/lib)
        get_filename_component(lzo2_dir "${lzo2_INCLUDE_DIRS}" DIRECTORY)
        message(STATUS "found lzo2.h at ${lzo2_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "lzo not found in conda environment. Install with: conda install lzo")
    endif()

    # snappy
    find_path(snappy_INCLUDE_DIR snappy.h PATHS $ENV{CONDA_PREFIX}/include NO_DEFAULT_PATH REQUIRED)
    find_library(snappy_LIBRARY snappy PATHS $ENV{CONDA_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
    if(snappy_INCLUDE_DIR AND snappy_LIBRARY)
        set(snappy_INCLUDE_DIRS ${snappy_INCLUDE_DIR})
        set(snappy_LIBRARIES snappy)
        set(snappy_LIBRARY_DIRS $ENV{CONDA_PREFIX}/lib)
        message(STATUS "found snappy.h at ${snappy_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "snappy not found in conda environment. Install with: conda install snappy")
    endif()

    # LibPressio - try find_package first (CMake-based), fall back to pkg-config
    # Make it optional since libpressio may not be installed
    set(HSHM_ENABLE_LIBPRESSIO OFF)
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
        set(HSHM_ENABLE_LIBPRESSIO ON)
    else()
        pkg_check_modules(libpressio QUIET libpressio)
        if(libpressio_FOUND)
            message(STATUS "found libpressio.h at ${libpressio_INCLUDE_DIRS}")
            set(HSHM_ENABLE_LIBPRESSIO ON)
        else()
            message(STATUS "libpressio not found - LibPressio wrapper will be disabled")
            set(libpressio_LIBRARIES "")
            set(libpressio_INCLUDE_DIRS "")
            set(libpressio_LIBRARY_DIRS "")
        endif()
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
    if(HSHM_ENABLE_LIBPRESSIO)
        list(APPEND COMPRESS_LIBS ${libpressio_LIBRARIES})
        list(APPEND COMPRESS_INCLUDES ${libpressio_INCLUDE_DIRS})
        list(APPEND COMPRESS_LIB_DIRS ${libpressio_LIBRARY_DIRS})
    endif()

    # Make HSHM_ENABLE_LIBPRESSIO available to subdirectories (HSHM)
    set(HSHM_ENABLE_LIBPRESSIO ${HSHM_ENABLE_LIBPRESSIO} CACHE BOOL "Enable LibPressio wrapper (auto-detected)" FORCE)
endif()

# Encryption libraries (conditional)
if(WRP_CORE_ENABLE_ENCRYPT)
    pkg_check_modules(libcrypto REQUIRED libcrypto)
    message(STATUS "found libcrypto.h at ${libcrypto_INCLUDE_DIRS}")

    set(ENCRYPT_LIBS ${libcrypto_LIBRARIES})
    set(ENCRYPT_INCLUDES ${libcrypto_INCLUDE_DIRS})
    set(ENCRYPT_LIB_DIRS ${libcrypto_LIBRARY_DIRS})
endif()

# Note: ELF library detection moved to after HSHM_ENABLE_ELF is set (line ~520)

# ZeroMQ (conditional) - try system first, fall back to submodule
if(WRP_CORE_ENABLE_ZMQ)
    pkg_check_modules(ZeroMQ QUIET libzmq)
    if(ZeroMQ_FOUND)
        message(STATUS "found zmq.h at ${ZeroMQ_INCLUDE_DIRS}")
        set(ZMQ_LIBS ${ZeroMQ_LIBRARIES})
        set(ZMQ_INCLUDES ${ZeroMQ_INCLUDE_DIRS})
        set(ZMQ_LIB_DIRS ${ZeroMQ_LIBRARY_DIRS})
    else()
        message(FATAL_ERROR "ZeroMQ not found. Please install ZeroMQ or run install.sh to build from submodules.")
    endif()
endif()

# Libfabric (conditional)
if(WRP_CORE_ENABLE_LIBFABRIC)
    pkg_check_modules(Libfabric REQUIRED libfabric)
    message(STATUS "found rdma/fabric.h at ${Libfabric_INCLUDE_DIRS}")
    set(LIBFABRIC_LIBS ${Libfabric_LIBRARIES})
    set(LIBFABRIC_INCLUDES ${Libfabric_INCLUDE_DIRS})
    set(LIBFABRIC_LIB_DIRS ${Libfabric_LIBRARY_DIRS})
endif()

# HDF5 (optional - for CAE component)
if(WRP_CORE_ENABLE_HDF5)
    # Try CONFIG mode first (for HDF5 installed to non-standard cmake/ directory)
    find_package(HDF5 COMPONENTS C QUIET CONFIG)
    if(NOT HDF5_FOUND)
        # Fallback to module mode (for system HDF5 installations)
        find_package(HDF5 COMPONENTS C QUIET)
    endif()
    if(HDF5_FOUND)
        message(STATUS "found HDF5 at ${HDF5_INCLUDE_DIRS}")

        # Ensure HDF5_LIBRARIES is set for both CONFIG and MODULE mode
        if(NOT HDF5_LIBRARIES)
            # CONFIG mode creates targets but may not set HDF5_LIBRARIES
            # Prefer shared libraries over static for better compatibility
            if(TARGET HDF5::HDF5)
                set(HDF5_LIBRARIES HDF5::HDF5)
            elseif(TARGET hdf5-shared)
                set(HDF5_LIBRARIES hdf5-shared)
            elseif(TARGET hdf5-static)
                set(HDF5_LIBRARIES hdf5-static)
            endif()
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

if (WRP_CTE_ENABLE_ADIOS2_ADAPTER OR WRP_CORE_ENABLE_GRAY_SCOTT)
    # Find ADIOS2
    find_package(ADIOS2 REQUIRED)
    if(ADIOS2_FOUND)
        message(STATUS "found ADIOS2 at ${ADIOS2_DIR}")
    else()
        message(FATAL_ERROR "WRP_CORE_ENABLE_GRAY_SCOTT is ON but ADIOS2 not found. Please install ADIOS2 or disable the option.")
    endif()
endif()

# ADIOS2 (optional - for Gray-Scott example)
if(WRP_CORE_ENABLE_GRAY_SCOTT)
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
if(WRP_CORE_ENABLE_PYTHON)
  message(STATUS "Python bindings enabled - configuring nanobind")

  # Find Python executable using helper from IowarpCoreCommon
  find_first_path_python()

  # Find Python for nanobind
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  # Add nanobind as subdirectory (git submodule)
  add_subdirectory(external/nanobind)

  message(STATUS "Python bindings enabled:")
  message(STATUS "  Install prefix:     ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "  Python bindings:    ${CMAKE_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
  message(STATUS "  Python_SITEARCH:    ${Python_SITEARCH} (for reference only)")
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
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # Clang 14+ supports coroutines in C++20 mode natively
  message(STATUS "Clang detected - coroutines enabled via C++20 mode")
endif()

# Export compile commands for IDE integration - Applied to all subdirectories
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Position Independent Code - Applied to all subdirectories
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# RPATH configuration - Conditionally enabled based on WRP_CORE_ENABLE_RPATH
if(WRP_CORE_ENABLE_RPATH)
message(STATUS "RPATH ENABLED with ABSOLUTE paths including all dependency locations")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  # CRITICAL: Include dependency library paths!

# Build list of ABSOLUTE RPATH entries
set(ABSOLUTE_RPATH_LIST "")

# 1. Installation lib directory (ABSOLUTE)
list(APPEND ABSOLUTE_RPATH_LIST "${CMAKE_INSTALL_PREFIX}/lib")

# 2. Conda environment lib directory (if Conda support is enabled)
if(IN_CONDA_BUILD)
    list(APPEND ABSOLUTE_RPATH_LIST "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "  Adding Conda lib to RPATH: $ENV{CONDA_PREFIX}/lib")
endif()

# 3. Add HDF5 library directories if enabled
if(WRP_CORE_ENABLE_HDF5 AND HDF5_LIBRARY_DIRS)
    foreach(HDF5_LIB_DIR ${HDF5_LIBRARY_DIRS})
        list(APPEND ABSOLUTE_RPATH_LIST "${HDF5_LIB_DIR}")
        message(STATUS "  Adding HDF5 lib to RPATH: ${HDF5_LIB_DIR}")
    endforeach()
endif()

# 5. Add any CMAKE_PREFIX_PATH lib directories
if(CMAKE_PREFIX_PATH)
    foreach(PREFIX_PATH ${CMAKE_PREFIX_PATH})
        if(EXISTS "${PREFIX_PATH}/lib")
            list(APPEND ABSOLUTE_RPATH_LIST "${PREFIX_PATH}/lib")
            message(STATUS "  Adding prefix lib to RPATH: ${PREFIX_PATH}/lib")
        endif()
        if(EXISTS "${PREFIX_PATH}/lib64")
            list(APPEND ABSOLUTE_RPATH_LIST "${PREFIX_PATH}/lib64")
            message(STATUS "  Adding prefix lib64 to RPATH: ${PREFIX_PATH}/lib64")
        endif()
    endforeach()
endif()

# Remove duplicates
list(REMOVE_DUPLICATES ABSOLUTE_RPATH_LIST)

# Platform-specific RPATH configuration
if(APPLE)
    set(CMAKE_MACOSX_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH "${ABSOLUTE_RPATH_LIST}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/bin")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "${ABSOLUTE_RPATH_LIST}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/bin")
endif()

message(STATUS "==================================================================")
message(STATUS "RPATH Configuration (ABSOLUTE paths):")
message(STATUS "  CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Final INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")
message(STATUS "  Final BUILD_RPATH: ${CMAKE_BUILD_RPATH}")
message(STATUS "")
message(STATUS "IMPORTANT: RPATH is baked in at BUILD time!")
message(STATUS "  If you install to a different --prefix, you MUST rebuild!")
message(STATUS "  Example: cmake --preset=release -DCMAKE_INSTALL_PREFIX=/new/path")
message(STATUS "==================================================================")
message(STATUS "")
else()
message(STATUS "RPATH DISABLED - libraries may require LD_LIBRARY_PATH configuration")
endif()

#------------------------------------------------------------------------------
# Compiler Flags
#------------------------------------------------------------------------------
# Add compiler flags following Google C++ style guide
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# Disable warnings that are false positives or from external dependencies
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-variable -Wno-reorder")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pedantic -Wno-sign-compare -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-unused-function -Wno-cast-function-type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-self-move -Wno-format")

# Debug configuration
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")

# Release configuration
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# AddressSanitizer configuration
if(WRP_CORE_ENABLE_ASAN)
    message(STATUS "AddressSanitizer (ASAN) enabled for iowarp-core")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# Code Coverage configuration
if(WRP_CORE_ENABLE_COVERAGE)
    message(STATUS "Code coverage instrumentation enabled for iowarp-core")
    add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
    add_link_options(--coverage)
endif()

#------------------------------------------------------------------------------
# Binary Output Directories
#------------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# For multi-config generators (Visual Studio, Xcode)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

#------------------------------------------------------------------------------
# Testing
#------------------------------------------------------------------------------
if(WRP_CORE_ENABLE_TESTS)
    enable_testing()
    include(CTest)
endif()

#------------------------------------------------------------------------------
# Component Subdirectories
#------------------------------------------------------------------------------

# Catch2 testing framework - find from system or install.sh installation
if(WRP_CORE_ENABLE_TESTS)
  message(STATUS "Finding Catch2 testing framework")
  find_package(Catch2 3 REQUIRED)
  if(Catch2_FOUND)
    message(STATUS "found Catch2 at ${Catch2_DIR}")
  endif()
endif()


# yaml-cpp is already found by IowarpCoreCommon (line 202)
if(NOT TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
  message(FATAL_ERROR "yaml-cpp was not found. Please run install.sh to install dependencies.")
endif()
message(STATUS "yaml-cpp already found by IowarpCoreCommon")

# Context Transport Primitives (formerly cte-hermes-shm) - Required by runtime
message(STATUS "Building context-transport-primitives component")

# Detect ELF library if enabled
if(WRP_CORE_ENABLE_ELF)
    pkg_check_modules(libelf REQUIRED libelf)
    message(STATUS "found libelf.h at ${libelf_INCLUDE_DIRS}")

    set(ELF_LIBS ${libelf_LIBRARIES})
    set(ELF_INCLUDES ${libelf_INCLUDE_DIRS})
    set(ELF_LIB_DIRS ${libelf_LIBRARY_DIRS})
endif()

add_subdirectory(context-transport-primitives)
# Set HSHM_ROOT for cross-component access
set(HSHM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/context-transport-primitives" CACHE PATH "Root directory for context-transport-primitives")

# Runtime component
if(WRP_CORE_ENABLE_RUNTIME)
    message(STATUS "Building runtime component")
    add_subdirectory(context-runtime)
    # Set CHIMAERA_ROOT for cross-component access (e.g., CTE needs bdev includes)
    set(CHIMAERA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/context-runtime" CACHE PATH "Root directory for runtime component")
else()
    message(STATUS "Runtime component disabled")
endif()

# Context Transfer Engine
if(WRP_CORE_ENABLE_CTE)
    message(STATUS "Building context-transfer-engine component")
    add_subdirectory(context-transfer-engine)
else()
    message(STATUS "Context-transfer-engine component disabled")
endif()

# Context Assimilation Engine
if(WRP_CORE_ENABLE_CAE)
    message(STATUS "Building context-assimilation-engine component")
    add_subdirectory(context-assimilation-engine)
else()
    message(STATUS "Context-assimilation-engine component disabled")
endif()

# Context Exploration Engine
if(WRP_CORE_ENABLE_CEE)
    message(STATUS "Building context-exploration-engine component")
    add_subdirectory(context-exploration-engine)
else()
    message(STATUS "Context-exploration-engine component disabled")
endif()

# Gray-Scott Example Application
if(WRP_CORE_ENABLE_GRAY_SCOTT)
    message(STATUS "Building Gray-Scott ADIOS2 example")
    add_subdirectory(external/iowarp-gray-scott)
else()
    message(STATUS "Gray-Scott example disabled")
endif()

#------------------------------------------------------------------------------
# Unified IOWarp Core Package Configuration
#------------------------------------------------------------------------------
# Generate and install the unified IowarpCore package configuration

include(CMakePackageConfigHelpers)

# Configure the package config file from template
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/IowarpCoreConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/iowarp-coreConfig.cmake"
    INSTALL_DESTINATION lib/cmake/iowarp-core
)

# Generate the version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/iowarp-coreConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install the package config files
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/iowarp-coreConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/iowarp-coreConfigVersion.cmake"
        "${CMAKE_SOURCE_DIR}/cmake/IowarpCoreCommon.cmake"
    DESTINATION
        lib/cmake/iowarp-core
)

message(STATUS "")
message(STATUS "IOWarp Core unified package configuration:")
message(STATUS "  Package name: iowarp-core")
message(STATUS "  Install destination: lib/cmake/iowarp-core")
message(STATUS "  Components included:")
message(STATUS "    - HermesShm (context-transport-primitives)")
message(STATUS "    - Chimaera (runtime)")
message(STATUS "    - Core ChiMods: admin, bdev")
if(WRP_CORE_ENABLE_CTE)
    message(STATUS "    - CTE ChiMod: wrp_cte_core")
endif()
if(WRP_CORE_ENABLE_CAE)
    message(STATUS "    - CAE ChiMod: wrp_cae_core")
endif()

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "IOWarp Core Configuration Summary:")
message(STATUS "  Build Shared Libraries:       ${BUILD_SHARED_LIBS}")
message(STATUS "  ")
message(STATUS "  Component Modules:")
message(STATUS "    Runtime:                    ${WRP_CORE_ENABLE_RUNTIME}")
message(STATUS "    Context Transfer Engine:    ${WRP_CORE_ENABLE_CTE}")
message(STATUS "    Context Assimilation:       ${WRP_CORE_ENABLE_CAE}")
message(STATUS "    Context Exploration:        ${WRP_CORE_ENABLE_CEE}")
message(STATUS "  ")
message(STATUS "  External Examples:")
message(STATUS "    Gray-Scott (ADIOS2):        ${WRP_CORE_ENABLE_GRAY_SCOTT}")
message(STATUS "  ")
message(STATUS "  Global Options:")
message(STATUS "    Enable Tests:               ${WRP_CORE_ENABLE_TESTS}")
message(STATUS "    Python Bindings:            ${WRP_CORE_ENABLE_PYTHON}")
message(STATUS "    AddressSanitizer:           ${WRP_CORE_ENABLE_ASAN}")
message(STATUS "    Code Coverage:              ${WRP_CORE_ENABLE_COVERAGE}")
message(STATUS "")


#------------------------------------------------------------------------------
# CPack Configuration
#------------------------------------------------------------------------------
if(WRP_CORE_ENABLE_CPACK)
    set(CPACK_PACKAGE_NAME "iowarp-core")
    set(CPACK_PACKAGE_VENDOR "IOWarp Team")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "IOWarp Core: High-performance distributed I/O and task execution runtime")
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
    set(CPACK_PACKAGE_CONTACT "grc@illinoistech.edu")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

    # Package generators
    set(CPACK_GENERATOR "TGZ;DEB")

    # DEB-specific settings
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "IOWarp Team <grc@illinoistech.edu>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libzmq3-dev, libyaml-cpp-dev")

    # Component installation
    set(CPACK_COMPONENTS_ALL libraries headers cmake)
    set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Runtime Libraries")
    set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "C++ Headers")
    set(CPACK_COMPONENT_CMAKE_DISPLAY_NAME "CMake Configuration Files")

    include(CPack)

    message(STATUS "CPack enabled - package generation available")
endif()
