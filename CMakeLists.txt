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
option(WRP_CORE_ENABLE_BENCHMARKS "Master switch: Enable benchmarks for all components" OFF)
option(WRP_CORE_ENABLE_PYTHON "Enable Python bindings for all components" OFF)
option(WRP_CORE_ENABLE_CMAKE_DOTENV "Enable reading environment variables from .env.cmake" OFF)
option(WRP_CORE_ENABLE_ASAN "Enable AddressSanitizer for debugging" OFF)
option(WRP_CORE_ENABLE_COVERAGE "Enable code coverage instrumentation for all components" OFF)
option(WRP_CORE_ENABLE_DOXYGEN "Check how well the code is documented" OFF)
option(WRP_CORE_ENABLE_RPATH "Enable RPATH for installed libraries" OFF)
option(WRP_CORE_ENABLE_CPACK "Enable CPack for package generation" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)

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
option(WRP_CORE_ENABLE_COMPRESS "Enable compression" OFF)
option(WRP_CORE_ENABLE_ENCRYPT "Enable encryption" OFF)
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
option(HSHM_ENABLE_TESTS "Enable tests for HermesShm" ON)
option(HSHM_ENABLE_BENCHMARKS "Enable benchmarks for HermesShm" ON)
option(HSHM_ENABLE_PTHREADS "Support spawning pthreads" OFF)
option(HSHM_ENABLE_WINDOWS_THREADS "Support spawning windows threads" OFF)
option(HSHM_DEBUG_LOCK "Used for debugging locks" OFF)
option(HSHM_NO_COMPILE "Disable compiling / installing this library" OFF)

#------------------------------------------------------------------------------
# Chimaera (context-runtime) Options
#------------------------------------------------------------------------------
option(CHIMAERA_ENABLE_TESTS "Enable tests for Chimaera runtime" ON)
option(CHIMAERA_ENABLE_BENCHMARKS "Enable benchmarks for Chimaera runtime" ON)

#------------------------------------------------------------------------------
# CTE (context-transfer-engine) Options
#------------------------------------------------------------------------------
option(WRP_CTE_ENABLE_TESTS "Enable tests for CTE" ON)
option(WRP_CTE_ENABLE_BENCHMARKS "Enable benchmarks for CTE" ON)
option(WRP_CTE_ENABLE_PYTHON "Enable Python bindings for CTE (requires WRP_CORE_ENABLE_PYTHON=ON)" ON)
option(WRP_CORE_INSTALL_WRAPPER_TO_VENV "Install Python wrappers to Python site-packages instead of lib/" ON)

# CTE Adapter Options (require WRP_CORE_ENABLE_ELF=ON)
option(WRP_CTE_ENABLE_POSIX_ADAPTER "Enable POSIX adapter" ON)
option(WRP_CTE_ENABLE_STDIO_ADAPTER "Enable STDIO adapter" OFF)
option(WRP_CTE_ENABLE_MPIIO_ADAPTER "Enable MPI-IO adapter" OFF)
option(WRP_CTE_ENABLE_VFD "Enable HDF5 VFD adapter" OFF)
option(WRP_CTE_ENABLE_NVIDIA_GDS_ADAPTER "Enable NVIDIA GDS adapter" OFF)

#------------------------------------------------------------------------------
# CAE (context-assimilation-engine) Options
#------------------------------------------------------------------------------
option(WRP_CAE_ENABLE_TESTS "Enable tests for CAE" ON)
option(WRP_CAE_ENABLE_BENCHMARKS "Enable benchmarks for CAE" ON)
option(CAE_ENABLE_GLOBUS "Enable Globus transfer support in CAE" OFF)

#------------------------------------------------------------------------------
# CEE (context-exploration-engine) Options
#------------------------------------------------------------------------------
option(WRP_CEE_ENABLE_TESTS "Enable tests for CEE" ON)
option(WRP_CEE_ENABLE_BENCHMARKS "Enable benchmarks for CEE" ON)

#==============================================================================
# END BUILD CONFIGURATION OPTIONS
#==============================================================================

# Hierarchical control logic:
# If WRP_CORE_ENABLE_TESTS is OFF, force all component tests OFF
# If WRP_CORE_ENABLE_BENCHMARKS is OFF, force all component benchmarks OFF
# Otherwise, use individual component values

# Apply master test switch
if(NOT WRP_CORE_ENABLE_TESTS)
    set(HSHM_ENABLE_TESTS OFF)
    set(CHIMAERA_ENABLE_TESTS OFF)
    set(WRP_CTE_ENABLE_TESTS OFF)
    set(WRP_CAE_ENABLE_TESTS OFF)
    set(WRP_CEE_ENABLE_TESTS OFF)
endif()

# Apply master benchmark switch
if(NOT WRP_CORE_ENABLE_BENCHMARKS)
    set(HSHM_ENABLE_BENCHMARKS OFF)
    set(CHIMAERA_ENABLE_BENCHMARKS OFF)
    set(WRP_CTE_ENABLE_BENCHMARKS OFF)
    set(WRP_CAE_ENABLE_BENCHMARKS OFF)
    set(WRP_CEE_ENABLE_BENCHMARKS OFF)
endif()

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
set(HSHM_ENABLE_COMPRESS ${WRP_CORE_ENABLE_COMPRESS} CACHE BOOL "Enable Compress (controlled by WRP_CORE_ENABLE_COMPRESS)" FORCE)
set(HSHM_ENABLE_ENCRYPT ${WRP_CORE_ENABLE_ENCRYPT} CACHE BOOL "Enable Encrypt (controlled by WRP_CORE_ENABLE_ENCRYPT)" FORCE)
set(HSHM_ENABLE_ELF ${WRP_CORE_ENABLE_ELF} CACHE BOOL "Enable ELF (controlled by WRP_CORE_ENABLE_ELF)" FORCE)
set(HSHM_ENABLE_CUDA ${WRP_CORE_ENABLE_CUDA} CACHE BOOL "Enable CUDA (controlled by WRP_CORE_ENABLE_CUDA)" FORCE)
set(HSHM_ENABLE_ROCM ${WRP_CORE_ENABLE_ROCM} CACHE BOOL "Enable ROCm (controlled by WRP_CORE_ENABLE_ROCM)" FORCE)
set(HSHM_ENABLE_COVERAGE ${WRP_CORE_ENABLE_COVERAGE} CACHE BOOL "Enable Coverage (controlled by WRP_CORE_ENABLE_COVERAGE)" FORCE)

# Also disable tests/benchmarks for disabled components
if(NOT WRP_CORE_ENABLE_CTE)
    set(WRP_CTE_ENABLE_TESTS OFF)
    set(WRP_CTE_ENABLE_BENCHMARKS OFF)
endif()

if(NOT WRP_CORE_ENABLE_CAE)
    set(WRP_CAE_ENABLE_TESTS OFF)
    set(WRP_CAE_ENABLE_BENCHMARKS OFF)
endif()

if(NOT WRP_CORE_ENABLE_CEE)
    set(WRP_CEE_ENABLE_TESTS OFF)
    set(WRP_CEE_ENABLE_BENCHMARKS OFF)
endif()

# Print final test/benchmark configuration
message(STATUS "Test/Benchmark Configuration:")
message(STATUS "  HSHM_ENABLE_TESTS: ${HSHM_ENABLE_TESTS}")
message(STATUS "  HSHM_ENABLE_BENCHMARKS: ${HSHM_ENABLE_BENCHMARKS}")
message(STATUS "  CHIMAERA_ENABLE_TESTS: ${CHIMAERA_ENABLE_TESTS}")
message(STATUS "  CHIMAERA_ENABLE_BENCHMARKS: ${CHIMAERA_ENABLE_BENCHMARKS}")
message(STATUS "  WRP_CTE_ENABLE_TESTS: ${WRP_CTE_ENABLE_TESTS}")
message(STATUS "  WRP_CTE_ENABLE_BENCHMARKS: ${WRP_CTE_ENABLE_BENCHMARKS}")
message(STATUS "  WRP_CAE_ENABLE_TESTS: ${WRP_CAE_ENABLE_TESTS}")
message(STATUS "  WRP_CAE_ENABLE_BENCHMARKS: ${WRP_CAE_ENABLE_BENCHMARKS}")
message(STATUS "  WRP_CEE_ENABLE_TESTS: ${WRP_CEE_ENABLE_TESTS}")
message(STATUS "  WRP_CEE_ENABLE_BENCHMARKS: ${WRP_CEE_ENABLE_BENCHMARKS}")

# Read environment variables from .env.cmake if enabled
if(WRP_CORE_ENABLE_CMAKE_DOTENV)
  if(EXISTS "${CMAKE_SOURCE_DIR}/.env.cmake")
    include("${CMAKE_SOURCE_DIR}/.env.cmake")
  endif()
endif()

#------------------------------------------------------------------------------
# External Library Discovery
#------------------------------------------------------------------------------

# Pkg-Config
find_package(PkgConfig REQUIRED)
if(PkgConfig_FOUND)
    message(STATUS "found pkg config")
endif()

# Doxygen (conditional)
if(WRP_CORE_ENABLE_DOXYGEN)
    find_package(Perl REQUIRED)
    find_package(Doxygen REQUIRED)
    message(STATUS "found doxygen at ${DOXYGEN_EXECUTABLE}")
endif()

# Catch2 is built from submodule - see component subdirectories section below
# No find_package needed here

# YAML-CPP - always use submodule (built later)

if (${YAML_CPP_LIBRARIES})
    set(YAML_CPP_LIBS ${YAML_CPP_LIBRARIES})
elseif(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_LIBS yaml-cpp::yaml-cpp)
else()
    set(YAML_CPP_LIBS yaml-cpp)
endif()

# Boost components - try system first, fall back to submodule
find_package(Boost QUIET COMPONENTS fiber context system)
if(Boost_FOUND)
    message(STATUS "found boost at ${Boost_INCLUDE_DIRS}")
else()
    message(STATUS "System Boost not found, building from submodule")

    # Initialize boost submodule if not already done
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/boost/CMakeLists.txt")
        message(STATUS "Initializing boost submodule...")
        execute_process(
            COMMAND git submodule update --init --recursive external/boost
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT
        )
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed for external/boost")
        endif()
    endif()

    # Build required Boost libraries from submodule
    set(BOOST_INCLUDE_LIBRARIES fiber context system)
    set(BOOST_ENABLE_CMAKE ON)
    add_subdirectory(external/boost EXCLUDE_FROM_ALL)
    message(STATUS "Building Boost fiber, context, and system from submodule")
endif()

# Cereal
# Skip find_package if target already exists (from submodule)
if(NOT TARGET cereal AND NOT TARGET cereal::cereal)
    find_package(cereal REQUIRED)
    if(cereal_FOUND)
        message(STATUS "found cereal at ${cereal_DIR}")
    endif()
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
if(WRP_CORE_ENABLE_COMPRESS)
    message(STATUS "WRP_CORE_ENABLE_COMPRESS is ON")
    pkg_check_modules(bzip2 REQUIRED bzip2)
    message(STATUS "found bz2.h at ${bzip2_INCLUDE_DIRS}")

    pkg_check_modules(lzo2 REQUIRED lzo2)
    message(STATUS "found lzo2.h at ${lzo2_INCLUDE_DIRS}")
    get_filename_component(lzo2_dir "${lzo2_INCLUDE_DIRS}" DIRECTORY)

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

    pkg_check_modules(snappy REQUIRED snappy)
    message(STATUS "found snappy.h at ${snappy_INCLUDE_DIRS}")

    pkg_check_modules(blosc2 REQUIRED blosc2)
    message(STATUS "found blosc2.h at ${blosc2_INCLUDE_DIRS}")

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

#------------------------------------------------------------------------------
# Load Common CMake Utilities (Functions Only)
#------------------------------------------------------------------------------
# Include IowarpCoreCommon.cmake which provides:
# - GPU support macros (wrp_core_enable_cuda, wrp_core_enable_rocm)
# - ChiMod build functions (add_chimod_client, add_chimod_runtime)
# - Utility functions (jarvis_repo_add, add_doxygen_doc, find_first_path_python)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(IowarpCoreCommon)

#------------------------------------------------------------------------------
# Python Bindings Configuration
#------------------------------------------------------------------------------
if(WRP_CORE_ENABLE_PYTHON)
  message(STATUS "Python bindings enabled - configuring nanobind")

  # Find Python executable using helper from IowarpCoreCommon
  find_first_path_python()

  # Find Python for nanobind
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  # Add nanobind subdirectory if it exists
  if(EXISTS "${CMAKE_SOURCE_DIR}/external/nanobind")
    add_subdirectory(external/nanobind)
    message(STATUS "Added nanobind from external/nanobind")
  else()
    message(FATAL_ERROR "nanobind not found in external/ - required when WRP_CORE_ENABLE_PYTHON=ON")
  endif()
else()
  message(STATUS "Python bindings disabled")
endif()

#------------------------------------------------------------------------------
# Global Settings
#------------------------------------------------------------------------------
# C++ Standard - Applied to all subdirectories
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDE integration - Applied to all subdirectories
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Position Independent Code - Applied to all subdirectories
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# RPATH configuration (optional)
if(WRP_CORE_ENABLE_RPATH)
    message(STATUS "RPATH enabled for installed libraries")
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

    # Platform-specific RPATH configuration
    if(APPLE)
        set(CMAKE_MACOSX_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH "@loader_path/../lib:@loader_path:${CMAKE_INSTALL_PREFIX}/lib")
    elseif(UNIX)
        set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN:${CMAKE_INSTALL_PREFIX}/lib")
    endif()
else()
    message(STATUS "RPATH disabled - libraries will not have embedded runtime paths")
    set(CMAKE_SKIP_INSTALL_RPATH TRUE)
endif()

#------------------------------------------------------------------------------
# Compiler Flags
#------------------------------------------------------------------------------
# Add compiler flags following Google C++ style guide
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# Disable some problematic warnings for external dependencies
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-variable -Wno-reorder")

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

# Add Catch2 testing framework if tests are enabled
if(WRP_CORE_ENABLE_TESTS)
  message(STATUS "Adding Catch2 testing framework")
  add_subdirectory(external/Catch2)

  # Force installation of Catch2 libraries even when used as subproject
  # This ensures libCatch2d.so and libCatch2Maind.so are installed to /usr/local/lib
  install(
    TARGETS
      Catch2
      Catch2WithMain
    EXPORT
      Catch2Targets
    LIBRARY DESTINATION
      ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION
      ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION
      ${CMAKE_INSTALL_BINDIR}
  )

  # Install Catch2 headers
  install(
    DIRECTORY
      "${CMAKE_CURRENT_SOURCE_DIR}/external/Catch2/src/catch2"
      "${CMAKE_CURRENT_BINARY_DIR}/external/Catch2/generated-includes/catch2"
    DESTINATION
      "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
      PATTERN "*.hpp"
  )

  # Install Catch2 CMake config files
  install(
    EXPORT
      Catch2Targets
    NAMESPACE
      Catch2::
    DESTINATION
      "${CMAKE_INSTALL_LIBDIR}/cmake/Catch2"
  )
endif()

# cereal header-only library - use installed version if available
if(NOT TARGET cereal AND NOT TARGET cereal::cereal)
    message(STATUS "cereal not found via find_package, checking submodule")
    if(EXISTS "${CMAKE_SOURCE_DIR}/external/cereal/CMakeLists.txt")
        message(STATUS "Building cereal library from submodule (header-only)")
        set(JUST_INSTALL_CEREAL ON CACHE BOOL "Just install cereal headers" FORCE)
        set(BUILD_DOC OFF CACHE BOOL "Disable cereal documentation" FORCE)
        set(BUILD_SANDBOX OFF CACHE BOOL "Disable cereal sandbox" FORCE)
        set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "Skip cereal performance comparison" FORCE)
        add_subdirectory(external/cereal)
    else()
        message(FATAL_ERROR "cereal not found. Please install cereal or run install.sh to build from submodules.")
    endif()
else()
    message(STATUS "Using installed cereal")
endif()

# Add yaml-cpp library (only if not already found by IowarpCoreCommon)
if(NOT TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
  message(STATUS "Building yaml-cpp library from submodule")
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib" FORCE)
  set(YAML_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build yaml-cpp as shared library" FORCE)
  add_subdirectory(external/yaml-cpp)
else()
  message(STATUS "yaml-cpp already found by IowarpCoreCommon, skipping submodule build")
endif()

# Context Transport Primitives (formerly cte-hermes-shm) - Required by runtime
message(STATUS "Building context-transport-primitives component")

# Detect ELF library if enabled (HSHM_ENABLE_ELF is set from WRP_CORE_ENABLE_ELF at line 86)
if(HSHM_ENABLE_ELF)
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
message(STATUS "  Global Options:")
message(STATUS "    Enable Tests:               ${WRP_CORE_ENABLE_TESTS}")
message(STATUS "    Python Bindings:            ${WRP_CORE_ENABLE_PYTHON}")
message(STATUS "    AddressSanitizer:           ${WRP_CORE_ENABLE_ASAN}")
message(STATUS "    Code Coverage:              ${WRP_CORE_ENABLE_COVERAGE}")
message(STATUS "")

#------------------------------------------------------------------------------
# Python Package Installation Support
#------------------------------------------------------------------------------
# When building with scikit-build-core (pip install), install Python package files

# Check if we're being built by scikit-build-core
if(DEFINED SKBUILD)
    message(STATUS "Building with scikit-build-core for Python package installation")

    # Install Python package __init__.py
    install(
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/iowarp/__init__.py"
        DESTINATION
            "iowarp"
    )

    # The Python extensions (wrp_cte_core) are already installed by the component CMakeLists.txt
    # We just need to ensure they go to the right location for Python import

    message(STATUS "Python package files will be installed to: ${CMAKE_INSTALL_PREFIX}/iowarp")
endif()

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
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libboost-fiber-dev, libboost-context-dev, libboost-filesystem-dev, libzmq3-dev, libyaml-cpp-dev")

    # Component installation
    set(CPACK_COMPONENTS_ALL libraries headers cmake)
    set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Runtime Libraries")
    set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "C++ Headers")
    set(CPACK_COMPONENT_CMAKE_DISPLAY_NAME "CMake Configuration Files")

    include(CPack)

    message(STATUS "CPack enabled - package generation available")
endif()
