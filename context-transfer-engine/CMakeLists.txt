cmake_minimum_required(VERSION 3.20)
project(content_transfer_engine)

# Set root directory for this component
set(CLIO_CTE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Note: CLIO_CTE_ENABLE_COMPRESS is defined in root CMakeLists.txt
# It controls both compressor chimod building and CTP compression support

# Python bindings are controlled by CLIO_CORE_ENABLE_PYTHON in root CMakeLists.txt
# CTE_ENABLE_ASAN is set by root CMakeLists.txt

# C++ standard, compile commands, RPATH, ASAN, and output directories are all set by root CMakeLists.txt

# CTE_ENABLE_CMAKE_DOTENV is set by root CMakeLists.txt
# DOTENV handling is done by root CMakeLists.txt

# CMake modules and ChiMod utilities are now provided by ClioCoreCommon.cmake
# All find_package calls are handled by root CMakeLists.txt
# Targets chimaera::cxx, clio_admin_client, etc. are already available

# Python bindings configuration
# nanobind is configured by root CMakeLists.txt when CLIO_CORE_ENABLE_PYTHON=ON
# No local nanobind setup needed - use the root configuration
if(CLIO_CORE_ENABLE_PYTHON)
  if(NOT COMMAND nanobind_add_module)
    message(WARNING "CLIO_CORE_ENABLE_PYTHON is ON but nanobind_add_module command not found. Python bindings may fail.")
  else()
    message(STATUS "Python bindings available - nanobind configured by root CMakeLists.txt")
  endif()
endif()

message(STATUS "Found Chimaera framework - building full ChiMod implementation")

# Ensure CUDA language is enabled in this directory scope so that chimod
# subdirectories (core, compressor) can use add_cuda_library when
# CLIO_CORE_ENABLE_CUDA is ON.  enable_language() is idempotent; the
# architecture and flag variables set by wrp_core_enable_cuda() in
# context-transport-primitives are already in the CMake cache.
if(CLIO_CORE_ENABLE_CUDA)
    enable_language(CUDA)
endif()

# Add the core ChiMod
add_subdirectory(core)

# Add the compressor ChiMod (optional - only if compression is enabled)
if(CLIO_CTE_ENABLE_COMPRESS)
    message(STATUS "CTE: Compression enabled - building compressor chimod")
    add_subdirectory(compressor)
else()
    message(STATUS "CTE: Compression disabled - skipping compressor chimod (use -DWRP_CTE_ENABLE_COMPRESS=ON to enable)")
endif()

# Add the filesystem ChiMod (a filesystem interface over the CTE; the I/O
# interceptors delegate to it). Default OFF while scaffolding (#552) so the
# in-progress chimod can't break the default build; flip ON to develop it.
option(CLIO_CTE_ENABLE_FILESYSTEM "Build the filesystem chimod (WIP, #552)" OFF)
# The interceptor adapters (libfuse, POSIX, STDIO, MPI-IO) now delegate to the
# filesystem chimod (#552), so they require the chimod to be built. Force it on
# whenever any of them is enabled.
if((CLIO_CTE_ENABLE_FUSE_ADAPTER OR CLIO_CTE_ENABLE_POSIX_ADAPTER OR
    CLIO_CTE_ENABLE_STDIO_ADAPTER OR CLIO_CTE_ENABLE_MPIIO_ADAPTER OR
    CLIO_CTE_ENABLE_VFD)
   AND NOT CLIO_CTE_ENABLE_FILESYSTEM)
    message(STATUS "CTE: enabling filesystem chimod (required by the I/O adapters)")
    set(CLIO_CTE_ENABLE_FILESYSTEM ON CACHE BOOL
        "Build the filesystem chimod (WIP, #552)" FORCE)
endif()
if(CLIO_CTE_ENABLE_FILESYSTEM)
    message(STATUS "CTE: building filesystem chimod")
    add_subdirectory(filesystem)
else()
    message(STATUS "CTE: filesystem chimod disabled (use -DCLIO_CTE_ENABLE_FILESYSTEM=ON)")
endif()

# Add the bdev ChiMod (when it exists)
# add_subdirectory(chimods/bdev)

# Add adapters
add_subdirectory(adapter)

# Add UVM (GPU virtual memory manager) -- requires CUDA
add_subdirectory(uvm)

# Enable testing
# CTest is included by root CMakeLists.txt
include(CTest)

# Add tests if enabled
# Tests are controlled by CLIO_CORE_ENABLE_TESTS in root CMakeLists.txt
if(CLIO_CORE_ENABLE_TESTS)
  add_subdirectory(test)
  message(STATUS "CTE tests enabled - added test subdirectory")
endif()

# Add wrapper components (includes Python bindings if enabled)
add_subdirectory(wrapper)

# Add benchmarks if enabled
# Benchmarks are controlled by CLIO_CORE_ENABLE_BENCHMARKS
if(CLIO_CORE_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
  message(STATUS "CTE benchmarks enabled - added benchmark subdirectory")
endif()

message(STATUS "CTE Core and Bdev ChiMods configured for compilation")

if(CLIO_CORE_ENABLE_JARVIS)
    jarvis_repo_add(${CLIO_CTE_ROOT}/test/jarvis_clio_core)
endif()
