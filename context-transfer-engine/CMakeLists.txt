cmake_minimum_required(VERSION 3.20)
project(content_transfer_engine)

# Set root directory for this component
set(WRP_CTE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Python bindings are controlled by WRP_CORE_ENABLE_PYTHON in root CMakeLists.txt
# WRP_CTE_ENABLE_PYTHON is defined in root CMakeLists.txt
# CTE_ENABLE_ASAN is set by root CMakeLists.txt

# C++ standard, compile commands, RPATH, ASAN, and output directories are all set by root CMakeLists.txt

# CTE_ENABLE_CMAKE_DOTENV is set by root CMakeLists.txt
# DOTENV handling is done by root CMakeLists.txt

# CMake modules and ChiMod utilities are now provided by IowarpCoreCommon.cmake
# All find_package calls are handled by root CMakeLists.txt
# Targets chimaera::cxx, chimaera_admin_client, etc. are already available

# Python bindings configuration
# nanobind is configured by root CMakeLists.txt when WRP_CORE_ENABLE_PYTHON=ON
# No local nanobind setup needed - use the root configuration
if(WRP_CORE_ENABLE_PYTHON)
  if(NOT COMMAND nanobind_add_module)
    message(WARNING "WRP_CORE_ENABLE_PYTHON is ON but nanobind_add_module command not found. Python bindings may fail.")
  else()
    message(STATUS "Python bindings available - nanobind configured by root CMakeLists.txt")
  endif()
endif()

message(STATUS "Found Chimaera framework - building full ChiMod implementation")

# Add the core ChiMod
add_subdirectory(core)

# Add the bdev ChiMod (when it exists)
# add_subdirectory(chimods/bdev)

# Add adapters
add_subdirectory(adapter)

# Enable testing
# CTest is included by root CMakeLists.txt
include(CTest)

# Add tests if enabled
# WRP_CTE_ENABLE_TESTS is set by root CMakeLists.txt
if(WRP_CTE_ENABLE_TESTS)
  add_subdirectory(test)
  message(STATUS "CTE tests enabled - added test subdirectory")
endif()

# Add wrapper components (includes Python bindings if enabled)
add_subdirectory(wrapper)

# Add benchmarks if enabled
# WRP_CTE_ENABLE_BENCHMARKS is set by root CMakeLists.txt
if(WRP_CTE_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
  message(STATUS "CTE benchmarks enabled - added benchmark subdirectory")
endif()

message(STATUS "CTE Core and Bdev ChiMods configured for compilation")

if(WRP_CORE_ENABLE_JARVIS)
    jarvis_repo_add(${WRP_CTE_ROOT}/test/jarvis_iowarp)
endif()
