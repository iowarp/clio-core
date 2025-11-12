cmake_minimum_required(VERSION 3.20)
project(content_transfer_engine)

# Python bindings are controlled by WRP_CORE_ENABLE_PYTHON in root CMakeLists.txt
# Component-specific option: only effective when WRP_CORE_ENABLE_PYTHON=ON
option(WRP_CTE_ENABLE_PYTHON "Enable Python bindings for CTE (requires WRP_CORE_ENABLE_PYTHON=ON)" ON)
# CTE_ENABLE_ASAN is set by root CMakeLists.txt

# C++ standard, compile commands, RPATH, ASAN, and output directories are all set by root CMakeLists.txt

# CTE_ENABLE_CMAKE_DOTENV is set by root CMakeLists.txt
# DOTENV handling is done by root CMakeLists.txt

# Add local cmake modules to path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find required Chimaera framework packages following MODULE_DEVELOPMENT_GUIDE.md patterns
# For internal ChiMod development within the main repository, we need the core framework
# Skip find_package if targets already exist (unified build as subdirectory)
if(NOT TARGET chimaera::cxx AND NOT TARGET chimaera_cxx)
  find_package(chimaera REQUIRED)              # Core Chimaera (automatically includes ChimaeraCommon.cmake)
  message(STATUS "Found Chimaera package: ${chimaera_FOUND}")
else()
  # When built as subdirectory, manually include ChimaeraCommon.cmake
  message(STATUS "Using Chimaera as subdirectory - including ChimaeraCommon.cmake")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/cmake")
  include(ChimaeraCommon)
endif()

# Determine which HermesShm target to use (for both installed and subdirectory modes)
include(FindHSHMTarget)

if(NOT TARGET chimaera::admin_client AND NOT TARGET chimaera_admin_client)
  find_package(chimaera_admin REQUIRED)        # Admin ChiMod (often required for ChiMod operations)
  message(STATUS "Found chimaera_admin package: ${chimaera_admin_FOUND}")
else()
  message(STATUS "Using chimaera_admin as subdirectory")
endif()

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

# Add tests (skip external tests during internal development)
add_subdirectory(test)

# Add wrapper components (includes Python bindings if enabled)
add_subdirectory(wrapper)

# Add benchmarks
add_subdirectory(benchmark)

message(STATUS "CTE Core and Bdev ChiMods configured for compilation")

jarvis_repo_add(${CMAKE_SOURCE_DIR}/test/jarvis_wrp_cte "")
