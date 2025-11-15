cmake_minimum_required(VERSION 3.25)
project(cae VERSION 1.0.0)
# CMAKE_CXX_STANDARD is set by root CMakeLists.txt

# Set root directory for this component
set(WRP_CAE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# -----------------------------------------------------------------------------
# Set CAE Version Variables
# -----------------------------------------------------------------------------
set(CAE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CAE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CAE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# -----------------------------------------------------------------------------
# Define Options
# -----------------------------------------------------------------------------
# Enable testing by default (can be disabled with -DBUILD_TESTING=OFF)
# CTest is included by root CMakeLists.txt
include(CTest)

# CAE Feature Options - External Library Support
# WRP_CAE_ENABLE_HDF5 is set from WRP_CORE_ENABLE_HDF5 (defined in root CMakeLists.txt)
set(WRP_CAE_ENABLE_HDF5 ${WRP_CORE_ENABLE_HDF5} CACHE BOOL "Enable HDF5 format support in CAE (controlled by WRP_CORE_ENABLE_HDF5)" FORCE)
# CAE_ENABLE_GLOBUS is defined in root CMakeLists.txt

# Coverage is controlled by WRP_CORE_ENABLE_COVERAGE in root CMakeLists.txt
# Root CMakeLists already applies coverage flags globally via add_compile_options/add_link_options

# -----------------------------------------------------------------------------
# Compiler Optimization
# -----------------------------------------------------------------------------
# CMAKE_CXX_STANDARD and CMAKE_POSITION_INDEPENDENT_CODE are set by root CMakeLists.txt

# CAE-specific debug/release flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("IN DEBUG MODE")
    add_compile_definitions(CAE_LOG_VERBOSITY=10)
else()
    message("IN RELEASE MODE")
    add_compile_definitions(CAE_LOG_VERBOSITY=1)
endif()

# ------------------------------------------------------------------------------
# Setup install and output Directories
# ------------------------------------------------------------------------------
# Installation directories use relative paths for proper --prefix override support
# Removed CAE_INSTALL_*_DIR macros - use lib, include, bin, share directly

# -----------------------------------------------------------------------------
# Find Packages
# -----------------------------------------------------------------------------

# CMAKE_EXPORT_COMPILE_COMMANDS and RPATH settings are set by root CMakeLists.txt

# Output directories are set by root CMakeLists.txt

# -----------------------------------------------------------------------------
# Build OMNI Module
# -----------------------------------------------------------------------------
# add_subdirectory(omni)

# -----------------------------------------------------------------------------
# Build ChiMods
# -----------------------------------------------------------------------------
# Note: core/CMakeLists.txt handles adding test subdirectory based on BUILD_TESTING
add_subdirectory(core)

# -----------------------------------------------------------------------------
# Build Benchmarks
# -----------------------------------------------------------------------------
# Add benchmarks if enabled (currently CAE has no benchmark directory)
# WRP_CAE_ENABLE_BENCHMARKS is set by root CMakeLists.txt
# if(WRP_CAE_ENABLE_BENCHMARKS)
#   add_subdirectory(benchmark)
#   message(STATUS "CAE benchmarks enabled - added benchmark subdirectory")
# endif()
