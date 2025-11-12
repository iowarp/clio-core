cmake_minimum_required(VERSION 3.25)
project(cae VERSION 1.0.0)
# CMAKE_CXX_STANDARD is set by root CMakeLists.txt

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
option(CAE_ENABLE_HDF5 "Enable HDF5 format support in CAE" ON)
option(CAE_ENABLE_GLOBUS "Enable Globus transfer support in CAE" OFF)
option(CAE_ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)

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

# Add coverage flags if enabled
if(CAE_ENABLE_COVERAGE)
    message(STATUS "Code coverage: ENABLED")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
else()
    message(STATUS "Code coverage: DISABLED")
endif()

# ------------------------------------------------------------------------------
# Setup install and output Directories
# ------------------------------------------------------------------------------
include(GNUInstallDirs)

if(NOT CAE_INSTALL_BIN_DIR)
    set(CAE_INSTALL_BIN_DIR ${CMAKE_INSTALL_BINDIR})
endif()

if(NOT CAE_INSTALL_LIB_DIR)
    set(CAE_INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR})
endif()

if(NOT CAE_INSTALL_INCLUDE_DIR)
    set(CAE_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(NOT CAE_INSTALL_DATA_DIR)
    set(CAE_INSTALL_DATA_DIR ${CMAKE_INSTALL_DATADIR})
endif()

# -----------------------------------------------------------------------------
# Find Packages
# -----------------------------------------------------------------------------

# CMAKE_EXPORT_COMPILE_COMMANDS and RPATH settings are set by root CMakeLists.txt
list(APPEND CMAKE_INSTALL_RPATH "${CAE_INSTALL_LIB_DIR}")

# Output directories are set by root CMakeLists.txt

# -----------------------------------------------------------------------------
# Build OMNI Module
# -----------------------------------------------------------------------------
# add_subdirectory(omni)

# -----------------------------------------------------------------------------
# Build ChiMods
# -----------------------------------------------------------------------------
add_subdirectory(core)
