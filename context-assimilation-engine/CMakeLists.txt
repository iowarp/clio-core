cmake_minimum_required(VERSION 3.25)
project(cae VERSION 1.0.0)
# CMAKE_CXX_STANDARD is set by root CMakeLists.txt

# Set root directory for this component
set(CLIO_CAE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

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
# CLIO_CAE_ENABLE_HDF5 is set from CLIO_CORE_ENABLE_HDF5 (defined in root CMakeLists.txt)
set(CLIO_CAE_ENABLE_HDF5 ${CLIO_CORE_ENABLE_HDF5} CACHE BOOL "Enable HDF5 format support in CAE (controlled by CLIO_CORE_ENABLE_HDF5)" FORCE)
# CAE_ENABLE_GLOBUS is defined in root CMakeLists.txt

# Coverage is controlled by CLIO_CORE_ENABLE_COVERAGE in root CMakeLists.txt
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
# Optional inference support (libcurl + nlohmann/json → OllamaGenerate)
# -----------------------------------------------------------------------------
# Detected HERE, above both chimods, because the summarizer chimod compiles
# against it AND the unit-test tree gates on it -- and those tests are added
# from core/CMakeLists.txt, a sibling scope that cannot see variables set in
# summarizer/CMakeLists.txt.
#
# nlohmann_json is always available (found or FetchContent'd by root). CURL is
# a system package; if missing, the summarizer's label_client compiles as a
# stub that returns false -- the module still builds and forwards, just
# without summarization.
find_package(CURL QUIET)
if(CURL_FOUND AND nlohmann_json_FOUND)
  message(STATUS "CAE summarization support: ENABLED (libcurl + nlohmann_json found)")
  set(CAE_SUMMARIZATION_ENABLED TRUE)
else()
  if(NOT CURL_FOUND)
    message(STATUS "CAE summarization support: DISABLED (libcurl not found)")
  else()
    message(STATUS "CAE summarization support: DISABLED (nlohmann_json not found)")
  endif()
  set(CAE_SUMMARIZATION_ENABLED FALSE)
endif()

# -----------------------------------------------------------------------------
# Build ChiMods
# -----------------------------------------------------------------------------
# Note: core/CMakeLists.txt handles adding test subdirectory based on BUILD_TESTING
add_subdirectory(core)

# Summarizer: the transparent LLM summarization interposer split out of core.
# Composed BETWEEN a CAE/CTE client and the CTE core; see its clio_mod.yaml.
add_subdirectory(summarizer)

# -----------------------------------------------------------------------------
# Build Benchmarks
# -----------------------------------------------------------------------------
# Benchmarks are controlled by CLIO_CORE_ENABLE_BENCHMARKS (set in root
# CMakeLists.txt, default ON). Currently provides clio_cae_label_bench
# which stresses the transparent-labeling PutBlob path.
if(CLIO_CORE_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
  message(STATUS "CAE benchmarks enabled - added benchmark subdirectory")
endif()
