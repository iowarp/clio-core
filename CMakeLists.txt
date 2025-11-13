cmake_minimum_required(VERSION 3.20)
project(iowarp-core VERSION 1.0.0 LANGUAGES C CXX)

#------------------------------------------------------------------------------
# Project Options
#------------------------------------------------------------------------------
# Build configuration
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)

# Component enable/disable
option(WRP_CORE_ENABLE_RUNTIME "Enable runtime component" ON)
option(WRP_CORE_ENABLE_CTE "Enable context-transfer-engine component" ON)
option(WRP_CORE_ENABLE_CAE "Enable context-assimilation-engine component" ON)
option(WRP_CORE_ENABLE_CEE "Enable context-exploration-engine component" ON)

# Global options
option(WRP_CORE_ENABLE_TESTS "Enable testing for all components" OFF)
option(WRP_CORE_ENABLE_CMAKE_DOTENV "Enable reading environment variables from .env.cmake" ON)
option(WRP_CORE_ENABLE_PYTHON "Enable Python bindings for all components" OFF)
option(WRP_CORE_ENABLE_ASAN "Enable AddressSanitizer for debugging" OFF)
option(WRP_CORE_ENABLE_COVERAGE "Enable code coverage instrumentation for all components" OFF)

# Automatically set component test/benchmark variables based on global flags and component enable status
# These are variables, not options - they're derived from WRP_CORE_ENABLE_TESTS and WRP_CORE_ENABLE_[COMPONENT]
# Tests are enabled if both the global test flag AND the component are enabled
set(HSHM_ENABLE_TESTS ${WRP_CORE_ENABLE_TESTS})
set(CHIMAERA_ENABLE_TESTS ${WRP_CORE_ENABLE_TESTS})

# CTE tests: enabled if global tests ON and CTE component ON
if(WRP_CORE_ENABLE_TESTS AND WRP_CORE_ENABLE_CTE)
    set(WRP_CTE_ENABLE_TESTS ON)
else()
    set(WRP_CTE_ENABLE_TESTS OFF)
endif()

# CAE tests: enabled if global tests ON and CAE component ON
if(WRP_CORE_ENABLE_TESTS AND WRP_CORE_ENABLE_CAE)
    set(WRP_CAE_ENABLE_TESTS ON)
else()
    set(WRP_CAE_ENABLE_TESTS OFF)
endif()

# CEE tests: enabled if global tests ON and CEE component ON
if(WRP_CORE_ENABLE_TESTS AND WRP_CORE_ENABLE_CEE)
    set(WRP_CEE_ENABLE_TESTS ON)
else()
    set(WRP_CEE_ENABLE_TESTS OFF)
endif()

# Benchmarks (currently always OFF, but follow same pattern for consistency)
set(HSHM_ENABLE_BENCHMARKS OFF)
set(CHIMAERA_ENABLE_BENCHMARKS OFF)
set(WRP_CTE_ENABLE_BENCHMARKS OFF)

# Read environment variables from .env.cmake if enabled
if(WRP_CORE_ENABLE_CMAKE_DOTENV)
  if(EXISTS "${CMAKE_SOURCE_DIR}/.env.cmake")
    include("${CMAKE_SOURCE_DIR}/.env.cmake")
  endif()
endif()

#------------------------------------------------------------------------------
# Python Bindings Configuration
#------------------------------------------------------------------------------
if(WRP_CORE_ENABLE_PYTHON)
  message(STATUS "Python bindings enabled - configuring nanobind")

  # Include helper function for finding Python from HSHM
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/context-transport-primitives/cmake")
  include(HermesShmCommonConfig)

  # Find Python executable using HSHM helper
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

# Enable RPATH
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

# Add cereal header-only library
message(STATUS "Building cereal library (header-only)")
set(JUST_INSTALL_CEREAL ON CACHE BOOL "Just install cereal headers" FORCE)
set(BUILD_DOC OFF CACHE BOOL "Disable cereal documentation" FORCE)
set(BUILD_SANDBOX OFF CACHE BOOL "Disable cereal sandbox" FORCE)
set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "Skip cereal performance comparison" FORCE)
add_subdirectory(external/cereal)

# Add yaml-cpp library
message(STATUS "Building yaml-cpp library")
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib" FORCE)
set(YAML_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL "Build yaml-cpp as shared library" FORCE)
add_subdirectory(external/yaml-cpp)

# Context Transport Primitives (formerly cte-hermes-shm) - Required by runtime
message(STATUS "Building context-transport-primitives component")
# Enable ELF support for adapter real API functionality
set(HSHM_ENABLE_ELF ON CACHE BOOL "Enable ELF support for adapters" FORCE)
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
