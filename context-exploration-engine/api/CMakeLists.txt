cmake_minimum_required(VERSION 3.20)
project(clio_cee_api VERSION 1.0.0)

#------------------------------------------------------------------------------
# Find Required Packages
#------------------------------------------------------------------------------
# All dependencies are provided by the unified build - no find_package needed
# chimaera::cxx, clio_cte_core_client, and clio_cae_core_client are already available

#------------------------------------------------------------------------------
# Context Interface Library
#------------------------------------------------------------------------------
add_library(clio_cee_api SHARED
  src/context_interface.cc
)

# Canonical alias matches the project's clio::<engine>:: scheme. The
# install(EXPORT ...) block below pins the exported name to `api` so
# downstream consumers see `clio::cee::api` after find_package().
add_library(clio::cee::api ALIAS clio_cee_api)

target_include_directories(clio_cee_api
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(clio_cee_api
  PUBLIC
    chimaera::cxx
    clio_cte_core_client
    clio_cae_core_client
)

# Set library properties
set_target_properties(clio_cee_api PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  # Drives the install-time exported name: combined with NAMESPACE clio::cee::
  # in install(EXPORT ...) it yields `clio::cee::api`.
  EXPORT_NAME api
  # Auto-export non-static symbols on Windows so the test binaries can
  # link against clio_cee_api.lib without each public function needing
  # an explicit __declspec(dllexport). Mirrors the same property on
  # every other shared target in this project.
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

#------------------------------------------------------------------------------
# Python Bindings (using nanobind)
#------------------------------------------------------------------------------
# Python bindings are controlled by CLIO_CORE_ENABLE_PYTHON in root CMakeLists.txt
# nanobind is configured by root CMakeLists.txt when CLIO_CORE_ENABLE_PYTHON=ON
if(CLIO_CORE_ENABLE_PYTHON)
  if(NOT COMMAND nanobind_add_module)
    message(WARNING "CLIO_CORE_ENABLE_PYTHON is ON but nanobind_add_module command not found. Python bindings will be skipped.")
  else()
    message(STATUS "Building CEE Python bindings - nanobind configured by root CMakeLists.txt")

    # Find Python for nanobind
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Create Python module
    nanobind_add_module(clio_cee
      src/python_bindings.cc
    )

    target_link_libraries(clio_cee
      PRIVATE
        clio_cee_api
    )

    # Install Python module to site-packages (PYTHON_SITE_PACKAGES set by root CMakeLists.txt)
    install(TARGETS clio_cee
      LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
    )
    message(STATUS "CEE Python wrapper will be installed to: ${PYTHON_SITE_PACKAGES}")
  endif()
else()
  message(STATUS "Python bindings disabled - set CLIO_CORE_ENABLE_PYTHON=ON to enable")
endif()

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------
install(TARGETS clio_cee_api
  EXPORT clio_cee_api_targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)

# Generate package config file with dependencies
include(CMakePackageConfigHelpers)

# Create the config file content
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/clio_cee_api-config.cmake.in"
"@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# Find required dependencies
find_dependency(chimaera REQUIRED)
find_dependency(clio_cte_core REQUIRED)
find_dependency(clio_cae_core REQUIRED)

# Include the exported targets
include(\"\${CMAKE_CURRENT_LIST_DIR}/clio_cee_api_targets.cmake\")

check_required_components(clio_cee_api)
")

configure_package_config_file(
  "${CMAKE_CURRENT_BINARY_DIR}/clio_cee_api-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/clio_cee_api-config.cmake"
  INSTALL_DESTINATION lib/cmake/clio_cee_api
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/clio_cee_api-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(EXPORT clio_cee_api_targets
  FILE clio_cee_api_targets.cmake
  NAMESPACE clio::cee::
  DESTINATION lib/cmake/clio_cee_api
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/clio_cee_api-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/clio_cee_api-config-version.cmake"
  DESTINATION lib/cmake/clio_cee_api
)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
