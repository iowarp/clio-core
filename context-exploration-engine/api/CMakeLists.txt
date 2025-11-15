cmake_minimum_required(VERSION 3.20)
project(wrp_cee_api VERSION 1.0.0)

#------------------------------------------------------------------------------
# Find Required Packages
#------------------------------------------------------------------------------
# All dependencies are provided by the unified build - no find_package needed
# chimaera::cxx, wrp_cte_core_client, and wrp_cae_core_client are already available

#------------------------------------------------------------------------------
# Context Interface Library
#------------------------------------------------------------------------------
add_library(wrp_cee_api SHARED
  src/context_interface.cc
)

add_library(wrp_cee::api ALIAS wrp_cee_api)

target_include_directories(wrp_cee_api
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(wrp_cee_api
  PUBLIC
    chimaera::cxx
    wrp_cte_core_client
    wrp_cae_core_client
)

# Set library properties
set_target_properties(wrp_cee_api PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

#------------------------------------------------------------------------------
# Python Bindings (using nanobind)
#------------------------------------------------------------------------------
# Python bindings are controlled by WRP_CORE_ENABLE_PYTHON in root CMakeLists.txt
# nanobind is configured by root CMakeLists.txt when WRP_CORE_ENABLE_PYTHON=ON
if(WRP_CORE_ENABLE_PYTHON)
  if(NOT COMMAND nanobind_add_module)
    message(WARNING "WRP_CORE_ENABLE_PYTHON is ON but nanobind_add_module command not found. Python bindings will be skipped.")
  else()
    message(STATUS "Building CEE Python bindings - nanobind configured by root CMakeLists.txt")

    # Find Python for nanobind
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Create Python module
    nanobind_add_module(wrp_cee
      src/python_bindings.cc
    )

    target_link_libraries(wrp_cee
      PRIVATE
        wrp_cee_api
    )

    # Install Python module to the appropriate location
    # If WRP_CORE_INSTALL_WRAPPER_TO_VENV is ON, install to Python site-packages
    # Otherwise, install to lib/ directory (for system-wide installations)
    if(WRP_CORE_INSTALL_WRAPPER_TO_VENV)
      install(TARGETS wrp_cee
        LIBRARY DESTINATION ${Python_SITEARCH}
      )
      message(STATUS "CEE Python wrapper will be installed to: ${Python_SITEARCH}")
    else()
      install(TARGETS wrp_cee
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
      )
      message(STATUS "CEE Python wrapper will be installed to: ${CMAKE_INSTALL_PREFIX}/lib")
    endif()
  endif()
else()
  message(STATUS "Python bindings disabled - set WRP_CORE_ENABLE_PYTHON=ON to enable")
endif()

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------
install(TARGETS wrp_cee_api
  EXPORT wrp_cee_api_targets
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
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config.cmake.in"
"@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# Find required dependencies
find_dependency(chimaera REQUIRED)
find_dependency(wrp_cte_core REQUIRED)
find_dependency(wrp_cae_core REQUIRED)

# Include the exported targets
include(\"\${CMAKE_CURRENT_LIST_DIR}/wrp_cee_api_targets.cmake\")

check_required_components(wrp_cee_api)
")

configure_package_config_file(
  "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config.cmake"
  INSTALL_DESTINATION lib/cmake/wrp_cee_api
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(EXPORT wrp_cee_api_targets
  FILE wrp_cee_api_targets.cmake
  NAMESPACE wrp_cee::
  DESTINATION lib/cmake/wrp_cee_api
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/wrp_cee_api-config-version.cmake"
  DESTINATION lib/cmake/wrp_cee_api
)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
