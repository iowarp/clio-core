# src/CMakeLists.txt
# Main Chimaera library source code

cmake_minimum_required(VERSION 3.10)

# Get all source files
file(GLOB_RECURSE CHIMAERA_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Create main library with namespace prefix to avoid conflicts
set(CHIMAERA_LIB_NAME "${CHIMAERA_NAMESPACE}_cxx")
add_library(${CHIMAERA_LIB_NAME} SHARED ${CHIMAERA_SOURCES})

# Set include directories as PUBLIC so they propagate to consumers
# Use CMAKE_CURRENT_SOURCE_DIR/.. to get runtime directory (this is runtime/src)
# Also include hermes_shm headers which are in context-transport-primitives/include
target_include_directories(${CHIMAERA_LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules/admin/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../context-transport-primitives/include>
  $<INSTALL_INTERFACE:include>
)

# Add runtime definition
# Removed CHIMAERA_RUNTIME compile definition - using runtime IsRuntime() checks instead

# Add CMAKE_INSTALL_PREFIX as a compile definition for ChiMod search paths
target_compile_definitions(${CHIMAERA_LIB_NAME} PRIVATE
  CHI_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
)

# Determine which HermesShm target to use (handles both installed and subdirectory modes)
# When used as subdirectory, prefer hermes_shm_host directly for proper include propagation
set(HSHM_TARGET "")
if(TARGET hermes_shm_host)
  # Building as subdirectory - use hermes_shm_host directly
  set(HSHM_TARGET hermes_shm_host)
  message(STATUS "Using HermesShm target: hermes_shm_host")
elseif(TARGET hshm::cxx)
  # Using installed package
  set(HSHM_TARGET hshm::cxx)
  message(STATUS "Using HermesShm target: hshm::cxx")
elseif(TARGET cxx)
  # Fallback to cxx
  set(HSHM_TARGET cxx)
  message(STATUS "Using HermesShm target: cxx")
else()
  message(FATAL_ERROR "No HermesShm target found")
endif()

# Link libraries
target_link_libraries(${CHIMAERA_LIB_NAME}
  PUBLIC
    ${HSHM_TARGET}
    cereal::cereal
    Boost::fiber
    Boost::context
    # chimods_admin_client  # Admin client library for global task routing - TODO: Enable when admin chimod is built
)

# Set target properties
set_target_properties(${CHIMAERA_LIB_NAME} PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER "${CHIMAERA_ROOT}/include/chimaera/chimaera.h"
  OUTPUT_NAME "${CHIMAERA_NAMESPACE}_cxx"
)

# Export target for parent scope
set_target_properties(${CHIMAERA_LIB_NAME} PROPERTIES EXPORT_NAME cxx)

# Create namespace alias for external consumption
add_library(${CHIMAERA_NAMESPACE}::cxx ALIAS ${CHIMAERA_LIB_NAME})