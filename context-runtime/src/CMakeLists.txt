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

# Link libraries
# Chimaera runtime links to hshm modular targets
# Clients only link to hshm::cxx
target_link_libraries(${CHIMAERA_LIB_NAME}
  PUBLIC
    hshm::cxx
    hshm::serialize  # For cereal
    Threads::Threads
    hshm::lightbeam  # For ZMQ transport
    # chimods_admin_client  # Admin client library for global task routing - TODO: Enable when admin chimod is built
)

# Set target properties
set_target_properties(${CHIMAERA_LIB_NAME} PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER "${CHIMAERA_ROOT}/include/chimaera/chimaera.h"
  OUTPUT_NAME "${CHIMAERA_NAMESPACE}_cxx"
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# Export target for parent scope
set_target_properties(${CHIMAERA_LIB_NAME} PROPERTIES EXPORT_NAME cxx)

# Create namespace alias for external consumption
add_library(${CHIMAERA_NAMESPACE}::cxx ALIAS ${CHIMAERA_LIB_NAME})