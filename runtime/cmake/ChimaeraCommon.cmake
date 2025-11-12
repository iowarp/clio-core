# ChimaeraCommon.cmake - Core shared CMake functionality for Chimaera

# Guard against multiple inclusions
if(CHIMAERA_COMMON_INCLUDED)
  return()
endif()
set(CHIMAERA_COMMON_INCLUDED TRUE)

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------

# Find HermesShm - skip if already available as subdirectory
# Check for both the namespaced alias (hshm::cxx) and the raw target (cxx)
if(NOT TARGET hshm::cxx AND NOT TARGET cxx)
  find_package(HermesShm CONFIG REQUIRED)
endif()

# Verify HSHM target exists (alias is created in context-transport-primitives/src/CMakeLists.txt)
if(TARGET hshm::cxx)
  message(STATUS "Using HermesShm target: hshm::cxx")
elseif(TARGET cxx)
  message(STATUS "Using HermesShm target: cxx")
else()
  message(FATAL_ERROR "Neither hshm::cxx nor cxx target found")
endif()

# Find Boost components
find_package(Boost REQUIRED COMPONENTS fiber context system)

# Find cereal
find_package(cereal REQUIRED)

# Find MPI (optional)
find_package(MPI QUIET)

# Thread support
find_package(Threads REQUIRED)

#------------------------------------------------------------------------------
# Common compile definitions and flags
#------------------------------------------------------------------------------

# Set common compile features
set(CHIMAERA_CXX_STANDARD 17)

# Common compile definitions
set(CHIMAERA_COMMON_COMPILE_DEFS
  $<$<CONFIG:Debug>:DEBUG>
  $<$<CONFIG:Release>:NDEBUG>
)

# Common include directories
set(CHIMAERA_COMMON_INCLUDES
  ${Boost_INCLUDE_DIRS}
  ${cereal_INCLUDE_DIRS}
)

# Common link libraries
set(CHIMAERA_COMMON_LIBS
  Boost::fiber
  Boost::context
  Boost::system
  Threads::Threads
)

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

# Helper function to link runtime to client library (called via DEFER)
# This allows linking to work regardless of which target is defined first
function(_chimaera_link_runtime_to_client RUNTIME_TARGET CLIENT_TARGET)
  if(TARGET ${CLIENT_TARGET})
    target_link_libraries(${RUNTIME_TARGET} PUBLIC ${CLIENT_TARGET})
    message(STATUS "Deferred linking: Runtime ${RUNTIME_TARGET} linked to client ${CLIENT_TARGET}")
  endif()
endfunction()

#------------------------------------------------------------------------------
# Module configuration parsing
#------------------------------------------------------------------------------

# Function to read repository namespace from chimaera_repo.yaml
# Searches up the directory tree from the given path to find chimaera_repo.yaml
function(read_repo_namespace output_var start_path)
  set(current_path "${start_path}")
  set(namespace "chimaera")  # Default fallback
  
  # Search up the directory tree for chimaera_repo.yaml
  while(NOT "${current_path}" STREQUAL "/" AND NOT "${current_path}" STREQUAL "")
    set(repo_file "${current_path}/chimaera_repo.yaml")
    if(EXISTS "${repo_file}")
      # Read and parse the YAML file
      file(READ "${repo_file}" REPO_YAML_CONTENT)
      string(REGEX MATCH "namespace: *([^\n\r]+)" NAMESPACE_MATCH "${REPO_YAML_CONTENT}")
      if(NAMESPACE_MATCH)
        string(REGEX REPLACE "namespace: *" "" namespace "${NAMESPACE_MATCH}")
        string(STRIP "${namespace}" namespace)
        break()
      endif()
    endif()
    
    # Move up one directory
    get_filename_component(current_path "${current_path}" DIRECTORY)
  endwhile()
  
  set(${output_var} "${namespace}" PARENT_SCOPE)
endfunction()

# Function to read module configuration from chimaera_mod.yaml
function(chimaera_read_module_config MODULE_DIR)
  set(CONFIG_FILE "${MODULE_DIR}/chimaera_mod.yaml")
  
  if(NOT EXISTS ${CONFIG_FILE})
    message(FATAL_ERROR "Missing chimaera_mod.yaml in ${MODULE_DIR}")
  endif()
  
  # Parse YAML file (simple regex parsing for key: value pairs)
  file(READ ${CONFIG_FILE} CONFIG_CONTENT)
  
  # Extract module_name
  string(REGEX MATCH "module_name:[ ]*([^\n\r]*)" MODULE_MATCH ${CONFIG_CONTENT})
  if(MODULE_MATCH)
    string(REGEX REPLACE "module_name:[ ]*" "" CHIMAERA_MODULE_NAME "${MODULE_MATCH}")
    string(STRIP "${CHIMAERA_MODULE_NAME}" CHIMAERA_MODULE_NAME)
  endif()
  set(CHIMAERA_MODULE_NAME ${CHIMAERA_MODULE_NAME} PARENT_SCOPE)
  
  # Extract namespace
  string(REGEX MATCH "namespace:[ ]*([^\n\r]*)" NAMESPACE_MATCH ${CONFIG_CONTENT})
  if(NAMESPACE_MATCH)
    string(REGEX REPLACE "namespace:[ ]*" "" CHIMAERA_NAMESPACE "${NAMESPACE_MATCH}")
    string(STRIP "${CHIMAERA_NAMESPACE}" CHIMAERA_NAMESPACE)
  endif()
  set(CHIMAERA_NAMESPACE ${CHIMAERA_NAMESPACE} PARENT_SCOPE)
  
  # Validate extracted values
  if(NOT CHIMAERA_MODULE_NAME)
    message(FATAL_ERROR "module_name not found in ${CONFIG_FILE}. Content preview: ${CONFIG_CONTENT}")
  endif()
  
  if(NOT CHIMAERA_NAMESPACE)
    message(FATAL_ERROR "namespace not found in ${CONFIG_FILE}. Content preview: ${CONFIG_CONTENT}")
  endif()
endfunction()

#------------------------------------------------------------------------------
# ChiMod Client Library Function
#------------------------------------------------------------------------------

# add_chimod_client - Create a ChiMod client library
#
# Parameters:
#   SOURCES             - Source files for the client library
#   COMPILE_DEFINITIONS - Additional compile definitions
#   LINK_LIBRARIES      - Additional libraries to link
#   LINK_DIRECTORIES    - Additional link directories
#   INCLUDE_LIBRARIES   - Libraries whose includes should be added
#   INCLUDE_DIRECTORIES - Additional include directories
#
function(add_chimod_client)
  cmake_parse_arguments(
    ARG
    ""
    ""
    "SOURCES;COMPILE_DEFINITIONS;LINK_LIBRARIES;LINK_DIRECTORIES;INCLUDE_LIBRARIES;INCLUDE_DIRECTORIES"
    ${ARGN}
  )
  
  # Read module configuration
  chimaera_read_module_config(${CMAKE_CURRENT_SOURCE_DIR})
  
  # Create target name
  set(TARGET_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_client")
  
  # Create the library
  add_library(${TARGET_NAME} SHARED ${ARG_SOURCES})
  
  # Set C++ standard
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CHIMAERA_CXX_STANDARD})
  
  # Add compile definitions
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      ${CHIMAERA_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )
  
  # Add include directories
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${CHIMAERA_COMMON_INCLUDES}
  )
  
  # Add additional include directories with BUILD_INTERFACE wrapper
  foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRECTORIES})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${INCLUDE_DIR}>
    )
  endforeach()
  
  # Add include directories from INCLUDE_LIBRARIES
  foreach(LIB ${ARG_INCLUDE_LIBRARIES})
    get_target_property(LIB_INCLUDES ${LIB} INTERFACE_INCLUDE_DIRECTORIES)
    if(LIB_INCLUDES)
      target_include_directories(${TARGET_NAME} PUBLIC ${LIB_INCLUDES})
    endif()
  endforeach()
  
  # Add link directories
  if(ARG_LINK_DIRECTORIES)
    target_link_directories(${TARGET_NAME} PUBLIC ${ARG_LINK_DIRECTORIES})
  endif()
  
  # Link libraries - use hermes_shm::cxx for internal builds, chimaera::cxx for external
  set(CORE_LIB "")
  if(TARGET chimaera::cxx)
    set(CORE_LIB chimaera::cxx)
  elseif(TARGET hermes_shm::cxx)
    set(CORE_LIB hermes_shm::cxx)
  elseif(TARGET HermesShm::cxx)
    set(CORE_LIB HermesShm::cxx)
  elseif(TARGET cxx)
    set(CORE_LIB cxx)
  else()
    message(FATAL_ERROR "Neither chimaera::cxx, hermes_shm::cxx, HermesShm::cxx nor cxx target found")
  endif()
  
  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${CORE_LIB}
      ${CHIMAERA_COMMON_LIBS}
      ${ARG_LINK_LIBRARIES}
  )
  
  # Create alias for external use
  add_library(${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}_client ALIAS ${TARGET_NAME})
  
  # Set properties for installation
  set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME "${CHIMAERA_MODULE_NAME}_client"
    OUTPUT_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_client"
  )
  
  # Install the client library
  set(MODULE_PACKAGE_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}")
  set(MODULE_EXPORT_NAME "${MODULE_PACKAGE_NAME}")
  
  install(TARGETS ${TARGET_NAME}
    EXPORT ${MODULE_EXPORT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )
  
  # Install headers
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      DESTINATION include
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
  endif()
  
  # Export module info to parent scope
  set(CHIMAERA_MODULE_CLIENT_TARGET ${TARGET_NAME} PARENT_SCOPE)
  set(CHIMAERA_MODULE_NAME ${CHIMAERA_MODULE_NAME} PARENT_SCOPE)
  set(CHIMAERA_NAMESPACE ${CHIMAERA_NAMESPACE} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# ChiMod Runtime Library Function
#------------------------------------------------------------------------------

# add_chimod_runtime - Create a ChiMod runtime library
#
# Parameters:
#   SOURCES             - Source files for the runtime library
#   COMPILE_DEFINITIONS - Additional compile definitions
#   LINK_LIBRARIES      - Additional libraries to link
#   LINK_DIRECTORIES    - Additional link directories
#   INCLUDE_LIBRARIES   - Libraries whose includes should be added
#   INCLUDE_DIRECTORIES - Additional include directories
#
function(add_chimod_runtime)
  cmake_parse_arguments(
    ARG
    ""
    ""
    "SOURCES;COMPILE_DEFINITIONS;LINK_LIBRARIES;LINK_DIRECTORIES;INCLUDE_LIBRARIES;INCLUDE_DIRECTORIES"
    ${ARGN}
  )
  
  # Read module configuration
  chimaera_read_module_config(${CMAKE_CURRENT_SOURCE_DIR})
  
  # Create target name
  set(TARGET_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_runtime")
  
  # Create the library
  add_library(${TARGET_NAME} SHARED ${ARG_SOURCES})
  
  # Set C++ standard
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CHIMAERA_CXX_STANDARD})
  
  # Add compile definitions (runtime always has CHIMAERA_RUNTIME=1)
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      CHIMAERA_RUNTIME=1
      ${CHIMAERA_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )
  
  # Add include directories
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${CHIMAERA_COMMON_INCLUDES}
  )
  
  # Add additional include directories with BUILD_INTERFACE wrapper
  foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRECTORIES})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${INCLUDE_DIR}>
    )
  endforeach()
  
  # Add include directories from INCLUDE_LIBRARIES
  foreach(LIB ${ARG_INCLUDE_LIBRARIES})
    get_target_property(LIB_INCLUDES ${LIB} INTERFACE_INCLUDE_DIRECTORIES)
    if(LIB_INCLUDES)
      target_include_directories(${TARGET_NAME} PUBLIC ${LIB_INCLUDES})
    endif()
  endforeach()
  
  # Add link directories
  if(ARG_LINK_DIRECTORIES)
    target_link_directories(${TARGET_NAME} PUBLIC ${ARG_LINK_DIRECTORIES})
  endif()
  
  # Link libraries - use hermes_shm::cxx for internal builds, chimaera::cxx for external
  set(CORE_LIB "")
  if(TARGET chimaera::cxx)
    set(CORE_LIB chimaera::cxx)
  elseif(TARGET hermes_shm::cxx)
    set(CORE_LIB hermes_shm::cxx)
  elseif(TARGET HermesShm::cxx)
    set(CORE_LIB HermesShm::cxx)
  elseif(TARGET cxx)
    set(CORE_LIB cxx)
  else()
    message(FATAL_ERROR "Neither chimaera::cxx, hermes_shm::cxx, HermesShm::cxx nor cxx target found")
  endif()

  # Automatically link to client library if it exists
  set(RUNTIME_LINK_LIBS ${CORE_LIB} ${CHIMAERA_COMMON_LIBS} ${ARG_LINK_LIBRARIES})

  # Try to find client target by name (handles cases where client was defined first)
  set(CLIENT_TARGET_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_client")
  if(TARGET ${CLIENT_TARGET_NAME})
    list(APPEND RUNTIME_LINK_LIBS ${CLIENT_TARGET_NAME})
    message(STATUS "Runtime ${TARGET_NAME} linking to client ${CLIENT_TARGET_NAME}")
  elseif(CHIMAERA_MODULE_CLIENT_TARGET AND TARGET ${CHIMAERA_MODULE_CLIENT_TARGET})
    # Fallback to variable-based approach for compatibility
    list(APPEND RUNTIME_LINK_LIBS ${CHIMAERA_MODULE_CLIENT_TARGET})
    message(STATUS "Runtime ${TARGET_NAME} linking to client ${CHIMAERA_MODULE_CLIENT_TARGET}")
  endif()
  
  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${RUNTIME_LINK_LIBS}
    PRIVATE
      rt  # POSIX real-time library for async I/O
  )
  
  # Create alias for external use
  add_library(${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}_runtime ALIAS ${TARGET_NAME})

  # Set properties for installation
  set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME "${CHIMAERA_MODULE_NAME}_runtime"
    OUTPUT_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}_runtime"
  )

  # Use cmake_language(DEFER) to link to client after all targets are processed
  # This works regardless of whether runtime or client is defined first
  # Use EVAL CODE to properly capture variable values
  cmake_language(EVAL CODE "
    cmake_language(DEFER CALL _chimaera_link_runtime_to_client \"${TARGET_NAME}\" \"${CLIENT_TARGET_NAME}\")
  ")
  
  # Install the runtime library (add to existing export set if client exists)
  set(MODULE_PACKAGE_NAME "${CHIMAERA_NAMESPACE}_${CHIMAERA_MODULE_NAME}")
  set(MODULE_EXPORT_NAME "${MODULE_PACKAGE_NAME}")
  
  install(TARGETS ${TARGET_NAME}
    EXPORT ${MODULE_EXPORT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )
  
  # Install headers (only if not already installed by client)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include" AND NOT CHIMAERA_MODULE_CLIENT_TARGET)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      DESTINATION include
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
  endif()
  
  # Generate and install package config files (only do this once per module)
  # Check if both client and runtime exist, or if this is runtime-only
  set(SHOULD_GENERATE_CONFIG FALSE)
  if(CHIMAERA_MODULE_CLIENT_TARGET AND TARGET ${CHIMAERA_MODULE_CLIENT_TARGET})
    # Both client and runtime exist, generate config
    set(SHOULD_GENERATE_CONFIG TRUE)
  elseif(NOT CHIMAERA_MODULE_CLIENT_TARGET)
    # Runtime-only module, generate config
    set(SHOULD_GENERATE_CONFIG TRUE)
  endif()
  
  if(SHOULD_GENERATE_CONFIG)
    # Export targets file
    install(EXPORT ${MODULE_EXPORT_NAME}
      FILE ${MODULE_EXPORT_NAME}.cmake
      NAMESPACE ${CHIMAERA_NAMESPACE}::
      DESTINATION lib/cmake/${MODULE_PACKAGE_NAME}
    )

    # Generate Config.cmake file
    set(CONFIG_CONTENT "
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# Find the core Chimaera package (handles all other dependencies)
find_dependency(chimaera REQUIRED)

# Include the exported targets
include(\"\${CMAKE_CURRENT_LIST_DIR}/${MODULE_EXPORT_NAME}.cmake\")

# Provide components
check_required_components(${MODULE_PACKAGE_NAME})
")

    # Write Config.cmake template
    set(CONFIG_IN_FILE "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}Config.cmake.in")
    file(WRITE "${CONFIG_IN_FILE}" "${CONFIG_CONTENT}")

    # Configure and install Config.cmake
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
      "${CONFIG_IN_FILE}"
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}Config.cmake"
      INSTALL_DESTINATION lib/cmake/${MODULE_PACKAGE_NAME}
    )

    # Generate ConfigVersion.cmake
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}ConfigVersion.cmake"
      VERSION 1.0.0
      COMPATIBILITY SameMajorVersion
    )

    # Install Config and ConfigVersion files
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}Config.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}ConfigVersion.cmake"
      DESTINATION lib/cmake/${MODULE_PACKAGE_NAME}
    )
    
    # Collect targets for status message
    set(INSTALLED_TARGETS ${TARGET_NAME})
    if(CHIMAERA_MODULE_CLIENT_TARGET AND TARGET ${CHIMAERA_MODULE_CLIENT_TARGET})
      list(APPEND INSTALLED_TARGETS ${CHIMAERA_MODULE_CLIENT_TARGET})
    endif()
    
    message(STATUS "Created module package: ${MODULE_PACKAGE_NAME}")
    message(STATUS "  Targets: ${INSTALLED_TARGETS}")
    message(STATUS "  Aliases: ${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}_client, ${CHIMAERA_NAMESPACE}::${CHIMAERA_MODULE_NAME}_runtime")
  endif()
  
  # Export module info to parent scope
  set(CHIMAERA_MODULE_RUNTIME_TARGET ${TARGET_NAME} PARENT_SCOPE)
  set(CHIMAERA_MODULE_NAME ${CHIMAERA_MODULE_NAME} PARENT_SCOPE)
  set(CHIMAERA_NAMESPACE ${CHIMAERA_NAMESPACE} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# Installation is now handled automatically within add_chimod_client/runtime
#------------------------------------------------------------------------------
