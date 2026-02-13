# IowarpCoreCommon.cmake - Common CMake functions for IOWarp Core and external repos
#
# This file provides shared utilities for both the IOWarp Core build and external
# repositories that depend on it.

# Guard against multiple inclusions
if(IOWARP_CORE_COMMON_INCLUDED)
  return()
endif()
set(IOWARP_CORE_COMMON_INCLUDED TRUE)

message(STATUS "Loading IowarpCoreCommon.cmake")

#------------------------------------------------------------------------------
# Dependency Target Resolution
#------------------------------------------------------------------------------

# Macro to resolve yaml-cpp target name across different versions
# Older versions use "yaml-cpp", newer versions use "yaml-cpp::yaml-cpp"
macro(resolve_yaml_cpp_target)
  if(NOT DEFINED YAML_CPP_LIBS)
    if(TARGET yaml-cpp::yaml-cpp)
      set(YAML_CPP_LIBS yaml-cpp::yaml-cpp)
      message(STATUS "Using yaml-cpp target: yaml-cpp::yaml-cpp")
    elseif(TARGET yaml-cpp)
      set(YAML_CPP_LIBS yaml-cpp)
      message(STATUS "Using yaml-cpp target: yaml-cpp")
    else()
      message(FATAL_ERROR "yaml-cpp target not found. Expected either 'yaml-cpp::yaml-cpp' or 'yaml-cpp'")
    endif()
  endif()
endmacro()

#------------------------------------------------------------------------------
# GPU Support Functions
#------------------------------------------------------------------------------

# Enable cuda boilerplate
macro(wrp_core_enable_cuda CXX_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    message(STATUS "USING CUDA ARCH: ${CMAKE_CUDA_ARCHITECTURES}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler -diag-suppress=177,20014,20011,20012")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-format,-Wno-pedantic,-Wno-sign-compare,-Wno-unused-but-set-variable")
    enable_language(CUDA)

    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
endmacro()

# Enable rocm boilerplate
macro(wrp_core_enable_rocm GPU_RUNTIME CXX_STANDARD)
    set(GPU_RUNTIME ${GPU_RUNTIME})
    enable_language(${GPU_RUNTIME})
    set(CMAKE_${GPU_RUNTIME}_STANDARD ${CXX_STANDARD})
    set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
    set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )

    if(GPU_RUNTIME STREQUAL "CUDA")
        include_directories("${ROCM_ROOT}/include")
    endif()

    if(NOT HIP_FOUND)
        find_package(HIP REQUIRED)
    endif()
endmacro()

# Function for setting source files for rocm
function(set_rocm_sources MODE DO_COPY SRC_FILES ROCM_SOURCE_FILES_VAR)
    set(ROCM_SOURCE_FILES ${${ROCM_SOURCE_FILES_VAR}} PARENT_SCOPE)
    set(GPU_RUNTIME ${GPU_RUNTIME} PARENT_SCOPE)

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            set(ROCM_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/rocm_${MODE}/${SOURCE})
            configure_file(${SOURCE} ${ROCM_SOURCE} COPYONLY)
        else()
            set(ROCM_SOURCE ${SOURCE})
        endif()

        list(APPEND ROCM_SOURCE_FILES ${ROCM_SOURCE})
        set_source_files_properties(${ROCM_SOURCE} PROPERTIES LANGUAGE ${GPU_RUNTIME})
    endforeach()

    set(${ROCM_SOURCE_FILES_VAR} ${ROCM_SOURCE_FILES} PARENT_SCOPE)
endfunction()

# Function for setting source files for cuda
function(set_cuda_sources DO_COPY SRC_FILES CUDA_SOURCE_FILES_VAR)
    set(CUDA_SOURCE_FILES ${${CUDA_SOURCE_FILES_VAR}} PARENT_SCOPE)

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            set(CUDA_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/cuda/${SOURCE})
            configure_file(${SOURCE} ${CUDA_SOURCE} COPYONLY)
        else()
            set(CUDA_SOURCE ${SOURCE})
        endif()

        list(APPEND CUDA_SOURCE_FILES ${CUDA_SOURCE})
        set_source_files_properties(${CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)
    endforeach()

    set(${CUDA_SOURCE_FILES_VAR} ${CUDA_SOURCE_FILES} PARENT_SCOPE)
endfunction()

# Function for adding a ROCm library
function(add_rocm_gpu_library TARGET SHARED DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(gpu "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_library(${TARGET} ${SHARED} ${ROCM_SOURCE_FILES})
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# Function for adding a ROCm host-only library
function(add_rocm_host_library TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(host "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_library(${TARGET} ${ROCM_SOURCE_FILES})
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# Function for adding a ROCm executable
function(add_rocm_host_executable TARGET)
    set(SRC_FILES ${ARGN})
    add_executable(${TARGET} ${SRC_FILES})
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
endfunction()

# Function for adding a ROCm executable
function(add_rocm_gpu_executable TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(exec "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_executable(${TARGET} ${ROCM_SOURCE_FILES})
    target_link_libraries(${TARGET} PUBLIC amdhip64 amd_comgr)
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
endfunction()

# Function for adding a CUDA library
function(add_cuda_library TARGET SHARED DO_COPY)
    set(SRC_FILES ${ARGN})
    set(CUDA_SOURCE_FILES "")
    set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)
    add_library(${TARGET} ${SHARED} ${CUDA_SOURCE_FILES})

    target_compile_options(${TARGET} PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    if(SHARED STREQUAL "SHARED")
        set_target_properties(${TARGET} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RUNTIME_LIBRARY Shared
        )
    else()
        set_target_properties(${TARGET} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RUNTIME_LIBRARY Static
        )
    endif()
endfunction()

# Function for adding a CUDA executable
function(add_cuda_executable TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(CUDA_SOURCE_FILES "")
    set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)
    add_executable(${TARGET} ${CUDA_SOURCE_FILES})
    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    # When copying sources, we need to add the include path back to the original source directory
    if(${DO_COPY})
        target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    target_compile_options(${TARGET} PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endfunction()

#------------------------------------------------------------------------------
# Jarvis Repo Management
#------------------------------------------------------------------------------

# Function for autoregistering a jarvis repo
macro(jarvis_repo_add REPO_PATH)
    # Get the file name of the source path
    get_filename_component(REPO_NAME ${REPO_PATH} NAME)

    # Install jarvis repo
    install(DIRECTORY ${REPO_PATH}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/jarvis)

    # Add jarvis repo after installation
    # Ensure install commands use env vars from host system, particularly PATH and PYTHONPATH
    install(CODE "execute_process(COMMAND env \"PATH=$ENV{PATH}\" \"PYTHONPATH=$ENV{PYTHONPATH}\" jarvis repo add ${CMAKE_INSTALL_PREFIX}/jarvis/${REPO_NAME})")
endmacro()

#------------------------------------------------------------------------------
# Doxygen Documentation
#------------------------------------------------------------------------------

function(add_doxygen_doc)
    set(options)
    set(oneValueArgs BUILD_DIR DOXY_FILE TARGET_NAME COMMENT)
    set(multiValueArgs)

    cmake_parse_arguments(DOXY_DOC
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    configure_file(
        ${DOXY_DOC_DOXY_FILE}
        ${DOXY_DOC_BUILD_DIR}/Doxyfile
        @ONLY
    )

    add_custom_target(${DOXY_DOC_TARGET_NAME}
        COMMAND
        ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY
        ${DOXY_DOC_BUILD_DIR}
        COMMENT
        "Building ${DOXY_DOC_COMMENT} with Doxygen"
        VERBATIM
    )

    message(STATUS "Added ${DOXY_DOC_TARGET_NAME} [Doxygen] target to build documentation")
endfunction()

#------------------------------------------------------------------------------
# Python Finding Utilities
#------------------------------------------------------------------------------

# FIND PYTHON
macro(find_first_path_python)
    if(DEFINED ENV{PATH})
        string(REPLACE ":" ";" PATH_LIST $ENV{PATH})

        foreach(PATH_ENTRY ${PATH_LIST})
            find_program(PYTHON_SCAN
                NAMES python3 python
                PATHS ${PATH_ENTRY}
                NO_DEFAULT_PATH
            )

            if(PYTHON_SCAN)
                message(STATUS "Found Python in PATH: ${PYTHON_SCAN}")
                set(Python_EXECUTABLE ${PYTHON_SCAN} CACHE FILEPATH "Python executable" FORCE)
                set(Python3_EXECUTABLE ${PYTHON_SCAN} CACHE FILEPATH "Python executable" FORCE)
                break()
            endif()
        endforeach()
    endif()

    set(Python_FIND_STRATEGY LOCATION)
    find_package(Python3 COMPONENTS Interpreter Development)

    if(Python3_FOUND)
        message(STATUS "Found Python3: ${Python3_EXECUTABLE}")
    else()
        message(FATAL_ERROR "Python3 not found")
    endif()
endmacro()

#------------------------------------------------------------------------------
# ChiMod Helper Functions
#------------------------------------------------------------------------------

# Helper function to link runtime to client library (called via DEFER)
# This allows linking to work regardless of which target is defined first
function(_chimaera_link_runtime_to_client RUNTIME_TARGET CLIENT_TARGET)
  if(TARGET ${CLIENT_TARGET})
    target_link_libraries(${RUNTIME_TARGET} PUBLIC ${CLIENT_TARGET})
    message(STATUS "Deferred linking: Runtime ${RUNTIME_TARGET} linked to client ${CLIENT_TARGET}")
  endif()
endfunction()

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
# Automatic Cross-Namespace Dependencies (Unified Builds):
#   For non-chimaera namespaces (e.g., wrp_cte, wrp_cae), this function automatically
#   links chimaera admin and bdev client libraries if they are available as targets.
#   This enables wrp_* ChiMods to use chimaera ChiMod headers and functionality without
#   explicit dependency declarations in their CMakeLists.txt files.
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
  set(CHIMAERA_CXX_STANDARD 17)
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CHIMAERA_CXX_STANDARD})

  # Common compile definitions
  set(CHIMAERA_COMMON_COMPILE_DEFS
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
  )

  # Add compile definitions
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      ${CHIMAERA_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )

  # Add include directories with proper BUILD_INTERFACE and INSTALL_INTERFACE
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )

  # Add additional include directories with BUILD_INTERFACE wrapper
  foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRECTORIES})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${INCLUDE_DIR}>
    )
  endforeach()

  # Add link directories
  if(ARG_LINK_DIRECTORIES)
    target_link_directories(${TARGET_NAME} PUBLIC ${ARG_LINK_DIRECTORIES})
  endif()

  # Link libraries - use chimaera::cxx for internal builds, hermes_shm::cxx for external
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

  # Automatically add chimaera ChiMod dependencies in unified builds
  set(CHIMAERA_CHIMOD_DEPS "")
  if(NOT "${CHIMAERA_NAMESPACE}" STREQUAL "chimaera")
    if(TARGET chimaera_admin_client)
      list(APPEND CHIMAERA_CHIMOD_DEPS chimaera_admin_client)
    endif()
    if(TARGET chimaera_bdev_client)
      list(APPEND CHIMAERA_CHIMOD_DEPS chimaera_bdev_client)
    endif()
  endif()

  # Clients only link to hshm::cxx (no Boost)
  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${CORE_LIB}
      ${ARG_LINK_LIBRARIES}
      ${CHIMAERA_CHIMOD_DEPS}
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
# Automatic Cross-Namespace Dependencies (Unified Builds):
#   For non-chimaera namespaces (e.g., wrp_cte, wrp_cae), this function automatically
#   links chimaera admin and bdev runtime libraries if they are available as targets.
#   This enables wrp_* ChiMods to use chimaera ChiMod headers and functionality without
#   explicit dependency declarations in their CMakeLists.txt files.
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
  set(CHIMAERA_CXX_STANDARD 17)
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CHIMAERA_CXX_STANDARD})

  # Common compile definitions
  set(CHIMAERA_COMMON_COMPILE_DEFS
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
  )

  # Add compile definitions (runtime always has CHIMAERA_RUNTIME=1)
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      CHIMAERA_RUNTIME=1
      ${CHIMAERA_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )

  # Add include directories with proper BUILD_INTERFACE and INSTALL_INTERFACE
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )

  # Add additional include directories with BUILD_INTERFACE wrapper
  foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRECTORIES})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${INCLUDE_DIR}>
    )
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

  # Runtime-specific link libraries
  set(CHIMAERA_RUNTIME_LIBS
    Threads::Threads
  )

  # Automatically link to client library if it exists
  set(RUNTIME_LINK_LIBS ${CORE_LIB} ${CHIMAERA_RUNTIME_LIBS} ${ARG_LINK_LIBRARIES})

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

  # Automatically add chimaera ChiMod dependencies in unified builds
  if(NOT "${CHIMAERA_NAMESPACE}" STREQUAL "chimaera")
    if(TARGET chimaera_admin_runtime)
      list(APPEND RUNTIME_LINK_LIBS chimaera_admin_runtime)
    endif()
    if(TARGET chimaera_bdev_runtime)
      list(APPEND RUNTIME_LINK_LIBS chimaera_bdev_runtime)
    endif()
  endif()

  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${RUNTIME_LINK_LIBS} 
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
  set(SHOULD_GENERATE_CONFIG FALSE)
  if(CHIMAERA_MODULE_CLIENT_TARGET AND TARGET ${CHIMAERA_MODULE_CLIENT_TARGET})
    set(SHOULD_GENERATE_CONFIG TRUE)
  elseif(NOT CHIMAERA_MODULE_CLIENT_TARGET)
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

message(STATUS "IowarpCoreCommon.cmake loaded successfully")
