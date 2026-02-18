cmake_minimum_required(VERSION 3.10)
project(chimaera VERSION 1.0.0)

# Set root directory for this component
set(CHIMAERA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Enable CUDA if requested
if(WRP_CORE_ENABLE_CUDA)
    wrp_core_enable_cuda(17)
endif()

# Enable ROCm if requested
if(WRP_CORE_ENABLE_ROCM)
    wrp_core_enable_rocm(HIP 17)
endif()

# Read namespace from chimaera_repo.yaml in project root (function defined in ChimaeraCommon.cmake)
# This will be called after the utilities are included below

#------------------------------------------------------------------------------
# SECTION 3: CHIMOD CMAKE UTILITIES AND DEPENDENCIES
#------------------------------------------------------------------------------
# ChiMod utilities are now provided by IowarpCoreCommon.cmake in /workspace/cmake
# No local includes needed - all find_package calls are in root CMakeLists.txt

#------------------------------------------------------------------------------
# SECTION 4: DEPENDENCY VERIFICATION
#------------------------------------------------------------------------------
# Verify that required dependencies were found by ChimaeraCommon.cmake
# Check for both package and target availability (supports both installed and subdirectory modes)
if(NOT HermesShm_FOUND AND NOT TARGET cxx AND NOT TARGET hshm::cxx)
  message(FATAL_ERROR "Hermes SHM (HSHM) not found")
endif()

if(HermesShm_FOUND)
  message(STATUS "found hermes_shm at ${HermesShm_PREFIX}")
else()
  message(STATUS "using hermes_shm as subdirectory (target: cxx)")
endif()

# Note: ZeroMQ support disabled for stub implementation

#------------------------------------------------------------------------------
# SECTION 5: SOURCE COMPILATION
#------------------------------------------------------------------------------
# Include directories
include_directories(${CHIMAERA_ROOT}/include)
include_directories(${HermesShm_INCLUDE_DIRS})

# Output directories are set by root CMakeLists.txt

# Read namespace from chimaera_repo.yaml BEFORE processing subdirectories
# This searches up the directory tree starting from CHIMAERA_ROOT
read_repo_namespace(CHIMAERA_NAMESPACE "${CHIMAERA_ROOT}")
# Set namespace as cache variable so it's available to all CMakeLists.txt
set(CHIMAERA_NAMESPACE "${CHIMAERA_NAMESPACE}" CACHE STRING "Project namespace for chimod targets")
message(STATUS "Chimaera namespace: ${CHIMAERA_NAMESPACE}")

# Add utilities subdirectory FIRST so chimaera repo refresh builds before everything else
add_subdirectory(util)

# Add source subdirectory to build main library
add_subdirectory(src)

# Add modules subdirectory
add_subdirectory(modules)

# Add benchmark subdirectory if benchmarks are enabled
# Benchmarks are controlled by WRP_CORE_ENABLE_BENCHMARKS
if(WRP_CORE_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
  message(STATUS "Benchmarks enabled - added benchmark subdirectory")
endif()

# Add test subdirectory if testing is enabled
# WRP_CORE_ENABLE_TESTS is set by root CMakeLists.txt
if(WRP_CORE_ENABLE_TESTS)
  # enable_testing() is handled by root CMakeLists.txt
  add_subdirectory(test)
  message(STATUS "Unit tests enabled - added test subdirectory")
endif()

#------------------------------------------------------------------------------
# SECTION 6: JARVIS INTEGRATION
#------------------------------------------------------------------------------
# Add Jarvis WRP runtime packages
if(WRP_CORE_ENABLE_JARVIS)
    jarvis_repo_add(${CHIMAERA_ROOT}/test/jarvis_iowarp)
endif()

#------------------------------------------------------------------------------
# SECTION 7: INSTALL CODE
#------------------------------------------------------------------------------
# Install the main chimaera library using the same dynamic system as modules
set(MAIN_EXPORT_NAME "${CHIMAERA_NAMESPACE}_coreTargets")
set(MAIN_PACKAGE_NAME "${CHIMAERA_NAMESPACE}")

# Include cereal in export if it was built as a submodule
set(_CEREAL_TARGET "")
if(TARGET cereal)
    set(_CEREAL_TARGET cereal)
endif()

# Install main library
install(TARGETS ${CHIMAERA_NAMESPACE}_cxx ${_CEREAL_TARGET}
  EXPORT ${MAIN_EXPORT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include/chimaera
  INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/chimaera
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)

# Install config files
install(DIRECTORY config/
  DESTINATION etc/chimaera
  FILES_MATCHING PATTERN "*"
)

# Seed ~/.chimaera/chimaera.yaml for the real user (handles sudo)
# Only creates the file if it doesn't already exist
install(CODE "
  # Determine real user's home directory
  set(_HOME \"\")
  if(DEFINED ENV{SUDO_USER})
    set(_SUDO_USER \"\$ENV{SUDO_USER}\")
    execute_process(
      COMMAND bash -c \"eval echo ~\${_SUDO_USER}\"
      OUTPUT_VARIABLE _HOME
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()
  if(_HOME STREQUAL \"\")
    set(_HOME \"\$ENV{HOME}\")
  endif()

  set(_DEST \"\${_HOME}/.chimaera\")
  set(_FILE \"\${_DEST}/chimaera.yaml\")
  if(NOT EXISTS \"\${_FILE}\")
    file(MAKE_DIRECTORY \"\${_DEST}\")
    file(COPY_FILE
      \"\${CMAKE_INSTALL_PREFIX}/etc/chimaera/chimaera_default.yaml\"
      \"\${_FILE}\"
    )
    message(STATUS \"Seeded default config: \${_FILE}\")
    # Fix ownership when running under sudo
    if(DEFINED ENV{SUDO_USER})
      execute_process(COMMAND chown -R \"\$ENV{SUDO_USER}:\$ENV{SUDO_USER}\" \"\${_DEST}\")
    endif()
  else()
    message(STATUS \"User config already exists: \${_FILE} (not overwriting)\")
  endif()
")

# Export targets file for main library
install(EXPORT ${MAIN_EXPORT_NAME}
  FILE ${MAIN_EXPORT_NAME}.cmake
  NAMESPACE ${CHIMAERA_NAMESPACE}::
  DESTINATION lib/cmake/${MAIN_PACKAGE_NAME}
)

# Generate and install package config file for main library
include(CMakePackageConfigHelpers)

# Generate ChimaeraConfig.cmake inline (no template file needed)
set(CHIMAERA_CONFIG_CONTENT "@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# All dependencies are handled by root CMakeLists.txt
# Include IowarpCoreCommon for external ChiMod development
include(\"\${CMAKE_CURRENT_LIST_DIR}/IowarpCoreCommon.cmake\")

# Include the exported targets
include(\"\${CMAKE_CURRENT_LIST_DIR}/@MAIN_EXPORT_NAME@.cmake\")

# Provide components
check_required_components(@MAIN_PACKAGE_NAME@)
")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${MAIN_PACKAGE_NAME}Config.cmake.in" "${CHIMAERA_CONFIG_CONTENT}")

configure_package_config_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${MAIN_PACKAGE_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${MAIN_PACKAGE_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${MAIN_PACKAGE_NAME}
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${MAIN_PACKAGE_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${MAIN_PACKAGE_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${MAIN_PACKAGE_NAME}ConfigVersion.cmake"
  "${CMAKE_SOURCE_DIR}/cmake/IowarpCoreCommon.cmake"
  DESTINATION lib/cmake/${MAIN_PACKAGE_NAME}
)

message(STATUS "Created core package: ${MAIN_PACKAGE_NAME}")
message(STATUS "  Export set: ${MAIN_EXPORT_NAME}")
message(STATUS "  Config location: lib/cmake/${MAIN_PACKAGE_NAME}")