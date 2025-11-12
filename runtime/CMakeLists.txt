cmake_minimum_required(VERSION 3.10)
project(chimaera VERSION 1.0.0)

# Read namespace from chimaera_repo.yaml in project root (function defined in ChimaeraCommon.cmake)
# This will be called after the utilities are included below

#------------------------------------------------------------------------------
# SECTION 3: CHIMOD CMAKE UTILITIES AND DEPENDENCIES
#------------------------------------------------------------------------------
# Include common utilities and dependencies early so find_package calls are available
include(cmake/ChimaeraCommon.cmake)

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
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${HermesShm_INCLUDE_DIRS})

# Output directories are set by root CMakeLists.txt

# Read namespace from chimaera_repo.yaml BEFORE processing subdirectories
# This searches up the directory tree starting from CMAKE_SOURCE_DIR
read_repo_namespace(CHIMAERA_NAMESPACE "${CMAKE_SOURCE_DIR}")
# Set namespace as cache variable so it's available to all CMakeLists.txt
set(CHIMAERA_NAMESPACE "${CHIMAERA_NAMESPACE}" CACHE STRING "Project namespace for chimod targets")
message(STATUS "Chimaera namespace: ${CHIMAERA_NAMESPACE}")

# Add utilities subdirectory FIRST so chi_refresh_repo builds before everything else
add_subdirectory(util)

# Add source subdirectory to build main library
add_subdirectory(src)

# Add modules subdirectory
add_subdirectory(modules)

# Add benchmark subdirectory
add_subdirectory(benchmark)

# Add test subdirectory if testing is enabled
if(CHIMAERA_ENABLE_TESTS)
  # enable_testing() is handled by root CMakeLists.txt
  add_subdirectory(test)
  message(STATUS "Unit tests enabled - added test subdirectory")
endif()

#------------------------------------------------------------------------------
# SECTION 6: JARVIS INTEGRATION
#------------------------------------------------------------------------------ 
# Add Jarvis WRP runtime packages
jarvis_repo_add(${CMAKE_SOURCE_DIR}/test/jarvis_wrp_runtime "")

#------------------------------------------------------------------------------
# SECTION 7: INSTALL CODE
#------------------------------------------------------------------------------
# Install the main chimaera library using the same dynamic system as modules
set(MAIN_EXPORT_NAME "${CHIMAERA_NAMESPACE}_coreTargets")
set(MAIN_PACKAGE_NAME "${CHIMAERA_NAMESPACE}")

# Install main library
install(TARGETS ${CHIMAERA_NAMESPACE}_cxx
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

# Export targets file for main library
install(EXPORT ${MAIN_EXPORT_NAME}
  FILE ${MAIN_EXPORT_NAME}.cmake
  NAMESPACE ${CHIMAERA_NAMESPACE}::
  DESTINATION lib/cmake/${MAIN_PACKAGE_NAME}
)

# Generate and install package config file for main library
include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ChimaeraConfig.cmake.in"
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
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ChimaeraCommon.cmake"
  DESTINATION lib/cmake/${MAIN_PACKAGE_NAME}
)

message(STATUS "Created core package: ${MAIN_PACKAGE_NAME}")
message(STATUS "  Export set: ${MAIN_EXPORT_NAME}")
message(STATUS "  Config location: lib/cmake/${MAIN_PACKAGE_NAME}")