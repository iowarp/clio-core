cmake_minimum_required(VERSION 3.10)
project(chimaera VERSION 1.0.0)

# Set root directory for this component
set(CLIO_RUN_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Enable CUDA if requested
if(CLIO_CORE_ENABLE_CUDA)
    wrp_core_enable_cuda(20)
endif()

# Enable ROCm if requested
if(CLIO_CORE_ENABLE_ROCM)
    wrp_core_enable_rocm(HIP 17)
endif()

# Enable SYCL if requested
if(CLIO_CORE_ENABLE_SYCL)
    wrp_core_enable_sycl(20)
endif()

# Read namespace from chimaera_repo.yaml in project root (function defined in ChimaeraCommon.cmake)
# This will be called after the utilities are included below

#------------------------------------------------------------------------------
# SECTION 3: CHIMOD CMAKE UTILITIES AND DEPENDENCIES
#------------------------------------------------------------------------------
# ChiMod utilities are now provided by ClioCoreCommon.cmake in /workspace/cmake
# No local includes needed - all find_package calls are in root CMakeLists.txt

#------------------------------------------------------------------------------
# SECTION 4: DEPENDENCY VERIFICATION
#------------------------------------------------------------------------------
# Verify that required dependencies were found by ChimaeraCommon.cmake
# Check for both package and target availability (supports both installed and subdirectory modes)
if(NOT ClioCtp_FOUND AND NOT TARGET cxx AND NOT TARGET ctp::cxx)
  message(FATAL_ERROR "Clio CTP not found")
endif()

if(ClioCtp_FOUND)
  message(STATUS "found clio_ctp at ${ClioCtp_PREFIX}")
else()
  message(STATUS "using clio_ctp as subdirectory (target: cxx)")
endif()

# Note: ZeroMQ support disabled for stub implementation

#------------------------------------------------------------------------------
# SECTION 5: SOURCE COMPILATION
#------------------------------------------------------------------------------
# Include directories
include_directories(${CLIO_RUN_ROOT}/include)
include_directories(${ClioCtp_INCLUDE_DIRS})

# Output directories are set by root CMakeLists.txt

# Read namespace from chimaera_repo.yaml BEFORE processing subdirectories
# This searches up the directory tree starting from CLIO_RUN_ROOT
read_repo_namespace(CLIO_RUN_NAMESPACE "${CLIO_RUN_ROOT}")
# Derive a filesystem-safe form: replace `::` with `_` so target prefixes
# like `clio::run` map to package paths `clio_run`.  See the same comment
# block in ClioCoreCommon.cmake's clio_run_read_module_config.
string(REPLACE "::" "_" CLIO_RUN_PACKAGE_NAME "${CLIO_RUN_NAMESPACE}")
# Set namespace as cache variable so it's available to all CMakeLists.txt
set(CLIO_RUN_NAMESPACE "${CLIO_RUN_NAMESPACE}" CACHE STRING "Project namespace for chimod targets (C++/target prefix; may contain `::`)")
set(CLIO_RUN_PACKAGE_NAME "${CLIO_RUN_PACKAGE_NAME}" CACHE STRING "Filesystem-safe form of CLIO_RUN_NAMESPACE (no `::`)")
message(STATUS "Chimaera namespace: ${CLIO_RUN_NAMESPACE} (package=${CLIO_RUN_PACKAGE_NAME})")

# Add utilities subdirectory FIRST so chimaera repo refresh builds before everything else
add_subdirectory(util)

# Add source subdirectory to build main library
add_subdirectory(src)

# Add modules subdirectory
add_subdirectory(modules)

# Add benchmark subdirectory if benchmarks are enabled
# Benchmarks are controlled by CLIO_CORE_ENABLE_BENCHMARKS
if(CLIO_CORE_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
  message(STATUS "Benchmarks enabled - added benchmark subdirectory")
endif()

# Add test subdirectory if testing is enabled
# CLIO_CORE_ENABLE_TESTS is set by root CMakeLists.txt
if(CLIO_CORE_ENABLE_TESTS)
  # enable_testing() is handled by root CMakeLists.txt
  add_subdirectory(test)
  message(STATUS "Unit tests enabled - added test subdirectory")
endif()

# Add Python bindings if enabled
if(CLIO_CORE_ENABLE_PYTHON)
  if(COMMAND nanobind_add_module)
    add_subdirectory(wrapper/python)
    message(STATUS "Chimaera runtime Python bindings enabled")
  else()
    message(WARNING "CLIO_CORE_ENABLE_PYTHON is ON but nanobind_add_module not found.")
  endif()
endif()

#------------------------------------------------------------------------------
# SECTION 6: JARVIS INTEGRATION
#------------------------------------------------------------------------------
# Add Jarvis WRP runtime packages
if(CLIO_CORE_ENABLE_JARVIS)
    jarvis_repo_add(${CLIO_RUN_ROOT}/test/jarvis_clio_core)
endif()

#------------------------------------------------------------------------------
# SECTION 7: INSTALL CODE
#------------------------------------------------------------------------------
# Install the main chimaera library using the same dynamic system as modules.
# Both names are filesystem paths (one ends up as `lib/cmake/${MAIN_PACKAGE_NAME}/`,
# the other as `${MAIN_EXPORT_NAME}.cmake` inside it) so we use the
# filesystem-safe PACKAGE_NAME, not the `::`-bearing NAMESPACE.
set(MAIN_EXPORT_NAME "${CLIO_RUN_PACKAGE_NAME}_coreTargets")
set(MAIN_PACKAGE_NAME "${CLIO_RUN_PACKAGE_NAME}")

# Include cereal in export if it was built as a submodule
set(_CEREAL_TARGET "")
if(TARGET cereal)
    set(_CEREAL_TARGET cereal)
endif()

# Collect GPU target if it exists.
#
# The GPU twin is defined in context-runtime/src/CMakeLists.txt as
# `${CHIMAERA_LIB_NAME}_gpu` where CHIMAERA_LIB_NAME is hardcoded to
# "clio_run_cxx" — so the literal target name is `clio_run_cxx_gpu`,
# NOT `${CLIO_RUN_NAMESPACE}_cxx_gpu`.  These can differ: the user-
# facing namespace deliberately stays `chimaera` for downstream
# find_package compat (see context-runtime/clio_repo.yaml), while the
# canonical .so / target names already use the new `clio_run_cxx`
# prefix.  Check the literal so this works regardless of the namespace.
set(_GPU_TARGET "")
if(TARGET clio_run_cxx_gpu)
  set(_GPU_TARGET clio_run_cxx_gpu)
endif()

# Install main library (and GPU library if built). The canonical target name
# is `clio_run_cxx` (see context-runtime/src/CMakeLists.txt); the
# install-time alias namespace stays `chimaera::` for backward compat with
# external find_package consumers like coeus-adapter.
install(TARGETS clio_run_cxx ${_GPU_TARGET} ${_CEREAL_TARGET}
  EXPORT ${MAIN_EXPORT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include/clio_runtime
  INCLUDES DESTINATION include
)

# Install libchimaera_cxx.{so,dylib} / libchimaera_cxx_gpu.{so,dylib}
# symlinks alongside the canonical libclio_run_cxx{,_gpu}.{so,dylib}* so
# any external binary that loaded the historical name via LD_LIBRARY_PATH
# or RPATH (dlopen / direct -lchimaera_cxx{,_gpu} link) keeps resolving.
install(CODE "
  if(WIN32)
    return()
  endif()
  set(_libdir \"\${CMAKE_INSTALL_PREFIX}/lib\")
  # SHARED libs land as e.g. libclio_run_cxx.so.1.5.7 on Linux with
  # libclio_run_cxx.so.1 and libclio_run_cxx.so symlinks; on macOS as
  # libclio_run_cxx.1.5.7.dylib with libclio_run_cxx.1.dylib and
  # libclio_run_cxx.dylib symlinks.
  # The wildcards catch both the host lib AND the GPU twin
  # (libclio_run_cxx_gpu) so the legacy symlink trio is created for
  # both — verified by the gpu-tests CI step which tests for
  # libchimaera_cxx_gpu.so.
  file(GLOB _real_files RELATIVE \"\${_libdir}\"
       \"\${_libdir}/libclio_run_cxx*.so*\"
       \"\${_libdir}/libclio_run_cxx*.dylib\")
  foreach(_real_name \${_real_files})
    string(REPLACE \"libclio_run_cxx\" \"libchimaera_cxx\"
           _link_name \"\${_real_name}\")
    if(NOT \"\${_link_name}\" STREQUAL \"\${_real_name}\"
       AND NOT EXISTS \"\${_libdir}/\${_link_name}\")
      execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
        \"\${_real_name}\" \"\${_libdir}/\${_link_name}\")
      message(STATUS \"Installed legacy symlink \${_link_name} -> \${_real_name}\")
    endif()
  endforeach()
")

# Install canonical headers (new name) AND the legacy forwarder-shim tree
# (so external code using `#include <chimaera/...>` still resolves).
install(DIRECTORY include/clio_runtime
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY include/chimaera
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)

# Install config files
install(DIRECTORY config/
  DESTINATION etc/chimaera
  FILES_MATCHING PATTERN "*"
)

# Seed the user's default config — both ~/.clio/clio.yaml (preferred) and
# ~/.chimaera/chimaera.yaml (legacy) so either location continues to work
# in the wild. Runtime lookup order is documented in
# ConfigManager::GetServerConfigPath; the new ~/.clio takes priority.
# Existing files are never overwritten.
install(CODE "
  # Determine real user's home directory (handles sudo install)
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

  # During CPack packaging, CMAKE_INSTALL_PREFIX points to /usr (or similar)
  # but files are actually staged under DESTDIR. If the source yaml does not
  # exist at the bare prefix path, this is a packaging install — skip seeding.
  set(_DEFAULT_SRC \"\${CMAKE_INSTALL_PREFIX}/etc/chimaera/chimaera_default.yaml\")
  if(NOT EXISTS \"\${_DEFAULT_SRC}\")
    message(STATUS \"Skipping user config seed (packaging install): \${_DEFAULT_SRC} not present\")
    return()
  endif()

  # Per-destination seeding helper: directory + filename + optional chown
  foreach(_entry
      \"\${_HOME}/.clio|clio.yaml\"       # new canonical location
      \"\${_HOME}/.chimaera|chimaera.yaml\")  # legacy location, kept seeded
    string(REPLACE \"|\" \";\" _parts \"\${_entry}\")
    list(GET _parts 0 _DEST)
    list(GET _parts 1 _NAME)
    set(_FILE \"\${_DEST}/\${_NAME}\")
    if(NOT EXISTS \"\${_FILE}\")
      file(MAKE_DIRECTORY \"\${_DEST}\")
      file(COPY_FILE \"\${_DEFAULT_SRC}\" \"\${_FILE}\")
      message(STATUS \"Seeded default config: \${_FILE}\")
      if(DEFINED ENV{SUDO_USER})
        execute_process(
          COMMAND chown -R \"\$ENV{SUDO_USER}:\$ENV{SUDO_USER}\" \"\${_DEST}\")
      endif()
    else()
      message(STATUS \"User config already exists: \${_FILE} (not overwriting)\")
    endif()
  endforeach()
")

# Export targets file for main library
install(EXPORT ${MAIN_EXPORT_NAME}
  FILE ${MAIN_EXPORT_NAME}.cmake
  NAMESPACE ${CLIO_RUN_NAMESPACE}::
  DESTINATION lib/cmake/${MAIN_PACKAGE_NAME}
)

# Generate and install package config file for main library
include(CMakePackageConfigHelpers)

# Generate ChimaeraConfig.cmake inline (no template file needed)
set(CHIMAERA_CONFIG_CONTENT "@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# All dependencies are handled by root CMakeLists.txt
# Include ClioCoreCommon for external Module development
include(\"\${CMAKE_CURRENT_LIST_DIR}/ClioCoreCommon.cmake\")

# Include the exported targets
include(\"\${CMAKE_CURRENT_LIST_DIR}/@MAIN_EXPORT_NAME@.cmake\")

# Backward-compat / forward-canonical aliases: chimaera::cxx is the
# install-time target name (pinned for downstream find_package compat),
# but new code should link via the canonical `clio::run::cxx`.  The
# intermediate `clio::run::cxx` waypoint is also kept for source-level
# back-compat with the pre-`::` rename spelling.
if(TARGET chimaera::cxx AND NOT TARGET clio::run::cxx)
  add_library(clio::run::cxx INTERFACE IMPORTED)
  set_target_properties(clio::run::cxx PROPERTIES
    INTERFACE_LINK_LIBRARIES chimaera::cxx)
endif()
if(TARGET chimaera::cxx AND NOT TARGET clio::run::cxx)
  add_library(clio::run::cxx INTERFACE IMPORTED)
  set_target_properties(clio::run::cxx PROPERTIES
    INTERFACE_LINK_LIBRARIES chimaera::cxx)
endif()

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
  "${CMAKE_SOURCE_DIR}/cmake/ClioCoreCommon.cmake"
  DESTINATION lib/cmake/${MAIN_PACKAGE_NAME}
)

message(STATUS "Created core package: ${MAIN_PACKAGE_NAME}")
message(STATUS "  Export set: ${MAIN_EXPORT_NAME}")
message(STATUS "  Config location: lib/cmake/${MAIN_PACKAGE_NAME}")