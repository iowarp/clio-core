# util/CMakeLists.txt
# Utility executables for Chimaera runtime

cmake_minimum_required(VERSION 3.10)

#==============================================================================
# Shared command library used by unified chimaera binary and legacy wrappers
#==============================================================================
add_library(chimaera_commands STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_runtime_start.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_runtime_stop.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_monitor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_compose.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_migrate.cc
)
target_link_libraries(chimaera_commands
  PUBLIC chimaera_cxx
  PUBLIC chimaera_admin_client
)
target_include_directories(chimaera_commands
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

#==============================================================================
# Unified chimaera binary
#==============================================================================
add_executable(chimaera
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera.cc
)
target_link_libraries(chimaera
  PUBLIC chimaera_commands
)

#==============================================================================
# Legacy binaries (thin wrappers for backward compatibility)
#==============================================================================

# Build chimaera_start_runtime
add_executable(chimaera_start_runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_start_runtime.cc
)
target_link_libraries(chimaera_start_runtime
  PUBLIC chimaera_commands
)

# Build chimaera_stop_runtime
add_executable(chimaera_stop_runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_stop_runtime.cc
)
target_link_libraries(chimaera_stop_runtime
  PUBLIC chimaera_commands
)

# Build chimaera_compose
add_executable(chimaera_compose
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_compose.cc
)
target_link_libraries(chimaera_compose
  PUBLIC chimaera_commands
)

# Build chimaera_monitor
add_executable(chimaera_monitor
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_monitor.cc
)
target_link_libraries(chimaera_monitor
  PUBLIC chimaera_commands
)

#==============================================================================
# Standalone utilities (not part of unified binary)
#==============================================================================

# Build chi_refresh_repo
add_executable(chi_refresh_repo
  ${CMAKE_CURRENT_SOURCE_DIR}/chi_refresh_repo.cc
)
target_link_libraries(chi_refresh_repo
  PUBLIC hshm::cxx
)

# Install utilities
install(TARGETS chimaera chimaera_start_runtime chimaera_stop_runtime chi_refresh_repo chimaera_compose chimaera_monitor
  RUNTIME DESTINATION bin
)
