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
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_refresh_repo.cc
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

# Install utilities
install(TARGETS chimaera
  RUNTIME DESTINATION bin
)
