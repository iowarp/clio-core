# util/CMakeLists.txt
# Utility executables for CLIO Runtime

cmake_minimum_required(VERSION 3.10)

#==============================================================================
# Shared command library used by the unified clio_run binary
#==============================================================================
add_library(clio_run_commands STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/clio_run_cmd_runtime_start.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/clio_run_cmd_runtime_stop.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/clio_run_cmd_monitor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/clio_run_cmd_compose.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/clio_run_cmd_migrate.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/clio_run_cmd_refresh_repo.cc
)
target_link_libraries(clio_run_commands
  PUBLIC clio_run_cxx
  PUBLIC clio_admin_client
)
target_include_directories(clio_run_commands
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

#==============================================================================
# Unified clio_run binary (canonical CLIO Runtime CLI)
#==============================================================================
add_executable(clio_run
  ${CMAKE_CURRENT_SOURCE_DIR}/clio_run.cc
)
target_link_libraries(clio_run
  PUBLIC clio_run_commands
)

# Regression test for issue #528: on multi-config generators, static
# archives must land in per-configuration directories (bin/<CONFIG>) so a
# Release build cannot clobber the Debug archive and poison the next Debug
# link, while executables stay in the flat bin/ layout tests depend on.
if(CLIO_CORE_ENABLE_TESTS AND CMAKE_CONFIGURATION_TYPES)
  add_test(NAME build_layout_per_config_archives
    COMMAND ${CMAKE_COMMAND}
      -DARCHIVE_FILE=$<TARGET_FILE:clio_run_commands>
      -DEXE_FILE=$<TARGET_FILE:clio_run>
      -DBIN_DIR=${CMAKE_BINARY_DIR}/bin
      -DCONFIG=$<CONFIG>
      -P ${CMAKE_SOURCE_DIR}/cmake/CheckPerConfigArchiveLayout.cmake
  )
endif()

# Install the binary under its canonical name `clio_run`.
install(TARGETS clio_run
  RUNTIME DESTINATION bin
)
