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
  PUBLIC chimaera_cxx
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

# Install the binary under its canonical name `clio_run`. The legacy name
# `chimaera` (predates the CLIO rebrand) is installed as a symlink in the
# same bin/ dir so existing pipelines and scripts keep working. The binary
# adapts its own usage messages via argv[0] so each invocation prints the
# right name. See docs/deprecation-notes.md for the full migration table.
install(TARGETS clio_run
  RUNTIME DESTINATION bin
)

# Install `chimaera` as a symlink to `clio_run` (Windows: copy fallback).
# Done at install time so it lands at ${CMAKE_INSTALL_PREFIX}/bin regardless
# of CPACK/DESTDIR. On Windows the installed binary is `clio_run.exe`; the
# EXISTS check must include that suffix or the legacy alias is silently
# skipped at install time and `chimaera.exe` never lands in the wheel.
if(WIN32)
  install(CODE "
    set(_src \"\${CMAKE_INSTALL_PREFIX}/bin/clio_run.exe\")
    if(EXISTS \"\${_src}\")
      file(INSTALL \"\${_src}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/bin\"
           RENAME chimaera.exe)
      message(STATUS \"Installed chimaera.exe -> clio_run.exe legacy copy\")
    endif()
  ")
else()
  install(CODE "
    set(_src \"\${CMAKE_INSTALL_PREFIX}/bin/clio_run\")
    set(_dst \"\${CMAKE_INSTALL_PREFIX}/bin/chimaera\")
    if(EXISTS \"\${_src}\")
      execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink clio_run \"\${_dst}\")
      message(STATUS \"Installed chimaera -> clio_run legacy symlink at \${_dst}\")
    endif()
  ")
endif()
