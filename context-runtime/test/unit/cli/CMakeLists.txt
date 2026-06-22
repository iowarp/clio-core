# CMakeLists.txt for clio_run CLI command unit tests
# Tests the clio_run_commands static library (refresh/monitor/compose/
# migrate/start/stop) plus the clio_run binary's argument dispatch.
cmake_minimum_required(VERSION 3.10)

# POSIX-only: the test drives the clio_run binary as a subprocess and
# inspects exit codes via <sys/wait.h>.
if(NOT WIN32)

set(CLI_TEST_TARGET clio_run_cli_command_tests)

add_executable(${CLI_TEST_TARGET} test_clio_run_cli.cc)

target_include_directories(${CLI_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
)

target_link_libraries(${CLI_TEST_TARGET}
  clio_run_commands          # Command entry points (RefreshRepo, ...)
  clio_run_cxx               # Main CLIO Runtime library
  ctp::cxx                   # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

# The test invokes the clio_run binary as a subprocess; ensure it is built
# and tell the test where it lives.
add_dependencies(${CLI_TEST_TARGET} clio_run)
target_compile_definitions(${CLI_TEST_TARGET} PRIVATE
  CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")

set_target_properties(${CLI_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(CLIO_CORE_ENABLE_TESTS)
  # CLIO_WAIT_SERVER=0 makes client-mode commands (stop/monitor/compose/
  # migrate) fail immediately when no runtime is up, so no daemon is ever
  # started or waited for. CLIO_PORT is off the default 10500 to avoid
  # colliding with concurrently running runtime tests.
  set(_cli_test_env
    "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_WAIT_SERVER=0;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10599")

  add_test(
    NAME cr_cli_refresh_repo_tests
    COMMAND ${CLI_TEST_TARGET} "RefreshRepo"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_cli_refresh_repo_tests PROPERTIES
    ENVIRONMENT "${_cli_test_env}"
    TIMEOUT 60
    LABELS "msan_skip"
  )

  add_test(
    NAME cr_cli_dispatch_tests
    COMMAND ${CLI_TEST_TARGET} "CliDispatch"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_cli_dispatch_tests PROPERTIES
    ENVIRONMENT "${_cli_test_env}"
    TIMEOUT 90
    LABELS "msan_skip"
  )

  add_test(
    NAME cr_cli_client_error_tests
    COMMAND ${CLI_TEST_TARGET} "CliClientError"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_cli_client_error_tests PROPERTIES
    ENVIRONMENT "${_cli_test_env}"
    TIMEOUT 90
    LABELS "msan_skip"
  )
  # Live-runtime variant: spawns its own clio_run daemon on port 10601 via
  # clio::run::test::RuntimeServer (always reaped — destructor SIGTERM/SIGKILLs),
  # then drives monitor/compose/migrate/stop through the binary. The shared
  # memory segment name is per-user, so serialize with the other tests that
  # run a runtime.
  set(_cli_live_env
    "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1")
  if(APPLE)
    # macOS (issue #482): the local ZMQ DEALER->ROUTER control plane does not
    # deliver ClientConnect replies on the macos-14/15/26 CI runners, so every
    # live-daemon client command times out. Drive the clients over the
    # unix-socket IPC transport instead — the path cr_ipc_transport_ipc and
    # cr_client_retry_*_ipc prove out on those same runners.
    string(APPEND _cli_live_env ";CLIO_IPC_MODE=ipc")
  endif()
  add_test(
    NAME cr_cli_runtime_command_tests
    COMMAND ${CLI_TEST_TARGET} "CliRuntime"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_cli_runtime_command_tests PROPERTIES
    ENVIRONMENT "${_cli_live_env}"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 110
    LABELS "msan_skip"
  )
  # CTE WAL restart test: full persistence cycle (compose with restart:true,
  # write WAL, snapshot, daemon restart, replay). Spawns its own daemons on
  # port 10602 — serialize with the other live-runtime tests.
  add_executable(clio_run_cte_wal_restart_test test_cte_wal_restart.cc)
  target_include_directories(clio_run_cte_wal_restart_test PRIVATE
    ${CLIO_RUN_ROOT}/include
    ${CLIO_RUN_ROOT}/test
  )
  target_link_libraries(clio_run_cte_wal_restart_test
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  target_compile_definitions(clio_run_cte_wal_restart_test PRIVATE
    CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")
  add_dependencies(clio_run_cte_wal_restart_test clio_run)
  set_target_properties(clio_run_cte_wal_restart_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(
    NAME cr_cli_cte_wal_restart
    COMMAND clio_run_cte_wal_restart_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_cli_cte_wal_restart PROPERTIES
    ENVIRONMENT "${_cli_live_env}"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 110
    LABELS "msan_skip"
  )
  # Filesystem chimod test: compose bdev->cte_core->filesystem, drive the cfs
  # client (open/write/getattr/read/truncate), assert EXACT logical size.
  # Only built when the filesystem chimod is enabled.
  if(CLIO_CTE_ENABLE_FILESYSTEM)
    add_executable(clio_run_cfs_test test_cfs.cc)
    target_include_directories(clio_run_cfs_test PRIVATE
      ${CLIO_RUN_ROOT}/include
      ${CLIO_RUN_ROOT}/test
    )
    target_link_libraries(clio_run_cfs_test
      clio_cte_filesystem_client
      clio_cte_core_client
      clio_admin_client
      clio_run_cxx
      ctp::cxx
      ${CMAKE_THREAD_LIBS_INIT}
    )
    target_compile_definitions(clio_run_cfs_test PRIVATE
      CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")
    add_dependencies(clio_run_cfs_test clio_run clio_cte_filesystem_runtime)
    set_target_properties(clio_run_cfs_test PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(
      NAME cr_cli_cfs
      COMMAND clio_run_cfs_test
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cr_cli_cfs PROPERTIES
      ENVIRONMENT "${_cli_live_env}"
      RESOURCE_LOCK "clio_runtime_port"
      TIMEOUT 110
      LABELS "msan_skip"
    )

    # Focused rename stress + concurrency test (issue #597): POSIX
    # rename-overwrite semantics and renames racing on shared destinations.
    add_executable(clio_run_cfs_rename_test test_cfs_rename.cc)
    target_include_directories(clio_run_cfs_rename_test PRIVATE
      ${CLIO_RUN_ROOT}/include
      ${CLIO_RUN_ROOT}/test
    )
    target_link_libraries(clio_run_cfs_rename_test
      clio_cte_filesystem_client
      clio_cte_core_client
      clio_admin_client
      clio_run_cxx
      ctp::cxx
      ${CMAKE_THREAD_LIBS_INIT}
    )
    target_compile_definitions(clio_run_cfs_rename_test PRIVATE
      CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")
    add_dependencies(clio_run_cfs_rename_test clio_run clio_cte_filesystem_runtime)
    set_target_properties(clio_run_cfs_rename_test PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(
      NAME cr_cli_cfs_rename
      COMMAND clio_run_cfs_rename_test
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cr_cli_cfs_rename PROPERTIES
      ENVIRONMENT "${_cli_live_env}"
      RESOURCE_LOCK "clio_runtime_port"
      TIMEOUT 180
      LABELS "msan_skip"
    )

    # Fallback-runtime CTE test: CTE on a USER runtime backed by a bdev on a
    # MAIN runtime. CTE's internal bdev calls are serialize-relayed across the
    # fallback. Two ephemeral runtimes on ports 10670/10680.
    add_executable(clio_run_cte_fallback_test test_cte_fallback.cc)
    target_include_directories(clio_run_cte_fallback_test PRIVATE
      ${CLIO_RUN_ROOT}/include
      ${CLIO_RUN_ROOT}/test
    )
    target_link_libraries(clio_run_cte_fallback_test
      clio_cte_core_client
      clio_bdev_client
      clio_admin_client
      clio_run_cxx
      ctp::cxx
      ${CMAKE_THREAD_LIBS_INIT}
    )
    target_compile_definitions(clio_run_cte_fallback_test PRIVATE
      CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")
    add_dependencies(clio_run_cte_fallback_test clio_run clio_bdev_runtime
                     clio_cte_core_runtime)
    set_target_properties(clio_run_cte_fallback_test PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(
      NAME cr_cli_cte_fallback
      COMMAND clio_run_cte_fallback_test
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cr_cli_cte_fallback PROPERTIES
      ENVIRONMENT "${_cli_live_env}"
      RESOURCE_LOCK "clio_runtime_port"
      TIMEOUT 180
      LABELS "msan_skip"
    )
  endif()

  # Per-RPC access-control test (private containers/methods). Composes ram bdev
  # pools marked private (whole-container and per-RPC) and asserts an external
  # client's direct Write/AllocateBlocks is rejected with EACCES while a public
  # control pool succeeds. Spawns its own daemon on port 10606 — serialize with
  # the other live-runtime tests.
  add_executable(clio_run_bdev_acl_test test_bdev_acl.cc)
  target_include_directories(clio_run_bdev_acl_test PRIVATE
    ${CLIO_RUN_ROOT}/include
    ${CLIO_RUN_ROOT}/test
  )
  target_link_libraries(clio_run_bdev_acl_test
    clio_bdev_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  target_compile_definitions(clio_run_bdev_acl_test PRIVATE
    CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")
  add_dependencies(clio_run_bdev_acl_test clio_run clio_bdev_runtime)
  set_target_properties(clio_run_bdev_acl_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(
    NAME cr_cli_bdev_acl
    COMMAND clio_run_bdev_acl_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_cli_bdev_acl PROPERTIES
    ENVIRONMENT "${_cli_live_env}"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 120
    LABELS "msan_skip"
  )

  # Fallback-runtime (crash isolation) test: spawns TWO runtimes (main on 10650
  # with a bdev, user on 10651 with CLIO_FALLBACK_PORT=10650 and no bdev), then
  # an SHM client calls the bdev through the user runtime, which punts to main.
  # Serialize with other live-runtime tests (shared per-user SHM segments).
  add_executable(clio_run_fallback_runtime_test test_fallback_runtime.cc)
  target_include_directories(clio_run_fallback_runtime_test PRIVATE
    ${CLIO_RUN_ROOT}/include
    ${CLIO_RUN_ROOT}/test
  )
  target_link_libraries(clio_run_fallback_runtime_test
    clio_bdev_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  target_compile_definitions(clio_run_fallback_runtime_test PRIVATE
    CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")
  add_dependencies(clio_run_fallback_runtime_test clio_run clio_bdev_runtime)
  set_target_properties(clio_run_fallback_runtime_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(
    NAME cr_cli_fallback_runtime
    COMMAND clio_run_fallback_runtime_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_cli_fallback_runtime PROPERTIES
    ENVIRONMENT "${_cli_live_env}"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 180
    LABELS "msan_skip"
  )

  # RestartLog unit test: pure file-level WAL behavior (no daemon).
  add_executable(clio_run_restart_log_test test_restart_log.cc)
  target_include_directories(clio_run_restart_log_test PRIVATE
    ${CLIO_RUN_ROOT}/include
    ${CLIO_RUN_ROOT}/test
  )
  target_link_libraries(clio_run_restart_log_test
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  set_target_properties(clio_run_restart_log_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(
    NAME cr_cli_restart_log
    COMMAND clio_run_restart_log_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_cli_restart_log PROPERTIES
    TIMEOUT 60
    LABELS "msan_skip"
  )

  # compose start/stop/rm/list + restart-WAL integration test: drives the
  # clio_run binary against a live daemon. Spawns daemons on ports 10610-10612
  # — serialize with the other live-runtime tests.
  add_executable(clio_run_compose_restart_test test_compose_restart.cc)
  target_include_directories(clio_run_compose_restart_test PRIVATE
    ${CLIO_RUN_ROOT}/include
    ${CLIO_RUN_ROOT}/test
  )
  target_link_libraries(clio_run_compose_restart_test
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  target_compile_definitions(clio_run_compose_restart_test PRIVATE
    CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")
  add_dependencies(clio_run_compose_restart_test clio_run)
  set_target_properties(clio_run_compose_restart_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(
    NAME cr_cli_compose_restart
    COMMAND clio_run_compose_restart_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_cli_compose_restart PROPERTIES
    ENVIRONMENT "${_cli_live_env}"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 240
    LABELS "msan_skip"
  )
endif()  # CLIO_CORE_ENABLE_TESTS

endif()  # NOT WIN32
