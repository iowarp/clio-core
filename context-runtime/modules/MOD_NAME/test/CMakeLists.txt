# CMakeLists.txt for MOD_NAME module unit tests
cmake_minimum_required(VERSION 3.10)

# Flush correctness test executable
set(FLUSH_TEST_TARGET clio_run_flush_correctness_tests)
set(FLUSH_TEST_SOURCES
  test_flush_correctness.cc
)

# CoMutex test executable
set(COMUTEX_TEST_TARGET clio_run_comutex_tests)
set(COMUTEX_TEST_SOURCES
  test_comutex.cc
)

# Wait Functionality test executable
set(WAIT_FUNCTIONALITY_TEST_TARGET clio_run_wait_functionality_tests)
set(WAIT_FUNCTIONALITY_TEST_SOURCES
  test_wait_functionality.cc
)

# Streaming test executable
set(STREAMING_TEST_TARGET clio_run_streaming_tests)
set(STREAMING_TEST_SOURCES
  test_streaming.cc
)

# GPU client-process / submission / queue-stress / parallelism tests
# removed along with the GPU runtime concept.

# Create flush correctness test executable
add_executable(${FLUSH_TEST_TARGET} ${FLUSH_TEST_SOURCES})

target_include_directories(${FLUSH_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin tasks
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks
)

target_link_libraries(${FLUSH_TEST_TARGET}
  cxx                             # Core CLIO Runtime
  clio_admin_runtime          # Admin module runtime
  clio_admin_client           # Admin module client
  clio_MOD_NAME_runtime       # MOD_NAME module runtime for work tracking
  clio_MOD_NAME_client        # MOD_NAME module client
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${FLUSH_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(${FLUSH_TEST_TARGET} PRIVATE CLIO_RUNTIME=1)

set_target_properties(${FLUSH_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create CoMutex test executable
add_executable(${COMUTEX_TEST_TARGET} ${COMUTEX_TEST_SOURCES})

target_include_directories(${COMUTEX_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin tasks
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

target_link_libraries(${COMUTEX_TEST_TARGET}
  clio_admin_runtime          # Admin module runtime
  clio_admin_client           # Admin module client
  clio_MOD_NAME_runtime       # MOD_NAME module runtime for CoMutex/CoRwLock
  clio_MOD_NAME_client        # MOD_NAME module client
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${COMUTEX_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${COMUTEX_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Wait Functionality test executable
add_executable(${WAIT_FUNCTIONALITY_TEST_TARGET} ${WAIT_FUNCTIONALITY_TEST_SOURCES})

target_include_directories(${WAIT_FUNCTIONALITY_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin tasks
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

target_link_libraries(${WAIT_FUNCTIONALITY_TEST_TARGET}
  clio_admin_runtime          # Admin module runtime
  clio_admin_client           # Admin module client
  clio_MOD_NAME_runtime       # MOD_NAME module runtime for WaitTest tasks
  clio_MOD_NAME_client        # MOD_NAME module client
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${WAIT_FUNCTIONALITY_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${WAIT_FUNCTIONALITY_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Streaming test executable
add_executable(${STREAMING_TEST_TARGET} ${STREAMING_TEST_SOURCES})

target_include_directories(${STREAMING_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin tasks
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

target_link_libraries(${STREAMING_TEST_TARGET}
  clio_admin_runtime          # Admin module runtime
  clio_admin_client           # Admin module client
  clio_MOD_NAME_runtime       # MOD_NAME module runtime for streaming tasks
  clio_MOD_NAME_client        # MOD_NAME module client
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${STREAMING_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${STREAMING_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable CTest integration if testing is enabled
if(CLIO_CORE_ENABLE_TESTS)
  # Each binary is registered once and runs all its Catch2 sections in a single
  # clio_run init/finalize cycle.  The previous setup added ~22 filtered ctest
  # entries over the same 4 binaries, which forced ~22 init/finalize cycles
  # serialized via RESOURCE_LOCK and exposed the clio_run shutdown ZMQ race.
  add_test(
    NAME cr_all_flush_tests
    COMMAND ${FLUSH_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  add_test(
    NAME cr_all_comutex_tests
    COMMAND ${COMUTEX_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  add_test(
    NAME cr_all_wait_functionality_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  add_test(
    NAME cr_all_streaming_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  set_tests_properties(
    cr_all_flush_tests
    cr_all_comutex_tests
    cr_all_wait_functionality_tests
    cr_all_streaming_tests
    PROPERTIES
      TIMEOUT 300
      ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
      LABELS "msan_skip;asan_skip"
  )

  # GPU client-process tests removed along with the GPU runtime.
endif()

# Custom targets for MOD_NAME tests
add_custom_target(test_flush
  COMMAND ${FLUSH_TEST_TARGET}
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio flush correctness tests"
)

add_custom_target(test_flush_basic
  COMMAND ${FLUSH_TEST_TARGET}
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio flush basic tests"
)

add_custom_target(test_flush_work_orchestrator
  COMMAND ${FLUSH_TEST_TARGET}
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio flush WorkOrchestrator integration tests"
)

add_custom_target(test_flush_stress
  COMMAND ${FLUSH_TEST_TARGET}
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio flush stress tests"
)

add_custom_target(test_comutex
  COMMAND ${COMUTEX_TEST_TARGET}
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio CoMutex and CoRwLock tests"
)

add_custom_target(test_comutex_basic
  COMMAND ${COMUTEX_TEST_TARGET} "[comutex][basic]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio CoMutex basic tests"
)

add_custom_target(test_corwlock_basic
  COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][basic]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio CoRwLock basic tests"
)

add_custom_target(test_comutex_concurrent
  COMMAND ${COMUTEX_TEST_TARGET} "[comutex][concurrent]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio concurrency tests"
)

add_custom_target(test_comutex_performance
  COMMAND ${COMUTEX_TEST_TARGET} "[comutex][performance]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio synchronization performance tests"
)

add_custom_target(test_streaming
  COMMAND ${STREAMING_TEST_TARGET}
  DEPENDS ${STREAMING_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio output streaming tests"
)

add_custom_target(test_streaming_small
  COMMAND ${STREAMING_TEST_TARGET}
  DEPENDS ${STREAMING_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio small output tests"
)

add_custom_target(test_streaming_large
  COMMAND ${STREAMING_TEST_TARGET}
  DEPENDS ${STREAMING_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio large output streaming tests"
)

add_custom_target(test_streaming_concurrent
  COMMAND ${STREAMING_TEST_TARGET}
  DEPENDS ${STREAMING_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio concurrent streaming tests"
)


# Migrate test executable (integration test binary for container migration)
set(MIGRATE_TEST_TARGET clio_run_migrate_tests)
set(MIGRATE_TEST_SOURCES
  ${CLIO_RUN_ROOT}/test/integration/migrate/test_migrate.cc
)

add_executable(${MIGRATE_TEST_TARGET} ${MIGRATE_TEST_SOURCES})

target_include_directories(${MIGRATE_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
)

target_link_libraries(${MIGRATE_TEST_TARGET}
  clio_admin_runtime
  clio_admin_client
  clio_MOD_NAME_runtime
  clio_MOD_NAME_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${MIGRATE_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Recovery test executable (integration test binary for node failure recovery)
set(RECOVERY_TEST_TARGET clio_run_recovery_tests)
set(RECOVERY_TEST_SOURCES
  ${CLIO_RUN_ROOT}/test/integration/recovery/test_recovery.cc
)

add_executable(${RECOVERY_TEST_TARGET} ${RECOVERY_TEST_SOURCES})

target_include_directories(${RECOVERY_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
)

target_link_libraries(${RECOVERY_TEST_TARGET}
  clio_admin_runtime
  clio_admin_client
  clio_MOD_NAME_runtime
  clio_MOD_NAME_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${RECOVERY_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Reconnect test executable (integration test binary for client failover)
set(RECONNECT_TEST_TARGET clio_run_reconnect_tests)
set(RECONNECT_TEST_SOURCES
  ${CLIO_RUN_ROOT}/test/integration/reconnect/test_reconnect.cc
)

add_executable(${RECONNECT_TEST_TARGET} ${RECONNECT_TEST_SOURCES})

target_include_directories(${RECONNECT_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
)

target_link_libraries(${RECONNECT_TEST_TARGET}
  clio_admin_runtime
  clio_admin_client
  clio_MOD_NAME_runtime
  clio_MOD_NAME_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${RECONNECT_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install test executables
install(TARGETS ${FLUSH_TEST_TARGET} ${COMUTEX_TEST_TARGET} ${WAIT_FUNCTIONALITY_TEST_TARGET} ${STREAMING_TEST_TARGET} ${MIGRATE_TEST_TARGET} ${RECOVERY_TEST_TARGET} ${RECONNECT_TEST_TARGET}
  RUNTIME DESTINATION bin
)

message(STATUS "MOD_NAME module tests configured:")
message(STATUS "  Flush test target: ${FLUSH_TEST_TARGET}")
message(STATUS "  CoMutex test target: ${COMUTEX_TEST_TARGET}")
message(STATUS "  Wait Functionality test target: ${WAIT_FUNCTIONALITY_TEST_TARGET}")
message(STATUS "  Streaming test target: ${STREAMING_TEST_TARGET}")
message(STATUS "  Migrate test target: ${MIGRATE_TEST_TARGET}")
message(STATUS "  Recovery test target: ${RECOVERY_TEST_TARGET}")
message(STATUS "  Reconnect test target: ${RECONNECT_TEST_TARGET}")
