# CMakeLists.txt for admin module unit tests
cmake_minimum_required(VERSION 3.10)

# Task archive test executable
set(TASK_ARCHIVE_TEST_TARGET chimaera_task_archive_tests)
set(TASK_ARCHIVE_TEST_SOURCES
  test_task_archive.cc
)

# Compose feature test executable
set(COMPOSE_TEST_TARGET chimaera_compose_tests)
set(COMPOSE_TEST_SOURCES
  test_compose.cc
)

# SubmitBatch test executable
set(SUBMIT_BATCH_TEST_TARGET chimaera_submit_batch_tests)
set(SUBMIT_BATCH_TEST_SOURCES
  test_submit_batch.cc
)

# Create compose test executable
add_executable(${COMPOSE_TEST_TARGET} ${COMPOSE_TEST_SOURCES})

target_include_directories(${COMPOSE_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin client
  ${CLIO_RUN_ROOT}/modules/bdev/include   # For bdev client
)

target_link_libraries(${COMPOSE_TEST_TARGET}
  cxx                             # Core CLIO Runtime
  clio_admin_runtime          # Admin module runtime
  clio_admin_client           # Admin module client
  clio_bdev_runtime           # BDev module runtime
  clio_bdev_client            # BDev module client
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${COMPOSE_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(${COMPOSE_TEST_TARGET} PRIVATE CHIMAERA_RUNTIME=1)

set_target_properties(${COMPOSE_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create task archive test executable
add_executable(${TASK_ARCHIVE_TEST_TARGET} ${TASK_ARCHIVE_TEST_SOURCES})

target_include_directories(${TASK_ARCHIVE_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin tasks
)

target_link_libraries(${TASK_ARCHIVE_TEST_TARGET}
  clio_admin_runtime          # Admin module runtime for task types
  clio_admin_client           # Admin module client for task types
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${TASK_ARCHIVE_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${TASK_ARCHIVE_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create SubmitBatch test executable
add_executable(${SUBMIT_BATCH_TEST_TARGET} ${SUBMIT_BATCH_TEST_SOURCES})

target_include_directories(${SUBMIT_BATCH_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin tasks
)

target_link_libraries(${SUBMIT_BATCH_TEST_TARGET}
  clio_admin_runtime          # Admin module runtime
  clio_admin_client           # Admin module client
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${SUBMIT_BATCH_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(${SUBMIT_BATCH_TEST_TARGET} PRIVATE CHIMAERA_RUNTIME=1)

set_target_properties(${SUBMIT_BATCH_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable CTest integration if testing is enabled
if(CLIO_CORE_ENABLE_TESTS)
  # Task Archive Tests
  add_test(
    NAME cr_task_archive_basic_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][input_in],[task_archive][output_in],[task_archive][input_out],[task_archive][output_out]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_task_archive_bulk_transfer_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][bulk_transfer]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_bulk_transfer_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_task_archive_non_task_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][non_task]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_non_task_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_task_archive_task_base_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][task_base]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_task_base_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_task_archive_admin_tasks_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][admin_tasks]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_admin_tasks_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_task_archive_bidirectional_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][bidirectional]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_bidirectional_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_task_archive_container_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][container]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_container_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_task_archive_error_handling_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][error_handling]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_error_handling_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_task_archive_performance_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][performance]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_performance_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_task_archive_integration_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][integration]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_task_archive_integration_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  # Comprehensive task archive test
  add_test(
    NAME cr_all_task_archive_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_task_archive_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  # Compose Tests
  add_test(
    NAME cr_compose_tests
    COMMAND ${COMPOSE_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_compose_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 180
  )

  # Submit Batch Tests
  add_test(
    NAME cr_submit_batch_tests
    COMMAND ${SUBMIT_BATCH_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_submit_batch_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 180
  )

  # Set test properties
  set_tests_properties(
    cr_task_archive_basic_tests
    cr_task_archive_bulk_transfer_tests
    cr_task_archive_non_task_tests
    cr_task_archive_task_base_tests
    cr_task_archive_admin_tasks_tests
    cr_task_archive_bidirectional_tests
    cr_task_archive_container_tests
    cr_task_archive_error_handling_tests
    cr_task_archive_performance_tests
    cr_task_archive_integration_tests
    cr_all_task_archive_tests
    PROPERTIES
      TIMEOUT 180
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  # Mark all admin tests as msan_skip (they use ZMQ which is uninstrumented)
  set_tests_properties(
    cr_task_archive_basic_tests
    cr_task_archive_bulk_transfer_tests
    cr_task_archive_non_task_tests
    cr_task_archive_task_base_tests
    cr_task_archive_admin_tasks_tests
    cr_task_archive_bidirectional_tests
    cr_task_archive_container_tests
    cr_task_archive_error_handling_tests
    cr_task_archive_performance_tests
    cr_task_archive_integration_tests
    cr_all_task_archive_tests
    cr_compose_tests
    cr_submit_batch_tests
    PROPERTIES LABELS "msan_skip"
  )

  # CLIO_FORCE_NET stress variants: route every task through the loopback
  # run-to-run network path so the serialize/send/recv/aggregate path is
  # exercised for all compose/batch task types on a single node.
  # NOTE: no CLIO_PORT here. The compose test writes its own runtime config
  # (port 9413); forcing CLIO_PORT=10500 made the server bind 10500 while the
  # composed pools' hosts kept the config port 9413, so under force_net every
  # reply was sent to a dead port and Wait() hung. Letting the test's config
  # own the port keeps bind and send consistent.
  clio_add_force_net_test(
    NAME cr_compose_force_net
    COMMAND ${COMPOSE_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1"
  )
  clio_add_force_net_test(
    NAME cr_submit_batch_force_net
    COMMAND ${SUBMIT_BATCH_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )
endif()

# Custom targets for admin tests
add_custom_target(test_task_archive
  COMMAND ${TASK_ARCHIVE_TEST_TARGET}
  DEPENDS ${TASK_ARCHIVE_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task archive tests"
)

add_custom_target(test_task_archive_basic
  COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][input_in],[task_archive][output_in],[task_archive][input_out],[task_archive][output_out]"
  DEPENDS ${TASK_ARCHIVE_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task archive basic tests"
)

add_custom_target(test_task_archive_serialization
  COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][task_base],[task_archive][admin_tasks]"
  DEPENDS ${TASK_ARCHIVE_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task serialization tests"
)

add_custom_target(test_task_archive_performance
  COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][performance]"
  DEPENDS ${TASK_ARCHIVE_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task archive performance tests"
)

# Install test executables
install(TARGETS ${TASK_ARCHIVE_TEST_TARGET} ${COMPOSE_TEST_TARGET} ${SUBMIT_BATCH_TEST_TARGET}
  RUNTIME DESTINATION bin
)

message(STATUS "Admin module tests configured:")
message(STATUS "  Task archive test target: ${TASK_ARCHIVE_TEST_TARGET}")
message(STATUS "  Compose test target: ${COMPOSE_TEST_TARGET}")
message(STATUS "  SubmitBatch test target: ${SUBMIT_BATCH_TEST_TARGET}")
