# CMakeLists.txt for bdev module unit tests
cmake_minimum_required(VERSION 3.10)

# Bdev ChiMod test executable
set(BDEV_TEST_TARGET clio_run_bdev_chimod_tests)
set(BDEV_TEST_SOURCES
  test_bdev_chimod.cc
)

# GPU bdev tests (test_bdev_gpucache, test_bdev_gpu_kernels) removed
# along with the GPU runtime concept.

# Create bdev test executable
add_executable(${BDEV_TEST_TARGET} ${BDEV_TEST_SOURCES})

# Include directories for bdev tests
target_include_directories(${BDEV_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin tasks
  ${CLIO_RUN_ROOT}/modules/bdev/include   # For bdev tasks and client
)

# Link against required libraries for bdev tests
target_link_libraries(${BDEV_TEST_TARGET}
  clio_admin_runtime          # Admin module runtime
  clio_admin_client           # Admin module client
  clio_bdev_runtime           # Bdev module runtime
  clio_bdev_client            # Bdev module client
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for bdev tests
set_target_properties(${BDEV_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

# Copy bdev test executable to bin directory
set_target_properties(${BDEV_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# GPU bdev test build rules removed.

# Enable CTest integration if testing is enabled
if(CLIO_CORE_ENABLE_TESTS)
  # Comprehensive bdev test (runs all tests; explicit TCP/IPC auto-skip
  # because default IPC mode is SHM and the skip guards filter them out)
  add_test(
    NAME cr_all_bdev_tests
    COMMAND ${BDEV_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  # The `bdev` label is what lets the prediction phase run this suite. That
  # phase scopes its coverage to context-runtime/modules/bdev but used to run
  # only the two prediction_e2e tests, so it reported ~21% on a subtree the
  # unit phase already covers at ~83%. A phase should run the tests for the
  # code it measures. This stays in the unit phase too: that selection
  # excludes by name (cr_bdev_) and by the slow/integration labels, neither
  # of which this matches.
  # Both labels must be set HERE, in the one and only set_tests_properties call
  # naming this test. LABELS is replaced, not appended, by each call, so listing
  # this test in a second call (as the shared block below used to do) silently
  # erases what the first call set. That is what happened: `bdev` was wiped by a
  # later LABELS "msan_skip", the prediction phase's `-L prediction_e2e|bdev`
  # stopped matching, and that phase measured the bdev subtree while never
  # running the suite that covers it -- 30.7% instead of ~83%. Keep this test
  # out of the shared block below.
  set_tests_properties(cr_all_bdev_tests PROPERTIES
    TIMEOUT 300
    ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    LABELS "bdev;msan_skip"
  )

  # Dedicated explicit backend tests — each sets CLIO_IPC_MODE so the
  # corresponding TEST_CASE runs while the other two auto-skip.
  add_test(
    NAME cr_bdev_explicit_backend_shm
    COMMAND ${BDEV_TEST_TARGET} "bdev_file_explicit_backend_shm"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_bdev_explicit_backend_shm PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_MODE=1;CLIO_IPC_MODE=SHM;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_bdev_explicit_backend_tcp
    COMMAND ${BDEV_TEST_TARGET} "bdev_file_explicit_backend_tcp"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_bdev_explicit_backend_tcp PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_MODE=1;CLIO_IPC_MODE=TCP;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_bdev_explicit_backend_ipc
    COMMAND ${BDEV_TEST_TARGET} "bdev_file_explicit_backend_ipc"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_bdev_explicit_backend_ipc PROPERTIES
    TIMEOUT 180
    ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_MODE=1;CLIO_IPC_MODE=IPC;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  # Set test properties
  set_tests_properties(
    cr_bdev_explicit_backend_shm
    cr_bdev_explicit_backend_tcp
    cr_bdev_explicit_backend_ipc
    PROPERTIES
      TIMEOUT 300
      LABELS "msan_skip"
      ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  # Force-net stress test: routes all bdev tasks through the network path
  # even on a single node, exercising SendIn/RecvIn/SendOut/RecvOut on loopback.
  add_test(
    NAME cr_bdev_force_net
    COMMAND ${BDEV_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_bdev_force_net PROPERTIES
    TIMEOUT 300
    LABELS "msan_skip"
    ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500;CLIO_FORCE_NET=1"
  )

  # Full-chain bdev -> ML prediction integration (Linux only). A driver script
  # stands up the live Python model server, then runs the
  # bdev_failure_prediction_integration case: a real bdev's Monitor("stats")
  # collects device health, derives the drive type, and calls the live model for
  # a real inference that must round-trip back through the bdev payload. SKIPs
  # (SKIP_RETURN_CODE 77) when python deps or the git-lfs models are absent, so
  # an ordinary build never fails on it; the dedicated CI section installs the
  # deps and runs `ctest -L prediction_e2e`. The env mirrors the other bdev
  # tests (embedded runtime needs CLIO_TEST_MODE/BIND_ADDR/REPO_PATH) with a
  # distinct CLIO_PORT so it can run alongside them.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(
      NAME cr_bdev_prediction_integration
      COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_bdev_prediction_integration.sh
              $<TARGET_FILE:${BDEV_TEST_TARGET}>
              ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/hw_failure_monitor
    )
    set_tests_properties(cr_bdev_prediction_integration PROPERTIES
      TIMEOUT 300
      SKIP_RETURN_CODE 77
      LABELS "prediction_e2e;msan_skip"
      ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10600"
    )
  endif()
endif()

# Custom targets for bdev tests
add_custom_target(test_bdev
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio bdev ChiMod tests"
)

add_custom_target(test_bdev_create
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio bdev container creation tests"
)

add_custom_target(test_bdev_allocate
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio bdev block allocation tests"
)

add_custom_target(test_bdev_io
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio bdev I/O operation tests"
)

add_custom_target(test_bdev_async
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio bdev asynchronous I/O tests"
)

add_custom_target(test_bdev_performance
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio bdev performance metrics tests"
)

add_custom_target(test_bdev_error
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Clio bdev error handling tests"
)

# Install test executable
install(TARGETS ${BDEV_TEST_TARGET}
  RUNTIME DESTINATION bin
)

message(STATUS "Bdev module tests configured:")
message(STATUS "  Test target: ${BDEV_TEST_TARGET}")
message(STATUS "  Test sources: ${BDEV_TEST_SOURCES}")
