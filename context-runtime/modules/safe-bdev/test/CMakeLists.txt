# CMakeLists.txt for safe-bdev module unit tests
cmake_minimum_required(VERSION 3.10)

set(SAFE_BDEV_TEST_TARGET chimaera_safe_bdev_chimod_tests)
set(SAFE_BDEV_TEST_SOURCES
  test_safe_bdev_chimod.cc
)

add_executable(${SAFE_BDEV_TEST_TARGET} ${SAFE_BDEV_TEST_SOURCES})

target_include_directories(${SAFE_BDEV_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include      # For admin tasks
  ${CLIO_RUN_ROOT}/modules/bdev/include       # For reused bdev tasks/client
  ${CLIO_RUN_ROOT}/modules/safe-bdev/include  # For safe_bdev tasks and client
)

target_link_libraries(${SAFE_BDEV_TEST_TARGET}
  clio_admin_runtime
  clio_admin_client
  clio_bdev_runtime
  clio_bdev_client
  clio_safe_bdev_runtime
  clio_safe_bdev_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${SAFE_BDEV_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Concurrent data-plane regression test (issue #909). Its own binary because
# reproducing the race needs more workers than the default 4: with 4, only two
# are general-purpose and every data-plane task lands on the same one, so the
# container is never entered concurrently. The ctest entry below raises
# CLIO_NUM_THREADS for this binary alone, leaving the other suites untouched.
set(SAFE_BDEV_CC_TEST_TARGET clio_safe_bdev_concurrency_tests)
add_executable(${SAFE_BDEV_CC_TEST_TARGET} test_safe_bdev_concurrency.cc)

target_include_directories(${SAFE_BDEV_CC_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include
  ${CLIO_RUN_ROOT}/modules/bdev/include
  ${CLIO_RUN_ROOT}/modules/safe-bdev/include
)

target_link_libraries(${SAFE_BDEV_CC_TEST_TARGET}
  clio_admin_runtime
  clio_admin_client
  clio_bdev_runtime
  clio_bdev_client
  clio_safe_bdev_runtime
  clio_safe_bdev_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${SAFE_BDEV_CC_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS ${SAFE_BDEV_CC_TEST_TARGET}
  RUNTIME DESTINATION bin
)

# Compose-time config tests. CreateParams::LoadConfig is header-only and the
# only safe-bdev entry point a deployment's YAML reaches, yet the suites above
# build their CreateParams in C++ and never parse a line of it. Its own binary
# for the same reason as the EC tests: no daemon, no shared memory, so it runs
# in milliseconds anywhere.
set(SAFE_BDEV_CONFIG_TEST_TARGET clio_safe_bdev_config_tests)
add_executable(${SAFE_BDEV_CONFIG_TEST_TARGET} test_safe_bdev_config.cc)

target_include_directories(${SAFE_BDEV_CONFIG_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include
  ${CLIO_RUN_ROOT}/modules/bdev/include
  ${CLIO_RUN_ROOT}/modules/safe-bdev/include
)

# Client-only: CreateParams lives in the header, so nothing here needs the
# runtime half of the chimod.
target_link_libraries(${SAFE_BDEV_CONFIG_TEST_TARGET}
  clio_safe_bdev_client
  clio_bdev_client
  clio_admin_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${SAFE_BDEV_CONFIG_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS ${SAFE_BDEV_CONFIG_TEST_TARGET}
  RUNTIME DESTINATION bin
)

# Erasure-coding core tests. These are daemon-independent: the EC core
# (GF(2^8), Reed-Solomon, EcArray) is header-only and pulls in no Chimaera
# runtime, so this target links nothing but the test harness. Keeping it
# separate from the runtime smoke test makes the add/remove/recover coverage
# fast and deterministic.
set(SAFE_BDEV_EC_TEST_TARGET chimaera_safe_bdev_ec_tests)
add_executable(${SAFE_BDEV_EC_TEST_TARGET} test_safe_bdev_ec.cc)
target_include_directories(${SAFE_BDEV_EC_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/test                        # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/safe-bdev/include   # For the EC core headers
)
# simple_test.h pulls in a clio_ctp header; ctp::cxx supplies those include
# dirs (and the few symbols simple_test.h's main references).
target_link_libraries(${SAFE_BDEV_EC_TEST_TARGET}
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(${SAFE_BDEV_EC_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
install(TARGETS ${SAFE_BDEV_EC_TEST_TARGET}
  RUNTIME DESTINATION bin
)

if(CLIO_CORE_ENABLE_TESTS)
  add_test(
    NAME cr_all_safe_bdev_tests
    COMMAND ${SAFE_BDEV_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_safe_bdev_tests PROPERTIES
    TIMEOUT 180
    LABELS "msan_skip"
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_safe_bdev_ec_tests
    COMMAND ${SAFE_BDEV_EC_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_safe_bdev_ec_tests PROPERTIES TIMEOUT 60)

  add_test(
    NAME cr_safe_bdev_config_tests
    COMMAND ${SAFE_BDEV_CONFIG_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_safe_bdev_config_tests PROPERTIES
    TIMEOUT 60
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
  )

  # CLIO_NUM_THREADS=16 is load-bearing: at the default 4 the scheduler places
  # every data-plane task on the same general-purpose worker and the race can
  # never occur. Distinct CLIO_PORT so it can run alongside the other suite.
  add_test(
    NAME cr_safe_bdev_concurrency_tests
    COMMAND ${SAFE_BDEV_CC_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_safe_bdev_concurrency_tests PROPERTIES
    TIMEOUT 180
    LABELS "msan_skip"
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10501;CLIO_NUM_THREADS=16"
  )
endif()

install(TARGETS ${SAFE_BDEV_TEST_TARGET}
  RUNTIME DESTINATION bin
)

message(STATUS "Safe-bdev module tests configured:")
message(STATUS "  Test target: ${SAFE_BDEV_TEST_TARGET}")

# Membership stress (grow 1 -> 2 -> 4 data disks, unplug, recover, with and
# without concurrent I/O). File members stand in for disks; their backing
# files go under $CLIO_SAFE_STRESS_DIR.
set(SAFE_BDEV_STRESS_TARGET clio_safe_bdev_stress_tests)
add_executable(${SAFE_BDEV_STRESS_TARGET} test_safe_bdev_stress.cc)
target_include_directories(${SAFE_BDEV_STRESS_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test
  ${CLIO_RUN_ROOT}/modules/admin/include
  ${CLIO_RUN_ROOT}/modules/bdev/include
  ${CLIO_RUN_ROOT}/modules/safe-bdev/include
)
target_link_libraries(${SAFE_BDEV_STRESS_TARGET}
  clio_admin_runtime
  clio_admin_client
  clio_bdev_runtime
  clio_bdev_client
  clio_safe_bdev_runtime
  clio_safe_bdev_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(${SAFE_BDEV_STRESS_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
if(CLIO_CORE_ENABLE_TESTS)
  add_test(
    NAME cr_safe_bdev_stress_tests
    COMMAND ${SAFE_BDEV_STRESS_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_safe_bdev_stress_tests PROPERTIES
    TIMEOUT 600
    LABELS "msan_skip"
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10502;CLIO_NUM_THREADS=16"
  )
endif()

# Distributed driver (not a ctest: it needs a live multi-node runtime and a
# harness that kills a node; see test/integration/safe-bdev/dist_stress.py).
add_executable(clio_safe_bdev_dist_stress test_safe_bdev_dist_stress.cc)
target_include_directories(clio_safe_bdev_dist_stress PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test
  ${CLIO_RUN_ROOT}/modules/admin/include
  ${CLIO_RUN_ROOT}/modules/bdev/include
  ${CLIO_RUN_ROOT}/modules/safe-bdev/include
)
target_link_libraries(clio_safe_bdev_dist_stress
  clio_admin_client
  clio_bdev_client
  clio_safe_bdev_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(clio_safe_bdev_dist_stress PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Member disks that fail at RUNTIME (issue #1117): the file bdev's test-only
# "<file>.fail" marker kills a member mid-run; the array must fault it, serve
# every byte degraded, keep it faulty across a restart, rebuild onto a fresh
# disk, and past max_failures return errors -- never wrong bytes. The array is
# composed from YAML with `parity: true` members (4 data + 2 parity).
set(SAFE_BDEV_DISK_FAIL_TARGET clio_safe_bdev_disk_fail_tests)
add_executable(${SAFE_BDEV_DISK_FAIL_TARGET} test_safe_bdev_disk_fail.cc)
target_include_directories(${SAFE_BDEV_DISK_FAIL_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test
  ${CLIO_RUN_ROOT}/modules/admin/include
  ${CLIO_RUN_ROOT}/modules/bdev/include
  ${CLIO_RUN_ROOT}/modules/safe-bdev/include
)
target_link_libraries(${SAFE_BDEV_DISK_FAIL_TARGET}
  clio_admin_runtime
  clio_admin_client
  clio_bdev_runtime
  clio_bdev_client
  clio_safe_bdev_runtime
  clio_safe_bdev_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(${SAFE_BDEV_DISK_FAIL_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
install(TARGETS ${SAFE_BDEV_DISK_FAIL_TARGET}
  RUNTIME DESTINATION bin
)
if(CLIO_CORE_ENABLE_TESTS)
  add_test(
    NAME cr_safe_bdev_disk_fail_tests
    COMMAND ${SAFE_BDEV_DISK_FAIL_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_safe_bdev_disk_fail_tests PROPERTIES
    TIMEOUT 300
    LABELS "msan_skip"
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHIMAERA_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10503"
  )
endif()
