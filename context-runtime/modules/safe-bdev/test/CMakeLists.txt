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
endif()

install(TARGETS ${SAFE_BDEV_TEST_TARGET}
  RUNTIME DESTINATION bin
)

message(STATUS "Safe-bdev module tests configured:")
message(STATUS "  Test target: ${SAFE_BDEV_TEST_TARGET}")
