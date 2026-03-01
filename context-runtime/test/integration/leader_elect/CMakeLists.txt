# CMakeLists.txt for Chimaera Leader Election Integration Tests
cmake_minimum_required(VERSION 3.10)

# These tests require Docker and docker-compose
# They spin up a 4-node cluster and test leader shutdown, failover, and restart:
# Phase 1: Client on node 1 shuts down local runtime, fails over to another node
# Phase 2: Node 1 runtime restarted externally, client re-inits and verifies health

# Build the test binary
set(LEADER_ELECT_TEST_TARGET chimaera_leader_elect_tests)

add_executable(${LEADER_ELECT_TEST_TARGET}
  ${CMAKE_CURRENT_SOURCE_DIR}/test_leader_elect.cc
)

target_include_directories(${LEADER_ELECT_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include
)

target_link_libraries(${LEADER_ELECT_TEST_TARGET}
  chimaera_admin_runtime
  chimaera_admin_client
  chimaera_MOD_NAME_runtime
  chimaera_MOD_NAME_client
  hshm::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${LEADER_ELECT_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Get the directory containing the test scripts
set(LEADER_ELECT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Register ctest
add_test(
    NAME cr_leader_elect_integration
    COMMAND ${LEADER_ELECT_TEST_DIR}/run_tests.sh all
    WORKING_DIRECTORY ${LEADER_ELECT_TEST_DIR}
)

set_tests_properties(cr_leader_elect_integration PROPERTIES
    LABELS "integration;docker;leader_elect"
    TIMEOUT 600  # 10 minute timeout for Docker-based tests
    ENVIRONMENT "NUM_NODES=4;TEST_FILTER=[leader_elect]"
)

message(STATUS "Chimaera leader election integration test configured")
message(STATUS "  Test directory: ${LEADER_ELECT_TEST_DIR}")
message(STATUS "  Run with: ctest -L integration")
