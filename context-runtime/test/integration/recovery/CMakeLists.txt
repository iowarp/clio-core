# CMakeLists.txt for Chimaera Recovery Integration Tests
cmake_minimum_required(VERSION 3.10)

# These tests require Docker and docker-compose
# They spin up a 4-node cluster and test node failure recovery:
# 1. Pre-failure: verify tasks work on all 4 nodes
# 2. Kill node 4 (simulated by docker stop)
# 3. Wait for SWIM detection + automatic recovery
# 4. Post-failure: verify tasks targeting dead node's container are re-routed

# Get the directory containing the test scripts
set(RECOVERY_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add integration test that runs the recovery test suite
add_test(
    NAME cr_recovery_integration
    COMMAND ${RECOVERY_TEST_DIR}/run_tests.sh all
    WORKING_DIRECTORY ${RECOVERY_TEST_DIR}
)

# Set test properties
set_tests_properties(cr_recovery_integration PROPERTIES
    LABELS "integration;docker;recovery"
    TIMEOUT 600  # 10 minute timeout for Docker-based tests
    ENVIRONMENT "NUM_NODES=4;TEST_FILTER=[recovery]"
)

message(STATUS "Chimaera recovery integration test configured")
message(STATUS "  Test directory: ${RECOVERY_TEST_DIR}")
message(STATUS "  Run with: ctest -L integration")
