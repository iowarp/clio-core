# CMakeLists.txt for Chimaera Reconnect Integration Tests
cmake_minimum_required(VERSION 3.10)

# These tests require Docker and docker-compose
# They spin up a 4-node cluster and test client failover:
# 1. Client on node 1 connects to local runtime
# 2. Client shuts down node 1's runtime via AsyncStopRuntime
# 3. Client sends a new task — WaitForServerAndReconnect fails over
# 4. Task completes on a different host from the hostfile

# Get the directory containing the test scripts
set(RECONNECT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add integration test that runs the reconnect test suite
add_test(
    NAME cr_reconnect_integration
    COMMAND ${RECONNECT_TEST_DIR}/run_tests.sh all
    WORKING_DIRECTORY ${RECONNECT_TEST_DIR}
)

# Set test properties
set_tests_properties(cr_reconnect_integration PROPERTIES
    LABELS "integration;docker;reconnect"
    TIMEOUT 600  # 10 minute timeout for Docker-based tests
    ENVIRONMENT "NUM_NODES=4;TEST_FILTER=[reconnect]"
)

message(STATUS "Chimaera reconnect integration test configured")
message(STATUS "  Test directory: ${RECONNECT_TEST_DIR}")
message(STATUS "  Run with: ctest -L integration")
