# CMakeLists.txt for Chimaera Migrate Integration Tests
cmake_minimum_required(VERSION 3.10)

# These tests require Docker and docker-compose
# They spin up a 4-node cluster and test container migration with retry queue

# Get the directory containing the test scripts
set(MIGRATE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add integration test that runs the migrate test suite
add_test(
    NAME cr_migrate_integration
    COMMAND ${MIGRATE_TEST_DIR}/run_tests.sh all
    WORKING_DIRECTORY ${MIGRATE_TEST_DIR}
)

# Set test properties
set_tests_properties(cr_migrate_integration PROPERTIES
    LABELS "integration;docker;migrate"
    TIMEOUT 600  # 10 minute timeout for Docker-based tests
    ENVIRONMENT "NUM_NODES=4;TEST_FILTER=[migrate]"
)

message(STATUS "Chimaera migrate integration test configured")
message(STATUS "  Test directory: ${MIGRATE_TEST_DIR}")
message(STATUS "  Run with: ctest -L integration")
