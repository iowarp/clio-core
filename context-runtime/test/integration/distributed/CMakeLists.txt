# CMakeLists.txt for CLIO Runtime Distributed Integration Tests
cmake_minimum_required(VERSION 3.10)

# These tests require Docker and docker-compose
# They spin up a multi-node cluster and run distributed tests

# Get the directory containing the test scripts
set(DISTRIBUTED_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add integration test that runs the distributed test suite
add_test(
    NAME cr_distributed_integration_docker
    COMMAND ${DISTRIBUTED_TEST_DIR}/run_tests.sh all
    WORKING_DIRECTORY ${DISTRIBUTED_TEST_DIR}
)

# Set test properties
set_tests_properties(cr_distributed_integration_docker PROPERTIES
    LABELS "integration;docker;distributed"
    TIMEOUT 600  # 10 minute timeout for Docker-based tests
    ENVIRONMENT "NUM_NODES=4;TEST_FILTER=bdev_file_explicit_backend"
    # TEST_FILTER matches all three mode variants:
    # bdev_file_explicit_backend_shm, _tcp, _ipc
)

message(STATUS "Clio distributed integration test configured")
message(STATUS "  Test directory: ${DISTRIBUTED_TEST_DIR}")
message(STATUS "  Run with: ctest -L integration")
