# CMakeLists.txt for CTE Jarvis Deployment Integration Tests
cmake_minimum_required(VERSION 3.10)

# These tests require Docker and docker-compose
# They spin up a 2-node cluster and run the Jarvis deployment flow:
# runtime start -> CTE compose -> benchmark

# Get the directory containing the test scripts
set(JARVIS_DEPLOY_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add integration test that runs the Jarvis deployment test
add_test(
    NAME cte_jarvis_deploy_integration
    COMMAND ${JARVIS_DEPLOY_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${JARVIS_DEPLOY_TEST_DIR}
)

# Set test properties
set_tests_properties(cte_jarvis_deploy_integration PROPERTIES
    LABELS "integration;docker;distributed;jarvis"
    TIMEOUT 120  # 2 minute timeout for Docker-based tests
)

message(STATUS "CTE Jarvis deployment integration test configured")
message(STATUS "  Test directory: ${JARVIS_DEPLOY_TEST_DIR}")
message(STATUS "  Run with: ctest -L integration")
