# CMakeLists.txt for the CTE 4-node memory-coherency integration test.
cmake_minimum_required(VERSION 3.10)

# Docker + docker-compose 4-node cluster: every node runs test_cte_coherency
# as a client against its local daemon. Prerequisite binaries (built before
# ctest runs): clio_run, test_cte_coherency.
set(COHERENCY_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_test(
    NAME cte_coherency_integration_docker
    COMMAND ${COHERENCY_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${COHERENCY_TEST_DIR}
)

set_tests_properties(cte_coherency_integration_docker PROPERTIES
    LABELS "integration;docker;distributed;coherency"
    TIMEOUT 900  # 15 min: 4-container bring-up + barriers + teardown
)

message(STATUS "CTE coherency integration test configured")
message(STATUS "  Test directory: ${COHERENCY_TEST_DIR}")
message(STATUS "  Run with: ctest -R cte_coherency_integration_docker")
