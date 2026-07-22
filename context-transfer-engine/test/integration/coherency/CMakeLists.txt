# CMakeLists.txt for the CTE multi-node coherency integration test.
#
# Docker-based: spins up a 4-container cluster and runs test_cte_coherency on
# every node (write-only / read-only / append-only). Portable counterpart to
# test/integration/distributed_slurm, which requires Slurm + Apptainer.
cmake_minimum_required(VERSION 3.10)

set(COHERENCY_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_test(
    NAME cte_coherency_integration_docker
    COMMAND ${COHERENCY_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${COHERENCY_TEST_DIR}
)

set_tests_properties(cte_coherency_integration_docker PROPERTIES
    LABELS "integration;docker;distributed;coherency"
    TIMEOUT 600
)

message(STATUS "CTE coherency integration test configured")
message(STATUS "  Test directory: ${COHERENCY_TEST_DIR}")
message(STATUS "  Run with: ctest -R cte_coherency_integration_docker")
