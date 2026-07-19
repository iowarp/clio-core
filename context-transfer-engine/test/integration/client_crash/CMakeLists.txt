# CMakeLists.txt for the CTE client-crash integration test (#722 / client dies
# mid-op). Requires Docker + docker-compose: it spins up a daemon and a
# separate-container external client that crashes mid-PutBlob, then asserts the
# runtime survives, stays responsive, and does not runaway re-queue.
cmake_minimum_required(VERSION 3.10)

set(CLIENT_CRASH_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# The driver container invokes the crash-client mode of this binary; make sure
# it (and the daemon) are built before the docker test runs.
add_test(
    NAME cte_client_crash_integration_docker
    COMMAND ${CLIENT_CRASH_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${CLIENT_CRASH_TEST_DIR}
)

set_tests_properties(cte_client_crash_integration_docker PROPERTIES
    LABELS "integration;docker;distributed;crash;leak"
    TIMEOUT 600
)

message(STATUS "CTE client-crash integration test configured (issue #722)")
message(STATUS "  Run with: ctest -R cte_client_crash_integration_docker")
