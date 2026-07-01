# CMakeLists.txt for the collective-restart reproducer (issue #628).
#
# A cross-node collective must not hang forever when a participant is restarted
# mid-flight and never declared dead by SWIM. run_tests.sh:
#   1. starts a 4-node Docker cluster (SWIM deliberately slow)
#   2. broadcasts a long-hold CoMutexTest
#   3. restarts node 4 while the replicas are holding
#   4. asserts the broadcast RETURNS (QueryTaskProgress completes the origin)
#      rather than hanging.
cmake_minimum_required(VERSION 3.10)

set(COLLECTIVE_RESTART_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_test(
    NAME cr_collective_restart_integration_docker
    COMMAND ${COLLECTIVE_RESTART_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${COLLECTIVE_RESTART_TEST_DIR}
)

set_tests_properties(cr_collective_restart_integration_docker PROPERTIES
    LABELS "integration;docker;collective_restart"
    TIMEOUT 300
)

message(STATUS "Clio collective-restart reproducer configured (issue #628)")
