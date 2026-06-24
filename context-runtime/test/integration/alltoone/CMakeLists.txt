# CMakeLists.txt for CLIO Runtime AllToOne Collective Integration Test
cmake_minimum_required(VERSION 3.10)

# This test requires Docker and docker-compose. It spins up a 4-node cluster
# and exercises PoolQuery::AllToOne: a client issues one AllToOne request per
# container in the pool; the neighborhood leader holds them in the BatchManager
# until tasks from ALL containers have arrived (the count-based barrier), then
# runs the aggregate once and broadcasts the result back to every submitter.

set(ALLTOONE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_test(
    NAME cr_alltoone_integration_docker
    COMMAND ${ALLTOONE_TEST_DIR}/run_tests.sh all
    WORKING_DIRECTORY ${ALLTOONE_TEST_DIR}
)

# The test binary reads CLIO_NUM_CONTAINERS to decide how many AllToOne requests
# to issue (one per node), which must equal the pool's container count so the
# barrier releases. NUM_NODES drives the cluster size.
set_tests_properties(cr_alltoone_integration_docker PROPERTIES
    LABELS "integration;docker;alltoone;collective"
    TIMEOUT 600  # 10 minute timeout for Docker-based tests
    ENVIRONMENT "NUM_NODES=4;TEST_FILTER=[alltoone]"
)

message(STATUS "Clio AllToOne integration test configured")
message(STATUS "  Test directory: ${ALLTOONE_TEST_DIR}")
message(STATUS "  Run with: ctest -R cr_alltoone_integration_docker")
