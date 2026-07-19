# CTE distributed filesystem (CFS) correctness test (issue #685).
#
# Builds a small writer/reader binary and registers a Docker-based ctest that
# spins up a 2-node cluster: node 1 writes a CFS file, node 2 must read it back
# cross-node. Proves the CFS tag namespace is shared cluster-wide (not per-node).

cmake_minimum_required(VERSION 3.10)

# The CFS adapter must be built for this test to be meaningful.
if(NOT CLIO_CTE_ENABLE_FILESYSTEM)
    message(STATUS "CTE cfs_distributed test skipped (CLIO_CTE_ENABLE_FILESYSTEM=OFF)")
    return()
endif()

# Writer/reader binary — lands in build/bin/ so the docker-compose (which mounts
# the workspace) can invoke /workspace/build/bin/test_cfs_distributed.
add_executable(test_cfs_distributed test_cfs_distributed.cc)
set_target_properties(test_cfs_distributed PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(test_cfs_distributed
    clio_cte_filesystem_client
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(test_cfs_distributed clio_run clio_cte_filesystem_runtime)

set(CFS_DIST_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_test(
    NAME cte_cfs_distributed_integration_docker
    COMMAND ${CFS_DIST_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${CFS_DIST_TEST_DIR}
)
set_tests_properties(cte_cfs_distributed_integration_docker PROPERTIES
    LABELS "integration;docker;distributed;cfs"
    TIMEOUT 480  # 8-minute cap for the Docker-based cross-node test
)

message(STATUS "CTE cfs_distributed integration test configured (issue #685)")
message(STATUS "  Run with: ctest -R cte_cfs_distributed_integration_docker")
