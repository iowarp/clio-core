# Multi-node MPI producer/consumer test (issue #714).
#
# Builds two small MPI binaries — one driving the CTE core client, one the CFS
# filesystem client — each run as `mpirun -np 2` across a 2-node Docker cluster
# (rank 0 producer, rank 1 consumer). They reproduce the cross-node blob-data
# gap: a blob/file written on node A must be readable on node B. Requires MPI
# (CLIO_CORE_ENABLE_MPI=ON) and the CFS adapter for the CFS variant.

cmake_minimum_required(VERSION 3.10)

if(NOT CLIO_CORE_ENABLE_MPI)
    message(STATUS "CTE mpi_producer_consumer test skipped (CLIO_CORE_ENABLE_MPI=OFF)")
    return()
endif()
if(NOT CLIO_CTE_ENABLE_FILESYSTEM)
    message(STATUS "CTE mpi_producer_consumer test skipped (CLIO_CTE_ENABLE_FILESYSTEM=OFF)")
    return()
endif()

set(PC_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Both binaries land in build/bin/ so the docker-compose (which mounts the
# workspace) can invoke /workspace/build/bin/<binary>.
add_executable(test_cte_producer_consumer test_cte_producer_consumer.cc)
set_target_properties(test_cte_producer_consumer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(test_cte_producer_consumer
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ctp::mpi
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(test_cte_producer_consumer clio_run)

add_executable(test_cfs_producer_consumer test_cfs_producer_consumer.cc)
set_target_properties(test_cfs_producer_consumer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(test_cfs_producer_consumer
    clio_cte_filesystem_client
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ctp::mpi
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(test_cfs_producer_consumer clio_run clio_cte_filesystem_runtime)

# Docker-based ctest: spins up the 2-node cluster and runs mpirun across it.
add_test(
    NAME cte_mpi_producer_consumer_integration_docker
    COMMAND ${PC_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${PC_TEST_DIR}
)
set_tests_properties(cte_mpi_producer_consumer_integration_docker PROPERTIES
    LABELS "integration;docker;distributed;mpi;cfs"
    TIMEOUT 600  # 10-minute cap for the Docker-based cross-node MPI test
)

message(STATUS "CTE mpi_producer_consumer integration test configured (issue #714)")
message(STATUS "  Run with: ctest -R cte_mpi_producer_consumer_integration_docker")
