# "Clio as a cache" multi-node integration test (issue #1015).
#
# Builds one MPI binary run as `mpirun -np 8` over a 2-node Docker cluster with
# no daemon on either node: every rank calls CLIO_INIT with CLIO_WITH_RUNTIME=1,
# one rank per node must start the runtime and the rest must attach to it, then
# all eight PutBlob/GetBlob through the CTE. Requires MPI and the CTE.

cmake_minimum_required(VERSION 3.10)

if(NOT CLIO_CORE_ENABLE_MPI)
    message(STATUS "with_runtime integration test skipped (CLIO_CORE_ENABLE_MPI=OFF)")
    return()
endif()
if(NOT CLIO_CORE_ENABLE_CTE)
    message(STATUS "with_runtime integration test skipped (CLIO_CORE_ENABLE_CTE=OFF)")
    return()
endif()

set(WR_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Lands in build/bin/ so the docker-compose (which mounts the workspace) can
# invoke /workspace/build/bin/test_with_runtime_cache.
add_executable(test_with_runtime_cache test_with_runtime_cache.cc)
set_target_properties(test_with_runtime_cache PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(test_with_runtime_cache
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ctp::mpi
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(test_with_runtime_cache clio_run)

add_test(
    NAME cr_with_runtime_cache_integration_docker
    COMMAND ${WR_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${WR_TEST_DIR}
)
set_tests_properties(cr_with_runtime_cache_integration_docker PROPERTIES
    LABELS "integration;docker;distributed;mpi"
    TIMEOUT 900  # 15-minute cap: two runtimes cold-start inside the MPI job
)

message(STATUS "Clio with_runtime (CLIO_WITH_RUNTIME=1 as a cache) integration test configured (issue #1015)")
message(STATUS "  Run with: ctest -R cr_with_runtime_cache_integration_docker")
