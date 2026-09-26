# Collective latency comparison: Clio PoolQuery::AllToOne vs MPI.
#
# Builds one MPI binary run as `mpirun -np 4` across a 4-node Docker cluster
# (one rank per node, each attached to its own local clio daemon). It times
# MPI_Barrier, MPI_Allreduce, and the MOD_NAME BarrierTask/AllReduceTask routed
# AllToOne, and reports how our collectives compare to MPI's.
#
# Requires MPI: the reference arms ARE MPI, so there is nothing to measure
# against without it.

cmake_minimum_required(VERSION 3.10)

if(NOT CLIO_CORE_ENABLE_MPI)
    message(STATUS "Clio collective_bench skipped (CLIO_CORE_ENABLE_MPI=OFF)")
    return()
endif()

set(COLL_BENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Lands in build/bin/ so docker-compose (which mounts the workspace) can invoke
# /workspace/build/bin/clio_collective_bench.
add_executable(clio_collective_bench collective_bench.cc)
set_target_properties(clio_collective_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(clio_collective_bench
    clio_MOD_NAME_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ctp::mpi
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(clio_collective_bench clio_run clio_MOD_NAME_runtime)

add_test(
    NAME cr_collective_bench_docker
    COMMAND ${COLL_BENCH_DIR}/run_tests.sh all
    WORKING_DIRECTORY ${COLL_BENCH_DIR}
)
set_tests_properties(cr_collective_bench_docker PROPERTIES
    LABELS "integration;docker;collective;mpi;benchmark"
    TIMEOUT 900  # 15 minutes: cluster bring-up plus four timed arms
    # Keep the ctest run short; the harness default is larger for a real
    # measurement run (see run_tests.sh / README.md).
    # The default 4 verification rounds run too, so this entry gates barrier
    # SEMANTICS (nobody exits before the last arrival, checked against a
    # negative control) and not just latency.
    ENVIRONMENT "NUM_NODES=4;COLL_BENCH_ITERS=200;COLL_BENCH_WARMUP=20"
)

message(STATUS "Clio collective_bench integration test configured")
message(STATUS "  Run with: ctest -R cr_collective_bench_docker")
