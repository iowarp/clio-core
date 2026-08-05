# Multi-node GPU blob-exchange test — one rank per node, one GPU per rank.
#
# Builds a single MPI+CUDA binary that attaches to an already-running Clio
# cluster (one daemon per node) and runs a GPU-produced / GPU-verified blob
# ring exchange across it. There is no add_test() here on purpose: the test
# needs a real multi-node allocation, which ctest cannot provision. Drive it
# from a batch script — clio-core-xnode-gpu-8node.slurm is the reference —
# with `mpirun -np <nodes> --map-by ppr:1:node .../test_cte_xnode_gpu`.
#
# The target still builds in every CUDA+MPI configuration so the code cannot
# rot unnoticed between cluster runs.

cmake_minimum_required(VERSION 3.10)

if(NOT CLIO_CORE_ENABLE_MPI)
    message(STATUS "CTE xnode_gpu test skipped (CLIO_CORE_ENABLE_MPI=OFF)")
    return()
endif()
if(NOT CLIO_CORE_ENABLE_CUDA)
    message(STATUS "CTE xnode_gpu test skipped (CLIO_CORE_ENABLE_CUDA=OFF)")
    return()
endif()

add_cuda_executable(test_cte_xnode_gpu FALSE
    test_cte_xnode_gpu.cc
    INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/context-runtime/include
        # core_client.h reaches into the bdev transport headers for the shm
        # metadata cache, and pulls admin types in transitively.
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    LINK_LIBS
        clio_cte_core_client
        clio_admin_client
        clio_bdev_client
        clio_run_cxx
        ctp::cuda_cxx
        ctp::mpi
        ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(test_cte_xnode_gpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# The batch script launches this binary against daemons started from the same
# build tree, so keep clio_run in step with it.
add_dependencies(test_cte_xnode_gpu clio_run)

install(TARGETS test_cte_xnode_gpu RUNTIME DESTINATION bin)

message(STATUS "CTE xnode_gpu multi-node GPU test configured (no ctest entry: "
               "needs a multi-node allocation)")
