# CMakeLists.txt for CTE Integration Tests
cmake_minimum_required(VERSION 3.10)

# Restart integration test (no Docker required)
add_subdirectory(restart)

# Indexer restart test (#905, no Docker required): docs put through the
# indexer, runtime reboot, SemanticSearch verified against the index the
# indexer rebuilt from file-backed storage. Gated on the indexer chimod.
if(TARGET clio_cte_indexer_client)
    add_subdirectory(indexer_restart)
endif()

# Replication persistence test (#886, no Docker required): 50 CachedPuts
# (DRAM primary + disk replica), runtime reboot, disk replicas verified.
# Gated on the replication chimod (default ON).
if(TARGET clio_cte_replication_client)
    add_subdirectory(replication_persist)
endif()

# Multi-node GPU blob exchange (no Docker: driven by a batch script on a real
# multi-node GPU allocation). Self-gates on CLIO_CORE_ENABLE_MPI + _CUDA.
add_subdirectory(xnode_gpu)

# Distributed integration tests require Docker
if(CLIO_CORE_ENABLE_DOCKER_CI)
    message(STATUS "CTE integration tests enabled (Docker CI)")
    add_subdirectory(distributed)
    # #886: 4-node cache-coherence suite (needs cache + replication chimods)
    if(TARGET clio_cte_cache_client AND TARGET clio_cte_replication_client)
        add_subdirectory(coherence)
    endif()
    add_subdirectory(cfs_distributed)  # #685: cross-node CFS namespace test
    add_subdirectory(mpi_producer_consumer)  # #714: cross-node blob data (MPI)
    add_subdirectory(client_crash)  # #722: client dies mid-PutBlob (runtime robustness)
    add_subdirectory(grayscott_congestion)  # #774: L=512 congestion-collapse repro
    add_subdirectory(jarvis_deploy)

    # FUSE integration test requires Docker with FUSE support
    if(CLIO_CTE_ENABLE_FUSE_ADAPTER)
        add_subdirectory(fuse)
    endif()
else()
    message(STATUS "CTE distributed integration tests disabled (set CLIO_CORE_ENABLE_DOCKER_CI=ON to enable)")
endif()
