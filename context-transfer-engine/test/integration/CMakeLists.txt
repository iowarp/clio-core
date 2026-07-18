# CMakeLists.txt for CTE Integration Tests
cmake_minimum_required(VERSION 3.10)

# Restart integration test (no Docker required)
add_subdirectory(restart)

# Distributed integration tests require Docker
if(CLIO_CORE_ENABLE_DOCKER_CI)
    message(STATUS "CTE integration tests enabled (Docker CI)")
    add_subdirectory(distributed)
    add_subdirectory(cfs_distributed)  # #685: cross-node CFS namespace test
    add_subdirectory(mpi_producer_consumer)  # #714: cross-node blob data (MPI)
    add_subdirectory(client_crash)  # #722: client dies mid-PutBlob (runtime robustness)
    add_subdirectory(jarvis_deploy)

    # FUSE integration test requires Docker with FUSE support
    if(CLIO_CTE_ENABLE_FUSE_ADAPTER)
        add_subdirectory(fuse)
    endif()
else()
    message(STATUS "CTE distributed integration tests disabled (set CLIO_CORE_ENABLE_DOCKER_CI=ON to enable)")
endif()
