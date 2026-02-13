# CMakeLists.txt for CTE Integration Tests
cmake_minimum_required(VERSION 3.10)

# Restart integration test (no Docker required)
add_subdirectory(restart)

# Distributed integration tests require Docker
if(WRP_CORE_ENABLE_DOCKER_CI)
    message(STATUS "CTE integration tests enabled (Docker CI)")
    add_subdirectory(distributed)
else()
    message(STATUS "CTE distributed integration tests disabled (set WRP_CORE_ENABLE_DOCKER_CI=ON to enable)")
endif()
