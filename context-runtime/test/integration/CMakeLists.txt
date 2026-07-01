# CMakeLists.txt for CLIO Runtime Integration Tests
cmake_minimum_required(VERSION 3.10)

# Integration tests require Docker
if(CLIO_CORE_ENABLE_DOCKER_CI)
    message(STATUS "Clio integration tests enabled (Docker CI)")
    add_subdirectory(distributed)
    add_subdirectory(expand)
    add_subdirectory(migrate)
    add_subdirectory(recovery)
    add_subdirectory(reconnect)
    add_subdirectory(leader_elect)
    add_subdirectory(alltoone)
    add_subdirectory(collective_restart)
else()
    message(STATUS "Clio integration tests disabled (set CLIO_CORE_ENABLE_DOCKER_CI=ON to enable)")
endif()
