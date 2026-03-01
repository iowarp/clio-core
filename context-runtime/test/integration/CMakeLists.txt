# CMakeLists.txt for Chimaera Integration Tests
cmake_minimum_required(VERSION 3.10)

# Integration tests require Docker
if(WRP_CORE_ENABLE_DOCKER_CI)
    message(STATUS "Chimaera integration tests enabled (Docker CI)")
    add_subdirectory(distributed)
    add_subdirectory(expand)
    add_subdirectory(migrate)
    add_subdirectory(recovery)
    add_subdirectory(reconnect)
    add_subdirectory(leader_elect)
else()
    message(STATUS "Chimaera integration tests disabled (set WRP_CORE_ENABLE_DOCKER_CI=ON to enable)")
endif()
