# CMakeLists.txt for CLIO Runtime tests
cmake_minimum_required(VERSION 3.10)

# Add unit test subdirectory
add_subdirectory(unit)

# Add integration tests (Docker-based distributed tests)
add_subdirectory(integration)

message(STATUS "Clio tests configured")