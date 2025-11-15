# External CTE Core Integration Test
# This CMakeLists.txt demonstrates how external applications can link to CTE Core
# following the MODULE_DEVELOPMENT_GUIDE.md patterns
cmake_minimum_required(VERSION 3.20)
project(cte_external_integration_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable debug build by default for testing
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Unified IOWarp Core package - includes everything
find_package(iowarp-core CONFIG REQUIRED)
# This automatically provides:
#   Core Components:
#     - All hshm::* modular targets (hshm::cxx, hshm::configure, hshm::serialize, etc.)
#     - chimaera::cxx (core runtime library)
#     - ChiMod build utilities (add_chimod_client, add_chimod_runtime, etc.)
#
#   Core ChiMods (Always Available):
#     - chimaera::admin_client, chimaera::admin_runtime
#     - chimaera::bdev_client, chimaera::bdev_runtime
#
#   Optional ChiMods (if enabled at build time):
#     - wrp_cte::core_client, wrp_cte::core_runtime (CTE - included if WRP_CORE_ENABLE_CTE=ON)
#     - wrp_cae::core_client, wrp_cae::core_runtime (CAE - included if WRP_CORE_ENABLE_CAE=ON)

# Find additional system dependencies
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Create the external integration test executable
add_executable(cte_external_test
    external_integration_test.cc
)

# Link against CTE Core libraries using modern CMake target naming
# These targets are created by the add_chimod_both() function in the core ChiMod
target_link_libraries(cte_external_test
    # CTE Core ChiMod libraries (from find_package(wrp_cte_core))
    wrp_cte::core_client                 # CTE Core client library
    wrp_cte::core_runtime                # CTE Core runtime library (if needed)
    
    # Core Chimaera framework (automatically included by ChiMod libraries)
    # chimaera::cxx                      # NOT needed - automatically included
    
    # Admin ChiMod (if needed for advanced functionality)
    # chimaera::admin_client             # Optional - uncomment if needed
)

# Set compile flags - use standard flags for external projects
target_compile_options(cte_external_test PRIVATE
    -Wall -Wextra -Wno-unused-parameter
)

# Add a custom target to build and run the test
add_custom_target(run_external_test
    COMMAND ./cte_external_test
    DEPENDS cte_external_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running external CTE integration test"
)

# Print configuration information for external project setup verification
message(STATUS "CTE External Integration Test Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "  IOWarp Core Found: ${iowarp-core_FOUND}")

# Instructions for external developers
message(STATUS "")
message(STATUS "External Developer Notes:")
message(STATUS "  This test demonstrates proper external linking to IOWarp libraries.")
message(STATUS "  Required package: iowarp-core (single unified package)")
message(STATUS "  iowarp-core automatically provides:")
message(STATUS "    - HermesShm: All hshm::* modular targets")
message(STATUS "    - Chimaera: chimaera::cxx and build utilities")
message(STATUS "    - Core ChiMods: admin, bdev (always available)")
message(STATUS "    - Optional ChiMods: CTE, CAE (if enabled at build time)")
message(STATUS "  Main targets used in this test:")
message(STATUS "    - wrp_cte::core_client (CTE ChiMod client)")
message(STATUS "    - wrp_cte::core_runtime (CTE ChiMod runtime)")
message(STATUS "")