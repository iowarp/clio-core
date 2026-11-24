# External CAE Core Integration Test
# This CMakeLists.txt demonstrates how external applications can link to CAE Core
# following the module_dev_guide.md patterns
cmake_minimum_required(VERSION 3.20)
project(cae_external_integration_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD_20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable debug build by default for testing
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Unified IOWarp Core package — `clio-core` is the canonical name; the
# legacy `iowarp-core` spelling still resolves to the same Config.
find_package(clio-core CONFIG REQUIRED)
# This automatically provides:
#   Core Components:
#     - All ctp::* modular targets (ctp::cxx, ctp::configure, ctp::serialize, etc.)
#     - clio::run::cxx (core runtime library)
#     - ChiMod build utilities (add_clio_module_client, add_clio_module_runtime, etc.)
#
#   Core ChiMods (Always Available):
#     - clio::run::admin_client, clio::run::admin_runtime
#     - clio::run::bdev_client, clio::run::bdev_runtime
#
#   Optional ChiMods (if enabled at build time):
#     - clio::cte::core_client, clio::cte::core_runtime (CTE - included if CLIO_CORE_ENABLE_CTE=ON)
#     - clio::cae::core_client, clio::cae::core_runtime (CAE - included if CLIO_CORE_ENABLE_CAE=ON)

# Find additional system dependencies
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Create the external integration test executable
add_executable(cae_external_test
    external_integration_test.cc
)

# Link against CAE Core libraries using modern CMake target naming
# These targets are created by the add_chimod_both() function in the core ChiMod
target_link_libraries(cae_external_test
    # CAE Core ChiMod libraries
    clio::cae::core_client                 # CAE Core client library

    # Core CLIO Runtime framework (automatically included by ChiMod libraries)
    # clio::run::cxx                     # NOT needed - automatically included

    # Admin ChiMod (if needed for advanced functionality)
    # clio::run::admin_client            # Optional - uncomment if needed
)

# Set compile flags - use standard flags for external projects
target_compile_options(cae_external_test PRIVATE
    -Wall -Wextra -Wno-unused-parameter
)

# Add a custom target to build and run the test
add_custom_target(run_external_test
    COMMAND ./cae_external_test
    DEPENDS cae_external_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running external CAE integration test"
)

# Print configuration information for external project setup verification
message(STATUS "CAE External Integration Test Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Clio Core Found: ${clio-core_FOUND}")

# Instructions for external developers
message(STATUS "")
message(STATUS "External Developer Notes:")
message(STATUS "  This test demonstrates proper external linking to IOWarp libraries.")
message(STATUS "  Required package: clio-core (canonical; iowarp-core is a legacy alias)")
message(STATUS "  clio-core automatically provides:")
message(STATUS "    - ClioCtp: All ctp::* modular targets")
message(STATUS "    - CLIO Runtime: clio::run::cxx and build utilities")
message(STATUS "    - Core ChiMods: admin, bdev (always available)")
message(STATUS "    - Optional ChiMods: CTE, CAE (if enabled at build time)")
message(STATUS "  Main targets used in this test:")
message(STATUS "    - clio::cae::core_client (CAE ChiMod client)")
message(STATUS "    - clio::cae::core_runtime (CAE ChiMod runtime)")
message(STATUS "")
