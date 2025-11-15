cmake_minimum_required(VERSION 3.20)
project(external_chimod_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Unified IOWarp Core package - includes HermesShm, Chimaera, and build utilities
find_package(iowarp-core REQUIRED)
# This automatically provides:
#   - All hshm::* modular targets (hshm::cxx, hshm::configure, hshm::serialize, etc.)
#   - chimaera::cxx (core runtime library)
#   - chimaera::admin_client and chimaera::admin_runtime
#   - ChiMod build utilities

# Create test executable
add_executable(test_external_chimod test_external_chimod.cc)

# Link against required libraries
target_link_libraries(test_external_chimod
  chimaera::admin_client                  # Admin client (required, includes chimaera::cxx)
  ${CMAKE_THREAD_LIBS_INIT}               # Threading support
)

message(STATUS "External ChiMod test configured successfully")
message(STATUS "  Target: test_external_chimod")
message(STATUS "  Links to: chimaera::admin_client (includes chimaera::cxx automatically)")

# Enable testing
enable_testing()
add_test(NAME external_chimod_test COMMAND test_external_chimod)