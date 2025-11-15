cmake_minimum_required(VERSION 3.20)
project(external_chimod_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required Chimaera packages
find_package(chimaera REQUIRED)
find_package(chimaera_admin REQUIRED)

# ChimaeraCommon.cmake utilities are included via chimaera package

# Add modules directory
add_subdirectory(modules/simple_mod)

# Create test executable
add_executable(test_simple_mod test_simple_mod.cc)

# Link against required libraries
target_link_libraries(test_simple_mod
  external_test::simple_mod_client        # Our ChiMod client
  chimaera::admin_client                  # Admin client (required) 
  ${CMAKE_THREAD_LIBS_INIT}               # Threading support
)

# Include directories
target_include_directories(test_simple_mod PUBLIC
  ${CHIMAERA_ROOT}/modules/simple_mod/include
)

# Enable testing
enable_testing()
add_test(NAME external_chimod_test COMMAND test_simple_mod)