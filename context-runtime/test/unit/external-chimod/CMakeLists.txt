cmake_minimum_required(VERSION 3.20)
project(external_chimod_test)

# Set C++ standard
set(CMAKE_CXX_STANDARD_20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required CLIO Runtime packages
find_package(clio_run REQUIRED)
find_package(clio_run_admin REQUIRED)

# ClioCoreCommon.cmake utilities are included via clio_run package

# Add modules directory
add_subdirectory(modules/simple_mod)

# Create test executable
add_executable(test_simple_mod test_simple_mod.cc)

# Link against required libraries
target_link_libraries(test_simple_mod
  external_test::simple_mod_client        # Our ChiMod client
  clio::run::admin_client                  # Admin client (required) 
  ${CMAKE_THREAD_LIBS_INIT}               # Threading support
)

# Include directories
target_include_directories(test_simple_mod PUBLIC
  ${CLIO_RUN_ROOT}/modules/simple_mod/include
)

# Enable testing
enable_testing()
add_test(NAME external_chimod_test COMMAND test_simple_mod)