cmake_minimum_required(VERSION 3.10)


# Create client library
add_chimod_client(
  SOURCES
    src/admin_client.cc
)

# Create runtime library
add_chimod_runtime(
  SOURCES
    src/admin_runtime.cc
    src/autogen/admin_lib_exec.cc
)

# Add unit tests subdirectory
if(WRP_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()