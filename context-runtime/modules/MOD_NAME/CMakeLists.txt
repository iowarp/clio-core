cmake_minimum_required(VERSION 3.10)


# Create client library
add_chimod_client(
  SOURCES
    src/MOD_NAME_client.cc
  INCLUDE_DIRECTORIES
    ${CHIMAERA_ROOT}/modules/admin/include  # MOD_NAME needs admin headers
)

# Create runtime library
add_chimod_runtime(
  SOURCES
    src/MOD_NAME_runtime.cc
    src/autogen/MOD_NAME_lib_exec.cc
  INCLUDE_DIRECTORIES
    ${CHIMAERA_ROOT}/modules/admin/include  # MOD_NAME needs admin headers
)

# Add unit tests subdirectory
if(WRP_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()