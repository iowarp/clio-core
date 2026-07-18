cmake_minimum_required(VERSION 3.10)

# Client library.
# LIB_NAME pins the library to `clio_safe_bdev_client`; the ALIAS keeps the
# brand-aligned `chimaera_safe_bdev_client` target name available for any
# in-tree or out-of-tree consumer that prefers it.
add_clio_module_client(
  LIB_NAME clio_safe_bdev
  SOURCES
    src/safe_bdev_client.cc
  LINK_LIBRARIES
    clio_admin_client
    clio_bdev_client
)
add_library(chimaera_safe_bdev_client ALIAS clio_safe_bdev_client)

# Runtime library (see client block above for the LIB_NAME + ALIAS rationale).
add_clio_module_runtime(
  LIB_NAME clio_safe_bdev
  SOURCES
    src/safe_bdev_runtime.cc
    src/autogen/safe_bdev_lib_exec.cc
  LINK_LIBRARIES
    clio_admin_runtime
    clio_bdev_runtime
    clio_bdev_client
)
add_library(chimaera_safe_bdev_runtime ALIAS clio_safe_bdev_runtime)

# Unit tests
if(CLIO_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()
