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
    src/safe_bdev_journal.cc
    src/safe_bdev_viz.cc
    src/autogen/safe_bdev_lib_exec.cc
  LINK_LIBRARIES
    clio_admin_runtime
    clio_bdev_runtime
    clio_bdev_client
)
add_library(chimaera_safe_bdev_runtime ALIAS clio_safe_bdev_runtime)

# Web dashboard page for this module (issue #990): mounted at
# /viz/clio_safe_bdev by the runtime as soon as a safe-bdev pool comes up.
clio_add_viz_assets(MOD_NAME clio_safe_bdev)

# Unit tests
if(CLIO_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()

# Operator tool: rebuild a failed member onto a replacement disk of a running
# array (RecoverBdev); used by clio_fs_suite's disk-replacement test.
add_executable(clio_safe_bdev_recover util/safe_bdev_recover.cc)
target_link_libraries(clio_safe_bdev_recover
  clio_safe_bdev_client
  clio_bdev_client
  clio_admin_client
  ctp::cxx
)
set_target_properties(clio_safe_bdev_recover PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
install(TARGETS clio_safe_bdev_recover RUNTIME DESTINATION bin)
