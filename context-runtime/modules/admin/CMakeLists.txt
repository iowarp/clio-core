cmake_minimum_required(VERSION 3.10)


# Create client library
# LIB_NAME pins the library to `chimaera_admin_*` so changing the
# C++ namespace (chimaera -> clio_run) doesn't cascade into renaming
# this load-bearing .so. See deprecation-notes.md.
add_clio_module_client(
  LIB_NAME clio_admin
  SOURCES
    src/admin_client.cc
)
# Legacy alias so out-of-tree CMake consumers that still reference the
# old `chimaera_admin_client` target keep linking.
add_library(chimaera_admin_client ALIAS clio_admin_client)

# Create runtime library (see client block above for LIB_NAME rationale).
add_clio_module_runtime(
  LIB_NAME clio_admin
  SOURCES
    src/admin_runtime.cc
    src/autogen/admin_lib_exec.cc
)
add_library(chimaera_admin_runtime ALIAS clio_admin_runtime)

# Coverage builds: the daemon dies via std::abort() in InitiateShutdown,
# which skips the gcov at-exit flush and silently discards all runtime-side
# coverage from daemon-based tests. Let admin_runtime.cc dump counters
# explicitly right before aborting.
if(CLIO_CORE_ENABLE_COVERAGE)
  target_compile_definitions(clio_admin_runtime PRIVATE CLIO_COVERAGE=1)
endif()

# Add unit tests subdirectory
if(CLIO_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()