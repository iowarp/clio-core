cmake_minimum_required(VERSION 3.10)


# Create client library
# LIB_NAME pins the library to `clio_run_admin_*` so changing the
# C++ namespace (clio_run -> clio_run) doesn't cascade into renaming
# this load-bearing .so. See deprecation-notes.md.
add_clio_module_client(
  LIB_NAME clio_admin
  SOURCES
    src/admin_client.cc
)

# Create runtime library (see client block above for LIB_NAME rationale).
add_clio_module_runtime(
  LIB_NAME clio_admin
  SOURCES
    src/admin_runtime.cc
    src/autogen/admin_lib_exec.cc
)

# RegisterMemory's GPU device-memory branch (admin_runtime.cc) needs
# CTP_ENABLE_CUDA/ROCM to see ctp::GpuApi's cudaIpcOpenMemHandle wrapper --
# without it, the #if CTP_ENABLE_CUDA guard is always false in this TU and
# the branch silently reports "GPU support not compiled in" even in a CUDA
# build. clio_admin_runtime only links clio::run::cxx, whose own link to
# its GPU library is PRIVATE (deliberately, so CUDA builtins don't leak
# into unrelated g++ TUs -- see context-runtime/src/CMakeLists.txt), so
# that define never reaches here on its own. Link ctp::cuda_cxx/rocm_cxx
# directly, PRIVATE, mirroring that same containment pattern one level
# down for just this module.
if(CLIO_CORE_ENABLE_CUDA)
  target_link_libraries(clio_admin_runtime PRIVATE ctp::cuda_cxx)
elseif(CLIO_CORE_ENABLE_ROCM)
  target_link_libraries(clio_admin_runtime PRIVATE ctp::rocm_cxx)
endif()

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