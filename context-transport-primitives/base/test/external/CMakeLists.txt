project(clio_ctp_external_test)

# ------------------------------------------------------------------------------
# External libraries
# ------------------------------------------------------------------------------
# Use unified IOWarp Core package which includes ClioCtp.
# `clio-core` is the canonical name; `iowarp-core` still works as a legacy alias.
find_package(clio-core CONFIG REQUIRED)
message(STATUS "found clio-core at ${clio-core_DIR}")
message(STATUS "  ClioCtp (context-transport-primitives) included in clio-core")

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
add_executable(test test.cc)
target_link_libraries(test ctp::cxx)

# hshm_enable_cuda(17)
# add_cuda_executable(test_cuda FALSE test_cuda.cc)
# target_link_libraries(test_cuda ctp::cudacxx)