project(hermes_shm_external_test)

# ------------------------------------------------------------------------------
# External libraries
# ------------------------------------------------------------------------------
# Use unified IOWarp Core package which includes HermesShm
find_package(iowarp-core CONFIG REQUIRED)
message(STATUS "found IowarpCore at ${IowarpCore_DIR}")
message(STATUS "  HermesShm (context-transport-primitives) included in IowarpCore")

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
add_executable(test test.cc)
target_link_libraries(test hshm::cxx)

# hshm_enable_cuda(17)
# add_cuda_executable(test_cuda FALSE test_cuda.cc)
# target_link_libraries(test_cuda hshm::cudacxx)