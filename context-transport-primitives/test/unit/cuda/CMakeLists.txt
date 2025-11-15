project(hermes_shm_cuda)

# ------------------------------------------------------------------------------
# Build Tests
# ------------------------------------------------------------------------------
add_cuda_executable(test_memory_cuda_exec FALSE backend_cuda.cc)
add_dependencies(test_memory_cuda_exec
        hermes_shm_cuda)
target_link_libraries(test_memory_cuda_exec PUBLIC
        hermes_shm_cuda)

add_cuda_executable(test_memory_cuda_mpi_exec FALSE backend_cuda_mpi.cc)
add_dependencies(test_memory_cuda_mpi_exec
        hermes_shm_cuda)
target_link_libraries(test_memory_cuda_mpi_exec PUBLIC
        hermes_shm_cuda MPI::MPI_CXX)

# ------------------------------------------------------------------------------
# Install Targets
# ------------------------------------------------------------------------------
install(TARGETS
        test_memory_cuda_exec
        test_memory_cuda_mpi_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

# -----------------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
        set_coverage_flags(test_memory_cuda_exec)
endif()