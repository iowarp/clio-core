project(hermes_shm)



#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------

if (HSHM_ENABLE_MPI)
    add_executable(test_allocator_mpi_exec
            ${TEST_MAIN}/main_mpi.cc
            test_init.cc
            allocator_mpi.cc)
    add_dependencies(test_allocator_mpi_exec hermes_shm_host)
    target_link_libraries(test_allocator_mpi_exec
            hermes_shm_host Catch2::Catch2 hshm::mpi)


#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

# Multi-Thread ALLOCATOR tests
set(MP_ALLOCATORS
        StackAllocator
        ScalablePageAllocator)
foreach(ALLOCATOR ${MP_ALLOCATORS})
    add_test(NAME test_${ALLOCATOR}_mpi COMMAND
            mpirun -n 4 ${CMAKE_BINARY_DIR}/bin/test_allocator_mpi_exec "${ALLOCATOR}Mpi")
endforeach()


#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
if (HSHM_ENABLE_MPI)
    install(TARGETS
            test_allocator_mpi_exec
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin)
endif()

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_allocator_mpi_exec)
endif()

endif()
