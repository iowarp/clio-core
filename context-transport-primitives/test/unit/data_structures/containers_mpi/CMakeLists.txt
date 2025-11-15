project(hermes_shm)

if (HSHM_ENABLE_MPI)
    #------------------------------------------------------------------------------
    # Test Cases
    #------------------------------------------------------------------------------

    add_executable(test_data_structure_mpi_exec
            ${TEST_MAIN}/main_mpi.cc
            test_init.cc
            mpsc_queue.cc
    )

    add_dependencies(test_data_structure_mpi_exec hermes_shm_host)
    target_link_libraries(test_data_structure_mpi_exec
            hermes_shm_host Catch2::Catch2 hshm::thread_all hshm::mpi)
    message(STATUS "MPI_LIBS: hshm::mpi")

    #------------------------------------------------------------------------------
    # Build Tests
    #------------------------------------------------------------------------------

    # LIST TESTS
    add_test(NAME test_mpsc_queue_mpi COMMAND
            mpirun -n 2 ${CMAKE_BINARY_DIR}/bin/test_data_structure_mpi_exec "TestMpscQueueMpi")

    #------------------------------------------------------------------------------
    # Install Targets
    #------------------------------------------------------------------------------
    install(TARGETS
            test_data_structure_mpi_exec
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin)

    #-----------------------------------------------------------------------------
    # Coverage
    #-----------------------------------------------------------------------------
    if(HSHM_ENABLE_COVERAGE)
        set_coverage_flags(test_data_structure_mpi_exec)
    endif()

endif()