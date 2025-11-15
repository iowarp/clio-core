project(hermes_shm)

if (HSHM_ENABLE_MPI)
    #------------------------------------------------------------------------------
    # Build Tests
    #------------------------------------------------------------------------------

    add_executable(test_backend_exec
            ${TEST_MAIN}/main_mpi.cc
            test_init.cc
            backend.cc
            memory_slots.cc
            memory_manager.cc)
    add_dependencies(test_backend_exec hermes_shm_host)
    target_link_libraries(test_backend_exec
            hermes_shm_host Catch2::Catch2 hshm::mpi)

    #------------------------------------------------------------------------------
    # Test Cases
    #------------------------------------------------------------------------------

    add_test(NAME test_memory_slots COMMAND
            mpirun -n 2 ${CMAKE_BINARY_DIR}/bin/test_backend_exec "MemorySlot")
    add_test(NAME test_reserve COMMAND
            ${CMAKE_BINARY_DIR}/bin/test_backend_exec "BackendReserve")
    add_test(NAME test_memory_manager COMMAND
            mpirun -n 2 ${CMAKE_BINARY_DIR}/bin/test_backend_exec "MemoryManager")

    #------------------------------------------------------------------------------
    # Install Targets
    #------------------------------------------------------------------------------
    install(TARGETS
            test_backend_exec
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin)

    #-----------------------------------------------------------------------------
    # Coverage
    #-----------------------------------------------------------------------------
    if(HSHM_ENABLE_COVERAGE)
        set_coverage_flags(test_backend_exec)
    endif()
endif()