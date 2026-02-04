project(hermes_shm)

if (HSHM_ENABLE_MPI)
    #------------------------------------------------------------------------------
    # Build Tests
    #------------------------------------------------------------------------------

    add_executable(test_backend_exec
            ${TEST_MAIN}/main_mpi.cc
            test_init.cc
            backend.cc
            memory_slots.cc)
    add_dependencies(test_backend_exec hermes_shm_host)
    target_link_libraries(test_backend_exec
            hermes_shm_host Catch2::Catch2 hshm::mpi)

    #------------------------------------------------------------------------------
    # Test Cases
    #------------------------------------------------------------------------------

    add_test(NAME ctp_memory_slots COMMAND
            mpirun -n 2 ${CMAKE_BINARY_DIR}/bin/test_backend_exec "MemorySlot")
    add_test(NAME ctp_reserve COMMAND
            ${CMAKE_BINARY_DIR}/bin/test_backend_exec "BackendReserve")

    #------------------------------------------------------------------------------
    # Install Targets
    #------------------------------------------------------------------------------
    install(TARGETS
            test_backend_exec
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin)

endif()