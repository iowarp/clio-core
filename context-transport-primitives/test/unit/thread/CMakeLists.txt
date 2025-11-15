project(hermes_shm)



if (HSHM_ENABLE_OPENMP)
    #------------------------------------------------------------------------------
    # Build Tests
    #------------------------------------------------------------------------------
    add_executable(test_thread_exec
            ${TEST_MAIN}/main.cc
            test_init.cc
            test_lock.cc)
    add_dependencies(test_thread_exec hermes_shm_host)
    target_link_libraries(test_thread_exec
            hermes_shm_host
            $<$<BOOL:${HSHM_ENABLE_THALLIUM}>:thallium>
            Catch2::Catch2
            hshm::thread_all)

    #------------------------------------------------------------------------------
    # Test Cases
    #------------------------------------------------------------------------------
    add_test(NAME test_thread COMMAND
            ${CMAKE_BINARY_DIR}/bin/test_thread_exec)

    #------------------------------------------------------------------------------
    # Install Targets
    #------------------------------------------------------------------------------
    install(TARGETS
            test_thread_exec
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin)

    #-----------------------------------------------------------------------------
    # Coverage
    #-----------------------------------------------------------------------------
    if(HSHM_ENABLE_COVERAGE)
        set_coverage_flags(test_thread_exec)
    endif()
endif()