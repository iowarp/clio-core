project(clio_ctp)



if (CLIO_CORE_ENABLE_OPENMP)
    #------------------------------------------------------------------------------
    # Build Tests
    #------------------------------------------------------------------------------
    add_executable(test_thread_exec
            ${TEST_MAIN}/main.cc
            test_init.cc
            test_lock.cc)
    add_dependencies(test_thread_exec clio_ctp_host)
    target_link_libraries(test_thread_exec
            clio_ctp_host
            $<$<BOOL:${CLIO_CORE_ENABLE_THALLIUM}>:thallium>
            Catch2::Catch2
            ctp::thread_all)

    #------------------------------------------------------------------------------
    # Test Cases
    #------------------------------------------------------------------------------
    add_test(NAME ctp_thread COMMAND
            ${CMAKE_BINARY_DIR}/bin/test_thread_exec)

    #------------------------------------------------------------------------------
    # Install Targets
    #------------------------------------------------------------------------------
    install(TARGETS
            test_thread_exec
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin)

endif()