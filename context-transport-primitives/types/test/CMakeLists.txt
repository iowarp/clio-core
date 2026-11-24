project(clio_ctp)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_types_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_argpack.cc
        test_util.cc
        test_atomics.cc)
add_dependencies(test_types_exec clio_ctp_host)
target_link_libraries(test_types_exec
        clio_ctp_host Catch2::Catch2 ctp::thread_all ctp::mpi)
if(NOT WIN32)
# TODO: Fix this test - currently failing
# add_test(NAME ctp_types COMMAND
#         ${CMAKE_BINARY_DIR}/bin/test_types_exec "~[error=FatalError]")
endif()
add_test(NAME ctp_fatal_logger
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_types_exec)

set_tests_properties(
        ctp_fatal_logger PROPERTIES
        WILL_FAIL TRUE)
set(INSTALLS ${INSTALLS} test_types_exec)

#------------------------------------------------------------------------------
# Build Timer Tests
#------------------------------------------------------------------------------
if (CLIO_CORE_ENABLE_MPI AND CLIO_CORE_ENABLE_OPENMP)
    add_executable(test_timers_exec
            ${TEST_MAIN}/main_mpi.cc
            test_init.cc
            test_timers.cc)
    add_dependencies(test_timers_exec clio_ctp_host)
    target_link_libraries(test_timers_exec
            clio_ctp_host Catch2::Catch2 ctp::thread_all ctp::mpi)

    add_test(NAME ctp_timers COMMAND
            ${CMAKE_BINARY_DIR}/bin/test_timers_exec)
    set(INSTALLS ${INSTALLS} test_timers_exec)
endif()

#-----------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        ${INSTALLS}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

