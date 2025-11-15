project(hermes_shm)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_types_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_argpack.cc
        test_util.cc
        test_atomics.cc)
add_dependencies(test_types_exec hermes_shm_host)
target_link_libraries(test_types_exec
        hermes_shm_host Catch2::Catch2 hshm::thread_all hshm::mpi)
if(NOT WIN32)	
add_test(NAME test_types COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_types_exec "~[error=FatalError]")
endif()
add_test(NAME test_fatal_logger
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_types_exec "[error=FatalError]")

set_tests_properties(
        test_fatal_logger PROPERTIES
        WILL_FAIL TRUE)
set(INSTALLS ${INSTALLS} test_types_exec)

#------------------------------------------------------------------------------
# Build Timer Tests
#------------------------------------------------------------------------------
if (HSHM_ENABLE_MPI AND HSHM_ENABLE_OPENMP)
    add_executable(test_timers_exec
            ${TEST_MAIN}/main_mpi.cc
            test_init.cc
            test_timers.cc)
    add_dependencies(test_timers_exec hermes_shm_host)
    target_link_libraries(test_timers_exec
            hermes_shm_host Catch2::Catch2 hshm::thread_all hshm::mpi)

    add_test(NAME test_timers COMMAND
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

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_types_exec)
    if (HSHM_ENABLE_MPI AND HSHM_ENABLE_OPENMP)
        set_coverage_flags(test_timers_exec)
    endif()
endif()
