project(hermes_shm)



#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------
set (LIBS
        hermes_shm_host
        Catch2::Catch2
        hshm::serialize
        hshm::mpi
        hshm::thread_all)
add_executable(test_cereal_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_cereal.cc)
add_dependencies(test_cereal_exec
        hermes_shm_host)
target_link_libraries(test_cereal_exec ${LIBS})

add_test(NAME test_cereal COMMAND ${CMAKE_BINARY_DIR}/bin/test_cereal_exec)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_cereal_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_cereal_exec)
endif()