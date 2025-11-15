project(hermes_shm)



#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_compress_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_compress.cc)
add_dependencies(test_compress_exec hermes_shm_host)
target_link_libraries(test_compress_exec
        hermes_shm_host Catch2::Catch2)

add_test(NAME test_compress COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_compress_exec "~[error=FatalError]")

#-----------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_compress_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_compress_exec)
endif()