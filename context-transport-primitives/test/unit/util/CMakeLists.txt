project(hermes_shm)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_config_parse_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_config_parse.cc)
add_dependencies(test_config_parse_exec hermes_shm_host)
target_link_libraries(test_config_parse_exec
        hermes_shm_host
        Catch2::Catch2)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------
add_test(NAME ctp_config_parse COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_config_parse_exec)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_config_parse_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
