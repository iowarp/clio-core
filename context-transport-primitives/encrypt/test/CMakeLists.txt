project(clio_ctp)



#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_encrypt_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_encrypt.cc)
add_dependencies(test_encrypt_exec clio_ctp_host)
target_link_libraries(test_encrypt_exec
        clio_ctp_host Catch2::Catch2)

add_test(NAME ctp_encrypt COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_encrypt_exec "~[error=FatalError]")

#-----------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_encrypt_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

