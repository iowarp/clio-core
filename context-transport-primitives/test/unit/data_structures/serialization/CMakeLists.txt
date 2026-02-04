project(hermes_shm)

#------------------------------------------------------------------------------
# Build Serialization Tests
#------------------------------------------------------------------------------

add_executable(test_local_serialize_exec
        test_local_serialize.cc)
add_dependencies(test_local_serialize_exec hermes_shm_host)
target_link_libraries(test_local_serialize_exec
        hermes_shm_host
        hshm::serialize)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_local_serialize_basic COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_local_serialize_exec)
add_test(NAME ctp_local_serialize_string COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_local_serialize_exec)
add_test(NAME ctp_local_serialize_vector COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_local_serialize_exec)
add_test(NAME ctp_local_serialize_list COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_local_serialize_exec)
add_test(NAME ctp_local_serialize_map COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_local_serialize_exec)
add_test(NAME ctp_local_serialize_mixed COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_local_serialize_exec)
add_test(NAME ctp_local_serialize_nested COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_local_serialize_exec)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_local_serialize_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

