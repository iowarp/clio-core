project(clio_ctp)

#------------------------------------------------------------------------------
# Build Serialization Tests
#------------------------------------------------------------------------------

add_executable(test_local_serialize_exec
        test_local_serialize.cc)
add_dependencies(test_local_serialize_exec clio_ctp_host)
target_link_libraries(test_local_serialize_exec
        clio_ctp_host
        ctp::serialize)

add_executable(test_msgpack_wrapper_exec
        test_msgpack_wrapper.cc)
add_dependencies(test_msgpack_wrapper_exec clio_ctp_host)
target_link_libraries(test_msgpack_wrapper_exec
        clio_ctp_host
        ctp::serialize)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_msgpack_wrapper COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_msgpack_wrapper_exec)
# Skip under MSan: msgpack::object payloads live in zones allocated by the
# uninstrumented libmsgpack-c.so, so object::convert<>() reads bytes MSan
# cannot prove initialized; those values then propagate into the uninstrumented
# libstdc++ ostream used by SimpleTest::print_summary. Both are unfixable
# third-party false positives.
set_tests_properties(ctp_msgpack_wrapper PROPERTIES LABELS "msan_skip")

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

# Skip under MSan: cereal serialization is uninstrumented; deserialized values
# retain uninitialized shadow that propagates to SimpleTest::print_summary output.
set_tests_properties(
  ctp_local_serialize_basic
  ctp_local_serialize_string
  ctp_local_serialize_vector
  ctp_local_serialize_list
  ctp_local_serialize_map
  ctp_local_serialize_mixed
  ctp_local_serialize_nested
  PROPERTIES LABELS "msan_skip"
)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_local_serialize_exec
        test_msgpack_wrapper_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

