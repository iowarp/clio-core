project(clio_ctp)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_msgpack_wrapper_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
