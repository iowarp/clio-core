project(hermes_shm)

#------------------------------------------------------------------------------
# Build Private Memory Data Structure Tests
#------------------------------------------------------------------------------

add_executable(test_priv_vector_exec
        test_priv_vector.cc)
add_dependencies(test_priv_vector_exec hermes_shm_host)
target_link_libraries(test_priv_vector_exec
        hermes_shm_host
        hshm::serialize)

add_executable(test_priv_string_exec
        test_priv_string.cc)
add_dependencies(test_priv_string_exec hermes_shm_host)
target_link_libraries(test_priv_string_exec
        hermes_shm_host
        hshm::serialize)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_priv_vector COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_vector_exec)
add_test(NAME ctp_priv_vector_pod COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_vector_exec)
add_test(NAME ctp_priv_vector_complex COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_vector_exec)
add_test(NAME ctp_priv_vector_edge COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_vector_exec)
add_test(NAME ctp_priv_vector_stress COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_vector_exec)
add_test(NAME ctp_priv_vector_serialization COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_vector_exec)
add_test(NAME ctp_priv_vector_local_serialize COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_vector_exec)

add_test(NAME ctp_priv_string COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_string_exec)
add_test(NAME ctp_priv_string_sso COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_string_exec)
add_test(NAME ctp_priv_string_operations COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_string_exec)
add_test(NAME ctp_priv_string_edge COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_string_exec)
add_test(NAME ctp_priv_string_serialization COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_string_exec)
add_test(NAME ctp_priv_string_local_serialize COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_priv_string_exec)

# Skip ctp_priv_string* under MSan: cereal deserialization is uninstrumented,
# leaving restored values with uninitialized shadow that propagates to test output.
set_tests_properties(
  ctp_priv_string
  ctp_priv_string_sso
  ctp_priv_string_operations
  ctp_priv_string_edge
  ctp_priv_string_serialization
  ctp_priv_string_local_serialize
  PROPERTIES LABELS "msan_skip"
)

# Skip ctp_priv_vector* under MSan: hshm priv::vector uses mmap-backed shared
# memory regions that MSan cannot instrument (no instrumented allocator backing),
# and cereal serialization leaves restored bytes with uninitialized MSan shadow.
set_tests_properties(
  ctp_priv_vector
  ctp_priv_vector_pod
  ctp_priv_vector_complex
  ctp_priv_vector_edge
  ctp_priv_vector_stress
  ctp_priv_vector_serialization
  ctp_priv_vector_local_serialize
  PROPERTIES LABELS "msan_skip"
)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_priv_vector_exec
        test_priv_string_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

