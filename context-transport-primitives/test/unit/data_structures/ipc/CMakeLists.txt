project(clio_ctp)

#------------------------------------------------------------------------------
# Build IPC Data Structure Tests
#------------------------------------------------------------------------------

add_executable(test_slist_pre_exec
        test_slist_pre.cc)
add_dependencies(test_slist_pre_exec clio_ctp_host)
target_link_libraries(test_slist_pre_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_rb_tree_pre_exec
        test_rb_tree_pre.cc)
add_dependencies(test_rb_tree_pre_exec clio_ctp_host)
target_link_libraries(test_rb_tree_pre_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_vector_exec
        vector.cc)
add_dependencies(test_vector_exec clio_ctp_host)
target_link_libraries(test_vector_exec
        clio_ctp_host)

add_executable(test_ring_buffer_exec
        test_ring_buffer.cc)
add_dependencies(test_ring_buffer_exec clio_ctp_host)
target_link_libraries(test_ring_buffer_exec
        clio_ctp_host)

add_executable(test_ring_buffer_spsc_exec
        test_ring_buffer_spsc.cc)
add_dependencies(test_ring_buffer_spsc_exec clio_ctp_host)
target_link_libraries(test_ring_buffer_spsc_exec
        clio_ctp_host)

add_executable(test_ring_buffer_ext_exec
        test_ring_buffer_ext.cc)
add_dependencies(test_ring_buffer_ext_exec clio_ctp_host)
target_link_libraries(test_ring_buffer_ext_exec
        clio_ctp_host)

add_executable(test_ring_buffer_mpsc_exec
        test_ring_buffer_mpsc.cc)
add_dependencies(test_ring_buffer_mpsc_exec clio_ctp_host)
target_link_libraries(test_ring_buffer_mpsc_exec
        clio_ctp_host)

add_executable(test_ring_buffer_mpmc_exec
        test_ring_buffer_mpmc.cc)
add_dependencies(test_ring_buffer_mpmc_exec clio_ctp_host)
target_link_libraries(test_ring_buffer_mpmc_exec
        clio_ctp_host)

add_executable(test_multi_ring_buffer_exec
        test_multi_ring_buffer.cc)
add_dependencies(test_multi_ring_buffer_exec clio_ctp_host)
target_link_libraries(test_multi_ring_buffer_exec
        clio_ctp_host)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_slist_pre COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_slist_pre_exec)
add_test(NAME ctp_slist_pre_atomic COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_slist_pre_exec)

add_test(NAME ctp_rb_tree_pre COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_rb_tree_pre_exec)
add_test(NAME ctp_rb_tree_pre_atomic COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_rb_tree_pre_exec)

add_test(NAME ctp_vector COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec)
add_test(NAME ctp_vector_pod COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec)
add_test(NAME ctp_vector_non_pod COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec)
add_test(NAME ctp_vector_edge COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec)
add_test(NAME ctp_vector_stress COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_vector_exec)

add_test(NAME ctp_ring_buffer COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ring_buffer_exec)
add_test(NAME ctp_ring_buffer_spsc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ring_buffer_spsc_exec)
add_test(NAME ctp_ring_buffer_ext COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ring_buffer_ext_exec)
add_test(NAME ctp_ring_buffer_mpsc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ring_buffer_mpsc_exec)
add_test(NAME ctp_ring_buffer_mpmc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ring_buffer_mpmc_exec)

add_test(NAME ctp_multi_ring_buffer COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_multi_ring_buffer_exec)

# Set timeout for MPSC ring buffer test (multi-threaded, needs more time)
set_tests_properties(
  ctp_ring_buffer_mpsc
  PROPERTIES TIMEOUT 15
)

# MPMC pop-stress test is heavier (more threads / more items) than MPSC.
set_tests_properties(
  ctp_ring_buffer_mpmc
  PROPERTIES TIMEOUT 30
)

# Skip under MSan: slist/rb_tree tests use precompiled Catch2 which
# has FatalConditionHandler false positives (sigaction filled by kernel).
set_tests_properties(
  ctp_slist_pre
  ctp_slist_pre_atomic
  ctp_rb_tree_pre
  ctp_rb_tree_pre_atomic
  PROPERTIES LABELS "msan_skip"
)

# Skip under MSan: vector and ring buffer tests use raw shared-memory and
# atomic operations that MSan cannot instrument (uninstrumented backing alloc).
set_tests_properties(
  ctp_vector
  ctp_vector_pod
  ctp_vector_non_pod
  ctp_vector_edge
  ctp_vector_stress
  ctp_ring_buffer
  ctp_ring_buffer_spsc
  ctp_ring_buffer_ext
  ctp_ring_buffer_mpsc
  ctp_ring_buffer_mpmc
  ctp_multi_ring_buffer
  PROPERTIES LABELS "msan_skip"
)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_slist_pre_exec
        test_rb_tree_pre_exec
        test_vector_exec
        test_ring_buffer_exec
        test_ring_buffer_spsc_exec
        test_ring_buffer_ext_exec
        test_ring_buffer_mpsc_exec
        test_ring_buffer_mpmc_exec
        test_multi_ring_buffer_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

