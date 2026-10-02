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

add_executable(test_rb_tree_pre_ext_exec
        test_rb_tree_pre_ext.cc)
add_dependencies(test_rb_tree_pre_ext_exec clio_ctp_host)
target_link_libraries(test_rb_tree_pre_ext_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_vector_exec
        vector.cc)
add_dependencies(test_vector_exec clio_ctp_host)
target_link_libraries(test_vector_exec
        clio_ctp_host)

add_executable(test_string_exec
        test_string.cc)
add_dependencies(test_string_exec clio_ctp_host)
target_link_libraries(test_string_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_unordered_map_exec
        test_unordered_map.cc)
add_dependencies(test_unordered_map_exec clio_ctp_host)
target_link_libraries(test_unordered_map_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_lease_exec
        test_lease.cc)
add_dependencies(test_lease_exec clio_ctp_host)
target_link_libraries(test_lease_exec
        clio_ctp_host Catch2::Catch2WithMain)

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
add_test(NAME ctp_rb_tree_pre_ext COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_rb_tree_pre_ext_exec)

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

add_test(NAME ctp_ipc_string COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_string_exec)

add_test(NAME ctp_ipc_unordered_map COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_unordered_map_exec)

add_test(NAME ctp_ipc_lease COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_lease_exec)
set_tests_properties(ctp_ipc_lease PROPERTIES TIMEOUT 120)

# msan_skip: these three assert cleanly under MSan, but Catch2 fails them
# anyway. Before each test case the prebuilt libCatch2.so installs its crash
# handler, and the `struct sigaction` it passes sits on stack our instrumented
# test bodies just poisoned -- MSan's sigaction interceptor then reports memory
# that belongs to a frame we do not compile. Measured on build-msan: 36 / 78 /
# 31 warnings per run, every one of them that same call.
#
# Nothing narrower works. The ignorelist cannot reach a check that fires inside
# libCatch2.so; ignorelisting the test bodies stops the poisoning but also
# leaves every object they construct unpoisoned, which is exactly the memory
# these tests exist to check; and building Catch2 from source under MSan just
# swaps the sigaction reports for reports from Catch2's own reporter against an
# uninstrumented libstdc++ (tried, still non-zero). An MSan-instrumented libc++
# would fix this properly; until then these run under asan/ubsan only.
set_tests_properties(ctp_ipc_string ctp_ipc_unordered_map ctp_ipc_lease
        PROPERTIES LABELS "msan_skip")

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

# MPMC pop-stress test is heavier (more threads / more items) than MPSC, and
# the locked Pop() path is slow under contention on the 2-core windows-2022
# runner (ran 30.11s there while finishing well under 30s on windows-2025 and
# windows-11-arm), so give it generous headroom.
set_tests_properties(
  ctp_ring_buffer_mpmc
  PROPERTIES TIMEOUT 120
)

# Skip under MSan: slist/rb_tree tests use precompiled Catch2 which
# has FatalConditionHandler false positives (sigaction filled by kernel).
set_tests_properties(
  ctp_slist_pre
  ctp_slist_pre_atomic
  ctp_rb_tree_pre
  ctp_rb_tree_pre_atomic
  ctp_rb_tree_pre_ext
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
        test_rb_tree_pre_ext_exec
        test_vector_exec
        test_string_exec
        test_unordered_map_exec
        test_lease_exec
        test_ring_buffer_exec
        test_ring_buffer_spsc_exec
        test_ring_buffer_ext_exec
        test_ring_buffer_mpsc_exec
        test_ring_buffer_mpmc_exec
        test_multi_ring_buffer_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

