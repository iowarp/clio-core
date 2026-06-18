project(clio_ctp)

#------------------------------------------------------------------------------
# Build Allocator Tests
#------------------------------------------------------------------------------

add_executable(test_sub_alloc_exec
        test_sub_alloc.cc)
add_dependencies(test_sub_alloc_exec clio_ctp_host)
target_link_libraries(test_sub_alloc_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_arena_alloc_exec
        test_arena_alloc.cc)
add_dependencies(test_arena_alloc_exec clio_ctp_host)
target_link_libraries(test_arena_alloc_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_buddy_allocator_exec
        test_buddy_allocator.cc)
add_dependencies(test_buddy_allocator_exec clio_ctp_host)
target_link_libraries(test_buddy_allocator_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_buddy_compaction_exec
        test_compaction.cc)
add_dependencies(test_buddy_compaction_exec clio_ctp_host)
target_link_libraries(test_buddy_compaction_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_ctp_malloc_exec
        test_ctp_malloc.cc)
add_dependencies(test_ctp_malloc_exec clio_ctp_host)
target_link_libraries(test_ctp_malloc_exec
        clio_ctp_host)

add_executable(test_thread_allocator_exec
        test_thread_allocator.cc)
add_dependencies(test_thread_allocator_exec clio_ctp_host)
target_link_libraries(test_thread_allocator_exec
        clio_ctp_host Catch2::Catch2WithMain)

if(NOT WIN32)
add_executable(test_mp_allocator_exec
        test_mp_allocator.cc)
add_dependencies(test_mp_allocator_exec clio_ctp_host)
target_link_libraries(test_mp_allocator_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_mp_reuse_exec
        test_mp_reuse.cc)
add_dependencies(test_mp_reuse_exec clio_ctp_host)
target_link_libraries(test_mp_reuse_exec
        clio_ctp_host Catch2::Catch2WithMain)

add_executable(test_mp_allocator_multiprocess_exec
        test_mp_allocator_multiprocess.cc)
add_dependencies(test_mp_allocator_multiprocess_exec clio_ctp_host)
target_link_libraries(test_mp_allocator_multiprocess_exec
        clio_ctp_host)

add_executable(test_buddy_allocator_multiprocess_exec_2
        test_buddy_allocator_multiprocess.cc)
add_dependencies(test_buddy_allocator_multiprocess_exec_2 clio_ctp_host)
target_link_libraries(test_buddy_allocator_multiprocess_exec_2
        clio_ctp_host)
endif()

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------

add_test(NAME ctp_sub_alloc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_sub_alloc_exec "~[nested]")
add_test(NAME ctp_sub_alloc_nested COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_sub_alloc_exec "[nested]")

add_test(NAME ctp_arena_alloc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_arena_alloc_exec)

add_test(NAME ctp_buddy_allocator COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_buddy_allocator_exec)

add_test(NAME ctp_buddy_compaction COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_buddy_compaction_exec)

add_test(NAME ctp_thread_allocator COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_thread_allocator_exec "~[multithread]")
add_test(NAME ctp_thread_allocator_multithread COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_thread_allocator_exec "[multithread]")

add_test(NAME ctp_ctp_malloc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_ctp_malloc_exec)

# Skip under MSan: buddy allocator, thread/partitioned allocator, and hshm
# malloc all use raw mmap/brk memory that MSan cannot instrument (no
# instrumented allocator backing).
set_tests_properties(
  ctp_buddy_allocator
  ctp_buddy_compaction
  ctp_ctp_malloc
  ctp_thread_allocator
  ctp_thread_allocator_multithread
  PROPERTIES LABELS "msan_skip"
)

if(NOT WIN32)
add_test(NAME ctp_mp_allocator COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_mp_allocator_exec "~[multithread]")
add_test(NAME ctp_mp_allocator_multithread COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_mp_allocator_exec "[multithread]")

add_test(NAME ctp_mp_reuse COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_mp_reuse_exec)

add_test(NAME ctp_mp_allocator_multiprocess
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_mp_allocator_multiprocess_test.sh
                ${CMAKE_BINARY_DIR}/bin/test_mp_allocator_multiprocess_exec)

add_test(NAME ctp_buddy_allocator_multiprocess COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_buddy_allocator_multiprocess_exec_2 0 5)

# Set timeouts and LD_LIBRARY_PATH for multiprocess tests
# LD_LIBRARY_PATH ensures subprocesses load the freshly built library,
# not the stale spack-installed one (which may lack newer symbols).
set_tests_properties(
  ctp_mp_allocator_multiprocess
  ctp_buddy_allocator_multiprocess
  PROPERTIES
    # 3 ranks x 2 threads each run a 5s contended alloc/free workload (rank 0
    # leads by a 2s stagger), so a healthy run is ~11-12s on macOS once the
    # ticket lock's backoff serializes the global allocator. 15s left almost no
    # CI headroom; 60s keeps it comfortably non-flaky.
    TIMEOUT 60
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

# Skip under MSan: Catch2 is precompiled without MSan instrumentation,
# and multiprocess tests don't propagate MSan shadow across fork boundaries.
set_tests_properties(
  ctp_sub_alloc
  ctp_sub_alloc_nested
  ctp_mp_allocator
  ctp_mp_allocator_multithread
  ctp_mp_allocator_multiprocess
  ctp_mp_reuse
  ctp_buddy_allocator_multiprocess
  PROPERTIES LABELS "msan_skip"
)

# macOS (issue #483, fixed): ctp_mp_allocator_multiprocess used to livelock
# under cross-process contention on the shared-memory ProducerConsumerAllocator.
# Its global ticket lock (ctp::Mutex) busy-spun with only a PAUSE hint, and the
# macOS scheduler would not preempt a spinning waiter to run the descheduled
# lock holder -- starving the holder for the full 15s timeout. The ticket lock's
# Backoff now additionally deschedules the OS thread (std::this_thread::sleep_for)
# once contention is sustained, so the holder makes progress. The test passes on
# macOS again and is enabled (it keeps the bumped timeout above for headroom).
endif()

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_sub_alloc_exec
        test_arena_alloc_exec
        test_buddy_allocator_exec
        test_buddy_compaction_exec
        test_ctp_malloc_exec
        test_thread_allocator_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

if(NOT WIN32)
install(TARGETS
        test_mp_allocator_exec
        test_mp_allocator_multiprocess_exec
        test_buddy_allocator_multiprocess_exec_2
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
endif()
