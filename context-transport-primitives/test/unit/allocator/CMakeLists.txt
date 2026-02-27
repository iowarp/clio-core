project(hermes_shm)

#------------------------------------------------------------------------------
# Build Allocator Tests
#------------------------------------------------------------------------------

add_executable(test_sub_alloc_exec
        test_sub_alloc.cc)
add_dependencies(test_sub_alloc_exec hermes_shm_host)
target_link_libraries(test_sub_alloc_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_arena_alloc_exec
        test_arena_alloc.cc)
add_dependencies(test_arena_alloc_exec hermes_shm_host)
target_link_libraries(test_arena_alloc_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_buddy_allocator_exec
        test_buddy_allocator.cc)
add_dependencies(test_buddy_allocator_exec hermes_shm_host)
target_link_libraries(test_buddy_allocator_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_hshm_malloc_exec
        test_hshm_malloc.cc)
add_dependencies(test_hshm_malloc_exec hermes_shm_host)
target_link_libraries(test_hshm_malloc_exec
        hermes_shm_host)

if(NOT WIN32)
add_executable(test_mp_allocator_exec
        test_mp_allocator.cc)
add_dependencies(test_mp_allocator_exec hermes_shm_host)
target_link_libraries(test_mp_allocator_exec
        hermes_shm_host Catch2::Catch2WithMain)

add_executable(test_mp_allocator_multiprocess_exec
        test_mp_allocator_multiprocess.cc)
add_dependencies(test_mp_allocator_multiprocess_exec hermes_shm_host)
target_link_libraries(test_mp_allocator_multiprocess_exec
        hermes_shm_host)

add_executable(test_buddy_allocator_multiprocess_exec_2
        test_buddy_allocator_multiprocess.cc)
add_dependencies(test_buddy_allocator_multiprocess_exec_2 hermes_shm_host)
target_link_libraries(test_buddy_allocator_multiprocess_exec_2
        hermes_shm_host)
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

add_test(NAME ctp_hshm_malloc COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_hshm_malloc_exec)

if(NOT WIN32)
add_test(NAME ctp_mp_allocator COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_mp_allocator_exec "~[multithread]")
add_test(NAME ctp_mp_allocator_multithread COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_mp_allocator_exec "[multithread]")

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
    TIMEOUT 15
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
  ctp_buddy_allocator_multiprocess
  PROPERTIES LABELS "msan_skip"
)
endif()

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_sub_alloc_exec
        test_arena_alloc_exec
        test_buddy_allocator_exec
        test_hshm_malloc_exec
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
