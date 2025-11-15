project(hermes_shm)

# ------------------------------------------------------------------------------
# Build Tests
# ------------------------------------------------------------------------------
add_executable(test_allocator_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        allocator.cc)
add_dependencies(test_allocator_exec hermes_shm_host)
target_link_libraries(test_allocator_exec
        hermes_shm_host Catch2::Catch2)

if(HSHM_ENABLE_OPENMP)
        add_executable(test_allocator_mp_exec
                ${TEST_MAIN}/main.cc
                test_init.cc
                allocator.cc
                allocator_thread.cc)
        add_dependencies(test_allocator_mp_exec hermes_shm_host)
        target_link_libraries(test_allocator_mp_exec
                hermes_shm_host Catch2::Catch2 hshm::thread_all)
endif()

# ------------------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------------------

# ALLOCATOR tests
set(ALLOCATORS
        StackAllocator
        MallocAllocator
        ScalablePageAllocator
        LocaFullPtrs)

foreach(ALLOCATOR ${ALLOCATORS})
        add_test(NAME test_${ALLOCATOR} COMMAND
                ${CMAKE_BINARY_DIR}/bin/test_allocator_exec "${ALLOCATOR}")
endforeach()

if(HSHM_ENABLE_OPENMP)
        # Multi-Thread ALLOCATOR tests
        set(MT_ALLOCATORS
                StackAllocator
                ScalablePageAllocator)

        foreach(ALLOCATOR ${MT_ALLOCATORS})
                add_test(NAME test_${ALLOCATOR}_4t COMMAND
                        ${CMAKE_BINARY_DIR}/bin/test_allocator_mp_exec "${ALLOCATOR}Multithreaded")
        endforeach()
endif()

# ------------------------------------------------------------------------------
# Install Targets
# ------------------------------------------------------------------------------
install(TARGETS
        test_allocator_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

# -----------------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
        set_coverage_flags(test_allocator_exec)
endif()
