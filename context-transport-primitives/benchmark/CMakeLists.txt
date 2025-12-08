project(hermes_shm_benchmark)

#------------------------------------------------------------------------------
# Build Allocator Benchmark
#------------------------------------------------------------------------------

add_executable(allocator_benchmark
        allocator_benchmark.cc)
add_dependencies(allocator_benchmark hermes_shm_host)
target_link_libraries(allocator_benchmark
        hermes_shm_host
        Threads::Threads)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        allocator_benchmark
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
