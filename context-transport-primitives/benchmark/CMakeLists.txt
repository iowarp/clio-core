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
# Build ZMQ IPC Latency Benchmark
#------------------------------------------------------------------------------

if(WRP_CORE_ENABLE_ZMQ)
  add_executable(zmq_ipc_latency_benchmark
          zmq_ipc_latency_benchmark.cc)
  add_dependencies(zmq_ipc_latency_benchmark hermes_shm_host)
  target_link_libraries(zmq_ipc_latency_benchmark
          hshm::lightbeam
          Threads::Threads)
  install(TARGETS zmq_ipc_latency_benchmark
          RUNTIME DESTINATION bin)
endif()

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        allocator_benchmark
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
