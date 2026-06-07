project(clio_ctp_benchmark)

#------------------------------------------------------------------------------
# Build Allocator Benchmark
#------------------------------------------------------------------------------

add_executable(allocator_benchmark
        allocator_benchmark.cc)
add_dependencies(allocator_benchmark clio_ctp_host)
target_link_libraries(allocator_benchmark
        clio_ctp_host
        Threads::Threads)

#------------------------------------------------------------------------------
# Build ZMQ IPC Latency Benchmark
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_ZMQ)
  add_executable(zmq_ipc_latency_benchmark
          zmq_ipc_latency_benchmark.cc)
  add_dependencies(zmq_ipc_latency_benchmark clio_ctp_host)
  target_link_libraries(zmq_ipc_latency_benchmark
          ctp::lightbeam
          Threads::Threads)
  install(TARGETS zmq_ipc_latency_benchmark
          RUNTIME DESTINATION bin)
endif()

#------------------------------------------------------------------------------
# Build CUDA Stream Create Cost Benchmark
#------------------------------------------------------------------------------

# Self-contained: depends only on the CUDA runtime (CUDA::cudart), not on
# nvcomp, since it measures the cudaStreamCreate primitive that ctp::NvComp
# (clio_ctp/compress/nvcomp.h) calls on every Compress/Decompress.
if(CLIO_CORE_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(TARGET CUDA::cudart)
    add_executable(cuda_stream_create_benchmark
            cuda_stream_create_benchmark.cc)
    target_link_libraries(cuda_stream_create_benchmark
            CUDA::cudart)
    install(TARGETS cuda_stream_create_benchmark
            RUNTIME DESTINATION bin)
  else()
    message(STATUS "CUDA::cudart not found - cuda_stream_create_benchmark disabled")
  endif()
endif()

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        allocator_benchmark
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
