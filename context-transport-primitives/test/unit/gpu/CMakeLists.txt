#------------------------------------------------------------------------------
# GPU Unit Tests
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
  message(STATUS "Building GPU unit tests")

  set(CLIO_CTP_GPU_INCLUDE_DIRS
    ${CLIO_CTP_ROOT}/include
    ${PROJECT_BINARY_DIR}/include
  )

  # GpuShmMmap test
  add_cuda_executable(test_gpu_shm_mmap TRUE test_gpu_shm_mmap.cc
    INCLUDE_DIRS ${CLIO_CTP_GPU_INCLUDE_DIRS})
  target_link_libraries(test_gpu_shm_mmap
    ctp::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_gpu_shm_mmap COMMAND test_gpu_shm_mmap)
  set_tests_properties(test_gpu_shm_mmap PROPERTIES TIMEOUT 60)

  # GpuMalloc test (commented out — not well-written)
  # add_cuda_executable(test_gpu_malloc TRUE test_gpu_malloc.cc)
  # target_link_libraries(test_gpu_malloc
  #   ctp::cuda_cxx
  #   Catch2::Catch2WithMain
  # )
  # add_test(NAME test_gpu_malloc COMMAND test_gpu_malloc)

  # LocalSerialize GPU test
  add_cuda_executable(test_local_serialize_gpu TRUE test_local_serialize_gpu.cc
    INCLUDE_DIRS ${CLIO_CTP_GPU_INCLUDE_DIRS})
  target_link_libraries(test_local_serialize_gpu
    ctp::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_local_serialize_gpu COMMAND test_local_serialize_gpu)
  set_tests_properties(test_local_serialize_gpu PROPERTIES TIMEOUT 60)

  # LocalTransfer GPU test
  add_cuda_executable(test_local_transfer_gpu TRUE test_local_transfer_gpu.cc
    INCLUDE_DIRS ${CLIO_CTP_GPU_INCLUDE_DIRS})
  target_link_libraries(test_local_transfer_gpu
    ctp::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_local_transfer_gpu COMMAND test_local_transfer_gpu)
  set_tests_properties(test_local_transfer_gpu PROPERTIES TIMEOUT 60)

  # Async copy primitives test (StreamQuery / MemcpyAsync / MemsetAsync) used by
  # the async memory-bdev transport.
  add_cuda_executable(test_gpu_async_copy_gpu TRUE test_gpu_async_copy_gpu.cc
    INCLUDE_DIRS ${CLIO_CTP_GPU_INCLUDE_DIRS})
  target_link_libraries(test_gpu_async_copy_gpu
    ctp::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_gpu_async_copy_gpu COMMAND test_gpu_async_copy_gpu)
  set_tests_properties(test_gpu_async_copy_gpu PROPERTIES TIMEOUT 60)

  # BuddyAllocator GPU test
  add_cuda_executable(test_buddy_alloc_gpu TRUE test_buddy_alloc_gpu.cc
    INCLUDE_DIRS ${CLIO_CTP_GPU_INCLUDE_DIRS})
  target_link_libraries(test_buddy_alloc_gpu
    ctp::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_buddy_alloc_gpu COMMAND test_buddy_alloc_gpu)

  # PrivateBuddyAllocator shifted (stack) GPU test
  add_cuda_executable(test_buddy_alloc_stack_gpu TRUE test_buddy_alloc_stack_gpu.cc
    INCLUDE_DIRS ${CLIO_CTP_GPU_INCLUDE_DIRS})
  target_link_libraries(test_buddy_alloc_stack_gpu
    ctp::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_buddy_alloc_stack_gpu COMMAND test_buddy_alloc_stack_gpu)

  # ThreadAllocator GPU test
  add_cuda_executable(test_thread_alloc_gpu TRUE test_thread_alloc_gpu.cc
    INCLUDE_DIRS ${CLIO_CTP_GPU_INCLUDE_DIRS})
  target_link_libraries(test_thread_alloc_gpu
    ctp::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_thread_alloc_gpu COMMAND test_thread_alloc_gpu)

  # RoundRobinAllocator CDP test. The test source picks its CDP flavor from the
  # toolkit: CDP2 (fire-and-forget children joined by a tail launch) when the
  # headers expose cudaStreamFireAndForget, else the CDP1 device-side
  # cudaDeviceSynchronize. So do NOT define CUDA_FORCE_CDP1_IF_SUPPORTED on
  # CUDA >= 12 — it both hides the CDP2 stream macros and fails to bring back
  # device-side cudaDeviceSynchronize, which CUDA 12 does not declare at all for
  # sm_90+ (see issue #828). On CUDA < 12 only CDP1 exists; silence its
  # 11.6-onwards deprecation warning.
  find_package(CUDAToolkit QUIET)
  set(CLIO_RR_CDP_DEFS "")
  if(CUDAToolkit_VERSION VERSION_LESS 12)
    set(CLIO_RR_CDP_DEFS __CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING)
  endif()
  add_cuda_executable(test_round_robin_alloc_gpu TRUE test_round_robin_alloc_gpu.cc
    INCLUDE_DIRS ${CLIO_CTP_GPU_INCLUDE_DIRS}
    DEFS ${CLIO_RR_CDP_DEFS})
  target_link_libraries(test_round_robin_alloc_gpu
    ctp::cuda_cxx
    Catch2::Catch2WithMain
  )
  target_compile_options(test_round_robin_alloc_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -rdc=true>)
  add_test(NAME test_round_robin_alloc_gpu COMMAND test_round_robin_alloc_gpu)
  set_tests_properties(test_round_robin_alloc_gpu PROPERTIES
    TIMEOUT 60 LABELS "gpu;allocator")

  add_subdirectory(runtime)

endif()

if(CLIO_CORE_ENABLE_SYCL)
  message(STATUS "Building SYCL GPU unit tests (Intel oneAPI)")
  add_executable(test_gpu_sycl test_gpu_sycl.cc)
  target_compile_definitions(test_gpu_sycl PRIVATE CTP_ENABLE_SYCL=1)
  wrp_core_apply_sycl_flags(test_gpu_sycl)
  target_link_libraries(test_gpu_sycl
    Catch2::Catch2WithMain
  )
  add_test(NAME test_gpu_sycl COMMAND test_gpu_sycl)
  set_tests_properties(test_gpu_sycl PROPERTIES LABELS "gpu;sycl")
endif()

if(NOT CLIO_CORE_ENABLE_CUDA AND NOT CLIO_CORE_ENABLE_ROCM AND NOT CLIO_CORE_ENABLE_SYCL)
  message(STATUS "GPU tests disabled (CLIO_CORE_ENABLE_CUDA, CLIO_CORE_ENABLE_ROCM, and CLIO_CORE_ENABLE_SYCL are all OFF)")
endif()
