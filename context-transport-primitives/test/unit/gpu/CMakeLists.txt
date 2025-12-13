#------------------------------------------------------------------------------
# GPU Unit Tests
#------------------------------------------------------------------------------

if(HSHM_ENABLE_CUDA OR HSHM_ENABLE_ROCM)
  message(STATUS "Building GPU unit tests")

  # GpuShmMmap test
  add_cuda_executable(test_gpu_shm_mmap TRUE test_gpu_shm_mmap.cc)
  target_link_libraries(test_gpu_shm_mmap
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_gpu_shm_mmap COMMAND test_gpu_shm_mmap)

  # GpuMalloc test
  add_cuda_executable(test_gpu_malloc TRUE test_gpu_malloc.cc)
  target_link_libraries(test_gpu_malloc
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_gpu_malloc COMMAND test_gpu_malloc)

else()
  message(STATUS "GPU tests disabled (HSHM_ENABLE_CUDA and HSHM_ENABLE_ROCM are both OFF)")
endif()
