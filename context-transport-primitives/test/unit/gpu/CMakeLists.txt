#------------------------------------------------------------------------------
# GPU Unit Tests
#------------------------------------------------------------------------------

if(WRP_CORE_ENABLE_CUDA OR WRP_CORE_ENABLE_ROCM)
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

  # LocalSerialize GPU test
  add_cuda_executable(test_local_serialize_gpu TRUE test_local_serialize_gpu.cc)
  target_link_libraries(test_local_serialize_gpu
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_local_serialize_gpu COMMAND test_local_serialize_gpu)

  # LocalTransfer GPU test
  add_cuda_executable(test_local_transfer_gpu TRUE test_local_transfer_gpu.cc)
  target_link_libraries(test_local_transfer_gpu
    hshm::cuda_cxx
    Catch2::Catch2WithMain
  )
  add_test(NAME test_local_transfer_gpu COMMAND test_local_transfer_gpu)

  add_subdirectory(runtime)

else()
  message(STATUS "GPU tests disabled (WRP_CORE_ENABLE_CUDA and WRP_CORE_ENABLE_ROCM are both OFF)")
endif()
