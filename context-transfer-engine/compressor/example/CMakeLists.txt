# ------------------------------------------------------------------------------
# Direct GPU-buffer compression example (issue #693): hands the compressor a
# CUDA-IPC-registered device pointer and lets NeuroPress pick the algorithm
# per chunk, with no HDF5 layer in between. Complements the VOL demo in
# adapter/hdf5_vol/example -- this is the shorter path and the one that
# actually keeps statistics on-device.
#
# Two translation units on purpose: the .cu holds only kernels (no Clio
# includes), the .cc holds all Clio/CTE code. clio_cte/compressor headers do
# not compile under nvcc. The kernel goes into its own small shared library
# via add_cuda_library so the executable links it like any other lib and
# needs no device-link step of its own.
# ------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_CUDA)
  add_cuda_library(neuropress_gpu_direct_kernel SHARED FALSE
    neuropress_gpu_direct_kernel.cu
  )

  add_executable(neuropress_gpu_direct neuropress_gpu_direct.cc)

  target_compile_definitions(neuropress_gpu_direct PRIVATE
    CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

  target_link_libraries(neuropress_gpu_direct
    neuropress_gpu_direct_kernel
    clio_cte_compressor_runtime
    clio_cte_compressor_client
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ctp::cuda_cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )

  install(TARGETS neuropress_gpu_direct neuropress_gpu_direct_kernel
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
  )

  # The K-way exploration sweep, with the lossy half of the action space
  # enabled. Reuses neuropress_gpu_direct_kernel for its data generation and
  # bound check rather than carrying a near-identical .cu of its own.
  add_executable(neuropress_explore_sweep neuropress_explore_sweep.cc)

  target_compile_definitions(neuropress_explore_sweep PRIVATE
    CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

  target_link_libraries(neuropress_explore_sweep
    neuropress_gpu_direct_kernel
    clio_cte_compressor_runtime
    clio_cte_compressor_client
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ctp::cuda_cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )

  install(TARGETS neuropress_explore_sweep
    RUNTIME DESTINATION bin
  )

  # Cross-project callback-trace equivalence harness. Skips itself when the
  # native NeuroPress build or CUPTI is absent, since neither is a Clio build
  # dependency.
  add_subdirectory(neuropress_gpu_chunk_equivalence)

  # Where do the bytes actually go? Traces one GPU-generated chunk through
  # Clio's real store path. Needs CUPTI but not native NeuroPress.
  add_subdirectory(neuropress_data_path_trace)

  # Which transfers are REDUNDANT? Same real store path at 2 x 256 MiB, with
  # NVTX ranges so nsys can attribute each copy to a phase. Needs NVTX, not
  # CUPTI -- the profiler here is external.
  add_subdirectory(neuropress_transfer_audit)
endif()
