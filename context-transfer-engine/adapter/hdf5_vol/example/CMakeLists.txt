# ------------------------------------------------------------------------------
# NeuroPress GPU demo (issue #693): generates data directly on the GPU and
# writes it through the Clio HDF5 VOL connector with a compressor pool
# configured, so each chunk is routed through NeuroPress's dynamic algorithm
# selection (compressor::Client::AsyncDynamicSchedule) instead of the raw
# uncompressed core-client path. Only built when CUDA is enabled -- without a
# real device pointer there is nothing GPU-resident to demonstrate.
#
# Two translation units, deliberately not merged: neuropress_gpu_demo_kernel.cu
# holds ONLY the simulation kernel (no Clio includes), built as its own SHARED
# CUDA library via add_cuda_library -- the same pattern context-transport-
# primitives/src/CMakeLists.txt uses for clio_ctp_cuda -- so its device code is
# self-contained (device-linked once, at the library's own build time). A
# plain add_executable then links it like any other shared library, with no
# device-link step of its own. neuropress_gpu_demo.cc (plain host TU, not
# compiled by nvcc) holds all the Clio/HDF5 driver code: clio_cte/compressor
# headers do not compile cleanly under nvcc (see the comment atop
# neuropress_gpu_demo_kernel.cu), the same reason
# context-transfer-engine/core/{include/.../core_client.h,src/core_client.cc}
# are split that way elsewhere in this project. Compiling the kernel and the
# executable as ONE mixed CXX+CUDA target (tried first) hit an unrelated nvlink
# failure ("unexpected object after cudadevrt") from device-linking against
# the other already-RDC-compiled GPU libraries this target also needs
# (ctp::cuda_cxx, clio_run_cxx_gpu); going through a small shared library
# sidesteps it entirely.
# ------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_CUDA)
  add_cuda_library(neuropress_gpu_demo_kernel SHARED FALSE
    neuropress_gpu_demo_kernel.cu
  )

  add_executable(neuropress_gpu_demo
    neuropress_gpu_demo.cc
  )

  target_include_directories(neuropress_gpu_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/hdf5_vol
    ${HDF5_INCLUDE_DIRS}
  )

  # Same trained weights path test_compressor_functional.cc's NeuroPress
  # test points at -- without this, CompressorConfig::neuropress_model_path_
  # stays empty and compressor_runtime.cc never loads the real predictor,
  # silently falling back to the legacy heuristic candidate list.
  target_compile_definitions(neuropress_gpu_demo PRIVATE
    CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

  target_link_libraries(neuropress_gpu_demo
    neuropress_gpu_demo_kernel
    clio_hdf5_vol
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
    ${HDF5_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
  )

  install(TARGETS neuropress_gpu_demo neuropress_gpu_demo_kernel
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
  )

  # ----------------------------------------------------------------------
  # NeuroPress GPU -> HDF5 end-to-end run: 1 GiB in 4 MiB chunks, inference
  # + online learning, every selection logged. Split into the same two
  # translation units and for the same reason as the demo above.
  # ----------------------------------------------------------------------
  add_cuda_library(neuropress_e2e_kernel SHARED FALSE
    neuropress_e2e_kernel.cu
  )

  add_executable(neuropress_e2e
    neuropress_e2e.cc
  )

  target_include_directories(neuropress_e2e PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/hdf5_vol
    ${HDF5_INCLUDE_DIRS}
  )

  target_compile_definitions(neuropress_e2e PRIVATE
    CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

  target_link_libraries(neuropress_e2e
    neuropress_e2e_kernel
    clio_hdf5_vol
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
    ${HDF5_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
  )

  install(TARGETS neuropress_e2e neuropress_e2e_kernel
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
  )
endif()
