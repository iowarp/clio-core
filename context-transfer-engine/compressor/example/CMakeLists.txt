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
# Only neuropress_grayscott_h5/ is tracked in the repository; the sibling
# harnesses here are development and parity tooling kept out of the branch
# (see .gitignore). Every entry point below is therefore guarded on the files
# it needs, so a checkout that has only the tracked demo still configures --
# same reason 009c6b67 guarded add_subdirectory(example) itself.
if(CLIO_CORE_ENABLE_CUDA)
 if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_gpu_direct_kernel.cu)
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

  # Phase bench: one fixed 2 GiB workload, any of the three NeuroPress phases,
  # every knob that distinguishes them on the command line. Reuses
  # neuropress_gpu_direct_kernel for data generation and the bound check.
  add_executable(neuropress_phase_bench neuropress_phase_bench.cc)

  target_compile_definitions(neuropress_phase_bench PRIVATE
    CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

  target_link_libraries(neuropress_phase_bench
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

  install(TARGETS neuropress_phase_bench
    RUNTIME DESTINATION bin
  )

  # LOCAL parity driver: same three modes as native, for apples-to-apples runs.
  add_executable(neuropress_mode_parity neuropress_mode_parity.cc)
  target_compile_definitions(neuropress_mode_parity PRIVATE
    CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")
  target_link_libraries(neuropress_mode_parity
    neuropress_gpu_direct_kernel clio_cte_compressor_runtime
    clio_cte_compressor_client clio_cte_core_runtime clio_cte_core_client
    clio::run::admin_runtime clio::run::admin_client
    clio::run::bdev_runtime clio::run::bdev_client
    ctp::cxx ctp::cuda_cxx ${CMAKE_THREAD_LIBS_INIT})

  endif()

  # Cross-project callback-trace equivalence harness. Skips itself when the
  # native NeuroPress build or CUPTI is absent, since neither is a Clio build
  # dependency.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_gpu_chunk_equivalence/CMakeLists.txt)
    add_subdirectory(neuropress_gpu_chunk_equivalence)
  endif()

  # Where do the bytes actually go? Traces one GPU-generated chunk through
  # Clio's real store path. Needs CUPTI but not native NeuroPress.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_data_path_trace/CMakeLists.txt)
    add_subdirectory(neuropress_data_path_trace)
  endif()

  # Which transfers are REDUNDANT? Same real store path at 2 x 256 MiB, with
  # NVTX ranges so nsys can attribute each copy to a phase. Needs NVTX, not
  # CUPTI -- the profiler here is external.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_transfer_audit/CMakeLists.txt)
    add_subdirectory(neuropress_transfer_audit)
  endif()

  # Gray-Scott through the HDF5 VOL with NO Clio code in the application --
  # the deployment shape, as opposed to the sibling harnesses which drive the
  # compressor API directly. See its README for the env recipe.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_grayscott_h5/CMakeLists.txt)
    add_subdirectory(neuropress_grayscott_h5)
  endif()

  # LAMMPS linked IN as a library, Clio's runtime hosted in the same process,
  # the driver reading Atom::x/v/f between `run` segments -- the coupling-code
  # shape, with no HDF5 and no LAMMPS patch. Its own CMakeLists skips itself
  # when no LAMMPS build is found (LAMMPS_SRC_DIR / LAMMPS_BUILD_DIR).
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_lammps_lib/CMakeLists.txt)
    add_subdirectory(neuropress_lammps_lib)
  endif()
endif()
