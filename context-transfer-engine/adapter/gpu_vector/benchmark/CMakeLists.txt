#------------------------------------------------------------------------------
# gpu_vector benchmarks
#
# One SUBDIRECTORY per science workload, each holding every substrate
# implementation of that workload (paged on the vector, plus mpi / nvshmem /
# nccl / bam baselines where they exist -- a baseline links nothing from
# clio, or it is not a baseline):
#
#   lammps_md/   LAMMPS melt        paged + mpi + nvshmem + nccl + bam
#   gmx/         PME spread+gather  paged
#   lbann/       MLP training       paged
#   grayscott/   reaction-diffusion paged
#   kmeans/      k-means            paged
#   weights/     weight streaming   paged
#
# The files remaining here are the vector's own micro-benchmarks (access,
# flush, overlap, stream) and the register-cap probes.
#------------------------------------------------------------------------------

add_subdirectory(lammps_md)
add_subdirectory(gmx)
add_subdirectory(lbann)
add_subdirectory(grayscott)
add_subdirectory(kmeans)
add_subdirectory(weights)

if(CLIO_CORE_ENABLE_CUDA)
  # The stream bench exercises COMPRESSED tiers, so it links the compressor
  # pool; without that pool the target cannot link.
  if(TARGET clio_cte_compressor_runtime)
    add_cuda_executable(clio_gpu_vector_stream_bench FALSE
        clio_gpu_vector_stream_bench.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
        LINK_LIBS
            clio_run_cxx
            clio_run_cxx_gpu
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            clio_cte_compressor_client
            clio_cte_compressor_runtime
            clio::cte::gpu_vector
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(clio_gpu_vector_stream_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_gpu_vector_stream_bench RUNTIME DESTINATION bin)
  endif()

    add_cuda_executable(clio_gpu_vector_overlap_bench FALSE
        clio_gpu_vector_overlap_bench.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
        LINK_LIBS
            clio_run_cxx
            clio_run_cxx_gpu
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            clio::cte::gpu_vector
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(clio_gpu_vector_overlap_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_gpu_vector_overlap_bench RUNTIME DESTINATION bin)

    add_cuda_executable(clio_gpu_vector_flush_bench FALSE
        clio_gpu_vector_flush_bench.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
        LINK_LIBS
            clio_run_cxx
            clio_run_cxx_gpu
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            clio::cte::gpu_vector
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(clio_gpu_vector_flush_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_gpu_vector_flush_bench RUNTIME DESTINATION bin)

    add_cuda_executable(clio_gpu_vector_access_bench FALSE
        clio_gpu_vector_access_bench.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
        LINK_LIBS
            clio_run_cxx
            clio_run_cxx_gpu
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            clio::cte::gpu_vector
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(clio_gpu_vector_access_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_gpu_vector_access_bench RUNTIME DESTINATION bin)
endif()
