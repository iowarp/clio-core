#------------------------------------------------------------------------------
# gpu_vector benchmarks
#
# One SUBDIRECTORY per science workload, each holding every substrate
# implementation of that workload (paged on the vector, plus mpi / nvshmem /
# nccl / bam baselines where they exist -- a baseline links nothing from
# clio, or it is not a baseline):
#
#   lammps_md/   LAMMPS melt        paged + mpi + nvshmem + nccl + bam
#   gmx/         PME spread+gather  paged + mpi + nvshmem + nccl + bam
#   lbann/       MLP training       paged + mpi + nvshmem + nccl + bam
#   grayscott/   reaction-diffusion paged + mpi + nvshmem + nccl + bam
#   kmeans/      k-means            paged + mpi + nvshmem + nccl + bam
#   weights/     weight streaming   paged + mpi + nvshmem + nccl + bam
#
# ALL SIX CARRY ALL FIVE SUBSTRATES. This list said "paged" only for five of
# them long after the mpi/nvshmem/bam editions landed, and the nccl column
# was genuinely lammps_md-only until the five NCCL editions were written --
# which is how a "compare paged against MPI/NVSHMEM/NCCL" campaign ran for
# weeks with no NCCL number for five of its six workloads. Keep this table
# honest; it is the only place the matrix is written down.
#   gnn/         GNN ingest + 1-hop aggregation tooling (host-side)
#
# The files remaining here are the vector's own micro-benchmarks (access,
# flush, overlap, stream) and the register-cap probes.
#------------------------------------------------------------------------------

# Workloads split into shared kernels plus per-backend launches
# (<wl>_kernels.h over <wl>_math.h, plus <wl>/cuda and <wl>/sycl). These are
# the ones a SYCL build can descend into -- which is now all six.
# NVSHMEM BOOTSTRAP. The MPI bootstrap is a dlopen'd plugin NVIDIA builds
# against OpenMPI, so it needs libmpi.so.40 and an MPI that can actually run
# multi-node. On a Cray machine neither holds: the system MPI is cray-mpich,
# and the HPC-X OpenMPI that ships with nvhpc cannot connect its UCX PML
# across Slingshot ("ucp_ep_create failed: Destination is unreachable").
# Turning this OFF builds the benches to call nvshmem_init() instead, which
# bootstraps through the LAUNCHER (NVSHMEM_BOOTSTRAP=PMI + srun --mpi=pmi2)
# -- the path NVIDIA's own perftest binaries use, and the one measured
# working multi-node on Delta at 11.3 GB/s put bandwidth over 2 nodes.
option(CLIO_GV_NVSHMEM_MPI_BOOTSTRAP
       "Bootstrap the NVSHMEM baselines through MPI (OFF = through the launcher, via nvshmem_init)"
       ON)

add_subdirectory(kmeans)
add_subdirectory(grayscott)
add_subdirectory(gmx)
add_subdirectory(weights)
add_subdirectory(lbann)
# lammps_md joined them: its seam split (md_kernels.h over md_common.h, plus
# lammps_md/cuda and lammps_md/sycl) was the last one, and the __shared__ usage
# that made it the outlier turned out to be only the reduction scratch -- the
# block-uniform tables were already on the global-backed CLIO_SHARED_PERSIST
# arena, which SYCL supports unchanged.
add_subdirectory(lammps_md)

# gnn is not a paged-vector workload at all -- it is an ingest/aggregate tool
# with a different shape -- so it has no SYCL row to port and stays here.
if(CLIO_CORE_ENABLE_CUDA OR CLIO_CORE_ENABLE_ROCM)
    add_subdirectory(gnn)
endif()

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
