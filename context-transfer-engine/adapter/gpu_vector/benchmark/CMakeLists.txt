#------------------------------------------------------------------------------
# gpu_vector benchmarks
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
  # The weights and stream benches exercise COMPRESSED tiers, so they link the
  # compressor pool. With CLIO_CTE_ENABLE_COMPRESS=OFF those targets do not
  # exist and CMake silently degrades the names to bare -l flags -- the GPU
  # conda job (cuda-release preset, compress off) then died at final link
  # ("cannot find -lclio_cte_compressor_client"). No compressor, no
  # compressed-mode benches; the raw-mode ones below still build.
  if(TARGET clio_cte_compressor_runtime)
    add_cuda_executable(clio_gpu_vector_weights_bench FALSE
        clio_gpu_vector_weights_bench.cc
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
    set_target_properties(clio_gpu_vector_weights_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_gpu_vector_weights_bench RUNTIME DESTINATION bin)

    add_cuda_executable(clio_gpu_vector_grayscott_bench FALSE
        clio_gpu_vector_grayscott_bench.cc
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
    set_target_properties(clio_gpu_vector_grayscott_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_gpu_vector_grayscott_bench RUNTIME DESTINATION bin)

    add_cuda_executable(clio_gpu_vector_kmeans_bench FALSE
        clio_gpu_vector_kmeans_bench.cc
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
    set_target_properties(clio_gpu_vector_kmeans_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_gpu_vector_kmeans_bench RUNTIME DESTINATION bin)

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
  endif()  # TARGET clio_cte_compressor_runtime

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

    # eternia-MD: the from-scratch LAMMPS-melt reimplementation with all
    # state device-canonical on paged vectors (design: core/eternia.md).
    add_cuda_executable(clio_gpu_vector_md_bench FALSE
        clio_gpu_vector_md_bench.cc
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
    set_target_properties(clio_gpu_vector_md_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_gpu_vector_md_bench RUNTIME DESTINATION bin)
    # REGISTER CEILING for the coroutine kernels, MEASURED rather than typed.
    # Clang's coroutine lowering allocates ~138 registers regardless of body,
    # which on sm_89 collapses these kernels to 1 block/SM -- and they are
    # LATENCY bound, not throughput bound: deleting the entire pair loop moved
    # wall time by 3% while device clocks attributed 56% of block time to it.
    # Occupancy is the lever.
    #
    # This used to be `-Xcuda-ptxas --maxrregcount=64`, which is translation-
    # unit wide: it clamped every kernel in the file to one hand-picked number,
    # plain kernels included. clio_coro_regcap() instead builds a probe copy of
    # this target with the ceiling off, reads the REAL per-kernel register
    # counts out of its cubin, and caps exactly the kernels that execute a
    # device coroutine and exceed the occupancy budget. Nothing here names a
    # register count. See cmake/ClioCoroRegCap.cmake; the occupancy target is
    # CLIO_CORO_TARGET_BLOCKS (default 4 blocks/SM at 256 threads = 64 regs,
    # which is what this target was validated at).
    clio_coro_regcap(clio_gpu_vector_md_bench)
    # MD_DEVTRACE=ON turns on the fault path's own device printf tracing
    # (CLIO_YCORO_DEVTRACE in device_vector.h) for debugging wedges.
    if(MD_DEVTRACE)
      target_compile_definitions(clio_gpu_vector_md_bench PRIVATE
          CLIO_YCORO_DEVTRACE)
    endif()

    # CORRECTNESS GATES AS TESTS. The bench already exits non-zero unless the
    # statics, resort and NVE gates all pass, so registering it is enough to
    # make the LAMMPS comparison a regression gate rather than something a
    # human has to remember to eyeball.
    #
    # Two entries, deliberately: one per REGIME. The resident entry asserts
    # the zero-fault contract (a cache that can hold the working set must
    # never page); the out-of-core entry runs the same physics through a
    # cache far too small for it and must produce the same energies. A
    # single entry could not check both, and separate entries are separate
    # processes -- CLIO_INIT runs once per process.
    add_test(NAME cte_gpu_vector_md_resident
        COMMAND clio_gpu_vector_md_bench --md --lattice 20 --steps 20
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_md_resident PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;md"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    # 11 pages of working set through an 8-slot cache. CLIO_GV_ADMIT=1 is
    # REQUIRED here, not incidental: without admission this configuration
    # (64 blocks against 8 slots) wedges rather than merely thrashing, which
    # is the documented concurrency-vs-capacity trade, not a defect in the
    # physics being gated.
    add_test(NAME cte_gpu_vector_md_ooc
        COMMAND clio_gpu_vector_md_bench --md --lattice 20 --steps 20
                --slots 8
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_md_ooc PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;md"
        ENVIRONMENT "CLIO_GV_ADMIT=1"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )

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

#------------------------------------------------------------------------------
# eternia-MD, NVSHMEM edition: the scale-out baseline for the paged MD bench.
#
# Built by NVCC through a custom rule rather than add_cuda_executable(), and
# the reason is not style. The NVSHMEM device library is a relocatable-device-
# code archive, so it needs -rdc=true and nvlink; this tree sets
# CMAKE_CUDA_COMPILER to clang for the device-coroutine kernels and turns
# separable compilation OFF there (see CLIO_CUDA_SEPARABLE in
# cmake/ClioCoreCommon.cmake). The two cannot share one target.
#
# It links NOTHING from clio -- no runtime, no CTE, no gpu_vector. A baseline
# that shares a substrate with the thing it is benchmarked against is not a
# baseline.
#------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_CUDA AND CLIO_CORE_ENABLE_NVSHMEM)
    find_path(NVSHMEM_INCLUDE_DIR nvshmem.h
        HINTS ENV NVSHMEM_HOME ENV NVSHMEM_PREFIX
        PATH_SUFFIXES include nvshmem/include)
    find_library(NVSHMEM_HOST_LIB NAMES nvshmem_host libnvshmem_host.so.3
        HINTS ENV NVSHMEM_HOME ENV NVSHMEM_PREFIX
        PATH_SUFFIXES lib lib64 nvshmem/lib)
    find_library(NVSHMEM_DEVICE_LIB NAMES nvshmem_device
        HINTS ENV NVSHMEM_HOME ENV NVSHMEM_PREFIX
        PATH_SUFFIXES lib lib64 nvshmem/lib)
    find_program(NVSHMEM_NVCC nvcc
        HINTS ${CUDAToolkit_BIN_DIR} ENV CUDA_HOME
        PATH_SUFFIXES bin)

    if(NVSHMEM_INCLUDE_DIR AND NVSHMEM_HOST_LIB AND NVSHMEM_DEVICE_LIB
       AND NVSHMEM_NVCC)
        get_filename_component(NVSHMEM_LIB_DIR ${NVSHMEM_HOST_LIB} DIRECTORY)

        # The arch list nvcc needs. CMAKE_CUDA_ARCHITECTURES entries carry
        # CMake suffixes ("89-real"); strip them.
        set(_nvs_arch_flags "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
            list(APPEND _nvs_arch_flags "-gencode=arch=compute_${_arch},code=sm_${_arch}")
        endforeach()
        if(NOT _nvs_arch_flags)
            list(APPEND _nvs_arch_flags "-arch=sm_80")
        endif()

        # MPI bootstrap when MPI is available: `mpirun -n N` then gives one PE
        # per GPU. Without it the bench still builds and runs single-PE on a
        # unique-ID bootstrap, which is what a one-GPU workstation gets.
        # Compile flags and link libraries are kept apart on purpose: nvcc
        # takes anything before -o as an INPUT, so an MPI .so listed with the
        # -I flags is handed to the preprocessor as a source file.
        find_package(MPI QUIET COMPONENTS CXX)
        set(_nvs_mpi_cflags "")
        set(_nvs_mpi_libs "")
        if(MPI_CXX_FOUND)
            set(_nvs_mpi_cflags -DMD_NVSHMEM_USE_MPI)
            foreach(_dir IN LISTS MPI_CXX_INCLUDE_DIRS)
                list(APPEND _nvs_mpi_cflags -I${_dir})
            endforeach()
            # Handed to the LINKER by absolute path. `-x cu` applies to every
            # input nvcc classifies as a source, and a bare .so on the command
            # line is one of those -- it gets preprocessed as CUDA and the
            # build dies in a wall of "null character(s) preserved in literal".
            foreach(_lib IN LISTS MPI_CXX_LIBRARIES)
                list(APPEND _nvs_mpi_libs -Xlinker ${_lib})
            endforeach()
        endif()

        set(_nvs_out ${CMAKE_BINARY_DIR}/bin/clio_md_nvshmem_bench)
        set(_nvs_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_md_nvshmem_bench.cc)
        add_custom_command(
            OUTPUT ${_nvs_out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
            COMMAND ${NVSHMEM_NVCC}
                -x cu -std=c++17 -O3 -rdc=true ${_nvs_arch_flags}
                -I${NVSHMEM_INCLUDE_DIR} ${_nvs_mpi_cflags}
                ${_nvs_src} -o ${_nvs_out}
                -L${NVSHMEM_LIB_DIR} -lnvshmem_device -lnvshmem_host
                ${_nvs_mpi_libs}
                -Xlinker -rpath -Xlinker ${NVSHMEM_LIB_DIR}
            DEPENDS ${_nvs_src}
            COMMENT "Building clio_md_nvshmem_bench (nvcc + NVSHMEM device rdc)"
            VERBATIM)
        add_custom_target(clio_md_nvshmem_bench ALL DEPENDS ${_nvs_out})
        install(PROGRAMS ${_nvs_out} DESTINATION bin)
        message(STATUS "eternia-MD NVSHMEM baseline: ON (${NVSHMEM_INCLUDE_DIR})")
    else()
        message(STATUS
            "CLIO_CORE_ENABLE_NVSHMEM=ON but NVSHMEM was not found "
            "(set NVSHMEM_HOME); skipping clio_md_nvshmem_bench")
    endif()
endif()

#------------------------------------------------------------------------------
# eternia-MD, MPI edition: the SAME kernels as the NVSHMEM baseline with the
# one variable changed -- two-sided, host-staged halo exchange instead of
# one-sided in-kernel RMA. Built by nvcc for symmetry with its sibling (it
# needs no rdc of its own), and like the sibling it links nothing from clio.
#
# Gated on MPI alone, NOT on CLIO_CORE_ENABLE_NVSHMEM: the whole point is
# that this baseline has no NVSHMEM dependency.
#------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_CUDA)
    find_package(MPI QUIET COMPONENTS CXX)
    find_program(MD_MPI_NVCC nvcc
        HINTS ${CUDAToolkit_BIN_DIR} ENV CUDA_HOME
        PATH_SUFFIXES bin)
    if(MPI_CXX_FOUND AND MD_MPI_NVCC)
        set(_mpi_arch_flags "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
            list(APPEND _mpi_arch_flags "-gencode=arch=compute_${_arch},code=sm_${_arch}")
        endforeach()
        if(NOT _mpi_arch_flags)
            list(APPEND _mpi_arch_flags "-arch=sm_80")
        endif()
        set(_mpi_cflags "")
        foreach(_dir IN LISTS MPI_CXX_INCLUDE_DIRS)
            list(APPEND _mpi_cflags -I${_dir})
        endforeach()
        # Link libs by absolute path through -Xlinker: with `-x cu` a bare
        # .so on the command line is classified as a SOURCE and handed to the
        # CUDA preprocessor.
        set(_mpi_libs "")
        foreach(_lib IN LISTS MPI_CXX_LIBRARIES)
            list(APPEND _mpi_libs -Xlinker ${_lib})
        endforeach()

        set(_mpi_out ${CMAKE_BINARY_DIR}/bin/clio_md_mpi_bench)
        set(_mpi_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_md_mpi_bench.cc)
        add_custom_command(
            OUTPUT ${_mpi_out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
            COMMAND ${MD_MPI_NVCC}
                -x cu -std=c++17 -O3 ${_mpi_arch_flags} ${_mpi_cflags}
                ${_mpi_src} -o ${_mpi_out} ${_mpi_libs}
            DEPENDS ${_mpi_src}
            COMMENT "Building clio_md_mpi_bench (nvcc + MPI)"
            VERBATIM)
        add_custom_target(clio_md_mpi_bench ALL DEPENDS ${_mpi_out})
        install(PROGRAMS ${_mpi_out} DESTINATION bin)
        message(STATUS "eternia-MD MPI baseline: ON")

        #----------------------------------------------------------------------
        # eternia-MD, NCCL edition: same kernels again, halo and migrants moved
        # by ncclSend/ncclRecv on DEVICE pointers. Needs MPI too (to bootstrap
        # the communicator and to carry migrant counts), so it is nested inside
        # the MPI block rather than duplicating that discovery.
        #----------------------------------------------------------------------
        find_path(NCCL_INCLUDE_DIR nccl.h
            HINTS ENV NCCL_HOME ENV NCCL_PREFIX
            PATH_SUFFIXES include nccl/include)
        find_library(NCCL_LIB NAMES nccl libnccl.so.2
            HINTS ENV NCCL_HOME ENV NCCL_PREFIX
            PATH_SUFFIXES lib lib64 nccl/lib)
        if(NCCL_INCLUDE_DIR AND NCCL_LIB)
            get_filename_component(NCCL_LIB_DIR ${NCCL_LIB} DIRECTORY)
            set(_nccl_out ${CMAKE_BINARY_DIR}/bin/clio_md_nccl_bench)
            set(_nccl_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_md_nccl_bench.cc)
            add_custom_command(
                OUTPUT ${_nccl_out}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
                COMMAND ${MD_MPI_NVCC}
                    -x cu -std=c++17 -O3 ${_mpi_arch_flags} ${_mpi_cflags}
                    -I${NCCL_INCLUDE_DIR}
                    ${_nccl_src} -o ${_nccl_out}
                    -L${NCCL_LIB_DIR} -lnccl ${_mpi_libs}
                    -Xlinker -rpath -Xlinker ${NCCL_LIB_DIR}
                DEPENDS ${_nccl_src}
                COMMENT "Building clio_md_nccl_bench (nvcc + NCCL + MPI)"
                VERBATIM)
            add_custom_target(clio_md_nccl_bench ALL DEPENDS ${_nccl_out})
            install(PROGRAMS ${_nccl_out} DESTINATION bin)
            message(STATUS "eternia-MD NCCL baseline: ON (${NCCL_INCLUDE_DIR})")
        else()
            message(STATUS
                "NCCL not found (set NCCL_HOME); skipping clio_md_nccl_bench")
        endif()
    else()
        message(STATUS "MPI not found; skipping clio_md_mpi_bench")
    endif()
endif()

#------------------------------------------------------------------------------
# eternia-MD, BaM edition: the capacity sibling rather than a transport one.
# Single rank, no MPI; the Verlet list lives behind BaM's GPU page cache over
# pinned host DRAM. Gated on CLIO_CORE_ENABLE_BAM, which is what builds the
# in-tree external/bam module this links against.
#
# It compiles BaM's four sources ITSELF rather than linking the `bam` target,
# and that is not gratuitous: external/bam passes --expt-relaxed-constexpr
# unconditionally, which clang rejects outright ("unknown argument"), so the
# bam target cannot build at all in a tree whose CMAKE_CUDA_COMPILER is clang
# -- as this one's is, for the device coroutines. Compiling here with nvcc
# keeps the bench buildable in every configuration and matches its three
# siblings, none of which link anything from the tree either.
#------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_CUDA AND EXISTS ${CMAKE_SOURCE_DIR}/external/bam/src/array.cu)
    find_program(MD_BAM_NVCC nvcc
        HINTS ${CUDAToolkit_BIN_DIR} ENV CUDA_HOME
        PATH_SUFFIXES bin)
    if(MD_BAM_NVCC)
        set(_bam_arch_flags "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
            list(APPEND _bam_arch_flags "-gencode=arch=compute_${_arch},code=sm_${_arch}")
        endforeach()
        if(NOT _bam_arch_flags)
            list(APPEND _bam_arch_flags "-arch=sm_80")
        endif()
        set(_bam_out ${CMAKE_BINARY_DIR}/bin/clio_md_bam_bench)
        set(_bam_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_md_bam_bench.cc)
        add_custom_command(
            OUTPUT ${_bam_out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
            COMMAND ${MD_BAM_NVCC}
                -x cu -std=c++17 -O3 -rdc=true --expt-relaxed-constexpr
                ${_bam_arch_flags}
                -I${CMAKE_SOURCE_DIR}/external/bam/include
                ${_bam_src}
                ${CMAKE_SOURCE_DIR}/external/bam/src/array.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/controller.cc
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache_host.cc
                -o ${_bam_out}
            # DEPENDS must list the BaM sources too. With only the bench
            # here, edits under external/bam/ never relinked and the build
            # silently kept handing back a stale binary.
            DEPENDS ${_bam_src}
                ${CMAKE_SOURCE_DIR}/external/bam/src/array.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/controller.cc
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache_host.cc
                ${CMAKE_SOURCE_DIR}/external/bam/include/bam/array.cuh
                ${CMAKE_SOURCE_DIR}/external/bam/include/bam/page_cache.cuh
                ${CMAKE_SOURCE_DIR}/external/bam/include/bam/types.h
                ${CMAKE_SOURCE_DIR}/external/bam/include/bam/page_cache_host.h
            COMMENT "Building clio_md_bam_bench (nvcc + BaM page cache)"
            VERBATIM)
        add_custom_target(clio_md_bam_bench ALL DEPENDS ${_bam_out})
        install(PROGRAMS ${_bam_out} DESTINATION bin)
        message(STATUS "eternia-MD BaM baseline: ON")
    endif()
endif()
