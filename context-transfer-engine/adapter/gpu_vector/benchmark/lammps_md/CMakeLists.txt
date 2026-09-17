#------------------------------------------------------------------------------
# lammps_md: the from-scratch LAMMPS-melt reimplementation, on five substrates.
#
#   paged    all state device-canonical on paged vectors (design: adapter/gpu_vector/eternia.md)
#   nvshmem  symmetric heap, one-sided in-kernel RMA        (links nothing from clio)
#   mpi      two-sided, host-staged halo exchange           (links nothing from clio)
#   nccl     halo + migrants via ncclSend/Recv on device ptrs
#   bam      capacity sibling: Verlet list behind BaM's GPU page cache
#
# The baselines are built by NVCC through custom rules rather than
# add_cuda_executable(), and the reason is not style: NVSHMEM's device library
# is an rdc archive needing nvlink, while this tree sets CMAKE_CUDA_COMPILER
# to clang for the device-coroutine kernels with separable compilation OFF.
# A baseline that shares a substrate with the thing it is benchmarked against
# is not a baseline, so they link nothing from clio.
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
    add_cuda_executable(clio_lammps_md_paged_bench FALSE
        clio_lammps_md_paged_bench.cc
        cuda/md_launch_cuda.cc
        INCLUDE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}
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
    set_target_properties(clio_lammps_md_paged_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_lammps_md_paged_bench RUNTIME DESTINATION bin)
    # REGISTER CEILING for the coroutine kernels, MEASURED rather than typed.
    # See cmake/ClioCoroRegCap.cmake; clang's coroutine lowering allocates
    # ~138 registers regardless of body and these kernels are latency bound,
    # so occupancy is the lever.
    clio_coro_regcap(clio_lammps_md_paged_bench)
    if(MD_DEVTRACE)
      target_compile_definitions(clio_lammps_md_paged_bench PRIVATE
          CLIO_YCORO_DEVTRACE)
    endif()

    # CORRECTNESS GATES AS TESTS: one entry per REGIME. The resident entry
    # asserts the zero-fault contract; the out-of-core entry runs the same
    # physics through a cache far too small for it and must produce the same
    # energies. Separate entries are separate processes -- CLIO_INIT runs
    # once per process.
    add_test(NAME cte_gpu_vector_lammps_md_resident
        COMMAND clio_lammps_md_paged_bench --md --lattice 20 --steps 20
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_lammps_md_resident PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;lammps_md"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    # 11 pages of working set through an 8-slot cache. CLIO_GV_ADMIT=1 is
    # historical here (admission is always on now) but harmless.
    add_test(NAME cte_gpu_vector_lammps_md_ooc
        COMMAND clio_lammps_md_paged_bench --md --lattice 20 --steps 20
                --slots 8
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    # THE _ooc ENTRY ABOVE DOES NOT EVICT. --slots 8 is clamped up to the
    # 28-frame pin floor and the working set fits, so it reports
    # "resident contract HELD: no evictions" -- it tests the clamp path, not
    # the paging path. THIS entry does: lattice 32 needs 54 pages against a
    # 32-slot cache, which faults ~2700 times and evicts ~2650, and the gates
    # still pass. Keep both; they cover different things.
    add_test(NAME cte_gpu_vector_lammps_md_evict
        COMMAND clio_lammps_md_paged_bench --md --lattice 32 --steps 10
                --blocks 8 --slots 32
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_lammps_md_evict PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )

    set_tests_properties(cte_gpu_vector_lammps_md_ooc PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;lammps_md"
        ENVIRONMENT "CLIO_GV_ADMIT=1"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
endif()

#------------------------------------------------------------------------------
# NVSHMEM edition.
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
        set(_nvs_arch_flags "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
            list(APPEND _nvs_arch_flags "-gencode=arch=compute_${_arch},code=sm_${_arch}")
        endforeach()
        if(NOT _nvs_arch_flags)
            list(APPEND _nvs_arch_flags "-arch=sm_80")
        endif()
        # MPI bootstrap when MPI is available. Compile flags and link libs are
        # kept apart: nvcc takes anything before -o as an INPUT, so an MPI .so
        # listed with the -I flags is preprocessed as a source file.
        find_package(MPI QUIET COMPONENTS CXX)
        set(_nvs_mpi_cflags "")
        set(_nvs_mpi_libs "")
        if(MPI_CXX_FOUND AND CLIO_GV_NVSHMEM_MPI_BOOTSTRAP)
            set(_nvs_mpi_cflags -DMD_NVSHMEM_USE_MPI)
            foreach(_dir IN LISTS MPI_CXX_INCLUDE_DIRS)
                list(APPEND _nvs_mpi_cflags -I${_dir})
            endforeach()
            foreach(_lib IN LISTS MPI_CXX_LIBRARIES)
                list(APPEND _nvs_mpi_libs -Xlinker ${_lib})
            endforeach()
        endif()

        set(_nvs_out ${CMAKE_BINARY_DIR}/bin/clio_lammps_md_nvshmem_bench)
        set(_nvs_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_lammps_md_nvshmem_bench.cc)
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
            COMMENT "Building clio_lammps_md_nvshmem_bench (nvcc + NVSHMEM rdc)"
            VERBATIM)
        add_custom_target(clio_lammps_md_nvshmem_bench ALL DEPENDS ${_nvs_out})
        install(PROGRAMS ${_nvs_out} DESTINATION bin)
        message(STATUS "lammps_md NVSHMEM baseline: ON (${NVSHMEM_INCLUDE_DIR})")
    else()
        message(STATUS
            "CLIO_CORE_ENABLE_NVSHMEM=ON but NVSHMEM was not found "
            "(set NVSHMEM_HOME); skipping clio_lammps_md_nvshmem_bench")
    endif()
endif()

#------------------------------------------------------------------------------
# MPI edition (+ nested NCCL edition, which needs MPI to bootstrap).
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
        set(_mpi_libs "")
        foreach(_lib IN LISTS MPI_CXX_LIBRARIES)
            list(APPEND _mpi_libs -Xlinker ${_lib})
        endforeach()

        set(_mpi_out ${CMAKE_BINARY_DIR}/bin/clio_lammps_md_mpi_bench)
        set(_mpi_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_lammps_md_mpi_bench.cc)
        add_custom_command(
            OUTPUT ${_mpi_out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
            COMMAND ${MD_MPI_NVCC}
                -x cu -std=c++17 -O3 ${_mpi_arch_flags} ${_mpi_cflags}
                ${_mpi_src} -o ${_mpi_out} ${_mpi_libs}
            DEPENDS ${_mpi_src}
            COMMENT "Building clio_lammps_md_mpi_bench (nvcc + MPI)"
            VERBATIM)
        add_custom_target(clio_lammps_md_mpi_bench ALL DEPENDS ${_mpi_out})
        install(PROGRAMS ${_mpi_out} DESTINATION bin)
        message(STATUS "lammps_md MPI baseline: ON")

        find_path(NCCL_INCLUDE_DIR nccl.h
            HINTS ENV NCCL_HOME ENV NCCL_PREFIX
            PATH_SUFFIXES include nccl/include)
        find_library(NCCL_LIB NAMES nccl libnccl.so.2
            HINTS ENV NCCL_HOME ENV NCCL_PREFIX
            PATH_SUFFIXES lib lib64 nccl/lib)
        if(NCCL_INCLUDE_DIR AND NCCL_LIB)
            get_filename_component(NCCL_LIB_DIR ${NCCL_LIB} DIRECTORY)
            set(_nccl_out ${CMAKE_BINARY_DIR}/bin/clio_lammps_md_nccl_bench)
            set(_nccl_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_lammps_md_nccl_bench.cc)
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
                COMMENT "Building clio_lammps_md_nccl_bench (nvcc + NCCL + MPI)"
                VERBATIM)
            add_custom_target(clio_lammps_md_nccl_bench ALL DEPENDS ${_nccl_out})
            install(PROGRAMS ${_nccl_out} DESTINATION bin)
            message(STATUS "lammps_md NCCL baseline: ON (${NCCL_INCLUDE_DIR})")
        else()
            message(STATUS
                "NCCL not found (set NCCL_HOME); skipping clio_lammps_md_nccl_bench")
        endif()
    else()
        message(STATUS "MPI not found; skipping clio_lammps_md_mpi_bench")
    endif()
endif()

#------------------------------------------------------------------------------
# BaM edition: the capacity sibling. Compiles BaM's sources ITSELF with nvcc
# because external/bam passes --expt-relaxed-constexpr, which clang rejects.
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
        set(_bam_out ${CMAKE_BINARY_DIR}/bin/clio_lammps_md_bam_bench)
        set(_bam_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_lammps_md_bam_bench.cc)
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
            DEPENDS ${_bam_src}
                ${CMAKE_SOURCE_DIR}/external/bam/src/array.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/controller.cc
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache_host.cc
                ${CMAKE_SOURCE_DIR}/external/bam/include/bam/array.cuh
                ${CMAKE_SOURCE_DIR}/external/bam/include/bam/page_cache.cuh
                ${CMAKE_SOURCE_DIR}/external/bam/include/bam/types.h
                ${CMAKE_SOURCE_DIR}/external/bam/include/bam/page_cache_host.h
            COMMENT "Building clio_lammps_md_bam_bench (nvcc + BaM page cache)"
            VERBATIM)
        add_custom_target(clio_lammps_md_bam_bench ALL DEPENDS ${_bam_out})
        install(PROGRAMS ${_bam_out} DESTINATION bin)
        message(STATUS "lammps_md BaM baseline: ON")
    endif()
endif()


#------------------------------------------------------------------------------
# eternia-MD, Kokkos edition: the performance-portability baseline.
# ONE SOURCE, whichever backend Kokkos was built for (-DKokkos_ROOT).
#------------------------------------------------------------------------------
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    find_package(MPI QUIET COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        add_executable(clio_lammps_md_kokkos_bench
            kokkos/clio_lammps_md_kokkos_bench.cc)
        target_link_libraries(clio_lammps_md_kokkos_bench PRIVATE
            Kokkos::kokkos MPI::MPI_CXX)
        set_target_properties(clio_lammps_md_kokkos_bench PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        install(TARGETS clio_lammps_md_kokkos_bench RUNTIME DESTINATION bin)

        # REGRESSION COVERAGE. These gates are self-checking (integer-exact
        # counts, or a host reference), so an entry here is worth having even
        # though the other baselines in this directory have none: without it
        # the row is only as correct as the last time someone ran it by hand.
        # No cleanup fixture -- a Kokkos baseline links nothing from clio and
        # so binds no runtime port. RUN_SERIAL only because they share a GPU.
        add_test(NAME bench_lammps_md_kokkos
            COMMAND clio_lammps_md_kokkos_bench --md --lattice 40 --steps 100 --rebin 10 --temp 3.0 --cap 48 --blocks 529 --threads 128 --rowchunk 1 --drift-tol 5e-3
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
        set_tests_properties(bench_lammps_md_kokkos PROPERTIES
            TIMEOUT 900 LABELS "bench;gpu;kokkos" RUN_SERIAL TRUE)

        # ...and the same deck across 2 ranks. The gates that matter here are
        # rank-count INVARIANT by construction (integer reductions), so a
        # divergence between these two entries is a real defect rather than
        # float noise.
        if(MPIEXEC_EXECUTABLE)
            add_test(NAME bench_lammps_md_kokkos_2rank
                COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe
                        ${MPIEXEC_NUMPROC_FLAG} 2
                        $<TARGET_FILE:clio_lammps_md_kokkos_bench> --md --lattice 40 --steps 100 --rebin 10 --temp 3.0 --cap 48 --blocks 529 --threads 128 --rowchunk 1 --drift-tol 5e-3
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            )
            set_tests_properties(bench_lammps_md_kokkos_2rank PROPERTIES
                TIMEOUT 900 LABELS "bench;gpu;kokkos;distributed" RUN_SERIAL TRUE)
        endif()
        message(STATUS "lammps_md Kokkos baseline: ON (backends: ${Kokkos_DEVICES})")
    endif()
endif()


#------------------------------------------------------------------------------
# eternia-MD over a paged vector -- SYCL.
#
# Same host driver source as the CUDA target, compiled once as ordinary C++.
# Only sycl/md_launch_sycl.cc is built with -fsycl; the workload
# (md_kernels.h over md_common.h) is one copy shared by both, and the SYCL
# launchers are GENERATED from the CUDA trampolines. SHARED library for the
# launches -- device code in a static archive does not reach the executable's
# device image. SPIR-V is refused: DPC++ cannot compile a coroutine for it.
#------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_SYCL AND NOT SYCL_TARGET MATCHES "spir64")
    set(_md_inc
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/context-runtime/include
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include)

    add_library(md_sycl_launch SHARED sycl/md_launch_sycl.cc)
    wrp_core_apply_sycl_flags(md_sycl_launch)
    set_target_properties(md_sycl_launch PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_include_directories(md_sycl_launch PUBLIC ${_md_inc})
    target_link_libraries(md_sycl_launch PUBLIC
        clio_run_cxx clio_run_cxx_gpu clio_cte_core_client)

    add_executable(clio_lammps_md_paged_bench_sycl clio_lammps_md_paged_bench.cc)
    target_link_options(clio_lammps_md_paged_bench_sycl PRIVATE
        -fsycl -fsycl-targets=${SYCL_TARGET})
    target_include_directories(clio_lammps_md_paged_bench_sycl PRIVATE
        ${CLIO_SYCL_INCLUDE_DIR} ${_md_inc})
    target_link_libraries(clio_lammps_md_paged_bench_sycl PRIVATE
        md_sycl_launch
        clio_run_cxx clio_run_cxx_gpu clio_admin_client clio_bdev_client
        clio_cte_core_client clio_cte_core_runtime clio::cte::gpu_vector
        ${CMAKE_THREAD_LIBS_INIT})
    set_target_properties(clio_lammps_md_paged_bench_sycl PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    install(TARGETS clio_lammps_md_paged_bench_sycl RUNTIME DESTINATION bin)

    # REGRESSION COVERAGE for the SYCL row. Without an entry here the row is
    # only ever as correct as the last time someone ran it by hand.
    # --blocks 8 is the SYCL backend's documented ceiling, not a taste.
    add_test(NAME cte_gpu_vector_lammps_md_sycl
        COMMAND clio_lammps_md_paged_bench_sycl --md --lattice 20 --steps 20 --blocks 8
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    # Same eviction geometry as cte_gpu_vector_lammps_md_evict. This is the
    # entry that puts the SYCL paging path under real pressure -- ~2900
    # faults and ~2850 evictions -- rather than merely faulting.
    add_test(NAME cte_gpu_vector_lammps_md_evict_sycl
        COMMAND clio_lammps_md_paged_bench_sycl --md --lattice 32 --steps 10
                --blocks 8 --slots 32
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_lammps_md_evict_sycl PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;sycl"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )

    set_tests_properties(cte_gpu_vector_lammps_md_sycl PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;sycl"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    message(STATUS "lammps_md paged bench (SYCL): ON")
endif()

# ---------------------------------------------------------------------------
# The MPI baseline, RUN at 2 ranks.
#
# It was built and never executed by ctest -- so nothing caught it rotting,
# and "the MPI edition works" rested on someone having run it by hand. It
# gates itself (see the bench), so an entry here is worth having.
#
# 2 ranks on one host: this is the baseline's PARALLEL path, not a cluster
# test -- the cross-machine gate for the paged edition lives in
# test/distributed_workloads/. RUN_SERIAL because the ranks want the GPU.
# ---------------------------------------------------------------------------
if(TARGET clio_lammps_md_mpi_bench AND MPIEXEC_EXECUTABLE)
    add_test(NAME bench_lammps_md_mpi_2rank
        COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe
                ${MPIEXEC_NUMPROC_FLAG} 2
                # NOT $<TARGET_FILE:>: the MPI baselines are custom targets
                # (nvcc + MPI driven by hand), not add_executable, so the
                # generator expression errors out at configure time.
                ${CMAKE_BINARY_DIR}/bin/clio_lammps_md_mpi_bench
                --lattice 20 --steps 20
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(bench_lammps_md_mpi_2rank PROPERTIES
        TIMEOUT 900 LABELS "bench;gpu;mpi;distributed" RUN_SERIAL TRUE)
endif()

# ---------------------------------------------------------------------------
# The NVSHMEM baseline, RUN at 2 PEs.
#
# Built and never executed, like the MPI edition was. Verified to run and
# self-gate at 2 PEs on this host before being registered -- NVSHMEM needs a
# working bootstrap, so an entry that cannot launch would be worse than none.
#
# Invoked by OUTPUT PATH, not $<TARGET_FILE:>: like the MPI baselines these
# are add_custom_target (nvcc + NVSHMEM rdc), and the generator expression
# fails at CONFIGURE time on a non-executable target, breaking the build for
# everyone rather than just this test.
# ---------------------------------------------------------------------------
if(TARGET clio_lammps_md_nvshmem_bench AND MPIEXEC_EXECUTABLE)
    add_test(NAME bench_lammps_md_nvshmem_2pe
        COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe
                ${MPIEXEC_NUMPROC_FLAG} 2
                ${CMAKE_BINARY_DIR}/bin/clio_lammps_md_nvshmem_bench
                --lattice 20 --steps 20
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(bench_lammps_md_nvshmem_2pe PROPERTIES
        TIMEOUT 900 LABELS "bench;gpu;nvshmem;distributed" RUN_SERIAL TRUE)
endif()

# ---------------------------------------------------------------------------
# The NCCL baseline, RUN at 2 ranks. lammps_md is the only workload with one.
#
# NCCL requires ONE GPU PER RANK -- it cannot share a device the way the MPI
# and NVSHMEM editions can -- so on a single-GPU host the bench exits 77 and
# ctest reports SKIP rather than failure. Verified: on this host it skips; the
# gate is real wherever there are 2+ visible devices.
# ---------------------------------------------------------------------------
if(TARGET clio_lammps_md_nccl_bench AND MPIEXEC_EXECUTABLE)
    add_test(NAME bench_lammps_md_nccl_2rank
        COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe
                ${MPIEXEC_NUMPROC_FLAG} 2
                ${CMAKE_BINARY_DIR}/bin/clio_lammps_md_nccl_bench
                --lattice 20 --steps 20
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(bench_lammps_md_nccl_2rank PROPERTIES
        TIMEOUT 900 LABELS "bench;gpu;nccl;distributed" RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77)
endif()
