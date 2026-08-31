#------------------------------------------------------------------------------
# Gray-Scott reaction-diffusion (sliding stencil window)
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
  # Compressed tiers need the compressor pool; without it these targets would
  # degrade their link names to bare -l flags and die at final link.
  if(TARGET clio_cte_compressor_runtime)
    add_cuda_executable(clio_grayscott_paged_bench FALSE
        clio_grayscott_paged_bench.cc
        cuda/grayscott_launch_cuda.cc
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
    set_target_properties(clio_grayscott_paged_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_grayscott_paged_bench RUNTIME DESTINATION bin)

    # The CUDA paged row had NO ctest entry while its SYCL twin did -- so a
    # break here would have been caught only on the backend that is harder to
    # build. Same flags as cte_gpu_vector_grayscott_sycl on purpose: the pair is
    # meant to be compared, not just to exist.
    add_test(NAME cte_gpu_vector_grayscott
        COMMAND clio_grayscott_paged_bench --blocks 8 --steps 2
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_grayscott PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
  endif()
endif()

#------------------------------------------------------------------------------
# grayscott, MPI edition: baseline, links nothing from clio.
#------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_CUDA)
    find_package(MPI QUIET COMPONENTS CXX)
    find_program(MD_MPI_NVCC nvcc
        HINTS ${CUDAToolkit_BIN_DIR} ENV CUDA_HOME
        PATH_SUFFIXES bin)
    if(MPI_CXX_FOUND AND MD_MPI_NVCC)
        set(_arch_flags "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
            list(APPEND _arch_flags "-gencode=arch=compute_${_arch},code=sm_${_arch}")
        endforeach()
        if(NOT _arch_flags)
            list(APPEND _arch_flags "-arch=sm_80")
        endif()
        set(_mpi_cflags "")
        foreach(_dir IN LISTS MPI_CXX_INCLUDE_DIRS)
            list(APPEND _mpi_cflags -I${_dir})
        endforeach()
        set(_mpi_libs "")
        foreach(_lib IN LISTS MPI_CXX_LIBRARIES)
            list(APPEND _mpi_libs -Xlinker ${_lib})
        endforeach()
        set(_out ${CMAKE_BINARY_DIR}/bin/clio_grayscott_mpi_bench)
        set(_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_grayscott_mpi_bench.cc)
        add_custom_command(
            OUTPUT ${_out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
            COMMAND ${MD_MPI_NVCC}
                -x cu -std=c++17 -O3 ${_arch_flags} ${_mpi_cflags}
                ${_src} -o ${_out} ${_mpi_libs}
            DEPENDS ${_src}
            COMMENT "Building clio_grayscott_mpi_bench (nvcc + MPI)"
            VERBATIM)
        add_custom_target(clio_grayscott_mpi_bench ALL DEPENDS ${_out})
        install(PROGRAMS ${_out} DESTINATION bin)
        message(STATUS "grayscott MPI baseline: ON")
    else()
        message(STATUS "MPI not found; skipping clio_grayscott_mpi_bench")
    endif()
endif()

#------------------------------------------------------------------------------
# grayscott, NVSHMEM edition.
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
        set(_arch_flags "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
            list(APPEND _arch_flags "-gencode=arch=compute_${_arch},code=sm_${_arch}")
        endforeach()
        if(NOT _arch_flags)
            list(APPEND _arch_flags "-arch=sm_80")
        endif()
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
        set(_out ${CMAKE_BINARY_DIR}/bin/clio_grayscott_nvshmem_bench)
        set(_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_grayscott_nvshmem_bench.cc)
        add_custom_command(
            OUTPUT ${_out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
            COMMAND ${NVSHMEM_NVCC}
                -x cu -std=c++17 -O3 -rdc=true ${_arch_flags}
                -I${NVSHMEM_INCLUDE_DIR} ${_nvs_mpi_cflags}
                ${_src} -o ${_out}
                -L${NVSHMEM_LIB_DIR} -lnvshmem_device -lnvshmem_host
                ${_nvs_mpi_libs}
                -Xlinker -rpath -Xlinker ${NVSHMEM_LIB_DIR}
            DEPENDS ${_src}
            COMMENT "Building clio_grayscott_nvshmem_bench (nvcc + NVSHMEM rdc)"
            VERBATIM)
        add_custom_target(clio_grayscott_nvshmem_bench ALL DEPENDS ${_out})
        install(PROGRAMS ${_out} DESTINATION bin)
        message(STATUS "grayscott NVSHMEM baseline: ON")
    else()
        message(STATUS "NVSHMEM not found; skipping clio_grayscott_nvshmem_bench")
    endif()
endif()

#------------------------------------------------------------------------------
# grayscott, BaM edition: compiles BaM's sources itself with nvcc (clang rejects
# --expt-relaxed-constexpr, see the lammps_md sibling).
#------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_CUDA AND EXISTS ${CMAKE_SOURCE_DIR}/external/bam/src/array.cu)
    find_program(MD_BAM_NVCC nvcc
        HINTS ${CUDAToolkit_BIN_DIR} ENV CUDA_HOME
        PATH_SUFFIXES bin)
    if(MD_BAM_NVCC)
        set(_arch_flags "")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
            list(APPEND _arch_flags "-gencode=arch=compute_${_arch},code=sm_${_arch}")
        endforeach()
        if(NOT _arch_flags)
            list(APPEND _arch_flags "-arch=sm_80")
        endif()
        set(_out ${CMAKE_BINARY_DIR}/bin/clio_grayscott_bam_bench)
        set(_src ${CMAKE_CURRENT_SOURCE_DIR}/clio_grayscott_bam_bench.cc)
        add_custom_command(
            OUTPUT ${_out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
            COMMAND ${MD_BAM_NVCC}
                -x cu -std=c++17 -O3 -rdc=true --expt-relaxed-constexpr
                ${_arch_flags}
                -I${CMAKE_SOURCE_DIR}/external/bam/include
                ${_src}
                ${CMAKE_SOURCE_DIR}/external/bam/src/array.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/controller.cc
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache_host.cc
                -o ${_out}
            DEPENDS ${_src}
                ${CMAKE_SOURCE_DIR}/external/bam/src/array.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache.cu
                ${CMAKE_SOURCE_DIR}/external/bam/src/controller.cc
                ${CMAKE_SOURCE_DIR}/external/bam/src/page_cache_host.cc
            COMMENT "Building clio_grayscott_bam_bench (nvcc + BaM page cache)"
            VERBATIM)
        add_custom_target(clio_grayscott_bam_bench ALL DEPENDS ${_out})
        install(PROGRAMS ${_out} DESTINATION bin)
        message(STATUS "grayscott BaM baseline: ON")
    endif()
endif()


#------------------------------------------------------------------------------
# Gray-Scott over a paged field -- SYCL.
#
# Same host driver source as the CUDA target, compiled once as ordinary C++.
# Only sycl/grayscott_launch_sycl.cc is built with -fsycl; the workload
# (grayscott_kernels.h over grayscott_math.h) is one copy shared by both.
# SHARED library for the launches -- device code in a static archive does not
# reach the executable's device image. SPIR-V is refused: DPC++ cannot compile
# a coroutine for it.
#------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_SYCL AND NOT SYCL_TARGET MATCHES "spir64")
    set(_gs_inc
        ${CMAKE_SOURCE_DIR}/context-runtime/include
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include)

    add_library(grayscott_sycl_launch SHARED sycl/grayscott_launch_sycl.cc)
    wrp_core_apply_sycl_flags(grayscott_sycl_launch)
    set_target_properties(grayscott_sycl_launch PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_include_directories(grayscott_sycl_launch PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} ${_gs_inc})
    target_link_libraries(grayscott_sycl_launch PUBLIC
        clio_run_cxx clio_run_cxx_gpu clio_cte_core_client)

    add_executable(clio_grayscott_paged_bench_sycl clio_grayscott_paged_bench.cc)
    target_link_options(clio_grayscott_paged_bench_sycl PRIVATE
        -fsycl -fsycl-targets=${SYCL_TARGET})
    target_include_directories(clio_grayscott_paged_bench_sycl PRIVATE
        ${CLIO_SYCL_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${_gs_inc})
    target_link_libraries(clio_grayscott_paged_bench_sycl PRIVATE
        grayscott_sycl_launch
        clio_run_cxx clio_run_cxx_gpu clio_admin_client clio_bdev_client
        clio_cte_core_client clio_cte_core_runtime clio::cte::gpu_vector
        ${CMAKE_THREAD_LIBS_INIT})
    set_target_properties(clio_grayscott_paged_bench_sycl PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    install(TARGETS clio_grayscott_paged_bench_sycl RUNTIME DESTINATION bin)

    # REGRESSION COVERAGE for the SYCL row. Without an entry here the row is
    # only ever as correct as the last time someone ran it by hand.
    # --blocks 8 is the SYCL backend's documented ceiling, not a taste.
    add_test(NAME cte_gpu_vector_grayscott_sycl
        COMMAND clio_grayscott_paged_bench_sycl --blocks 8 --steps 2
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_grayscott_sycl PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;sycl"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    message(STATUS "grayscott paged bench (SYCL): ON")
endif()



#------------------------------------------------------------------------------
# Gray-Scott, Kokkos edition: the performance-portability baseline.
#
# ONE SOURCE, WHICHEVER BACKEND KOKKOS WAS BUILT FOR -- point CMake at an
# install with -DKokkos_ROOT=<path>. Links nothing from clio, like every
# other baseline here.
#------------------------------------------------------------------------------
find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    find_package(MPI QUIET COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        add_executable(clio_grayscott_kokkos_bench
            kokkos/clio_grayscott_kokkos_bench.cc)
        target_link_libraries(clio_grayscott_kokkos_bench PRIVATE
            Kokkos::kokkos MPI::MPI_CXX)
        set_target_properties(clio_grayscott_kokkos_bench PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
        install(TARGETS clio_grayscott_kokkos_bench RUNTIME DESTINATION bin)

        # REGRESSION COVERAGE. These gates are self-checking (integer-exact
        # counts, or a host reference), so an entry here is worth having even
        # though the other baselines in this directory have none: without it
        # the row is only as correct as the last time someone ran it by hand.
        # No cleanup fixture -- a Kokkos baseline links nothing from clio and
        # so binds no runtime port. RUN_SERIAL only because they share a GPU.
        add_test(NAME bench_grayscott_kokkos
            COMMAND clio_grayscott_kokkos_bench --data-mb 512 --steps 4 --check-csum 381349.522809
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
        set_tests_properties(bench_grayscott_kokkos PROPERTIES
            TIMEOUT 900 LABELS "bench;gpu;kokkos" RUN_SERIAL TRUE)

        # ...and the same deck across 2 ranks. The gates that matter here are
        # rank-count INVARIANT by construction (integer reductions), so a
        # divergence between these two entries is a real defect rather than
        # float noise.
        if(MPIEXEC_EXECUTABLE)
            add_test(NAME bench_grayscott_kokkos_2rank
                COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe
                        ${MPIEXEC_NUMPROC_FLAG} 2
                        $<TARGET_FILE:clio_grayscott_kokkos_bench> --data-mb 512 --steps 4 --check-csum 381349.522809
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            )
            set_tests_properties(bench_grayscott_kokkos_2rank PROPERTIES
                TIMEOUT 900 LABELS "bench;gpu;kokkos;distributed" RUN_SERIAL TRUE)
        endif()
        message(STATUS "grayscott Kokkos baseline: ON (backends: ${Kokkos_DEVICES})")
    endif()
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
if(TARGET clio_grayscott_mpi_bench AND MPIEXEC_EXECUTABLE)
    add_test(NAME bench_grayscott_mpi_2rank
        COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe
                ${MPIEXEC_NUMPROC_FLAG} 2
                # NOT $<TARGET_FILE:>: the MPI baselines are custom targets
                # (nvcc + MPI driven by hand), not add_executable, so the
                # generator expression errors out at configure time.
                ${CMAKE_BINARY_DIR}/bin/clio_grayscott_mpi_bench
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(bench_grayscott_mpi_2rank PROPERTIES
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
if(TARGET clio_grayscott_nvshmem_bench AND MPIEXEC_EXECUTABLE)
    add_test(NAME bench_grayscott_nvshmem_2pe
        COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe
                ${MPIEXEC_NUMPROC_FLAG} 2
                ${CMAKE_BINARY_DIR}/bin/clio_grayscott_nvshmem_bench
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(bench_grayscott_nvshmem_2pe PROPERTIES
        TIMEOUT 900 LABELS "bench;gpu;nvshmem;distributed" RUN_SERIAL TRUE)
endif()
