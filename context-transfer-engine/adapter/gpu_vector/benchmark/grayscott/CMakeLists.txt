#------------------------------------------------------------------------------
# Gray-Scott reaction-diffusion (sliding stencil window)
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
  # Compressed tiers need the compressor pool; without it these targets would
  # degrade their link names to bare -l flags and die at final link.
  if(TARGET clio_cte_compressor_runtime)
    add_cuda_executable(clio_grayscott_paged_bench FALSE
        clio_grayscott_paged_bench.cc
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
        if(MPI_CXX_FOUND)
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
