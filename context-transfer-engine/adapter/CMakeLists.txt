#------------------------------------------------------------------------------
# Build adapters
#------------------------------------------------------------------------------

# CTE adapter singleton implementation - moved to filesystem directory
# to leverage existing working build configuration

# The HDF5 VFD is a plugin the library dlopen's, not an interceptor: it
# never touches real_api.h, so it does not need ELF support. Gate it on
# UNIX -- H5FDclio.cc uses <sys/file.h>, <unistd.h> and flock, so Windows
# needs porting work rather than a gate change.
if(UNIX AND CLIO_CTE_ENABLE_VFD)
    add_subdirectory(vfd)
endif()

# Adapters require ELF support (for real_api.h)
# Disable all adapters when ELF is disabled
if(CLIO_CORE_ENABLE_ELF)
    message(STATUS "CTE Adapters: ENABLED (ELF support available)")

    # Optional adapters - controlled by root CMakeLists.txt options
    if (CLIO_CTE_ENABLE_POSIX_ADAPTER)
        add_subdirectory(posix)
    endif()
    if (CLIO_CTE_ENABLE_STDIO_ADAPTER)
        add_subdirectory(stdio)
    endif()
    if (CLIO_CTE_ENABLE_MPIIO_ADAPTER)
        add_subdirectory(mpiio)
    endif()
    if (CLIO_CTE_ENABLE_NVIDIA_GDS_ADAPTER)
        add_subdirectory(nvidia_gds)
    endif()
else()
    message(STATUS "CTE Adapters: DISABLED (ELF support not enabled)")
    message(STATUS "  To enable adapters, configure with -DWRP_CORE_ENABLE_ELF=ON")
endif()

# ADIOS2 adapter does not require ELF support (it's a plugin engine)
if (CLIO_CTE_ENABLE_ADIOS2_ADAPTER)
    add_subdirectory(adios2)
endif()

# FUSE adapter does not require ELF support (standalone daemon)
if (CLIO_CTE_ENABLE_FUSE_ADAPTER)
    add_subdirectory(libfuse)
endif()

# HDF5 VOL connector (requires HDF5 >= 2.0, does not require ELF)
if (CLIO_CTE_ENABLE_HDF5_VOL)
    add_subdirectory(hdf5_vol)
endif()

# GPU Vector adapter (device-paged vector over the CTE).
#
# CLANG ONLY, AND SILENT ABOUT IT RATHER THAN BROKEN. Its device API is C++20
# coroutines end to end, which nvcc cannot compile -- so under nvcc every one
# of its targets fails with "namespace clio::run::gpu has no member
# YCoroTask", halfway through a build that had no way to know better. That is
# exactly what a CI job naming test_gpu_vector_* in its --target list got.
#
# Not built at all unless the GPU toolchain is clang. A build that cannot
# compile the vector should not offer its targets.
# SYCL qualifies for the same reason clang-CUDA does, and needs no flag to
# say so: a SYCL device compiler is always clang (DPC++) or AdaptiveCpp, so
# the toolchain question CLIO_GPU_CLANG answers for CUDA has only one
# possible answer here.
if (((CLIO_CORE_ENABLE_CUDA OR CLIO_CORE_ENABLE_ROCM) AND CLIO_GPU_CLANG)
    OR CLIO_CORE_ENABLE_SYCL)
    add_subdirectory(gpu_vector)
elseif (CLIO_CORE_ENABLE_CUDA OR CLIO_CORE_ENABLE_ROCM)
    message(STATUS
            "gpu_vector: DISABLED -- needs CLIO_GPU_CLANG=ON (its device API "
            "is C++20 coroutines, which only clang compiles for the device)")
endif()

# kvhdf5 adapter (CUDA only; HDF5-like producer surface over the CTE KV store)
if (CLIO_CORE_ENABLE_CUDA)
    add_subdirectory(kvhdf5)
endif()

#-----------------------------------------------------------------------------
# Install adapter headers
#-----------------------------------------------------------------------------
file(GLOB_RECURSE CLIO_CTE_HEADERS "*.h")
install(
        FILES
        ${CLIO_CTE_HEADERS}
        DESTINATION
        include/adapter
        COMPONENT
        headers
)
install(DIRECTORY mapper DESTINATION include/adapter)