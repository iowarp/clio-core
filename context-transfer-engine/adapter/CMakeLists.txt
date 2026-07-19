#------------------------------------------------------------------------------
# Build adapters
#------------------------------------------------------------------------------

# CTE adapter singleton implementation - moved to filesystem directory
# to leverage existing working build configuration

# Adapters require ELF support (for real_api.h)
# Disable all adapters when ELF is disabled
if(CLIO_CORE_ENABLE_ELF)
    message(STATUS "CTE Adapters: ENABLED (ELF support available)")

    # Shared cfs-backed I/O core the interceptors delegate to (replaces the
    # old clio_cte_fs_base filesystem layer; backed by the context-filesystem
    # chimod via clio::cte::filesystem::Client).
    add_subdirectory(cfs)

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
    if (CLIO_CTE_ENABLE_VFD)
        add_subdirectory(vfd)
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

# GPU Vector adapter (CUDA/ROCm only; producer-only kernel API)
if (CLIO_CORE_ENABLE_CUDA OR CLIO_CORE_ENABLE_ROCM)
    add_subdirectory(gpu_vector)
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