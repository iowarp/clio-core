# Adapter Unit Tests

# Add subdirectories for each adapter based on enabled options
# Note: All adapters require CLIO_CORE_ENABLE_ELF=ON except ADIOS2
if(CLIO_CORE_ENABLE_ELF AND CLIO_CTE_ENABLE_POSIX_ADAPTER)
    add_subdirectory(posix)
endif()

if(CLIO_CORE_ENABLE_ELF AND CLIO_CTE_ENABLE_STDIO_ADAPTER)
    add_subdirectory(stdio)
endif()

if(CLIO_CORE_ENABLE_ELF AND CLIO_CTE_ENABLE_MPIIO_ADAPTER AND CLIO_CORE_ENABLE_MPI)
    add_subdirectory(mpiio)
endif()

# HDF5 VFD smoke test (requires the VFD target, which requires HDF5 >= 1.14).
if(CLIO_CTE_ENABLE_VFD AND TARGET clio_cte_hdf5_vfd)
    add_subdirectory(vfd)
endif()

# ADIOS2 adapter tests (doesn't require ELF interception)
# Only add if ADIOS2 adapter is enabled
if(CLIO_CTE_ENABLE_ADIOS2_ADAPTER)
    add_subdirectory(adios2)
endif()

# FUSE adapter tests (doesn't require ELF or libfuse3 — tests internal components)
if(CLIO_CTE_ENABLE_FUSE_ADAPTER)
    add_subdirectory(libfuse)
endif()

# HDF5 VOL connector tests (doesn't require ELF or a running runtime).
# The TARGET guard skips them when the connector itself was skipped
# (HDF5 missing or < 1.14).
if(CLIO_CTE_ENABLE_HDF5_VOL AND TARGET iowarp_hdf5_vol)
    add_subdirectory(hdf5_vol)
endif()