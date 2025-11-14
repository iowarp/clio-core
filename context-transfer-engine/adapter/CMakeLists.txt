#------------------------------------------------------------------------------
# Build adapters
#------------------------------------------------------------------------------

# Create the CAE configuration library
add_library(wrp_cte_cae_config SHARED
    cae_config.cc
    cae_config.h
)

# Note: wrp_cte_core_client provides core/include via transitive includes
# LOCAL includes for this library's own headers and adapter/ path resolution
target_include_directories(wrp_cte_cae_config PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For cae_config.h in same directory
    ${WRP_CTE_ROOT}              # For adapter/cae_config.h and adapter/adapter_types.h include paths
)

target_link_libraries(wrp_cte_cae_config
    hshm::configure
    hshm::cxx
    wrp_cte_core_client  # Provides core headers transitively
)

# CTE adapter singleton implementation - moved to filesystem directory
# to leverage existing working build configuration

# Core adapters that are always built
add_subdirectory(filesystem)
add_subdirectory(posix)

# Optional adapters - only build if explicitly enabled
# These are currently disabled for CTE as they may need additional updates
option(WRP_CTE_ENABLE_STDIO_ADAPTER "Enable STDIO adapter" OFF)
option(WRP_CTE_ENABLE_MPIIO_ADAPTER "Enable MPI-IO adapter" OFF)
option(WRP_CTE_ENABLE_VFD "Enable HDF5 VFD adapter" OFF)
option(WRP_CTE_ENABLE_NVIDIA_GDS_ADAPTER "Enable NVIDIA GDS adapter" OFF)

if (WRP_CTE_ENABLE_STDIO_ADAPTER)
    add_subdirectory(stdio)
endif()
if (WRP_CTE_ENABLE_MPIIO_ADAPTER)
    add_subdirectory(mpiio)
endif()
if (WRP_CTE_ENABLE_VFD)
    add_subdirectory(vfd)
endif()
if (WRP_CTE_ENABLE_NVIDIA_GDS_ADAPTER)
    add_subdirectory(nvidia_gds)
endif()

#-----------------------------------------------------------------------------
# Install adapter headers
#-----------------------------------------------------------------------------
file(GLOB_RECURSE WRP_CTE_HEADERS "*.h")
install(
        FILES
        ${WRP_CTE_HEADERS}
        DESTINATION
        include/adapter
        COMPONENT
        headers
)
install(DIRECTORY mapper DESTINATION include/adapter)

#-----------------------------------------------------------------------------
# Install CAE config library
#-----------------------------------------------------------------------------
install(TARGETS wrp_cte_cae_config
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)