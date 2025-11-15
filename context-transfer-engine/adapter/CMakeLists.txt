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

# Adapters require ELF support (for real_api.h)
# Disable all adapters when ELF is disabled
if(WRP_CORE_ENABLE_ELF)
    message(STATUS "CTE Adapters: ENABLED (ELF support available)")

    # Core adapters that are built when ELF is enabled
    add_subdirectory(filesystem)

    # Optional adapters - controlled by root CMakeLists.txt options
    if (WRP_CTE_ENABLE_POSIX_ADAPTER)
        add_subdirectory(posix)
    endif()
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
else()
    message(STATUS "CTE Adapters: DISABLED (ELF support not enabled)")
    message(STATUS "  To enable adapters, configure with -DWRP_CORE_ENABLE_ELF=ON")
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