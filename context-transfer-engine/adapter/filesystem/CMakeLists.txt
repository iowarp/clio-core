# Create the metadata manager singleton + FS base class
add_library(wrp_cte_fs_base SHARED
        filesystem.cc
        filesystem.h
        filesystem_io_client.h
        filesystem_mdm.h)

# Local includes for adapter headers
# Note: bdev headers come transitively via chimaera_bdev_client
# Note: core headers come transitively via wrp_cte_core_client
target_include_directories(wrp_cte_fs_base PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}       # For local headers
  ${WRP_CTE_ROOT}                   # For adapter/adapter_types.h include path
)

target_link_libraries(wrp_cte_fs_base
        MPI::MPI_CXX
        wrp_cte_core_client      # Provides core and bdev headers transitively
        wrp_cte_cae_config
        hshm::cxx
        hshm::interceptor)       # Provides ELF support for posix_api.h

# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
install(
        TARGETS
        wrp_cte_fs_base
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)
install(
        FILES
        filesystem_io_client.h
        DESTINATION
        include/adapter/filesystem
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
# Install headers
# -----------------------------------------------------------------------------
file(GLOB_RECURSE WRP_CTE_HEADERS "*.h")
install(
        FILES
        ${WRP_CTE_HEADERS}
        DESTINATION
        include/adapter/filesystem
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
# Add Target(s) to Coverage
# -----------------------------------------------------------------------------
# Coverage flags removed for CTE build
