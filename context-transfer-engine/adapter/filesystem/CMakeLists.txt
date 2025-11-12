include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/context-transfer-engine  # For adapter/adapter_types.h
        .)

# Include directories are handled by target dependencies

# Create the metadata manager singleton + FS base class
add_library(wrp_cte_fs_base SHARED
        filesystem.cc
        filesystem.h
        filesystem_io_client.h
        filesystem_mdm.h)

# Add bdev ChiMod include directory and adapter parent directory (unified build mode)
target_include_directories(wrp_cte_fs_base PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/modules/bdev/include
  ${CMAKE_SOURCE_DIR}/context-transfer-engine  # For adapter/adapter_types.h
)

target_link_libraries(wrp_cte_fs_base
        MPI::MPI_CXX
        wrp_cte_core_client
        wrp_cte_cae_config
        ${HSHM_TARGET})

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
        ${CMAKE_INSTALL_PREFIX}/include/adapter/filesystem
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
# Add Target(s) to Coverage
# -----------------------------------------------------------------------------
# Coverage flags removed for CTE build
