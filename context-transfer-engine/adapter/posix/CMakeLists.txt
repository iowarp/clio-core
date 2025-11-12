include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/context-transfer-engine  # For adapter/ includes
        .)

# Include directories are handled by target dependencies

# Create the POSIX interceptor
add_library(wrp_cte_posix SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/posix_api.cc)

# Add bdev ChiMod include directory (unified build mode)
target_include_directories(wrp_cte_posix PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/modules/bdev/include
  ${CMAKE_SOURCE_DIR}/context-transfer-engine  # For adapter/adapter_types.h
)

add_dependencies(wrp_cte_posix wrp_cte_fs_base)
target_link_libraries(wrp_cte_posix
        MPI::MPI_CXX
        stdc++fs
        wrp_cte_fs_base
        wrp_cte_core_client
        wrp_cte_cae_config
        ${HSHM_TARGET})
set_target_properties(wrp_cte_posix PROPERTIES POSITION_INDEPENDENT_CODE ON)


# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
install(
        TARGETS
        wrp_cte_posix
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# -----------------------------------------------------------------------------
# Install headers
# -----------------------------------------------------------------------------
file(GLOB_RECURSE WRP_CTE_HEADERS "*.h")
install(
        FILES
        ${WRP_CTE_HEADERS}
        DESTINATION
        ${CMAKE_INSTALL_PREFIX}/include/adapter/posix
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
# Add Target(s) to Coverage
# -----------------------------------------------------------------------------
# Coverage flags removed for CTE build
