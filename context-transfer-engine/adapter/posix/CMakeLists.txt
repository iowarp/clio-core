# Create the POSIX interceptor
add_library(wrp_cte_posix SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/posix_api.cc)

# Local includes for adapter headers
# Note: bdev headers come transitively via wrp_cte_core_client
# Note: core headers come transitively via wrp_cte_core_client
target_include_directories(wrp_cte_posix PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}       # For local headers
  ${WRP_CTE_ROOT}                   # For adapter/adapter_types.h include path
)

add_dependencies(wrp_cte_posix wrp_cte_fs_base)
target_link_libraries(wrp_cte_posix
        MPI::MPI_CXX
        stdc++fs
        wrp_cte_fs_base          # Provides filesystem headers
        wrp_cte_core_client      # Provides core and bdev headers transitively
        wrp_cte_cae_config
        hshm::cxx
        hshm::interceptor)       # Provides ELF support and HSHM_ENABLE_ELF definition
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
        include/adapter/posix
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
# Add Target(s) to Coverage
# -----------------------------------------------------------------------------
# Coverage flags removed for CTE build
