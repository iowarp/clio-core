# Create the POSIX interceptor
add_library(clio_cte_posix SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/posix_api.cc)

# Local includes for adapter headers
# Note: bdev headers come transitively via clio_cte_core_client
# Note: core headers come transitively via clio_cte_core_client
target_include_directories(clio_cte_posix PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}       # For local headers
  ${CLIO_CTE_ROOT}                   # For adapter/adapter_types.h include path
)

target_link_libraries(clio_cte_posix
        clio_cte_cfs_adapter      # cfs-backed I/O core (context-filesystem chimod)
        clio_cte_core_client      # Provides core and bdev headers transitively
        ctp::cxx
        ctp::interceptor)       # Provides ELF support and CTP_ENABLE_ELF definition
set_target_properties(clio_cte_posix PROPERTIES POSITION_INDEPENDENT_CODE ON)


# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
install(
        TARGETS
        clio_cte_posix
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# -----------------------------------------------------------------------------
# Install headers
# -----------------------------------------------------------------------------
file(GLOB_RECURSE CLIO_CTE_HEADERS "*.h")
install(
        FILES
        ${CLIO_CTE_HEADERS}
        DESTINATION
        include/adapter/posix
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
# Add Target(s) to Coverage
# -----------------------------------------------------------------------------
# Coverage flags removed for CTE build
