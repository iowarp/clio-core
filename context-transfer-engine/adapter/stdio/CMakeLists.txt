include_directories(
        ${CLIO_CTE_ROOT}
        ${HERMES_SRC_DIR}
        ${HERMES_ADAPTER_DIR}
        ${HERMES_IO_CLIENT_DIR}
        .)

# Create the STDIO interceptor
add_library(clio_cte_stdio SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/stdio_api.cc)
target_link_libraries(clio_cte_stdio
        clio_cte_cfs_adapter      # cfs-backed I/O core (context-filesystem chimod)
        clio_cte_core_client
        ctp::cxx
        ctp::interceptor)
set_target_properties(clio_cte_stdio PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
install(
        TARGETS
        clio_cte_stdio
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
        include/adapter/stdio
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
