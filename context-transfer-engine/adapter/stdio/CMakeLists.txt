include_directories(
        ${WRP_CTE_ROOT}
        ${HERMES_SRC_DIR}
        ${HERMES_ADAPTER_DIR}
        ${HERMES_IO_CLIENT_DIR}
        .)

# Create the STDIO interceptor
set(INTERCEPTOR_DEPS
        ${Hermes_CLIENT_DEPS} wrp_cte_fs_base)
add_library(wrp_cte_stdio SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/stdio_api.cc)
add_dependencies(wrp_cte_stdio ${INTERCEPTOR_DEPS})
target_link_libraries(wrp_cte_stdio MPI::MPI_CXX stdc++fs ${libelf_LIBRARIES} ${INTERCEPTOR_DEPS})

# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
install(
        TARGETS
        wrp_cte_stdio
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
        include/adapter/stdio
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
