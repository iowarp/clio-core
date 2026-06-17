# Shared, byte-oriented I/O core for the CTE interceptor adapters (POSIX,
# STDIO, MPI-IO). Replaces the old clio_cte_fs_base: every op delegates to the
# context-filesystem chimod via clio::cte::filesystem::Client.
add_library(clio_cte_cfs_adapter STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cfs_io.cc)

set_target_properties(clio_cte_cfs_adapter PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# PUBLIC so the POSIX/STDIO/MPI-IO interceptors that link this target inherit
# the include paths needed by cfs_io.h.
target_include_directories(clio_cte_cfs_adapter PUBLIC
    ${CLIO_CTE_ROOT}                                                   # adapter/cfs/cfs_io.h
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/filesystem/include     # filesystem client
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)          # cte core client

target_link_libraries(clio_cte_cfs_adapter PUBLIC
    clio_cte_filesystem_client   # the chimod client every op delegates to
    clio_cte_core_client         # core client + CTE_MANAGER init
    ctp::cxx)

install(
    TARGETS clio_cte_cfs_adapter
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
