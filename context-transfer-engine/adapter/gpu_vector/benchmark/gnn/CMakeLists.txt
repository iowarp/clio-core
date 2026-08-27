#------------------------------------------------------------------------------
# GNN workload tooling.
#
# gnn_ingest streams a raw array from stdin into a vector's CTE backing store,
# so a matrix far larger than memory can be loaded without ever materialising a
# flat copy on disk. Host-only (no CUDA): it just writes blobs.
#------------------------------------------------------------------------------

add_executable(gnn_ingest gnn_ingest.cc)

target_include_directories(gnn_ingest PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
)

target_link_libraries(gnn_ingest PRIVATE
    clio_run_cxx
    clio_admin_client
    clio_bdev_client
    clio_cte_core_client
    ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(gnn_ingest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# gnn_aggregate: 1-hop mean aggregation over a CTE-resident matrix, without a
# flat copy at either end. Multi-pass scatter so every feature read stays
# sequential (see the file header for why gather is the wrong shape here).
add_executable(gnn_aggregate gnn_aggregate.cc)

target_include_directories(gnn_aggregate PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
)

target_link_libraries(gnn_aggregate PRIVATE
    clio_run_cxx
    clio_admin_client
    clio_bdev_client
    clio_cte_core_client
    ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(gnn_aggregate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS gnn_ingest gnn_aggregate RUNTIME DESTINATION bin)
