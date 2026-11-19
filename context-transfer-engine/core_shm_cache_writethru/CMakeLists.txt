cmake_minimum_required(VERSION 3.10)
project(clio_cte_core_shm_cache_writethru)

#------------------------------------------------------------------------------
# Shared-memory write-through cache chimod: the top of the CTE chain. Clients
# write and read blobs in its segment directly; FlushLog aggregates the
# logged ranges into MultiPutBlob tasks down the chain. See DESIGN.md.
#------------------------------------------------------------------------------

add_clio_module_runtime(
  SOURCES
    src/core_shm_cache_writethru_runtime.cc
    src/core_shm_cache_writethru_data.cc
    src/core_shm_cache_writethru_flush.cc
    src/autogen/core_shm_cache_writethru_lib_exec.cc
)

add_clio_module_client(
  SOURCES
    src/core_shm_cache_writethru_client.cc
)

target_link_libraries(clio_cte_core_shm_cache_writethru_runtime PUBLIC
    clio_cte_core_client)
target_link_libraries(clio_cte_core_shm_cache_writethru_client PUBLIC
    clio_cte_core_client)
target_include_directories(clio_cte_core_shm_cache_writethru_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cte_core_shm_cache_writethru_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
