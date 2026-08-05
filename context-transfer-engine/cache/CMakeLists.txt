cmake_minimum_required(VERSION 3.10)
project(clio_cte_cache)

#------------------------------------------------------------------------------
# Cache chimod (issue #886) — node-local uncompressed copies over the CTE
# core's REPLICA_CACHE slot, write-back to the authoritative chain.
#------------------------------------------------------------------------------

add_clio_module_runtime(
  SOURCES
    src/cache_runtime.cc
    src/autogen/cache_lib_exec.cc
)

add_clio_module_client(
  SOURCES
    src/cache_client.cc
)

target_link_libraries(clio_cte_cache_runtime PUBLIC clio_cte_core_client)
target_link_libraries(clio_cte_cache_client PUBLIC clio_cte_core_client)
target_include_directories(clio_cte_cache_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cte_cache_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
