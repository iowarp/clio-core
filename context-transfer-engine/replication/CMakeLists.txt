cmake_minimum_required(VERSION 3.10)
project(clio_cte_replication)

#------------------------------------------------------------------------------
# Replication chimod (issue #886) — policy layer over the CTE core's blob
# replicas. Built via the standard ChiMod build functions.
#------------------------------------------------------------------------------

add_clio_module_runtime(
  SOURCES
    src/replication_runtime.cc
    src/autogen/replication_lib_exec.cc
)

add_clio_module_client(
  SOURCES
    src/replication_client.cc
)

# The replication chimod drives the CTE core through its public client.
target_link_libraries(clio_cte_replication_runtime PUBLIC clio_cte_core_client)
target_link_libraries(clio_cte_replication_client PUBLIC clio_cte_core_client)
target_include_directories(clio_cte_replication_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cte_replication_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
