cmake_minimum_required(VERSION 3.10)
project(clio_cte_indexer)

#------------------------------------------------------------------------------
# Indexer chimod (issue #905) — owns the SemanticSearch index as an
# interposer above the CTE core. Built via the standard ChiMod build
# functions.
#------------------------------------------------------------------------------

add_clio_module_runtime(
  SOURCES
    src/indexer_runtime.cc
    src/autogen/indexer_lib_exec.cc
)

add_clio_module_client(
  SOURCES
    src/indexer_client.cc
)

# The indexer chimod drives the CTE core through its public client.
target_link_libraries(clio_cte_indexer_runtime PUBLIC clio_cte_core_client)
target_link_libraries(clio_cte_indexer_client PUBLIC clio_cte_core_client)
target_include_directories(clio_cte_indexer_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cte_indexer_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
