cmake_minimum_required(VERSION 3.10)
project(clio_cte_checkpoint)

#------------------------------------------------------------------------------
# Checkpoint chimod -- the fault handler behind vector.Copy's lazy copies:
# materialises a copy-tag's missing blobs from the source tag on first touch.
#------------------------------------------------------------------------------

add_clio_module_runtime(
  SOURCES
    src/checkpoint_runtime.cc
    src/autogen/checkpoint_lib_exec.cc
)

add_clio_module_client(
  SOURCES
    src/checkpoint_client.cc
)

target_link_libraries(clio_cte_checkpoint_runtime PUBLIC clio_cte_core_client)
target_link_libraries(clio_cte_checkpoint_client PUBLIC clio_cte_core_client)
target_include_directories(clio_cte_checkpoint_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cte_checkpoint_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
