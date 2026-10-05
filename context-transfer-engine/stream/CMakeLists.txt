cmake_minimum_required(VERSION 3.10)
project(clio_cte_stream)

#------------------------------------------------------------------------------
# Stream chimod: byte-stream layer over CTE tags with deferred append pipeline
#------------------------------------------------------------------------------

add_clio_module_runtime(
  SOURCES
    src/stream_runtime.cc
    src/stream_home.cc
    src/stream_log.cc
    src/autogen/stream_lib_exec.cc
)

add_clio_module_client(
  SOURCES
    src/stream_client.cc
)

target_link_libraries(clio_cte_stream_runtime PUBLIC clio_cte_core_client)
target_link_libraries(clio_cte_stream_client PUBLIC clio_cte_core_client)

target_include_directories(clio_cte_stream_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cte_stream_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
