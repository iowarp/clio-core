cmake_minimum_required(VERSION 3.10)
project(clio_cte_lz4)

#------------------------------------------------------------------------------
# LZ4 chimod -- pages encoded on the host (ctp gpu_lz4), decoded on the GPU by
# the runtime GPU library's ctp_gpu_lz4_decode_launch (CUDA/ROCm/SYCL twins in
# context-runtime/src/gpu/gpu_lz4_*.cc). Builds and runs CPU-only too: device
# reads then decode on the host.
#------------------------------------------------------------------------------

add_clio_module_runtime(
  SOURCES
    src/lz4_runtime.cc
    src/autogen/lz4_lib_exec.cc
)

add_clio_module_client(
  SOURCES
    src/lz4_client.cc
)

target_link_libraries(clio_cte_lz4_runtime PUBLIC clio_cte_core_client)
target_link_libraries(clio_cte_lz4_client PUBLIC clio_cte_core_client)
target_include_directories(clio_cte_lz4_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_SOURCE_DIR}/context-transport-primitives/compress/include)
target_include_directories(clio_cte_lz4_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
# The decode launch lives in the runtime GPU library.
if(TARGET clio_run_cxx_gpu)
  target_link_libraries(clio_cte_lz4_runtime PUBLIC clio_run_cxx_gpu)
endif()
