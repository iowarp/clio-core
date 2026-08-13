# ------------------------------------------------------------------------------
# Clio-NeuroPress data path trace (issue #693).
#
# Follows a GPU-generated chunk through Clio's real store path and records every
# byte that moves, in order, with the thread that moved it -- answering whether
# the payload stays device-resident until it has been compressed.
#
# Needs CUPTI (the transfer record IS the output, so building without it would
# produce a report that looks complete and is empty). Unlike the sibling
# equivalence harness this does NOT need native NeuroPress: it traces Clio only.
# ------------------------------------------------------------------------------

if(NOT CLIO_CORE_ENABLE_CUDA)
  message(STATUS "NeuroPress data path trace: skipped (CUDA off)")
  return()
endif()

find_path(CLIO_CUPTI_INCLUDE_DIR cupti.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
        /usr/local/cuda/targets/x86_64-linux/include
        /usr/local/cuda/extras/CUPTI/include)
find_library(CLIO_CUPTI_LIB
  NAMES cupti
  HINTS /usr/local/cuda/targets/x86_64-linux/lib
        /usr/local/cuda/extras/CUPTI/lib64
        /usr/local/cuda/lib64)

if(NOT CLIO_CUPTI_INCLUDE_DIR OR NOT CLIO_CUPTI_LIB)
  message(STATUS
    "NeuroPress data path trace: skipped (CUPTI not found; the transfer "
    "record is this tool's entire output)")
  return()
endif()

message(STATUS "NeuroPress data path trace: enabled (cupti=${CLIO_CUPTI_LIB})")

add_cuda_library(npeq_data_path_kernel SHARED FALSE data_path_kernel.cu)

add_executable(neuropress_data_path_trace
  neuropress_data_path_trace.cc
  path_tracer.cc
)

target_include_directories(neuropress_data_path_trace PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CLIO_CUPTI_INCLUDE_DIR}
)

target_compile_definitions(neuropress_data_path_trace PRIVATE
  CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

target_link_libraries(neuropress_data_path_trace
  npeq_data_path_kernel
  ${CLIO_CUPTI_LIB}
  clio_cte_compressor_runtime
  clio_cte_compressor_client
  clio_cte_core_runtime
  clio_cte_core_client
  clio::run::admin_runtime
  clio::run::admin_client
  clio::run::bdev_runtime
  clio::run::bdev_client
  ctp::cxx
  ctp::cuda_cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS neuropress_data_path_trace npeq_data_path_kernel
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
)
