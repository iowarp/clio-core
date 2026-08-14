# ------------------------------------------------------------------------------
# Clio-NeuroPress large-chunk transfer audit (issue #693).
#
# Runs 2 x 256 MiB of device-generated data through Clio's real store path under
# NeuroPress inference-only selection with a ratio cost model, wrapping each
# phase in an NVTX range so Nsight Systems can attribute every cudaMemcpy to the
# phase that caused it.
#
# Unlike the sibling neuropress_data_path_trace this needs NO CUPTI: the
# profiler is nsys, which is external. It does need NVTX, which ships with the
# CUDA toolkit as a header-only interface in recent versions.
# ------------------------------------------------------------------------------

if(NOT CLIO_CORE_ENABLE_CUDA)
  message(STATUS "NeuroPress transfer audit: skipped (CUDA off)")
  return()
endif()

find_path(CLIO_NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
        /usr/local/cuda/include
        /usr/local/cuda/targets/x86_64-linux/include)

if(NOT CLIO_NVTX_INCLUDE_DIR)
  message(STATUS
    "NeuroPress transfer audit: skipped (nvtx3 headers not found; the NVTX "
    "ranges are what make the nsys profile attributable)")
  return()
endif()

message(STATUS "NeuroPress transfer audit: enabled (nvtx=${CLIO_NVTX_INCLUDE_DIR})")

add_cuda_library(npaudit_kernel SHARED FALSE audit_kernel.cu)

add_executable(neuropress_transfer_audit
  neuropress_transfer_audit.cc
)

target_include_directories(neuropress_transfer_audit PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CLIO_NVTX_INCLUDE_DIR}
)

target_compile_definitions(neuropress_transfer_audit PRIVATE
  CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

target_link_libraries(neuropress_transfer_audit
  npaudit_kernel
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

install(TARGETS neuropress_transfer_audit npaudit_kernel
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
)
