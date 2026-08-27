#------------------------------------------------------------------------------
# weight streaming (full re-read every pass)
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
  # Compressed tiers need the compressor pool; without it these targets would
  # degrade their link names to bare -l flags and die at final link.
  if(TARGET clio_cte_compressor_runtime)
    add_cuda_executable(clio_weights_paged_bench FALSE
        clio_weights_paged_bench.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
        LINK_LIBS
            clio_run_cxx
            clio_run_cxx_gpu
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            clio_cte_compressor_client
            clio_cte_compressor_runtime
            clio::cte::gpu_vector
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(clio_weights_paged_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_weights_paged_bench RUNTIME DESTINATION bin)
  endif()
endif()
