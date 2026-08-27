#------------------------------------------------------------------------------
# LBANN science kernel: MLP training with the weights paged
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
  # Compressed tiers need the compressor pool; without it these targets would
  # degrade their link names to bare -l flags and die at final link.
  if(TARGET clio_cte_compressor_runtime)
    add_cuda_executable(clio_lbann_paged_bench FALSE
        clio_lbann_paged_bench.cc
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
    set_target_properties(clio_lbann_paged_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_lbann_paged_bench RUNTIME DESTINATION bin)

    # BIT-EQUALITY against a dense reference: per-step losses and a digest of
    # the final weight bits, with in-place SGD on paged weights.
    add_test(NAME cte_gpu_vector_lbann
        COMMAND clio_lbann_paged_bench
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_lbann PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    add_test(NAME cte_gpu_vector_lbann_ooc
        COMMAND clio_lbann_paged_bench --cap 24
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_lbann_ooc PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
  endif()
endif()
