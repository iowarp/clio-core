#------------------------------------------------------------------------------
# GROMACS science kernel: PME charge spread + gather over a paged mesh
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
  # Compressed tiers need the compressor pool; without it these targets would
  # degrade their link names to bare -l flags and die at final link.
  if(TARGET clio_cte_compressor_runtime)
    add_cuda_executable(clio_gmx_paged_bench FALSE
        clio_gmx_paged_bench.cc
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
    set_target_properties(clio_gmx_paged_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_gmx_paged_bench RUNTIME DESTINATION bin)

    # BIT-EQUALITY against a dense in-VRAM reference: integer-exact charge
    # conservation and mesh checksums for PME spread/gather. The ooc entry
    # runs the same physics through a cache far smaller than the mesh.
    add_test(NAME cte_gpu_vector_gmx
        COMMAND clio_gmx_paged_bench --page-kb 128 --blocks 8 --atoms 200000
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_gmx PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    add_test(NAME cte_gpu_vector_gmx_ooc
        COMMAND clio_gmx_paged_bench --page-kb 128 --blocks 8 --atoms 200000
                --cap 40
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_gmx_ooc PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
  endif()
endif()
