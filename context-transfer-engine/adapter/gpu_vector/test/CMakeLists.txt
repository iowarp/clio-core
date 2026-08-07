#------------------------------------------------------------------------------
# gpu_vector tests
#------------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
    add_cuda_executable(test_gpu_vector_smoke FALSE
        test_gpu_vector_smoke_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
        LINK_LIBS
            clio_run_cxx
            clio_run_cxx_gpu
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            clio::cte::gpu_vector
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(test_gpu_vector_smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_smoke COMMAND test_gpu_vector_smoke
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_smoke PROPERTIES
        TIMEOUT 300
        LABELS "cte;gpu;gpu_vector"
    )
endif()
