#-----------------------------------------------------------------------------
# GPU Vector tests — CUDA / ROCm only.
#-----------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
    add_cuda_executable(test_gpu_vector FALSE
        test_gpu_vector_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
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
    set_target_properties(test_gpu_vector PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_cuda COMMAND test_gpu_vector
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
        TIMEOUT 120
        LABELS "gpu;cuda;cte;gpu_vector"
    )
endif()

if(CLIO_CORE_ENABLE_ROCM)
    add_rocm_gpu_executable(test_gpu_vector FALSE
        test_gpu_vector_gpu.cc
    )
    # CUDA_ARCHITECTURES via set_target_properties wasn't reaching the
    # generated flags.make for this target (probably because the test
    # add_subdirectory chain is detached from the wrp_core_enable_rocm
    # scope). Pass --generate-code explicitly so atomicAdd_system / system-
    # scope intrinsics resolve. sm_89 covers the local NVIDIA target.
    target_compile_options(test_gpu_vector PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code=arch=compute_89,code=sm_89>"
    )
    set_target_properties(test_gpu_vector PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(test_gpu_vector PUBLIC
        ${CMAKE_SOURCE_DIR}/context-runtime/include
        ${CMAKE_SOURCE_DIR}/context-runtime/test
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
    )
    target_link_libraries(test_gpu_vector PUBLIC
        clio_run_cxx
        clio_run_cxx_gpu
        clio_admin_client
        clio_bdev_client
        clio_cte_core_client
        clio_cte_core_runtime
        clio::cte::gpu_vector
        ctp::rocm_cxx
        ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(test_gpu_vector PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_rocm COMMAND test_gpu_vector
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_rocm PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
        TIMEOUT 120
        LABELS "gpu;rocm;cte;gpu_vector"
    )
endif()
