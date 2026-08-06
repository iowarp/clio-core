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

    # On-device faulting under oversubscription, UNCOMPRESSED: cache == 1/2 the

    add_cuda_executable(test_gpu_vector_matrix FALSE
        test_gpu_vector_matrix_gpu.cc
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
    set_target_properties(test_gpu_vector_matrix PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_matrix COMMAND test_gpu_vector_matrix
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_matrix PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10517"
        TIMEOUT 300
        LABELS "gpu;cuda;cte;gpu_vector"
    )

    add_cuda_executable(test_gpu_vector_oversubscribe FALSE
        test_gpu_vector_oversubscribe_gpu.cc
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
    set_target_properties(test_gpu_vector_oversubscribe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_oversubscribe_cuda COMMAND test_gpu_vector_oversubscribe
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_oversubscribe_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10640"
        TIMEOUT 180
        LABELS "gpu;cuda;cte;gpu_vector;fault;oversubscribe"
    )

    # MULTI-BLOCK cold faulting against a CAE-sharded external tag — the
    # llama.cpp weight-streaming configuration in miniature (the coverage gap
    # behind the family-mismatch deadlock: every other multi-block test writes
    # through the vector, every external-writer test is nblocks=1).
    add_cuda_executable(test_gpu_vector_llama_geom FALSE
        test_gpu_vector_llama_geom_gpu.cc
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
    set_target_properties(test_gpu_vector_llama_geom PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_llama_geom_cuda COMMAND test_gpu_vector_llama_geom
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_llama_geom_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10660"
    )

    # Real .gguf model tensors (qwen2.5) mapped as gpu_vectors and demand
    # faulted. Opt-in via CLIO_TEST_GGUF; skips cleanly when unset so a
    # multi-GB model is never a CI dependency.
    add_cuda_executable(test_gpu_vector_gguf_map FALSE
        test_gpu_vector_gguf_map_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
            ${CMAKE_CURRENT_SOURCE_DIR}
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
    set_target_properties(test_gpu_vector_gguf_map PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_gguf_map_cuda COMMAND test_gpu_vector_gguf_map
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_gguf_map_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10648"
        TIMEOUT 900
        LABELS "gpu;cuda;cte;gpu_vector;tensor;gguf;fault"
    )

    # ------------------------------------------------------------------
    # Compressor-dependent tests. These link clio_cte_compressor_{client,
    # runtime} and set Context::dynamic_compress_ / compress_preset_, which
    # only exist under CTP_ENABLE_COMPRESS. Without this guard they are
    # configured even in a no-compressor build and fail at link/compile time.
    # Same idiom as the sibling benchmark/CMakeLists.txt.
    # ------------------------------------------------------------------
    if(CLIO_CTE_ENABLE_COMPRESS)
    endif()  # CLIO_CTE_ENABLE_COMPRESS
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
