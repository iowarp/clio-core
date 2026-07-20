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
    # logical extent, reads served purely by the cold-miss fault path.
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

    # Model tensors mapped as gpu_vectors, across every ggml tensor format
    # (F32/F16/BF16 + Q4_0/Q8_0/Q4_K/Q5_K/Q6_K block-quantized). Uncompressed;
    # cache is far smaller than each tensor so every page demand-faults.
    add_cuda_executable(test_gpu_vector_tensor_map FALSE
        test_gpu_vector_tensor_map_gpu.cc
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
    set_target_properties(test_gpu_vector_tensor_map PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_tensor_map_cuda COMMAND test_gpu_vector_tensor_map
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_tensor_map_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10642"
        TIMEOUT 600
        LABELS "gpu;cuda;cte;gpu_vector;tensor;fault"
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
    # Transparent LOSSLESS-compressed page faulting: sequential read of an
    # extent many times larger than the cache, every page decompressed on the
    # fault path. Pins zstd (lossless) rather than cuszp (lossy) because the
    # target workload is model weights, which must round-trip bit-exactly.
    add_cuda_executable(test_gpu_vector_compress_fault FALSE
        test_gpu_vector_compress_fault_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
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
    set_target_properties(test_gpu_vector_compress_fault PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_compress_fault_cuda COMMAND test_gpu_vector_compress_fault
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_compress_fault_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10644;CLIO_CTE_COMPRESS_LIB=zstd"
        TIMEOUT 600
        LABELS "gpu;cuda;cte;gpu_vector;compress;fault;lossless"
    )

    # LARGER-THAN-VRAM compressed read via one long-running kernel. Writes and
    # reads tens of GiB through a file tier: minutes, not seconds. Labelled
    # "huge" so ordinary runs can exclude it (-LE huge).
    add_cuda_executable(test_gpu_vector_huge_compress FALSE
        test_gpu_vector_huge_compress_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
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
    set_target_properties(test_gpu_vector_huge_compress PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_huge_compress_cuda COMMAND test_gpu_vector_huge_compress
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_huge_compress_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10646;CLIO_CTE_COMPRESS_LIB=zstd"
        TIMEOUT 7200
        LABELS "gpu;cuda;cte;gpu_vector;compress;fault;lossless;huge"
    )

    # Compressed gpu_vector: page traffic routed through the compressor chimod.
    add_cuda_executable(test_gpu_vector_compress FALSE
        test_gpu_vector_compress_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
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
    set_target_properties(test_gpu_vector_compress PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_compress_cuda COMMAND test_gpu_vector_compress
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_compress_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10520;CLIO_CTE_COMPRESS_LIB=cuszp"
        TIMEOUT 180
        LABELS "gpu;cuda;cte;gpu_vector;compress"
    )

    # Capacity stress: dataset larger than the HBM budget stored, compressed,
    # entirely in GPU HBM (kHbm bdev).
    add_cuda_executable(test_gpu_vector_capacity FALSE
        test_gpu_vector_capacity_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
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
    set_target_properties(test_gpu_vector_capacity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_capacity_cuda COMMAND test_gpu_vector_capacity
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_capacity_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10560;CLIO_CTE_COMPRESS_LIB=cuszp"
        TIMEOUT 300
        LABELS "gpu;cuda;cte;gpu_vector;compress;capacity"
    )

    # SequentialTransaction windowed prefetch: sweep a dataset larger than the
    # HBM cache, host-orchestrated (the #700 direction; no on-device fault).
    add_cuda_executable(test_gpu_vector_transaction FALSE
        test_gpu_vector_transaction_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
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
    set_target_properties(test_gpu_vector_transaction PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_transaction_cuda COMMAND test_gpu_vector_transaction
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_transaction_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10600;CLIO_CTE_COMPRESS_LIB=cuszp"
        TIMEOUT 240
        LABELS "gpu;cuda;cte;gpu_vector;compress;transaction"
    )

    # Distributed compressed GPU vector: compress in HBM on one GPU node, store
    # compressed pages across a multi-node cte_core, read back + decompress.
    # Runs as a client under test/integration/distributed_slurm/run_slurm_gpu.
    add_cuda_executable(test_gpu_vector_distributed FALSE
        test_gpu_vector_distributed_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
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
    set_target_properties(test_gpu_vector_distributed PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    # No ctest entry: it is a multi-node client, driven by the Slurm harness.

    # IO-bound reader benchmark: prefetch vs no-prefetch on an auto-tiered
    # (fast RAM + slow Lustre file) store.
    add_cuda_executable(test_gpu_vector_prefetch FALSE
        test_gpu_vector_prefetch_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
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
    set_target_properties(test_gpu_vector_prefetch PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_prefetch_cuda COMMAND test_gpu_vector_prefetch
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_prefetch_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10620;CLIO_CTE_COMPRESS_LIB=cuszp"
        TIMEOUT 300
        LABELS "gpu;cuda;cte;gpu_vector;compress;prefetch"
    )
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
