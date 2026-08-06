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

    add_cuda_executable(test_gpu_vector_spill FALSE
        test_gpu_vector_spill_gpu.cc
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
    set_target_properties(test_gpu_vector_spill PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_spill_cuda COMMAND test_gpu_vector_spill
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_spill_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10580;CLIO_CTE_COMPRESS_LIB=cuszp"
        TIMEOUT 300
        LABELS "gpu;cuda;cte;gpu_vector;compress;spill"
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

    add_cuda_executable(test_gpu_vector_kmeans FALSE
        test_gpu_vector_kmeans_gpu.cc
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
    set_target_properties(test_gpu_vector_kmeans PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_kmeans_cuda COMMAND test_gpu_vector_kmeans
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_kmeans_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10590;CLIO_CTE_COMPRESS_LIB=cuszp"
        TIMEOUT 400
        LABELS "gpu;cuda;cte;gpu_vector;compress;kmeans"
    )

    # Real-data counterpart: k-means over a compressed SIFT1M dataset, checked
    # against an uncompressed in-memory baseline (no synthetic ground truth).
    add_cuda_executable(test_gpu_vector_kmeans_real FALSE
        test_gpu_vector_kmeans_real_gpu.cc
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
    set_target_properties(test_gpu_vector_kmeans_real PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_kmeans_real_cuda COMMAND test_gpu_vector_kmeans_real
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_kmeans_real_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10592;CLIO_CTE_COMPRESS_LIB=cuszp"
        TIMEOUT 600
        LABELS "gpu;cuda;cte;gpu_vector;compress;kmeans;real"
    )

    # GNN (GraphSAGE-mean) forward over a LOSSLESS zstd-compressed ogbn-arxiv
    # feature matrix, host-staged through the compressor, checked bit-for-bit
    # against an uncompressed baseline.
    add_cuda_executable(test_gpu_vector_gnn FALSE
        test_gpu_vector_gnn_gpu.cc
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
    set_target_properties(test_gpu_vector_gnn PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_gnn_cuda COMMAND test_gpu_vector_gnn
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_gnn_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10594;CLIO_CTE_COMPRESS_LIB=zstd"
        TIMEOUT 600
        LABELS "gpu;cuda;cte;gpu_vector;compress;gnn;zstd"
    )

    # Option B: the same GNN forward, but features sourced through the real
    # gv::Vector<float> + SequentialTransaction streaming path with lossless zstd
    # (exercises the compressor's CPU-codec D2H/H2D staging). Bit-exact vs baseline.
    add_cuda_executable(test_gpu_vector_gnn_stream FALSE
        test_gpu_vector_gnn_stream_gpu.cc
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
    set_target_properties(test_gpu_vector_gnn_stream PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_gnn_stream_cuda COMMAND test_gpu_vector_gnn_stream
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_gnn_stream_cuda PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10596;CLIO_CTE_COMPRESS_LIB=zstd"
        TIMEOUT 600
        LABELS "gpu;cuda;cte;gpu_vector;compress;gnn;zstd;stream"
    )

    # Capacity comparison: traditional in-core GPU k-means (whole dataset resident,
    # OOMs past HBM) vs Eternia (compressed/tiered, streams a fixed window). Bench,
    # not registered as a ctest (needs the SIFT dataset + a size env); build only.
    add_cuda_executable(test_gpu_vector_kmeans_capacity FALSE
        test_gpu_vector_kmeans_capacity_gpu.cc
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
    set_target_properties(test_gpu_vector_kmeans_capacity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # GNN capacity comparison: GraphSAGE aggregation+readout over a feature matrix
    # larger than HBM -- traditional in-core (OOMs) vs Eternia zstd-compressed +
    # tiered streaming (bounded window). Build-only (needs data + size env).
    add_cuda_executable(test_gpu_vector_gnn_capacity FALSE
        test_gpu_vector_gnn_capacity_gpu.cc
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
    set_target_properties(test_gpu_vector_gnn_capacity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # GNN TRAINING over a compressed/streamed feature matrix: full-batch GD of a
    # SIGN-style MLP classifier; bit-exact vs in-core; runs beyond HBM. Build-only.
    add_cuda_executable(test_gpu_vector_gnn_train FALSE
        test_gpu_vector_gnn_train_gpu.cc
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
    set_target_properties(test_gpu_vector_gnn_train PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
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
