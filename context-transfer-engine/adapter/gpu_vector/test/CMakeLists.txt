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
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        # Without the cleanup fixture nothing reclaims the runtime's TCP port
        # or its SHM between test BINARIES, and the second one to run inherits
        # them. Measured: each passes 5/5 alone, together they failed 4 of 5.
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    # Vector::Copy (lazy checkpoint through the checkpoint chimod fault
    # handler): host + device reads of a copy tag materialise from the source.
    add_cuda_executable(test_gpu_vector_copy FALSE
        test_gpu_vector_copy_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/test
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/checkpoint/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
        LINK_LIBS
            clio_run_cxx
            clio_run_cxx_gpu
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            clio_cte_checkpoint_client
            clio_cte_checkpoint_runtime
            clio::cte::gpu_vector
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(test_gpu_vector_copy PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_copy COMMAND test_gpu_vector_copy
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_copy PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )

    add_cuda_executable(test_gpu_vector_generation FALSE
        test_gpu_vector_generation_gpu.cc
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
    set_target_properties(test_gpu_vector_generation PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_generation COMMAND test_gpu_vector_generation
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_generation PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        # Without the cleanup fixture nothing reclaims the runtime's TCP port
        # or its SHM between test BINARIES, and the second one to run inherits
        # them. Measured: each passes 5/5 alone, together they failed 4 of 5.
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )

        add_cuda_executable(test_gpu_vector_shared_persist FALSE
        test_gpu_vector_shared_persist_gpu.cc
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
    set_target_properties(test_gpu_vector_shared_persist PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_shared_persist COMMAND test_gpu_vector_shared_persist
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_shared_persist PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        # Without the cleanup fixture nothing reclaims the runtime's TCP port
        # or its SHM between test BINARIES, and the second one to run inherits
        # them. Measured: each passes 5/5 alone, together they failed 4 of 5.
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
        add_cuda_executable(test_gpu_vector_update_range FALSE
        test_gpu_vector_update_range_gpu.cc
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
    set_target_properties(test_gpu_vector_update_range PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_update_range COMMAND test_gpu_vector_update_range
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_update_range PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        # Without the cleanup fixture nothing reclaims the runtime's TCP port
        # or its SHM between test BINARIES, and the second one to run inherits
        # them. Measured: each passes 5/5 alone, together they failed 4 of 5.
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
        add_cuda_executable(test_gpu_vector_shared_cache FALSE
        test_gpu_vector_shared_cache_gpu.cc
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
    set_target_properties(test_gpu_vector_shared_cache PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_shared_cache COMMAND test_gpu_vector_shared_cache
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_shared_cache PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        # Without the cleanup fixture nothing reclaims the runtime's TCP port
        # or its SHM between test BINARIES, and the second one to run inherits
        # them. Measured: each passes 5/5 alone, together they failed 4 of 5.
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
        add_cuda_executable(test_cte_generation FALSE
        test_cte_generation_gpu.cc
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
    set_target_properties(test_cte_generation PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_generation COMMAND test_cte_generation
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_generation PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        # Without the cleanup fixture nothing reclaims the runtime's TCP port
        # or its SHM between test BINARIES, and the second one to run inherits
        # them. Measured: each passes 5/5 alone, together they failed 4 of 5.
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    add_cuda_executable(test_gpu_vector_workloads FALSE
        test_gpu_vector_workloads_gpu.cc
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
    set_target_properties(test_gpu_vector_workloads PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_workloads COMMAND test_gpu_vector_workloads
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_workloads PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    add_cuda_executable(test_gpu_vector_race FALSE
        test_gpu_vector_race_gpu.cc
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
    set_target_properties(test_gpu_vector_race PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_race COMMAND test_gpu_vector_race
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_race PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
        # The binary's DEFAULT shape (48 blocks x 8192 iters) is a manual
        # wedge REPRODUCER cranked far past this budget (seeding alone runs
        # ~6 min; iterations ~2 s each under storm backpressure). The
        # registered test caps iterations so the byte-exactness verdict
        # fits TIMEOUT; hunt wedges by running the binary directly.
        ENVIRONMENT "CLIO_RACE_ITERS=48"
    )
    add_cuda_executable(test_gpu_vector_semantics FALSE
        test_gpu_vector_semantics_gpu.cc
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
    set_target_properties(test_gpu_vector_semantics PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_semantics COMMAND test_gpu_vector_semantics
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_semantics PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    # 4-node distributed stress (see distributed/run_distributed.sh). Built
    # always so compile rot is caught; RUN only by the docker harness -- it
    # needs a cluster and a GPU, so there is deliberately no add_test here.
    add_cuda_executable(test_gpu_vector_distributed FALSE
        test_gpu_vector_distributed_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
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
    set_target_properties(test_gpu_vector_distributed PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS test_gpu_vector_distributed RUNTIME DESTINATION bin)

    # Distributed paged HALO EXCHANGE (see distributed_md_paged/). Same
    # rule as the test above: built always so compile rot is caught, RUN
    # only by the docker harness -- it needs a cluster and a GPU.
    add_cuda_executable(test_gpu_vector_md_distributed FALSE
        test_gpu_vector_md_distributed_gpu.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
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
    set_target_properties(test_gpu_vector_md_distributed PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS test_gpu_vector_md_distributed RUNTIME DESTINATION bin)

    # Distributed generational put/get -- CTE only, no GPU in the way
    # (see distributed_generation/). Built always, run by the harness.
    add_cuda_executable(test_cte_generation_distributed FALSE
        test_cte_generation_distributed.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
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
    set_target_properties(test_cte_generation_distributed PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS test_cte_generation_distributed RUNTIME DESTINATION bin)


    # Compressed-tier tests link the compressor pool; with
    # CLIO_CTE_ENABLE_COMPRESS=OFF (the GPU conda job's cuda-release preset)
    # those targets do not exist and CMake degrades them to bare -l flags
    # that fail at final link. Same guard as the benchmark side.
    if(TARGET clio_cte_compressor_runtime)
    add_cuda_executable(test_gpu_vector_compress FALSE
        test_gpu_vector_compress_gpu.cc
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
            clio_cte_compressor_client
            clio_cte_compressor_runtime
            clio::cte::gpu_vector
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(test_gpu_vector_compress PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_compress COMMAND test_gpu_vector_compress
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_compress PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    # ------------------------------------------------------------------
    # The GPU-ONLY path: kHBM tier with no host tier under it, nvcomp, and
    # a yielding fault path. Its own binary because CLIO_INIT runs once per
    # PROCESS and the storage tiers are fixed at that moment -- an HBM-only
    # runtime cannot be a second TEST_CASE beside a RAM-tier one.
    # ------------------------------------------------------------------
    add_cuda_executable(test_gpu_vector_hbm_nvcomp FALSE
        test_gpu_vector_hbm_nvcomp_gpu.cc
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
    set_target_properties(test_gpu_vector_hbm_nvcomp PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME cte_gpu_vector_hbm_nvcomp COMMAND test_gpu_vector_hbm_nvcomp
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_hbm_nvcomp PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;nvcomp"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    # GNN (GraphSAGE) forward over a lossless-zstd feature store. The features
    # are sourced through the CTE + compressor as raw device buffers, and the
    # result is checked bit-for-bit against an uncompressed baseline -- lossless
    # means EXACTLY equal, so this is an equality assert, not a tolerance.
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
    add_test(NAME cte_gpu_vector_gnn COMMAND test_gpu_vector_gnn
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_gnn PROPERTIES
        ENVIRONMENT "CLIO_CTE_COMPRESS_LIB=zstd"
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;compress;gnn;zstd"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    # Option B: the same GraphSAGE forward, but the features are STREAMED
    # through a real gv::Vector<float> backed by the compressor, with a cache
    # far smaller than the matrix so every page is a fault + decompress.
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
    add_test(NAME cte_gpu_vector_gnn_stream COMMAND test_gpu_vector_gnn_stream
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_gnn_stream PROPERTIES
        TIMEOUT 900
        LABELS "cte;gpu;gpu_vector;compress;gnn;zstd"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    # GNN training: mini-batch SGD over the compressed feature store, compared
    # against an in-core run (bit-exact where in-core fits, and where it OOMs
    # the vector is the only path that trained at all).
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
    add_test(NAME cte_gpu_vector_gnn_train COMMAND test_gpu_vector_gnn_train
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_gnn_train PROPERTIES
        TIMEOUT 1800
        LABELS "cte;gpu;gpu_vector;compress;gnn;zstd;train"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    # GNN capacity: the poster's headline -- an in-core run OOMs where the
    # vector streams the same matrix and produces the identical readout.
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
    add_test(NAME cte_gpu_vector_gnn_capacity COMMAND test_gpu_vector_gnn_capacity
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cte_gpu_vector_gnn_capacity PROPERTIES
        TIMEOUT 1800
        LABELS "cte;gpu;gpu_vector;compress;gnn;zstd;capacity"
        FIXTURES_REQUIRED clio_test_cleanup_fixture
        RUN_SERIAL TRUE
    )
    endif()  # TARGET clio_cte_compressor_runtime
endif()

# Peer node for the workload cluster tests (workloads/). Node 1 runs
# LAMMPS/GROMACS/LBANN against the paged path; nodes 2..N run this so their
# RAM tiers join the cluster and the workload's pages hash onto other
# machines.
#
# Deliberately OUTSIDE the CLIO_CORE_ENABLE_CUDA guard above. It contains no
# device code, and a peer's job is to hold blobs, not to run kernels -- so it
# must build in a CPU-only image, which is exactly what the cluster harness
# runs the peers in. Putting it inside the guard silently produced no target
# at all in a CPU build.
add_executable(eternia_cluster_node workloads/eternia_cluster_node.cc)
target_include_directories(eternia_cluster_node PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
)
target_link_libraries(eternia_cluster_node PRIVATE
    clio_run_cxx
    clio_admin_client
    clio_bdev_client
    clio_cte_core_client
    clio_cte_core_runtime
    ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(eternia_cluster_node PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
install(TARGETS eternia_cluster_node RUNTIME DESTINATION bin)

#------------------------------------------------------------------------------
# Distributed paged-workload gates, as LOCAL ctest entries.
#
# These stand up a 2-node docker cluster and check each workload's distributed
# result against its own single-node reference. They are real tests, not a
# parse: kmeans, weights, gmx and grayscott all pass, and grayscott's is backed
# by a negative control that fails the gate if disabling the halo exchange
# leaves the answer unchanged.
#
# All six paged workloads now pass here (lammps_md has its own harness under
# distributed_md_bench/).
#
# OPT-IN, because they need docker, the nvidia container toolkit and a GPU, and
# a machine without those should skip rather than fail. Turn them on with
#   -DCLIO_CORE_ENABLE_DOCKER_TESTS=ON
# and run with     ctest -L gv_distributed
#
# BUILD_DIR must name the build tree the containers mount, which is why it is
# passed explicitly rather than assumed to be "build".
#------------------------------------------------------------------------------
option(CLIO_CORE_ENABLE_DOCKER_TESTS
       "Run the 2-node docker distributed workload gates from ctest" OFF)

if(CLIO_CORE_ENABLE_CUDA AND CLIO_CORE_ENABLE_DOCKER_TESTS)
    get_filename_component(_gv_build_name "${CMAKE_BINARY_DIR}" NAME)
    foreach(_wl kmeans weights gmx grayscott lbann)
        add_test(NAME cte_gv_distributed_${_wl}
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/distributed_workloads/run_workloads_distributed.sh ${_wl}
        )
        set_tests_properties(cte_gv_distributed_${_wl} PROPERTIES
            # Two containers, a single-node reference run, and for grayscott a
            # control run as well -- the slowest is a few minutes.
            TIMEOUT 1800
            LABELS "cte;gpu;gpu_vector;gv_distributed;docker"
            # One cluster at a time: they share a fixed subnet and port.
            RUN_SERIAL TRUE
            ENVIRONMENT "BUILD_DIR=${_gv_build_name}"
        )
    endforeach()
    message(STATUS "gpu_vector distributed docker gates: ON "
                   "(ctest -L gv_distributed, BUILD_DIR=${_gv_build_name})")
endif()
