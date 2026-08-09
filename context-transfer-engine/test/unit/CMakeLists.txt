# ------------------------------------------------------------------------------
# CTE Core Unit Tests
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.10)

# Note: Binary output directory is configured in root CMakeLists.txt to use build/bin
# The test executable will be placed in build/bin directory due to global
# CMAKE_RUNTIME_OUTPUT_DIRECTORY setting

# Find required test dependencies
# Catch2 is already found by root CMakeLists.txt

# Create the test executable — one file per target to avoid TEST_CASE
# line-number collisions that produce duplicate symbol link errors.
add_executable(cte_core_unit_tests
    test_core_minimal.cc
)

add_executable(cte_core_functionality_simple_tests
    test_core_functionality_simple.cc
)

# Create separate executable for config/DPE tests (avoids TEST_CASE line number collisions)
add_executable(test_cte_config_dpe
    test_cte_config_dpe.cc
)

# Create single version of test_core_functionality that uses environment variable
# CLIO_WITH_RUNTIME to control whether runtime is initialized (default: yes)
add_executable(test_core_functionality
    test_core_functionality.cc
)

# Create test_query executable for query API tests (TagQuery, BlobQuery)
add_executable(test_query
    test_query.cc
)

# Create test_tag_operations executable for Tag API coverage tests
add_executable(test_tag_operations
    test_tag_operations.cc
)

# Create test_core_client_config executable for Client and Config API coverage tests
add_executable(test_core_client_config
    test_core_client_config.cc
)

# Create test_core_runtime_coverage executable for Runtime method coverage tests
add_executable(test_core_runtime_coverage
    test_core_runtime_coverage.cc
)

# Create test_tiered_storage_stress executable for tiered storage stress tests
add_executable(test_tiered_storage_stress
    test_tiered_storage_stress.cc
)

# Create test_tiered_storage_dram_default executable — stresses the
# "0g" -> 80% DRAM RAM-tier default through the CTE tiering path.
add_executable(test_tiered_storage_dram_default
    test_tiered_storage_dram_default.cc
)

# Regression: PutBlob(1MB)+DelBlob frees blocks back to their bdev
# allocator partition so the next same-size Put reuses them.
add_executable(test_blob_block_reuse
    test_blob_block_reuse.cc
)

# Create test_reorganize_blob executable for ReorganizeBlob API tests
add_executable(test_reorganize_blob
    test_reorganize_blob.cc
)

# Score-driven capacity eviction (Method::kEvict) API tests
add_executable(test_evict
    test_evict.cc
)

# Internal data organizer (issue #738): factory/frecency-score/config units
# plus the end-to-end periodic DynamicReorganize promote/demote flow.
add_executable(test_data_organizer
    test_data_organizer.cc
)

# I/O emulation (issue #747): Context::emulate_ on PutBlob/GetBlob (modeled
# time, buffer untouched, WAL skipped) + bdev perf-stats persistence.
add_executable(test_io_emulation
    test_io_emulation.cc
)

# BDEV leak stress: PutBlob/DelBlob loop for a wall-clock minute, then assert
# the bdev used capacity is 0 and the runtime leak-checker heap counter is flat.
add_executable(test_bdev_leak_stress
    test_bdev_leak_stress.cc
)

add_executable(test_concurrent_io_stress
    test_concurrent_io_stress.cc
)

# Same-blob write-race regression (#680 / xfstests generic/074). Many threads
# write disjoint regions of ONE shared blob; the per-blob write token must keep
# each region intact.
add_executable(test_wakeup_never_miss
    test_wakeup_never_miss.cc
)

add_executable(test_concurrent_same_blob
    test_concurrent_same_blob.cc
)

# Vectored PutBlob/GetBlob (#820): N regions in one task, one write-token
# acquire. Asserts equivalence with N single puts, list-order overlap
# resolution, and union-range allocation.
add_executable(test_vectored_blob_io
    test_vectored_blob_io.cc
)

# bdev fragmentation reuse (#820): large alloc served from fragmented small
# free extents -- the decisive regression test for moving anti-fragmentation
# from the CTE into the bdev allocator.
add_executable(test_bdev_fragmentation
    test_bdev_fragmentation.cc
)

# Freed space must reach the DPE without waiting for a StatTargets tick (#919).
# Drives the stale-mirror window that made cte_bdev_fragmentation flaky.
add_executable(test_bdev_free_visibility
    test_bdev_free_visibility.cc
)

# Batching benchmark (#820): many small writes at ONE blob, A/B'd on the same
# binary via CLIO_TASK_BATCHING.
add_executable(bench_same_blob_batching
    bench_same_blob_batching.cc
)

# Parallel blob-metadata overlap stress: many threads race
# PutBlob/GetBlob/DelBlob/GetBlobInfo over only 4 tags x 4 blobs (heavy
# overlap). Success = no crashes, hangs, or leaks.
add_executable(test_cte_parallel_overlap_stress
    test_cte_parallel_overlap_stress.cc
)

# GPU-runtime-era CUDA tests were removed when the GPU runtime concept
# was retired. The kernel→CPU submission path is now exercised by:
#   - context-runtime/test/unit/gpu/test_gpu_kernel_stress_*
#     (kPinnedHost task submission to a dummy chimod).
#   - context-transfer-engine/adapter/gpu_vector/test/test_gpu_vector_gpu.cc
#     (kDeviceMem-backed pages flushed to bdev kRam via PutBlob).

# Include directories for all tests - using unified simple_test.h framework
target_include_directories(cte_core_unit_tests PRIVATE

)

target_include_directories(cte_core_functionality_simple_tests PRIVATE

)

target_include_directories(test_core_functionality PRIVATE

)

target_include_directories(test_query PRIVATE

)

target_include_directories(test_cte_config_dpe PRIVATE

)

target_include_directories(test_tag_operations PRIVATE

)

target_include_directories(test_core_client_config PRIVATE

)

target_include_directories(test_core_runtime_coverage PRIVATE

)

target_include_directories(test_tiered_storage_stress PRIVATE

)

target_include_directories(test_tiered_storage_dram_default PRIVATE

)

target_include_directories(test_blob_block_reuse PRIVATE

)

target_include_directories(test_reorganize_blob PRIVATE

)

# GPU runtime tests + their include directories removed.

# Link against required libraries using simple_test.h framework (NOT Catch2)
target_link_libraries(cte_core_unit_tests
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

target_link_libraries(cte_core_functionality_simple_tests
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Link test_core_functionality - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_core_functionality
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_cte_config_dpe - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_cte_config_dpe
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_query - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_query
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_tag_operations - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_tag_operations
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_core_client_config - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_core_client_config
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_core_runtime_coverage - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_core_runtime_coverage
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_tiered_storage_stress - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_tiered_storage_stress
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_tiered_storage_dram_default - using simple_test.h framework
target_link_libraries(test_tiered_storage_dram_default
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum + policy helper
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_blob_block_reuse - using simple_test.h framework
target_link_libraries(test_blob_block_reuse
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_reorganize_blob - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_reorganize_blob
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_evict - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_evict
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_data_organizer - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_data_organizer
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_io_emulation - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_io_emulation
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (perf-stats persistence helpers)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_bdev_leak_stress - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_bdev_leak_stress
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_concurrent_io_stress (#680 concurrency repro) - same deps as the
# leak stress test (embedded runtime + CTE core + bdev + threads).
target_link_libraries(test_concurrent_io_stress
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(test_concurrent_io_stress PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
)

# Link test_wakeup_never_miss (gated signalling park/wake race) - same deps.
target_link_libraries(test_wakeup_never_miss
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(test_wakeup_never_miss PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
)

# Link test_concurrent_same_blob (#680 same-blob write race) - same deps.
target_link_libraries(test_concurrent_same_blob
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Link test_vectored_blob_io (#820 vectored blob I/O) - same deps.
target_link_libraries(test_vectored_blob_io
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Link test_bdev_fragmentation (#820) - same deps.
target_link_libraries(test_bdev_fragmentation
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Link test_bdev_free_visibility (#919) - same deps.
target_link_libraries(test_bdev_free_visibility
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Link bench_same_blob_batching (#820) - same deps.
target_link_libraries(bench_same_blob_batching
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(test_concurrent_same_blob PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
)

# Link test_cte_parallel_overlap_stress - same deps (embedded runtime + CTE
# core + bdev + threads).
target_link_libraries(test_cte_parallel_overlap_stress
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(test_cte_parallel_overlap_stress PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
)

# GPU-runtime-era link rules removed.

# Note: Include directories are provided automatically via linked ChiMod targets
# clio_cte_core_client provides ${CLIO_CTE_ROOT}/core/include via transitive includes

# Manual test registration for specific test cases
add_test(NAME cte_core_pool_creation
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_target_registration
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_blob_put
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_blob_get
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_integration
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_client_creation
    COMMAND cte_core_functionality_simple_tests)

add_test(NAME cte_core_params
    COMMAND cte_core_functionality_simple_tests)

add_test(NAME cte_core_target_config
    COMMAND cte_core_functionality_simple_tests)

add_test(NAME cte_core_tag_info
    COMMAND cte_core_functionality_simple_tests)

add_test(NAME cte_core_blob_info
    COMMAND cte_core_functionality_simple_tests)

add_test(NAME cte_core_tasks
    COMMAND cte_core_functionality_simple_tests)

add_test(NAME cte_core_helpers
    COMMAND cte_core_functionality_simple_tests)

add_test(NAME cte_core_workflow
    COMMAND cte_core_functionality_simple_tests)

add_test(NAME cte_core_performance
    COMMAND cte_core_functionality_simple_tests)

# CTE Config and DPE tests (separate executable)
add_test(NAME cte_config_tests
    COMMAND test_cte_config_dpe "[cte][config]")
add_test(NAME cte_dpe_tests
    COMMAND test_cte_config_dpe "[cte][dpe]")

# Add test_core_functionality as a single test (the binary runs all 9 internal
# test cases in one invocation; duplicating CTest entries causes parallel runs
# that fight over shared memory and ports).
add_test(NAME cte_functional_all
    COMMAND test_core_functionality)

# Add test_query tests - use test names for filtering (simple_test.h filters by name, not tags)
add_test(NAME cte_query_tag_exact
    COMMAND test_query "TagQuery - Exact Match")

add_test(NAME cte_query_tag_wildcard
    COMMAND test_query "TagQuery - Wildcard Pattern")

add_test(NAME cte_query_tag_alternation
    COMMAND test_query "TagQuery - Alternation Pattern")

add_test(NAME cte_query_tag_matchall
    COMMAND test_query "TagQuery - Match All Pattern")

add_test(NAME cte_query_tag_nomatch
    COMMAND test_query "TagQuery - No Matches")

add_test(NAME cte_query_blob_exact
    COMMAND test_query "BlobQuery - Exact Match")

add_test(NAME cte_query_blob_wildcard
    COMMAND test_query "BlobQuery - Wildcard Patterns")

add_test(NAME cte_query_blob_multitag
    COMMAND test_query "BlobQuery - Multiple Tags")

add_test(NAME cte_query_blob_matchall
    COMMAND test_query "BlobQuery - Match All")

add_test(NAME cte_query_blob_no_blob
    COMMAND test_query "BlobQuery - No Blob Matches")

add_test(NAME cte_query_blob_no_tag
    COMMAND test_query "BlobQuery - No Tag Matches")

# Note: cte_query_blob_extension test removed - not implemented in simple_test.h version

add_test(NAME cte_query_local_poolquery
    COMMAND test_query "Query - Local Pool Query")

# Add test_tag_operations tests - comprehensive Tag API coverage tests
add_test(NAME cte_tag_construction
    COMMAND test_tag_operations "Tag - Construction")

add_test(NAME cte_tag_putblob
    COMMAND test_tag_operations "Tag - PutBlob")

add_test(NAME cte_tag_getblob
    COMMAND test_tag_operations "Tag - GetBlob")

add_test(NAME cte_tag_errors
    COMMAND test_tag_operations "Tag - Error")

add_test(NAME cte_tag_metadata
    COMMAND test_tag_operations "Tag - Metadata")

add_test(NAME cte_tag_reorganize
    COMMAND test_tag_operations "Tag - Reorganize")

add_test(NAME cte_tag_async
    COMMAND test_tag_operations "Tag - Async")

add_test(NAME cte_tag_large
    COMMAND test_tag_operations "Tag - Large")

add_test(NAME cte_tag_edge
    COMMAND test_tag_operations "Tag - Edge")

# Single-process runs of test_core_client_config / test_core_runtime_coverage.
# Previously each Catch2 SECTION was registered as a standalone ctest entry
# (27 + 19), which serialized via RESOURCE_LOCK but multiplied clio_run
# init/finalize cycles and exposed shutdown ZMQ races.
add_test(NAME cte_client_config_all
    COMMAND test_core_client_config)

add_test(NAME cte_runtime_coverage_all
    COMMAND test_core_runtime_coverage)

# Set test properties for proper execution environment
set_tests_properties(
    cte_core_pool_creation
    cte_core_target_registration
    cte_core_blob_put
    cte_core_blob_get
    cte_core_integration
    cte_core_client_creation
    cte_core_params
    cte_core_target_config
    cte_core_tag_info
    cte_core_blob_info
    cte_core_tasks
    cte_core_helpers
    cte_config_tests
    cte_dpe_tests
    cte_core_workflow
    cte_core_performance
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout for each test
        LABELS "unit;core;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
)

set_tests_properties(
    cte_functional_all
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout
        LABELS "functional;core;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

set_tests_properties(
    cte_query_tag_exact
    cte_query_tag_wildcard
    cte_query_tag_alternation
    cte_query_tag_matchall
    cte_query_tag_nomatch
    cte_query_blob_exact
    cte_query_blob_wildcard
    cte_query_blob_multitag
    cte_query_blob_matchall
    cte_query_blob_no_blob
    cte_query_blob_no_tag
    cte_query_local_poolquery
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout for each test
        LABELS "query;core;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

set_tests_properties(
    cte_tag_construction
    cte_tag_putblob
    cte_tag_getblob
    cte_tag_errors
    cte_tag_metadata
    cte_tag_reorganize
    cte_tag_async
    cte_tag_large
    cte_tag_edge
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout for each test
        LABELS "unit;tag;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

set_tests_properties(cte_client_config_all PROPERTIES
    TIMEOUT 600
    LABELS "unit;core;cte;client;config"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
)

set_tests_properties(cte_runtime_coverage_all PROPERTIES
    TIMEOUT 600
    LABELS "unit;core;cte;runtime;coverage"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

# BDEV leak stress: a 1-minute PutBlob/DelBlob churn that asserts the bdev used
# capacity returns to 0 and the runtime leak-checker heap counter stays flat.
# Name contains "leak" so the Leak Check CI (-R "leak|...force_net") runs it.
add_test(NAME cte_bdev_leak_stress
    COMMAND test_bdev_leak_stress)
set_tests_properties(cte_bdev_leak_stress PROPERTIES
    TIMEOUT 180
    LABELS "unit;cte;bdev;leak;stress;msan_skip"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;CTE_LEAK_STRESS_SECONDS=60"
)

# Concurrent I/O stress: many threads hammer the shared per-process allocator
# via AllocateBuffer/FreeBuffer (+ optional Put/Get) to reproduce the #680
# multi-threaded buffer/task race without fsx/FUSE. NOT registered with add_test
# yet: it still fails until the shared_ptr<Task>/Future completion race in the
# client-submit -> worker-process path is fixed (TSan-confirmed root cause). Run
# it manually as a repro: build/bin/test_concurrent_io_stress.

# Add tiered storage stress tests
# NOTE: These tests are chained (Put -> Reorganize -> Integrity -> Cleanup)
# and share global state via g_fixture, so they must run together in a single
# process. Running them as separate ctest entries would lose shared state.
add_test(NAME cte_tiered_storage_all
    COMMAND test_tiered_storage_stress)

set_tests_properties(
    cte_tiered_storage_all
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout
        LABELS "stress;tiered;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

# Stresses the "0g" -> 80% DRAM RAM-tier default through the CTE tiering
# path. Chained (policy -> Put -> Reorganize -> Integrity -> Cleanup) and
# share global state via g_fixture, so they run together in one process.
add_test(NAME cte_tiered_dram_default_all
    COMMAND test_tiered_storage_dram_default)

set_tests_properties(
    cte_tiered_dram_default_all
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout
        LABELS "stress;tiered;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

# Parallel overlap stress: Put/Get/Del/GetInfo race on 4 tags x 4 blobs.
# Pass = the process finishes (no crash) within the timeout (no hang); run under
# sanitizers for the no-leak metric. Labeled concurrent (NOT "stress") so it is
# included in the default CI ctest run -- like test_concurrent_same_blob -- and
# a CI-friendly iteration count keeps it fast even in the instrumented coverage
# build (override STRESS_ITERS/STRESS_THREADS locally for a heavier soak).
add_test(NAME cte_parallel_overlap_stress
    COMMAND test_cte_parallel_overlap_stress)

set_tests_properties(
    cte_parallel_overlap_stress
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout -> a hang trips this
        LABELS "concurrent;cte;msan_skip"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;STRESS_THREADS=8;STRESS_ITERS=1500"
)

# Regression: blocks freed by DelBlob return to their bdev partition.
add_test(NAME cte_block_reuse_all
    COMMAND test_blob_block_reuse)

set_tests_properties(
    cte_block_reuse_all
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout
        LABELS "regression;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

# Regression: concurrent writes to disjoint regions of ONE shared blob must not
# corrupt the shared block layout (#680 / xfstests generic/074 fsx mismatches).
# Every writer serializes on the same per-blob write token, so the workload is
# effectively a single serial stream of PutBlobs. The 1500-iter default writes
# ~800 MB and cannot finish inside the 300s budget on an instrumented Debug/gcov
# build (measured: ~33% done at timeout on the coverage runner), so -- exactly
# like cte_parallel_overlap_stress above -- pin a CI-friendly iteration count in
# the environment. 8 threads x 100 iters still races ~800 concurrent same-blob
# extends, which is what detects the corruption; the 1500-iter soak is a local
# knob (override SAME_BLOB_ITERS/SAME_BLOB_THREADS for a heavier run).
# Targeted never-miss-wakeup regression (gated producer->worker signalling).
# Cold-submits after every worker parks; a missed wake shows as the ~50ms
# max_sleep self-heal latency signature on nearly every op (budgeted assert),
# and a lost-wake hang trips the timeout. CI pins a lighter iteration count;
# the 800-iter soak is the local default.
add_test(NAME cr_wakeup_never_miss
    COMMAND test_wakeup_never_miss)
set_tests_properties(
    cr_wakeup_never_miss
    PROPERTIES
        TIMEOUT 300
        LABELS "regression;wakeup;runtime"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;WAKEUP_ITERS=400"
)

add_test(NAME cte_concurrent_same_blob_all
    COMMAND test_concurrent_same_blob)

set_tests_properties(
    cte_concurrent_same_blob_all
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout -> a hang trips this
        LABELS "regression;concurrent;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;SAME_BLOB_THREADS=8;SAME_BLOB_ITERS=100"
)

# issue #820: run the SAME rigorous same-blob correctness test again, this time
# THROUGH the batching merge path (CLIO_CTE_BATCHING + CLIO_BATCH_LANE). Without
# these env vars BuildBatch returns at its policy gate and SmashBatch /
# CompleteBatchParents / the inline-exec sink never run -- which is why coverage
# of the merge path was ~0. Reusing this test means the merged path is checked
# by the strongest correctness assertion there is (each thread's disjoint region
# must survive intact), not a bespoke smoke test.
add_test(NAME cte_concurrent_same_blob_batched
    COMMAND test_concurrent_same_blob)
set_tests_properties(
    cte_concurrent_same_blob_batched
    PROPERTIES
        TIMEOUT 300
        LABELS "regression;concurrent;cte;batching"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;SAME_BLOB_THREADS=8;SAME_BLOB_ITERS=100;CLIO_CTE_BATCHING=1;CLIO_BATCH_LANE=1"
)

# issue #820: vectored PutBlob/GetBlob correctness (equivalence with N single
# puts, list-order overlap, union-range allocation, get-scatter, malformed-
# segment rejection).
add_test(NAME cte_vectored_blob_io
    COMMAND test_vectored_blob_io)
set_tests_properties(cte_vectored_blob_io PROPERTIES
    TIMEOUT 300
    LABELS "unit;cte;vectored;820"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1")

# issue #820: bdev fragmentation reuse -- large alloc served from fragmented
# small free extents. Decisive regression test for moving anti-fragmentation
# from the CTE into the bdev allocator (the 074/521/522 scenario).
add_test(NAME cte_bdev_fragmentation
    COMMAND test_bdev_fragmentation)
set_tests_properties(cte_bdev_fragmentation PROPERTIES
    TIMEOUT 300
    LABELS "regression;bdev;cte;820"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1")

# issue #919: placement must see space freed since the last StatTargets tick.
# Four fill/stat/delete/put cycles; a stray periodic tick can rescue one, not all.
add_test(NAME cte_bdev_free_visibility
    COMMAND test_bdev_free_visibility)
set_tests_properties(cte_bdev_free_visibility PROPERTIES
    TIMEOUT 600
    LABELS "regression;bdev;cte;919"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1")

# Add reorganize blob tests
# NOTE: These tests are chained (Put -> Move -> Integrity -> Promote -> Cleanup)
# and share global state via g_fixture, so they must run together in a single
# process. Running them as separate ctest entries would lose shared state.
add_test(NAME cte_reorganize_all
    COMMAND test_reorganize_blob)

set_tests_properties(
    cte_reorganize_all
    PROPERTIES
        # 600s: the #753 race tests seed a multi-chunk victim and fill the
        # disk tier; Windows Debug runners need the headroom even with the
        # in-test geometry scaling (cte_reorganize_all timed out at 300s on
        # windows-2025).
        TIMEOUT 600
        LABELS "unit;reorganize;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

# Deferred-put pipeline + multiput batches (issue #862)
add_executable(test_multiput_defer test_multiput_defer.cc)
target_link_libraries(test_multiput_defer
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_multiput_defer
    COMMAND test_multiput_defer)

set_tests_properties(
    cte_multiput_defer
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;multiput;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

# Blob replicas + replication chimod (issue #886). Gated on the module: the
# mechanism tests also exercise the policy verbs, and the module is ON by
# default, so the gate only matters for trimmed builds.
if(TARGET clio_cte_replication_client)
add_executable(test_blob_replicas test_blob_replicas.cc)
target_include_directories(test_blob_replicas PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/replication/include
)
target_link_libraries(test_blob_replicas
    clio_cte_core_runtime
    clio_cte_core_client
    clio_cte_replication_runtime
    clio_cte_replication_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_blob_replicas
    COMMAND test_blob_replicas)

set_tests_properties(
    cte_blob_replicas
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;replicas;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)
endif()  # TARGET clio_cte_replication_client

# Compressor interposition (issue #886). Gated on the module (build with
# CLIO_CTE_ENABLE_COMPRESS=ON): a plain core client against the compressor
# pool — transparent compress on whole-blob puts, decompress on reads incl.
# partial/vectored, logical sizes, forwarded batches and metadata.
if(TARGET clio_cte_compressor_client)
add_executable(test_compressor_interpose test_compressor_interpose.cc)
target_include_directories(test_compressor_interpose PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
)
target_link_libraries(test_compressor_interpose
    clio_cte_core_runtime
    clio_cte_core_client
    clio_cte_compressor_runtime
    clio_cte_compressor_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_compressor_interpose
    COMMAND test_compressor_interpose)

set_tests_properties(
    cte_compressor_interpose
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;compressor;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)
endif()  # TARGET clio_cte_compressor_client

# Full interposer stack (issue #886): compressor(562) -> replication(561) ->
# core(512) under ONE plain core client with the SHM mirror attached —
# fast path functional for raw blobs, REFUSED for compressed ones, and
# drop+heal end to end. Needs both interposer modules.
if(TARGET clio_cte_compressor_client AND TARGET clio_cte_replication_client)
add_executable(test_interposer_stack test_interposer_stack.cc)
target_include_directories(test_interposer_stack PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/replication/include
)
target_link_libraries(test_interposer_stack
    clio_cte_core_runtime
    clio_cte_core_client
    clio_cte_compressor_runtime
    clio_cte_compressor_client
    clio_cte_replication_runtime
    clio_cte_replication_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_interposer_stack
    COMMAND test_interposer_stack)

set_tests_properties(
    cte_interposer_stack
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;compressor;replicas;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)
endif()  # full interposer stack

# Cache chimod (issue #886 cache/replication split): cache(563) -> core(512)
# under a plain core client — write-back semantics (ack before primary),
# overwrite coalescing, SHM fast path served from the cache replica, and
# miss + re-populate.
if(TARGET clio_cte_cache_client)
add_executable(test_cache_interpose test_cache_interpose.cc)
target_include_directories(test_cache_interpose PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/cache/include
)
target_link_libraries(test_cache_interpose
    clio_cte_core_runtime
    clio_cte_core_client
    clio_cte_cache_runtime
    clio_cte_cache_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_cache_interpose
    COMMAND test_cache_interpose)

set_tests_properties(
    cte_cache_interpose
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;cache;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)
endif()  # TARGET clio_cte_cache_client

#------------------------------------------------------------------------------
# Indexer chimod behavioral coverage (issue #905): scope, async drain +
# search barrier, maintenance verbs, kReindexScan, first-insertion backfill.
#------------------------------------------------------------------------------
if(TARGET clio_cte_indexer_client)
add_executable(test_indexer_ops test_indexer_ops.cc)
target_include_directories(test_indexer_ops PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/indexer/include
)
target_link_libraries(test_indexer_ops
    clio_cte_core_runtime
    clio_cte_core_client
    clio_cte_indexer_runtime
    clio_cte_indexer_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_indexer_ops
    COMMAND test_indexer_ops)

set_tests_properties(
    cte_indexer_ops
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;indexer;cte;semantic-search;coverage"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)
endif()  # TARGET clio_cte_indexer_client

# Full 4-layer stack (issue #886 cache/replication split): cache(563) ->
# compressor(562) -> replication(561) -> core(512). A compressed blob KEEPS
# the SHM fast path via its raw cache replica; the persistent replica holds
# the stored form marked transformed; evict + refill round-trips.
if(TARGET clio_cte_cache_client AND TARGET clio_cte_compressor_client
   AND TARGET clio_cte_replication_client)
add_executable(test_cache_stack test_cache_stack.cc)
target_include_directories(test_cache_stack PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/cache/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/replication/include
)
target_link_libraries(test_cache_stack
    clio_cte_core_runtime
    clio_cte_core_client
    clio_cte_cache_runtime
    clio_cte_cache_client
    clio_cte_compressor_runtime
    clio_cte_compressor_client
    clio_cte_replication_runtime
    clio_cte_replication_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_cache_stack
    COMMAND test_cache_stack)

set_tests_properties(
    cte_cache_stack
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;cache;compressor;replicas;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)
endif()  # full 4-layer cache stack

# Add score-driven eviction tests (Method::kEvict)
add_test(NAME cte_evict
    COMMAND test_evict)

set_tests_properties(
    cte_evict
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;evict;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

# Add data organizer tests (issue #738)
# NOTE: like the reorganize tests, the cases share global fixture state
# (blobs put in one case are checked/cleaned in later cases), so they run
# together in a single process.
add_test(NAME cte_data_organizer_all
    COMMAND test_data_organizer)

set_tests_properties(
    cte_data_organizer_all
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout
        LABELS "unit;organizer;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

# Add I/O emulation tests (issue #747)
# NOTE: like the reorganize tests, the cases share global fixture state, so
# they run together in a single process.
add_test(NAME cte_io_emulation_all
    COMMAND test_io_emulation)

set_tests_properties(
    cte_io_emulation_all
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout
        LABELS "unit;emulation;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)

# ---------------------------------------------------------------------------
# CLIO_FORCE_NET stress variants (one per CTE task-path test binary)
# Route every non-Local task through the loopback run-to-run network path so
# the serialize/send/recv/aggregate path is exercised for all CTE task types
# (PutBlob/GetBlob/Tag/Query/Reorganize/...) on a single node. GPU-only and
# pure-stress binaries are intentionally excluded.
# ---------------------------------------------------------------------------
set(_cte_fn_env_basic "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1")
# CLIO_NET_TRACE exercises the env-gated per-stage net telemetry (issue
# #892) on the wire path these force_net tests already traverse.
set(_cte_fn_env_data "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;CLIO_NET_TRACE=1")
clio_add_force_net_test(NAME cte_core_unit_force_net
    COMMAND cte_core_unit_tests ENVIRONMENT "${_cte_fn_env_basic}")
clio_add_force_net_test(NAME cte_core_simple_force_net
    COMMAND cte_core_functionality_simple_tests ENVIRONMENT "${_cte_fn_env_basic}")
clio_add_force_net_test(NAME cte_functional_force_net
    COMMAND test_core_functionality ENVIRONMENT "${_cte_fn_env_data}")
clio_add_force_net_test(NAME cte_tag_force_net
    COMMAND test_tag_operations ENVIRONMENT "${_cte_fn_env_data}")
clio_add_force_net_test(NAME cte_query_force_net
    COMMAND test_query ENVIRONMENT "${_cte_fn_env_data}")
# 600s: the #753 race tests seed a multi-chunk victim + fill the disk tier,
# and every one of those ops is a loopback net round trip in this variant —
# the geometry already shrinks under CLIO_FORCE_NET (see the test), but the
# slowest Windows runners still need headroom over the 300s default.
clio_add_force_net_test(NAME cte_reorganize_force_net TIMEOUT 600
    COMMAND test_reorganize_blob ENVIRONMENT "${_cte_fn_env_data}")
# Put(1MB)+DelBlob exercises the 1 MiB-bulk + free-to-partition net round-trip.
# The full 1024-cycle regression is ~300 ms/cycle over the loopback net path
# (~5 min), so cap the force-net variant at 64 cycles via BLOCK_REUSE_CYCLES —
# enough to stress the path without blowing the timeout. Passes.
clio_add_force_net_test(NAME cte_block_reuse_force_net
    COMMAND test_blob_block_reuse
    ENVIRONMENT "${_cte_fn_env_data};BLOCK_REUSE_CYCLES=64")
clio_add_force_net_test(NAME cte_runtime_coverage_force_net
    COMMAND test_core_runtime_coverage ENVIRONMENT "${_cte_fn_env_data}" TIMEOUT 600)
clio_add_force_net_test(NAME cte_client_config_force_net
    COMMAND test_core_client_config ENVIRONMENT "${_cte_fn_env_basic}" TIMEOUT 600)
clio_add_force_net_test(NAME cte_config_dpe_force_net
    COMMAND test_cte_config_dpe ENVIRONMENT "${_cte_fn_env_basic}")
# BDEV leak stress over the loopback net path. Each Put/Del round-trips the net
# transport (far slower than in-process), so use a shorter window — still enough
# churn to expose a per-op leak, while keeping the CI job bounded.
clio_add_force_net_test(NAME cte_bdev_leak_stress_force_net
    COMMAND test_bdev_leak_stress
    ENVIRONMENT "${_cte_fn_env_data};CTE_LEAK_STRESS_SECONDS=20" TIMEOUT 180)
# Blob replicas (issue #886) over the loopback net path: the only variant that
# serializes the replication chimod's tasks (CachedPut/CachedGet/
# ReplicateBlob/FlushTag) through their autogen Save/Load wire path.
if(TARGET clio_cte_replication_client)
clio_add_force_net_test(NAME cte_blob_replicas_force_net
    COMMAND test_blob_replicas ENVIRONMENT "${_cte_fn_env_data}" TIMEOUT 600)
endif()

# NOTE: the legacy GPU CTE tests cte_gpu_coroutine / cte_gpu_core_all /
# cte_gpu_create were removed — their executables (test_cte_gpu_coroutine /
# test_gpu_core / test_gpu_create) no longer exist, so the old add_test()
# registrations were dead (ctest reported them "Not Run: Could not find
# executable"). The current GPU CTE test is cte_devmem_putget_cuda
# (test_cte_devmem_putget, defined below).

# Mark all CTE tests as msan_skip (they use ZMQ via clio_run runtime which is
# uninstrumented).
# NOTE: set_tests_properties(LABELS ...) REPLACES each test's label list (the
# same clobbering trap called out for cte_functional_all below), so these tests
# lose their own labels (e.g. "regression;cte", "query;core;cte"). Fixing it is
# NOT a drive-by: the coverage gate excludes -LE "query|stress|functional", so
# restoring labels changes which tests the gate runs. Tracked separately.
set_tests_properties(
    cte_core_pool_creation
    cte_core_target_registration
    cte_core_blob_put
    cte_core_blob_get
    cte_core_integration
    cte_core_client_creation
    cte_core_params
    cte_core_target_config
    cte_core_tag_info
    cte_core_blob_info
    cte_core_tasks
    cte_core_helpers
    cte_core_workflow
    cte_core_performance
    cte_config_tests
    cte_dpe_tests
    cte_functional_all
    cte_query_tag_exact
    cte_query_tag_wildcard
    cte_query_tag_alternation
    cte_query_tag_matchall
    cte_query_tag_nomatch
    cte_query_blob_exact
    cte_query_blob_wildcard
    cte_query_blob_multitag
    cte_query_blob_matchall
    cte_query_blob_no_blob
    cte_query_blob_no_tag
    cte_query_local_poolquery
    cte_tag_construction
    cte_tag_putblob
    cte_tag_getblob
    cte_tag_errors
    cte_tag_metadata
    cte_tag_reorganize
    cte_tag_async
    cte_tag_large
    cte_tag_edge
    cte_client_config_all
    cte_runtime_coverage_all
    cte_tiered_storage_all
    cte_tiered_dram_default_all
    cte_block_reuse_all
    cte_reorganize_all
    cte_data_organizer_all
    cte_io_emulation_all
    PROPERTIES LABELS "msan_skip"
)

# Tests that start a full CLIO Runtime runtime share /tmp/clio_iowarp SHM
# segments and TCP/IPC ports, so they must NOT run in parallel. They also
# need the clio_run cleanup fixture so a previous runtime process holding
# 0.0.0.0:9416 doesn't break the next test's `IpcManager::ServerInit` with
# "Address in use" (especially on Windows, where TIME_WAIT can hang on for
# minutes and SIMPLE_TEST_PROCESS_EXIT skips libzmq's own socket close).
set_tests_properties(
    cte_query_tag_exact
    cte_query_tag_wildcard
    cte_query_tag_alternation
    cte_query_tag_matchall
    cte_query_tag_nomatch
    cte_query_blob_exact
    cte_query_blob_wildcard
    cte_query_blob_multitag
    cte_query_blob_matchall
    cte_query_blob_no_blob
    cte_query_blob_no_tag
    cte_query_local_poolquery
    cte_tag_construction
    cte_tag_putblob
    cte_tag_getblob
    cte_tag_errors
    cte_tag_metadata
    cte_tag_reorganize
    cte_tag_async
    cte_tag_large
    cte_tag_edge
    cte_functional_all
    cte_client_config_all
    cte_runtime_coverage_all
    cte_tiered_storage_all
    cte_tiered_dram_default_all
    cte_block_reuse_all
    cte_reorganize_all
    cte_data_organizer_all
    cte_io_emulation_all
    PROPERTIES
        RESOURCE_LOCK "clio_runtime"
        FIXTURES_REQUIRED "clio_test_cleanup_fixture"
)
# Move the runtime off port 9413 (and its derivatives 9414/9416) onto
# a high range so that an IDE / random local server squatting on the
# default port doesn't break the test run. libzmq's Windows socket
# handle is inheritable, which lets an ancestor process (e.g. the
# IDE that spawned ctest) accidentally keep a listener alive across
# test runs. Until that's fixed upstream, just dodge.
# APPEND: setting ENVIRONMENT via set_tests_properties REPLACED the env list
# set alongside each add_test above, silently dropping CLIO_TEST_DATA_DIR and
# CLIO_BIND_ADDR for every test in this group (cte_block_reuse_all then wrote
# its config to "./" and bound 0.0.0.0 instead of 127.0.0.1).
set_property(TEST
    cte_query_tag_exact
    cte_query_tag_wildcard
    cte_query_tag_alternation
    cte_query_tag_matchall
    cte_query_tag_nomatch
    cte_query_blob_exact
    cte_query_blob_wildcard
    cte_query_blob_multitag
    cte_query_blob_matchall
    cte_query_blob_no_blob
    cte_query_blob_no_tag
    cte_query_local_poolquery
    cte_tag_construction
    cte_tag_putblob
    cte_tag_getblob
    cte_tag_errors
    cte_tag_metadata
    cte_tag_reorganize
    cte_tag_async
    cte_tag_large
    cte_tag_edge
    cte_functional_all
    cte_client_config_all
    cte_runtime_coverage_all
    cte_tiered_storage_all
    cte_tiered_dram_default_all
    cte_block_reuse_all
    cte_reorganize_all
    cte_data_organizer_all
    cte_io_emulation_all
    APPEND PROPERTY ENVIRONMENT "CLIO_PORT=10500"
)
# All cte_functional_* tests run test_core_functionality (9 test cases, each
# initializing and finalizing the full clio_run runtime).  After multiple
# runtime init/finalize cycles, ZMQ background threads fire callbacks on
# already-freed runtime memory, producing non-deterministic heap-use-after-free
# defects under ASan.  The same uninstrumented-runtime category caused all CTE
# tests to be msan_skip'd.  Skip the whole functional group under ASan.
#
# IMPORTANT: keep the "functional" label here. set_tests_properties(LABELS ...)
# REPLACES the label list, so this must repeat functional;core;cte (set above)
# or it clobbers them -- which previously dropped cte_functional_all out of the
# `ctest -LE functional` filter used by both the coverage gate
# (CI/calculate_coverage.sh) and the macOS workflow, letting this heavy
# functional test leak into those runs. On macOS it then SEGFAULTs
# intermittently via a coroutine_handle use-after-free in the worker
# (issue #485, the same UAF noted above). Restoring the label filters it out
# consistently, as originally intended.
set_tests_properties(
    cte_functional_all
    PROPERTIES LABELS "functional;core;cte;msan_skip;asan_skip"
)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Install Targets
# ------------------------------------------------------------------------------
set(_CTE_TEST_TARGETS cte_core_unit_tests cte_core_functionality_simple_tests test_core_functionality test_query test_cte_config_dpe test_tag_operations test_core_client_config test_core_runtime_coverage test_tiered_storage_stress test_tiered_storage_dram_default test_blob_block_reuse test_reorganize_blob test_evict test_data_organizer test_io_emulation)
# (The legacy test_cte_gpu_coroutine / test_gpu_core / test_gpu_create targets
# and their ctest registrations were removed along with the GPU runtime. CTE
# GPU coverage now lives in test_cte_devmem_putget (cte_devmem_putget_cuda).)
install(TARGETS ${_CTE_TEST_TARGETS}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# ------------------------------------------------------------------------------
# CTE PutBlob+GetBlob device-memory round-trip (CUDA / ROCm).
# Validates that the producer-only gpu2cpu path correctly handles tasks
# AND blob payload data that live entirely in kDeviceMem (no pinned
# host fallback).
# ------------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_CUDA)
  add_cuda_executable(test_cte_devmem_putget FALSE
    gpu/test_cte_devmem_putget.cc
    INCLUDE_DIRS
      ${CMAKE_SOURCE_DIR}/context-runtime/include
      ${CMAKE_SOURCE_DIR}/context-runtime/test
      ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
      ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    LINK_LIBS
      clio_run_cxx
      clio_run_cxx_gpu
      clio_admin_client
      clio_bdev_client
      clio_cte_core_client
      clio_cte_core_runtime
      ctp::cuda_cxx
      ${CMAKE_THREAD_LIBS_INIT}
  )
  set_target_properties(test_cte_devmem_putget PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(NAME cte_devmem_putget_cuda COMMAND test_cte_devmem_putget
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cte_devmem_putget_cuda PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1"
    TIMEOUT 120
    LABELS "gpu;cuda;cte;devmem;putblob"
  )
endif()

if(CLIO_CORE_ENABLE_ROCM)
  add_rocm_gpu_executable(test_cte_devmem_putget FALSE
    gpu/test_cte_devmem_putget.cc
  )
  target_compile_options(test_cte_devmem_putget PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code=arch=compute_89,code=sm_89>"
  )
  set_target_properties(test_cte_devmem_putget PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  target_include_directories(test_cte_devmem_putget PUBLIC
    ${CMAKE_SOURCE_DIR}/context-runtime/include
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
  )
  target_link_libraries(test_cte_devmem_putget PUBLIC
    clio_run_cxx
    clio_run_cxx_gpu
    clio_admin_client
    clio_bdev_client
    clio_cte_core_client
    clio_cte_core_runtime
    ctp::rocm_cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  add_test(NAME cte_devmem_putget_rocm COMMAND test_cte_devmem_putget
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cte_devmem_putget_rocm PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1"
    TIMEOUT 120
    LABELS "gpu;rocm;cte;devmem;putblob"
  )
endif()

# ------------------------------------------------------------------------------
# Note: Compression tests have been moved to compressor chimod tests
# See context-transfer-engine/compressor/test/ for compression functionality tests
# ------------------------------------------------------------------------------
# Add External Integration Tests
# ------------------------------------------------------------------------------
# External test demonstrates proper external linking patterns
# Temporarily commented out until ChiMod is installed and can be found as a package
# add_subdirectory(external)

# ------------------------------------------------------------------------------
# Add Adapter Unit Tests
# ------------------------------------------------------------------------------
add_subdirectory(adapters)
# ------------------------------------------------------------------------------
# Transaction log (WAL) unit tests — simple_test.h framework (NOT Catch2)
# ------------------------------------------------------------------------------
add_executable(test_cte_transaction_log
    test_cte_transaction_log.cc
)

target_link_libraries(test_cte_transaction_log
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio::run::admin_runtime      # Admin module runtime (required for runtime mode)
    clio::run::admin_client       # Admin module client
    clio::run::bdev_runtime       # Bdev module runtime (required for runtime mode)
    clio::run::bdev_client        # Bdev client for BdevType enum
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

add_test(NAME cte_transaction_log_all
    COMMAND test_cte_transaction_log)

set_tests_properties(
    cte_transaction_log_all
    PROPERTIES
        TIMEOUT 120
        # msan_skip: persists records through std::ofstream and reads them back
        # via std::ifstream (both in the uninstrumented libstdc++.so), and prints
        # counts through SimpleTest::print_summary (uninstrumented libstdc++
        # ostream). All reported reads are third-party false positives.
        LABELS "unit;core;cte;msan_skip"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
)

# ------------------------------------------------------------------------------
# GPU metadata cache unit tests — pure POD header, CPU-testable
# ------------------------------------------------------------------------------
add_executable(test_cte_shm_metadata_cache
    test_cte_shm_metadata_cache.cc
)

target_link_libraries(test_cte_shm_metadata_cache
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_shm_metadata_cache_all
    COMMAND test_cte_shm_metadata_cache)

set_tests_properties(
    cte_shm_metadata_cache_all
    PROPERTIES TIMEOUT 120
)

# ------------------------------------------------------------------------------
# clio-fs zero-IPC read path (issue #817): correctness + latency
#
# UNIX-only: the descriptor layer it drives is built on unistd.h / sys/stat.h /
# off_t and has no Windows port, so the filesystem client compiles it out there
# exist there.
# ------------------------------------------------------------------------------
if(UNIX)
add_executable(test_cfs_shm_read
    test_cfs_shm_read.cc
)

target_include_directories(test_cfs_shm_read PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
)

# Spawns its own clio_run daemon (RuntimeServer) and composes it from YAML, so
# the fast path is exercised across a real process boundary. A co-located
# runtime cannot prove that -- it shares the address space and hand-registers
# its targets, which is precisely how the production-dead locality check
# (issue #817) stayed invisible.
target_compile_definitions(test_cfs_shm_read PRIVATE
    CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")

target_link_libraries(test_cfs_shm_read
    clio_cte_filesystem_client  # descriptor layer + chimod client
    clio_cte_filesystem_client
    clio_cte_core_client
    # The client library's Config helpers are defined in the RUNTIME library,
    # so it has to come after the client on the link line: this binary calls
    # nothing in it directly, and --as-needed would otherwise drop it before
    # the client's undefined reference is even seen.
    clio_cte_core_runtime
    clio::run::admin_client
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# The co-located runtime dlopens the chimods from CLIO_REPO_PATH.
add_dependencies(test_cfs_shm_read
    clio_run
    clio_cte_core_runtime
    clio_cte_filesystem_runtime
    clio_bdev_runtime
    clio_admin_runtime)

add_test(NAME cfs_shm_read
    COMMAND test_cfs_shm_read
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# This test spawns a live daemon, so it inherits the macOS caveat from the CLI
# live-runtime tests (issue #482): the local ZMQ DEALER->ROUTER control plane
# does not deliver ClientConnect replies on the macOS runners, so drive the
# client over the unix-socket IPC transport there.
set(_cfs_shm_read_env
    "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1")
if(APPLE)
  string(APPEND _cfs_shm_read_env ";CLIO_IPC_MODE=ipc")
endif()

set_tests_properties(cfs_shm_read PROPERTIES
    ENVIRONMENT "${_cfs_shm_read_env}"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 180
    LABELS "unit;cte;cfs;msan_skip")
endif()  # UNIX

# Private-memory AsyncGetBlob (issue #823). UNIX-only because the "separate"
# configuration forks a real clio_run daemon (RuntimeServer / unistd.h), the
# same shape the cfs_shm_read test uses. The two ctest entries below run the
# SAME binary in the two runtime configurations the issue calls for: inline
# (co-located runtime -> direct write into private memory) and separate (pure
# client -> SHM staging + PostWait copy).
if(UNIX)
add_executable(test_getblob_priv
    test_getblob_priv.cc
)

target_include_directories(test_getblob_priv PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
)

target_compile_definitions(test_getblob_priv PRIVATE
    CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")

target_link_libraries(test_getblob_priv
    clio_cte_core_client
    # Config helpers used by the client live in the runtime library; keep it
    # after the client on the link line (see the cfs_shm_read note above).
    clio_cte_core_runtime
    clio::run::admin_client
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# The co-located (inline) runtime dlopens the chimods from CLIO_REPO_PATH; the
# separate configuration forks clio_run. Both need the runtime modules built.
add_dependencies(test_getblob_priv
    clio_run
    clio_cte_core_runtime
    clio_bdev_runtime
    clio_admin_runtime)

set(_getblob_priv_env
    "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1")
if(APPLE)
  string(APPEND _getblob_priv_env ";CLIO_IPC_MODE=ipc")
endif()

# Inline: this process starts the runtime co-located (direct-write path).
add_test(NAME cte_getblob_priv_inline
    COMMAND test_getblob_priv
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_tests_properties(cte_getblob_priv_inline PROPERTIES
    ENVIRONMENT "${_getblob_priv_env}"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 120
    LABELS "unit;core;cte;msan_skip")

# Separate: forks a real daemon and attaches as a pure client (staging path).
add_test(NAME cte_getblob_priv_separate
    COMMAND test_getblob_priv
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_tests_properties(cte_getblob_priv_separate PROPERTIES
    ENVIRONMENT "${_getblob_priv_env};CLIO_GETBLOB_PRIV_MODE=separate"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 180
    LABELS "unit;core;cte;msan_skip")

# Write-side mirror (issue #830): private-memory AsyncPutBlob. Same two
# configurations as the getblob_priv pair above — inline (co-located runtime,
# zero-copy FromRaw write) and separate (pure client, TASK_DATA_OWNER staging).
add_executable(test_putblob_priv
    test_putblob_priv.cc
)

target_include_directories(test_putblob_priv PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
)

target_compile_definitions(test_putblob_priv PRIVATE
    CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")

target_link_libraries(test_putblob_priv
    clio_cte_core_client
    clio_cte_core_runtime
    clio::run::admin_client
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_dependencies(test_putblob_priv
    clio_run
    clio_cte_core_runtime
    clio_bdev_runtime
    clio_admin_runtime)

add_test(NAME cte_putblob_priv_inline
    COMMAND test_putblob_priv
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_tests_properties(cte_putblob_priv_inline PROPERTIES
    ENVIRONMENT "${_getblob_priv_env}"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 120
    LABELS "unit;core;cte;msan_skip")

add_test(NAME cte_putblob_priv_separate
    COMMAND test_putblob_priv
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_tests_properties(cte_putblob_priv_separate PROPERTIES
    ENVIRONMENT "${_getblob_priv_env};CLIO_PUTBLOB_PRIV_MODE=separate"
    RESOURCE_LOCK "clio_runtime_port"
    TIMEOUT 180
    LABELS "unit;core;cte;msan_skip")
endif()  # UNIX

add_executable(test_cte_gpu_metadata_cache
    test_cte_gpu_metadata_cache.cc
)

target_link_libraries(test_cte_gpu_metadata_cache
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_gpu_metadata_cache_all
    COMMAND test_cte_gpu_metadata_cache)

set_tests_properties(
    cte_gpu_metadata_cache_all
    PROPERTIES
        TIMEOUT 120
        # msan_skip: prints test counts through SimpleTest::print_summary, whose
        # std::ostream number formatting lives in the uninstrumented libstdc++.so
        # (std::num_put / std::ctype widening) — unfixable third-party false
        # positives.
        LABELS "unit;core;cte;msan_skip"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
)

# ------------------------------------------------------------------------------
# Config error-path unit tests — simple_test.h framework (NOT Catch2)
# ------------------------------------------------------------------------------
add_executable(test_cte_config_errors
    test_cte_config_errors.cc
)

target_link_libraries(test_cte_config_errors
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_config_errors_all
    COMMAND test_cte_config_errors)

set_tests_properties(
    cte_config_errors_all
    PROPERTIES
        TIMEOUT 120
        # msan_skip: exercises config validation that reads files via
        # std::ifstream and parses with std::isspace/std::tolower (locale facets),
        # all in the uninstrumented libstdc++.so, then prints counts through
        # SimpleTest::print_summary. Reported reads are third-party false
        # positives.
        LABELS "unit;core;cte;msan_skip"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
)

# ------------------------------------------------------------------------------
# CTE binds an ALREADY-EXISTING pool (a safe-bdev pool) as a storage target.
# Composes a RAM bdev -> safe-bdev over it -> CTE target bound via
# attach_existing=1, then PutBlob/GetBlob round-trips through the safe-bdev pool.
# Links the safe-bdev runtime+client so the safe_bdev chimod is available
# in-process (mirrors how the other CTE tests link bdev_runtime/bdev_client).
# ------------------------------------------------------------------------------
# Requires the safe-bdev module (WIP, see context-runtime/modules/CMakeLists.txt).
if(TARGET clio_safe_bdev_runtime)
add_executable(test_cte_existing_pool
    test_cte_existing_pool.cc
)

target_link_libraries(test_cte_existing_pool
    clio_cte_core_runtime         # CTE core runtime library
    clio_cte_core_client          # CTE core client library
    clio_admin_runtime      # Admin module runtime (required for runtime mode)
    clio_admin_client       # Admin module client
    clio_bdev_runtime       # Bdev module runtime (member bdev)
    clio_bdev_client        # Bdev client for BdevType enum
    clio_safe_bdev_runtime       # safe-bdev runtime (existing pool target)
    clio_safe_bdev_client        # safe-bdev client (compose the pool)
    ctp::cxx                    # ClioCtp library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

add_test(NAME cte_existing_pool_all
    COMMAND test_cte_existing_pool)

set_tests_properties(
    cte_existing_pool_all
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;core;cte;safe_bdev;msan_skip"
        RESOURCE_LOCK "chimaera_runtime"
        FIXTURES_REQUIRED "chimaera_test_cleanup_fixture"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHI_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
)

install(TARGETS test_cte_existing_pool
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)


# ------------------------------------------------------------------------------
# Safe-BDEV recovery stress test: 4 file bdevs + 1 RAM bdev under safe-bdev
# (max_failures=1) bound to CTE. Injects a bdev fault and verifies CTE can
# still read data back via EC reconstruction.
# ------------------------------------------------------------------------------
add_executable(test_safe_bdev_recovery_e2e
    test_safe_bdev_recovery_e2e.cc
)

target_link_libraries(test_safe_bdev_recovery_e2e
    clio_cte_core_runtime
    clio_cte_core_client
    clio_admin_runtime
    clio_admin_client
    clio_bdev_runtime
    clio_bdev_client
    clio_safe_bdev_runtime
    clio_safe_bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_safe_bdev_recovery_e2e
    COMMAND test_safe_bdev_recovery_e2e)

set_tests_properties(
    cte_safe_bdev_recovery_e2e
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;core;cte;safe_bdev;recovery;msan_skip"
        RESOURCE_LOCK "chimaera_runtime"
        FIXTURES_REQUIRED "chimaera_test_cleanup_fixture"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHI_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
)

install(TARGETS test_safe_bdev_recovery_e2e
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

endif()  # TARGET clio_safe_bdev_runtime

# Batched POD paging: PodMultiPutBlob / PodMultiGetBlob / PodMultiScore.
# Needs the compressor client for the interposed-batch case.
if(TARGET clio_cte_compressor_client)
add_executable(test_pod_multi_paging test_pod_multi_paging.cc)
target_include_directories(test_pod_multi_paging PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
)
target_link_libraries(test_pod_multi_paging
    clio_cte_core_runtime
    clio_cte_core_client
    clio_cte_compressor_runtime
    clio_cte_compressor_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME cte_pod_multi_paging
    COMMAND test_pod_multi_paging)

set_tests_properties(
    cte_pod_multi_paging
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;cte;pod"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)
endif()
