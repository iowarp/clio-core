cmake_minimum_required(VERSION 3.10)

#------------------------------------------------------------------------------
# Compressor unit tests
#------------------------------------------------------------------------------

# Test distribution classifier
add_executable(test_distribution_classifier_exec
  test_distribution_classifier.cc
)
target_link_libraries(test_distribution_classifier_exec
  clio::cte::compressor_runtime
  ctp::compress
  Catch2::Catch2WithMain
)
add_test(NAME test_distribution_classifier COMMAND test_distribution_classifier_exec)

# Test linear regression table predictor
add_executable(test_linreg_table_predictor_exec
  test_linreg_table_predictor.cc
)
target_link_libraries(test_linreg_table_predictor_exec
  clio::cte::compressor_runtime
  ctp::compress
  Catch2::Catch2WithMain
)
add_test(NAME test_linreg_table_predictor COMMAND test_linreg_table_predictor_exec)

# Test Q-table predictor
add_executable(test_qtable_predictor_exec
  test_qtable_predictor.cc
)
target_link_libraries(test_qtable_predictor_exec
  clio::cte::compressor_runtime
  ctp::compress
  Catch2::Catch2WithMain
)
add_test(NAME test_qtable_predictor COMMAND test_qtable_predictor_exec)

# Compose-YAML configuration (issue #693). Pure parse test: no runtime, no
# GPU, no pool -- CompressorConfig::LoadConfig against the entries a pipeline
# actually writes. Guards the invariant that every serialized field stays
# reachable from YAML, which is how five model paths came to be silently
# discarded.
add_executable(test_compressor_compose_config_exec
  test_compressor_compose_config.cc
)
target_link_libraries(test_compressor_compose_config_exec
  clio::cte::compressor_runtime
  ctp::compress
  Catch2::Catch2WithMain
)
add_test(NAME test_compressor_compose_config
         COMMAND test_compressor_compose_config_exec)
set_tests_properties(test_compressor_compose_config PROPERTIES
  LABELS "693;compressor;config;unit")

# Test NeuroPress bridge (issue #693 cycle 3): NeuroPressCandidateStats
# ranking GPU/NeuroPress candidates for dynamic selection.
add_executable(test_neuropress_bridge_exec
  test_neuropress_bridge.cc
)
target_link_libraries(test_neuropress_bridge_exec
  clio::cte::compressor_runtime
  ctp::compress
  ctp::compress_model
  Catch2::Catch2WithMain
)
target_include_directories(test_neuropress_bridge_exec PRIVATE
  ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
# Real, shipped NeuroPress .nnwt weights (see context-transport-primitives/
# src/compress/model/weights/) -- exercises the actual pretrained model.
target_compile_definitions(test_neuropress_bridge_exec PRIVATE
  CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")
add_test(NAME test_neuropress_bridge COMMAND test_neuropress_bridge_exec)

# NOTE: test_compression_contention has been moved to the end of this file
# as a standalone benchmark without ChiMod runtime dependency

#------------------------------------------------------------------------------
# Compressor functional tests - test actual chimod functionality
#------------------------------------------------------------------------------

# Test compressor chimod functionality (Compress, Decompress, DynamicSchedule tasks)
add_executable(test_compressor_functional
  test_compressor_functional.cc
)


# Link against compressor runtime and dependencies
target_link_libraries(test_compressor_functional
  clio::cte::compressor_runtime
  clio::cte::compressor_client
  clio::cte::core_runtime         # For Context type
  clio::cte::core_client          # For core client if needed
  clio::run::admin_runtime       # For CLIO Runtime
  clio::run::admin_client        # For admin client
  clio::run::bdev_runtime        # For bdev runtime
  clio::run::bdev_client         # For bdev client
  ctp::compress                # For compression libraries
  ctp::cxx                     # For ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}     # Threading support
)

# "Static Compress - real GPU device pointer as input" stages a genuine
# cudaMalloc'd buffer via ctp::DeviceAwareMemcpy/IsDevicePointer. Without
# ctp::cuda_cxx (which carries CTP_ENABLE_CUDA=1 plus the real CUDA
# runtime), gpu_api.h's device-aware helpers compile down to their
# CPU-only fallback (IsDevicePointer always false, DeviceAwareMemcpy always
# plain std::memcpy) -- which segfaults on a real device pointer instead of
# actually exercising the staging path, same class of gap the VOL adapter
# had before its own GPU-pointer fix.
if(CLIO_CORE_ENABLE_CUDA)
    target_link_libraries(test_compressor_functional ctp::cuda_cxx)
elseif(CLIO_CORE_ENABLE_ROCM)
    target_link_libraries(test_compressor_functional ctp::rocm_cxx)
endif()

target_compile_definitions(test_compressor_functional PRIVATE
  CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

# Register functional tests - use test names (simple_test.h filters by name).
#
# simple_test.h's TEST_CASE registers under "<title> <tags>"; a plain-text
# filter (no leading '[') is a substring match against that WHOLE string, and
# a filter that doesn't match anything runs 0 tests and exits 0 -- a silent
# vacuous PASS, not a build/test failure ctest would ever flag. Six of these
# eight registrations had drifted from the actual TEST_CASE titles below them
# (renamed at some point without updating this file) and had been running
# zero assertions ever since; only compressor_error_handling's filter still
# matched for real, which is exactly the one that was already known to hang
# (see the class comment on "Error Handling - Invalid Parameters" above).
# Verified against the current titles in test_compressor_functional.cc.
add_test(NAME compressor_compress_lz4
    COMMAND test_compressor_functional "Basic Compress and Store")

add_test(NAME compressor_decompress_lz4
    COMMAND test_compressor_functional "Decompress and Retrieve")

# No standalone "round-trip" or "compression ratio" TEST_CASE exists (and
# none appears to ever have, going by the titles actually defined in this
# file) -- point these at the closest real coverage rather than continue
# matching nothing.
add_test(NAME compressor_roundtrip_integrity
    COMMAND test_compressor_functional "[basic]")

add_test(NAME compressor_compression_ratios
    COMMAND test_compressor_functional "[libraries]")

add_test(NAME compressor_dynamic_schedule
    COMMAND test_compressor_functional "Dynamic Schedule Compression")

add_test(NAME compressor_dynamic_neuropress
    COMMAND test_compressor_functional "[neuropress][693]")

# Exploration gets its own entry as well as being swept up by the tag filter
# above: it is the only case that exercises the K-way search, the
# winner-adoption path that RE-STORES a blob with a different codec's bytes,
# and the Train() samples exploration produces. Worth failing under its own
# name rather than inside a broader bucket.
add_test(NAME compressor_exploration
    COMMAND test_compressor_functional "[exploration]")

# Residency invariance: the same bytes must select the same algorithm whether
# they arrive in CPU shared memory or a registered device backend. The two
# routes compute the three selection features with DIFFERENT code (host
# DataStatisticsFactory vs the device stats kernels), and the parity suite
# cannot catch a divergence between them because it feeds both sides the same
# bytes rather than driving the two residencies.
add_test(NAME compressor_residency_invariance
    COMMAND test_compressor_functional "[residency]")

add_test(NAME compressor_nvcomp_gpu_roundtrip
    COMMAND test_compressor_functional "[nvcomp][gpu]")

add_test(NAME compressor_multiple_libraries
    COMMAND test_compressor_functional "Multiple Compression Libraries")

# Never registered at all (a gap, not a rename drift like the others above).
add_test(NAME compressor_no_compression_passthrough
    COMMAND test_compressor_functional "No Compression Passthrough")

add_test(NAME compressor_error_handling
    COMMAND test_compressor_functional "Error Handling - Invalid Parameters")

# Static selection of every GPU compressor at 1 GiB (nvcomp's 8 + the
# explicit-selection-only cuSZ/ndzip/cuSZp) -- a much heavier run than the
# other GPU tests, so it gets its own longer TIMEOUT below rather than
# sharing the 300s default.
add_test(NAME compressor_gpu_1gb_sweep
    COMMAND test_compressor_functional "[1gb]")
set_tests_properties(compressor_gpu_1gb_sweep PROPERTIES
    TIMEOUT 900
    LABELS "compressor;functional;chimod;gpu")

# Static (explicit compress_lib_) compression with a REAL cudaMalloc'd
# device pointer as the source -- see the TEST_CASE comment for why none
# of the tests above (including the 1GiB sweep) actually cover this.
add_test(NAME compressor_static_gpu_device_ptr
    COMMAND test_compressor_functional "[gpu][static][693]")
set_tests_properties(compressor_static_gpu_device_ptr PROPERTIES
    TIMEOUT 300
    LABELS "compressor;functional;chimod;gpu;693")

# Set test properties
set_tests_properties(
    compressor_compress_lz4
    compressor_decompress_lz4
    compressor_roundtrip_integrity
    compressor_compression_ratios
    compressor_dynamic_schedule
    compressor_dynamic_neuropress
    compressor_nvcomp_gpu_roundtrip
    compressor_multiple_libraries
    compressor_no_compression_passthrough
    compressor_error_handling
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout
        LABELS "compressor;functional;chimod"
)

# Install functional test
install(TARGETS test_compressor_functional
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

#------------------------------------------------------------------------------
# Transparent Compression Integration Test
#------------------------------------------------------------------------------

add_executable(test_transparent_compress
  test_transparent_compress.cc
)

target_link_libraries(test_transparent_compress
  clio::cte::compressor_runtime
  clio::cte::compressor_client
  clio::cte::core_runtime
  clio::cte::core_client
  clio::run::admin_runtime
  clio::run::admin_client
  clio::run::bdev_runtime
  clio::run::bdev_client
  ctp::compress
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

# Copy compose config next to binary
add_custom_command(TARGET test_transparent_compress POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/test_transparent_compress_config.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/test_transparent_compress_config.yaml
)

# One entry, both cases, in declaration order. The GetBlob case reads the
# blob the PutBlob case writes ("the tag we created in the previous test"),
# so they are an ordered pair by design -- but as separate ctest entries they
# ran in separate PROCESSES with the runtime cleanup fixtures firing in
# between, so GetBlob always started against a wiped runtime and could never
# find the blob. It failed on every run for that reason alone, with nothing
# wrong in the compressor. Catch2 takes multiple test specs and runs them in
# declaration order, which keeps the pair in one process.
add_test(NAME compressor_transparent_roundtrip
    COMMAND test_transparent_compress "Transparent PutBlob" "Transparent GetBlob")

set_tests_properties(
    compressor_transparent_roundtrip
    PROPERTIES
        TIMEOUT 120
        LABELS "compressor;transparent;integration;msan_skip"
        RESOURCE_LOCK "clio_runtime"
)

install(TARGETS test_transparent_compress RUNTIME DESTINATION bin)

#------------------------------------------------------------------------------
# NeuroPress-via-compose integration test (issue #693)
#------------------------------------------------------------------------------
# Proves a compressor composed from YAML actually loads the NeuroPress model
# and selects with it. Every other NeuroPress test and example sets
# neuropress_model_path_ programmatically, so nothing covered the config-file
# route -- the one a deployment uses, and the one that was broken.
add_executable(test_neuropress_compose
  test_neuropress_compose.cc
)

target_link_libraries(test_neuropress_compose
  clio::cte::compressor_runtime
  clio::cte::compressor_client
  clio::cte::core_runtime
  clio::cte::core_client
  clio::run::admin_runtime
  clio::run::admin_client
  clio::run::bdev_runtime
  clio::run::bdev_client
  ctp::compress
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_definitions(test_neuropress_compose PRIVATE
  CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

# One entry, both cases in declaration order: the round-trip case reads the
# blob the selection case writes, so separate ctest entries would run in
# separate processes against a wiped runtime and the read could never find it.
#
# A SINGLE filter, matching both by tag. simple_test.h's SIMPLE_TEST_MAIN
# reads argv[1] only, so passing one spec per case silently runs just the
# first -- which is what compressor_transparent_roundtrip above does today,
# leaving its GetBlob half dead. Filtering on the shared [compose] tag keeps
# both in one process and in declaration order.
add_test(NAME compressor_neuropress_compose
    COMMAND test_neuropress_compose "[compose]")

set_tests_properties(
    compressor_neuropress_compose
    PROPERTIES
        TIMEOUT 180
        LABELS "693;compressor;neuropress;compose;integration;gpu;msan_skip"
        RESOURCE_LOCK "clio_runtime"
)

#------------------------------------------------------------------------------
# ADIOS2-style Transparent Compression Test (direct CTE client API, no Tag)
#------------------------------------------------------------------------------

add_executable(test_adios2_transparent_compress
  test_adios2_transparent_compress.cc
)

target_link_libraries(test_adios2_transparent_compress
  clio::cte::compressor_runtime
  clio::cte::compressor_client
  clio::cte::core_runtime
  clio::cte::core_client
  clio::run::admin_runtime
  clio::run::admin_client
  clio::run::bdev_runtime
  clio::run::bdev_client
  ctp::compress
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

# Same ordered-pair rule as compressor_transparent_roundtrip above: the
# GetBlob case reads what the PutBlob case wrote, so splitting them across
# processes guaranteed a failure. multi-step below is self-contained and
# stays its own entry.
add_test(NAME adios2_transparent_compress_roundtrip
    COMMAND test_adios2_transparent_compress
            "ADIOS2 transparent compress - PutBlob"
            "ADIOS2 transparent compress - GetBlob")

add_test(NAME adios2_transparent_compress_multistep
    COMMAND test_adios2_transparent_compress "ADIOS2 transparent compress - multi-step")

set_tests_properties(
    adios2_transparent_compress_roundtrip
    adios2_transparent_compress_multistep
    PROPERTIES
        TIMEOUT 120
        LABELS "compressor;adios2;transparent;integration;msan_skip"
        RESOURCE_LOCK "clio_runtime"
)

install(TARGETS test_adios2_transparent_compress RUNTIME DESTINATION bin)

#------------------------------------------------------------------------------
# CTE Compression Benchmark
#------------------------------------------------------------------------------

# Create clio_cte_compress_bench executable
add_executable(clio_cte_compress_bench
    clio_cte_compress_bench.cc
)

# Add compression compile definition
target_compile_definitions(clio_cte_compress_bench PRIVATE CLIO_CORE_ENABLE_COMPRESS)

# Link required libraries
target_link_libraries(clio_cte_compress_bench
    clio::cte::core_client          # CTE core client library
    clio::cte::core_runtime         # CTE core runtime (for compression context)
    clio::run::cxx                 # Core CLIO Runtime library
    ctp::compress                # Compression library
    ${CMAKE_THREAD_LIBS_INIT}     # Threading support (for pthread_setaffinity_np)
)

# Set C++ standard
target_compile_features(clio_cte_compress_bench PRIVATE cxx_std_17)

# Install the compression benchmark executable
install(TARGETS clio_cte_compress_bench
    RUNTIME DESTINATION bin
)

#------------------------------------------------------------------------------
# Compression Contention Benchmark (standalone, no ChiMod runtime dependency)
#------------------------------------------------------------------------------

# Enable the contention benchmark (no longer has Singleton issues)
add_executable(test_compression_contention_exec
    test_compression_contention.cc
)
target_link_libraries(test_compression_contention_exec
    ctp::compress
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(test_compression_contention_exec PRIVATE cxx_std_17)
add_test(NAME test_compression_contention COMMAND test_compression_contention_exec
    --chunk-size-kb 4
    --sim-threads 1
    --compress-threads 1
    --queue-depth 4
    --busy-time 0
    --num-outputs 2)
install(TARGETS test_compression_contention_exec
    RUNTIME DESTINATION bin
)

#------------------------------------------------------------------------------
# Compression Benchmark (unified benchmark for all compression libraries)
#------------------------------------------------------------------------------

add_executable(test_compression_benchmark_exec
    test_compression_benchmark.cc
)
target_include_directories(test_compression_benchmark_exec PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transport-primitives/test/unit  # For basic_test.h
)
target_link_libraries(test_compression_benchmark_exec
    clio::cte::compressor_runtime
    ctp::compress
    ctp::cxx
    Catch2::Catch2WithMain
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(test_compression_benchmark_exec PRIVATE cxx_std_17)
add_test(NAME test_compression_benchmark COMMAND test_compression_benchmark_exec)
install(TARGETS test_compression_benchmark_exec
    RUNTIME DESTINATION bin
)

#------------------------------------------------------------------------------
# Feature-Targeted Data Generator (standalone test)
#------------------------------------------------------------------------------

# Commented out - source file missing
# add_executable(test_feature_targeted_generator_exec
#     test_feature_targeted_generator.cc
# )
#
# target_compile_features(test_feature_targeted_generator_exec PRIVATE cxx_std_17)
#
# install(TARGETS test_feature_targeted_generator_exec
#     RUNTIME DESTINATION bin
# )

# message(STATUS "Feature-targeted data generator enabled")

#------------------------------------------------------------------------------
# Synthetic Workload Generator (MPI-based benchmark)
#------------------------------------------------------------------------------

# Only build if MPI is available
find_package(MPI QUIET)
if(MPI_FOUND)
    add_executable(synthetic_workload_exec
        synthetic_workload.cc
    )

    target_include_directories(synthetic_workload_exec PRIVATE
        ${MPI_CXX_INCLUDE_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/../generator
    )

    target_link_libraries(synthetic_workload_exec
        clio::cte::compressor_client
        clio::cte::core_client
        clio::cte::core_runtime
        clio::run::cxx
        ctp::compress
        ctp::cxx
        ${MPI_CXX_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    target_compile_features(synthetic_workload_exec PRIVATE cxx_std_17)

    # Add MPI compile flags
    if(MPI_CXX_COMPILE_FLAGS)
        target_compile_options(synthetic_workload_exec PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    endif()

    # Add MPI link flags
    if(MPI_CXX_LINK_FLAGS)
        set_target_properties(synthetic_workload_exec PROPERTIES
            LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
        )
    endif()

    install(TARGETS synthetic_workload_exec
        RUNTIME DESTINATION bin
    )

    message(STATUS "Synthetic workload generator enabled (MPI found)")
else()
    message(STATUS "Synthetic workload generator disabled (MPI not found)")
endif()

#------------------------------------------------------------------------------
# Exploration must store exactly what the adopted codec needs (issue #693)
#------------------------------------------------------------------------------
# Guards the invariant that a rejected candidate's footprint is not left
# allocated: PutBlob overwrites bytes but does not shorten a longer blob, so
# adopting a winner smaller than the primary used to leave 4-9x the needed
# space behind and made every ratio read off the tier wrong by that factor.
add_executable(test_explore_single_write
  test_explore_single_write.cc
)

target_link_libraries(test_explore_single_write
  clio::cte::compressor_runtime
  clio::cte::compressor_client
  clio::cte::core_runtime
  clio::cte::core_client
  clio::run::admin_runtime
  clio::run::admin_client
  clio::run::bdev_runtime
  clio::run::bdev_client
  ctp::compress
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

target_include_directories(test_explore_single_write PRIVATE
  ${CMAKE_SOURCE_DIR}/context-runtime/test)   # simple_test.h

target_compile_definitions(test_explore_single_write PRIVATE
  CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/")

add_test(NAME compressor_explore_single_write
    COMMAND test_explore_single_write "[explore]")

set_tests_properties(
    compressor_explore_single_write
    PROPERTIES
        TIMEOUT 180
        LABELS "693;compressor;neuropress;explore;integration;gpu;msan_skip"
        RESOURCE_LOCK "clio_runtime"
)
