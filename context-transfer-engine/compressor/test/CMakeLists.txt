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

# Register functional tests - use test names (simple_test.h filters by name)
add_test(NAME compressor_compress_lz4
    COMMAND test_compressor_functional "CompressTask - LZ4 Compression")

add_test(NAME compressor_decompress_lz4
    COMMAND test_compressor_functional "DecompressTask - LZ4 Decompression")

add_test(NAME compressor_roundtrip_integrity
    COMMAND test_compressor_functional "Round-trip - Data Integrity")

add_test(NAME compressor_compression_ratios
    COMMAND test_compressor_functional "Compression Ratio - Data Patterns")

add_test(NAME compressor_dynamic_schedule
    COMMAND test_compressor_functional "DynamicSchedule - Compression Selection")

add_test(NAME compressor_multiple_libraries
    COMMAND test_compressor_functional "Multiple Libraries - Compression")

add_test(NAME compressor_error_handling
    COMMAND test_compressor_functional "Error Handling - Invalid Parameters")

# Set test properties
set_tests_properties(
    compressor_compress_lz4
    compressor_decompress_lz4
    compressor_roundtrip_integrity
    compressor_compression_ratios
    compressor_dynamic_schedule
    compressor_multiple_libraries
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

add_test(NAME compressor_transparent_putblob
    COMMAND test_transparent_compress "Transparent PutBlob")

add_test(NAME compressor_transparent_getblob
    COMMAND test_transparent_compress "Transparent GetBlob")

set_tests_properties(
    compressor_transparent_putblob
    compressor_transparent_getblob
    PROPERTIES
        TIMEOUT 120
        LABELS "compressor;transparent;integration;msan_skip"
        RESOURCE_LOCK "clio_runtime"
)

install(TARGETS test_transparent_compress RUNTIME DESTINATION bin)

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

add_test(NAME adios2_transparent_compress_put
    COMMAND test_adios2_transparent_compress "ADIOS2 transparent compress - PutBlob")

add_test(NAME adios2_transparent_compress_get
    COMMAND test_adios2_transparent_compress "ADIOS2 transparent compress - GetBlob")

add_test(NAME adios2_transparent_compress_multistep
    COMMAND test_adios2_transparent_compress "ADIOS2 transparent compress - multi-step")

set_tests_properties(
    adios2_transparent_compress_put
    adios2_transparent_compress_get
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
