cmake_minimum_required(VERSION 3.10)

#------------------------------------------------------------------------------
# Compressor unit tests
#------------------------------------------------------------------------------

# Test distribution classifier
add_executable(test_distribution_classifier_exec
  test_distribution_classifier.cc
)
target_link_libraries(test_distribution_classifier_exec
  wrp_cte::compressor_runtime
  hshm::compress
  Catch2::Catch2WithMain
)
add_test(NAME test_distribution_classifier COMMAND test_distribution_classifier_exec)

# Test linear regression table predictor
add_executable(test_linreg_table_predictor_exec
  test_linreg_table_predictor.cc
)
target_link_libraries(test_linreg_table_predictor_exec
  wrp_cte::compressor_runtime
  hshm::compress
  Catch2::Catch2WithMain
)
add_test(NAME test_linreg_table_predictor COMMAND test_linreg_table_predictor_exec)

# Test Q-table predictor
add_executable(test_qtable_predictor_exec
  test_qtable_predictor.cc
)
target_link_libraries(test_qtable_predictor_exec
  wrp_cte::compressor_runtime
  hshm::compress
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

# Include simple_test.h framework
target_include_directories(test_compressor_functional PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

# Link against compressor runtime and dependencies
target_link_libraries(test_compressor_functional
  wrp_cte::compressor_runtime
  wrp_cte::compressor_client
  wrp_cte::core_runtime         # For Context type
  wrp_cte::core_client          # For core client if needed
  chimaera::admin_runtime       # For Chimaera runtime
  chimaera::admin_client        # For admin client
  chimaera::bdev_runtime        # For bdev runtime
  chimaera::bdev_client         # For bdev client
  hshm::compress                # For compression libraries
  hshm::cxx                     # For HermesShm library
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
# CTE Compression Benchmark
#------------------------------------------------------------------------------

# Create wrp_cte_compress_bench executable
add_executable(wrp_cte_compress_bench
    wrp_cte_compress_bench.cc
)

# Add compression compile definition
target_compile_definitions(wrp_cte_compress_bench PRIVATE WRP_CORE_ENABLE_COMPRESS)

# Link required libraries
target_link_libraries(wrp_cte_compress_bench
    wrp_cte::core_client          # CTE core client library
    wrp_cte::core_runtime         # CTE core runtime (for compression context)
    chimaera::cxx                 # Core Chimaera library
    hshm::compress                # Compression library
    ${CMAKE_THREAD_LIBS_INIT}     # Threading support (for pthread_setaffinity_np)
)

# Set C++ standard
target_compile_features(wrp_cte_compress_bench PRIVATE cxx_std_17)

# Install the compression benchmark executable
install(TARGETS wrp_cte_compress_bench
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
    hshm::compress
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(test_compression_contention_exec PRIVATE cxx_std_17)
add_test(NAME test_compression_contention COMMAND test_compression_contention_exec)
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
    wrp_cte::compressor_runtime
    hshm::compress
    hshm::cxx
    Catch2::Catch2WithMain
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(test_compression_benchmark_exec PRIVATE cxx_std_17)
add_test(NAME test_compression_benchmark COMMAND test_compression_benchmark_exec)
install(TARGETS test_compression_benchmark_exec
    RUNTIME DESTINATION bin
)

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
    )

    target_link_libraries(synthetic_workload_exec
        wrp_cte::compressor_client
        wrp_cte::core_client
        wrp_cte::core_runtime
        chimaera::cxx
        hshm::compress
        hshm::cxx
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
