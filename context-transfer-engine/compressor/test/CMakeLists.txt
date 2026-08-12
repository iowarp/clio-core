cmake_minimum_required(VERSION 3.10)

#------------------------------------------------------------------------------
# Compressor unit tests
#------------------------------------------------------------------------------

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

# Stage the functional tests' compose config next to the binary.
add_custom_command(TARGET test_compressor_functional POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compressor_functional_config.yaml
    ${CMAKE_BINARY_DIR}/bin/test_compressor_functional_config.yaml
)

# ONE entry running every case in order, in a single process.
#
# Two things forced this. simple_test.h filters by name, and six of the seven
# filters here named tests that do not exist -- those entries ran nothing and
# passed vacuously. And "Decompress and Retrieve" reads the blob that "Basic
# Compress and Store" wrote, so splitting them across ctest entries puts them
# in separate PROCESSES with separate runtimes and the blob is gone. Running
# the binary unfiltered executes the cases in order against one runtime, which
# is what they were written for.
add_test(NAME compressor_functional
    COMMAND test_compressor_functional
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set_tests_properties(
    compressor_functional
    PROPERTIES
        TIMEOUT 300
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

# ONE test, running both cases in a single process.
#
# They were two ctest entries invoking this same binary with different name
# filters, i.e. two separate PROCESSES. The GetBlob case reads the blob "we
# created in the previous test", but that blob lives in the runtime the
# PutBlob process started and took with it, so GetBlob could only ever fail
# with a non-zero return code. Running the binary unfiltered executes both
# cases in order against one runtime, which is what the test was written for.
add_test(NAME compressor_transparent
    COMMAND test_transparent_compress)

set_tests_properties(
    compressor_transparent
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

