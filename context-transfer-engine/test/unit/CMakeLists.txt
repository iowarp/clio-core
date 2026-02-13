# ------------------------------------------------------------------------------
# CTE Core Unit Tests
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.10)

# Note: Binary output directory is configured in root CMakeLists.txt to use build/bin
# The test executable will be placed in build/bin directory due to global
# CMAKE_RUNTIME_OUTPUT_DIRECTORY setting

# Find required test dependencies
# Catch2 is already found by root CMakeLists.txt

# Create the test executable
add_executable(cte_core_unit_tests
    test_core_minimal.cc
    test_core_functionality_simple.cc
)

# Create separate executable for config/DPE tests (avoids TEST_CASE line number collisions)
add_executable(test_cte_config_dpe
    test_cte_config_dpe.cc
)

# Create single version of test_core_functionality that uses environment variable
# CHIMAERA_WITH_RUNTIME to control whether runtime is initialized (default: yes)
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

# Create test_reorganize_blob executable for ReorganizeBlob API tests
add_executable(test_reorganize_blob
    test_reorganize_blob.cc
)

# Include directories for all tests - using unified simple_test.h framework
target_include_directories(cte_core_unit_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

target_include_directories(test_core_functionality PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

target_include_directories(test_query PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

target_include_directories(test_cte_config_dpe PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

target_include_directories(test_tag_operations PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

target_include_directories(test_core_client_config PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

target_include_directories(test_core_runtime_coverage PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

target_include_directories(test_tiered_storage_stress PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

target_include_directories(test_reorganize_blob PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
)

# Link against required libraries using simple_test.h framework (NOT Catch2)
target_link_libraries(cte_core_unit_tests
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client for BdevType enum
    hshm::cxx                    # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_core_functionality - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_core_functionality
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client for BdevType enum
    hshm::cxx                    # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_cte_config_dpe - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_cte_config_dpe
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client for BdevType enum
    hshm::cxx                    # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_query - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_query
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client for BdevType enum
    hshm::cxx                    # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_tag_operations - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_tag_operations
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client for BdevType enum
    hshm::cxx                    # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_core_client_config - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_core_client_config
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client for BdevType enum
    hshm::cxx                    # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_core_runtime_coverage - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_core_runtime_coverage
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client for BdevType enum
    hshm::cxx                    # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_tiered_storage_stress - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_tiered_storage_stress
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client for BdevType enum
    hshm::cxx                    # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Link test_reorganize_blob - using simple_test.h framework (NOT Catch2)
target_link_libraries(test_reorganize_blob
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client for BdevType enum
    hshm::cxx                    # HermesShm library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Note: Include directories are provided automatically via linked ChiMod targets
# wrp_cte_core_client provides ${WRP_CTE_ROOT}/core/include via transitive includes

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
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_params
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_target_config
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_tag_info
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_blob_info
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_tasks
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_helpers
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_workflow
    COMMAND cte_core_unit_tests)

add_test(NAME cte_core_performance
    COMMAND cte_core_unit_tests)

# CTE Config and DPE tests (separate executable)
add_test(NAME cte_config_tests
    COMMAND test_cte_config_dpe "[cte][config]")
add_test(NAME cte_dpe_tests
    COMMAND test_cte_config_dpe "[cte][dpe]")

# Add test_core_functionality tests
add_test(NAME cte_functional_pool_creation
    COMMAND test_core_functionality)

add_test(NAME cte_functional_target_registration
    COMMAND test_core_functionality)

add_test(NAME cte_functional_putblob
    COMMAND test_core_functionality)

add_test(NAME cte_functional_getblob
    COMMAND test_core_functionality)

add_test(NAME cte_functional_putget_integration
    COMMAND test_core_functionality)

add_test(NAME cte_functional_putget_comprehensive
    COMMAND test_core_functionality)

add_test(NAME cte_functional_reorganize
    COMMAND test_core_functionality)

add_test(NAME cte_functional_e2e_workflow
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

# Add test_core_client_config tests - comprehensive Client and Config API coverage tests
add_test(NAME cte_client_config_default
    COMMAND test_core_client_config "Config - Default")

add_test(NAME cte_client_config_file
    COMMAND test_core_client_config "Config - Load from Valid YAML File")

add_test(NAME cte_client_config_invalid_file
    COMMAND test_core_client_config "Config - Load from Invalid File Path")

add_test(NAME cte_client_config_empty_path
    COMMAND test_core_client_config "Config - Load from Empty Path")

add_test(NAME cte_client_config_string
    COMMAND test_core_client_config "Config - Load from Valid YAML String")

add_test(NAME cte_client_config_empty_string
    COMMAND test_core_client_config "Config - Load from Empty String")

add_test(NAME cte_client_config_invalid_yaml
    COMMAND test_core_client_config "Config - Load from Invalid YAML")

add_test(NAME cte_client_config_environment
    COMMAND test_core_client_config "Config - Load from Environment")

add_test(NAME cte_client_config_save
    COMMAND test_core_client_config "Config - Save to File")

add_test(NAME cte_client_list_targets
    COMMAND test_core_client_config "Client - AsyncListTargets")

add_test(NAME cte_client_stat_targets
    COMMAND test_core_client_config "Client - AsyncStatTargets")

add_test(NAME cte_client_get_target_info
    COMMAND test_core_client_config "Client - AsyncGetTargetInfo")

add_test(NAME cte_client_unregister_target
    COMMAND test_core_client_config "Client - AsyncUnregisterTarget")

add_test(NAME cte_client_get_or_create_tag
    COMMAND test_core_client_config "Client - AsyncGetOrCreateTag Direct")

add_test(NAME cte_client_del_tag_by_id
    COMMAND test_core_client_config "Client - AsyncDelTag by ID")

add_test(NAME cte_client_del_tag_by_name
    COMMAND test_core_client_config "Client - AsyncDelTag by Name")

add_test(NAME cte_client_get_tag_size
    COMMAND test_core_client_config "Client - AsyncGetTagSize")

add_test(NAME cte_client_putblob
    COMMAND test_core_client_config "Client - AsyncPutBlob Direct")

add_test(NAME cte_client_getblob
    COMMAND test_core_client_config "Client - AsyncGetBlob Direct")

add_test(NAME cte_client_delblob
    COMMAND test_core_client_config "Client - AsyncDelBlob")

add_test(NAME cte_client_reorganize_blob
    COMMAND test_core_client_config "Client - AsyncReorganizeBlob Direct")

add_test(NAME cte_client_get_blob_score
    COMMAND test_core_client_config "Client - AsyncGetBlobScore Direct")

add_test(NAME cte_client_get_blob_size
    COMMAND test_core_client_config "Client - AsyncGetBlobSize Direct")

add_test(NAME cte_client_get_contained_blobs
    COMMAND test_core_client_config "Client - AsyncGetContainedBlobs Direct")

add_test(NAME cte_client_tag_query
    COMMAND test_core_client_config "Client - AsyncTagQuery")

add_test(NAME cte_client_blob_query
    COMMAND test_core_client_config "Client - AsyncBlobQuery")

add_test(NAME cte_client_poll_telemetry
    COMMAND test_core_client_config "Client - AsyncPollTelemetryLog")

# Add test_core_runtime_coverage tests - comprehensive runtime method coverage
add_test(NAME cte_runtime_get_target_info_success
    COMMAND test_core_runtime_coverage "Runtime - GetTargetInfo Success Path")

add_test(NAME cte_runtime_get_target_info_error
    COMMAND test_core_runtime_coverage "Runtime - GetTargetInfo NonExistent Target")

add_test(NAME cte_runtime_unregister_success
    COMMAND test_core_runtime_coverage "Runtime - UnregisterTarget Success")

add_test(NAME cte_runtime_unregister_error
    COMMAND test_core_runtime_coverage "Runtime - UnregisterTarget NonExistent")

add_test(NAME cte_runtime_delblob_success
    COMMAND test_core_runtime_coverage "Runtime - DelBlob Success Path")

add_test(NAME cte_runtime_delblob_nonexistent
    COMMAND test_core_runtime_coverage "Runtime - DelBlob NonExistent Blob")

add_test(NAME cte_runtime_delblob_empty
    COMMAND test_core_runtime_coverage "Runtime - DelBlob Empty Name")

add_test(NAME cte_runtime_get_blob_score_success
    COMMAND test_core_runtime_coverage "Runtime - GetBlobScore Success")

add_test(NAME cte_runtime_get_blob_score_nonexistent
    COMMAND test_core_runtime_coverage "Runtime - GetBlobScore NonExistent Blob")

add_test(NAME cte_runtime_get_blob_score_empty
    COMMAND test_core_runtime_coverage "Runtime - GetBlobScore Empty Name")

add_test(NAME cte_runtime_get_blob_size_success
    COMMAND test_core_runtime_coverage "Runtime - GetBlobSize Success")

add_test(NAME cte_runtime_get_blob_size_error
    COMMAND test_core_runtime_coverage "Runtime - GetBlobSize NonExistent")

add_test(NAME cte_runtime_get_tag_size
    COMMAND test_core_runtime_coverage "Runtime - GetTagSize Success")

add_test(NAME cte_runtime_deltag_by_name
    COMMAND test_core_runtime_coverage "Runtime - DelTag by Name Success")

add_test(NAME cte_runtime_deltag_by_id
    COMMAND test_core_runtime_coverage "Runtime - DelTag by ID Success")

add_test(NAME cte_runtime_get_contained_blobs_success
    COMMAND test_core_runtime_coverage "Runtime - GetContainedBlobs Success")

add_test(NAME cte_runtime_get_contained_blobs_empty
    COMMAND test_core_runtime_coverage "Runtime - GetContainedBlobs Empty Tag")

add_test(NAME cte_runtime_list_targets
    COMMAND test_core_runtime_coverage "Runtime - ListTargets Success")

add_test(NAME cte_runtime_stat_targets
    COMMAND test_core_runtime_coverage "Runtime - StatTargets Success")

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
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

set_tests_properties(
    cte_functional_pool_creation
    cte_functional_target_registration
    cte_functional_putblob
    cte_functional_getblob
    cte_functional_putget_integration
    cte_functional_putget_comprehensive
    cte_functional_reorganize
    cte_functional_e2e_workflow
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout for each test
        LABELS "functional;core;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
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
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
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
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

set_tests_properties(
    cte_client_config_default
    cte_client_config_file
    cte_client_config_invalid_file
    cte_client_config_empty_path
    cte_client_config_string
    cte_client_config_empty_string
    cte_client_config_invalid_yaml
    cte_client_config_environment
    cte_client_config_save
    cte_client_list_targets
    cte_client_stat_targets
    cte_client_get_target_info
    cte_client_unregister_target
    cte_client_get_or_create_tag
    cte_client_del_tag_by_id
    cte_client_del_tag_by_name
    cte_client_get_tag_size
    cte_client_putblob
    cte_client_getblob
    cte_client_delblob
    cte_client_reorganize_blob
    cte_client_get_blob_score
    cte_client_get_blob_size
    cte_client_get_contained_blobs
    cte_client_tag_query
    cte_client_blob_query
    cte_client_poll_telemetry
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout for each test
        LABELS "unit;core;cte;client;config"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

set_tests_properties(
    cte_runtime_get_target_info_success
    cte_runtime_get_target_info_error
    cte_runtime_unregister_success
    cte_runtime_unregister_error
    cte_runtime_delblob_success
    cte_runtime_delblob_nonexistent
    cte_runtime_delblob_empty
    cte_runtime_get_blob_score_success
    cte_runtime_get_blob_score_nonexistent
    cte_runtime_get_blob_score_empty
    cte_runtime_get_blob_size_success
    cte_runtime_get_blob_size_error
    cte_runtime_get_tag_size
    cte_runtime_deltag_by_name
    cte_runtime_deltag_by_id
    cte_runtime_get_contained_blobs_success
    cte_runtime_get_contained_blobs_empty
    cte_runtime_list_targets
    cte_runtime_stat_targets
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout for each test
        LABELS "unit;core;cte;runtime;coverage"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

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
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

# Add reorganize blob tests
# NOTE: These tests are chained (Put -> Move -> Integrity -> Promote -> Cleanup)
# and share global state via g_fixture, so they must run together in a single
# process. Running them as separate ctest entries would lose shared state.
add_test(NAME cte_reorganize_all
    COMMAND test_reorganize_blob)

set_tests_properties(
    cte_reorganize_all
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout
        LABELS "unit;reorganize;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Install Targets
# ------------------------------------------------------------------------------
install(TARGETS cte_core_unit_tests test_core_functionality test_query test_cte_config_dpe test_tag_operations test_core_client_config test_core_runtime_coverage test_tiered_storage_stress test_reorganize_blob
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

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