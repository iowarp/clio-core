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

# Create single version of test_core_functionality that uses environment variable
# CHIMAERA_WITH_RUNTIME to control whether runtime is initialized (default: yes)
add_executable(test_core_functionality
    test_core_functionality.cc
)

# Create test_query executable for query API tests (TagQuery, BlobQuery)
add_executable(test_query
    test_query.cc
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

# Note: Include directories are provided automatically via linked ChiMod targets
# wrp_cte_core_client provides ${WRP_CTE_ROOT}/core/include via transitive includes

# Manual test registration for specific test cases
add_test(NAME cte_core_pool_creation
    COMMAND cte_core_unit_tests "[core][cte][pool]")

add_test(NAME cte_core_target_registration
    COMMAND cte_core_unit_tests "[core][cte][target]")

add_test(NAME cte_core_blob_put
    COMMAND cte_core_unit_tests "[core][cte][putblob]")

add_test(NAME cte_core_blob_get
    COMMAND cte_core_unit_tests "[core][cte][getblob]")

add_test(NAME cte_core_integration
    COMMAND cte_core_unit_tests "[core][cte][integration]")

add_test(NAME cte_core_client_creation
    COMMAND cte_core_unit_tests "[client][core][creation][cte]")

add_test(NAME cte_core_params
    COMMAND cte_core_unit_tests "[core][cte][params]")

add_test(NAME cte_core_target_config
    COMMAND cte_core_unit_tests "[config][core][cte][target]")

add_test(NAME cte_core_tag_info
    COMMAND cte_core_unit_tests "[core][cte][info][tag]")

add_test(NAME cte_core_blob_info
    COMMAND cte_core_unit_tests "[blob][core][cte][info]")

add_test(NAME cte_core_tasks
    COMMAND cte_core_unit_tests "[core][cte][tasks]")

add_test(NAME cte_core_helpers
    COMMAND cte_core_unit_tests "[core][cte][helpers]")

add_test(NAME cte_core_workflow
    COMMAND cte_core_unit_tests "[core][cte][workflow]")

add_test(NAME cte_core_performance
    COMMAND cte_core_unit_tests "[core][cte][performance]")

# Add test_core_functionality tests
add_test(NAME cte_functional_pool_creation
    COMMAND test_core_functionality "[core][creation][cte][pool]")

add_test(NAME cte_functional_target_registration
    COMMAND test_core_functionality "[core][cte][registration][target]")

add_test(NAME cte_functional_putblob
    COMMAND test_core_functionality "[blob][core][cte][functional][put]")

add_test(NAME cte_functional_getblob
    COMMAND test_core_functionality "[blob][core][cte][functional][get]")

add_test(NAME cte_functional_putget_integration
    COMMAND test_core_functionality "[blob][core][cte][integration][put-get]")

add_test(NAME cte_functional_putget_comprehensive
    COMMAND test_core_functionality "[blob][comprehensive][core][cte][integration][put-get]")

add_test(NAME cte_functional_reorganize
    COMMAND test_core_functionality "[blob][core][cte][functional][reorganize]")

add_test(NAME cte_functional_e2e_workflow
    COMMAND test_core_functionality "[core][cte][integration]")

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
    cte_core_workflow
    cte_core_performance
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout for each test
        LABELS "unit;core;cte"
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
)

# ------------------------------------------------------------------------------
# Coverage Support
# ------------------------------------------------------------------------------
if(WRP_CTE_ENABLE_COVERAGE)
    set_coverage_flags(cte_core_unit_tests)
endif()

# ------------------------------------------------------------------------------
# Install Targets
# ------------------------------------------------------------------------------
install(TARGETS cte_core_unit_tests test_core_functionality test_query
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