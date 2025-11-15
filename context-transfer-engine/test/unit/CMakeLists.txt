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
# CTE_INIT_RUNTIME to control whether runtime is initialized (default: yes)
add_executable(test_core_functionality
    test_core_functionality.cc
)

# Create test_query executable for query API tests (TagQuery, BlobQuery)
add_executable(test_query
    test_query.cc
)

# Link against required libraries using modern target names
target_link_libraries(cte_core_unit_tests
    wrp_cte_core_client          # CTE core client library
    wrp_cte_core_runtime         # CTE core runtime library
    chimaera::cxx                # Modern Chimaera core framework target
    chimaera::admin_client       # Modern Chimaera admin target (if needed)
    chimaera::bdev_client        # Bdev client for BdevType enum
    Catch2::Catch2WithMain       # Catch2 testing framework with main
)

# Link test_core_functionality
target_link_libraries(test_core_functionality
    wrp_cte_core_client          # CTE core client library
    wrp_cte_core_runtime         # CTE core runtime library
    chimaera::cxx                # Modern Chimaera core framework target
    chimaera::admin_client       # Modern Chimaera admin target (if needed)
    chimaera::bdev_client        # Bdev client for BdevType enum
    Catch2::Catch2WithMain       # Catch2 testing framework with main
)

# Link test_query
target_link_libraries(test_query
    wrp_cte_core_client          # CTE core client library
    wrp_cte_core_runtime         # CTE core runtime library
    chimaera::cxx                # Modern Chimaera core framework target
    chimaera::admin_client       # Modern Chimaera admin target (if needed)
    chimaera::bdev_client        # Bdev client for BdevType enum
    Catch2::Catch2WithMain       # Catch2 testing framework with main
)

# Set compile definitions for runtime initialization control
# Note: Runtime initialization is now controlled by CTE_INIT_RUNTIME environment variable
# at runtime, not at compile time
target_compile_definitions(cte_core_unit_tests PRIVATE CTE_INIT_RUNTIME=1)

# Set compilation standards and warnings
target_compile_features(cte_core_unit_tests PRIVATE cxx_std_17)
target_compile_options(cte_core_unit_tests PRIVATE
    -Wall -Wextra -Werror        # Strict warnings as per CLAUDE.md
    $<$<CONFIG:Debug>:-g -O0>    # Debug symbols and no optimization for debug
    $<$<CONFIG:Release>:-O3>     # Optimization for release
)

target_compile_features(test_core_functionality PRIVATE cxx_std_17)
target_compile_options(test_core_functionality PRIVATE
    -Wall -Wextra -Werror        # Strict warnings as per CLAUDE.md
    $<$<CONFIG:Debug>:-g -O0>    # Debug symbols and no optimization for debug
    $<$<CONFIG:Release>:-O3>     # Optimization for release
)

# Compile options for test_query
target_compile_features(test_query PRIVATE cxx_std_17)
target_compile_options(test_query PRIVATE
    -Wall -Wextra -Werror        # Strict warnings as per CLAUDE.md
    $<$<CONFIG:Debug>:-g -O0>    # Debug symbols and no optimization for debug
    $<$<CONFIG:Release>:-O3>     # Optimization for release
)

# Note: Include directories are provided automatically via linked ChiMod targets
# wrp_cte_core_client provides ${WRP_CTE_ROOT}/core/include via transitive includes

# Add test discovery for CTest integration
include(CTest)
if(BUILD_TESTING)
    # Note: Manual test registration is used below instead of auto-discovery
    # to ensure compatibility with the current Catch2 setup
endif()

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

# Add test_query tests
add_test(NAME cte_query_tag_exact
    COMMAND test_query "[query][tagquery][exact]")

add_test(NAME cte_query_tag_wildcard
    COMMAND test_query "[query][tagquery][wildcard]")

add_test(NAME cte_query_tag_alternation
    COMMAND test_query "[query][tagquery][alternation]")

add_test(NAME cte_query_tag_matchall
    COMMAND test_query "[query][tagquery][matchall]")

add_test(NAME cte_query_tag_nomatch
    COMMAND test_query "[query][tagquery][nomatch]")

add_test(NAME cte_query_blob_exact
    COMMAND test_query "[query][blobquery][exact]")

add_test(NAME cte_query_blob_wildcard
    COMMAND test_query "[query][blobquery][wildcard]")

add_test(NAME cte_query_blob_multitag
    COMMAND test_query "[query][blobquery][multitag]")

add_test(NAME cte_query_blob_matchall
    COMMAND test_query "[query][blobquery][matchall]")

add_test(NAME cte_query_blob_no_blob
    COMMAND test_query "[query][blobquery][noblob]")

add_test(NAME cte_query_blob_no_tag
    COMMAND test_query "[query][blobquery][notag]")

add_test(NAME cte_query_blob_extension
    COMMAND test_query "[query][blobquery][extension]")

add_test(NAME cte_query_local_poolquery
    COMMAND test_query "[query][poolquery][local]")

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
    cte_query_blob_extension
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
# Add External Integration Tests
# ------------------------------------------------------------------------------
# External test demonstrates proper external linking patterns
# Temporarily commented out until ChiMod is installed and can be found as a package
# add_subdirectory(external)

# ------------------------------------------------------------------------------
# Add Adapter Unit Tests
# ------------------------------------------------------------------------------
add_subdirectory(adapters)