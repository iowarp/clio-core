#------------------------------------------------------------------------------
# Unit Tests CMakeLists.txt
#------------------------------------------------------------------------------
# This file defines all unit tests for the CAE (Content Assimilation Engine)
# and configures them as CTest targets for automated testing.
#
# Tests include:
# - Binary file assimilation
# - Range-based partial transfers
# - Error handling and validation
# - HDF5 format assimilation
# - Globus transfer integration (conditional)
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.10)

# Enable testing for CTest
enable_testing()

# Find required dependencies
# yaml-cpp is already found by root CMakeLists.txt

# Conditionally find HDF5 if enabled
if(WRP_CAE_ENABLE_HDF5)
  # HDF5 is already found by root CMakeLists.txt
  if(NOT HDF5_FOUND)
    message(WARNING "WRP_CAE_ENABLE_HDF5 is ON but HDF5 not found. HDF5 tests will be disabled.")
    set(HDF5_ENABLED FALSE)
  else()
    set(HDF5_ENABLED TRUE)
  endif()
else()
  set(HDF5_ENABLED FALSE)
endif()

#------------------------------------------------------------------------------
# Helper function to create a unit test
#------------------------------------------------------------------------------
# Creates an executable, links dependencies, and registers with CTest
#
# Arguments:
#   TEST_NAME - Name of the test executable
#   TEST_SOURCE - Path to the test source file (relative to unit/ directory)
#   EXTRA_LIBS - Additional libraries to link (optional)
#   EXTRA_INCLUDES - Additional include directories (optional)
#------------------------------------------------------------------------------
function(add_cae_unit_test TEST_NAME TEST_SOURCE)
  # Parse optional arguments
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs EXTRA_LIBS EXTRA_INCLUDES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Create test executable
  add_executable(${TEST_NAME} ${TEST_SOURCE})

  # Link common dependencies
  target_link_libraries(${TEST_NAME}
    PRIVATE
      wrp_cae_core_client
      wrp_cte_core_client
      chimaera::cxx
      hshm::configure
      ${ARG_EXTRA_LIBS}
  )

  # Add common include directories
  target_include_directories(${TEST_NAME}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${ARG_EXTRA_INCLUDES}
  )

  # Install test binary
  install(TARGETS ${TEST_NAME} DESTINATION bin)

  # Register with CTest
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

  # Set test environment variables
  set_tests_properties(${TEST_NAME} PROPERTIES
    ENVIRONMENT "CHIMAERA_WITH_RUNTIME=1;CHI_SERVER_CONF=/workspace/context-assimilation-engine/test/unit/wrp_config.yaml"
    TIMEOUT 300  # 5 minute timeout
  )
endfunction()

#------------------------------------------------------------------------------
# Test 1: Binary File Assimilation
#------------------------------------------------------------------------------
# Tests basic binary file transfer functionality
# - Creates test binary file with patterned data
# - Loads OMNI configuration from YAML
# - Calls ParseOmni to transfer to CTE
# - Validates data integrity in CTE
#------------------------------------------------------------------------------
add_cae_unit_test(cae_binary_assim
  ${CMAKE_CURRENT_SOURCE_DIR}/binary_assim/test_binary_assim.cc
)

# Install test configuration
install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/binary_assim/binary_assim_omni.yaml
  DESTINATION share/wrp_cae/test/binary_assim
)

#------------------------------------------------------------------------------
# Test 2: Range-Based Partial Transfers
#------------------------------------------------------------------------------
# Tests partial file transfers using range_off and range_size
# - Creates large test file
# - Transfers specific byte ranges
# - Validates correct data and size in CTE
#------------------------------------------------------------------------------
add_cae_unit_test(cae_range_assim
  ${CMAKE_CURRENT_SOURCE_DIR}/range_test/test_range_assim.cc
)

# Set smaller file size for range test to speed up CI
set_tests_properties(cae_range_assim PROPERTIES
  ENVIRONMENT "CHIMAERA_WITH_RUNTIME=1;CHI_SERVER_CONF=/workspace/context-assimilation-engine/test/unit/wrp_config.yaml;TEST_FILE_SIZE=10"  # 10MB instead of default
)

#------------------------------------------------------------------------------
# Test 3: Error Handling and Validation
#------------------------------------------------------------------------------
# Tests error cases and validation logic
# - Invalid source paths
# - Invalid format types
# - Missing required fields
# - Malformed URIs
#------------------------------------------------------------------------------
add_cae_unit_test(cae_error_handling
  ${CMAKE_CURRENT_SOURCE_DIR}/error_test/test_error_handling.cc
)

#------------------------------------------------------------------------------
# Test 4: HDF5 Format Assimilation (Conditional)
#------------------------------------------------------------------------------
# Tests HDF5 hierarchical data format support
# - Creates HDF5 file with multiple datasets
# - Tests dataset discovery and transfer
# - Validates metadata and tensor information
# - Tests hierarchical group structures
# Note: Requires WRP_CORE_ENABLE_HDF5=ON
#------------------------------------------------------------------------------
if(HDF5_ENABLED)
  add_cae_unit_test(cae_hdf5_assim
    ${CMAKE_CURRENT_SOURCE_DIR}/hdf5_assim/test_hdf5_assim.cc
    EXTRA_LIBS ${HDF5_LIBRARIES}
    EXTRA_INCLUDES ${HDF5_INCLUDE_DIRS}
  )

  # Set test labels
  set_tests_properties(cae_hdf5_assim PROPERTIES LABELS "format;hdf5")
endif()

#------------------------------------------------------------------------------
# Test 5: Globus Transfer Integration (Conditional) - REMOVED
#------------------------------------------------------------------------------
# Globus integration testing has been moved to integration tests only
# due to the complexity of OAuth2 authentication and collection-specific scopes.
# See test/integration/globus_matsci/ for Globus integration testing.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Custom CTest Configuration
#------------------------------------------------------------------------------

# Create test labels for selective execution
set_tests_properties(cae_binary_assim PROPERTIES LABELS "core;basic")
set_tests_properties(cae_range_assim PROPERTIES LABELS "core;advanced")
set_tests_properties(cae_error_handling PROPERTIES LABELS "core;validation")

#------------------------------------------------------------------------------
# Test Summary Target
#------------------------------------------------------------------------------
# Custom target to run all tests with verbose output
add_custom_target(run_unit_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running all CAE unit tests with verbose output"
  USES_TERMINAL
)

# Custom target to run only core tests
add_custom_target(run_core_tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose -L core
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running core CAE unit tests"
  USES_TERMINAL
)

#------------------------------------------------------------------------------
# Print Configuration Summary
#------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "CAE Unit Tests Configuration:")
message(STATUS "  Binary Assimilation Test: ENABLED")
message(STATUS "  Range Transfer Test:      ENABLED")
message(STATUS "  Error Handling Test:      ENABLED")
if(HDF5_ENABLED)
  message(STATUS "  HDF5 Format Test:         ENABLED")
else()
  message(STATUS "  HDF5 Format Test:         DISABLED (set -DWRP_CORE_ENABLE_HDF5=ON to enable)")
endif()
message(STATUS "")
message(STATUS "Run tests with:")
message(STATUS "  make test                 - Run all tests")
message(STATUS "  make run_unit_tests       - Run all tests with verbose output")
message(STATUS "  make run_core_tests       - Run only core tests")
message(STATUS "  ctest -L core             - Run tests with 'core' label")
if(HDF5_ENABLED)
  message(STATUS "  ctest -L hdf5             - Run tests with 'hdf5' label")
endif()
message(STATUS "")
