cmake_minimum_required(VERSION 3.20)

# Enable testing for CTest
enable_testing()

#------------------------------------------------------------------------------
# Helper function to create a unit test
#------------------------------------------------------------------------------
# Creates an executable, links dependencies, and registers with CTest
#------------------------------------------------------------------------------
function(add_cee_api_unit_test TEST_NAME TEST_SOURCE)
  # Create test executable
  add_executable(${TEST_NAME} ${TEST_SOURCE})

  # Link common dependencies
  target_link_libraries(${TEST_NAME}
    PRIVATE
      clio_cee_api
  )

  # Install test binary
  install(TARGETS ${TEST_NAME} DESTINATION bin/tests)

  # Register with CTest
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

  # Set test environment variables
  set_tests_properties(${TEST_NAME} PROPERTIES
    ENVIRONMENT "INIT_CHIMAERA=1;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300  # 5 minute timeout
  )
endfunction()

#------------------------------------------------------------------------------
# Unit Tests
#------------------------------------------------------------------------------

# Test for context_bundle
add_cee_api_unit_test(test_context_bundle test_context_bundle.cc)

# Test for context_query
add_cee_api_unit_test(test_context_query test_context_query.cc)

# Test for context_destroy
add_cee_api_unit_test(test_context_destroy test_context_destroy.cc)

# Comprehensive coverage test
add_executable(test_context_comprehensive test_context_comprehensive.cc)
target_link_libraries(test_context_comprehensive
  PRIVATE
    clio_cee_api
)
install(TARGETS test_context_comprehensive DESTINATION bin/tests)

# Register individual test cases with CTest
add_test(NAME cee_init COMMAND test_context_comprehensive "[cee][init]")
add_test(NAME cee_retrieve_basic COMMAND test_context_comprehensive "[cee][retrieve][basic]")
add_test(NAME cee_retrieve_empty COMMAND test_context_comprehensive "[cee][retrieve][empty]")
add_test(NAME cee_retrieve_smallbuffer COMMAND test_context_comprehensive "[cee][retrieve][buffer]")
add_test(NAME cee_retrieve_custombatch COMMAND test_context_comprehensive "[cee][retrieve][batch]")
add_test(NAME cee_retrieve_maxresults COMMAND test_context_comprehensive "[cee][retrieve][limit]")
add_test(NAME cee_bundle_multi COMMAND test_context_comprehensive "[cee][bundle][multi]")
add_test(NAME cee_bundle_range COMMAND test_context_comprehensive "[cee][bundle][range]")
add_test(NAME cee_bundle_invalid COMMAND test_context_comprehensive "[cee][bundle][error]")
add_test(NAME cee_query_regex COMMAND test_context_comprehensive "[cee][query][regex]")
add_test(NAME cee_query_limit COMMAND test_context_comprehensive "[cee][query][limit]")
add_test(NAME cee_destroy_multi COMMAND test_context_comprehensive "[cee][destroy][multi]")
add_test(NAME cee_destroy_partial COMMAND test_context_comprehensive "[cee][destroy][partial]")
add_test(NAME cee_splice_stub COMMAND test_context_comprehensive "[cee][splice][stub]")
add_test(NAME cee_integration COMMAND test_context_comprehensive "[cee][integration]")

# Set test properties for comprehensive tests
set_tests_properties(
  cee_init cee_retrieve_basic cee_retrieve_empty cee_retrieve_smallbuffer
  cee_retrieve_custombatch cee_retrieve_maxresults cee_bundle_multi
  cee_bundle_range cee_bundle_invalid cee_query_regex cee_query_limit
  cee_destroy_multi cee_destroy_partial cee_splice_stub cee_integration
  PROPERTIES
    TIMEOUT 180
    LABELS "cee;comprehensive;coverage;msan_skip"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHI_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    RESOURCE_LOCK "chimaera_runtime"
)

#------------------------------------------------------------------------------
# Python Tests (if Python bindings are enabled)
#------------------------------------------------------------------------------
if(TARGET clio_cee)
  # Copy Python test to build directory
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_context_interface.py
    ${CMAKE_CURRENT_BINARY_DIR}/test_context_interface.py
    COPYONLY
  )

  # Add Python test to CTest
  add_test(
    NAME cee_python_interface
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_context_interface.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # Set test properties
  set_tests_properties(cee_python_interface PROPERTIES
    LABELS "cee;api;python;interface"
    TIMEOUT 600  # 10 minute timeout for Python tests
  )

  # Install Python test
  install(FILES test_context_interface.py
    DESTINATION share/clio_cee/test
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

  # Copy enhanced Python test to build directory
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_context_interface_enhanced.py
    ${CMAKE_CURRENT_BINARY_DIR}/test_context_interface_enhanced.py
    COPYONLY
  )

  # Add enhanced Python test to CTest
  add_test(
    NAME cee_python_interface_enhanced
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_context_interface_enhanced.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # Set test properties
  set_tests_properties(cee_python_interface_enhanced PROPERTIES
    LABELS "cee;api;python;enhanced;coverage"
    TIMEOUT 600  # 10 minute timeout for Python tests
  )

  # Install enhanced Python test
  install(FILES test_context_interface_enhanced.py
    DESTINATION share/clio_cee/test
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

  # Copy retrieve roundtrip regression test to build directory
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_context_retrieve_roundtrip.py
    ${CMAKE_CURRENT_BINARY_DIR}/test_context_retrieve_roundtrip.py
    COPYONLY
  )

  # Add retrieve roundtrip test to CTest
  add_test(
    NAME cee_python_retrieve_roundtrip
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_context_retrieve_roundtrip.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(cee_python_retrieve_roundtrip PROPERTIES
    LABELS "cee;api;python;retrieve;regression;manual"
    TIMEOUT 120
    DISABLED TRUE  # Requires externally running IOWarp runtime
  )

  install(FILES test_context_retrieve_roundtrip.py
    DESTINATION share/clio_cee/test
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

  message(STATUS "CEE Python tests (basic, enhanced, and retrieve roundtrip) configured")
else()
  message(STATUS "CEE Python test skipped (Python bindings not built)")
endif()

#------------------------------------------------------------------------------
# Test labels for selective execution
#------------------------------------------------------------------------------
# Skip under MSan: CEE tests use uninstrumented chimaera runtime (ZMQ, shared
# memory, forked processes) that produce false-positive MSan reports.
set_tests_properties(test_context_bundle PROPERTIES LABELS "cee;api;bundle;msan_skip")
set_tests_properties(test_context_query PROPERTIES LABELS "cee;api;query;msan_skip")
set_tests_properties(test_context_destroy PROPERTIES LABELS "cee;api;destroy;msan_skip")
