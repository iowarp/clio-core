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
      wrp_cee_api
  )

  # Install test binary
  install(TARGETS ${TEST_NAME} DESTINATION bin/tests)

  # Register with CTest
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

  # Set test environment variables
  set_tests_properties(${TEST_NAME} PROPERTIES
    ENVIRONMENT "INIT_CHIMAERA=1"
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

#------------------------------------------------------------------------------
# Python Tests (if Python bindings are enabled)
#------------------------------------------------------------------------------
if(TARGET wrp_cee)
  # Copy Python test to build directory
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_context_interface.py
    ${CMAKE_CURRENT_BINARY_DIR}/test_context_interface.py
    COPYONLY
  )

  # Add Python test to CTest
  add_test(
    NAME test_context_interface_python
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_context_interface.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # Set test properties
  set_tests_properties(test_context_interface_python PROPERTIES
    LABELS "cee;api;python;interface"
    TIMEOUT 600  # 10 minute timeout for Python tests
  )

  # Install Python test
  install(FILES test_context_interface.py
    DESTINATION share/wrp_cee/test
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

  message(STATUS "CEE Python test (test_context_interface.py) configured")
else()
  message(STATUS "CEE Python test skipped (Python bindings not built)")
endif()

#------------------------------------------------------------------------------
# Test labels for selective execution
#------------------------------------------------------------------------------
set_tests_properties(test_context_bundle PROPERTIES LABELS "cee;api;bundle")
set_tests_properties(test_context_query PROPERTIES LABELS "cee;api;query")
set_tests_properties(test_context_destroy PROPERTIES LABELS "cee;api;destroy")
