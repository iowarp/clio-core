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
    ENVIRONMENT "INIT_CHIMAERA=1;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
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
target_include_directories(test_context_comprehensive
  PRIVATE
    ${CMAKE_SOURCE_DIR}/test  # For simple_test.h
)
target_link_libraries(test_context_comprehensive
  PRIVATE
    wrp_cee_api
)
install(TARGETS test_context_comprehensive DESTINATION bin/tests)

# Register individual test cases with CTest
add_test(NAME CEE_Init COMMAND test_context_comprehensive "[cee][init]")
add_test(NAME CEE_Retrieve_Basic COMMAND test_context_comprehensive "[cee][retrieve][basic]")
add_test(NAME CEE_Retrieve_Empty COMMAND test_context_comprehensive "[cee][retrieve][empty]")
add_test(NAME CEE_Retrieve_SmallBuffer COMMAND test_context_comprehensive "[cee][retrieve][buffer]")
add_test(NAME CEE_Retrieve_CustomBatch COMMAND test_context_comprehensive "[cee][retrieve][batch]")
add_test(NAME CEE_Retrieve_MaxResults COMMAND test_context_comprehensive "[cee][retrieve][limit]")
add_test(NAME CEE_Bundle_Multi COMMAND test_context_comprehensive "[cee][bundle][multi]")
add_test(NAME CEE_Bundle_Range COMMAND test_context_comprehensive "[cee][bundle][range]")
add_test(NAME CEE_Bundle_Invalid COMMAND test_context_comprehensive "[cee][bundle][error]")
add_test(NAME CEE_Query_Regex COMMAND test_context_comprehensive "[cee][query][regex]")
add_test(NAME CEE_Query_Limit COMMAND test_context_comprehensive "[cee][query][limit]")
add_test(NAME CEE_Destroy_Multi COMMAND test_context_comprehensive "[cee][destroy][multi]")
add_test(NAME CEE_Destroy_Partial COMMAND test_context_comprehensive "[cee][destroy][partial]")
add_test(NAME CEE_Splice_Stub COMMAND test_context_comprehensive "[cee][splice][stub]")
add_test(NAME CEE_Integration COMMAND test_context_comprehensive "[cee][integration]")

# Set test properties for comprehensive tests
set_tests_properties(
  CEE_Init CEE_Retrieve_Basic CEE_Retrieve_Empty CEE_Retrieve_SmallBuffer
  CEE_Retrieve_CustomBatch CEE_Retrieve_MaxResults CEE_Bundle_Multi
  CEE_Bundle_Range CEE_Bundle_Invalid CEE_Query_Regex CEE_Query_Limit
  CEE_Destroy_Multi CEE_Destroy_Partial CEE_Splice_Stub CEE_Integration
  PROPERTIES
    TIMEOUT 180
    LABELS "cee;comprehensive;coverage"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

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

  # Copy enhanced Python test to build directory
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_context_interface_enhanced.py
    ${CMAKE_CURRENT_BINARY_DIR}/test_context_interface_enhanced.py
    COPYONLY
  )

  # Add enhanced Python test to CTest
  add_test(
    NAME test_context_interface_enhanced_python
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_context_interface_enhanced.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # Set test properties
  set_tests_properties(test_context_interface_enhanced_python PROPERTIES
    LABELS "cee;api;python;enhanced;coverage"
    TIMEOUT 600  # 10 minute timeout for Python tests
  )

  # Install enhanced Python test
  install(FILES test_context_interface_enhanced.py
    DESTINATION share/wrp_cee/test
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
    NAME test_context_retrieve_roundtrip_python
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_context_retrieve_roundtrip.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(test_context_retrieve_roundtrip_python PROPERTIES
    LABELS "cee;api;python;retrieve;regression;manual"
    TIMEOUT 120
    DISABLED TRUE  # Requires externally running IOWarp runtime
  )

  install(FILES test_context_retrieve_roundtrip.py
    DESTINATION share/wrp_cee/test
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

  message(STATUS "CEE Python tests (basic, enhanced, and retrieve roundtrip) configured")
else()
  message(STATUS "CEE Python test skipped (Python bindings not built)")
endif()

#------------------------------------------------------------------------------
# Test labels for selective execution
#------------------------------------------------------------------------------
set_tests_properties(test_context_bundle PROPERTIES LABELS "cee;api;bundle")
set_tests_properties(test_context_query PROPERTIES LABELS "cee;api;query")
set_tests_properties(test_context_destroy PROPERTIES LABELS "cee;api;destroy")
