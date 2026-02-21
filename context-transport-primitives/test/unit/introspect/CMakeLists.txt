# Introspect tests

if(NOT WIN32)
# Test for MapMixedMemory functionality
add_executable(test_mixed_mapping test_mixed_mapping.cc)
target_link_libraries(test_mixed_mapping hshm::cxx)

# Install to build/bin
install(TARGETS test_mixed_mapping
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Add CTest
add_test(
  NAME ctp_mixed_mapping
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_mixed_mapping_test.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Set timeout for mixed mapping test (multi-process, needs more time)
set_tests_properties(
  ctp_mixed_mapping
  PROPERTIES TIMEOUT 15
)
endif()
