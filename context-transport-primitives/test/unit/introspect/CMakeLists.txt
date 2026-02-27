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

# Set timeout and LD_LIBRARY_PATH for mixed mapping test (multi-process)
# LD_LIBRARY_PATH ensures subprocesses load the freshly built library.
# CMAKE_BINARY_DIR is passed so the shell script finds the test binary.
set_tests_properties(
  ctp_mixed_mapping
  PROPERTIES
    TIMEOUT 15
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}"
)

# Skip under MSan: multi-process test; MSan shadow doesn't propagate across fork.
set_tests_properties(
  ctp_mixed_mapping
  PROPERTIES LABELS "msan_skip"
)
endif()
