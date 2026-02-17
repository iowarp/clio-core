# CMakeLists.txt for Chimaera Expand/Induction Integration Tests
cmake_minimum_required(VERSION 3.10)

# Get the directory containing the test scripts
set(EXPAND_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add integration test for node induction
add_test(
    NAME cr_expand_induction
    COMMAND ${EXPAND_TEST_DIR}/run_tests.sh all
    WORKING_DIRECTORY ${EXPAND_TEST_DIR}
)

# Set test properties
set_tests_properties(cr_expand_induction PROPERTIES
    LABELS "integration;docker;expand"
    TIMEOUT 600  # 10 minute timeout for Docker-based tests
)

message(STATUS "Chimaera expand/induction integration test configured")
message(STATUS "  Test directory: ${EXPAND_TEST_DIR}")
message(STATUS "  Run with: ctest -R cr_expand_induction")
