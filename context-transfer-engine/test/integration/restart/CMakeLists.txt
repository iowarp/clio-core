# CMakeLists.txt for CTE Restart Integration Test
cmake_minimum_required(VERSION 3.10)

# Restart test executable (two-mode: --put-blobs / --verify-blobs)
add_executable(test_restart test_restart.cc)

target_include_directories(test_restart PRIVATE
  ${CMAKE_SOURCE_DIR}/test
)

target_link_libraries(test_restart
    wrp_cte_core_runtime
    wrp_cte_core_client
    chimaera::admin_runtime
    chimaera::admin_client
    chimaera::bdev_runtime
    chimaera::bdev_client
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Get the directory containing the test scripts
set(RESTART_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Shell script test that orchestrates multi-process restart workflow
add_test(NAME cte_restart_integration
    COMMAND bash ${RESTART_TEST_DIR}/test_restart.sh
    WORKING_DIRECTORY ${RESTART_TEST_DIR}
)

set_tests_properties(cte_restart_integration
    PROPERTIES
        TIMEOUT 120
        LABELS "integration;restart;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};BIN_DIR=${CMAKE_BINARY_DIR}/bin"
)

install(TARGETS test_restart
    RUNTIME DESTINATION bin
)

message(STATUS "CTE restart integration test configured")
message(STATUS "  Test directory: ${RESTART_TEST_DIR}")
message(STATUS "  Run with: ctest -L restart")
