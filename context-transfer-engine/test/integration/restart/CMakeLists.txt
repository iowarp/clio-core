# CMakeLists.txt for CTE Restart Integration Test
cmake_minimum_required(VERSION 3.10)

# Restart test executable (two-mode: --put-blobs / --verify-blobs)
add_executable(test_restart test_restart.cc)

target_link_libraries(test_restart
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Get the directory containing the test scripts
set(RESTART_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Shell script test that orchestrates multi-process restart workflow.
# POSIX-only: the .sh launches background processes via bash, sleeps, kills
# them with `kill -9`, and inspects /tmp. No clean Windows equivalent; the
# C++ binary still builds so it can be invoked manually if needed.
if(UNIX)
    add_test(NAME cte_restart_integration
        COMMAND bash ${RESTART_TEST_DIR}/test_restart.sh
        WORKING_DIRECTORY ${RESTART_TEST_DIR}
    )

    set_tests_properties(cte_restart_integration
        PROPERTIES
            TIMEOUT 300
            LABELS "integration;restart;cte;msan_skip"
            ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};BIN_DIR=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    )
endif()

install(TARGETS test_restart
    RUNTIME DESTINATION bin
)

message(STATUS "CTE restart integration test configured")
message(STATUS "  Test directory: ${RESTART_TEST_DIR}")
message(STATUS "  Run with: ctest -L restart")
