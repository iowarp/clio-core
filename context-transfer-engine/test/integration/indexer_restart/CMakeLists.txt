# CMakeLists.txt for the indexer restart integration test (issue #905)
cmake_minimum_required(VERSION 3.10)

# Restart test client (two-mode: --put / --verify)
add_executable(test_indexer_restart test_indexer_restart.cc)

target_link_libraries(test_indexer_restart
    clio_cte_core_client
    clio::run::admin_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

set(INDEXER_RESTART_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Shell script test that orchestrates the multi-process restart workflow.
# POSIX-only, same constraints as the cte_restart_integration script.
if(UNIX)
    add_test(NAME cte_indexer_restart
        COMMAND bash ${INDEXER_RESTART_TEST_DIR}/test_indexer_restart.sh
        WORKING_DIRECTORY ${INDEXER_RESTART_TEST_DIR}
    )

    set_tests_properties(cte_indexer_restart
        PROPERTIES
            TIMEOUT 300
            LABELS "integration;restart;cte;indexer;semantic-search;msan_skip"
            ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};BIN_DIR=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
            RESOURCE_LOCK "clio_runtime"
    )
endif()

install(TARGETS test_indexer_restart
    RUNTIME DESTINATION bin
)

message(STATUS "Indexer restart integration test configured")
