# CMakeLists.txt for CTE Replication Persistence Test (issue #886)
cmake_minimum_required(VERSION 3.10)

# Two-mode executable (--put-blobs / --verify-blobs); reboot orchestrated by
# the shell script.
add_executable(test_repl_persist test_repl_persist.cc)

target_include_directories(test_repl_persist PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/replication/include
)

target_link_libraries(test_repl_persist
    clio_cte_core_runtime
    clio_cte_core_client
    clio_cte_replication_runtime
    clio_cte_replication_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

set(REPL_PERSIST_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# POSIX-only orchestration (bash, background daemons, kill -9), same as the
# restart integration test.
if(UNIX)
    add_test(NAME cte_replication_persist_integration
        COMMAND bash ${REPL_PERSIST_TEST_DIR}/test_repl_persist.sh
        WORKING_DIRECTORY ${REPL_PERSIST_TEST_DIR}
    )

    set_tests_properties(cte_replication_persist_integration
        PROPERTIES
            TIMEOUT 300
            LABELS "integration;restart;replicas;cte;msan_skip"
            ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};BIN_DIR=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    )
endif()

install(TARGETS test_repl_persist
    RUNTIME DESTINATION bin
)
