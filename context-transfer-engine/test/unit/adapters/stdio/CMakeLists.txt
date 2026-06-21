# STDIO Adapter Unit Tests

include_directories(
    ${CLIO_CTE_ROOT}
    .
)

add_executable(clio_cte_stdio_unit_tests
    test_stdio_adapter.cc
)

target_link_libraries(clio_cte_stdio_unit_tests
    clio_cte_stdio
    clio_cte_cfs_adapter      # cfs-backed I/O core (context-filesystem chimod)
    clio_cte_filesystem_client
    clio_cte_core_client
    ctp::cxx
    ctp::interceptor
    Catch2::Catch2WithMain
)

if(CLIO_CORE_ENABLE_MPI)
    target_link_libraries(clio_cte_stdio_unit_tests MPI::MPI_CXX)
endif()

add_dependencies(clio_cte_stdio_unit_tests
    clio_cte_core_runtime
    clio_cte_filesystem_runtime
    clio_bdev_runtime
    clio_admin_runtime)

install(
    TARGETS clio_cte_stdio_unit_tests
    RUNTIME DESTINATION bin
)

if(CLIO_CORE_ENABLE_TESTS)
    add_test(
        NAME clio_cte_stdio_unit_tests
        COMMAND clio_cte_stdio_unit_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(clio_cte_stdio_unit_tests PROPERTIES
        ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
        RESOURCE_LOCK "clio_runtime_port"
        TIMEOUT 120
        LABELS "msan_skip")
endif()
