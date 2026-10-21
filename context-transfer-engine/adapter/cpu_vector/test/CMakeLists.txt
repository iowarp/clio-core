#------------------------------------------------------------------------------
# cpu_vector tests: host-only, against an in-process runtime with a RAM tier.
#------------------------------------------------------------------------------

add_executable(test_cpu_vector test_cpu_vector.cc)
target_link_libraries(test_cpu_vector
    clio::cte::cpu_vector
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(test_cpu_vector PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test   # simple_test.h
)

add_test(NAME cte_cpu_vector COMMAND test_cpu_vector)
set_tests_properties(cte_cpu_vector PROPERTIES
    TIMEOUT 300
    LABELS "cpu_vector;cte"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1"
)
