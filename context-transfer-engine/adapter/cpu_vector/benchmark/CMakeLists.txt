#------------------------------------------------------------------------------
# cpu_vector workloads: CPU ports of the gpu_vector benchmarks.
#------------------------------------------------------------------------------

set(CPU_VECTOR_BENCH_LIBS
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

# k-means: the agent_eval task's workload, and its independent CPU reference.
add_executable(clio_kmeans_cpu_bench kmeans/clio_kmeans_cpu_bench.cc)
target_link_libraries(clio_kmeans_cpu_bench ${CPU_VECTOR_BENCH_LIBS})

add_executable(clio_kmeans_cpu_ref
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gpu_vector/agent_eval/tasks/kmeans/ref.cc)

install(TARGETS clio_kmeans_cpu_bench clio_kmeans_cpu_ref
        RUNTIME DESTINATION bin)

if(CLIO_CORE_ENABLE_TESTS)
    # Out of core: a 32 MB point set through an 8 MB cache must match the
    # reference exactly on counts and to float noise on csum.
    add_test(NAME cte_cpu_vector_kmeans
        COMMAND ${CMAKE_COMMAND}
            -DBENCH=$<TARGET_FILE:clio_kmeans_cpu_bench>
            -DREF=$<TARGET_FILE:clio_kmeans_cpu_ref>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/kmeans/check_kmeans.cmake)
    set_tests_properties(cte_cpu_vector_kmeans PROPERTIES
        TIMEOUT 600
        LABELS "cpu_vector;cte;workload"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1")
endif()
