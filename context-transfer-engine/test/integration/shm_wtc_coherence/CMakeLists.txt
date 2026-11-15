cmake_minimum_required(VERSION 3.10)

# 4-node shm write-through cache coherence suite (run_tests.sh drives it).
add_executable(test_shm_wtc_coherence test_shm_wtc_coherence.cc)
set_target_properties(test_shm_wtc_coherence PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_include_directories(test_shm_wtc_coherence PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test)
target_link_libraries(test_shm_wtc_coherence
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(test_shm_wtc_coherence
    clio_run clio_cte_core_runtime clio_cte_replication_runtime
    clio_cte_cache_runtime clio_cte_core_shm_cache_writethru_runtime)

# Multi-node stress (rotating writers, torn reads, eviction, tag delete).
add_executable(test_shm_wtc_stress test_shm_wtc_stress.cc)
set_target_properties(test_shm_wtc_stress PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_include_directories(test_shm_wtc_stress PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test)
target_link_libraries(test_shm_wtc_stress
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(test_shm_wtc_stress
    clio_run clio_cte_core_runtime clio_cte_replication_runtime
    clio_cte_cache_runtime clio_cte_core_shm_cache_writethru_runtime)

# Restart / crash checker (one phase per run; the harness restarts and
# kills runtimes between phases).
add_executable(test_shm_wtc_restart test_shm_wtc_restart.cc)
set_target_properties(test_shm_wtc_restart PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(test_shm_wtc_restart
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(test_shm_wtc_restart
    clio_run clio_cte_core_runtime clio_cte_replication_runtime
    clio_cte_cache_runtime clio_cte_core_shm_cache_writethru_runtime)
