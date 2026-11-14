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
    clio_cte_cache_runtime clio_cte_shm_writethru_cache_runtime)
