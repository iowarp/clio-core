# 4-node cache-coherence suite (issue #886 distributed coherence).
#
# Builds the collective coherence binary (every node runs it) and registers
# a Docker-based ctest that spins up the 4-node cluster with the full
# default chain (cache -> replication -> core) composed on every node.

cmake_minimum_required(VERSION 3.10)

add_executable(test_cache_coherence test_cache_coherence.cc)
set_target_properties(test_cache_coherence PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_include_directories(test_cache_coherence PRIVATE
    ${CMAKE_SOURCE_DIR}/context-runtime/test)
target_link_libraries(test_cache_coherence
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(test_cache_coherence
    clio_run clio_cte_core_runtime
    clio_cte_replication_runtime clio_cte_cache_runtime)

# IOR-like chain benchmark (not a ctest; driven manually / by bench scripts).
add_executable(bench_chain bench_chain.cc)
set_target_properties(bench_chain PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(bench_chain
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(bench_chain
    clio_run clio_cte_core_runtime
    clio_cte_replication_runtime clio_cte_cache_runtime)

set(COHERENCE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_test(
    NAME cte_cache_coherence_integration_docker
    COMMAND ${COHERENCE_TEST_DIR}/run_tests.sh
    WORKING_DIRECTORY ${COHERENCE_TEST_DIR}
)
set_tests_properties(cte_cache_coherence_integration_docker PROPERTIES
    LABELS "integration;docker;distributed;coherence;cache"
    TIMEOUT 900  # 15-minute cap for the 4-node docker suite
)
