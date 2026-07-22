# Gray-Scott L=512 congestion-collapse reproducer (issue #774).
#
# Builds the symmetric per-node producer binary and registers a MANUAL
# docker-based repro harness: 2-node cluster (4 daemon worker threads, Delta
# parity), each node Puts blobs whose names HashBlobToContainer-scatter ~50%
# cross-node, then Gets them back twice per output (hermes_derived analog).
# Phase 1 (0.5 MB blobs, L=256 analog) must complete; phase 2 (4 MB blobs,
# L=512 analog) probes for the throughput-collapse cliff.
#
# This is a REPRO HARNESS, not a CI gate: the ctest below runs only the small
# phase (harness sanity). Run the full probe manually:
#   ./run_tests.sh          # both phases + repro report
#   ./run_tests.sh large    # just the L=512 analog probe

cmake_minimum_required(VERSION 3.10)

add_executable(test_grayscott_congestion test_grayscott_congestion.cc)
set_target_properties(test_grayscott_congestion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(test_grayscott_congestion
    clio_cte_core_client
    clio_admin_client
    clio_run_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(test_grayscott_congestion clio_run clio_cte_core_runtime)

set(GS_CONGEST_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_test(
    NAME cte_grayscott_congestion_docker_small
    COMMAND ${GS_CONGEST_TEST_DIR}/run_tests.sh small
    WORKING_DIRECTORY ${GS_CONGEST_TEST_DIR}
)
set_tests_properties(cte_grayscott_congestion_docker_small PROPERTIES
    LABELS "integration;docker;distributed;grayscott"
    TIMEOUT 900
)

message(STATUS "CTE grayscott_congestion repro harness configured (issue #774)")
message(STATUS "  Full probe: ${GS_CONGEST_TEST_DIR}/run_tests.sh")
