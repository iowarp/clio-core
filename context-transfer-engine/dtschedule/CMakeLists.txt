cmake_minimum_required(VERSION 3.10)
project(clio_cte_dtschedule)

#------------------------------------------------------------------------------
# DTSchedule chimod — decision + codec + tier + location selection layer for
# transparent compression. Phase 1: skeleton with fixed-codec PutBlob/GetBlob,
# config parsing, tier tables, trace writer.
#------------------------------------------------------------------------------

add_clio_module_runtime(
  SOURCES
    src/dtschedule_runtime.cc
    src/dag_spec.cc
    src/autogen/dtschedule_lib_exec.cc
    src/ccm/data_stats.cc
    src/ccm/qtable_predictor.cc
    src/ccm/ema_predictor.cc
    src/ccm/fixed_predictor.cc
    src/ccm/oracle_predictor.cc
    src/ccm/candidates.cc
    src/ccm/ranker.cc
    src/ccm/ccm_manager.cc
)

add_clio_module_client(
  SOURCES
    src/dtschedule_client.cc
)

# DTSchedule drives the CTE core through its public client
target_link_libraries(clio_cte_dtschedule_runtime PUBLIC clio_cte_core_client)
target_link_libraries(clio_cte_dtschedule_client PUBLIC clio_cte_core_client)

# Link compression support and JSON for Q-table
target_link_libraries(clio_cte_dtschedule_runtime PRIVATE ctp::compress)
target_link_libraries(clio_cte_dtschedule_runtime PRIVATE nlohmann_json::nlohmann_json)

# Set compile definition for Q-table model directory (relative to source)
# Default points to qtable_v1; users can override with qtable_model_path in YAML
target_compile_definitions(clio_cte_dtschedule_runtime PRIVATE
    DTSCHEDULE_MODEL_DIR="${CMAKE_SOURCE_DIR}/context-transfer-engine/dtschedule/models/qtable_v1")

# Include directories
target_include_directories(clio_cte_dtschedule_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transport-primitives/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/dtschedule/include)
target_include_directories(clio_cte_dtschedule_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/dtschedule/include)

# Add unit tests if testing is enabled
if(BUILD_TESTING OR CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()

# Co-located MPI load generator for the evaluations (optional: needs MPI).
add_subdirectory(tools/compress_load)
find_package(MPI COMPONENTS C QUIET)
if(MPI_C_FOUND)
    add_subdirectory(tools/mpi_stress)
    add_subdirectory(tools/prodcons)
endif()
