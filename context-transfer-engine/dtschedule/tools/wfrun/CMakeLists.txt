# Limited WfCommons executor over the CTE API (MPI): runs a placement.json
# from tools/wfcommons/dt_translator.py --api through the dtschedule pool.
# Reuses the prodcons CTE-API backend (pc_cte_api.cc) as-is.
if(TARGET clio_cte_core_client)
  add_executable(dtschedule_wfrun wfrun.cc wf_dag.cc wf_data.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../prodcons/pc_cte_api.cc)
  target_link_libraries(dtschedule_wfrun
    PRIVATE MPI::MPI_C clio_cte_core_client nlohmann_json::nlohmann_json)
  set_target_properties(dtschedule_wfrun
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
