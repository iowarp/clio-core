# Producer-consumer workload for the DTSchedule evaluations (MPI).
add_executable(dtschedule_heat_producer heat_producer.c)
target_link_libraries(dtschedule_heat_producer PRIVATE MPI::MPI_C m)
add_executable(dtschedule_checksum_consumer checksum_consumer.c)
target_link_libraries(dtschedule_checksum_consumer PRIVATE MPI::MPI_C)
set_target_properties(dtschedule_heat_producer dtschedule_checksum_consumer
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Same workload over the CTE API instead of POSIX (pc_cte_api.cc): files are
# CTE tags put/get through the dtschedule pool directly, so placement is
# dtschedule's alone (clio-fs routes every read/write to the file's home node).
if(TARGET clio_cte_core_client)
  foreach(prog heat_producer checksum_consumer)
    add_executable(dtschedule_${prog}_api ${prog}.c pc_cte_api.cc)
    target_compile_definitions(dtschedule_${prog}_api PRIVATE PC_USE_CTE_API)
    target_link_libraries(dtschedule_${prog}_api
      PRIVATE MPI::MPI_C m clio_cte_core_client)
    set_target_properties(dtschedule_${prog}_api
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  endforeach()
endif()
