# Producer-consumer workload for the DTSchedule evaluations (MPI).
add_executable(dtschedule_heat_producer heat_producer.c)
target_link_libraries(dtschedule_heat_producer PRIVATE MPI::MPI_C m)
add_executable(dtschedule_checksum_consumer checksum_consumer.c)
target_link_libraries(dtschedule_checksum_consumer PRIVATE MPI::MPI_C)
set_target_properties(dtschedule_heat_producer dtschedule_checksum_consumer
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
