# Co-located MPI load for the DTSchedule evaluations (see mpi_stress.c).
add_executable(dtschedule_mpi_stress mpi_stress.c)
target_link_libraries(dtschedule_mpi_stress PRIVATE MPI::MPI_C)
set_target_properties(dtschedule_mpi_stress PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
