# Background compression load for the interference benchmark.
add_executable(dtschedule_compress_load compress_load.cc)
target_link_libraries(dtschedule_compress_load PRIVATE ctp::compress ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(dtschedule_compress_load PRIVATE cxx_std_17)
set_target_properties(dtschedule_compress_load PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
