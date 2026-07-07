project(clio_ctp)

#------------------------------------------------------------------------------
# Build Benchmark
#------------------------------------------------------------------------------

# Unified synthetic compressor benchmark combining NeuroPress and CTE roles
add_executable(benchmark_compress_metrics_exec benchmark_compress_metrics.cc)

#------------------------------------------------------------------------------
# Link Dependencies
#------------------------------------------------------------------------------

target_link_libraries(benchmark_compress_metrics_exec
        clio_ctp_host ctp::compress)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------

install(TARGETS
        benchmark_compress_metrics_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
