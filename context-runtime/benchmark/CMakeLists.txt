# Benchmark directory

# BDev I/O throughput benchmark
add_executable(wrp_run_thrpt_benchmark
  wrp_run_thrpt_benchmark.cc
)

target_link_libraries(wrp_run_thrpt_benchmark
  cxx                             # Core Chimaera runtime
  chimaera::bdev_client           # BDev client for I/O operations
  chimaera::MOD_NAME_client       # MOD_NAME template module
  chimaera::admin_client          # Admin client for container creation
  hshm::cxx                       # HermesShm shared memory framework
)

# Install benchmark executable
install(TARGETS wrp_run_thrpt_benchmark
  DESTINATION bin
)
