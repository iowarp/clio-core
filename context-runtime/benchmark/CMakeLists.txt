# Benchmark directory

# BDev I/O throughput benchmark
add_executable(clio_run_thrpt_bench
  clio_run_thrpt_bench.cc
)

target_link_libraries(clio_run_thrpt_bench
  cxx                             # Core CLIO Runtime
  clio::run::bdev_client           # BDev client for I/O operations
  clio::run::MOD_NAME_client       # MOD_NAME template module
  clio::run::admin_client          # Admin client for container creation
  ctp::cxx                       # ClioCtp shared memory framework
)

# Install benchmark executable
install(TARGETS clio_run_thrpt_bench
  DESTINATION bin
)

# Cross-node bdev throughput benchmark (MPI). Each rank targets its
# +1 neighbour's bdev container via DirectHash so all writes cross
# the network — isolates clio_run's cross-node task path from CTE/
# FUSE/IOR.
find_package(MPI QUIET)
if(MPI_CXX_FOUND)
  add_executable(clio_xnode_bdev_bench clio_xnode_bdev_bench.cc)
  target_link_libraries(clio_xnode_bdev_bench
    cxx
    clio::run::bdev_client
    clio::run::admin_client
    ctp::cxx
    MPI::MPI_CXX
  )
  install(TARGETS clio_xnode_bdev_bench DESTINATION bin)
else()
  message(STATUS
    "MPI not found — skipping clio_xnode_bdev_bench target (cross-node "
    "stress test requires MPI for cluster launch).")
endif()

# GPU runtime / CDP / SYCL benchmark suites removed along with the GPU
# runtime concept. Kernels now submit tasks to the CPU runtime via the
# gpu2cpu_queue; benchmark coverage of that path is via the unit tests
# in test/unit/gpu/test_sycl_*.cc.
