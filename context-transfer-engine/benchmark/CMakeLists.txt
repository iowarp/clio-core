# Benchmark directory CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Create clio_cte_bench executable
add_executable(clio_cte_bench
  clio_cte_bench.cc
)

# Link required libraries (MPI no longer required - uses worker threads instead)
target_link_libraries(clio_cte_bench
  clio::cte::core_client
  clio::run::cxx
)

# Set C++ standard
target_compile_features(clio_cte_bench PRIVATE cxx_std_17)

# Install the benchmark executable
install(TARGETS clio_cte_bench
  RUNTIME DESTINATION bin
)

# Create clio_cte_semantic_bench executable — ingests N keyword blobs under a
# single tag, then runs one broadcast SemanticSearch over them.
add_executable(clio_cte_semantic_bench
  clio_cte_semantic_bench.cc
)
target_link_libraries(clio_cte_semantic_bench
  clio::cte::core_client
  clio::run::cxx
)
target_compile_features(clio_cte_semantic_bench PRIVATE cxx_std_17)
install(TARGETS clio_cte_semantic_bench
  RUNTIME DESTINATION bin
)

# Create clio_cte_score_bench executable (MPI-based score/demotion benchmark)
# Only build if MPI is enabled
if(CLIO_CORE_ENABLE_MPI)
  add_executable(clio_cte_score_bench
    clio_cte_score_bench.cc
  )

  # Link required libraries including MPI
  target_link_libraries(clio_cte_score_bench
    clio::cte::core_client
    clio::run::cxx
    MPI::MPI_CXX
  )

  # Set C++ standard
  target_compile_features(clio_cte_score_bench PRIVATE cxx_std_17)

  # Install the score benchmark executable
  install(TARGETS clio_cte_score_bench
    RUNTIME DESTINATION bin
  )
else()
  message(STATUS "Skipping clio_cte_score_bench - MPI not enabled (set CLIO_CORE_ENABLE_MPI=ON)")
endif()

# Redis comparison benchmark — apples-to-apples mirror of clio_cte_bench
# (shares bench_common.h). Built only when CLIO_CORE_ENABLE_REDIS=ON, which
# locates hiredis and sets CLIO_REDIS_LIBS (see top-level CMakeLists.txt).
if(CLIO_CORE_ENABLE_REDIS)
  add_executable(clio_redis_bench
    clio_redis_bench.cc
  )
  target_link_libraries(clio_redis_bench
    ${CLIO_REDIS_LIBS}
    ctp::cxx
  )
  target_compile_features(clio_redis_bench PRIVATE cxx_std_17)
  install(TARGETS clio_redis_bench
    RUNTIME DESTINATION bin
  )
else()
  message(STATUS "Skipping clio_redis_bench - Redis not enabled "
                 "(set CLIO_CORE_ENABLE_REDIS=ON; requires hiredis)")
endif()

# NOTE: Compression benchmark (clio_cte_compress_bench) has been moved to
# context-transfer-engine/compressor/test/
