# CAE benchmark directory CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# clio_cae_label_bench — stresses the summarizer chimod's PutBlob path
# (CAE core -> summarizer -> CTE core).
# Uses POSIX-only APIs (mkstemps, /tmp); skip on Windows.
if(NOT WIN32)
  add_executable(clio_cae_label_bench
    clio_cae_label_bench.cc
  )

  target_link_libraries(clio_cae_label_bench
    clio::cte::core_client
    clio::run::cxx
  )

  target_compile_features(clio_cae_label_bench PRIVATE cxx_std_17)

  install(TARGETS clio_cae_label_bench
    RUNTIME DESTINATION bin
  )

  # clio_s3_read_bench — times CAE's s3:// -> CTE read path against real S3.
  # Gated on CAE_ENABLE_S3: the S3 assimilator source is only compiled in then,
  # and the benchmark fork+execs cae_s3_tool, built by that same gate.
  if(CAE_ENABLE_S3)
    add_executable(clio_s3_read_bench
      clio_s3_read_bench.cc
    )

    # Deliberately NOT linking aws-cpp-sdk-*: this process calls CLIO_INIT, and
    # loading libaws-cpp-sdk-core.so into a CLIO-initializing process corrupts
    # runtime startup. All S3 I/O is out-of-process via cae_s3_tool.
    target_link_libraries(clio_s3_read_bench
      clio_cae_core_client
      clio_cte_core_client
      clio::run::cxx
      ctp::configure
    )

    # bench_common.h (PrintResults) is not an exported include dir; reach it by
    # source-tree path so the results wording stays byte-identical to
    # clio_cte_bench, whose Jarvis regex table this benchmark reuses.
    target_include_directories(clio_s3_read_bench PRIVATE
      ${CMAKE_SOURCE_DIR}/context-transfer-engine/benchmark
    )

    target_compile_features(clio_s3_read_bench PRIVATE cxx_std_17)

    # Ordering only — the helper is exec'd by path at runtime, never linked.
    add_dependencies(clio_s3_read_bench cae_s3_tool)

    install(TARGETS clio_s3_read_bench
      RUNTIME DESTINATION bin
    )
  endif()
endif()
