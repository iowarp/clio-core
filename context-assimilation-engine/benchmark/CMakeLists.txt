# CAE benchmark directory CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# clio_cae_label_bench — stresses CAE's transparent-labeling PutBlob path.
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
endif()
