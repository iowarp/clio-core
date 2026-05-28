# CAE benchmark directory CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# clio_cae_label_bench — stresses CAE's transparent-labeling PutBlob path.
# Connects through the standard CTE client API (so CAE's interceptor
# handlers are exercised); labeling rule is built at runtime from CLI
# args and shipped via a tmp compose YAML at CLIO_SERVER_CONF.
add_executable(clio_cae_label_bench
  clio_cae_label_bench.cc
)

target_link_libraries(clio_cae_label_bench
  clio::cte::core_client
  chimaera::cxx
)

target_compile_features(clio_cae_label_bench PRIVATE cxx_std_17)

install(TARGETS clio_cae_label_bench
  RUNTIME DESTINATION bin
)
