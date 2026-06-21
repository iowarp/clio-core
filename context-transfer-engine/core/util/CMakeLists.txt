cmake_minimum_required(VERSION 3.10)

add_executable(cte_search cte_search.cc)

target_link_libraries(cte_search
    clio::cte::core_client
    clio::run::cxx
)

target_compile_features(cte_search PRIVATE cxx_std_17)

install(TARGETS cte_search RUNTIME DESTINATION bin)
