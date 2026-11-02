project(clio_ctp)

add_library(my_lib1 STATIC my_lib1.cc)
target_link_libraries(my_lib1 PRIVATE clio_ctp_host)
add_library(my_lib2 STATIC my_lib2.cc)
target_link_libraries(my_lib2 PRIVATE clio_ctp_host)

add_executable(test_singleton ${TEST_MAIN}/main.cc test_init.cc test_singleton.cc)
add_dependencies(test_singleton my_lib1 my_lib2)
target_link_libraries(test_singleton PRIVATE clio_ctp_host my_lib1 my_lib2 Catch2::Catch2)