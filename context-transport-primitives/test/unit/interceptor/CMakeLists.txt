project(hermes_shm)

add_library(test_interceptor_lib_stat STATIC test_lib.cc)
add_library(test_interceptor_my_lib_stat STATIC test_my_lib.cc)
add_library(test_interceptor_lib SHARED test_lib.cc)
add_library(test_interceptor_my_lib SHARED test_my_lib.cc)

add_executable(test_interceptor test_interceptor.cc)
add_dependencies(test_interceptor test_interceptor_lib)
target_link_libraries(test_interceptor PRIVATE test_interceptor_lib)

add_executable(test_interceptor_stat test_interceptor.cc)
add_dependencies(test_interceptor_stat test_interceptor_lib_stat)
target_link_libraries(test_interceptor_stat PRIVATE test_interceptor_lib_stat)

install(TARGETS
    test_interceptor
    test_interceptor_lib
    test_interceptor_my_lib
    test_interceptor_stat
    test_interceptor_lib_stat
    test_interceptor_my_lib_stat
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)