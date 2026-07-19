project(clio_ctp)

#------------------------------------------------------------------------------
# Build Search Index Tests
#------------------------------------------------------------------------------
# The regex search engine is a header-only, std-only template, so the test only
# needs Catch2 (with its own main) and the CTP include path. Threads is for the
# concurrent insert/delete/search stress test.
add_executable(test_regex_search_engine_exec
        test_regex_search_engine.cc)
find_package(Threads REQUIRED)
target_link_libraries(test_regex_search_engine_exec
        Catch2::Catch2WithMain Threads::Threads)

#------------------------------------------------------------------------------
# Test Cases
#------------------------------------------------------------------------------
add_test(NAME ctp_regex_search_engine COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_regex_search_engine_exec)

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_regex_search_engine_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
