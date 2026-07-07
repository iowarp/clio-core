# Compression preprocessor tests (issue #693). Header-only preprocessors, so
# this links only the CTP host lib + Catch2.
project(clio_ctp)

add_executable(test_compress_preprocess_exec
        ${TEST_MAIN}/main.cc
        test_preprocess.cc)

target_link_libraries(test_compress_preprocess_exec
        clio_ctp_host Catch2::Catch2)

add_test(NAME ctp_compress_preprocess COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_compress_preprocess_exec "~[error=FatalError]")

install(TARGETS test_compress_preprocess_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
