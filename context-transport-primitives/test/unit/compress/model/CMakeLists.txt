# Unified compression-metric model tests (issue #693). Pure C++: depends only
# on clio_ctp_compress_model, so it builds/runs regardless of the compressor
# backends (CLIO_CTE_ENABLE_COMPRESS).
project(clio_ctp)

add_executable(test_compress_model_exec
        ${TEST_MAIN}/main.cc
        test_models.cc)

add_dependencies(test_compress_model_exec clio_ctp_compress_model)
target_link_libraries(test_compress_model_exec
        clio_ctp_host clio_ctp_compress_model Catch2::Catch2)

# Real, shipped NeuroPress .nnwt weights (see src/compress/model/weights/) --
# used by the "loads real weights" test so it exercises the actual pretrained
# model rather than a synthetic stand-in.
target_compile_definitions(test_compress_model_exec PRIVATE
        CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CLIO_CTP_ROOT}/src/compress/model/weights/")

add_test(NAME ctp_compress_model COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_compress_model_exec "~[error=FatalError]")

install(TARGETS test_compress_model_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
