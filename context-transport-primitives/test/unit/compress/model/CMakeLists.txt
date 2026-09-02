# Unified compression-metric model tests (issue #693). Pure C++: depends only
# on clio_ctp_compress_model, so it builds/runs regardless of the compressor
# backends (CLIO_CTE_ENABLE_COMPRESS).
project(clio_ctp)

add_executable(test_compress_model_exec
        ${TEST_MAIN}/main.cc
        test_models.cc
        test_data_stats_gpu.cc
        test_block_evolution.cc
        test_compress_factory_gpu.cc)

add_dependencies(test_compress_model_exec clio_ctp_compress_model)
target_link_libraries(test_compress_model_exec
        clio_ctp_host clio_ctp_compress_model Catch2::Catch2)

# This TU is pure C++ and does not use CUDA itself, but clio_ctp.h
# unconditionally includes <cuda_runtime_api.h> whenever the project-wide
# CTP_ENABLE_CUDA is on (macros.h) -- without this, the include fails as
# soon as any build enables CUDA anywhere, even though this target never
# calls into it.
if(CLIO_CTP_ENABLE_CUDA AND TARGET ctp::cuda_cxx)
    target_link_libraries(test_compress_model_exec ctp::cuda_cxx)
endif()

# Real, shipped NeuroPress .nnwt weights (see src/compress/model/weights/) --
# used by the "loads real weights" test so it exercises the actual pretrained
# model rather than a synthetic stand-in.
target_compile_definitions(test_compress_model_exec PRIVATE
        CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CLIO_CTP_ROOT}/src/compress/model/weights/")

add_test(NAME ctp_compress_model COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_compress_model_exec "~[error=FatalError]")

# The adaptive online-SGD rule is env-gated and its knobs are parsed once per
# process, so the case that exercises it is hidden ([.np_adaptive]) and gets
# its own run of the same binary with the rule selected.
add_test(NAME ctp_neuropress_adaptive_sgd COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_compress_model_exec "[.np_adaptive]")
set_tests_properties(ctp_neuropress_adaptive_sgd PROPERTIES
        ENVIRONMENT "CLIO_NEUROPRESS_SGD_RULE=adaptive")

install(TARGETS test_compress_model_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

# Cross-project NeuroPress differential parity, skipped when the tree is
# absent -- it is not part of this repository.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/parity/CMakeLists.txt")
    add_subdirectory(parity)
endif()
