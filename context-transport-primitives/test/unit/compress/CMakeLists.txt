project(clio_ctp)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------

# Basic compression unit tests
add_executable(test_compress_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_compress.cc)

# Note: Unified compression benchmark (benchmark_compression_unified_exec) has been removed
# Note: AI model predictor tests (Q-Table, LinReg, Distribution, Contention)
# have been moved to context-transfer-engine/compressor/test

#------------------------------------------------------------------------------
# Link Dependencies
#------------------------------------------------------------------------------

add_dependencies(test_compress_exec clio_ctp_host)
target_link_libraries(test_compress_exec
        clio_ctp_host ctp::compress Catch2::Catch2)
target_compile_definitions(test_compress_exec
        PRIVATE CTP_ENABLE_COMPRESS=1)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

add_test(NAME ctp_compress COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_compress_exec "~[error=FatalError]")

# Device quantizer: needs a GPU, so registered but labelled.
if(CLIO_CORE_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(test_quantize_device_exec
            ${TEST_MAIN}/main.cc
            test_init.cc
            test_quantize_device.cc)
    target_link_libraries(test_quantize_device_exec
            clio_ctp_host ctp::compress CUDA::cudart Catch2::Catch2)
    add_test(NAME ctp_quantize_device COMMAND
            ${CMAKE_BINARY_DIR}/bin/test_quantize_device_exec)
    set_tests_properties(ctp_quantize_device PROPERTIES LABELS "gpu")

    # Look-ahead (temporal) codec: needs a GPU, cuSZp for anchors and nvcomp
    # for the residual packer.
    add_executable(test_lookahead_gpu_exec
            ${TEST_MAIN}/main.cc
            test_init.cc
            test_lookahead_gpu.cc)
    target_link_libraries(test_lookahead_gpu_exec
            clio_ctp_host ctp::compress CUDA::cudart Catch2::Catch2)
    add_test(NAME ctp_lookahead_gpu COMMAND
            ${CMAKE_BINARY_DIR}/bin/test_lookahead_gpu_exec)
    set_tests_properties(ctp_lookahead_gpu PROPERTIES LABELS "gpu")

    # NeuroPress v2 predictor: loading, host network vs the Python model,
    # GPU features/ranking vs the host, online learning.
    add_executable(test_neuropress_v2_exec
            ${TEST_MAIN}/main.cc
            test_init.cc
            test_neuropress_v2.cc)
    target_link_libraries(test_neuropress_v2_exec
            clio_ctp_host ctp::compress CUDA::cudart Catch2::Catch2)
    target_compile_definitions(test_neuropress_v2_exec PRIVATE
            CLIO_CTP_NEUROPRESS_V2_WEIGHTS="${CLIO_CTP_ROOT}/src/compress/model/weights/v2")
    add_test(NAME ctp_neuropress_v2 COMMAND
            ${CMAKE_BINARY_DIR}/bin/test_neuropress_v2_exec)
    set_tests_properties(ctp_neuropress_v2 PROPERTIES LABELS "gpu")
endif()

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------

install(TARGETS
        test_compress_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#------------------------------------------------------------------------------
