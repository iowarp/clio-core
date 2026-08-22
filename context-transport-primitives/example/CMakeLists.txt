# Runnable examples. Not tests: nothing here asserts, and nothing is
# registered with ctest -- they exist to be read and run by hand.
project(clio_ctp)

if(NOT CLIO_CORE_ENABLE_CUDA)
    message(STATUS "ctp examples: skipped (CUDA not enabled)")
    return()
endif()

add_executable(neuropress_gpu_inference neuropress_gpu_inference.cu)

target_include_directories(neuropress_gpu_inference PRIVATE
    ${CLIO_CTP_ROOT}/include)

target_compile_definitions(neuropress_gpu_inference PRIVATE
    CTP_ENABLE_NEUROPRESS_GPU=1
    CTP_ENABLE_CUDA=1
    CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CLIO_CTP_ROOT}/src/compress/model/weights")

target_link_libraries(neuropress_gpu_inference clio_ctp_compress_model)
if(TARGET ctp::cuda_cxx)
    target_link_libraries(neuropress_gpu_inference ctp::cuda_cxx)
endif()
