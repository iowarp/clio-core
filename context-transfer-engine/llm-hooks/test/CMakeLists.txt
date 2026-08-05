cmake_minimum_required(VERSION 3.20)

# KV cache round-trip test
add_executable(test_iowarp_kvcache test_kvcache.cc)
target_link_libraries(test_iowarp_kvcache PRIVATE clio_llm_kvcache Catch2::Catch2WithMain)
add_test(NAME test_iowarp_kvcache COMMAND test_iowarp_kvcache)

# Weight offloading smoke test. Gated on the TARGET, not on CLIO_CORE_ENABLE_CUDA:
# the legacy GpuVMM backend it exercises lives in weights/, which is only added
# when CLIO_CORE_ENABLE_LLM_WEIGHTS_GPUVMM=ON (default OFF — superseded by the
# gpu_vector read_range path). With CUDA on and that option off, the old guard
# still declared this test, and linking a non-existent clio_llm_weights target
# propagated no include dirs, so the build failed on a missing
# clio_llm/weights/weight_manager.h.
if(TARGET clio_llm_weights)
    add_executable(test_iowarp_weights test_weights.cc)
    target_link_libraries(test_iowarp_weights PRIVATE clio_llm_weights Catch2::Catch2WithMain)
    add_test(NAME test_iowarp_weights COMMAND test_iowarp_weights)
endif()
