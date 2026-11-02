#------------------------------------------------------------------------------
# GPU Dynamic Runtime Virtual Dispatch Test
#------------------------------------------------------------------------------

# Shared CUDA library (dynamically loaded)
add_cuda_library(gpu_runtime_lib SHARED TRUE lib.cc
  INCLUDE_DIRS
    ${CLIO_CTP_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/include
)
target_link_libraries(gpu_runtime_lib ctp::cuda_cxx)
target_include_directories(gpu_runtime_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS gpu_runtime_lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

# Test executable
# lib.cc is compiled directly into the executable so CUDA virtual dispatch
# works correctly. CUDA does not support cross-dlopen virtual dispatch for
# device code; the vtable for Sum::Run must be in the same linked image.
add_cuda_executable(test_gpu_runtime TRUE main.cc lib.cc)
target_link_libraries(test_gpu_runtime ctp::cuda_cxx)
target_include_directories(test_gpu_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(test_gpu_runtime PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared)
add_test(NAME test_gpu_runtime COMMAND test_gpu_runtime)
set_tests_properties(test_gpu_runtime PROPERTIES TIMEOUT 60)
install(TARGETS test_gpu_runtime RUNTIME DESTINATION bin)
