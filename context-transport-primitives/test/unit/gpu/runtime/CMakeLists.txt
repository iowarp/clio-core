#------------------------------------------------------------------------------
# GPU Dynamic Runtime Virtual Dispatch Test
#------------------------------------------------------------------------------

# Shared CUDA library (dynamically loaded)
add_cuda_library(gpu_runtime_lib SHARED TRUE lib.cc)
target_link_libraries(gpu_runtime_lib hshm::cuda_cxx)
target_include_directories(gpu_runtime_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS gpu_runtime_lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

# Test executable
add_cuda_executable(test_gpu_runtime TRUE main.cc)
target_link_libraries(test_gpu_runtime hshm::cuda_cxx ${CMAKE_DL_LIBS})
target_include_directories(test_gpu_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_gpu_runtime PRIVATE
    GPU_RUNTIME_LIB_PATH="$<TARGET_FILE:gpu_runtime_lib>")
set_target_properties(test_gpu_runtime PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared)
add_dependencies(test_gpu_runtime gpu_runtime_lib)
add_test(NAME test_gpu_runtime COMMAND test_gpu_runtime)
install(TARGETS test_gpu_runtime RUNTIME DESTINATION bin)
