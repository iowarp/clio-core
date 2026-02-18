cmake_minimum_required(VERSION 3.20)

# UVM module: Software-managed GPU demand paging
# Only builds when CUDA support is enabled
if(NOT WRP_CORE_ENABLE_CUDA)
    message(STATUS "CTE UVM: Skipping (WRP_CORE_ENABLE_CUDA is OFF)")
    return()
endif()

message(STATUS "CTE UVM: Building GPU virtual memory manager (CUDA enabled)")

# Library: wrp_cte_uvm
add_cuda_library(wrp_cte_uvm SHARED TRUE src/gpu_vmm.cu)
target_include_directories(wrp_cte_uvm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(wrp_cte_uvm PUBLIC cuda)

install(TARGETS wrp_cte_uvm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/wrp_cte/uvm
    DESTINATION include/wrp_cte
    FILES_MATCHING PATTERN "*.h"
)

# Test executable
if(WRP_CORE_ENABLE_TESTS)
    add_cuda_executable(test_gpu_vmm TRUE test/test_gpu_vmm.cu)
    target_link_libraries(test_gpu_vmm PRIVATE wrp_cte_uvm cuda)
    add_test(NAME test_gpu_vmm COMMAND test_gpu_vmm)
    message(STATUS "CTE UVM: Test target 'test_gpu_vmm' added")
endif()
