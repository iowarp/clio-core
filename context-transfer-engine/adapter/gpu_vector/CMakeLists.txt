#------------------------------------------------------------------------------
# gpu_vector adapter: header-only device-paged vector over the CTE.
#------------------------------------------------------------------------------

add_library(clio_cte_gpu_vector INTERFACE)
add_library(clio::cte::gpu_vector ALIAS clio_cte_gpu_vector)

target_include_directories(clio_cte_gpu_vector INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Vector::Copy creates the checkpoint fault-handler pool.
target_link_libraries(clio_cte_gpu_vector INTERFACE
    clio_cte_core_client
    clio_cte_checkpoint_client
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include)

# Each workload's CMakeLists guards its own targets, so a SYCL build only
# gets the ones that have been ported. The rest are CUDA sources -- __global__
# functions launched with <<<>>> -- and sycl_cuda_compat.h making them PARSE
# under SYCL is worse than useless: they would link against libsycl and then
# launch nothing, because a __global__ function is not a SYCL kernel.
add_subdirectory(benchmark)

if(CLIO_CORE_ENABLE_TESTS)
    if(CLIO_CORE_ENABLE_CUDA OR CLIO_CORE_ENABLE_ROCM)
        add_subdirectory(test)
    endif()
    # SYCL kernels driving the same DeviceVector. Its own directory because
    # SYCL kernel bodies are lambdas rather than __global__ functions, so the
    # test SOURCES cannot be shared even though every header they use is.
    add_subdirectory(test/sycl)
endif()
