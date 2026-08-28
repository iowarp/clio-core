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

add_subdirectory(benchmark)

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
