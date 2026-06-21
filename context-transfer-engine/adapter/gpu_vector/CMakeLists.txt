#-----------------------------------------------------------------------------
# GPU Vector adapter — header-only library.
#
# All kernel launches and device functions live in
# include/clio_cte/gpu_vector/*.h and are instantiated by the user's
# CUDA / HIP TU. This INTERFACE library carries the include path and
# transitive link to the producer-only clio_run GPU stack
# (clio_run_cxx + clio_run_cxx_gpu + clio_cte_core_client).
#-----------------------------------------------------------------------------

add_library(clio_cte_gpu_vector INTERFACE)

target_include_directories(clio_cte_gpu_vector INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(clio_cte_gpu_vector INTERFACE
    clio_run_cxx
    clio_run_cxx_gpu
    clio_admin_client
    clio_cte_core_client
)

add_library(clio::cte::gpu_vector ALIAS clio_cte_gpu_vector)

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()

#-----------------------------------------------------------------------------
# Benchmarks
#-----------------------------------------------------------------------------
add_subdirectory(benchmark)

#-----------------------------------------------------------------------------
# Install headers
#-----------------------------------------------------------------------------
install(DIRECTORY include/clio_cte/gpu_vector
    DESTINATION include/clio_cte
    COMPONENT headers
    FILES_MATCHING PATTERN "*.h"
)
