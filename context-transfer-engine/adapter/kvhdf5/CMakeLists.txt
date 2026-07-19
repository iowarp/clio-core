#-----------------------------------------------------------------------------
# kvhdf5 adapter — header-only HDF5-like API over the CTE key-value store.
#
# The device-facing producer surface (GpuCteDataset, GpuDatasetHandle, chunking,
# tag_path) lives entirely in include/clio_cte/kvhdf5/*.h and is instantiated by
# the user's CUDA translation unit. This INTERFACE library carries the include
# path and the transitive link to the producer-only clio_run GPU stack
# (clio_run_cxx + clio_run_cxx_gpu + clio_cte_core_client), mirroring the
# gpu_vector adapter. Only added when CUDA is enabled (see adapter/CMakeLists.txt);
# the headers pull in cuda/std + the clio GPU runtime.
#-----------------------------------------------------------------------------

add_library(clio_cte_kvhdf5 INTERFACE)

target_include_directories(clio_cte_kvhdf5 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(clio_cte_kvhdf5 INTERFACE
    clio_run_cxx
    clio_run_cxx_gpu
    clio_admin_client
    clio_cte_core_client
)

add_library(clio::cte::kvhdf5 ALIAS clio_cte_kvhdf5)

#-----------------------------------------------------------------------------
# Tests — CPU unit tests + the clio contract canary + kvhdf5 GPU e2e tests.
#-----------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()

#-----------------------------------------------------------------------------
# Install headers
#-----------------------------------------------------------------------------
install(DIRECTORY include/clio_cte/kvhdf5
    DESTINATION include/clio_cte
    COMPONENT headers
    FILES_MATCHING PATTERN "*.h"
)
