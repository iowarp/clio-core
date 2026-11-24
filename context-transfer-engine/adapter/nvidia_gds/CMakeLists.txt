# cuFile (GDS) interceptor: bounces each cuFileRead/cuFileWrite through a host
# buffer and pread/pwrite. Host-only code (no kernels), so it is plain CXX
# against the CUDA runtime -- the add_cuda_library + ctp::cudacxx build this
# replaced named a target that does not exist, so enabling the adapter failed
# to configure (#1191). Batch I/O is passed to the real libcufile untouched.
if(NOT CLIO_CORE_ENABLE_CUDA)
  message(STATUS "CTE NVIDIA GDS adapter: skipped (needs CLIO_CORE_ENABLE_CUDA)")
  return()
endif()
find_package(CUDAToolkit REQUIRED)

add_library(clio_cte_nvidia_gds SHARED cufile_api.cc)
target_include_directories(clio_cte_nvidia_gds PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(clio_cte_nvidia_gds PUBLIC
  ctp::cxx
  ctp::interceptor   # RealApi (dlsym of the real libcufile)
  CUDA::cudart)
set_target_properties(clio_cte_nvidia_gds PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

install(TARGETS clio_cte_nvidia_gds
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        )
