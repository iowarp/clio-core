include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_cuda_library(wrp_cte_nvidia_gds SHARED TRUE cufile_api.cc)
target_link_libraries(wrp_cte_nvidia_gds PUBLIC hshm::cudacxx MPI::MPI_CXX)

install(TARGETS wrp_cte_nvidia_gds
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        )