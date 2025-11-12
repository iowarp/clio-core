# POSIX Adapter Unit Tests

include_directories(
    ${CMAKE_SOURCE_DIR}
    .
)

# Create the POSIX adapter unit test
add_executable(wrp_cte_posix_unit_tests
    test_posix_adapter.cc
)

target_link_libraries(wrp_cte_posix_unit_tests
    MPI::MPI_CXX
    wrp_cte_posix
    wrp_cte_fs_base
    wrp_cte_core_client
    wrp_cte_cae_config
    ${HSHM_TARGET}
    Catch2::Catch2WithMain
)

# Install the test executable
install(
    TARGETS wrp_cte_posix_unit_tests
    RUNTIME DESTINATION bin
)