# HDF5 VFD adapter smoke test (requires HDF5 >= 1.14, same as the VFD).
find_package(HDF5 REQUIRED COMPONENTS C)

include_directories(
    ${CLIO_CTE_ROOT}
    .
)

add_executable(clio_cte_vfd_unit_tests
    test_vfd_adapter.cc
)

target_include_directories(clio_cte_vfd_unit_tests SYSTEM PUBLIC
    ${HDF5_INCLUDE_DIRS})

target_link_libraries(clio_cte_vfd_unit_tests
    clio_cte_hdf5_vfd         # the driver under test
    clio_cte_cfs_adapter
    clio_cte_filesystem_client
    clio_cte_core_client
    ctp::cxx
    ctp::interceptor
    ${HDF5_LIBRARIES}
)

add_dependencies(clio_cte_vfd_unit_tests
    clio_cte_core_runtime
    clio_cte_filesystem_runtime
    clio_bdev_runtime
    clio_admin_runtime)

install(
    TARGETS clio_cte_vfd_unit_tests
    RUNTIME DESTINATION bin
)

if(CLIO_CORE_ENABLE_TESTS)
    add_test(
        NAME clio_cte_vfd_unit_tests
        COMMAND clio_cte_vfd_unit_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(clio_cte_vfd_unit_tests PROPERTIES
        ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
        RESOURCE_LOCK "chimaera_runtime_port"
        TIMEOUT 120
        LABELS "msan_skip")
endif()
