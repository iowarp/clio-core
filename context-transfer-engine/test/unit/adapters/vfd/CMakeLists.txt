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
    clio_vfd         # the driver under test
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

# The driver's behaviour when no CLIO runtime is reachable. Separate binary on
# purpose: the suite above starts a runtime before its first case, so it cannot
# express "no runtime" from the inside. No runtime dependencies, no fixtures.
add_executable(clio_cte_vfd_no_runtime_tests
    test_vfd_no_runtime.cc
)

target_include_directories(clio_cte_vfd_no_runtime_tests SYSTEM PUBLIC
    ${HDF5_INCLUDE_DIRS})

target_link_libraries(clio_cte_vfd_no_runtime_tests
    clio_vfd
    clio_cte_cfs_adapter
    clio_cte_filesystem_client
    clio_cte_core_client
    ctp::cxx
    ctp::interceptor
    ${HDF5_LIBRARIES}
)

install(
    TARGETS clio_cte_vfd_unit_tests clio_cte_vfd_no_runtime_tests
    RUNTIME DESTINATION bin
)

if(CLIO_CORE_ENABLE_TESTS)
    add_test(
        NAME clio_cte_vfd_unit_tests
        COMMAND clio_cte_vfd_unit_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(clio_cte_vfd_unit_tests PROPERTIES
        ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
        RESOURCE_LOCK "clio_runtime_port"
        TIMEOUT 120
        LABELS "msan_skip")

    # No runtime, no fixtures, no resource lock -- the point is that this one
    # needs nothing brought up first.
    add_test(
        NAME clio_cte_vfd_no_runtime_tests
        COMMAND clio_cte_vfd_no_runtime_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(clio_cte_vfd_no_runtime_tests PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
        TIMEOUT 120
        LABELS "msan_skip")
endif()
