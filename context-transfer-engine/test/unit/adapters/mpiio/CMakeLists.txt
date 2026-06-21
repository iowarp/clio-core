# MPI-IO Adapter Unit Tests
#
# This test exercises the WRP CTE MPI-IO adapter via direct linking
# (no LD_PRELOAD). It only builds when MPI is enabled.

if(NOT CLIO_CORE_ENABLE_MPI)
    return()
endif()

include_directories(
    ${CLIO_CTE_ROOT}
    .
)

add_executable(clio_cte_mpiio_unit_tests
    test_mpiio_adapter.cc
)

target_link_libraries(clio_cte_mpiio_unit_tests
    clio_cte_mpiio
    clio_cte_cfs_adapter      # cfs-backed I/O core (context-filesystem chimod)
    clio_cte_filesystem_client
    clio_cte_core_client
    MPI::MPI_CXX
    ctp::cxx
    ctp::interceptor
    Catch2::Catch2WithMain
)

add_dependencies(clio_cte_mpiio_unit_tests
    clio_cte_core_runtime
    clio_cte_filesystem_runtime
    clio_bdev_runtime
    clio_admin_runtime)

install(
    TARGETS clio_cte_mpiio_unit_tests
    RUNTIME DESTINATION bin
)

if(CLIO_CORE_ENABLE_TESTS)
    add_test(
        NAME clio_cte_mpiio_unit_tests
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:clio_cte_mpiio_unit_tests>
                ${MPIEXEC_POSTFLAGS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(clio_cte_mpiio_unit_tests PROPERTIES
        ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
        RESOURCE_LOCK "clio_runtime_port"
        TIMEOUT 120
        LABELS "msan_skip")
endif()
