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
    clio_cte_filesystem_client  # descriptor layer + chimod client
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
    clio_cte_filesystem_client  # descriptor layer + chimod client
    clio_cte_filesystem_client
    clio_cte_core_client
    ctp::cxx
    ctp::interceptor
    ${HDF5_LIBRARIES}
)

if(WIN32)
    foreach(_vfd_test clio_cte_vfd_unit_tests clio_cte_vfd_no_runtime_tests)
        # Same reason the driver needs it: HDF5's public globals (H5T_NATIVE_*_g,
        # H5P_CLS_*_g) are H5_DLLVAR, which only becomes __declspec(dllimport)
        # once the consumer defines this. Functions link either way, so the
        # failure shows up as LNK2019 on the type/class ids alone.
        if(NOT HDF5_USE_STATIC_LIBRARIES)
            target_compile_definitions(${_vfd_test} PRIVATE H5_BUILT_AS_DYNAMIC_LIB)
        endif()
        # The Win32 half of the driver's platform layer. The tests call the same
        # clio_vfd_* shims the driver does (flock/pread have no MSVC spelling),
        # and these are internal glue rather than driver API -- so compile the TU
        # in here rather than widening clio_vfd's exported surface for tests.
        target_sources(${_vfd_test} PRIVATE
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/vfd/H5FDclio_compat_win.cc)
    endforeach()
endif()

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
        # CLIO_RESTART_LOG comes from the top-level per-test isolation default;
        # see the VOL I/O suite's note.
        ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
        RESOURCE_LOCK "clio_runtime"
        # 300 rather than 120, matching the other runtime-based suites here.
        # Warm, this binary runs in ~15s; the first run after a reboot took over
        # 120s and was cut off mid-suite while still making steady progress
        # (cold page-faulting of the runtime's shm segments, plus a virus
        # scanner meeting freshly linked binaries). A timeout that fires on a
        # healthy run reports as a failure with no failing assertion, which is
        # the least actionable signal available -- the ceiling wants to be high
        # enough that TIMEOUT here means genuinely stuck.
        TIMEOUT 300
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
