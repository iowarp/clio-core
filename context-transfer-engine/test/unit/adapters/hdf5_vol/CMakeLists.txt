# HDF5 VOL Connector Unit Tests
#
# test_hdf5_vol_adapter exercises the connector's plugin entry points, class
# definition, registration and property-list integration with HDF5. No Clio
# runtime is required (the connector's initialize callback is nullptr), so it
# runs standalone without RESOURCE_LOCK / cleanup fixtures.
#
# test_hdf5_vol_io, added later in this same file, is the opposite: it brings up
# an in-process runtime and DOES need the lock. Do not read the paragraph above
# as applying to every test registered here.
#
# Linking this test also regression-checks issue #518: it consumes
# ${HDF5_LIBRARIES}, which was empty when HDF5 was found in CONFIG mode
# (vcpkg exports only the namespaced hdf5::hdf5-shared target), making the
# connector DLL fail to link with unresolved H5* symbols on Windows.

add_executable(test_hdf5_vol_adapter
    test_hdf5_vol_adapter.cc
)

target_include_directories(test_hdf5_vol_adapter PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/hdf5_vol
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(test_hdf5_vol_adapter
    clio_hdf5_vol
    ctp::cxx
    ${HDF5_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# simple_test.h filters by substring of the registered test name
add_test(NAME cte_hdf5_vol_plugin_entry
    COMMAND test_hdf5_vol_adapter "HDF5 VOL - Plugin Entry Points")

add_test(NAME cte_hdf5_vol_class
    COMMAND test_hdf5_vol_adapter "HDF5 VOL - Connector Class Definition")

add_test(NAME cte_hdf5_vol_register
    COMMAND test_hdf5_vol_adapter "HDF5 VOL - Registration Lifecycle")

add_test(NAME cte_hdf5_vol_fapl
    COMMAND test_hdf5_vol_adapter "HDF5 VOL - File Access Property List")

add_test(NAME cte_hdf5_vol_introspect
    COMMAND test_hdf5_vol_adapter "HDF5 VOL - Introspect Callbacks")

add_test(NAME cte_hdf5_vol_info
    COMMAND test_hdf5_vol_adapter "HDF5 VOL - Info Copy And Free")

set_tests_properties(
    cte_hdf5_vol_plugin_entry
    cte_hdf5_vol_class
    cte_hdf5_vol_register
    cte_hdf5_vol_fapl
    cte_hdf5_vol_introspect
    cte_hdf5_vol_info
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;adapter;hdf5_vol;cte"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
)

#------------------------------------------------------------------------------
# End-to-end I/O test: drives real HDF5 file/dataset/group/attr operations
# through the connector against an in-process Chimaera/CTE runtime. Unlike the
# registration-only suite above, this links the CTE core + admin + bdev
# runtimes so CLIO_INIT(kClient, true) stands up an in-process runtime.
#------------------------------------------------------------------------------
add_executable(test_hdf5_vol_io
    test_hdf5_vol_io.cc
)

target_include_directories(test_hdf5_vol_io PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/hdf5_vol
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(test_hdf5_vol_io
    clio_hdf5_vol
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${HDF5_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# The HDF5-VOL runtime I/O tests are not yet functional on Windows (the VOL
# I/O path fails at runtime there); the test_hdf5_vol_io executable still
# builds on Windows for coverage, but only register/run these on non-Windows
# until the Windows VOL I/O path is ported.
if(NOT WIN32)
  add_test(NAME cte_hdf5_vol_io_dataset
      COMMAND test_hdf5_vol_io "HDF5 VOL IO - Create-Write-Read whole dataset")

  add_test(NAME cte_hdf5_vol_io_groups_attrs
      COMMAND test_hdf5_vol_io "HDF5 VOL IO - Groups and attributes")

  add_test(NAME cte_hdf5_vol_io_passthrough
      COMMAND test_hdf5_vol_io "HDF5 VOL IO - Passthrough callbacks")

  set_tests_properties(
      cte_hdf5_vol_io_dataset
      cte_hdf5_vol_io_groups_attrs
      cte_hdf5_vol_io_passthrough
      PROPERTIES
          TIMEOUT 300
          LABELS "unit;adapter;hdf5_vol;cte"
          # These stand up a real in-process runtime (CLIO_INIT with an
          # embedded server), so they are NOT the lock-free standalone tests
          # this file's header describes -- that text predates test_hdf5_vol_io
          # and only ever applied to test_hdf5_vol_adapter. Every other
          # runtime-using suite takes the lock; this one did not.
          #
          # Hygiene, not a bug fix, and the distinction is worth keeping. An
          # earlier version of this comment claimed the missing lock caused the
          # intermittent 30 s "RouteTask: inline retry exhausted" failure at
          # CLIO_INIT, via two runtimes colliding on port 9413's shared-memory
          # segment name. That was WRONG and is recorded here so it is not
          # re-derived: ctest runs serially (no -j, no CTEST_PARALLEL_LEVEL in
          # CI), so these never overlapped with the VFD suite in the first
          # place, and the missing lock was inert. The failure did not reproduce
          # in 140 attempts with the pre-fix configuration restored.
          #
          # The lock and the distinct port still belong here -- they are what
          # every sibling suite does, and they matter the moment anyone runs
          # ctest with -j -- but nothing here is known to fix that failure.
          RESOURCE_LOCK "clio_runtime"
          # HDF5_USE_FILE_LOCKING=FALSE: these tests close a file and
          # immediately reopen it. H5Fclose() returns success before the
          # OS-level flock is released, so the reopen intermittently fails on
          # macOS with:
          #   H5FD__sec2_lock(): unable to lock file, errno = 35,
          #                      'Resource temporarily unavailable'
          # Each test owns its own .h5 path and is single-writer, so the lock
          # buys nothing here; it only adds a race.
          #
          # CAVEAT: this is a mitigation, not a root-cause fix. The holder of
          # the lingering descriptor was not identified, and one candidate is
          # the clio VOL connector keeping a runtime-side handle open past
          # H5Fclose(). If that is real, disabling the lock hides it rather
          # than fixing it -- worth chasing separately.
          # CLIO_RESTART_LOG is NOT set here: the top-level CMakeLists stamps a
          # per-test WAL path on every test now, so isolation from the shared
          # ~/.clio/restart_log.bin is the default rather than something each
          # suite remembers to ask for. Setting it here too would be a second
          # mechanism for one job.
          ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10620;HDF5_USE_FILE_LOCKING=FALSE"
  )
endif()

install(
    TARGETS test_hdf5_vol_adapter test_hdf5_vol_io
    RUNTIME DESTINATION bin
)

# ---------------------------------------------------------------------------
# Differential HDF5 compatibility suite (Python). The NATIVE stack is the
# oracle; each feature case is written/read through a clio connector and
# compared to native, plus the §1.1 native-compatibility gates (h5diff,
# h5clear+reopen structural validity, and h5dump/h5ls/h5stat output parity).
# Unlike the C++ unit tests above, this exercises real data round-trips end to
# end and REQUIRES a running clio_run — the suite starts/stops its own
# (pkill + start), so it must run alone (RUN_SERIAL).
#
# ONE harness, TWO connectors, but TWO test registrations rather than one run
# with --modes vol,vfd. The connectors fail for unrelated reasons and are owned
# as separate deliverables; a single test would attribute a VFD regression to
# the VOL on the dashboard and would make one connector's failure hide the
# other's result. Each registration also gets its own results JSON.
#
# Both report the real compatibility state rather than masking gaps, so any
# future regression fails the test. Run them for the authoritative,
# always-current status; do not rely on numbers copied into comments/prose.
#
# NEITHER arm carries an --expect-fail. The VOL arm used to carry exactly one,
# for the cache serving a pre-corruption copy of a file damaged behind HDF5's
# back; the coherence stamp closed it and the suite itself said so, printing
# "previously-failing cases now PASS (drop from --expect-fail)". That mechanism
# working as designed is why the allowlist is empty rather than forgotten. If a
# gap is ever added back, name it individually and put its reason in GAP_NOTES.
#
# Finding Python3 is necessary but NOT sufficient: the suite is a differential
# test against a native HDF5 stack and also needs the h5py module plus the HDF5
# CLI tools (h5diff/h5dump/h5ls/h5stat/h5clear/h5cc). Those are absent on some
# CDash builders (e.g. macmini-m1). Rather than gate registration on every
# dependency here, the suite self-detects a missing toolchain and exits
# SKIP_RETURN_CODE, so CDash shows it as "Not Run" instead of a wall of false
# failures. CI, where the toolchain IS guaranteed, additionally asserts that the
# suite did not take the skip path — see ci-adapters.yml.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    # Name kept as-is: CDash history and ci-adapters.yml both key on it.
    add_test(NAME cte_hdf5_vol_compat_suite
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/benchmarks/hdf5-ingest/hdf5_compat_suite.py
            --modes vol
            --bin $<TARGET_FILE_DIR:clio_hdf5_vol>
            --out ${CMAKE_CURRENT_BINARY_DIR}/vol_compat_results.json
    )
    set_tests_properties(cte_hdf5_vol_compat_suite PROPERTIES
        TIMEOUT 900
        RUN_SERIAL TRUE
        RESOURCE_LOCK "clio_runtime"
        SKIP_RETURN_CODE 125
        LABELS "integration;adapter;hdf5_vol;compat;cte"
        ENVIRONMENT "CLIO_HOME_DIR=$ENV{HOME};LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
    )

    # The VFD arm over the same corpus. The clio_vfd target lives in the vfd
    # adapter directory; guard on it so this only registers where it is built.
    if(TARGET clio_vfd)
        add_test(NAME cte_hdf5_vfd_compat_suite
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/benchmarks/hdf5-ingest/hdf5_compat_suite.py
                --modes vfd
                --bin $<TARGET_FILE_DIR:clio_vfd>
                --out ${CMAKE_CURRENT_BINARY_DIR}/vfd_compat_results.json
        )
        set_tests_properties(cte_hdf5_vfd_compat_suite PROPERTIES
            TIMEOUT 900
            RUN_SERIAL TRUE
            RESOURCE_LOCK "clio_runtime"
            SKIP_RETURN_CODE 125
            LABELS "integration;adapter;hdf5_vfd;compat;cte"
            ENVIRONMENT "CLIO_HOME_DIR=$ENV{HOME};LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
        )
    else()
        message(STATUS "hdf5_vol: clio_vfd target absent — skipping VFD compat suite test")
    endif()
else()
    message(STATUS "hdf5_vol: Python3 not found — skipping HDF5 compat suite tests")
endif()
