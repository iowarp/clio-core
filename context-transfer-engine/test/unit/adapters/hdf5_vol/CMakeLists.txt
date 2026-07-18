# HDF5 VOL Connector Unit Tests
#
# Exercises the connector's plugin entry points, class definition,
# registration and property-list integration with HDF5. No Clio runtime
# is required (the connector's initialize callback is nullptr), so these
# tests run standalone without RESOURCE_LOCK / cleanup fixtures.
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
          ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1"
  )
endif()

install(
    TARGETS test_hdf5_vol_adapter test_hdf5_vol_io
    RUNTIME DESTINATION bin
)

# ---------------------------------------------------------------------------
# Differential VOL compatibility suite (Python). Native VOL is the oracle;
# each feature case is written/read through the clio VOL and compared to
# native, plus h5diff/h5dump. Unlike the C++ unit tests above, this exercises
# real data round-trips end to end and REQUIRES a running clio_run — the suite
# starts/stops its own (pkill + start), so it must run alone (RUN_SERIAL).
#
# Runs in HONEST mode (no --expect-fail allowlist): it reports the VOL's real
# compatibility state rather than masking gaps, so any future regression fails the
# test. Run it for the authoritative, always-current status; do not rely on numbers
# copied into comments/prose.
#
# Finding Python3 is necessary but NOT sufficient: the suite is a differential
# test against a native HDF5 stack and also needs the h5py module plus the HDF5
# CLI tools (h5diff/h5dump/h5ls/h5cc). Those are absent on some CDash builders
# (e.g. macmini-m1). Rather than gate registration on every dependency here, the
# suite self-detects a missing toolchain and exits SKIP_RETURN_CODE, so CDash
# shows it as "Not Run" instead of a wall of false failures.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    add_test(NAME cte_hdf5_vol_compat_suite
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/benchmarks/hdf5-ingest/vol_compat_suite.py
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
else()
    message(STATUS "hdf5_vol: Python3 not found — skipping VOL compat suite test")
endif()
