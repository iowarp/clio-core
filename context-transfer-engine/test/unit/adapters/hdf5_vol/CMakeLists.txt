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
    iowarp_hdf5_vol
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
# runtimes so CHIMAERA_INIT(kClient, true) stands up an in-process runtime.
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
    iowarp_hdf5_vol
    clio_cte_core_runtime
    clio_cte_core_client
    chimaera::admin_runtime
    chimaera::admin_client
    chimaera::bdev_runtime
    chimaera::bdev_client
    ctp::cxx
    ${HDF5_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

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

install(
    TARGETS test_hdf5_vol_adapter test_hdf5_vol_io
    RUNTIME DESTINATION bin
)
