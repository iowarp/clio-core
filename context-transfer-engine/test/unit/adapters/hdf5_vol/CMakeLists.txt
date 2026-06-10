# HDF5 VOL Connector Unit Tests
#
# Exercises the connector's plugin entry points, class definition,
# registration and property-list integration with HDF5. No Chimaera runtime
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

install(
    TARGETS test_hdf5_vol_adapter
    RUNTIME DESTINATION bin
)
