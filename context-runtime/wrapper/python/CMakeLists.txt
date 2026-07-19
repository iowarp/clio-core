cmake_minimum_required(VERSION 3.20)

# Python and nanobind are configured by root CMakeLists.txt via add_subdirectory(external/nanobind)

# Create the Python module
nanobind_add_module(
    clio_runtime_ext
    runtime_bindings.cc
)

# Link against the admin client library (provides MonitorTask, admin::Client, clio_run headers)
# and the safe_bdev client (AddBdev/RemoveBdev/RecoverBdev for dashboard controls).
target_link_libraries(clio_runtime_ext PRIVATE clio_admin_client clio_safe_bdev_client)

# Local includes only
target_include_directories(clio_runtime_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Ensure C++17 standard
target_compile_features(clio_runtime_ext PRIVATE cxx_std_17)

# Install Python module to site-packages (PYTHON_SITE_PACKAGES set by root CMakeLists.txt)
install(
    TARGETS clio_runtime_ext
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
)
message(STATUS "Clio runtime Python wrapper will be installed to: ${PYTHON_SITE_PACKAGES}")

# Add Python bindings tests
if(BUILD_TESTING)
    add_test(
        NAME test_monitor_python
        COMMAND ${Python_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/test_monitor.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:clio_runtime_ext>
    )

    set_tests_properties(
        test_monitor_python
        PROPERTIES
        LABELS "python;bindings;monitor;msan_skip"
        TIMEOUT 120
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    )

    set_tests_properties(
        test_monitor_python
        PROPERTIES
        DEPENDS clio_runtime_ext
    )
endif()
