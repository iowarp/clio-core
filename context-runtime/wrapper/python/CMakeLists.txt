cmake_minimum_required(VERSION 3.20)

# Python and nanobind are configured by root CMakeLists.txt via add_subdirectory(external/nanobind)

# Create the Python module
nanobind_add_module(
    chimaera_runtime_ext
    runtime_bindings.cc
)

# Link against the admin client library (provides MonitorTask, admin::Client, chimaera headers)
target_link_libraries(chimaera_runtime_ext PRIVATE chimaera_admin_client)

# Local includes only
target_include_directories(chimaera_runtime_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Ensure C++17 standard
target_compile_features(chimaera_runtime_ext PRIVATE cxx_std_17)

# Install Python module to site-packages within CMAKE_INSTALL_PREFIX
set(PYTHON_SITE_PACKAGES "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
install(
    TARGETS chimaera_runtime_ext
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
)
message(STATUS "Chimaera runtime Python wrapper will be installed to: ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES}")

# Add Python bindings tests
if(BUILD_TESTING)
    add_test(
        NAME test_monitor_python
        COMMAND ${Python_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/test_monitor.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:chimaera_runtime_ext>
    )

    set_tests_properties(
        test_monitor_python
        PROPERTIES
        LABELS "python;bindings;monitor;msan_skip"
        TIMEOUT 120
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
    )

    set_tests_properties(
        test_monitor_python
        PROPERTIES
        DEPENDS chimaera_runtime_ext
    )
endif()
