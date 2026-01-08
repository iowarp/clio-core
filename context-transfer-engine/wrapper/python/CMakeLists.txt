cmake_minimum_required(VERSION 3.20)

# Find Python for nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Create the Python module
nanobind_add_module(
    wrp_cte_core_ext
    core_bindings.cc
)

# Link against the core client library
target_link_libraries(wrp_cte_core_ext PRIVATE wrp_cte_core_client)

# Note: core/include is provided transitively via wrp_cte_core_client
# Note: nanobind includes are provided via nanobind::module target
# Local includes only
target_include_directories(wrp_cte_core_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For local headers in wrapper/python
)

# Ensure C++17 standard
target_compile_features(wrp_cte_core_ext PRIVATE cxx_std_17)

# Install Python module to site-packages within CMAKE_INSTALL_PREFIX
# For conda: This will be ${CMAKE_INSTALL_PREFIX}/lib/pythonX.Y/site-packages
set(PYTHON_SITE_PACKAGES "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
install(
    TARGETS wrp_cte_core_ext
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
)
message(STATUS "CTE Python wrapper will be installed to: ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES}")

# Generate stub file for type hints
# Note: Stub generation is disabled because the module requires runtime initialization
# (Chimaera runtime, singletons, etc.) before it can be imported. The nanobind stub
# generator imports the module, which fails with std::bad_cast when singletons aren't
# initialized. Users should rely on the inline documentation and Python type checkers
# that can work without stubs.
# nanobind_add_stub(
#     wrp_cte_core_ext_stub
#     MODULE wrp_cte_core_ext
#     OUTPUT core_stub.pyi
#     PYTHON_PATH $<TARGET_FILE_DIR:wrp_cte_core_ext>
#     DEPENDS wrp_cte_core_ext
# )

# Install the stub file alongside the module (renamed to match module)
# Disabled along with stub generation above
# install(
#     FILES ${CMAKE_CURRENT_BINARY_DIR}/core_stub.pyi
#     DESTINATION lib
#     RENAME wrp_cte_core_ext.pyi
#     OPTIONAL
# )

# Add Python bindings tests
# NOTE: These tests may conflict with system-installed versions of the module.
# If you see "refusing to add duplicate key" errors, uninstall the system version:
#   sudo rm /usr/local/lib/wrp_cte_core_ext*.so
# Or run tests after: make uninstall
if(BUILD_TESTING)
    # Comprehensive Python bindings test (includes all bindings and optional runtime initialization)
    # Use -I flag to run in isolated mode (ignore user site-packages)
    add_test(
        NAME python_bindings_test
        COMMAND ${Python_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/test_bindings.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:wrp_cte_core_ext>
    )

    # Set environment variable to skip runtime initialization by default
    # This allows tests to pass even without full Chimaera runtime setup
    set_tests_properties(
        python_bindings_test
        PROPERTIES
        LABELS "python;bindings"
        TIMEOUT 120
        ENVIRONMENT "CHIMAERA_WITH_RUNTIME=0"
    )

    # Ensure test depends on the built module
    set_tests_properties(
        python_bindings_test
        PROPERTIES
        DEPENDS wrp_cte_core_ext
    )

    # PollTelemetryLog focused test (requires runtime initialization)
    # This test specifically validates telemetry logging functionality
    # Requires: CHIMAERA_WITH_RUNTIME=1 and WRP_RUNTIME_CONF set
    add_test(
        NAME python_telemetry_test
        COMMAND ${Python_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/test_cte_telemetry.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:wrp_cte_core_ext>
    )

    # Configure telemetry test with runtime initialization
    # Uses the same config as ADIOS2 adapter tests
    # CRITICAL: Put conda libs FIRST in LD_LIBRARY_PATH to avoid OpenSSL version conflicts
    set_tests_properties(
        python_telemetry_test
        PROPERTIES
        LABELS "python;bindings;telemetry"
        TIMEOUT 180
        ENVIRONMENT "CHIMAERA_WITH_RUNTIME=1;WRP_RUNTIME_CONF=${CMAKE_SOURCE_DIR}/context-transfer-engine/test/unit/adapters/adios2/cte_config.yaml;LD_LIBRARY_PATH=/home/iowarp/miniconda3/lib"
    )

    # Ensure test depends on the built module
    set_tests_properties(
        python_telemetry_test
        PROPERTIES
        DEPENDS wrp_cte_core_ext
    )
endif()