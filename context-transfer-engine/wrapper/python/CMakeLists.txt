cmake_minimum_required(VERSION 3.20)

# Python and nanobind are configured by root CMakeLists.txt via add_subdirectory(external/nanobind)

# Create the Python module
nanobind_add_module(
    clio_cte_core_ext
    core_bindings.cc
)

# Link against the core client library
target_link_libraries(clio_cte_core_ext PRIVATE clio_cte_core_client)

# Note: core/include is provided transitively via clio_cte_core_client
# Note: nanobind includes are provided via nanobind::module target
# Local includes only
target_include_directories(clio_cte_core_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For local headers in wrapper/python
)

# Ensure C++17 standard
target_compile_features(clio_cte_core_ext PRIVATE cxx_std_17)

# Install Python module to site-packages (PYTHON_SITE_PACKAGES set by root CMakeLists.txt)
install(
    TARGETS clio_cte_core_ext
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
)
message(STATUS "CTE Python wrapper will be installed to: ${PYTHON_SITE_PACKAGES}")

# Generate stub file for type hints
# Note: Stub generation is disabled because the module requires runtime initialization
# (CLIO Runtime runtime, singletons, etc.) before it can be imported. The nanobind stub
# generator imports the module, which fails with std::bad_cast when singletons aren't
# initialized. Users should rely on the inline documentation and Python type checkers
# that can work without stubs.
# nanobind_add_stub(
#     clio_cte_core_ext_stub
#     MODULE clio_cte_core_ext
#     OUTPUT core_stub.pyi
#     PYTHON_PATH $<TARGET_FILE_DIR:clio_cte_core_ext>
#     DEPENDS clio_cte_core_ext
# )

# Install the stub file alongside the module (renamed to match module)
# Disabled along with stub generation above
# install(
#     FILES ${CMAKE_CURRENT_BINARY_DIR}/core_stub.pyi
#     DESTINATION lib
#     RENAME clio_cte_core_ext.pyi
#     OPTIONAL
# )

# Add Python bindings tests
# NOTE: These tests may conflict with system-installed versions of the module.
# If you see "refusing to add duplicate key" errors, uninstall the system version:
#   sudo rm /usr/local/lib/clio_cte_core_ext*.so
# Or run tests after: make uninstall
if(BUILD_TESTING)
    # Comprehensive Python bindings test (includes all bindings and optional runtime initialization)
    # Use -I flag to run in isolated mode (ignore user site-packages)
    add_test(
        NAME python_bindings_test
        COMMAND ${Python_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/test_bindings.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:clio_cte_core_ext>
    )

    # Set environment variable to enable runtime initialization
    # This allows tests to run with full CLIO Runtime runtime setup
    set_tests_properties(
        python_bindings_test
        PROPERTIES
        LABELS "python;bindings"
        TIMEOUT 120
        ENVIRONMENT "CHI_WITH_RUNTIME=1"
    )

    # Ensure test depends on the built module
    set_tests_properties(
        python_bindings_test
        PROPERTIES
        DEPENDS clio_cte_core_ext
    )

    # PollTelemetryLog focused test (requires runtime initialization).
    # Requires CLIO_WITH_RUNTIME=1 and CLIO_SERVER_CONF set.
    # NOTE: test reads the CLIO_-prefixed names verbatim (not GetCompat),
    # so the env vars below MUST use the CLIO_ spelling or the test
    # will fast-fail with "CLIO_SERVER_CONF not set".
    add_test(
        NAME python_telemetry_test
        COMMAND ${Python_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/test_cte_telemetry.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:clio_cte_core_ext>
    )

    # Use the same ADIOS2 adapter config (it already has a clio_cte_core
    # compose entry with a RAM storage tier so PutBlob/GetBlob work).
    # LD_LIBRARY_PATH is left empty here so the loader falls back to
    # the .so's rpath ($ORIGIN); previous attempts to pin it to a
    # conda prefix broke the import outright on systems without that
    # prefix.
    set_tests_properties(
        python_telemetry_test
        PROPERTIES
        LABELS "python;bindings;telemetry"
        TIMEOUT 180
        ENVIRONMENT "CLIO_WITH_RUNTIME=1;CLIO_SERVER_CONF=${CMAKE_SOURCE_DIR}/context-transfer-engine/test/unit/adapters/adios2/cte_config.yaml;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    )

    # Ensure test depends on the built module
    set_tests_properties(
        python_telemetry_test
        PROPERTIES
        DEPENDS clio_cte_core_ext
    )

    # SemanticSearch focused test — exercises the Client.SemanticSearch
    # binding end-to-end: puts 10 documents, runs BM25, verifies the
    # returned list[SemanticSearchResult] is sorted and the relevant
    # docs make the top-k.
    add_test(
        NAME python_semantic_search_test
        COMMAND ${Python_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/test_cte_semantic_search.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:clio_cte_core_ext>
    )
    set_tests_properties(
        python_semantic_search_test
        PROPERTIES
        LABELS "python;bindings;semantic-search"
        TIMEOUT 120
        ENVIRONMENT "CHI_WITH_RUNTIME=1;CLIO_BIND_ADDR=127.0.0.1"
        DEPENDS clio_cte_core_ext
    )

    # TemporalSearch focused test — puts blobs in three time-stamped groups,
    # then exercises windows, open bounds, max_entries cap, and regex filtering.
    add_test(
        NAME python_temporal_search_test
        COMMAND ${Python_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/test_cte_temporal_search.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:clio_cte_core_ext>
    )
    set_tests_properties(
        python_temporal_search_test
        PROPERTIES
        LABELS "python;bindings;temporal-search"
        TIMEOUT 120
        ENVIRONMENT "CHI_WITH_RUNTIME=1;CLIO_BIND_ADDR=127.0.0.1"
        DEPENDS clio_cte_core_ext
    )

    # GetBlob binary round-trip — regression guard for the nb::bytes return
    # type (a std::string return decoded non-UTF-8 blob data and raised
    # UnicodeDecodeError).
    add_test(
        NAME python_getblob_bytes_test
        COMMAND ${Python_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/test_cte_getblob_bytes.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:clio_cte_core_ext>
    )
    set_tests_properties(
        python_getblob_bytes_test
        PROPERTIES
        LABELS "python;bindings;getblob"
        TIMEOUT 120
        ENVIRONMENT "CHI_WITH_RUNTIME=1;CLIO_BIND_ADDR=127.0.0.1"
        DEPENDS clio_cte_core_ext
    )
endif()
