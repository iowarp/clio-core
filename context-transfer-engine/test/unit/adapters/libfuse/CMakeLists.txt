# ------------------------------------------------------------------------------
# FUSE Adapter Unit Tests
# ------------------------------------------------------------------------------

message(STATUS "Building FUSE adapter tests")

# Tests do NOT require libfuse3 — they test the CTE helper functions
# (tag lifecycle, implicit directories, page-based I/O) directly.
add_executable(test_fuse_adapter
    test_fuse_adapter.cc
)

target_include_directories(test_fuse_adapter PRIVATE
    ${CLIO_CTE_ROOT}
)

target_link_libraries(test_fuse_adapter
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Tag lifecycle tests
add_test(NAME fuse_cte_tag_exists
    COMMAND test_fuse_adapter "Tag create and exists")
add_test(NAME fuse_cte_tag_delete
    COMMAND test_fuse_adapter "Tag deletion")

# Implicit directory tests
add_test(NAME fuse_cte_dir_exists
    COMMAND test_fuse_adapter "CteDirExists")
add_test(NAME fuse_cte_list_children
    COMMAND test_fuse_adapter "CteListDirectChildren")
add_test(NAME fuse_cte_list_subdirs
    COMMAND test_fuse_adapter "CteListSubdirs")

# Page-based I/O tests
add_test(NAME fuse_cte_small_rw
    COMMAND test_fuse_adapter "Small write and read")
add_test(NAME fuse_cte_multipage
    COMMAND test_fuse_adapter "Multi-page write and read")
add_test(NAME fuse_cte_partial_page
    COMMAND test_fuse_adapter "Partial page write")
add_test(NAME fuse_cte_cross_page
    COMMAND test_fuse_adapter "Cross-page write")

# Integration test
add_test(NAME fuse_integration
    COMMAND test_fuse_adapter "Full file lifecycle")

set_tests_properties(
    fuse_cte_tag_exists
    fuse_cte_tag_delete
    fuse_cte_dir_exists
    fuse_cte_list_children
    fuse_cte_list_subdirs
    fuse_cte_small_rw
    fuse_cte_multipage
    fuse_cte_partial_page
    fuse_cte_cross_page
    fuse_integration
    PROPERTIES
        TIMEOUT 300
        LABELS "unit;adapter;fuse"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
)

# ------------------------------------------------------------------------------
# Copy-workspace test: ingests /workspace into CTE, reads back, verifies
# ------------------------------------------------------------------------------

add_executable(test_fuse_copy_workspace
    test_fuse_copy_workspace.cc
)

target_include_directories(test_fuse_copy_workspace PRIVATE
    ${CLIO_CTE_ROOT}
)

target_link_libraries(test_fuse_copy_workspace
    clio_cte_core_runtime
    clio_cte_core_client
    clio::run::admin_runtime
    clio::run::admin_client
    clio::run::bdev_runtime
    clio::run::bdev_client
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Copy CTE config to build directory
add_custom_command(TARGET test_fuse_copy_workspace POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/cte_config.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/cte_config.yaml
)

add_test(NAME fuse_copy_workspace
    COMMAND test_fuse_copy_workspace "Ingest and verify")

set_tests_properties(
    fuse_copy_workspace
    PROPERTIES
        TIMEOUT 600
        LABELS "unit;adapter;fuse;copy_workspace"
        ENVIRONMENT "CLIO_WITH_RUNTIME=1;CTP_LOG_LEVEL=error;CLIO_SERVER_CONF=${CMAKE_CURRENT_SOURCE_DIR}/cte_config.yaml;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
        RESOURCE_LOCK "clio_runtime"
)

install(TARGETS test_fuse_adapter test_fuse_copy_workspace RUNTIME DESTINATION bin)

# ------------------------------------------------------------------------------
# Callback coverage test: drives the fuse_cte.cc FUSE operation callbacks
# directly (in-process, no real mount) against a live embedded runtime. Only
# built when libfuse3 is available (the callbacks and the cte_fuse_ops table
# need <fuse3/fuse.h>); the adapter binary (clio_cte_fuse) is the signal for
# that having succeeded at configure time.
# ------------------------------------------------------------------------------
if(TARGET clio_cte_fuse)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(FUSE3 QUIET fuse3)
  endif()

  if(FUSE3_FOUND)
    add_executable(test_fuse_ops test_fuse_ops.cc)

    target_compile_definitions(test_fuse_ops PRIVATE
        FUSE_USE_VERSION=35
        CLIO_CTE_FUSE_ENABLED)

    target_include_directories(test_fuse_ops PRIVATE
        ${CLIO_CTE_ROOT}
        ${CLIO_CTE_ROOT}/adapter/libfuse
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/filesystem/include
        ${FUSE3_INCLUDE_DIRS})

    target_link_directories(test_fuse_ops PRIVATE ${FUSE3_LIBRARY_DIRS})
    target_link_libraries(test_fuse_ops
        clio_cte_filesystem_client
        clio_cte_core_runtime
        clio_cte_core_client
        clio::run::admin_runtime
        clio::run::admin_client
        clio::run::bdev_runtime
        clio::run::bdev_client
        ctp::cxx
        ${FUSE3_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT})

    add_custom_command(TARGET test_fuse_ops POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/cte_config.yaml
            ${CMAKE_CURRENT_BINARY_DIR}/cte_config.yaml)

    add_test(NAME fuse_ops COMMAND test_fuse_ops)
    set_tests_properties(fuse_ops PROPERTIES
        TIMEOUT 300
        LABELS "unit;adapter;fuse"
        ENVIRONMENT "CLIO_WITH_RUNTIME=1;CTP_LOG_LEVEL=error;CLIO_SERVER_CONF=${CMAKE_CURRENT_SOURCE_DIR}/cte_config.yaml;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
        RESOURCE_LOCK "clio_runtime")

    install(TARGETS test_fuse_ops RUNTIME DESTINATION bin)
  endif()
endif()

message(STATUS "FUSE adapter tests configured successfully")
