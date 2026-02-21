# ------------------------------------------------------------------------------
# ADIOS2 Adapter Unit Tests
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.10)

# Find ADIOS2 (only if not already found)
if(NOT ADIOS2_FOUND)
  find_package(ADIOS2 QUIET)
  if(NOT ADIOS2_FOUND)
    message(WARNING "ADIOS2 not found. ADIOS2 adapter tests will not be built.")
    return()
  endif()
endif()

message(STATUS "Building ADIOS2 adapter tests")

# Create the ADIOS2 adapter test executable
add_executable(test_adios2_adapter
    test_adios2_adapter.cc
)

# Include directories
target_include_directories(test_adios2_adapter PRIVATE
  ${CMAKE_SOURCE_DIR}/test  # For unified simple_test.h
  ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/adios2  # For iowarp_engine.h
  ${ADIOS2_INCLUDE_DIR}
)

# Link against required libraries
target_link_libraries(test_adios2_adapter
    iowarp_engine                # IOWarp ADIOS2 engine adapter
    wrp_cte_core_runtime         # CTE core runtime library
    wrp_cte_core_client          # CTE core client library
    chimaera::admin_runtime      # Admin module runtime (required for runtime mode)
    chimaera::admin_client       # Admin module client
    chimaera::bdev_runtime       # Bdev module runtime (required for runtime mode)
    chimaera::bdev_client        # Bdev client
    hshm::cxx                    # HermesShm library
    adios2::adios2               # ADIOS2 library
    ${CMAKE_THREAD_LIBS_INIT}    # Threading support
)

# Add individual test cases to CTest
# Note: simple_test.h filters by test NAME, not tags
add_test(NAME adios2_adapter_init
    COMMAND test_adios2_adapter "ADIOS2 Engine Initialization")

add_test(NAME adios2_adapter_step
    COMMAND test_adios2_adapter "ADIOS2 BeginStep and EndStep")

add_test(NAME adios2_adapter_putsync
    COMMAND test_adios2_adapter "ADIOS2 DoPutSync")

add_test(NAME adios2_adapter_putdeferred
    COMMAND test_adios2_adapter "ADIOS2 DoPutDeferred")

add_test(NAME adios2_adapter_multistep
    COMMAND test_adios2_adapter "ADIOS2 Multi-Step Write Workflow")

add_test(NAME adios2_adapter_currentstep
    COMMAND test_adios2_adapter "ADIOS2 CurrentStep Management")

add_test(NAME adios2_adapter_close
    COMMAND test_adios2_adapter "ADIOS2 DoClose")

add_test(NAME adios2_adapter_variable
    COMMAND test_adios2_adapter "ADIOS2 Variable Definition")

add_test(NAME adios2_adapter_large
    COMMAND test_adios2_adapter "ADIOS2 Large Data Handling")

add_test(NAME adios2_adapter_integration
    COMMAND test_adios2_adapter "ADIOS2 Comprehensive Integration")

# Set test properties for proper execution environment
# Use CMAKE_CURRENT_SOURCE_DIR to get the path to cte_config.yaml
set(ADIOS2_TEST_LD_PATH "${CMAKE_BINARY_DIR}/bin:${CMAKE_INSTALL_PREFIX}/lib")

set_tests_properties(
    adios2_adapter_init
    adios2_adapter_step
    adios2_adapter_putsync
    adios2_adapter_putdeferred
    adios2_adapter_multistep
    adios2_adapter_currentstep
    adios2_adapter_close
    adios2_adapter_variable
    adios2_adapter_large
    adios2_adapter_integration
    PROPERTIES
        TIMEOUT 300  # 5 minute timeout for each test
        LABELS "unit;adapter;adios2"
        ENVIRONMENT "ADIOS2_PLUGIN_PATH=${CMAKE_BINARY_DIR}/bin;CHI_WITH_RUNTIME=1;WRP_RUNTIME_CONF=${CMAKE_CURRENT_SOURCE_DIR}/cte_config.yaml;LD_LIBRARY_PATH=${ADIOS2_TEST_LD_PATH}:$ENV{LD_LIBRARY_PATH}"
)

# Install test target
install(TARGETS test_adios2_adapter
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

message(STATUS "ADIOS2 adapter tests configured successfully")
