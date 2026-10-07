# DTSchedule unit tests

# Test: dtschedule interposition
add_executable(test_dtschedule_interpose test_dtschedule_interpose.cc)
target_link_libraries(test_dtschedule_interpose
    clio_cte_dtschedule_client
    clio_cte_dtschedule_runtime
    clio_cte_core_client
    nlohmann_json::nlohmann_json)

# Q-table model directory for the direct Predict() timing case.
target_compile_definitions(test_dtschedule_interpose PRIVATE
    DTSCHEDULE_TEST_MODEL_DIR="${CMAKE_SOURCE_DIR}/context-transfer-engine/dtschedule/models/qtable_v1")

# Register test
# The runtime header pulls in the shared compression header and the core
# interposer; neither include dir is exported by the module targets.
target_include_directories(test_dtschedule_interpose PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include)

add_test(
    NAME test_dtschedule_interpose
    COMMAND test_dtschedule_interpose)

# Set working directory for test
# Same environment as the compressor interposer test: the build-tree
# libraries must win over any installed iowarp, and the runtime needs a
# writable config dir and a loopback bind address.
set_tests_properties(test_dtschedule_interpose
    PROPERTIES
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_TEST_DATA_DIR=${CMAKE_BINARY_DIR};CLIO_BIND_ADDR=127.0.0.1")
