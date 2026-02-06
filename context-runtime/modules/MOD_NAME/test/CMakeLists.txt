# CMakeLists.txt for MOD_NAME module unit tests
cmake_minimum_required(VERSION 3.10)

# Flush correctness test executable
set(FLUSH_TEST_TARGET chimaera_flush_correctness_tests)
set(FLUSH_TEST_SOURCES
  test_flush_correctness.cc
)

# CoMutex test executable
set(COMUTEX_TEST_TARGET chimaera_comutex_tests)
set(COMUTEX_TEST_SOURCES
  test_comutex.cc
)

# Wait Functionality test executable
set(WAIT_FUNCTIONALITY_TEST_TARGET chimaera_wait_functionality_tests)
set(WAIT_FUNCTIONALITY_TEST_SOURCES
  test_wait_functionality.cc
)

# Create flush correctness test executable
add_executable(${FLUSH_TEST_TARGET} ${FLUSH_TEST_SOURCES})

target_include_directories(${FLUSH_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks
)

target_link_libraries(${FLUSH_TEST_TARGET}
  cxx                             # Core Chimaera runtime library
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for work tracking
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${FLUSH_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(${FLUSH_TEST_TARGET} PRIVATE CHIMAERA_RUNTIME=1)

set_target_properties(${FLUSH_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create CoMutex test executable
add_executable(${COMUTEX_TEST_TARGET} ${COMUTEX_TEST_SOURCES})

target_include_directories(${COMUTEX_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

target_link_libraries(${COMUTEX_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for CoMutex/CoRwLock
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${COMUTEX_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${COMUTEX_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Wait Functionality test executable
add_executable(${WAIT_FUNCTIONALITY_TEST_TARGET} ${WAIT_FUNCTIONALITY_TEST_SOURCES})

target_include_directories(${WAIT_FUNCTIONALITY_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

target_link_libraries(${WAIT_FUNCTIONALITY_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for WaitTest tasks
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${WAIT_FUNCTIONALITY_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${WAIT_FUNCTIONALITY_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable CTest integration if testing is enabled
if(CHIMAERA_ENABLE_TESTS)
  # Flush Correctness Tests
  add_test(
    NAME cr_flush_basic_tests
    COMMAND ${FLUSH_TEST_TARGET} "[flush][admin]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_flush_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_flush_work_orchestrator_tests
    COMMAND ${FLUSH_TEST_TARGET} "[flush][work_orchestrator]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_flush_work_orchestrator_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_flush_stress_tests
    COMMAND ${FLUSH_TEST_TARGET} "[flush][stress]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_flush_stress_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Comprehensive flush test
  add_test(
    NAME cr_all_flush_tests
    COMMAND ${FLUSH_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_flush_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # CoMutex Tests
  add_test(
    NAME cr_comutex_basic_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_comutex_concurrent_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][concurrent]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_concurrent_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_corwlock_basic_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_corwlock_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_corwlock_readers_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][readers]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_corwlock_readers_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_corwlock_writers_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][writers]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_corwlock_writers_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_corwlock_interaction_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][interaction]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_corwlock_interaction_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_tasknode_grouping_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[tasknode][grouping]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_tasknode_grouping_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_comutex_error_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][error],[corwlock][error]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_error_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_comutex_performance_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][performance],[corwlock][performance]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_performance_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_comutex_integration_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[integration]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_comutex_integration_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Comprehensive CoMutex test
  add_test(
    NAME cr_all_comutex_tests
    COMMAND ${COMUTEX_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_comutex_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Wait Functionality Tests
  add_test(
    NAME cr_wait_test_basic_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_wait_test_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_wait_test_recursive_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][recursive]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_wait_test_recursive_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_wait_test_async_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][async]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_wait_test_async_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_wait_test_edge_cases_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][edge_cases]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_wait_test_edge_cases_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Comprehensive Wait functionality test
  add_test(
    NAME cr_all_wait_functionality_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_wait_functionality_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Set test properties
  set_tests_properties(
    cr_flush_basic_tests
    cr_flush_work_orchestrator_tests
    cr_flush_stress_tests
    cr_all_flush_tests
    cr_comutex_basic_tests
    cr_comutex_concurrent_tests
    cr_corwlock_basic_tests
    cr_corwlock_readers_tests
    cr_corwlock_writers_tests
    cr_corwlock_interaction_tests
    cr_tasknode_grouping_tests
    cr_comutex_error_tests
    cr_comutex_performance_tests
    cr_comutex_integration_tests
    cr_all_comutex_tests
    cr_wait_test_basic_tests
    cr_wait_test_recursive_tests
    cr_wait_test_async_tests
    cr_wait_test_edge_cases_tests
    cr_all_wait_functionality_tests
    PROPERTIES
      TIMEOUT 180
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CHIMAERA_TEST_MODE=1"
  )
endif()

# Custom targets for MOD_NAME tests
add_custom_target(test_flush
  COMMAND ${FLUSH_TEST_TARGET}
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera flush correctness tests"
)

add_custom_target(test_flush_basic
  COMMAND ${FLUSH_TEST_TARGET} "[flush][admin]"
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera flush basic tests"
)

add_custom_target(test_flush_work_orchestrator
  COMMAND ${FLUSH_TEST_TARGET} "[flush][work_orchestrator]"
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera flush WorkOrchestrator integration tests"
)

add_custom_target(test_flush_stress
  COMMAND ${FLUSH_TEST_TARGET} "[flush][stress]"
  DEPENDS ${FLUSH_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera flush stress tests"
)

add_custom_target(test_comutex
  COMMAND ${COMUTEX_TEST_TARGET}
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera CoMutex and CoRwLock tests"
)

add_custom_target(test_comutex_basic
  COMMAND ${COMUTEX_TEST_TARGET} "[comutex][basic]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera CoMutex basic tests"
)

add_custom_target(test_corwlock_basic
  COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][basic]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera CoRwLock basic tests"
)

add_custom_target(test_comutex_concurrent
  COMMAND ${COMUTEX_TEST_TARGET} "[comutex][concurrent],[corwlock][readers],[corwlock][writers]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera concurrency tests"
)

add_custom_target(test_comutex_performance
  COMMAND ${COMUTEX_TEST_TARGET} "[comutex][performance],[corwlock][performance]"
  DEPENDS ${COMUTEX_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera synchronization performance tests"
)

# Install test executables
install(TARGETS ${FLUSH_TEST_TARGET} ${COMUTEX_TEST_TARGET} ${WAIT_FUNCTIONALITY_TEST_TARGET}
  RUNTIME DESTINATION bin
)

message(STATUS "MOD_NAME module tests configured:")
message(STATUS "  Flush test target: ${FLUSH_TEST_TARGET}")
message(STATUS "  CoMutex test target: ${COMUTEX_TEST_TARGET}")
message(STATUS "  Wait Functionality test target: ${WAIT_FUNCTIONALITY_TEST_TARGET}")
