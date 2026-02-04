# CMakeLists.txt for Chimaera core unit tests
# Note: Module-specific tests are now in their respective module directories:
#   - modules/admin/test/
#   - modules/bdev/test/
#   - modules/MOD_NAME/test/
cmake_minimum_required(VERSION 3.10)

# Include common utilities for dependency management
if(NOT COMMAND add_chimod_client)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/ChimaeraCommon.cmake)
endif()

# Core runtime test executable
set(TEST_TARGET chimaera_unit_tests)
set(TEST_SOURCES
  test_chimaera_runtime_simple.cc
)

# IPC AllocateBuffer test executable
set(IPC_ALLOCATE_BUFFER_TEST_TARGET chimaera_ipc_allocate_buffer_tests)
set(IPC_ALLOCATE_BUFFER_TEST_SOURCES
  test_ipc_allocate_buffer.cc
)

# Unordered Map LL test executable
set(UNORDERED_MAP_LL_TEST_TARGET test_unordered_map_ll)
set(UNORDERED_MAP_LL_TEST_SOURCES
  test_unordered_map_ll.cc
)

# SaveTask and LoadTask test executable
set(SAVE_LOAD_TASK_TEST_TARGET chimaera_save_load_task_tests)
set(SAVE_LOAD_TASK_TEST_SOURCES
  test_save_load_task.cc
)

# Local Task Archive test executable
set(LOCAL_TASK_ARCHIVE_TEST_TARGET chimaera_local_task_archive_tests)
set(LOCAL_TASK_ARCHIVE_TEST_SOURCES
  test_local_task_archive.cc
)

# Per-Process Shared Memory test executable
set(PER_PROCESS_SHM_TEST_TARGET chimaera_per_process_shm_tests)
set(PER_PROCESS_SHM_TEST_SOURCES
  test_per_process_shm.cc
)

# Output Streaming test executable
set(STREAMING_TEST_TARGET chimaera_output_streaming_tests)
set(STREAMING_TEST_SOURCES
  test_streaming.cc
)

# LocalTransfer test executable
set(LOCAL_TRANSFER_TEST_TARGET chimaera_local_transfer_tests)
set(LOCAL_TRANSFER_TEST_SOURCES
  test_local_transfer.cc
)

# External Client test executable
set(EXTERNAL_CLIENT_TEST_TARGET chimaera_external_client_tests)
set(EXTERNAL_CLIENT_TEST_SOURCES
  test_external_client.cc
)

# Runtime Cleanup test executable
set(RUNTIME_CLEANUP_TEST_TARGET chimaera_runtime_cleanup_tests)
set(RUNTIME_CLEANUP_TEST_SOURCES
  test_runtime_cleanup.cc
)

# IPC Errors test executable
set(IPC_ERRORS_TEST_TARGET chimaera_ipc_errors_tests)
set(IPC_ERRORS_TEST_SOURCES
  test_ipc_errors.cc
)

# Create core test executable
add_executable(${TEST_TARGET} ${TEST_SOURCES})

target_include_directories(${TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
)

target_link_libraries(${TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create IPC AllocateBuffer test executable
add_executable(${IPC_ALLOCATE_BUFFER_TEST_TARGET} ${IPC_ALLOCATE_BUFFER_TEST_SOURCES})

target_include_directories(${IPC_ALLOCATE_BUFFER_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For test utilities
)

target_link_libraries(${IPC_ALLOCATE_BUFFER_TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${IPC_ALLOCATE_BUFFER_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${IPC_ALLOCATE_BUFFER_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Unordered Map LL test executable
add_executable(${UNORDERED_MAP_LL_TEST_TARGET} ${UNORDERED_MAP_LL_TEST_SOURCES})

target_link_libraries(${UNORDERED_MAP_LL_TEST_TARGET}
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
  chimaera::cxx              # Core library with runtime headers
)

set_target_properties(${UNORDERED_MAP_LL_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${UNORDERED_MAP_LL_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create SaveTask and LoadTask test executable
add_executable(${SAVE_LOAD_TASK_TEST_TARGET} ${SAVE_LOAD_TASK_TEST_SOURCES})

target_include_directories(${SAVE_LOAD_TASK_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/bdev/include   # For bdev tasks
)

target_link_libraries(${SAVE_LOAD_TASK_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_bdev_runtime           # Bdev module runtime
  chimaera_bdev_client            # Bdev module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${SAVE_LOAD_TASK_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${SAVE_LOAD_TASK_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Local Task Archive test executable
add_executable(${LOCAL_TASK_ARCHIVE_TEST_TARGET} ${LOCAL_TASK_ARCHIVE_TEST_SOURCES})

target_include_directories(${LOCAL_TASK_ARCHIVE_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/admin/include  # For admin tasks
  ${CHIMAERA_ROOT}/modules/bdev/include   # For bdev tasks
)

target_link_libraries(${LOCAL_TASK_ARCHIVE_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_bdev_runtime           # Bdev module runtime
  chimaera_bdev_client            # Bdev module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${LOCAL_TASK_ARCHIVE_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${LOCAL_TASK_ARCHIVE_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Per-Process Shared Memory test executable
add_executable(${PER_PROCESS_SHM_TEST_TARGET} ${PER_PROCESS_SHM_TEST_SOURCES})

target_include_directories(${PER_PROCESS_SHM_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
)

target_link_libraries(${PER_PROCESS_SHM_TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${PER_PROCESS_SHM_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${PER_PROCESS_SHM_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Streaming test executable
add_executable(${STREAMING_TEST_TARGET} ${STREAMING_TEST_SOURCES})

target_include_directories(${STREAMING_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
  ${CHIMAERA_ROOT}/modules/MOD_NAME/include  # For MOD_NAME client and tasks
)

target_link_libraries(${STREAMING_TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  chimaera_MOD_NAME_client   # MOD_NAME client
  chimaera_MOD_NAME_runtime  # MOD_NAME runtime
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${STREAMING_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${STREAMING_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create LocalTransfer test executable
add_executable(${LOCAL_TRANSFER_TEST_TARGET} ${LOCAL_TRANSFER_TEST_SOURCES})

target_include_directories(${LOCAL_TRANSFER_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
)

target_link_libraries(${LOCAL_TRANSFER_TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${LOCAL_TRANSFER_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${LOCAL_TRANSFER_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create External Client test executable
add_executable(${EXTERNAL_CLIENT_TEST_TARGET} ${EXTERNAL_CLIENT_TEST_SOURCES})

target_include_directories(${EXTERNAL_CLIENT_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
)

target_link_libraries(${EXTERNAL_CLIENT_TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${EXTERNAL_CLIENT_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${EXTERNAL_CLIENT_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Runtime Cleanup test executable
add_executable(${RUNTIME_CLEANUP_TEST_TARGET} ${RUNTIME_CLEANUP_TEST_SOURCES})

target_include_directories(${RUNTIME_CLEANUP_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
)

target_link_libraries(${RUNTIME_CLEANUP_TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${RUNTIME_CLEANUP_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${RUNTIME_CLEANUP_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create IPC Errors test executable
add_executable(${IPC_ERRORS_TEST_TARGET} ${IPC_ERRORS_TEST_SOURCES})

target_include_directories(${IPC_ERRORS_TEST_TARGET} PRIVATE
  ${CHIMAERA_ROOT}/include
  ${CHIMAERA_ROOT}/test  # For simple_test.h
)

target_link_libraries(${IPC_ERRORS_TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${IPC_ERRORS_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${IPC_ERRORS_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable CTest integration if testing is enabled
if(CHIMAERA_ENABLE_TESTS)
  # Core Runtime Tests
  add_test(
    NAME cr_runtime_initialization_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_runtime_initialization_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_client_initialization_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_client_initialization_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_mod_name_task_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_mod_name_task_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_error_handling_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_error_handling_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_concurrent_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_concurrent_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_memory_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_memory_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_performance_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_performance_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Comprehensive core test
  add_test(
    NAME cr_all_chimaera_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_chimaera_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # IPC AllocateBuffer Tests
  add_test(
    NAME cr_ipc_allocate_buffer_basic_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_types_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_types_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_sizes_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_sizes_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_multiple_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_multiple_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_client_runtime_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_client_runtime_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_error_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_error_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_alignment_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_alignment_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_documentation_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_documentation_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Per-Process Shared Memory Tests
  add_test(
    NAME cr_per_process_shm_increase_memory_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_increase_memory_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_per_process_shm_medium_alloc_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_medium_alloc_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_per_process_shm_large_alloc_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_large_alloc_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 300  # 5 minute timeout for large allocation test
  )

  add_test(
    NAME cr_per_process_shm_multiple_large_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_multiple_large_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 300
  )

  add_test(
    NAME cr_per_process_shm_patterns_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_patterns_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 300
  )

  add_test(
    NAME cr_per_process_shm_free_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_free_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_per_process_shm_tofullptr_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_tofullptr_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME cr_per_process_shm_stress_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_stress_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 600  # 10 minute timeout for stress test
  )

  add_test(
    NAME cr_per_process_shm_info_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_info_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Streaming Tests
  add_test(
    NAME cr_streaming_small_output_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_streaming_small_output_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 60
  )

  add_test(
    NAME cr_streaming_large_output_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_streaming_large_output_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 60
  )

  add_test(
    NAME cr_streaming_bitfield_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_streaming_bitfield_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 60
  )

  add_test(
    NAME cr_all_streaming_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_streaming_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 180
  )

  # LocalTransfer Tests
  add_test(
    NAME cr_local_transfer_construct_tests
    COMMAND ${LOCAL_TRANSFER_TEST_TARGET} "construct"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_local_transfer_construct_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 60
  )

  add_test(
    NAME cr_local_transfer_single_tests
    COMMAND ${LOCAL_TRANSFER_TEST_TARGET} "single"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_local_transfer_single_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 60
  )

  add_test(
    NAME cr_local_transfer_multi_tests
    COMMAND ${LOCAL_TRANSFER_TEST_TARGET} "multi"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_local_transfer_multi_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 120
  )

  add_test(
    NAME cr_local_transfer_large_tests
    COMMAND ${LOCAL_TRANSFER_TEST_TARGET} "large"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_local_transfer_large_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 120
  )

  add_test(
    NAME cr_local_transfer_edge_tests
    COMMAND ${LOCAL_TRANSFER_TEST_TARGET} "edge"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_local_transfer_edge_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 60
  )

  add_test(
    NAME cr_local_transfer_error_tests
    COMMAND ${LOCAL_TRANSFER_TEST_TARGET} "error"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_local_transfer_error_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 60
  )

  add_test(
    NAME cr_local_transfer_threaded_tests
    COMMAND ${LOCAL_TRANSFER_TEST_TARGET} "threaded"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_local_transfer_threaded_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 120
  )

  add_test(
    NAME cr_all_local_transfer_tests
    COMMAND ${LOCAL_TRANSFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_local_transfer_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 300
  )

  # External Client Tests
  add_test(
    NAME cr_external_client_connection_tests
    COMMAND ${EXTERNAL_CLIENT_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_external_client_connection_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 120
  )

  # Runtime Cleanup Tests
  add_test(
    NAME cr_runtime_cleanup_tests
    COMMAND ${RUNTIME_CLEANUP_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_runtime_cleanup_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 120
  )

  # IPC Errors Tests
  add_test(
    NAME cr_ipc_errors_tests
    COMMAND ${IPC_ERRORS_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_errors_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 120
  )

  # Set test properties for timeout and environment
  set_tests_properties(
    cr_runtime_initialization_tests
    cr_client_initialization_tests
    cr_mod_name_task_tests
    cr_error_handling_tests
    cr_concurrent_tests
    cr_memory_tests
    cr_performance_tests
    cr_all_chimaera_tests
    PROPERTIES
      TIMEOUT 180
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CHIMAERA_TEST_MODE=1"
  )

  # Unordered Map LL Tests (standalone test with custom main)
  add_test(
    NAME cr_unordered_map_ll_tests
    COMMAND ${UNORDERED_MAP_LL_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_unordered_map_ll_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
    TIMEOUT 180
  )
endif()

# Custom targets for running specific test categories
add_custom_target(test_runtime
  COMMAND ${TEST_TARGET}
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera runtime tests"
)

add_custom_target(test_client
  COMMAND ${TEST_TARGET}
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera client tests"
)

add_custom_target(test_tasks
  COMMAND ${TEST_TARGET}
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task tests"
)

add_custom_target(test_errors
  COMMAND ${TEST_TARGET}
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera error handling tests"
)

add_custom_target(test_performance
  COMMAND ${TEST_TARGET}
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera performance tests"
)

# Custom targets for per-process shared memory tests
add_custom_target(test_per_process_shm
  COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
  DEPENDS ${PER_PROCESS_SHM_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera per-process shared memory tests"
)

add_custom_target(test_per_process_shm_large
  COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
  DEPENDS ${PER_PROCESS_SHM_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera per-process shared memory large allocation tests (>1GB)"
)

add_custom_target(test_per_process_shm_stress
  COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
  DEPENDS ${PER_PROCESS_SHM_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera per-process shared memory stress tests"
)

# Custom target to run all core unit tests
add_custom_target(test_all_unit
  COMMAND ${TEST_TARGET}
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running all Chimaera core unit tests"
)

# Install test executables
install(TARGETS
  ${TEST_TARGET}
  ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
  ${UNORDERED_MAP_LL_TEST_TARGET}
  ${SAVE_LOAD_TASK_TEST_TARGET}
  ${LOCAL_TASK_ARCHIVE_TEST_TARGET}
  ${PER_PROCESS_SHM_TEST_TARGET}
  ${STREAMING_TEST_TARGET}
  ${LOCAL_TRANSFER_TEST_TARGET}
  ${EXTERNAL_CLIENT_TEST_TARGET}
  ${RUNTIME_CLEANUP_TEST_TARGET}
  ${IPC_ERRORS_TEST_TARGET}
  RUNTIME DESTINATION bin
)

# Note: Module-specific tests are now in their respective module directories
# Note: Distributed tests are run via Docker using the chimaera_bdev_chimod_tests binary
# See test/unit/distributed/docker-compose.yml and run_tests.sh for distributed test execution

# Print configuration summary
message(STATUS "Chimaera core unit tests configured:")
message(STATUS "  Test target: ${TEST_TARGET}")
message(STATUS "  Test sources: ${TEST_SOURCES}")
message(STATUS "  IPC AllocateBuffer test target: ${IPC_ALLOCATE_BUFFER_TEST_TARGET}")
message(STATUS "  Per-Process SHM test target: ${PER_PROCESS_SHM_TEST_TARGET}")
message(STATUS "  CTest enabled: ${CHIMAERA_ENABLE_TESTS}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "")
message(STATUS "Module-specific tests are in:")
message(STATUS "  - modules/admin/test/")
message(STATUS "  - modules/bdev/test/")
message(STATUS "  - modules/MOD_NAME/test/")
