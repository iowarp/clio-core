# CMakeLists.txt for Chimaera unit tests
cmake_minimum_required(VERSION 3.10)

# Include common utilities for dependency management
# ChimaeraCommon.cmake is already included by parent CMakeLists.txt, but include it again if not found
if(NOT COMMAND add_chimod_client)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/ChimaeraCommon.cmake)
endif()

# Test executable name
set(TEST_TARGET chimaera_unit_tests)

# Test source files
set(TEST_SOURCES
  test_chimaera_runtime_simple.cc
)

# Task archive test executable (separate from main tests)
set(TASK_ARCHIVE_TEST_TARGET chimaera_task_archive_tests)
set(TASK_ARCHIVE_TEST_SOURCES
  test_task_archive.cc
)

# Flush correctness test executable (separate test for flush functionality)
set(FLUSH_TEST_TARGET chimaera_flush_correctness_tests)
set(FLUSH_TEST_SOURCES
  test_flush_correctness.cc
)

# Bdev ChiMod test executable (separate test for bdev functionality)
set(BDEV_TEST_TARGET chimaera_bdev_chimod_tests)
set(BDEV_TEST_SOURCES
  test_bdev_chimod.cc
)

# Boost Fiber test executable (separate test for boost context validation)
set(FIBER_TEST_TARGET chimaera_boost_fiber_tests)
set(FIBER_TEST_SOURCES
  test_boost_fiber.cc
)

# Worker Fiber Pattern test executable (test for worker.cc/task.cc pattern)
set(WORKER_FIBER_TEST_TARGET chimaera_worker_fiber_pattern_tests)
set(WORKER_FIBER_TEST_SOURCES
  test_worker_fiber_pattern.cc
)

# Segfault Debug test executable (debug segfault in ExecTask is_started)
set(SEGFAULT_DEBUG_TEST_TARGET chimaera_segfault_debug_tests)
set(SEGFAULT_DEBUG_TEST_SOURCES
  test_segfault_debug.cc
)

# CoMutex test executable (test for CoMutex and CoRwLock synchronization primitives)
set(COMUTEX_TEST_TARGET chimaera_comutex_tests)
set(COMUTEX_TEST_SOURCES
  test_comutex.cc
)

# Wait Functionality test executable (test for task->Wait() recursive functionality)
set(WAIT_FUNCTIONALITY_TEST_TARGET chimaera_wait_functionality_tests)
set(WAIT_FUNCTIONALITY_TEST_SOURCES
  test_wait_functionality.cc
)

# IPC AllocateBuffer test executable (test for CHI_IPC->AllocateBuffer functionality)
set(IPC_ALLOCATE_BUFFER_TEST_TARGET chimaera_ipc_allocate_buffer_tests)
set(IPC_ALLOCATE_BUFFER_TEST_SOURCES
  test_ipc_allocate_buffer.cc
)

# Unordered Map LL test executable (test for chi::unordered_map_ll container)
set(UNORDERED_MAP_LL_TEST_TARGET test_unordered_map_ll)
set(UNORDERED_MAP_LL_TEST_SOURCES
  test_unordered_map_ll.cc
)

# SaveTask and LoadTask test executable (test for Container SaveTask/LoadTask methods)
set(SAVE_LOAD_TASK_TEST_TARGET chimaera_save_load_task_tests)
set(SAVE_LOAD_TASK_TEST_SOURCES
  test_save_load_task.cc
)

# Compose feature test executable (test for compose configuration and pool creation)
set(COMPOSE_TEST_TARGET chimaera_compose_tests)
set(COMPOSE_TEST_SOURCES
  test_compose.cc
)

add_executable(${COMPOSE_TEST_TARGET} ${COMPOSE_TEST_SOURCES})

# Include directories for compose tests
target_include_directories(${COMPOSE_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test  # For simple_test.h
  ${CMAKE_SOURCE_DIR}/modules/admin/include  # For admin client
  ${CMAKE_SOURCE_DIR}/modules/bdev/include   # For bdev client
)

# Link against required libraries for compose tests
target_link_libraries(${COMPOSE_TEST_TARGET}
  cxx                             # Core Chimaera runtime library
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_bdev_runtime           # BDev module runtime
  chimaera_bdev_client            # BDev module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for compose tests
set_target_properties(${COMPOSE_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Define CHIMAERA_RUNTIME for runtime mode
target_compile_definitions(${COMPOSE_TEST_TARGET} PRIVATE CHIMAERA_RUNTIME=1)

# Copy compose test executable to bin directory
set_target_properties(${COMPOSE_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(${TASK_ARCHIVE_TEST_TARGET} ${TASK_ARCHIVE_TEST_SOURCES})

# Include directories for task archive tests
target_include_directories(${TASK_ARCHIVE_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test  # For simple_test.h
  ${CMAKE_SOURCE_DIR}/modules/admin/include  # For admin tasks
)

# Link against required libraries for task archive tests
target_link_libraries(${TASK_ARCHIVE_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime for task types
  chimaera_admin_client           # Admin module client for task types
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for task archive tests
set_target_properties(${TASK_ARCHIVE_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Removed CHIMAERA_RUNTIME compile definition - using runtime IsRuntime() checks instead

# Copy task archive test executable to bin directory
set_target_properties(${TASK_ARCHIVE_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create flush correctness test executable
add_executable(${FLUSH_TEST_TARGET} ${FLUSH_TEST_SOURCES})

# Include directories for flush tests
target_include_directories(${FLUSH_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/runtime/test  # For simple_test.h
  ${CMAKE_SOURCE_DIR}/modules/admin/include  # For admin tasks
  ${CMAKE_SOURCE_DIR}/modules/MOD_NAME/include  # For MOD_NAME tasks
)

# Link against required libraries for flush tests
target_link_libraries(${FLUSH_TEST_TARGET}
  cxx                             # Core Chimaera runtime library
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for work tracking
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for flush tests
set_target_properties(${FLUSH_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Define CHIMAERA_RUNTIME for runtime mode
target_compile_definitions(${FLUSH_TEST_TARGET} PRIVATE CHIMAERA_RUNTIME=1)

# Copy flush test executable to bin directory
set_target_properties(${FLUSH_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create bdev test executable
add_executable(${BDEV_TEST_TARGET} ${BDEV_TEST_SOURCES})

# Include directories for bdev tests
target_include_directories(${BDEV_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test  # For simple_test.h
  ${CMAKE_SOURCE_DIR}/modules/admin/include  # For admin tasks
  ${CMAKE_SOURCE_DIR}/modules/bdev/include   # For bdev tasks and client
)

# Link against required libraries for bdev tests
target_link_libraries(${BDEV_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_bdev_runtime           # Bdev module runtime
  chimaera_bdev_client            # Bdev module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for bdev tests
set_target_properties(${BDEV_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Removed CHIMAERA_RUNTIME compile definition - using runtime IsRuntime() checks instead

# Copy bdev test executable to bin directory
set_target_properties(${BDEV_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create boost fiber test executable
add_executable(${FIBER_TEST_TARGET} ${FIBER_TEST_SOURCES})

# Include directories for fiber tests
target_include_directories(${FIBER_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# Find Boost context library for fiber tests
find_package(Boost REQUIRED COMPONENTS context)

# Link against hermes-shm target for fiber tests (which includes boost)
target_link_libraries(${FIBER_TEST_TARGET}
  hshm::cxx                       # Hermes-SHM target (includes boost dependencies)
  Boost::context                  # Boost context library for fiber operations
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for fiber tests
set_target_properties(${FIBER_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Copy fiber test executable to bin directory
set_target_properties(${FIBER_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create worker fiber pattern test executable
add_executable(${WORKER_FIBER_TEST_TARGET} ${WORKER_FIBER_TEST_SOURCES})

# Include directories for worker fiber pattern tests
target_include_directories(${WORKER_FIBER_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# Link against hermes-shm target for worker fiber pattern tests
target_link_libraries(${WORKER_FIBER_TEST_TARGET}
  hshm::cxx                       # Hermes-SHM target (includes boost dependencies)
  Boost::context                  # Boost context library for fiber operations
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for worker fiber pattern tests
set_target_properties(${WORKER_FIBER_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Copy worker fiber pattern test executable to bin directory
set_target_properties(${WORKER_FIBER_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create segfault debug test executable
add_executable(${SEGFAULT_DEBUG_TEST_TARGET} ${SEGFAULT_DEBUG_TEST_SOURCES})

# Include directories for segfault debug tests
target_include_directories(${SEGFAULT_DEBUG_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

# Link against hermes-shm target for segfault debug tests
target_link_libraries(${SEGFAULT_DEBUG_TEST_TARGET}
  hshm::cxx                       # Hermes-SHM target (includes boost dependencies)
  Boost::context                  # Boost context library for fiber operations
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for segfault debug tests
set_target_properties(${SEGFAULT_DEBUG_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Copy segfault debug test executable to bin directory
set_target_properties(${SEGFAULT_DEBUG_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create CoMutex test executable
add_executable(${COMUTEX_TEST_TARGET} ${COMUTEX_TEST_SOURCES})

# Include directories for CoMutex tests
target_include_directories(${COMUTEX_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test  # For simple_test.h
  ${CMAKE_SOURCE_DIR}/modules/admin/include  # For admin tasks
  ${CMAKE_SOURCE_DIR}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

# Link against required libraries for CoMutex tests
target_link_libraries(${COMUTEX_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for CoMutex/CoRwLock
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for CoMutex tests
set_target_properties(${COMUTEX_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Removed CHIMAERA_RUNTIME compile definition - using runtime IsRuntime() checks instead

# Copy CoMutex test executable to bin directory
set_target_properties(${COMUTEX_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Wait Functionality test executable
add_executable(${WAIT_FUNCTIONALITY_TEST_TARGET} ${WAIT_FUNCTIONALITY_TEST_SOURCES})

# Include directories for Wait Functionality tests
target_include_directories(${WAIT_FUNCTIONALITY_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test  # For simple_test.h
  ${CMAKE_SOURCE_DIR}/modules/admin/include  # For admin tasks
  ${CMAKE_SOURCE_DIR}/modules/MOD_NAME/include  # For MOD_NAME tasks and client
)

# Link against required libraries for Wait Functionality tests
target_link_libraries(${WAIT_FUNCTIONALITY_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_MOD_NAME_runtime       # MOD_NAME module runtime for WaitTest tasks
  chimaera_MOD_NAME_client        # MOD_NAME module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for Wait Functionality tests
set_target_properties(${WAIT_FUNCTIONALITY_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Copy Wait Functionality test executable to bin directory
set_target_properties(${WAIT_FUNCTIONALITY_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create IPC AllocateBuffer test executable
add_executable(${IPC_ALLOCATE_BUFFER_TEST_TARGET} ${IPC_ALLOCATE_BUFFER_TEST_SOURCES})

# Include directories for IPC AllocateBuffer tests
target_include_directories(${IPC_ALLOCATE_BUFFER_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test  # For test utilities
)

# Link against required libraries for IPC AllocateBuffer tests
target_link_libraries(${IPC_ALLOCATE_BUFFER_TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

# Set C++ standard for IPC AllocateBuffer tests
set_target_properties(${IPC_ALLOCATE_BUFFER_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Copy IPC AllocateBuffer test executable to bin directory
set_target_properties(${IPC_ALLOCATE_BUFFER_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Unordered Map LL test executable
add_executable(${UNORDERED_MAP_LL_TEST_TARGET} ${UNORDERED_MAP_LL_TEST_SOURCES})

# Include directories for Unordered Map LL tests
target_include_directories(${UNORDERED_MAP_LL_TEST_TARGET} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include  # Runtime include directory for chimaera headers
)

# Link against required libraries for Unordered Map LL tests
target_link_libraries(${UNORDERED_MAP_LL_TEST_TARGET}
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

# Set C++ standard for Unordered Map LL tests
set_target_properties(${UNORDERED_MAP_LL_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Copy Unordered Map LL test executable to bin directory
set_target_properties(${UNORDERED_MAP_LL_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create SaveTask and LoadTask test executable
add_executable(${SAVE_LOAD_TASK_TEST_TARGET} ${SAVE_LOAD_TASK_TEST_SOURCES})

# Include directories for SaveTask/LoadTask tests
target_include_directories(${SAVE_LOAD_TASK_TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test  # For simple_test.h
  ${CMAKE_SOURCE_DIR}/modules/admin/include  # For admin tasks
  ${CMAKE_SOURCE_DIR}/modules/bdev/include   # For bdev tasks
)

# Link against required libraries for SaveTask/LoadTask tests
target_link_libraries(${SAVE_LOAD_TASK_TEST_TARGET}
  chimaera_admin_runtime          # Admin module runtime
  chimaera_admin_client           # Admin module client
  chimaera_bdev_runtime           # Bdev module runtime
  chimaera_bdev_client            # Bdev module client
  hshm::cxx                       # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

# Set C++ standard for SaveTask/LoadTask tests
set_target_properties(${SAVE_LOAD_TASK_TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Copy SaveTask/LoadTask test executable to bin directory
set_target_properties(${SAVE_LOAD_TASK_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create test executable
add_executable(${TEST_TARGET} ${TEST_SOURCES})

# Include directories
target_include_directories(${TEST_TARGET} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/test  # For simple_test.h
)

# Link against required libraries
target_link_libraries(${TEST_TARGET}
  chimaera_cxx               # Main Chimaera library
  hshm::cxx                  # HermesShm library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

# Set C++ standard
set_target_properties(${TEST_TARGET} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Removed CHIMAERA_RUNTIME compile definition - using runtime IsRuntime() checks instead

# Copy test executable to bin directory for easier access
set_target_properties(${TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable CTest integration if testing is enabled
if(CHIMAERA_ENABLE_TESTS)
  # Task Archive Tests
  add_test(
    NAME task_archive_basic_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][input_in],[task_archive][output_in],[task_archive][input_out],[task_archive][output_out]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  add_test(
    NAME task_archive_bulk_transfer_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][bulk_transfer]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_bulk_transfer_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  add_test(
    NAME task_archive_non_task_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][non_task]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_non_task_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  add_test(
    NAME task_archive_task_base_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][task_base]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_task_base_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  add_test(
    NAME task_archive_admin_tasks_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][admin_tasks]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_admin_tasks_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  add_test(
    NAME task_archive_bidirectional_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][bidirectional]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_bidirectional_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  add_test(
    NAME task_archive_container_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][container]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_container_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  add_test(
    NAME task_archive_error_handling_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][error_handling]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_error_handling_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  add_test(
    NAME task_archive_performance_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][performance]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_performance_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  add_test(
    NAME task_archive_integration_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][integration]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(task_archive_integration_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )
  
  
  # Comprehensive task archive test
  add_test(
    NAME all_task_archive_tests
    COMMAND ${TASK_ARCHIVE_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(all_task_archive_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )


  # Add test cases with different tags
  add_test(
    NAME runtime_initialization_tests
    COMMAND ${TEST_TARGET} "[runtime][initialization]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(runtime_initialization_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME client_initialization_tests  
    COMMAND ${TEST_TARGET} "[client][initialization]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(client_initialization_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME mod_name_task_tests
    COMMAND ${TEST_TARGET} "[task][mod_name]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(mod_name_task_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME error_handling_tests
    COMMAND ${TEST_TARGET} "[error][edge_cases]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(error_handling_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME concurrent_tests
    COMMAND ${TEST_TARGET} "[concurrent][stress]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(concurrent_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME memory_tests
    COMMAND ${TEST_TARGET} "[memory][cleanup]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(memory_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME performance_tests
    COMMAND ${TEST_TARGET} "[performance][timing]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(performance_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  # Flush Correctness Tests
  add_test(
    NAME flush_basic_tests
    COMMAND ${FLUSH_TEST_TARGET} "[flush][admin]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(flush_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME flush_work_orchestrator_tests
    COMMAND ${FLUSH_TEST_TARGET} "[flush][work_orchestrator]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(flush_work_orchestrator_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  add_test(
    NAME flush_stress_tests
    COMMAND ${FLUSH_TEST_TARGET} "[flush][stress]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(flush_stress_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Comprehensive flush test
  add_test(
    NAME all_flush_tests
    COMMAND ${FLUSH_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(all_flush_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  # Bdev ChiMod Tests
  add_test(
    NAME bdev_container_creation_tests
    COMMAND ${BDEV_TEST_TARGET} "[bdev][create]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(bdev_container_creation_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME bdev_block_allocation_tests
    COMMAND ${BDEV_TEST_TARGET} "[bdev][allocate]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(bdev_block_allocation_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME bdev_io_operations_tests
    COMMAND ${BDEV_TEST_TARGET} "[bdev][io]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(bdev_io_operations_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME bdev_async_operations_tests
    COMMAND ${BDEV_TEST_TARGET} "[bdev][async]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(bdev_async_operations_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME bdev_free_management_tests
    COMMAND ${BDEV_TEST_TARGET} "[bdev][free]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(bdev_free_management_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME bdev_performance_tests
    COMMAND ${BDEV_TEST_TARGET} "[bdev][performance]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(bdev_performance_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME bdev_error_handling_tests
    COMMAND ${BDEV_TEST_TARGET} "[bdev][error]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(bdev_error_handling_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME bdev_cleanup_tests
    COMMAND ${BDEV_TEST_TARGET} "[bdev][cleanup]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(bdev_cleanup_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  # Comprehensive bdev test
  add_test(
    NAME all_bdev_tests
    COMMAND ${BDEV_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(all_bdev_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )


  # CoMutex Tests
  add_test(
    NAME comutex_basic_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(comutex_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME comutex_concurrent_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][concurrent]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(comutex_concurrent_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME corwlock_basic_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(corwlock_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME corwlock_readers_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][readers]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(corwlock_readers_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME corwlock_writers_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][writers]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(corwlock_writers_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME corwlock_interaction_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[corwlock][interaction]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(corwlock_interaction_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME tasknode_grouping_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[tasknode][grouping]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(tasknode_grouping_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME comutex_error_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][error],[corwlock][error]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(comutex_error_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME comutex_performance_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[comutex][performance],[corwlock][performance]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(comutex_performance_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME comutex_integration_tests
    COMMAND ${COMUTEX_TEST_TARGET} "[integration]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(comutex_integration_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  # Comprehensive CoMutex test
  add_test(
    NAME all_comutex_tests
    COMMAND ${COMUTEX_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(all_comutex_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )


  # Wait Functionality Tests
  add_test(
    NAME wait_test_basic_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(wait_test_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME wait_test_recursive_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][recursive]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(wait_test_recursive_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME wait_test_async_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][async]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(wait_test_async_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME wait_test_edge_cases_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET} "[wait_test][edge_cases]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(wait_test_edge_cases_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  # Comprehensive Wait functionality test
  add_test(
    NAME all_wait_functionality_tests
    COMMAND ${WAIT_FUNCTIONALITY_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(all_wait_functionality_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  # IPC AllocateBuffer Tests
  add_test(
    NAME ipc_allocate_buffer_basic_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET} "[ipc][allocate_buffer][basic]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(ipc_allocate_buffer_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME ipc_allocate_buffer_types_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET} "[ipc][allocate_buffer][types]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(ipc_allocate_buffer_types_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME ipc_allocate_buffer_sizes_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET} "[ipc][allocate_buffer][sizes]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(ipc_allocate_buffer_sizes_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME ipc_allocate_buffer_multiple_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET} "[ipc][allocate_buffer][multiple]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(ipc_allocate_buffer_multiple_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME ipc_allocate_buffer_client_runtime_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET} "[ipc][allocate_buffer][client_runtime]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(ipc_allocate_buffer_client_runtime_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME ipc_allocate_buffer_error_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET} "[ipc][allocate_buffer][error]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(ipc_allocate_buffer_error_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME ipc_allocate_buffer_alignment_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET} "[ipc][allocate_buffer][alignment]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(ipc_allocate_buffer_alignment_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  add_test(
    NAME ipc_allocate_buffer_documentation_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET} "[ipc][allocate_buffer][documentation]"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(ipc_allocate_buffer_documentation_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  # Comprehensive IPC AllocateBuffer test
  add_test(
    NAME all_ipc_allocate_buffer_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(all_ipc_allocate_buffer_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )


  # Add a comprehensive test that runs all tests
  add_test(
    NAME all_chimaera_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(all_chimaera_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin"
  )

  
  # Set test properties for timeout and environment
  set_tests_properties(
    task_archive_basic_tests
    task_archive_bulk_transfer_tests
    task_archive_non_task_tests
    task_archive_task_base_tests
    task_archive_admin_tasks_tests
    task_archive_bidirectional_tests
    task_archive_container_tests
    task_archive_error_handling_tests
    task_archive_performance_tests
    task_archive_integration_tests
    all_task_archive_tests
    runtime_initialization_tests
    client_initialization_tests
    mod_name_task_tests
    error_handling_tests
    concurrent_tests
    memory_tests
    performance_tests
    flush_basic_tests
    flush_work_orchestrator_tests
    flush_stress_tests
    all_flush_tests
    bdev_container_creation_tests
    bdev_block_allocation_tests
    bdev_io_operations_tests
    bdev_async_operations_tests
    bdev_free_management_tests
    bdev_performance_tests
    bdev_error_handling_tests
    bdev_cleanup_tests
    all_bdev_tests
    comutex_basic_tests
    comutex_concurrent_tests
    corwlock_basic_tests
    corwlock_readers_tests
    corwlock_writers_tests
    corwlock_interaction_tests
    tasknode_grouping_tests
    comutex_error_tests
    comutex_performance_tests
    comutex_integration_tests
    all_comutex_tests
    all_chimaera_tests
    PROPERTIES
      TIMEOUT 180  # 180 second timeout for each test (increased for synchronization tests)
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CHIMAERA_TEST_MODE=1"
  )
endif()

# Custom targets for running specific test categories

# Task Archive Test Targets
add_custom_target(test_task_archive
  COMMAND ${TASK_ARCHIVE_TEST_TARGET}
  DEPENDS ${TASK_ARCHIVE_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task archive tests"
)

add_custom_target(test_task_archive_basic
  COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][input_in],[task_archive][output_in],[task_archive][input_out],[task_archive][output_out]"
  DEPENDS ${TASK_ARCHIVE_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task archive basic tests"
)

add_custom_target(test_task_archive_serialization
  COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][task_base],[task_archive][admin_tasks]"
  DEPENDS ${TASK_ARCHIVE_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task serialization tests"
)

add_custom_target(test_task_archive_performance
  COMMAND ${TASK_ARCHIVE_TEST_TARGET} "[task_archive][performance]"
  DEPENDS ${TASK_ARCHIVE_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task archive performance tests"
)

add_custom_target(test_runtime
  COMMAND ${TEST_TARGET} "[runtime]"
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera runtime tests"
)

add_custom_target(test_client
  COMMAND ${TEST_TARGET} "[client]"
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera client tests"
)

add_custom_target(test_tasks
  COMMAND ${TEST_TARGET} "[task]"
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera task tests"
)

add_custom_target(test_errors
  COMMAND ${TEST_TARGET} "[error]"
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera error handling tests"
)

add_custom_target(test_performance
  COMMAND ${TEST_TARGET} "[performance]"
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera performance tests"
)

# Custom targets for flush tests
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

# Custom targets for bdev tests
add_custom_target(test_bdev
  COMMAND ${BDEV_TEST_TARGET}
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev ChiMod tests"
)

add_custom_target(test_bdev_create
  COMMAND ${BDEV_TEST_TARGET} "[bdev][create]"
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev container creation tests"
)

add_custom_target(test_bdev_allocate
  COMMAND ${BDEV_TEST_TARGET} "[bdev][allocate]"
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev block allocation tests"
)

add_custom_target(test_bdev_io
  COMMAND ${BDEV_TEST_TARGET} "[bdev][io]"
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev I/O operation tests"
)

add_custom_target(test_bdev_async
  COMMAND ${BDEV_TEST_TARGET} "[bdev][async]"
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev asynchronous I/O tests"
)

add_custom_target(test_bdev_performance
  COMMAND ${BDEV_TEST_TARGET} "[bdev][performance]"
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev performance metrics tests"
)

add_custom_target(test_bdev_error
  COMMAND ${BDEV_TEST_TARGET} "[bdev][error]"
  DEPENDS ${BDEV_TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Chimaera bdev error handling tests"
)

# Custom targets for CoMutex tests
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

# Custom target to run all unit tests
add_custom_target(test_all_unit
  COMMAND ${TEST_TARGET}
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running all Chimaera unit tests"
)

# Install test executables (optional)
install(TARGETS ${TEST_TARGET} ${TASK_ARCHIVE_TEST_TARGET} ${FLUSH_TEST_TARGET} ${BDEV_TEST_TARGET} ${FIBER_TEST_TARGET} ${WORKER_FIBER_TEST_TARGET} ${SEGFAULT_DEBUG_TEST_TARGET} ${COMUTEX_TEST_TARGET} ${WAIT_FUNCTIONALITY_TEST_TARGET} ${IPC_ALLOCATE_BUFFER_TEST_TARGET} ${UNORDERED_MAP_LL_TEST_TARGET} ${SAVE_LOAD_TASK_TEST_TARGET} ${COMPOSE_TEST_TARGET}
  RUNTIME DESTINATION bin
)

# Note: Distributed tests are run via Docker using the chimaera_bdev_chimod_tests binary
# See test/unit/distributed/docker-compose.yml and run_tests.sh for distributed test execution

# Print configuration summary
message(STATUS "Chimaera unit tests configured:")
message(STATUS "  Test target: ${TEST_TARGET}")
message(STATUS "  Test sources: ${TEST_SOURCES}")
message(STATUS "  Task archive test target: ${TASK_ARCHIVE_TEST_TARGET}")
message(STATUS "  Task archive test sources: ${TASK_ARCHIVE_TEST_SOURCES}")
message(STATUS "  Flush test target: ${FLUSH_TEST_TARGET}")
message(STATUS "  Flush test sources: ${FLUSH_TEST_SOURCES}")
message(STATUS "  Bdev test target: ${BDEV_TEST_TARGET}")
message(STATUS "  Bdev test sources: ${BDEV_TEST_SOURCES}")
message(STATUS "  CoMutex test target: ${COMUTEX_TEST_TARGET}")
message(STATUS "  CoMutex test sources: ${COMUTEX_TEST_SOURCES}")
message(STATUS "  Wait Functionality test target: ${WAIT_FUNCTIONALITY_TEST_TARGET}")
message(STATUS "  Wait Functionality test sources: ${WAIT_FUNCTIONALITY_TEST_SOURCES}")
message(STATUS "  IPC AllocateBuffer test target: ${IPC_ALLOCATE_BUFFER_TEST_TARGET}")
message(STATUS "  IPC AllocateBuffer test sources: ${IPC_ALLOCATE_BUFFER_TEST_SOURCES}")
message(STATUS "  Save/Load Task test target: ${SAVE_LOAD_TASK_TEST_TARGET}")
message(STATUS "  Save/Load Task test sources: ${SAVE_LOAD_TASK_TEST_SOURCES}")
message(STATUS "  Compose test target: ${COMPOSE_TEST_TARGET}")
message(STATUS "  Compose test sources: ${COMPOSE_TEST_SOURCES}")
message(STATUS "  CTest enabled: ${CHIMAERA_ENABLE_TESTS}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/bin")