# CMakeLists.txt for CLIO Runtime core unit tests
# Note: Module-specific tests are now in their respective module directories:
#   - modules/admin/test/
#   - modules/bdev/test/
#   - modules/MOD_NAME/test/
cmake_minimum_required(VERSION 3.10)

# Include common utilities for dependency management
if(NOT COMMAND add_clio_module_client)
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

# Autogen Coverage test executable
set(AUTOGEN_COVERAGE_TEST_TARGET chimaera_autogen_coverage_tests)
set(AUTOGEN_COVERAGE_TEST_SOURCES
  test_autogen_coverage.cc
)

# Output Streaming test executable
set(STREAMING_TEST_TARGET chimaera_output_streaming_tests)
set(STREAMING_TEST_SOURCES
  test_streaming.cc
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

# IPC Transport Modes test executable
set(IPC_TRANSPORT_MODES_TEST_TARGET chimaera_ipc_transport_modes_tests)
set(IPC_TRANSPORT_MODES_TEST_SOURCES
  test_ipc_transport_modes.cc
)

# Client Retry test executable
set(CLIENT_RETRY_TEST_TARGET chimaera_client_retry_tests)
set(CLIENT_RETRY_TEST_SOURCES
  test_client_retry.cc
)


# GPU IPC AllocateBuffer test executable (only if CUDA or HIP is enabled)
set(IPC_ALLOCATE_BUFFER_GPU_TEST_TARGET test_ipc_allocate_buffer_gpu)
set(IPC_ALLOCATE_BUFFER_GPU_TEST_SOURCES
  gpu/test_ipc_allocate_buffer_gpu.cc
)

# Create core test executable
add_executable(${TEST_TARGET} ${TEST_SOURCES})

target_include_directories(${TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
)

target_link_libraries(${TEST_TARGET}
  chimaera_cxx               # Main CLIO Runtime library
  ctp::cxx                  # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create IPC AllocateBuffer test executable
add_executable(${IPC_ALLOCATE_BUFFER_TEST_TARGET} ${IPC_ALLOCATE_BUFFER_TEST_SOURCES})

target_include_directories(${IPC_ALLOCATE_BUFFER_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For test utilities
)

target_link_libraries(${IPC_ALLOCATE_BUFFER_TEST_TARGET}
  chimaera_cxx               # Main CLIO Runtime library
  ctp::cxx                  # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${IPC_ALLOCATE_BUFFER_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
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
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${UNORDERED_MAP_LL_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create SaveTask and LoadTask test executable
add_executable(${SAVE_LOAD_TASK_TEST_TARGET} ${SAVE_LOAD_TASK_TEST_SOURCES})

target_include_directories(${SAVE_LOAD_TASK_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin tasks
  ${CLIO_RUN_ROOT}/modules/bdev/include   # For bdev tasks
)

target_link_libraries(${SAVE_LOAD_TASK_TEST_TARGET}
  clio_admin_runtime          # Admin module runtime
  clio_admin_client           # Admin module client
  clio_bdev_runtime           # Bdev module runtime
  clio_bdev_client            # Bdev module client
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${SAVE_LOAD_TASK_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${SAVE_LOAD_TASK_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Local Task Archive test executable
add_executable(${LOCAL_TASK_ARCHIVE_TEST_TARGET} ${LOCAL_TASK_ARCHIVE_TEST_SOURCES})

target_include_directories(${LOCAL_TASK_ARCHIVE_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include  # For admin tasks
  ${CLIO_RUN_ROOT}/modules/bdev/include   # For bdev tasks
)

target_link_libraries(${LOCAL_TASK_ARCHIVE_TEST_TARGET}
  clio_admin_runtime          # Admin module runtime
  clio_admin_client           # Admin module client
  clio_bdev_runtime           # Bdev module runtime
  clio_bdev_client            # Bdev module client
  ctp::cxx                       # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}       # Threading support
)

set_target_properties(${LOCAL_TASK_ARCHIVE_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${LOCAL_TASK_ARCHIVE_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Per-Process Shared Memory test executable.
# POSIX-only: uses fork()/waitpid()/setenv() — no Windows equivalent that
# keeps the same shared-page semantics (CreateProcess gives a fresh address
# space, defeating the inheritance the test relies on).
if(NOT WIN32)
add_executable(${PER_PROCESS_SHM_TEST_TARGET} ${PER_PROCESS_SHM_TEST_SOURCES})

target_include_directories(${PER_PROCESS_SHM_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
)

target_link_libraries(${PER_PROCESS_SHM_TEST_TARGET}
  chimaera_cxx               # Main CLIO Runtime library
  ctp::cxx                  # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

# RuntimeServer launches clio_run; ensure it is built and discoverable.
add_dependencies(${PER_PROCESS_SHM_TEST_TARGET} clio_run)
target_compile_definitions(${PER_PROCESS_SHM_TEST_TARGET} PRIVATE
  CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")

set_target_properties(${PER_PROCESS_SHM_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${PER_PROCESS_SHM_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif()

# Create Autogen Coverage test executable (requires CAE)
if(CLIO_CORE_ENABLE_CAE)
add_executable(${AUTOGEN_COVERAGE_TEST_TARGET} ${AUTOGEN_COVERAGE_TEST_SOURCES})

  target_include_directories(${AUTOGEN_COVERAGE_TEST_TARGET} PRIVATE
    ${CLIO_RUN_ROOT}/include
    ${CLIO_RUN_ROOT}/test  # For simple_test.h
    ${CLIO_RUN_ROOT}/modules/admin/include
    ${CLIO_RUN_ROOT}/modules/bdev/include
    ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
  )

  target_link_libraries(${AUTOGEN_COVERAGE_TEST_TARGET}
    chimaera_MOD_NAME_runtime
    chimaera_MOD_NAME_client
    clio_cte_core_runtime
    clio_cae_core_runtime
    clio_cae_core_client
    clio_bdev_runtime
    clio_admin_runtime
    clio_cte_core_client
    clio_bdev_client
    clio_admin_client
    chimaera_cxx
    ctp::cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )

  set_target_properties(${AUTOGEN_COVERAGE_TEST_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )

set_target_properties(${AUTOGEN_COVERAGE_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif()  # CLIO_CORE_ENABLE_CAE

# Delayed Task test executable
set(DELAYED_TASK_TEST_TARGET chimaera_delayed_task_tests)
set(DELAYED_TASK_TEST_SOURCES
  test_delayed_task.cc
)

# Create Streaming test executable
add_executable(${STREAMING_TEST_TARGET} ${STREAMING_TEST_SOURCES})

target_include_directories(${STREAMING_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include  # For MOD_NAME client and tasks
)

target_link_libraries(${STREAMING_TEST_TARGET}
  chimaera_cxx               # Main CLIO Runtime library
  chimaera_MOD_NAME_client   # MOD_NAME client
  chimaera_MOD_NAME_runtime  # MOD_NAME runtime
  ctp::cxx                  # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${STREAMING_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${STREAMING_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create Delayed Task test executable
add_executable(${DELAYED_TASK_TEST_TARGET} ${DELAYED_TASK_TEST_SOURCES})

target_include_directories(${DELAYED_TASK_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
)

target_link_libraries(${DELAYED_TASK_TEST_TARGET}
  chimaera_cxx
  chimaera_MOD_NAME_client
  chimaera_MOD_NAME_runtime
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${DELAYED_TASK_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${DELAYED_TASK_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create External Client test executable.
# POSIX-only: drives a forked client process via fork()/waitpid()/SIGKILL.
if(NOT WIN32)
add_executable(${EXTERNAL_CLIENT_TEST_TARGET} ${EXTERNAL_CLIENT_TEST_SOURCES})

target_include_directories(${EXTERNAL_CLIENT_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
)

target_link_libraries(${EXTERNAL_CLIENT_TEST_TARGET}
  chimaera_cxx               # Main CLIO Runtime library
  ctp::cxx                  # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

# RuntimeServer launches clio_run; ensure it is built and discoverable.
add_dependencies(${EXTERNAL_CLIENT_TEST_TARGET} clio_run)
target_compile_definitions(${EXTERNAL_CLIENT_TEST_TARGET} PRIVATE
  CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")

set_target_properties(${EXTERNAL_CLIENT_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${EXTERNAL_CLIENT_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif()

# Create Runtime Cleanup test executable
add_executable(${RUNTIME_CLEANUP_TEST_TARGET} ${RUNTIME_CLEANUP_TEST_SOURCES})

target_include_directories(${RUNTIME_CLEANUP_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
)

target_link_libraries(${RUNTIME_CLEANUP_TEST_TARGET}
  chimaera_cxx               # Main CLIO Runtime library
  ctp::cxx                  # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${RUNTIME_CLEANUP_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${RUNTIME_CLEANUP_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create IPC Errors test executable
add_executable(${IPC_ERRORS_TEST_TARGET} ${IPC_ERRORS_TEST_SOURCES})

target_include_directories(${IPC_ERRORS_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
)

target_link_libraries(${IPC_ERRORS_TEST_TARGET}
  chimaera_cxx               # Main CLIO Runtime library
  ctp::cxx                  # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

set_target_properties(${IPC_ERRORS_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${IPC_ERRORS_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create IPC Transport Modes test executable.
# Launches the runtime out-of-process via chi::test::RuntimeServer (clio_run
# start) instead of fork()+CHIMAERA_INIT(kServer). POSIX-only for now: the
# RuntimeServer helper's Windows path pulls in <windows.h>, which collides with
# the ctp lightbeam headers' <winsock2.h> and leaks the Yield/min/max macros
# into widely-included headers (the project deliberately keeps <windows.h> out
# of shared headers -- see simple_test.h). Making this build on Windows is part
# of the macOS/Windows transport follow-up (see issue #476).
if(NOT WIN32)
add_executable(${IPC_TRANSPORT_MODES_TEST_TARGET} ${IPC_TRANSPORT_MODES_TEST_SOURCES})

target_include_directories(${IPC_TRANSPORT_MODES_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h / runtime_server.h
  ${CLIO_RUN_ROOT}/modules/bdev/include
  ${CLIO_RUN_ROOT}/modules/admin/include
)

target_link_libraries(${IPC_TRANSPORT_MODES_TEST_TARGET}
  chimaera_cxx               # Main CLIO Runtime library
  clio_bdev_client        # Bdev module client
  clio_admin_client       # Admin module client
  ctp::cxx                  # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

# RuntimeServer launches this binary; ensure it is built and tell the helper
# where it lives.
add_dependencies(${IPC_TRANSPORT_MODES_TEST_TARGET} clio_run)
target_compile_definitions(${IPC_TRANSPORT_MODES_TEST_TARGET} PRIVATE
  CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")

set_target_properties(${IPC_TRANSPORT_MODES_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${IPC_TRANSPORT_MODES_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif()

# Create Client Retry test executable.
# POSIX-only: uses <sys/wait.h> / fork() for parent-child retry orchestration.
if(NOT WIN32)
add_executable(${CLIENT_RETRY_TEST_TARGET} ${CLIENT_RETRY_TEST_SOURCES})

target_include_directories(${CLIENT_RETRY_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/bdev/include
  ${CLIO_RUN_ROOT}/modules/admin/include
)

target_link_libraries(${CLIENT_RETRY_TEST_TARGET}
  chimaera_cxx               # Main CLIO Runtime library
  clio_bdev_client        # Bdev module client
  clio_admin_client       # Admin module client
  ctp::cxx                  # ClioCtp library
  ${CMAKE_THREAD_LIBS_INIT}  # Threading support
)

# RuntimeServer launches clio_run; ensure it is built and discoverable.
add_dependencies(${CLIENT_RETRY_TEST_TARGET} clio_run)
target_compile_definitions(${CLIENT_RETRY_TEST_TARGET} PRIVATE
  CLIO_RUN_EXE="$<TARGET_FILE:clio_run>")

set_target_properties(${CLIENT_RETRY_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

set_target_properties(${CLIENT_RETRY_TEST_TARGET} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
endif()


# GPU runtime tests removed — kernels now submit tasks to the CPU
# runtime via gpu2cpu_queue (see test/unit/gpu/test_sycl_*.cc).
if(FALSE)
  add_cuda_executable(${IPC_ALLOCATE_BUFFER_GPU_TEST_TARGET} TRUE ${IPC_ALLOCATE_BUFFER_GPU_TEST_SOURCES}
    INCLUDE_DIRS
      ${CLIO_RUN_ROOT}/test
      ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
    LINK_LIBS chimaera_cxx ctp::cuda_cxx
  )
  target_include_directories(${IPC_ALLOCATE_BUFFER_GPU_TEST_TARGET} PRIVATE
    ${CLIO_RUN_ROOT}/include
    ${CLIO_RUN_ROOT}/test
    ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
  )
  target_link_libraries(${IPC_ALLOCATE_BUFFER_GPU_TEST_TARGET}
    chimaera_cxx
    ctp::cuda_cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  set_target_properties(${IPC_ALLOCATE_BUFFER_GPU_TEST_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(NAME ${IPC_ALLOCATE_BUFFER_GPU_TEST_TARGET} COMMAND ${IPC_ALLOCATE_BUFFER_GPU_TEST_TARGET})

  # GPU runtime / buddy-benchmark CUDA-only tests removed along with
  # the GPU runtime concept.
endif()

# GPU producer-only stress test.
#
# Three flavors compile from the same `test_gpu_kernel_stress_common.h`
# host fixture, one per backend. Each runs the same end-to-end pipeline
# (allocate registered backend -> place N tasks -> launch a kernel
# that submits all N concurrently via gpu2cpu_queue -> CPU worker
# dispatches MOD_NAME::Runtime::GpuSubmit -> verify formula).
if(CLIO_CORE_ENABLE_CUDA)
  add_cuda_executable(test_gpu_kernel_stress FALSE
    gpu/test_gpu_kernel_stress_gpu.cc
    INCLUDE_DIRS
      ${CLIO_RUN_ROOT}/include
      ${CLIO_RUN_ROOT}/test
      ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
      ${CLIO_RUN_ROOT}/modules/admin/include
    LINK_LIBS
      chimaera_cxx
      chimaera_cxx_gpu
      clio_admin_client
      chimaera_MOD_NAME_client
      ctp::cuda_cxx
      ${CMAKE_THREAD_LIBS_INIT}
  )
  set_target_properties(test_gpu_kernel_stress PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(NAME cr_gpu_kernel_stress_cuda COMMAND test_gpu_kernel_stress
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_gpu_kernel_stress_cuda PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 120
    LABELS "gpu;cuda;stress"
  )
endif()

if(CLIO_CORE_ENABLE_ROCM)
  add_rocm_gpu_executable(test_gpu_kernel_stress FALSE
    gpu/test_gpu_kernel_stress_gpu.cc
  )
  target_include_directories(test_gpu_kernel_stress PUBLIC
    ${CLIO_RUN_ROOT}/include
    ${CLIO_RUN_ROOT}/test
    ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
    ${CLIO_RUN_ROOT}/modules/admin/include
  )
  target_link_libraries(test_gpu_kernel_stress PUBLIC
    chimaera_cxx
    chimaera_cxx_gpu
    clio_admin_client
    chimaera_MOD_NAME_client
    ctp::rocm_cxx
    ${CMAKE_THREAD_LIBS_INIT}
  )
  set_target_properties(test_gpu_kernel_stress PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(NAME cr_gpu_kernel_stress_rocm COMMAND test_gpu_kernel_stress
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_gpu_kernel_stress_rocm PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 120
    LABELS "gpu;rocm;stress"
  )
endif()

if(CLIO_CORE_ENABLE_SYCL)
  add_sycl_executable(test_gpu_kernel_stress FALSE
    gpu/test_gpu_kernel_stress_sycl.cc
    INCLUDE_DIRS
      ${CLIO_RUN_ROOT}/include
      ${CLIO_RUN_ROOT}/test
      ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
      ${CLIO_RUN_ROOT}/modules/admin/include
    LINK_LIBS
      chimaera_cxx
      chimaera_cxx_gpu
      clio_admin_client
      chimaera_MOD_NAME_client
      ctp::cxx
      ${CMAKE_THREAD_LIBS_INIT}
  )
  set_target_properties(test_gpu_kernel_stress PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  add_test(NAME cr_gpu_kernel_stress_sycl COMMAND test_gpu_kernel_stress
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_gpu_kernel_stress_sycl PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 120
    LABELS "gpu;sycl;stress"
  )
endif()

# GPU task-trace, POD-transfer, and runtime tests were removed along
# with the GPU runtime. Coverage of the kernel→CPU path now lives in
# the SYCL kernel-side tests (test_sycl_chimod_to_cpu, test_sycl_cte_*).

# GPU coroutine test removed along with the GPU runtime concept.

# Enable CTest integration if testing is enabled
if(CLIO_CORE_ENABLE_TESTS)
  # Core Runtime Tests
  # Single ctest entry runs the whole binary in one chimaera init/finalize cycle.
  # Previously 8 redundant entries each invoked the full unfiltered binary, which
  # serialized via RESOURCE_LOCK but multiplied chimaera shutdown ZMQ-race exposure.
  add_test(
    NAME cr_unit_tests
    COMMAND ${TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_unit_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CHIMAERA_TEST_MODE=1;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 600
  )

  # IPC AllocateBuffer Tests
  add_test(
    NAME cr_ipc_allocate_buffer_basic_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_basic_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_types_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_types_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_sizes_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_sizes_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_multiple_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_multiple_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_client_runtime_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_client_runtime_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_error_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_error_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_alignment_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_alignment_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  add_test(
    NAME cr_ipc_allocate_buffer_documentation_tests
    COMMAND ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_allocate_buffer_documentation_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
  )

  # Per-Process Shared Memory Tests (POSIX fork-based — Linux/macOS only)
  if(NOT WIN32)
  add_test(
    NAME cr_per_process_shm_tests
    COMMAND ${PER_PROCESS_SHM_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_per_process_shm_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 600
  )
  endif()

  # Autogen Coverage Tests (requires CAE)
  if(CLIO_CORE_ENABLE_CAE)
  add_test(
    NAME cr_autogen_coverage_tests
    COMMAND ${AUTOGEN_COVERAGE_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_autogen_coverage_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
    LABELS "msan_skip"
  )
  endif()  # CLIO_CORE_ENABLE_CAE

  # Streaming Tests
  add_test(
    NAME cr_streaming_small_output_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_streaming_small_output_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 60
  )

  add_test(
    NAME cr_streaming_large_output_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_streaming_large_output_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 60
  )

  add_test(
    NAME cr_streaming_bitfield_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_streaming_bitfield_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 60
  )

  add_test(
    NAME cr_all_streaming_tests
    COMMAND ${STREAMING_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_all_streaming_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 180
  )

  # Delayed Task Tests — runtime starts, idles 3s, then fires a MOD_NAME task
  add_test(
    NAME cr_delayed_task_tests
    COMMAND ${DELAYED_TASK_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_delayed_task_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 60
    LABELS "msan_skip"
  )

  # External Client Tests (POSIX fork-based — Linux/macOS only)
  # NOTE: Each test case must run in its own process because CHIMAERA_INIT has
  # a static guard that prevents re-initialization. Running all tests in one
  # process causes the static guard to return true for subsequent calls,
  # breaking tests that expect initialization failure.
  if(NOT WIN32)
  add_test(
    NAME cr_external_client_basic_connection
    COMMAND ${EXTERNAL_CLIENT_TEST_TARGET} "ExternalClient - Basic Connection"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_external_client_basic_connection PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 120
  )

  add_test(
    NAME cr_external_client_multiple_clients
    COMMAND ${EXTERNAL_CLIENT_TEST_TARGET} "ExternalClient - Multiple Clients"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_external_client_multiple_clients PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 120
  )

  add_test(
    NAME cr_external_client_without_server
    COMMAND ${EXTERNAL_CLIENT_TEST_TARGET} "ExternalClient - Connection Without Server"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_external_client_without_server PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 120
  )

  add_test(
    NAME cr_external_client_operations
    COMMAND ${EXTERNAL_CLIENT_TEST_TARGET} "ExternalClient - Client Operations"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_external_client_operations PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 120
  )
  endif()

  # Runtime Cleanup Tests
  add_test(
    NAME cr_runtime_cleanup_tests
    COMMAND ${RUNTIME_CLEANUP_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_runtime_cleanup_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 120
  )

  # IPC Errors Tests
  add_test(
    NAME cr_ipc_errors_tests
    COMMAND ${IPC_ERRORS_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_errors_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 120
  )

  # IPC Transport Mode Tests (POSIX fork-based — Linux/macOS only)
  # NOTE: Each test case must run in its own process because CHIMAERA_INIT has
  # a static guard that prevents re-initialization.
  if(NOT WIN32)
  add_test(
    NAME cr_ipc_transport_shm
    COMMAND ${IPC_TRANSPORT_MODES_TEST_TARGET} "IpcTransportMode - SHM Client Connection"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_transport_shm PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )

  add_test(
    NAME cr_ipc_transport_tcp
    COMMAND ${IPC_TRANSPORT_MODES_TEST_TARGET} "IpcTransportMode - TCP Client Connection"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_transport_tcp PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )

  add_test(
    NAME cr_ipc_transport_ipc
    COMMAND ${IPC_TRANSPORT_MODES_TEST_TARGET} "IpcTransportMode - IPC Client Connection"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_transport_ipc PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )

  add_test(
    NAME cr_ipc_transport_default
    COMMAND ${IPC_TRANSPORT_MODES_TEST_TARGET} "IpcTransportMode - Default Mode Is TCP"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_transport_default PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )
  endif()  # NOT WIN32 (ipc_transport_modes tests)

  # Client Retry Tests (POSIX fork-based — Linux/macOS only)
  # NOTE: Each test case must run in its own process because CHIMAERA_INIT has
  # a static guard that prevents re-initialization.
  if(NOT WIN32)
  add_test(
    NAME cr_client_retry_server_restart_tcp
    COMMAND ${CLIENT_RETRY_TEST_TARGET} "ClientRetry - Server Restart TCP"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_client_retry_server_restart_tcp PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )

  add_test(
    NAME cr_client_retry_server_restart_ipc
    COMMAND ${CLIENT_RETRY_TEST_TARGET} "ClientRetry - Server Restart IPC"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_client_retry_server_restart_ipc PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )

  add_test(
    NAME cr_client_retry_server_restart_shm
    COMMAND ${CLIENT_RETRY_TEST_TARGET} "ClientRetry - Server Restart SHM"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_client_retry_server_restart_shm PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )

  add_test(
    NAME cr_client_retry_client_death_tcp
    COMMAND ${CLIENT_RETRY_TEST_TARGET} "ClientRetry - Client Death TCP"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_client_retry_client_death_tcp PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )

  add_test(
    NAME cr_client_retry_client_death_ipc
    COMMAND ${CLIENT_RETRY_TEST_TARGET} "ClientRetry - Client Death IPC"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_client_retry_client_death_ipc PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )

  add_test(
    NAME cr_client_retry_client_death_shm
    COMMAND ${CLIENT_RETRY_TEST_TARGET} "ClientRetry - Client Death SHM"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_client_retry_client_death_shm PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 300
  )
  endif()

  # Unordered Map LL Tests (standalone test with custom main)
  add_test(
    NAME cr_unordered_map_ll_tests
    COMMAND ${UNORDERED_MAP_LL_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_unordered_map_ll_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
    TIMEOUT 180
  )

  # Mark all CR runtime tests as msan_skip (they use ZMQ which is uninstrumented).
  # The fork-based tests are excluded on Windows (not built there).
  set(_msan_skip_tests
    cr_unit_tests
    cr_ipc_allocate_buffer_basic_tests
    cr_ipc_allocate_buffer_types_tests
    cr_ipc_allocate_buffer_sizes_tests
    cr_ipc_allocate_buffer_multiple_tests
    cr_ipc_allocate_buffer_client_runtime_tests
    cr_ipc_allocate_buffer_error_tests
    cr_ipc_allocate_buffer_alignment_tests
    cr_ipc_allocate_buffer_documentation_tests
    cr_streaming_small_output_tests
    cr_streaming_large_output_tests
    cr_streaming_bitfield_tests
    cr_all_streaming_tests
    cr_runtime_cleanup_tests
    cr_ipc_errors_tests
    cr_unordered_map_ll_tests
  )
  if(NOT WIN32)
    list(APPEND _msan_skip_tests
      cr_per_process_shm_tests
      cr_external_client_basic_connection
      cr_external_client_multiple_clients
      cr_external_client_without_server
      cr_external_client_operations
      cr_client_retry_server_restart_tcp
      cr_client_retry_server_restart_ipc
      cr_client_retry_server_restart_shm
      cr_client_retry_client_death_tcp
      cr_client_retry_client_death_ipc
      cr_client_retry_client_death_shm
      cr_ipc_transport_shm
      cr_ipc_transport_tcp
      cr_ipc_transport_ipc
      cr_ipc_transport_default
    )
  endif()
  set_tests_properties(${_msan_skip_tests} PROPERTIES LABELS "msan_skip")
endif()

# macOS: the cross-process client/transport tests reach the runtime through a
# ZeroMQ DEALER->ROUTER handshake (WaitForLocalServer always probes over ZMQ
# regardless of the test's data mode). On macOS the daemon receives the request
# but every ROUTER reply to the connected DEALER's known identity fails
# EHOSTUNREACH -- a libzmq-on-Darwin outbound-routing anomaly that persists even
# when the send is issued from the same thread that received the request and
# even with an explicit ZMQ_IDENTITY (see issue #482). Disable (not silently
# exclude) the affected tests on macOS so the rest of the suite can gate; the
# IPC unix-socket transport variants are unaffected and stay enabled.
if(APPLE)
  set(_macos_zmq_router_disabled
    cr_per_process_shm_tests
    cr_external_client_basic_connection
    cr_external_client_multiple_clients
    cr_external_client_operations
    cr_ipc_transport_shm
    cr_ipc_transport_tcp
    cr_ipc_transport_default
    cr_client_retry_server_restart_tcp
    cr_client_retry_client_death_tcp
    cr_client_retry_client_death_shm
  )
  set_tests_properties(${_macos_zmq_router_disabled} PROPERTIES DISABLED TRUE)
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

# Custom targets for per-process shared memory tests (Linux/macOS only — the
# underlying binary is fork-based so it isn't built on Windows).
if(NOT WIN32)
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
endif()  # NOT WIN32

# Custom target to run all core unit tests
add_custom_target(test_all_unit
  COMMAND ${TEST_TARGET}
  DEPENDS ${TEST_TARGET}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running all Chimaera core unit tests"
)

# Install test executables.
# Fork-based POSIX-only tests are skipped on Windows (see add_executable guards).
set(_install_test_targets
  ${TEST_TARGET}
  ${IPC_ALLOCATE_BUFFER_TEST_TARGET}
  ${UNORDERED_MAP_LL_TEST_TARGET}
  ${SAVE_LOAD_TASK_TEST_TARGET}
  ${LOCAL_TASK_ARCHIVE_TEST_TARGET}
  ${STREAMING_TEST_TARGET}
  ${DELAYED_TASK_TEST_TARGET}
  ${RUNTIME_CLEANUP_TEST_TARGET}
  ${IPC_ERRORS_TEST_TARGET}
)
if(NOT WIN32)
  list(APPEND _install_test_targets
    ${PER_PROCESS_SHM_TEST_TARGET}
    ${EXTERNAL_CLIENT_TEST_TARGET}
    ${CLIENT_RETRY_TEST_TARGET}
    ${IPC_TRANSPORT_MODES_TEST_TARGET}
  )
endif()
if(CLIO_CORE_ENABLE_CAE)
  list(APPEND _install_test_targets ${AUTOGEN_COVERAGE_TEST_TARGET})
endif()
install(TARGETS ${_install_test_targets} RUNTIME DESTINATION bin)

# Note: Module-specific tests are now in their respective module directories
# Note: Distributed tests are run via Docker using the chimaera_bdev_chimod_tests binary
# See test/unit/distributed/docker-compose.yml and run_tests.sh for distributed test execution

# MPI BDev Transport Integration Test
if(CLIO_CORE_ENABLE_MPI)
  set(MPI_BDEV_TRANSPORT_TEST_TARGET chimaera_mpi_bdev_transport_tests)
  add_executable(${MPI_BDEV_TRANSPORT_TEST_TARGET} test_mpi_bdev_transport.cc)

  target_include_directories(${MPI_BDEV_TRANSPORT_TEST_TARGET} PRIVATE
    ${CLIO_RUN_ROOT}/include
    ${CLIO_RUN_ROOT}/test
    ${CLIO_RUN_ROOT}/modules/bdev/include
    ${CLIO_RUN_ROOT}/modules/admin/include
  )

  target_link_libraries(${MPI_BDEV_TRANSPORT_TEST_TARGET}
    chimaera_cxx
    clio_bdev_client
    clio_admin_client
    ctp::cxx
    ctp::mpi
    ${CMAKE_THREAD_LIBS_INIT}
  )

  set_target_properties(${MPI_BDEV_TRANSPORT_TEST_TARGET} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  install(TARGETS ${MPI_BDEV_TRANSPORT_TEST_TARGET}
    RUNTIME DESTINATION bin
  )

  if(CLIO_CORE_ENABLE_TESTS)
    # 4-rank test: server + SHM + TCP + IPC clients
    add_test(
      NAME cr_mpi_bdev_transport_all_modes
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
              ${MPIEXEC_PREFLAGS}
              ${CMAKE_BINARY_DIR}/bin/${MPI_BDEV_TRANSPORT_TEST_TARGET}
              ${MPIEXEC_POSTFLAGS}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cr_mpi_bdev_transport_all_modes PROPERTIES
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
      RESOURCE_LOCK "chimaera_runtime_port"
      TIMEOUT 300
    )

    # 2-rank test: server + TCP client (minimal)
    add_test(
      NAME cr_mpi_bdev_transport_minimal
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              ${MPIEXEC_PREFLAGS}
              ${CMAKE_BINARY_DIR}/bin/${MPI_BDEV_TRANSPORT_TEST_TARGET}
              ${MPIEXEC_POSTFLAGS}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    set_tests_properties(cr_mpi_bdev_transport_minimal PROPERTIES
      ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10500"
      RESOURCE_LOCK "chimaera_runtime_port"
      TIMEOUT 180
    )
  endif()
endif()

# Print configuration summary
message(STATUS "Chimaera core unit tests configured:")
message(STATUS "  Test target: ${TEST_TARGET}")
message(STATUS "  Test sources: ${TEST_SOURCES}")
message(STATUS "  IPC AllocateBuffer test target: ${IPC_ALLOCATE_BUFFER_TEST_TARGET}")
message(STATUS "  Per-Process SHM test target: ${PER_PROCESS_SHM_TEST_TARGET}")
message(STATUS "  CTest enabled: ${CLIO_CORE_ENABLE_TESTS}")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "")
message(STATUS "Module-specific tests are in:")
message(STATUS "  - modules/admin/test/")
message(STATUS "  - modules/bdev/test/")
message(STATUS "  - modules/MOD_NAME/test/")
