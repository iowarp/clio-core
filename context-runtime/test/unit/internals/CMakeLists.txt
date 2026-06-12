# IpcManager / runtime internals unit tests.
set(IPC_INTERNALS_TEST_TARGET chimaera_ipc_internals_tests)

add_executable(${IPC_INTERNALS_TEST_TARGET}
  test_ipc_manager_internals.cc
)

target_include_directories(${IPC_INTERNALS_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
)

target_link_libraries(${IPC_INTERNALS_TEST_TARGET}
  clio_admin_runtime
  clio_admin_client
  clio_bdev_runtime
  clio_bdev_client
  chimaera_cxx
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${IPC_INTERNALS_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(CLIO_CORE_ENABLE_TESTS)
  add_test(
    NAME cr_ipc_internals_tests
    COMMAND ${IPC_INTERNALS_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_ipc_internals_tests PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10596"
    TIMEOUT 120
    LABELS "msan_skip"
  )
endif()
