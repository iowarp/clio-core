# Autogen dispatch sweep tests — exercises every method id of the admin,
# bdev, and MOD_NAME modules through the full autogen lib_exec dispatch
# battery (Save/Load/AllocLoad/LocalSave/LocalLoad/LocalAllocLoad/NewCopy/
# Aggregate).
# Requires CAE (which implies CTE): the sweep includes and links the CTE and
# CAE core runtimes, so skip it in CAE-less configs (e.g. the cuda-release
# preset used for the conda GPU package).
if(CLIO_CORE_ENABLE_CAE)

set(AUTOGEN_SWEEP_TEST_TARGET clio_run_autogen_sweep_tests)

add_executable(${AUTOGEN_SWEEP_TEST_TARGET}
  test_autogen_methods_sweep.cc
)

target_include_directories(${AUTOGEN_SWEEP_TEST_TARGET} PRIVATE
  ${CLIO_RUN_ROOT}/include
  ${CLIO_RUN_ROOT}/test  # For simple_test.h
  ${CLIO_RUN_ROOT}/modules/admin/include
  ${CLIO_RUN_ROOT}/modules/bdev/include
  ${CLIO_RUN_ROOT}/modules/MOD_NAME/include
)

target_link_libraries(${AUTOGEN_SWEEP_TEST_TARGET}
  clio_MOD_NAME_runtime
  clio_MOD_NAME_client
  clio_cte_core_runtime
  clio_cte_core_client
  clio_cae_core_runtime
  clio_cae_core_client
  clio_bdev_runtime
  clio_admin_runtime
  clio_bdev_client
  clio_admin_client
  clio_run_cxx
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(${AUTOGEN_SWEEP_TEST_TARGET} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(CLIO_CORE_ENABLE_TESTS)
  add_test(
    NAME cr_autogen_sweep_tests
    COMMAND ${AUTOGEN_SWEEP_TEST_TARGET}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
  set_tests_properties(cr_autogen_sweep_tests PROPERTIES
    ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH};CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10597"
    TIMEOUT 120
    LABELS "msan_skip"
  )
endif()

endif()  # CLIO_CORE_ENABLE_CAE
