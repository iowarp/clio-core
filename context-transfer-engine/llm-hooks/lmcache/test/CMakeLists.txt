cmake_minimum_required(VERSION 3.20)

add_executable(test_iowarp_lmcache_store test_lmcache_store.cc)
target_link_libraries(test_iowarp_lmcache_store PRIVATE
    clio_llm_lmcache
    Catch2::Catch2WithMain
)
add_test(NAME test_iowarp_lmcache_store COMMAND test_iowarp_lmcache_store)
# Run the CTE runtime-integration cases (embedded runtime, RAM-only config)
# instead of skipping them; MakeStore still SKIPs gracefully when the runtime
# cannot initialize on a given platform.
set_tests_properties(test_iowarp_lmcache_store PROPERTIES
    ENVIRONMENT "CLIO_LMCACHE_TEST_RUNTIME=1;CLIO_WITH_RUNTIME=1;CLIO_SERVER_CONF=${CMAKE_CURRENT_SOURCE_DIR}/lmcache_cte_config.yaml;CLIO_IPC_MODE=SHM"
    TIMEOUT 300
)

if(CLIO_CORE_ENABLE_PYTHON AND TARGET clio_cte_lmcache_ext)
    # -E -s isolates from the user's env/site-packages but (unlike -I) keeps
    # the script/cwd path lookup; PYTHONPATH would be ignored under -I, and
    # the extension module lives in the binary dir.
    add_test(
        NAME python_lmcache_store_bytes_test
        COMMAND ${Python_EXECUTABLE} -s ${CMAKE_CURRENT_SOURCE_DIR}/test_lmcache_store.py
        WORKING_DIRECTORY $<TARGET_FILE_DIR:clio_cte_lmcache_ext>
    )
    set_tests_properties(
        python_lmcache_store_bytes_test
        PROPERTIES
        LABELS "python;bindings;lmcache"
        TIMEOUT 120
        DEPENDS clio_cte_lmcache_ext
        ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:clio_cte_lmcache_ext>;CLIO_LMCACHE_TEST_RUNTIME=1;CLIO_WITH_RUNTIME=1;CLIO_SERVER_CONF=${CMAKE_CURRENT_SOURCE_DIR}/lmcache_cte_config.yaml;CLIO_IPC_MODE=SHM"
    )
endif()
