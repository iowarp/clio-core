project(hermes_shm)

# ------------------------------------------------------------------------------
# Build Tests
# ------------------------------------------------------------------------------
set(SOURCES
        string.cc
        pair.cc

        # tuple.cc
        list.cc
        slist.cc
        vector.cc
        lifo_list_queue.cc
        unordered_map.cc
        charwrap.cc
        chararr.cc
        namespace.cc
        key_set.cc)

if(HSHM_ENABLE_OPENMP)
        list(APPEND SOURCES queue.cc)
endif()

add_executable(test_data_structure_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        ${SOURCES}
)

add_dependencies(test_data_structure_exec hermes_shm_host)
target_link_libraries(test_data_structure_exec
        hermes_shm_host
        Catch2::Catch2
        hshm::thread_all)

# ------------------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------------------

# STRING TESTS
add_test(NAME test_string COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "String")

# CHARBUF TESTS
add_test(NAME test_charbuf COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "Charbuf")

# VECTOR TESTS
add_test(NAME test_vector COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "Vector*")

# LIST TESTS
add_test(NAME test_list COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "*::List*")

# SLIST TESTS
add_test(NAME test_slist COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "Slist*")

# UNORDERED_MAP TESTS
add_test(NAME test_unordered_map COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "UnorderedMap*")

# PAIR TESTS
add_test(NAME test_pair COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "Pair*")

# lifo_list_queue TESTS
add_test(NAME test_lifo_list_queue COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "lifo_list_queueOfMpPage")

if(HSHM_ENABLE_OPENMP)
        # SPSC TESTS
        add_test(NAME test_spsc COMMAND
                ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "TestSpsc*")

        # MPSC TESTS
        add_test(NAME test_mpsc COMMAND
                ${CMAKE_BINARY_DIR}/bin/test_data_structure_exec "TestMpsc*")
endif()

# ------------------------------------------------------------------------------
# Install Targets
# ------------------------------------------------------------------------------
install(TARGETS
        test_data_structure_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

# -----------------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
        set_coverage_flags(test_data_structure_exec)
endif()
