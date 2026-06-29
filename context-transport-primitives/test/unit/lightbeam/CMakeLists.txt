if(CLIO_CORE_ENABLE_ZMQ)
    add_executable(lightbeam_transport_test lightbeam_transport_test.cc)
    target_link_libraries(lightbeam_transport_test clio_ctp_host ctp::lightbeam ctp::serialize)
    target_compile_definitions(lightbeam_transport_test PRIVATE CTP_ENABLE_ZMQ)
    add_test(NAME lightbeam_transport_test COMMAND lightbeam_transport_test)
    set_tests_properties(lightbeam_transport_test PROPERTIES
        LABELS "msan_skip"
        RESOURCE_LOCK "lightbeam_port"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
    install(TARGETS lightbeam_transport_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    add_executable(test_lightbeam_new test_lightbeam_new.cc)
    target_link_libraries(test_lightbeam_new clio_ctp_host ctp::lightbeam ctp::serialize)
    target_compile_definitions(test_lightbeam_new PRIVATE CTP_ENABLE_ZMQ)
    add_test(NAME ctp_lightbeam_new COMMAND test_lightbeam_new)
    set_tests_properties(ctp_lightbeam_new PROPERTIES
        LABELS "msan_skip"
        RESOURCE_LOCK "lightbeam_port"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
    install(TARGETS test_lightbeam_new
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

add_executable(socket_transport_test socket_transport_test.cc)
target_link_libraries(socket_transport_test clio_ctp_host ctp::lightbeam ctp::serialize)
add_test(NAME ctp_socket_transport COMMAND socket_transport_test)
set_tests_properties(ctp_socket_transport PROPERTIES
    LABELS "msan_skip"
    RESOURCE_LOCK "lightbeam_port"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
install(TARGETS socket_transport_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_executable(shm_transport_test shm_transport_test.cc)
target_link_libraries(shm_transport_test clio_ctp_host ctp::lightbeam ctp::serialize)
add_test(NAME ctp_shm_transport COMMAND shm_transport_test)
# Skip under MSan: shm_transport uses mmap-backed shared memory regions that
# MSan cannot instrument (no instrumented allocator backing).
set_tests_properties(ctp_shm_transport PROPERTIES
    LABELS "msan_skip"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
install(TARGETS shm_transport_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_executable(shm_transfer_test shm_transfer_test.cc)
target_link_libraries(shm_transfer_test clio_ctp_host ctp::lightbeam Catch2::Catch2WithMain)
add_test(NAME ctp_shm_transfer COMMAND shm_transfer_test)
# Skip under MSan: Catch2 precompiled without MSan ignorelist; ConsoleReporter
# and FatalConditionHandler produce false positives.
set_tests_properties(ctp_shm_transfer PROPERTIES
    LABELS "msan_skip"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
install(TARGETS shm_transfer_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Named MPSC SHM transport test (issue #642): 16KB/1MB single-thread + multi-
# producer 1MB into one consumer.
add_executable(shm_mpsc_transport_test shm_mpsc_transport_test.cc)
target_link_libraries(shm_mpsc_transport_test clio_ctp_host ctp::lightbeam Catch2::Catch2WithMain)
add_test(NAME ctp_shm_mpsc_transport COMMAND shm_mpsc_transport_test)
# Skip under MSan: Catch2 precompiled without MSan ignorelist; mmap-backed SHM.
set_tests_properties(ctp_shm_mpsc_transport PROPERTIES
    LABELS "msan_skip"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
install(TARGETS shm_mpsc_transport_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# EventManager test (always enabled - uses pipe/epoll + socket transport)
add_executable(event_manager_test event_manager_test.cc)
target_link_libraries(event_manager_test clio_ctp_host ctp::lightbeam ctp::serialize)
if(CLIO_CORE_ENABLE_ZMQ)
    target_compile_definitions(event_manager_test PRIVATE CTP_ENABLE_ZMQ)
endif()
add_test(NAME ctp_event_manager COMMAND event_manager_test)
# Skip under MSan: uses socket/epoll system calls with uninstrumented kernel data.
set_tests_properties(ctp_event_manager PROPERTIES
    LABELS "msan_skip"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
install(TARGETS event_manager_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# ZMQ transport full test (conditional on ZMQ)
if(CLIO_CORE_ENABLE_ZMQ)
    add_executable(zmq_transport_full_test zmq_transport_full_test.cc)
    target_link_libraries(zmq_transport_full_test clio_ctp_host ctp::lightbeam ctp::serialize)
    target_compile_definitions(zmq_transport_full_test PRIVATE CTP_ENABLE_ZMQ)
    add_test(NAME ctp_zmq_transport_full COMMAND zmq_transport_full_test)
    set_tests_properties(ctp_zmq_transport_full PROPERTIES
        LABELS "msan_skip"
        RESOURCE_LOCK "lightbeam_port"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
    install(TARGETS zmq_transport_full_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    # Isolated reproduction of the runtime's concurrent ZMQ topology:
    # 4 client senders, 1 client receiver, 1 server receiver, 1 server sender,
    # the two receivers on separate ports. Driven per-scenario via argv.
    add_executable(zmq_runtime_pattern_test zmq_runtime_pattern_test.cc)
    target_link_libraries(zmq_runtime_pattern_test clio_ctp_host ctp::lightbeam ctp::serialize)
    target_compile_definitions(zmq_runtime_pattern_test PRIVATE CTP_ENABLE_ZMQ)
    add_test(NAME ctp_zmq_runtime_pattern COMMAND zmq_runtime_pattern_test both)
    set_tests_properties(ctp_zmq_runtime_pattern PROPERTIES
        LABELS "msan_skip"
        RESOURCE_LOCK "lightbeam_port"
        TIMEOUT 60
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
    install(TARGETS zmq_runtime_pattern_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# Thallium transport test (conditional on Thallium)
if(CLIO_CORE_ENABLE_THALLIUM)
    add_executable(thallium_transport_test thallium_transport_test.cc)
    target_link_libraries(thallium_transport_test clio_ctp_host ctp::lightbeam ctp::serialize)
    add_test(NAME ctp_thallium_transport COMMAND thallium_transport_test)
    set_tests_properties(ctp_thallium_transport PROPERTIES
        LABELS "msan_skip"
        RESOURCE_LOCK "lightbeam_port"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
    install(TARGETS thallium_transport_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# Socket transport full test (always enabled)
add_executable(socket_transport_full_test socket_transport_full_test.cc)
target_link_libraries(socket_transport_full_test clio_ctp_host ctp::lightbeam ctp::serialize)
add_test(NAME ctp_socket_transport_full COMMAND socket_transport_full_test)
set_tests_properties(ctp_socket_transport_full PROPERTIES
    LABELS "msan_skip"
    RESOURCE_LOCK "lightbeam_port"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
install(TARGETS socket_transport_full_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# NIXL transport test (conditional on NIXL)
if(CLIO_CORE_ENABLE_NIXL)
    find_library(NIXL_LIB_TEST NAMES nixl PATHS /usr/local/lib /usr/local/lib/x86_64-linux-gnu
                 /usr/lib /usr/lib/x86_64-linux-gnu)
    find_library(NIXL_BUILD_LIB_TEST NAMES nixl_build PATHS /usr/local/lib /usr/local/lib/x86_64-linux-gnu
                 /usr/lib /usr/lib/x86_64-linux-gnu)
    find_path(NIXL_INCLUDE_TEST NAMES nixl.h PATHS /usr/local/include /usr/include)
    if(NIXL_LIB_TEST AND NIXL_INCLUDE_TEST)
        set(NIXL_LINK_LIBS_TEST ${NIXL_LIB_TEST})
        if(NIXL_BUILD_LIB_TEST)
            list(APPEND NIXL_LINK_LIBS_TEST ${NIXL_BUILD_LIB_TEST})
        endif()
        add_executable(nixl_transport_test nixl_transport_test.cc)
        target_link_libraries(nixl_transport_test clio_ctp_host ctp::lightbeam ${NIXL_LINK_LIBS_TEST})
        target_include_directories(nixl_transport_test PRIVATE ${NIXL_INCLUDE_TEST})
        target_compile_definitions(nixl_transport_test PRIVATE CTP_ENABLE_NIXL=1)
        add_test(NAME ctp_nixl_transport COMMAND nixl_transport_test)
        set_tests_properties(ctp_nixl_transport PROPERTIES
            LABELS "msan_skip"
            ENVIRONMENT "LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
        install(TARGETS nixl_transport_test
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
endif()

# distributed_lightbeam_test requires MPI
if(CLIO_CORE_ENABLE_MPI AND CLIO_CORE_ENABLE_ZMQ)
    add_executable(distributed_lightbeam_test distributed_lightbeam_test.cc)
    target_link_libraries(distributed_lightbeam_test
        clio_ctp_host
        ctp::lightbeam
        ctp::mpi
        ctp::serialize
    )
    target_compile_definitions(distributed_lightbeam_test PRIVATE CTP_ENABLE_ZMQ)
    install(TARGETS distributed_lightbeam_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

