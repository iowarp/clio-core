if(WRP_CORE_ENABLE_ZMQ)
    add_executable(lightbeam_transport_test lightbeam_transport_test.cc)
    target_link_libraries(lightbeam_transport_test hermes_shm_host hshm::lightbeam hshm::serialize)
    target_compile_definitions(lightbeam_transport_test PRIVATE HSHM_ENABLE_ZMQ)
    add_test(NAME lightbeam_transport_test COMMAND lightbeam_transport_test)
    set_tests_properties(lightbeam_transport_test PROPERTIES
        LABELS "msan_skip"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
    install(TARGETS lightbeam_transport_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    add_executable(test_lightbeam_new test_lightbeam_new.cc)
    target_link_libraries(test_lightbeam_new hermes_shm_host hshm::lightbeam hshm::serialize)
    target_compile_definitions(test_lightbeam_new PRIVATE HSHM_ENABLE_ZMQ)
    add_test(NAME ctp_lightbeam_new COMMAND test_lightbeam_new)
    set_tests_properties(ctp_lightbeam_new PROPERTIES
        LABELS "msan_skip"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
    install(TARGETS test_lightbeam_new
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

add_executable(socket_transport_test socket_transport_test.cc)
target_link_libraries(socket_transport_test hermes_shm_host hshm::lightbeam hshm::serialize)
add_test(NAME ctp_socket_transport COMMAND socket_transport_test)
set_tests_properties(ctp_socket_transport PROPERTIES
    LABELS "msan_skip"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
install(TARGETS socket_transport_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_executable(shm_transport_test shm_transport_test.cc)
target_link_libraries(shm_transport_test hermes_shm_host hshm::lightbeam hshm::serialize)
add_test(NAME ctp_shm_transport COMMAND shm_transport_test)
set_tests_properties(ctp_shm_transport PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
install(TARGETS shm_transport_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_executable(shm_transfer_test shm_transfer_test.cc)
target_link_libraries(shm_transfer_test hermes_shm_host hshm::lightbeam Catch2::Catch2WithMain)
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

# EventManager test (always enabled - uses pipe/epoll + socket transport)
add_executable(event_manager_test event_manager_test.cc)
target_link_libraries(event_manager_test hermes_shm_host hshm::lightbeam hshm::serialize)
if(WRP_CORE_ENABLE_ZMQ)
    target_compile_definitions(event_manager_test PRIVATE HSHM_ENABLE_ZMQ)
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
if(WRP_CORE_ENABLE_ZMQ)
    add_executable(zmq_transport_full_test zmq_transport_full_test.cc)
    target_link_libraries(zmq_transport_full_test hermes_shm_host hshm::lightbeam hshm::serialize)
    target_compile_definitions(zmq_transport_full_test PRIVATE HSHM_ENABLE_ZMQ)
    add_test(NAME ctp_zmq_transport_full COMMAND zmq_transport_full_test)
    set_tests_properties(ctp_zmq_transport_full PROPERTIES
        LABELS "msan_skip"
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
    install(TARGETS zmq_transport_full_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# Socket transport full test (always enabled)
add_executable(socket_transport_full_test socket_transport_full_test.cc)
target_link_libraries(socket_transport_full_test hermes_shm_host hshm::lightbeam hshm::serialize)
add_test(NAME ctp_socket_transport_full COMMAND socket_transport_full_test)
set_tests_properties(ctp_socket_transport_full PROPERTIES
    LABELS "msan_skip"
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin:$ENV{LD_LIBRARY_PATH}")
install(TARGETS socket_transport_full_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# distributed_lightbeam_test requires MPI
if(WRP_CORE_ENABLE_MPI AND WRP_CORE_ENABLE_ZMQ)
    add_executable(distributed_lightbeam_test distributed_lightbeam_test.cc)
    target_link_libraries(distributed_lightbeam_test
        hermes_shm_host
        hshm::lightbeam
        hshm::mpi
        hshm::serialize
    )
    target_compile_definitions(distributed_lightbeam_test PRIVATE HSHM_ENABLE_ZMQ)
    install(TARGETS distributed_lightbeam_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

