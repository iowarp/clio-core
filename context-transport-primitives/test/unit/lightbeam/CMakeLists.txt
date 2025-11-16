if(HSHM_ENABLE_ZMQ)
    add_executable(lightbeam_transport_test lightbeam_transport_test.cc)
    target_link_libraries(lightbeam_transport_test hermes_shm_host hshm::lightbeam hshm::serialize)
    target_compile_definitions(lightbeam_transport_test PRIVATE HSHM_ENABLE_ZMQ)
    add_test(NAME lightbeam_transport_test COMMAND lightbeam_transport_test)
    install(TARGETS lightbeam_transport_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    add_executable(test_lightbeam_new test_lightbeam_new.cc)
    target_link_libraries(test_lightbeam_new hermes_shm_host hshm::lightbeam hshm::serialize)
    target_compile_definitions(test_lightbeam_new PRIVATE HSHM_ENABLE_ZMQ)
    add_test(NAME ctp_lightbeam_new COMMAND test_lightbeam_new)
    install(TARGETS test_lightbeam_new
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# distributed_lightbeam_test requires MPI
if(HSHM_ENABLE_MPI AND HSHM_ENABLE_ZMQ)
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

