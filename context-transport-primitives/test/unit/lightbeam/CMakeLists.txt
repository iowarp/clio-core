if(HSHM_ENABLE_ZMQ)
    add_executable(lightbeam_transport_test lightbeam_transport_test.cc)
    target_include_directories(lightbeam_transport_test PRIVATE ${ZeroMQ_INCLUDE_DIRS})
    target_link_directories(lightbeam_transport_test PRIVATE ${ZeroMQ_LIBRARY_DIRS})
    target_link_libraries(lightbeam_transport_test hermes_shm_host ${ZeroMQ_LIBRARIES} cereal::cereal)
    target_compile_definitions(lightbeam_transport_test PRIVATE HSHM_ENABLE_ZMQ)
    add_test(NAME lightbeam_transport_test COMMAND lightbeam_transport_test)
    install(TARGETS lightbeam_transport_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    add_executable(test_lightbeam_new test_lightbeam_new.cc)
    target_include_directories(test_lightbeam_new PRIVATE ${ZeroMQ_INCLUDE_DIRS})
    target_link_directories(test_lightbeam_new PRIVATE ${ZeroMQ_LIBRARY_DIRS})
    target_link_libraries(test_lightbeam_new hermes_shm_host ${ZeroMQ_LIBRARIES} cereal::cereal)
    target_compile_definitions(test_lightbeam_new PRIVATE HSHM_ENABLE_ZMQ)
    add_test(NAME test_lightbeam_new COMMAND test_lightbeam_new)
    install(TARGETS test_lightbeam_new
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# distributed_lightbeam_test requires MPI
if(HSHM_ENABLE_MPI)
    add_executable(distributed_lightbeam_test distributed_lightbeam_test.cc)
    target_include_directories(distributed_lightbeam_test PRIVATE
        $<$<BOOL:${HSHM_ENABLE_ZMQ}>:${ZeroMQ_INCLUDE_DIRS}>
        $<$<BOOL:${HSHM_ENABLE_THALLIUM}>:${Thallium_INCLUDE_DIRS}>
        $<$<BOOL:${HSHM_ENABLE_LIBFABRIC}>:${Libfabric_INCLUDE_DIRS}>
    )
    target_link_directories(distributed_lightbeam_test PRIVATE
        $<$<BOOL:${HSHM_ENABLE_ZMQ}>:${ZeroMQ_LIBRARY_DIRS}>
        $<$<BOOL:${HSHM_ENABLE_THALLIUM}>:${Thallium_LIBRARY_DIRS}>
        $<$<BOOL:${HSHM_ENABLE_LIBFABRIC}>:${Libfabric_LIBRARY_DIRS}>
    )
    target_link_libraries(distributed_lightbeam_test
        hermes_shm_host
        $<$<BOOL:${HSHM_ENABLE_ZMQ}>:${ZeroMQ_LIBRARIES}>
        $<$<BOOL:${HSHM_ENABLE_THALLIUM}>:${Thallium_LIBRARIES}>
        $<$<BOOL:${HSHM_ENABLE_LIBFABRIC}>:${Libfabric_LIBRARIES}>
        ${MPI_LIBS}
        $<$<BOOL:${HSHM_ENABLE_THALLIUM}>:thallium>
        cereal::cereal
    )
    install(TARGETS distributed_lightbeam_test
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

