# Introspect tests

#------------------------------------------------------------------------------
# Drive-failure prediction contract test (cross-platform)
#------------------------------------------------------------------------------
# Links clio_ctp_host, which propagates CTP_HAS_POCO + Poco::Net when Poco was
# found (root find_package(Poco ...)); the test then exercises the real HTTP
# client's graceful-degradation path. Where Poco is absent it asserts the "{}"
# fallback, so the same test is valid on every supported OS.
add_executable(test_drive_prediction_exec test_drive_prediction.cc)
add_dependencies(test_drive_prediction_exec clio_ctp_host)
target_link_libraries(test_drive_prediction_exec
    clio_ctp_host
    Catch2::Catch2WithMain)

add_test(NAME ctp_drive_prediction COMMAND test_drive_prediction_exec)

# Catch2's precompiled main installs a SIGABRT/sigaction handler that MSan
# flags as a false positive (see util/CMakeLists.txt); skip there.
set_tests_properties(ctp_drive_prediction PROPERTIES LABELS "msan_skip")

#------------------------------------------------------------------------------
# Drive-failure prediction END-TO-END test (Linux only)
#------------------------------------------------------------------------------
# Drives a live Python prediction server (real LightGBM models) through the
# production Poco HTTP path. The driver script SKIPs (ctest SKIP_RETURN_CODE 77)
# when prerequisites are absent — non-Linux, git-lfs models not pulled, or the
# python deps not installed — so it never fails a plain build. The dedicated CI
# section installs those deps and runs `ctest -L prediction_e2e`.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_drive_prediction_e2e_exec test_drive_prediction_e2e.cc)
    add_dependencies(test_drive_prediction_e2e_exec clio_ctp_host)
    target_link_libraries(test_drive_prediction_e2e_exec clio_ctp_host)

    add_test(NAME ctp_drive_prediction_e2e
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_prediction_e2e.sh
                $<TARGET_FILE:test_drive_prediction_e2e_exec>
                ${CMAKE_SOURCE_DIR}/prediction_server)
    set_tests_properties(ctp_drive_prediction_e2e PROPERTIES
        LABELS "prediction_e2e"
        SKIP_RETURN_CODE 77
        TIMEOUT 180)
endif()
