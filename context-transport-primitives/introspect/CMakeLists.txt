# introspect: system information and drive-failure prediction
clio_ctp_add_sources(src/system_info.cc)

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
