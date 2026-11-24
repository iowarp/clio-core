# lightbeam: network and shared-memory transports and event managers
if(WIN32)
    # Impls that pull in <windows.h>; the headers stay Win32-clean.
    clio_ctp_add_sources(src/socket_win.cc src/event_manager_win.cc)
else()
    clio_ctp_add_sources(src/socket_posix.cc)
endif()

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
