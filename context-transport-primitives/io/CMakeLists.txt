# io: asynchronous I/O backends (io_uring, libaio, POSIX AIO, IOCP, NIXL)
if(WIN32)
    # The IOCP impl pulls in <windows.h>; its header stays Win32-clean.
    clio_ctp_add_sources(src/iocp_io.cc)
endif()

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
