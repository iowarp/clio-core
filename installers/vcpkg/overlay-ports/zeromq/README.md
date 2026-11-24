# zeromq overlay port

A verbatim copy of vcpkg's `ports/zeromq` (4.3.5#3) plus one upstream patch,
`windows-ipc-connect-fallback.patch`. Nothing else differs; diff it against
the registry port before touching anything here.

## Why

On Windows, libzmq's mailbox signaler is an `AF_UNIX` socketpair created by
`make_fdpair()` in `src/ip.cpp`. That function has a tcp/ip fallback for
systems where `AF_UNIX` does not work, but it clears the fallback flag as soon
as `bind()` succeeds -- on the assumption that a successful bind means IPC
works. It does not always: a later `listen()` or `connect()` on the bound
socket can still fail, and the function then returns -1 with no fallback left.

`signaler_t`'s constructor does not check that return value. It leaves `_r`
and `_w` at `retired_fd`, `get_fd()` hands that invalid descriptor to the
poller, and `epoll_ctl(EPOLL_CTL_ADD)` aborts the process:

    Bad file descriptor (src/epoll.cpp:73)
    not a socket (src/epoll.cpp:73)

The two spellings are the same assert with a different `errno`.

This is zeromq/libzmq#4730, fixed by zeromq/libzmq#4734 (merged 2024-08-26).
The fix has never been released: v4.3.5 (2023-10-09) is still the newest tag,
so every vcpkg build of libzmq carries the bug. vcpkg's port already passes
`-DZMQ_HAVE_IPC=0` for MinGW, which sidesteps it there; MSVC keeps IPC on.

It reaches clio because every client process builds a ZMQ context during
`IpcManager::ClientInit`, so any workload that spawns short-lived clients can
hit it. It was found via the netCDF-C suite, where `ncdump_tst_nccopy5` failed
under both the CLIO VFD and the CLIO VOL while passing on the baseline --
reproducibly, and even with the test run in isolation on a clean runner.

## Removing this

Drop the whole directory once a libzmq release past v4.3.5 exists and vcpkg's
port has moved to it. Check with:

    gh api repos/zeromq/libzmq/releases --jq '.[0].tag_name'
