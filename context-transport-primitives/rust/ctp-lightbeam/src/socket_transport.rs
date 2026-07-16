// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Socket transport backend — Rust port of
//! `clio_ctp/lightbeam/socket_transport.h` plus the parts of
//! `clio_ctp/lightbeam/posix_socket.h` (and its `socket_posix.cc` /
//! `socket_win.cc` implementations) that the transport consumes.
//!
//! A [`SocketTransport`] is either a client (one connected stream) or a
//! server (a listener plus a set of accepted connections). Its wire framing
//! is unchanged from C++:
//!
//! ```text
//! [4-byte big-endian metadata length][metadata][bulk0][bulk1]...
//! ```
//!
//! Only `send` descriptors carrying `BULK_XFER` contribute payload bytes;
//! `BULK_EXPOSE` descriptors are metadata-only. The whole frame goes out in
//! one vectored write, mirroring the C++ single-`sendmsg`/`writev` path.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::lbm`) | Rust | Notes |
//! |---|---|---|
//! | `SocketTransport` | [`SocketTransport`] | |
//! | `SocketTransport::SocketTransport(mode, addr, protocol, port)` | [`SocketTransport::new`] | throws → returns [`Result`] |
//! | `~SocketTransport` | `impl Drop for SocketTransport` | unlinks the `ipc` path; streams close themselves |
//! | `Expose` | [`SocketTransport::expose`] | associated fn (does not touch `this` in C++ either) |
//! | `Send` | [`SocketTransport::send`] | |
//! | `Recv` | [`SocketTransport::recv`] | |
//! | `RecvAll` / `RecvMetadata` / `RecvBulks` | `recv_all` / `recv_metadata` / `recv_bulks` | private, take `&mut Stream` |
//! | `AcceptNewClient` | `accept_new_client` | private |
//! | `RemoveFiredEvent` | `remove_fired_event` | private |
//! | `ClearRecvHandles` | [`SocketTransport::clear_recv_handles`] | |
//! | `GetAddress` | [`SocketTransport::get_address`] | |
//! | `IsServerAlive` (member) | [`SocketTransport::is_server_alive`] | |
//! | `RegisterEventManager` / `UnregisterEventManager` | [`SocketTransport::register_event_manager`] / [`SocketTransport::unregister_event_manager`] | see divergence 2 |
//! | `IsClient` / `IsServer` | [`SocketTransport::is_client`] / [`SocketTransport::is_server`] | |
//! | `Transport` (base), `TransportType`, `TransportMode` | [`TransportType`], [`TransportMode`] | see divergence 1 |
//! | `Bulk`, `ClientInfo`, `LbmMeta<>`, `LbmContext` | [`Bulk`], [`ClientInfo`], [`LbmMeta`], [`LbmContext`] | see divergence 1 |
//! | `SocketFiredAction` | [`SocketTransport::push_fired_event`] | see divergence 2 |
//! | `EventInfo` / `EventTrigger` | [`EventInfo`] / [`EventTrigger`] | no `EventAction*` member |
//! | `BULK_EXPOSE` / `BULK_XFER` | [`BULK_EXPOSE`] / [`BULK_XFER`] | same bit positions (0, 1) |
//! | `sock::socket_t` / `sock::kInvalidSocket` | [`SocketId`] / [`INVALID_SOCKET`] | see divergence 3 |
//! | `sock::Connect` / `Listen` / `Accept` | `connect` / `listen` / `accept` | private free fns |
//! | `sock::SendV` / `sock::IoBuffer` | `send_v` / [`std::io::IoSlice`] | |
//! | `sock::RecvExact` | `recv_exact` | same `0` / `EAGAIN` / `-1` returns |
//! | `sock::PollRead` | `poll_read` | see divergence 4 |
//! | `sock::HostToNet32` / `NetToHost32` | [`host_to_net32`] / [`net_to_host32`] | |
//! | `sock::IsServerAlive` (free) | [`is_server_alive`] | |
//! | `sock::UnlinkPath` | `unlink_path` | |
//! | `sock::Close` | `drop` | RAII |
//! | `sock::InitSocketLib` / `CleanupSocketLib` | — | see divergence 5 |
//! | `sock::SetNonBlocking` / `SetTcpNoDelay` | `Stream::set_nonblocking` / `Stream::set_nodelay` | std APIs |
//! | `sock::GetError` / `GetErrorString` | [`std::io::Error`] | see divergence 6 |
//!
//! # Semantic divergences from the C++
//!
//! 1. **Shared lightbeam types are defined locally — and now overlap
//!    `transport.rs`.** [`TransportType`], [`TransportMode`], [`Bulk`],
//!    [`ClientInfo`], [`LbmMeta`], [`LbmContext`], [`EventInfo`] and the bulk
//!    flags all live in `lightbeam.h` / `event_manager.h` on the C++ side.
//!    `transport.rs` was an empty stub when this port was written, so they are
//!    declared here; it has since landed with its own copies plus a
//!    `Transport` trait, and the two now **coexist as distinct types**
//!    (nothing re-exports across the modules, so the crate still compiles).
//!    Unifying them is deliberate follow-up work, not a mechanical
//!    re-export: `transport::Bulk::data` is a `FullPtr` — an *opaque address*
//!    that, by its own contract, is never dereferenced and owns nothing —
//!    whereas [`Bulk::data`] here is an owned `Vec<u8>`. A socket transport
//!    must actually read and write those bytes, so adopting `transport::Bulk`
//!    means raw-pointer `unsafe` through `send`/`recv_bulks` plus a
//!    hand-rolled allocate/free protocol reproducing the C++ malloc sentinel
//!    (divergence 7). The right reconciliation is to give the shared `Bulk` an
//!    ownership-carrying buffer (or pair `FullPtr` with an ownership side
//!    table) and then delete these local copies; until then this module stays
//!    safe and self-contained. Likewise `SocketTransport` does not implement
//!    the `transport::Transport` trait: the trait's `expose`/`send` signatures
//!    are expressed in terms of `FullPtr`/`transport::LbmMeta`. C++ `Transport`
//!    is a non-virtual base dispatched via `type_`, so [`SocketTransport`]
//!    carries `mode_`/`type_` as fields and exposes `IsClient`/`IsServer`
//!    inherently — faithful to the C++, and trait-ready once the `Bulk`
//!    ownership question is settled.
//! 2. **No `EventManager`.** The C++ registers fds with an epoll/kqueue/WSAPoll
//!    `EventManager` and learns about fired events through a
//!    `SocketFiredAction` vtable callback. That module is not ported (and this
//!    crate has no polling dependency), so registration is modelled as
//!    bookkeeping: [`SocketTransport::register_event_manager`] marks the
//!    transport registered and records the fds it *would* watch (readable via
//!    [`SocketTransport::event_registrations`]); an external event loop calls
//!    [`SocketTransport::push_fired_event`] where `SocketFiredAction::Run`
//!    would have pushed. The `Recv` drain/erase/close ordering — including
//!    dropping the registration *before* closing an fd, so a recycled fd
//!    number cannot leave a stale registry entry behind — is ported exactly.
//!    `EventInfo` therefore has no `action_` pointer, and
//!    `UnregisterEventManager`'s "detach without RemoveEvent" (a
//!    use-after-free guard in C++) is a plain flag clear here.
//! 3. **`socket_t` → `SocketId` (`u64`).** C++ has `int` fds on POSIX and
//!    `uintptr_t` SOCKETs on Windows; `ClientInfo::fd_` is an `int` (which
//!    truncates a Windows SOCKET). Rust uses one `u64` id on both platforms,
//!    derived from `AsRawFd`/`AsRawSocket`, with `INVALID_SOCKET = u64::MAX`
//!    (the Windows `~0` convention). Ids are handles for routing/lookup only;
//!    the sockets themselves are owned `TcpStream`/`UnixStream` values.
//! 4. **`poll(2)`/`WSAPoll` are emulated.** `libc` is not a dependency of this
//!    crate and std exposes no `poll`. `poll_read` probes readability with a
//!    1-byte `MSG_PEEK` (`TcpStream::peek`) on a bounded exponential backoff
//!    (50 µs → 1 ms), preserving the C++ contract (`>0` ready — including at
//!    EOF, which `poll` reports as `POLLIN` — `0` timeout, `-1` error,
//!    negative timeout = block forever). Write readiness
//!    (`poll(POLLOUT, 1000)` inside `SendV`) cannot be peeked, so `send_v`
//!    instead retries the vectored write on the same backoff and fails once
//!    writes have stalled for the full 1000 ms — the same observable outcome,
//!    reached by retry rather than by poll. Consequence: readiness is noticed
//!    within ≤1 ms rather than immediately, and both paths burn a little CPU
//!    while waiting.
//! 5. **No `InitSocketLib`/`CleanupSocketLib`.** Rust's std runs `WSAStartup`
//!    itself, so the C++ init/cleanup calls (and the
//!    `SetSocketLibShutdown`/`IsSocketLibShutdown` pair, which exists only to
//!    keep libzmq's teardown from tripping over `WSACleanup`) have no port.
//! 6. **Errors.** C++ reports `int` rc values from `errno`/`WSAGetLastError`.
//!    That is kept for the `Send`/`Recv` surface (`0`, [`EAGAIN`], `-1`, or an
//!    OS code), but constructor failures — where C++ throws
//!    `std::runtime_error` — return [`SocketTransportError`]. Note the C++
//!    already mixes conventions on Windows: `RecvExact` returns `<cerrno>`
//!    `EAGAIN` (11) while `GetError()` yields `WSAEWOULDBLOCK` (10035);
//!    [`EAGAIN`] here is 11, matching what the C++ `rc == EAGAIN` comparisons
//!    actually test against.
//! 7. **`FullPtr<char>` → owned `Vec<u8>`.** `Bulk::data` is an owned buffer.
//!    The C++ tags self-allocated recv buffers with the
//!    `AllocatorId(UINT32_MAX-1, UINT32_MAX-1)` sentinel purely so
//!    `ClearRecvHandles` knows which pointers are `std::malloc`'d and safe to
//!    `std::free` (freeing a CTP `MallocAllocator` pointer there is a
//!    heap-corrupting bad-free). Rust replaces the sentinel with
//!    [`Bulk::owned`], and `Vec` frees itself —
//!    [`SocketTransport::clear_recv_handles`] keeps the C++ shape (clearing
//!    only owned buffers) but can neither leak nor double-free. A zero-sized
//!    `BULK_XFER` bulk allocates nothing here (C++ `malloc(0)`s and tags it);
//!    the observable result is identical.
//! 8. **Metadata encoding is not `GlobalSerialize`-compatible.**
//!    `ctp-serialize` is not a dependency of this crate, so
//!    `encode_meta`/`decode_meta` implement the exact field shape the C++
//!    serializes — `send`, `recv`, `send_bulks`, `recv_bulks`, each `Bulk` as
//!    `(size, flags)` — in a fixed little-endian layout. A Rust peer
//!    interoperates with a Rust peer; it will **not** interoperate with a C++
//!    peer until this routes through the ported `GlobalSerialize`. The framing
//!    around it (4-byte big-endian length prefix) is wire-identical.
//!    Truncated/absurd metadata decodes to an error → `-1`, matching the C++
//!    `catch (const std::exception&)` arm; the length-prefixed allocation uses
//!    `try_reserve`, so a hostile 4 GiB length returns `-1` instead of
//!    throwing `bad_alloc` (C++ allocates `meta_buf` outside its `try` block,
//!    so that throw escapes `RecvMetadata` entirely) or aborting the process.
//! 9. **`SO_SNDBUF`/`SO_RCVBUF` (4 MiB) and the listen backlog are not set.**
//!    std has no API for either, and neither `socket2` nor `libc` is a
//!    dependency. The OS defaults apply, and std's backlog is 128 (C++ asks
//!    for 16). `TCP_NODELAY` *is* set (`set_nodelay`), matching C++.
//!    `SO_REUSEADDR`: std sets it on Unix but not on Windows; C++ sets it
//!    explicitly on both. Throughput, not correctness, is affected.
//! 10. **`SIGPIPE`.** `SetNoSigPipe`/`MSG_NOSIGNAL` have no port: Rust's std
//!     already sets `SO_NOSIGPIPE` on macOS/BSD and ignores `SIGPIPE`
//!     process-wide at startup, so writes to a dead peer return `EPIPE`
//!     instead of killing the process — the same guarantee the C++ works for.
//! 11. **`ipc` (`AF_UNIX`) is Unix-only.** std's `UnixStream`/`UnixListener`
//!     are `cfg(unix)`. The C++ supports `AF_UNIX` on Windows too, so
//!     `protocol == "ipc"` on Windows returns
//!     [`SocketTransportError::UnsupportedProtocol`] instead of connecting.
//!     TCP is fully supported everywhere.
//! 12. **Addresses stay numeric-IPv4-only** (`inet_pton` parity — no DNS, no
//!     IPv6). Servers bind `INADDR_ANY:port`, ignoring `addr`, exactly as C++
//!     `Listen` does. One C++ quirk is *not* reproduced: its free
//!     `IsServerAlive` ignores `inet_pton`'s return value, so an unparseable
//!     address probes `0.0.0.0:port` — which on Linux reaches a local listener
//!     and reports a bogus "alive". [`is_server_alive`] returns `false` for an
//!     unparseable address.
//! 13. **`port` truncation is preserved.** C++ takes `int port` and applies
//!     `htons(static_cast<uint16_t>(port))`; [`SocketTransport::new`] takes
//!     `i32` and truncates with `as u16` identically.
//! 14. **`send_v` fixes two C++ hazards.** (a) C++ caps its `iovec` array at
//!     64 entries but computes `total` over *all* buffers, so a frame with
//!     more than 64 `BULK_XFER` bulks spins forever on a 0-length `sendmsg`;
//!     the Rust port passes every slice and has no cap. (b) A zero-byte write with
//!     bytes still outstanding likewise spins forever in C++; here it returns
//!     `-1`. Additionally, [`SocketTransport::send`] rejects (`-1`) a
//!     `BULK_XFER` bulk whose `size` disagrees with its buffer length — C++
//!     reads `size` bytes from the raw pointer, an out-of-bounds read when the
//!     two disagree.
//! 15. **`Recv`'s server-mode fired-event bookkeeping is ported verbatim**,
//!     including the subtlety that a *successful* read returns immediately
//!     **without** erasing the fired event (more data may be pending on that
//!     fd), while the `EAGAIN` and error paths erase it.
//! 16. **`LbmContext` is ignored**, as in C++ (`(void)ctx` throughout the
//!     socket path); only the fields the socket transport could plausibly
//!     consume are declared. `Transport::PollRecv`/`GetBoundPort` are not part
//!     of `SocketTransport`'s C++ surface (they are ZMQ-only; the base returns
//!     0), so they are not ported. [`SocketTransport::bound_port`] is a Rust
//!     **addition** — with no `GetBoundPort` for sockets, an ephemeral
//!     (`port = 0`) bind would otherwise be unusable, and tests need it.

use std::fmt;
use std::io::{self, IoSlice, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;

use ctp_memory::ShmPtr;
use ctp_types::bit_opt;

// ---------------------------------------------------------------------------
// Constants (lightbeam.h / event_manager.h / posix_socket.h)
// ---------------------------------------------------------------------------

/// `BULK_EXPOSE` — bulk metadata is sent, but no payload bytes are.
pub const BULK_EXPOSE: u32 = bit_opt(0);

/// `BULK_XFER` — bulk is marked for data transmission.
pub const BULK_XFER: u32 = bit_opt(1);

/// `LBM_SYNC` — retained as a no-op flag, exactly as in C++.
pub const LBM_SYNC: u32 = 0x1;

/// `<cerrno>` `EAGAIN`. 11 on Linux and MSVC alike; this is the value the C++
/// `rc == EAGAIN` comparisons test against on every platform (divergence 6).
pub const EAGAIN: i32 = 11;

pub use crate::transport::INVALID_SOCKET;

/// `kDefaultReadEvent` — `EPOLLIN` on POSIX.
#[cfg(not(windows))]
pub const DEFAULT_READ_EVENT: u32 = 0x001;
/// `kDefaultReadEvent` — `POLLRDNORM` on Windows.
#[cfg(windows)]
pub const DEFAULT_READ_EVENT: u32 = 0x0100;

/// Default port of the C++ constructor's `int port = 8193`.
pub const DEFAULT_PORT: i32 = 8193;

/// The C++ `PollRead(fd, 1000)` / `poll(POLLOUT, 1000)` stall budget, in **ms**.
const STALL_TIMEOUT_MS: i64 = 1000;

/// `IsServerAlive`'s connect timeout, in **ms** (C++ `SO_SNDTIMEO` 500000 µs).
const ALIVE_PROBE_TIMEOUT_MS: u64 = 500;

/// Readiness-backoff bounds for the emulated `poll` (divergence 4).
const POLL_MIN_SLEEP: Duration = Duration::from_micros(50);
const POLL_MAX_SLEEP: Duration = Duration::from_millis(1);

/// Bytes each serialized `Bulk` occupies: `size: u64` + `flags: u32`.
const BULK_WIRE_BYTES: usize = 12;

pub use crate::transport::SocketId;

// ---------------------------------------------------------------------------
// lightbeam.h types
// ---------------------------------------------------------------------------

/// `ctp::lbm::TransportType` / `ctp::lbm::TransportMode`.
///
/// These were "declared locally until transport.rs lands". It has landed, so
/// they come from [`crate::transport`] — the module that ports `lightbeam.h`,
/// the header declaring them once that this module's C++ counterpart
/// includes. Re-exported so this module's spelling is the same type.
pub use crate::transport::{TransportMode, TransportType};

/// `ctp::lbm::Bulk` — from [`crate::transport`] (`lightbeam.h`), which
/// declares it once. This module used to declare its own, holding an owned
/// `Vec<u8>` plus an `owned: bool`; see the type's docs and
/// [`RECV_ALLOCATED_ID`] for why the buffer is pointed at rather than owned,
/// and why ownership rides on the allocator id rather than a flag.
pub use crate::transport::{Bulk, FullPtr, RECV_ALLOCATED_ID};

/// `ctp::lbm::ClientInfo` and `LbmMeta` — from [`crate::transport`]
/// (`lightbeam.h`), which declares them once.
///
/// This module used to declare its own of each. `ClientInfo` differed only in
/// `fd`'s width, and that was the last thing keeping `LbmMeta` split three
/// ways: `LbmMeta` embeds a `ClientInfo`, so it could not unify until the
/// width was settled. It is settled in favour of the `u64` this module argued
/// for — see [`SocketId`].
pub use crate::transport::{ClientInfo, LbmMeta};
use crate::transport::{alloc_recv_buffer, bulk_bytes, free_recv_buffer};

/// `ctp::lbm::LbmContext`. Ignored by the socket backend (divergence 16).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LbmContext {
    /// Combination of `LBM_*` flags.
    pub flags: u32,
    /// Timeout in **milliseconds** (0 = no timeout).
    pub timeout_ms: i32,
}

impl LbmContext {
    pub const fn new(flags: u32, timeout_ms: i32) -> Self {
        Self { flags, timeout_ms }
    }

    /// `IsSync()`.
    pub const fn is_sync(&self) -> bool {
        (self.flags & LBM_SYNC) != 0
    }

    /// `HasTimeout()`.
    pub const fn has_timeout(&self) -> bool {
        self.timeout_ms > 0
    }
}

/// `ctp::lbm::EventTrigger`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventTrigger {
    pub fd: SocketId,
    pub event_id: i32,
}

/// `ctp::lbm::EventInfo`, minus the `EventAction*` (divergence 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventInfo {
    pub trigger: EventTrigger,
    pub events: u32,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Constructor failures. C++ throws `std::runtime_error` (divergence 6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SocketTransportError {
    /// `"SocketTransport: failed to connect to {addr}:{port}"`.
    Connect { addr: String, port: i32 },
    /// `"SocketTransport: failed to listen on {addr}:{port}"`.
    Listen { addr: String, port: i32 },
    /// `ipc` requested on a platform whose std has no `AF_UNIX`
    /// (divergence 11).
    UnsupportedProtocol(String),
}

impl fmt::Display for SocketTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect { addr, port } => {
                write!(f, "SocketTransport: failed to connect to {addr}:{port}")
            }
            Self::Listen { addr, port } => {
                write!(f, "SocketTransport: failed to listen on {addr}:{port}")
            }
            Self::UnsupportedProtocol(p) => write!(
                f,
                "SocketTransport: protocol '{p}' is unsupported on this platform"
            ),
        }
    }
}

impl std::error::Error for SocketTransportError {}

// ---------------------------------------------------------------------------
// `sock` layer: one stream/listener abstraction over TCP and AF_UNIX
// ---------------------------------------------------------------------------

/// A connected socket — `sock::socket_t` in owning form.
#[derive(Debug)]
enum Stream {
    Tcp(TcpStream),
    #[cfg(unix)]
    Uds(UnixStream),
}

impl Stream {
    /// `sock::socket_t` identity of this stream (divergence 3).
    fn id(&self) -> SocketId {
        #[cfg(unix)]
        {
            match self {
                Stream::Tcp(s) => s.as_raw_fd() as SocketId,
                Stream::Uds(s) => s.as_raw_fd() as SocketId,
            }
        }
        #[cfg(windows)]
        {
            match self {
                Stream::Tcp(s) => s.as_raw_socket() as SocketId,
            }
        }
    }

    /// `sock::SetNonBlocking`.
    fn set_nonblocking(&self, enable: bool) -> io::Result<()> {
        match self {
            Stream::Tcp(s) => s.set_nonblocking(enable),
            #[cfg(unix)]
            Stream::Uds(s) => s.set_nonblocking(enable),
        }
    }

    /// `sock::SetTcpNoDelay` (a no-op for `AF_UNIX`, which C++ also skips).
    fn set_nodelay(&self) -> io::Result<()> {
        match self {
            Stream::Tcp(s) => s.set_nodelay(true),
            #[cfg(unix)]
            Stream::Uds(_) => Ok(()),
        }
    }

    /// Readability probe backing the emulated `poll` (divergence 4).
    fn peek(&self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Stream::Tcp(s) => s.peek(buf),
            #[cfg(unix)]
            Stream::Uds(s) => s.peek(buf),
        }
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Stream::Tcp(s) => s.read(buf),
            #[cfg(unix)]
            Stream::Uds(s) => s.read(buf),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Stream::Tcp(s) => s.write(buf),
            #[cfg(unix)]
            Stream::Uds(s) => s.write(buf),
        }
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        match self {
            Stream::Tcp(s) => s.write_vectored(bufs),
            #[cfg(unix)]
            Stream::Uds(s) => s.write_vectored(bufs),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Stream::Tcp(s) => s.flush(),
            #[cfg(unix)]
            Stream::Uds(s) => s.flush(),
        }
    }
}

/// A listening socket.
#[derive(Debug)]
enum Listener {
    Tcp(TcpListener),
    #[cfg(unix)]
    Uds(UnixListener),
}

impl Listener {
    fn set_nonblocking(&self, enable: bool) -> io::Result<()> {
        match self {
            Listener::Tcp(l) => l.set_nonblocking(enable),
            #[cfg(unix)]
            Listener::Uds(l) => l.set_nonblocking(enable),
        }
    }

    /// `sock::Accept` — one pending connection, or `None` if none is pending.
    fn accept(&self) -> Option<Stream> {
        match self {
            Listener::Tcp(l) => l.accept().ok().map(|(s, _)| Stream::Tcp(s)),
            #[cfg(unix)]
            Listener::Uds(l) => l.accept().ok().map(|(s, _)| Stream::Uds(s)),
        }
    }

    /// Identity of this listening socket (`listen_fd_`).
    fn id(&self) -> SocketId {
        #[cfg(unix)]
        {
            match self {
                Listener::Tcp(l) => l.as_raw_fd() as SocketId,
                Listener::Uds(l) => l.as_raw_fd() as SocketId,
            }
        }
        #[cfg(windows)]
        {
            match self {
                Listener::Tcp(l) => l.as_raw_socket() as SocketId,
            }
        }
    }

    /// Actual bound TCP port (see divergence 16). 0 for `ipc`.
    fn port(&self) -> u16 {
        match self {
            Listener::Tcp(l) => l.local_addr().map(|a| a.port()).unwrap_or(0),
            #[cfg(unix)]
            Listener::Uds(_) => 0,
        }
    }
}

/// `sock::HostToNet32`.
#[inline]
pub const fn host_to_net32(host: u32) -> u32 {
    host.to_be()
}

/// `sock::NetToHost32`.
#[inline]
pub const fn net_to_host32(net: u32) -> u32 {
    u32::from_be(net)
}

/// `sock::UnlinkPath`.
fn unlink_path(path: &str) {
    // C++ ignores unlink()/DeleteFileA failure; so do we.
    let _ = std::fs::remove_file(path);
}

/// `sock::GetError()` for the most recent failure, or -1 when the OS reports
/// no code (C++ would return a stale/zero `errno` here).
fn last_os_error() -> i32 {
    io::Error::last_os_error()
        .raw_os_error()
        .filter(|c| *c != 0)
        .unwrap_or(-1)
}

/// `sock::PollRead(fd, timeout_ms)`: `>0` ready, `0` timeout, `-1` error.
///
/// Negative `timeout_ms` blocks indefinitely, as `poll(2)` does. See
/// divergence 4 for how readiness is observed without `poll`.
fn poll_read(s: &Stream, timeout_ms: i64) -> i32 {
    let deadline = if timeout_ms < 0 {
        None
    } else {
        Some(Instant::now() + Duration::from_millis(timeout_ms as u64))
    };
    let mut backoff = POLL_MIN_SLEEP;
    let mut probe = [0u8; 1];
    loop {
        match s.peek(&mut probe) {
            // Data pending, or EOF — poll(2) reports POLLIN for both.
            Ok(_) => return 1,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(_) => return -1,
        }
        if let Some(deadline) = deadline {
            let now = Instant::now();
            if now >= deadline {
                return 0;
            }
            std::thread::sleep(backoff.min(deadline - now));
        } else {
            std::thread::sleep(backoff);
        }
        backoff = (backoff * 2).min(POLL_MAX_SLEEP);
    }
}

/// `sock::RecvExact(fd, buf, len)`: 0 on success, [`EAGAIN`] when *nothing* has
/// been read yet and the socket would block, -1 on error/short read.
///
/// A zero-length request succeeds immediately, as the C++ `while (received <
/// len)` loop does.
fn recv_exact(s: &mut Stream, buf: &mut [u8]) -> i32 {
    let len = buf.len();
    let mut received = 0usize;
    while received < len {
        match s.read(&mut buf[received..]) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if received == 0 {
                    return EAGAIN;
                }
                // Mid-message stall: wait for the rest, then keep reading.
                if poll_read(s, STALL_TIMEOUT_MS) <= 0 {
                    return -1;
                }
            }
            Err(_) => return -1,
            // Peer closed mid-message.
            Ok(0) => return -1,
            Ok(n) => received += n,
        }
    }
    0
}

/// `sock::SendV(fd, iov, count)`: total bytes sent, or -1 on error.
///
/// Every slice is passed (no 64-entry cap) and a zero-byte write is an error
/// rather than a spin — see divergence 14. A write stalled for
/// [`STALL_TIMEOUT_MS`] fails, mirroring `poll(POLLOUT, 1000) <= 0 → -1`.
fn send_v(s: &mut Stream, bufs: &mut [IoSlice<'_>]) -> isize {
    let total: usize = bufs.iter().map(|b| b.len()).sum();
    let mut sent = 0usize;
    let mut rest = bufs;
    let mut stall_since: Option<Instant> = None;
    let mut backoff = POLL_MIN_SLEEP;
    while sent < total {
        match s.write_vectored(rest) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                let since = *stall_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= Duration::from_millis(STALL_TIMEOUT_MS as u64) {
                    return -1;
                }
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(POLL_MAX_SLEEP);
            }
            Err(_) => return -1,
            // Would spin forever in C++; treat as a failed write.
            Ok(0) => return -1,
            Ok(n) => {
                sent += n;
                IoSlice::advance_slices(&mut rest, n);
                stall_since = None;
                backoff = POLL_MIN_SLEEP;
            }
        }
    }
    sent as isize
}

/// Parse the C++ `inet_pton(AF_INET, ...)` address space: numeric IPv4 only.
fn parse_ipv4(addr: &str) -> Option<Ipv4Addr> {
    addr.parse::<Ipv4Addr>().ok()
}

/// `sock::Connect(addr, port, protocol)`.
fn connect(addr: &str, port: i32, protocol: &str) -> Result<Stream, SocketTransportError> {
    if protocol == "ipc" {
        #[cfg(unix)]
        {
            return UnixStream::connect(addr)
                .map(Stream::Uds)
                .map_err(|_| SocketTransportError::Connect {
                    addr: addr.to_string(),
                    port,
                });
        }
        #[cfg(not(unix))]
        {
            return Err(SocketTransportError::UnsupportedProtocol(
                protocol.to_string(),
            ));
        }
    }
    let ip = parse_ipv4(addr).ok_or_else(|| SocketTransportError::Connect {
        addr: addr.to_string(),
        port,
    })?;
    // C++ truncates the int port via static_cast<uint16_t> (divergence 13).
    let sa = SocketAddrV4::new(ip, port as u16);
    let s = TcpStream::connect(sa).map_err(|_| SocketTransportError::Connect {
        addr: addr.to_string(),
        port,
    })?;
    // sock::SetTcpNoDelay; SetSendBuf(4MB) has no std equivalent (divergence 9).
    let _ = s.set_nodelay(true);
    Ok(Stream::Tcp(s))
}

/// `sock::Listen(addr, port, protocol)`. TCP binds `INADDR_ANY`, ignoring
/// `addr`, exactly as the C++ does.
fn listen(addr: &str, port: i32, protocol: &str) -> Result<Listener, SocketTransportError> {
    if protocol == "ipc" {
        #[cfg(unix)]
        {
            unlink_path(addr);
            return UnixListener::bind(addr)
                .map(Listener::Uds)
                .map_err(|_| SocketTransportError::Listen {
                    addr: addr.to_string(),
                    port,
                });
        }
        #[cfg(not(unix))]
        {
            return Err(SocketTransportError::UnsupportedProtocol(
                protocol.to_string(),
            ));
        }
    }
    let sa = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port as u16);
    TcpListener::bind(sa)
        .map(Listener::Tcp)
        .map_err(|_| SocketTransportError::Listen {
            addr: addr.to_string(),
            port,
        })
}

/// `sock::IsServerAlive(addr, port, protocol)` — a connect probe.
pub fn is_server_alive(addr: &str, port: i32, protocol: &str) -> bool {
    if protocol == "ipc" {
        #[cfg(unix)]
        {
            return UnixStream::connect(addr).is_ok();
        }
        #[cfg(not(unix))]
        {
            return false;
        }
    }
    // Unparseable → false; the C++ would probe 0.0.0.0:port (divergence 12).
    let Some(ip) = parse_ipv4(addr) else {
        return false;
    };
    let sa = SocketAddr::V4(SocketAddrV4::new(ip, port as u16));
    TcpStream::connect_timeout(&sa, Duration::from_millis(ALIVE_PROBE_TIMEOUT_MS)).is_ok()
}

// ---------------------------------------------------------------------------
// Metadata codec (divergence 8)
// ---------------------------------------------------------------------------

/// Serialize `send, recv, send_bulks, recv_bulks` — the exact field shape of
/// the C++ `LbmMeta::serialize` / `Bulk::serialize`.
fn encode_meta(meta: &LbmMeta, out: &mut Vec<u8>) {
    fn put_bulks(out: &mut Vec<u8>, bulks: &[Bulk]) {
        out.extend_from_slice(&(bulks.len() as u64).to_le_bytes());
        for b in bulks {
            out.extend_from_slice(&(b.size as u64).to_le_bytes());
            out.extend_from_slice(&b.flags.bits().to_le_bytes());
        }
    }
    put_bulks(out, &meta.send);
    put_bulks(out, &meta.recv);
    out.extend_from_slice(&(meta.send_bulks as u64).to_le_bytes());
    out.extend_from_slice(&(meta.recv_bulks as u64).to_le_bytes());
}

/// Reader over a metadata buffer; every read is bounds-checked, so truncated
/// input becomes an error rather than a panic.
struct MetaCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> MetaCursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn u32(&mut self) -> Result<u32, ()> {
        let end = self.pos.checked_add(4).ok_or(())?;
        let bytes = self.buf.get(self.pos..end).ok_or(())?;
        self.pos = end;
        Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| ())?))
    }

    fn u64(&mut self) -> Result<u64, ()> {
        let end = self.pos.checked_add(8).ok_or(())?;
        let bytes = self.buf.get(self.pos..end).ok_or(())?;
        self.pos = end;
        Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| ())?))
    }

    fn bulks(&mut self) -> Result<Vec<Bulk>, ()> {
        let count = self.u64()?;
        // Bound the allocation by what the buffer could actually hold — a
        // hostile count must not make us reserve gigabytes.
        if count > (self.remaining() / BULK_WIRE_BYTES) as u64 {
            return Err(());
        }
        let mut out = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let size = self.u64()?;
            let flags = self.u32()?;
            // No buffer yet: recv_bulks allocates one if this is a XFER.
            out.push(Bulk::new(
                FullPtr::null(),
                usize::try_from(size).map_err(|_| ())?,
                flags,
            ));
        }
        Ok(out)
    }
}

/// Inverse of [`encode_meta`]. `Err(())` stands in for the C++
/// `catch (const std::exception&)` arm, which logs and returns -1.
fn decode_meta(buf: &[u8], meta: &mut LbmMeta) -> Result<(), ()> {
    let mut cur = MetaCursor::new(buf);
    let send = cur.bulks()?;
    let recv = cur.bulks()?;
    let send_bulks = cur.u64()?;
    let recv_bulks = cur.u64()?;
    meta.send = send;
    meta.recv = recv;
    meta.send_bulks = usize::try_from(send_bulks).map_err(|_| ())?;
    meta.recv_bulks = usize::try_from(recv_bulks).map_err(|_| ())?;
    Ok(())
}

/// Allocate `len` zeroed bytes, or `None` if the allocation would fail —
/// std's `vec![0; len]` aborts the process instead (divergence 8).
/// A zeroed `Vec` of `len` bytes, or `None` if the allocation fails.
///
/// Fallible on purpose: `len` comes off the wire, so a bogus length must fail
/// rather than abort the process. Used for the metadata buffer, which is
/// genuinely owned by its local variable — unlike a bulk payload, which is
/// pointed at rather than owned.
fn try_alloc_zeroed(len: usize) -> Option<Vec<u8>> {
    let mut v: Vec<u8> = Vec::new();
    v.try_reserve_exact(len).ok()?;
    v.resize(len, 0);
    Some(v)
}

// ---------------------------------------------------------------------------
// SocketTransport
// ---------------------------------------------------------------------------

/// An accepted server-side connection (a `client_fds_` entry).
#[derive(Debug)]
struct Conn {
    id: SocketId,
    stream: Stream,
}

/// `ctp::lbm::SocketTransport`.
///
/// # Example
///
/// ```no_run
/// use ctp_lightbeam::socket_transport::*;
///
/// let mut server =
///     SocketTransport::new(TransportMode::Server, "0.0.0.0", "tcp", 0).unwrap();
/// let port = server.bound_port() as i32;
/// let mut client =
///     SocketTransport::new(TransportMode::Client, "127.0.0.1", "tcp", port).unwrap();
///
/// // A Bulk points at memory rather than owning it, so the payload has to
/// // outlive the transfer — here the local `payload`, in the runtime the
/// // task archive that exposed it.
/// let payload = b"hello".to_vec();
/// let ptr = FullPtr::from_local(payload.as_ptr() as usize);
///
/// let mut out = LbmMeta::default();
/// out.send.push(client.expose(ptr, payload.len(), BULK_XFER));
/// assert_eq!(client.send(&out, &LbmContext::default()), 0);
///
/// let mut got = LbmMeta::default();
/// let info = server.recv(&mut got, &LbmContext::default());
/// if info.rc == 0 {
///     // recv allocated this buffer, so the server must release it.
///     assert_eq!(got.recv[0].size, 5);
///     server.clear_recv_handles(&mut got);
/// }
/// ```
#[derive(Debug)]
pub struct SocketTransport {
    /// `Transport::type_` — always [`TransportType::Socket`].
    type_: TransportType,
    /// `Transport::mode_`.
    mode_: TransportMode,
    addr: String,
    protocol: String,
    port: i32,
    /// Client mode: the connected socket (`fd_`).
    stream: Option<Stream>,
    /// Server mode: the listening socket (`listen_fd_`).
    listener: Option<Listener>,
    /// Server mode: accepted clients (`client_fds_`), in accept order.
    clients: Vec<Conn>,
    /// `em_ != nullptr`.
    em_registered: bool,
    /// Fds handed to the EventManager via `AddEvent` (divergence 2).
    event_regs: Vec<SocketId>,
    /// `fired_events_`, populated where `SocketFiredAction::Run` would push.
    fired_events: Vec<EventInfo>,
}

impl SocketTransport {
    /// `SocketTransport::SocketTransport(mode, addr, protocol, port)`.
    ///
    /// `protocol` is `"ipc"` (an `AF_UNIX` socket at `addr`) or anything else
    /// (TCP); the C++ defaults are `protocol = "tcp"` and `port = 8193`
    /// ([`DEFAULT_PORT`]). Client mode connects; server mode binds and
    /// listens. Both sockets are set non-blocking, as in C++.
    pub fn new(
        mode: TransportMode,
        addr: &str,
        protocol: &str,
        port: i32,
    ) -> Result<Self, SocketTransportError> {
        let mut t = Self {
            type_: TransportType::Socket,
            mode_: mode,
            addr: addr.to_string(),
            protocol: protocol.to_string(),
            port,
            stream: None,
            listener: None,
            clients: Vec::new(),
            em_registered: false,
            event_regs: Vec::new(),
            fired_events: Vec::new(),
        };
        // sock::InitSocketLib() has no port — std runs WSAStartup itself.
        match mode {
            TransportMode::Client => {
                let s = connect(addr, port, protocol)?;
                let _ = s.set_nonblocking(true);
                t.stream = Some(s);
            }
            TransportMode::Server => {
                let l = listen(addr, port, protocol)?;
                let _ = l.set_nonblocking(true);
                t.listener = Some(l);
            }
        }
        Ok(t)
    }

    /// `Transport::IsClient`.
    pub fn is_client(&self) -> bool {
        self.mode_ == TransportMode::Client
    }

    /// `Transport::IsServer`.
    pub fn is_server(&self) -> bool {
        self.mode_ == TransportMode::Server
    }

    /// `Transport::type_`.
    pub fn transport_type(&self) -> TransportType {
        self.type_
    }

    /// `GetAddress()`.
    pub fn get_address(&self) -> &str {
        &self.addr
    }

    /// The configured port (`port_`).
    pub fn port(&self) -> i32 {
        self.port
    }

    /// The **actually bound** TCP port — a Rust addition (divergence 16) that
    /// makes an ephemeral `port = 0` server usable. 0 for clients and `ipc`.
    pub fn bound_port(&self) -> u16 {
        self.listener.as_ref().map(|l| l.port()).unwrap_or(0)
    }

    /// `Expose(ptr, data_size, flags)`. Takes ownership of the buffer, since
    /// `Bulk::data` is an owned `Vec` here (divergence 7).
    /// `Expose(ptr, data_size, flags)` — describe caller-owned memory.
    ///
    /// The C++ body is exactly this: copy the pointer in, record size and
    /// flags. The memory belongs to the caller and must outlive the transfer.
    pub fn expose(&self, ptr: FullPtr, data_size: usize, flags: u32) -> Bulk {
        Bulk::new(ptr, data_size, flags)
    }

    /// `ClearRecvHandles(meta)` — release the buffers this transport allocated.
    ///
    /// C++ must consult the malloc sentinel to avoid a bad-free of a CTP
    /// allocator pointer; here the `owned` flag preserves the same shape and
    /// `Vec` handles the rest (divergence 7).
    pub fn clear_recv_handles(&self, meta: &mut LbmMeta) {
        for bulk in &mut meta.recv {
            // Only free buffers recv_bulks allocated itself, tagged with
            // RECV_ALLOCATED_ID. A CTP-allocator buffer swapped in by the task
            // archive must NOT be freed here — its user pointer sits inside a
            // larger region, so this would be a bad-free rather than a leak.
            if bulk.is_recv_allocated() {
                // SAFETY: is_recv_allocated proves the buffer came from
                // alloc_recv_buffer; `size` is unchanged since it was set.
                unsafe { free_recv_buffer(bulk.data.ptr, bulk.size) };
                // Null only the local half, as the C++ does: that alone makes
                // is_recv_allocated false, so a second call is a no-op.
                bulk.data.ptr = 0;
            }
        }
    }

    /// `IsServerAlive(ctx)` — connect probe against this transport's endpoint.
    pub fn is_server_alive(&self, _ctx: &LbmContext) -> bool {
        is_server_alive(&self.addr, self.port, &self.protocol)
    }

    // -- EventManager stand-in (divergence 2) -------------------------------

    /// `RegisterEventManager(em)`: register this transport's fds for read
    /// events. Client mode registers `fd_`; server mode registers `listen_fd_`
    /// (which only wakes the loop for new connections) plus every client fd.
    pub fn register_event_manager(&mut self) {
        self.em_registered = true;
        self.event_regs.clear();
        if self.is_client() {
            if let Some(s) = &self.stream {
                self.event_regs.push(s.id());
            }
        } else {
            if let Some(l) = &self.listener {
                // listen_fd_: no action — it just wakes the loop.
                self.event_regs.push(l.id());
            }
            for c in &self.clients {
                self.event_regs.push(c.id);
            }
        }
    }

    /// `UnregisterEventManager()`: detach *without* `RemoveEvent`, because the
    /// caller is about to destroy the EventManager (touching it would be a
    /// use-after-free in C++).
    pub fn unregister_event_manager(&mut self) {
        self.em_registered = false;
        self.event_regs.clear();
    }

    /// Fds currently registered for read events (test/diagnostic view).
    pub fn event_registrations(&self) -> &[SocketId] {
        &self.event_regs
    }

    /// `SocketFiredAction::Run(event)` — where an event loop reports that `fd`
    /// became readable.
    pub fn push_fired_event(&mut self, fd: SocketId) {
        self.fired_events.push(EventInfo {
            trigger: EventTrigger { fd, event_id: 0 },
            events: DEFAULT_READ_EVENT,
        });
    }

    /// Queued fired events (test/diagnostic view).
    pub fn fired_events(&self) -> &[EventInfo] {
        &self.fired_events
    }

    /// `RemoveFiredEvent(fd)` — drop every queued event for `fd`.
    fn remove_fired_event(&mut self, fd: SocketId) {
        self.fired_events.retain(|e| e.trigger.fd != fd);
    }

    /// `em_->RemoveEvent(fd)`: drop the registration and any queued events.
    fn remove_event(&mut self, fd: SocketId) {
        self.event_regs.retain(|id| *id != fd);
        self.remove_fired_event(fd);
    }

    // -- Send --------------------------------------------------------------

    /// `Send(meta, ctx)`: 0 on success, -1 or an OS error code on failure.
    ///
    /// Emits `[4-byte BE length][metadata][bulk0]...` in one vectored write.
    pub fn send(&mut self, meta: &LbmMeta, _ctx: &LbmContext) -> i32 {
        // 1. Determine which fd to send on.
        let idx = if self.is_client() {
            if self.stream.is_none() {
                return -1;
            }
            None
        } else {
            // Server mode: use client_info_.fd_ as set by Recv.
            let fd = meta.client_info.fd;
            if fd == INVALID_SOCKET {
                // HLOG(kError, "SocketTransport::Send(server) - no
                //       client_info_.fd_ set")
                return -1;
            }
            match self.clients.iter().position(|c| c.id == fd) {
                Some(i) => Some(i),
                // C++ would sendmsg() on a stale fd and fail with EBADF.
                None => return -1,
            }
        };

        // 2. Serialize the metadata straight into the buffer writev gets.
        let mut meta_buf: Vec<u8> = Vec::new();
        encode_meta(meta, &mut meta_buf);

        // 3. Build the iovec: [4-byte BE length prefix][metadata][XFER bulks]
        let meta_len = host_to_net32(meta_buf.len() as u32).to_ne_bytes();
        let mut iov: Vec<IoSlice<'_>> = Vec::with_capacity(2 + meta.send.len());
        iov.push(IoSlice::new(&meta_len));
        iov.push(IoSlice::new(&meta_buf));
        for b in &meta.send {
            if !b.flags.any(BULK_XFER) {
                continue;
            }
            // A zero-length bulk carries no bytes, so a null pointer is fine
            // and there is nothing to read.
            if b.size == 0 {
                continue;
            }
            // C++ reads `size` bytes straight from the pointer. Refuse a null
            // one rather than inherit that dereference — but note this is the
            // only check left: `size` is now the sender's word, exactly as in
            // C++, because a pointer has no length to cross-check it against.
            if b.data.ptr == 0 {
                return -1;
            }
            // SAFETY: the sender exposed this region and keeps it alive for
            // the call; `size` is the length it exposed.
            iov.push(IoSlice::new(unsafe { bulk_bytes(b) }));
        }

        // 4. Single vectored write.
        let stream = match idx {
            Some(i) => &mut self.clients[i].stream,
            None => self.stream.as_mut().expect("client stream checked above"),
        };
        if send_v(stream, &mut iov) < 0 {
            // HLOG(kError, "SocketTransport::Send - writev failed: {}")
            return last_os_error();
        }
        0
    }

    // -- Recv --------------------------------------------------------------

    /// `Recv(meta, ctx)`: accept, iterate fired events, receive metadata+bulks.
    ///
    /// `rc` is 0 on success, [`EAGAIN`] when nothing was available, else an
    /// error. On error in server mode the offending client is deregistered and
    /// closed.
    pub fn recv(&mut self, meta: &mut LbmMeta, ctx: &LbmContext) -> ClientInfo {
        let mut info = ClientInfo::default();
        if self.is_client() {
            let fd = match &self.stream {
                Some(s) => s.id(),
                // The C++ dispatch's failure value.
                None => return ClientInfo::failed(),
            };
            if self.em_registered {
                if let Some(pos) = self.fired_events.iter().position(|e| e.trigger.fd == fd) {
                    self.fired_events.remove(pos);
                }
            }
            let stream = self.stream.as_mut().expect("checked above");
            info.rc = Self::recv_all(stream, fd, meta, ctx);
            info.fd = fd;
            return info;
        }

        // Server mode: accept new clients (non-blocking).
        self.accept_new_client();

        if self.em_registered {
            // Iterate fired_events_ (only client fds — listen_fd_ has no
            // action, so it never lands here). As in the C++ loop, every path
            // either erases the current entry and continues, or returns — the
            // cursor never advances past a surviving entry.
            let i = 0usize;
            while i < self.fired_events.len() {
                let fd = self.fired_events[i].trigger.fd;
                let Some(ci) = self.clients.iter().position(|c| c.id == fd) else {
                    // No live client for this fd: nothing to read from.
                    self.fired_events.remove(i);
                    continue;
                };
                info.rc = Self::recv_all(&mut self.clients[ci].stream, fd, meta, ctx);
                info.fd = fd;
                if info.rc == EAGAIN {
                    self.fired_events.remove(i);
                    continue;
                }
                if info.rc != 0 {
                    // Drop the registration BEFORE closing, so the kernel
                    // cannot recycle the fd number and leave a stale entry in
                    // the registry (which would later trip MOD->ENOENT when a
                    // fresh accept() picked the same number).
                    self.remove_event(fd);
                    self.clients.retain(|c| c.id != fd); // closes via Drop
                    self.remove_fired_event(fd);
                    return info;
                }
                // Success: return with the event still queued — more data may
                // be pending on this fd (divergence 15).
                return info;
            }
            // Fall through to polling every client fd directly once
            // fired_events_ is exhausted: data may sit on sockets epoll has
            // not re-armed yet.
        }

        // Poll all client fds directly.
        let mut i = 0usize;
        while i < self.clients.len() {
            let fd = self.clients[i].id;
            info.rc = Self::recv_all(&mut self.clients[i].stream, fd, meta, ctx);
            info.fd = fd;
            if info.rc == EAGAIN {
                i += 1;
                continue;
            }
            if info.rc != 0 {
                if self.em_registered {
                    self.remove_event(fd);
                }
                self.clients.remove(i); // closes via Drop
                return info;
            }
            return info;
        }
        info.rc = EAGAIN;
        info.fd = INVALID_SOCKET;
        info
    }

    /// `RecvAll(fd, meta, ctx)` — metadata, then bulks, on one fd.
    fn recv_all(s: &mut Stream, fd: SocketId, meta: &mut LbmMeta, ctx: &LbmContext) -> i32 {
        let rc = Self::recv_metadata(s, meta, ctx);
        if rc != 0 {
            return rc;
        }
        meta.client_info.fd = fd;
        // Set up recv entries from the send descriptors.
        for i in 0..meta.send.len() {
            let recv_bulk = Bulk::new(
                FullPtr::null(),
                meta.send[i].size,
                meta.send[i].flags.bits(),
            );
            meta.recv.push(recv_bulk);
        }
        Self::recv_bulks(s, meta, ctx)
    }

    /// `RecvMetadata(fd, meta, ctx)` — the length prefix and metadata only.
    fn recv_metadata(s: &mut Stream, meta: &mut LbmMeta, _ctx: &LbmContext) -> i32 {
        let mut net_len = [0u8; 4];
        let rc = recv_exact(s, &mut net_len);
        if rc == EAGAIN {
            return EAGAIN;
        }
        if rc != 0 {
            return -1;
        }
        let meta_len = net_to_host32(u32::from_ne_bytes(net_len));
        // Recv straight into the deserialize buffer — no string intermediate.
        let Some(mut meta_buf) = try_alloc_zeroed(meta_len as usize) else {
            return -1;
        };
        if recv_exact(s, &mut meta_buf) != 0 {
            return -1;
        }
        if decode_meta(&meta_buf, meta).is_err() {
            // HLOG(kFatal, "Socket RecvMetadata: Deserialization failed - {}")
            return -1;
        }
        0
    }

    /// `RecvBulks(fd, meta, ctx)` — payload for each `BULK_XFER` recv entry.
    fn recv_bulks(s: &mut Stream, meta: &mut LbmMeta, _ctx: &LbmContext) -> i32 {
        for bulk in &mut meta.recv {
            if !bulk.flags.any(BULK_XFER) {
                continue;
            }
            let size = bulk.size;
            // Nothing to receive, so nothing to allocate. Rust's zero-length
            // allocations hand back a dangling-but-non-null pointer, which
            // would look like a real buffer to clear_recv_handles and cost a
            // pointless round trip through the allocator.
            if size == 0 {
                continue;
            }
            // C++: `buf == nullptr` → malloc, tagged with the sentinel so
            // ClearRecvHandles knows this buffer is ours to free.
            let allocated = bulk.data.ptr == 0;
            if allocated {
                let Some(addr) = alloc_recv_buffer(size) else {
                    return -1;
                };
                bulk.data = FullPtr::new(addr, ShmPtr::new(RECV_ALLOCATED_ID, addr as u64));
            }

            // SAFETY: either we just allocated `size` bytes at this address, or
            // the caller supplied a region it promised is `size` bytes. No
            // other reference to it exists while we fill it.
            let dst = unsafe { std::slice::from_raw_parts_mut(bulk.data.ptr as *mut u8, size) };
            let mut rc;
            loop {
                rc = recv_exact(s, dst);
                if rc != EAGAIN {
                    break;
                }
                if poll_read(s, STALL_TIMEOUT_MS) <= 0 {
                    rc = -1;
                    break;
                }
            }

            if rc != 0 {
                if allocated {
                    // SAFETY: allocated by us above, size unchanged.
                    unsafe { free_recv_buffer(bulk.data.ptr, size) };
                    bulk.data = FullPtr::null();
                }
                return last_os_error();
            }
        }
        0
    }

    /// `AcceptNewClient()` — one pending connection, non-blocking, no loop.
    fn accept_new_client(&mut self) {
        if self.is_client() {
            return;
        }
        let Some(listener) = &self.listener else {
            return;
        };
        let Some(stream) = listener.accept() else {
            return;
        };
        if self.protocol != "ipc" {
            let _ = stream.set_nodelay();
        }
        // sock::SetRecvBuf(fd, 4MB) has no std equivalent (divergence 9).
        let _ = stream.set_nonblocking(true);
        let id = stream.id();
        self.clients.push(Conn { id, stream });
        if self.em_registered {
            self.event_regs.push(id);
        }
    }
}

impl Drop for SocketTransport {
    /// `~SocketTransport()`. Sockets close themselves (RAII), so the C++
    /// `RemoveEvent`/`Close` calls have no counterpart — but the `ipc` path
    /// unlink does.
    fn drop(&mut self) {
        if self.is_server() && self.protocol == "ipc" {
            unlink_path(&self.addr);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ctp_memory::AllocatorId;

    const LOCALHOST: &str = "127.0.0.1";

    /// Spin `recv` until it yields something other than EAGAIN, or the budget
    /// (in **ms**) runs out. Sockets are non-blocking, so a fresh connection's
    /// bytes may not have landed yet.
    fn recv_until(t: &mut SocketTransport, meta: &mut LbmMeta, budget_ms: u64) -> ClientInfo {
        let deadline = Instant::now() + Duration::from_millis(budget_ms);
        loop {
            let info = t.recv(meta, &LbmContext::default());
            if info.rc != EAGAIN || Instant::now() >= deadline {
                return info;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// Drive `recv` until the server has accepted a client (AcceptNewClient
    /// takes one pending connection per call).
    ///
    /// Only safe with a *silent* client: `Recv` falls through to polling every
    /// client fd, so it would consume a pending frame into the throwaway meta.
    fn accept_one(server: &mut SocketTransport) {
        let deadline = Instant::now() + Duration::from_millis(5_000);
        while server.clients.is_empty() && Instant::now() < deadline {
            let _ = server.recv(&mut LbmMeta::default(), &LbmContext::default());
        }
        assert!(!server.clients.is_empty(), "the client should be accepted");
    }

    fn server_on_ephemeral_port() -> SocketTransport {
        SocketTransport::new(TransportMode::Server, "0.0.0.0", "tcp", 0)
            .expect("ephemeral bind must succeed")
    }

    fn client_to(port: u16) -> SocketTransport {
        SocketTransport::new(TransportMode::Client, LOCALHOST, "tcp", port as i32)
            .expect("connecting to a listening port must succeed")
    }

    /// A XFER bulk over `data`.
    ///
    /// `Bulk` points at memory rather than owning it, so the bytes must
    /// outlive the transfer — a real sender keeps them alive in the task
    /// archive. Tests leak instead, which costs a few bytes per case and
    /// keeps the lifetime out of the assertions.
    fn xfer(data: &[u8]) -> Bulk {
        let leaked: &'static mut [u8] = Box::leak(data.to_vec().into_boxed_slice());
        let addr = leaked.as_mut_ptr() as usize;
        Bulk::new(FullPtr::from_local(addr), data.len(), BULK_XFER)
    }

    /// An EXPOSE-only bulk of `size` bytes over no buffer.
    fn expose_only(size: usize) -> Bulk {
        Bulk::new(FullPtr::null(), size, BULK_EXPOSE)
    }

    /// The bytes a bulk points at, for assertions.
    fn bytes(b: &Bulk) -> &[u8] {
        if b.data.ptr == 0 {
            return &[];
        }
        // SAFETY: every bulk in these tests points at `size` live bytes —
        // either leaked by `xfer` or allocated by the transport's recv path.
        unsafe { bulk_bytes(b) }
    }

    /// True when the bulk has no buffer at all.
    fn is_empty(b: &Bulk) -> bool {
        b.data.ptr == 0
    }

    // -- byte order --------------------------------------------------------

    #[test]
    fn host_net_32_roundtrip() {
        for v in [0u32, 1, 0xFF, 0x0102_0304, u32::MAX] {
            assert_eq!(net_to_host32(host_to_net32(v)), v);
        }
        // Big-endian byte layout is what actually goes on the wire.
        assert_eq!(host_to_net32(0x0102_0304).to_ne_bytes(), [1u8, 2, 3, 4]);
    }

    // -- metadata codec ----------------------------------------------------

    #[test]
    fn meta_codec_roundtrip_with_mixed_flags() {
        let mut meta = LbmMeta {
            send: vec![
                xfer(&[1, 2, 3]),
                expose_only(4096),
            ],
            recv: vec![Bulk::new(FullPtr::null(), 7, BULK_XFER | BULK_EXPOSE)],
            send_bulks: 1,
            recv_bulks: 1,
            client_info: ClientInfo::default(),
        };
        let mut buf = Vec::new();
        encode_meta(&meta, &mut buf);

        let mut out = LbmMeta::default();
        decode_meta(&buf, &mut out).expect("roundtrip must decode");
        assert_eq!(out.send.len(), 2);
        assert_eq!(out.send[0].size, 3);
        assert!(out.send[0].flags.any(BULK_XFER));
        assert_eq!(out.send[1].size, 4096);
        assert!(out.send[1].flags.any(BULK_EXPOSE));
        assert!(!out.send[1].flags.any(BULK_XFER));
        assert_eq!(out.recv.len(), 1);
        assert!(out.recv[0].flags.all(BULK_XFER | BULK_EXPOSE));
        assert_eq!(out.send_bulks, 1);
        assert_eq!(out.recv_bulks, 1);
        // Only descriptors travel — never the payload.
        assert!(is_empty(&out.send[0]));

        // client_info_ is not serialized, so encoding stays byte-identical.
        let before = buf.clone();
        meta.client_info.fd = 42;
        buf.clear();
        encode_meta(&meta, &mut buf);
        assert_eq!(buf, before);
    }

    #[test]
    fn meta_codec_roundtrip_empty() {
        let meta = LbmMeta::default();
        let mut buf = Vec::new();
        encode_meta(&meta, &mut buf);
        // 2 vector counts + 2 tallies, all u64.
        assert_eq!(buf.len(), 32);
        let mut out = LbmMeta {
            send: vec![Bulk::default()],
            ..Default::default()
        };
        decode_meta(&buf, &mut out).expect("empty meta must decode");
        assert!(out.send.is_empty(), "decode replaces the vectors wholesale");
        assert!(out.recv.is_empty());
        assert_eq!(out.send_bulks, 0);
        assert_eq!(out.recv_bulks, 0);
    }

    #[test]
    fn meta_codec_rejects_truncated_input() {
        let meta = LbmMeta {
            send: vec![xfer(&[9])],
            ..Default::default()
        };
        let mut buf = Vec::new();
        encode_meta(&meta, &mut buf);
        for cut in 0..buf.len() {
            let mut out = LbmMeta::default();
            assert!(
                decode_meta(&buf[..cut], &mut out).is_err(),
                "truncation at {cut} must be an error, not a panic"
            );
        }
        let mut out = LbmMeta::default();
        assert!(decode_meta(&buf, &mut out).is_ok());
    }

    #[test]
    fn meta_codec_rejects_absurd_bulk_count() {
        // A hostile count must not make us reserve gigabytes.
        let mut buf = Vec::new();
        buf.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut out = LbmMeta::default();
        assert!(decode_meta(&buf, &mut out).is_err());

        // A count of exactly what the buffer holds is fine.
        let mut ok = Vec::new();
        ok.extend_from_slice(&1u64.to_le_bytes()); // send count
        ok.extend_from_slice(&5u64.to_le_bytes()); // size
        ok.extend_from_slice(&BULK_XFER.to_le_bytes()); // flags
        ok.extend_from_slice(&0u64.to_le_bytes()); // recv count
        ok.extend_from_slice(&0u64.to_le_bytes()); // send_bulks
        ok.extend_from_slice(&0u64.to_le_bytes()); // recv_bulks
        let mut out2 = LbmMeta::default();
        decode_meta(&ok, &mut out2).expect("an exact-fit count decodes");
        assert_eq!(out2.send[0].size, 5);
    }

    // -- construction ------------------------------------------------------

    #[test]
    fn server_binds_and_reports_its_ephemeral_port() {
        let s = server_on_ephemeral_port();
        assert!(s.is_server());
        assert!(!s.is_client());
        assert_eq!(s.transport_type(), TransportType::Socket);
        assert_eq!(s.get_address(), "0.0.0.0");
        assert_eq!(s.port(), 0, "the configured port is preserved verbatim");
        assert_ne!(s.bound_port(), 0, "bound_port resolves the ephemeral bind");
    }

    #[test]
    fn client_connect_to_dead_port_fails() {
        // Bind then drop, so the port is (almost certainly) free.
        let port = server_on_ephemeral_port().bound_port();
        let err = SocketTransport::new(TransportMode::Client, LOCALHOST, "tcp", port as i32)
            .expect_err("connecting to a dead port must fail");
        assert!(matches!(err, SocketTransportError::Connect { .. }));
        assert!(format!("{err}").contains("failed to connect"));
    }

    #[test]
    fn client_rejects_non_numeric_address() {
        // inet_pton parity: hostnames are not resolved (divergence 12).
        let err = SocketTransport::new(TransportMode::Client, "localhost", "tcp", 9)
            .expect_err("a hostname must not resolve");
        assert!(matches!(err, SocketTransportError::Connect { .. }));
    }

    #[test]
    fn client_has_no_bound_port() {
        let server = server_on_ephemeral_port();
        let client = client_to(server.bound_port());
        assert!(client.is_client());
        assert_eq!(client.bound_port(), 0);
    }

    // -- loopback round trips ---------------------------------------------

    #[test]
    fn loopback_roundtrip_single_xfer_bulk() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        let payload = b"hello lightbeam".to_vec();
        let out = LbmMeta {
            send: vec![xfer(&payload)],
            send_bulks: 1,
            ..Default::default()
        };
        assert_eq!(client.send(&out, &LbmContext::default()), 0);

        let mut got = LbmMeta::default();
        let info = recv_until(&mut server, &mut got, 5_000);
        assert_eq!(info.rc, 0, "the server must receive the frame");
        assert_ne!(info.fd, INVALID_SOCKET);
        assert_eq!(got.client_info.fd, info.fd, "the routing fd lands on meta");

        // recv entries are appended one per send descriptor.
        assert_eq!(got.recv.len(), 1);
        assert_eq!(got.recv[0].size, payload.len());
        assert_eq!(bytes(&got.recv[0]), &payload[..]);
        assert!(got.recv[0].is_recv_allocated(), "the transport allocated this buffer");

        // ClearRecvHandles releases only what the transport owns.
        server.clear_recv_handles(&mut got);
        assert!(is_empty(&got.recv[0]));
        assert!(!got.recv[0].is_recv_allocated());
    }

    #[test]
    fn loopback_roundtrip_mixed_expose_and_xfer() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        let a = vec![0xABu8; 64];
        let b = vec![0xCDu8; 3];
        let out = LbmMeta {
            send: vec![
                xfer(&a),
                // EXPOSE-only: the descriptor travels, the payload does not.
                expose_only(999),
                xfer(&b),
            ],
            send_bulks: 2,
            ..Default::default()
        };
        assert_eq!(client.send(&out, &LbmContext::default()), 0);

        let mut got = LbmMeta::default();
        assert_eq!(recv_until(&mut server, &mut got, 5_000).rc, 0);
        assert_eq!(got.recv.len(), 3);
        assert_eq!(bytes(&got.recv[0]), &a[..]);
        assert!(got.recv[0].is_recv_allocated());
        // The EXPOSE-only descriptor keeps its size but gets no buffer.
        assert_eq!(got.recv[1].size, 999);
        assert!(is_empty(&got.recv[1]));
        assert!(!got.recv[1].is_recv_allocated());
        assert_eq!(bytes(&got.recv[2]), &b[..]);
    }

    #[test]
    fn loopback_roundtrip_no_bulks_at_all() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        assert_eq!(client.send(&LbmMeta::default(), &LbmContext::default()), 0);

        let mut got = LbmMeta::default();
        assert_eq!(recv_until(&mut server, &mut got, 5_000).rc, 0);
        assert!(got.recv.is_empty());
        assert!(got.send.is_empty());
    }

    #[test]
    fn loopback_roundtrip_zero_sized_xfer_bulk() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        let out = LbmMeta {
            send: vec![Bulk::new(FullPtr::null(), 0, BULK_XFER)],
            send_bulks: 1,
            ..Default::default()
        };
        assert_eq!(client.send(&out, &LbmContext::default()), 0);

        let mut got = LbmMeta::default();
        assert_eq!(recv_until(&mut server, &mut got, 5_000).rc, 0);
        assert_eq!(got.recv.len(), 1);
        assert_eq!(got.recv[0].size, 0);
        assert!(is_empty(&got.recv[0]));
    }

    #[test]
    fn loopback_roundtrip_large_payload_spans_many_writes() {
        // 4 MiB comfortably exceeds any socket buffer, so this exercises the
        // partial-write advance in send_v and the stall loop in recv_exact.
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        let payload: Vec<u8> = (0..4 * 1024 * 1024u32).map(|i| (i % 251) as u8).collect();
        let expected = payload.clone();
        let out = LbmMeta {
            send: vec![xfer(&payload)],
            send_bulks: 1,
            ..Default::default()
        };

        // send blocks until the peer drains, so the reader must run
        // concurrently.
        let sender = std::thread::spawn(move || client.send(&out, &LbmContext::default()));

        let mut got = LbmMeta::default();
        let info = recv_until(&mut server, &mut got, 30_000);
        assert_eq!(sender.join().expect("sender thread"), 0);
        assert_eq!(info.rc, 0);
        assert_eq!(bytes(&got.recv[0]).len(), expected.len());
        assert_eq!(bytes(&got.recv[0]), &expected[..]);
    }

    #[test]
    fn loopback_roundtrip_many_bulks_exceeds_cpp_iov_cap() {
        // >64 bulks: the C++ caps its iovec array at 64 while summing `total`
        // over all of them, and spins forever. The Rust port sends them all
        // (divergence 14a).
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        let n = 100usize;
        let bulks: Vec<Bulk> = (0..n).map(|i| xfer(&[i as u8; 8])).collect();
        let out = LbmMeta {
            send: bulks,
            send_bulks: n,
            ..Default::default()
        };
        assert_eq!(client.send(&out, &LbmContext::default()), 0);

        let mut got = LbmMeta::default();
        assert_eq!(recv_until(&mut server, &mut got, 5_000).rc, 0);
        assert_eq!(got.recv.len(), n);
        for (i, b) in got.recv.iter().enumerate() {
            assert_eq!(bytes(b), &vec![i as u8; 8][..], "bulk {i} must survive intact");
        }
    }

    #[test]
    fn server_routes_a_reply_back_to_the_sending_client() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        let req = LbmMeta {
            send: vec![xfer(b"ping")],
            send_bulks: 1,
            ..Default::default()
        };
        assert_eq!(client.send(&req, &LbmContext::default()), 0);

        let mut got = LbmMeta::default();
        let info = recv_until(&mut server, &mut got, 5_000);
        assert_eq!(info.rc, 0);
        assert_eq!(bytes(&got.recv[0]), b"ping");

        // Reply on the fd Recv reported.
        let reply = LbmMeta {
            send: vec![xfer(b"pong")],
            send_bulks: 1,
            client_info: ClientInfo {
                rc: 0,
                fd: info.fd,
                ..ClientInfo::default()
            },
            ..Default::default()
        };
        assert_eq!(server.send(&reply, &LbmContext::default()), 0);

        let mut back = LbmMeta::default();
        assert_eq!(recv_until(&mut client, &mut back, 5_000).rc, 0);
        assert_eq!(bytes(&back.recv[0]), b"pong");
    }

    #[test]
    fn server_serves_multiple_clients() {
        let mut server = server_on_ephemeral_port();
        let port = server.bound_port();
        let mut c1 = client_to(port);
        let mut c2 = client_to(port);

        for (c, tag) in [(&mut c1, 1u8), (&mut c2, 2u8)] {
            let out = LbmMeta {
                send: vec![xfer(&[tag; 4])],
                send_bulks: 1,
                ..Default::default()
            };
            assert_eq!(c.send(&out, &LbmContext::default()), 0);
        }

        // AcceptNewClient takes one pending connection per Recv, so the two
        // frames arrive across successive calls.
        let mut seen: Vec<u8> = Vec::new();
        let deadline = Instant::now() + Duration::from_millis(5_000);
        while seen.len() < 2 && Instant::now() < deadline {
            let mut got = LbmMeta::default();
            let info = server.recv(&mut got, &LbmContext::default());
            if info.rc == 0 {
                seen.push(bytes(&got.recv[0])[0]);
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 2], "both clients must be served");
    }

    // -- error / EAGAIN paths ---------------------------------------------

    #[test]
    fn recv_is_eagain_when_idle() {
        let mut server = server_on_ephemeral_port();
        let mut meta = LbmMeta::default();

        // No clients at all.
        let info = server.recv(&mut meta, &LbmContext::default());
        assert_eq!(info.rc, EAGAIN);
        assert_eq!(info.fd, INVALID_SOCKET);

        // A connected but silent client is still EAGAIN.
        let _client = client_to(server.bound_port());
        let deadline = Instant::now() + Duration::from_millis(2_000);
        let mut accepted = false;
        while Instant::now() < deadline {
            let info = server.recv(&mut meta, &LbmContext::default());
            assert_eq!(info.rc, EAGAIN, "a silent client yields no data");
            if !server.clients.is_empty() {
                accepted = true;
                break;
            }
        }
        assert!(accepted, "the client should have been accepted");
    }

    #[test]
    fn server_send_without_client_info_fd_fails() {
        let mut server = server_on_ephemeral_port();
        // client_info.fd == INVALID_SOCKET
        assert_eq!(server.send(&LbmMeta::default(), &LbmContext::default()), -1);

        // An fd that is not a live client is equally unroutable.
        let stale = LbmMeta {
            client_info: ClientInfo {
                rc: 0,
                fd: 12345,
                ..ClientInfo::default()
            },
            ..Default::default()
        };
        assert_eq!(server.send(&stale, &LbmContext::default()), -1);
    }

    #[test]
    fn send_refuses_a_null_buffer_but_trusts_the_size() {
        // `size` is the sender's word. This port previously carried a Vec and
        // could reject `data.len() != size`; a Bulk now holds a FullPtr, which
        // has no length to cross-check against, so that net is gone — the same
        // contract C++ has always had. It is the price of letting a bulk point
        // into a shared segment instead of owning a copy.
        let server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        // The one thing still checkable: a non-zero size over a null pointer.
        let bad = LbmMeta {
            send: vec![Bulk::new(FullPtr::null(), 64, BULK_XFER)],
            send_bulks: 1,
            ..Default::default()
        };
        assert_eq!(client.send(&bad, &LbmContext::default()), -1);

        // A zero-sized bulk over a null pointer is legitimate, not an error.
        let empty = LbmMeta {
            send: vec![Bulk::new(FullPtr::null(), 0, BULK_XFER)],
            send_bulks: 1,
            ..Default::default()
        };
        assert_eq!(client.send(&empty, &LbmContext::default()), 0);
    }

    #[test]
    fn server_drops_a_client_that_disconnects_mid_frame() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        // Send a length prefix promising a body, then vanish.
        {
            let stream = client.stream.as_mut().unwrap();
            let prefix = host_to_net32(64).to_ne_bytes();
            let mut iov = [IoSlice::new(&prefix)];
            assert!(send_v(stream, &mut iov) > 0);
        }
        drop(client);

        // Faithful C++ fragility: RecvMetadata maps EAGAIN to EAGAIN only for
        // the 4-byte prefix. Once the prefix is in, a body that is not there
        // yet is -1, not a retry — so this connection dies either on the
        // absent body or on the EOF that follows.
        let mut meta = LbmMeta::default();
        let deadline = Instant::now() + Duration::from_millis(5_000);
        let mut rc = EAGAIN;
        while Instant::now() < deadline {
            let info = server.recv(&mut meta, &LbmContext::default());
            if info.rc != EAGAIN {
                rc = info.rc;
                break;
            }
        }
        assert_ne!(rc, EAGAIN, "a half-frame then EOF must surface an error");
        assert_ne!(rc, 0);
        assert!(server.clients.is_empty(), "the dead client must be closed");
    }

    #[test]
    fn recv_rejects_undecodable_metadata() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        // A well-framed but garbage metadata body.
        {
            let stream = client.stream.as_mut().unwrap();
            let body = [0xFFu8; 8];
            let prefix = host_to_net32(body.len() as u32).to_ne_bytes();
            let mut iov = [IoSlice::new(&prefix), IoSlice::new(&body)];
            assert!(send_v(stream, &mut iov) > 0);
        }

        let mut meta = LbmMeta::default();
        let info = recv_until(&mut server, &mut meta, 5_000);
        assert_eq!(info.rc, -1, "a decode failure maps to the C++ -1");
    }

    // -- liveness ----------------------------------------------------------

    #[test]
    fn is_server_alive_tracks_the_listener() {
        let server = server_on_ephemeral_port();
        let port = server.bound_port() as i32;
        assert!(is_server_alive(LOCALHOST, port, "tcp"));

        let probe = SocketTransport::new(TransportMode::Client, LOCALHOST, "tcp", port).unwrap();
        assert!(probe.is_server_alive(&LbmContext::default()));

        drop(probe);
        drop(server);
        assert!(!is_server_alive(LOCALHOST, port, "tcp"));
    }

    #[test]
    fn is_server_alive_rejects_an_unparseable_address() {
        // Divergence 12: the C++ would probe 0.0.0.0:port here.
        assert!(!is_server_alive("not-an-ip", DEFAULT_PORT, "tcp"));
    }

    // -- poll emulation ----------------------------------------------------

    #[test]
    fn poll_read_times_out_on_an_idle_socket() {
        let mut server = server_on_ephemeral_port();
        let client = client_to(server.bound_port());

        let start = Instant::now();
        let rc = poll_read(client.stream.as_ref().unwrap(), 50);
        let elapsed = start.elapsed();
        assert_eq!(rc, 0, "no data → timeout");
        assert!(elapsed >= Duration::from_millis(50), "must honor the budget");

        // A zero timeout returns immediately.
        let start = Instant::now();
        assert_eq!(poll_read(client.stream.as_ref().unwrap(), 0), 0);
        assert!(start.elapsed() < Duration::from_millis(50));

        accept_one(&mut server);
    }

    #[test]
    fn poll_read_reports_ready_when_data_lands() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());

        let out = LbmMeta {
            send: vec![xfer(b"x")],
            send_bulks: 1,
            ..Default::default()
        };
        assert_eq!(client.send(&out, &LbmContext::default()), 0);
        let mut got = LbmMeta::default();
        let info = recv_until(&mut server, &mut got, 5_000);
        assert_eq!(info.rc, 0);

        // Reply, so the client side has data pending.
        let reply = LbmMeta {
            send: vec![xfer(b"y")],
            send_bulks: 1,
            client_info: ClientInfo {
                rc: 0,
                fd: info.fd,
                ..ClientInfo::default()
            },
            ..Default::default()
        };
        assert_eq!(server.send(&reply, &LbmContext::default()), 0);

        // 1000 ms is ample for a loopback byte.
        assert_eq!(poll_read(client.stream.as_ref().unwrap(), 1_000), 1);
    }

    #[test]
    fn poll_read_reports_ready_at_eof() {
        // poll(2) reports POLLIN at EOF; the peek emulation must agree.
        let mut server = server_on_ephemeral_port();
        let client = client_to(server.bound_port());
        accept_one(&mut server);
        drop(server);

        assert_eq!(poll_read(client.stream.as_ref().unwrap(), 1_000), 1);
    }

    #[test]
    fn recv_exact_zero_length_is_success() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());
        let mut empty: [u8; 0] = [];
        // C++'s `while (received < len)` never runs for len == 0.
        assert_eq!(recv_exact(client.stream.as_mut().unwrap(), &mut empty), 0);
        accept_one(&mut server);
    }

    #[test]
    fn recv_exact_is_eagain_when_nothing_arrived() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());
        let mut buf = [0u8; 4];
        assert_eq!(recv_exact(client.stream.as_mut().unwrap(), &mut buf), EAGAIN);
        accept_one(&mut server);
    }

    // -- EventManager stand-in --------------------------------------------

    #[test]
    fn register_event_manager_records_the_client_fd() {
        let server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());
        assert!(client.event_registrations().is_empty());
        client.register_event_manager();
        let fd = client.stream.as_ref().unwrap().id();
        assert_eq!(client.event_registrations(), &[fd]);

        client.unregister_event_manager();
        assert!(client.event_registrations().is_empty());
        assert!(!client.em_registered);
    }

    #[test]
    fn register_event_manager_records_the_listener_then_accepted_clients() {
        let mut server = server_on_ephemeral_port();
        server.register_event_manager();
        assert_eq!(
            server.event_registrations().len(),
            1,
            "the listen fd is registered first"
        );

        let _client = client_to(server.bound_port());
        accept_one(&mut server);
        assert_eq!(
            server.event_registrations().len(),
            2,
            "AcceptNewClient registers each new client fd"
        );
    }

    #[test]
    fn a_fired_event_drives_recv() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());
        server.register_event_manager();

        // Accept while the client is still silent, so the frame below is not
        // swallowed by the accept loop's throwaway meta.
        accept_one(&mut server);
        let fd = server.clients[0].id;

        let out = LbmMeta {
            send: vec![xfer(b"data")],
            send_bulks: 1,
            ..Default::default()
        };
        assert_eq!(client.send(&out, &LbmContext::default()), 0);

        server.push_fired_event(fd);
        assert_eq!(server.fired_events().len(), 1);
        assert_eq!(server.fired_events()[0].events, DEFAULT_READ_EVENT);
        assert_eq!(server.fired_events()[0].trigger.fd, fd);

        let mut got = LbmMeta::default();
        let info = recv_until(&mut server, &mut got, 5_000);
        assert_eq!(info.rc, 0);
        assert_eq!(bytes(&got.recv[0]), b"data");
    }

    #[test]
    fn a_fired_event_for_an_unknown_fd_is_dropped() {
        let mut server = server_on_ephemeral_port();
        server.register_event_manager();
        server.push_fired_event(999_999);
        let mut meta = LbmMeta::default();
        let info = server.recv(&mut meta, &LbmContext::default());
        assert_eq!(info.rc, EAGAIN);
        assert!(
            server.fired_events().is_empty(),
            "an event with no live client must not be retried forever"
        );
    }

    #[test]
    fn eagain_erases_the_fired_event_but_success_keeps_it() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());
        server.register_event_manager();
        accept_one(&mut server);
        let fd = server.clients[0].id;

        // Nothing to read: the event is consumed (C++ erases on EAGAIN).
        server.push_fired_event(fd);
        let mut meta = LbmMeta::default();
        assert_eq!(server.recv(&mut meta, &LbmContext::default()).rc, EAGAIN);
        assert!(server.fired_events().is_empty());

        // Data available: Recv returns immediately and LEAVES the event queued
        // (divergence 15) — more data may still be pending on that fd.
        let out = LbmMeta {
            send: vec![xfer(b"z")],
            send_bulks: 1,
            ..Default::default()
        };
        assert_eq!(client.send(&out, &LbmContext::default()), 0);
        server.push_fired_event(fd);
        let mut got = LbmMeta::default();
        assert_eq!(recv_until(&mut server, &mut got, 5_000).rc, 0);
        assert_eq!(
            server.fired_events().len(),
            1,
            "a successful read must not erase the fired event"
        );
    }

    #[test]
    fn the_error_path_deregisters_before_closing() {
        let mut server = server_on_ephemeral_port();
        let mut client = client_to(server.bound_port());
        server.register_event_manager();

        // Accept while the client is silent, so the fd is known before the
        // half-frame below kills the connection.
        accept_one(&mut server);
        let fd = server.clients[0].id;
        assert!(server.event_registrations().contains(&fd));

        // Half a frame, then EOF.
        {
            let stream = client.stream.as_mut().unwrap();
            let prefix = host_to_net32(32).to_ne_bytes();
            let mut iov = [IoSlice::new(&prefix)];
            assert!(send_v(stream, &mut iov) > 0);
        }
        drop(client);

        let mut meta = LbmMeta::default();
        let mut rc = EAGAIN;
        let deadline = Instant::now() + Duration::from_millis(5_000);
        while Instant::now() < deadline {
            server.push_fired_event(fd);
            let info = server.recv(&mut meta, &LbmContext::default());
            if info.rc != EAGAIN {
                rc = info.rc;
                break;
            }
        }
        assert_ne!(rc, EAGAIN);
        assert_ne!(rc, 0);
        assert!(
            !server.event_registrations().contains(&fd),
            "the registration must be dropped before the fd is closed"
        );
        assert!(server.fired_events().is_empty());
        assert!(server.clients.is_empty());
    }

    // -- misc --------------------------------------------------------------

    #[test]
    fn expose_builds_the_descriptor() {
        let t = server_on_ephemeral_port();
        let src = xfer(&[1, 2, 3, 4]);
        let b = t.expose(src.data, 4, BULK_XFER | BULK_EXPOSE);
        assert_eq!(b.size, 4);
        assert!(b.flags.all(BULK_XFER | BULK_EXPOSE));
        assert!(
            !b.is_recv_allocated(),
            "Expose never allocates on the transport's behalf — it only              describes memory the caller already owns"
        );
        assert_eq!(bytes(&b), &[1, 2, 3, 4][..], "the pointer is passed through");
        assert_eq!(b.data, src.data, "Expose copies the FullPtr verbatim");

        // Bit positions must match the C++ BIT_OPT(u32, 0/1).
        assert_eq!(BULK_EXPOSE, 1);
        assert_eq!(BULK_XFER, 2);
    }

    #[test]
    fn lbm_context_flags_and_timeout_are_ms() {
        let c = LbmContext::default();
        assert!(!c.is_sync());
        assert!(!c.has_timeout());
        let c = LbmContext::new(LBM_SYNC, 250);
        assert!(c.is_sync());
        assert!(c.has_timeout(), "250 ms is a timeout");
        assert!(!LbmContext::new(0, 0).has_timeout());
        assert!(!LbmContext::new(0, -1).has_timeout());
    }

    #[test]
    fn clear_recv_handles_frees_only_what_the_transport_allocated() {
        let server = server_on_ephemeral_port();

        // 1. A buffer the recv path allocated: tagged with the sentinel, so
        //    clear_recv_handles owns it and must free it.
        let ours = alloc_recv_buffer(3).expect("alloc");
        // 2. A caller-provided buffer (private memory: null allocator id).
        let theirs = xfer(&[4, 5, 6]);
        // 3. The case the C++ comment is about: a CTP-allocator buffer swapped
        //    in by the task archive. Its id is neither null nor the sentinel,
        //    and freeing it here would be a bad-free, not a leak — the task
        //    reclaims it via TASK_DATA_OWNER.
        let ctp_addr = xfer(&[7, 8, 9]).data.ptr;
        let ctp_owned = Bulk::new(
            FullPtr::new(ctp_addr, ShmPtr::new(AllocatorId::new(1, 0), 16)),
            3,
            BULK_XFER,
        );

        let mut meta = LbmMeta {
            recv: vec![
                Bulk::new(
                    FullPtr::new(ours, ShmPtr::new(RECV_ALLOCATED_ID, ours as u64)),
                    3,
                    BULK_XFER,
                ),
                theirs,
                ctp_owned,
            ],
            ..Default::default()
        };
        assert!(meta.recv[0].is_recv_allocated());
        assert!(!meta.recv[1].is_recv_allocated());
        assert!(!meta.recv[2].is_recv_allocated());

        server.clear_recv_handles(&mut meta);

        // Ours is released and the entry is inert.
        assert!(is_empty(&meta.recv[0]));
        assert!(!meta.recv[0].is_recv_allocated());
        // Everyone else's is untouched.
        assert_eq!(bytes(&meta.recv[1]), &[4, 5, 6][..]);
        assert_eq!(bytes(&meta.recv[2]), &[7, 8, 9][..]);
        assert_eq!(meta.recv[2].data.shm.alloc_id, AllocatorId::new(1, 0));

        // Idempotent: a second call must not double-free.
        server.clear_recv_handles(&mut meta);
        assert!(is_empty(&meta.recv[0]));
        assert_eq!(bytes(&meta.recv[1]), &[4, 5, 6][..]);
    }

    #[test]
    fn ipc_protocol_availability_matches_the_platform() {
        let r = SocketTransport::new(TransportMode::Client, "ctp-rs-no-such-sock", "ipc", 0);
        #[cfg(unix)]
        assert!(
            matches!(r, Err(SocketTransportError::Connect { .. })),
            "unix: ipc is supported — this path simply does not exist"
        );
        #[cfg(not(unix))]
        assert!(
            matches!(r, Err(SocketTransportError::UnsupportedProtocol(_))),
            "windows: std has no AF_UNIX (divergence 11)"
        );
    }
}
