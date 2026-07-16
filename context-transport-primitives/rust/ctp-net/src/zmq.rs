// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Safe RAII bindings for **libzmq** (ZeroMQ), the C messaging library behind
//! CTP's lightbeam ZeroMQ transport (`clio_ctp/lightbeam/zmq_transport.h`).
//!
//! # Library, version, and ABI assumptions
//!
//! Wraps the **libzmq C API** (`zmq.h`, SONAME `libzmq.so.5`). The extern
//! declarations, option ids, and struct layouts below are hand-transcribed —
//! no bindgen, no `zmq-sys` — and were verified against **libzmq 4.3.5**
//! (`libzmq.so.5.2.5`) as shipped in the CTP devcontainer. Assumptions:
//!
//! * **Stable 4.x C ABI.** Everything used here (contexts, sockets, `zmq_msg_t`,
//!   `zmq_poll`) has been ABI-stable across libzmq 4.x; the option ids are
//!   frozen by the RFC. Only stable API is wrapped — no DRAFT sockets
//!   (`ZMQ_SERVER`/`ZMQ_RADIO`/…), no `zmq_poller`, no CURVE/GSSAPI.
//! * **`zmq_msg_t` is 64 opaque bytes** aligned to `sizeof(void*)`. This is the
//!   one layout that would silently corrupt memory if it ever changed, so it is
//!   not taken on faith: [`Context::msg_t_size`] queries the *live* library
//!   (`ZMQ_MSG_T_SIZE`) and [`abi_check`] compares it against
//!   `size_of::<ZmqMsgT>()`. The test suite asserts it.
//! * **`zmq_poll`'s timeout is C `long`** (64-bit on LP64 Unix, 32-bit on
//!   Windows) — hence `c_long`, not `i32`.
//! * **`EAGAIN`/`EINTR` are platform errno values, not libzmq constants.**
//!   `zmq.h` defines only the `ZMQ_HAUSNUMERO`-based codes (`ETERM`, `EFSM`, …)
//!   and falls back to `<errno.h>` for the POSIX ones, so [`errno::EAGAIN`] is
//!   selected per target (11 on Linux/Windows, 35 on the BSDs/macOS).
//! * **Linking** is via `#[link(name = "zmq")]` — `ctp-net` has no `build.rs`,
//!   so the attribute is what emits `-lzmq`. It cannot emit a link *search*
//!   path, though: where the CTP devcontainer installs libzmq
//!   (`/usr/local/lib`) is on the runtime loader's path (`ld.so.conf`) but not
//!   on rust-lld's, so linking needs `-L /usr/local/lib` from outside this file
//!   — e.g. `RUSTFLAGS='-L /usr/local/lib'`, or, better, a `build.rs` emitting
//!   `cargo:rustc-link-search=native=…` from `pkg-config --libs libzmq`, the
//!   way `ctp-gpu/build.rs` does for CUDA.
//!
//! # Safety model
//!
//! * [`Socket`] borrows its [`Context`] (`Socket<'ctx>`). The borrow checker
//!   therefore *proves* every socket is closed before `zmq_ctx_term` runs —
//!   which is exactly the ordering libzmq requires, and skipping it is what
//!   hangs `zmq_ctx_term` forever.
//! * [`Context`] is `Send + Sync` (libzmq contexts are documented thread-safe).
//!   [`Socket`] is `Send` but **not** `Sync`: a socket may be *migrated* between
//!   threads but never used by two at once. This is the same constraint the C++
//!   `ZeroMqTransport` enforces at runtime with `sock_mtx_`; here it is a
//!   compile-time property, so no mutex is needed to make misuse impossible.
//! * No raw pointers or `zmq_msg_t` handles escape into the safe surface.
//!
//! # Divergence from the C++ `ZeroMqTransport`
//!
//! 1. **The context is really terminated; the C++ one is deliberately leaked.**
//!    `GetSharedContext`'s `CtxOwner` only calls `zmq_ctx_shutdown` at exit and
//!    lets the OS reclaim the context, because CLIO sockets can outlive it and
//!    `zmq_ctx_term` would then deadlock (and on Windows abort inside libzmq's
//!    signaler). That leak buys a lifetime guarantee Rust gets for free, so
//!    [`Context::drop`] does the full `zmq_ctx_shutdown` + `zmq_ctx_term`
//!    (retrying on `EINTR`, as the C API requires).
//! 2. **`ZMQ_BLOCKY = 0` is set in [`Context::new`]**, mirroring the C++ shared
//!    context. New sockets then default to `LINGER = 0`, so neither `zmq_close`
//!    nor the `zmq_ctx_term` in `Drop` can block on undelivered messages.
//!    Override per socket with [`Socket::set_linger`].
//! 3. **No env sniffing.** C++ reads `CLIO_ZMQ_IO_THREADS` and defaults to 8 I/O
//!    threads; a library wrapper should not read the environment, so I/O threads
//!    stay at libzmq's default unless set via [`Context::with_io_threads`].
//! 4. **Errors are values.** C++ ignores most `zmq_setsockopt`/`zmq_close`
//!    return codes and returns bare `zmq_errno()` ints from `Send`/`Recv`; every
//!    call here returns [`ZmqError`], which carries the errno *and* the
//!    `zmq_strerror` text. Nothing panics on a library error.
//! 5. **`EINTR` is retried internally** by [`Socket::send`], [`Socket::recv_into`],
//!    [`Socket::send_msg`], and [`Socket::recv_msg`] — the C++ `zmq_send_eintr`
//!    /`zmq_msg_send_eintr`/`zmq_msg_recv_eintr` helpers, hoisted into the
//!    wrapper so no caller can forget them. (That bug is documented in the C++:
//!    a stray `SIGCHLD` poisons an in-flight multipart message and wedges the
//!    socket.)
//! 6. **Send is copy-based only.** `zmq_msg_init_data` (zero-copy send with a
//!    free callback) is not wrapped, matching the C++ after its async
//!    send-completion machinery was removed. Zero-copy *recv* is available via
//!    [`Socket::recv_msg`], which is how `RecvBulks` avoids a memcpy.

use std::ffi::{c_char, c_int, c_long, c_short, c_void, CStr, CString};
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Raw FFI: hand-rolled from zmq.h (libzmq 4.3.5). See module docs.
// ---------------------------------------------------------------------------

/// `zmq_fd_t`: `SOCKET` (pointer-sized) on Windows, `int` elsewhere.
#[cfg(windows)]
pub type SocketFd = usize;
/// `zmq_fd_t`: `SOCKET` (pointer-sized) on Windows, `int` elsewhere.
#[cfg(not(windows))]
pub type SocketFd = c_int;

/// `zmq_msg_t`: 64 opaque bytes, aligned to `sizeof(void*)`.
///
/// Over-aligned to 8 on 32-bit targets (where C asks for 4); harmless, since we
/// only ever hand libzmq a pointer to one of these. The size is checked against
/// the live library by [`abi_check`].
#[repr(C, align(8))]
struct ZmqMsgT {
    _opaque: [u8; 64],
}

/// `zmq_pollitem_t` — verified layout: `{ void*@0, zmq_fd_t@8, short@12, short@14 }`, size 16.
#[repr(C)]
struct ZmqPollItemT {
    socket: *mut c_void,
    fd: SocketFd,
    events: c_short,
    revents: c_short,
}

#[link(name = "zmq")]
extern "C" {
    fn zmq_errno() -> c_int;
    fn zmq_strerror(errnum: c_int) -> *const c_char;
    fn zmq_version(major: *mut c_int, minor: *mut c_int, patch: *mut c_int);

    fn zmq_ctx_new() -> *mut c_void;
    fn zmq_ctx_term(context: *mut c_void) -> c_int;
    fn zmq_ctx_shutdown(context: *mut c_void) -> c_int;
    fn zmq_ctx_set(context: *mut c_void, option: c_int, optval: c_int) -> c_int;
    fn zmq_ctx_get(context: *mut c_void, option: c_int) -> c_int;

    fn zmq_socket(context: *mut c_void, type_: c_int) -> *mut c_void;
    fn zmq_close(s: *mut c_void) -> c_int;
    fn zmq_setsockopt(
        s: *mut c_void,
        option: c_int,
        optval: *const c_void,
        optvallen: usize,
    ) -> c_int;
    fn zmq_getsockopt(
        s: *mut c_void,
        option: c_int,
        optval: *mut c_void,
        optvallen: *mut usize,
    ) -> c_int;
    fn zmq_bind(s: *mut c_void, addr: *const c_char) -> c_int;
    fn zmq_connect(s: *mut c_void, addr: *const c_char) -> c_int;
    fn zmq_send(s: *mut c_void, buf: *const c_void, len: usize, flags: c_int) -> c_int;
    fn zmq_recv(s: *mut c_void, buf: *mut c_void, len: usize, flags: c_int) -> c_int;

    fn zmq_msg_init(msg: *mut ZmqMsgT) -> c_int;
    fn zmq_msg_init_size(msg: *mut ZmqMsgT, size: usize) -> c_int;
    fn zmq_msg_close(msg: *mut ZmqMsgT) -> c_int;
    fn zmq_msg_data(msg: *mut ZmqMsgT) -> *mut c_void;
    fn zmq_msg_size(msg: *const ZmqMsgT) -> usize;
    fn zmq_msg_more(msg: *const ZmqMsgT) -> c_int;
    fn zmq_msg_send(msg: *mut ZmqMsgT, s: *mut c_void, flags: c_int) -> c_int;
    fn zmq_msg_recv(msg: *mut ZmqMsgT, s: *mut c_void, flags: c_int) -> c_int;

    fn zmq_poll(items: *mut ZmqPollItemT, nitems: c_int, timeout: c_long) -> c_int;
}

// ---------------------------------------------------------------------------
// Errno
// ---------------------------------------------------------------------------

/// Error numbers reported by [`ZmqError::code`].
///
/// The `ZMQ_HAUSNUMERO`-based codes are libzmq's own and identical everywhere.
/// `EAGAIN`/`EINTR` are *platform* values: `zmq.h` only defines the POSIX names
/// when the platform doesn't, so they must be selected per target.
pub mod errno {
    /// libzmq's private errno base (`zmq.h`).
    pub const HAUSNUMERO: i32 = 156_384_712;
    /// Operation cannot be performed in this state (e.g. REQ/REP out of turn).
    pub const EFSM: i32 = HAUSNUMERO + 51;
    /// The peer's protocol is not compatible.
    pub const ENOCOMPATPROTO: i32 = HAUSNUMERO + 52;
    /// The context was terminated (`zmq_ctx_shutdown`/`zmq_ctx_term`).
    pub const ETERM: i32 = HAUSNUMERO + 53;
    /// No I/O thread available.
    pub const EMTHREAD: i32 = HAUSNUMERO + 54;

    /// Interrupted by a signal. 4 on Linux, the BSDs/macOS, and the MSVC CRT.
    pub const EINTR: i32 = 4;

    /// Would block (non-blocking call, or `ZMQ_SNDTIMEO`/`ZMQ_RCVTIMEO` expiry).
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly"
    ))]
    pub const EAGAIN: i32 = 35;
    /// Would block (non-blocking call, or `ZMQ_SNDTIMEO`/`ZMQ_RCVTIMEO` expiry).
    #[cfg(not(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly"
    )))]
    pub const EAGAIN: i32 = 11;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A libzmq failure: the `zmq_errno()` value plus the call that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZmqError {
    code: i32,
    what: String,
}

impl ZmqError {
    /// Capture `zmq_errno()` for a failed call.
    fn last(what: impl Into<String>) -> Self {
        // SAFETY: zmq_errno reads this thread's errno; no preconditions.
        Self {
            code: unsafe { zmq_errno() },
            what: what.into(),
        }
    }

    /// A wrapper-level error with no libzmq errno behind it.
    fn wrapper(what: impl Into<String>) -> Self {
        Self {
            code: 0,
            what: what.into(),
        }
    }

    /// The raw errno. Compare against [`errno`] constants.
    pub fn code(&self) -> i32 {
        self.code
    }

    /// `zmq_strerror` text for [`Self::code`] (empty for wrapper-level errors).
    pub fn message(&self) -> String {
        if self.code == 0 {
            return String::new();
        }
        // SAFETY: zmq_strerror accepts any int and returns a static
        // NUL-terminated string ("Undefined error" for unknown codes).
        let p = unsafe { zmq_strerror(self.code) };
        if p.is_null() {
            return String::new();
        }
        // SAFETY: non-null pointer to a static NUL-terminated string.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// `EAGAIN` — the call would block, or a send/recv timeout expired. The C++
    /// `SendOut` path re-queues on this.
    pub fn is_again(&self) -> bool {
        self.code == errno::EAGAIN
    }

    /// `EINTR` — interrupted by a signal. Retried internally by this wrapper's
    /// send/recv; still observable from [`Socket::poll`].
    pub fn is_intr(&self) -> bool {
        self.code == errno::EINTR
    }

    /// `ETERM` — the context was shut down; the socket is finished.
    pub fn is_term(&self) -> bool {
        self.code == errno::ETERM
    }
}

impl std::fmt::Display for ZmqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.code == 0 {
            write!(f, "ZmqError: {}", self.what)
        } else {
            write!(
                f,
                "ZmqError: {}: {} (errno {})",
                self.what,
                self.message(),
                self.code
            )
        }
    }
}

impl std::error::Error for ZmqError {}

/// Result of a libzmq call.
pub type Result<T> = std::result::Result<T, ZmqError>;

/// Endpoints must be NUL-free to cross the C boundary.
fn to_cstring(s: &str, what: &str) -> Result<CString> {
    CString::new(s).map_err(|_| ZmqError::wrapper(format!("{what}: interior NUL in '{s}'")))
}

// ---------------------------------------------------------------------------
// Constants: socket types and options (verified against libzmq 4.3.5 zmq.h)
// ---------------------------------------------------------------------------

/// ZMQ socket types (`ZMQ_PAIR`…`ZMQ_STREAM`). Values are wire-frozen by the
/// ZMTP RFC and verified against `zmq.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SocketType {
    Pair = 0,
    Pub = 1,
    Sub = 2,
    Req = 3,
    Rep = 4,
    /// Client half of CTP's fixed DEALER ↔ ROUTER topology.
    Dealer = 5,
    /// Server half of CTP's fixed DEALER ↔ ROUTER topology.
    Router = 6,
    Pull = 7,
    Push = 8,
    XPub = 9,
    XSub = 10,
    Stream = 11,
}

impl SocketType {
    fn from_raw(v: c_int) -> Option<Self> {
        Some(match v {
            0 => Self::Pair,
            1 => Self::Pub,
            2 => Self::Sub,
            3 => Self::Req,
            4 => Self::Rep,
            5 => Self::Dealer,
            6 => Self::Router,
            7 => Self::Pull,
            8 => Self::Push,
            9 => Self::XPub,
            10 => Self::XSub,
            11 => Self::Stream,
            _ => return None,
        })
    }
}

/// Socket options used by CTP's transport (`zmq_setsockopt`/`zmq_getsockopt`).
///
/// `ZMQ_IDENTITY` is libzmq's own alias for [`SockOpt::RoutingId`] (both 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SockOpt {
    /// `ZMQ_ROUTING_ID` / `ZMQ_IDENTITY` — bytes.
    RoutingId = 5,
    /// `ZMQ_SUBSCRIBE` — bytes.
    Subscribe = 6,
    SndBuf = 11,
    RcvBuf = 12,
    /// `ZMQ_RCVMORE` — more frames in the current multipart message.
    RcvMore = 13,
    /// `ZMQ_FD` — pointer-sized on Windows; use [`Socket::fd`].
    Fd = 14,
    Events = 15,
    /// `ZMQ_TYPE` — read-only.
    Type = 16,
    Linger = 17,
    SndHwm = 23,
    RcvHwm = 24,
    RcvTimeo = 27,
    SndTimeo = 28,
    /// `ZMQ_LAST_ENDPOINT` — string; use [`Socket::last_endpoint`].
    LastEndpoint = 32,
    RouterMandatory = 33,
    Immediate = 39,
    RouterHandover = 56,
    HeartbeatIvl = 75,
    HeartbeatTtl = 76,
    HeartbeatTimeout = 77,
}

/// Context options (`zmq_ctx_set`/`zmq_ctx_get`).
///
/// `ZMQ_THREAD_PRIORITY` is omitted: it shares id 3 with `ZMQ_SOCKET_LIMIT` in
/// `zmq.h` (set-only vs get-only), which a Rust enum cannot express.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum CtxOpt {
    IoThreads = 1,
    MaxSockets = 2,
    /// `ZMQ_SOCKET_LIMIT` — read-only.
    SocketLimit = 3,
    MaxMsgSz = 5,
    /// `ZMQ_MSG_T_SIZE` — read-only `sizeof(zmq_msg_t)`; see [`abi_check`].
    MsgTSize = 6,
    /// `ZMQ_BLOCKY` — when 0, new sockets get `LINGER = 0`.
    Blocky = 70,
}

/// Flags for [`Socket::send`] / [`Socket::recv_into`] / the `_msg` forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Flags(c_int);

impl Flags {
    /// Blocking (subject to `ZMQ_SNDTIMEO`/`ZMQ_RCVTIMEO`).
    pub const NONE: Flags = Flags(0);
    /// `ZMQ_DONTWAIT` — fail with `EAGAIN` rather than block.
    pub const DONTWAIT: Flags = Flags(1);
    /// `ZMQ_SNDMORE` — another frame of this multipart message follows.
    pub const SNDMORE: Flags = Flags(2);

    /// Raw bit value passed to libzmq.
    pub fn bits(self) -> i32 {
        self.0
    }

    /// True if every bit of `other` is set.
    pub fn contains(self, other: Flags) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for Flags {
    type Output = Flags;
    fn bitor(self, rhs: Flags) -> Flags {
        Flags(self.0 | rhs.0)
    }
}

/// Poll readiness bits for [`Socket::poll`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PollEvents(c_short);

impl PollEvents {
    /// `ZMQ_POLLIN`.
    pub const IN: PollEvents = PollEvents(1);
    /// `ZMQ_POLLOUT`.
    pub const OUT: PollEvents = PollEvents(2);
    /// `ZMQ_POLLERR`.
    pub const ERR: PollEvents = PollEvents(4);

    /// Raw bit value.
    pub fn bits(self) -> i16 {
        self.0
    }

    /// True if every bit of `other` is set.
    pub fn contains(self, other: PollEvents) -> bool {
        self.0 & other.0 == other.0
    }

    /// True if no bits are set (a poll timeout).
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for PollEvents {
    type Output = PollEvents;
    fn bitor(self, rhs: PollEvents) -> PollEvents {
        PollEvents(self.0 | rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Library-level helpers
// ---------------------------------------------------------------------------

/// Runtime libzmq version as `(major, minor, patch)` — e.g. `(4, 3, 5)`.
pub fn version() -> (i32, i32, i32) {
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    // SAFETY: three valid out-pointers; zmq_version cannot fail.
    unsafe { zmq_version(&mut major, &mut minor, &mut patch) };
    (major, minor, patch)
}

/// Verify this module's hand-rolled `zmq_msg_t` against the loaded library.
///
/// `size_of::<ZmqMsgT>()` is the single layout assumption that could corrupt
/// memory if libzmq ever changed it, and the library will tell us its own
/// `sizeof(zmq_msg_t)` via `ZMQ_MSG_T_SIZE`. Cheap insurance against linking a
/// libzmq whose ABI doesn't match these decls.
pub fn abi_check() -> Result<()> {
    let ctx = Context::new()?;
    let reported = ctx.msg_t_size()?;
    let ours = std::mem::size_of::<ZmqMsgT>();
    if reported != ours {
        return Err(ZmqError::wrapper(format!(
            "zmq_msg_t ABI mismatch: libzmq {:?} reports {reported} bytes, these \
             bindings assume {ours}",
            version()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// A ZMQ context: the socket factory and owner of the I/O thread pool.
///
/// Dropping terminates the context (`zmq_ctx_shutdown` + `zmq_ctx_term`). That
/// can only be reached once every [`Socket`] borrowed from it is gone, because
/// `Socket<'ctx>` holds a `&'ctx Context` — so the "close sockets before the
/// context or `zmq_ctx_term` hangs" rule is enforced by the borrow checker
/// rather than by discipline.
pub struct Context {
    ctx: *mut c_void,
}

// SAFETY: libzmq contexts are explicitly documented as thread-safe — creating
// sockets from, and terminating, one context from multiple threads is
// supported. (Sockets are not; see `Socket`.)
unsafe impl Send for Context {}
// SAFETY: as above — `&Context` may be shared across threads.
unsafe impl Sync for Context {}

impl Context {
    /// Create a context with `ZMQ_BLOCKY = 0`.
    ///
    /// `ZMQ_BLOCKY = 0` gives every new socket `LINGER = 0`, so no close and no
    /// `Drop` can block on undelivered messages. This mirrors the C++
    /// `GetSharedContext`, which sets the same option for the same reason.
    pub fn new() -> Result<Self> {
        // SAFETY: zmq_ctx_new takes no arguments; returns NULL on failure.
        let ctx = unsafe { zmq_ctx_new() };
        if ctx.is_null() {
            return Err(ZmqError::last("zmq_ctx_new"));
        }
        let this = Self { ctx };
        // On failure `this` drops here and terminates the fresh context.
        this.set(CtxOpt::Blocky, 0)?;
        Ok(this)
    }

    /// Create a context with `n` I/O threads (`ZMQ_IO_THREADS`).
    ///
    /// The C++ transport defaults this to 8 — 2 saturates at 64+ nodes during
    /// SWIM probe rounds. That policy belongs to the caller, not the wrapper.
    pub fn with_io_threads(n: i32) -> Result<Self> {
        let ctx = Self::new()?;
        ctx.set(CtxOpt::IoThreads, n)?;
        Ok(ctx)
    }

    /// `zmq_ctx_set`. Options affecting the I/O pool must be set before any
    /// socket is created.
    pub fn set(&self, option: CtxOpt, value: i32) -> Result<()> {
        // SAFETY: valid context; libzmq validates the option id and value and
        // reports EINVAL rather than misbehaving.
        let rc = unsafe { zmq_ctx_set(self.ctx, option as c_int, value) };
        if rc != 0 {
            return Err(ZmqError::last(format!("zmq_ctx_set({option:?})")));
        }
        Ok(())
    }

    /// `zmq_ctx_get`.
    pub fn get(&self, option: CtxOpt) -> Result<i32> {
        // SAFETY: valid context; unknown option ids yield -1/EINVAL.
        let rc = unsafe { zmq_ctx_get(self.ctx, option as c_int) };
        if rc < 0 {
            return Err(ZmqError::last(format!("zmq_ctx_get({option:?})")));
        }
        Ok(rc)
    }

    /// `sizeof(zmq_msg_t)` as reported by the loaded library.
    pub fn msg_t_size(&self) -> Result<usize> {
        Ok(self.get(CtxOpt::MsgTSize)? as usize)
    }

    /// Open a socket on this context. The returned socket borrows the context.
    pub fn socket(&self, socket_type: SocketType) -> Result<Socket<'_>> {
        // SAFETY: valid context and a verified socket-type constant; returns
        // NULL on failure.
        let sock = unsafe { zmq_socket(self.ctx, socket_type as c_int) };
        if sock.is_null() {
            return Err(ZmqError::last(format!("zmq_socket({socket_type:?})")));
        }
        Ok(Socket {
            sock,
            _ctx: PhantomData,
        })
    }

    /// `zmq_ctx_shutdown`: make blocking calls on this context's sockets fail
    /// with `ETERM` so worker threads can unwind. Does not wait for sockets to
    /// close; `Drop` still terminates the context.
    pub fn shutdown(&self) -> Result<()> {
        // SAFETY: valid context; shutdown is idempotent and safe to race with
        // socket operations (that is its entire purpose).
        if unsafe { zmq_ctx_shutdown(self.ctx) } != 0 {
            return Err(ZmqError::last("zmq_ctx_shutdown"));
        }
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: `ctx` came from zmq_ctx_new and is terminated exactly once.
        // Every Socket borrowed from it is already dropped (and thus closed) —
        // the `Socket<'ctx>` borrow makes that a compile-time fact — so
        // zmq_ctx_term cannot block waiting on an open socket. Sockets default
        // to LINGER=0 via ZMQ_BLOCKY=0, so it cannot block on unsent data
        // either. zmq_ctx_term is documented to fail with EINTR if a signal
        // arrives and must then be restarted; any other error is unrecoverable
        // in a destructor and is dropped.
        unsafe {
            zmq_ctx_shutdown(self.ctx);
            loop {
                if zmq_ctx_term(self.ctx) == 0 || zmq_errno() != errno::EINTR {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Socket
// ---------------------------------------------------------------------------

/// A ZMQ socket, closed on drop.
///
/// The `'ctx` borrow ties the socket to its [`Context`]: the context cannot be
/// dropped while a socket is live, which is precisely libzmq's teardown rule.
pub struct Socket<'ctx> {
    sock: *mut c_void,
    _ctx: PhantomData<&'ctx Context>,
}

// SAFETY: libzmq sockets may be migrated between threads provided a full memory
// barrier separates the two threads' use of them. Moving a value across a Rust
// thread boundary always implies that synchronization, so `Send` is sound.
//
// `Socket` is deliberately NOT `Sync` (the raw pointer field keeps it that way):
// libzmq sockets are not thread-safe, and concurrent send-on-one-thread +
// recv-on-another corrupts libzmq's internal pipe_t/msg_t state — the exact bug
// the C++ transport's `sock_mtx_` exists to prevent (it documents SIGSEGVs in
// zmq::pipe_t::read and "double free or corruption" from that race). Here the
// type system prevents sharing outright.
unsafe impl Send for Socket<'_> {}

impl Socket<'_> {
    /// `zmq_bind` — e.g. `tcp://127.0.0.1:8192`, `tcp://0.0.0.0:0`, `ipc:///tmp/x`.
    pub fn bind(&self, endpoint: &str) -> Result<()> {
        let c = to_cstring(endpoint, "zmq_bind")?;
        // SAFETY: live socket; `c` is a NUL-terminated string valid for the call.
        if unsafe { zmq_bind(self.sock, c.as_ptr()) } != 0 {
            return Err(ZmqError::last(format!("zmq_bind({endpoint})")));
        }
        Ok(())
    }

    /// `zmq_connect`. Asynchronous: it returns before the peer is reachable.
    pub fn connect(&self, endpoint: &str) -> Result<()> {
        let c = to_cstring(endpoint, "zmq_connect")?;
        // SAFETY: live socket; `c` is a NUL-terminated string valid for the call.
        if unsafe { zmq_connect(self.sock, c.as_ptr()) } != 0 {
            return Err(ZmqError::last(format!("zmq_connect({endpoint})")));
        }
        Ok(())
    }

    // -- options ------------------------------------------------------------

    /// `zmq_setsockopt` for an `int`-valued option.
    pub fn set_int(&self, option: SockOpt, value: i32) -> Result<()> {
        let v: c_int = value;
        // SAFETY: live socket; we pass a valid pointer to `v` and its true
        // size. libzmq validates option/size pairs and returns EINVAL on a
        // mismatch rather than reading past the value.
        let rc = unsafe {
            zmq_setsockopt(
                self.sock,
                option as c_int,
                &v as *const c_int as *const c_void,
                std::mem::size_of::<c_int>(),
            )
        };
        if rc != 0 {
            return Err(ZmqError::last(format!("zmq_setsockopt({option:?})")));
        }
        Ok(())
    }

    /// `zmq_setsockopt` for a byte-valued option (`RoutingId`, `Subscribe`, …).
    pub fn set_bytes(&self, option: SockOpt, value: &[u8]) -> Result<()> {
        // A zero-length value is meaningful (e.g. SUBSCRIBE "" = all topics).
        // Hand libzmq a real pointer rather than a dangling one so its internal
        // memcpy never sees a bogus source, even with len 0.
        static EMPTY: [u8; 1] = [0];
        let ptr = if value.is_empty() {
            EMPTY.as_ptr()
        } else {
            value.as_ptr()
        };
        // SAFETY: live socket; `ptr` is valid for `value.len()` bytes and
        // libzmq only reads that many.
        let rc =
            unsafe { zmq_setsockopt(self.sock, option as c_int, ptr as *const c_void, value.len()) };
        if rc != 0 {
            return Err(ZmqError::last(format!("zmq_setsockopt({option:?})")));
        }
        Ok(())
    }

    /// `zmq_getsockopt` for an `int`-valued option.
    pub fn get_int(&self, option: SockOpt) -> Result<i32> {
        let mut v: c_int = 0;
        let mut len = std::mem::size_of::<c_int>();
        // SAFETY: live socket; `v`/`len` are a valid out-param pair describing
        // exactly the storage libzmq may write. Non-int options fail with
        // EINVAL (size mismatch) instead of overflowing `v`.
        let rc = unsafe {
            zmq_getsockopt(
                self.sock,
                option as c_int,
                &mut v as *mut c_int as *mut c_void,
                &mut len,
            )
        };
        if rc != 0 {
            return Err(ZmqError::last(format!("zmq_getsockopt({option:?})")));
        }
        Ok(v)
    }

    /// `zmq_getsockopt` for a byte/string-valued option. `max_len` bounds the
    /// buffer libzmq may fill; too small a buffer fails with `EINVAL`.
    pub fn get_bytes(&self, option: SockOpt, max_len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; max_len.max(1)];
        let mut len = buf.len();
        // SAFETY: live socket; `buf` is valid for `len` bytes and libzmq writes
        // at most that many, updating `len` to what it wrote.
        let rc = unsafe {
            zmq_getsockopt(
                self.sock,
                option as c_int,
                buf.as_mut_ptr() as *mut c_void,
                &mut len,
            )
        };
        if rc != 0 {
            return Err(ZmqError::last(format!("zmq_getsockopt({option:?})")));
        }
        buf.truncate(len);
        Ok(buf)
    }

    /// `ZMQ_IDENTITY`/`ZMQ_ROUTING_ID`: the routing prefix a peer ROUTER uses to
    /// address this socket. Must be set **before** [`Self::connect`].
    ///
    /// CTP requires this to be unique per DEALER: identical identities make the
    /// ROUTER coalesce connections and silently drop all but one sender.
    pub fn set_identity(&self, identity: &[u8]) -> Result<()> {
        self.set_bytes(SockOpt::RoutingId, identity)
    }

    /// `ZMQ_SUBSCRIBE` (SUB sockets). Empty prefix subscribes to everything.
    pub fn subscribe(&self, prefix: &[u8]) -> Result<()> {
        self.set_bytes(SockOpt::Subscribe, prefix)
    }

    /// `ZMQ_LINGER` in ms: how long a close waits for unsent messages. `0`
    /// (this crate's default, via `ZMQ_BLOCKY = 0`) discards; `-1` blocks
    /// forever — with `-1`, `Context::drop` can hang on undelivered data.
    pub fn set_linger(&self, millis: i32) -> Result<()> {
        self.set_int(SockOpt::Linger, millis)
    }

    /// `ZMQ_SNDTIMEO` in ms (`-1` blocks forever). The C++ ROUTER caps this at
    /// 1 s so a back-pressured peer cannot deadlock the single net worker.
    pub fn set_sndtimeo(&self, millis: i32) -> Result<()> {
        self.set_int(SockOpt::SndTimeo, millis)
    }

    /// `ZMQ_RCVTIMEO` in ms (`-1` blocks forever).
    pub fn set_rcvtimeo(&self, millis: i32) -> Result<()> {
        self.set_int(SockOpt::RcvTimeo, millis)
    }

    /// `ZMQ_SNDHWM`: outbound queue depth in messages (default 1000).
    pub fn set_sndhwm(&self, messages: i32) -> Result<()> {
        self.set_int(SockOpt::SndHwm, messages)
    }

    /// `ZMQ_RCVHWM`: inbound queue depth in messages (default 1000).
    pub fn set_rcvhwm(&self, messages: i32) -> Result<()> {
        self.set_int(SockOpt::RcvHwm, messages)
    }

    /// `ZMQ_SNDBUF`: kernel send buffer in bytes (0 = OS default).
    pub fn set_sndbuf(&self, bytes: i32) -> Result<()> {
        self.set_int(SockOpt::SndBuf, bytes)
    }

    /// `ZMQ_RCVBUF`: kernel receive buffer in bytes (0 = OS default).
    pub fn set_rcvbuf(&self, bytes: i32) -> Result<()> {
        self.set_int(SockOpt::RcvBuf, bytes)
    }

    /// `ZMQ_IMMEDIATE`: when true, queue messages only to completed
    /// connections instead of buffering for pending ones.
    pub fn set_immediate(&self, immediate: bool) -> Result<()> {
        self.set_int(SockOpt::Immediate, immediate as i32)
    }

    /// `ZMQ_ROUTER_MANDATORY`: make a send to an unroutable identity fail
    /// (`EHOSTUNREACH`) instead of dropping silently.
    pub fn set_router_mandatory(&self, mandatory: bool) -> Result<()> {
        self.set_int(SockOpt::RouterMandatory, mandatory as i32)
    }

    /// `ZMQ_ROUTER_HANDOVER`: let a reconnecting peer with an existing identity
    /// take over that identity (typical after a client restart).
    pub fn set_router_handover(&self, handover: bool) -> Result<()> {
        self.set_int(SockOpt::RouterHandover, handover as i32)
    }

    /// ZMTP heartbeat: `ZMQ_HEARTBEAT_IVL` (PING period),
    /// `ZMQ_HEARTBEAT_TIMEOUT` (silence before the peer is declared dead), and
    /// `ZMQ_HEARTBEAT_TTL` — all in ms.
    pub fn set_heartbeat(&self, ivl_ms: i32, timeout_ms: i32, ttl_ms: i32) -> Result<()> {
        self.set_int(SockOpt::HeartbeatIvl, ivl_ms)?;
        self.set_int(SockOpt::HeartbeatTimeout, timeout_ms)?;
        self.set_int(SockOpt::HeartbeatTtl, ttl_ms)
    }

    /// `ZMQ_TYPE`: this socket's type.
    pub fn socket_type(&self) -> Result<SocketType> {
        let raw = self.get_int(SockOpt::Type)?;
        SocketType::from_raw(raw)
            .ok_or_else(|| ZmqError::wrapper(format!("zmq_getsockopt(TYPE): unknown type {raw}")))
    }

    /// `ZMQ_LAST_ENDPOINT`: the endpoint actually bound. This is how CTP
    /// resolves the real port after binding to ephemeral `:0`.
    pub fn last_endpoint(&self) -> Result<String> {
        let mut b = self.get_bytes(SockOpt::LastEndpoint, 512)?;
        // libzmq NUL-terminates and counts the NUL in the returned length.
        if b.last() == Some(&0) {
            b.pop();
        }
        String::from_utf8(b)
            .map_err(|_| ZmqError::wrapper("zmq_getsockopt(LAST_ENDPOINT): invalid UTF-8"))
    }

    /// `ZMQ_RCVMORE`: true if more frames of the current multipart message remain.
    pub fn has_more(&self) -> Result<bool> {
        Ok(self.get_int(SockOpt::RcvMore)? != 0)
    }

    /// `ZMQ_EVENTS`: readiness bits. Reading this is what re-arms the
    /// edge-triggered [`Self::fd`], as libzmq requires.
    pub fn events(&self) -> Result<PollEvents> {
        Ok(PollEvents(self.get_int(SockOpt::Events)? as c_short))
    }

    /// `ZMQ_FD`: the edge-triggered descriptor to register with epoll/kqueue.
    ///
    /// It signals only that [`Self::events`] should be re-read — it is never a
    /// substitute for it, and readability of the fd does not mean a message is
    /// available. Prefer [`Self::poll`], which works on Windows too.
    pub fn fd(&self) -> Result<SocketFd> {
        let mut v: SocketFd = 0;
        let mut len = std::mem::size_of::<SocketFd>();
        // SAFETY: live socket; ZMQ_FD's value is `zmq_fd_t`, which `SocketFd`
        // matches per target, so `v`/`len` describe exactly libzmq's write.
        let rc = unsafe {
            zmq_getsockopt(
                self.sock,
                SockOpt::Fd as c_int,
                &mut v as *mut SocketFd as *mut c_void,
                &mut len,
            )
        };
        if rc != 0 {
            return Err(ZmqError::last("zmq_getsockopt(FD)"));
        }
        Ok(v)
    }

    // -- transfer -----------------------------------------------------------

    /// Send one frame, copying `data` into libzmq's own buffer before
    /// returning — so `data` may be reused or freed immediately after.
    ///
    /// Retries `EINTR` internally (the C++ `zmq_send_eintr`). Pass
    /// [`Flags::SNDMORE`] for every frame of a multipart message but the last.
    pub fn send(&self, data: &[u8], flags: Flags) -> Result<()> {
        // Empty frames are load-bearing in CTP's wire format (the DEALER/ROUTER
        // delimiter). Give libzmq a real pointer for them rather than the
        // dangling one an empty slice carries.
        static EMPTY: [u8; 1] = [0];
        let ptr = if data.is_empty() {
            EMPTY.as_ptr()
        } else {
            data.as_ptr()
        };
        loop {
            // SAFETY: live socket; `ptr` is valid for `data.len()` bytes, which
            // is all libzmq reads, and it copies them before returning.
            let rc = unsafe { zmq_send(self.sock, ptr as *const c_void, data.len(), flags.bits()) };
            if rc >= 0 {
                return Ok(());
            }
            let err = ZmqError::last("zmq_send");
            // A signal (e.g. SIGCHLD from a child exit) must not poison an
            // in-flight multipart message; just retry.
            if err.is_intr() {
                continue;
            }
            return Err(err);
        }
    }

    /// Receive one frame into `buf`, returning the message's size **on the
    /// wire**. A return value greater than `buf.len()` means the message was
    /// truncated and the excess discarded — matching `zmq_recv`. Use
    /// [`Self::recv_msg`] to avoid truncation and the copy.
    ///
    /// Retries `EINTR` internally.
    pub fn recv_into(&self, buf: &mut [u8], flags: Flags) -> Result<usize> {
        loop {
            // SAFETY: live socket; `buf` is valid for `buf.len()` bytes, which
            // bounds libzmq's copy. For an empty `buf` the pointer is dangling
            // but libzmq copies min(len, size) = 0 bytes and never reads it.
            let rc = unsafe {
                zmq_recv(
                    self.sock,
                    buf.as_mut_ptr() as *mut c_void,
                    buf.len(),
                    flags.bits(),
                )
            };
            if rc >= 0 {
                return Ok(rc as usize);
            }
            let err = ZmqError::last("zmq_recv");
            if err.is_intr() {
                continue;
            }
            return Err(err);
        }
    }

    /// Send a [`Message`], transferring its payload to libzmq.
    ///
    /// On success `msg` is left empty (libzmq nullifies it) — still valid, just
    /// zero-length. On failure `msg` is untouched and can be retried, which is
    /// what makes `EAGAIN` recoverable here.
    ///
    /// Retries `EINTR` internally (the C++ `zmq_msg_send_eintr`).
    pub fn send_msg(&self, msg: &mut Message, flags: Flags) -> Result<()> {
        loop {
            // SAFETY: live socket; `msg.raw` is an initialized zmq_msg_t we
            // exclusively own (&mut). libzmq either takes the payload and
            // re-inits the struct, or leaves it untouched on error; either way
            // it stays valid for Message::drop's zmq_msg_close.
            let rc = unsafe { zmq_msg_send(&mut msg.raw, self.sock, flags.bits()) };
            if rc >= 0 {
                return Ok(());
            }
            let err = ZmqError::last("zmq_msg_send");
            if err.is_intr() {
                continue;
            }
            return Err(err);
        }
    }

    /// Receive one frame as a [`Message`], with no copy of the payload — the
    /// zero-copy recv path CTP's `RecvBulks` uses.
    ///
    /// Retries `EINTR` internally (the C++ `zmq_msg_recv_eintr`).
    pub fn recv_msg(&self, flags: Flags) -> Result<Message> {
        let mut msg = Message::new()?;
        loop {
            // SAFETY: live socket; `msg.raw` is an initialized zmq_msg_t we
            // exclusively own. On error it remains initialized, so dropping
            // `msg` (zmq_msg_close) stays correct on every path.
            let rc = unsafe { zmq_msg_recv(&mut msg.raw, self.sock, flags.bits()) };
            if rc >= 0 {
                return Ok(msg);
            }
            let err = ZmqError::last("zmq_msg_recv");
            if err.is_intr() {
                continue;
            }
            return Err(err);
        }
    }

    /// Wait up to `timeout_ms` for readiness (`-1` waits forever, `0` polls).
    /// Returns the events that fired; empty means the timeout expired.
    ///
    /// This is the correct readiness primitive for a ZMQ socket: it consults
    /// `ZMQ_EVENTS` rather than the raw [`Self::fd`], so it also works on
    /// Windows, where that fd cannot be watched by epoll/WSAEventSelect.
    pub fn poll(&self, events: PollEvents, timeout_ms: i32) -> Result<PollEvents> {
        let mut item = ZmqPollItemT {
            socket: self.sock,
            // Ignored when `socket` is non-null.
            fd: 0 as SocketFd,
            events: events.bits(),
            revents: 0,
        };
        // SAFETY: one valid, fully-initialized pollitem is passed with nitems=1;
        // libzmq writes only `revents`. The timeout is C `long` (see module docs).
        let rc = unsafe { zmq_poll(&mut item, 1, timeout_ms as c_long) };
        if rc < 0 {
            return Err(ZmqError::last("zmq_poll"));
        }
        Ok(PollEvents(item.revents))
    }

    /// Convenience: true if a message can be received within `timeout_ms`.
    pub fn poll_in(&self, timeout_ms: i32) -> Result<bool> {
        Ok(self.poll(PollEvents::IN, timeout_ms)?.contains(PollEvents::IN))
    }
}

impl Drop for Socket<'_> {
    fn drop(&mut self) {
        // SAFETY: `sock` came from zmq_socket and is closed exactly once; the
        // context it belongs to is still alive (we hold a borrow of it).
        // Nothing to report from a destructor, and LINGER=0 (ZMQ_BLOCKY=0)
        // means this cannot block.
        unsafe { zmq_close(self.sock) };
    }
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// An owned `zmq_msg_t`, closed on drop.
///
/// Received messages own libzmq's buffer directly, so [`Socket::recv_msg`] +
/// [`Message::as_slice`] reads a frame without copying it.
pub struct Message {
    raw: ZmqMsgT,
}

impl Message {
    /// An empty message (`zmq_msg_init`).
    pub fn new() -> Result<Self> {
        let mut raw = ZmqMsgT { _opaque: [0u8; 64] };
        // SAFETY: `raw` is 64 writable bytes with the alignment libzmq expects
        // (asserted against ZMQ_MSG_T_SIZE by `abi_check`). zmq_msg_init fully
        // initializes it. Moving the struct afterwards is fine: libzmq stores
        // no pointer into the struct itself — small messages are inline and
        // addressed relative to the struct, larger ones point at the heap.
        if unsafe { zmq_msg_init(&mut raw) } != 0 {
            return Err(ZmqError::last("zmq_msg_init"));
        }
        Ok(Self { raw })
    }

    /// A message with `size` bytes of uninitialized-but-owned payload
    /// (`zmq_msg_init_size`).
    pub fn with_size(size: usize) -> Result<Self> {
        let mut raw = ZmqMsgT { _opaque: [0u8; 64] };
        // SAFETY: as in `new`; zmq_msg_init_size allocates the payload and
        // initializes `raw`, or fails with ENOMEM leaving it uninitialized
        // (in which case we never construct a Message and never close it).
        if unsafe { zmq_msg_init_size(&mut raw, size) } != 0 {
            return Err(ZmqError::last("zmq_msg_init_size"));
        }
        let mut msg = Self { raw };
        // zmq_msg_init_size leaves the payload uninitialized; zero it so
        // as_slice never exposes stale heap bytes.
        msg.as_mut_slice().fill(0);
        Ok(msg)
    }

    /// A message holding a copy of `data`.
    pub fn from_slice(data: &[u8]) -> Result<Self> {
        let mut msg = Self::with_size(data.len())?;
        msg.as_mut_slice().copy_from_slice(data);
        Ok(msg)
    }

    /// Payload length in bytes.
    pub fn len(&self) -> usize {
        // SAFETY: `raw` is initialized; zmq_msg_size only reads it.
        unsafe { zmq_msg_size(&self.raw) }
    }

    /// True if the payload is empty (e.g. a DEALER/ROUTER delimiter frame).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True if more frames of this multipart message follow (`zmq_msg_more`).
    pub fn has_more(&self) -> bool {
        // SAFETY: `raw` is initialized; zmq_msg_more only reads it.
        unsafe { zmq_msg_more(&self.raw) != 0 }
    }

    /// The payload.
    pub fn as_slice(&self) -> &[u8] {
        let len = self.len();
        if len == 0 {
            return &[];
        }
        // SAFETY: zmq_msg_data takes `zmq_msg_t*` only because the C API
        // predates const-correct accessors — it does not mutate the message, so
        // deriving the pointer from `&self` is sound. It returns a pointer to
        // `len` initialized bytes owned by this message, and the returned slice
        // borrows `self`, so it cannot outlive the message or a move of it.
        unsafe {
            let p = zmq_msg_data(&self.raw as *const ZmqMsgT as *mut ZmqMsgT);
            std::slice::from_raw_parts(p as *const u8, len)
        }
    }

    /// The payload, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.len();
        if len == 0 {
            return &mut [];
        }
        // SAFETY: as in `as_slice`, but `&mut self` guarantees exclusive access
        // to the `len` bytes this message owns.
        unsafe {
            let p = zmq_msg_data(&mut self.raw);
            std::slice::from_raw_parts_mut(p as *mut u8, len)
        }
    }
}

impl Drop for Message {
    fn drop(&mut self) {
        // SAFETY: `raw` was initialized by a zmq_msg_init* call in every
        // constructor and is closed exactly once. Closing a message that
        // zmq_msg_send nullified is valid — it is an empty, initialized message.
        unsafe { zmq_msg_close(&mut self.raw) };
    }
}

impl std::fmt::Debug for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Message")
            .field("len", &self.len())
            .field("more", &self.has_more())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Unique loopback/ipc names so concurrently-run tests never collide.
    fn unique_id() -> String {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        format!(
            "{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        )
    }

    /// Bind a PULL to an ephemeral loopback port and report the real endpoint.
    fn bind_ephemeral<'c>(ctx: &'c Context, ty: SocketType) -> (Socket<'c>, String) {
        let sock = ctx.socket(ty).unwrap();
        sock.bind("tcp://127.0.0.1:0").unwrap();
        let ep = sock.last_endpoint().unwrap();
        (sock, ep)
    }

    /// The layout assumption that could corrupt memory: ask the live libzmq for
    /// its own sizeof(zmq_msg_t) and compare with our hand-rolled struct.
    #[test]
    fn msg_t_abi_matches_live_library() {
        let (major, minor, patch) = version();
        eprintln!("ctp-net: libzmq {major}.{minor}.{patch}");
        assert_eq!(major, 4, "these bindings target the libzmq 4.x C ABI");

        let ctx = Context::new().unwrap();
        assert_eq!(ctx.msg_t_size().unwrap(), std::mem::size_of::<ZmqMsgT>());
        assert_eq!(std::mem::size_of::<ZmqMsgT>(), 64);
        assert_eq!(std::mem::align_of::<ZmqMsgT>(), 8);
        abi_check().unwrap();
    }

    /// Every socket-type constant must round-trip through ZMQ_TYPE, which
    /// proves both the type ids and ZMQ_TYPE=16 are transcribed correctly.
    #[test]
    fn socket_type_constants_round_trip() {
        let ctx = Context::new().unwrap();
        for ty in [
            SocketType::Pair,
            SocketType::Pub,
            SocketType::Sub,
            SocketType::Req,
            SocketType::Rep,
            SocketType::Dealer,
            SocketType::Router,
            SocketType::Pull,
            SocketType::Push,
            SocketType::XPub,
            SocketType::XSub,
            SocketType::Stream,
        ] {
            let s = ctx.socket(ty).unwrap();
            assert_eq!(s.socket_type().unwrap(), ty, "ZMQ_TYPE mismatch for {ty:?}");
        }
        // The values the issue asked us to verify, against zmq.h 4.3.5.
        assert_eq!(SocketType::Push as i32, 8);
        assert_eq!(SocketType::Pull as i32, 7);
        assert_eq!(SocketType::Dealer as i32, 5);
        assert_eq!(SocketType::Router as i32, 6);
        assert_eq!(SocketType::Pub as i32, 1);
        assert_eq!(SocketType::Sub as i32, 2);
        assert_eq!(SocketType::Req as i32, 3);
        assert_eq!(SocketType::Rep as i32, 4);
    }

    /// The headline test: a real PUSH → PULL round trip over TCP loopback,
    /// including the ephemeral-port resolution CTP does via ZMQ_LAST_ENDPOINT.
    #[test]
    fn push_pull_roundtrip_tcp() {
        let ctx = Context::new().unwrap();
        let (pull, endpoint) = bind_ephemeral(&ctx, SocketType::Pull);
        assert!(
            endpoint.starts_with("tcp://127.0.0.1:") && !endpoint.ends_with(":0"),
            "ZMQ_LAST_ENDPOINT should resolve the ephemeral port, got {endpoint}"
        );
        pull.set_rcvtimeo(5_000).unwrap();

        let push = ctx.socket(SocketType::Push).unwrap();
        push.set_sndtimeo(5_000).unwrap();
        push.connect(&endpoint).unwrap();

        for i in 0..8u8 {
            push.send(&[i; 4], Flags::NONE).unwrap();
        }
        for i in 0..8u8 {
            let msg = pull.recv_msg(Flags::NONE).unwrap();
            assert_eq!(msg.as_slice(), &[i; 4], "payload {i} corrupted in transit");
            assert!(!msg.has_more());
        }

        // Same round trip through the copy-in/copy-out path.
        push.send(b"lightbeam", Flags::NONE).unwrap();
        let mut buf = [0u8; 32];
        let n = pull.recv_into(&mut buf, Flags::NONE).unwrap();
        assert_eq!(&buf[..n], b"lightbeam");
    }

    /// Multipart framing: SNDMORE + RCVMORE, the shape of every CTP message
    /// (delimiter / meta / bulk…).
    #[test]
    fn push_pull_multipart_roundtrip() {
        let ctx = Context::new().unwrap();
        let (pull, endpoint) = bind_ephemeral(&ctx, SocketType::Pull);
        pull.set_rcvtimeo(5_000).unwrap();
        let push = ctx.socket(SocketType::Push).unwrap();
        push.set_sndtimeo(5_000).unwrap();
        push.connect(&endpoint).unwrap();

        // Empty delimiter, then meta, then a 1 MiB "bulk".
        let bulk = vec![0xABu8; 1 << 20];
        push.send(b"", Flags::SNDMORE).unwrap();
        push.send(b"meta", Flags::SNDMORE).unwrap();
        push.send(&bulk, Flags::NONE).unwrap();

        let delim = pull.recv_msg(Flags::NONE).unwrap();
        assert!(delim.is_empty());
        assert!(delim.has_more());
        assert!(pull.has_more().unwrap());

        let meta = pull.recv_msg(Flags::NONE).unwrap();
        assert_eq!(meta.as_slice(), b"meta");
        assert!(meta.has_more());

        let got = pull.recv_msg(Flags::NONE).unwrap();
        assert_eq!(got.len(), bulk.len());
        assert_eq!(got.as_slice(), &bulk[..], "1 MiB bulk frame corrupted");
        assert!(!got.has_more());
        assert!(!pull.has_more().unwrap());
    }

    /// PUSH/PULL over ipc:// — CTP's same-host protocol.
    #[cfg(unix)]
    #[test]
    fn push_pull_roundtrip_ipc() {
        let path = std::env::temp_dir().join(format!("ctp-net-zmq-{}.ipc", unique_id()));
        let endpoint = format!("ipc://{}", path.display());

        let ctx = Context::new().unwrap();
        {
            let pull = ctx.socket(SocketType::Pull).unwrap();
            pull.bind(&endpoint).unwrap();
            pull.set_rcvtimeo(5_000).unwrap();

            let push = ctx.socket(SocketType::Push).unwrap();
            push.set_sndtimeo(5_000).unwrap();
            push.connect(&endpoint).unwrap();

            push.send(b"over ipc", Flags::NONE).unwrap();
            let msg = pull.recv_msg(Flags::NONE).unwrap();
            assert_eq!(msg.as_slice(), b"over ipc");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// PUSH is send-only and PULL is recv-only. This is what distinguishes
    /// PUSH=8/PULL=7 from, say, PAIR: a wrong constant would block-then-EAGAIN
    /// instead of refusing outright.
    #[test]
    fn push_and_pull_are_unidirectional() {
        let ctx = Context::new().unwrap();
        let (pull, endpoint) = bind_ephemeral(&ctx, SocketType::Pull);
        // Bounded, so a mis-transcribed constant fails the assert instead of
        // hanging the suite.
        pull.set_sndtimeo(100).unwrap();
        let push = ctx.socket(SocketType::Push).unwrap();
        push.set_rcvtimeo(100).unwrap();
        push.connect(&endpoint).unwrap();

        let err = push.recv_into(&mut [0u8; 4], Flags::NONE).unwrap_err();
        assert!(!err.is_again(), "PUSH must refuse recv outright, got {err}");
        let err = pull.send(b"nope", Flags::NONE).unwrap_err();
        assert!(!err.is_again(), "PULL must refuse send outright, got {err}");
    }

    /// CTP's real topology: a DEALER's identity arrives at the ROUTER as the
    /// routing prefix, and a reply addressed to it comes back. Exercises
    /// DEALER=5, ROUTER=6, ZMQ_ROUTING_ID=5 and ZMQ_ROUTER_MANDATORY=33.
    #[test]
    fn dealer_router_identity_roundtrip() {
        let ctx = Context::new().unwrap();
        let (router, endpoint) = bind_ephemeral(&ctx, SocketType::Router);
        router.set_router_mandatory(true).unwrap();
        router.set_router_handover(true).unwrap();
        router.set_rcvtimeo(5_000).unwrap();
        router.set_sndtimeo(5_000).unwrap();

        let identity = format!("host:{}", unique_id());
        let dealer = ctx.socket(SocketType::Dealer).unwrap();
        dealer.set_identity(identity.as_bytes()).unwrap();
        dealer.set_rcvtimeo(5_000).unwrap();
        dealer.set_sndtimeo(5_000).unwrap();
        dealer.connect(&endpoint).unwrap();

        // DEALER → ROUTER: [delimiter, payload]
        dealer.send(b"", Flags::SNDMORE).unwrap();
        dealer.send(b"request", Flags::NONE).unwrap();

        // ROUTER sees [identity, delimiter, payload]
        let id_frame = router.recv_msg(Flags::NONE).unwrap();
        assert_eq!(id_frame.as_slice(), identity.as_bytes());
        assert!(id_frame.has_more());
        let delim = router.recv_msg(Flags::NONE).unwrap();
        assert!(delim.is_empty() && delim.has_more());
        let body = router.recv_msg(Flags::NONE).unwrap();
        assert_eq!(body.as_slice(), b"request");
        assert!(!body.has_more());

        // ROUTER → DEALER: prefix the identity to route the reply back.
        router.send(identity.as_bytes(), Flags::SNDMORE).unwrap();
        router.send(b"", Flags::SNDMORE).unwrap();
        router.send(b"response", Flags::NONE).unwrap();

        let delim = dealer.recv_msg(Flags::NONE).unwrap();
        assert!(delim.is_empty() && delim.has_more());
        let reply = dealer.recv_msg(Flags::NONE).unwrap();
        assert_eq!(reply.as_slice(), b"response");
    }

    /// Send a Message rather than a slice; libzmq nullifies it on success.
    #[test]
    fn send_msg_transfers_payload_and_empties_message() {
        let ctx = Context::new().unwrap();
        let (pull, endpoint) = bind_ephemeral(&ctx, SocketType::Pull);
        pull.set_rcvtimeo(5_000).unwrap();
        let push = ctx.socket(SocketType::Push).unwrap();
        push.set_sndtimeo(5_000).unwrap();
        push.connect(&endpoint).unwrap();

        let mut msg = Message::from_slice(b"owned payload").unwrap();
        assert_eq!(msg.len(), 13);
        push.send_msg(&mut msg, Flags::NONE).unwrap();
        // Nullified by libzmq: still a valid (empty) message, safe to drop.
        assert!(msg.is_empty());

        let got = pull.recv_msg(Flags::NONE).unwrap();
        assert_eq!(got.as_slice(), b"owned payload");
    }

    /// A message survives being moved (returned, pushed into a Vec) — the
    /// relocatability the `Message` wrapper relies on.
    #[test]
    fn message_survives_moves() {
        let mut msg = Message::from_slice(b"relocate me").unwrap();
        msg.as_mut_slice()[0] = b'R';
        // Move the (already initialized) zmq_msg_t onto the heap, then read it
        // back from its new address: proves libzmq keeps no pointer into the
        // struct itself, which is what makes `Message` movable.
        let moved = Box::new(msg);
        assert_eq!(moved.as_slice(), b"Relocate me");

        let empty = Message::new().unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.as_slice(), b"");
        assert_eq!(Message::with_size(8).unwrap().as_slice(), &[0u8; 8]);
    }

    /// Timeouts and non-blocking recv surface as EAGAIN, not a panic — the
    /// error CTP's SendOut/RecvIn paths re-queue on.
    #[test]
    fn timeout_and_dontwait_report_eagain() {
        let ctx = Context::new().unwrap();
        let (pull, _endpoint) = bind_ephemeral(&ctx, SocketType::Pull);

        pull.set_rcvtimeo(50).unwrap();
        let err = pull.recv_msg(Flags::NONE).unwrap_err();
        assert!(err.is_again(), "expected EAGAIN, got {err}");
        assert_eq!(err.code(), errno::EAGAIN);
        assert!(!err.message().is_empty());
        // The failing call is named in the error, so a log line identifies it.
        assert!(format!("{err}").contains("zmq_msg_recv"), "got {err}");

        let err = pull.recv_into(&mut [0u8; 4], Flags::DONTWAIT).unwrap_err();
        assert!(err.is_again(), "expected EAGAIN, got {err}");
        assert!(format!("{err}").contains("zmq_recv"), "got {err}");

        // Nothing to read => poll times out and reports no events.
        assert!(pull.poll(PollEvents::IN, 10).unwrap().is_empty());
        assert!(!pull.poll_in(10).unwrap());
    }

    /// poll/events report writability once a peer is attached, and ZMQ_FD is
    /// retrievable for the epoll-registration path.
    #[test]
    fn poll_reports_readiness() {
        let ctx = Context::new().unwrap();
        let (pull, endpoint) = bind_ephemeral(&ctx, SocketType::Pull);
        pull.set_rcvtimeo(5_000).unwrap();
        let push = ctx.socket(SocketType::Push).unwrap();
        push.set_sndtimeo(5_000).unwrap();
        push.connect(&endpoint).unwrap();

        assert!(push.poll(PollEvents::OUT, 5_000).unwrap().contains(PollEvents::OUT));
        push.send(b"ready", Flags::NONE).unwrap();
        assert!(pull.poll_in(5_000).unwrap(), "message should arrive");
        assert!(pull.events().unwrap().contains(PollEvents::IN));
        let _fd = pull.fd().unwrap();
        assert_eq!(pull.recv_msg(Flags::NONE).unwrap().as_slice(), b"ready");
    }

    /// Context options: the ZMQ_BLOCKY=0 default (so Drop can't hang) and
    /// with_io_threads.
    #[test]
    fn context_options() {
        let ctx = Context::new().unwrap();
        assert_eq!(ctx.get(CtxOpt::Blocky).unwrap(), 0, "new sockets get LINGER=0");
        assert!(ctx.get(CtxOpt::SocketLimit).unwrap() > 0);

        let ctx = Context::with_io_threads(4).unwrap();
        assert_eq!(ctx.get(CtxOpt::IoThreads).unwrap(), 4);

        // Bad values are errors, not panics.
        assert!(ctx.set(CtxOpt::IoThreads, -1).is_err());
    }

    /// Endpoint mistakes are reported, not panicked on, and the socket stays
    /// usable afterwards.
    #[test]
    fn bad_endpoints_return_errors() {
        let ctx = Context::new().unwrap();
        let sock = ctx.socket(SocketType::Push).unwrap();

        let err = sock.connect("not-a-protocol://x").unwrap_err();
        assert!(format!("{err}").contains("zmq_connect"));
        assert!(sock.bind("tcp://127.0.0.1:not-a-port").is_err());
        // Interior NUL is caught in Rust, before reaching C.
        let err = sock.bind("tcp://127.0.0.1:\0 1234").unwrap_err();
        assert_eq!(err.code(), 0);
        assert!(format!("{err}").contains("interior NUL"));

        // Still healthy.
        let (pull, endpoint) = bind_ephemeral(&ctx, SocketType::Pull);
        pull.set_rcvtimeo(5_000).unwrap();
        sock.set_sndtimeo(5_000).unwrap();
        sock.connect(&endpoint).unwrap();
        sock.send(b"ok", Flags::NONE).unwrap();
        assert_eq!(pull.recv_msg(Flags::NONE).unwrap().as_slice(), b"ok");
    }

    /// ETERM: after shutdown, blocking calls fail with a live-context error
    /// instead of hanging — this is how CTP unblocks its recv threads.
    #[test]
    fn shutdown_reports_eterm() {
        let ctx = Context::new().unwrap();
        let (pull, _endpoint) = bind_ephemeral(&ctx, SocketType::Pull);
        ctx.shutdown().unwrap();
        let err = pull.recv_msg(Flags::NONE).unwrap_err();
        assert!(err.is_term(), "expected ETERM, got {err}");
        assert_eq!(errno::ETERM, errno::HAUSNUMERO + 53);
    }

    /// Sockets are Send: a socket may be migrated to another thread (never
    /// shared with one). Also proves Context: Sync + Send.
    #[test]
    fn socket_migrates_to_another_thread() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Socket<'_>>();
        assert_send::<Context>();
        assert_sync::<Context>();
        assert_send::<Message>();

        let ctx = Context::new().unwrap();
        let (pull, endpoint) = bind_ephemeral(&ctx, SocketType::Pull);
        pull.set_rcvtimeo(5_000).unwrap();

        std::thread::scope(|s| {
            // The PULL socket is moved into the thread; the context is shared.
            let receiver = s.spawn(move || pull.recv_msg(Flags::NONE).unwrap());
            let push = ctx.socket(SocketType::Push).unwrap();
            push.set_sndtimeo(5_000).unwrap();
            push.connect(&endpoint).unwrap();
            push.send(b"cross-thread", Flags::NONE).unwrap();
            assert_eq!(receiver.join().unwrap().as_slice(), b"cross-thread");
        });
    }
}
