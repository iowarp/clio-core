// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Safe Rust wrapper over **thallium** (Mochi), the C++ RPC/RDMA library CTP
//! uses for its lightbeam thallium transport.
//!
//! # What is wrapped
//!
//! thallium sits on top of margo -> mercury (network) + argobots (user-level
//! threads). It is a **header-only C++14/17 template library**: `tl::engine`,
//! `tl::endpoint`, `tl::remote_procedure` and `tl::bulk` are C++ classes with
//! no C entry points, and its RPC (de)serialization is compile-time template
//! machinery. There is therefore nothing for Rust to bind to directly, so this
//! module binds `shim/thallium_shim.cc`, a C ABI (`ctp_th_*`) written against
//! the real thallium API — the same arrangement as
//! `ctp-coroutine/shim/boost_fiber_shim.cc`. Per the FFI convention used by
//! `ctp-gpu/src/cuda.rs`, the `extern "C"` block below is hand-rolled: no
//! bindgen, no external crates.
//!
//! The wrapped surface is the one
//! `clio_ctp/lightbeam/thallium_transport.h` actually exercises:
//!
//! | C++ | Rust |
//! |---|---|
//! | `tl::engine(proto, THALLIUM_SERVER_MODE, progress, rpc_threads)` | [`Engine::init`] |
//! | `engine.self()` | [`Engine::self_addr`] |
//! | `engine.lookup(addr)` | [`Engine::lookup`] -> [`Endpoint`] |
//! | `engine.finalize()` | [`Engine::finalize`] |
//! | `engine.define(name)` | [`Engine::define`] -> [`Rpc`] |
//! | `engine.define(name, handler)` | [`Engine::define_handler`] -> [`RpcHandler`] |
//! | `engine.expose(segs, read_only)` | [`Engine::expose_read`] -> [`Bulk`] |
//! | `rpc.on(ep)(meta, bulks)` | [`Rpc::call`] |
//! | `remote.on(sender) >> local` | [`RemoteBulk::pull_into`] / [`RemoteBulk::pull_to_vec`] |
//!
//! # Version / ABI assumptions
//!
//! * **thallium 0.10+** (Mochi stack: margo 0.9+, mercury 2.x, argobots 1.x),
//!   located by CMake as `find_package(thallium CONFIG REQUIRED)` exactly as
//!   the C++ tree does (root `CMakeLists.txt`, `CLIO_CORE_ENABLE_THALLIUM`).
//! * The shim assumes these thallium signatures: the four-argument engine
//!   constructor `engine(std::string, int mode, bool use_progress_thread,
//!   std::int32_t rpc_thread_count)`; `THALLIUM_SERVER_MODE` /
//!   `THALLIUM_CLIENT_MODE`; `engine::expose(std::vector<std::pair<void*,
//!   size_t>>, tl::bulk_mode)`; `bulk::size()`; `bulk::on(endpoint) >> bulk`;
//!   `request::get_endpoint()` / `request::respond(int)`.
//! * thallium itself has **no stable C ABI**; the stable ABI here is the shim's
//!   `ctp_th_*`, so Rust and the shim must be rebuilt together. Status codes
//!   and bulk-mode values are duplicated in both files and must stay in sync.
//! * `i32`/`c_int` response codes and `usize`/`size_t` sizes are assumed to
//!   match, which holds on every platform CTP targets.
//!
//! # Divergences from the C++ wrapper (`thallium_transport.h`)
//!
//! * **Scope**: this wraps the thallium primitives, not lightbeam. The C++
//!   `ThalliumTransport::Send`/`Recv`, its cereal `GlobalSerialize` of
//!   `LbmMeta`, and its `ThalliumRecvQueue` are *not* reproduced — metadata is
//!   an opaque `&[u8]` here. Callers build the transport on top.
//! * **Configuration**: the C++ engine reads `CLIO_LBM_THALLIUM_PROTOCOL` and
//!   `CLIO_LBM_THALLIUM_RPC_THREADS` from the environment.
//!   [`EngineConfig::default`] hard-codes the same defaults
//!   (`ofi+tcp;ofi_rxm`, progress thread on, 4 RPC threads) but takes them as
//!   explicit parameters; reading env vars is left to the caller.
//! * **Singleton**: [`Engine::init`] keeps the C++ `call_once` semantics —
//!   first init wins, later configs are silently ignored — but reports that
//!   case via [`Engine::init_was_first`] rather than hiding it.
//! * **Memory**: the C++ handler `malloc`s a buffer per bulk and hands the raw
//!   pointers to `Recv`, which frees them later in `ClearRecvHandles`.
//!   [`RemoteBulk::pull_to_vec`] returns an owned `Vec<u8>` instead, so there
//!   is no manual free and no leak on the error paths.
//! * **Errors**: the C++ side throws (`std::runtime_error` from a failed
//!   `lookup`) and logs via `HLOG`. Nothing throws across this ABI: every
//!   fallible call returns [`ThalliumError`] carrying the C++ `what()` text.
//!   Library errors never panic.
//! * **Handler lifetime**: registering a handler intentionally leaks the boxed
//!   closure — see [`Engine::define_handler`].
//! * **`Bulk` safety**: `ThalliumTransport::Expose` takes a raw `FullPtr<char>`
//!   and stashes a `new tl::bulk` in `Bulk::desc` for `Send` to `delete`.
//!   [`Bulk`] instead borrows the buffer (`Bulk<'a>`) and deregisters on drop,
//!   so a registration cannot outlive the memory it exposes and cannot be
//!   double-freed.
//!
//! # Availability
//!
//! Feature-gated behind `thallium` (off by default), mirroring
//! `CLIO_CORE_ENABLE_THALLIUM=OFF`. Nothing here is compiled or linked in a
//! default build.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---------------------------------------------------------------------------
// Raw C ABI exposed by shim/thallium_shim.cc (hand-rolled; keep in sync)
// ---------------------------------------------------------------------------

const CTP_TH_OK: c_int = 0;
const CTP_TH_ALREADY_INIT: c_int = 1;
const CTP_TH_ERR: c_int = -1;
const CTP_TH_ERR_INVAL: c_int = -2;
const CTP_TH_ERR_NOT_INIT: c_int = -3;

const CTP_TH_BULK_READ_ONLY: c_int = 0;

/// Handler trampoline: `(user, sender_ep, meta, meta_len, bulks, n_bulks)`.
/// The return value becomes the RPC response (`req.respond(rc)`).
type CtpThHandlerFn = unsafe extern "C" fn(
    user: *mut c_void,
    sender_ep: *mut c_void,
    meta: *const c_char,
    meta_len: usize,
    remote_bulks: *const *mut c_void,
    n_bulks: usize,
) -> c_int;

extern "C" {
    fn ctp_th_engine_init(
        protocol: *const c_char,
        server_mode: c_int,
        use_progress_thread: c_int,
        rpc_thread_count: c_int,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_th_engine_is_initialized() -> c_int;
    fn ctp_th_engine_self(
        buf: *mut c_char,
        buflen: usize,
        needed: *mut usize,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_th_engine_finalize();

    fn ctp_th_endpoint_lookup(
        addr: *const c_char,
        out_ep: *mut *mut c_void,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_th_endpoint_free(ep: *mut c_void);
    fn ctp_th_endpoint_to_string(
        ep: *mut c_void,
        buf: *mut c_char,
        buflen: usize,
        needed: *mut usize,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;

    fn ctp_th_rpc_define(
        name: *const c_char,
        out_rpc: *mut *mut c_void,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_th_rpc_define_handler(
        name: *const c_char,
        cb: CtpThHandlerFn,
        user: *mut c_void,
        out_rpc: *mut *mut c_void,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_th_rpc_free(rpc: *mut c_void);
    #[allow(clippy::too_many_arguments)]
    fn ctp_th_rpc_call(
        rpc: *mut c_void,
        ep: *mut c_void,
        meta: *const c_char,
        meta_len: usize,
        bulks: *const *mut c_void,
        n_bulks: usize,
        out_rc: *mut c_int,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;

    fn ctp_th_bulk_expose(
        ptr: *mut c_void,
        size: usize,
        mode: c_int,
        out_bulk: *mut *mut c_void,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_th_bulk_free(bulk: *mut c_void);
    fn ctp_th_bulk_size(bulk: *mut c_void) -> usize;
    fn ctp_th_bulk_pull_into(
        remote_bulk: *mut c_void,
        sender_ep: *mut c_void,
        dst: *mut c_void,
        dst_len: usize,
        errbuf: *mut c_char,
        errlen: usize,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Any failure reported by the shim, carrying the C++ exception text when the
/// underlying thallium call threw. The crate has no shared error type (see
/// `GpuError` in `ctp-gpu/src/cuda.rs` for the same per-module pattern).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThalliumError(pub String);

impl std::fmt::Display for ThalliumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ThalliumError: {}", self.0)
    }
}
impl std::error::Error for ThalliumError {}

/// Size of the per-call error-message buffer handed to the shim. The shim
/// truncates and NUL-terminates within this bound.
const ERR_LEN: usize = 512;

/// Stack buffer for the shim's `errbuf`/`errlen` out-parameter pair. This is
/// how error text crosses the ABI: no thread_local "last error" slot anywhere
/// (project rule, and header thread_locals duplicate per-DLL on Windows).
struct ErrBuf([c_char; ERR_LEN]);

impl ErrBuf {
    fn new() -> Self {
        ErrBuf([0; ERR_LEN])
    }

    fn ptr(&mut self) -> *mut c_char {
        self.0.as_mut_ptr()
    }

    fn msg(&self) -> String {
        // SAFETY: the buffer is zero-initialized and the shim only ever writes
        // a NUL-terminated string within ERR_LEN, so a NUL is always present.
        unsafe { CStr::from_ptr(self.0.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Map a shim status to a `Result`. Nonnegative statuses are success
/// (`CTP_TH_ALREADY_INIT` is 1).
fn th_check(rc: c_int, err: &ErrBuf, what: &str) -> Result<(), ThalliumError> {
    if rc >= CTP_TH_OK {
        return Ok(());
    }
    let kind = match rc {
        CTP_TH_ERR_INVAL => "invalid argument",
        CTP_TH_ERR_NOT_INIT => "engine not initialized (call Engine::init first)",
        CTP_TH_ERR => "thallium error",
        _ => "unknown error",
    };
    let msg = err.msg();
    Err(ThalliumError(if msg.is_empty() {
        format!("{what}: {kind} (rc={rc})")
    } else {
        format!("{what}: {kind}: {msg}")
    }))
}

/// Reject interior NULs before handing a string to C.
fn to_cstring(s: &str, what: &str) -> Result<CString, ThalliumError> {
    CString::new(s).map_err(|_| ThalliumError(format!("{what}: contains an interior NUL byte")))
}

/// Shared `needed`-then-retry idiom for the shim's string out-parameters.
fn read_c_string<F>(what: &str, mut fill: F) -> Result<String, ThalliumError>
where
    F: FnMut(*mut c_char, usize, *mut usize, *mut c_char) -> c_int,
{
    let mut buf = vec![0u8; 256];
    for _ in 0..2 {
        let mut needed: usize = 0;
        let mut err = ErrBuf::new();
        let rc = fill(
            buf.as_mut_ptr() as *mut c_char,
            buf.len(),
            &mut needed,
            err.ptr(),
        );
        th_check(rc, &err, what)?;
        if needed == 0 {
            return Ok(String::new());
        }
        if needed <= buf.len() {
            // `needed` counts the NUL; the Rust string excludes it.
            buf.truncate(needed - 1);
            return Ok(String::from_utf8_lossy(&buf).into_owned());
        }
        buf.resize(needed, 0);
    }
    Err(ThalliumError(format!("{what}: address length is unstable")))
}

// ---------------------------------------------------------------------------
// Engine: the process-singleton tl::engine (ThalliumEngine::Get analog)
// ---------------------------------------------------------------------------

/// Engine mode. CTP creates the engine in [`EngineMode::Server`] even for
/// client-only transports, because both share the one process singleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineMode {
    /// `THALLIUM_CLIENT_MODE`: can issue RPCs, cannot serve them.
    Client,
    /// `THALLIUM_SERVER_MODE`: can also serve RPCs. Requires [`Engine::finalize`]
    /// before exit.
    Server,
}

/// Engine parameters. [`Default`] reproduces `ThalliumEngine::Get()`'s
/// defaults, which that code reads from `CLIO_LBM_THALLIUM_PROTOCOL` and
/// `CLIO_LBM_THALLIUM_RPC_THREADS`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    /// Mercury protocol, e.g. `ofi+tcp;ofi_rxm`, `ofi+verbs;ofi_rxm` (RDMA),
    /// or `na+sm` (intra-node shared memory).
    pub protocol: String,
    pub mode: EngineMode,
    /// Dedicated Mercury progress thread. Without it the calling thread must
    /// drive progress and blocking RPCs deadlock — CTP always enables it.
    pub progress_thread: bool,
    /// Argobots execution streams for RPC handlers. More handlers means more
    /// peers can RDMA-pull in parallel. Clamped to >= 1 by the shim.
    pub rpc_threads: i32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            protocol: "ofi+tcp;ofi_rxm".to_string(),
            mode: EngineMode::Server,
            progress_thread: true,
            rpc_threads: 4,
        }
    }
}

/// Handle to the process-singleton engine. Zero-sized: it is a capability
/// token proving [`Engine::init`] succeeded, so no engine operation can be
/// reached before initialization.
#[derive(Debug, Clone, Copy)]
pub struct Engine {
    /// True if *this* call constructed the engine; false if one already
    /// existed and this config was ignored.
    was_first: bool,
}

impl Engine {
    /// Initialize the process-singleton engine.
    ///
    /// **First init wins**, mirroring `ThalliumEngine::Get()`'s
    /// `std::call_once`: if an engine already exists, `cfg` is ignored and the
    /// existing one is returned with [`Engine::init_was_first`] `== false`.
    pub fn init(cfg: &EngineConfig) -> Result<Engine, ThalliumError> {
        let protocol = to_cstring(&cfg.protocol, "protocol")?;
        let mut err = ErrBuf::new();
        // SAFETY: `protocol` is a live NUL-terminated string for the call;
        // errbuf is a valid ERR_LEN buffer. The shim catches every C++
        // exception, so nothing unwinds through here.
        let rc = unsafe {
            ctp_th_engine_init(
                protocol.as_ptr(),
                c_int::from(cfg.mode == EngineMode::Server),
                c_int::from(cfg.progress_thread),
                cfg.rpc_threads as c_int,
                err.ptr(),
                ERR_LEN,
            )
        };
        th_check(rc, &err, "ctp_th_engine_init")?;
        Ok(Engine {
            was_first: rc != CTP_TH_ALREADY_INIT,
        })
    }

    /// Whether [`Engine::init`] actually built the engine (`true`) or found an
    /// existing one and discarded the supplied config (`false`).
    pub fn init_was_first(&self) -> bool {
        self.was_first
    }

    /// Whether an engine exists and has not been finalized.
    pub fn is_initialized() -> bool {
        // SAFETY: no arguments, no preconditions.
        unsafe { ctp_th_engine_is_initialized() != 0 }
    }

    /// The engine's own address (`engine.self()`) — what a server publishes
    /// for peers to [`Engine::lookup`].
    pub fn self_addr(&self) -> Result<String, ThalliumError> {
        read_c_string("ctp_th_engine_self", |buf, len, needed, errbuf| {
            // SAFETY: buf/len describe one live allocation, `needed` and
            // `errbuf` are valid out-pointers for the duration of the call.
            unsafe { ctp_th_engine_self(buf, len, needed, errbuf, ERR_LEN) }
        })
    }

    /// Resolve a peer address to an [`Endpoint`] (`engine.lookup`).
    ///
    /// The C++ transport treats a failed lookup as "server not alive"; this
    /// returns `Err` for the same condition.
    pub fn lookup(&self, addr: &str) -> Result<Endpoint, ThalliumError> {
        let addr_c = to_cstring(addr, "address")?;
        let mut raw: *mut c_void = std::ptr::null_mut();
        let mut err = ErrBuf::new();
        // SAFETY: `addr_c` is live and NUL-terminated; `raw` is a valid
        // out-pointer that the shim sets only on success.
        let rc = unsafe { ctp_th_endpoint_lookup(addr_c.as_ptr(), &mut raw, err.ptr(), ERR_LEN) };
        th_check(rc, &err, "ctp_th_endpoint_lookup")?;
        if raw.is_null() {
            return Err(ThalliumError(format!(
                "lookup({addr}) returned no endpoint"
            )));
        }
        Ok(Endpoint { raw })
    }

    /// Declare an RPC by name without a handler — the client half of
    /// `engine.define(name)`.
    pub fn define(&self, name: &str) -> Result<Rpc, ThalliumError> {
        let name_c = to_cstring(name, "rpc name")?;
        let mut raw: *mut c_void = std::ptr::null_mut();
        let mut err = ErrBuf::new();
        // SAFETY: `name_c` is live and NUL-terminated; `raw` is a valid
        // out-pointer set only on success.
        let rc = unsafe { ctp_th_rpc_define(name_c.as_ptr(), &mut raw, err.ptr(), ERR_LEN) };
        th_check(rc, &err, "ctp_th_rpc_define")?;
        Ok(Rpc { raw })
    }

    /// Register an RPC handler — the server half, `engine.define(name, cb)`.
    ///
    /// `handler` is invoked on one of the engine's `rpc_threads` Argobots
    /// execution streams (hence `Send + Sync`), receiving the sender's
    /// endpoint, the metadata blob, and the sender's remote bulks. Its return
    /// value becomes the RPC response; the C++ transport uses `0` for success
    /// and an errno such as `EIO`/`ENOMEM` for failure. A panic is caught and
    /// answered with `-1` rather than unwinding into C++.
    ///
    /// Pull any bulk you want **inside** the handler, as the C++ transport
    /// does: the sender blocks on the response, which is not sent until the
    /// handler returns, so its exposed memory stays alive for exactly that
    /// window. A [`RemoteBulk`] borrowed here cannot escape it.
    ///
    /// The boxed closure is **intentionally leaked**: margo keeps the RPC
    /// registered for the engine's lifetime and dropping the returned
    /// [`RpcHandler`] does not deregister it, so freeing the closure would
    /// leave a late delivery calling a dangling pointer. One leak per
    /// registration, and CTP registers once per process.
    pub fn define_handler<F>(&self, name: &str, handler: F) -> Result<RpcHandler, ThalliumError>
    where
        F: Fn(&Endpoint, &[u8], &[RemoteBulk<'_>]) -> i32 + Send + Sync + 'static,
    {
        let name_c = to_cstring(name, "rpc name")?;
        // Double box: `Box<dyn Fn>` is a fat pointer, so box it again to get a
        // thin `*mut c_void` for the shim's `user` argument.
        let boxed: Box<HandlerFn> = Box::new(handler);
        let user = Box::into_raw(Box::new(boxed)) as *mut c_void;
        let mut raw: *mut c_void = std::ptr::null_mut();
        let mut err = ErrBuf::new();
        // SAFETY: `name_c` is live and NUL-terminated; `user` points to a
        // leaked `Box<HandlerFn>` that therefore outlives every delivery, and
        // handler_trampoline is the matching extern "C" reader for it.
        let rc = unsafe {
            ctp_th_rpc_define_handler(
                name_c.as_ptr(),
                handler_trampoline,
                user,
                &mut raw,
                err.ptr(),
                ERR_LEN,
            )
        };
        if let Err(e) = th_check(rc, &err, "ctp_th_rpc_define_handler") {
            // Registration failed, so no delivery can reference `user`: this
            // is the one path where reclaiming the closure is sound.
            // SAFETY: `user` came from Box::into_raw above and the shim did
            // not store it (it returns before registering on failure).
            drop(unsafe { Box::from_raw(user as *mut Box<HandlerFn>) });
            return Err(e);
        }
        Ok(RpcHandler { rpc: Rpc { raw } })
    }

    /// Register `buf` for RDMA reads by a peer (`engine.expose(segs,
    /// read_only)`) — the analog of `ThalliumTransport::Expose`, which only
    /// ever exposes read_only for the send side.
    ///
    /// The returned [`Bulk`] borrows `buf`, so the registration cannot outlive
    /// the memory it publishes; it is deregistered on drop.
    pub fn expose_read<'a>(&self, buf: &'a [u8]) -> Result<Bulk<'a>, ThalliumError> {
        if buf.is_empty() {
            return Err(ThalliumError(
                "expose_read: cannot expose an empty slice".into(),
            ));
        }
        let mut raw: *mut c_void = std::ptr::null_mut();
        let mut err = ErrBuf::new();
        // SAFETY: buf is a live slice borrowed for 'a, and the returned Bulk
        // holds that borrow, so the region stays valid and immovable while
        // registered. The cast to *mut is required by thallium's `expose`
        // signature (`std::vector<std::pair<void*, size_t>>`); the shim passes
        // bulk_mode::read_only, so the region is never written through it.
        let rc = unsafe {
            ctp_th_bulk_expose(
                buf.as_ptr() as *mut c_void,
                buf.len(),
                CTP_TH_BULK_READ_ONLY,
                &mut raw,
                err.ptr(),
                ERR_LEN,
            )
        };
        th_check(rc, &err, "ctp_th_bulk_expose")?;
        Ok(Bulk {
            raw,
            _buf: PhantomData,
        })
    }

    /// Stop the progress thread and RPC streams (`engine.finalize()`).
    ///
    /// A `THALLIUM_SERVER_MODE` engine parks in its destructor until this is
    /// called, so a server process must call it before exit. Idempotent.
    /// Endpoints, RPCs and bulks must not be used afterwards; calls that
    /// follow report [`ThalliumError`] instead of touching a dead engine.
    pub fn finalize(&self) {
        // SAFETY: no arguments; the shim is idempotent and swallows teardown
        // exceptions.
        unsafe { ctp_th_engine_finalize() }
    }
}

// ---------------------------------------------------------------------------
// Endpoint (tl::endpoint)
// ---------------------------------------------------------------------------

/// A resolved peer address. Owned when produced by [`Engine::lookup`]; the
/// handler sees a borrowed one it cannot free.
#[derive(Debug)]
pub struct Endpoint {
    raw: *mut c_void,
}

impl Endpoint {
    /// The endpoint's address string.
    pub fn addr(&self) -> Result<String, ThalliumError> {
        read_c_string("ctp_th_endpoint_to_string", |buf, len, needed, errbuf| {
            // SAFETY: self.raw is a live tl::endpoint for &self; buf/len
            // describe one live allocation and the out-pointers are valid.
            unsafe { ctp_th_endpoint_to_string(self.raw, buf, len, needed, errbuf, ERR_LEN) }
        })
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        // SAFETY: raw came from ctp_th_endpoint_lookup and is freed once —
        // Endpoint is not Clone, and the handler's borrowed endpoint is held
        // in a ManuallyDrop so this never runs for it.
        unsafe { ctp_th_endpoint_free(self.raw) }
    }
}

// ---------------------------------------------------------------------------
// Bulk (tl::bulk)
// ---------------------------------------------------------------------------

/// A locally registered RDMA region, borrowing the buffer it exposes.
///
/// Where the C++ side stows a `new tl::bulk` in `Bulk::desc` for `Send` to
/// `delete` later, this ties registration to the buffer's borrow and
/// deregisters on drop.
#[derive(Debug)]
pub struct Bulk<'a> {
    raw: *mut c_void,
    _buf: PhantomData<&'a [u8]>,
}

impl Bulk<'_> {
    /// Registered length in bytes (`tl::bulk::size()`).
    pub fn len(&self) -> usize {
        // SAFETY: raw is a live tl::bulk for &self.
        unsafe { ctp_th_bulk_size(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for Bulk<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from ctp_th_bulk_expose and is freed once (Bulk is
        // not Clone). Deregistration happens while the borrow is still live.
        unsafe { ctp_th_bulk_free(self.raw) }
    }
}

/// A *peer's* bulk region, as seen inside an RPC handler. Borrowed from the
/// handler's argument vector: it cannot outlive the call, which is exactly the
/// window in which the sender's memory is guaranteed alive.
#[derive(Debug)]
pub struct RemoteBulk<'a> {
    raw: *mut c_void,
    _call: PhantomData<&'a ()>,
}

impl RemoteBulk<'_> {
    /// The remote region's length in bytes.
    pub fn len(&self) -> usize {
        // SAFETY: raw is a live tl::bulk owned by the caller's handler frame.
        unsafe { ctp_th_bulk_size(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Synchronous RDMA pull of the whole region into `dst`
    /// (`remote.on(sender) >> local`).
    ///
    /// `dst` must be at least [`RemoteBulk::len`] bytes; exactly that many are
    /// written. The transient write_only registration of `dst` is created and
    /// released inside the shim, so it never outlives this call.
    pub fn pull_into(&self, from: &Endpoint, dst: &mut [u8]) -> Result<(), ThalliumError> {
        let need = self.len();
        if dst.len() < need {
            return Err(ThalliumError(format!(
                "pull_into: destination is {} bytes, remote bulk is {need}",
                dst.len()
            )));
        }
        let mut err = ErrBuf::new();
        // SAFETY: self.raw and from.raw are live for the call; dst is a live
        // exclusive slice of >= need bytes, so the pull cannot overrun it and
        // no other Rust reference aliases it while the RDMA write lands.
        let rc = unsafe {
            ctp_th_bulk_pull_into(
                self.raw,
                from.raw,
                dst.as_mut_ptr() as *mut c_void,
                dst.len(),
                err.ptr(),
                ERR_LEN,
            )
        };
        th_check(rc, &err, "ctp_th_bulk_pull_into")
    }

    /// [`RemoteBulk::pull_into`] into a fresh `Vec<u8>`.
    ///
    /// This replaces the C++ handler's `malloc(remote.size())`, expose, pull,
    /// then hand-off-a-raw-pointer sequence: the buffer is owned and freed by
    /// Rust, including on the error paths where the C++ code frees by hand.
    pub fn pull_to_vec(&self, from: &Endpoint) -> Result<Vec<u8>, ThalliumError> {
        let mut buf = vec![0u8; self.len()];
        if buf.is_empty() {
            return Ok(buf);
        }
        self.pull_into(from, &mut buf)?;
        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// Rpc (tl::remote_procedure)
// ---------------------------------------------------------------------------

/// A declared RPC. Call it at a peer with [`Rpc::call`].
#[derive(Debug)]
pub struct Rpc {
    raw: *mut c_void,
}

impl Rpc {
    /// Blocking `rpc.on(ep)(meta, bulks)`, as `ThalliumTransport::Send` issues
    /// it. Returns the peer handler's response code (`0` = success in CTP's
    /// convention); an `Err` means the RPC itself failed to complete.
    ///
    /// The peer pulls each bulk inside its handler, so every [`Bulk`] here
    /// stays registered — and its buffer borrowed — until this call returns.
    pub fn call(
        &self,
        ep: &Endpoint,
        meta: &[u8],
        bulks: &[&Bulk<'_>],
    ) -> Result<i32, ThalliumError> {
        let raw_bulks: Vec<*mut c_void> = bulks.iter().map(|b| b.raw).collect();
        let mut out_rc: c_int = 0;
        let mut err = ErrBuf::new();
        // SAFETY: self.raw and ep.raw are live for the call; meta is a live
        // slice of meta.len() bytes (the shim copies it into a std::string);
        // raw_bulks is a live array of raw_bulks.len() live tl::bulk pointers,
        // each kept alive by the &Bulk borrows for the whole call.
        let rc = unsafe {
            ctp_th_rpc_call(
                self.raw,
                ep.raw,
                meta.as_ptr() as *const c_char,
                meta.len(),
                raw_bulks.as_ptr(),
                raw_bulks.len(),
                &mut out_rc,
                err.ptr(),
                ERR_LEN,
            )
        };
        th_check(rc, &err, "ctp_th_rpc_call")?;
        Ok(out_rc as i32)
    }
}

impl Drop for Rpc {
    fn drop(&mut self) {
        // SAFETY: raw came from ctp_th_rpc_define/_define_handler and is freed
        // once (Rpc is not Clone). Freeing the handle does not deregister a
        // handler, so no delivery is left pointing at freed state.
        unsafe { ctp_th_rpc_free(self.raw) }
    }
}

/// A registered RPC handler. Dropping this frees the handle but, matching
/// margo, does **not** deregister the handler: it keeps serving for the
/// engine's lifetime (see [`Engine::define_handler`]).
#[derive(Debug)]
pub struct RpcHandler {
    rpc: Rpc,
}

impl RpcHandler {
    /// The underlying RPC, for calling this same name at a peer.
    pub fn rpc(&self) -> &Rpc {
        &self.rpc
    }
}

// ---------------------------------------------------------------------------
// Handler trampoline
// ---------------------------------------------------------------------------

type HandlerFn = dyn Fn(&Endpoint, &[u8], &[RemoteBulk<'_>]) -> i32 + Send + Sync + 'static;

/// Called by the shim on an Argobots execution stream. Converts the C ABI
/// arguments into borrowed Rust views and returns the handler's response code.
/// Nothing unwinds out of here: a panic becomes `CTP_TH_ERR`.
unsafe extern "C" fn handler_trampoline(
    user: *mut c_void,
    sender_ep: *mut c_void,
    meta: *const c_char,
    meta_len: usize,
    remote_bulks: *const *mut c_void,
    n_bulks: usize,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `user` is the leaked Box<HandlerFn> from define_handler; it
        // is never freed, so this reference is valid for the whole process and
        // the closure is Sync, so concurrent ES deliveries may share it.
        let f: &HandlerFn = unsafe { &**(user as *const Box<HandlerFn>) };

        let meta_slice: &[u8] = if meta_len == 0 || meta.is_null() {
            &[]
        } else {
            // SAFETY: the shim passes std::string::data()/size() of a live
            // string owned by the handler frame, which outlives this call.
            unsafe { std::slice::from_raw_parts(meta as *const u8, meta_len) }
        };

        // Borrowed, not owned: ManuallyDrop stops Endpoint's Drop from freeing
        // an endpoint the shim owns on its own stack.
        let ep = ManuallyDrop::new(Endpoint { raw: sender_ep });

        let bulks: Vec<RemoteBulk<'_>> = (0..n_bulks)
            .map(|i| RemoteBulk {
                // SAFETY: remote_bulks points to n_bulks live tl::bulk* owned
                // by the handler frame for the duration of this call.
                raw: unsafe { *remote_bulks.add(i) },
                _call: PhantomData,
            })
            .collect();

        f(&ep, meta_slice, &bulks)
    }));
    match result {
        Ok(rc) => rc as c_int,
        // The RPC response is the only place to report this: the sender is
        // blocked on it, and unwinding into C++ is undefined.
        Err(_) => CTP_TH_ERR,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Bring up the shared-memory engine, or skip. `na+sm` needs no NIC, so
    /// this exercises the real RPC + bulk path on any machine that has
    /// thallium; it is the same intra-node loopback setup as the C++
    /// `thallium_bench`. Mirrors `driver_available()` in ctp-gpu/src/cuda.rs:
    /// the suite stays green where the dependency is absent.
    fn engine_or_skip() -> Option<Engine> {
        let cfg = EngineConfig {
            protocol: "na+sm".to_string(),
            mode: EngineMode::Server,
            progress_thread: true,
            rpc_threads: 2,
        };
        match Engine::init(&cfg) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("ctp-net/thallium: engine unavailable ({e}); skipping");
                None
            }
        }
    }

    #[test]
    fn config_defaults_match_cxx_transport() {
        // ThalliumEngine::Get(): "ofi+tcp;ofi_rxm", progress thread on,
        // rpc_thread_count 4, THALLIUM_SERVER_MODE.
        let cfg = EngineConfig::default();
        assert_eq!(cfg.protocol, "ofi+tcp;ofi_rxm");
        assert_eq!(cfg.mode, EngineMode::Server);
        assert!(cfg.progress_thread);
        assert_eq!(cfg.rpc_threads, 4);
    }

    #[test]
    fn error_reports_context_and_cxx_message() {
        let e = ThalliumError("lookup failed".to_string());
        assert_eq!(e.to_string(), "ThalliumError: lookup failed");
        let err = ErrBuf::new();
        assert!(th_check(CTP_TH_OK, &err, "op").is_ok());
        // Success codes are nonnegative: ALREADY_INIT must not read as failure.
        assert!(th_check(CTP_TH_ALREADY_INIT, &err, "op").is_ok());
        let msg = th_check(CTP_TH_ERR_NOT_INIT, &err, "op")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("engine not initialized"), "{msg}");
        let msg = th_check(CTP_TH_ERR_INVAL, &err, "op")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("invalid argument"), "{msg}");
    }

    /// Interior NULs must be rejected before reaching C, not truncated.
    #[test]
    fn interior_nul_is_rejected() {
        assert!(to_cstring("na+sm", "protocol").is_ok());
        let e = to_cstring("na\0+sm", "protocol").unwrap_err();
        assert!(e.to_string().contains("interior NUL"), "{e}");

        let cfg = EngineConfig {
            protocol: "na\0+sm".to_string(),
            ..Default::default()
        };
        // Fails on the Rust side, so it needs no engine and runs everywhere.
        assert!(Engine::init(&cfg).is_err());
    }

    /// End-to-end over `na+sm`: register a handler, look up our own address,
    /// expose a payload, call the RPC, and confirm the handler RDMA-pulled the
    /// exact bytes. This is the C++ transport's Send/Recv path minus lightbeam
    /// serialization.
    #[test]
    fn loopback_rpc_pulls_bulk() {
        let Some(engine) = engine_or_skip() else {
            return;
        };

        let payload: Vec<u8> = (0..64 * 1024u32).map(|i| (i % 251) as u8).collect();
        let got: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_meta = Arc::new(Mutex::new(Vec::<u8>::new()));
        let calls = Arc::new(AtomicUsize::new(0));

        let got_h = Arc::clone(&got);
        let meta_h = Arc::clone(&seen_meta);
        let calls_h = Arc::clone(&calls);
        let handler = engine
            .define_handler("ctp_test_pull", move |sender, meta, bulks| {
                calls_h.fetch_add(1, Ordering::SeqCst);
                *meta_h.lock().unwrap() = meta.to_vec();
                let mut out = Vec::new();
                for b in bulks {
                    match b.pull_to_vec(sender) {
                        Ok(v) => out.push(v),
                        // errno, matching the C++ handler's req.respond(EIO).
                        Err(_) => return 5,
                    }
                }
                *got_h.lock().unwrap() = out;
                0
            })
            .unwrap();

        let self_addr = engine.self_addr().unwrap();
        assert!(!self_addr.is_empty());
        let peer = engine.lookup(&self_addr).unwrap();

        let bulk = engine.expose_read(&payload).unwrap();
        assert_eq!(bulk.len(), payload.len());

        let meta = b"lbm-meta".to_vec();
        let rc = handler.rpc().call(&peer, &meta, &[&bulk]).unwrap();
        assert_eq!(rc, 0, "handler responded with an error");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(*seen_meta.lock().unwrap(), meta);
        let pulled = got.lock().unwrap();
        assert_eq!(pulled.len(), 1);
        assert_eq!(pulled[0], payload, "RDMA pull delivered wrong bytes");
        drop(pulled);

        // A zero-bulk RPC is legal: metadata-only messages take this path.
        let rc = handler.rpc().call(&peer, b"", &[]).unwrap();
        assert_eq!(rc, 0);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(got.lock().unwrap().is_empty());

        // A server-mode engine parks in its destructor until finalized.
        engine.finalize();
    }
}
