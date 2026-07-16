// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! NIXL (NVIDIA Inference Xfer Library) bindings, hand-rolled and
//! dependency-free, plus the safe wrappers over them.
//!
//! # What is wrapped
//!
//! Upstream: <https://github.com/ai-dynamo/nixl> (Apache-2.0). CTP's C++ side
//! (`include/clio_ctp/lightbeam/nixl_transport.h`, `include/clio_ctp/io/nixl_io.h`)
//! includes `<nixl.h>` and drives the **C++** `nixlAgent` class directly. That
//! class is not FFI-callable from Rust: its API takes `std::string`,
//! `nixl_b_params_t` (an `std::map`), and template-instantiated descriptor
//! lists, none of which have a stable C ABI.
//!
//! NIXL ships a C ABI for exactly this reason: `src/bindings/rust/wrapper.cpp`
//! is compiled by `src/bindings/meson.build` into an installed shared library
//! `libnixl_capi.so` exporting the `nixl_capi_*` symbols (the same surface the
//! upstream `nixl-sys` crate binds). This module binds **that** library. Every
//! `nixl_capi_*` entry point below is a thin C++ shim over the same `nixlAgent`
//! method CTP's C++ calls, so the semantics match one-for-one.
//!
//! # Version / ABI assumptions
//!
//! * Declarations transcribed from `src/bindings/rust/wrapper.h` @
//!   `0797ede5c477d5832202659bd4dea9316b67952d` (NIXL `main`, meson
//!   `version: '1.4.0'`; newest tagged release at time of writing: v1.3.1).
//!   The `nixl_capi_*` surface has been ABI-stable across the 1.x line, but
//!   NIXL makes no formal ABI guarantee — re-check `wrapper.h` when bumping.
//! * `nixl_capi_status_t` has negative enumerators, so it is `int` (`c_int`).
//!   `nixl_capi_mem_type_t` / `nixl_capi_xfer_op_t` / `nixl_capi_thread_sync_t`
//!   have only non-negative enumerators, so GCC/Clang give them `unsigned int`
//!   (`c_uint`). Both are 32 bits and occupy the same argument slot on every
//!   ABI NIXL supports, so the distinction is cosmetic — it is honored anyway.
//! * C `bool` (`<stdbool.h>`) is one byte and ABI-compatible with Rust `bool`.
//! * `nixl_capi_agent_config_t` is `#[repr(C)]`-mirrored field-for-field.
//! * NIXL is Linux-only in practice (its POSIX/UCX plugins use POSIX I/O).
//!   Nothing here is `cfg(unix)`-gated so the module still type-checks
//!   elsewhere, but only the tests that touch a real file descriptor are.
//!
//! # Linking
//!
//! `ctp-net` has no build script, so the extern block carries its own
//! `#[link(name = "nixl_capi")]`. The library lives under NIXL's install
//! prefix (default `/opt/nvidia/nvda_nixl`, e.g. `$NIXL_PREFIX/lib64` or
//! `$NIXL_PREFIX/lib/x86_64-linux-gnu`); point the linker at it with
//! `RUSTFLAGS="-L $NIXL_PREFIX/lib64"` and the loader with `LD_LIBRARY_PATH`.
//! Nothing is linked unless the `nixl` feature is on.
//!
//! # Divergence from the C++ wrapper
//!
//! 1. **Error fidelity.** `NixlTransport::RunXfer` returns the raw
//!    `nixl_status_t`. The C API collapses every backend failure to
//!    `NIXL_CAPI_ERROR_BACKEND` (-2), so [`NixlError::status`] reports the
//!    coarse C code, not NIXL's specific `NIXL_ERR_*`. Nothing in this module
//!    can recover the finer code — it is destroyed inside `wrapper.cpp`.
//! 2. **No DRAM→DRAM `memcpy` path.** `NixlTransport::SendToMem` performs a
//!    host `std::memcpy` because the POSIX backend cannot do DRAM→DRAM. That
//!    is not a NIXL operation, so it is deliberately not reproduced here;
//!    this module exposes only real NIXL transfers. Callers wanting the
//!    loopback semantics use `slice::copy_from_slice` themselves.
//! 3. **Request teardown.** The C++ code calls `releaseXferReq` only. Through
//!    the C API that leaks the handle wrapper: `nixl_capi_release_xfer_req`
//!    frees the NIXL request and NULLs it, and `nixl_capi_destroy_xfer_req`
//!    then frees the handle struct (and refuses, with
//!    `NIXL_CAPI_ERROR_INVALID_STATE`, if release has not run). [`XferRequest`]
//!    does both in `Drop`.
//! 4. **`Drop` waits.** NIXL may DMA into a posted request's local buffers, so
//!    dropping an in-flight [`XferRequest`] first polls to completion (the same
//!    unbounded `while (st == NIXL_IN_PROG)` loop `RunXfer` uses) before
//!    releasing. This keeps the safe API sound at the cost of a blocking drop.
//! 5. **Descriptor shadowing.** `RunXfer` rebuilds a registration list by
//!    reading descriptors back out of the transfer list (`local_descs[i]`).
//!    The C API has no descriptor read-back, so [`XferDescList`] keeps a
//!    Rust-side copy of what was added and [`Agent::run_xfer`] builds the
//!    registration lists from it.
//! 6. **`nixl_capi_create_configured_agent` ignores `num_workers`** in this
//!    NIXL revision (the field exists in the struct but `wrapper.cpp` never
//!    reads it). [`AgentConfig::num_workers`] is carried for ABI fidelity.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Raw C API (wrapper.h)
// ---------------------------------------------------------------------------

type CapiStatus = c_int;
type CapiAgent = *mut c_void;
type CapiParams = *mut c_void;
type CapiBackendH = *mut c_void;
type CapiOptArgsH = *mut c_void;
type CapiStringList = *mut c_void;
type CapiXferDlist = *mut c_void;
type CapiRegDlist = *mut c_void;
type CapiXferReq = *mut c_void;

const NIXL_CAPI_SUCCESS: CapiStatus = 0;
const NIXL_CAPI_IN_PROG: CapiStatus = 1;
const NIXL_CAPI_ERROR_INVALID_PARAM: CapiStatus = -1;
const NIXL_CAPI_ERROR_BACKEND: CapiStatus = -2;
const NIXL_CAPI_ERROR_INVALID_STATE: CapiStatus = -3;
const NIXL_CAPI_ERROR_EXCEPTION: CapiStatus = -4;
const NIXL_CAPI_ERROR_NO_TELEMETRY: CapiStatus = -5;

const NIXL_CAPI_MEM_DRAM: c_uint = 0;
const NIXL_CAPI_MEM_VRAM: c_uint = 1;
const NIXL_CAPI_MEM_BLOCK: c_uint = 2;
const NIXL_CAPI_MEM_OBJECT: c_uint = 3;
const NIXL_CAPI_MEM_FILE: c_uint = 4;
const NIXL_CAPI_MEM_UNKNOWN: c_uint = 5;

const NIXL_CAPI_XFER_OP_READ: c_uint = 0;
const NIXL_CAPI_XFER_OP_WRITE: c_uint = 1;

const NIXL_CAPI_THREAD_SYNC_NONE: c_uint = 0;
const NIXL_CAPI_THREAD_SYNC_STRICT: c_uint = 1;
const NIXL_CAPI_THREAD_SYNC_RW: c_uint = 2;

/// Mirror of `nixl_capi_agent_config_t`. Field order and types are load-bearing.
#[repr(C)]
struct CapiAgentConfig {
    enable_prog_thread: bool,
    enable_listen_thread: bool,
    listen_port: c_int,
    thread_sync: c_uint,
    num_workers: c_uint,
    pthr_delay_us: u64,
    lthr_delay_us: u64,
    capture_telemetry: bool,
}

#[link(name = "nixl_capi")]
extern "C" {
    fn nixl_capi_create_configured_agent(
        name: *const c_char,
        cfg: *const CapiAgentConfig,
        agent: *mut CapiAgent,
    ) -> CapiStatus;
    fn nixl_capi_destroy_agent(agent: CapiAgent) -> CapiStatus;

    fn nixl_capi_get_available_plugins(
        agent: CapiAgent,
        plugins: *mut CapiStringList,
    ) -> CapiStatus;
    fn nixl_capi_destroy_string_list(list: CapiStringList) -> CapiStatus;
    fn nixl_capi_string_list_size(list: CapiStringList, size: *mut usize) -> CapiStatus;
    fn nixl_capi_string_list_get(
        list: CapiStringList,
        index: usize,
        str_: *mut *const c_char,
    ) -> CapiStatus;

    fn nixl_capi_create_params(params: *mut CapiParams) -> CapiStatus;
    fn nixl_capi_params_add(
        params: CapiParams,
        key: *const c_char,
        value: *const c_char,
    ) -> CapiStatus;
    fn nixl_capi_destroy_params(params: CapiParams) -> CapiStatus;

    fn nixl_capi_create_backend(
        agent: CapiAgent,
        plugin_name: *const c_char,
        params: CapiParams,
        backend: *mut CapiBackendH,
    ) -> CapiStatus;
    fn nixl_capi_destroy_backend(backend: CapiBackendH) -> CapiStatus;

    fn nixl_capi_create_opt_args(args: *mut CapiOptArgsH) -> CapiStatus;
    fn nixl_capi_destroy_opt_args(args: CapiOptArgsH) -> CapiStatus;
    fn nixl_capi_opt_args_add_backend(args: CapiOptArgsH, backend: CapiBackendH) -> CapiStatus;

    fn nixl_capi_register_mem(
        agent: CapiAgent,
        dlist: CapiRegDlist,
        opt_args: CapiOptArgsH,
    ) -> CapiStatus;
    fn nixl_capi_deregister_mem(
        agent: CapiAgent,
        dlist: CapiRegDlist,
        opt_args: CapiOptArgsH,
    ) -> CapiStatus;

    fn nixl_capi_create_xfer_dlist(mem_type: c_uint, dlist: *mut CapiXferDlist) -> CapiStatus;
    fn nixl_capi_destroy_xfer_dlist(dlist: CapiXferDlist) -> CapiStatus;
    fn nixl_capi_xfer_dlist_get_type(dlist: CapiXferDlist, mem_type: *mut c_uint) -> CapiStatus;
    fn nixl_capi_xfer_dlist_add_desc(
        dlist: CapiXferDlist,
        addr: usize,
        len: usize,
        dev_id: u64,
    ) -> CapiStatus;
    fn nixl_capi_xfer_dlist_desc_count(dlist: CapiXferDlist, count: *mut usize) -> CapiStatus;

    fn nixl_capi_create_reg_dlist(mem_type: c_uint, dlist: *mut CapiRegDlist) -> CapiStatus;
    fn nixl_capi_destroy_reg_dlist(dlist: CapiRegDlist) -> CapiStatus;
    fn nixl_capi_reg_dlist_add_desc(
        dlist: CapiRegDlist,
        addr: usize,
        len: usize,
        dev_id: u64,
        metadata: *const c_void,
        metadata_len: usize,
    ) -> CapiStatus;
    fn nixl_capi_reg_dlist_desc_count(dlist: CapiRegDlist, count: *mut usize) -> CapiStatus;

    fn nixl_capi_create_xfer_req(
        agent: CapiAgent,
        operation: c_uint,
        local_descs: CapiXferDlist,
        remote_descs: CapiXferDlist,
        remote_agent: *const c_char,
        req_hndl: *mut CapiXferReq,
        opt_args: CapiOptArgsH,
    ) -> CapiStatus;
    fn nixl_capi_post_xfer_req(
        agent: CapiAgent,
        req_hndl: CapiXferReq,
        opt_args: CapiOptArgsH,
    ) -> CapiStatus;
    fn nixl_capi_get_xfer_status(agent: CapiAgent, req_hndl: CapiXferReq) -> CapiStatus;
    fn nixl_capi_release_xfer_req(agent: CapiAgent, req: CapiXferReq) -> CapiStatus;
    fn nixl_capi_destroy_xfer_req(req: CapiXferReq) -> CapiStatus;

    fn nixl_capi_is_stub() -> bool;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure of a NIXL call, or of argument marshalling on the way in.
///
/// Library errors never panic; they surface here. Note divergence (1) in the
/// module docs: `status` is NIXL's coarse C status, not its internal
/// `nixl_status_t`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixlError {
    status: i32,
    context: String,
}

impl NixlError {
    // `CapiStatus` is `c_int`, i.e. `i32` on every target Rust supports, so
    // the status is stored as a plain i32 and needs no conversion.
    fn new(status: CapiStatus, context: impl Into<String>) -> Self {
        Self {
            status,
            context: context.into(),
        }
    }

    /// Bad argument detected on the Rust side (never reached the C API).
    fn invalid(context: impl Into<String>) -> Self {
        Self::new(NIXL_CAPI_ERROR_INVALID_PARAM, context)
    }

    /// The raw `nixl_capi_status_t` value.
    pub fn status(&self) -> i32 {
        self.status
    }

    /// The `NIXL_CAPI_*` spelling of [`NixlError::status`].
    pub fn status_name(&self) -> &'static str {
        status_name(self.status)
    }

    /// Which call failed.
    pub fn context(&self) -> &str {
        &self.context
    }
}

impl std::fmt::Display for NixlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NixlError: {}: {} ({})",
            self.context,
            self.status_name(),
            self.status
        )
    }
}
impl std::error::Error for NixlError {}

/// `NIXL_CAPI_*` name for a raw status code.
pub fn status_name(status: i32) -> &'static str {
    match status {
        NIXL_CAPI_SUCCESS => "NIXL_CAPI_SUCCESS",
        NIXL_CAPI_IN_PROG => "NIXL_CAPI_IN_PROG",
        NIXL_CAPI_ERROR_INVALID_PARAM => "NIXL_CAPI_ERROR_INVALID_PARAM",
        NIXL_CAPI_ERROR_BACKEND => "NIXL_CAPI_ERROR_BACKEND",
        NIXL_CAPI_ERROR_INVALID_STATE => "NIXL_CAPI_ERROR_INVALID_STATE",
        NIXL_CAPI_ERROR_EXCEPTION => "NIXL_CAPI_ERROR_EXCEPTION",
        NIXL_CAPI_ERROR_NO_TELEMETRY => "NIXL_CAPI_ERROR_NO_TELEMETRY",
        _ => "NIXL_CAPI_ERROR_UNKNOWN",
    }
}

fn capi_check(rc: CapiStatus, what: &str) -> Result<(), NixlError> {
    if rc == NIXL_CAPI_SUCCESS {
        Ok(())
    } else {
        Err(NixlError::new(rc, what.to_string()))
    }
}

fn to_cstring(s: &str, what: &str) -> Result<CString, NixlError> {
    CString::new(s).map_err(|_| NixlError::invalid(format!("{what}: interior NUL byte")))
}

/// True if the linked `libnixl_capi` is the LD_PRELOAD-able stub build
/// (`stubs.cpp`) rather than a real NIXL. Stub symbols resolve NIXL lazily at
/// runtime, so this distinguishes "NIXL absent" from "NIXL misconfigured".
pub fn library_is_stub() -> bool {
    // SAFETY: nixl_capi_is_stub takes no arguments and only returns a constant.
    unsafe { nixl_capi_is_stub() }
}

// ---------------------------------------------------------------------------
// Plain-data enums
// ---------------------------------------------------------------------------

/// `nixl_capi_mem_type_t` / NIXL's `nixl_mem_t` (`DRAM_SEG` … `UNKNOWN_SEG`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemType {
    Dram,
    Vram,
    Block,
    Object,
    File,
    Unknown,
}

impl MemType {
    fn as_raw(self) -> c_uint {
        match self {
            MemType::Dram => NIXL_CAPI_MEM_DRAM,
            MemType::Vram => NIXL_CAPI_MEM_VRAM,
            MemType::Block => NIXL_CAPI_MEM_BLOCK,
            MemType::Object => NIXL_CAPI_MEM_OBJECT,
            MemType::File => NIXL_CAPI_MEM_FILE,
            MemType::Unknown => NIXL_CAPI_MEM_UNKNOWN,
        }
    }

    fn from_raw(raw: c_uint) -> MemType {
        match raw {
            NIXL_CAPI_MEM_DRAM => MemType::Dram,
            NIXL_CAPI_MEM_VRAM => MemType::Vram,
            NIXL_CAPI_MEM_BLOCK => MemType::Block,
            NIXL_CAPI_MEM_OBJECT => MemType::Object,
            NIXL_CAPI_MEM_FILE => MemType::File,
            _ => MemType::Unknown,
        }
    }
}

/// `nixl_capi_xfer_op_t`. `Write` pushes local→remote, `Read` pulls remote→local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XferOp {
    Read,
    Write,
}

impl XferOp {
    fn as_raw(self) -> c_uint {
        match self {
            XferOp::Read => NIXL_CAPI_XFER_OP_READ,
            XferOp::Write => NIXL_CAPI_XFER_OP_WRITE,
        }
    }
}

/// `nixl_capi_thread_sync_t` (`nixlAgentConfig::syncMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadSync {
    None,
    Strict,
    ReadWrite,
}

impl ThreadSync {
    fn as_raw(self) -> c_uint {
        match self {
            ThreadSync::None => NIXL_CAPI_THREAD_SYNC_NONE,
            ThreadSync::Strict => NIXL_CAPI_THREAD_SYNC_STRICT,
            ThreadSync::ReadWrite => NIXL_CAPI_THREAD_SYNC_RW,
        }
    }
}

/// Result of polling a posted transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    Complete,
    InProgress,
}

// ---------------------------------------------------------------------------
// Agent configuration
// ---------------------------------------------------------------------------

/// Safe mirror of `nixlAgentConfig` (as reachable through
/// `nixl_capi_agent_config_t`).
///
/// [`AgentConfig::default`] reproduces what CTP's `NixlTransport` and
/// `NixlAsyncIO` construct: no progress thread, no listen thread, RW sync mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    /// `nixlAgentConfig::useProgThread`.
    pub prog_thread: bool,
    /// `nixlAgentConfig::useListenThread`.
    pub listen_thread: bool,
    /// `nixlAgentConfig::listenPort` (only meaningful with `listen_thread`).
    pub listen_port: u16,
    /// `nixlAgentConfig::syncMode`.
    pub thread_sync: ThreadSync,
    /// Carried for ABI fidelity; ignored by NIXL 1.x's `wrapper.cpp`.
    pub num_workers: u32,
    /// `nixlAgentConfig::pthrDelay`, microseconds.
    pub pthr_delay_us: u64,
    /// `nixlAgentConfig::lthrDelay`, microseconds.
    pub lthr_delay_us: u64,
    /// `nixlAgentConfig::captureTelemetry`.
    pub capture_telemetry: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        // Mirrors nixl_transport.h / nixl_io.h:
        //   cfg.useProgThread = false;
        //   cfg.useListenThread = false;
        //   cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        Self {
            prog_thread: false,
            listen_thread: false,
            listen_port: 0,
            thread_sync: ThreadSync::ReadWrite,
            num_workers: 0,
            pthr_delay_us: 0,
            lthr_delay_us: 0,
            capture_telemetry: false,
        }
    }
}

impl AgentConfig {
    fn as_raw(&self) -> CapiAgentConfig {
        CapiAgentConfig {
            enable_prog_thread: self.prog_thread,
            enable_listen_thread: self.listen_thread,
            listen_port: c_int::from(self.listen_port),
            thread_sync: self.thread_sync.as_raw(),
            num_workers: self.num_workers as c_uint,
            pthr_delay_us: self.pthr_delay_us,
            lthr_delay_us: self.lthr_delay_us,
            capture_telemetry: self.capture_telemetry,
        }
    }
}

/// `NixlTransport::MakeAgentName` analog: a process-unique agent name.
pub fn make_agent_name() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("lbm_nixl_agent_{id}")
}

// ---------------------------------------------------------------------------
// Descriptor lists
//
// The C API is add-only: descriptors can be counted but never read back, so
// each list keeps a Rust-side shadow of what was added (see divergence 5).
// ---------------------------------------------------------------------------

/// One `nixlBasicDesc`: address (or file offset), length, device id (or fd).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Desc {
    addr: usize,
    len: usize,
    dev_id: u64,
}

/// `nixl_xfer_dlist_t` — the descriptors naming one side of a transfer.
///
/// `'buf` is the lifetime of any host buffer added via
/// [`XferDescList::add_buffer`], which keeps those buffers borrowed for as
/// long as the list (and anything derived from it) lives.
pub struct XferDescList<'buf> {
    raw: CapiXferDlist,
    mem_type: MemType,
    descs: Vec<Desc>,
    _buf: PhantomData<&'buf mut [u8]>,
}

impl<'buf> XferDescList<'buf> {
    /// `nixl_xfer_dlist_t(mem_type)`.
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        let mut raw: CapiXferDlist = std::ptr::null_mut();
        // SAFETY: valid out-pointer; mem_type is a valid enumerator.
        capi_check(
            unsafe { nixl_capi_create_xfer_dlist(mem_type.as_raw(), &mut raw) },
            "nixl_capi_create_xfer_dlist",
        )?;
        Ok(Self {
            raw,
            mem_type,
            descs: Vec::new(),
            _buf: PhantomData,
        })
    }

    /// Segment type this list was created with.
    pub fn mem_type(&self) -> MemType {
        self.mem_type
    }

    /// Segment type as NIXL itself reports it (`nixl_xfer_dlist_t::getType()`,
    /// which `RunXfer` uses to type the remote registration list). Always
    /// agrees with [`XferDescList::mem_type`] unless the enum mapping drifts.
    pub fn queried_mem_type(&self) -> Result<MemType, NixlError> {
        let mut raw: c_uint = 0;
        // SAFETY: live list + valid out-pointer.
        capi_check(
            unsafe { nixl_capi_xfer_dlist_get_type(self.raw, &mut raw) },
            "nixl_capi_xfer_dlist_get_type",
        )?;
        Ok(MemType::from_raw(raw))
    }

    /// Describe a host buffer (`DRAM_SEG`, `devId=0` — as CTP's C++ does).
    ///
    /// The buffer is borrowed mutably for `'buf`: NIXL may read from it (an
    /// [`XferOp::Write`] source) or DMA into it (an [`XferOp::Read`]
    /// destination), so exclusive access is required either way.
    pub fn add_buffer(&mut self, buf: &'buf mut [u8]) -> Result<(), NixlError> {
        if self.mem_type != MemType::Dram {
            return Err(NixlError::invalid(format!(
                "add_buffer: host buffers need MemType::Dram, list is {:?}",
                self.mem_type
            )));
        }
        let (addr, len) = (buf.as_mut_ptr() as usize, buf.len());
        // SAFETY: `buf` is a live slice borrowed for 'buf, which outlives this
        // list; addr/len describe exactly it.
        unsafe { self.add_desc(addr, len, 0) }
    }

    /// Describe a byte range of an open file (`FILE_SEG`: `addr` is the file
    /// offset, `devId` is the file descriptor), as CTP's `SendToFile` and
    /// `NixlAsyncIO::Write` do.
    ///
    /// A bogus `fd` is a NIXL-level error, not memory unsafety, so this is safe.
    pub fn add_file_extent(&mut self, fd: i32, offset: u64, len: usize) -> Result<(), NixlError> {
        if self.mem_type != MemType::File {
            return Err(NixlError::invalid(format!(
                "add_file_extent: needs MemType::File, list is {:?}",
                self.mem_type
            )));
        }
        let addr = usize::try_from(offset)
            .map_err(|_| NixlError::invalid("add_file_extent: offset exceeds uintptr_t"))?;
        // SAFETY: FILE_SEG descriptors are (offset, len, fd) triples; NIXL does
        // not dereference `addr` — it seeks to it.
        unsafe { self.add_desc(addr, len, u64::from(fd as u32)) }
    }

    /// Add a raw descriptor (device pointers, pre-registered memory, …).
    ///
    /// # Safety
    /// For memory segments (`DRAM_SEG`/`VRAM_SEG`), `addr`/`len` must describe
    /// a region that stays allocated — and untouched by other code — for as
    /// long as this list and any transfer built from it live; NIXL may read or
    /// write it. Same contract as handing the address to `nixlBasicDesc`.
    pub unsafe fn add_desc(
        &mut self,
        addr: usize,
        len: usize,
        dev_id: u64,
    ) -> Result<(), NixlError> {
        // SAFETY: `self.raw` is a live list from new(); the caller upholds the
        // meaning of addr/len/dev_id.
        capi_check(
            unsafe { nixl_capi_xfer_dlist_add_desc(self.raw, addr, len, dev_id) },
            "nixl_capi_xfer_dlist_add_desc",
        )?;
        self.descs.push(Desc { addr, len, dev_id });
        Ok(())
    }

    /// Descriptor count as NIXL sees it (`descCount()`).
    pub fn desc_count(&self) -> Result<usize, NixlError> {
        let mut count = 0usize;
        // SAFETY: live list + valid out-pointer.
        capi_check(
            unsafe { nixl_capi_xfer_dlist_desc_count(self.raw, &mut count) },
            "nixl_capi_xfer_dlist_desc_count",
        )?;
        Ok(count)
    }

    /// True when no descriptor has been added.
    pub fn is_empty(&self) -> bool {
        self.descs.is_empty()
    }

    /// Registration list covering these descriptors — `RunXfer`'s
    /// `nixl_reg_dlist_t local_reg(...); for (...) addDesc(...)`, built from
    /// the shadow because the C API cannot read descriptors back.
    fn to_reg_list(&self) -> Result<RegDescList<'buf>, NixlError> {
        let mut reg = RegDescList::new(self.mem_type)?;
        for d in &self.descs {
            // SAFETY: each descriptor was validated when added to `self`, whose
            // 'buf borrow outlives the returned list (to_reg_list is private and
            // every caller keeps `self` borrowed for at least as long).
            unsafe { reg.add_desc(d.addr, d.len, d.dev_id, &[])? };
        }
        Ok(reg)
    }
}

impl Drop for XferDescList<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from nixl_capi_create_xfer_dlist and is freed once.
        unsafe { nixl_capi_destroy_xfer_dlist(self.raw) };
    }
}

/// `nixl_reg_dlist_t` — descriptors for `registerMem`/`deregisterMem`.
pub struct RegDescList<'buf> {
    raw: CapiRegDlist,
    mem_type: MemType,
    descs: Vec<Desc>,
    _buf: PhantomData<&'buf mut [u8]>,
}

impl<'buf> RegDescList<'buf> {
    /// `nixl_reg_dlist_t(mem_type)`.
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        let mut raw: CapiRegDlist = std::ptr::null_mut();
        // SAFETY: valid out-pointer; mem_type is a valid enumerator.
        capi_check(
            unsafe { nixl_capi_create_reg_dlist(mem_type.as_raw(), &mut raw) },
            "nixl_capi_create_reg_dlist",
        )?;
        Ok(Self {
            raw,
            mem_type,
            descs: Vec::new(),
            _buf: PhantomData,
        })
    }

    /// Segment type this list was created with.
    pub fn mem_type(&self) -> MemType {
        self.mem_type
    }

    /// Register a host buffer (`DRAM_SEG`, `devId=0`), borrowed for `'buf`.
    pub fn add_buffer(&mut self, buf: &'buf mut [u8]) -> Result<(), NixlError> {
        if self.mem_type != MemType::Dram {
            return Err(NixlError::invalid(format!(
                "add_buffer: host buffers need MemType::Dram, list is {:?}",
                self.mem_type
            )));
        }
        let (addr, len) = (buf.as_mut_ptr() as usize, buf.len());
        // SAFETY: `buf` is live for 'buf, which outlives this list.
        unsafe { self.add_desc(addr, len, 0, &[]) }
    }

    /// Register a byte range of an open file (`FILE_SEG`).
    pub fn add_file_extent(&mut self, fd: i32, offset: u64, len: usize) -> Result<(), NixlError> {
        if self.mem_type != MemType::File {
            return Err(NixlError::invalid(format!(
                "add_file_extent: needs MemType::File, list is {:?}",
                self.mem_type
            )));
        }
        let addr = usize::try_from(offset)
            .map_err(|_| NixlError::invalid("add_file_extent: offset exceeds uintptr_t"))?;
        // SAFETY: FILE_SEG descriptors carry no dereferenceable address.
        unsafe { self.add_desc(addr, len, u64::from(fd as u32), &[]) }
    }

    /// Add a raw `nixlBlobDesc` (descriptor plus opaque metadata blob).
    ///
    /// # Safety
    /// Same contract as [`XferDescList::add_desc`].
    pub unsafe fn add_desc(
        &mut self,
        addr: usize,
        len: usize,
        dev_id: u64,
        metadata: &[u8],
    ) -> Result<(), NixlError> {
        let (meta_ptr, meta_len) = if metadata.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (metadata.as_ptr() as *const c_void, metadata.len())
        };
        // SAFETY: live list; metadata (if any) is a valid slice that
        // wrapper.cpp copies into an std::string before returning; the caller
        // upholds the meaning of addr/len/dev_id.
        capi_check(
            unsafe {
                nixl_capi_reg_dlist_add_desc(self.raw, addr, len, dev_id, meta_ptr, meta_len)
            },
            "nixl_capi_reg_dlist_add_desc",
        )?;
        self.descs.push(Desc { addr, len, dev_id });
        Ok(())
    }

    /// Descriptor count as NIXL sees it.
    pub fn desc_count(&self) -> Result<usize, NixlError> {
        let mut count = 0usize;
        // SAFETY: live list + valid out-pointer.
        capi_check(
            unsafe { nixl_capi_reg_dlist_desc_count(self.raw, &mut count) },
            "nixl_capi_reg_dlist_desc_count",
        )?;
        Ok(count)
    }

    /// True when no descriptor has been added.
    pub fn is_empty(&self) -> bool {
        self.descs.is_empty()
    }
}

impl Drop for RegDescList<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from nixl_capi_create_reg_dlist and is freed once.
        unsafe { nixl_capi_destroy_reg_dlist(self.raw) };
    }
}

// ---------------------------------------------------------------------------
// Optional args (nixl_opt_args_t)
// ---------------------------------------------------------------------------

struct OptArgs {
    raw: CapiOptArgsH,
}

impl OptArgs {
    /// `nixl_opt_args_t{ .backends = { backend } }`.
    fn with_backend(backend: &Backend<'_>) -> Result<Self, NixlError> {
        let mut raw: CapiOptArgsH = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        capi_check(
            unsafe { nixl_capi_create_opt_args(&mut raw) },
            "nixl_capi_create_opt_args",
        )?;
        let args = Self { raw };
        // SAFETY: both handles are live; wrapper.cpp only push_backs the
        // backend pointer, which the agent keeps alive.
        capi_check(
            unsafe { nixl_capi_opt_args_add_backend(args.raw, backend.raw) },
            "nixl_capi_opt_args_add_backend",
        )?;
        Ok(args)
    }
}

impl Drop for OptArgs {
    fn drop(&mut self) {
        // SAFETY: raw came from nixl_capi_create_opt_args and is freed once.
        unsafe { nixl_capi_destroy_opt_args(self.raw) };
    }
}

/// `nullptr` opt-args means "no backend hint", exactly as CTP's C++ passes.
fn opt_raw(args: Option<&OptArgs>) -> CapiOptArgsH {
    args.map_or(std::ptr::null_mut(), |a| a.raw)
}

fn make_opt(backend: Option<&Backend<'_>>) -> Result<Option<OptArgs>, NixlError> {
    match backend {
        Some(b) => Ok(Some(OptArgs::with_backend(b)?)),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Backend (nixlBackendH)
// ---------------------------------------------------------------------------

/// A backend plugin instantiated on an [`Agent`] (`agent->createBackend`).
///
/// The agent owns the underlying `nixlBackendH`; this is a borrowed handle,
/// so it cannot outlive the agent.
pub struct Backend<'a> {
    raw: CapiBackendH,
    _agent: PhantomData<&'a Agent>,
}

impl Drop for Backend<'_> {
    fn drop(&mut self) {
        // SAFETY: frees only the C handle wrapper — wrapper.cpp's
        // nixl_capi_destroy_backend does `delete backend;` and never touches
        // the nixlBackendH, which the agent owns and destroys.
        unsafe { nixl_capi_destroy_backend(self.raw) };
    }
}

// ---------------------------------------------------------------------------
// Memory registration guard
// ---------------------------------------------------------------------------

/// RAII guard for `agent->registerMem` / `deregisterMem`.
///
/// `RunXfer` deregisters on every exit path; dropping this does the same.
pub struct Registration<'a, 'buf> {
    agent: &'a Agent,
    list: &'a RegDescList<'buf>,
    opt: Option<OptArgs>,
}

impl Drop for Registration<'_, '_> {
    fn drop(&mut self) {
        // SAFETY: agent and list are alive (both borrowed for 'a); this mirrors
        // the registerMem that created the guard. Drop cannot report errors.
        unsafe {
            nixl_capi_deregister_mem(self.agent.raw, self.list.raw, opt_raw(self.opt.as_ref()))
        };
    }
}

// ---------------------------------------------------------------------------
// Transfer request (nixlXferReqH)
// ---------------------------------------------------------------------------

/// A created (and possibly posted) transfer request.
///
/// `'a` borrows the agent and the descriptor lists it was built from; `'buf`
/// is those lists' buffer lifetime, so any host memory NIXL may touch stays
/// alive at least as long as the request.
pub struct XferRequest<'a, 'buf> {
    agent: &'a Agent,
    raw: CapiXferReq,
    _lists: PhantomData<&'a XferDescList<'buf>>,
}

impl XferRequest<'_, '_> {
    /// `agent->postXferReq`. Returns [`Progress::InProgress`] when NIXL took
    /// the request but has not finished it.
    pub fn post(&self, backend: Option<&Backend<'_>>) -> Result<Progress, NixlError> {
        let opt = make_opt(backend)?;
        // SAFETY: agent and request are live; `opt` (if any) outlives the call.
        let rc =
            unsafe { nixl_capi_post_xfer_req(self.agent.raw, self.raw, opt_raw(opt.as_ref())) };
        match rc {
            NIXL_CAPI_SUCCESS => Ok(Progress::Complete),
            NIXL_CAPI_IN_PROG => Ok(Progress::InProgress),
            _ => Err(NixlError::new(rc, "nixl_capi_post_xfer_req")),
        }
    }

    /// `agent->getXferStatus`.
    pub fn status(&self) -> Result<Progress, NixlError> {
        // SAFETY: agent and request are live.
        let rc = unsafe { nixl_capi_get_xfer_status(self.agent.raw, self.raw) };
        match rc {
            NIXL_CAPI_SUCCESS => Ok(Progress::Complete),
            NIXL_CAPI_IN_PROG => Ok(Progress::InProgress),
            _ => Err(NixlError::new(rc, "nixl_capi_get_xfer_status")),
        }
    }

    /// Poll to completion — `RunXfer`'s `while (st == NIXL_IN_PROG)` loop.
    pub fn wait(&self) -> Result<(), NixlError> {
        loop {
            match self.status()? {
                Progress::Complete => return Ok(()),
                Progress::InProgress => std::hint::spin_loop(),
            }
        }
    }
}

impl Drop for XferRequest<'_, '_> {
    fn drop(&mut self) {
        // Wait out an in-flight transfer before releasing: NIXL may still be
        // reading or writing the borrowed buffers (divergence 4). A failing
        // status ends the wait — there is nothing left to complete.
        while let Ok(Progress::InProgress) = self.status() {
            std::hint::spin_loop();
        }
        // SAFETY: agent and request are live. release frees the nixlXferReqH
        // and NULLs it inside the handle; destroy then frees the handle itself,
        // and refuses (leaking it, not double-freeing) if release failed.
        unsafe {
            nixl_capi_release_xfer_req(self.agent.raw, self.raw);
            nixl_capi_destroy_xfer_req(self.raw);
        }
    }
}

// ---------------------------------------------------------------------------
// Agent (nixlAgent)
// ---------------------------------------------------------------------------

/// A NIXL agent — the `std::unique_ptr<nixlAgent>` CTP's `NixlTransport` and
/// `NixlAsyncIO` each own one of.
///
/// Not `Send`/`Sync`: NIXL's thread-safety depends on
/// [`AgentConfig::thread_sync`] and is not asserted here.
pub struct Agent {
    raw: CapiAgent,
    name: String,
}

impl Agent {
    /// Create an agent with CTP's configuration ([`AgentConfig::default`]).
    pub fn new(name: &str) -> Result<Self, NixlError> {
        Self::with_config(name, &AgentConfig::default())
    }

    /// Create an agent with an auto-generated unique name, as
    /// `NixlTransport` does when constructed with an empty `agent_name`.
    pub fn with_generated_name() -> Result<Self, NixlError> {
        Self::new(&make_agent_name())
    }

    /// `new nixlAgent(name, cfg)`.
    pub fn with_config(name: &str, config: &AgentConfig) -> Result<Self, NixlError> {
        if name.is_empty() {
            return Err(NixlError::invalid("Agent::with_config: empty agent name"));
        }
        let c_name = to_cstring(name, "Agent::with_config")?;
        let raw_cfg = config.as_raw();
        let mut raw: CapiAgent = std::ptr::null_mut();
        // SAFETY: `c_name` is NUL-terminated and outlives the call (wrapper.cpp
        // copies it into an std::string); `raw_cfg` is a valid #[repr(C)]
        // mirror read through a const pointer; the out-pointer is valid.
        capi_check(
            unsafe { nixl_capi_create_configured_agent(c_name.as_ptr(), &raw_cfg, &mut raw) },
            "nixl_capi_create_configured_agent",
        )?;
        Ok(Self {
            raw,
            name: name.to_string(),
        })
    }

    /// The agent name — `NixlTransport::GetAddress`'s logical address.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `agent->getAvailPlugins`.
    pub fn available_plugins(&self) -> Result<Vec<String>, NixlError> {
        let mut list: CapiStringList = std::ptr::null_mut();
        // SAFETY: live agent + valid out-pointer.
        capi_check(
            unsafe { nixl_capi_get_available_plugins(self.raw, &mut list) },
            "nixl_capi_get_available_plugins",
        )?;

        // Every path below must reach the destroy_string_list at the end.
        let collect = || -> Result<Vec<String>, NixlError> {
            let mut size = 0usize;
            // SAFETY: live list + valid out-pointer.
            capi_check(
                unsafe { nixl_capi_string_list_size(list, &mut size) },
                "nixl_capi_string_list_size",
            )?;
            let mut out = Vec::with_capacity(size);
            for i in 0..size {
                let mut s: *const c_char = std::ptr::null();
                // SAFETY: i < size; valid out-pointer.
                capi_check(
                    unsafe { nixl_capi_string_list_get(list, i, &mut s) },
                    "nixl_capi_string_list_get",
                )?;
                if s.is_null() {
                    return Err(NixlError::new(
                        NIXL_CAPI_ERROR_BACKEND,
                        "nixl_capi_string_list_get: null string",
                    ));
                }
                // SAFETY: `s` points at an std::string inside `list`: NUL-
                // terminated and alive until destroy_string_list below. The
                // bytes are copied into an owned String here.
                out.push(unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned());
            }
            Ok(out)
        };
        let result = collect();
        // SAFETY: `list` came from get_available_plugins and is freed exactly
        // once, after the last read of its strings.
        unsafe { nixl_capi_destroy_string_list(list) };
        result
    }

    /// `agent->createBackend(plugin, params, backend)`.
    ///
    /// `params` are the `nixl_b_params_t` key/value pairs; CTP's C++ passes an
    /// empty map for the POSIX backend.
    pub fn create_backend(
        &self,
        plugin: &str,
        params: &[(&str, &str)],
    ) -> Result<Backend<'_>, NixlError> {
        let c_plugin = to_cstring(plugin, "create_backend: plugin name")?;

        // nixl_capi_create_backend rejects a null params handle, so even an
        // empty map is an allocated object.
        let mut raw_params: CapiParams = std::ptr::null_mut();
        // SAFETY: valid out-pointer.
        capi_check(
            unsafe { nixl_capi_create_params(&mut raw_params) },
            "nixl_capi_create_params",
        )?;

        let build = || -> Result<CapiBackendH, NixlError> {
            for (k, v) in params {
                let ck = to_cstring(k, "create_backend: param key")?;
                let cv = to_cstring(v, "create_backend: param value")?;
                // SAFETY: live params handle; both strings are NUL-terminated
                // and outlive the call (wrapper.cpp copies into std::string).
                capi_check(
                    unsafe { nixl_capi_params_add(raw_params, ck.as_ptr(), cv.as_ptr()) },
                    "nixl_capi_params_add",
                )?;
            }
            let mut backend: CapiBackendH = std::ptr::null_mut();
            // SAFETY: live agent and params; `c_plugin` is NUL-terminated and
            // outlives the call; the out-pointer is valid.
            capi_check(
                unsafe {
                    nixl_capi_create_backend(self.raw, c_plugin.as_ptr(), raw_params, &mut backend)
                },
                "nixl_capi_create_backend",
            )?;
            Ok(backend)
        };
        let built = build();
        // SAFETY: `raw_params` came from create_params; createBackend takes the
        // map by const reference and copies what it keeps, so freeing is safe.
        unsafe { nixl_capi_destroy_params(raw_params) };

        Ok(Backend {
            raw: built?,
            _agent: PhantomData,
        })
    }

    /// `agent->registerMem(dlist)`; the returned guard deregisters on drop.
    pub fn register_mem<'a, 'buf>(
        &'a self,
        list: &'a RegDescList<'buf>,
        backend: Option<&Backend<'a>>,
    ) -> Result<Registration<'a, 'buf>, NixlError> {
        let opt = make_opt(backend)?;
        // SAFETY: live agent and list; `opt` (if any) outlives the call and is
        // moved into the guard, which uses it again on deregister.
        capi_check(
            unsafe { nixl_capi_register_mem(self.raw, list.raw, opt_raw(opt.as_ref())) },
            "nixl_capi_register_mem",
        )?;
        Ok(Registration {
            agent: self,
            list,
            opt,
        })
    }

    /// `agent->createXferReq(op, local, remote, remote_agent, req)`.
    ///
    /// `remote_agent` of `None` means this agent — the loopback peer `RunXfer`
    /// picks with `remote_name.empty() ? agent_name_ : remote_name`. The
    /// memory behind `local`/`remote` must already be registered.
    pub fn create_xfer_req<'a, 'buf>(
        &'a self,
        op: XferOp,
        local: &'a XferDescList<'buf>,
        remote: &'a XferDescList<'buf>,
        remote_agent: Option<&str>,
        backend: Option<&Backend<'a>>,
    ) -> Result<XferRequest<'a, 'buf>, NixlError> {
        let peer = remote_agent.unwrap_or(&self.name);
        let c_peer = to_cstring(peer, "create_xfer_req: remote agent")?;
        let opt = make_opt(backend)?;
        let mut raw: CapiXferReq = std::ptr::null_mut();
        // SAFETY: live agent and both lists (borrowed for 'a); `c_peer` is
        // NUL-terminated and copied by wrapper.cpp; out-pointer is valid.
        capi_check(
            unsafe {
                nixl_capi_create_xfer_req(
                    self.raw,
                    op.as_raw(),
                    local.raw,
                    remote.raw,
                    c_peer.as_ptr(),
                    &mut raw,
                    opt_raw(opt.as_ref()),
                )
            },
            "nixl_capi_create_xfer_req",
        )?;
        Ok(XferRequest {
            agent: self,
            raw,
            _lists: PhantomData,
        })
    }

    /// `NixlTransport::RunXfer`: register both sides, create and post the
    /// request, poll to completion, then release and deregister.
    ///
    /// Cleanup order matches the C++ (request released before its memory is
    /// deregistered): `req` is declared last, so it drops first.
    pub fn run_xfer(
        &self,
        op: XferOp,
        local: &XferDescList<'_>,
        remote: &XferDescList<'_>,
        remote_agent: Option<&str>,
        backend: Option<&Backend<'_>>,
    ) -> Result<(), NixlError> {
        let local_reg = local.to_reg_list()?;
        let remote_reg = remote.to_reg_list()?;

        let _local_guard = self.register_mem(&local_reg, backend)?;
        let _remote_guard = self.register_mem(&remote_reg, backend)?;

        let req = self.create_xfer_req(op, local, remote, remote_agent, backend)?;
        req.post(backend)?;
        req.wait()
    }

    /// Blocking DRAM→FILE write of one buffer: `NixlTransport::SendToFile` /
    /// `NixlAsyncIO::Write`.
    ///
    /// `src` is only ever read by NIXL (that is what [`XferOp::Write`] means)
    /// and stays borrowed for the whole transfer, so a shared slice is sound.
    pub fn write_at(
        &self,
        backend: Option<&Backend<'_>>,
        src: &[u8],
        fd: i32,
        offset: u64,
    ) -> Result<(), NixlError> {
        let mut local = XferDescList::new(MemType::Dram)?;
        // SAFETY: `src` is a live slice borrowed for this call; the transfer
        // completes before returning (run_xfer polls to completion, and even a
        // dropped request waits), and XferOp::Write only reads local memory —
        // so NIXL never writes through this shared borrow.
        unsafe { local.add_desc(src.as_ptr() as usize, src.len(), 0)? };

        let mut remote = XferDescList::new(MemType::File)?;
        remote.add_file_extent(fd, offset, src.len())?;

        self.run_xfer(XferOp::Write, &local, &remote, None, backend)
    }

    /// Blocking FILE→DRAM read into one buffer: `NixlAsyncIO::Read`.
    pub fn read_at(
        &self,
        backend: Option<&Backend<'_>>,
        dst: &mut [u8],
        fd: i32,
        offset: u64,
    ) -> Result<(), NixlError> {
        let len = dst.len();
        let mut local = XferDescList::new(MemType::Dram)?;
        // SAFETY: `dst` is a live, exclusively borrowed slice; the transfer
        // completes before this function returns, so NIXL's writes land while
        // the borrow is still held and nothing else can observe them mid-flight.
        unsafe { local.add_desc(dst.as_mut_ptr() as usize, len, 0)? };

        let mut remote = XferDescList::new(MemType::File)?;
        remote.add_file_extent(fd, offset, len)?;

        self.run_xfer(XferOp::Read, &local, &remote, None, backend)
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        // SAFETY: raw came from nixl_capi_create_configured_agent and is freed
        // once; backends and requests borrowed from it cannot outlive it.
        unsafe { nixl_capi_destroy_agent(self.raw) };
    }
}

/// Hand-written rather than derived: the derive would print the raw agent
/// pointer, which is noise in logs and a needless address leak.
impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent").field("name", &self.name).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Why the NIXL-touching tests cannot run here, if they cannot. Mirrors
    /// ctp-gpu's `driver_available()` skip so the suite stays green when the
    /// library is present but has no usable plugins.
    fn skip_reason() -> Option<String> {
        if library_is_stub() {
            return Some("libnixl_capi is the stub build (no real NIXL loaded)".to_string());
        }
        None
    }

    #[test]
    fn agent_names_are_unique() {
        let a = make_agent_name();
        let b = make_agent_name();
        assert_ne!(a, b);
        assert!(a.starts_with("lbm_nixl_agent_"));
        assert!(b.starts_with("lbm_nixl_agent_"));
    }

    /// The C++ transports build exactly this config; the wrapper must not drift.
    #[test]
    fn default_config_matches_cpp_transport() {
        let cfg = AgentConfig::default();
        assert!(!cfg.prog_thread);
        assert!(!cfg.listen_thread);
        assert_eq!(cfg.thread_sync, ThreadSync::ReadWrite);

        let raw = cfg.as_raw();
        assert!(!raw.enable_prog_thread);
        assert!(!raw.enable_listen_thread);
        assert_eq!(raw.thread_sync, NIXL_CAPI_THREAD_SYNC_RW);
    }

    #[test]
    fn status_names_cover_the_capi_enum() {
        assert_eq!(status_name(0), "NIXL_CAPI_SUCCESS");
        assert_eq!(status_name(1), "NIXL_CAPI_IN_PROG");
        assert_eq!(status_name(-1), "NIXL_CAPI_ERROR_INVALID_PARAM");
        assert_eq!(status_name(-2), "NIXL_CAPI_ERROR_BACKEND");
        assert_eq!(status_name(-3), "NIXL_CAPI_ERROR_INVALID_STATE");
        assert_eq!(status_name(-4), "NIXL_CAPI_ERROR_EXCEPTION");
        assert_eq!(status_name(-5), "NIXL_CAPI_ERROR_NO_TELEMETRY");
        assert_eq!(status_name(42), "NIXL_CAPI_ERROR_UNKNOWN");

        let e = NixlError::new(NIXL_CAPI_ERROR_BACKEND, "createBackend");
        assert_eq!(e.status(), -2);
        assert_eq!(e.context(), "createBackend");
        assert_eq!(
            e.to_string(),
            "NixlError: createBackend: NIXL_CAPI_ERROR_BACKEND (-2)"
        );
    }

    #[test]
    fn mem_type_roundtrips_through_the_c_enum() {
        for mt in [
            MemType::Dram,
            MemType::Vram,
            MemType::Block,
            MemType::Object,
            MemType::File,
            MemType::Unknown,
        ] {
            assert_eq!(MemType::from_raw(mt.as_raw()), mt);
        }
        // NIXL's nixl_mem_t ordering, which the C enum casts straight into.
        assert_eq!(MemType::Dram.as_raw(), 0);
        assert_eq!(MemType::File.as_raw(), 4);
        assert_eq!(MemType::from_raw(99), MemType::Unknown);
    }

    /// Bad marshalling must be an error, never a panic and never a C call.
    #[test]
    fn invalid_arguments_are_errors_not_panics() {
        let e = Agent::new("").unwrap_err();
        assert_eq!(e.status(), NIXL_CAPI_ERROR_INVALID_PARAM);
        let e = Agent::new("has\0nul").unwrap_err();
        assert_eq!(e.status(), NIXL_CAPI_ERROR_INVALID_PARAM);
    }

    #[test]
    fn agent_lists_plugins_and_creates_posix_backend() {
        if let Some(why) = skip_reason() {
            eprintln!("ctp-net/nixl: skipping ({why})");
            return;
        }
        let agent = match Agent::with_generated_name() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("ctp-net/nixl: no usable NIXL agent ({e}); skipping");
                return;
            }
        };
        assert!(agent.name().starts_with("lbm_nixl_agent_"));

        let plugins = match agent.available_plugins() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("ctp-net/nixl: getAvailPlugins failed ({e}); skipping");
                return;
            }
        };
        eprintln!("ctp-net/nixl: plugins = {plugins:?}");
        if !plugins.iter().any(|p| p == "POSIX") {
            eprintln!("ctp-net/nixl: POSIX plugin not built; skipping");
            return;
        }
        // The call NixlTransport's ctor makes: createBackend("POSIX", {}).
        let backend = agent.create_backend("POSIX", &[]);
        assert!(backend.is_ok(), "createBackend(POSIX): {:?}", backend.err());
    }

    /// End-to-end proof of the wrapped path: DRAM→FILE then FILE→DRAM over
    /// NIXL's POSIX backend — the transfer CTP's SendToFile / NixlAsyncIO do.
    #[cfg(unix)]
    #[test]
    fn dram_file_roundtrip_over_posix_backend() {
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;

        if let Some(why) = skip_reason() {
            eprintln!("ctp-net/nixl: skipping ({why})");
            return;
        }
        let agent = match Agent::with_generated_name() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("ctp-net/nixl: no usable NIXL agent ({e}); skipping");
                return;
            }
        };
        let backend = match agent.create_backend("POSIX", &[]) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("ctp-net/nixl: no POSIX backend ({e}); skipping");
                return;
            }
        };

        const N: usize = 4096;
        let path = std::env::temp_dir().join(format!("{}.bin", make_agent_name()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .expect("open temp file");
        file.set_len(N as u64).expect("size temp file");
        let fd = file.as_raw_fd();

        let src: Vec<u8> = (0..N).map(|i| (i % 251) as u8).collect();
        agent
            .write_at(Some(&backend), &src, fd, 0)
            .expect("NIXL DRAM→FILE write");

        let mut dst = vec![0u8; N];
        agent
            .read_at(Some(&backend), &mut dst, fd, 0)
            .expect("NIXL FILE→DRAM read");

        assert_eq!(src, dst);
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    /// Segment-type guards are wrapper-side checks: they must reject a
    /// mismatched descriptor rather than hand NIXL a nonsense one.
    #[test]
    fn descriptor_lists_reject_mismatched_segments() {
        if let Some(why) = skip_reason() {
            eprintln!("ctp-net/nixl: skipping ({why})");
            return;
        }
        // Buffers are declared before the lists that borrow them: a list holds
        // its buffers borrowed for as long as it lives, so the reverse order
        // would (correctly) not compile.
        let mut buf = [0u8; 16];
        let mut buf2 = [0u8; 16];

        let mut file_list = match XferDescList::new(MemType::File) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("ctp-net/nixl: cannot build dlist ({e}); skipping");
                return;
            }
        };
        let e = file_list.add_buffer(&mut buf).unwrap_err();
        assert_eq!(e.status(), NIXL_CAPI_ERROR_INVALID_PARAM);

        let mut dram_list = XferDescList::new(MemType::Dram).expect("dram dlist");
        let e = dram_list.add_file_extent(3, 0, 16).unwrap_err();
        assert_eq!(e.status(), NIXL_CAPI_ERROR_INVALID_PARAM);

        dram_list.add_buffer(&mut buf2).expect("add DRAM buffer");
        assert_eq!(dram_list.desc_count().expect("desc_count"), 1);
        assert!(!dram_list.is_empty());
        assert_eq!(dram_list.mem_type(), MemType::Dram);
        // NIXL's own view of the segment type must match our enum mapping.
        assert_eq!(
            dram_list.queried_mem_type().expect("getType"),
            MemType::Dram
        );
        assert_eq!(
            file_list.queried_mem_type().expect("getType"),
            MemType::File
        );
    }
}
