// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! SYCL GPU backend: the `GpuApi` SYCL branch of
//! `clio_ctp/util/gpu_api.h`, wrapped for Rust.
//!
//! # What is wrapped
//!
//! **SYCL 2020** (Khronos spec, revision 8+), as implemented by **Intel
//! oneAPI DPC++** (`icpx -fsycl`, 2021.4 or newer) or **AdaptiveCpp**
//! (formerly hipSYCL/Open SYCL). Covered: queue create/destroy, USM
//! `malloc_device` / `malloc_shared` / `malloc_host` / `free`, blocking
//! and async `memcpy`, async `memset`, `wait`, GPU device enumeration,
//! and the `get_pointer_type` device-pointer query.
//!
//! # Why there is a shim
//!
//! SYCL is a **C++-only API** with no C ABI: `sycl::queue` is an RAII
//! class, the allocators are overloaded/templated free functions, and
//! errors arrive as thrown `sycl::exception`s. None of that is bindable
//! the way [`crate::hip`] or `ctp-gpu`'s CUDA driver bindings are, so
//! this module talks to `shim/sycl_shim.cc`, which projects the API onto
//! a flat C ABI over opaque handles (modelled on
//! `ctp-coroutine/shim/boost_fiber_shim.cc`). The `extern "C"` block
//! below declares that shim's ABI by hand — no bindgen, no crates.
//!
//! # ABI / version assumptions
//!
//! * **SYCL 2020 only.** `sycl::gpu_selector_v` (a value, per SYCL 2020)
//!   is used, not SYCL 1.2.1's `sycl::gpu_selector{}` class. This
//!   deliberately matches `gpu_api.h`, which is also SYCL 2020.
//! * The stable ABI here is the **shim's**, not SYCL's. The C++ standard
//!   library and SYCL runtime types never cross the boundary — only
//!   `void*`, `size_t`, `int`, and `char` buffers — so the shim absorbs
//!   SYCL's unstable C++ ABI. Status codes must stay in sync with the
//!   `CTP_SYCL_*` defines in the shim.
//! * `shim/sycl_shim.cc` must be compiled by a SYCL compiler and linked
//!   against the SYCL runtime (`-lsycl`). See the note in [Divergences]
//!   about the missing build script.
//!
//! # Divergences from the C++ wrapper
//!
//! 1. **`MemcpyAsync` actually copies.** `GpuApi::MemcpyAsync` has CUDA
//!    and ROCm branches but **no SYCL branch**, so on a SYCL build the
//!    C++ call compiles to nothing and silently copies no data. This
//!    wrapper's [`UsmBuffer::copy_from_host_async`] performs the copy.
//! 2. **No hidden default queue.** `GpuApi::SyclQueue()` is a
//!    function-static `sycl::queue`, created on first use and never
//!    destroyed, that implicitly backs `Malloc`/`Free`/`Memcpy`. Here the
//!    queue is an explicit RAII [`Queue`] the caller owns, and a buffer
//!    borrows the queue it was allocated from — so `sycl::free` is always
//!    paired with the allocating context, which the C++ side only gets
//!    right by virtue of that one global.
//! 3. **Errors are values.** SYCL throws; `gpu_api.h` lets exceptions
//!    escape (and its CUDA/ROCm peers `FATAL` via `*_ERROR_CHECK`). The
//!    shim catches everything at the boundary and returns a status code,
//!    which becomes [`SyclError`]. Constructing a [`Queue`] on a host
//!    with no GPU is an `Err`, not a crash. Nothing here panics or
//!    aborts on a library error.
//! 4. **`in_order` is explicit.** `GpuApi` conflates two queue kinds: the
//!    static `SyclQueue()` is out-of-order, while `CreateStream()` builds
//!    an `in_order` queue to imitate CUDA streams. Both are reachable, as
//!    [`Queue::new`] and [`Queue::new_in_order`].
//! 5. **IPC is not wrapped.** `GetIpcMemHandle`/`OpenIpcMemHandle` have no
//!    SYCL branch in C++ (`GpuIpcMemHandle::sycl_ptr_` is declared but
//!    never assigned), so there is nothing to port.
//!
//! [Divergences]: #divergences-from-the-c-wrapper

use std::ffi::{c_char, c_int, c_void, CStr};

// Status codes — mirror the CTP_SYCL_* defines in shim/sycl_shim.cc.
const CTP_SYCL_OK: c_int = 0;
const CTP_SYCL_ERR_EXCEPTION: c_int = 1;
const CTP_SYCL_ERR_INVALID: c_int = 2;
const CTP_SYCL_ERR_ALLOC: c_int = 3;

extern "C" {
    fn ctp_sycl_device_count(out_count: *mut c_int, err: *mut c_char, errlen: usize) -> c_int;
    fn ctp_sycl_queue_create(
        in_order: c_int,
        out_queue: *mut *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_sycl_queue_destroy(queue: *mut c_void);
    fn ctp_sycl_queue_wait(queue: *mut c_void, err: *mut c_char, errlen: usize) -> c_int;
    fn ctp_sycl_queue_device_name(
        queue: *mut c_void,
        buf: *mut c_char,
        buflen: usize,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_sycl_malloc_device(
        queue: *mut c_void,
        bytes: usize,
        out_ptr: *mut *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_sycl_malloc_shared(
        queue: *mut c_void,
        bytes: usize,
        out_ptr: *mut *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_sycl_malloc_host(
        queue: *mut c_void,
        bytes: usize,
        out_ptr: *mut *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_sycl_free(
        queue: *mut c_void,
        ptr: *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_sycl_memcpy(
        queue: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_sycl_memcpy_async(
        queue: *mut c_void,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_sycl_memset_async(
        queue: *mut c_void,
        dst: *mut c_void,
        value: c_int,
        bytes: usize,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ctp_sycl_is_device_ptr(
        queue: *mut c_void,
        ptr: *const c_void,
        out_is_device: *mut c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A SYCL failure: a caught `sycl::exception`, or a rejected argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyclError(pub String);

impl std::fmt::Display for SyclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SyclError: {}", self.0)
    }
}
impl std::error::Error for SyclError {}

/// Human-readable name for a shim status code.
fn status_name(rc: c_int) -> &'static str {
    match rc {
        CTP_SYCL_ERR_EXCEPTION => "sycl::exception",
        CTP_SYCL_ERR_INVALID => "invalid argument",
        CTP_SYCL_ERR_ALLOC => "allocation failed",
        _ => "unknown error",
    }
}

/// Length of the message buffer handed to the shim. SYCL `what()` strings
/// are short; the shim truncates rather than overruns.
const ERR_LEN: usize = 512;

/// Caller-owned message buffer. The shim keeps no `thread_local` "last
/// error" (project rule, and per-DLL duplication on Windows), so every
/// fallible call carries its own buffer — which also keeps the shim
/// reentrant.
struct ErrBuf([c_char; ERR_LEN]);

impl ErrBuf {
    fn new() -> Self {
        // Zeroed, so it reads back as an empty C string even if the shim
        // never writes to it.
        ErrBuf([0; ERR_LEN])
    }

    fn as_mut_ptr(&mut self) -> *mut c_char {
        self.0.as_mut_ptr()
    }

    fn message(&self) -> String {
        // SAFETY: the buffer is zero-initialized and the shim's SetErr
        // always NUL-terminates within ERR_LEN, so a NUL is present.
        unsafe { CStr::from_ptr(self.0.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Turn a shim status code into a `Result`, attaching the message the
/// shim captured from `what()`.
fn check(rc: c_int, buf: &ErrBuf, what: &str) -> Result<(), SyclError> {
    if rc == CTP_SYCL_OK {
        return Ok(());
    }
    let msg = buf.message();
    if msg.is_empty() {
        Err(SyclError(format!("{what}: {}", status_name(rc))))
    } else {
        Err(SyclError(format!("{what}: {msg}")))
    }
}

// ---------------------------------------------------------------------------
// Device enumeration (GpuApi::GetDeviceCount analog)
// ---------------------------------------------------------------------------

/// Number of SYCL GPU devices across all platforms.
///
/// A host with a SYCL runtime but no GPU reports `Ok(0)`; only a failing
/// platform query is an `Err`.
pub fn device_count() -> Result<usize, SyclError> {
    let mut err = ErrBuf::new();
    let mut n: c_int = 0;
    // SAFETY: both out-pointers are valid for the call; the shim writes
    // at most ERR_LEN bytes into the message buffer.
    let rc = unsafe { ctp_sycl_device_count(&mut n, err.as_mut_ptr(), ERR_LEN) };
    check(rc, &err, "ctp_sycl_device_count")?;
    Ok(n.max(0) as usize)
}

/// True if at least one SYCL GPU device is present. The `driver_available`
/// analog used to skip GPU tests on GPU-less CI runners.
pub fn gpu_available() -> bool {
    matches!(device_count(), Ok(n) if n > 0)
}

// ---------------------------------------------------------------------------
// Queue (GpuApi::SyclQueue / CreateStream / DestroyStream / Synchronize)
// ---------------------------------------------------------------------------

/// An owned `sycl::queue` on the default GPU device.
///
/// Dropping it destroys the underlying queue, so — unlike the C++
/// `GpuApi::SyclQueue()` static — the runtime's queue does not outlive the
/// program's use of it.
///
/// Neither `Send` nor `Sync`: the raw handle field opts out of both. SYCL
/// queues are in fact thread-safe, so this is conservative rather than
/// required; sharing across threads can be added later behind an explicit
/// `unsafe impl` once there is a caller that needs it.
pub struct Queue {
    handle: *mut c_void,
}

impl Queue {
    /// Out-of-order queue on the default GPU device — the `GpuApi::SyclQueue()`
    /// equivalent.
    ///
    /// Returns `Err` (never panics) when no SYCL GPU is available.
    pub fn new() -> Result<Self, SyclError> {
        Self::create(false)
    }

    /// In-order queue: submissions complete in submission order, matching
    /// CUDA stream semantics. The `GpuApi::CreateStream()` equivalent.
    pub fn new_in_order() -> Result<Self, SyclError> {
        Self::create(true)
    }

    fn create(in_order: bool) -> Result<Self, SyclError> {
        let mut err = ErrBuf::new();
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointers; the shim catches every C++
        // exception, so nothing unwinds across this boundary.
        let rc = unsafe {
            ctp_sycl_queue_create(
                c_int::from(in_order),
                &mut handle,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        check(rc, &err, "ctp_sycl_queue_create")?;
        if handle.is_null() {
            return Err(SyclError("ctp_sycl_queue_create: null queue".into()));
        }
        Ok(Self { handle })
    }

    /// Name of the device this queue targets; empty if the query fails.
    pub fn device_name(&self) -> String {
        let mut err = ErrBuf::new();
        let mut buf = [0 as c_char; 256];
        // SAFETY: handle is live; buf/err are valid for their lengths and
        // the shim NUL-terminates within each.
        let rc = unsafe {
            ctp_sycl_queue_device_name(
                self.handle,
                buf.as_mut_ptr(),
                buf.len(),
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        if rc != CTP_SYCL_OK {
            return String::new();
        }
        // SAFETY: on success the shim wrote a NUL-terminated string.
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// Block until all work submitted to this queue completes.
    /// The `GpuApi::Synchronize(stream)` analog.
    ///
    /// The shim uses `wait_and_throw()`, so an async error raised by the
    /// SYCL runtime surfaces here as `Err` instead of being swallowed.
    pub fn wait(&self) -> Result<(), SyclError> {
        let mut err = ErrBuf::new();
        // SAFETY: handle is live for &self.
        let rc = unsafe { ctp_sycl_queue_wait(self.handle, err.as_mut_ptr(), ERR_LEN) };
        check(rc, &err, "ctp_sycl_queue_wait")
    }

    /// `sycl::malloc_device` — device USM. Fast for the GPU, and **not**
    /// host-dereferenceable, so [`UsmBuffer::as_slice`] refuses it; move
    /// data with [`UsmBuffer::copy_from_host`] / [`UsmBuffer::copy_to_host`].
    pub fn malloc_device(&self, len: usize) -> Result<UsmBuffer<'_>, SyclError> {
        // SAFETY (fn ptr choice): each shim allocator has the same
        // signature and returns a pointer owned by this queue's context.
        self.malloc_with(
            ctp_sycl_malloc_device,
            len,
            UsmKind::Device,
            "malloc_device",
        )
    }

    /// `sycl::malloc_shared` — USM shared between host and device.
    /// The `GpuApi::MallocManaged` analog.
    pub fn malloc_shared(&self, len: usize) -> Result<UsmBuffer<'_>, SyclError> {
        self.malloc_with(
            ctp_sycl_malloc_shared,
            len,
            UsmKind::Shared,
            "malloc_shared",
        )
    }

    /// `sycl::malloc_host` — pinned host USM, device-accessible.
    /// The `GpuApi::MallocHost` analog.
    pub fn malloc_host(&self, len: usize) -> Result<UsmBuffer<'_>, SyclError> {
        self.malloc_with(ctp_sycl_malloc_host, len, UsmKind::Host, "malloc_host")
    }

    fn malloc_with(
        &self,
        alloc: unsafe extern "C" fn(
            *mut c_void,
            usize,
            *mut *mut c_void,
            *mut c_char,
            usize,
        ) -> c_int,
        len: usize,
        kind: UsmKind,
        what: &str,
    ) -> Result<UsmBuffer<'_>, SyclError> {
        let mut err = ErrBuf::new();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `alloc` is one of the three shim allocators (all with
        // this exact signature); handle is live; out-pointers are valid.
        // A zero-length request is rounded to 1 byte so the shim never
        // sees a 0-byte USM allocation (unspecified in SYCL 2020).
        let rc = unsafe { alloc(self.handle, len.max(1), &mut ptr, err.as_mut_ptr(), ERR_LEN) };
        check(rc, &err, what)?;
        if ptr.is_null() {
            return Err(SyclError(format!("{what}: null pointer")));
        }
        Ok(UsmBuffer {
            ptr,
            len,
            kind,
            queue: self,
        })
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        // SAFETY: handle came from ctp_sycl_queue_create and is owned
        // uniquely by self; the shim swallows any destructor exception.
        // Buffers borrow the Queue, so all frees have already run.
        unsafe { ctp_sycl_queue_destroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// UsmBuffer (Malloc / MallocManaged / MallocHost / Free / Memcpy)
// ---------------------------------------------------------------------------

/// Which USM flavor an allocation is, and hence whether the host may read
/// it directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsmKind {
    /// `malloc_device`: device-only; the host must not dereference it.
    Device,
    /// `malloc_shared`: migrates between host and device.
    Shared,
    /// `malloc_host`: pinned host memory the device can reach.
    Host,
}

impl UsmKind {
    /// True when the host may dereference this allocation.
    pub fn is_host_accessible(self) -> bool {
        matches!(self, UsmKind::Shared | UsmKind::Host)
    }
}

/// A USM allocation, freed on drop with the queue that created it.
///
/// The `'q` borrow is load-bearing: `sycl::free` must be paired with the
/// context that allocated the pointer, so the buffer cannot outlive — nor
/// be freed by — the wrong [`Queue`].
pub struct UsmBuffer<'q> {
    ptr: *mut c_void,
    len: usize,
    kind: UsmKind,
    queue: &'q Queue,
}

impl UsmBuffer<'_> {
    /// Requested length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Which USM flavor this is.
    pub fn kind(&self) -> UsmKind {
        self.kind
    }

    /// Ask the SYCL runtime whether this really is device USM — the
    /// `GpuApi::IsDevicePointer` analog. Independent of [`kind`]: it
    /// round-trips through `sycl::get_pointer_type`.
    ///
    /// [`kind`]: UsmBuffer::kind
    pub fn is_device_pointer(&self) -> Result<bool, SyclError> {
        let mut err = ErrBuf::new();
        let mut out: c_int = 0;
        // SAFETY: queue handle and ptr are live; out-pointers valid.
        let rc = unsafe {
            ctp_sycl_is_device_ptr(
                self.queue.handle,
                self.ptr as *const c_void,
                &mut out,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        check(rc, &err, "ctp_sycl_is_device_ptr")?;
        Ok(out != 0)
    }

    /// Host view of a `Shared`/`Host` allocation. `Err` for `Device`
    /// memory, which the host cannot dereference — this is what keeps the
    /// safe surface free of raw pointers without hiding the distinction.
    ///
    /// Requires `&self`, and any async transfer borrows the buffer
    /// mutably, so no in-flight copy can be observed through this view.
    pub fn as_slice(&self) -> Result<&[u8], SyclError> {
        if !self.kind.is_host_accessible() {
            return Err(SyclError(
                "as_slice: device USM is not host-accessible".into(),
            ));
        }
        // SAFETY: shared/host USM is host-dereferenceable for `len` bytes
        // (SYCL 2020 §4.8); the allocation is live for 'self; u8 has no
        // alignment requirement beyond 1, which USM always satisfies; and
        // no device work can be in flight (see doc comment).
        Ok(unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) })
    }

    /// Mutable host view of a `Shared`/`Host` allocation. `Err` for
    /// `Device` memory.
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8], SyclError> {
        if !self.kind.is_host_accessible() {
            return Err(SyclError(
                "as_mut_slice: device USM is not host-accessible".into(),
            ));
        }
        // SAFETY: as in as_slice; &mut self guarantees exclusive access.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) })
    }

    /// Blocking host→device copy: submit, then wait. Exactly what
    /// `GpuApi::Memcpy`'s SYCL branch does.
    pub fn copy_from_host<T: Copy>(&mut self, src: &[T]) -> Result<(), SyclError> {
        let bytes = std::mem::size_of_val(src);
        if bytes > self.len {
            return Err(SyclError("copy_from_host: src larger than buffer".into()));
        }
        let mut err = ErrBuf::new();
        // SAFETY: src is a live host slice of `bytes`; dst has capacity
        // (checked); the queue owns dst's context; the call blocks until
        // the copy completes, so src cannot move out from under it.
        let rc = unsafe {
            ctp_sycl_memcpy(
                self.queue.handle,
                self.ptr,
                src.as_ptr() as *const c_void,
                bytes,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        check(rc, &err, "ctp_sycl_memcpy")
    }

    /// Blocking device→host copy.
    pub fn copy_to_host<T: Copy>(&self, dst: &mut [T]) -> Result<(), SyclError> {
        let bytes = std::mem::size_of_val(dst);
        if bytes > self.len {
            return Err(SyclError("copy_to_host: dst larger than buffer".into()));
        }
        let mut err = ErrBuf::new();
        // SAFETY: dst is a live, exclusive host slice of `bytes`; the
        // source allocation is at least that long; the call blocks.
        let rc = unsafe {
            ctp_sycl_memcpy(
                self.queue.handle,
                dst.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                bytes,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        check(rc, &err, "ctp_sycl_memcpy")
    }

    /// Async host→device copy: submits and returns immediately. Completion
    /// is observed with [`Queue::wait`].
    ///
    /// Unlike the C++ `GpuApi::MemcpyAsync`, which has no SYCL branch and
    /// therefore copies nothing, this really submits the copy.
    ///
    /// # Safety
    ///
    /// The copy is still in flight when this returns. Until
    /// [`Queue::wait`] reports completion, the caller must not
    /// * drop, move, or write `src`, or
    /// * read or write this buffer (including via [`as_mut_slice`]), or
    /// * drop this buffer (dropping frees the allocation the DMA targets).
    ///
    /// This is the same contract as raw `sycl::queue::memcpy`, and the
    /// reason the async path is `unsafe` while the blocking path is not: a
    /// borrow-based guard cannot enforce it, since `std::mem::forget` on
    /// the guard would end the borrow with the transfer still running.
    ///
    /// [`as_mut_slice`]: UsmBuffer::as_mut_slice
    pub unsafe fn copy_from_host_async<T: Copy>(&mut self, src: &[T]) -> Result<(), SyclError> {
        let bytes = std::mem::size_of_val(src);
        if bytes > self.len {
            return Err(SyclError(
                "copy_from_host_async: src larger than buffer".into(),
            ));
        }
        let mut err = ErrBuf::new();
        // SAFETY: pointers and length are valid at submission; the caller
        // upholds the liveness contract documented above for the rest of
        // the transfer's lifetime.
        let rc = unsafe {
            ctp_sycl_memcpy_async(
                self.queue.handle,
                self.ptr,
                src.as_ptr() as *const c_void,
                bytes,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        check(rc, &err, "ctp_sycl_memcpy_async")
    }

    /// Async device→host copy: submits and returns immediately.
    ///
    /// # Safety
    ///
    /// As [`copy_from_host_async`]: until [`Queue::wait`] reports
    /// completion the caller must not read, write, move, or drop `dst`,
    /// nor drop this buffer.
    ///
    /// [`copy_from_host_async`]: UsmBuffer::copy_from_host_async
    pub unsafe fn copy_to_host_async<T: Copy>(&self, dst: &mut [T]) -> Result<(), SyclError> {
        let bytes = std::mem::size_of_val(dst);
        if bytes > self.len {
            return Err(SyclError(
                "copy_to_host_async: dst larger than buffer".into(),
            ));
        }
        let mut err = ErrBuf::new();
        // SAFETY: pointers/length valid at submission; caller upholds the
        // liveness contract above.
        let rc = unsafe {
            ctp_sycl_memcpy_async(
                self.queue.handle,
                dst.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                bytes,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        check(rc, &err, "ctp_sycl_memcpy_async")
    }

    /// Async fill with a byte value. The `GpuApi::MemsetAsync` SYCL branch
    /// analog.
    ///
    /// # Safety
    ///
    /// The fill is in flight on return: until [`Queue::wait`] reports
    /// completion the caller must not read or write this buffer, nor drop
    /// it.
    pub unsafe fn memset_async(&mut self, value: u8) -> Result<(), SyclError> {
        let mut err = ErrBuf::new();
        // SAFETY: dst/len describe this live allocation; caller upholds
        // the liveness contract above.
        let rc = unsafe {
            ctp_sycl_memset_async(
                self.queue.handle,
                self.ptr,
                c_int::from(value),
                self.len,
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        check(rc, &err, "ctp_sycl_memset_async")
    }
}

impl Drop for UsmBuffer<'_> {
    fn drop(&mut self) {
        let mut err = ErrBuf::new();
        // SAFETY: ptr came from this queue's allocator and is freed once
        // (self owns it); the queue outlives the buffer by the 'q borrow.
        // Errors cannot be propagated out of drop and a free failure is
        // not actionable, so the status is dropped.
        unsafe {
            ctp_sycl_free(self.queue.handle, self.ptr, err.as_mut_ptr(), ERR_LEN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Tests that need no SYCL runtime -----------------------------------

    #[test]
    fn only_device_usm_is_host_inaccessible() {
        assert!(!UsmKind::Device.is_host_accessible());
        assert!(UsmKind::Shared.is_host_accessible());
        assert!(UsmKind::Host.is_host_accessible());
    }

    #[test]
    fn status_names_cover_the_shim_codes() {
        assert_eq!(status_name(CTP_SYCL_ERR_EXCEPTION), "sycl::exception");
        assert_eq!(status_name(CTP_SYCL_ERR_INVALID), "invalid argument");
        assert_eq!(status_name(CTP_SYCL_ERR_ALLOC), "allocation failed");
        assert_eq!(status_name(42), "unknown error");
    }

    #[test]
    fn ok_status_is_never_an_error() {
        let buf = ErrBuf::new();
        assert!(check(CTP_SYCL_OK, &buf, "noop").is_ok());
    }

    /// A zeroed buffer must read back as an empty string (not garbage, not
    /// UB) — `check` relies on this when the shim reports a code without a
    /// message.
    #[test]
    fn empty_err_buf_falls_back_to_the_status_name() {
        let buf = ErrBuf::new();
        assert_eq!(buf.message(), "");
        let e = check(CTP_SYCL_ERR_ALLOC, &buf, "malloc_device").unwrap_err();
        assert_eq!(e.0, "malloc_device: allocation failed");
        assert_eq!(e.to_string(), "SyclError: malloc_device: allocation failed");
    }

    /// The shim's `what()` text must win over the generic status name, and
    /// truncation must stay in-bounds and NUL-terminated.
    #[test]
    fn err_buf_reads_back_a_shim_message() {
        let mut buf = ErrBuf::new();
        let msg = b"Native API failed\0";
        for (slot, byte) in buf.0.iter_mut().zip(msg.iter()) {
            *slot = *byte as c_char;
        }
        assert_eq!(buf.message(), "Native API failed");
        let e = check(CTP_SYCL_ERR_EXCEPTION, &buf, "ctp_sycl_queue_create").unwrap_err();
        assert_eq!(e.0, "ctp_sycl_queue_create: Native API failed");
    }

    // -- Tests that need a SYCL GPU ----------------------------------------
    //
    // These skip (rather than fail) without one, so the suite stays green
    // on GPU-less runners — the ctp-gpu convention. They have never been
    // executed: no SYCL runtime is installed on the development machine.

    #[test]
    fn device_usm_roundtrip() {
        if !gpu_available() {
            eprintln!("ctp-gpu-backends: no SYCL GPU available; skipping");
            return;
        }
        let q = Queue::new().unwrap();
        eprintln!("ctp-gpu-backends: SYCL device {}", q.device_name());

        const N: usize = 4096;
        let data: Vec<u8> = (0..N).map(|i| (i % 251) as u8).collect();
        let mut buf = q.malloc_device(N).unwrap();
        assert_eq!(buf.len(), N);
        assert!(buf.is_device_pointer().unwrap());
        // Device USM must not be reachable as a host slice.
        assert!(buf.as_slice().is_err());

        buf.copy_from_host(&data).unwrap();
        let mut back = vec![0u8; N];
        buf.copy_to_host(&mut back).unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn shared_usm_is_host_accessible() {
        if !gpu_available() {
            eprintln!("ctp-gpu-backends: no SYCL GPU available; skipping");
            return;
        }
        let q = Queue::new().unwrap();
        const N: usize = 1024;
        let mut buf = q.malloc_shared(N).unwrap();
        // Shared USM is written directly by the host, no memcpy needed.
        for (i, b) in buf.as_mut_slice().unwrap().iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        assert_eq!(buf.as_slice().unwrap()[7], 7);
        // Shared USM is not *device* USM per get_pointer_type.
        assert!(!buf.is_device_pointer().unwrap());
    }

    /// The divergence that matters: the C++ `MemcpyAsync` has no SYCL
    /// branch and copies nothing. This asserts the data really lands.
    #[test]
    fn async_memcpy_actually_copies() {
        if !gpu_available() {
            eprintln!("ctp-gpu-backends: no SYCL GPU available; skipping");
            return;
        }
        let q = Queue::new_in_order().unwrap();
        const N: usize = 2048;
        let data: Vec<u8> = (0..N).map(|i| (i % 97) as u8).collect();
        let mut buf = q.malloc_device(N).unwrap();
        let mut back = vec![0u8; N];
        // SAFETY: `data`, `back`, and `buf` all outlive the wait() below,
        // and nothing touches them until the transfers have completed.
        unsafe {
            buf.copy_from_host_async(&data).unwrap();
            buf.copy_to_host_async(&mut back).unwrap();
        }
        q.wait().unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn queue_wait_on_idle_queue_is_ok() {
        if !gpu_available() {
            eprintln!("ctp-gpu-backends: no SYCL GPU available; skipping");
            return;
        }
        let q = Queue::new().unwrap();
        q.wait().unwrap();
    }
}
