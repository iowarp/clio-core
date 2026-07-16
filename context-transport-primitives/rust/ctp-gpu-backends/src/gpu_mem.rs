// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! GPU memory backends — Rust port of the C++ `ctp::ipc::GpuMalloc`
//! (`memory/backend/gpu_malloc.h`) and `ctp::ipc::GpuShmMmap`
//! (`memory/backend/gpu_shm_mmap.h`), the GPU-side analogs of the CPU
//! shared-memory backend wrapped by `ctp-memory`'s `SharedMemBackend`.
//!
//! # Wrapped library
//!
//! The **NVIDIA CUDA Driver API** — `nvcuda.dll` (Windows) /
//! `libcuda.so.1` (Linux) — the same library `ctp-gpu/src/cuda.rs` wraps,
//! and the library behind the `GpuApi::Malloc` / `MallocManaged` / `Memcpy`
//! / `GetIpcMemHandle` calls the two C++ backends make (`util/gpu_api.h`).
//! Declarations are hand-rolled here (no bindgen, no external crates), in
//! the style of `ctp-gpu/src/cuda.rs`.
//!
//! # Version / ABI assumptions
//!
//! * **Driver API 2.0 entry points**: the `_v2` exports (`cuMemAlloc_v2`,
//!   `cuMemFree_v2`, `cuMemcpy{HtoD,DtoH}_v2`, `cuMemsetD8_v2`,
//!   `cuIpcOpenMemHandle_v2`, `cuDevicePrimaryCtxRelease_v2`) — the names
//!   the CUDA headers' `#define`s resolve to. CUDA 11+; validated here on
//!   CUDA 13.3 / driver 591.86 / RTX 5080 (`sm_120`).
//! * `CUresult` is `int` and `CUDA_SUCCESS == 0`; `CUdeviceptr` is 64-bit
//!   (`unsigned long long`); `CUdevice` is `int`; `size_t` is 64-bit. This
//!   module is therefore **64-bit only**, which matches CTP's targets.
//! * `CUipcMemHandle` is an opaque 64-byte POD (`CU_IPC_HANDLE_SIZE`).
//! * [`MemoryBackendHeader`] mirrors the C++ struct byte-for-byte (40
//!   bytes: `MemoryBackendId{u32,u32}`, `bitfield64_t` == a single `u64`,
//!   two `size_t`, one `int` + tail padding), and [`MemoryBackendId`] is
//!   byte-compatible with `ctp_memory::AllocatorId` per MEMORY_DESIGN.md's
//!   frozen-pointer-ABI pillar.
//! * Device attribute ids are driver constants: 75/76 (compute capability),
//!   83 (`MANAGED_MEMORY`), 89 (`CONCURRENT_MANAGED_ACCESS`).
//!
//! # Divergences from the C++ backends
//!
//! 1. **`GpuShmMmap` uses managed memory; the C++ uses pinned host memory.**
//!    Issue #756 scopes this module to the unified-memory path, so
//!    [`GpuShmMmap::create`] allocates with `cuMemAllocManaged`
//!    (`cudaMallocManaged`-equivalent). The C++ class *documents*
//!    `cudaMallocManaged` but its code actually calls `GpuApi::MallocHost`
//!    (`cudaMallocHost`), having found managed memory does not give
//!    coherent GPU→CPU visibility for `atomicExch_system` on PCIe-only
//!    hosts. **Consequence:** this backend carries bulk segment data with
//!    explicit synchronization; it does *not* inherit the C++ backend's
//!    lock-free CPU↔GPU atomic-handoff property. See divergence 6 — that
//!    C++ decision is corroborated by measurement here.
//! 2. **Symbols are resolved at runtime**, not at link time. `ctp-gpu` has
//!    a `build.rs` emitting `cargo:rustc-link-lib=cuda`; `ctp-gpu-backends`
//!    has none, and this workflow owns only this file — so the driver is
//!    loaded via `LoadLibraryA`/`dlopen` + `GetProcAddress`/`dlsym`. The
//!    signatures are identical to a link-time binding; the bonus is that a
//!    driver-less host gets a [`GpuMemError`] instead of a link failure.
//! 3. **No leak on drop.** C++ `GpuMalloc::shm_init` clears `flags_` and
//!    then only calls `SetOwner()`, never setting
//!    `MEMORY_BACKEND_INITIALIZED` — so `~GpuMalloc` → `_Destroy()` hits its
//!    `if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) return;` guard and the
//!    device allocation is never freed. (`GpuShmMmap::shm_init` *does* set
//!    the bit.) Here freeing is RAII via [`Drop`] and cannot be skipped; the
//!    persisted header records `OWNED | INITIALIZED`.
//! 4. **Header size is 64 KiB, not 4 KiB.** Both C++ GPU headers' layout
//!    comments say `[4KB MemoryBackendHeader | Data]`, but the shared
//!    constant `kBackendHeaderSize` in `memory_backend.h` is `65536` (sized
//!    for Windows' 64 KiB `MapViewOfFile` allocation granularity). This port
//!    follows the constant — see [`BACKEND_HEADER_SIZE`].
//! 5. **URL attach is absent, and IPC is Linux-only *by construction*.** C++
//!    `shm_attach(url)` exists only to log an error and return `false` on both
//!    backends, so it is expressed here by its absence;
//!    [`GpuMalloc::attach_ipc`] is the real cross-process path. That path, and
//!    [`GpuMalloc::ipc_handle`], are compiled to an immediate error on
//!    Windows and their symbols are **not bound there at all** — a divergence
//!    forced by measurement, not preference. CUDA IPC is documented Linux-only,
//!    yet `nvcuda.dll` exports `cuIpcGetMemHandle` *and returns
//!    `CUDA_SUCCESS`*; the resulting handle is unusable, and the call leaves
//!    state behind that faults a **concurrent `cuMemFree` on another thread**
//!    (reproduced: SIGSEGV inside `cuMemFree_v2` in 9 of 12 runs at
//!    `--test-threads=2`, 0 of 12 with the call removed). Since the C++
//!    `GpuMalloc` reaches IPC through `GpuApi::GetIpcMemHandle` unconditionally,
//!    the same hazard is latent there on Windows.
//! 6. **Host access to a managed heap is capability-gated** — the one place
//!    where physics, not taste, dictates the API. The allocation always uses
//!    `CU_MEM_ATTACH_GLOBAL`, so the heap genuinely is GPU-accessible
//!    unified memory (issue #756's requirement). But CUDA only permits the
//!    CPU to dereference an `ATTACH_GLOBAL` managed pointer while device work
//!    is in flight when the device reports `CONCURRENT_MANAGED_ACCESS`.
//!    Measured on the validation host (RTX 5080, Windows/WDDM): that
//!    attribute is **0**, and a CPU write racing another thread's device work
//!    faults the process (`STATUS_IN_PAGE_ERROR`). Safe Rust must not be able
//!    to do that, so [`GpuShmMmap::heap`]/[`heap_mut`](GpuShmMmap::heap_mut)
//!    hand out a slice only where the hardware permits concurrent access and
//!    return a [`GpuMemError`] otherwise (never UB), while
//!    [`GpuShmMmap::read_heap`]/[`write_heap`](GpuShmMmap::write_heap) go
//!    through `cuMemcpy`, are always legal, and are the portable path.
//!    `CU_MEM_ATTACH_HOST` would make CPU access unconditionally legal, but
//!    was measured to leave the heap unreachable from the device
//!    (`cuMemcpyDtoH` → `CUDA_ERROR_INVALID_VALUE`) until stream-attached,
//!    which would defeat the backend's purpose — hence it is not used.
//! 7. **ROCm/HIP is out of scope** for this module (the crate's `hip`
//!    feature covers it); `#[cfg(feature = "cuda")]` gates everything below.

/// True when CUDA support is compiled in **and** the driver library loads
/// with at least one visible device. Mirrors `ctp_gpu::is_available`.
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::driver_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{
    GpuDevice, GpuIpcMemHandle, GpuMalloc, GpuMemError, GpuShmMmap, MemoryBackendHeader,
    MemoryBackendId, BACKEND_HEADER_SIZE, MIN_BACKEND_SIZE,
};

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
    use std::sync::OnceLock;

    // -----------------------------------------------------------------------
    // Raw driver types + constants (CUDA driver API ABI)
    // -----------------------------------------------------------------------

    type CuResult = c_int;
    type CuDevice = c_int;
    type CuContext = *mut c_void;
    type CuDeviceptr = u64;

    const CUDA_SUCCESS: CuResult = 0;

    /// `CU_MEM_ATTACH_GLOBAL` — the managed allocation is accessible from any
    /// stream on any device. See module doc, divergence 6, for why this and
    /// not `CU_MEM_ATTACH_HOST`.
    const CU_MEM_ATTACH_GLOBAL: c_uint = 1;
    /// `CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS`, the only legal flag for
    /// `cuIpcOpenMemHandle` — which is Unix-only here (divergence 5).
    #[cfg(unix)]
    const CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: c_uint = 1;
    /// `CU_IPC_HANDLE_SIZE`.
    const CU_IPC_HANDLE_SIZE: usize = 64;

    /// Why the Windows IPC entry points are refused before reaching the
    /// driver — see [`GpuMalloc::ipc_handle`] and module doc, divergence 5.
    #[cfg(windows)]
    const IPC_UNSUPPORTED_ON_WINDOWS: &str =
        "CUDA IPC is not supported on Windows; nvcuda.dll exports \
         cuIpcGetMemHandle and returns CUDA_SUCCESS, but the handle is \
         unusable and a concurrent cuMemFree then faults inside the driver, \
         so this wrapper refuses without calling it";

    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
    const CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: c_int = 83;
    const CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: c_int = 89;

    // -----------------------------------------------------------------------
    // Errors
    // -----------------------------------------------------------------------

    /// Failure from the CUDA driver, or from this wrapper's own validation.
    ///
    /// Library errors are always returned, never panicked on: a missing
    /// driver, an absent device, an exhausted heap, an out-of-range transfer,
    /// or an unsupported operation (e.g. CUDA IPC on Windows) all surface as
    /// this type.
    #[derive(Debug)]
    pub struct GpuMemError(pub String);

    impl std::fmt::Display for GpuMemError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "GpuMemError: {}", self.0)
        }
    }
    impl std::error::Error for GpuMemError {}

    // -----------------------------------------------------------------------
    // Dynamic binding to the driver (see module doc, divergence 2)
    // -----------------------------------------------------------------------

    #[cfg(windows)]
    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    #[cfg(unix)]
    extern "C" {
        fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    /// Hand-rolled `extern "C"` signatures for every driver entry point this
    /// module calls. Each field reproduces the C declaration written above it;
    /// the table is built once and shared (fn pointers are `Send + Sync`).
    struct Driver {
        // CUresult cuInit(unsigned int Flags)
        init: unsafe extern "C" fn(c_uint) -> CuResult,
        // CUresult cuDeviceGetCount(int *count)
        device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
        // CUresult cuDeviceGet(CUdevice *device, int ordinal)
        device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
        // CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
        device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
        // CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute a, CUdevice d)
        device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
        // CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
        primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
        // CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
        primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult,
        // CUresult cuCtxSetCurrent(CUcontext ctx)
        ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
        // CUresult cuCtxSynchronize(void)
        ctx_synchronize: unsafe extern "C" fn() -> CuResult,
        // CUresult cuGetErrorName(CUresult error, const char **pStr)
        get_error_name: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
        // CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
        mem_alloc: unsafe extern "C" fn(*mut CuDeviceptr, usize) -> CuResult,
        // CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytes, unsigned flags)
        mem_alloc_managed: unsafe extern "C" fn(*mut CuDeviceptr, usize, c_uint) -> CuResult,
        // CUresult cuMemFree(CUdeviceptr dptr)
        mem_free: unsafe extern "C" fn(CuDeviceptr) -> CuResult,
        // CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t n)
        memcpy_htod: unsafe extern "C" fn(CuDeviceptr, *const c_void, usize) -> CuResult,
        // CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t n)
        memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize) -> CuResult,
        // CUresult cuMemsetD8(CUdeviceptr dst, unsigned char uc, size_t n)
        memset_d8: unsafe extern "C" fn(CuDeviceptr, u8, usize) -> CuResult,
        // CUDA IPC — Linux only, and deliberately not even *bound* elsewhere
        // (module doc, divergence 5). `Option` because a driver need not
        // export them; absence yields a GpuMemError, never a panic.
        //
        // CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr)
        #[cfg(unix)]
        ipc_get_mem_handle:
            Option<unsafe extern "C" fn(*mut GpuIpcMemHandle, CuDeviceptr) -> CuResult>,
        // CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle h, unsigned Flags)
        #[cfg(unix)]
        ipc_open_mem_handle:
            Option<unsafe extern "C" fn(*mut CuDeviceptr, GpuIpcMemHandle, c_uint) -> CuResult>,
        // CUresult cuIpcCloseMemHandle(CUdeviceptr dptr)
        #[cfg(unix)]
        ipc_close_mem_handle: Option<unsafe extern "C" fn(CuDeviceptr) -> CuResult>,
    }

    /// Look up one symbol. `name` must be NUL-terminated.
    fn raw_symbol(handle: *mut c_void, name: &str) -> *mut c_void {
        debug_assert!(name.ends_with('\0'));
        let ptr = name.as_ptr() as *const c_char;
        // SAFETY: `handle` came from LoadLibraryA/dlopen in `load_library` and
        // is still loaded (the library is never unloaded); `ptr` is a
        // NUL-terminated C string valid for the call. Both APIs return null
        // for an absent symbol, which every caller checks.
        #[cfg(windows)]
        unsafe {
            GetProcAddress(handle, ptr)
        }
        #[cfg(unix)]
        unsafe {
            dlsym(handle, ptr)
        }
    }

    macro_rules! sym {
        ($handle:expr, $name:literal) => {{
            let p = raw_symbol($handle, concat!($name, "\0"));
            if p.is_null() {
                return None;
            }
            // SAFETY: `$name` is a documented CUDA Driver API entry point and
            // the destination field's fn-pointer type reproduces its C
            // signature exactly (spelled out in the comment above each
            // field). Transmuting a non-null code address from
            // GetProcAddress/dlsym to that fn pointer is the standard idiom.
            unsafe { std::mem::transmute(p) }
        }};
    }

    // Only IPC uses this, and IPC is bound on Unix alone.
    #[allow(unused_macros)]
    macro_rules! opt_sym {
        ($handle:expr, $name:literal) => {{
            let p = raw_symbol($handle, concat!($name, "\0"));
            if p.is_null() {
                None
            } else {
                // SAFETY: as `sym!`, but the caller tolerates absence.
                Some(unsafe { std::mem::transmute(p) })
            }
        }};
    }

    /// Load the driver library. The handle is intentionally never freed: the
    /// driver stays mapped for the process lifetime, so every fn pointer in
    /// [`Driver`] remains callable (and `Driver` stays `Send + Sync`).
    fn load_library() -> Option<*mut c_void> {
        #[cfg(windows)]
        const CANDIDATES: &[&str] = &["nvcuda.dll\0"];
        #[cfg(unix)]
        const CANDIDATES: &[&str] = &["libcuda.so.1\0", "libcuda.so\0"];

        for name in CANDIDATES {
            let ptr = name.as_ptr() as *const c_char;
            // SAFETY: `ptr` is a NUL-terminated C string valid for the call.
            // Both loaders return null on failure, which we check.
            let handle = unsafe {
                #[cfg(windows)]
                {
                    LoadLibraryA(ptr)
                }
                #[cfg(unix)]
                {
                    const RTLD_NOW: c_int = 2;
                    dlopen(ptr, RTLD_NOW)
                }
            };
            if !handle.is_null() {
                return Some(handle);
            }
        }
        None
    }

    impl Driver {
        // Each `sym!` transmutes to the *field's* declared fn-pointer type,
        // which is the single source of truth for these signatures. Spelling
        // the type again at each call site (what the lint asks for) would
        // duplicate all nineteen signatures and let the two copies drift.
        #[allow(clippy::missing_transmute_annotations)]
        fn load() -> Option<Driver> {
            let h = load_library()?;
            Some(Driver {
                init: sym!(h, "cuInit"),
                device_get_count: sym!(h, "cuDeviceGetCount"),
                device_get: sym!(h, "cuDeviceGet"),
                device_get_name: sym!(h, "cuDeviceGetName"),
                device_get_attribute: sym!(h, "cuDeviceGetAttribute"),
                primary_ctx_retain: sym!(h, "cuDevicePrimaryCtxRetain"),
                primary_ctx_release: sym!(h, "cuDevicePrimaryCtxRelease_v2"),
                ctx_set_current: sym!(h, "cuCtxSetCurrent"),
                ctx_synchronize: sym!(h, "cuCtxSynchronize"),
                get_error_name: sym!(h, "cuGetErrorName"),
                mem_alloc: sym!(h, "cuMemAlloc_v2"),
                mem_alloc_managed: sym!(h, "cuMemAllocManaged"),
                mem_free: sym!(h, "cuMemFree_v2"),
                memcpy_htod: sym!(h, "cuMemcpyHtoD_v2"),
                memcpy_dtoh: sym!(h, "cuMemcpyDtoH_v2"),
                memset_d8: sym!(h, "cuMemsetD8_v2"),
                #[cfg(unix)]
                ipc_get_mem_handle: opt_sym!(h, "cuIpcGetMemHandle"),
                #[cfg(unix)]
                ipc_open_mem_handle: opt_sym!(h, "cuIpcOpenMemHandle_v2"),
                #[cfg(unix)]
                ipc_close_mem_handle: opt_sym!(h, "cuIpcCloseMemHandle"),
            })
        }
    }

    static DRIVER: OnceLock<Option<Driver>> = OnceLock::new();

    fn driver() -> Result<&'static Driver, GpuMemError> {
        DRIVER.get_or_init(Driver::load).as_ref().ok_or_else(|| {
            GpuMemError(
                "CUDA driver library not found or missing required symbols \
                 (looked for nvcuda.dll / libcuda.so.1)"
                    .into(),
            )
        })
    }

    /// Map a nonzero `CUresult` to a [`GpuMemError`] carrying the driver's own
    /// error name, mirroring `cu_check` in `ctp-gpu/src/cuda.rs`.
    fn cu_check(rc: CuResult, what: &str) -> Result<(), GpuMemError> {
        if rc == CUDA_SUCCESS {
            return Ok(());
        }
        let Ok(d) = driver() else {
            return Err(GpuMemError(format!("{what}: CUDA error {rc}")));
        };
        let mut name: *const c_char = std::ptr::null();
        // SAFETY: valid out-pointer; on failure the driver leaves `name` null.
        unsafe { (d.get_error_name)(rc, &mut name) };
        if name.is_null() {
            return Err(GpuMemError(format!("{what}: CUDA error {rc}")));
        }
        // SAFETY: the driver returns a pointer to a static NUL-terminated
        // string that outlives this borrow.
        let text = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        Err(GpuMemError(format!("{what}: {text}")))
    }

    /// True if the driver library loads, initializes, and sees a device.
    pub fn driver_available() -> bool {
        let Ok(d) = driver() else {
            return false;
        };
        // SAFETY: cuInit/cuDeviceGetCount have no preconditions beyond a valid
        // out-pointer.
        unsafe {
            if (d.init)(0) != CUDA_SUCCESS {
                return false;
            }
            let mut n: c_int = 0;
            (d.device_get_count)(&mut n) == CUDA_SUCCESS && n > 0
        }
    }

    // -----------------------------------------------------------------------
    // GpuDevice — `GpuApi::SetDevice` + primary context
    // -----------------------------------------------------------------------

    /// A CUDA device with its primary context retained and made current.
    ///
    /// Not `Send`/`Sync` by construction (it holds the raw `CUcontext`):
    /// `cuCtxSetCurrent` binds the context to the *calling thread*, so a
    /// device bound on one thread must not be used from another.
    pub struct GpuDevice {
        device: CuDevice,
        _ctx: CuContext,
    }

    impl GpuDevice {
        /// Initialize the driver and bind the primary context of `ordinal`.
        pub fn new(ordinal: i32) -> Result<Self, GpuMemError> {
            let d = driver()?;
            // SAFETY: the standard driver bring-up sequence; every call is
            // checked and each out-pointer is a valid local.
            unsafe {
                cu_check((d.init)(0), "cuInit")?;
                let mut device: CuDevice = 0;
                cu_check((d.device_get)(&mut device, ordinal), "cuDeviceGet")?;
                let mut ctx: CuContext = std::ptr::null_mut();
                cu_check(
                    (d.primary_ctx_retain)(&mut ctx, device),
                    "cuDevicePrimaryCtxRetain",
                )?;
                cu_check((d.ctx_set_current)(ctx), "cuCtxSetCurrent")?;
                Ok(Self { device, _ctx: ctx })
            }
        }

        /// Device name, e.g. `NVIDIA GeForce RTX 5080`.
        pub fn name(&self) -> String {
            let Ok(d) = driver() else {
                return String::new();
            };
            let mut buf = [0 as c_char; 256];
            // SAFETY: valid buffer with its true length; the driver
            // NUL-terminates within `len`.
            unsafe {
                if (d.device_get_name)(buf.as_mut_ptr(), buf.len() as c_int, self.device)
                    != CUDA_SUCCESS
                {
                    return String::new();
                }
                CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
            }
        }

        fn attribute(&self, attr: c_int) -> i32 {
            let Ok(d) = driver() else {
                return 0;
            };
            let mut v: c_int = 0;
            // SAFETY: valid out-pointer and a documented attribute id; on
            // failure `v` stays 0, which every caller treats as "no".
            unsafe {
                if (d.device_get_attribute)(&mut v, attr, self.device) != CUDA_SUCCESS {
                    return 0;
                }
            }
            v
        }

        /// Compute capability, e.g. `(12, 0)` for Blackwell.
        pub fn compute_capability(&self) -> (i32, i32) {
            (
                self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
            )
        }

        /// Whether this device can allocate managed (unified) memory, i.e.
        /// whether [`GpuShmMmap::create`] can work at all.
        pub fn supports_managed_memory(&self) -> bool {
            self.attribute(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY) != 0
        }

        /// Whether the CPU may dereference managed memory *while device work
        /// is in flight* (`CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`).
        ///
        /// 0 on Windows/WDDM (including the RTX 5080 this was validated on)
        /// and on pre-Pascal devices. When false, [`GpuShmMmap::heap`] refuses
        /// to hand out a slice and callers must use the `cuMemcpy`-based
        /// [`GpuShmMmap::read_heap`]/[`GpuShmMmap::write_heap`]. See module
        /// doc, divergence 6.
        pub fn supports_concurrent_managed_access(&self) -> bool {
            self.attribute(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS) != 0
        }

        /// Block until all work on the context completes (`GpuApi::Synchronize`).
        pub fn synchronize(&self) -> Result<(), GpuMemError> {
            let d = driver()?;
            // SAFETY: a context was made current in `new`.
            cu_check(unsafe { (d.ctx_synchronize)() }, "cuCtxSynchronize")
        }
    }

    impl Drop for GpuDevice {
        fn drop(&mut self) {
            let Ok(d) = driver() else {
                return;
            };
            // SAFETY: releasing exactly the primary context retained in `new`.
            // Errors are unrecoverable here and deliberately ignored.
            unsafe { (d.primary_ctx_release)(self.device) };
        }
    }

    // -----------------------------------------------------------------------
    // Backend header — mirrors memory_backend.h
    // -----------------------------------------------------------------------

    /// C++ `kBackendHeaderSize`: the region prefix reserved for the header.
    ///
    /// 64 KiB, not the 4 KiB the C++ GPU backends' comments claim (module
    /// doc, divergence 4).
    pub const BACKEND_HEADER_SIZE: usize = 65536;

    /// Smallest region both C++ backends accept; smaller requests are rounded
    /// up rather than rejected, as in `shm_init`.
    pub const MIN_BACKEND_SIZE: usize = 1024 * 1024;

    const MEMORY_BACKEND_INITIALIZED: u64 = 1 << 0;
    const MEMORY_BACKEND_OWNED: u64 = 1 << 1;

    /// C++ `ctp::ipc::MemoryBackendId`; byte-compatible with
    /// `ctp_memory::AllocatorId` (MEMORY_DESIGN.md pillar 2).
    #[repr(C)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct MemoryBackendId {
        pub major: u32,
        pub minor: u32,
    }

    impl MemoryBackendId {
        pub const fn new(major: u32, minor: u32) -> Self {
            Self { major, minor }
        }

        /// C++ `MemoryBackendId::GetRoot()`.
        pub const fn root() -> Self {
            Self::new(0, 0)
        }

        /// C++ `MemoryBackendId::GetNull()`.
        pub const fn null() -> Self {
            Self::new(u32::MAX, u32::MAX)
        }

        pub const fn is_null(&self) -> bool {
            self.major == u32::MAX && self.minor == u32::MAX
        }
    }

    /// C++ `ctp::ipc::MemoryBackendHeader`, persisted at region offset 0.
    ///
    /// `flags` is the C++ `bitfield64_t`, whose sole member is a `u64`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct MemoryBackendHeader {
        pub id: MemoryBackendId,
        pub flags: u64,
        pub backend_size: u64,
        pub data_capacity: u64,
        pub data_id: i32,
        _pad: u32,
    }

    impl MemoryBackendHeader {
        fn new(id: MemoryBackendId, backend_size: usize, gpu_id: i32) -> Self {
            Self {
                id,
                flags: MEMORY_BACKEND_OWNED | MEMORY_BACKEND_INITIALIZED,
                backend_size: backend_size as u64,
                data_capacity: (backend_size - BACKEND_HEADER_SIZE) as u64,
                data_id: gpu_id,
                _pad: 0,
            }
        }

        /// True if the header carries `MEMORY_BACKEND_INITIALIZED`.
        pub fn is_initialized(&self) -> bool {
            self.flags & MEMORY_BACKEND_INITIALIZED != 0
        }

        /// True if the header carries `MEMORY_BACKEND_OWNED`.
        pub fn is_owner(&self) -> bool {
            self.flags & MEMORY_BACKEND_OWNED != 0
        }
    }

    /// Round a requested region size up to the backends' minimum, matching the
    /// `if (size < kMin) size = kMin;` in both C++ `shm_init`s.
    fn clamp_backend_size(requested: usize) -> usize {
        requested.max(MIN_BACKEND_SIZE)
    }

    /// Opaque `CUipcMemHandle` (64 bytes), the C++ `GpuIpcMemHandle`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct GpuIpcMemHandle {
        reserved: [u8; CU_IPC_HANDLE_SIZE],
    }

    impl GpuIpcMemHandle {
        /// Serialize for transport to another process (the C++ ships the raw
        /// 64 bytes too).
        pub fn as_bytes(&self) -> &[u8; CU_IPC_HANDLE_SIZE] {
            &self.reserved
        }

        /// Rebuild a handle received from another process.
        pub fn from_bytes(bytes: [u8; CU_IPC_HANDLE_SIZE]) -> Self {
            Self { reserved: bytes }
        }
    }

    impl std::fmt::Debug for GpuIpcMemHandle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "GpuIpcMemHandle(<{CU_IPC_HANDLE_SIZE} bytes>)")
        }
    }

    // -----------------------------------------------------------------------
    // Shared region helpers (both backends share the same layout + transfers)
    // -----------------------------------------------------------------------

    /// Bounds-check `off + len` against `cap`, returning an error (never
    /// panicking, never wrapping) on overflow or overrun.
    fn check_range(cap: usize, off: usize, len: usize) -> Result<(), GpuMemError> {
        let end = off
            .checked_add(len)
            .ok_or_else(|| GpuMemError(format!("range {off}+{len} overflows")))?;
        if end > cap {
            return Err(GpuMemError(format!(
                "range {off}+{len} exceeds capacity {cap}"
            )));
        }
        Ok(())
    }

    /// `GpuApi::Memcpy` host→device into a bounds-checked region slot.
    fn region_write(base: CuDeviceptr, cap: usize, off: usize, src: &[u8]) -> Result<(), GpuMemError> {
        let d = driver()?;
        check_range(cap, off, src.len())?;
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: `src` is a valid host slice of `src.len()` bytes, and the
        // destination range was just bounds-checked against the allocation.
        cu_check(
            unsafe { (d.memcpy_htod)(base + off as u64, src.as_ptr() as *const c_void, src.len()) },
            "cuMemcpyHtoD",
        )
    }

    /// `GpuApi::Memcpy` device→host from a bounds-checked region slot.
    fn region_read(base: CuDeviceptr, cap: usize, off: usize, dst: &mut [u8]) -> Result<(), GpuMemError> {
        let d = driver()?;
        check_range(cap, off, dst.len())?;
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: `dst` is a valid mutable host slice of `dst.len()` bytes and
        // the source range was just bounds-checked against the allocation.
        cu_check(
            unsafe { (d.memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, base + off as u64, dst.len()) },
            "cuMemcpyDtoH",
        )
    }

    /// Zero a freshly allocated region and stamp the backend header into it.
    ///
    /// Both steps run device-side (`cuMemsetD8` + `cuMemcpyHtoD`), which is
    /// legal for device *and* managed memory regardless of the host's managed
    /// access rights — see module doc, divergence 6.
    fn init_region(
        device: &GpuDevice,
        ptr: CuDeviceptr,
        id: MemoryBackendId,
        backend_size: usize,
        gpu_id: i32,
    ) -> Result<(), GpuMemError> {
        let d = driver()?;
        // SAFETY: `ptr` owns `backend_size` bytes from the caller's allocation.
        cu_check(unsafe { (d.memset_d8)(ptr, 0, backend_size) }, "cuMemsetD8")?;

        let header = MemoryBackendHeader::new(id, backend_size, gpu_id);
        // SAFETY: `header` is a live `#[repr(C)]` POD of exactly
        // size_of::<MemoryBackendHeader>() bytes, and the region is far larger
        // (>= BACKEND_HEADER_SIZE == 64 KiB).
        cu_check(
            unsafe {
                (d.memcpy_htod)(
                    ptr,
                    &header as *const MemoryBackendHeader as *const c_void,
                    std::mem::size_of::<MemoryBackendHeader>(),
                )
            },
            "cuMemcpyHtoD(header)",
        )?;
        device.synchronize()
    }

    /// Read the backend header back out of a region.
    fn read_header(ptr: CuDeviceptr) -> Result<MemoryBackendHeader, GpuMemError> {
        let d = driver()?;
        let mut header = MemoryBackendHeader::default();
        // SAFETY: `&mut header` is a valid destination of exactly
        // size_of::<MemoryBackendHeader>() bytes; the region begins with that
        // header and is at least 64 KiB.
        cu_check(
            unsafe {
                (d.memcpy_dtoh)(
                    &mut header as *mut MemoryBackendHeader as *mut c_void,
                    ptr,
                    std::mem::size_of::<MemoryBackendHeader>(),
                )
            },
            "cuMemcpyDtoH(header)",
        )?;
        Ok(header)
    }

    // -----------------------------------------------------------------------
    // GpuShmMmap — managed-memory segment (gpu_shm_mmap.h analog)
    // -----------------------------------------------------------------------

    /// A segment whose heap lives in **unified memory** (`cuMemAllocManaged`
    /// with `CU_MEM_ATTACH_GLOBAL`), the Rust analog of C++
    /// `ctp::ipc::GpuShmMmap`.
    ///
    /// Layout matches the C++ backend:
    /// `region -> [BACKEND_HEADER_SIZE header | heap]`.
    ///
    /// The heap is addressable by kernels at [`heap_device_ptr`](Self::heap_device_ptr)
    /// and, under CUDA's Unified Virtual Addressing, by the CPU at the same
    /// address. Host access comes in two forms, and which one you get is a
    /// property of the *hardware* (module doc, divergence 6):
    ///
    /// * [`heap`](Self::heap)/[`heap_mut`](Self::heap_mut) — zero-copy slices,
    ///   available only where the device reports
    ///   [`supports_concurrent_managed_access`](GpuDevice::supports_concurrent_managed_access);
    /// * [`read_heap`](Self::read_heap)/[`write_heap`](Self::write_heap) —
    ///   `cuMemcpy`, always available and always legal.
    ///
    /// Process-local: managed memory has no IPC handle (see [`GpuMalloc`] for
    /// the cross-process backend), mirroring the C++ backend's "not shareable
    /// across processes". Freed on drop (`cuMemFree`).
    pub struct GpuShmMmap {
        ptr: CuDeviceptr,
        backend_size: usize,
        id: MemoryBackendId,
        data_id: i32,
        url: String,
        /// Cached at construction: whether the CPU may touch the heap while
        /// device work is in flight.
        host_coherent: bool,
    }

    impl GpuShmMmap {
        /// C++ `shm_init`: allocate `backend_size` bytes of managed memory
        /// (rounded up to [`MIN_BACKEND_SIZE`]), zero it, and persist the
        /// backend header at offset 0.
        ///
        /// `url` is stored for identification only — the C++ backend does the
        /// same (it has no URL-addressable form). `gpu_id` is recorded in the
        /// header's `data_id`.
        pub fn create(
            device: &GpuDevice,
            id: MemoryBackendId,
            backend_size: usize,
            url: &str,
            gpu_id: i32,
        ) -> Result<Self, GpuMemError> {
            let d = driver()?;
            if !device.supports_managed_memory() {
                return Err(GpuMemError(
                    "device does not support managed memory (CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY=0)"
                        .into(),
                ));
            }
            let backend_size = clamp_backend_size(backend_size);

            let mut ptr: CuDeviceptr = 0;
            // SAFETY: valid out-pointer; size is nonzero (>= MIN_BACKEND_SIZE);
            // CU_MEM_ATTACH_GLOBAL is the documented flag for an allocation
            // usable from any stream on any device. A context is current (held
            // by `device`).
            cu_check(
                unsafe { (d.mem_alloc_managed)(&mut ptr, backend_size, CU_MEM_ATTACH_GLOBAL) },
                "cuMemAllocManaged",
            )?;

            // Bind ownership to `seg` before any fallible step, so an error
            // frees the allocation via Drop rather than leaking it.
            let seg = Self {
                ptr,
                backend_size,
                id,
                data_id: gpu_id,
                url: url.to_string(),
                host_coherent: device.supports_concurrent_managed_access(),
            };
            init_region(device, ptr, id, backend_size, gpu_id)?;
            Ok(seg)
        }

        /// Backend id (C++ `GetId`).
        pub fn id(&self) -> MemoryBackendId {
            self.id
        }

        /// Identifier this backend was created with (C++ `url_`).
        pub fn url(&self) -> &str {
            &self.url
        }

        /// Total region size: header + heap (C++ `backend_size_`).
        pub fn backend_size(&self) -> usize {
            self.backend_size
        }

        /// Heap size available to an allocator (C++ `data_capacity_`).
        pub fn data_capacity(&self) -> usize {
            self.backend_size - BACKEND_HEADER_SIZE
        }

        /// GPU id recorded in the header (C++ `data_id_`).
        pub fn data_id(&self) -> i32 {
            self.data_id
        }

        /// Address of the whole region (header included), for kernels that
        /// want the backend header. A `u64`, not a raw pointer — exactly as
        /// `ctp_gpu::DeviceBuffer::device_ptr` does.
        pub fn device_ptr(&self) -> u64 {
            self.ptr
        }

        /// Address of the heap (region + [`BACKEND_HEADER_SIZE`]) — the
        /// pointer a kernel should receive to work on segment data.
        pub fn heap_device_ptr(&self) -> u64 {
            self.ptr + BACKEND_HEADER_SIZE as u64
        }

        /// Whether zero-copy host slices are available on this device; when
        /// false, [`heap`](Self::heap)/[`heap_mut`](Self::heap_mut) return an
        /// error and the memcpy accessors are the way in.
        pub fn host_slices_available(&self) -> bool {
            self.host_coherent
        }

        /// Read back the persisted header.
        pub fn header(&self) -> Result<MemoryBackendHeader, GpuMemError> {
            read_header(self.ptr)
        }

        /// The heap as a zero-copy CPU slice.
        ///
        /// `Err` when the device lacks `CONCURRENT_MANAGED_ACCESS` (e.g. any
        /// Windows/WDDM device): there, the CPU may not dereference managed
        /// memory while *any* device work is in flight, which safe code cannot
        /// guarantee process-wide — so a slice is never handed out and
        /// [`read_heap`](Self::read_heap) is the portable path.
        pub fn heap(&self) -> Result<&[u8], GpuMemError> {
            if !self.host_coherent {
                return Err(Self::no_host_slice_error());
            }
            // SAFETY: `create` allocated `backend_size` bytes; the heap starts
            // at BACKEND_HEADER_SIZE and runs for `data_capacity()` bytes, so
            // the range lies inside that one allocation. Managed memory is
            // host-addressable under UVA, `host_coherent` establishes that the
            // CPU may read it even with kernels in flight, and `&self` ties the
            // slice's lifetime to the live segment.
            Ok(unsafe {
                std::slice::from_raw_parts(
                    (self.ptr as *const u8).add(BACKEND_HEADER_SIZE),
                    self.data_capacity(),
                )
            })
        }

        /// The heap as a mutable zero-copy CPU slice. Same availability rule as
        /// [`heap`](Self::heap); `&mut self` additionally excludes aliasing
        /// host readers.
        pub fn heap_mut(&mut self) -> Result<&mut [u8], GpuMemError> {
            if !self.host_coherent {
                return Err(Self::no_host_slice_error());
            }
            let cap = self.data_capacity();
            // SAFETY: as `heap`, plus `&mut self` guarantees no other Rust
            // reference into this segment exists for the borrow's duration.
            Ok(unsafe {
                std::slice::from_raw_parts_mut((self.ptr as *mut u8).add(BACKEND_HEADER_SIZE), cap)
            })
        }

        fn no_host_slice_error() -> GpuMemError {
            GpuMemError(
                "zero-copy host slices need CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS \
                 (0 on this device, e.g. Windows/WDDM); use read_heap/write_heap instead"
                    .into(),
            )
        }

        /// `GpuApi::Memcpy` host→heap at `off`. Always available.
        pub fn write_heap(&mut self, off: usize, src: &[u8]) -> Result<(), GpuMemError> {
            region_write(self.heap_device_ptr(), self.data_capacity(), off, src)
        }

        /// `GpuApi::Memcpy` heap→host from `off`. Always available.
        pub fn read_heap(&self, off: usize, dst: &mut [u8]) -> Result<(), GpuMemError> {
            region_read(self.heap_device_ptr(), self.data_capacity(), off, dst)
        }
    }

    impl Drop for GpuShmMmap {
        fn drop(&mut self) {
            let Ok(d) = driver() else {
                return;
            };
            if self.ptr != 0 {
                // SAFETY: `ptr` came from cuMemAllocManaged and is freed once
                // (Drop runs at most once).
                unsafe { (d.mem_free)(self.ptr) };
                self.ptr = 0;
            }
        }
    }

    // -----------------------------------------------------------------------
    // GpuMalloc — device-only segment (gpu_malloc.h analog)
    // -----------------------------------------------------------------------

    /// A GPU-only segment (`cuMemAlloc`), the Rust analog of C++
    /// `ctp::ipc::GpuMalloc`. Layout is the same
    /// `[BACKEND_HEADER_SIZE header | data]`, but the region lives in device
    /// memory, so the host reaches it only through
    /// [`write_data`](Self::write_data)/[`read_data`](Self::read_data)
    /// (`GpuApi::Memcpy`) — there is no host slice at any time.
    ///
    /// Unlike [`GpuShmMmap`] this backend *can* be shared across processes:
    /// the owner publishes an [`ipc_handle`](Self::ipc_handle) and a peer maps
    /// it with [`attach_ipc`](Self::attach_ipc). CUDA IPC is Linux-only; on
    /// Windows both calls return a [`GpuMemError`] rather than panicking.
    ///
    /// Drop frees the region (owner) or closes the IPC mapping (attacher) —
    /// unconditionally, unlike the C++ dtor (module doc, divergence 3).
    pub struct GpuMalloc {
        ptr: CuDeviceptr,
        backend_size: usize,
        id: MemoryBackendId,
        data_id: i32,
        url: String,
        owner: bool,
    }

    impl GpuMalloc {
        /// C++ `shm_init`: allocate device memory (rounded up to
        /// [`MIN_BACKEND_SIZE`]) and copy the backend header into it.
        pub fn create(
            device: &GpuDevice,
            id: MemoryBackendId,
            backend_size: usize,
            url: &str,
            gpu_id: i32,
        ) -> Result<Self, GpuMemError> {
            let d = driver()?;
            let backend_size = clamp_backend_size(backend_size);

            let mut ptr: CuDeviceptr = 0;
            // SAFETY: valid out-pointer; size is nonzero. A context is current
            // (held by `device`).
            cu_check(unsafe { (d.mem_alloc)(&mut ptr, backend_size) }, "cuMemAlloc")?;

            // Bind ownership before any fallible step so errors free via Drop.
            let seg = Self {
                ptr,
                backend_size,
                id,
                data_id: gpu_id,
                url: url.to_string(),
                owner: true,
            };
            init_region(device, ptr, id, backend_size, gpu_id)?;
            Ok(seg)
        }

        /// C++ `shm_attach_ipc` — **unavailable on Windows**, where CUDA IPC
        /// does not exist. Fails without calling the driver; see
        /// [`ipc_handle`](Self::ipc_handle) for why that matters.
        #[cfg(windows)]
        pub fn attach_ipc(_device: &GpuDevice, _handle: &GpuIpcMemHandle) -> Result<Self, GpuMemError> {
            Err(GpuMemError(IPC_UNSUPPORTED_ON_WINDOWS.into()))
        }

        /// C++ `shm_attach_ipc`: open a handle exported by the owning process
        /// and recover the metadata from the header stored in the region.
        ///
        /// Returns a non-owning segment: drop closes the mapping instead of
        /// freeing the memory.
        #[cfg(unix)]
        pub fn attach_ipc(_device: &GpuDevice, handle: &GpuIpcMemHandle) -> Result<Self, GpuMemError> {
            let d = driver()?;
            let open = d.ipc_open_mem_handle.ok_or_else(|| {
                GpuMemError("cuIpcOpenMemHandle unavailable (CUDA IPC is Linux-only)".into())
            })?;

            let mut ptr: CuDeviceptr = 0;
            // SAFETY: valid out-pointer; `handle` is passed by value as the C
            // API expects (CUipcMemHandle is a 64-byte POD);
            // LAZY_ENABLE_PEER_ACCESS is the only legal flag.
            cu_check(
                unsafe { open(&mut ptr, *handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) },
                "cuIpcOpenMemHandle",
            )?;

            // Own the mapping before the next fallible step, so a header read
            // failure closes it via Drop instead of leaking it.
            let mut seg = Self {
                ptr,
                backend_size: 0,
                id: MemoryBackendId::null(),
                data_id: -1,
                url: String::new(),
                owner: false,
            };
            let header = read_header(ptr)?;
            seg.backend_size = header.backend_size as usize;
            seg.id = header.id;
            seg.data_id = header.data_id;
            Ok(seg)
        }

        /// C++ `GpuApi::GetIpcMemHandle` — **unavailable on Windows**.
        ///
        /// This returns an error *without calling the driver*, which is
        /// deliberate and load-bearing. CUDA IPC is a Linux-only facility, but
        /// `nvcuda.dll` still exports `cuIpcGetMemHandle` and it still returns
        /// `CUDA_SUCCESS` — while leaving state behind such that a later
        /// `cuMemFree` on *another thread* faults inside the driver. That was
        /// reproduced here (SIGSEGV in `cuMemFree_v2`, 9 of 12 runs at
        /// `--test-threads=2`; 0 of 12 once this call was removed), so the
        /// symbol is not even bound on Windows. See module doc, divergence 5.
        #[cfg(windows)]
        pub fn ipc_handle(&self) -> Result<GpuIpcMemHandle, GpuMemError> {
            Err(GpuMemError(IPC_UNSUPPORTED_ON_WINDOWS.into()))
        }

        /// C++ `GpuApi::GetIpcMemHandle`: export this region for another
        /// process.
        #[cfg(unix)]
        pub fn ipc_handle(&self) -> Result<GpuIpcMemHandle, GpuMemError> {
            let d = driver()?;
            let get = d.ipc_get_mem_handle.ok_or_else(|| {
                GpuMemError("cuIpcGetMemHandle unavailable (CUDA IPC is Linux-only)".into())
            })?;
            let mut handle = GpuIpcMemHandle {
                reserved: [0u8; CU_IPC_HANDLE_SIZE],
            };
            // SAFETY: valid out-pointer to a 64-byte POD; `ptr` is a device
            // allocation from cuMemAlloc.
            cu_check(unsafe { get(&mut handle, self.ptr) }, "cuIpcGetMemHandle")?;
            Ok(handle)
        }

        /// True for the process that allocated the region (C++ `IsOwner`).
        pub fn is_owner(&self) -> bool {
            self.owner
        }

        pub fn id(&self) -> MemoryBackendId {
            self.id
        }

        pub fn url(&self) -> &str {
            &self.url
        }

        pub fn backend_size(&self) -> usize {
            self.backend_size
        }

        pub fn data_capacity(&self) -> usize {
            self.backend_size.saturating_sub(BACKEND_HEADER_SIZE)
        }

        pub fn data_id(&self) -> i32 {
            self.data_id
        }

        /// Device address of the whole region (header included).
        pub fn device_ptr(&self) -> u64 {
            self.ptr
        }

        /// Device address of the data region — what a kernel should receive.
        pub fn data_device_ptr(&self) -> u64 {
            self.ptr + BACKEND_HEADER_SIZE as u64
        }

        /// Read back the header the owner persisted in device memory.
        pub fn header(&self) -> Result<MemoryBackendHeader, GpuMemError> {
            read_header(self.ptr)
        }

        /// `GpuApi::Memcpy` host→device, into the data region at `off`.
        pub fn write_data(&mut self, off: usize, src: &[u8]) -> Result<(), GpuMemError> {
            region_write(self.data_device_ptr(), self.data_capacity(), off, src)
        }

        /// `GpuApi::Memcpy` device→host, from the data region at `off`.
        pub fn read_data(&self, off: usize, dst: &mut [u8]) -> Result<(), GpuMemError> {
            region_read(self.data_device_ptr(), self.data_capacity(), off, dst)
        }
    }

    impl Drop for GpuMalloc {
        fn drop(&mut self) {
            let Ok(d) = driver() else {
                return;
            };
            if self.ptr == 0 {
                return;
            }
            if self.owner {
                // SAFETY: `ptr` came from cuMemAlloc; freed exactly once.
                unsafe { (d.mem_free)(self.ptr) };
            } else {
                // Non-owners only exist on Unix: `attach_ipc` is the sole way
                // to make one, and it always fails on Windows.
                #[cfg(unix)]
                if let Some(close) = d.ipc_close_mem_handle {
                    // SAFETY: `ptr` came from cuIpcOpenMemHandle; closed once.
                    unsafe { close(self.ptr) };
                }
            }
            self.ptr = 0;
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Acquire device 0, or `None` when this host has no GPU — every
        /// device test skips (prints + returns) in that case, so the suite
        /// stays green on GPU-less runners and in the devcontainer.
        fn device_or_skip(what: &str) -> Option<GpuDevice> {
            if !driver_available() {
                eprintln!("ctp-gpu-backends: no CUDA device available; skipping {what}");
                return None;
            }
            match GpuDevice::new(0) {
                Ok(d) => Some(d),
                Err(e) => {
                    eprintln!("ctp-gpu-backends: device unavailable ({e}); skipping {what}");
                    None
                }
            }
        }

        /// The header ABI is a contract with the C++ side, so pin it. These
        /// values come from memory_backend.h, not from Rust's layout choices.
        #[test]
        fn header_abi_matches_cpp() {
            assert_eq!(std::mem::size_of::<MemoryBackendId>(), 8);
            // MemoryBackendId{u32,u32} + bitfield64_t{u64} + 2*size_t + int + pad
            assert_eq!(std::mem::size_of::<MemoryBackendHeader>(), 40);
            assert_eq!(std::mem::align_of::<MemoryBackendHeader>(), 8);
            assert_eq!(std::mem::size_of::<GpuIpcMemHandle>(), 64); // CU_IPC_HANDLE_SIZE
                                                                    // kBackendHeaderSize — 64 KiB, not the 4 KiB the C++ GPU headers'
                                                                    // comments claim (module doc, divergence 4).
            assert_eq!(BACKEND_HEADER_SIZE, 65536);

            let h = MemoryBackendHeader::new(MemoryBackendId::new(7, 9), MIN_BACKEND_SIZE, 3);
            assert_eq!(h.id, MemoryBackendId::new(7, 9));
            assert_eq!(h.backend_size, MIN_BACKEND_SIZE as u64);
            assert_eq!(h.data_capacity, (MIN_BACKEND_SIZE - BACKEND_HEADER_SIZE) as u64);
            assert_eq!(h.data_id, 3);
            // Unlike C++ GpuMalloc::shm_init, INITIALIZED is always set.
            assert!(h.is_initialized() && h.is_owner());
        }

        #[test]
        fn backend_id_null_and_root() {
            assert!(MemoryBackendId::null().is_null());
            assert!(!MemoryBackendId::root().is_null());
            assert_eq!(MemoryBackendId::root(), MemoryBackendId::new(0, 0));
        }

        /// Both C++ `shm_init`s round a small request up to 1 MiB.
        #[test]
        fn size_clamped_to_minimum() {
            assert_eq!(clamp_backend_size(0), MIN_BACKEND_SIZE);
            assert_eq!(clamp_backend_size(1024), MIN_BACKEND_SIZE);
            assert_eq!(clamp_backend_size(4 * MIN_BACKEND_SIZE), 4 * MIN_BACKEND_SIZE);
        }

        /// Range checks are pure arithmetic: no wrap, no panic, no device.
        #[test]
        fn range_checks_reject_overflow_and_overrun() {
            assert!(check_range(1024, 0, 1024).is_ok());
            assert!(check_range(1024, 1023, 1).is_ok());
            assert!(check_range(1024, 1024, 0).is_ok());
            assert!(check_range(1024, 1024, 1).is_err());
            assert!(check_range(1024, 0, 1025).is_err());
            // usize::MAX + len must not wrap into a "valid" range.
            assert!(check_range(1024, usize::MAX, 8).is_err());
        }

        /// A driver-less host must produce an error, never a panic/abort.
        #[test]
        fn no_driver_is_an_error_not_a_panic() {
            match GpuDevice::new(0) {
                Ok(_) => assert!(driver_available()),
                Err(e) => assert!(!e.to_string().is_empty()),
            }
        }

        /// The core deliverable: a segment whose heap is GPU-accessible
        /// unified memory, carrying data in and out from the host.
        #[test]
        fn managed_segment_heap_roundtrip() {
            let Some(dev) = device_or_skip("managed_segment_heap_roundtrip") else {
                return;
            };
            eprintln!(
                "ctp-gpu-backends: {} (cc {:?}, managed={}, concurrent_managed={})",
                dev.name(),
                dev.compute_capability(),
                dev.supports_managed_memory(),
                dev.supports_concurrent_managed_access()
            );

            let id = MemoryBackendId::new(std::process::id(), 1);
            let mut seg = GpuShmMmap::create(&dev, id, 2 * 1024 * 1024, "ctp_test_managed", 0)
                .expect("managed segment");

            assert_eq!(seg.backend_size(), 2 * 1024 * 1024);
            assert_eq!(seg.data_capacity(), 2 * 1024 * 1024 - BACKEND_HEADER_SIZE);
            assert_eq!(seg.url(), "ctp_test_managed");
            assert_eq!(seg.heap_device_ptr(), seg.device_ptr() + 65536);
            assert_eq!(seg.data_id(), 0);

            // Persisted header round-trips through unified memory.
            let h = seg.header().expect("header");
            assert_eq!(h.id, id);
            assert_eq!(h.backend_size, 2 * 1024 * 1024);
            assert_eq!(h.data_capacity, (2 * 1024 * 1024 - BACKEND_HEADER_SIZE) as u64);
            assert!(h.is_initialized() && h.is_owner());

            // The heap is zeroed, so offset-based allocators start clean.
            let mut zeros = vec![0xFFu8; 4096];
            seg.read_heap(0, &mut zeros).unwrap();
            assert!(zeros.iter().all(|&b| b == 0));

            // Data round-trips, including the very last byte of the heap.
            let pattern: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
            seg.write_heap(0, &pattern).unwrap();
            let last = seg.data_capacity() - 1;
            seg.write_heap(last, &[0xEE]).unwrap();

            let mut back = vec![0u8; 4096];
            seg.read_heap(0, &mut back).unwrap();
            assert_eq!(back, pattern);
            let mut tail = [0u8; 1];
            seg.read_heap(last, &mut tail).unwrap();
            assert_eq!(tail[0], 0xEE);
        }

        /// Zero-copy host slices: usable exactly where the hardware allows,
        /// a clean error (never a fault) everywhere else. This is the API
        /// contract from module-doc divergence 6.
        #[test]
        fn heap_slices_track_device_capability() {
            let Some(dev) = device_or_skip("heap_slices_track_device_capability") else {
                return;
            };
            let mut seg = GpuShmMmap::create(
                &dev,
                MemoryBackendId::new(std::process::id(), 2),
                MIN_BACKEND_SIZE,
                "ctp_test_slices",
                0,
            )
            .expect("managed segment");

            let concurrent = dev.supports_concurrent_managed_access();
            assert_eq!(seg.host_slices_available(), concurrent);

            if !concurrent {
                // Windows/WDDM lands here: no slice, but a descriptive error.
                let err = seg.heap().unwrap_err().to_string();
                assert!(err.contains("CONCURRENT_MANAGED_ACCESS"), "{err}");
                assert!(seg.heap_mut().is_err());
                eprintln!("ctp-gpu-backends: host slices unavailable (expected on WDDM): {err}");
                return;
            }

            // Concurrent-managed-access devices (Linux/Pascal+): zero-copy.
            let cap = seg.data_capacity();
            assert_eq!(seg.heap().unwrap().len(), cap);
            let pattern: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
            seg.heap_mut().unwrap()[..4096].copy_from_slice(&pattern);
            assert_eq!(&seg.heap().unwrap()[..4096], &pattern[..]);

            // The memcpy path observes what the slice wrote — same memory.
            let mut back = vec![0u8; 4096];
            seg.read_heap(0, &mut back).unwrap();
            assert_eq!(back, pattern);
        }

        /// Prove the heap really is device memory, not just a host buffer:
        /// read it back over the *device* copy path at the heap's device
        /// address and confirm the bytes written through the backend are there.
        #[test]
        fn managed_heap_is_visible_to_device() {
            let Some(dev) = device_or_skip("managed_heap_is_visible_to_device") else {
                return;
            };
            let mut seg = GpuShmMmap::create(
                &dev,
                MemoryBackendId::new(std::process::id(), 3),
                MIN_BACKEND_SIZE,
                "ctp_test_uvm_device",
                0,
            )
            .expect("managed segment");

            let pattern: Vec<u8> = (0..1024u32).map(|i| (i % 97) as u8).collect();
            seg.write_heap(0, &pattern).unwrap();
            dev.synchronize().unwrap();

            // Same bytes via a raw cuMemcpyDtoH from the heap's device address
            // — the address a kernel would be handed.
            let d = driver().unwrap();
            let mut back = vec![0u8; 1024];
            // SAFETY: valid host destination of 1024 bytes; the heap holds at
            // least that much at `heap_device_ptr()`.
            let rc = unsafe {
                (d.memcpy_dtoh)(back.as_mut_ptr() as *mut c_void, seg.heap_device_ptr(), 1024)
            };
            cu_check(rc, "cuMemcpyDtoH").unwrap();
            assert_eq!(back, pattern, "device view must observe the backend's writes");
        }

        /// A tiny request is rounded up rather than rejected (C++ parity).
        #[test]
        fn managed_segment_enforces_minimum_size() {
            let Some(dev) = device_or_skip("managed_segment_enforces_minimum_size") else {
                return;
            };
            let seg = GpuShmMmap::create(&dev, MemoryBackendId::root(), 4096, "ctp_test_min", 0)
                .expect("managed segment");
            assert_eq!(seg.backend_size(), MIN_BACKEND_SIZE);
            assert_eq!(seg.header().unwrap().backend_size, MIN_BACKEND_SIZE as u64);
        }

        /// Out-of-range heap access is a returned error, not a panic or UB.
        #[test]
        fn managed_heap_rejects_out_of_bounds() {
            let Some(dev) = device_or_skip("managed_heap_rejects_out_of_bounds") else {
                return;
            };
            let mut seg = GpuShmMmap::create(
                &dev,
                MemoryBackendId::new(std::process::id(), 4),
                MIN_BACKEND_SIZE,
                "ctp_test_heap_bounds",
                0,
            )
            .expect("managed segment");
            let cap = seg.data_capacity();

            let mut buf = vec![0u8; 16];
            assert!(seg.read_heap(cap - 8, &mut buf).is_err());
            assert!(seg.write_heap(cap, &[1u8; 8]).is_err());
            assert!(seg.write_heap(usize::MAX, &[1u8; 8]).is_err());
            assert!(seg.read_heap(0, &mut []).is_ok());
        }

        /// Device-only backend: host reaches the data region via memcpy.
        #[test]
        fn gpu_malloc_data_roundtrip() {
            let Some(dev) = device_or_skip("gpu_malloc_data_roundtrip") else {
                return;
            };
            let id = MemoryBackendId::new(std::process::id(), 5);
            let mut seg = GpuMalloc::create(&dev, id, MIN_BACKEND_SIZE, "ctp_test_devmem", 0)
                .expect("device segment");

            assert!(seg.is_owner());
            assert_eq!(seg.data_capacity(), MIN_BACKEND_SIZE - BACKEND_HEADER_SIZE);
            assert_eq!(seg.data_device_ptr(), seg.device_ptr() + 65536);

            // Header was memcpy'd into device memory and reads back intact.
            let h = seg.header().expect("header");
            assert_eq!(h.id, id);
            assert_eq!(h.data_capacity, (MIN_BACKEND_SIZE - BACKEND_HEADER_SIZE) as u64);
            assert!(h.is_initialized() && h.is_owner());

            let pattern: Vec<u8> = (0..2048u32).map(|i| (i % 253) as u8).collect();
            seg.write_data(64, &pattern).unwrap();
            let mut back = vec![0u8; 2048];
            seg.read_data(64, &mut back).unwrap();
            assert_eq!(back, pattern);

            // Untouched bytes are still zero (the region was memset).
            let mut zeros = vec![0xFFu8; 64];
            seg.read_data(0, &mut zeros).unwrap();
            assert!(zeros.iter().all(|&b| b == 0));
        }

        /// Out-of-range access is a returned error, not a panic or UB.
        #[test]
        fn gpu_malloc_rejects_out_of_bounds() {
            let Some(dev) = device_or_skip("gpu_malloc_rejects_out_of_bounds") else {
                return;
            };
            let mut seg = GpuMalloc::create(
                &dev,
                MemoryBackendId::new(std::process::id(), 6),
                MIN_BACKEND_SIZE,
                "ctp_test_bounds",
                0,
            )
            .expect("device segment");
            let cap = seg.data_capacity();

            let mut buf = vec![0u8; 16];
            assert!(seg.read_data(cap - 8, &mut buf).is_err());
            assert!(seg.write_data(cap, &[1u8; 8]).is_err());
            assert!(seg.write_data(usize::MAX, &[1u8; 8]).is_err());
            // The last valid byte still works.
            assert!(seg.write_data(cap - 1, &[0xAB]).is_ok());
            let mut one = [0u8; 1];
            seg.read_data(cap - 1, &mut one).unwrap();
            assert_eq!(one[0], 0xAB);
            // Empty transfers are no-ops.
            assert!(seg.read_data(0, &mut []).is_ok());
        }

        /// The IPC wire format is pure data — no driver, no device.
        #[test]
        fn ipc_handle_bytes_round_trip() {
            let bytes = [7u8; 64];
            let h = GpuIpcMemHandle::from_bytes(bytes);
            assert_eq!(h.as_bytes(), &bytes);
            assert!(format!("{h:?}").contains("64 bytes"));
        }

        /// IPC is platform-gated: Linux may export a handle, Windows must
        /// refuse *without calling the driver* (module doc, divergence 5 —
        /// calling it there faults a concurrent `cuMemFree`).
        #[test]
        fn gpu_malloc_ipc_handle_is_platform_gated() {
            let Some(dev) = device_or_skip("gpu_malloc_ipc_handle_is_platform_gated") else {
                return;
            };
            let seg = GpuMalloc::create(
                &dev,
                MemoryBackendId::new(std::process::id(), 7),
                MIN_BACKEND_SIZE,
                "ctp_test_ipc",
                0,
            )
            .expect("device segment");

            let res = seg.ipc_handle();

            #[cfg(windows)]
            {
                let e = res.expect_err("Windows must refuse CUDA IPC").to_string();
                assert!(e.contains("not supported on Windows"), "{e}");
                eprintln!("ctp-gpu-backends: IPC correctly refused on Windows");
            }
            #[cfg(unix)]
            match res {
                Ok(h) => {
                    // Round-trip the wire form the C++ ships between processes.
                    let bytes = *h.as_bytes();
                    assert_eq!(GpuIpcMemHandle::from_bytes(bytes).as_bytes(), &bytes);
                    eprintln!("ctp-gpu-backends: CUDA IPC supported here");
                }
                Err(e) => eprintln!("ctp-gpu-backends: CUDA IPC unavailable ({e})"),
            }
        }

        /// Managed memory has no IPC handle by design, which is why
        /// [`GpuShmMmap`] is documented process-local and deliberately exposes
        /// no `ipc_handle()`. On Linux, prove the driver agrees; on Windows the
        /// IPC entry points are never called at all (module doc, divergence 5).
        #[test]
        #[cfg(unix)]
        fn managed_segment_is_process_local() {
            let Some(dev) = device_or_skip("managed_segment_is_process_local") else {
                return;
            };
            let seg = GpuShmMmap::create(
                &dev,
                MemoryBackendId::new(std::process::id(), 8),
                MIN_BACKEND_SIZE,
                "ctp_test_local",
                0,
            )
            .expect("managed segment");

            let d = driver().unwrap();
            if let Some(get) = d.ipc_get_mem_handle {
                let mut h = GpuIpcMemHandle {
                    reserved: [0u8; CU_IPC_HANDLE_SIZE],
                };
                // SAFETY: valid out-pointer; `device_ptr()` is a live managed
                // allocation. The call is expected to fail — that is the point.
                let rc = unsafe { get(&mut h, seg.device_ptr()) };
                assert_ne!(rc, CUDA_SUCCESS, "managed memory must not export an IPC handle");
            }
        }

        /// Many segments alive at once, each independently freed on drop.
        #[test]
        fn segments_are_independent_and_freed() {
            let Some(dev) = device_or_skip("segments_are_independent_and_freed") else {
                return;
            };
            let mut segs: Vec<GpuShmMmap> = (0..4)
                .map(|i| {
                    GpuShmMmap::create(
                        &dev,
                        MemoryBackendId::new(std::process::id(), 100 + i),
                        MIN_BACKEND_SIZE,
                        "ctp_test_multi",
                        0,
                    )
                    .expect("managed segment")
                })
                .collect();

            for (i, seg) in segs.iter_mut().enumerate() {
                seg.write_heap(0, &[i as u8]).unwrap();
            }
            for (i, seg) in segs.iter().enumerate() {
                let mut b = [0u8; 1];
                seg.read_heap(0, &mut b).unwrap();
                assert_eq!(b[0], i as u8, "segments must not alias");
                assert_eq!(seg.id().minor, 100 + i as u32);
            }
            // Dropping in a loop must not double-free or leak.
            drop(segs);

            // Allocation still works afterwards.
            let seg = GpuShmMmap::create(
                &dev,
                MemoryBackendId::root(),
                MIN_BACKEND_SIZE,
                "ctp_test_after",
                0,
            )
            .expect("managed segment after drops");
            let mut b = [0xFFu8; 1];
            seg.read_heap(0, &mut b).unwrap();
            assert_eq!(b[0], 0);
        }
    }
}
