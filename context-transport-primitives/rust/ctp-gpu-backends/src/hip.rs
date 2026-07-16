// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! HIP / ROCm runtime + HIPRTC bindings (hand-rolled, dependency-free) and
//! the safe wrappers over them. This is the ROCm sibling of
//! `ctp-gpu/src/cuda.rs` and deliberately mirrors its structure, naming and
//! safety style.
//!
//! # What is wrapped
//!
//! * **HIP runtime API** — `libamdhip64.so` (Linux) / `amdhip64.dll`
//!   (Windows), i.e. the C API declared by ROCm's `hip/hip_runtime_api.h`.
//!   Covers the `ctp::GpuApi` (`clio_ctp/util/gpu_api.h`) `CTP_ENABLE_ROCM`
//!   branch: `hipSetDevice`, `hipGetDeviceCount`, `hipDeviceSynchronize`,
//!   `hipStreamCreate`/`hipStreamCreateWithFlags`/`hipStreamSynchronize`/
//!   `hipStreamQuery`/`hipStreamDestroy`, `hipMalloc`, `hipMallocManaged`,
//!   `hipFree`, `hipMemcpy`, `hipMemcpyAsync`, `hipGetErrorString`.
//! * **HIP module API** — `hipModuleLoadData`, `hipModuleGetFunction`,
//!   `hipModuleLaunchKernel`, `hipModuleUnload`.
//! * **HIPRTC** — `hiprtcCreateProgram`/`hiprtcCompileProgram`/
//!   `hiprtcGetCodeSize`/`hiprtcGetCode`/`hiprtcGetProgramLog*`/
//!   `hiprtcDestroyProgram`, for runtime JIT of HIP C++ kernel source.
//!
//! Per the migration policy kernels REMAIN HIP C++ ([`KERNEL_SOURCE`]); this
//! module only wraps them for Rust callers.
//!
//! # Version / ABI assumptions
//!
//! These are the values the C++ side gets from ROCm's headers by symbol. A
//! hand-rolled binding must hard-code them, so they are pinned here:
//!
//! * ROCm **5.x / 6.x**, 64-bit only (the only configurations ROCm ships).
//!   Handles (`hipStream_t`, `hipModule_t`, `hipFunction_t`, `hiprtcProgram`)
//!   are opaque pointers; `hipDevice_t` and `hipError_t` are `int`-sized.
//! * `hipSuccess == 0`, `hipErrorNotReady == 600`, `HIPRTC_SUCCESS == 0`.
//! * `hipMemcpyDefault == 4`, `hipStreamNonBlocking == 0x01`,
//!   `hipMemAttachGlobal == 0x01`.
//! * `hipMallocManaged` is C-linkage with **three** parameters — the header's
//!   `__dparm(hipMemAttachGlobal)` default only exists in C++, so the flag is
//!   passed explicitly to match what the C++ CTP code gets implicitly.
//! * Device pointers are returned to safe code as `u64` (see
//!   [`DeviceBuffer::device_ptr`]), which mirrors `cuda.rs`'s `CUdeviceptr`
//!   and is exact on any 64-bit ROCm target.
//!
//! **Linkage**: the symbols below resolve from `amdhip64` (which also exports
//! the `hiprtc*` entry points on Linux; the Windows HIP SDK ships an
//! additional `hiprtc` import library). Like `cuda.rs`, this module declares
//! bare `extern "C"` blocks and leaves `-l`/`-L` to the crate's build script
//! rather than guessing library names via `#[link]`.
//!
//! NOTE: unlike `ctp-gpu`, this crate currently has **no `build.rs`**, so
//! while `cargo build --features hip` produces an rlib fine (an rlib does not
//! link), anything that actually links — including `cargo test --features
//! hip` — fails with unresolved `hip*`/`hiprtc*` externals until a build
//! script emits `cargo:rustc-link-lib=amdhip64` (plus `hiprtc` on Windows)
//! and a `cargo:rustc-link-search=native=$ROCM_PATH/lib` path, mirroring
//! `ctp-gpu/build.rs`. That script is out of this module's scope.
//!
//! # Divergence from the C++ wrapper (`gpu_api.h`, ROCm branch)
//!
//! 1. **Errors are values.** `HIP_ERROR_CHECK` logs FATAL and aborts; every
//!    wrapper here returns [`GpuError`] instead. Nothing panics on a library
//!    error.
//! 2. **Ownership is RAII.** `GpuApi::Malloc`/`CreateStream` hand back raw
//!    handles the caller must pass to `Free`/`DestroyStream`; [`DeviceBuffer`],
//!    [`Stream`] and [`Module`] free themselves on `Drop`.
//! 3. **`GetDeviceCount` does not swallow errors.** The C++ version maps any
//!    failure to `0`; here [`device_count`] returns `Result` and the
//!    swallowing probe is [`runtime_available`].
//! 4. **Async copies are `unsafe`.** `GpuApi::MemcpyAsync` is unchecked; a
//!    *safe* Rust fn cannot take a host slice for an async copy (dropping it
//!    before the stream syncs would be UB), so those are `unsafe fn` with an
//!    explicit "keep `src` alive until synchronize" contract.
//! 5. **JIT differs from `cuda.rs` by necessity.** NVRTC splits PTX vs CUBIN
//!    and needs an `sm_XY` computed from the compute capability; HIPRTC has a
//!    single `hiprtcGetCode` code-object output and defaults to the *current*
//!    device's arch, so no capability query is performed. `CTP_GPU_ARCH`
//!    (e.g. `gfx90a`) overrides it via `--offload-arch=`, mirroring the same
//!    env var in `cuda.rs`.
//! 6. **`hipGetErrorString` returns `const char*` directly**, unlike
//!    `cuGetErrorName`'s status + out-param, so error formatting is simpler.
//! 7. **Out of scope here** (present in `gpu_api.h`, not part of the #756 HIP
//!    FFI surface): IPC mem handles, host register/unregister, pinned host
//!    alloc, memset, and pointer-attribute queries.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};

type HipResult = c_int;
type HiprtcResult = c_int;
type HipDevice = c_int;
type HipStreamT = *mut c_void;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HiprtcProgram = *mut c_void;

const HIP_SUCCESS: HipResult = 0;
const HIP_ERROR_NOT_READY: HipResult = 600;
const HIPRTC_SUCCESS: HiprtcResult = 0;

/// `hipMemcpyKind::hipMemcpyDefault` — direction inferred from the pointers,
/// exactly as the C++ `GpuApi::Memcpy`/`MemcpyAsync` use it.
const HIP_MEMCPY_DEFAULT: c_int = 4;
/// `hipStreamNonBlocking` — matches `GpuApi::CreateStream`.
const HIP_STREAM_NON_BLOCKING: c_uint = 0x01;
/// `hipMemAttachGlobal` — the header default for `hipMallocManaged`.
const HIP_MEM_ATTACH_GLOBAL: c_uint = 0x01;

extern "C" {
    fn hipInit(flags: c_uint) -> HipResult;
    fn hipGetDeviceCount(count: *mut c_int) -> HipResult;
    fn hipSetDevice(device_id: c_int) -> HipResult;
    fn hipDeviceGet(device: *mut HipDevice, ordinal: c_int) -> HipResult;
    fn hipDeviceGetName(name: *mut c_char, len: c_int, device: HipDevice) -> HipResult;
    fn hipDeviceSynchronize() -> HipResult;
    fn hipGetLastError() -> HipResult;
    fn hipGetErrorString(hip_error: HipResult) -> *const c_char;

    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipResult;
    fn hipMallocManaged(dev_ptr: *mut *mut c_void, size: usize, flags: c_uint) -> HipResult;
    fn hipFree(ptr: *mut c_void) -> HipResult;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size_bytes: usize, kind: c_int)
        -> HipResult;
    fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: c_int,
        stream: HipStreamT,
    ) -> HipResult;

    fn hipStreamCreate(stream: *mut HipStreamT) -> HipResult;
    fn hipStreamCreateWithFlags(stream: *mut HipStreamT, flags: c_uint) -> HipResult;
    fn hipStreamSynchronize(stream: HipStreamT) -> HipResult;
    fn hipStreamQuery(stream: HipStreamT) -> HipResult;
    fn hipStreamDestroy(stream: HipStreamT) -> HipResult;

    fn hipModuleLoadData(module: *mut HipModule, image: *const c_void) -> HipResult;
    fn hipModuleUnload(module: HipModule) -> HipResult;
    fn hipModuleGetFunction(
        function: *mut HipFunction,
        module: HipModule,
        kname: *const c_char,
    ) -> HipResult;
    #[allow(clippy::too_many_arguments)]
    fn hipModuleLaunchKernel(
        f: HipFunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: HipStreamT,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> HipResult;

    fn hiprtcCreateProgram(
        prog: *mut HiprtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> HiprtcResult;
    fn hiprtcCompileProgram(
        prog: HiprtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> HiprtcResult;
    fn hiprtcGetCodeSize(prog: HiprtcProgram, code_size_ret: *mut usize) -> HiprtcResult;
    fn hiprtcGetCode(prog: HiprtcProgram, code: *mut c_char) -> HiprtcResult;
    fn hiprtcGetProgramLogSize(prog: HiprtcProgram, log_size_ret: *mut usize) -> HiprtcResult;
    fn hiprtcGetProgramLog(prog: HiprtcProgram, log: *mut c_char) -> HiprtcResult;
    fn hiprtcDestroyProgram(prog: *mut HiprtcProgram) -> HiprtcResult;
    fn hiprtcGetErrorString(result: HiprtcResult) -> *const c_char;
}

/// The wrapped HIP C++ kernel source (kernels stay C++ per the migration
/// policy; see `rust/MIGRATION.md`). This is the HIP mirror of
/// `ctp-gpu/kernels/ctp_kernels.cu` — kept as a literal rather than
/// `include_str!`-ing across crate roots, so `ctp-gpu-backends` stays
/// self-contained until the two crates are folded together at integration.
/// HIPRTC pre-includes the HIP device headers, so no `#include` is needed;
/// `extern "C"` keeps the names unmangled for `hipModuleGetFunction`.
pub const KERNEL_SOURCE: &str = r#"
extern "C" __global__ void ctp_vector_add(const float *a, const float *b,
                                          float *out, unsigned long long n) {
  unsigned long long i =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

extern "C" __global__ void ctp_fill_u8(unsigned char *dst, unsigned char val,
                                       unsigned long long n) {
  unsigned long long i =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = val;
  }
}
"#;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GpuError(pub String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuError: {}", self.0)
    }
}
impl std::error::Error for GpuError {}

fn hip_check(rc: HipResult, what: &str) -> Result<(), GpuError> {
    if rc == HIP_SUCCESS {
        return Ok(());
    }
    // SAFETY: hipGetErrorString accepts any hipError_t (unknown codes map to
    // the "hipErrorUnknown" string) and returns a static NUL-terminated
    // string owned by the runtime; we only read it.
    let msg = unsafe { hipGetErrorString(rc) };
    let err = if msg.is_null() {
        format!("{what}: HIP error {rc}")
    } else {
        // SAFETY: non-null static NUL-terminated string from the runtime.
        format!(
            "{what}: {}",
            unsafe { CStr::from_ptr(msg) }.to_string_lossy()
        )
    };
    Err(GpuError(err))
}

fn hiprtc_check(rc: HiprtcResult, what: &str) -> Result<(), GpuError> {
    if rc == HIPRTC_SUCCESS {
        return Ok(());
    }
    // SAFETY: hiprtcGetErrorString accepts any hiprtcResult and returns a
    // static NUL-terminated string.
    let msg = unsafe { hiprtcGetErrorString(rc) };
    let err = if msg.is_null() {
        format!("{what}: HIPRTC error {rc}")
    } else {
        // SAFETY: non-null static NUL-terminated string from HIPRTC.
        format!(
            "{what}: {}",
            unsafe { CStr::from_ptr(msg) }.to_string_lossy()
        )
    };
    Err(GpuError(err))
}

/// Number of visible ROCm devices. `GpuApi::GetDeviceCount` analog, except
/// errors surface instead of collapsing to `0` — see [`runtime_available`]
/// for the C++ swallow-the-error behavior.
pub fn device_count() -> Result<i32, GpuError> {
    let mut n: c_int = 0;
    // SAFETY: valid out-pointer; hipGetDeviceCount has no other precondition.
    hip_check(unsafe { hipGetDeviceCount(&mut n) }, "hipGetDeviceCount")?;
    Ok(n)
}

/// True if the HIP runtime initializes and at least one device exists.
/// Mirrors `cuda::driver_available` and the C++ "no GPU is not fatal" rule:
/// a failed query clears the sticky error and reports "no GPU" so a
/// ROCm-enabled build still runs the CPU path on a driver-less host.
pub fn runtime_available() -> bool {
    // SAFETY: hipInit/hipGetDeviceCount/hipGetLastError have no
    // preconditions; the out-pointer is valid.
    unsafe {
        if hipInit(0) != HIP_SUCCESS {
            hipGetLastError();
            return false;
        }
        let mut n: c_int = 0;
        if hipGetDeviceCount(&mut n) != HIP_SUCCESS {
            hipGetLastError(); // Clear the error state.
            return false;
        }
        n > 0
    }
}

/// Grid size for a 1-D launch of `n` elements at `block` threads/block.
pub fn grid_1d(n: usize, block: u32) -> u32 {
    if block == 0 || n == 0 {
        return 0;
    }
    (n as u64).div_ceil(block as u64) as u32
}

// ---------------------------------------------------------------------------
// Gpu: device selection (GpuApi::SetDevice / Synchronize analog)
// ---------------------------------------------------------------------------

pub struct Gpu {
    device: HipDevice,
}

impl Gpu {
    /// Initialize the HIP runtime and make `ordinal` the current device.
    /// Unlike CUDA there is no explicit primary-context retain: HIP binds the
    /// device's context implicitly on `hipSetDevice`.
    pub fn new(ordinal: i32) -> Result<Self, GpuError> {
        // SAFETY: standard HIP bring-up; every call is checked and the
        // out-pointer is valid.
        unsafe {
            hip_check(hipInit(0), "hipInit")?;
            let mut device: HipDevice = 0;
            hip_check(hipDeviceGet(&mut device, ordinal), "hipDeviceGet")?;
            hip_check(hipSetDevice(ordinal), "hipSetDevice")?;
            Ok(Self { device })
        }
    }

    pub fn device_name(&self) -> String {
        let mut buf = [0 as c_char; 256];
        // SAFETY: valid buffer + matching length; device came from
        // hipDeviceGet.
        if unsafe { hipDeviceGetName(buf.as_mut_ptr(), buf.len() as c_int, self.device) }
            != HIP_SUCCESS
        {
            return String::new();
        }
        // SAFETY: the runtime NUL-terminates within len.
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// `GpuApi::Synchronize()` analog.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        // SAFETY: device selected in new(); no preconditions.
        hip_check(unsafe { hipDeviceSynchronize() }, "hipDeviceSynchronize")
    }
}

// ---------------------------------------------------------------------------
// Stream: GpuApi::CreateStream / Synchronize(stream) / StreamQuery analogs
// ---------------------------------------------------------------------------

pub struct Stream {
    stream: HipStreamT,
}

impl Stream {
    /// `hipStreamCreate` — a blocking stream that serializes with the NULL
    /// stream.
    pub fn new(_gpu: &Gpu) -> Result<Self, GpuError> {
        let mut stream: HipStreamT = std::ptr::null_mut();
        // SAFETY: valid out-pointer; device is current via Gpu.
        hip_check(unsafe { hipStreamCreate(&mut stream) }, "hipStreamCreate")?;
        Ok(Self { stream })
    }

    /// `GpuApi::CreateStream()` analog: a non-blocking stream that does not
    /// serialize against the NULL stream — the flag the C++ side uses so a
    /// copy cannot deadlock behind a kernel parked on the default stream.
    pub fn new_non_blocking(_gpu: &Gpu) -> Result<Self, GpuError> {
        let mut stream: HipStreamT = std::ptr::null_mut();
        // SAFETY: valid out-pointer; HIP_STREAM_NON_BLOCKING is a valid flag.
        hip_check(
            unsafe { hipStreamCreateWithFlags(&mut stream, HIP_STREAM_NON_BLOCKING) },
            "hipStreamCreateWithFlags",
        )?;
        Ok(Self { stream })
    }

    /// `GpuApi::Synchronize(stream)` analog: block until all work submitted
    /// to this stream completes.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        // SAFETY: stream handle is live for &self.
        hip_check(
            unsafe { hipStreamSynchronize(self.stream) },
            "hipStreamSynchronize",
        )
    }

    /// `GpuApi::StreamQuery` analog: non-blocking poll. `Ok(true)` once every
    /// submitted operation has completed, `Ok(false)` while work is in
    /// flight. The "not ready" status is consumed so this stays a pure query
    /// and never trips a later error check.
    pub fn query(&self) -> Result<bool, GpuError> {
        // SAFETY: stream handle is live for &self.
        let rc = unsafe { hipStreamQuery(self.stream) };
        if rc == HIP_SUCCESS {
            return Ok(true);
        }
        if rc == HIP_ERROR_NOT_READY {
            // SAFETY: no preconditions; clears the sticky "not ready".
            unsafe { hipGetLastError() };
            return Ok(false);
        }
        hip_check(rc, "hipStreamQuery")?;
        Ok(true)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            // SAFETY: handle came from hipStreamCreate*; destroyed once,
            // since Stream is not Copy/Clone.
            unsafe { hipStreamDestroy(self.stream) };
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceBuffer: Malloc / MallocManaged / Memcpy analogs
// ---------------------------------------------------------------------------

pub struct DeviceBuffer {
    ptr: *mut c_void,
    len: usize,
}

impl DeviceBuffer {
    /// `GpuApi::Malloc` analog (device memory).
    pub fn alloc(_gpu: &Gpu, len: usize) -> Result<Self, GpuError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer; a zero-size request is rounded to 1 so
        // the returned handle is always freeable.
        hip_check(unsafe { hipMalloc(&mut ptr, len.max(1)) }, "hipMalloc")?;
        Ok(Self { ptr, len })
    }

    /// `GpuApi::MallocManaged` analog (unified memory).
    pub fn alloc_managed(_gpu: &Gpu, len: usize) -> Result<Self, GpuError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer; HIP_MEM_ATTACH_GLOBAL is the header's
        // default flag for hipMallocManaged.
        hip_check(
            unsafe { hipMallocManaged(&mut ptr, len.max(1), HIP_MEM_ATTACH_GLOBAL) },
            "hipMallocManaged",
        )?;
        Ok(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device address (for kernel parameters). Returned as `u64` rather
    /// than a pointer: it is not host-dereferenceable, and this matches the
    /// `CUdeviceptr` shape of `cuda.rs`.
    pub fn device_ptr(&self) -> u64 {
        self.ptr as u64
    }

    /// `GpuApi::Memcpy` host→device (blocking, `hipMemcpyDefault`).
    pub fn copy_from_host<T: Copy>(&mut self, src: &[T]) -> Result<(), GpuError> {
        let bytes = std::mem::size_of_val(src);
        if bytes > self.len {
            return Err(GpuError("copy_from_host: src larger than buffer".into()));
        }
        // SAFETY: src is a live host slice of `bytes`; dst has capacity for
        // it (checked above). hipMemcpy blocks, so src outlives the copy.
        hip_check(
            unsafe { hipMemcpy(self.ptr, src.as_ptr() as *const c_void, bytes, HIP_MEMCPY_DEFAULT) },
            "hipMemcpy(HtoD)",
        )
    }

    /// `GpuApi::Memcpy` device→host (blocking, `hipMemcpyDefault`).
    pub fn copy_to_host<T: Copy>(&self, dst: &mut [T]) -> Result<(), GpuError> {
        let bytes = std::mem::size_of_val(dst);
        if bytes > self.len {
            return Err(GpuError("copy_to_host: dst larger than buffer".into()));
        }
        // SAFETY: dst is a live mutable host slice of `bytes`; src holds at
        // least that many bytes (checked above). hipMemcpy blocks.
        hip_check(
            unsafe {
                hipMemcpy(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    bytes,
                    HIP_MEMCPY_DEFAULT,
                )
            },
            "hipMemcpy(DtoH)",
        )
    }

    /// `GpuApi::MemcpyAsync` host→device on `stream`.
    ///
    /// # Safety
    /// The copy is still in flight when this returns: `src` must remain
    /// allocated and unmodified until [`Stream::synchronize`] (or
    /// [`Stream::query`] returning `true`) reports the stream drained. This
    /// cannot be a safe fn — dropping `src` first would let the device read
    /// freed host memory.
    pub unsafe fn copy_from_host_async<T: Copy>(
        &mut self,
        src: &[T],
        stream: &Stream,
    ) -> Result<(), GpuError> {
        let bytes = std::mem::size_of_val(src);
        if bytes > self.len {
            return Err(GpuError(
                "copy_from_host_async: src larger than buffer".into(),
            ));
        }
        // SAFETY: dst has capacity (checked); the caller guarantees src stays
        // live until the stream is synchronized, per this fn's contract.
        hip_check(
            unsafe {
                hipMemcpyAsync(
                    self.ptr,
                    src.as_ptr() as *const c_void,
                    bytes,
                    HIP_MEMCPY_DEFAULT,
                    stream.stream,
                )
            },
            "hipMemcpyAsync(HtoD)",
        )
    }

    /// `GpuApi::MemcpyAsync` device→host on `stream`.
    ///
    /// # Safety
    /// As [`DeviceBuffer::copy_from_host_async`]: `dst` must stay alive and
    /// untouched until the stream is synchronized — reading it earlier
    /// observes a partially written buffer, and dropping it earlier lets the
    /// device write freed memory.
    pub unsafe fn copy_to_host_async<T: Copy>(
        &self,
        dst: &mut [T],
        stream: &Stream,
    ) -> Result<(), GpuError> {
        let bytes = std::mem::size_of_val(dst);
        if bytes > self.len {
            return Err(GpuError("copy_to_host_async: dst larger than buffer".into()));
        }
        // SAFETY: src holds `bytes` (checked); the caller guarantees dst
        // stays live until the stream is synchronized, per the contract.
        hip_check(
            unsafe {
                hipMemcpyAsync(
                    dst.as_mut_ptr() as *mut c_void,
                    self.ptr,
                    bytes,
                    HIP_MEMCPY_DEFAULT,
                    stream.stream,
                )
            },
            "hipMemcpyAsync(DtoH)",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from hipMalloc/hipMallocManaged and is freed
            // once (DeviceBuffer is not Copy/Clone). hipFree implicitly
            // synchronizes the device, so an in-flight async copy naming this
            // buffer has drained before the memory is released.
            unsafe { hipFree(self.ptr) };
        }
    }
}

// ---------------------------------------------------------------------------
// Module + Kernel: HIPRTC JIT of the wrapped HIP C++ source
// ---------------------------------------------------------------------------

pub struct Module {
    module: HipModule,
}

impl Module {
    /// JIT-compile HIP C++ `src` via HIPRTC and load the resulting code
    /// object. HIPRTC targets the *current* device's arch by default, which
    /// is exactly what we want when compiling on the machine that owns the
    /// GPU; `CTP_GPU_ARCH` (e.g. `gfx90a`, `gfx1100`) overrides it via
    /// `--offload-arch=`, mirroring the same env var in `cuda.rs`.
    ///
    /// Note the NVRTC divergence: there is no PTX/CUBIN split on ROCm — code
    /// objects come out of the single `hiprtcGetCode` entry point.
    pub fn compile(_gpu: &Gpu, src: &str) -> Result<Self, GpuError> {
        let src_c = CString::new(src).map_err(|_| GpuError("NUL in kernel source".into()))?;
        let name_c = CString::new("ctp_kernels.hip").unwrap();

        let arch_opt = match std::env::var("CTP_GPU_ARCH") {
            Ok(arch) => Some(
                CString::new(format!("--offload-arch={arch}"))
                    .map_err(|_| GpuError("NUL in CTP_GPU_ARCH".into()))?,
            ),
            Err(_) => None,
        };
        let opts: Vec<*const c_char> = arch_opt.iter().map(|o| o.as_ptr()).collect();

        // SAFETY: every pointer handed to HIPRTC is valid for the call, the
        // options array matches its length, and the program handle is
        // destroyed on every path out (success, compile failure, or an
        // early return from the checks below).
        unsafe {
            let mut prog: HiprtcProgram = std::ptr::null_mut();
            hiprtc_check(
                hiprtcCreateProgram(
                    &mut prog,
                    src_c.as_ptr(),
                    name_c.as_ptr(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                ),
                "hiprtcCreateProgram",
            )?;

            let rc = hiprtcCompileProgram(prog, opts.len() as c_int, opts.as_ptr());
            if rc != HIPRTC_SUCCESS {
                // Surface the compile log — kernel authors need the errors.
                let mut log_size = 0usize;
                hiprtcGetProgramLogSize(prog, &mut log_size);
                let mut log = vec![0u8; log_size.max(1)];
                hiprtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char);
                hiprtcDestroyProgram(&mut prog);
                let log = String::from_utf8_lossy(&log).into_owned();
                return Err(GpuError(format!("HIPRTC compile failed:\n{log}")));
            }

            let mut size = 0usize;
            if let Err(e) = hiprtc_check(hiprtcGetCodeSize(prog, &mut size), "hiprtcGetCodeSize") {
                hiprtcDestroyProgram(&mut prog);
                return Err(e);
            }
            let mut image = vec![0u8; size.max(1)];
            if let Err(e) = hiprtc_check(
                hiprtcGetCode(prog, image.as_mut_ptr() as *mut c_char),
                "hiprtcGetCode",
            ) {
                hiprtcDestroyProgram(&mut prog);
                return Err(e);
            }
            hiprtcDestroyProgram(&mut prog);

            let mut module: HipModule = std::ptr::null_mut();
            hip_check(
                hipModuleLoadData(&mut module, image.as_ptr() as *const c_void),
                "hipModuleLoadData",
            )?;
            Ok(Self { module })
        }
    }

    /// Look up an `extern "C" __global__` kernel by name.
    pub fn kernel(&self, name: &str) -> Result<Kernel<'_>, GpuError> {
        let name_c = CString::new(name).map_err(|_| GpuError("NUL in kernel name".into()))?;
        let mut f: HipFunction = std::ptr::null_mut();
        // SAFETY: module is loaded; out-pointer and NUL-terminated name are
        // valid for the call.
        hip_check(
            unsafe { hipModuleGetFunction(&mut f, self.module, name_c.as_ptr()) },
            "hipModuleGetFunction",
        )?;
        Ok(Kernel {
            f,
            _module: std::marker::PhantomData,
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.module.is_null() {
            // SAFETY: module came from hipModuleLoadData; unloaded once.
            unsafe { hipModuleUnload(self.module) };
        }
    }
}

pub struct Kernel<'m> {
    f: HipFunction,
    _module: std::marker::PhantomData<&'m Module>,
}

impl Kernel<'_> {
    /// Launch with a 1-D grid on the NULL stream.
    ///
    /// # Safety
    /// `params` must match the kernel's signature: one pointer per formal
    /// parameter, each pointing at a live value of the right type — the same
    /// contract as raw `hipModuleLaunchKernel`. Any device memory named by
    /// those values must stay allocated until the launch completes.
    pub unsafe fn launch_1d(
        &self,
        grid: u32,
        block: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), GpuError> {
        // SAFETY: caller upholds the parameter contract; the null stream is
        // a valid hipStream_t. Function is live for the module's lifetime
        // (PhantomData ties the borrow).
        unsafe { self.launch_1d_raw(grid, block, std::ptr::null_mut(), params) }
    }

    /// Launch with a 1-D grid on `stream`.
    ///
    /// # Safety
    /// As [`Kernel::launch_1d`]; additionally the launch is asynchronous, so
    /// everything `params` points into must stay live until the stream is
    /// synchronized.
    pub unsafe fn launch_1d_on_stream(
        &self,
        grid: u32,
        block: u32,
        stream: &Stream,
        params: &mut [*mut c_void],
    ) -> Result<(), GpuError> {
        // SAFETY: caller upholds the parameter contract; stream.stream is a
        // live handle for &stream.
        unsafe { self.launch_1d_raw(grid, block, stream.stream, params) }
    }

    /// # Safety
    /// See [`Kernel::launch_1d`]; `stream` must be null or a live
    /// `hipStream_t`.
    unsafe fn launch_1d_raw(
        &self,
        grid: u32,
        block: u32,
        stream: HipStreamT,
        params: &mut [*mut c_void],
    ) -> Result<(), GpuError> {
        // SAFETY: caller upholds the parameter and stream contracts; the
        // params array is valid for the call's duration.
        hip_check(
            unsafe {
                hipModuleLaunchKernel(
                    self.f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Runtime-free tests (no ROCm device or driver needed) -------------

    #[test]
    fn grid_1d_covers_all_elements() {
        assert_eq!(grid_1d(0, 256), 0);
        assert_eq!(grid_1d(1, 256), 1);
        assert_eq!(grid_1d(256, 256), 1);
        assert_eq!(grid_1d(257, 256), 2);
        assert_eq!(grid_1d(1 << 20, 256), 4096);
        // Degenerate block size must not divide by zero.
        assert_eq!(grid_1d(10, 0), 0);
        // Every element is covered: grid * block >= n.
        for n in [1usize, 255, 256, 1000, 65_537] {
            assert!(grid_1d(n, 256) as usize * 256 >= n);
        }
    }

    #[test]
    fn kernel_source_is_unmangled_hip_cpp() {
        // Kernels stay HIP C++ and must be extern "C" for name lookup.
        assert!(KERNEL_SOURCE.contains(r#"extern "C" __global__ void ctp_vector_add"#));
        assert!(KERNEL_SOURCE.contains(r#"extern "C" __global__ void ctp_fill_u8"#));
        // HIPRTC rejects a source containing an interior NUL; CString::new in
        // compile() would error out, so prove the shipped source is clean.
        assert!(CString::new(KERNEL_SOURCE).is_ok());
    }

    #[test]
    fn error_displays_with_context() {
        let e = GpuError("hipMalloc: out of memory".into());
        assert_eq!(e.to_string(), "GpuError: hipMalloc: out of memory");
    }

    // --- Device tests: skip (like ctp-gpu) when no ROCm device exists -----

    /// End-to-end proof of the wrapped-kernel path: HIPRTC-compile the HIP
    /// C++ source, run ctp_vector_add on the device, verify on host.
    #[test]
    fn vector_add_on_device() {
        if !runtime_available() {
            eprintln!("ctp-gpu-backends: no ROCm device available; skipping");
            return;
        }
        let gpu = Gpu::new(0).unwrap();
        eprintln!("ctp-gpu-backends: running on {}", gpu.device_name());

        let module = Module::compile(&gpu, KERNEL_SOURCE).unwrap();
        let kernel = module.kernel("ctp_vector_add").unwrap();

        const N: usize = 1 << 20;
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();

        let mut da = DeviceBuffer::alloc(&gpu, N * 4).unwrap();
        let mut db = DeviceBuffer::alloc(&gpu, N * 4).unwrap();
        let dout = DeviceBuffer::alloc(&gpu, N * 4).unwrap();
        da.copy_from_host(&a).unwrap();
        db.copy_from_host(&b).unwrap();

        let mut pa = da.device_ptr();
        let mut pb = db.device_ptr();
        let mut pout = dout.device_ptr();
        let mut n = N as u64;
        let mut params = [
            &mut pa as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut pout as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        // SAFETY: params matches ctp_vector_add(const float*, const float*,
        // float*, unsigned long long); all buffers outlive the synchronize.
        unsafe {
            kernel
                .launch_1d(grid_1d(N, block), block, &mut params)
                .unwrap()
        };
        gpu.synchronize().unwrap();

        let mut out = vec![0f32; N];
        dout.copy_to_host(&mut out).unwrap();
        for i in (0..N).step_by(97_213) {
            assert_eq!(out[i], (i + 2 * i) as f32);
        }
        assert_eq!(out[N - 1], ((N - 1) + 2 * (N - 1)) as f32);
    }

    #[test]
    fn managed_alloc_roundtrip() {
        if !runtime_available() {
            eprintln!("ctp-gpu-backends: no ROCm device available; skipping");
            return;
        }
        let gpu = Gpu::new(0).unwrap();
        let mut buf = DeviceBuffer::alloc_managed(&gpu, 4096).unwrap();
        let data: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        buf.copy_from_host(&data).unwrap();
        let mut back = vec![0u8; 4096];
        buf.copy_to_host(&mut back).unwrap();
        assert_eq!(data, back);
    }

    /// The stream path the C++ compressors use: async H2D, launch, async D2H,
    /// synchronize (see compress/cusz.h's create → memcpyAsync → sync shape).
    #[test]
    fn stream_async_roundtrip() {
        if !runtime_available() {
            eprintln!("ctp-gpu-backends: no ROCm device available; skipping");
            return;
        }
        let gpu = Gpu::new(0).unwrap();
        let stream = Stream::new_non_blocking(&gpu).unwrap();
        let src: Vec<u8> = (0..8192u32).map(|i| (i % 253) as u8).collect();
        let mut buf = DeviceBuffer::alloc(&gpu, src.len()).unwrap();
        let mut back = vec![0u8; src.len()];
        // SAFETY: src and back outlive the synchronize below, so the device
        // never touches freed host memory.
        unsafe {
            buf.copy_from_host_async(&src, &stream).unwrap();
            buf.copy_to_host_async(&mut back, &stream).unwrap();
        }
        stream.synchronize().unwrap();
        assert!(stream.query().unwrap(), "drained stream must report ready");
        assert_eq!(src, back);
    }

    #[test]
    fn device_count_matches_availability() {
        if !runtime_available() {
            eprintln!("ctp-gpu-backends: no ROCm device available; skipping");
            return;
        }
        assert!(device_count().unwrap() > 0);
    }

    #[test]
    fn compile_error_surfaces_the_log() {
        if !runtime_available() {
            eprintln!("ctp-gpu-backends: no ROCm device available; skipping");
            return;
        }
        let gpu = Gpu::new(0).unwrap();
        // Errors are returned, never fatal — the key divergence from
        // HIP_ERROR_CHECK. (Matched rather than unwrap_err()'d: Module is an
        // owning handle and deliberately not Debug.)
        match Module::compile(&gpu, "this is not valid HIP C++") {
            Ok(_) => panic!("invalid HIP C++ must not compile"),
            Err(e) => assert!(e.to_string().contains("HIPRTC compile failed")),
        }
    }
}
