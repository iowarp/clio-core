// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// CUDA driver API + NVRTC bindings (hand-rolled, dependency-free) and the
// safe wrappers over them. Kernels are compiled from CUDA C++ source via
// NVRTC at runtime, targeting the LOCAL device's exact SM (CUBIN), then
// loaded and launched through the driver API. Compiling on the machine
// that owns the GPU sidesteps toolkit arch floors (CUDA 13 dropped
// pre-Turing) and driver-JIT PTX-version skew, so one binary serves the
// devcontainer's CUDA 12.6 and the host's 13.3/Blackwell alike.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};

type CuResult = c_int;
type NvrtcResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuDeviceptr = u64;
type NvrtcProgram = *mut c_void;

const CU_MEM_ATTACH_GLOBAL: c_uint = 1;

extern "C" {
    fn cuInit(flags: c_uint) -> CuResult;
    fn cuDeviceGetCount(count: *mut c_int) -> CuResult;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int) -> CuResult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CuContext, dev: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRelease_v2(dev: CuDevice) -> CuResult;
    fn cuCtxSetCurrent(ctx: CuContext) -> CuResult;
    fn cuCtxSynchronize() -> CuResult;
    fn cuGetErrorName(error: CuResult, name: *mut *const c_char) -> CuResult;

    fn cuMemAlloc_v2(dptr: *mut CuDeviceptr, size: usize) -> CuResult;
    fn cuMemAllocManaged(dptr: *mut CuDeviceptr, size: usize, flags: c_uint) -> CuResult;
    fn cuMemFree_v2(dptr: CuDeviceptr) -> CuResult;
    fn cuMemcpyHtoD_v2(dst: CuDeviceptr, src: *const c_void, size: usize) -> CuResult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CuDeviceptr, size: usize) -> CuResult;

    fn cuModuleLoadData(module: *mut CuModule, image: *const c_void) -> CuResult;
    fn cuModuleUnload(module: CuModule) -> CuResult;
    fn cuModuleGetFunction(f: *mut CuFunction, module: CuModule, name: *const c_char) -> CuResult;
    #[allow(clippy::too_many_arguments)]
    fn cuLaunchKernel(
        f: CuFunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CuStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CuResult;

    fn nvrtcCreateProgram(
        prog: *mut NvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> NvrtcResult;
    fn nvrtcCompileProgram(
        prog: NvrtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> NvrtcResult;
    fn nvrtcGetPTXSize(prog: NvrtcProgram, size: *mut usize) -> NvrtcResult;
    fn nvrtcGetPTX(prog: NvrtcProgram, ptx: *mut c_char) -> NvrtcResult;
    fn nvrtcGetProgramLogSize(prog: NvrtcProgram, size: *mut usize) -> NvrtcResult;
    fn nvrtcGetProgramLog(prog: NvrtcProgram, log: *mut c_char) -> NvrtcResult;
    fn nvrtcDestroyProgram(prog: *mut NvrtcProgram) -> NvrtcResult;
    fn nvrtcGetCUBINSize(prog: NvrtcProgram, size: *mut usize) -> NvrtcResult;
    fn nvrtcGetCUBIN(prog: NvrtcProgram, cubin: *mut c_char) -> NvrtcResult;
    fn cuDeviceGetAttribute(value: *mut c_int, attrib: c_int, dev: CuDevice) -> CuResult;
}

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

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

fn cu_check(rc: CuResult, what: &str) -> Result<(), GpuError> {
    if rc == 0 {
        return Ok(());
    }
    let mut name: *const c_char = std::ptr::null();
    // SAFETY: out-pointer is valid; on failure `name` stays null.
    unsafe { cuGetErrorName(rc, &mut name) };
    let err = if name.is_null() {
        format!("{what}: CUDA error {rc}")
    } else {
        // SAFETY: driver returns a static NUL-terminated string.
        format!(
            "{what}: {}",
            unsafe { CStr::from_ptr(name) }.to_string_lossy()
        )
    };
    Err(GpuError(err))
}

/// True if the driver initializes and at least one device exists.
pub fn driver_available() -> bool {
    // SAFETY: cuInit/cuDeviceGetCount have no preconditions.
    unsafe {
        if cuInit(0) != 0 {
            return false;
        }
        let mut n = 0;
        cuDeviceGetCount(&mut n) == 0 && n > 0
    }
}

// ---------------------------------------------------------------------------
// Gpu: device + primary context (GpuApi::SetDevice analog)
// ---------------------------------------------------------------------------

pub struct Gpu {
    device: CuDevice,
    _ctx: CuContext,
}

impl Gpu {
    /// Initialize the driver and bind the primary context of `ordinal`.
    pub fn new(ordinal: i32) -> Result<Self, GpuError> {
        // SAFETY: standard driver-API bring-up sequence; each call checked.
        unsafe {
            cu_check(cuInit(0), "cuInit")?;
            let mut device: CuDevice = 0;
            cu_check(cuDeviceGet(&mut device, ordinal), "cuDeviceGet")?;
            let mut ctx: CuContext = std::ptr::null_mut();
            cu_check(
                cuDevicePrimaryCtxRetain(&mut ctx, device),
                "cuDevicePrimaryCtxRetain",
            )?;
            cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            Ok(Self { device, _ctx: ctx })
        }
    }

    pub fn device_name(&self) -> String {
        let mut buf = [0i8 as c_char; 256];
        // SAFETY: valid buffer + length for the call.
        if unsafe { cuDeviceGetName(buf.as_mut_ptr(), buf.len() as c_int, self.device) } != 0 {
            return String::new();
        }
        // SAFETY: driver NUL-terminates within len.
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// `GpuApi::Synchronize` analog.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        // SAFETY: context bound in new().
        cu_check(unsafe { cuCtxSynchronize() }, "cuCtxSynchronize")
    }

    /// Compute capability of this device, e.g. (12, 0) for Blackwell.
    pub fn compute_capability(&self) -> (i32, i32) {
        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        // SAFETY: valid out-pointers + attribute ids.
        unsafe {
            cuDeviceGetAttribute(
                &mut major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self.device,
            );
            cuDeviceGetAttribute(
                &mut minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self.device,
            );
        }
        (major, minor)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: releasing the primary ctx we retained.
        unsafe { cuDevicePrimaryCtxRelease_v2(self.device) };
    }
}

// ---------------------------------------------------------------------------
// DeviceBuffer: Malloc/MallocManaged/Memcpy analogs
// ---------------------------------------------------------------------------

pub struct DeviceBuffer {
    ptr: CuDeviceptr,
    len: usize,
}

impl DeviceBuffer {
    /// `GpuApi::Malloc` analog (device memory).
    pub fn alloc(_gpu: &Gpu, len: usize) -> Result<Self, GpuError> {
        let mut ptr: CuDeviceptr = 0;
        // SAFETY: valid out-pointer; nonzero size enforced below.
        cu_check(unsafe { cuMemAlloc_v2(&mut ptr, len.max(1)) }, "cuMemAlloc")?;
        Ok(Self { ptr, len })
    }

    /// `GpuApi::MallocManaged` analog (unified memory).
    pub fn alloc_managed(_gpu: &Gpu, len: usize) -> Result<Self, GpuError> {
        let mut ptr: CuDeviceptr = 0;
        // SAFETY: valid out-pointer.
        cu_check(
            unsafe { cuMemAllocManaged(&mut ptr, len.max(1), CU_MEM_ATTACH_GLOBAL) },
            "cuMemAllocManaged",
        )?;
        Ok(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer (for kernel parameters).
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// `GpuApi::Memcpy` host→device.
    pub fn copy_from_host<T: Copy>(&mut self, src: &[T]) -> Result<(), GpuError> {
        let bytes = std::mem::size_of_val(src);
        if bytes > self.len {
            return Err(GpuError("copy_from_host: src larger than buffer".into()));
        }
        // SAFETY: src is a valid host slice of `bytes`; dst has capacity.
        cu_check(
            unsafe { cuMemcpyHtoD_v2(self.ptr, src.as_ptr() as *const c_void, bytes) },
            "cuMemcpyHtoD",
        )
    }

    /// `GpuApi::Memcpy` device→host.
    pub fn copy_to_host<T: Copy>(&self, dst: &mut [T]) -> Result<(), GpuError> {
        let bytes = std::mem::size_of_val(dst);
        if bytes > self.len {
            return Err(GpuError("copy_to_host: dst larger than buffer".into()));
        }
        // SAFETY: dst is a valid mutable host slice of `bytes`.
        cu_check(
            unsafe { cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, self.ptr, bytes) },
            "cuMemcpyDtoH",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: ptr was returned by cuMemAlloc*/cuMemAllocManaged.
            unsafe { cuMemFree_v2(self.ptr) };
        }
    }
}

// ---------------------------------------------------------------------------
// Module + Kernel: NVRTC JIT of the wrapped CUDA C++ source
// ---------------------------------------------------------------------------

pub struct Module {
    module: CuModule,
}

impl Module {
    /// JIT-compile CUDA C++ `src` for the LOCAL device via NVRTC and load
    /// it. Compilation happens at runtime on the machine that owns the GPU,
    /// so the target is simply the device's own SM (`sm_XY` → CUBIN) — this
    /// sidesteps both toolkit arch floors (CUDA 13 dropped pre-Turing) and
    /// driver-JIT PTX-version skew (a 13.x NVRTC emits PTX a 12.x-era
    /// driver cannot load). CTP_GPU_ARCH overrides the target if ever
    /// needed (a `compute_XY` value produces PTX instead of CUBIN).
    pub fn compile(gpu: &Gpu, src: &str) -> Result<Self, GpuError> {
        let arch = std::env::var("CTP_GPU_ARCH").unwrap_or_else(|_| {
            let (major, minor) = gpu.compute_capability();
            format!("sm_{major}{minor}")
        });
        let want_ptx = arch.starts_with("compute_");
        let src_c = CString::new(src).map_err(|_| GpuError("NUL in kernel source".into()))?;
        let name_c = CString::new("ctp_kernels.cu").unwrap();
        let arch_opt = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        let opts = [arch_opt.as_ptr()];

        // SAFETY: NVRTC calls use valid pointers throughout; program handle
        // is destroyed on every path.
        unsafe {
            let mut prog: NvrtcProgram = std::ptr::null_mut();
            if nvrtcCreateProgram(
                &mut prog,
                src_c.as_ptr(),
                name_c.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            ) != 0
            {
                return Err(GpuError("nvrtcCreateProgram failed".into()));
            }

            let rc = nvrtcCompileProgram(prog, opts.len() as c_int, opts.as_ptr());
            if rc != 0 {
                // Surface the compile log — kernel authors need the errors.
                let mut log_size = 0usize;
                nvrtcGetProgramLogSize(prog, &mut log_size);
                let mut log = vec![0u8; log_size.max(1)];
                nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char);
                nvrtcDestroyProgram(&mut prog);
                let log = String::from_utf8_lossy(&log).into_owned();
                return Err(GpuError(format!("NVRTC compile failed:\n{log}")));
            }

            let mut image: Vec<u8>;
            if want_ptx {
                let mut size = 0usize;
                nvrtcGetPTXSize(prog, &mut size);
                image = vec![0u8; size.max(1)];
                nvrtcGetPTX(prog, image.as_mut_ptr() as *mut c_char);
            } else {
                let mut size = 0usize;
                nvrtcGetCUBINSize(prog, &mut size);
                image = vec![0u8; size.max(1)];
                nvrtcGetCUBIN(prog, image.as_mut_ptr() as *mut c_char);
            }
            nvrtcDestroyProgram(&mut prog);

            let mut module: CuModule = std::ptr::null_mut();
            cu_check(
                cuModuleLoadData(&mut module, image.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
            Ok(Self { module })
        }
    }

    /// Look up an `extern "C" __global__` kernel by name.
    pub fn kernel(&self, name: &str) -> Result<Kernel<'_>, GpuError> {
        let name_c = CString::new(name).map_err(|_| GpuError("NUL in kernel name".into()))?;
        let mut f: CuFunction = std::ptr::null_mut();
        // SAFETY: module is loaded; out-pointer valid.
        cu_check(
            unsafe { cuModuleGetFunction(&mut f, self.module, name_c.as_ptr()) },
            "cuModuleGetFunction",
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
            // SAFETY: module was loaded by cuModuleLoadData.
            unsafe { cuModuleUnload(self.module) };
        }
    }
}

pub struct Kernel<'m> {
    f: CuFunction,
    _module: std::marker::PhantomData<&'m Module>,
}

impl Kernel<'_> {
    /// Launch with a 1-D grid.
    ///
    /// # Safety
    /// `params` must match the kernel's signature: one pointer per formal
    /// parameter, each pointing at a live value of the right type — the
    /// same contract as raw `cuLaunchKernel`.
    pub unsafe fn launch_1d(
        &self,
        grid: u32,
        block: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), GpuError> {
        // SAFETY: caller upholds the parameter contract; function is live
        // for the module's lifetime (PhantomData ties the borrow).
        cu_check(
            unsafe {
                cuLaunchKernel(
                    self.f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end proof of the wrapped-kernel path: NVRTC-compile the CUDA
    /// C++ source, run ctp_vector_add on the device, verify on host.
    /// Skips (with a notice) when no CUDA device is present, so the suite
    /// stays green on GPU-less CI runners.
    #[test]
    fn vector_add_on_device() {
        if !driver_available() {
            eprintln!("ctp-gpu: no CUDA device available; skipping");
            return;
        }
        let gpu = Gpu::new(0).unwrap();
        eprintln!("ctp-gpu: running on {}", gpu.device_name());

        let module = Module::compile(&gpu, crate::KERNEL_SOURCE).unwrap();
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
        let grid = ((N as u32) + block - 1) / block;
        unsafe { kernel.launch_1d(grid, block, &mut params).unwrap() };
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
        if !driver_available() {
            eprintln!("ctp-gpu: no CUDA device available; skipping");
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
}
