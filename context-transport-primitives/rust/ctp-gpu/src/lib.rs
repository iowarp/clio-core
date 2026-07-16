// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// GPU extension wrapper. Policy: GPU kernels REMAIN CUDA C++
// (kernels/ctp_kernels.cu); this crate wraps them for Rust callers,
// mirroring the C++ `ctp::GpuApi` surface (gpu_api.h): device/managed
// allocation, memcpy, kernel launch, synchronize.
//
// Mechanism: the kernel source is embedded at build time (include_str!) and
// JIT-compiled by NVRTC at runtime; launches go through the CUDA driver
// API. No nvcc or host C++ compiler is needed to build this crate — only
// the CUDA toolkit's libraries at link time (`--features cuda`).
//
// Without the `cuda` feature the crate compiles to a stub whose
// `is_available()` returns false, so the workspace builds everywhere.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{DeviceBuffer, Gpu, GpuError, Kernel, Module};

/// True when CUDA support is compiled in AND a usable device exists.
pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda::driver_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// The wrapped CUDA C++ kernel source (kernels stay C++ per the migration
/// policy; see rust/MIGRATION.md).
pub const KERNEL_SOURCE: &str = include_str!("../kernels/ctp_kernels.cu");
