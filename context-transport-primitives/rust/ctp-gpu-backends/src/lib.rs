// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
// STUB: implemented by the FFI-wrapper workflow.

//! GPU FFI wrappers completing the story `ctp-gpu` began (which covers
//! CUDA via the driver API + NVRTC, with kernels remaining CUDA C++).
//!
//! | Backend | Nature | Feature | Module |
//! |---|---|---|---|
//! | HIP / ROCm | C API | `hip` | [`hip`] |
//! | SYCL | C++-only API → needs a C++ shim | `sycl` | [`sycl`] |
//! | GPU memory backends (gpu_malloc / gpu_shm_mmap) | C/C++ | `cuda`/`hip` | [`gpu_mem`] |
//!
//! Folds into `ctp-gpu` at integration; kept separate during development
//! so the two porting workflows never touch the same crate.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "hip")]
pub mod hip;
#[cfg(feature = "sycl")]
pub mod sycl;
pub mod gpu_mem;
