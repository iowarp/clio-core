// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
// STUB: implemented by the FFI-wrapper workflow.

//! FFI wrappers for CTP's C/C++ networking dependencies.
//!
//! | Library | Nature | Feature | Module |
//! |---|---|---|---|
//! | libzmq | C API | `zmq` | [`zmq`] |
//! | thallium (Mochi/Margo/Argobots) | C++ API → needs a C++ shim | `thallium` | [`thallium`] |
//! | NIXL | C/C++ API | `nixl` | [`nixl`] |
//!
//! Policy (rust/MIGRATION.md): these libraries are NOT ported — they are
//! wrapped. Each module exposes a safe Rust surface over the raw FFI.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "zmq")]
pub mod zmq;
#[cfg(feature = "thallium")]
pub mod thallium;
#[cfg(feature = "nixl")]
pub mod nixl;
