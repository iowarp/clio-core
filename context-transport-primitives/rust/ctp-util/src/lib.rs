// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// Rust port of clio_ctp/util: one module per C++ header. See
// rust/MIGRATION.md for conventions (no thread_local, timing in ms, C++
// parity documented per module).

#![deny(unsafe_op_in_unsafe_fn)]

pub mod config;
pub mod error;
pub mod logging;
pub mod random;
pub mod timer;
