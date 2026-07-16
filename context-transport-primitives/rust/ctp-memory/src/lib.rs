// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Shared-memory primitives: backends, the frozen `ShmPtr` ABI, the
//! allocator registry, and the v1 free-list allocator.
//!
//! The design contract lives in `rust/MEMORY_DESIGN.md`. Three pillars:
//! **one segment, one owner** (allocator metadata is mutated only by the
//! implementation that created the segment); **frozen pointer ABI**
//! (`ShmPtr = { alloc_id: (u32,u32), off: u64 }`, byte-compatible with the
//! C++ `ShmPtrBase`); **position-independence by construction** (segments
//! contain offsets only, never absolute pointers).

#![deny(unsafe_op_in_unsafe_fn)]

pub mod leak_checker;

pub mod smart_ptr;

pub mod arena;

pub mod slab;

pub mod buddy;

pub mod allocator;
pub mod backend;
pub mod ptr;
pub mod registry;

pub use allocator::FreeListAllocator;
pub use backend::SharedMemBackend;
pub use ptr::{AllocatorId, FullPtr, OffsetPtr, ShmPtr, ShmSafe, NULL_OFFSET};
pub use registry::{register_allocator, resolve, unregister_allocator};
