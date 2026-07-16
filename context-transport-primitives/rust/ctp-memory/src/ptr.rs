// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! The frozen shared-memory pointer ABI (MEMORY_DESIGN.md pillar 2).
//!
//! C++ → Rust mapping: `MemoryBackendId`/`AllocatorId` → [`AllocatorId`];
//! `ShmPtrBase` → [`ShmPtr`]; `OffsetPtrBase` → [`OffsetPtr`]. Layouts are
//! `#[repr(C)]` and byte-compatible; `off == u64::MAX` is null.

use std::marker::PhantomData;

/// Null sentinel for offsets (C++ `IS_NULL` convention).
pub const NULL_OFFSET: u64 = u64::MAX;

/// Allocator/backend identity (C++ `MemoryBackendId` == `AllocatorId`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AllocatorId {
    pub major: u32,
    pub minor: u32,
}

impl AllocatorId {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// C++ `GetNull()` — both fields zero.
    pub const fn null() -> Self {
        Self { major: 0, minor: 0 }
    }

    pub const fn is_null(&self) -> bool {
        self.major == 0 && self.minor == 0
    }
}

/// Marker for types that may live in shared memory: POD-shaped only —
/// no references, no Drop, no interior pointers into process-local memory.
/// Implement manually per type, mirroring the task-ABI POD discipline.
///
/// # Safety
/// Implementors assert the type is valid for any bit pattern written by a
/// cooperating process and contains no process-local addresses.
pub unsafe trait ShmSafe: Copy + 'static {}

// The obvious primitives.
unsafe impl ShmSafe for u8 {}
unsafe impl ShmSafe for u16 {}
unsafe impl ShmSafe for u32 {}
unsafe impl ShmSafe for u64 {}
unsafe impl ShmSafe for i8 {}
unsafe impl ShmSafe for i16 {}
unsafe impl ShmSafe for i32 {}
unsafe impl ShmSafe for i64 {}
unsafe impl ShmSafe for f32 {}
unsafe impl ShmSafe for f64 {}
unsafe impl ShmSafe for usize {}
unsafe impl ShmSafe for AllocatorId {}
unsafe impl<T: ShmSafe> ShmSafe for ShmPtr<T> {}
unsafe impl<T: ShmSafe> ShmSafe for OffsetPtr<T> {}
unsafe impl<T: ShmSafe, const N: usize> ShmSafe for [T; N] {}

/// Cross-process pointer: `(alloc_id, offset)` — the interop currency.
///
/// `Copy`, no `Drop`: this is a *reference* into shared state; memory is
/// released only by an explicit `free` on the owning allocator.
#[repr(C)]
pub struct ShmPtr<T = u8> {
    pub alloc_id: AllocatorId,
    pub off: u64,
    _marker: PhantomData<*const T>,
}

// PhantomData<*const T> suppresses auto-Send/Sync; ShmPtr is a plain
// (id, offset) value with no aliasing of its own — restore them.
unsafe impl<T> Send for ShmPtr<T> {}
unsafe impl<T> Sync for ShmPtr<T> {}

impl<T> Clone for ShmPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ShmPtr<T> {}

impl<T> std::fmt::Debug for ShmPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // C++ operator<< prints "alloc_id::off".
        write!(
            f,
            "{}.{}::{}",
            self.alloc_id.major, self.alloc_id.minor, self.off
        )
    }
}

impl<T> ShmPtr<T> {
    pub const fn new(alloc_id: AllocatorId, off: u64) -> Self {
        Self {
            alloc_id,
            off,
            _marker: PhantomData,
        }
    }

    /// C++ `ShmPtr::GetNull()`.
    pub const fn null() -> Self {
        Self::new(AllocatorId::null(), NULL_OFFSET)
    }

    pub const fn is_null(&self) -> bool {
        self.off == NULL_OFFSET
    }

    /// Reinterpret the pointee type (C++ `Cast<U>()`); offset unchanged.
    pub const fn cast<U>(self) -> ShmPtr<U> {
        ShmPtr::new(self.alloc_id, self.off)
    }

    /// Drop to an allocator-implied offset pointer.
    pub const fn to_offset(self) -> OffsetPtr<T> {
        OffsetPtr::new(self.off)
    }
}

/// Offset-only pointer; the allocator is implied by context (C++
/// `OffsetPtrBase`). Used inside segments so stored links never carry the
/// redundant 8-byte alloc_id.
#[repr(C)]
pub struct OffsetPtr<T = u8> {
    pub off: u64,
    _marker: PhantomData<*const T>,
}

unsafe impl<T> Send for OffsetPtr<T> {}
unsafe impl<T> Sync for OffsetPtr<T> {}

impl<T> Clone for OffsetPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for OffsetPtr<T> {}

impl<T> std::fmt::Debug for OffsetPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "off::{}", self.off)
    }
}

impl<T> OffsetPtr<T> {
    pub const fn new(off: u64) -> Self {
        Self {
            off,
            _marker: PhantomData,
        }
    }

    pub const fn null() -> Self {
        Self::new(NULL_OFFSET)
    }

    pub const fn is_null(&self) -> bool {
        self.off == NULL_OFFSET
    }

    /// Bind to an allocator, producing the cross-process form.
    pub const fn to_shm(self, alloc_id: AllocatorId) -> ShmPtr<T> {
        ShmPtr::new(alloc_id, self.off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_layout_is_frozen() {
        // The C++ ShmPtrBase is {u32, u32, u64} = 16 bytes; OffsetPtr is 8.
        assert_eq!(std::mem::size_of::<AllocatorId>(), 8);
        assert_eq!(std::mem::size_of::<ShmPtr<u8>>(), 16);
        assert_eq!(std::mem::size_of::<ShmPtr<[u64; 32]>>(), 16);
        assert_eq!(std::mem::size_of::<OffsetPtr<u8>>(), 8);
        assert_eq!(std::mem::align_of::<ShmPtr<u8>>(), 8);
        // Field order: alloc_id then off (serialization order parity).
        let p = ShmPtr::<u8>::new(AllocatorId::new(1, 2), 0xABCD);
        let bytes: [u8; 16] = unsafe { std::mem::transmute(p) };
        assert_eq!(&bytes[0..4], &1u32.to_ne_bytes());
        assert_eq!(&bytes[4..8], &2u32.to_ne_bytes());
        assert_eq!(&bytes[8..16], &0xABCDu64.to_ne_bytes());
    }

    #[test]
    fn null_semantics() {
        assert!(ShmPtr::<u8>::null().is_null());
        assert!(OffsetPtr::<u8>::null().is_null());
        // A null-id pointer with a real offset is NOT null (offset decides).
        assert!(!ShmPtr::<u8>::new(AllocatorId::null(), 0).is_null());
    }

    #[test]
    fn cast_and_conversion_preserve_location() {
        let p = ShmPtr::<u64>::new(AllocatorId::new(3, 4), 128);
        let q: ShmPtr<u8> = p.cast();
        assert_eq!(q.off, 128);
        assert_eq!(q.alloc_id, AllocatorId::new(3, 4));
        assert_eq!(p.to_offset().to_shm(p.alloc_id).off, p.off);
    }
}
