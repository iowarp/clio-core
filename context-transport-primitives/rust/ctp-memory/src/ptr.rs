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

// ---------------------------------------------------------------------------
// FullPtr
// ---------------------------------------------------------------------------

/// Both halves of a pointer: the process-local address and the cross-process
/// [`ShmPtr`] (C++ `ctp::ipc::FullPtr<T>`, `memory/allocator/allocator.h`).
///
/// Field order matches the C++ (`T* ptr_` then `ShmPtrBase<T> shm_`).
///
/// `ptr` is an **opaque address**, not a `*mut T`: this type carries
/// locations, it never dereferences them. That keeps it a plain value —
/// `Copy`, `Send`, `Sync`, no lifetime — which is what the C++ struct is too.
/// Resolve it with [`crate::registry::resolve`] when you actually need memory.
///
/// Only the `shm` half is serialized (C++ `serialize` writes `shm_` alone):
/// `ptr` is meaningless in another process. See [`Bulk`]-style users, which
/// re-resolve `ptr` locally after a load.
///
/// # Note on the private-memory encoding
///
/// [`FullPtr::from_local`] wraps ordinary (stack/heap) memory as a null
/// allocator id with **the address itself as the offset** — the
/// `MallocAllocator` convention. So a `FullPtr` over private memory is not
/// null, and `shm.off` is an address rather than an offset. This is why the
/// C++ takes such care to use integer arithmetic rather than `char* +
/// offset`: `GetBackendData()` is null for the `MallocAllocator`, and
/// `nullptr + off` is UB that macOS clang folds to null at -O2 (issue #620).
/// Rust's `wrapping_add` on `usize` sidesteps that whole class.
#[repr(C)]
pub struct FullPtr<T = u8> {
    /// C++ `T* ptr_` — process-local address; 0 is `nullptr`.
    pub ptr: usize,
    /// C++ `ShmPtrBase<T> shm_` — the cross-process half.
    pub shm: ShmPtr<T>,
}

impl<T> Clone for FullPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for FullPtr<T> {}

impl<T> std::fmt::Debug for FullPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullPtr")
            .field("ptr", &self.ptr)
            .field("shm", &self.shm)
            .finish()
    }
}

impl<T> FullPtr<T> {
    /// C++ `FullPtr(const T*, const PointerT&)`.
    pub const fn new(ptr: usize, shm: ShmPtr<T>) -> Self {
        Self { ptr, shm }
    }

    /// C++ `FullPtr::GetNull()` / the default ctor.
    pub const fn null() -> Self {
        Self {
            ptr: 0,
            shm: ShmPtr::null(),
        }
    }

    /// C++ `explicit FullPtr(T* ptr)` — wrap private/stack/heap memory: null
    /// allocator id, with the address itself as the "offset". See the type
    /// docs on this encoding.
    pub const fn from_local(addr: usize) -> Self {
        Self {
            ptr: addr,
            shm: ShmPtr::new(AllocatorId::null(), addr as u64),
        }
    }

    /// C++ `IsNull()`: **either** half being null makes the pair null.
    pub const fn is_null(&self) -> bool {
        self.ptr == 0 || self.shm.is_null()
    }

    /// C++ `SetNull()`.
    pub fn set_null(&mut self) {
        *self = Self::null();
    }

    /// C++ `operator+(size_t)`: advances **both** halves.
    ///
    /// Wraps on overflow: C++ unsigned arithmetic wraps, and a debug-build
    /// panic here would change behavior rather than preserve it.
    pub const fn offset_by(&self, size: usize) -> Self {
        Self {
            ptr: self.ptr.wrapping_add(size),
            shm: ShmPtr::new(self.shm.alloc_id, self.shm.off.wrapping_add(size as u64)),
        }
    }

    /// C++ `operator-(size_t)`: rewinds both halves. Wraps — see
    /// [`FullPtr::offset_by`].
    pub const fn rewind_by(&self, size: usize) -> Self {
        Self {
            ptr: self.ptr.wrapping_sub(size),
            shm: ShmPtr::new(self.shm.alloc_id, self.shm.off.wrapping_sub(size as u64)),
        }
    }

    /// Reinterpret the pointee type, as [`ShmPtr::cast`] does.
    pub const fn cast<U>(self) -> FullPtr<U> {
        FullPtr {
            ptr: self.ptr,
            shm: self.shm.cast::<U>(),
        }
    }
}

impl<T> Default for FullPtr<T> {
    fn default() -> Self {
        Self::null()
    }
}

impl<T> PartialEq for FullPtr<T> {
    /// C++ `operator==`: both halves must match.
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
            && self.shm.alloc_id == other.shm.alloc_id
            && self.shm.off == other.shm.off
    }
}

impl<T> Eq for FullPtr<T> {}

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

    #[test]
    fn full_ptr_null_semantics() {
        assert!(FullPtr::<u8>::null().is_null());
        assert!(FullPtr::<u8>::default().is_null());

        // EITHER half null makes the pair null, per the C++ IsNull().
        let shm = ShmPtr::<u8>::new(AllocatorId::new(1, 0), 64);
        assert!(FullPtr::new(0, shm).is_null(), "null local half");
        assert!(
            FullPtr::new(0x1000, ShmPtr::<u8>::null()).is_null(),
            "null shm half"
        );
        assert!(!FullPtr::new(0x1000, shm).is_null());

        let mut p = FullPtr::new(0x1000, shm);
        p.set_null();
        assert!(p.is_null());
    }

    #[test]
    fn full_ptr_from_local_uses_the_address_as_offset() {
        // The MallocAllocator convention: private memory is a null allocator
        // id with the address itself as the offset. So a FullPtr over private
        // memory is NOT null, despite the null allocator id.
        let addr = 0xDEAD_BEEFusize;
        let p = FullPtr::<u8>::from_local(addr);
        assert_eq!(p.ptr, addr);
        assert_eq!(p.shm.alloc_id, AllocatorId::null());
        assert_eq!(p.shm.off, addr as u64);
        assert!(!p.is_null(), "private memory is addressable, not null");

        // Address 0 is the one case that is null on both halves.
        assert!(FullPtr::<u8>::from_local(0).is_null());
    }

    #[test]
    fn full_ptr_offset_moves_both_halves_and_wraps() {
        let p = FullPtr::new(0x1000, ShmPtr::<u8>::new(AllocatorId::new(2, 1), 64));
        let q = p.offset_by(16);
        assert_eq!(q.ptr, 0x1010);
        assert_eq!(q.shm.off, 80);
        assert_eq!(q.shm.alloc_id, AllocatorId::new(2, 1), "id is preserved");
        assert_eq!(q.rewind_by(16), p, "offset then rewind round-trips");

        // Wraps like C++ unsigned arithmetic rather than panicking.
        let hi = FullPtr::new(usize::MAX, ShmPtr::<u8>::new(AllocatorId::new(1, 1), u64::MAX));
        assert_eq!(hi.offset_by(1).ptr, 0);
        assert_eq!(hi.offset_by(1).shm.off, 0);
    }

    #[test]
    fn full_ptr_equality_needs_both_halves() {
        let a = FullPtr::new(0x1000, ShmPtr::<u8>::new(AllocatorId::new(1, 0), 8));
        assert_eq!(a, a);
        assert_ne!(a, FullPtr::new(0x2000, a.shm), "local half differs");
        assert_ne!(
            a,
            FullPtr::new(0x1000, ShmPtr::<u8>::new(AllocatorId::new(1, 0), 9)),
            "offset differs"
        );
        assert_ne!(
            a,
            FullPtr::new(0x1000, ShmPtr::<u8>::new(AllocatorId::new(2, 0), 8)),
            "allocator differs"
        );
    }

    #[test]
    fn full_ptr_cast_keeps_both_halves() {
        let p = FullPtr::<u8>::new(0x1000, ShmPtr::new(AllocatorId::new(1, 2), 24));
        let q: FullPtr<u64> = p.cast();
        assert_eq!(q.ptr, p.ptr);
        assert_eq!(q.shm.off, p.shm.off);
        assert_eq!(q.shm.alloc_id, p.shm.alloc_id);
    }
}
