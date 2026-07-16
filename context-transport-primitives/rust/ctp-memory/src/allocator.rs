// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! The v1 in-segment free-list allocator (MEMORY_DESIGN.md "Segment layout").
//!
//! Position-independent by construction: the header and every block link
//! are offsets relative to the heap base; cross-process mutual exclusion is
//! an in-segment `AtomicU32` spinlock. First-fit with block split; frees
//! push to the list head. Coalescing deliberately deferred to the buddy
//! allocator port (see design doc).

use crate::backend::SharedMemBackend;
use crate::ptr::{AllocatorId, OffsetPtr, ShmPtr, ShmSafe, NULL_OFFSET};
use crate::registry;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

const MAGIC: u64 = 0x4354_505f_5348_4d31; // "CTP_SHM1"
const VERSION: u32 = 1;
const OWNER_RUST: u32 = 2;
/// Header size, 64-byte aligned; the heap starts here.
const HEADER_SIZE: u64 = 64;
/// Per-allocation block header preceding every payload.
const BLOCK_HEADER: u64 = 16;
/// Payload alignment.
const ALIGN: u64 = 16;

#[repr(C)]
struct SegmentHeader {
    magic: u64,
    version: u32,
    owner_impl: u32,
    alloc_major: u32,
    alloc_minor: u32,
    heap_size: u64,
    lock: u32, // accessed as AtomicU32
    _pad: u32,
    free_head: u64, // offset of first free block header (NULL_OFFSET = none)
    bump: u64,      // high-water mark (heap-relative)
}

const _: () = assert!(std::mem::size_of::<SegmentHeader>() as u64 <= HEADER_SIZE);

#[repr(C)]
struct BlockHeader {
    size: u64,      // payload capacity in bytes
    next_free: u64, // valid only while on the free list
}

/// The v1 allocator. Owns its backend mapping; registers itself in the
/// process registry on create/open and unregisters on drop.
pub struct FreeListAllocator {
    backend: SharedMemBackend,
    id: AllocatorId,
}

impl FreeListAllocator {
    /// Initialize a fresh segment (this process/language becomes the owner
    /// per design pillar 1) and register it.
    pub fn create(backend: SharedMemBackend, id: AllocatorId) -> Self {
        assert!(backend.size() as u64 > HEADER_SIZE + BLOCK_HEADER + ALIGN);
        let hdr = backend.base() as *mut SegmentHeader;
        // SAFETY: freshly mapped, exclusively owned segment large enough
        // for the header (asserted above).
        unsafe {
            hdr.write(SegmentHeader {
                magic: MAGIC,
                version: VERSION,
                owner_impl: OWNER_RUST,
                alloc_major: id.major,
                alloc_minor: id.minor,
                heap_size: backend.size() as u64 - HEADER_SIZE,
                lock: 0,
                _pad: 0,
                free_head: NULL_OFFSET,
                bump: 0,
            });
        }
        let me = Self { backend, id };
        me.register();
        me
    }

    /// Attach to an existing segment (created by any process implementing
    /// this format) and register it.
    pub fn open(backend: SharedMemBackend, expect_id: AllocatorId) -> std::io::Result<Self> {
        let hdr = backend.base() as *const SegmentHeader;
        // SAFETY: mapped segment at least HEADER_SIZE (checked by size).
        let ok = backend.size() as u64 > HEADER_SIZE
            && unsafe {
                (*hdr).magic == MAGIC
                    && (*hdr).version == VERSION
                    && (*hdr).alloc_major == expect_id.major
                    && (*hdr).alloc_minor == expect_id.minor
            };
        if !ok {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a CTP_SHM1 segment or id mismatch",
            ));
        }
        let me = Self {
            backend,
            id: expect_id,
        };
        me.register();
        Ok(me)
    }

    fn register(&self) {
        let heap_base = unsafe { self.backend.base().add(HEADER_SIZE as usize) };
        let heap_size = self.backend.size() as u64 - HEADER_SIZE;
        registry::register_allocator(self.id, heap_base, heap_size);
    }

    pub fn id(&self) -> AllocatorId {
        self.id
    }

    pub fn backend(&self) -> &SharedMemBackend {
        &self.backend
    }

    fn header(&self) -> *mut SegmentHeader {
        self.backend.base() as *mut SegmentHeader
    }

    fn heap_base(&self) -> *mut u8 {
        // SAFETY: segment > HEADER_SIZE, asserted at create/open.
        unsafe { self.backend.base().add(HEADER_SIZE as usize) }
    }

    /// In-segment spinlock (cross-process capable).
    fn lock(&self) -> SegmentLockGuard<'_> {
        // SAFETY: `lock` field is within the mapped header and 4-aligned.
        let atom = unsafe { AtomicU32::from_ptr(std::ptr::addr_of_mut!((*self.header()).lock)) };
        while atom
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
            std::thread::yield_now();
        }
        SegmentLockGuard { atom }
    }

    /// Allocate `size` payload bytes (16-aligned). Returns a heap-relative
    /// offset pointer, or `None` when the segment is exhausted.
    pub fn alloc_bytes(&self, size: u64) -> Option<OffsetPtr<u8>> {
        let want = size.max(1).div_ceil(ALIGN) * ALIGN;
        let _g = self.lock();
        // SAFETY: header mapped and exclusively locked for metadata writes.
        unsafe {
            let hdr = &mut *self.header();
            // First-fit over the free list.
            let mut prev: u64 = NULL_OFFSET;
            let mut cur = hdr.free_head;
            while cur != NULL_OFFSET {
                let bh = self.block_at(cur);
                if (*bh).size >= want {
                    // Split when the remainder can hold another block.
                    if (*bh).size >= want + BLOCK_HEADER + ALIGN {
                        let rem_off = cur + BLOCK_HEADER + want;
                        let rem = self.block_at(rem_off);
                        (*rem).size = (*bh).size - want - BLOCK_HEADER;
                        (*rem).next_free = (*bh).next_free;
                        (*bh).size = want;
                        if prev == NULL_OFFSET {
                            hdr.free_head = rem_off;
                        } else {
                            (*self.block_at(prev)).next_free = rem_off;
                        }
                    } else if prev == NULL_OFFSET {
                        hdr.free_head = (*bh).next_free;
                    } else {
                        (*self.block_at(prev)).next_free = (*bh).next_free;
                    }
                    return Some(OffsetPtr::new(cur + BLOCK_HEADER));
                }
                prev = cur;
                cur = (*bh).next_free;
            }
            // Bump path.
            let need = BLOCK_HEADER + want;
            if hdr.bump + need > hdr.heap_size {
                return None;
            }
            let off = hdr.bump;
            hdr.bump += need;
            let bh = self.block_at(off);
            (*bh).size = want;
            (*bh).next_free = NULL_OFFSET;
            Some(OffsetPtr::new(off + BLOCK_HEADER))
        }
    }

    /// Return an allocation to the free list.
    ///
    /// # Safety
    /// `ptr` must have been returned by `alloc_bytes`/`alloc` on THIS
    /// allocator (any process) and not freed since.
    pub unsafe fn free_bytes(&self, ptr: OffsetPtr<u8>) {
        if ptr.is_null() || ptr.off < BLOCK_HEADER {
            return;
        }
        let _g = self.lock();
        // SAFETY: caller contract — a live block header precedes the payload.
        unsafe {
            let hdr = &mut *self.header();
            let block_off = ptr.off - BLOCK_HEADER;
            let bh = self.block_at(block_off);
            (*bh).next_free = hdr.free_head;
            hdr.free_head = block_off;
        }
    }

    /// Typed allocation, zero-initialized, as the cross-process form.
    pub fn alloc<T: ShmSafe>(&self) -> Option<ShmPtr<T>> {
        let off = self.alloc_bytes(std::mem::size_of::<T>() as u64)?;
        // SAFETY: freshly allocated in-range block of >= size_of::<T>().
        unsafe {
            std::ptr::write_bytes(
                self.heap_base().add(off.off as usize),
                0,
                std::mem::size_of::<T>(),
            );
        }
        Some(off.to_shm(self.id).cast())
    }

    /// Free a typed allocation.
    ///
    /// # Safety
    /// Same contract as [`Self::free_bytes`].
    pub unsafe fn free<T>(&self, ptr: ShmPtr<T>) {
        // SAFETY: forwarded caller contract.
        unsafe { self.free_bytes(OffsetPtr::new(ptr.off)) };
    }

    /// Remaining never-allocated space (diagnostic; ignores the free list).
    pub fn bump_remaining(&self) -> u64 {
        let _g = self.lock();
        // SAFETY: header mapped; read under the lock.
        unsafe {
            let hdr = &*self.header();
            hdr.heap_size - hdr.bump
        }
    }

    fn block_at(&self, off: u64) -> *mut BlockHeader {
        // SAFETY: callers pass offsets previously produced by this
        // allocator's metadata, all within heap bounds.
        unsafe { self.heap_base().add(off as usize) as *mut BlockHeader }
    }

    /// The bump high-water mark, exposed for cross-process tests that need
    /// a mailbox location agreed upon out-of-band.
    pub fn heap_size(&self) -> u64 {
        // SAFETY: header mapped.
        unsafe { (*self.header()).heap_size }
    }
}

impl Drop for FreeListAllocator {
    fn drop(&mut self) {
        registry::unregister_allocator(self.id);
    }
}

struct SegmentLockGuard<'a> {
    atom: &'a AtomicU32,
}

impl Drop for SegmentLockGuard<'_> {
    fn drop(&mut self) {
        self.atom.store(0, Ordering::Release);
    }
}

/// In-segment atomic u64 view, for cross-process mailboxes/counters.
///
/// # Safety
/// `ptr` must resolve to a live, 8-aligned u64 inside a registered segment.
pub unsafe fn atomic_u64_at(ptr: *mut u64) -> &'static AtomicU64 {
    // SAFETY: forwarded caller contract; AtomicU64 has the same layout.
    unsafe { AtomicU64::from_ptr(ptr) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh(name_tag: &str, bytes: usize) -> FreeListAllocator {
        // Unique AllocatorId per call: the registry is process-global and
        // the test harness runs tests in parallel — a shared id would let
        // one test's Drop unregister another test's live mapping.
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_MINOR: AtomicU32 = AtomicU32::new(1);
        let minor = NEXT_MINOR.fetch_add(1, Ordering::Relaxed);
        let name = format!("ctp_rs_alloc_{}_{}", name_tag, std::process::id());
        let backend = SharedMemBackend::create(&name, bytes).unwrap();
        backend.destroy(); // unlink name eagerly (POSIX); mapping stays
        FreeListAllocator::create(backend, AllocatorId::new(700, minor))
    }

    #[test]
    fn alloc_free_reuse_cycle() {
        let a = fresh("cycle", 1 << 16);
        let p1 = a.alloc_bytes(100).unwrap();
        let p2 = a.alloc_bytes(100).unwrap();
        assert_ne!(p1.off, p2.off);
        let before = a.bump_remaining();
        // Freed block is reused without consuming new bump space.
        unsafe { a.free_bytes(p1) };
        let p3 = a.alloc_bytes(64).unwrap();
        assert_eq!(a.bump_remaining(), before);
        assert_eq!(p3.off, p1.off); // first-fit finds the freed block
                                    // Exhaustion returns None, not UB.
        assert!(a.alloc_bytes(1 << 20).is_none());
    }

    #[test]
    fn typed_alloc_resolves_via_registry() {
        let a = fresh("typed", 1 << 16);
        let p: ShmPtr<u64> = a.alloc().unwrap();
        let raw = crate::registry::resolve(p).unwrap();
        // SAFETY: freshly allocated, zeroed u64 in a registered segment.
        unsafe {
            assert_eq!(*raw, 0);
            *raw = 0xDEAD_BEEF;
            assert_eq!(*crate::registry::resolve(p).unwrap(), 0xDEAD_BEEF);
            a.free(p);
        }
    }

    #[test]
    fn split_leaves_usable_remainder() {
        let a = fresh("split", 1 << 16);
        let big = a.alloc_bytes(1024).unwrap();
        unsafe { a.free_bytes(big) };
        // Small alloc splits the freed 1024-block; remainder is reusable.
        let small = a.alloc_bytes(64).unwrap();
        assert_eq!(small.off, big.off);
        let rest = a.alloc_bytes(512).unwrap();
        assert!(rest.off > small.off && rest.off < big.off + 1024 + BLOCK_HEADER);
    }

    #[test]
    fn concurrent_alloc_free_is_consistent() {
        use std::sync::Arc;
        let a = Arc::new(fresh("mt", 1 << 20));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let a = Arc::clone(&a);
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        let p = a.alloc_bytes(64).unwrap();
                        let raw = crate::registry::resolve(p.to_shm(a.id())).unwrap();
                        // SAFETY: live 64-byte allocation.
                        unsafe {
                            std::ptr::write_bytes(raw, 0x77, 64);
                            a.free_bytes(p);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Heap not exhausted: 8*200 allocations of 64B were recycled.
        assert!(a.bump_remaining() > (1 << 20) - 8 * 200 * 80);
    }
}
