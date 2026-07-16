// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! `ctp::shared_ptr` / `ctp::unique_ptr` — allocator-carved smart pointers for
//! shared-memory objects (port of `clio_ctp/memory/smart_ptr/shared_ptr.h` and
//! `.../unique_ptr.h`).
//!
//! # Ownership contract (read this before using the module)
//!
//! Unlike `std::sync::Arc`, **the reference count is not process-local**. A
//! [`SharedPtrHeader`] is carved from the owning allocator's segment,
//! immediately in front of the payload — `make_shared` performs ONE allocation
//! laid out as `[SharedPtrHeader | pad | T]`, so header and payload share one
//! lifetime and one free. The count is an in-segment `AtomicU32` (MEMORY_DESIGN
//! pillar 3), which means:
//!
//! 1. **Any process may be the last releaser.** Process A can `make_shared`,
//!    hand the [`ShmPtr`] to process B via [`SharedPtr::to_shm`], B can
//!    [`SharedPtr::attach`] (count 1 → 2), and whichever of the two drops last
//!    performs the free. There is no "creator must outlive" rule for the
//!    *object*; the rule is on the *allocator* (see 2).
//! 2. **The free goes to the owning allocator** (MEMORY_DESIGN pillar 1:
//!    one segment, one owner). The handle borrows `&'a A`, so the allocator —
//!    and hence the mapping — is statically guaranteed to outlive every handle
//!    in this process. A releaser in a *non-owning* language/process must
//!    delegate the free rather than mutate foreign allocator metadata; in Rust
//!    that delegation is expressed by simply not attaching, or by handing the
//!    `ShmPtr` back to the owner.
//! 3. **The payload has no `Drop`.** `T: ShmSafe` is `Copy + 'static` POD, so
//!    the last releaser has nothing to destruct — it only frees the block.
//!    Handles (`SharedPtr`/`UniquePtr`) are process-local RAII objects and DO
//!    have `Drop`; it calls the explicit `free` on the owning allocator. Only
//!    the in-segment types (`SharedPtrHeader`, the payload) are `Drop`-free.
//! 4. **A count of zero is unreachable while a handle exists.** `reset()`,
//!    `drop`, and `cast()` are the only count transitions; a `wrap_non_owning`
//!    handle has a null header and participates in no counting at all.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`clio_ctp/memory/smart_ptr/*.h`) | Rust (`ctp_memory::smart_ptr`) | Notes |
//! |---|---|---|
//! | `ctp::SharedPtrHeader` | [`SharedPtrHeader`] | in-segment control block, `repr(C)`, 16 B |
//! | `SharedPtrHeader::ref_count_` (`ipc::atomic<u32>`) | `SharedPtrHeader::ref_count` | `u32` field viewed via `AtomicU32::from_ptr` |
//! | `SharedPtrHeader::destroy_` (`void(*)(void*)`) | `SharedPtrHeader::destroy` | reserved `u64`, always 0 — divergence 1 |
//! | `ctp::SharedPtrDestroyAs<T>` | *(removed)* | divergence 1 |
//! | `ctp::shared_ptr<T, AllocT>` | [`SharedPtr<'a, T, A>`] | |
//! | `AllocT *alloc_` (template param) | `&'a A where A: `[`BlockAllocator`] | |
//! | `ctp::make_shared<T>(alloc, args...)` | [`make_shared`] | `Option`, single value — divergences 2, 3 |
//! | copy ctor / copy assign | `Clone` | `fetch_add(1)` |
//! | move ctor / move assign | Rust move | divergence 4 |
//! | `~shared_ptr()` / `Release()` | `Drop` / [`SharedPtr::reset`] | |
//! | `WrapNonOwning(T*)` | [`SharedPtr::wrap_non_owning`] (`unsafe`) | |
//! | `Cast<U>()` | [`SharedPtr::cast`] | by value — divergence 5 |
//! | `get()` | [`SharedPtr::get`] | |
//! | `operator->` / `operator*` | [`SharedPtr::as_ref_unchecked`] / [`SharedPtr::as_mut_unchecked`] / [`SharedPtr::read`] / [`SharedPtr::write`] | all `unsafe` — divergence 6 |
//! | `IsNull()` / `explicit operator bool` | [`SharedPtr::is_null`] | |
//! | `use_count()` | [`SharedPtr::use_count`] | |
//! | `operator==` / `operator!=` | `PartialEq` | payload identity |
//! | `operator ctp::ipc::FullPtr<T>()` | [`SharedPtr::to_shm`] | non-owning — divergence 7 |
//! | *(none)* | [`SharedPtr::attach`] | addition: cross-process co-ownership |
//! | `ctp::unique_ptr<T, AllocT>` | [`UniquePtr<'a, T, A>`] | |
//! | `ctp::make_unique<T>(alloc, args...)` | [`make_unique`] | divergences 2, 3, 9 |
//! | `UniquePtrDestroyAs<T>` / `destroy_` | *(removed)* | divergence 1 |
//! | `unique_ptr::release()` | [`UniquePtr::release`] / [`UniquePtr::release_shm`] | |
//! | `unique_ptr::reset()` | [`UniquePtr::reset`] | |
//!
//! # Semantic divergences from the C++
//!
//! 1. **The type-erased destructor is gone.** C++ stores `destroy_`, a raw
//!    function pointer, *inside the shared block*. That is only valid in the
//!    process that installed it: a different process (or a different DLL/ASLR
//!    base) that becomes the last releaser would call a wild pointer, so the
//!    C++ header is only cross-process-safe when `destroy_ == nullptr`. This
//!    port keeps the 8-byte slot for block-layout parity but always writes 0,
//!    which C++ `Release()` already interprets as "trivially destructible / no
//!    destruction needed" — so a C++ last-releaser of a Rust-made block does
//!    the right thing (frees without destructing). Rust never calls a
//!    `destroy_` it finds in a C++-made block. Destruction of non-POD payloads
//!    is therefore out of scope here and belongs to the task-ABI/delegation
//!    path (MEMORY_DESIGN pillar 1). This also removes the C++ feature where a
//!    `shared_ptr<Task>` base view destructs the concrete derived type — Rust
//!    has no inheritance and `T: ShmSafe` has no destructor.
//! 2. **`T: ShmSafe` bound; one value, not variadic ctor args.** C++
//!    `make_shared(alloc, args...)` placement-news `T(args...)`. Rust has no
//!    variadic generics: [`make_shared`] takes an already-constructed `T` and
//!    moves (bitwise-copies) it into the segment. Restricting to `ShmSafe`
//!    (POD, `Copy`, no `Drop`) is what makes divergence 1 sound.
//! 3. **Allocation failure is `None`, not a null handle.** C++ returns a
//!    default-constructed (null) `shared_ptr`. Rust returns `Option`, matching
//!    `FreeListAllocator::alloc_bytes`. A null handle is still representable
//!    via [`SharedPtr::null`] / `Default` (the C++ default ctor).
//! 4. **Moves do not null the source at runtime.** C++ move ctor/assign clear
//!    the source's three fields; in Rust a moved-from handle is statically
//!    unusable, so the stores are unnecessary. Observable behavior matches.
//! 5. **`Cast<U>()` consumes instead of reinterpreting in place.** C++ returns
//!    a *reference* to `*this` reinterpreted as `shared_ptr<U, AllocT>`, which
//!    relies on layout equality across instantiations. `repr(Rust)` gives no
//!    such guarantee, so [`SharedPtr::cast`] takes `self` and returns a new
//!    handle. The reference count is unchanged either way (the C++ contract);
//!    the difference is that the source handle is consumed rather than aliased.
//! 6. **No `Deref`/`DerefMut`; payload access is `unsafe`.** Another process
//!    may be writing the same bytes, so handing out `&T` cannot be safe (the
//!    crate's existing `registry::resolve` takes the same position: resolution
//!    is safe, dereference is the caller's `unsafe`). Callers synchronize with
//!    in-segment atomics. C++ `operator*`/`operator->` on a null pointer is UB;
//!    the Rust accessors document a non-null precondition instead.
//! 7. **`FullPtr<T>` → `ShmPtr<T>`.** The C++ implicit conversion yields a
//!    non-owning `FullPtr` (raw pointer + allocator). The Rust currency is the
//!    frozen `(alloc_id, off)` ABI, so [`SharedPtr::to_shm`] returns a
//!    non-owning `ShmPtr<T>`. It is explicit rather than implicit (Rust has no
//!    implicit conversions), and unlike `FullPtr` it is meaningful in another
//!    process — see [`SharedPtr::attach`].
//! 8. **Reference-count details.** C++ `ipc::atomic` defaults every op to
//!    `seq_cst`. This port uses the `Arc` idiom: `Relaxed` for the increment,
//!    `Release` for the decrement plus an `Acquire` fence on the final release
//!    (which is what actually establishes happens-before with the freeing
//!    process); `use_count` loads `Relaxed` and is a racy snapshot, as in C++.
//!    Additionally, C++ `fetch_add` wraps silently at `u32::MAX` (a latent
//!    use-after-free); this port detects the wrap, undoes it, and panics.
//! 9. **No device path.** C++ `make_unique` is `CTP_CROSS_FUN` and returns a
//!    null `unique_ptr` when compiled for device (`#if CTP_IS_HOST`). The Rust
//!    port is host-only, so the `#else` branch has no analogue.
//! 10. **Additions for cross-process ownership:** [`SharedPtr::attach`] (the
//!     only way a second process can join the count — C++ has no equivalent
//!     because a `shared_ptr`'s raw pointers cannot cross a process boundary)
//!     and [`UniquePtr::release_shm`], which returns the `ShmPtr` a later
//!     explicit free needs (C++ `release()` returns a bare `T*`, leaving the
//!     caller to recover the offset out-of-band; that form is ported too).
//! 11. **Handle representation.** C++ fixes the layout `{alloc_, header_,
//!     data_}` so `Cast()` can reinterpret. The Rust handle additionally
//!     stores the block's `OffsetPtr` (C++ recomputes it in `Release` via the
//!     `FullPtr(alloc, raw_ptr)` ctor, which this crate's allocator API does
//!     not offer). Handles are process-local, so their layout is not ABI.
//! 12. **Over-aligned payloads panic.** [`BlockAllocator`] guarantees 16-byte
//!     payload alignment ([`MAX_ALIGN`]), so `align_of::<T>() > 16` is
//!     rejected with a clear panic. The C++ computes the same `kDataOff`
//!     formula but silently yields a misaligned object when the allocator's
//!     guarantee is weaker than `alignof(T)`.
//! 13. **Header ABI is 64-bit-host parity.** C++ `SharedPtrHeader` is
//!     `{atomic<u32>, void(*)(void*)}` = 16 B/align 8 on 64-bit hosts,
//!     matching this port exactly. On a 32-bit build the C++ header would
//!     shrink to 8 B while the Rust one stays 16 B; 32-bit hosts are not a
//!     supported target.
//! 14. `header_->~Header()` in C++ `Release()` is a no-op for the POD header
//!     and has no Rust analogue.

use crate::allocator::FreeListAllocator;
use crate::ptr::{AllocatorId, OffsetPtr, ShmPtr, ShmSafe};
use crate::registry;
use std::sync::atomic::{fence, AtomicU32, Ordering};

/// Payload alignment guaranteed by [`BlockAllocator`] implementations (the v1
/// free-list allocator's `ALIGN`). Payloads needing more are rejected.
pub const MAX_ALIGN: usize = 16;

/// Control block for [`SharedPtr`], co-located in front of the payload.
///
/// Lives **inside the segment**: `repr(C)`, no `Drop`, no pointers — only the
/// reference count is meaningful. Byte-compatible with the C++
/// `ctp::SharedPtrHeader` on 64-bit hosts (divergence 13).
#[repr(C)]
pub struct SharedPtrHeader {
    /// Strong reference count; starts at 1 (set by [`make_shared`]). Accessed
    /// through `AtomicU32::from_ptr` — it is an in-segment atomic, mutated by
    /// every process holding a reference.
    pub ref_count: u32,
    /// Padding matching the C++ `atomic<u32>` → function-pointer gap.
    _pad: u32,
    /// Reserved: the C++ `destroy_` slot. Rust always writes 0 ("no
    /// destruction needed"). See divergence 1 — never call this.
    pub destroy: u64,
}

// Block-layout parity with the 64-bit C++ SharedPtrHeader.
const _: () = assert!(std::mem::size_of::<SharedPtrHeader>() == 16);
const _: () = assert!(std::mem::align_of::<SharedPtrHeader>() == 8);

/// The C++ `AllocT` template parameter, narrowed to what the smart pointers
/// need: carve a block, resolve it in this process, hand it back.
///
/// # Safety
///
/// Implementors must guarantee that:
/// - `alloc_block(n)` returns an offset to at least `n` writable bytes,
///   aligned to at least [`MAX_ALIGN`], valid until `free_block`;
/// - `resolve_block(p)` maps an offset produced by `alloc_block` to a live
///   address **in this process**, and returns non-null for a live block;
/// - offsets are stable across every process mapping the same segment (they
///   are relative to the segment's heap base, per MEMORY_DESIGN pillar 3);
/// - `id()` is the [`AllocatorId`] under which those offsets resolve.
pub unsafe trait BlockAllocator {
    /// Identity used to build the cross-process [`ShmPtr`] form.
    fn id(&self) -> AllocatorId;

    /// Carve `size` bytes; `None` when the segment is exhausted.
    fn alloc_block(&self, size: u64) -> Option<OffsetPtr<u8>>;

    /// Return a block to the allocator.
    ///
    /// # Safety
    /// `block` came from `alloc_block` on this allocator (in any process) and
    /// has not been freed since.
    unsafe fn free_block(&self, block: OffsetPtr<u8>);

    /// Resolve a block offset to an address in this process; null if unknown.
    fn resolve_block(&self, block: OffsetPtr<u8>) -> *mut u8;
}

// SAFETY: FreeListAllocator hands out 16-byte-aligned, heap-base-relative
// offsets (MEMORY_DESIGN "Segment layout") and resolves them through the
// process registry; free_bytes forwards our identical caller contract.
unsafe impl BlockAllocator for FreeListAllocator {
    fn id(&self) -> AllocatorId {
        FreeListAllocator::id(self)
    }

    fn alloc_block(&self, size: u64) -> Option<OffsetPtr<u8>> {
        self.alloc_bytes(size)
    }

    unsafe fn free_block(&self, block: OffsetPtr<u8>) {
        // SAFETY: forwarded caller contract (block from alloc_block, live).
        unsafe { self.free_bytes(block) }
    }

    fn resolve_block(&self, block: OffsetPtr<u8>) -> *mut u8 {
        registry::resolve(block.to_shm(FreeListAllocator::id(self))).unwrap_or(std::ptr::null_mut())
    }
}

/// Offset of the payload within the block — the C++ `kDataOff`:
/// `align_up(sizeof(SharedPtrHeader), max(alignof(T), alignof(Header)))`.
const fn data_offset<T>() -> usize {
    let ha = std::mem::align_of::<SharedPtrHeader>();
    let ta = std::mem::align_of::<T>();
    let align = if ta > ha { ta } else { ha };
    std::mem::size_of::<SharedPtrHeader>().div_ceil(align) * align
}

/// Reject payloads the allocator cannot align (divergence 12).
fn check_align<T>() {
    assert!(
        std::mem::align_of::<T>() <= MAX_ALIGN,
        "ctp smart_ptr: align_of::<{}>() = {} exceeds the {}-byte allocator guarantee",
        std::any::type_name::<T>(),
        std::mem::align_of::<T>(),
        MAX_ALIGN
    );
}

/// View the in-segment reference count as an atomic.
///
/// # Safety
/// `header` must point at a live [`SharedPtrHeader`] inside a mapped segment.
unsafe fn ref_count_of(header: *mut SharedPtrHeader) -> &'static AtomicU32 {
    // SAFETY: caller contract; ref_count is at offset 0 of a repr(C) header
    // and 4-byte aligned, which is AtomicU32's layout requirement.
    unsafe { AtomicU32::from_ptr(std::ptr::addr_of_mut!((*header).ref_count)) }
}

/// An intrusively-allocated, reference-counted handle to a shared-memory
/// object (C++ `ctp::shared_ptr<T, AllocT>`).
///
/// The count lives in the segment, not in this handle — see the module-level
/// ownership contract. Construct via [`make_shared`], [`SharedPtr::attach`], or
/// [`SharedPtr::wrap_non_owning`]; there is deliberately no raw-pointer-adopting
/// constructor, mirroring the C++ private ctor.
pub struct SharedPtr<'a, T: ShmSafe, A: BlockAllocator = FreeListAllocator> {
    /// Allocator the block was carved from — `None` for null/non-owning
    /// handles (C++ `alloc_`).
    alloc: Option<&'a A>,
    /// Block start (== the header) as a heap-relative offset (divergence 11).
    block: OffsetPtr<u8>,
    /// Control block resolved in this process (C++ `header_`); null means "no
    /// reference counting" (null or non-owning handle).
    header: *mut SharedPtrHeader,
    /// The managed object resolved in this process (C++ `data_`).
    data: *mut T,
}

// SAFETY: the reference count is an in-segment atomic, so ownership
// transitions are thread- (and process-) safe; the payload is `ShmSafe` POD
// and every access to it is an `unsafe` fn with a documented contract. Holding
// `&'a A` across threads is what requires `A: Sync`.
unsafe impl<T: ShmSafe, A: BlockAllocator + Sync> Send for SharedPtr<'_, T, A> {}
// SAFETY: as above; `&SharedPtr` exposes only counts and raw pointers.
unsafe impl<T: ShmSafe, A: BlockAllocator + Sync> Sync for SharedPtr<'_, T, A> {}

impl<'a, T: ShmSafe, A: BlockAllocator> SharedPtr<'a, T, A> {
    /// Null handle (C++ default constructor).
    pub const fn null() -> Self {
        Self {
            alloc: None,
            block: OffsetPtr::null(),
            header: std::ptr::null_mut(),
            data: std::ptr::null_mut(),
        }
    }

    /// Build a NON-OWNING view over an existing object (C++ `WrapNonOwning`).
    ///
    /// The result has a null control header: it participates in no reference
    /// counting and frees nothing on drop — a purely typed handle for interop
    /// at boundaries where storage is owned elsewhere. [`Self::use_count`]
    /// reports 0 and [`Self::to_shm`] returns null. Prefer [`make_shared`];
    /// this is the deliberate, named exception to the "no raw pointers" rule.
    ///
    /// # Safety
    /// `data` must be valid for reads/writes of a `T` for as long as this
    /// handle (and any clone of it) is used, and must not be freed while in
    /// use. Ownership is NOT transferred.
    pub unsafe fn wrap_non_owning(data: *mut T) -> Self {
        Self {
            alloc: None,
            block: OffsetPtr::null(),
            header: std::ptr::null_mut(),
            data,
        }
    }

    /// Join ownership of an object created by [`make_shared`], possibly in
    /// another process (increments the in-segment count).
    ///
    /// This is the mechanism behind ownership-contract point 1 and has no C++
    /// counterpart (divergence 10): pass the [`ShmPtr`] from [`Self::to_shm`]
    /// through any channel, then attach on the far side against the allocator
    /// that owns the segment. Returns `None` if `data` is null, belongs to a
    /// different allocator, cannot be a `make_shared` payload (offset below the
    /// header), or does not resolve in this process.
    ///
    /// # Safety
    /// `data` must be a live payload pointer produced by [`Self::to_shm`] on a
    /// [`make_shared`] handle for the same `T` and the same segment, whose
    /// reference count has not yet reached zero (i.e. the sender still holds a
    /// reference, or otherwise guarantees liveness across the handoff).
    pub unsafe fn attach(alloc: &'a A, data: ShmPtr<T>) -> Option<Self> {
        check_align::<T>();
        if data.is_null() || data.alloc_id != alloc.id() {
            return None;
        }
        let doff = data_offset::<T>() as u64;
        let block_off = data.off.checked_sub(doff)?;
        let block = OffsetPtr::<u8>::new(block_off);
        let base = alloc.resolve_block(block);
        if base.is_null() {
            return None;
        }
        let header = base as *mut SharedPtrHeader;
        // SAFETY: caller contract — `base` is the live block of a make_shared
        // allocation, whose header sits at offset 0 and count is non-zero.
        unsafe {
            bump_ref_count(header);
            Some(Self {
                alloc: Some(alloc),
                block,
                header,
                data: base.add(doff as usize) as *mut T,
            })
        }
    }

    /// Reinterpret the payload type (C++ `Cast<U>()`).
    ///
    /// The reference count is unchanged: this consumes the handle and returns
    /// the same ownership stake viewed as `U` (divergence 5).
    pub fn cast<U: ShmSafe>(self) -> SharedPtr<'a, U, A> {
        check_align::<U>();
        // The block offset only stays meaningful if U's payload lands where
        // T's did; guaranteed because data_offset() is identical for every
        // type aligned to <= MAX_ALIGN, which check_align enforces.
        let me = std::mem::ManuallyDrop::new(self);
        SharedPtr {
            alloc: me.alloc,
            block: me.block,
            header: me.header,
            data: me.data as *mut U,
        }
    }

    /// Release this reference and become null (C++ `reset()`/`Release()`).
    /// Frees the block if this was the last reference. Idempotent.
    pub fn reset(&mut self) {
        if !self.header.is_null() {
            // SAFETY: non-null header ⇒ this handle holds a live reference,
            // so the block (and its segment) is still mapped.
            let rc = unsafe { ref_count_of(self.header) };
            // fetch_sub returns the PRIOR value: ==1 means we dropped the last
            // reference (C++ Release()).
            if rc.fetch_sub(1, Ordering::Release) == 1 {
                // Acquire the writes of every other releaser before reusing
                // the memory.
                fence(Ordering::Acquire);
                // No destructor to run: T is ShmSafe POD and `destroy` is
                // always 0 (divergences 1, 2). Just free the one block that
                // holds [header | pad | T].
                if let Some(alloc) = self.alloc {
                    // SAFETY: `block` came from alloc_block on this allocator
                    // in make_shared and is freed exactly once — here, by the
                    // thread/process that observed the count reach zero.
                    unsafe { alloc.free_block(self.block) };
                }
            }
        }
        self.alloc = None;
        self.block = OffsetPtr::null();
        self.header = std::ptr::null_mut();
        self.data = std::ptr::null_mut();
    }

    /// Raw pointer to the managed object, valid in this process (C++ `get()`).
    /// Null for a null handle. Dereferencing is the caller's `unsafe`, per the
    /// crate's `registry::resolve` convention.
    pub fn get(&self) -> *mut T {
        self.data
    }

    /// Non-owning cross-process handle to the payload (C++
    /// `operator FullPtr<T>()`, divergence 7). Null for null/non-owning
    /// handles. The object must outlive any use of the result — this transfers
    /// no ownership; see [`Self::attach`] to acquire one.
    pub fn to_shm(&self) -> ShmPtr<T> {
        match self.alloc {
            Some(alloc) if !self.block.is_null() => {
                ShmPtr::new(alloc.id(), self.block.off + data_offset::<T>() as u64)
            }
            _ => ShmPtr::null(),
        }
    }

    /// True when this handle manages nothing (C++ `IsNull()`; the negation is
    /// C++ `explicit operator bool`).
    pub fn is_null(&self) -> bool {
        self.data.is_null()
    }

    /// Current reference count; 0 when null or non-owning (C++ `use_count()`).
    /// A racy snapshot — only 1 (sole owner) and 0 are actionable.
    pub fn use_count(&self) -> u32 {
        if self.header.is_null() {
            0
        } else {
            // SAFETY: non-null header ⇒ live reference held by this handle.
            unsafe { ref_count_of(self.header) }.load(Ordering::Relaxed)
        }
    }

    /// Shared reference to the payload (C++ `operator*`).
    ///
    /// # Safety
    /// The handle must be non-null and no other thread/process may write the
    /// payload for the lifetime of the returned reference (divergence 6).
    pub unsafe fn as_ref_unchecked(&self) -> &T {
        debug_assert!(!self.data.is_null());
        // SAFETY: caller contract (non-null, no concurrent writer).
        unsafe { &*self.data }
    }

    /// Exclusive reference to the payload (C++ `operator*` on a mutable view).
    ///
    /// # Safety
    /// The handle must be non-null and no other thread/process may access the
    /// payload for the lifetime of the returned reference. Note that `&mut
    /// self` proves exclusivity of the *handle*, never of the segment bytes.
    pub unsafe fn as_mut_unchecked(&mut self) -> &mut T {
        debug_assert!(!self.data.is_null());
        // SAFETY: caller contract (non-null, no concurrent accessor).
        unsafe { &mut *self.data }
    }

    /// Copy the payload out. `None` when null.
    ///
    /// # Safety
    /// No other thread/process may be writing the payload concurrently; a
    /// torn read is UB, not merely a stale value. Synchronize with in-segment
    /// atomics.
    pub unsafe fn read(&self) -> Option<T> {
        if self.data.is_null() {
            return None;
        }
        // SAFETY: caller contract; `data` is a live, aligned, initialized T.
        Some(unsafe { std::ptr::read(self.data) })
    }

    /// Overwrite the payload. Returns false when null.
    ///
    /// Takes `&self` because shared-memory payloads are shared by
    /// construction — the handle's Rust-level mutability says nothing about
    /// other processes.
    ///
    /// # Safety
    /// No other thread/process may be reading or writing the payload
    /// concurrently.
    pub unsafe fn write(&self, value: T) -> bool {
        if self.data.is_null() {
            return false;
        }
        // SAFETY: caller contract; `data` is live, aligned, and POD.
        unsafe { std::ptr::write(self.data, value) };
        true
    }
}

/// Increment an in-segment count, refusing to wrap (divergence 8).
///
/// # Safety
/// `header` must point at a live header whose count is non-zero.
unsafe fn bump_ref_count(header: *mut SharedPtrHeader) {
    // SAFETY: caller contract.
    let rc = unsafe { ref_count_of(header) };
    if rc.fetch_add(1, Ordering::Relaxed) == u32::MAX {
        // We just wrapped MAX -> 0, which would make the next release free a
        // live object. Undo, then fail loudly instead of silently (C++ wraps).
        rc.fetch_sub(1, Ordering::Relaxed);
        panic!("ctp shared_ptr: reference count overflow");
    }
}

impl<T: ShmSafe, A: BlockAllocator> Clone for SharedPtr<'_, T, A> {
    /// C++ copy constructor: share ownership, incrementing the in-segment
    /// count. A non-owning/null handle clones without any count change.
    fn clone(&self) -> Self {
        if !self.header.is_null() {
            // SAFETY: non-null header ⇒ this handle holds a live reference,
            // so the count is >= 1 and the segment is mapped.
            unsafe { bump_ref_count(self.header) };
        }
        Self {
            alloc: self.alloc,
            block: self.block,
            header: self.header,
            data: self.data,
        }
    }
}

impl<T: ShmSafe, A: BlockAllocator> Drop for SharedPtr<'_, T, A> {
    /// C++ `~shared_ptr()`: drop this reference.
    fn drop(&mut self) {
        self.reset();
    }
}

impl<T: ShmSafe, A: BlockAllocator> Default for SharedPtr<'_, T, A> {
    fn default() -> Self {
        Self::null()
    }
}

impl<T: ShmSafe, A: BlockAllocator> PartialEq for SharedPtr<'_, T, A> {
    /// C++ `operator==`: identity of the managed object.
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: ShmSafe, A: BlockAllocator> Eq for SharedPtr<'_, T, A> {}

impl<T: ShmSafe, A: BlockAllocator> std::fmt::Debug for SharedPtr<'_, T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedPtr")
            .field("shm", &self.to_shm())
            .field("use_count", &self.use_count())
            .finish()
    }
}

/// Allocate ONE block holding a [`SharedPtrHeader`] and a `T`, initialize
/// both, and return a handle owning it with a reference count of 1 (C++
/// `ctp::make_shared`).
///
/// The payload sub-region is placed at the C++ `kDataOff`. Returns `None` when
/// the allocator is exhausted (divergence 3). Panics if `T` needs more than
/// [`MAX_ALIGN`] alignment (divergence 12).
pub fn make_shared<'a, T: ShmSafe, A: BlockAllocator>(
    alloc: &'a A,
    value: T,
) -> Option<SharedPtr<'a, T, A>> {
    check_align::<T>();
    let doff = data_offset::<T>();
    let total = (doff + std::mem::size_of::<T>()) as u64;
    let block = alloc.alloc_block(total)?;
    let base = alloc.resolve_block(block);
    if base.is_null() {
        // Registry has no mapping for this allocator: give the block back
        // rather than leaking it.
        // SAFETY: `block` was just returned by alloc_block and never shared.
        unsafe { alloc.free_block(block) };
        return None;
    }
    let header = base as *mut SharedPtrHeader;
    // SAFETY: `base` is a fresh, exclusively-owned block of >= `total` bytes,
    // aligned to MAX_ALIGN >= align_of::<SharedPtrHeader>(), and `doff` is
    // aligned for T (check_align + the kDataOff formula). Nothing else can
    // observe the block until the handle below is returned.
    unsafe {
        header.write(SharedPtrHeader {
            ref_count: 1,
            _pad: 0,
            // Always 0: "no destruction needed" for C++ and Rust alike.
            destroy: 0,
        });
        let data = base.add(doff) as *mut T;
        data.write(value);
        Some(SharedPtr {
            alloc: Some(alloc),
            block,
            header,
            data,
        })
    }
}

/// A move-only, single-owner handle to an allocator-carved object (C++
/// `ctp::unique_ptr<T, AllocT>`).
///
/// Mirrors [`SharedPtr`] without the reference count: the sole owner frees the
/// object on drop. Unlike `shared_ptr`, there is no header — the payload sits
/// at the start of the block. Construct via [`make_unique`]; non-copyable by
/// construction (no `Clone`), which is the C++ `= delete` pair.
pub struct UniquePtr<'a, T: ShmSafe, A: BlockAllocator = FreeListAllocator> {
    /// Allocator the object was carved from (C++ `alloc_`).
    alloc: Option<&'a A>,
    /// Block/payload offset (the C++ `data_`'s cross-process form).
    block: OffsetPtr<u8>,
    /// The managed object resolved in this process (C++ `data_`).
    data: *mut T,
}

// SAFETY: single-owner handle to POD payload; all payload access is unsafe
// with a documented contract. `&'a A` across threads requires `A: Sync`.
unsafe impl<T: ShmSafe, A: BlockAllocator + Sync> Send for UniquePtr<'_, T, A> {}
// SAFETY: as above; `&UniquePtr` exposes only raw pointers and offsets.
unsafe impl<T: ShmSafe, A: BlockAllocator + Sync> Sync for UniquePtr<'_, T, A> {}

impl<T: ShmSafe, A: BlockAllocator> UniquePtr<'_, T, A> {
    /// Null handle (C++ default constructor).
    pub const fn null() -> Self {
        Self {
            alloc: None,
            block: OffsetPtr::null(),
            data: std::ptr::null_mut(),
        }
    }

    /// Free the owned object and become null (C++ `reset()`/`DoReset()`).
    /// Idempotent.
    pub fn reset(&mut self) {
        if !self.data.is_null() {
            // No destructor: T is ShmSafe POD (divergences 1, 2).
            if let Some(alloc) = self.alloc {
                // SAFETY: `block` came from alloc_block in make_unique, is
                // owned solely by this handle, and is freed exactly once (the
                // fields are cleared below).
                unsafe { alloc.free_block(self.block) };
            }
        }
        self.alloc = None;
        self.block = OffsetPtr::null();
        self.data = std::ptr::null_mut();
    }

    /// Relinquish ownership without freeing; returns the raw pointer (C++
    /// `release()`).
    ///
    /// The object is now unowned: freeing it requires its offset, which this
    /// form discards — see [`Self::release_shm`] (divergence 10).
    pub fn release(&mut self) -> *mut T {
        let d = self.data;
        self.alloc = None;
        self.block = OffsetPtr::null();
        self.data = std::ptr::null_mut();
        d
    }

    /// Relinquish ownership without freeing; returns the cross-process handle
    /// (Rust addition). The caller becomes responsible for an explicit
    /// `free` on the owning allocator.
    pub fn release_shm(&mut self) -> ShmPtr<T> {
        let p = self.to_shm();
        self.alloc = None;
        self.block = OffsetPtr::null();
        self.data = std::ptr::null_mut();
        p
    }

    /// Non-owning cross-process handle to the payload; null when null.
    pub fn to_shm(&self) -> ShmPtr<T> {
        match self.alloc {
            Some(alloc) if !self.block.is_null() => ShmPtr::new(alloc.id(), self.block.off),
            _ => ShmPtr::null(),
        }
    }

    /// Raw pointer to the managed object (C++ `get()`).
    pub fn get(&self) -> *mut T {
        self.data
    }

    /// True when this handle owns nothing (C++ `IsNull()`).
    pub fn is_null(&self) -> bool {
        self.data.is_null()
    }

    /// Shared reference to the payload (C++ `operator*`).
    ///
    /// # Safety
    /// Non-null handle; no concurrent writer in any process (divergence 6).
    pub unsafe fn as_ref_unchecked(&self) -> &T {
        debug_assert!(!self.data.is_null());
        // SAFETY: caller contract.
        unsafe { &*self.data }
    }

    /// Exclusive reference to the payload.
    ///
    /// # Safety
    /// Non-null handle; no concurrent accessor in any process. `&mut self`
    /// proves exclusivity of the handle, not of the segment bytes.
    pub unsafe fn as_mut_unchecked(&mut self) -> &mut T {
        debug_assert!(!self.data.is_null());
        // SAFETY: caller contract.
        unsafe { &mut *self.data }
    }

    /// Copy the payload out; `None` when null.
    ///
    /// # Safety
    /// No concurrent writer in any process.
    pub unsafe fn read(&self) -> Option<T> {
        if self.data.is_null() {
            return None;
        }
        // SAFETY: caller contract; live, aligned, initialized T.
        Some(unsafe { std::ptr::read(self.data) })
    }

    /// Overwrite the payload; false when null.
    ///
    /// # Safety
    /// No concurrent accessor in any process.
    pub unsafe fn write(&self, value: T) -> bool {
        if self.data.is_null() {
            return false;
        }
        // SAFETY: caller contract; live, aligned POD T.
        unsafe { std::ptr::write(self.data, value) };
        true
    }
}

impl<T: ShmSafe, A: BlockAllocator> Drop for UniquePtr<'_, T, A> {
    /// C++ `~unique_ptr()`: free the owned object.
    fn drop(&mut self) {
        self.reset();
    }
}

impl<T: ShmSafe, A: BlockAllocator> Default for UniquePtr<'_, T, A> {
    fn default() -> Self {
        Self::null()
    }
}

impl<T: ShmSafe, A: BlockAllocator> std::fmt::Debug for UniquePtr<'_, T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UniquePtr").field("shm", &self.to_shm()).finish()
    }
}

/// Allocate and initialize a single `T` from `alloc`, returning the sole owner
/// (C++ `ctp::make_unique`). `None` when the allocator is exhausted
/// (divergence 3); host-only (divergence 9).
pub fn make_unique<'a, T: ShmSafe, A: BlockAllocator>(
    alloc: &'a A,
    value: T,
) -> Option<UniquePtr<'a, T, A>> {
    check_align::<T>();
    let block = alloc.alloc_block(std::mem::size_of::<T>() as u64)?;
    let base = alloc.resolve_block(block);
    if base.is_null() {
        // SAFETY: `block` was just allocated and never shared.
        unsafe { alloc.free_block(block) };
        return None;
    }
    let data = base as *mut T;
    // SAFETY: fresh, exclusively-owned block of >= size_of::<T>() bytes,
    // aligned to MAX_ALIGN >= align_of::<T>() (check_align).
    unsafe { data.write(value) };
    Some(UniquePtr {
        alloc: Some(alloc),
        block,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SharedMemBackend;
    use std::sync::atomic::AtomicU32 as StdAtomicU32;

    /// A fresh, uniquely-named segment + allocator per call: the registry and
    /// (on Windows) the segment namespace are process-global, and tests run in
    /// parallel — a shared id/name would let one test's Drop unregister
    /// another's live mapping, or alias two "distinct" segments.
    fn fresh(bytes: usize) -> FreeListAllocator {
        static NEXT: StdAtomicU32 = StdAtomicU32::new(1);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("ctp_rs_smartptr_{}_{}", std::process::id(), n);
        let backend = SharedMemBackend::create(&name, bytes).unwrap();
        backend.destroy(); // unlink the name eagerly (POSIX); mapping stays
        FreeListAllocator::create(backend, AllocatorId::new(756, n))
    }

    /// A payload with interior structure, to catch offset/alignment mistakes.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    #[repr(C)]
    struct Payload {
        a: u64,
        b: u32,
        c: u32,
    }
    // SAFETY: POD — repr(C), Copy, no Drop, no references or host addresses.
    unsafe impl ShmSafe for Payload {}

    /// Zero-sized payload: the "empty" edge case.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Zst;
    // SAFETY: no bytes at all; trivially valid for any bit pattern.
    unsafe impl ShmSafe for Zst {}

    #[test]
    fn header_abi_matches_cpp_64bit() {
        // C++: { ipc::atomic<u32> ref_count_; void(*destroy_)(void*); }
        assert_eq!(std::mem::size_of::<SharedPtrHeader>(), 16);
        assert_eq!(std::mem::align_of::<SharedPtrHeader>(), 8);
        let h = SharedPtrHeader {
            ref_count: 1,
            _pad: 0,
            destroy: 0,
        };
        let base = &h as *const SharedPtrHeader as usize;
        assert_eq!(std::ptr::addr_of!(h.ref_count) as usize - base, 0);
        assert_eq!(std::ptr::addr_of!(h.destroy) as usize - base, 8);
        // kDataOff is 16 for every type the allocator can align.
        assert_eq!(data_offset::<u8>(), 16);
        assert_eq!(data_offset::<u64>(), 16);
        assert_eq!(data_offset::<Payload>(), 16);
        assert_eq!(data_offset::<[u64; 16]>(), 16);
    }

    #[test]
    fn make_shared_starts_at_one_and_round_trips() {
        let a = fresh(1 << 16);
        let p = make_shared(&a, Payload { a: 7, b: 8, c: 9 }).unwrap();
        assert_eq!(p.use_count(), 1);
        assert!(!p.is_null());
        // SAFETY: sole owner, no other thread/process touches this object.
        unsafe {
            assert_eq!(p.read(), Some(Payload { a: 7, b: 8, c: 9 }));
            assert_eq!(p.as_ref_unchecked().a, 7);
            p.write(Payload { a: 1, b: 2, c: 3 });
            assert_eq!(p.read().unwrap().c, 3);
        }
        // The Rust-written destroy slot must read as "nothing to destruct" so
        // that a C++ last-releaser frees without calling a wild pointer.
        // SAFETY: live header owned by `p`.
        assert_eq!(unsafe { (*p.header).destroy }, 0);
        // Payload is placed at kDataOff past the header, inside one block.
        assert_eq!(p.to_shm().off, p.block.off + 16);
        assert_eq!(p.to_shm().alloc_id, a.id());
    }

    #[test]
    fn clone_and_drop_track_count_then_free_once() {
        let a = fresh(1 << 16);
        let p = make_shared(&a, 0xDEAD_BEEF_u64).unwrap();
        let off = p.to_shm().off;
        {
            let q = p.clone();
            assert_eq!(p.use_count(), 2);
            assert_eq!(q.use_count(), 2);
            assert_eq!(p, q); // identity of the managed object
                              // SAFETY: no concurrent writer.
            assert_eq!(unsafe { q.read() }, Some(0xDEAD_BEEF));
            let r = q.clone();
            assert_eq!(r.use_count(), 3);
        } // q and r drop
        assert_eq!(p.use_count(), 1);
        drop(p);
        // The block is now free: the next same-size allocation reuses it
        // (first-fit over the free list), proving the last releaser freed.
        let p2 = make_shared(&a, 1_u64).unwrap();
        assert_eq!(p2.to_shm().off, off);
        assert_eq!(p2.use_count(), 1);
    }

    #[test]
    fn reset_is_idempotent_and_null_handles_are_inert() {
        let a = fresh(1 << 16);
        let mut p = make_shared(&a, 5_u64).unwrap();
        p.reset();
        assert!(p.is_null());
        assert_eq!(p.use_count(), 0);
        assert!(p.to_shm().is_null());
        assert!(p.get().is_null());
        p.reset(); // second reset must not double-free
        p.reset();
        assert!(p.is_null());

        // A default/null handle answers everything without touching memory.
        let n = SharedPtr::<u64, FreeListAllocator>::null();
        assert!(n.is_null());
        assert_eq!(n.use_count(), 0);
        assert!(n.to_shm().is_null());
        // SAFETY: null handle — read/write must report, not deref.
        unsafe {
            assert_eq!(n.read(), None);
            assert!(!n.write(1));
        }
        assert_eq!(n, SharedPtr::<u64, FreeListAllocator>::default());
        // Cloning a null handle is a no-op, not a null-header deref.
        let c = n.clone();
        assert_eq!(c.use_count(), 0);
        // Dropping twice-over is fine.
        drop(c);
        drop(n);
    }

    #[test]
    fn wrap_non_owning_counts_nothing_and_frees_nothing() {
        let a = fresh(1 << 16);
        let mut local: u64 = 42;
        {
            // SAFETY: `local` outlives the handle and is not freed by it.
            let w = unsafe {
                SharedPtr::<u64, FreeListAllocator>::wrap_non_owning(&mut local as *mut u64)
            };
            assert!(!w.is_null());
            assert_eq!(w.use_count(), 0); // no control block at all
            assert!(w.to_shm().is_null()); // not addressable cross-process
                                           // SAFETY: exclusive access in this test.
            assert_eq!(unsafe { w.read() }, Some(42));
            let w2 = w.clone(); // no count to touch
            assert_eq!(w2.use_count(), 0);
            assert_eq!(w, w2);
        } // drops must not free `local` (it is a stack object)
        assert_eq!(local, 42);
        // The allocator never saw a free: a fresh alloc still bump-allocates.
        let before = a.bump_remaining();
        let p = make_shared(&a, 1_u64).unwrap();
        assert!(a.bump_remaining() < before);
        drop(p);
    }

    #[test]
    fn cast_preserves_count_and_location() {
        let a = fresh(1 << 16);
        let p = make_shared(&a, Payload { a: 1, b: 2, c: 3 }).unwrap();
        let keep = p.clone();
        assert_eq!(p.use_count(), 2);
        let shm_before = p.to_shm();
        let raw_before = p.get() as usize;

        // Cast consumes and does NOT change the count (C++ Cast contract).
        let as_u64: SharedPtr<u64, _> = p.cast();
        assert_eq!(as_u64.use_count(), 2);
        assert_eq!(as_u64.get() as usize, raw_before);
        assert_eq!(as_u64.to_shm().off, shm_before.off);
        // SAFETY: sole test thread; Payload's first field is a u64.
        assert_eq!(unsafe { as_u64.read() }, Some(1));

        drop(as_u64);
        assert_eq!(keep.use_count(), 1);
        // Round-tripping back preserves the payload.
        let back: SharedPtr<Payload, _> = keep.cast();
        // SAFETY: sole test thread.
        assert_eq!(unsafe { back.read() }, Some(Payload { a: 1, b: 2, c: 3 }));
    }

    #[test]
    fn attach_shares_ownership_so_any_holder_can_be_last_releaser() {
        let a = fresh(1 << 16);
        let p = make_shared(&a, 0xABCD_u64).unwrap();
        let shm = p.to_shm(); // the value another process would receive
        let off = shm.off;

        // Stand-in for the far side: attach against the same allocator.
        // SAFETY: `p` still holds a reference, so the object is live.
        let attached = unsafe { SharedPtr::attach(&a, shm) }.unwrap();
        assert_eq!(p.use_count(), 2);
        assert_eq!(attached.use_count(), 2);
        assert_eq!(attached.get(), p.get());
        // SAFETY: no concurrent writer.
        assert_eq!(unsafe { attached.read() }, Some(0xABCD));

        // The CREATOR drops first; the attacher is the last releaser.
        drop(p);
        assert_eq!(attached.use_count(), 1);
        // SAFETY: still alive — this is the whole point of the contract.
        assert_eq!(unsafe { attached.read() }, Some(0xABCD));
        drop(attached);

        // Freed exactly once by the non-creator: block is reusable.
        let p2 = make_shared(&a, 1_u64).unwrap();
        assert_eq!(p2.to_shm().off, off);
    }

    #[test]
    fn attach_rejects_bad_input() {
        let a = fresh(1 << 16);
        let b = fresh(1 << 16);
        let p = make_shared(&a, 1_u64).unwrap();
        let shm = p.to_shm();

        // SAFETY: each call is rejected before any dereference.
        unsafe {
            // Null pointer.
            assert!(SharedPtr::<u64, _>::attach(&a, ShmPtr::<u64>::null()).is_none());
            // Foreign allocator id (would resolve to the wrong segment).
            assert!(SharedPtr::<u64, _>::attach(&b, shm).is_none());
            // Offset below the header: cannot be a make_shared payload.
            let bogus = ShmPtr::<u64>::new(a.id(), 8);
            assert!(SharedPtr::<u64, _>::attach(&a, bogus).is_none());
            // Offset past the end of the heap: registry refuses to resolve.
            let far = ShmPtr::<u64>::new(a.id(), a.heap_size() + 1024);
            assert!(SharedPtr::<u64, _>::attach(&a, far).is_none());
        }
        // The real object was untouched by the rejected attaches.
        assert_eq!(p.use_count(), 1);
    }

    #[test]
    fn make_shared_returns_none_when_exhausted() {
        // Tiny segment: the header+payload will not fit.
        let a = fresh(4096);
        assert!(make_shared(&a, [0u64; 4096]).is_none());
        // Failure must not have leaked the heap: a small alloc still works.
        let p = make_shared(&a, 1_u64).unwrap();
        assert_eq!(p.use_count(), 1);
    }

    #[test]
    fn zero_sized_payload_is_addressable_and_freed() {
        let a = fresh(1 << 16);
        let p = make_shared(&a, Zst).unwrap();
        assert!(!p.is_null());
        assert_eq!(p.use_count(), 1);
        // Header still precedes the (empty) payload.
        assert_eq!(p.to_shm().off, p.block.off + 16);
        let q = p.clone();
        assert_eq!(p.use_count(), 2);
        drop(q);
        // SAFETY: sole owner.
        assert_eq!(unsafe { p.read() }, Some(Zst));
        let off = p.to_shm().off;
        drop(p);
        // Block was freed and is reused.
        let r = make_shared(&a, Zst).unwrap();
        assert_eq!(r.to_shm().off, off);
    }

    #[test]
    fn refcount_overflow_panics_instead_of_wrapping() {
        let a = fresh(1 << 16);
        let p = make_shared(&a, 1_u64).unwrap();
        // Drive the in-segment count to the boundary (a corrupt/hostile
        // segment could present this; C++ would wrap to 0 and later free a
        // live object).
        // SAFETY: `p` holds a live reference, so the header is mapped.
        let rc = unsafe { ref_count_of(p.header) };
        rc.store(u32::MAX, Ordering::Relaxed);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _c = p.clone();
        }));
        assert!(res.is_err(), "clone at u32::MAX must panic");
        // The failed clone left the count intact (not wrapped to 0).
        assert_eq!(rc.load(Ordering::Relaxed), u32::MAX);
        // Restore so the drop below frees rather than merely decrementing.
        rc.store(1, Ordering::Relaxed);
        drop(p);
    }

    #[test]
    fn concurrent_clone_drop_keeps_object_alive() {
        use std::sync::Arc;
        let a = Arc::new(fresh(1 << 16));
        let p = make_shared(&*a, 0x5A5A_u64).unwrap();
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..500 {
                        let c = p.clone();
                        assert!(c.use_count() >= 1);
                        // SAFETY: nobody writes the payload in this test, and
                        // holding `c` keeps it alive.
                        assert_eq!(unsafe { c.read() }, Some(0x5A5A));
                        drop(c);
                    }
                });
            }
        });
        // Every clone was matched by a drop: back to the original reference.
        assert_eq!(p.use_count(), 1);
        // SAFETY: still the sole owner.
        assert_eq!(unsafe { p.read() }, Some(0x5A5A));
        let off = p.to_shm().off;
        drop(p);
        let p2 = make_shared(&*a, 1_u64).unwrap();
        assert_eq!(p2.to_shm().off, off); // freed exactly once
    }

    #[test]
    fn unique_ptr_owns_moves_and_frees() {
        let a = fresh(1 << 16);
        let u = make_unique(&a, Payload { a: 1, b: 2, c: 3 }).unwrap();
        assert!(!u.is_null());
        // No header: the payload sits at the block start.
        assert_eq!(u.to_shm().off, u.block.off);
        // SAFETY: sole owner, single-threaded test.
        unsafe {
            assert_eq!(u.read(), Some(Payload { a: 1, b: 2, c: 3 }));
            u.write(Payload { a: 9, b: 9, c: 9 });
        }
        let off = u.to_shm().off;
        // Move transfers ownership (C++ move ctor); no free happens here.
        let moved = u;
        // SAFETY: sole owner.
        assert_eq!(unsafe { moved.as_ref_unchecked().a }, 9);
        drop(moved);
        // Freed: block is reused.
        let u2 = make_unique(&a, 0_u64).unwrap();
        assert_eq!(u2.to_shm().off, off);
    }

    #[test]
    fn unique_ptr_reset_and_null_are_inert() {
        let a = fresh(1 << 16);
        let mut u = make_unique(&a, 7_u64).unwrap();
        u.reset();
        assert!(u.is_null());
        u.reset(); // must not double-free
        assert!(u.to_shm().is_null());
        assert!(u.get().is_null());

        let n = UniquePtr::<u64, FreeListAllocator>::default();
        assert!(n.is_null());
        // SAFETY: null handle — must report rather than deref.
        unsafe {
            assert_eq!(n.read(), None);
            assert!(!n.write(1));
        }
    }

    #[test]
    fn unique_ptr_release_relinquishes_without_freeing() {
        let a = fresh(1 << 16);
        let mut u = make_unique(&a, 0x1234_u64).unwrap();
        let off = u.to_shm().off;
        let raw = u.release();
        assert!(u.is_null()); // handle gave up ownership
        assert!(!raw.is_null());
        // SAFETY: memory was NOT freed by release(), so the object is live.
        assert_eq!(unsafe { *raw }, 0x1234);
        // Nothing was returned to the allocator: the next alloc takes new
        // bump space rather than reusing the released block.
        let other = make_unique(&a, 1_u64).unwrap();
        assert_ne!(other.to_shm().off, off);

        // release_shm hands back what an explicit free actually needs.
        let mut v = make_unique(&a, 0x99_u64).unwrap();
        let voff = v.to_shm().off;
        let shm = v.release_shm();
        assert!(v.is_null());
        assert_eq!(shm.off, voff);
        assert_eq!(shm.alloc_id, a.id());
        // SAFETY: the block is live and was allocated by this allocator; we
        // are the only claimant since the handle released it.
        unsafe { a.free(shm) };
        let w = make_unique(&a, 2_u64).unwrap();
        assert_eq!(w.to_shm().off, voff); // reused ⇒ the free landed
    }
}
