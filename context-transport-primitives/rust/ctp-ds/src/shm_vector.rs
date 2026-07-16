// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Shared-memory vector — the port of `ctp::ipc::vector<T, AllocT>`
//! (`include/clio_ctp/data_structures/ipc/vector.h`).
//!
//! Storage is a heap-relative [`OffsetPtr`], never a raw pointer, so the
//! header may itself live inside a segment and be read by any process that
//! maps it (MEMORY_DESIGN.md pillar 3). The type is `#[repr(C)]`, has no
//! `Drop`, and releases its block only through an explicit
//! [`ShmVector::destroy`] — the same discipline the C++ FreeBuffer/task
//! teardown protocol uses.
//!
//! The C++ header ships two vectors. This module ports the **ipc** one; the
//! **priv** one (`data_structures/priv/vector.h`) is private-memory-only and
//! is described under "priv::vector" below.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::ipc`) | Rust | Notes |
//! |---|---|---|
//! | `vector<T, AllocT>` | [`ShmVector<T>`] | allocator is a per-call argument, not a type parameter |
//! | `ShmContainer<AllocT>::this_` | `ShmVector::alloc_id` | divergence 1 |
//! | `size_`, `capacity_`, `data_` | `size`, `capacity`, `data` | `u64` (frozen ABI width), not `size_t` |
//! | `vector(AllocT*)` | [`ShmVector::new`] | |
//! | `vector(alloc, size, args...)` | [`ShmVector::filled`] | |
//! | `vector(alloc, first, last)`, `vector(alloc, init_list)` | [`ShmVector::from_slice`] | |
//! | `~vector()` | [`ShmVector::destroy`] | explicit; there is no `Drop` |
//! | `push_back`, `emplace_back` | [`ShmVector::push`] | |
//! | `insert(pos, v)`, `emplace(pos, args)` | [`ShmVector::insert`] | index instead of iterator |
//! | `erase(pos)` | [`ShmVector::remove`] | |
//! | `erase(first, last)` | [`ShmVector::erase_range`] | |
//! | `clear()` | [`ShmVector::clear`] | |
//! | `reserve(n)` | [`ShmVector::reserve`] | |
//! | `resize(n)` | [`ShmVector::resize`] | zero-fills, as the C++ `memset` path does |
//! | `resize(n, v)` | [`ShmVector::resize_with`] | |
//! | `shrink_to_fit()` | [`ShmVector::shrink_to_fit`] | |
//! | `size()`, `capacity()`, `empty()` | [`len`](ShmVector::len), [`capacity`](ShmVector::capacity), [`is_empty`](ShmVector::is_empty) | |
//! | `at(i)`, `operator[](i)` | [`get`](ShmVector::get), [`set`](ShmVector::set) | bounds-checked; divergence 7 |
//! | `front()`, `back()` | [`first`](ShmVector::first), [`last`](ShmVector::last) | `Option`, not UB on empty |
//! | `data()` | [`data_offset`](ShmVector::data_offset), [`as_slice`](ShmVector::as_slice) | |
//! | `begin/end/cbegin/cend`, `iterator`, `const_iterator` | [`as_slice`](ShmVector::as_slice), [`as_mut_slice`](ShmVector::as_mut_slice), [`to_vec`](ShmVector::to_vec) | divergence 8 |
//! | `operator==`, `operator!=` | `PartialEq` | divergence 15 |
//! | `AllocateStorage`, `DeallocateStorage`, `CopyElements`, `MoveElements`, `DestroyElements` | `realloc_to`, `deallocate_storage` | private helpers |
//! | (priv) `pop_back()` | [`ShmVector::pop`] | absent from `ipc::vector`; carried over |
//!
//! # Semantic divergences
//!
//! 1. **Allocator handle.** C++ `ShmContainer` stores `this_ = (size_t)this -
//!    (size_t)alloc`: a self-relative distance to a *process-local* allocator
//!    object, recovered by `GetAllocator()`. This port stores an
//!    [`AllocatorId`] and resolves `data` through the process-global registry
//!    (`base + off`), so nothing process-local is ever written into a
//!    segment (pillar 3). Consequence: `ShmVector` is **not** byte-compatible
//!    with C++ `ipc::vector` — same 32 bytes, but field 4 is an id rather
//!    than a `this_` delta, and the field order differs (C++ puts the base
//!    class first). Every operation that allocates/frees takes
//!    `&FreeListAllocator` explicitly.
//! 2. **`priv::vector` is not ported here.** It is a *private-memory* vector
//!    whose state is absolute pointers (`FullPtr<T> data_`, `AllocT* alloc_`)
//!    plus an inline SVO buffer — none of which may appear in a `#[repr(C)]`
//!    shm type (pillar 3), so it belongs in a future private-memory module.
//!    Its behavioral differences from the port, for the record: `SVO_SIZE = 8`
//!    elements live inline, so `capacity()` starts at 8 and the first heap
//!    growth is 16; `Grow(min)` doubles *in a loop* until `>= min` where the
//!    ipc path reserves the exact request; `alloc_ == nullptr` makes
//!    `reserve()` a silent success; it has copy/move ctors, `swap`,
//!    `shrink_to_fit` back into the SVO buffer, `FixupSvoPtr`,
//!    `resize_no_init`, allocator propagation to elements constructible from
//!    `AllocT*`, and an `at()` that throws `std::out_of_range`.
//! 3. **Growth policy (matched).** [`push`](ShmVector::push) and
//!    [`insert`](ShmVector::insert) grow `capacity == 0 ? 1 : capacity * 2`
//!    → 1, 2, 4, 8, 16, … exactly like `ipc::emplace_back`/`emplace`.
//!    [`reserve`](ShmVector::reserve)/[`resize`](ShmVector::resize) allocate
//!    the **exact** requested capacity with no doubling, also as in C++.
//!    (`priv::vector` would yield 8, 16, 32, … because of its SVO.)
//! 4. **Fallible operations return `bool` and never lose data.** C++
//!    `reserve`/`resize`/`push_back` return `void` and no-op on OOM. Worse,
//!    C++ `reserve` nulls `data_`/`size_`/`capacity_` *before* calling
//!    `AllocateStorage`, so an allocation failure **silently empties the
//!    vector and leaks the old block**; and C++ `resize` assigns
//!    `size_ = new_size` even when its `reserve` failed, leaving
//!    `size_ > capacity_ == 0` over null storage (a loaded gun). This port
//!    allocates first and commits only on success: on OOM the vector is left
//!    exactly as it was and `false` is returned. Deliberate divergence —
//!    the C++ behavior is a latent corruption bug, not a contract.
//! 5. **Leaks fixed.** C++ `reserve` frees the old block only when
//!    `old_size > 0`, so `clear(); reserve(n)` leaks; C++
//!    `operator=(const vector&)` calls `AllocateStorage` over a live block
//!    without freeing it. This port frees the old block whenever one exists.
//! 6. **Overflow is checked.** `capacity * 2` and `capacity * sizeof(T)` wrap
//!    silently in C++ (undersized allocation → heap overflow). Here both are
//!    checked; overflow returns `false` and changes nothing.
//! 7. **Bounds are checked.** C++ `ipc::at()` documents bounds checking but
//!    performs none (it is `operator[]` verbatim), and `front()`/`back()` on
//!    an empty vector are UB. [`get`](ShmVector::get)/[`set`](ShmVector::set)/
//!    [`first`](ShmVector::first)/[`last`](ShmVector::last)/
//!    [`insert`](ShmVector::insert)/[`remove`](ShmVector::remove)/
//!    [`erase_range`](ShmVector::erase_range) all reject out-of-range indices
//!    (`None`/`false`) instead of corrupting the segment.
//! 8. **Iterators → slices/indices.** The C++ `iterator`/`const_iterator`
//!    classes are raw-`T*` wrappers; Rust gets real iterators via
//!    [`as_slice`](ShmVector::as_slice)/[`as_mut_slice`](ShmVector::as_mut_slice)
//!    (both `unsafe`: they hand out a borrow whose lifetime the compiler
//!    cannot tie to the segment mapping) and a safe copying
//!    [`to_vec`](ShmVector::to_vec).
//! 9. **`T: ShmSafe` (POD, `Copy`).** Every non-POD C++ path — placement new,
//!    move construction, `~T()`, and the `IS_SHM_CONTAINER(T)` branch that
//!    passes the allocator to element constructors — collapses:
//!    [`clear`](ShmVector::clear)/[`destroy`](ShmVector::destroy) never run
//!    element destructors. Consequently **nested shm containers**
//!    (`ipc::vector<ipc::vector<int>>`) are unsupported: `ShmSafe: Copy` and
//!    `ShmVector` is deliberately not `Copy`/`Clone` (C++ deletes its copy
//!    and move constructors for the same reason).
//! 10. **No `Drop`.** A dropped `ShmVector` leaks its block until
//!     [`destroy`](ShmVector::destroy) is called — the explicit-teardown rule
//!     of MEMORY_DESIGN pillar 2, and the reason `~vector()` becomes a named
//!     method.
//! 11. **A zeroed struct is not a valid vector.** Null is `u64::MAX`, and
//!     `off == 0` is a legitimate heap offset, so zeroed shm bytes are *not*
//!     an empty vector. Same as C++ (`OffsetPtr::GetNull()`), but worth
//!     stating for a `#[repr(C)]` type that looks zero-constructible.
//! 12. **Allocator mismatch is rejected.** Passing an allocator whose id
//!     differs from the vector's is a no-op returning `false` (C++ would
//!     compute offsets against the wrong heap). `ipc::vector` has no
//!     equivalent check.
//! 13. **Alignment.** The v1 allocator guarantees 16-byte payload alignment,
//!     so [`new`](ShmVector::new) asserts `align_of::<T>() <= 16`. C++ has no
//!     such check.
//! 14. **`insert` into never-allocated storage fails** — faithfully. C++
//!     `emplace` resolves `FullPtr(alloc, data_)` and returns a null iterator
//!     *before* considering growth, so inserting into a freshly constructed
//!     `ipc::vector` (`data_` null) does nothing. Preserved, quirk and all;
//!     use [`push`](ShmVector::push) or [`reserve`](ShmVector::reserve) first.
//! 15. **`PartialEq` quirk preserved.** C++ `operator==` returns `false` when
//!     one side has storage and the other does not — *even when both are
//!     empty* (a `clear()`ed vector != a fresh one). Mirrored, including the
//!     `!alloc → size_ == 0` fallback, which here means "unresolvable
//!     allocator".
//! 16. **Serialization not ported.** `priv::vector::save`/`load` (cereal via
//!     `save_vec`/`load_vec`) belong to the `ctp-serialize` surface.
//! 17. **`emplace_back`/`emplace` fold into `push`/`insert`.** With `T: Copy`
//!     there is no observable difference between in-place construction from
//!     forwarded arguments and a by-value write.
//!
//! # Safety model
//!
//! Resolution (`registry::resolve`) is safe and bounds-checked; *dereferencing*
//! the result is not — exactly the split `ctp-memory` defines. Every method
//! below resolves and reads/writes within the call, under this contract: **the
//! owning allocator must remain registered and its segment mapped for the
//! duration of the call** (the C++ `FullPtr(alloc, off)` contract). If the
//! allocator is gone, resolution fails and the operation reports failure
//! instead of touching memory.

#![deny(unsafe_op_in_unsafe_fn)]

use ctp_memory::{resolve, AllocatorId, FreeListAllocator, OffsetPtr, ShmSafe};

/// Payload alignment guaranteed by the v1 allocator (`ALIGN` in
/// `ctp-memory/src/allocator.rs`). Element types must not exceed it.
const MAX_ALIGN: usize = 16;

/// Shared-memory vector (C++ `ctp::ipc::vector<T, AllocT>`).
///
/// 32 bytes of pure position-independent state: two counters, a heap-relative
/// offset, and the id of the allocator that offset belongs to. Not `Copy`,
/// not `Clone` (C++ deletes both), and no `Drop` — call
/// [`destroy`](Self::destroy) to release the block.
#[repr(C)]
pub struct ShmVector<T> {
    /// C++ `size_` — number of live elements.
    size: u64,
    /// C++ `capacity_` — elements the current block can hold.
    capacity: u64,
    /// C++ `data_` — heap-relative offset of the element array.
    data: OffsetPtr<T>,
    /// Replaces C++ `ShmContainer::this_` (divergence 1).
    alloc_id: AllocatorId,
}

impl<T: ShmSafe> ShmVector<T> {
    /// C++ `vector(AllocT *alloc)` — empty, no storage allocated.
    ///
    /// # Panics
    /// If `align_of::<T>() > 16`, which the v1 allocator cannot satisfy.
    pub fn new(alloc: &FreeListAllocator) -> Self {
        Self::with_alloc_id(alloc.id())
    }

    /// Construct against an allocator known only by id (the cross-process
    /// form: the segment may be owned by another process or language).
    ///
    /// # Panics
    /// If `align_of::<T>() > 16`.
    pub fn with_alloc_id(alloc_id: AllocatorId) -> Self {
        assert!(
            std::mem::align_of::<T>() <= MAX_ALIGN,
            "ShmVector element alignment {} exceeds the allocator's {}-byte guarantee",
            std::mem::align_of::<T>(),
            MAX_ALIGN
        );
        Self {
            size: 0,
            capacity: 0,
            data: OffsetPtr::null(),
            alloc_id,
        }
    }

    /// C++ `vector(AllocT *alloc, size_t size, Args&&... args)` — `count`
    /// copies of `value`, capacity exactly `count`.
    ///
    /// Mirrors the C++ contract that an allocation failure yields an empty
    /// vector rather than an error: check [`len`](Self::len) if it matters.
    pub fn filled(alloc: &FreeListAllocator, count: usize, value: T) -> Self {
        let mut v = Self::new(alloc);
        if count > 0 && v.realloc_to(alloc, count as u64) {
            if let Some(ptr) = v.data_ptr() {
                for i in 0..count {
                    // SAFETY: the block just allocated holds `count`
                    // elements; `i < count` and the segment is mapped for
                    // the duration of this call.
                    unsafe { ptr.add(i).write(value) };
                }
                v.size = count as u64;
            }
        }
        v
    }

    /// C++ `vector(alloc, first, last)` / `vector(alloc, init_list)` —
    /// capacity exactly `src.len()`. Empty on allocation failure (as C++).
    pub fn from_slice(alloc: &FreeListAllocator, src: &[T]) -> Self {
        let mut v = Self::new(alloc);
        if !src.is_empty() && v.realloc_to(alloc, src.len() as u64) {
            if let Some(dst) = v.data_ptr() {
                // SAFETY: `dst` owns a fresh block of `src.len()` elements in
                // the mapped segment; `src` is a live slice elsewhere, so the
                // ranges cannot overlap.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
                v.size = src.len() as u64;
            }
        }
        v
    }

    /// The allocator this vector's offsets belong to.
    pub fn alloc_id(&self) -> AllocatorId {
        self.alloc_id
    }

    /// C++ `data()`, in the offset form that is safe to publish.
    pub fn data_offset(&self) -> OffsetPtr<T> {
        self.data
    }

    /// C++ `size()`.
    pub fn len(&self) -> usize {
        self.size as usize
    }

    /// C++ `capacity()`.
    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// C++ `empty()`.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Bounds-checked read (C++ `at`/`operator[]`, which check nothing).
    pub fn get(&self, idx: usize) -> Option<T> {
        if idx as u64 >= self.size {
            return None;
        }
        let ptr = self.data_ptr()?;
        // SAFETY: `idx < size <= capacity`, so the element is inside the
        // resolved block, and `T: ShmSafe` is valid for any bit pattern.
        Some(unsafe { ptr.add(idx).read() })
    }

    /// Bounds-checked write (C++ `operator[]` assignment). `false` when out
    /// of range or the allocator is unresolvable.
    pub fn set(&mut self, idx: usize, value: T) -> bool {
        if idx as u64 >= self.size {
            return false;
        }
        match self.data_ptr() {
            // SAFETY: `idx < size <= capacity` — inside the resolved block.
            Some(ptr) => {
                unsafe { ptr.add(idx).write(value) };
                true
            }
            None => false,
        }
    }

    /// C++ `front()` — `None` instead of UB on empty.
    pub fn first(&self) -> Option<T> {
        self.get(0)
    }

    /// C++ `back()` — `None` instead of UB on empty.
    pub fn last(&self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        self.get((self.size - 1) as usize)
    }

    /// C++ `push_back`/`emplace_back`: grow `cap == 0 ? 1 : cap * 2` when
    /// full, then append. `false` on OOM/overflow/allocator mismatch, with
    /// the vector left untouched (divergence 4).
    pub fn push(&mut self, alloc: &FreeListAllocator, value: T) -> bool {
        if alloc.id() != self.alloc_id {
            return false;
        }
        if self.size >= self.capacity && !self.grow_double(alloc) {
            return false;
        }
        let ptr = match self.data_ptr() {
            Some(p) => p,
            // Mirrors the C++ `if (fp.ptr_)` guard: no storage, no append.
            None => return false,
        };
        // SAFETY: growth above guarantees `size < capacity`, so slot `size`
        // is inside the resolved block.
        unsafe { ptr.add(self.size as usize).write(value) };
        self.size += 1;
        true
    }

    /// `priv::vector::pop_back()`, which returns the removed element here
    /// because `T: Copy` makes it free. `None` when empty (C++ no-ops).
    /// Capacity is unchanged.
    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        let val = self.get((self.size - 1) as usize)?;
        self.size -= 1;
        Some(val)
    }

    /// C++ `insert(pos, value)`/`emplace(pos, args...)` by index.
    ///
    /// Returns `false` — changing nothing — when `idx > len()`, on
    /// OOM/overflow/allocator mismatch, or when the vector has never
    /// allocated storage (divergence 14: the C++ quirk, preserved).
    pub fn insert(&mut self, alloc: &FreeListAllocator, idx: usize, value: T) -> bool {
        if alloc.id() != self.alloc_id {
            return false;
        }
        // C++ emplace resolves data_ and bails out before growing.
        if self.data.is_null() {
            return false;
        }
        if idx as u64 > self.size {
            return false;
        }
        if self.size >= self.capacity && !self.grow_double(alloc) {
            return false;
        }
        let ptr = match self.data_ptr() {
            Some(p) => p,
            None => return false,
        };
        let tail = (self.size - idx as u64) as usize;
        // SAFETY: `size < capacity` after growth, so the shifted range
        // `[idx, size]` and the write at `idx` are inside the block.
        // `copy` is memmove — the ranges overlap by design.
        unsafe {
            std::ptr::copy(ptr.add(idx), ptr.add(idx + 1), tail);
            ptr.add(idx).write(value);
        }
        self.size += 1;
        true
    }

    /// C++ `erase(pos)` — shift the tail left. Returns the removed element
    /// (C++ returns an iterator to the follower); `None` when out of range,
    /// empty, or unresolvable. Capacity is unchanged.
    pub fn remove(&mut self, idx: usize) -> Option<T> {
        if idx as u64 >= self.size {
            return None;
        }
        let ptr = self.data_ptr()?;
        // SAFETY: `idx < size <= capacity`; the shifted tail `[idx+1, size)`
        // is inside the block. `copy` is memmove (overlapping by design).
        let val = unsafe {
            let val = ptr.add(idx).read();
            std::ptr::copy(
                ptr.add(idx + 1),
                ptr.add(idx),
                (self.size - idx as u64 - 1) as usize,
            );
            val
        };
        self.size -= 1;
        Some(val)
    }

    /// C++ `erase(first, last)` — remove `[first, last)`. `false` (no change)
    /// when the range is invalid or the allocator is unresolvable. An empty
    /// range succeeds without touching memory, as in C++.
    pub fn erase_range(&mut self, first: usize, last: usize) -> bool {
        if first > last || last as u64 > self.size {
            return false;
        }
        let count = (last - first) as u64;
        if count == 0 {
            return true;
        }
        let ptr = match self.data_ptr() {
            Some(p) => p,
            None => return false,
        };
        // SAFETY: `last <= size <= capacity`, so both the erased range and
        // the shifted tail `[last, size)` are inside the block.
        unsafe {
            std::ptr::copy(
                ptr.add(last),
                ptr.add(first),
                (self.size - last as u64) as usize,
            );
        }
        self.size -= count;
        true
    }

    /// C++ `clear()` — size to zero, storage and capacity retained.
    /// (`DestroyElements` is a no-op for `T: ShmSafe`.)
    pub fn clear(&mut self) {
        self.size = 0;
    }

    /// C++ `reserve(n)` — allocate **exactly** `n` elements when `n` exceeds
    /// the current capacity; no-op (returning `true`) otherwise. `false` on
    /// OOM/overflow/allocator mismatch, vector unchanged (divergence 4).
    pub fn reserve(&mut self, alloc: &FreeListAllocator, new_capacity: usize) -> bool {
        if new_capacity as u64 <= self.capacity {
            return true;
        }
        self.realloc_to(alloc, new_capacity as u64)
    }

    /// C++ `shrink_to_fit()` — capacity down to exactly `len()`; an empty
    /// vector releases its block entirely. `false` on OOM/mismatch (C++
    /// returns `void` and would drop the elements — divergence 4).
    pub fn shrink_to_fit(&mut self, alloc: &FreeListAllocator) -> bool {
        if self.capacity == self.size || self.size == 0 {
            if self.size == 0 {
                self.deallocate_storage(alloc);
            }
            return true;
        }
        self.realloc_to(alloc, self.size)
    }

    /// C++ `resize(n)` — grow with **zero-filled** elements (the C++
    /// `memset` path for trivially constructible types; sound here because
    /// `ShmSafe` asserts validity for any bit pattern) or shrink. `false` on
    /// OOM/overflow/mismatch, vector unchanged.
    pub fn resize(&mut self, alloc: &FreeListAllocator, new_size: usize) -> bool {
        self.resize_impl(alloc, new_size, None)
    }

    /// C++ `resize(n, value)` — grow by copying `value`, or shrink.
    pub fn resize_with(&mut self, alloc: &FreeListAllocator, new_size: usize, value: T) -> bool {
        self.resize_impl(alloc, new_size, Some(value))
    }

    /// C++ `~vector()` (`DestroyElements` + `DeallocateStorage`), made
    /// explicit: no `Drop` for a shared-memory type. Idempotent; the vector
    /// stays usable (empty, no storage) afterwards.
    pub fn destroy(&mut self, alloc: &FreeListAllocator) {
        self.deallocate_storage(alloc);
    }

    /// Copy the elements out into private memory — the safe way to read a
    /// vector whole (no borrow of shared state escapes).
    pub fn to_vec(&self) -> Vec<T> {
        match self.data_ptr() {
            // SAFETY: `size <= capacity` elements are initialized inside the
            // resolved block; the copy finishes before this call returns.
            Some(ptr) => unsafe {
                std::slice::from_raw_parts(ptr as *const T, self.size as usize).to_vec()
            },
            None => Vec::new(),
        }
    }

    /// Borrow the elements in place (C++ `begin()`/`end()`).
    ///
    /// # Safety
    /// The returned borrow lives in shared memory: the owning allocator must
    /// stay registered and its segment mapped for the whole lifetime of the
    /// slice, and no other process/thread may mutate or free the block
    /// meanwhile. Rust cannot check any of that — hence `unsafe`. Prefer
    /// [`to_vec`](Self::to_vec) or [`get`](Self::get) when in doubt.
    pub unsafe fn as_slice(&self) -> &[T] {
        match self.data_ptr() {
            // SAFETY: forwarded caller contract; `size` elements of the
            // resolved block are initialized.
            Some(ptr) => unsafe { std::slice::from_raw_parts(ptr as *const T, self.size as usize) },
            None => &[],
        }
    }

    /// Mutable in-place borrow (C++ non-const `begin()`/`end()`).
    ///
    /// # Safety
    /// As [`as_slice`](Self::as_slice), plus: the caller must ensure no other
    /// view of this block is live, in this or any other process.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        match self.data_ptr() {
            // SAFETY: forwarded caller contract; `size` elements of the
            // resolved block are initialized and uniquely borrowed.
            Some(ptr) => unsafe { std::slice::from_raw_parts_mut(ptr, self.size as usize) },
            None => &mut [],
        }
    }

    // ---- private helpers (C++ AllocateStorage/DeallocateStorage/Copy...) ----

    /// Resolve `data` in this process (C++ `FullPtr(GetAllocator(), data_)`).
    fn data_ptr(&self) -> Option<*mut T> {
        if self.data.is_null() {
            return None;
        }
        resolve(self.data.to_shm(self.alloc_id))
    }

    /// The C++ `capacity_ == 0 ? 1 : capacity_ * 2` growth step shared by
    /// `emplace_back` and `emplace`.
    fn grow_double(&mut self, alloc: &FreeListAllocator) -> bool {
        let new_capacity = if self.capacity == 0 {
            1
        } else {
            match self.capacity.checked_mul(2) {
                Some(c) => c,
                None => return false,
            }
        };
        self.realloc_to(alloc, new_capacity)
    }

    /// Move to a block of exactly `new_capacity` elements, keeping
    /// `min(size, new_capacity)` of them, and free the old block.
    ///
    /// Unlike C++ `reserve` this allocates *before* disturbing any state, so
    /// a failure leaves the vector byte-identical (divergences 4, 5, 6).
    fn realloc_to(&mut self, alloc: &FreeListAllocator, new_capacity: u64) -> bool {
        if alloc.id() != self.alloc_id {
            return false;
        }
        if new_capacity == 0 {
            self.deallocate_storage(alloc);
            return true;
        }
        let bytes = match new_capacity.checked_mul(std::mem::size_of::<T>() as u64) {
            Some(b) => b,
            None => return false,
        };
        let new_off = match alloc.alloc_bytes(bytes) {
            Some(o) => o,
            None => return false,
        };
        let new_data: OffsetPtr<T> = OffsetPtr::new(new_off.off);
        let keep = self.size.min(new_capacity);
        if keep > 0 {
            match (self.data_ptr(), resolve(new_data.to_shm(self.alloc_id))) {
                // SAFETY: `keep <= size` elements are initialized in the old
                // block; the new block is a distinct fresh allocation of
                // `new_capacity >= keep` elements, so the ranges are disjoint.
                (Some(src), Some(dst)) => unsafe {
                    std::ptr::copy_nonoverlapping(src, dst, keep as usize)
                },
                _ => {
                    // Unresolvable segment: give the new block back and keep
                    // the vector exactly as it was.
                    // SAFETY: `new_off` came from this allocator moments ago
                    // and has not been freed.
                    unsafe { alloc.free_bytes(new_off) };
                    return false;
                }
            }
        }
        if !self.data.is_null() {
            // SAFETY: `data` was allocated by this same allocator (id checked
            // above) and is live until now.
            unsafe { alloc.free_bytes(OffsetPtr::new(self.data.off)) };
        }
        self.data = new_data;
        self.capacity = new_capacity;
        self.size = keep;
        true
    }

    /// C++ `DeallocateStorage()`.
    fn deallocate_storage(&mut self, alloc: &FreeListAllocator) {
        if self.data.is_null() || alloc.id() != self.alloc_id {
            return;
        }
        // SAFETY: `data` was allocated by this allocator (id checked) and is
        // live; it is nulled immediately below, so no double free.
        unsafe { alloc.free_bytes(OffsetPtr::new(self.data.off)) };
        self.data = OffsetPtr::null();
        self.capacity = 0;
        self.size = 0;
    }

    /// Shared body of C++ `resize(n)` (fill `None` → zero) and
    /// `resize(n, value)` (fill `Some(value)`).
    fn resize_impl(
        &mut self,
        alloc: &FreeListAllocator,
        new_size: usize,
        value: Option<T>,
    ) -> bool {
        let n = new_size as u64;
        if n == self.size {
            return true;
        }
        if alloc.id() != self.alloc_id {
            return false;
        }
        if n < self.size {
            // Element destructors are a no-op for `T: ShmSafe`.
            self.size = n;
            return true;
        }
        if n > self.capacity && !self.reserve(alloc, new_size) {
            return false;
        }
        let ptr = match self.data_ptr() {
            Some(p) => p,
            None => return false,
        };
        let start = self.size as usize;
        match value {
            // SAFETY: `[size, n)` lies inside the block (`n <= capacity`).
            None => unsafe { std::ptr::write_bytes(ptr.add(start), 0, new_size - start) },
            Some(v) => {
                for i in start..new_size {
                    // SAFETY: as above, one element at a time.
                    unsafe { ptr.add(i).write(v) };
                }
            }
        }
        self.size = n;
        true
    }
}

/// C++ `operator==`, quirks included (divergence 15).
impl<T: ShmSafe + PartialEq> PartialEq for ShmVector<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        match (self.data_ptr(), other.data_ptr()) {
            // C++: `!fp.ptr_ && !other_fp.ptr_` → true.
            (None, None) => true,
            // SAFETY: both blocks hold `size` initialized elements and stay
            // mapped for this call (module safety model).
            (Some(a), Some(b)) => unsafe {
                std::slice::from_raw_parts(a as *const T, self.size as usize)
                    == std::slice::from_raw_parts(b as *const T, other.size as usize)
            },
            // C++: one side unresolvable/never-allocated → false, even when
            // both vectors are empty.
            _ => false,
        }
    }
}

impl<T> std::fmt::Debug for ShmVector<T> {
    /// Prints the header only — never resolves, so it is valid even when the
    /// segment is gone.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShmVector")
            .field("size", &self.size)
            .field("capacity", &self.capacity)
            .field("data", &self.data)
            .field("alloc_id", &self.alloc_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ctp_memory::SharedMemBackend;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Module-private allocator major: the registry is process-global and
    /// every module in this crate shares one test binary, so a major unique
    /// to shm_vector keeps parallel tests from unregistering each other.
    const TEST_MAJOR: u32 = 0x5348_5657; // "SHVW"

    fn fresh(tag: &str, bytes: usize) -> FreeListAllocator {
        static NEXT_MINOR: AtomicU32 = AtomicU32::new(1);
        let minor = NEXT_MINOR.fetch_add(1, Ordering::Relaxed);
        let name = format!("ctp_rs_shmvec_{}_{}_{}", tag, std::process::id(), minor);
        let backend = SharedMemBackend::create(&name, bytes).unwrap();
        backend.destroy(); // unlink the name eagerly (POSIX); mapping stays
        FreeListAllocator::create(backend, AllocatorId::new(TEST_MAJOR, minor))
    }

    #[test]
    fn layout_is_frozen_and_dropless() {
        // C++ ipc::vector is {this_(8), size_(8), capacity_(8), data_(8)}.
        assert_eq!(std::mem::size_of::<ShmVector<u32>>(), 32);
        assert_eq!(std::mem::size_of::<ShmVector<[u64; 16]>>(), 32);
        assert_eq!(std::mem::align_of::<ShmVector<u32>>(), 8);
        // Shared-memory types never run destructors implicitly.
        assert!(!std::mem::needs_drop::<ShmVector<u32>>());
    }

    #[test]
    fn new_is_empty_with_no_storage() {
        let a = fresh("new", 1 << 16);
        let v = ShmVector::<u64>::new(&a);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        assert!(v.is_empty());
        assert!(v.data_offset().is_null());
        assert_eq!(v.alloc_id(), a.id());
        assert_eq!(v.get(0), None);
        assert_eq!(v.first(), None);
        assert_eq!(v.last(), None);
        assert_eq!(v.to_vec(), Vec::<u64>::new());
    }

    #[test]
    fn push_growth_doubles_from_one() {
        // C++ emplace_back: capacity_ == 0 ? 1 : capacity_ * 2.
        let a = fresh("growth", 1 << 16);
        let mut v = ShmVector::<u32>::new(&a);
        let expected = [1usize, 2, 4, 4, 8, 8, 8, 8, 16, 16];
        for (i, &want) in expected.iter().enumerate() {
            assert!(v.push(&a, i as u32));
            assert_eq!(v.len(), i + 1);
            assert_eq!(v.capacity(), want, "capacity after push #{}", i + 1);
        }
        assert_eq!(v.to_vec(), (0..10u32).collect::<Vec<_>>());
        v.destroy(&a);
    }

    #[test]
    fn index_read_write_and_bounds() {
        let a = fresh("index", 1 << 16);
        let mut v = ShmVector::<u64>::new(&a);
        for i in 0..8u64 {
            assert!(v.push(&a, i * 10));
        }
        assert_eq!(v.get(0), Some(0));
        assert_eq!(v.get(7), Some(70));
        assert_eq!(v.first(), Some(0));
        assert_eq!(v.last(), Some(70));
        // Boundary: index == len is out of range.
        assert_eq!(v.get(8), None);
        assert_eq!(v.get(usize::MAX), None);
        assert!(v.set(3, 999));
        assert_eq!(v.get(3), Some(999));
        assert!(!v.set(8, 1));
        // The in-place borrow sees the same bytes.
        // SAFETY: `a` outlives the borrow and nothing else touches the block.
        unsafe {
            assert_eq!(v.as_slice().len(), 8);
            assert_eq!(v.as_slice()[3], 999);
            v.as_mut_slice()[3] = 4;
        }
        assert_eq!(v.get(3), Some(4));
        v.destroy(&a);
    }

    #[test]
    fn reserve_is_exact_and_never_shrinks() {
        let a = fresh("reserve", 1 << 16);
        let mut v = ShmVector::<u32>::new(&a);
        assert!(v.push(&a, 7));
        assert_eq!(v.capacity(), 1);
        // C++ reserve allocates exactly what was asked for — no doubling.
        assert!(v.reserve(&a, 10));
        assert_eq!(v.capacity(), 10);
        assert_eq!(v.len(), 1);
        assert_eq!(v.get(0), Some(7)); // elements survive the move
                                       // Smaller/equal requests are no-ops that still succeed.
        assert!(v.reserve(&a, 10));
        assert!(v.reserve(&a, 5));
        assert!(v.reserve(&a, 0));
        assert_eq!(v.capacity(), 10);
        // Growth after reserve resumes doubling from the reserved capacity.
        for i in 0..10u32 {
            assert!(v.push(&a, i));
        }
        assert_eq!(v.capacity(), 20);
        v.destroy(&a);
    }

    #[test]
    fn reserve_reuses_the_freed_block_and_does_not_leak() {
        // C++ reserve() skips the free when old_size == 0, so `clear();
        // reserve(n)` leaks the old block (divergence 5). Ours frees it.
        let a = fresh("noleak", 1 << 16);
        let mut v = ShmVector::<u64>::new(&a);
        assert!(v.reserve(&a, 4));
        let first_off = v.data_offset().off; // 4 * 8 = 32 payload bytes
        v.clear();
        assert!(v.reserve(&a, 8)); // the empty block must be freed here
        assert_ne!(v.data_offset().off, first_off);
        // Proof it reached the free list: the next 32-byte request is served
        // from that exact block and consumes no new bump space.
        let before = a.bump_remaining();
        let reused = a.alloc_bytes(32).unwrap();
        assert_eq!(reused.off, first_off);
        assert_eq!(a.bump_remaining(), before);
        // SAFETY: just allocated by this allocator, not freed since.
        unsafe { a.free_bytes(reused) };
        v.destroy(&a);
    }

    #[test]
    fn clear_keeps_capacity_and_storage() {
        let a = fresh("clear", 1 << 16);
        let mut v = ShmVector::<u32>::new(&a);
        for i in 0..5u32 {
            assert!(v.push(&a, i));
        }
        let off = v.data_offset().off;
        let cap = v.capacity();
        v.clear();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), cap); // C++ clear() only destroys elements
        assert_eq!(v.data_offset().off, off);
        assert_eq!(v.get(0), None);
        // Refilling reuses the same block.
        assert!(v.push(&a, 42));
        assert_eq!(v.data_offset().off, off);
        assert_eq!(v.get(0), Some(42));
        v.destroy(&a);
    }

    #[test]
    fn destroy_releases_the_block_and_is_idempotent() {
        let a = fresh("destroy", 1 << 16);
        let mut v = ShmVector::<u64>::new(&a);
        for i in 0..20u64 {
            assert!(v.push(&a, i));
        }
        let before = a.bump_remaining();
        v.destroy(&a);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);
        assert!(v.data_offset().is_null());
        assert_eq!(v.to_vec(), Vec::<u64>::new());
        // Destroying twice must not double free.
        v.destroy(&a);
        assert!(v.data_offset().is_null());
        // The vector is still usable afterwards (C++ dtor leaves a shell;
        // ours leaves a valid empty vector) and consumes no new bump space:
        // the freed blocks are reused.
        assert!(v.push(&a, 1));
        assert_eq!(a.bump_remaining(), before);
        assert_eq!(v.to_vec(), vec![1]);
        v.destroy(&a);
    }

    #[test]
    fn resize_zero_fills_grows_and_shrinks() {
        let a = fresh("resize", 1 << 16);
        let mut v = ShmVector::<u64>::new(&a);
        assert!(v.push(&a, 0xFFFF));
        // Grow: new elements are zeroed (the C++ memset path).
        assert!(v.resize(&a, 4));
        assert_eq!(v.to_vec(), vec![0xFFFF, 0, 0, 0]);
        // resize reserves the exact size — no doubling.
        assert_eq!(v.capacity(), 4);
        // Shrink: size only, capacity retained.
        assert!(v.resize(&a, 2));
        assert_eq!(v.to_vec(), vec![0xFFFF, 0]);
        assert_eq!(v.capacity(), 4);
        // Same-size resize is a no-op.
        assert!(v.resize(&a, 2));
        assert_eq!(v.len(), 2);
        // Zero.
        assert!(v.resize(&a, 0));
        assert!(v.is_empty());
        assert_eq!(v.capacity(), 4);
        // Growing from empty re-fills with zeros.
        assert!(v.resize(&a, 3));
        assert_eq!(v.to_vec(), vec![0, 0, 0]);
        v.destroy(&a);
    }

    #[test]
    fn resize_with_fills_with_value() {
        let a = fresh("resizeval", 1 << 16);
        let mut v = ShmVector::<u32>::new(&a);
        assert!(v.resize_with(&a, 3, 9));
        assert_eq!(v.to_vec(), vec![9, 9, 9]);
        assert!(v.resize_with(&a, 5, 4));
        assert_eq!(v.to_vec(), vec![9, 9, 9, 4, 4]);
        assert!(v.resize_with(&a, 1, 0));
        assert_eq!(v.to_vec(), vec![9]);
        // Growing a vector with no storage at all.
        let mut e = ShmVector::<u32>::new(&a);
        assert!(e.resize_with(&a, 2, 7));
        assert_eq!(e.to_vec(), vec![7, 7]);
        e.destroy(&a);
        v.destroy(&a);
    }

    #[test]
    fn shrink_to_fit_matches_cpp_cases() {
        let a = fresh("shrink", 1 << 16);
        let mut v = ShmVector::<u32>::new(&a);
        for i in 0..5u32 {
            assert!(v.push(&a, i));
        }
        assert_eq!(v.capacity(), 8);
        // capacity > size → exact refit.
        assert!(v.shrink_to_fit(&a));
        assert_eq!(v.capacity(), 5);
        assert_eq!(v.to_vec(), (0..5u32).collect::<Vec<_>>());
        // capacity == size → no-op.
        let off = v.data_offset().off;
        assert!(v.shrink_to_fit(&a));
        assert_eq!(v.capacity(), 5);
        assert_eq!(v.data_offset().off, off);
        // size == 0 → storage released entirely (C++ DeallocateStorage).
        v.clear();
        assert!(v.shrink_to_fit(&a));
        assert_eq!(v.capacity(), 0);
        assert!(v.data_offset().is_null());
        // ... and on a vector that never allocated.
        let mut e = ShmVector::<u32>::new(&a);
        assert!(e.shrink_to_fit(&a));
        assert_eq!(e.capacity(), 0);
        v.destroy(&a);
    }

    #[test]
    fn insert_shifts_right_and_honors_the_cpp_null_storage_quirk() {
        let a = fresh("insert", 1 << 16);
        // Divergence 14: C++ emplace() bails out when data_ is null, so a
        // fresh vector cannot be inserted into.
        let mut fresh_v = ShmVector::<u32>::new(&a);
        assert!(!fresh_v.insert(&a, 0, 1));
        assert!(fresh_v.is_empty());

        let mut v = ShmVector::<u32>::from_slice(&a, &[1, 2, 4]);
        assert!(v.insert(&a, 2, 3)); // middle
        assert_eq!(v.to_vec(), vec![1, 2, 3, 4]);
        assert!(v.insert(&a, 0, 0)); // front
        assert_eq!(v.to_vec(), vec![0, 1, 2, 3, 4]);
        let n = v.len();
        assert!(v.insert(&a, n, 5)); // back (idx == len is legal)
        assert_eq!(v.to_vec(), vec![0, 1, 2, 3, 4, 5]);
        // Out of range changes nothing.
        assert!(!v.insert(&a, v.len() + 1, 99));
        assert_eq!(v.to_vec(), vec![0, 1, 2, 3, 4, 5]);
        v.destroy(&a);
    }

    #[test]
    fn remove_and_pop_shift_left() {
        let a = fresh("remove", 1 << 16);
        let mut v = ShmVector::<u32>::from_slice(&a, &[0, 1, 2, 3, 4]);
        assert_eq!(v.remove(2), Some(2)); // middle
        assert_eq!(v.to_vec(), vec![0, 1, 3, 4]);
        assert_eq!(v.remove(0), Some(0)); // front
        assert_eq!(v.to_vec(), vec![1, 3, 4]);
        assert_eq!(v.remove(2), Some(4)); // back
        assert_eq!(v.to_vec(), vec![1, 3]);
        assert_eq!(v.remove(2), None); // out of range
        assert_eq!(v.to_vec(), vec![1, 3]);
        assert_eq!(v.pop(), Some(3));
        assert_eq!(v.pop(), Some(1));
        assert_eq!(v.pop(), None); // empty
        assert!(v.is_empty());
        // Capacity is untouched by erasure.
        assert_eq!(v.capacity(), 5);
        v.destroy(&a);
    }

    #[test]
    fn erase_range_covers_empty_partial_and_full() {
        let a = fresh("eraserange", 1 << 16);
        let mut v = ShmVector::<u32>::from_slice(&a, &[0, 1, 2, 3, 4, 5]);
        assert!(v.erase_range(2, 2)); // empty range: no-op
        assert_eq!(v.to_vec(), vec![0, 1, 2, 3, 4, 5]);
        assert!(v.erase_range(1, 3)); // partial
        assert_eq!(v.to_vec(), vec![0, 3, 4, 5]);
        assert!(v.erase_range(2, 4)); // tail
        assert_eq!(v.to_vec(), vec![0, 3]);
        assert!(!v.erase_range(1, 5)); // last > len
        assert!(!v.erase_range(2, 1)); // first > last
        assert_eq!(v.to_vec(), vec![0, 3]);
        assert!(v.erase_range(0, 2)); // full
        assert!(v.is_empty());
        assert!(v.erase_range(0, 0)); // empty range on empty vector
        v.destroy(&a);
    }

    #[test]
    fn from_slice_and_filled_allocate_exactly() {
        let a = fresh("ctors", 1 << 16);
        let mut v = ShmVector::<u64>::from_slice(&a, &[10, 20, 30]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.capacity(), 3); // exact, per the C++ init-list ctor
        assert_eq!(v.to_vec(), vec![10, 20, 30]);

        let mut e = ShmVector::<u64>::from_slice(&a, &[]);
        assert!(e.is_empty());
        assert_eq!(e.capacity(), 0);
        assert!(e.data_offset().is_null());

        let mut f = ShmVector::<u64>::filled(&a, 4, 7);
        assert_eq!(f.to_vec(), vec![7, 7, 7, 7]);
        assert_eq!(f.capacity(), 4);

        let mut z = ShmVector::<u64>::filled(&a, 0, 7);
        assert!(z.is_empty());
        assert!(z.data_offset().is_null());

        v.destroy(&a);
        e.destroy(&a);
        f.destroy(&a);
        z.destroy(&a);
    }

    #[test]
    fn out_of_memory_leaves_the_vector_intact() {
        // Divergence 4: C++ would empty the vector and leak the block here.
        let a = fresh("oom", 4096);
        let mut v = ShmVector::<u64>::new(&a);
        let mut n = 0u64;
        while v.push(&a, n) {
            n += 1;
            assert!(n < 1 << 20, "segment should have been exhausted by now");
        }
        let len = v.len();
        assert!(len > 0);
        // Every element survived the failed growth.
        assert_eq!(v.to_vec(), (0..len as u64).collect::<Vec<_>>());
        assert_eq!(v.last(), Some(len as u64 - 1));
        // Failure is sticky, not corrupting.
        assert!(!v.push(&a, 12345));
        assert_eq!(v.len(), len);
        // A doomed reserve also changes nothing.
        let cap = v.capacity();
        let off = v.data_offset().off;
        assert!(!v.reserve(&a, 1 << 20));
        assert_eq!(v.capacity(), cap);
        assert_eq!(v.data_offset().off, off);
        assert_eq!(v.len(), len);
        v.destroy(&a);
    }

    #[test]
    fn capacity_overflow_is_rejected_not_wrapped() {
        // C++ computes capacity * sizeof(T) in wrapping size_t arithmetic.
        let a = fresh("overflow", 1 << 16);
        let mut v = ShmVector::<u64>::new(&a);
        assert!(v.push(&a, 1));
        // usize::MAX * 8 overflows u64 → rejected before any allocation.
        assert!(!v.reserve(&a, usize::MAX));
        assert_eq!(v.capacity(), 1);
        assert_eq!(v.to_vec(), vec![1]);
        // No overflow, but far beyond the segment → clean failure.
        assert!(!v.reserve(&a, 1 << 40));
        assert!(!v.resize(&a, usize::MAX));
        assert_eq!(v.to_vec(), vec![1]);
        v.destroy(&a);
    }

    #[test]
    fn foreign_allocator_is_rejected() {
        // Divergence 12: C++ would resolve offsets against the wrong heap.
        let a = fresh("own", 1 << 16);
        let b = fresh("other", 1 << 16);
        let mut v = ShmVector::<u32>::from_slice(&a, &[1, 2, 3]);
        let off = v.data_offset().off;
        assert!(!v.push(&b, 4));
        assert!(!v.reserve(&b, 100));
        assert!(!v.insert(&b, 0, 0));
        assert!(!v.resize(&b, 10));
        assert!(!v.resize_with(&b, 10, 1));
        v.destroy(&b); // no-op: not this vector's allocator
        assert_eq!(v.data_offset().off, off);
        assert_eq!(v.to_vec(), vec![1, 2, 3]);
        v.destroy(&a);
    }

    #[test]
    fn equality_preserves_the_cpp_storage_quirk() {
        let a = fresh("eq", 1 << 16);
        let mut x = ShmVector::<u32>::from_slice(&a, &[1, 2, 3]);
        let mut y = ShmVector::<u32>::from_slice(&a, &[1, 2, 3]);
        let mut z = ShmVector::<u32>::from_slice(&a, &[1, 2, 4]);
        let mut short = ShmVector::<u32>::from_slice(&a, &[1, 2]);
        assert_eq!(x, y);
        assert_ne!(x, z); // same size, different elements
        assert_ne!(x, short); // different size
        assert_eq!(x, x); // reflexive

        // Divergence 15: cleared (has storage) != fresh (no storage), even
        // though both are empty — exactly what C++ operator== does.
        let mut cleared = ShmVector::<u32>::from_slice(&a, &[1]);
        cleared.clear();
        let fresh_v = ShmVector::<u32>::new(&a);
        assert!(cleared.is_empty() && fresh_v.is_empty());
        assert_ne!(cleared, fresh_v);
        // Two storage-less empties compare equal.
        let other_fresh = ShmVector::<u32>::new(&a);
        assert_eq!(fresh_v, other_fresh);

        x.destroy(&a);
        y.destroy(&a);
        z.destroy(&a);
        short.destroy(&a);
        cleared.destroy(&a);
    }

    #[test]
    fn header_lives_in_the_segment_and_resolves_from_any_view() {
        // The point of the offset-only design (pillar 3): the vector's own
        // 32-byte header can sit inside the segment — as a C++ ipc::vector
        // embedded in an shm struct does — and any independently resolved
        // view of it sees the same elements.
        let a = fresh("inseg", 1 << 16);
        let hdr_off = a
            .alloc_bytes(std::mem::size_of::<ShmVector<u32>>() as u64)
            .unwrap();
        let hdr_ptr = ctp_memory::resolve(hdr_off.to_shm(a.id()).cast::<ShmVector<u32>>()).unwrap();
        // SAFETY: a fresh 16-aligned block of exactly size_of::<ShmVector>()
        // bytes; nothing else refers to it yet.
        unsafe {
            hdr_ptr.write(ShmVector::new(&a));
            let v = &mut *hdr_ptr;
            for i in 0..50u32 {
                assert!(v.push(&a, i * 3));
            }
        }
        // A second resolution of the same offset — the "other process" view.
        let view = ctp_memory::resolve(hdr_off.to_shm(a.id()).cast::<ShmVector<u32>>()).unwrap();
        // SAFETY: same live header; the &mut above has expired.
        unsafe {
            let v2 = &*view;
            assert_eq!(v2.len(), 50);
            assert_eq!(v2.get(49), Some(147));
            assert_eq!(v2.to_vec(), (0..50u32).map(|i| i * 3).collect::<Vec<_>>());
            // Elements live in the segment, not in the header.
            assert!(!v2.data_offset().is_null());
            assert_ne!(v2.data_offset().off, hdr_off.off);
        }
        // SAFETY: still the same live header, uniquely borrowed here.
        unsafe {
            (*hdr_ptr).destroy(&a);
            a.free_bytes(hdr_off);
        }
    }

    #[test]
    fn many_vectors_grow_concurrently_in_one_segment() {
        // A ShmVector is no more thread-safe than std::vector (C++ parity):
        // what must hold under concurrency is that independent vectors
        // sharing one allocator never corrupt each other.
        use std::sync::Arc;
        let a = Arc::new(fresh("mt", 1 << 20));
        let handles: Vec<_> = (0..8u32)
            .map(|t| {
                let a = Arc::clone(&a);
                std::thread::spawn(move || {
                    let mut v = ShmVector::<u64>::new(&a);
                    for i in 0..200u64 {
                        assert!(v.push(&a, t as u64 * 1000 + i));
                    }
                    assert_eq!(v.len(), 200);
                    let want: Vec<u64> = (0..200).map(|i| t as u64 * 1000 + i).collect();
                    assert_eq!(v.to_vec(), want);
                    v.destroy(&a);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
