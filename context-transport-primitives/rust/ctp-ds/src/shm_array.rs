// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Fixed-capacity, inline-buffer containers — the Rust port of
//! `clio_ctp/data_structures/priv/array.h` (`ctp::ipc::array<T, N>`) and
//! `clio_ctp/data_structures/priv/array_vector.h` (`ctp::priv::array_vector<N>`).
//!
//! Both are allocator-free, heap-free, `#[repr(C)]`, `Copy`, and have **no
//! `Drop`** — they are byte-for-byte placeable in a shared-memory segment
//! (MEMORY_DESIGN.md pillars 2 and 3: no absolute pointers, everything is
//! inline storage plus a length). Each carries a `ShmSafe` impl gated on its
//! element type, so a `Array<T, N>` may live in a segment exactly when `T` may.
//!
//! # C++ → Rust name mapping
//!
//! | C++ | Rust | Notes |
//! |---|---|---|
//! | `ctp::ipc::array<T, N>` | [`Array<T, N>`] | `size_t N` → `const N: usize` |
//! | `array::array()` | [`Array::new`] / `Default` | see divergence 1 |
//! | `array::data()` | [`Array::as_ptr`] / [`Array::as_mut_ptr`] | raw `T*` over the whole `[T; N]` |
//! | (no C++ equivalent) | [`Array::buffer`] / [`Array::buffer_mut`] | safe `&[T; N]` form of `data()` |
//! | `array::size()` | [`Array::len`] | raw field read; see divergence 3 |
//! | `array::capacity()` | [`Array::capacity`] / [`Array::CAPACITY`] | `static constexpr` → assoc. const |
//! | `array::resize(n)` | [`Array::resize`] | unchecked, returns `()`; divergence 3 |
//! | (addition) | [`Array::try_resize`] | checked form, returns `bool` |
//! | `array::reserve(n)` | [`Array::reserve`] | no-op, as in C++ |
//! | `array::operator[]` | `Index` / `IndexMut` | divergence 2 |
//! | (no C++ equivalent) | [`Array::as_slice`] / [`Array::as_mut_slice`] | live `[0, size)` view; divergence 4 |
//! | `ctp::priv::array_vector<N>` | [`ArrayVector<N>`] | `alignas(8)` → `#[repr(C, align(8))]` |
//! | `array_vector::array_vector()` | [`ArrayVector::new`] / `Default` | zeroes the buffer; divergence 1 |
//! | `array_vector::data()` | [`ArrayVector::as_ptr`] / [`ArrayVector::as_mut_ptr`] | `char*` → `*const u8` / `*mut u8` |
//! | (no C++ equivalent) | [`ArrayVector::buffer`] / [`ArrayVector::buffer_mut`] | safe `&[u8; N]` form of `data()` |
//! | `array_vector::size()` | [`ArrayVector::len`] | raw field read; see divergence 3 |
//! | `array_vector::capacity()` | [`ArrayVector::capacity`] / [`ArrayVector::CAPACITY`] | |
//! | `array_vector::empty()` | [`ArrayVector::is_empty`] | |
//! | `array_vector::reserve(n)` | [`ArrayVector::reserve`] | `bool` = `n <= N`, no state change |
//! | `array_vector::resize(n)` | [`ArrayVector::resize`] | assigns then reports; divergence 3 |
//! | `array_vector::resize_no_init(n)` | [`ArrayVector::resize_no_init`] | identical body in C++; divergence 6 |
//! | `array_vector::clear()` | [`ArrayVector::clear`] | |
//! | `array_vector::push_back(c)` | [`ArrayVector::push_back`] | divergence 5 |
//! | `array_vector::operator[]` | `Index` / `IndexMut` | divergence 2 |
//! | `array_vector::begin()/end()` | [`ArrayVector::as_slice`] / [`ArrayVector::iter`] | divergence 4 |
//!
//! `char` maps to `u8`: `array_vector` is a byte buffer for `LocalSerialize`,
//! and C++ `char` is implementation-signed — `u8` fixes the sign and keeps the
//! one-byte layout the ABI depends on.
//!
//! # Semantic divergences from the C++
//!
//! 1. **Construction initializes the buffer.** The C++ ctors only set
//!    `size_ = 0`; `data_` is left default-initialized, i.e. *indeterminate*
//!    for the POD element types these containers are used with. Rust has no
//!    safe way to hold an uninitialized `[T; N]` in a `Copy` struct, so
//!    [`Array::new`] fills with `T::default()` (hence its extra
//!    `T: Copy + Default` bound, which the bare struct does not carry) and
//!    [`ArrayVector::new`] zero-fills. Reading element `i` where
//!    `i >= size` therefore yields a defined value here and an indeterminate
//!    one in C++. Cost is an `N`-element init the C++ skips; for in-segment
//!    instances the allocator already hands back zeroed memory, so callers
//!    that place these types in shared memory need not construct through
//!    `new()` at all.
//!
//! 2. **Indexing is bounds-checked against `N`, not unchecked.** Both C++
//!    `operator[]`s index `data_` with no check: `i < N` is well-defined
//!    (possibly reading indeterminate bytes past `size_`), `i >= N` is UB.
//!    The Rust `Index`/`IndexMut` reproduce the defined half exactly — `i < N`
//!    is allowed regardless of `size` — and turn the UB half into a panic.
//!    Note the bound is **capacity, not length**: `a.resize(2); a[7]` is legal
//!    for `N == 10`, matching C++.
//!
//! 3. **`resize` still assigns a too-large size.** `array_vector::resize(n)`
//!    sets `size_ = n` *and then* returns `n <= N`; `array::resize(n)` assigns
//!    with no report at all. Both are preserved verbatim, so [`len`](ArrayVector::len)
//!    can exceed [`capacity`](ArrayVector::capacity) after an ignored `false`.
//!    C++ makes every later `end()`/iteration UB in that state; see divergence 4
//!    for what Rust does instead. [`Array::try_resize`] is a non-C++ addition
//!    for callers that want the checked form `array` never offered.
//!
//! 4. **Slice views saturate at capacity instead of invoking UB.** C++
//!    `end()` is `data_ + size_`, which walks off the buffer once `size_ > N`.
//!    A Rust slice cannot express that, so [`as_slice`](ArrayVector::as_slice)
//!    and friends clamp to `min(size, N)` — never UB, per the
//!    "resolve to `None`, never UB" rule in MEMORY_DESIGN.md, which also guards
//!    against a corrupt or hostile `size` field read out of a segment.
//!    [`ArrayVector::try_as_slice`] / [`Array::try_as_slice`] expose the
//!    overflow explicitly as `None` for callers that must detect it.
//!
//! 5. **`push_back` is checked and reports.** C++ `push_back` is
//!    `data_[size_++] = c` — an out-of-bounds write and UB when full. Rust
//!    cannot reproduce that, so [`ArrayVector::push_back`] returns `bool`:
//!    on a full buffer it writes nothing, leaves `size` untouched, and returns
//!    `false`. This is the one place the container silently *drops* work the
//!    C++ would have (corruptly) performed — callers must check the result.
//!
//! 6. **`resize_no_init` is a true alias.** The two C++ bodies are character
//!    for character identical (neither initializes anything); the Rust keeps
//!    both names for call-site compatibility and forwards one to the other.
//!
//! 7. **`is_empty`/`iter` on `Array`** have no C++ counterpart (`array` exposes
//!    no `empty()`); they are added because Clippy requires `is_empty` beside
//!    `len`, and iteration is the idiomatic spelling of the pointer walk.
//!
//! 8. **`N == 0` is representable.** `T data_[0]` is ill-formed in standard
//!    C++ (a GCC extension); `[T; 0]` is ordinary Rust. A zero-capacity
//!    container behaves consistently here: every `push_back` fails, every
//!    slice is empty, every index panics.

use ctp_memory::ShmSafe;
use std::ops::{Index, IndexMut};

// ---------------------------------------------------------------------------
// array<T, N>  (ctp::ipc::array)
// ---------------------------------------------------------------------------

/// Fixed-size inline array — port of `ctp::ipc::array<T, N>`.
///
/// No allocator, no heap. A drop-in stand-in for a `vector<T, AllocT>` whose
/// maximum size is known at compile time, and the shape `LocalSerialize` /
/// `LocalDeserialize` consume (`data()` + `size()` + `resize()`).
///
/// Layout mirrors the C++ exactly: `data_` then `size_`, `#[repr(C)]`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Array<T, const N: usize> {
    /// C++ `T data_[N]` — the whole inline buffer, live or not.
    pub data: [T; N],
    /// C++ `size_t size_` — the live prefix length. May exceed `N`; see
    /// divergence 3 in the module docs.
    pub size: usize,
}

impl<T, const N: usize> Array<T, N> {
    /// C++ `static constexpr size_t capacity()`.
    pub const CAPACITY: usize = N;

    /// C++ `size_t size() const` — the raw field, *not* clamped to `N`.
    #[inline]
    pub const fn len(&self) -> usize {
        self.size
    }

    /// No C++ counterpart (`array` exposes no `empty()`); see divergence 7.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// C++ `static constexpr size_t capacity()`.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// C++ `void resize(size_t new_size)` — assigns unconditionally, no bounds
    /// check and no report, exactly as the C++ does (divergence 3).
    #[inline]
    pub fn resize(&mut self, new_size: usize) {
        self.size = new_size;
    }

    /// Checked `resize`: assigns and returns `true` only when it fits, leaving
    /// `size` untouched otherwise. Not in the C++ (divergence 3).
    #[inline]
    pub fn try_resize(&mut self, new_size: usize) -> bool {
        if new_size <= N {
            self.size = new_size;
            true
        } else {
            false
        }
    }

    /// C++ `void reserve(size_t)` — a no-op there and here; capacity is fixed.
    #[inline]
    pub fn reserve(&mut self, _n: usize) {}

    /// C++ `const T *data() const`.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// C++ `T *data()`.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// The whole `[T; N]` buffer — the safe form of C++ `data()`, ignoring
    /// `size` just as a raw `T*` would.
    #[inline]
    pub const fn buffer(&self) -> &[T; N] {
        &self.data
    }

    /// Mutable whole-buffer view; see [`buffer`](Self::buffer).
    #[inline]
    pub fn buffer_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }

    /// The live `[0, size)` prefix, **clamped to `N`** (divergence 4).
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.live_len()]
    }

    /// Mutable live prefix; see [`as_slice`](Self::as_slice).
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let n = self.live_len();
        &mut self.data[..n]
    }

    /// The live prefix, or `None` when `size > N` — the state C++ turns into
    /// UB at `end()` (divergence 4).
    #[inline]
    pub fn try_as_slice(&self) -> Option<&[T]> {
        if self.size <= N {
            Some(&self.data[..self.size])
        } else {
            None
        }
    }

    /// Iterate the live prefix (divergence 7).
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Mutably iterate the live prefix (divergence 7).
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// `size` clamped into the buffer — the guard behind divergence 4.
    #[inline]
    const fn live_len(&self) -> usize {
        if self.size < N {
            self.size
        } else {
            N
        }
    }
}

impl<T: Copy + Default, const N: usize> Array<T, N> {
    /// C++ `array() : size_(0) {}`, plus the buffer init Rust requires
    /// (divergence 1).
    #[inline]
    pub fn new() -> Self {
        Self {
            data: [T::default(); N],
            size: 0,
        }
    }
}

impl<T: Copy + Default, const N: usize> Default for Array<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// C++ `T &operator[](size_t i)` — bounds-checked against `N` (divergence 2).
impl<T, const N: usize> Index<usize> for Array<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

/// C++ `const T &operator[](size_t i) const` — see divergence 2.
impl<T, const N: usize> IndexMut<usize> for Array<T, N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Array<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Shows the live prefix, not the whole buffer — the trailing `N - size`
/// elements are not part of the value.
impl<T: std::fmt::Debug, const N: usize> std::fmt::Debug for Array<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("size", &self.size)
            .field("capacity", &N)
            .field("data", &self.as_slice())
            .finish()
    }
}

// SAFETY: `Array<T, N>` is `#[repr(C)]` inline storage plus a `usize` length —
// no references, no Drop, no process-local addresses — so it is segment-placeable
// exactly when its elements are. `T: ShmSafe` supplies `Copy + 'static` and the
// any-bit-pattern guarantee for `data`; `size` is a `usize`, itself `ShmSafe`,
// and every read of it is clamped (`live_len`) or checked (`try_as_slice`)
// before it reaches memory, so a hostile in-segment value cannot cause UB.
unsafe impl<T: ShmSafe, const N: usize> ShmSafe for Array<T, N> {}

// ---------------------------------------------------------------------------
// array_vector<N>  (ctp::priv::array_vector)
// ---------------------------------------------------------------------------

/// Fixed-capacity byte vector over an inline buffer — port of
/// `ctp::priv::array_vector<N>`.
///
/// The minimal surface `LocalSerialize`/`LocalDeserialize` need, with no
/// allocator and no heap. `alignas(8)` on both the C++ class and its `data_`
/// member becomes `#[repr(C, align(8))]`, which pins the same size/alignment
/// for every `N`.
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArrayVector<const N: usize> {
    /// C++ `alignas(8) char data_[N]` — `char` → `u8` (see module docs).
    pub data: [u8; N],
    /// C++ `size_t size_ = 0`. May exceed `N`; see divergence 3.
    pub size: usize,
}

impl<const N: usize> ArrayVector<N> {
    /// C++ `static constexpr size_t capacity()`.
    pub const CAPACITY: usize = N;

    /// C++ `array_vector() : size_(0) {}`, plus the zero-fill Rust requires
    /// (divergence 1).
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0u8; N],
            size: 0,
        }
    }

    /// C++ `const char *data() const`.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// C++ `char *data()`.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// The whole `[u8; N]` buffer — the safe form of C++ `data()`.
    #[inline]
    pub const fn buffer(&self) -> &[u8; N] {
        &self.data
    }

    /// Mutable whole-buffer view; see [`buffer`](Self::buffer).
    #[inline]
    pub fn buffer_mut(&mut self) -> &mut [u8; N] {
        &mut self.data
    }

    /// C++ `size_t size() const` — the raw field, *not* clamped to `N`.
    #[inline]
    pub const fn len(&self) -> usize {
        self.size
    }

    /// C++ `static constexpr size_t capacity()`.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// C++ `bool empty() const` — `size_ == 0`.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// C++ `bool reserve(size_t n)` — pure predicate `n <= N`; capacity is
    /// fixed, so nothing is allocated or changed.
    #[inline]
    pub fn reserve(&mut self, n: usize) -> bool {
        n <= N
    }

    /// C++ `bool resize(size_t n) { size_ = n; return n <= N; }` — note the
    /// assignment happens even when the answer is `false` (divergence 3).
    #[inline]
    pub fn resize(&mut self, n: usize) -> bool {
        self.size = n;
        n <= N
    }

    /// C++ `resize_no_init` — an exact alias of [`resize`](Self::resize) in the
    /// C++ too (divergence 6); neither initializes the new bytes.
    #[inline]
    pub fn resize_no_init(&mut self, n: usize) -> bool {
        self.resize(n)
    }

    /// C++ `void clear() { size_ = 0; }` — length only; bytes are left alone.
    #[inline]
    pub fn clear(&mut self) {
        self.size = 0;
    }

    /// C++ `void push_back(char c)`, made total: returns `false` and changes
    /// nothing when the buffer is full, where the C++ writes out of bounds
    /// (divergence 5).
    #[inline]
    pub fn push_back(&mut self, c: u8) -> bool {
        if self.size < N {
            self.data[self.size] = c;
            self.size += 1;
            true
        } else {
            false
        }
    }

    /// C++ `begin()` .. `end()`, **clamped to `N`** (divergence 4).
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.live_len()]
    }

    /// Mutable live view; see [`as_slice`](Self::as_slice).
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let n = self.live_len();
        &mut self.data[..n]
    }

    /// The live bytes, or `None` when `size > N` — the state C++ turns into UB
    /// at `end()` (divergence 4).
    #[inline]
    pub fn try_as_slice(&self) -> Option<&[u8]> {
        if self.size <= N {
            Some(&self.data[..self.size])
        } else {
            None
        }
    }

    /// C++ `begin()`/`end()` walk, spelled as an iterator.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, u8> {
        self.as_slice().iter()
    }

    /// Mutable form of [`iter`](Self::iter).
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, u8> {
        self.as_mut_slice().iter_mut()
    }

    /// `size` clamped into the buffer — the guard behind divergence 4.
    #[inline]
    const fn live_len(&self) -> usize {
        if self.size < N {
            self.size
        } else {
            N
        }
    }
}

impl<const N: usize> Default for ArrayVector<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// C++ `char &operator[](size_t i)` — bounds-checked against `N` (divergence 2).
impl<const N: usize> Index<usize> for ArrayVector<N> {
    type Output = u8;

    #[inline]
    fn index(&self, i: usize) -> &u8 {
        &self.data[i]
    }
}

/// C++ `const char &operator[](size_t i) const` — see divergence 2.
impl<const N: usize> IndexMut<usize> for ArrayVector<N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut u8 {
        &mut self.data[i]
    }
}

impl<'a, const N: usize> IntoIterator for &'a ArrayVector<N> {
    type Item = &'a u8;
    type IntoIter = std::slice::Iter<'a, u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Shows the live bytes, not the whole buffer.
impl<const N: usize> std::fmt::Debug for ArrayVector<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayVector")
            .field("size", &self.size)
            .field("capacity", &N)
            .field("data", &self.as_slice())
            .finish()
    }
}

// SAFETY: `ArrayVector<N>` is `#[repr(C, align(8))]` inline bytes plus a `usize`
// length — no references, no Drop, no process-local addresses. Every `u8` bit
// pattern is valid, and `size` is only ever clamped (`live_len`) or checked
// (`try_as_slice`) before indexing, so a corrupt in-segment length yields a
// short slice rather than UB.
unsafe impl<const N: usize> ShmSafe for ArrayVector<N> {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    // ---- Array: layout -----------------------------------------------------

    #[test]
    fn array_layout_matches_cpp() {
        // C++: `T data_[N]; size_t size_;` with no alignas — data at 0, size
        // after it at the natural size_t alignment.
        assert_eq!(size_of::<Array<u32, 3>>(), 24); // 12 data + 4 pad + 8 size
        assert_eq!(align_of::<Array<u32, 3>>(), align_of::<usize>());
        assert_eq!(size_of::<Array<u8, 8>>(), 16); // 8 data + 8 size
        assert_eq!(size_of::<Array<u64, 4>>(), 40); // 32 data + 8 size

        let a = Array::<u32, 3>::new();
        let base = &a as *const _ as usize;
        assert_eq!(&a.data as *const _ as usize - base, 0, "data_ is first");
        assert_eq!(
            &a.size as *const _ as usize - base,
            16,
            "size_ follows the padded buffer"
        );
    }

    #[test]
    fn array_is_copy_and_has_no_drop() {
        assert!(!std::mem::needs_drop::<Array<u64, 4>>());
        let a = Array::<u64, 4>::new();
        let b = a; // Copy, not a move-out
        assert_eq!(a.len(), b.len());
    }

    // ---- Array: basics -----------------------------------------------------

    #[test]
    fn array_new_is_empty_with_full_capacity() {
        let a = Array::<u32, 5>::new();
        assert_eq!(a.len(), 0);
        assert!(a.is_empty());
        assert_eq!(a.capacity(), 5);
        assert_eq!(Array::<u32, 5>::CAPACITY, 5);
        assert!(a.as_slice().is_empty());
        // Divergence 1: the buffer is defined, unlike the C++ ctor's.
        assert_eq!(a.buffer(), &[0u32; 5]);
    }

    #[test]
    fn array_resize_then_index_roundtrips() {
        let mut a = Array::<u32, 4>::new();
        a.resize(3);
        assert_eq!(a.len(), 3);
        a[0] = 10;
        a[1] = 20;
        a[2] = 30;
        assert_eq!(a.as_slice(), &[10, 20, 30]);
        assert_eq!(a.iter().copied().sum::<u32>(), 60);
        assert_eq!((&a).into_iter().count(), 3);
    }

    #[test]
    fn array_reserve_is_a_noop() {
        let mut a = Array::<u8, 4>::new();
        a.resize(2);
        a.reserve(1000); // C++ ignores the argument entirely
        assert_eq!(a.len(), 2);
        assert_eq!(a.capacity(), 4);
    }

    #[test]
    fn array_resize_shrink_keeps_bytes_but_hides_them() {
        let mut a = Array::<u8, 4>::new();
        a.resize(4);
        a.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
        a.resize(2);
        assert_eq!(a.as_slice(), &[1, 2]);
        // C++ resize only touches size_; the tail bytes survive and are still
        // addressable via operator[] below capacity (divergence 2).
        assert_eq!(a[3], 4);
        a.resize(4);
        assert_eq!(a.as_slice(), &[1, 2, 3, 4], "grow-back re-exposes the tail");
    }

    // ---- Array: capacity / overflow edges ----------------------------------

    #[test]
    fn array_resize_to_exact_capacity_is_ok() {
        let mut a = Array::<u8, 4>::new();
        a.resize(4);
        assert_eq!(a.len(), 4);
        assert_eq!(a.as_slice().len(), 4);
        assert_eq!(a.try_as_slice().map(<[u8]>::len), Some(4));
        assert!(a.try_resize(4), "boundary is inclusive");
    }

    #[test]
    fn array_resize_past_capacity_assigns_anyway() {
        // Divergence 3: the C++ `void resize` has no check and no report.
        let mut a = Array::<u8, 4>::new();
        a.resize(9);
        assert_eq!(a.len(), 9, "the raw field is honoured, as in C++");
        // Divergence 4: views saturate rather than walking off the buffer.
        assert_eq!(a.as_slice().len(), 4);
        assert_eq!(a.as_mut_slice().len(), 4);
        assert_eq!(a.try_as_slice(), None, "overflow is detectable");
    }

    #[test]
    fn array_try_resize_is_the_checked_form() {
        let mut a = Array::<u8, 4>::new();
        assert!(a.try_resize(4));
        assert_eq!(a.len(), 4);
        assert!(!a.try_resize(5), "5 > 4 fails");
        assert_eq!(a.len(), 4, "and leaves size untouched, unlike resize()");
        assert!(a.try_resize(0));
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn array_index_is_bounded_by_capacity_not_length() {
        // Divergence 2: C++ allows i < N regardless of size_; so do we.
        let mut a = Array::<u8, 8>::new();
        a.resize(2);
        a[7] = 42; // past size_, inside the buffer: fine in both languages
        assert_eq!(a[7], 42);
        assert_eq!(a.len(), 2, "indexing never moves size_");
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn array_index_past_capacity_panics_instead_of_ub() {
        let a = Array::<u8, 4>::new();
        let _ = a[4]; // C++: UB. Rust: panic (divergence 2).
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn array_index_mut_past_capacity_panics_instead_of_ub() {
        let mut a = Array::<u8, 4>::new();
        a.resize(9); // even an overflowed size_ cannot unlock OOB writes
        a[4] = 1;
    }

    #[test]
    fn array_zero_capacity_is_degenerate_but_defined() {
        // Divergence 8: `T data_[0]` is not standard C++ at all.
        let mut a = Array::<u8, 0>::new();
        assert_eq!(a.capacity(), 0);
        assert!(a.is_empty());
        assert!(a.as_slice().is_empty());
        assert!(a.try_resize(0));
        assert!(!a.try_resize(1));
        assert_eq!(size_of::<Array<u8, 0>>(), size_of::<usize>());
    }

    #[test]
    fn array_debug_shows_only_live_elements() {
        let mut a = Array::<u8, 8>::new();
        a.resize(2);
        a[0] = 1;
        a[1] = 2;
        let s = format!("{a:?}");
        assert!(s.contains("[1, 2]"), "got {s}");
        assert!(s.contains("capacity: 8"), "got {s}");
    }

    #[test]
    fn array_raw_pointers_address_the_whole_buffer() {
        let mut a = Array::<u32, 4>::new();
        a.resize(1);
        assert_eq!(a.as_ptr(), a.buffer().as_ptr());
        assert_eq!(a.as_mut_ptr().cast_const(), a.as_slice().as_ptr());
    }

    #[test]
    fn array_of_non_default_bound_still_constructs_via_literal() {
        // The bare struct carries no Copy/Default bound; only new() does.
        let a: Array<i64, 2> = Array {
            data: [-1, -2],
            size: 2,
        };
        assert_eq!(a.as_slice(), &[-1, -2]);
    }

    // ---- ArrayVector: layout -----------------------------------------------

    #[test]
    fn array_vector_layout_matches_cpp() {
        // C++: alignas(8) class { alignas(8) char data_[N]; size_t size_; }
        assert_eq!(align_of::<ArrayVector<64>>(), 8);
        assert_eq!(size_of::<ArrayVector<64>>(), 72); // 64 + 8
        assert_eq!(size_of::<ArrayVector<10>>(), 24); // 10 + 6 pad + 8
        assert_eq!(align_of::<ArrayVector<1>>(), 8, "alignas(8) holds for any N");
        assert_eq!(size_of::<ArrayVector<1>>(), 16);

        let v = ArrayVector::<10>::new();
        let base = &v as *const _ as usize;
        assert_eq!(&v.data as *const _ as usize - base, 0);
        assert_eq!(&v.size as *const _ as usize - base, 16);
    }

    #[test]
    fn array_vector_is_copy_and_has_no_drop() {
        assert!(!std::mem::needs_drop::<ArrayVector<32>>());
        let v = ArrayVector::<32>::new();
        let w = v;
        assert_eq!(v.len(), w.len());
    }

    // ---- ArrayVector: basics -----------------------------------------------

    #[test]
    fn array_vector_new_is_empty_and_zeroed() {
        let v = ArrayVector::<16>::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 16);
        assert_eq!(ArrayVector::<16>::CAPACITY, 16);
        assert!(v.as_slice().is_empty());
        assert_eq!(v.buffer(), &[0u8; 16]); // divergence 1
        assert_eq!(v, ArrayVector::<16>::default());
    }

    #[test]
    fn array_vector_push_back_fills_then_reports_full() {
        let mut v = ArrayVector::<4>::new();
        for (i, c) in b"abcd".iter().enumerate() {
            assert!(v.push_back(*c), "push {i} should fit");
            assert_eq!(v.len(), i + 1);
            assert!(!v.is_empty());
        }
        assert_eq!(v.as_slice(), b"abcd");

        // Divergence 5: C++ would write past the buffer here.
        assert!(!v.push_back(b'e'));
        assert_eq!(v.len(), 4, "size_ is not advanced on a failed push");
        assert_eq!(v.as_slice(), b"abcd", "and nothing is clobbered");
    }

    #[test]
    fn array_vector_push_back_on_zero_capacity_always_fails() {
        // Divergence 8.
        let mut v = ArrayVector::<0>::new();
        assert_eq!(v.capacity(), 0);
        assert!(!v.push_back(b'x'));
        assert!(v.is_empty());
        assert!(v.as_slice().is_empty());
        assert_eq!(size_of::<ArrayVector<0>>(), 8);
    }

    #[test]
    fn array_vector_clear_resets_length_only() {
        let mut v = ArrayVector::<4>::new();
        v.push_back(b'x');
        v.push_back(b'y');
        v.clear();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert!(v.as_slice().is_empty());
        // C++ clear() never touches data_; the bytes are still readable.
        assert_eq!(v[0], b'x');
        assert_eq!(v.buffer()[1], b'y');
        // ...and push_back overwrites from the front again.
        assert!(v.push_back(b'z'));
        assert_eq!(v.as_slice(), b"z");
    }

    #[test]
    fn array_vector_reserve_is_a_pure_predicate() {
        let mut v = ArrayVector::<8>::new();
        v.push_back(b'a');
        assert!(v.reserve(0));
        assert!(v.reserve(8), "boundary n == N is reservable");
        assert!(!v.reserve(9));
        assert!(!v.reserve(usize::MAX));
        assert_eq!(v.len(), 1, "reserve never changes state");
        assert_eq!(v.capacity(), 8);
    }

    #[test]
    fn array_vector_resize_reports_fit_and_boundaries() {
        let mut v = ArrayVector::<8>::new();
        assert!(v.resize(0));
        assert_eq!(v.len(), 0);
        assert!(v.resize(8), "n == N fits");
        assert_eq!(v.len(), 8);
        assert_eq!(v.as_slice().len(), 8);
    }

    #[test]
    fn array_vector_resize_past_capacity_assigns_and_returns_false() {
        // Divergence 3: C++ assigns size_ = n *before* returning n <= N.
        let mut v = ArrayVector::<8>::new();
        assert!(!v.resize(9));
        assert_eq!(v.len(), 9, "the bad size is retained, exactly as in C++");
        // Divergence 4: the C++ end() would be data_ + 9 (UB); we saturate.
        assert_eq!(v.as_slice().len(), 8);
        assert_eq!(v.as_mut_slice().len(), 8);
        assert_eq!(v.try_as_slice(), None);
        assert_eq!(v.iter().count(), 8);

        // A saturating view must not hide recovery: resizing back is fine.
        assert!(v.resize(2));
        assert_eq!(v.try_as_slice().map(<[u8]>::len), Some(2));
    }

    #[test]
    fn array_vector_resize_extreme_size_does_not_overflow_views() {
        let mut v = ArrayVector::<8>::new();
        assert!(!v.resize(usize::MAX));
        assert_eq!(v.len(), usize::MAX);
        assert_eq!(v.as_slice().len(), 8, "saturates, never UB");
        assert_eq!(v.try_as_slice(), None);
    }

    #[test]
    fn array_vector_resize_no_init_is_an_alias() {
        // Divergence 6: identical bodies in the C++.
        let mut a = ArrayVector::<8>::new();
        let mut b = ArrayVector::<8>::new();
        for n in [0usize, 1, 8, 9, 100] {
            assert_eq!(a.resize(n), b.resize_no_init(n), "n = {n}");
            assert_eq!(a.len(), b.len(), "n = {n}");
        }
        assert_eq!(a, b);
    }

    #[test]
    fn array_vector_resize_does_not_initialize_new_bytes() {
        // Both C++ resize forms only move size_; growing re-exposes old bytes.
        let mut v = ArrayVector::<4>::new();
        v.resize(4);
        v.as_mut_slice().copy_from_slice(b"junk");
        v.clear();
        assert!(v.resize_no_init(4));
        assert_eq!(v.as_slice(), b"junk", "no zero-fill on resize");
    }

    #[test]
    fn array_vector_index_is_bounded_by_capacity_not_length() {
        let mut v = ArrayVector::<8>::new();
        v.push_back(b'a');
        v[7] = b'z'; // past size_, inside data_ — legal in both (divergence 2)
        assert_eq!(v[7], b'z');
        assert_eq!(v.len(), 1);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn array_vector_index_past_capacity_panics_instead_of_ub() {
        let v = ArrayVector::<4>::new();
        let _ = v[4];
    }

    #[test]
    fn array_vector_iteration_matches_begin_end() {
        let mut v = ArrayVector::<8>::new();
        for c in b"hey" {
            v.push_back(*c);
        }
        assert_eq!(v.iter().copied().collect::<Vec<u8>>(), b"hey");
        assert_eq!((&v).into_iter().count(), 3);
        for b in v.iter_mut() {
            *b = b.to_ascii_uppercase();
        }
        assert_eq!(v.as_slice(), b"HEY");
    }

    #[test]
    fn array_vector_debug_shows_only_live_bytes() {
        let mut v = ArrayVector::<8>::new();
        v.push_back(1);
        v.push_back(2);
        let s = format!("{v:?}");
        assert!(s.contains("[1, 2]"), "got {s}");
        assert!(s.contains("capacity: 8"), "got {s}");
    }

    #[test]
    fn array_vector_serialize_shaped_roundtrip() {
        // The LocalSerialize usage pattern: reserve, resize, memcpy through
        // the slice, then read back via data()/size().
        let payload = b"iowarp-ctp";
        let mut v = ArrayVector::<32>::new();
        assert!(v.reserve(payload.len()));
        assert!(v.resize(payload.len()));
        v.as_mut_slice().copy_from_slice(payload);
        assert_eq!(v.len(), payload.len());
        assert_eq!(v.as_slice(), payload);
        assert_eq!(v.try_as_slice(), Some(&payload[..]));
    }

    // ---- ShmSafe / placement ----------------------------------------------

    #[test]
    fn shm_safe_is_implemented_for_both() {
        fn assert_shm_safe<T: ShmSafe>() {}
        assert_shm_safe::<Array<u64, 4>>();
        assert_shm_safe::<Array<u8, 0>>();
        assert_shm_safe::<ArrayVector<64>>();
        assert_shm_safe::<ArrayVector<0>>();
        // Nested placement: an Array of ShmSafe elements is itself ShmSafe.
        assert_shm_safe::<Array<Array<u32, 2>, 2>>();
    }

    #[test]
    fn types_survive_a_raw_byte_roundtrip() {
        // Stands in for a segment write/read by another process: the value is
        // reconstructible from its bytes alone (no process-local state).
        let mut v = ArrayVector::<16>::new();
        for c in b"shm" {
            v.push_back(*c);
        }
        let bytes: [u8; size_of::<ArrayVector<16>>()] =
            // SAFETY: `ArrayVector<16>` is `repr(C)`, `Copy`, `ShmSafe` and has
            // no padding-sensitive reads here — every byte of the source is
            // initialized (the ctor zero-fills), and the destination is a plain
            // byte array of exactly the same size.
            unsafe { std::mem::transmute(v) };
        // SAFETY: the bytes came from a valid `ArrayVector<16>` moments ago and
        // the sizes match exactly, so this reverses the transmute above.
        let back: ArrayVector<16> = unsafe { std::mem::transmute(bytes) };
        assert_eq!(back.as_slice(), b"shm");
        assert_eq!(back.len(), 3);
        assert_eq!(back, v);
    }

    #[test]
    fn array_in_a_zeroed_segment_reads_as_empty() {
        // Allocators hand back zeroed memory (MEMORY_DESIGN.md), so the
        // all-zero bit pattern must be a valid, empty container.
        let zeroed = [0u8; size_of::<ArrayVector<16>>()];
        // SAFETY: `ArrayVector<16>` is `ShmSafe`, which asserts validity for any
        // bit pattern; all-zero is what a freshly allocated segment block holds.
        let v: ArrayVector<16> = unsafe { std::mem::transmute(zeroed) };
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert!(v.as_slice().is_empty());
    }
}
