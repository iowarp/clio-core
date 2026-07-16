// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! `ctp::priv::basic_string` — an allocator-backed string with Short String
//! Optimization (SSO), ported from
//! `include/clio_ctp/data_structures/priv/string.h`.
//!
//! Small strings live inline in the object; longer ones *spill* into a block
//! obtained from a [`FreeListAllocator`] and are addressed by a **heap-relative
//! offset**, never a pointer (MEMORY_DESIGN.md pillar 3). There is no `Drop`:
//! a spilled string owns a segment block that must be released by an explicit
//! [`ShmString::destroy`], mirroring the task-teardown discipline in
//! MIGRATION.md.
//!
//! # Layout
//!
//! ```text
//! #[repr(C)] ShmString<SSO_SIZE = 32>      offset  size   C++ counterpart
//!   storage: [u8; SSO_SIZE]                     0  SSO_SIZE  union SsoStorage
//!       using_sso == 1: the characters, NUL-terminated
//!       using_sso == 0: HeapInfo, little/native-endian in the first 16 bytes
//!           storage[0..8]   = capacity (elements, INCLUDING the NUL slot)
//!           storage[8..16]  = off      (heap-relative allocator offset)
//!   size:      u64                             32     8   size_type size_
//!   using_sso: u8   (0 = spilled, 1 = inline)  40     1   bool using_sso_
//!   _pad:      [u8; 7]                         41     7   (tail padding)
//! total: 48 bytes, align 8 (C++ is 64: it also caches `alloc_` and `data_`)
//! ```
//!
//! The C++ `union SsoStorage { T buffer_[SSOSize]; HeapInfo heap_; }` is
//! emulated by the byte array + `to_ne_bytes`/`from_ne_bytes` accessors: the
//! bytes are identical to the C++ union on the same host, and no `unsafe`
//! union punning is needed. Because `HeapInfo` needs 16 bytes, `SSO_SIZE >= 16`
//! is enforced by a const assertion (C++ silently pads the union up to 16).
//!
//! # SSO threshold and spill rules (ported verbatim, asymmetry included)
//!
//! The C++ uses **two different thresholds**, and this port keeps both:
//!
//! | Operation | C++ predicate | Max inline `size` (SSO_SIZE = 32) |
//! |---|---|---|
//! | constructors, copy-ctor, assignment, `substr`, `shrink_to_fit` | `len < SSOSize - 1` | [`SSO_CTOR_MAX`] = 30 |
//! | `append`/`push`/`replace` | `new_size <= SSOSize - 1` | [`SSO_APPEND_MAX`] = 31 |
//!
//! So a 31-byte string *built by appending* stays inline, while the same
//! 31-byte string *constructed* from a slice spills — and `clone_with` of an
//! inline 31-byte string produces a spilled copy. This is faithful to the C++
//! (see `basic_string(AllocT*, const T*)` vs `AppendCStr`), not a Rust artifact.
//!
//! Spill rules:
//! - Inline → spilled (`transition_to_heap`): copy the inline bytes to a stack
//!   temporary, allocate `total_capacity`, copy back, clear `using_sso`.
//! - Spilled → larger block (`heap_reserve`): no-op if `new_cap <= capacity`,
//!   else grow to `max(capacity * 2, new_cap)`, copy `size` bytes, free the old
//!   block (2x growth amortizes appends, exactly as the C++ does).
//! - Spilled → inline: only `shrink_to_fit`, and only when `size < SSO_SIZE-1`.
//! - **Inline strings are always NUL-terminated; spilled ones are not** — the
//!   terminator is written on demand by [`ShmString::c_str`], as in the C++.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::priv`) | Rust | Notes |
//! |---|---|---|
//! | `basic_string<T, AllocT, SSOSize>` | [`ShmString<SSO_SIZE>`] | `T` fixed to `u8`; allocator passed per call |
//! | `string<AllocT, SSOSize>` (typedef) | [`ShmString`] | default `SSO_SIZE = 32` |
//! | `npos` | [`NPOS`] | `find` returns `Option<usize>`; `NPOS` kept for `count` args |
//! | `basic_string()` | [`ShmString::new`] / `Default` | empty, inline |
//! | `basic_string(alloc)` | [`ShmString::new`] | allocator is not stored |
//! | `basic_string(alloc, const T*)` | [`ShmString::from_bytes`] | slice, not NUL-scanned |
//! | `basic_string(alloc, std::basic_string)` | [`ShmString::from_str`] | |
//! | `basic_string(count, c, alloc)` | [`ShmString::from_repeat`] | |
//! | `basic_string(other, pos, count, alloc)` / `substr` | [`ShmString::substr`] | |
//! | copy ctor / `operator=(const&)` | [`ShmString::clone_with`] | needs the allocator; no `Clone` impl |
//! | move ctor / `operator=(&&)` | Rust move (`let b = a;`) | no fixup needed — see divergences |
//! | `~basic_string()` | [`ShmString::destroy`] | explicit, idempotent; no `Drop` |
//! | `at` | [`ShmString::at`] / [`ShmString::at_mut`] | `Option`, not `std::out_of_range` |
//! | `operator[]` | `as_bytes(alloc)[i]` | no `Index` impl (needs the allocator) |
//! | `front` / `back` | [`ShmString::front`] / [`ShmString::back`] | `Option`, not UB on empty |
//! | `data` / `size` | [`ShmString::as_bytes`] / [`ShmString::len`] | |
//! | `c_str` | [`ShmString::c_str`] | `&mut self` (it can reallocate) |
//! | `empty` / `length` / `capacity` | [`ShmString::is_empty`] / [`ShmString::len`] / [`ShmString::capacity`] | |
//! | `UsingSso` | [`ShmString::using_sso`] | |
//! | `reserve` / `shrink_to_fit` | [`ShmString::reserve`] / [`ShmString::shrink_to_fit`] | |
//! | `push_back` / `pop_back` / `clear` | [`ShmString::push`] / [`ShmString::pop`] / [`ShmString::clear`] | |
//! | `resize` / `resize_no_init` | [`ShmString::resize`] / [`ShmString::resize_no_init`] | |
//! | `append(...)` / `operator+=` | [`ShmString::append_bytes`] / [`ShmString::append_repeat`] | |
//! | `operator==` / `compare` | [`ShmString::eq_bytes`] / [`ShmString::compare`] | `Ordering`, not `int` |
//! | `starts_with` / `ends_with` | [`ShmString::starts_with`] / [`ShmString::ends_with`] | |
//! | `find(str/c, pos)` | [`ShmString::find`] / [`ShmString::find_byte`] | |
//! | `replace` / `erase` / `swap` | [`ShmString::replace`] / [`ShmString::erase`] / [`ShmString::swap`] | |
//! | `str()` / `operator std::string` | [`ShmString::str`] | UTF-8-lossy `String`; see divergences |
//! | `GetSsoState` / `SetSsoState` / `SsoState` | [`ShmString::sso_state`] / [`ShmString::from_sso_state`] / [`SsoState`] | the only `ShmSafe` type here |
//! | `NumberToStr` / `UintToStr` | [`ShmString::int_to_str`] / [`ShmString::uint_to_str`] / [`ShmString::float_to_str`] | |
//! | `FromNumber` | [`ShmString::from_i64`] / [`ShmString::from_u64`] / [`ShmString::from_f64`] | C++ version is uninstantiable — see divergences |
//! | `ctp::hash<basic_string>` | [`ShmString::fnv1a`] | |
//! | `iterator` / `const_iterator` / reverse | `as_bytes(alloc).iter()` etc. | not ported — see divergences |
//! | `save` / `load` (cereal) | — | not ported — see divergences |
//! | `FixupSsoPointer` / `InitFromSso` | — | unnecessary — see divergences |
//!
//! # Semantic divergences from the C++
//!
//! 1. **No cached `alloc_`/`data_`.** The C++ object stores a raw `AllocT*`
//!    and a cached `T* data_`. Both are process-local addresses, which pillar 3
//!    forbids inside a segment (they are exactly why the C++ needs
//!    `FixupSsoPointer`/`InitFromSso` after a bulk memcpy). This port stores
//!    only `(capacity, off)` and takes `&FreeListAllocator` as an explicit
//!    parameter on every method that may touch spilled bytes. Consequence: the
//!    Rust struct is 48 bytes vs the C++ 64, and the two are **not**
//!    byte-compatible — this type is not part of the frozen ABI.
//! 2. **Moves are free.** Because nothing is cached, a Rust move (a 48-byte
//!    memcpy) is correct as-is; the C++ move ctor/assignment must re-point
//!    `data_` and null out the source. `FixupSsoPointer`/`InitFromSso` have no
//!    Rust counterpart. Note a moved-from Rust value is statically unusable,
//!    so the C++ "leave `other` empty" step is unnecessary too.
//! 3. **No `Drop`, no `Copy`, no `Clone`.** `destroy(&alloc)` replaces the
//!    destructor. `ShmSafe` requires `Copy`, and a bitwise copy of a spilled
//!    string would alias (and later double-free) its block — so `ShmString`
//!    itself is deliberately **not** `ShmSafe`/`Copy`/`Clone` (the C++ type is
//!    likewise not trivially copyable: it has a deep copy ctor and a dtor).
//!    Only the pointer-free [`SsoState`] is `ShmSafe`. Copying goes through
//!    [`ShmString::clone_with`], which needs an allocator.
//! 4. **`destroy` is idempotent**; it leaves an empty inline string, so a
//!    double `destroy` cannot double-free. The C++ dtor has no such guard.
//! 5. **`using_sso_: bool` → `using_sso: u8`.** `bool` has invalid bit patterns
//!    (anything but 0/1 is UB to read), which is unsound for memory a peer
//!    process can write. The u8 flag treats "non-zero" as inline, like C++
//!    `bool` reads do in practice.
//! 6. **`size_type` (`size_t`) → `u64` in storage.** Frozen at 8 bytes like the
//!    rest of the pointer ABI rather than being platform-width; the public API
//!    speaks `usize`.
//! 7. **`T` is fixed to `u8`.** Only the `priv::string` (i.e. `char`)
//!    instantiation is used in-tree. `u8` also makes the `memcmp` ordering
//!    unambiguous — the C++ `compare` uses `memcmp` (unsigned) even where
//!    `char` is signed; this port matches that unsigned ordering.
//! 8. **Slices, not NUL-terminated C strings.** `from_bytes`/`append_bytes`
//!    take `&[u8]` and preserve embedded NULs; the C++ `const T*` ctors stop at
//!    the first NUL. `null` C-string arguments (which C++ treats as "" for
//!    `append`/`==`, as "match" for `starts_with`, and as `erase` for
//!    `replace`) are expressed as empty slices; the `starts_with(nullptr) ==
//!    true` quirk becomes the equivalent `starts_with(&[])` (also true).
//! 9. **Exceptions → panics / `Option`.** `CTP_THROW(std::out_of_range)` maps to
//!    a panic in `substr`/`replace`/`erase`; `at`/`front`/`back` return `Option`
//!    instead (the C++ `at` throws while `operator[]`/`front`/`back` are UB on
//!    empty — the Rust versions are total).
//! 10. **Allocation failure panics.** `FreeListAllocator::alloc_bytes` returns
//!     `None` on exhaustion; the infallible C++-shaped API panics with
//!     "allocator exhausted" (the C++ propagates the allocator's exception or
//!     aborts). A resolve failure (segment not mapped in this process) panics
//!     rather than reading a wild pointer.
//! 11. **`find` overflow is checked.** C++ computes `pos + needle.size() >
//!     size_`, which wraps for huge `pos`; the Rust port uses `checked_add` and
//!     yields `None`. Same answer for all in-range inputs.
//! 12. **`str()` is UTF-8-lossy.** C++ `str()`/`operator std::string` copies raw
//!     bytes (`std::string` is not UTF-8-checked). Rust `String` must be UTF-8,
//!     so invalid sequences become U+FFFD. Use [`ShmString::as_bytes`] for
//!     byte-exact access, or [`ShmString::as_str`] for a checked `&str`.
//! 13. **`sso_state()` returns `Option`.** The C++ `GetSsoState` happily exports
//!     the union's `HeapInfo` bytes as if they were characters when the string
//!     is spilled; this port returns `None` instead.
//! 14. **C++ `FromNumber` is uninstantiable** — it calls
//!     `basic_string(alloc, buf, len)`, a 3-argument ctor that does not exist in
//!     `string.h` (it only compiles because the template is never instantiated).
//!     The Rust `from_i64`/`from_u64`/`from_f64` implement the evident intent.
//!     `float_to_str` also differs where C++ is UB: `(unsigned long long)val`
//!     for an out-of-range/NaN `val` is UB in C++, while Rust's `as u64`
//!     saturates (NaN → 0). Digit generation and truncation are otherwise
//!     byte-identical, including `uint_to_str` keeping the *most significant*
//!     digits when the buffer is too small.
//! 15. **Not ported:** iterators (`begin`/`end`/reverse — use
//!     `as_bytes(alloc).iter()`; Rust slices give random access for free),
//!     cereal `save`/`load` (needs `ctp-serialize`, not a dependency of this
//!     crate), `std::basic_string` conversions/initializer lists (no C++ types),
//!     and `operator+`/`operator[]` overloads (an allocator argument is
//!     required, which operator traits cannot carry).
//! 16. **Cross-allocator operations.** C++ compares two `basic_string`s
//!     directly because each caches its own `data_`. Here, resolve the other
//!     side first: `a.compare(&alloc_a, b.as_bytes(&alloc_b))`. Likewise `swap`
//!     is a plain `mem::swap` and does **not** exchange allocators (the C++
//!     swaps `alloc_` too) — swapping strings from different allocators is the
//!     caller's problem in both languages.

use ctp_memory::{resolve, FreeListAllocator, OffsetPtr, ShmSafe};
use std::cmp::Ordering;

/// Default SSO buffer size in bytes (C++ `SSOSize = 32`).
pub const DEFAULT_SSO_SIZE: usize = 32;

/// C++ `basic_string::npos`. `find` returns `Option<usize>`; this constant
/// remains meaningful for `count` arguments ("to the end").
pub const NPOS: usize = usize::MAX;

/// FNV-1a offset basis / prime (C++ `ctp::hash<basic_string>`).
const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

/// POD snapshot of an inline string (C++ `basic_string::SsoState`).
///
/// Pointer-free and `Copy`, so it is the one type here that may be memcpy'd
/// into shared memory or serialized as a flat range.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsoState<const SSO_SIZE: usize = DEFAULT_SSO_SIZE> {
    /// The full inline buffer (C++ copies all `SSOSize` bytes, not just `size`).
    pub buffer: [u8; SSO_SIZE],
    /// Character count, excluding the NUL terminator.
    pub size: u64,
}

// SAFETY: a fixed-size byte array plus a u64 — no references, no Drop, valid
// for every bit pattern a cooperating process can write.
unsafe impl<const SSO_SIZE: usize> ShmSafe for SsoState<SSO_SIZE> {}

/// Allocator-backed string with Short String Optimization.
///
/// See the module docs for the layout, the two SSO thresholds, and the spill
/// rules. Every method that can touch spilled bytes takes the owning
/// [`FreeListAllocator`]; passing a *different* allocator than the one the
/// string spilled into is a contract violation (it will panic on resolve rather
/// than corrupt memory, since offsets are validated against the segment).
#[repr(C)]
#[derive(Debug)]
pub struct ShmString<const SSO_SIZE: usize = DEFAULT_SSO_SIZE> {
    /// Inline characters, or `HeapInfo{capacity, off}` in the first 16 bytes.
    storage: [u8; SSO_SIZE],
    /// Character count, excluding the NUL terminator.
    size: u64,
    /// 0 = spilled to the allocator, non-zero = inline.
    using_sso: u8,
    /// Explicit tail padding, so the layout is fully determined.
    _pad: [u8; 7],
}

impl<const SSO_SIZE: usize> Default for ShmString<SSO_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SSO_SIZE: usize> ShmString<SSO_SIZE> {
    /// Longest string the *constructors* keep inline (C++ `len < SSOSize - 1`).
    pub const SSO_CTOR_MAX: usize = SSO_SIZE - 2;
    /// Longest string *appends* keep inline (C++ `new_size <= SSOSize - 1`).
    pub const SSO_APPEND_MAX: usize = SSO_SIZE - 1;

    /// The union emulation needs 16 bytes for `HeapInfo`; C++ gets this for
    /// free by padding the union. Forced by every constructor.
    const LAYOUT_CHECK: () = assert!(
        SSO_SIZE >= 16,
        "ShmString: SSO_SIZE must be >= 16 (HeapInfo is capacity + off = 16 bytes)"
    );

    // ---------------------------------------------------------------- ctors

    /// C++ `basic_string()` / `basic_string(AllocT*)`: an empty inline string.
    /// The allocator is not stored (divergence 1), so both ctors collapse here.
    pub fn new() -> Self {
        let () = Self::LAYOUT_CHECK;
        Self {
            storage: [0u8; SSO_SIZE],
            size: 0,
            using_sso: 1,
            _pad: [0u8; 7],
        }
    }

    /// C++ `basic_string(AllocT* alloc, const T* s)` — spills when
    /// `len >= SSO_CTOR_MAX + 1` (i.e. `len < SSOSize - 1` stays inline).
    pub fn from_bytes(alloc: &FreeListAllocator, s: &[u8]) -> Self {
        let mut me = Self::new();
        let len = s.len();
        if len < SSO_SIZE - 1 {
            me.storage[..len].copy_from_slice(s);
            me.storage[len] = 0;
            me.size = len as u64;
        } else {
            me.heap_alloc(alloc, (len + 1) as u64);
            me.heap_bytes_mut(alloc, len).copy_from_slice(s);
            me.size = len as u64;
            me.using_sso = 0;
        }
        me
    }

    /// C++ `basic_string(AllocT*, const std::basic_string&)`.
    pub fn from_str(alloc: &FreeListAllocator, s: &str) -> Self {
        Self::from_bytes(alloc, s.as_bytes())
    }

    /// C++ `basic_string(size_type count, T c, AllocT* alloc)`.
    pub fn from_repeat(alloc: &FreeListAllocator, count: usize, c: u8) -> Self {
        let mut me = Self::new();
        if count < SSO_SIZE - 1 {
            me.storage[..count].fill(c);
            me.storage[count] = 0;
            me.size = count as u64;
        } else {
            me.heap_alloc(alloc, (count + 1) as u64);
            me.heap_bytes_mut(alloc, count).fill(c);
            me.size = count as u64;
            me.using_sso = 0;
        }
        me
    }

    /// C++ copy constructor / `operator=(const basic_string&)`.
    ///
    /// Note the C++ threshold quirk: an inline string of exactly
    /// `SSO_APPEND_MAX` bytes yields a **spilled** copy (divergence: none —
    /// this is faithful).
    pub fn clone_with(&self, alloc: &FreeListAllocator) -> Self {
        Self::from_bytes(alloc, self.as_bytes(alloc))
    }

    /// C++ `basic_string(const basic_string&, pos, count, AllocT*)` / `substr`.
    /// `count == NPOS` means "to the end".
    ///
    /// # Panics
    /// If `pos > len()` (C++ throws `std::out_of_range`).
    pub fn substr(&self, alloc: &FreeListAllocator, pos: usize, count: usize) -> Self {
        let n = self.len();
        assert!(pos <= n, "ShmString::substr: position out of range");
        let take = if count == NPOS { n - pos } else { count.min(n - pos) };
        Self::from_bytes(alloc, &self.as_bytes(alloc)[pos..pos + take])
    }

    /// C++ `~basic_string()`: release the spilled block, if any.
    ///
    /// Idempotent (divergence 4): the string is left empty and inline, so
    /// calling it twice — or on an inline string — is a no-op. **Must** be
    /// called before a spilled string goes out of scope; there is no `Drop`.
    pub fn destroy(&mut self, alloc: &FreeListAllocator) {
        if !self.using_sso() {
            self.heap_free(alloc);
        }
        self.storage = [0u8; SSO_SIZE];
        self.size = 0;
        self.using_sso = 1;
    }

    // ------------------------------------------------------------ observers

    /// C++ `UsingSso()`.
    #[inline]
    pub fn using_sso(&self) -> bool {
        self.using_sso != 0
    }

    /// C++ `size()` / `length()`.
    #[inline]
    pub fn len(&self) -> usize {
        self.size as usize
    }

    /// C++ `empty()`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// C++ `capacity()`: `SSOSize` while inline, else the allocated capacity
    /// (which includes the NUL slot).
    pub fn capacity(&self) -> usize {
        if self.using_sso() {
            SSO_SIZE
        } else {
            self.heap_info().0 as usize
        }
    }

    /// C++ `data()` + `size()`.
    pub fn as_bytes<'a>(&'a self, alloc: &'a FreeListAllocator) -> &'a [u8] {
        let n = self.len();
        if self.using_sso() {
            &self.storage[..n]
        } else {
            self.heap_bytes(alloc, n)
        }
    }

    /// Mutable view of the characters.
    pub fn as_bytes_mut<'a>(&'a mut self, alloc: &'a FreeListAllocator) -> &'a mut [u8] {
        let n = self.len();
        if self.using_sso() {
            &mut self.storage[..n]
        } else {
            self.heap_bytes_mut(alloc, n)
        }
    }

    /// UTF-8-checked view (no C++ counterpart; the checked half of `str()`).
    pub fn as_str<'a>(&'a self, alloc: &'a FreeListAllocator) -> Result<&'a str, std::str::Utf8Error> {
        std::str::from_utf8(self.as_bytes(alloc))
    }

    /// C++ `str()` / `operator std::string()`. Lossy: invalid UTF-8 becomes
    /// U+FFFD (divergence 12).
    pub fn str(&self, alloc: &FreeListAllocator) -> String {
        String::from_utf8_lossy(self.as_bytes(alloc)).into_owned()
    }

    /// C++ `at(pos)`, but total: `None` instead of `std::out_of_range`.
    pub fn at(&self, alloc: &FreeListAllocator, pos: usize) -> Option<u8> {
        if pos >= self.len() {
            return None;
        }
        Some(self.as_bytes(alloc)[pos])
    }

    /// Mutable C++ `at(pos)`.
    pub fn at_mut<'a>(&'a mut self, alloc: &'a FreeListAllocator, pos: usize) -> Option<&'a mut u8> {
        if pos >= self.len() {
            return None;
        }
        self.as_bytes_mut(alloc).get_mut(pos)
    }

    /// C++ `front()` (UB on empty there, `None` here).
    pub fn front(&self, alloc: &FreeListAllocator) -> Option<u8> {
        self.as_bytes(alloc).first().copied()
    }

    /// C++ `back()` (UB on empty there, `None` here).
    pub fn back(&self, alloc: &FreeListAllocator) -> Option<u8> {
        self.as_bytes(alloc).last().copied()
    }

    /// C++ `c_str()`: guarantee a NUL terminator and return the characters
    /// **plus** that terminator (`len() + 1` bytes).
    ///
    /// Takes `&mut self` (divergence): the C++ is `const` but `const_cast`s to
    /// grow a spilled buffer whose capacity leaves no room for the terminator.
    pub fn c_str<'a>(&'a mut self, alloc: &'a FreeListAllocator) -> &'a [u8] {
        let n = self.len();
        if !self.using_sso() && self.heap_info().0 <= self.size {
            self.heap_reserve(alloc, self.size + 1);
        }
        if self.using_sso() {
            self.storage[n] = 0;
            &self.storage[..n + 1]
        } else {
            let buf = self.heap_bytes_mut(alloc, n + 1);
            buf[n] = 0;
            buf
        }
    }

    /// C++ `GetSsoState()`, but `None` when spilled (divergence 13).
    pub fn sso_state(&self) -> Option<SsoState<SSO_SIZE>> {
        if !self.using_sso() {
            return None;
        }
        Some(SsoState {
            buffer: self.storage,
            size: self.size,
        })
    }

    /// C++ `SetSsoState()`: adopt a POD snapshot as an inline string.
    ///
    /// # Panics
    /// If `state.size` cannot be inline (`>= SSO_SIZE`), which would mean the
    /// snapshot did not come from an inline string.
    pub fn from_sso_state(state: &SsoState<SSO_SIZE>) -> Self {
        let () = Self::LAYOUT_CHECK;
        assert!(
            (state.size as usize) < SSO_SIZE,
            "ShmString::from_sso_state: size does not fit the inline buffer"
        );
        Self {
            storage: state.buffer,
            size: state.size,
            using_sso: 1,
            _pad: [0u8; 7],
        }
    }

    /// C++ `ctp::hash<basic_string>`: FNV-1a over the characters.
    pub fn fnv1a(&self, alloc: &FreeListAllocator) -> u64 {
        let mut h = FNV_OFFSET_BASIS;
        for &b in self.as_bytes(alloc) {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    }

    // ------------------------------------------------------------- capacity

    /// C++ `reserve(new_capacity)`.
    ///
    /// Faithful quirk: an inline string reports `capacity() == SSO_SIZE`, so
    /// `reserve(SSO_SIZE)` does nothing even though only `SSO_APPEND_MAX`
    /// characters actually fit inline.
    pub fn reserve(&mut self, alloc: &FreeListAllocator, new_capacity: usize) {
        if new_capacity <= self.capacity() {
            return;
        }
        if self.using_sso() {
            self.transition_to_heap(alloc, new_capacity as u64);
        } else {
            self.heap_reserve(alloc, new_capacity as u64);
        }
    }

    /// C++ `shrink_to_fit()`: return to inline storage when `len < SSO_SIZE-1`,
    /// otherwise reallocate the spilled block to exactly `len + 1`.
    pub fn shrink_to_fit(&mut self, alloc: &FreeListAllocator) {
        if self.using_sso() {
            return;
        }
        let n = self.len();
        if n < SSO_SIZE - 1 {
            let mut heap_copy = [0u8; SSO_SIZE];
            heap_copy[..n].copy_from_slice(self.heap_bytes(alloc, n));
            self.heap_free(alloc);
            self.storage[..n].copy_from_slice(&heap_copy[..n]);
            self.storage[n] = 0;
            self.using_sso = 1;
        } else if self.heap_info().0 > self.size + 1 {
            let old_off = self.heap_info().1;
            self.heap_alloc(alloc, self.size + 1);
            self.copy_between_blocks(alloc, old_off, n);
            // SAFETY: `old_off` came from `alloc_bytes` on this allocator, is
            // still live (we only overwrote our record of it), and is freed
            // exactly once here.
            unsafe { alloc.free_bytes(OffsetPtr::new(old_off)) };
        }
    }

    // ------------------------------------------------------------ modifiers

    /// C++ `push_back(c)` / `operator+=(T)`.
    pub fn push(&mut self, alloc: &FreeListAllocator, c: u8) {
        let n = self.len();
        if self.using_sso() {
            // C++: `size_ + 1 <= SSOSize - 1` (equivalent, clippy-preferred form).
            if n < SSO_SIZE - 1 {
                self.storage[n] = c;
                self.storage[n + 1] = 0;
                self.size += 1;
            } else {
                self.transition_to_heap(alloc, (n + 2) as u64);
                self.heap_bytes_mut(alloc, n + 1)[n] = c;
                self.size += 1;
            }
        } else {
            self.heap_reserve(alloc, (n + 2) as u64);
            self.heap_bytes_mut(alloc, n + 1)[n] = c;
            self.size += 1;
        }
    }

    /// C++ `pop_back()`: a no-op on an empty string.
    pub fn pop(&mut self) {
        if self.size > 0 {
            self.size -= 1;
            if self.using_sso() {
                self.storage[self.size as usize] = 0;
            }
        }
    }

    /// C++ `clear()`: release any spilled block and become empty + inline.
    pub fn clear(&mut self, alloc: &FreeListAllocator) {
        if !self.using_sso() {
            self.heap_free(alloc);
        }
        self.size = 0;
        self.using_sso = 1;
        self.storage[0] = 0;
    }

    /// C++ `resize(count)`: truncate, or extend with NUL bytes.
    pub fn resize(&mut self, alloc: &FreeListAllocator, count: usize) {
        let n = self.len();
        match count.cmp(&n) {
            Ordering::Less => {
                self.size = count as u64;
                if self.using_sso() {
                    self.storage[count] = 0;
                } else {
                    // capacity >= old size > count, so index `count` is in range.
                    self.heap_bytes_mut(alloc, count + 1)[count] = 0;
                }
            }
            Ordering::Greater => {
                self.reserve(alloc, count + 1);
                if self.using_sso() {
                    self.storage[n..count].fill(0);
                } else {
                    self.heap_bytes_mut(alloc, count)[n..count].fill(0);
                }
                self.size = count as u64;
            }
            Ordering::Equal => {}
        }
    }

    /// C++ `resize_no_init(count)`: grow without initializing.
    ///
    /// Safe in the Rust sense — segment bytes are always initialized memory —
    /// but the new region may expose stale data; overwrite it immediately.
    pub fn resize_no_init(&mut self, alloc: &FreeListAllocator, count: usize) {
        self.reserve(alloc, count + 1);
        self.size = count as u64;
    }

    /// C++ `AppendCStr` / `append(...)` / `operator+=`.
    ///
    /// Stays inline while `len() + s.len() <= SSO_APPEND_MAX` — one byte more
    /// than the constructors allow (see the module docs).
    pub fn append_bytes(&mut self, alloc: &FreeListAllocator, s: &[u8]) {
        let len = s.len();
        if len == 0 {
            return;
        }
        let n = self.len();
        if self.using_sso() && n + len <= SSO_SIZE - 1 {
            self.storage[n..n + len].copy_from_slice(s);
            self.size = (n + len) as u64;
            self.storage[n + len] = 0;
        } else {
            if self.using_sso() {
                self.transition_to_heap(alloc, (n + len + 1) as u64);
            } else {
                self.heap_reserve(alloc, (n + len + 1) as u64);
            }
            self.heap_bytes_mut(alloc, n + len)[n..n + len].copy_from_slice(s);
            self.size = (n + len) as u64;
        }
    }

    /// C++ `append(const basic_string&, pos, count)`.
    ///
    /// # Panics
    /// If `pos > s.len()` (C++ throws `std::out_of_range`).
    pub fn append_slice_from(&mut self, alloc: &FreeListAllocator, s: &[u8], pos: usize, count: usize) {
        assert!(pos <= s.len(), "ShmString::append: position out of range");
        let take = if count == NPOS {
            s.len() - pos
        } else {
            count.min(s.len() - pos)
        };
        self.append_bytes(alloc, &s[pos..pos + take]);
    }

    /// C++ `append(size_type count, T c)`.
    pub fn append_repeat(&mut self, alloc: &FreeListAllocator, count: usize, c: u8) {
        if count == 0 {
            return;
        }
        let n = self.len();
        let new_size = n + count;
        // C++: `new_size <= SSOSize - 1` (equivalent, clippy-preferred form).
        if self.using_sso() && new_size < SSO_SIZE {
            self.storage[n..new_size].fill(c);
            self.size = new_size as u64;
            self.storage[new_size] = 0;
        } else {
            if self.using_sso() {
                self.transition_to_heap(alloc, (new_size + 1) as u64);
            } else {
                self.heap_reserve(alloc, (new_size + 1) as u64);
            }
            self.heap_bytes_mut(alloc, new_size)[n..new_size].fill(c);
            self.size = new_size as u64;
        }
    }

    /// C++ `replace(pos, count, str)`.
    ///
    /// # Panics
    /// If `pos > len()` (C++ throws `std::out_of_range`).
    pub fn replace(&mut self, alloc: &FreeListAllocator, pos: usize, count: usize, s: &[u8]) {
        let n = self.len();
        assert!(pos <= n, "ShmString::replace: position out of range");
        let rep = count.min(n - pos);
        let tail = n - pos - rep;
        let new_size = n - rep + s.len();

        // C++: `new_size <= SSOSize - 1` (equivalent, clippy-preferred form).
        if self.using_sso() && new_size < SSO_SIZE {
            self.storage.copy_within(pos + rep..pos + rep + tail, pos + s.len());
            self.storage[pos..pos + s.len()].copy_from_slice(s);
            self.size = new_size as u64;
            self.storage[new_size] = 0;
        } else {
            if self.using_sso() {
                self.transition_to_heap(alloc, (new_size + 1) as u64);
            } else {
                self.heap_reserve(alloc, (new_size + 1) as u64);
            }
            let buf = self.heap_bytes_mut(alloc, new_size.max(n));
            buf.copy_within(pos + rep..pos + rep + tail, pos + s.len());
            buf[pos..pos + s.len()].copy_from_slice(s);
            self.size = new_size as u64;
        }
    }

    /// C++ `erase(pos, count)`. `count == NPOS` erases to the end.
    ///
    /// Faithful quirk: like the C++, no NUL terminator is written in the
    /// spilled case (only [`ShmString::c_str`] guarantees one).
    ///
    /// # Panics
    /// If `pos > len()` (C++ throws `std::out_of_range`).
    pub fn erase(&mut self, alloc: &FreeListAllocator, pos: usize, count: usize) {
        let n = self.len();
        assert!(pos <= n, "ShmString::erase: position out of range");
        let erase_count = if count == NPOS { n - pos } else { count.min(n - pos) };
        if self.using_sso() {
            self.storage.copy_within(pos + erase_count..n, pos);
            self.size -= erase_count as u64;
            self.storage[self.size as usize] = 0;
        } else {
            self.heap_bytes_mut(alloc, n).copy_within(pos + erase_count..n, pos);
            self.size -= erase_count as u64;
        }
    }

    /// C++ `swap(other)`.
    ///
    /// A plain `mem::swap`: no `data_` fixups are needed (divergence 2), and no
    /// allocator is exchanged (divergence 16).
    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
    }

    // ----------------------------------------------------------- comparison

    /// C++ `operator==(const basic_string&)` / `operator==(const T*)`.
    pub fn eq_bytes(&self, alloc: &FreeListAllocator, other: &[u8]) -> bool {
        self.as_bytes(alloc) == other
    }

    /// C++ `compare(...)`: lexicographic, unsigned-byte ordering.
    /// Returns an `Ordering` rather than a `memcmp`-style `int`.
    pub fn compare(&self, alloc: &FreeListAllocator, other: &[u8]) -> Ordering {
        self.as_bytes(alloc).cmp(other)
    }

    /// C++ `starts_with(...)`. An empty prefix matches (as `nullptr` does there).
    pub fn starts_with(&self, alloc: &FreeListAllocator, prefix: &[u8]) -> bool {
        self.as_bytes(alloc).starts_with(prefix)
    }

    /// C++ `ends_with(...)`. An empty suffix matches (as `nullptr` does there).
    pub fn ends_with(&self, alloc: &FreeListAllocator, suffix: &[u8]) -> bool {
        self.as_bytes(alloc).ends_with(suffix)
    }

    /// C++ `find(const basic_string&, pos)`: first match at or after `pos`,
    /// `None` for the C++ `npos`. An empty needle matches at `pos` when
    /// `pos <= len()`.
    pub fn find(&self, alloc: &FreeListAllocator, needle: &[u8], pos: usize) -> Option<usize> {
        let n = self.len();
        if needle.is_empty() {
            return if pos <= n { Some(pos) } else { None };
        }
        // C++ computes `pos + needle.size() > size_`, which wraps; checked here.
        if pos.checked_add(needle.len())? > n {
            return None;
        }
        let data = self.as_bytes(alloc);
        data[pos..]
            .windows(needle.len())
            .position(|w| w == needle)
            .map(|i| i + pos)
    }

    /// C++ `find(T c, pos)`.
    pub fn find_byte(&self, alloc: &FreeListAllocator, c: u8, pos: usize) -> Option<usize> {
        if pos >= self.len() {
            return None;
        }
        self.as_bytes(alloc)[pos..]
            .iter()
            .position(|&b| b == c)
            .map(|i| i + pos)
    }

    // ------------------------------------------------- number → string (C++
    // NumberToStr/UintToStr/FromNumber)

    /// C++ `UintToStr(buf, buf_size, val)`: decimal digits, NUL-terminated when
    /// there is room. Returns the digit count written (excluding the NUL).
    ///
    /// Faithful quirk: when the buffer is too small the **most significant**
    /// digits are kept (`buf_size - 1` of them), not the least significant.
    pub fn uint_to_str(buf: &mut [u8], mut val: u64) -> usize {
        let buf_size = buf.len();
        if buf_size == 0 {
            return 0;
        }
        if val == 0 {
            buf[0] = b'0';
            if buf_size > 1 {
                buf[1] = 0;
            }
            return 1;
        }
        let mut tmp = [0u8; 32];
        let mut nd = 0usize;
        while val > 0 && nd < 32 {
            tmp[nd] = b'0' + (val % 10) as u8;
            nd += 1;
            val /= 10;
        }
        let len = if nd < buf_size { nd } else { buf_size - 1 };
        for (i, b) in buf.iter_mut().take(len).enumerate() {
            *b = tmp[nd - 1 - i];
        }
        if len < buf_size {
            buf[len] = 0;
        }
        len
    }

    /// C++ `NumberToStr(buf, buf_size, IntT val)` for signed integers.
    /// `i64::MIN` is negated through the unsigned type, as the C++ does.
    pub fn int_to_str(buf: &mut [u8], val: i64) -> usize {
        if buf.is_empty() {
            return 0;
        }
        if val < 0 {
            buf[0] = b'-';
            // C++: `UIntT uval = (UIntT)0 - (UIntT)val` — wrapping, so INT_MIN
            // is well defined.
            let uval = (val as u64).wrapping_neg();
            return 1 + Self::uint_to_str(&mut buf[1..], uval);
        }
        Self::uint_to_str(buf, val as u64)
    }

    /// C++ `NumberToStr(buf, buf_size, FloatT val, precision)`.
    ///
    /// Divergence 14: C++ `(unsigned long long)val` is UB when `val` is NaN or
    /// out of range; Rust's `as u64` saturates (NaN → 0).
    pub fn float_to_str(buf: &mut [u8], mut val: f64, precision: i32) -> usize {
        let buf_size = buf.len();
        if buf_size == 0 {
            return 0;
        }
        let mut pos = 0usize;
        if val < 0.0 {
            buf[pos] = b'-';
            pos += 1;
            val = -val;
        }
        let int_part = val as u64;
        pos += Self::uint_to_str(&mut buf[pos..], int_part);
        if precision > 0 && pos + 1 < buf_size {
            buf[pos] = b'.';
            pos += 1;
            let mut frac = val - int_part as f64;
            let mut d = 0;
            while d < precision && pos < buf_size - 1 {
                frac *= 10.0;
                let digit = frac as i32;
                buf[pos] = b'0' + digit as u8;
                pos += 1;
                frac -= digit as f64;
                d += 1;
            }
        }
        if pos < buf_size {
            buf[pos] = 0;
        }
        pos
    }

    /// C++ `FromNumber(alloc, IntT val)` (signed).
    pub fn from_i64(alloc: &FreeListAllocator, val: i64) -> Self {
        let mut buf = [0u8; 32];
        let len = Self::int_to_str(&mut buf, val);
        Self::from_bytes(alloc, &buf[..len])
    }

    /// C++ `FromNumber(alloc, IntT val)` (unsigned).
    pub fn from_u64(alloc: &FreeListAllocator, val: u64) -> Self {
        let mut buf = [0u8; 32];
        let len = Self::uint_to_str(&mut buf, val);
        Self::from_bytes(alloc, &buf[..len])
    }

    /// C++ `FromNumber(alloc, FloatT val, precision = 6)`.
    pub fn from_f64(alloc: &FreeListAllocator, val: f64, precision: i32) -> Self {
        let mut buf = [0u8; 64];
        let len = Self::float_to_str(&mut buf, val, precision);
        Self::from_bytes(alloc, &buf[..len])
    }

    // ------------------------------------------------------ heap internals
    // (C++ HeapInfo / HeapAlloc / HeapFree / HeapReserve / TransitionToHeap)

    /// Decode the `HeapInfo` union member: `(capacity, off)`.
    fn heap_info(&self) -> (u64, u64) {
        let mut c = [0u8; 8];
        let mut o = [0u8; 8];
        c.copy_from_slice(&self.storage[0..8]);
        o.copy_from_slice(&self.storage[8..16]);
        (u64::from_ne_bytes(c), u64::from_ne_bytes(o))
    }

    /// Encode the `HeapInfo` union member, clobbering the inline buffer —
    /// exactly as writing `storage_.heap_` does in C++.
    fn set_heap_info(&mut self, capacity: u64, off: u64) {
        self.storage[0..8].copy_from_slice(&capacity.to_ne_bytes());
        self.storage[8..16].copy_from_slice(&off.to_ne_bytes());
    }

    /// C++ `HeapAlloc(capacity)`: allocate and record `(capacity, off)`.
    /// Does not touch `using_sso`/`size`, matching the C++.
    fn heap_alloc(&mut self, alloc: &FreeListAllocator, capacity: u64) {
        let p = alloc
            .alloc_bytes(capacity)
            .expect("ShmString: allocator exhausted");
        self.set_heap_info(capacity, p.off);
    }

    /// C++ `HeapFree()`. Caller must ensure the string is spilled.
    fn heap_free(&mut self, alloc: &FreeListAllocator) {
        let (_, off) = self.heap_info();
        // SAFETY: `off` was produced by `alloc_bytes` on this allocator and is
        // recorded exactly once; callers (`clear`/`destroy`/`shrink_to_fit`)
        // drop the record immediately after, so it cannot be freed twice.
        unsafe { alloc.free_bytes(OffsetPtr::new(off)) };
    }

    /// Resolve the spilled block to a pointer valid in THIS process.
    fn heap_ptr(alloc: &FreeListAllocator, off: u64) -> *mut u8 {
        resolve(OffsetPtr::<u8>::new(off).to_shm(alloc.id()))
            .expect("ShmString: spilled block does not resolve (wrong allocator?)")
    }

    /// Borrow `len` bytes of the spilled block.
    fn heap_bytes<'a>(&'a self, alloc: &'a FreeListAllocator, len: usize) -> &'a [u8] {
        let (cap, off) = self.heap_info();
        assert!(len as u64 <= cap, "ShmString: read beyond spilled capacity");
        let p = Self::heap_ptr(alloc, off);
        // SAFETY: `p` resolves inside the registered segment and the allocator
        // guarantees the block holds at least `cap >= len` bytes (asserted).
        // The `'a` lifetime ties the slice to both `self` (sole owner of the
        // block) and `alloc` (keeps the mapping registered), so it cannot
        // outlive either.
        unsafe { std::slice::from_raw_parts(p, len) }
    }

    /// Mutably borrow `len` bytes of the spilled block.
    fn heap_bytes_mut<'a>(&'a mut self, alloc: &'a FreeListAllocator, len: usize) -> &'a mut [u8] {
        let (cap, off) = self.heap_info();
        assert!(len as u64 <= cap, "ShmString: write beyond spilled capacity");
        let p = Self::heap_ptr(alloc, off);
        // SAFETY: as `heap_bytes`, plus uniqueness — `&mut self` is the sole
        // owner of this block, so no other live slice can alias it.
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    /// Copy `len` bytes from a previous block into the current one.
    fn copy_between_blocks(&self, alloc: &FreeListAllocator, old_off: u64, len: usize) {
        let (_, new_off) = self.heap_info();
        let src = Self::heap_ptr(alloc, old_off);
        let dst = Self::heap_ptr(alloc, new_off);
        // SAFETY: `old_off` and `new_off` are two *distinct* live blocks of the
        // same allocator (the new one was just allocated, so first-fit cannot
        // have returned the still-owned old one), each at least `len` bytes.
        // Distinctness makes `copy_nonoverlapping` sound.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
    }

    /// C++ `HeapReserve(new_cap)`: grow to `max(capacity * 2, new_cap)`.
    fn heap_reserve(&mut self, alloc: &FreeListAllocator, new_cap: u64) {
        let (cap, off) = self.heap_info();
        if new_cap <= cap {
            return;
        }
        let grow_cap = (cap * 2).max(new_cap);
        let n = self.len();
        self.heap_alloc(alloc, grow_cap);
        self.copy_between_blocks(alloc, off, n);
        // SAFETY: `off` is the previous block: live, allocated by this
        // allocator, no longer recorded anywhere, and freed exactly once.
        unsafe { alloc.free_bytes(OffsetPtr::new(off)) };
    }

    /// C++ `TransitionToHeap(total_capacity)`: spill the inline bytes.
    fn transition_to_heap(&mut self, alloc: &FreeListAllocator, total_capacity: u64) {
        let n = self.len();
        // The C++ stages through a stack temporary for the same reason: the
        // HeapInfo write below clobbers the inline buffer we are copying from.
        let mut sso_copy = [0u8; SSO_SIZE];
        sso_copy[..n].copy_from_slice(&self.storage[..n]);
        self.heap_alloc(alloc, total_capacity);
        self.heap_bytes_mut(alloc, n).copy_from_slice(&sso_copy[..n]);
        self.using_sso = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ctp_memory::{AllocatorId, SharedMemBackend};
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

    /// A private segment per test. Unique `AllocatorId` per call: the registry
    /// is process-global and cargo runs tests in parallel, so a shared id would
    /// let one test's `Drop` unregister another's live mapping (same reasoning
    /// as ctp-memory's allocator tests).
    fn fresh(tag: &str) -> FreeListAllocator {
        static NEXT_MINOR: AtomicU32 = AtomicU32::new(1);
        let minor = NEXT_MINOR.fetch_add(1, AtomicOrdering::Relaxed);
        let name = format!("ctp_rs_shmstr_{}_{}_{}", tag, std::process::id(), minor);
        let backend = SharedMemBackend::create(&name, 1 << 20).unwrap();
        backend.destroy(); // unlink the name eagerly (POSIX); the mapping stays
        FreeListAllocator::create(backend, AllocatorId::new(7561, minor))
    }

    // -------------------------------------------------------------- layout

    #[test]
    fn layout_is_documented_shape() {
        assert_eq!(std::mem::size_of::<ShmString>(), 48);
        assert_eq!(std::mem::align_of::<ShmString>(), 8);
        // Pointer-free: the whole struct is bytes + integers, so a move is a
        // plain memcpy (divergence 2) and no Drop glue exists (divergence 3).
        assert!(!std::mem::needs_drop::<ShmString>());
        assert!(!std::mem::needs_drop::<SsoState>());
        assert_eq!(std::mem::size_of::<SsoState>(), 40);
        // The two SSO thresholds from the C++.
        assert_eq!(ShmString::<32>::SSO_CTOR_MAX, 30);
        assert_eq!(ShmString::<32>::SSO_APPEND_MAX, 31);
    }

    #[test]
    fn empty_string_needs_no_allocator() {
        let s = ShmString::<32>::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(s.using_sso());
        assert_eq!(s.capacity(), 32);
        assert_eq!(ShmString::<32>::default().len(), 0);
    }

    // ------------------------------------------------------- SSO boundary

    #[test]
    fn ctor_sso_boundary_is_len_lt_sso_size_minus_one() {
        let a = fresh("ctor_bound");
        // 30 bytes (SSO_CTOR_MAX) — inline.
        let s30 = ShmString::<32>::from_bytes(&a, &[b'a'; 30]);
        assert!(s30.using_sso());
        assert_eq!(s30.len(), 30);
        assert_eq!(s30.capacity(), 32);
        // 31 bytes — the C++ ctor predicate `len < SSOSize - 1` fails: spills.
        let mut s31 = ShmString::<32>::from_bytes(&a, &[b'a'; 31]);
        assert!(!s31.using_sso());
        assert_eq!(s31.len(), 31);
        assert_eq!(s31.capacity(), 32); // exactly len + 1
        assert_eq!(s31.as_bytes(&a), &[b'a'; 31]);
        s31.destroy(&a);
    }

    #[test]
    fn append_sso_boundary_is_one_byte_more_than_ctor() {
        let a = fresh("append_bound");
        // Faithful C++ asymmetry: appends fit 31 inline, ctors only 30.
        let mut s = ShmString::<32>::new();
        for _ in 0..31 {
            s.push(&a, b'x');
        }
        assert!(s.using_sso(), "31 chars must still be inline after appends");
        assert_eq!(s.len(), 31);
        // The NUL terminator lives at storage[31] — the last inline byte.
        assert_eq!(s.storage[31], 0);
        // The 32nd character cannot fit: spill.
        s.push(&a, b'y');
        assert!(!s.using_sso());
        assert_eq!(s.len(), 32);
        assert_eq!(&s.as_bytes(&a)[..31], &[b'x'; 31]);
        assert_eq!(s.as_bytes(&a)[31], b'y');
        s.destroy(&a);
    }

    #[test]
    fn empty_and_one_byte_edges() {
        let a = fresh("edges");
        let s = ShmString::<32>::from_bytes(&a, &[]);
        assert!(s.using_sso() && s.is_empty());
        assert_eq!(s.as_bytes(&a), b"");
        let s1 = ShmString::<32>::from_bytes(&a, b"z");
        assert_eq!(s1.len(), 1);
        assert_eq!(s1.front(&a), Some(b'z'));
        assert_eq!(s1.back(&a), Some(b'z'));
        // front/back are total here (C++ is UB on empty).
        assert_eq!(s.front(&a), None);
        assert_eq!(s.back(&a), None);
        // pop on empty is a no-op, not an underflow.
        let mut e = ShmString::<32>::new();
        e.pop();
        assert_eq!(e.len(), 0);
    }

    // -------------------------------------------------------------- spill

    #[test]
    fn spill_preserves_content_and_grows_geometrically() {
        let a = fresh("spill");
        let mut s = ShmString::<32>::from_bytes(&a, b"hello");
        assert!(s.using_sso());
        s.append_bytes(&a, &[b'q'; 100]);
        assert!(!s.using_sso());
        assert_eq!(s.len(), 105);
        assert_eq!(&s.as_bytes(&a)[..5], b"hello");
        assert_eq!(&s.as_bytes(&a)[5..], &[b'q'; 100]);
        // 2x growth: capacity was 106, appending 1 more byte doubles it.
        assert_eq!(s.capacity(), 106);
        s.push(&a, b'!');
        assert_eq!(s.capacity(), 212);
        assert_eq!(s.len(), 106);
        assert_eq!(s.back(&a), Some(b'!'));
        s.destroy(&a);
    }

    #[test]
    fn spill_via_append_repeat_and_reserve() {
        let a = fresh("spill2");
        let mut s = ShmString::<32>::from_bytes(&a, b"ab");
        s.append_repeat(&a, 50, b'z');
        assert!(!s.using_sso());
        assert_eq!(s.len(), 52);
        assert_eq!(&s.as_bytes(&a)[..2], b"ab");
        assert!(s.as_bytes(&a)[2..].iter().all(|&b| b == b'z'));
        s.destroy(&a);

        // reserve() spills an inline string; C++ quirk: reserve(<= SSO_SIZE)
        // is a no-op because capacity() reports SSO_SIZE while inline.
        let mut r = ShmString::<32>::from_bytes(&a, b"hi");
        r.reserve(&a, 32);
        assert!(r.using_sso());
        r.reserve(&a, 33);
        assert!(!r.using_sso());
        assert_eq!(r.capacity(), 33);
        assert_eq!(r.as_bytes(&a), b"hi");
        r.destroy(&a);
    }

    #[test]
    fn shrink_to_fit_returns_to_sso_and_trims() {
        let a = fresh("shrink");
        let mut s = ShmString::<32>::from_bytes(&a, &[b'k'; 100]);
        assert!(!s.using_sso());
        // Trim to exact size while still too long to inline.
        s.resize(&a, 40);
        s.shrink_to_fit(&a);
        assert!(!s.using_sso());
        assert_eq!(s.capacity(), 41);
        assert_eq!(s.len(), 40);
        // Now short enough (< SSO_SIZE - 1): comes back inline, freeing the block.
        s.resize(&a, 10);
        s.shrink_to_fit(&a);
        assert!(s.using_sso());
        assert_eq!(s.as_bytes(&a), &[b'k'; 10]);
        s.destroy(&a);
    }

    // -------------------------------------------------------------- clear

    #[test]
    fn clear_releases_block_and_returns_to_sso() {
        let a = fresh("clear");
        let mut s = ShmString::<32>::from_bytes(&a, &[b'c'; 200]);
        assert!(!s.using_sso());
        let (_, off) = s.heap_info();
        s.clear(&a);
        assert!(s.using_sso());
        assert!(s.is_empty());
        assert_eq!(s.as_bytes(&a), b"");
        // The freed block is reusable: first-fit hands the same offset back.
        let reused = a.alloc_bytes(200).unwrap();
        assert_eq!(reused.off, off, "clear() must return the block to the allocator");
        // Clearing an inline string is a harmless no-op.
        let mut inline = ShmString::<32>::from_bytes(&a, b"short");
        inline.clear(&a);
        assert!(inline.is_empty() && inline.using_sso());
    }

    #[test]
    fn clear_then_reuse_is_usable() {
        let a = fresh("clear_reuse");
        let mut s = ShmString::<32>::from_bytes(&a, &[b'c'; 64]);
        s.clear(&a);
        s.append_bytes(&a, b"fresh start");
        assert!(s.using_sso());
        assert_eq!(s.str(&a), "fresh start");
        s.destroy(&a);
    }

    // ---------------------------------------------------- str() round-trip

    #[test]
    fn str_round_trip_inline_and_spilled() {
        let a = fresh("roundtrip");
        // Inline.
        let short = ShmString::<32>::from_str(&a, "hello, world");
        assert_eq!(short.str(&a), "hello, world");
        assert_eq!(short.as_str(&a).unwrap(), "hello, world");
        assert!(short.using_sso());
        // Spilled.
        let long_src = "the quick brown fox jumps over the lazy dog, repeatedly";
        let mut long = ShmString::<32>::from_str(&a, long_src);
        assert!(!long.using_sso());
        assert_eq!(long.str(&a), long_src);
        assert_eq!(long.as_str(&a).unwrap(), long_src);
        // Round-trip through a copy in the same allocator.
        let mut copy = long.clone_with(&a);
        assert_eq!(copy.str(&a), long_src);
        copy.destroy(&a);
        long.destroy(&a);
        // Embedded NULs survive (divergence 8: slices, not C strings).
        let mut nul = ShmString::<32>::from_bytes(&a, b"a\0b");
        assert_eq!(nul.len(), 3);
        assert_eq!(nul.as_bytes(&a), b"a\0b");
        nul.destroy(&a);
        // Invalid UTF-8 is lossy in str() but exact in as_bytes() (divergence 12).
        let mut bad = ShmString::<32>::from_bytes(&a, &[0xff, b'a']);
        assert_eq!(bad.as_bytes(&a), &[0xff, b'a']);
        assert!(bad.as_str(&a).is_err());
        assert_eq!(bad.str(&a), "\u{fffd}a");
        bad.destroy(&a);
    }

    #[test]
    fn c_str_terminates_inline_and_spilled() {
        let a = fresh("cstr");
        let mut s = ShmString::<32>::from_bytes(&a, b"abc");
        assert_eq!(s.c_str(&a), b"abc\0");
        // Spilled strings are not eagerly terminated; c_str() fixes that.
        let mut long = ShmString::<32>::from_bytes(&a, &[b'm'; 50]);
        let cs = long.c_str(&a);
        assert_eq!(cs.len(), 51);
        assert_eq!(cs[50], 0);
        assert!(cs[..50].iter().all(|&b| b == b'm'));
        // Also correct when capacity leaves no room for the terminator.
        long.resize_no_init(&a, long.capacity());
        let n = long.len();
        let cs = long.c_str(&a);
        assert_eq!(cs[n], 0);
        long.destroy(&a);
    }

    // ------------------------------------------- destroy releases the block

    #[test]
    fn destroy_releases_the_spill_block() {
        let a = fresh("destroy");
        let before = a.bump_remaining();
        let mut s = ShmString::<32>::from_bytes(&a, &[b'd'; 512]);
        assert!(!s.using_sso());
        let (cap, off) = s.heap_info();
        assert_eq!(cap, 513);
        assert!(a.bump_remaining() < before, "spilling must consume segment space");

        s.destroy(&a);

        // Reset to an empty inline string...
        assert!(s.using_sso());
        assert!(s.is_empty());
        // ...and the block is back on the free list: first-fit returns it.
        let reused = a.alloc_bytes(512).unwrap();
        assert_eq!(reused.off, off, "destroy() must free the spilled block");
    }

    #[test]
    fn destroy_is_idempotent_and_safe_on_inline() {
        let a = fresh("destroy_idem");
        let mut s = ShmString::<32>::from_bytes(&a, &[b'e'; 300]);
        let (_, off) = s.heap_info();
        s.destroy(&a);
        // A second destroy must NOT free the block again (divergence 4).
        s.destroy(&a);
        s.destroy(&a);
        // If the block had been double-freed, it would appear twice on the free
        // list and two allocations would alias the same offset.
        let p1 = a.alloc_bytes(300).unwrap();
        let p2 = a.alloc_bytes(300).unwrap();
        assert_eq!(p1.off, off);
        assert_ne!(p1.off, p2.off, "double-free would alias these allocations");
        // destroy on a never-spilled string is a no-op.
        let mut inline = ShmString::<32>::from_bytes(&a, b"tiny");
        inline.destroy(&a);
        assert!(inline.is_empty());
    }

    #[test]
    fn moving_a_spilled_string_needs_no_fixup() {
        // Divergence 2: the C++ move ctor re-points data_; a Rust move is a
        // plain memcpy and the offset stays valid.
        let a = fresh("move");
        let s = ShmString::<32>::from_bytes(&a, &[b'v'; 64]);
        let mut moved = s; // memcpy; `s` is statically dead afterwards
        assert_eq!(moved.as_bytes(&a), &[b'v'; 64]);
        let boxed = Box::new(moved);
        assert_eq!(boxed.as_bytes(&a), &[b'v'; 64]);
        moved = *boxed;
        moved.destroy(&a);
    }

    // ----------------------------------------------------------- modifiers

    #[test]
    fn resize_truncates_and_extends() {
        let a = fresh("resize");
        let mut s = ShmString::<32>::from_bytes(&a, b"abcdef");
        s.resize(&a, 3);
        assert_eq!(s.as_bytes(&a), b"abc");
        assert_eq!(s.storage[3], 0); // truncation writes the terminator
        s.resize(&a, 6);
        assert_eq!(s.as_bytes(&a), b"abc\0\0\0"); // extension fills with NUL
        s.resize(&a, 6);
        assert_eq!(s.len(), 6);
        // Extension past the inline capacity spills.
        s.resize(&a, 100);
        assert!(!s.using_sso());
        assert_eq!(s.len(), 100);
        assert_eq!(&s.as_bytes(&a)[..3], b"abc");
        assert!(s.as_bytes(&a)[3..].iter().all(|&b| b == 0));
        // Truncating a spilled string keeps it spilled (as in C++).
        s.resize(&a, 2);
        assert!(!s.using_sso());
        assert_eq!(s.as_bytes(&a), b"ab");
        s.destroy(&a);
    }

    #[test]
    fn erase_and_replace_inline_and_spilled() {
        let a = fresh("erase_replace");
        let mut s = ShmString::<32>::from_bytes(&a, b"hello world");
        s.erase(&a, 5, 6);
        assert_eq!(s.as_bytes(&a), b"hello");
        s.erase(&a, 1, NPOS); // to the end
        assert_eq!(s.as_bytes(&a), b"h");
        s.erase(&a, 0, 0); // no-op
        assert_eq!(s.as_bytes(&a), b"h");

        // replace, staying inline.
        let mut r = ShmString::<32>::from_bytes(&a, b"abcXYZghi");
        r.replace(&a, 3, 3, b"-");
        assert_eq!(r.as_bytes(&a), b"abc-ghi");
        // Grow-in-place inline (shorter → longer).
        r.replace(&a, 3, 1, b"12345");
        assert_eq!(r.as_bytes(&a), b"abc12345ghi");
        // replace that overflows the inline buffer spills.
        r.replace(&a, 0, 0, &[b'p'; 40]);
        assert!(!r.using_sso());
        assert_eq!(&r.as_bytes(&a)[..40], &[b'p'; 40]);
        assert_eq!(&r.as_bytes(&a)[40..], b"abc12345ghi");
        // replace on a spilled string, shrinking.
        r.replace(&a, 0, 40, b"");
        assert_eq!(r.as_bytes(&a), b"abc12345ghi");
        r.destroy(&a);

        // erase on a spilled string.
        let mut e = ShmString::<32>::from_bytes(&a, &[b'w'; 60]);
        e.erase(&a, 10, 40);
        assert_eq!(e.len(), 20);
        assert_eq!(e.as_bytes(&a), &[b'w'; 20]);
        e.destroy(&a);
    }

    #[test]
    #[should_panic(expected = "position out of range")]
    fn erase_past_end_panics_like_cpp_throws() {
        let a = fresh("erase_panic");
        let mut s = ShmString::<32>::from_bytes(&a, b"abc");
        s.erase(&a, 4, 1); // C++: CTP_THROW(std::out_of_range)
    }

    #[test]
    fn push_pop_and_at() {
        let a = fresh("push_pop");
        let mut s = ShmString::<32>::new();
        for c in b"abc" {
            s.push(&a, *c);
        }
        assert_eq!(s.as_bytes(&a), b"abc");
        assert_eq!(s.at(&a, 1), Some(b'b'));
        assert_eq!(s.at(&a, 3), None); // C++ at() throws; total here
        *s.at_mut(&a, 1).unwrap() = b'B';
        assert_eq!(s.as_bytes(&a), b"aBc");
        s.pop();
        assert_eq!(s.as_bytes(&a), b"aB");
        assert_eq!(s.storage[2], 0); // pop re-terminates inline strings
    }

    #[test]
    fn swap_exchanges_inline_and_spilled() {
        let a = fresh("swap");
        let mut short = ShmString::<32>::from_bytes(&a, b"tiny");
        let mut long = ShmString::<32>::from_bytes(&a, &[b'L'; 80]);
        short.swap(&mut long);
        assert!(!short.using_sso());
        assert_eq!(short.as_bytes(&a), &[b'L'; 80]);
        assert!(long.using_sso());
        assert_eq!(long.as_bytes(&a), b"tiny");
        short.destroy(&a);
        long.destroy(&a);
    }

    #[test]
    fn substr_and_clone_thresholds() {
        let a = fresh("substr");
        let src = ShmString::<32>::from_bytes(&a, b"hello world");
        let sub = src.substr(&a, 6, NPOS);
        assert_eq!(sub.as_bytes(&a), b"world");
        let sub2 = src.substr(&a, 0, 5);
        assert_eq!(sub2.as_bytes(&a), b"hello");
        // count past the end clamps.
        let sub3 = src.substr(&a, 6, 999);
        assert_eq!(sub3.as_bytes(&a), b"world");
        // pos == len is legal and yields "".
        assert!(src.substr(&a, 11, NPOS).is_empty());

        // Faithful C++ quirk: copying an inline 31-byte string SPILLS the copy,
        // because the copy ctor uses the stricter `< SSOSize - 1` predicate.
        let mut built = ShmString::<32>::new();
        built.append_repeat(&a, 31, b'j');
        assert!(built.using_sso());
        let mut copy = built.clone_with(&a);
        assert!(!copy.using_sso(), "copy ctor threshold is one byte stricter");
        assert_eq!(copy.as_bytes(&a), built.as_bytes(&a));
        copy.destroy(&a);
    }

    // ---------------------------------------------------------- comparison

    #[test]
    fn compare_eq_and_affixes() {
        let a = fresh("compare");
        let s = ShmString::<32>::from_bytes(&a, b"banana");
        assert!(s.eq_bytes(&a, b"banana"));
        assert!(!s.eq_bytes(&a, b"banan"));
        assert!(!s.eq_bytes(&a, b"bananas"));
        assert_eq!(s.compare(&a, b"banana"), Ordering::Equal);
        assert_eq!(s.compare(&a, b"apple"), Ordering::Greater);
        assert_eq!(s.compare(&a, b"cherry"), Ordering::Less);
        // Prefix ordering: shorter sorts first (C++ compare's length tiebreak).
        assert_eq!(s.compare(&a, b"ban"), Ordering::Greater);
        assert_eq!(s.compare(&a, b"bananas"), Ordering::Less);
        // memcmp ordering is UNSIGNED even for high bytes (divergence 7).
        let hi = ShmString::<32>::from_bytes(&a, &[0x80]);
        assert_eq!(hi.compare(&a, &[0x7f]), Ordering::Greater);

        assert!(s.starts_with(&a, b"ban"));
        assert!(!s.starts_with(&a, b"nan"));
        assert!(s.ends_with(&a, b"ana"));
        assert!(!s.ends_with(&a, b"anb"));
        // Empty affixes always match (the C++ nullptr behavior).
        assert!(s.starts_with(&a, b""));
        assert!(s.ends_with(&a, b""));
        // Longer-than-self affixes are false, never an overflow.
        assert!(!s.starts_with(&a, b"bananas!!"));
        assert!(!s.ends_with(&a, b"bananas!!"));
        let e = ShmString::<32>::new();
        assert!(!e.ends_with(&a, b"x"));
    }

    #[test]
    fn find_matches_cpp_edge_cases() {
        let a = fresh("find");
        let s = ShmString::<32>::from_bytes(&a, b"abcabc");
        assert_eq!(s.find(&a, b"abc", 0), Some(0));
        assert_eq!(s.find(&a, b"abc", 1), Some(3));
        assert_eq!(s.find(&a, b"abc", 4), None);
        assert_eq!(s.find(&a, b"zzz", 0), None);
        // Empty needle: matches at pos while pos <= size, else npos.
        assert_eq!(s.find(&a, b"", 0), Some(0));
        assert_eq!(s.find(&a, b"", 6), Some(6));
        assert_eq!(s.find(&a, b"", 7), None);
        // Huge pos must not overflow (divergence 11; C++ wraps here).
        assert_eq!(s.find(&a, b"abc", NPOS), None);
        assert_eq!(s.find(&a, b"", NPOS), None);
        // find(char, pos).
        assert_eq!(s.find_byte(&a, b'c', 0), Some(2));
        assert_eq!(s.find_byte(&a, b'c', 3), Some(5));
        assert_eq!(s.find_byte(&a, b'z', 0), None);
        assert_eq!(s.find_byte(&a, b'a', 6), None); // pos >= size
        // Spilled strings search the segment block.
        let mut long = ShmString::<32>::from_bytes(&a, b"0123456789012345678901234567890123456789needle");
        assert_eq!(long.find(&a, b"needle", 0), Some(40));
        assert_eq!(long.find_byte(&a, b'n', 0), Some(40));
        long.destroy(&a);
    }

    #[test]
    fn fnv1a_matches_cpp_reference_values() {
        let a = fresh("hash");
        // Reference FNV-1a/64 values for the C++ ctp::hash implementation.
        let empty = ShmString::<32>::new();
        assert_eq!(empty.fnv1a(&a), 0xcbf2_9ce4_8422_2325);
        let s = ShmString::<32>::from_bytes(&a, b"a");
        assert_eq!(s.fnv1a(&a), 0xaf63_dc4c_8601_ec8c);
        let s = ShmString::<32>::from_bytes(&a, b"foobar");
        assert_eq!(s.fnv1a(&a), 0x85944171f73967e8);
        // Same content ⇒ same hash regardless of inline/spilled storage.
        let inline = ShmString::<32>::from_bytes(&a, &[b'h'; 20]);
        let mut spilled = ShmString::<32>::from_bytes(&a, &[b'h'; 20]);
        spilled.reserve(&a, 200);
        assert!(!spilled.using_sso() && inline.using_sso());
        assert_eq!(inline.fnv1a(&a), spilled.fnv1a(&a));
        spilled.destroy(&a);
    }

    // ---------------------------------------------------------- SsoState

    #[test]
    fn sso_state_round_trip() {
        let a = fresh("sso_state");
        let s = ShmString::<32>::from_bytes(&a, b"pod-safe");
        let state = s.sso_state().unwrap();
        assert_eq!(state.size, 8);
        let restored = ShmString::<32>::from_sso_state(&state);
        assert_eq!(restored.as_bytes(&a), b"pod-safe");
        assert!(restored.using_sso());
        // Spilled strings have no POD state (divergence 13).
        let mut long = ShmString::<32>::from_bytes(&a, &[b'p'; 90]);
        assert!(long.sso_state().is_none());
        long.destroy(&a);
    }

    // ------------------------------------------------------ number → string

    #[test]
    fn number_to_str_covers_zero_negatives_and_min() {
        let mut buf = [0u8; 32];
        let n = ShmString::<32>::uint_to_str(&mut buf, 0);
        assert_eq!(&buf[..n], b"0");
        let n = ShmString::<32>::uint_to_str(&mut buf, 1234567890);
        assert_eq!(&buf[..n], b"1234567890");
        let n = ShmString::<32>::uint_to_str(&mut buf, u64::MAX);
        assert_eq!(&buf[..n], b"18446744073709551615");
        let n = ShmString::<32>::int_to_str(&mut buf, -42);
        assert_eq!(&buf[..n], b"-42");
        let n = ShmString::<32>::int_to_str(&mut buf, 0);
        assert_eq!(&buf[..n], b"0");
        // i64::MIN negates through the unsigned type, as the C++ does.
        let n = ShmString::<32>::int_to_str(&mut buf, i64::MIN);
        assert_eq!(&buf[..n], b"-9223372036854775808");
        let n = ShmString::<32>::int_to_str(&mut buf, i64::MAX);
        assert_eq!(&buf[..n], b"9223372036854775807");
    }

    #[test]
    fn number_to_str_buffer_edges_match_cpp() {
        // buf_size == 0 → nothing written.
        assert_eq!(ShmString::<32>::uint_to_str(&mut [], 5), 0);
        assert_eq!(ShmString::<32>::int_to_str(&mut [], -5), 0);
        assert_eq!(ShmString::<32>::float_to_str(&mut [], 1.5, 2), 0);
        // val == 0 in a 1-byte buffer: digit written, no room for the NUL.
        let mut one = [0xffu8; 1];
        assert_eq!(ShmString::<32>::uint_to_str(&mut one, 0), 1);
        assert_eq!(one[0], b'0');
        // Truncation keeps the MOST significant digits (faithful C++ quirk).
        let mut small = [0u8; 4];
        let n = ShmString::<32>::uint_to_str(&mut small, 123456);
        assert_eq!(n, 3);
        assert_eq!(&small[..n], b"123");
        assert_eq!(small[3], 0);
        // "-" alone when only one byte is available for a negative.
        let mut tiny = [0u8; 1];
        assert_eq!(ShmString::<32>::int_to_str(&mut tiny, -7), 1);
        assert_eq!(tiny[0], b'-');
    }

    #[test]
    fn float_to_str_and_from_number() {
        let a = fresh("numbers");
        let mut buf = [0u8; 64];
        let n = ShmString::<32>::float_to_str(&mut buf, 3.5, 2);
        assert_eq!(&buf[..n], b"3.50");
        let n = ShmString::<32>::float_to_str(&mut buf, -0.25, 3);
        assert_eq!(&buf[..n], b"-0.250");
        // precision 0 → no decimal point at all.
        let n = ShmString::<32>::float_to_str(&mut buf, 12.9, 0);
        assert_eq!(&buf[..n], b"12");
        // NaN: C++ is UB here; Rust saturates to 0 (divergence 14).
        let n = ShmString::<32>::float_to_str(&mut buf, f64::NAN, 1);
        assert_eq!(&buf[..n], b"0.0");

        // FromNumber equivalents (the C++ template is uninstantiable —
        // divergence 14).
        let s = ShmString::<32>::from_i64(&a, -12345);
        assert_eq!(s.str(&a), "-12345");
        assert!(s.using_sso());
        let s = ShmString::<32>::from_u64(&a, u64::MAX);
        assert_eq!(s.str(&a), "18446744073709551615");
        let s = ShmString::<32>::from_f64(&a, 1.5, 6);
        assert_eq!(s.str(&a), "1.500000");
    }

    // ------------------------------------------------ non-default SSO sizes

    #[test]
    fn custom_sso_size_shifts_both_thresholds() {
        let a = fresh("sso16");
        assert_eq!(ShmString::<16>::SSO_CTOR_MAX, 14);
        assert_eq!(ShmString::<16>::SSO_APPEND_MAX, 15);
        // 14 inline, 15 spills (ctor predicate).
        let s14 = ShmString::<16>::from_bytes(&a, &[b'a'; 14]);
        assert!(s14.using_sso());
        let mut s15 = ShmString::<16>::from_bytes(&a, &[b'a'; 15]);
        assert!(!s15.using_sso());
        assert_eq!(s15.as_bytes(&a), &[b'a'; 15]);
        s15.destroy(&a);
        // ...but appending reaches 15 inline.
        let mut built = ShmString::<16>::new();
        built.append_repeat(&a, 15, b'b');
        assert!(built.using_sso());
        assert_eq!(built.len(), 15);
        built.push(&a, b'c');
        assert!(!built.using_sso());
        assert_eq!(built.len(), 16);
        built.destroy(&a);
        // A 64-byte inline buffer.
        let mut big = ShmString::<64>::from_bytes(&a, &[b'z'; 62]);
        assert!(big.using_sso());
        assert_eq!(big.capacity(), 64);
        big.append_bytes(&a, b"x");
        assert!(big.using_sso()); // 63 == SSO_APPEND_MAX
        big.append_bytes(&a, b"y");
        assert!(!big.using_sso());
        assert_eq!(big.len(), 64);
        big.destroy(&a);
    }

    // ------------------------------------------------------------ churn

    #[test]
    fn heavy_churn_does_not_leak_or_corrupt() {
        let a = fresh("churn");
        let before = a.bump_remaining();
        for i in 0..200usize {
            let mut s = ShmString::<32>::new();
            for j in 0..(i % 90) {
                s.push(&a, b'a' + (j % 26) as u8);
            }
            let expect: Vec<u8> = (0..(i % 90)).map(|j| b'a' + (j % 26) as u8).collect();
            assert_eq!(s.as_bytes(&a), &expect[..]);
            s.destroy(&a);
        }
        // Every spilled block was returned: the bump pointer stops advancing
        // once the free list can satisfy the churn.
        assert!(
            a.bump_remaining() > before - (1 << 16),
            "spilled blocks are not being recycled"
        );
    }

    #[test]
    fn strings_in_one_segment_are_independent() {
        let a = fresh("multi");
        let mut xs: Vec<ShmString<32>> = (0..16)
            .map(|i| ShmString::<32>::from_repeat(&a, 40 + i, b'0' + i as u8))
            .collect();
        for (i, s) in xs.iter().enumerate() {
            assert!(!s.using_sso());
            assert_eq!(s.len(), 40 + i);
            assert!(s.as_bytes(&a).iter().all(|&b| b == b'0' + i as u8));
        }
        for s in xs.iter_mut() {
            s.destroy(&a);
        }
    }
}
