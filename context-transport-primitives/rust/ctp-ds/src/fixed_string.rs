// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! `ctp::priv::fixed_string<N>` → [`FixedString<N>`]: the POD, GPU-safe inline
//! string carried by the `Pod*Blob` tasks (issue #556).
//!
//! No heap, no SSO, no self-referential pointer, no `Drop` — the type is
//! bitwise-relocatable, so a raw `memcpy`/`cudaMemcpy` of a struct embedding a
//! `FixedString` is correct with **zero post-copy fixup**. That is precisely why
//! GPU tasks can carry it without the `priv::string` SSO-pointer fixup dance.
//!
//! # Byte layout (frozen — this type crosses SHM and the wire)
//!
//! The C++ type derives from `array_vector<N>` and adds **no** data members, so
//! both languages describe the same bytes:
//!
//! ```text
//! C++:  class alignas(8) array_vector<N> { alignas(8) char data_[N]; size_t size_; };
//! Rust: #[repr(C, align(8))] struct FixedString<N> { data: [u8; N], size: u64 }
//!
//! offset 0                    align_up(N, 8)        align_up(N, 8) + 8
//!   +-------------------------+---- pad ----+---------------+
//!   | data: N bytes           |  0..7 bytes | size: u64     |
//!   +-------------------------+-------------+---------------+
//!
//! align_of = 8            size_of = align_up(N, 8) + 8
//! ```
//!
//! For the default `N = 32`: `data` at offset 0 (32 bytes), `size` at offset 32,
//! `size_of` = 40, `align_of` = 8. Both fields are **native-endian**, matching
//! the C++ archives (which `memcpy` a `size_t`); no cross-endian interop is
//! offered by either side.
//!
//! ## Invariants
//!
//! - `size <= MAX_LEN` where `MAX_LEN == N - 1`;
//! - `data[size] == 0` — one byte is reserved for the NUL terminator so
//!   [`c_str`](FixedString::c_str) is always valid. Inputs longer than `MAX_LEN`
//!   are **truncated** (never an error, matching C++ `assign`).
//!
//! Every constructor and mutator here upholds both. A byte pattern arriving from
//! shared memory can violate them (a hostile or buggy writer), so the accessors
//! are defensive — see the divergence list.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::priv::fixed_string<N>`)      | Rust (`FixedString<N>`)                    |
//! |-----------------------------------------|--------------------------------------------|
//! | `fixed_string<N>`                       | `FixedString<N>` (`N` defaults to 32)      |
//! | `kMaxLen`                               | `FixedString::<N>::MAX_LEN`                |
//! | `CStrLen(const char*)`                  | [`c_str_len`] (module-level fn)            |
//! | `fixed_string()`                        | [`FixedString::new`] / `Default::default`  |
//! | `fixed_string(const char *s)`           | `From<&CStr>`                              |
//! | `fixed_string(const char *s, size_t n)` | `From<&[u8]>` / [`FixedString::assign`]    |
//! | `fixed_string(const std::string&)`      | `From<&str>`                               |
//! | `fixed_string(std::string_view)`        | `From<&str>`                               |
//! | `operator=(const char*)`                | `From<&CStr>` + assignment                 |
//! | `operator=(const std::string&)`         | `From<&str>` + assignment                  |
//! | `assign(const char *s, size_t len)`     | [`FixedString::assign`] (`&[u8]`)          |
//! | `assign(s.data(), s.size())`            | [`FixedString::assign_str`] (`&str`)       |
//! | `c_str()`                               | [`FixedString::c_str`] → `&CStr`           |
//! | `data()` / `view()` (const)             | [`FixedString::as_bytes`] → `&[u8]`        |
//! | `data()` (mutable)                      | [`FixedString::as_bytes_mut`] (see below)  |
//! | `str()` / `operator std::string()`      | [`FixedString::as_str`] / `TryFrom<_> for String` |
//! | `size()` / `length()`                   | [`FixedString::len`]                       |
//! | `empty()`                               | [`FixedString::is_empty`]                  |
//! | `max_size()`                            | [`FixedString::max_size`] (== `MAX_LEN`)   |
//! | `capacity()` (from `array_vector`)      | [`FixedString::capacity`] (== `N`)         |
//! | `reserve(n)` (from `array_vector`)      | [`FixedString::reserve`]                   |
//! | `operator[](i)`                         | `Index<usize>` (`Output = u8`)             |
//! | `clear()`                               | [`FixedString::clear`]                     |
//! | `push_back(char)`                       | [`FixedString::push_back`]                 |
//! | `operator==(const fixed_string&)`       | `PartialEq<FixedString<N>>`                |
//! | `operator==(const char*)`               | `PartialEq<CStr>`                          |
//! | `operator==(const std::string&)`        | `PartialEq<str>` / `PartialEq<&str>`       |
//! | `save(Archive&)`                        | [`FixedString::save_into`]                 |
//! | `load(Archive&)`                        | [`FixedString::load_from`]                 |
//! | `CalculateSizeArchive` accounting       | [`FixedString::serialized_size`]           |
//! | `std::hash<fixed_string<N>>`            | `impl Hash` (hashes `as_bytes()`)          |
//! | `begin()` / `end()`                     | [`FixedString::iter`]                      |
//!
//! # Semantic divergences from the C++
//!
//! 1. **`size_t` → `u64` (deliberate pin).** C++ `size_` is `size_t`, i.e.
//!    platform-width. The port pins the field to `u64` so the SHM/wire layout is
//!    identical regardless of target width. On every supported (64-bit) target
//!    this is byte-identical to the C++; on a hypothetical 32-bit build the Rust
//!    struct would be 4 bytes larger than the C++ one and would NOT interop.
//! 2. **`char` → `u8`.** C++ `char` has implementation-defined signedness; the
//!    port uses `u8` throughout. Byte values are unaffected; only the sign of a
//!    single-element comparison would differ, and no such comparison is exposed.
//! 3. **The whole buffer is zero-filled** by `new`/`assign`/`clear`. C++ leaves
//!    `data_[size_+1 .. N]` indeterminate (its default ctor writes only
//!    `data_[0]`). Since the primary transport for this type is a raw memcpy of
//!    an embedding struct, C++ ships stale/uninitialised tail bytes over SHM and
//!    the GPU bus; the port makes those bytes deterministically zero. No
//!    accessor's result changes, and C++ readers are unaffected (they read at
//!    most `size_` bytes / up to the first NUL) — this is hardening, not a
//!    behaviour change.
//! 4. **`load` buffer overflow is not reproduced.** C++ `load` calls
//!    `resize_no_init(size)` with the *stream-declared* size and then reads that
//!    many bytes into the `N`-byte inline buffer — overflowing it — and only
//!    *afterwards* clamps `size_` to `kMaxLen`. [`load_from`] copies at most
//!    `MAX_LEN` bytes and then skips the cursor past the declared length, which
//!    yields exactly the C++ *post-clamp* value (first `MAX_LEN` bytes,
//!    `size == MAX_LEN`) without the undefined behaviour.
//! 5. **Corrupt byte patterns are handled, not UB.** `FixedString` is `ShmSafe`,
//!    so it may be materialised from arbitrary bytes. [`len`](FixedString::len)
//!    clamps a stored `size > MAX_LEN` down to `MAX_LEN` (C++ would read out of
//!    bounds), and [`c_str`](FixedString::c_str) returns an empty `CStr` when the
//!    buffer holds no NUL at all (C++ `c_str()` would run off the end).
//!    [`is_valid`](FixedString::is_valid) — which has no C++ counterpart — reports
//!    whether the invariants actually hold.
//! 6. **`str()` is fallible.** C++ `str()`/`view()` hand back the bytes with no
//!    encoding check. [`as_str`](FixedString::as_str) validates UTF-8 and returns
//!    `Err(Utf8Error)`; the byte-exact equivalent is
//!    [`as_bytes`](FixedString::as_bytes), and [`to_string_lossy`](FixedString::to_string_lossy)
//!    is the infallible convenience. This matters because `assign` truncates on a
//!    **byte** boundary and can split a multi-byte character (faithful to C++).
//! 7. **Public data members are not exposed.** C++ inherits `data_`/`size_` as
//!    public members; keeping them private is what makes the invariants
//!    enforceable. `char *data()` (whole-buffer mutable) and `array_vector`'s
//!    `resize`/`resize_no_init` are likewise **not** ported: they exist to let the
//!    serializer fill a buffer and set a length independently, which is precisely
//!    how the invariant gets broken (and how divergence 4 happens). That use case
//!    is served by [`assign`](FixedString::assign) and [`load_from`]. In-place
//!    mutation of existing content is available via
//!    [`as_bytes_mut`](FixedString::as_bytes_mut), which cannot change the length.
//! 8. **`save`/`load` are byte-slice functions, not `Archive` generics.** The C++
//!    templates dispatch over any archive. The port emits/consumes exactly the
//!    archive's byte stream (`u64` native-endian length prefix, then that many
//!    payload bytes, no NUL) via [`save_into`](FixedString::save_into) /
//!    [`load_from`]; binding these to the `local_serialize`/`global_serialize`
//!    archive traits is deferred to those modules (separate ports).
//! 9. **`push_back` returns `bool`** (`true` when the byte was appended). C++
//!    returns `void` and silently drops the byte when full; the drop behaviour is
//!    preserved, the return value is purely additive.
//! 10. **`std::hash` parity is not attempted.** The impl hashes `as_bytes()` (the
//!     C++ specialisation hashes `view()`, i.e. the same bytes), but the digest
//!     algorithm is Rust's — hash *values* never matched across languages anyway
//!     (`std::hash` is not portable). Only `Eq`/`Hash` consistency is guaranteed.
//! 11. **Comparison against `FixedString<M>` where `M != N` is not offered**, as
//!     in C++ (`operator==` takes the same `N`). Compare `as_bytes()` for that.
//! 12. **`c_str_len` is bounded by the slice.** C++ `CStrLen` scans an unbounded
//!     `const char*` (and returns 0 for `nullptr`); the port takes a `&[u8]` and
//!     returns `bytes.len()` when no NUL is present. Rust has no null slice; the
//!     `nullptr` case corresponds to an empty slice, which also yields 0.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Index;
use std::borrow::Cow;
use std::ffi::CStr;
use std::str::Utf8Error;

use ctp_memory::ShmSafe;

/// Port of `fixed_string::CStrLen` — length of the NUL-terminated string at the
/// start of `bytes`, i.e. the index of the first NUL, or `bytes.len()` if there
/// is none (the C++ would keep scanning past the end; see divergence 12).
#[inline]
#[must_use]
pub fn c_str_len(bytes: &[u8]) -> usize {
    match bytes.iter().position(|&b| b == 0) {
        Some(i) => i,
        None => bytes.len(),
    }
}

/// Fixed-capacity, inline (POD) string — `ctp::priv::fixed_string<N>`.
///
/// `N` is the **total buffer capacity in bytes**; the usable string length is at
/// most `N - 1` ([`MAX_LEN`](Self::MAX_LEN)) because one byte is reserved for the
/// NUL terminator. See the [module docs](self) for the exact byte layout and the
/// divergence list.
///
/// `Copy`, no `Drop`, `#[repr(C, align(8))]`, [`ShmSafe`]: bitwise-relocatable
/// across processes, hosts, and the GPU bus.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct FixedString<const N: usize = 32> {
    /// The inline buffer. `data[len()]` is always 0 for a valid value.
    data: [u8; N],
    /// Live byte count. Pinned to `u64` (C++ `size_t`) — divergence 1.
    size: u64,
}

// SAFETY: `FixedString` is `Copy + 'static`, `#[repr(C)]`, has no `Drop`, holds
// no references and no process-local addresses (the whole point of the type —
// it is the GPU/SHM-transportable string). Every bit pattern of `[u8; N]` and
// `u64` is a valid value of those field types, and the accessors defensively
// clamp a `size` that violates the invariant rather than trusting it, so a
// hostile writer cannot induce UB through this type (divergence 5).
unsafe impl<const N: usize> ShmSafe for FixedString<N> {}

impl<const N: usize> FixedString<N> {
    /// Rejects `FixedString<0>`, which cannot hold even a NUL terminator (C++
    /// would silently underflow `kMaxLen = N - 1` to `SIZE_MAX`). Referenced from
    /// [`new`](Self::new) so the assertion is forced at monomorphisation.
    const CHECK_CAPACITY: () = assert!(
        N >= 1,
        "FixedString<N> requires N >= 1: one byte is reserved for the NUL terminator"
    );

    /// Max usable string length — C++ `kMaxLen` (`N - 1`).
    pub const MAX_LEN: usize = N - 1;

    /// C++ `fixed_string()` — empty string, `data[0] == 0`.
    ///
    /// Unlike the C++ ctor, the entire buffer is zeroed (divergence 3).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        let () = Self::CHECK_CAPACITY;
        Self {
            data: [0u8; N],
            size: 0,
        }
    }

    /// C++ `assign(const char *s, size_t len)` — copy `min(src.len(), MAX_LEN)`
    /// bytes and NUL-terminate. Truncation is silent, exactly as in C++.
    ///
    /// Truncation is by **byte**, so a multi-byte UTF-8 character straddling the
    /// boundary is split (divergence 6).
    #[inline]
    pub fn assign(&mut self, src: &[u8]) {
        let len = if src.len() > Self::MAX_LEN {
            Self::MAX_LEN
        } else {
            src.len()
        };
        // Zero-fill first (divergence 3): this both scrubs the stale tail and
        // plants the NUL terminator at `data[len]` for free.
        self.data = [0u8; N];
        self.data[..len].copy_from_slice(&src[..len]);
        self.size = len as u64;
    }

    /// C++ `assign(s.data(), s.size())` for `std::string`/`std::string_view`.
    #[inline]
    pub fn assign_str(&mut self, src: &str) {
        self.assign(src.as_bytes());
    }

    /// C++ `c_str()` — the NUL-terminated view, truncated at the **first** NUL.
    ///
    /// For an embedded NUL this is shorter than [`as_bytes`](Self::as_bytes);
    /// that asymmetry is faithful to C++. Returns an empty `CStr` for a corrupt
    /// buffer that contains no NUL at all (divergence 5).
    #[inline]
    #[must_use]
    pub fn c_str(&self) -> &CStr {
        CStr::from_bytes_until_nul(&self.data).unwrap_or(c"")
    }

    /// C++ `data()` (const) / `view()` — the live bytes, embedded NULs included.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len()]
    }

    /// In-place mutable view of the live bytes. The length cannot change, so the
    /// invariants survive any write. (C++'s `char *data()` hands out the whole
    /// buffer; see divergence 7.)
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let len = self.len();
        &mut self.data[..len]
    }

    /// The raw `N`-byte buffer, including the terminator and the zeroed tail.
    /// Useful for layout/parity checks; prefer [`as_bytes`](Self::as_bytes).
    #[inline]
    #[must_use]
    pub const fn buffer(&self) -> &[u8; N] {
        &self.data
    }

    /// C++ `str()` / `operator std::string()`, but UTF-8-checked (divergence 6).
    #[inline]
    pub fn as_str(&self) -> Result<&str, Utf8Error> {
        core::str::from_utf8(self.as_bytes())
    }

    /// Infallible companion to [`as_str`](Self::as_str): invalid sequences become
    /// U+FFFD. No C++ counterpart.
    #[inline]
    #[must_use]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(self.as_bytes())
    }

    /// C++ `size()` / `length()`.
    ///
    /// Defensively clamped to [`MAX_LEN`](Self::MAX_LEN): a `size` field arriving
    /// from shared memory is untrusted (divergence 5). Use
    /// [`raw_size`](Self::raw_size) to inspect the stored value verbatim.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        let n = self.size as usize;
        if n > Self::MAX_LEN {
            Self::MAX_LEN
        } else {
            n
        }
    }

    /// The `size` field exactly as stored, with no clamp — for layout parity
    /// checks and corruption diagnostics. No C++ counterpart.
    #[inline]
    #[must_use]
    pub const fn raw_size(&self) -> u64 {
        self.size
    }

    /// C++ `empty()`.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Whether the type's invariants actually hold for these bytes: `size` is in
    /// range and the terminator is present. Always `true` for a value built
    /// through this API; may be `false` for one materialised from raw shared
    /// memory. No C++ counterpart (divergence 5).
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (self.size as usize) <= Self::MAX_LEN && self.data[self.size as usize] == 0
    }

    /// C++ `max_size()` — the longest string that fits (`N - 1`).
    #[inline]
    #[must_use]
    pub const fn max_size() -> usize {
        Self::MAX_LEN
    }

    /// C++ `array_vector::capacity()` — the total buffer size (`N`).
    #[inline]
    #[must_use]
    pub const fn capacity() -> usize {
        N
    }

    /// C++ `array_vector::reserve(n)` — a pure predicate; reserves nothing.
    #[inline]
    #[must_use]
    pub const fn reserve(n: usize) -> bool {
        n <= N
    }

    /// C++ `clear()`. Also scrubs the buffer (divergence 3).
    #[inline]
    pub fn clear(&mut self) {
        self.data = [0u8; N];
        self.size = 0;
    }

    /// C++ `push_back(char c)` — appends unless full, in which case the byte is
    /// silently dropped. Returns whether it was appended (divergence 9).
    #[inline]
    pub fn push_back(&mut self, c: u8) -> bool {
        let len = self.len();
        if len < Self::MAX_LEN {
            self.data[len] = c;
            self.size = (len + 1) as u64;
            self.data[len + 1] = 0;
            true
        } else {
            false
        }
    }

    /// C++ `begin()`/`end()` over the live bytes.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, u8> {
        self.as_bytes().iter()
    }

    /// Bytes this value contributes to an archive — C++ `CalculateSizeArchive`
    /// accounting (`sizeof(size_t) + len`, with `size_t` pinned to 8).
    #[inline]
    #[must_use]
    pub const fn serialized_size(&self) -> usize {
        core::mem::size_of::<u64>() + self.len()
    }

    /// C++ `save(Archive&)` → `save_string` → `ar.save_string_fused(data, size)`:
    /// a native-endian `u64` length prefix followed by that many payload bytes
    /// (no NUL). Appends to `out`; see divergence 8.
    pub fn save_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.len() as u64).to_ne_bytes());
        out.extend_from_slice(self.as_bytes());
    }

    /// C++ `load(Archive&)`. Returns the value and the number of bytes consumed
    /// (`8 + declared_len`), or `None` if `buf` is too short to hold the prefix
    /// and the full declared payload.
    ///
    /// A declared length above [`MAX_LEN`](Self::MAX_LEN) truncates to the first
    /// `MAX_LEN` bytes and consumes the whole declared payload — the C++
    /// post-clamp result, minus the buffer overflow it takes to get there
    /// (divergence 4).
    pub fn load_from(buf: &[u8]) -> Option<(Self, usize)> {
        const PREFIX: usize = core::mem::size_of::<u64>();
        let prefix = buf.get(..PREFIX)?;
        let mut raw = [0u8; PREFIX];
        raw.copy_from_slice(prefix);
        let declared = usize::try_from(u64::from_ne_bytes(raw)).ok()?;
        let end = PREFIX.checked_add(declared)?;
        let body = buf.get(PREFIX..end)?;
        let mut out = Self::new();
        // `assign` performs the clamp to MAX_LEN, keeping the leading bytes.
        out.assign(body);
        Some((out, end))
    }
}

impl<const N: usize> Default for FixedString<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// --- Converting constructors (C++ implicit ctors / operator=) ---

impl<const N: usize> From<&[u8]> for FixedString<N> {
    /// C++ `fixed_string(const char *s, size_t len)`. Truncates silently.
    #[inline]
    fn from(src: &[u8]) -> Self {
        let mut out = Self::new();
        out.assign(src);
        out
    }
}

impl<const N: usize> From<&str> for FixedString<N> {
    /// C++ `fixed_string(const std::string&)` / `fixed_string(std::string_view)`.
    /// Truncates silently, on a byte boundary (divergence 6).
    #[inline]
    fn from(src: &str) -> Self {
        Self::from(src.as_bytes())
    }
}

impl<const N: usize> From<&String> for FixedString<N> {
    #[inline]
    fn from(src: &String) -> Self {
        Self::from(src.as_str())
    }
}

impl<const N: usize> From<&CStr> for FixedString<N> {
    /// C++ `fixed_string(const char *s)` — length via `CStrLen`.
    #[inline]
    fn from(src: &CStr) -> Self {
        Self::from(src.to_bytes())
    }
}

impl<const N: usize> TryFrom<FixedString<N>> for String {
    type Error = Utf8Error;

    /// C++ `operator std::string()`, but UTF-8-checked (divergence 6).
    #[inline]
    fn try_from(src: FixedString<N>) -> Result<Self, Self::Error> {
        src.as_str().map(ToOwned::to_owned)
    }
}

// --- Accessors / comparison ---

impl<const N: usize> Index<usize> for FixedString<N> {
    type Output = u8;

    /// C++ `operator[](size_t i)`, which indexes the raw buffer (so `i >= size()`
    /// is legal up to `N - 1`). Rust panics for `i >= N` where C++ is UB.
    #[inline]
    fn index(&self, i: usize) -> &u8 {
        &self.data[i]
    }
}

impl<const N: usize> PartialEq for FixedString<N> {
    /// C++ `operator==(const fixed_string&)` — same length, same live bytes. The
    /// buffer tail is never compared.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl<const N: usize> Eq for FixedString<N> {}

impl<const N: usize> PartialEq<str> for FixedString<N> {
    /// C++ `operator==(const std::string&)` — a full byte compare of `view()`,
    /// so embedded NULs participate.
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl<const N: usize> PartialEq<&str> for FixedString<N> {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl<const N: usize> PartialEq<FixedString<N>> for str {
    #[inline]
    fn eq(&self, other: &FixedString<N>) -> bool {
        other == self
    }
}

impl<const N: usize> PartialEq<String> for FixedString<N> {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl<const N: usize> PartialEq<[u8]> for FixedString<N> {
    #[inline]
    fn eq(&self, other: &[u8]) -> bool {
        self.as_bytes() == other
    }
}

impl<const N: usize> PartialEq<CStr> for FixedString<N> {
    /// C++ `operator==(const char *s)`: compares `CStrLen(s)` against `size()`
    /// first, so a value with an embedded NUL can never equal a C string.
    #[inline]
    fn eq(&self, other: &CStr) -> bool {
        self.as_bytes() == other.to_bytes()
    }
}

impl<const N: usize> Hash for FixedString<N> {
    /// C++ `std::hash<fixed_string<N>>` hashes `view()`; this hashes the same
    /// bytes with Rust's hasher. Digests do not match across languages
    /// (divergence 10).
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

impl<const N: usize> fmt::Debug for FixedString<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedString")
            .field("capacity", &N)
            .field("len", &self.len())
            .field("value", &self.to_string_lossy())
            .finish()
    }
}

impl<const N: usize> fmt::Display for FixedString<N> {
    /// Lossy: invalid UTF-8 renders as U+FFFD (divergence 6).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_lossy())
    }
}

impl<'a, const N: usize> IntoIterator for &'a FixedString<N> {
    type Item = &'a u8;
    type IntoIter = core::slice::Iter<'a, u8>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashMap;
    use std::mem::{align_of, offset_of, size_of};

    /// Reinterpret raw bytes as a `FixedString<8>` — the SHM/wire arrival path,
    /// used here to forge byte patterns the safe API cannot produce.
    fn from_raw8(raw: [u8; 16]) -> FixedString<8> {
        // SAFETY: `size_of::<FixedString<8>>() == 16` (asserted in
        // `layout_matches_cpp_array_vector`), and `FixedString` is `ShmSafe`:
        // every bit pattern of its `[u8; 8]` and `u64` fields is a valid value,
        // so any 16 bytes are a valid (possibly invariant-violating) instance.
        unsafe { core::mem::transmute::<[u8; 16], FixedString<8>>(raw) }
    }

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    // --- Layout parity (the reason this type exists) ---

    #[test]
    fn layout_matches_cpp_array_vector() {
        // C++: alignas(8) { alignas(8) char data_[N]; size_t size_; }
        // => data_ at 0, size_ at align_up(N, 8), size = align_up(N, 8) + 8.
        assert_eq!(size_of::<FixedString<32>>(), 40);
        assert_eq!(align_of::<FixedString<32>>(), 8);
        assert_eq!(offset_of!(FixedString<32>, data), 0);
        assert_eq!(offset_of!(FixedString<32>, size), 32);

        assert_eq!(size_of::<FixedString<8>>(), 16);
        assert_eq!(offset_of!(FixedString<8>, size), 8);

        // N = 1: the smallest legal capacity (empty strings only).
        assert_eq!(size_of::<FixedString<1>>(), 16);
        assert_eq!(align_of::<FixedString<1>>(), 8);
        assert_eq!(offset_of!(FixedString<1>, size), 8);

        // Non-multiple-of-8 N pads up to the u64's alignment, as in C++.
        assert_eq!(size_of::<FixedString<10>>(), 24);
        assert_eq!(offset_of!(FixedString<10>, size), 16);
        assert_eq!(size_of::<FixedString<16>>(), 24);
        assert_eq!(offset_of!(FixedString<16>, size), 16);
    }

    #[test]
    fn raw_bytes_are_exactly_the_wire_image() {
        let s = FixedString::<8>::from("hi");
        // SAFETY: see `from_raw8` — the type is ShmSafe/POD, so viewing its 16
        // bytes as a byte array is sound and is precisely what memcpy transports.
        let raw = unsafe { core::mem::transmute::<FixedString<8>, [u8; 16]>(s) };
        assert_eq!(&raw[..8], b"hi\0\0\0\0\0\0", "buffer, NUL-terminated + zeroed tail");
        assert_eq!(&raw[8..], &2u64.to_ne_bytes(), "size_ at offset 8, native-endian");
    }

    #[test]
    fn is_copy_and_has_no_drop() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<FixedString<32>>();
        assert!(!std::mem::needs_drop::<FixedString<32>>());

        // Bitwise relocation is the contract: a copy is independent.
        let a = FixedString::<32>::from("original");
        let mut b = a;
        b.assign_str("changed");
        assert_eq!(a, "original");
        assert_eq!(b, "changed");
    }

    #[test]
    fn is_shm_safe() {
        fn assert_shm_safe<T: ShmSafe>() {}
        assert_shm_safe::<FixedString<32>>();
        assert_shm_safe::<FixedString<64>>();
    }

    // --- Construction / empty ---

    #[test]
    fn new_is_empty_and_zeroed() {
        let s = FixedString::<32>::new();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert!(s.is_valid());
        assert_eq!(s.as_bytes(), b"");
        assert_eq!(s.c_str(), c"");
        assert_eq!(s.as_str().unwrap(), "");
        // Divergence 3: the whole buffer is zeroed, not just data[0].
        assert_eq!(s.buffer(), &[0u8; 32]);
        assert_eq!(FixedString::<32>::default(), s);
    }

    #[test]
    fn empty_str_round_trips() {
        let s = FixedString::<32>::from("");
        assert!(s.is_empty());
        assert_eq!(s.as_str().unwrap(), "");
        assert_eq!(s.c_str(), c"");
    }

    #[test]
    fn str_round_trip() {
        let s = FixedString::<32>::from("hello world");
        assert_eq!(s.len(), 11);
        assert_eq!(s.as_str().unwrap(), "hello world");
        assert_eq!(s.to_string_lossy(), "hello world");
        assert_eq!(String::try_from(s).unwrap(), "hello world");
        assert_eq!(s.c_str(), c"hello world");
        assert_eq!(s.as_bytes(), b"hello world");
        assert_eq!(s.to_string(), "hello world");
        assert!(s.is_valid());
    }

    #[test]
    fn unicode_round_trips_when_it_fits() {
        let s = FixedString::<32>::from("héllo → wörld");
        assert_eq!(s.as_str().unwrap(), "héllo → wörld");
        assert_eq!(s.len(), "héllo → wörld".len());
    }

    #[test]
    fn from_c_str_uses_c_str_len() {
        let s = FixedString::<32>::from(c"abc");
        assert_eq!(s.len(), 3);
        assert_eq!(s, "abc");
    }

    #[test]
    fn from_bytes_and_string() {
        assert_eq!(FixedString::<32>::from(&b"bytes"[..]), "bytes");
        assert_eq!(FixedString::<32>::from(&String::from("owned")), "owned");
    }

    // --- c_str_len (C++ CStrLen) ---

    #[test]
    fn c_str_len_semantics() {
        assert_eq!(c_str_len(b"abc\0def"), 3);
        assert_eq!(c_str_len(b"\0abc"), 0);
        // No NUL: bounded by the slice (divergence 12); C++ would keep scanning.
        assert_eq!(c_str_len(b"abc"), 3);
        // The `nullptr` case: C++ returns 0, and so does the empty slice.
        assert_eq!(c_str_len(b""), 0);
    }

    // --- Capacity boundaries / truncation ---

    #[test]
    fn capacity_constants() {
        assert_eq!(FixedString::<32>::capacity(), 32);
        assert_eq!(FixedString::<32>::max_size(), 31);
        assert_eq!(FixedString::<32>::MAX_LEN, 31);
        assert_eq!(FixedString::<8>::MAX_LEN, 7);
        // N = 1 can hold only the empty string.
        assert_eq!(FixedString::<1>::MAX_LEN, 0);
        assert!(FixedString::<8>::reserve(8));
        assert!(!FixedString::<8>::reserve(9));
    }

    #[test]
    fn exact_fit_at_max_len_is_not_truncated() {
        // N = 8 => MAX_LEN = 7.
        let s = FixedString::<8>::from("abcdefg");
        assert_eq!(s.len(), 7);
        assert_eq!(s, "abcdefg");
        assert_eq!(s.c_str(), c"abcdefg");
        // The reserved terminator byte is the last one in the buffer.
        assert_eq!(s.buffer()[7], 0);
        assert!(s.is_valid());
    }

    #[test]
    fn one_past_max_len_truncates() {
        let s = FixedString::<8>::from("abcdefgh");
        assert_eq!(s.len(), 7);
        assert_eq!(s, "abcdefg");
        assert_eq!(s.buffer()[7], 0);
    }

    #[test]
    fn far_past_max_len_truncates() {
        let s = FixedString::<8>::from("abcdefghijklmnopqrstuvwxyz");
        assert_eq!(s.len(), FixedString::<8>::MAX_LEN);
        assert_eq!(s, "abcdefg");
        assert!(s.is_valid());
    }

    #[test]
    fn n_one_holds_only_the_empty_string() {
        let s = FixedString::<1>::from("anything");
        assert!(s.is_empty());
        assert_eq!(s.c_str(), c"");
        assert!(s.is_valid());
    }

    #[test]
    fn truncation_can_split_a_utf8_character() {
        // "aaaaaa" (6) + '€' (3 bytes) = 9 bytes; MAX_LEN = 7 keeps one stray byte.
        let s = FixedString::<8>::from("aaaaaa€");
        assert_eq!(s.len(), 7);
        // Faithful to C++: truncation is by byte, so the result is not valid UTF-8.
        assert!(s.as_str().is_err());
        assert_eq!(s.as_bytes(), b"aaaaaa\xe2");
        // The lossy path still works (divergence 6).
        assert_eq!(s.to_string_lossy(), "aaaaaa\u{fffd}");
    }

    #[test]
    fn assign_shrinking_scrubs_the_tail() {
        let mut s = FixedString::<8>::from("abcdefg");
        s.assign_str("xy");
        assert_eq!(s.len(), 2);
        assert_eq!(s, "xy");
        // Divergence 3: C++ would leave "xy\0defg"; the port zeroes the tail so
        // the memcpy'd image never carries stale bytes.
        assert_eq!(s.buffer(), b"xy\0\0\0\0\0\0");
    }

    // --- Embedded NUL ---

    #[test]
    fn embedded_nul_is_part_of_the_value() {
        let s = FixedString::<32>::from(&b"ab\0cd"[..]);
        // size()/view() count the NUL...
        assert_eq!(s.len(), 5);
        assert_eq!(s.as_bytes(), b"ab\0cd");
        assert_eq!(s.as_str().unwrap(), "ab\0cd");
        // ...but c_str() stops at the first one (faithful to C++).
        assert_eq!(s.c_str(), c"ab");
        assert_eq!(s.c_str().to_bytes().len(), 2);
        assert!(s.is_valid());
    }

    #[test]
    fn embedded_nul_comparison_overloads_differ() {
        let s = FixedString::<32>::from(&b"ab\0cd"[..]);
        // operator==(const std::string&) compares the full view => equal.
        assert_eq!(s, "ab\0cd");
        assert!(s == *"ab\0cd");
        let owned = String::from("ab\0cd");
        assert!(s == owned, "PartialEq<String> compares the full view too");
        // operator==(const char*) compares CStrLen(s) against size() first, so a
        // value with an embedded NUL never equals a C string: a C string ends at
        // its first NUL, so the longest one that could match is "ab", whose length
        // already differs from size(). (This is why the two C++ overloads disagree.)
        assert_eq!(c_str_len(s.as_bytes()), 2);
        assert_ne!(c_str_len(s.as_bytes()), s.len());
        assert!(s != *c"ab");
        assert!(s != "ab");
    }

    #[test]
    fn embedded_nul_at_the_end_is_kept() {
        let mut s = FixedString::<8>::new();
        s.assign(b"ab\0");
        assert_eq!(s.len(), 3);
        assert_eq!(s.as_bytes(), b"ab\0");
        assert_eq!(s.c_str(), c"ab");
    }

    // --- Mutators ---

    #[test]
    fn clear_resets_and_scrubs() {
        let mut s = FixedString::<8>::from("abcdefg");
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.c_str(), c"");
        assert_eq!(s.buffer(), &[0u8; 8]);
        assert!(s.is_valid());
    }

    #[test]
    fn push_back_fills_then_silently_drops() {
        let mut s = FixedString::<8>::new();
        for (i, c) in b"abcdefg".iter().enumerate() {
            assert!(s.push_back(*c), "byte {i} should fit");
            assert_eq!(s.len(), i + 1);
            // The terminator invariant holds after every push.
            assert!(s.is_valid());
            assert_eq!(s.c_str().to_bytes().len(), i + 1);
        }
        assert_eq!(s, "abcdefg");
        // Full: C++ silently drops (the bool is the additive divergence 9).
        assert!(!s.push_back(b'h'));
        assert_eq!(s.len(), 7);
        assert_eq!(s, "abcdefg");
        assert!(s.is_valid());
    }

    #[test]
    fn push_back_on_capacity_one_always_drops() {
        let mut s = FixedString::<1>::new();
        assert!(!s.push_back(b'x'));
        assert!(s.is_empty());
    }

    #[test]
    fn push_back_accepts_a_nul_byte() {
        let mut s = FixedString::<8>::from("ab");
        assert!(s.push_back(0));
        assert_eq!(s.len(), 3);
        assert_eq!(s.as_bytes(), b"ab\0");
        assert_eq!(s.c_str(), c"ab");
    }

    #[test]
    fn as_bytes_mut_edits_in_place() {
        let mut s = FixedString::<32>::from("abc");
        s.as_bytes_mut()[0] = b'X';
        assert_eq!(s, "Xbc");
        assert_eq!(s.len(), 3);
        assert!(s.is_valid());
    }

    #[test]
    fn assign_str_and_reassignment() {
        let mut s = FixedString::<32>::new();
        s.assign_str("first");
        assert_eq!(s, "first");
        s = FixedString::from("second");
        assert_eq!(s, "second");
        s.assign(b"third");
        assert_eq!(s, "third");
    }

    // --- Indexing / iteration ---

    #[test]
    fn index_reads_the_raw_buffer() {
        let s = FixedString::<8>::from("ab");
        assert_eq!(s[0], b'a');
        assert_eq!(s[1], b'b');
        // C++ indexes data_ raw, so past-the-end-but-in-buffer is legal.
        assert_eq!(s[2], 0);
        assert_eq!(s[7], 0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn index_past_the_buffer_panics() {
        let s = FixedString::<8>::from("ab");
        let _ = s[8]; // C++ is UB here; Rust panics (divergence: see Index docs).
    }

    #[test]
    fn iteration_covers_live_bytes_only() {
        let s = FixedString::<8>::from("abc");
        assert_eq!(s.iter().copied().collect::<Vec<_>>(), b"abc");
        assert_eq!((&s).into_iter().count(), 3);
    }

    // --- Comparison / hashing ---

    #[test]
    fn equality_ignores_the_buffer_tail() {
        let mut a = FixedString::<8>::from("abcdefg");
        a.assign_str("ab");
        let b = FixedString::<8>::from("ab");
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn tail_garbage_affects_neither_eq_nor_hash() {
        // Two forged values with identical live bytes but different tails — the
        // C++ compares/hashes only data_[0..size_], and so must the port.
        let clean = from_raw8([b'h', b'i', 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
        let dirty = from_raw8([b'h', b'i', 0, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(clean, dirty);
        assert_eq!(hash_of(&clean), hash_of(&dirty));
        assert_eq!(dirty, "hi");
    }

    #[test]
    fn inequality_by_length_and_by_content() {
        let a = FixedString::<32>::from("abc");
        assert!(a != FixedString::<32>::from("abcd"));
        assert!(a != FixedString::<32>::from("abd"));
        assert!(a != FixedString::<32>::from(""));
        assert_eq!(a, FixedString::<32>::from("abc"));
    }

    #[test]
    fn comparison_against_str_and_bytes() {
        let s = FixedString::<32>::from("abc");
        assert!(s == *"abc");
        assert!(s == "abc");
        assert!(*"abc" == s);
        assert!(s == b"abc"[..]);
        assert!(s == *c"abc");
        assert!(s != *"abcd");
    }

    #[test]
    fn usable_as_a_hash_map_key() {
        let mut m: HashMap<FixedString<32>, u32> = HashMap::new();
        m.insert(FixedString::from("alpha"), 1);
        m.insert(FixedString::from("beta"), 2);
        assert_eq!(m.get(&FixedString::from("alpha")), Some(&1));
        assert_eq!(m.get(&FixedString::from("gamma")), None);
        // Re-inserting an equal key overwrites rather than duplicating.
        m.insert(FixedString::from("alpha"), 9);
        assert_eq!(m.len(), 2);
        assert_eq!(m[&FixedString::<32>::from("alpha")], 9);
    }

    // --- Corrupt / hostile byte patterns (divergence 5) ---

    #[test]
    fn oversized_size_field_is_clamped_not_ub() {
        // size_ = 99 with N = 8: C++ len()/view() would read out of bounds.
        let mut raw = [0xFFu8; 16];
        raw[8..].copy_from_slice(&99u64.to_ne_bytes());
        let s = from_raw8(raw);
        assert_eq!(s.raw_size(), 99, "the stored field is reported verbatim");
        assert_eq!(s.len(), FixedString::<8>::MAX_LEN, "but len() clamps");
        assert_eq!(s.as_bytes().len(), 7, "and as_bytes() cannot overrun");
        assert!(!s.is_valid());
        assert!(!s.is_empty());
    }

    #[test]
    fn size_field_of_u64_max_is_clamped() {
        let mut raw = [b'z'; 16];
        raw[8..].copy_from_slice(&u64::MAX.to_ne_bytes());
        let s = from_raw8(raw);
        assert_eq!(s.len(), FixedString::<8>::MAX_LEN);
        assert_eq!(s.as_bytes(), b"zzzzzzz");
        assert!(!s.is_valid());
    }

    #[test]
    fn buffer_without_a_nul_yields_an_empty_c_str() {
        // Every buffer byte non-zero: C++ c_str() would run off the end.
        let mut raw = [b'x'; 16];
        raw[8..].copy_from_slice(&7u64.to_ne_bytes());
        let s = from_raw8(raw);
        assert_eq!(s.c_str(), c"", "no NUL in the buffer => empty, never an overrun");
        assert_eq!(s.as_bytes(), b"xxxxxxx", "the live bytes are still readable");
        assert!(!s.is_valid(), "the terminator invariant is broken");
    }

    #[test]
    fn valid_bytes_from_the_wire_are_accepted() {
        let raw = [b'o', b'k', 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
        let s = from_raw8(raw);
        assert!(s.is_valid());
        assert_eq!(s, "ok");
        assert_eq!(s.c_str(), c"ok");
    }

    // --- Serialization (C++ save/load) ---

    #[test]
    fn save_emits_the_cpp_archive_image() {
        let s = FixedString::<32>::from("hey");
        let mut buf = Vec::new();
        s.save_into(&mut buf);
        // save_string_fused(data, size): u64 length prefix, then size bytes, no NUL.
        assert_eq!(&buf[..8], &3u64.to_ne_bytes());
        assert_eq!(&buf[8..], b"hey");
        assert_eq!(buf.len(), s.serialized_size());
        // CalculateSizeArchive parity: sizeof(size_t) + len.
        assert_eq!(s.serialized_size(), 8 + 3);
    }

    #[test]
    fn save_load_round_trip() {
        for text in ["", "a", "hello world", "abcdefghijklmnopqrstuvwxyz012345"] {
            let s = FixedString::<32>::from(text);
            let mut buf = Vec::new();
            s.save_into(&mut buf);
            let (back, used) = FixedString::<32>::load_from(&buf).unwrap();
            assert_eq!(back, s, "round trip for {text:?}");
            assert_eq!(used, buf.len());
            assert!(back.is_valid());
        }
    }

    #[test]
    fn save_load_round_trip_preserves_embedded_nul() {
        let s = FixedString::<32>::from(&b"a\0b\0c"[..]);
        let mut buf = Vec::new();
        s.save_into(&mut buf);
        let (back, _) = FixedString::<32>::load_from(&buf).unwrap();
        assert_eq!(back.as_bytes(), b"a\0b\0c");
        assert_eq!(back, s);
    }

    #[test]
    fn load_consumes_exactly_its_record() {
        let a = FixedString::<32>::from("one");
        let b = FixedString::<32>::from("two!");
        let mut buf = Vec::new();
        a.save_into(&mut buf);
        b.save_into(&mut buf);

        let (got_a, used_a) = FixedString::<32>::load_from(&buf).unwrap();
        assert_eq!(got_a, a);
        assert_eq!(used_a, 11);
        let (got_b, used_b) = FixedString::<32>::load_from(&buf[used_a..]).unwrap();
        assert_eq!(got_b, b);
        assert_eq!(used_a + used_b, buf.len());
    }

    #[test]
    fn load_of_an_oversized_record_truncates_like_the_cpp_post_clamp() {
        // A stream claiming 20 bytes into an N = 8 buffer. C++ overflows the
        // inline buffer, then clamps size_ to kMaxLen (divergence 4); the port
        // reaches the same value without the overflow.
        let payload = b"abcdefghijklmnopqrst";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(payload.len() as u64).to_ne_bytes());
        buf.extend_from_slice(payload);

        let (s, used) = FixedString::<8>::load_from(&buf).unwrap();
        assert_eq!(s.len(), FixedString::<8>::MAX_LEN);
        assert_eq!(s, "abcdefg", "the leading bytes are kept, as after the C++ clamp");
        assert!(s.is_valid(), "load re-establishes the NUL terminator");
        assert_eq!(used, buf.len(), "the whole declared payload is consumed");
    }

    #[test]
    fn load_of_an_empty_record() {
        let buf = 0u64.to_ne_bytes();
        let (s, used) = FixedString::<32>::load_from(&buf).unwrap();
        assert!(s.is_empty());
        assert_eq!(used, 8);
    }

    #[test]
    fn load_rejects_a_short_buffer() {
        assert!(FixedString::<32>::load_from(&[]).is_none());
        assert!(FixedString::<32>::load_from(&[0u8; 7]).is_none());
        // Prefix says 5 bytes follow, but only 3 do.
        let mut buf = Vec::from(5u64.to_ne_bytes());
        buf.extend_from_slice(b"abc");
        assert!(FixedString::<32>::load_from(&buf).is_none());
    }

    #[test]
    fn load_rejects_a_length_that_would_overflow_the_cursor() {
        let buf = {
            let mut b = Vec::from(u64::MAX.to_ne_bytes());
            b.extend_from_slice(b"abc");
            b
        };
        // usize::MAX-ish declared length: no panic, no wraparound, just None.
        assert!(FixedString::<32>::load_from(&buf).is_none());
    }

    // --- Formatting ---

    #[test]
    fn display_and_debug() {
        let s = FixedString::<32>::from("text");
        assert_eq!(format!("{s}"), "text");
        let dbg = format!("{s:?}");
        assert!(dbg.contains("capacity: 32"), "{dbg}");
        assert!(dbg.contains("len: 4"), "{dbg}");
        assert!(dbg.contains("text"), "{dbg}");
    }

    #[test]
    fn display_is_lossy_for_invalid_utf8() {
        let s = FixedString::<8>::from(&b"a\xffb"[..]);
        assert!(s.as_str().is_err());
        assert_eq!(format!("{s}"), "a\u{fffd}b");
    }
}
