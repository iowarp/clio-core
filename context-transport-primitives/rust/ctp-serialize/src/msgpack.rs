// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Port of the **encoder** half of `clio_ctp/serialize/msgpack_wrapper.h`.
//!
//! The C++ header is a thin wrapper over the pure-C `msgpack-c` library
//! (`msgpack_sbuffer` + `msgpack_packer`). This module reimplements that
//! encoder in pure Rust — no external crate — emitting spec-correct MessagePack
//! bytes that are **byte-for-byte identical** to what `msgpack-c` produces for
//! the surface CTP actually uses (the `pack_map` / `pack(str)` / `pack(u32)` /
//! `pack(u64)` / `pack(f64)` / `pack(bool)` pattern in the bdev `Monitor` task,
//! `context-runtime/modules/bdev/src/bdev_runtime.cc:452`).
//!
//! Encoding rules were transcribed from the exact library this build links
//! (vcpkg `msgpack-c`: `include/msgpack/pack_template.h` and `pack.h`), not
//! from the spec alone, so that the C++ and Rust producers agree bit for bit
//! during the migration window when both may write the same wire records.
//!
//! C++ → Rust parity map
//! ---------------------
//! | C++ name                                   | Rust name                          |
//! |--------------------------------------------|------------------------------------|
//! | `msgpack::sbuffer`                         | [`Sbuffer`]                        |
//! | `sbuffer::data()` / `size()` / `clear()`   | [`Sbuffer::data`] / [`Sbuffer::size`] / [`Sbuffer::clear`] |
//! | `sbuffer::c_ptr()` (raw `msgpack_sbuffer*`)| *(omitted — no C library to feed)* |
//! | `msgpack::packer<Buffer>`                  | [`Packer<'_, W>`](Packer)          |
//! | `packer::packer(Buffer&)`                  | [`Packer::new`] (takes `&mut W`)    |
//! | `msgpack_packer::callback` (write fn ptr)  | [`PackWrite`] trait                |
//! | `packer::pack_array(size_t)`               | [`Packer::pack_array`]             |
//! | `packer::pack_map(size_t)`                 | [`Packer::pack_map`]               |
//! | `packer::pack_nil()`                       | [`Packer::pack_nil`]               |
//! | `packer::pack(T)` (11 overloads)           | [`Packer::pack`] + [`Pack`] trait  |
//! | `packer::pack(bool)`                       | `impl Pack for bool`               |
//! | `packer::pack(uint8_t .. uint64_t)`        | `impl Pack for u8 ..= u64`         |
//! | `packer::pack(int8_t .. int64_t)`          | `impl Pack for i8 ..= i64`         |
//! | `packer::pack(float)` / `pack(double)`     | `impl Pack for f32` / `f64`        |
//! | `packer::pack(const std::string&)`         | `impl Pack for &String` / `String`  |
//! | `packer::pack(const char*)` (via `strlen`) | `impl Pack for &str`               |
//! | `packer::pack(unsigned long)` / `pack(long)` (the `__APPLE__`/`_WIN32` LP64-vs-LLP64 disambiguation overloads) | `impl Pack for usize` / `isize` |
//! | `packer::pack_raw_msgpack(const char*, size_t)` | [`Packer::pack_raw_msgpack`]  |
//! | `packer::pack_raw_msgpack(const std::string&)`  | [`Packer::pack_raw_msgpack`] (same fn, `&[u8]`) |
//!
//! Wire-format fidelity notes
//! --------------------------
//! * **Integer encoding is value-driven, not width-driven.** `msgpack-c` has a
//!   separate macro per C width (`msgpack_pack_real_uint8` … `_uint64`,
//!   `_int8` … `_int64`), but each narrower macro is a strict narrowing of the
//!   64-bit one: the emitted bytes depend only on the *value*. So
//!   `pack(200u8)` and `pack(200u64)` both emit `cc c8`. This port funnels all
//!   unsigned widths through [`Packer::pack_uint_value`] and all signed widths
//!   through [`Packer::pack_int_value`], which reproduces every per-width macro
//!   exactly. (Verified case-by-case against `pack_template.h`.)
//! * **Non-negative signed values are emitted as *unsigned* msgpack types.**
//!   `msgpack_pack_int64(42)` emits fixint `2a` and `msgpack_pack_int64(300)`
//!   emits `cd 01 2c` (uint16), i.e. they decode as `POSITIVE_INTEGER`, not
//!   `NEGATIVE_INTEGER`. The existing C++ unit test pins this
//!   (`test_msgpack_wrapper.cc`: `PackUnpack<int64_t>(42)` →
//!   `type == POSITIVE_INTEGER`). [`Packer::pack_int_value`] delegates to the
//!   unsigned path for `d >= 0` to match.
//! * **`str8` (`0xd9`) is emitted for 32..=255 byte strings.** The C++ `msgpack`
//!   library gates `0xd9` behind `use_bin_type`, but the *C* `msgpack_packer`
//!   struct has no such field (`pack.h:38`) and its `msgpack_pack_str` emits
//!   `0xd9` unconditionally. Since the wrapper uses the C API, this port does
//!   the same. A decoder predating msgpack spec v5 would reject these bytes.
//! * Float/double use raw IEEE-754 bit reinterpretation (`to_bits`), matching
//!   the C union punning — NaN payloads and `-0.0` are preserved verbatim.
//!
//! Semantic divergences (explicit)
//! -------------------------------
//! 1. **Decoder not ported.** The C++ header also carries an unpacking surface
//!    — `msgpack::object`, `object_kv`, `object_array/map/str/bin`,
//!    `object_union`, `msgpack::type::*` constants, `object::convert<T>` (12
//!    specialisations), `object_handle`, and the free `msgpack::unpack()`. None
//!    of it is in this file; this module is encode-only, per the scope of the
//!    porting task. The C++ side must remain authoritative for unpacking until
//!    a `msgpack::unpack` port lands. Those `convert<T>` specialisations throw
//!    `std::runtime_error` on type mismatch, so their Rust port should return
//!    `Result`, not panic — noted here for whoever picks it up.
//! 2. **Borrow discipline replaces `Buffer&`.** C++ `packer` stores a raw
//!    `Buffer&` and lets you read `sbuf.data()` while the packer is still
//!    alive. [`Packer`] holds `&mut W`, so the buffer is unreadable until the
//!    packer's borrow ends. In practice NLL ends the borrow at the packer's
//!    last use, so the C++ call pattern transliterates unchanged; only
//!    genuinely interleaved read-while-packing needs an explicit scope.
//! 3. **No error codes.** Every `msgpack_pack_*` returns `int` (the callback's
//!    status; `msgpack_sbuffer_write` returns -1 on `malloc` failure) — and the
//!    C++ wrapper discards all of them, so a mid-record allocation failure is
//!    silently ignored and yields a truncated buffer. The Rust methods return
//!    `()`: allocation failure aborts the process (Rust's `Vec` OOM behaviour)
//!    rather than silently truncating. Strictly safer, never observable as a
//!    different byte stream.
//! 4. **Faithful `u32` truncation of container/string lengths retained.**
//!    `msgpack_pack_map` / `_array` / `_str` cast the `size_t` length to
//!    `uint32_t` with no range check, so on a 64-bit host a length >= 2^32
//!    silently wraps (e.g. `pack_map(2^32 + 5)` emits a header claiming 5
//!    entries). This port reproduces that wrap rather than panicking, to stay
//!    byte-identical (`map_len_truncates_above_u32_like_msgpack_c` pins it).
//!    Unreachable in practice — CTP's maps are tens of entries — but it is a
//!    latent upstream hazard, recorded here rather than silently "fixed".
//! 5. **`usize`/`isize` collapse the LP64/LLP64 overload split.** The C++ needs
//!    `#if defined(__APPLE__) || defined(_WIN32)` overloads on
//!    `unsigned long`/`long` with a `sizeof` dispatch, purely to resolve C++
//!    overload ambiguity across platforms. Because encoding is value-driven
//!    (see above), Rust just widens `usize`/`isize` to `u64`/`i64`; the emitted
//!    bytes are identical on both 64- and 32-bit hosts, and the conditional
//!    compilation disappears.
//! 6. **`Sbuffer` is `Clone`.** C++ `sbuffer` deletes its copy ctor (it owns a
//!    `malloc`'d pointer, so copying risks a double free) and is move-only.
//!    Rust's `Vec<u8>` clones safely, so the restriction has no purpose here.
//!    Move semantics (C++'s move ctor nulling `data`/`size`/`alloc`) are the
//!    language default.
//! 7. **`c_ptr()` accessors omitted.** `sbuffer::c_ptr()` exists only to hand a
//!    raw `msgpack_sbuffer*` to the C library; with no C library underneath,
//!    there is nothing to expose. [`Sbuffer::into_vec`] / [`Sbuffer::data`]
//!    cover the legitimate uses.
//! 8. **The arm-oabi double byte-swap is dropped.** `msgpack_pack_double` has a
//!    `defined(__arm__) && !(__ARM_EABI__)` branch swapping the two 32-bit
//!    halves for legacy ARM OABI mixed-endian doubles. CTP does not target ARM
//!    OABI (and Rust has no such target), so the port always emits plain
//!    big-endian IEEE-754.
//!
//! No `unsafe`, no `thread_local!` (per project rule) — this module is pure
//! logic over an owned `Vec<u8>`.

// ---- write sink (C++ `msgpack_packer::callback`) ---------------------------

/// Sink a [`Packer`] appends encoded bytes to.
///
/// Mirrors the `msgpack_packer_write` callback that C++ `packer` installs via
/// `msgpack_packer_init(&pk_, buf_.c_ptr(), msgpack_sbuffer_write)` — a trait
/// here instead of a function pointer, so the sink is chosen statically.
pub trait PackWrite {
    /// Append `bytes` verbatim to the sink.
    fn write_bytes(&mut self, bytes: &[u8]);
}

// ---- sbuffer ---------------------------------------------------------------

/// Growable output buffer — port of C++ `msgpack::sbuffer` (`msgpack_sbuffer`).
///
/// # Examples
/// ```
/// use ctp_serialize::msgpack::{Packer, Sbuffer};
///
/// let mut sbuf = Sbuffer::new();
/// let mut pk = Packer::new(&mut sbuf);
/// pk.pack_map(1);
/// pk.pack("iops");
/// pk.pack(128u32);
/// assert_eq!(sbuf.data(), &[0x81, 0xa4, b'i', b'o', b'p', b's', 0xcc, 0x80]);
/// ```
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Sbuffer {
    buf: Vec<u8>,
}

impl Sbuffer {
    /// C++ `sbuffer::sbuffer()` (`msgpack_sbuffer_init`).
    #[must_use]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Preallocate `cap` bytes. No C++ counterpart (`msgpack_sbuffer_init`
    /// starts empty and grows on demand); offered because CTP records have a
    /// predictable size and this avoids the growth reallocs.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    /// C++ `sbuffer::data()`.
    ///
    /// Returns an empty slice when nothing has been packed; the C++ returns a
    /// null `const char*` in that state, which callers pair with `size() == 0`.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.buf
    }

    /// C++ `sbuffer::size()`.
    #[must_use]
    pub fn size(&self) -> usize {
        self.buf.len()
    }

    /// True when nothing has been packed. (No C++ counterpart; idiomatic Rust.)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// C++ `sbuffer::clear()` (`msgpack_sbuffer_clear`): resets the length,
    /// retaining the allocation — matching `Vec::clear`.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Consume the buffer, yielding the encoded bytes.
    ///
    /// Replaces the C++ `std::string(sbuf.data(), sbuf.size())` copy that
    /// `bdev_runtime.cc` performs to hand the record off — here it is a move.
    #[must_use]
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }
}

impl PackWrite for Sbuffer {
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
}

impl PackWrite for Vec<u8> {
    fn write_bytes(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

// ---- packer ----------------------------------------------------------------

/// MessagePack encoder — port of C++ `msgpack::packer<Buffer>`.
///
/// Generic over the sink like the C++ template is over `Buffer`, but bound by
/// [`PackWrite`] rather than duck-typed on `c_ptr()`.
#[derive(Debug)]
pub struct Packer<'a, W: PackWrite> {
    buf: &'a mut W,
}

impl<'a, W: PackWrite> Packer<'a, W> {
    /// C++ `explicit packer(Buffer& buf)`.
    pub fn new(buf: &'a mut W) -> Self {
        Self { buf }
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.buf.write_bytes(bytes);
    }

    // -- containers ----------------------------------------------------------

    /// C++ `pack_array(size_t n)` → `msgpack_pack_array`.
    ///
    /// `n >= 2^32` wraps to `n as u32`, faithfully to msgpack-c (divergence 4).
    pub fn pack_array(&mut self, n: usize) {
        if n < 16 {
            self.write(&[0x90 | (n as u8)]);
        } else if n < 65536 {
            let mut b = [0xdc, 0, 0];
            b[1..].copy_from_slice(&(n as u16).to_be_bytes());
            self.write(&b);
        } else {
            let mut b = [0xdd, 0, 0, 0, 0];
            b[1..].copy_from_slice(&(n as u32).to_be_bytes());
            self.write(&b);
        }
    }

    /// C++ `pack_map(size_t n)` → `msgpack_pack_map`.
    ///
    /// `n >= 2^32` wraps to `n as u32`, faithfully to msgpack-c (divergence 4).
    pub fn pack_map(&mut self, n: usize) {
        if n < 16 {
            self.write(&[0x80 | (n as u8)]);
        } else if n < 65536 {
            let mut b = [0xde, 0, 0];
            b[1..].copy_from_slice(&(n as u16).to_be_bytes());
            self.write(&b);
        } else {
            let mut b = [0xdf, 0, 0, 0, 0];
            b[1..].copy_from_slice(&(n as u32).to_be_bytes());
            self.write(&b);
        }
    }

    /// C++ `pack_nil()` → `msgpack_pack_nil`.
    pub fn pack_nil(&mut self) {
        self.write(&[0xc0]);
    }

    // -- scalars -------------------------------------------------------------

    /// C++ `pack(T)`. Dispatches through the [`Pack`] trait, standing in for
    /// the header's overload set.
    pub fn pack<T: Pack>(&mut self, v: T) {
        v.pack_to(self);
    }

    /// Shared unsigned path — reproduces `msgpack_pack_real_uint8/16/32/64`
    /// (each is a narrowing of this; see the fidelity notes).
    pub fn pack_uint_value(&mut self, d: u64) {
        if d < (1 << 8) {
            if d < (1 << 7) {
                self.write(&[d as u8]); // positive fixint
            } else {
                self.write(&[0xcc, d as u8]); // uint 8
            }
        } else if d < (1 << 16) {
            let mut b = [0xcd, 0, 0];
            b[1..].copy_from_slice(&(d as u16).to_be_bytes());
            self.write(&b); // uint 16
        } else if d < (1 << 32) {
            let mut b = [0xce, 0, 0, 0, 0];
            b[1..].copy_from_slice(&(d as u32).to_be_bytes());
            self.write(&b); // uint 32
        } else {
            let mut b = [0xcf, 0, 0, 0, 0, 0, 0, 0, 0];
            b[1..].copy_from_slice(&d.to_be_bytes());
            self.write(&b); // uint 64
        }
    }

    /// Shared signed path — reproduces `msgpack_pack_real_int8/16/32/64`.
    ///
    /// Non-negative values route to [`Packer::pack_uint_value`], matching
    /// msgpack-c's choice to emit unsigned type tags for them.
    pub fn pack_int_value(&mut self, d: i64) {
        if d >= 0 {
            self.pack_uint_value(d as u64);
        } else if d >= -32 {
            self.write(&[d as u8]); // negative fixint (0xe0..=0xff)
        } else if d >= -128 {
            self.write(&[0xd0, d as u8]); // int 8
        } else if d >= -32768 {
            let mut b = [0xd1, 0, 0];
            b[1..].copy_from_slice(&(d as i16).to_be_bytes());
            self.write(&b); // int 16
        } else if d >= -(1 << 31) {
            let mut b = [0xd2, 0, 0, 0, 0];
            b[1..].copy_from_slice(&(d as i32).to_be_bytes());
            self.write(&b); // int 32
        } else {
            let mut b = [0xd3, 0, 0, 0, 0, 0, 0, 0, 0];
            b[1..].copy_from_slice(&d.to_be_bytes());
            self.write(&b); // int 64
        }
    }

    /// C++ `msgpack_pack_str(len)` — the string *header* only.
    ///
    /// `len >= 2^32` wraps to `len as u32`, faithfully to msgpack-c.
    pub fn pack_str_header(&mut self, len: usize) {
        if len < 32 {
            self.write(&[0xa0 | (len as u8)]); // fixstr
        } else if len < 256 {
            self.write(&[0xd9, len as u8]); // str 8
        } else if len < 65536 {
            let mut b = [0xda, 0, 0];
            b[1..].copy_from_slice(&(len as u16).to_be_bytes());
            self.write(&b); // str 16
        } else {
            let mut b = [0xdb, 0, 0, 0, 0];
            b[1..].copy_from_slice(&(len as u32).to_be_bytes());
            self.write(&b); // str 32
        }
    }

    /// C++ `msgpack_pack_str_body(b, l)` — raw string bytes, no header.
    pub fn pack_str_body(&mut self, body: &[u8]) {
        self.write(body);
    }

    /// C++ `pack_raw_msgpack(const char*, size_t)` / `(const std::string&)`:
    /// splice already-encoded msgpack through, bypassing the type tagging
    /// (the C++ calls `pk_.callback` directly).
    pub fn pack_raw_msgpack(&mut self, data: &[u8]) {
        self.write(data);
    }
}

// ---- Pack trait (C++ `packer::pack` overload set) ---------------------------

/// A value the C++ header has a `packer::pack()` overload for.
pub trait Pack {
    /// Emit `self` into `pk`.
    fn pack_to<W: PackWrite>(self, pk: &mut Packer<'_, W>);
}

impl Pack for bool {
    /// C++ `pack(bool)` → `msgpack_pack_true` / `msgpack_pack_false`.
    fn pack_to<W: PackWrite>(self, pk: &mut Packer<'_, W>) {
        pk.write(&[if self { 0xc3 } else { 0xc2 }]);
    }
}

/// `impl Pack` for the unsigned widths — all route through the shared
/// value-driven unsigned encoder (see the fidelity notes).
macro_rules! impl_pack_unsigned {
    ($($t:ty),* $(,)?) => {$(
        impl Pack for $t {
            fn pack_to<W: PackWrite>(self, pk: &mut Packer<'_, W>) {
                pk.pack_uint_value(self as u64);
            }
        }
    )*};
}
impl_pack_unsigned!(u8, u16, u32, u64, usize);

/// `impl Pack` for the signed widths — all route through the shared
/// value-driven signed encoder.
macro_rules! impl_pack_signed {
    ($($t:ty),* $(,)?) => {$(
        impl Pack for $t {
            fn pack_to<W: PackWrite>(self, pk: &mut Packer<'_, W>) {
                pk.pack_int_value(self as i64);
            }
        }
    )*};
}
impl_pack_signed!(i8, i16, i32, i64, isize);

impl Pack for f32 {
    /// C++ `pack(float)` → `msgpack_pack_float`: `0xca` + big-endian IEEE-754.
    fn pack_to<W: PackWrite>(self, pk: &mut Packer<'_, W>) {
        let mut b = [0xca, 0, 0, 0, 0];
        b[1..].copy_from_slice(&self.to_bits().to_be_bytes());
        pk.write(&b);
    }
}

impl Pack for f64 {
    /// C++ `pack(double)` → `msgpack_pack_double`: `0xcb` + big-endian IEEE-754.
    fn pack_to<W: PackWrite>(self, pk: &mut Packer<'_, W>) {
        let mut b = [0xcb, 0, 0, 0, 0, 0, 0, 0, 0];
        b[1..].copy_from_slice(&self.to_bits().to_be_bytes());
        pk.write(&b);
    }
}

impl Pack for &str {
    /// C++ `pack(const char*)` — header + body. Length is in **bytes**, as in
    /// C++ (`char_traits::length`); `str::len()` is likewise a byte count, so
    /// multi-byte UTF-8 agrees with the C++ automatically.
    fn pack_to<W: PackWrite>(self, pk: &mut Packer<'_, W>) {
        pk.pack_str_header(self.len());
        pk.pack_str_body(self.as_bytes());
    }
}

impl Pack for &String {
    /// C++ `pack(const std::string&)`.
    fn pack_to<W: PackWrite>(self, pk: &mut Packer<'_, W>) {
        self.as_str().pack_to(pk);
    }
}

impl Pack for String {
    /// Owned-value convenience; C++ takes `const std::string&`.
    fn pack_to<W: PackWrite>(self, pk: &mut Packer<'_, W>) {
        self.as_str().pack_to(pk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack one value and return the bytes.
    fn packed<T: Pack>(v: T) -> Vec<u8> {
        let mut sbuf = Sbuffer::new();
        Packer::new(&mut sbuf).pack(v);
        sbuf.into_vec()
    }

    // ---- sbuffer ----------------------------------------------------------

    #[test]
    fn empty_sbuffer_has_no_bytes() {
        let sbuf = Sbuffer::new();
        assert_eq!(sbuf.size(), 0);
        assert!(sbuf.is_empty());
        assert_eq!(sbuf.data(), &[] as &[u8]);
    }

    #[test]
    fn clear_resets_length_and_allows_reuse() {
        let mut sbuf = Sbuffer::new();
        Packer::new(&mut sbuf).pack(1u32);
        assert_eq!(sbuf.size(), 1);
        sbuf.clear();
        assert_eq!(sbuf.size(), 0);
        assert!(sbuf.is_empty());
        // C++ msgpack_sbuffer_clear keeps the buffer usable.
        Packer::new(&mut sbuf).pack(2u32);
        assert_eq!(sbuf.data(), &[0x02]);
    }

    #[test]
    fn packer_appends_across_separate_packer_instances() {
        // C++ constructs one packer per sbuffer, but nothing resets the buffer;
        // a second packer must continue where the first stopped.
        let mut sbuf = Sbuffer::new();
        Packer::new(&mut sbuf).pack(1u32);
        Packer::new(&mut sbuf).pack(2u32);
        assert_eq!(sbuf.data(), &[0x01, 0x02]);
    }

    #[test]
    fn vec_u8_is_a_usable_sink() {
        let mut v: Vec<u8> = Vec::new();
        Packer::new(&mut v).pack(true);
        assert_eq!(v, vec![0xc3]);
    }

    // ---- nil / bool -------------------------------------------------------

    #[test]
    fn nil_and_bools_match_spec() {
        let mut sbuf = Sbuffer::new();
        Packer::new(&mut sbuf).pack_nil();
        assert_eq!(sbuf.data(), &[0xc0]);
        assert_eq!(packed(true), vec![0xc3]);
        assert_eq!(packed(false), vec![0xc2]);
    }

    // ---- unsigned integers ------------------------------------------------

    #[test]
    fn unsigned_boundaries_match_spec() {
        // positive fixint: 0x00..=0x7f
        assert_eq!(packed(0u32), vec![0x00]);
        assert_eq!(packed(1u32), vec![0x01]);
        assert_eq!(packed(127u32), vec![0x7f]);
        // uint 8: 0xcc
        assert_eq!(packed(128u32), vec![0xcc, 0x80]);
        assert_eq!(packed(255u32), vec![0xcc, 0xff]);
        // uint 16: 0xcd
        assert_eq!(packed(256u32), vec![0xcd, 0x01, 0x00]);
        assert_eq!(packed(65535u32), vec![0xcd, 0xff, 0xff]);
        // uint 32: 0xce
        assert_eq!(packed(65536u32), vec![0xce, 0x00, 0x01, 0x00, 0x00]);
        assert_eq!(packed(u32::MAX), vec![0xce, 0xff, 0xff, 0xff, 0xff]);
        // uint 64: 0xcf
        assert_eq!(
            packed(4_294_967_296u64),
            vec![0xcf, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            packed(u64::MAX),
            vec![0xcf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]
        );
    }

    #[test]
    fn unsigned_encoding_depends_on_value_not_declared_width() {
        // msgpack-c's per-width macros are narrowings of the 64-bit one, so the
        // same value must encode identically regardless of the Rust type.
        assert_eq!(packed(200u8), vec![0xcc, 0xc8]);
        assert_eq!(packed(200u16), packed(200u8));
        assert_eq!(packed(200u32), packed(200u8));
        assert_eq!(packed(200u64), packed(200u8));
        assert_eq!(packed(200usize), packed(200u8));
        assert_eq!(packed(0u8), packed(0u64));
        assert_eq!(packed(u8::MAX), packed(255u64));
        assert_eq!(packed(u16::MAX), packed(65535u64));
    }

    #[test]
    fn cpp_unit_test_unsigned_vectors() {
        // The values the C++ test_msgpack_wrapper.cc round-trips.
        assert_eq!(packed(200u8), vec![0xcc, 0xc8]);
        assert_eq!(packed(60000u16), vec![0xcd, 0xea, 0x60]);
        assert_eq!(packed(4_000_000_000u32), vec![0xce, 0xee, 0x6b, 0x28, 0x00]);
        assert_eq!(
            packed(123_456_789_012_345u64),
            vec![0xcf, 0x00, 0x00, 0x70, 0x48, 0x86, 0x0d, 0xdf, 0x79]
        );
    }

    // ---- signed integers --------------------------------------------------

    #[test]
    fn negative_boundaries_match_spec() {
        // negative fixint: 0xe0..=0xff covers -32..=-1
        assert_eq!(packed(-1i64), vec![0xff]);
        assert_eq!(packed(-32i64), vec![0xe0]);
        // int 8: 0xd0
        assert_eq!(packed(-33i64), vec![0xd0, 0xdf]);
        assert_eq!(packed(-128i64), vec![0xd0, 0x80]);
        // int 16: 0xd1
        assert_eq!(packed(-129i64), vec![0xd1, 0xff, 0x7f]);
        assert_eq!(packed(-32768i64), vec![0xd1, 0x80, 0x00]);
        // int 32: 0xd2
        assert_eq!(packed(-32769i64), vec![0xd2, 0xff, 0xff, 0x7f, 0xff]);
        assert_eq!(
            packed(-2_147_483_648i64),
            vec![0xd2, 0x80, 0x00, 0x00, 0x00]
        );
        // int 64: 0xd3
        assert_eq!(
            packed(-2_147_483_649i64),
            vec![0xd3, 0xff, 0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff]
        );
        assert_eq!(
            packed(i64::MIN),
            vec![0xd3, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn non_negative_signed_values_emit_unsigned_tags() {
        // Pinned by the C++ test: PackUnpack<int64_t>(42) is POSITIVE_INTEGER.
        assert_eq!(packed(42i64), vec![0x2a]);
        assert_eq!(packed(0i64), vec![0x00]);
        assert_eq!(packed(127i64), vec![0x7f]);
        assert_eq!(packed(128i64), vec![0xcc, 0x80]); // uint8 tag, not int16
        assert_eq!(packed(300i64), vec![0xcd, 0x01, 0x2c]); // uint16 tag
        assert_eq!(packed(i64::MAX), packed(i64::MAX as u64)); // uint64 tag

        // Signed and unsigned agree for every non-negative value.
        let non_negative = [0i64, 1, 127, 128, 255, 256, 65535, 65536, 4_294_967_295];
        for v in non_negative {
            assert_eq!(packed(v), packed(v as u64), "mismatch at {v}");
        }
    }

    #[test]
    fn signed_encoding_depends_on_value_not_declared_width() {
        assert_eq!(packed(-100i8), vec![0xd0, 0x9c]);
        assert_eq!(packed(-100i16), packed(-100i8));
        assert_eq!(packed(-100i32), packed(-100i8));
        assert_eq!(packed(-100i64), packed(-100i8));
        assert_eq!(packed(-100isize), packed(-100i8));
        // Extremes of each narrower type.
        assert_eq!(packed(i8::MIN), vec![0xd0, 0x80]);
        assert_eq!(packed(i16::MIN), vec![0xd1, 0x80, 0x00]);
        assert_eq!(packed(i32::MIN), vec![0xd2, 0x80, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn cpp_unit_test_signed_vectors() {
        assert_eq!(packed(-100i8), vec![0xd0, 0x9c]);
        assert_eq!(packed(-30000i16), vec![0xd1, 0x8a, 0xd0]);
        assert_eq!(
            packed(-2_000_000_000i32),
            vec![0xd2, 0x88, 0xca, 0x6c, 0x00]
        );
        assert_eq!(
            packed(-123_456_789_012_345i64),
            vec![0xd3, 0xff, 0xff, 0x8f, 0xb7, 0x79, 0xf2, 0x20, 0x87]
        );
    }

    // ---- floats -----------------------------------------------------------

    #[test]
    fn floats_match_spec() {
        // 1.5f32 == 0x3fc00000
        assert_eq!(packed(1.5f32), vec![0xca, 0x3f, 0xc0, 0x00, 0x00]);
        assert_eq!(packed(0.0f32), vec![0xca, 0x00, 0x00, 0x00, 0x00]);
        // 1.5f64 == 0x3ff8000000000000
        assert_eq!(
            packed(1.5f64),
            vec![0xcb, 0x3f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            packed(0.0f64),
            vec![0xcb, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        // 1.0f64 == 0x3ff0000000000000
        assert_eq!(
            packed(1.0f64),
            vec![0xcb, 0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        // -2.0f64 == 0xc000000000000000
        assert_eq!(
            packed(-2.0f64),
            vec![0xcb, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn float_bit_punning_preserves_negative_zero_and_nan() {
        // Union punning in C keeps the sign bit of -0.0 (a value-based encoder
        // that normalised to 0.0 would silently differ from the C++).
        assert_eq!(
            packed(-0.0f64),
            vec![0xcb, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        assert_eq!(packed(-0.0f32), vec![0xca, 0x80, 0x00, 0x00, 0x00]);
        // Infinities.
        assert_eq!(
            packed(f64::INFINITY),
            vec![0xcb, 0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            packed(f64::NEG_INFINITY),
            vec![0xcb, 0xff, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        // NaN: tag plus the exact bit pattern, whatever this platform's is.
        let nan = packed(f64::NAN);
        assert_eq!(nan[0], 0xcb);
        assert_eq!(&nan[1..], &f64::NAN.to_bits().to_be_bytes());
    }

    // ---- strings ----------------------------------------------------------

    #[test]
    fn empty_string_is_a_bare_fixstr_header() {
        assert_eq!(packed(""), vec![0xa0]);
    }

    #[test]
    fn short_strings_use_fixstr() {
        assert_eq!(packed("a"), vec![0xa1, b'a']);
        assert_eq!(packed("score"), vec![0xa5, b's', b'c', b'o', b'r', b'e']);
    }

    #[test]
    fn string_length_boundaries_match_spec() {
        // 31 bytes: last fixstr (0xa0 | 31 == 0xbf)
        let s31 = "a".repeat(31);
        let out = packed(s31.as_str());
        assert_eq!(out[0], 0xbf);
        assert_eq!(out.len(), 32);

        // 32 bytes: first str8 (0xd9). Emitted unconditionally by the C API —
        // see the str8 fidelity note.
        let s32 = "a".repeat(32);
        let out = packed(s32.as_str());
        assert_eq!(&out[..2], &[0xd9, 0x20]);
        assert_eq!(out.len(), 34);

        // 255 bytes: last str8
        let s255 = "a".repeat(255);
        let out = packed(s255.as_str());
        assert_eq!(&out[..2], &[0xd9, 0xff]);
        assert_eq!(out.len(), 257);

        // 256 bytes: first str16 (0xda)
        let s256 = "a".repeat(256);
        let out = packed(s256.as_str());
        assert_eq!(&out[..3], &[0xda, 0x01, 0x00]);
        assert_eq!(out.len(), 259);

        // 65535 bytes: last str16
        let s = "a".repeat(65535);
        let out = packed(s.as_str());
        assert_eq!(&out[..3], &[0xda, 0xff, 0xff]);
        assert_eq!(out.len(), 65538);

        // 65536 bytes: first str32 (0xdb)
        let s = "a".repeat(65536);
        let out = packed(s.as_str());
        assert_eq!(&out[..5], &[0xdb, 0x00, 0x01, 0x00, 0x00]);
        assert_eq!(out.len(), 65541);
    }

    #[test]
    fn string_length_is_bytes_not_chars() {
        // C++ std::string::size() is a byte count; str::len() must agree.
        // "é" is 2 UTF-8 bytes -> fixstr of length 2, not 1.
        assert_eq!(packed("é"), vec![0xa2, 0xc3, 0xa9]);
        // 4-byte emoji.
        assert_eq!(packed("😀"), vec![0xa4, 0xf0, 0x9f, 0x98, 0x80]);
        // A 16-char string of 2-byte chars is 32 bytes -> crosses into str8.
        let s = "é".repeat(16);
        assert_eq!(s.len(), 32);
        assert_eq!(&packed(s.as_str())[..2], &[0xd9, 0x20]);
    }

    #[test]
    fn string_and_str_and_string_ref_agree() {
        let owned = String::from("pool_name");
        let expected = packed("pool_name");
        assert_eq!(packed(&owned), expected);
        assert_eq!(packed(owned), expected);
    }

    #[test]
    fn embedded_nul_is_packed_verbatim() {
        // Rust &str may contain NUL. The C++ pack(const std::string&) overload
        // is length-based and also keeps it (only pack(const char*), which uses
        // strlen, would truncate) -- so this matches the std::string overload
        // that bdev_runtime.cc actually calls.
        assert_eq!(packed("a\0b"), vec![0xa3, b'a', 0x00, b'b']);
    }

    // ---- containers -------------------------------------------------------

    fn map_header(n: usize) -> Vec<u8> {
        let mut sbuf = Sbuffer::new();
        Packer::new(&mut sbuf).pack_map(n);
        sbuf.into_vec()
    }

    fn array_header(n: usize) -> Vec<u8> {
        let mut sbuf = Sbuffer::new();
        Packer::new(&mut sbuf).pack_array(n);
        sbuf.into_vec()
    }

    #[test]
    fn map_header_boundaries_match_spec() {
        assert_eq!(map_header(0), vec![0x80]); // empty fixmap
        assert_eq!(map_header(1), vec![0x81]);
        assert_eq!(map_header(15), vec![0x8f]); // last fixmap
        assert_eq!(map_header(16), vec![0xde, 0x00, 0x10]); // first map16
        assert_eq!(map_header(65535), vec![0xde, 0xff, 0xff]); // last map16
        assert_eq!(map_header(65536), vec![0xdf, 0x00, 0x01, 0x00, 0x00]); // map32
    }

    #[test]
    fn array_header_boundaries_match_spec() {
        assert_eq!(array_header(0), vec![0x90]); // empty fixarray
        assert_eq!(array_header(15), vec![0x9f]); // last fixarray
        assert_eq!(array_header(16), vec![0xdc, 0x00, 0x10]); // first array16
        assert_eq!(array_header(65535), vec![0xdc, 0xff, 0xff]);
        assert_eq!(array_header(65536), vec![0xdd, 0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn map_len_truncates_above_u32_like_msgpack_c() {
        // Divergence 4: msgpack_pack_map casts size_t -> uint32_t unchecked, so
        // 2^32 + 5 emits a header claiming 5 entries. Reproduced deliberately
        // to stay byte-identical; unreachable in CTP (no allocation involved in
        // writing the header, so this is cheap to pin).
        assert_eq!(
            map_header((1usize << 32) + 5),
            vec![0xdf, 0x00, 0x00, 0x00, 0x05]
        );
        assert_eq!(
            array_header((1usize << 32) + 5),
            vec![0xdd, 0x00, 0x00, 0x00, 0x05]
        );
        // Exactly 2^32 wraps to a zero-length container.
        assert_eq!(map_header(1usize << 32), vec![0xdf, 0x00, 0x00, 0x00, 0x00]);
    }

    // ---- raw passthrough --------------------------------------------------

    #[test]
    fn pack_raw_msgpack_splices_bytes_verbatim() {
        let mut sbuf = Sbuffer::new();
        let mut pk = Packer::new(&mut sbuf);
        pk.pack_map(1);
        pk.pack("nested");
        // A pre-encoded {"k": 1} record.
        pk.pack_raw_msgpack(&[0x81, 0xa1, b'k', 0x01]);
        assert_eq!(
            sbuf.data(),
            &[0x81, 0xa6, b'n', b'e', b's', b't', b'e', b'd', 0x81, 0xa1, b'k', 0x01]
        );
    }

    #[test]
    fn pack_raw_msgpack_with_empty_input_is_a_noop() {
        let mut sbuf = Sbuffer::new();
        Packer::new(&mut sbuf).pack_raw_msgpack(&[]);
        assert_eq!(sbuf.size(), 0);
    }

    // ---- split header/body (C++ msgpack_pack_str + _str_body) -------------

    #[test]
    fn split_str_header_and_body_equals_pack() {
        let mut sbuf = Sbuffer::new();
        let mut pk = Packer::new(&mut sbuf);
        pk.pack_str_header(5);
        pk.pack_str_body(b"score");
        assert_eq!(sbuf.data(), packed("score").as_slice());
    }

    // ---- end-to-end: the bdev Monitor record ------------------------------

    #[test]
    fn bdev_monitor_shaped_record_matches_hand_computed_bytes() {
        // Mirrors bdev_runtime.cc:452 in miniature: pack_map(n) then
        // alternating str keys and typed values.
        let mut sbuf = Sbuffer::new();
        let mut pk = Packer::new(&mut sbuf);
        pk.pack_map(3);
        pk.pack("pool_name");
        pk.pack("nvme0");
        pk.pack("bdev_type");
        pk.pack(2u32);
        pk.pack("read_bandwidth_mbps");
        pk.pack(1.5f64);

        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            0x83,                                     // fixmap, 3 entries
            0xa9, b'p', b'o', b'o', b'l', b'_', b'n', b'a', b'm', b'e', // "pool_name"
            0xa5, b'n', b'v', b'm', b'e', b'0',       // "nvme0"
            0xa9, b'b', b'd', b'e', b'v', b'_', b't', b'y', b'p', b'e', // "bdev_type"
            0x02,                                     // 2 -> positive fixint
            0xb3, b'r', b'e', b'a', b'd', b'_', b'b', b'a', b'n', b'd', b'w',
            b'i', b'd', b't', b'h', b'_', b'm', b'b', b'p', b's', // 19 bytes -> 0xa0|19
            0xcb, 0x3f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1.5f64
        ];
        assert_eq!(sbuf.data(), expected.as_slice());
    }

    #[test]
    fn monitor_record_with_16_entries_uses_map16() {
        // bdev_runtime.cc packs pack_map(16) -- exactly the fixmap/map16 edge.
        let mut sbuf = Sbuffer::new();
        let mut pk = Packer::new(&mut sbuf);
        pk.pack_map(16);
        for _ in 0..16 {
            pk.pack("k");
            pk.pack(0u64);
        }
        assert_eq!(&sbuf.data()[..3], &[0xde, 0x00, 0x10]);
        assert_eq!(sbuf.size(), 3 + 16 * 3);
    }

    #[test]
    fn nested_maps_compose() {
        // The compressor_runtime.cc shape: pack_map(n) of str -> pack_map(4).
        let mut sbuf = Sbuffer::new();
        let mut pk = Packer::new(&mut sbuf);
        pk.pack_map(1);
        pk.pack("t0");
        pk.pack_map(2);
        pk.pack("score");
        pk.pack(0u32);
        pk.pack("written");
        pk.pack(true);
        assert_eq!(
            sbuf.data(),
            &[
                0x81, 0xa2, b't', b'0', 0x82, 0xa5, b's', b'c', b'o', b'r', b'e', 0x00, 0xa7, b'w',
                b'r', b'i', b't', b't', b'e', b'n', 0xc3,
            ]
        );
    }
}
