// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! FFI wrapper for **fpzip** (LLNL), the lossless/lossy floating-point array
//! compressor — the Rust face of C++ `ctp::Fpzip`
//! (`clio_ctp/compress/fpzip.h`).
//!
//! # Version / ABI assumptions
//!
//! Written against **fpzip 1.3.0** (`libfpzip.so.1.3.0`, the devcontainer's
//! `/usr/local/lib`), whose ABI has been stable across the 1.x line:
//!
//! * `FPZ` is a plain 6-`int` struct — `{type, prec, nx, ny, nz, nf}` — that
//!   the caller mutates directly (our `Fpz` mirrors it `#[repr(C)]`). This is
//!   the library's *public* handle type, so its layout is part of the ABI.
//! * `fpzip_errno` / `fpzip_errstr[]` are exported globals; `fpzip_errstr`
//!   has 8 entries (`fpzipSuccess` … `fpzipErrorInternal`).
//! * fpzip's core is C++, so the crate links `stdc++` (see `build.rs`); the
//!   entry points themselves are `extern "C"`.
//!
//! [`library_version`] checks the linked library at run time; the round-trip
//! tests assert it is 1.x so an ABI-incompatible bump fails loudly rather
//! than silently corrupting data.
//!
//! # Parameters and framing (matched to the C++)
//!
//! Framing is **byte-identical to `ctp::Fpzip`**, so blobs cross the C++/Rust
//! boundary in either direction: an fpzip header (`fpzip_write_header`)
//! followed by one `fpzip_write` field, with the payload declared as a 1-D
//! `FPZIP_TYPE_FLOAT` array (`nx = input_size / 4`, `ny = nz = nf = 1`).
//! Bytes are interpreted as native-endian `f32`, matching the C++
//! `reinterpret_cast<float*>`.
//!
//! `prec` comes from the [`Preset`], reproducing the *code* in the C++
//! `LibPressioWithModes::ConfigureFPZIP` (which is what the `"fpzip"` factory
//! row actually builds), not the stale doc comment above it — that comment
//! claims 8/16/20, the code sets 12/18/21. We follow the code:
//!
//! `prec` counts bits retained of the whole 32-bit word (sign + exponent = 9,
//! the rest mantissa), so the measured relative errors below — from
//! `lossy_round_trip_within_preset_tolerance` on a smooth 1-D field — are the
//! numbers to design against:
//!
//! | [`Preset`] | `prec` | mantissa bits kept | measured max rel. error | ratio |
//! |---|---|---|---|---|
//! | `Fast` | 12 | 3 | 1.0e-1 | ~180x |
//! | `Balanced` | 18 | 9 | 1.7e-3 | ~16x |
//! | `Best` | 21 | 12 | 2.1e-4 | ~6x |
//!
//! (Ratios are for that smooth field and are *not* representative of noisy
//! data; `Fast` in particular is far too lossy for anything but previewing.)
//!
//! All three presets are therefore **lossy** (consistent with
//! `LibraryId::Fpzip.is_lossy()`). fpzip's *lossless* mode — the default of
//! the standalone C++ `ctp::Fpzip(int precision = 0)` class — is reachable
//! via [`FpzipCompressor::with_precision`]`(0)`, which round-trips bit-exactly.
//!
//! Note fpzip is designed to exploit 2-D/3-D spatial structure and its own
//! docs warn it "may not perform well on 1D data". The C++ wrapper flattens
//! everything to 1-D anyway, and we inherit that deliberately: the shape is
//! part of the bitstream, so changing it would break compatibility with blobs
//! the C++ stack has already written.
//!
//! # Divergences from the C++ wrapper
//!
//! 1. **Decompressed size comes from the stream header, not the caller.**
//!    C++ `Decompress` derives `num_floats` from the *caller's* `output_size`
//!    and leaves it unchanged; it therefore silently mis-decodes when the
//!    caller's guess disagrees with the stream. We read the header, validate
//!    `type`/`prec`/dims, require `nx*ny*nz*nf*4 <= output.len()`, and return
//!    the true byte count. `output` is a capacity here, per the trait.
//! 2. **Errors carry fpzip's reason** (`fpzip_errstr[fpzip_errno]`) instead of
//!    being printed to `std::cerr` and reduced to `false`.
//! 3. **Empty input is not an error**: `compress(&[])` → `Ok(0)` and
//!    `decompress(&[])` → `Ok(0)`. The C++ would hand fpzip `nx = 0`, which
//!    fpzip rejects. The empty encoding is empty, so this round-trips.
//! 4. **Unaligned buffers are handled.** C++ `reinterpret_cast<float*>`es the
//!    caller's `void*`; we copy through an aligned scratch when a buffer is
//!    not 4-byte aligned, so no unaligned `f32` access reaches the library.
//! 5. **Decompression input is copied into a padded scratch.**
//!    `fpzip_read_from_buffer` takes *no length* — the C++ hands it a raw
//!    pointer and trusts the stream, so a corrupt/truncated stream reads off
//!    the end of the caller's buffer. See the safety note below.
//!
//! # Safety of `decompress` on untrusted input
//!
//! `fpzip_read_from_buffer(const void*)` has no length parameter, so fpzip
//! cannot bound its own reads: only the bytes the *stream* claims stop it.
//! To keep the safe surface sound, [`FpzipCompressor::decompress`]:
//!
//! * copies `input` into a scratch buffer padded with zeros out to
//!   `read_scratch_len` — an over-estimate of the largest stream that could
//!   decode into `output` — so a lying header over-reads into *our* zeroed
//!   padding rather than past the allocation, and
//! * validates the header's element count against `output`'s capacity
//!   *before* `fpzip_read`, which bounds the write side exactly.
//!
//! Corrupt input can thus yield garbage floats or an error, but not UB.
//!
//! # Concurrency
//!
//! [`FpzipCompressor`] is stateless (a `prec` value) and every call owns its
//! own `FPZ` stream, so concurrent use is safe. `fpzip_errno` is a process
//! global, however: under concurrent *failures* the reported reason may come
//! from another thread's error. Only the message is affected — never the
//! success/failure verdict, which comes from each call's own return value.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

use crate::{CompressError, Compressor, LibraryId, Preset};

// ---------------------------------------------------------------------------
// Raw FFI — hand-rolled from fpzip.h (fpzip 1.3.0)
// ---------------------------------------------------------------------------

/// `FPZ` — fpzip's public array metadata + stream handle.
///
/// Layout is fpzip's, verbatim; callers set the fields directly (that is how
/// fpzip's API is meant to be driven). Allocated and freed by fpzip, never by
/// us, so this is only ever seen behind a pointer.
#[repr(C)]
struct Fpz {
    /// `FPZIP_TYPE_FLOAT` (0) or `FPZIP_TYPE_DOUBLE` (1).
    type_: c_int,
    /// Bits of precision to retain; 0 = lossless.
    prec: c_int,
    nx: c_int,
    ny: c_int,
    nz: c_int,
    nf: c_int,
}

/// `FPZIP_TYPE_FLOAT` — single precision. CTP only ever writes floats.
const FPZIP_TYPE_FLOAT: c_int = 0;

/// Number of entries in `fpzip_errstr[]` (`fpzipSuccess` … `fpzipErrorInternal`).
const FPZIP_ERRSTR_LEN: usize = 8;

extern "C" {
    // -- write side --
    fn fpzip_write_to_buffer(buffer: *mut c_void, size: usize) -> *mut Fpz;
    fn fpzip_write_header(fpz: *mut Fpz) -> c_int;
    fn fpzip_write(fpz: *mut Fpz, data: *const c_void) -> usize;
    fn fpzip_write_close(fpz: *mut Fpz);

    // -- read side --
    fn fpzip_read_from_buffer(buffer: *const c_void) -> *mut Fpz;
    fn fpzip_read_header(fpz: *mut Fpz) -> c_int;
    fn fpzip_read(fpz: *mut Fpz, data: *mut c_void) -> usize;
    fn fpzip_read_close(fpz: *mut Fpz);

    // -- globals --
    /// `fpzipError fpzip_errno` — set by the call that failed.
    static fpzip_errno: c_int;
    /// `const char* const fpzip_errstr[]` — message per error code.
    static fpzip_errstr: [*const c_char; FPZIP_ERRSTR_LEN];
    /// `const unsigned int fpzip_library_version` — `(major << 8) | (minor << 4) | patch`.
    static fpzip_library_version: c_uint;
}

/// Version of the *linked* libfpzip, as `(major, minor, patch)`.
pub fn library_version() -> (u32, u32, u32) {
    // SAFETY: `fpzip_library_version` is a `const unsigned int` exported by
    // libfpzip; reading it is a plain load of an immutable global.
    let v = unsafe { fpzip_library_version };
    ((v >> 8) & 0xf, (v >> 4) & 0xf, v & 0xf)
}

/// Build a [`CompressError`] from `fpzip_errno`, naming the call that failed.
fn last_error(what: &str) -> CompressError {
    // SAFETY: `fpzip_errno` is an `int`-sized global exported by libfpzip.
    // Racy only in the benign sense documented in the module docs (a
    // concurrent failure may overwrite the reason; the load itself is sound).
    let code = unsafe { fpzip_errno };
    let msg = match usize::try_from(code) {
        Ok(i) if i < FPZIP_ERRSTR_LEN => {
            // SAFETY: `fpzip_errstr` has FPZIP_ERRSTR_LEN entries (bounds
            // checked above), each a static NUL-terminated string literal in
            // libfpzip's rodata — valid for the process lifetime.
            unsafe { CStr::from_ptr(fpzip_errstr[i]) }
                .to_string_lossy()
                .into_owned()
        }
        _ => format!("unknown error code {code}"),
    };
    CompressError(format!("fpzip: {what}: {msg}"))
}

// ---------------------------------------------------------------------------
// RAII stream wrappers — an FPZ is a resource; close it on every path
// ---------------------------------------------------------------------------

/// Owns an `FPZ` from `fpzip_write_to_buffer`; closes it on drop.
struct WriteStream(*mut Fpz);

impl WriteStream {
    /// # Safety
    /// `buffer` must be valid for writes of `size` bytes for as long as the
    /// returned stream lives.
    unsafe fn to_buffer(buffer: *mut c_void, size: usize) -> Result<Self, CompressError> {
        // SAFETY: caller guarantees `buffer` is writable for `size` bytes.
        let fpz = unsafe { fpzip_write_to_buffer(buffer, size) };
        if fpz.is_null() {
            return Err(last_error("fpzip_write_to_buffer"));
        }
        Ok(Self(fpz))
    }

    fn fpz(&self) -> *mut Fpz {
        self.0
    }
}

impl Drop for WriteStream {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a non-null FPZ from fpzip_write_to_buffer, not
        // yet closed (ownership is unique and Drop runs once).
        unsafe { fpzip_write_close(self.0) };
    }
}

/// Owns an `FPZ` from `fpzip_read_from_buffer`; closes it on drop.
struct ReadStream(*mut Fpz);

impl ReadStream {
    /// # Safety
    /// `buffer` must be valid for reads of every byte the compressed stream
    /// claims — fpzip takes no length. Callers pass a padded scratch (see
    /// [`read_scratch_len`]).
    unsafe fn from_buffer(buffer: *const c_void) -> Result<Self, CompressError> {
        // SAFETY: caller guarantees `buffer` is readable for the stream's extent.
        let fpz = unsafe { fpzip_read_from_buffer(buffer) };
        if fpz.is_null() {
            return Err(last_error("fpzip_read_from_buffer"));
        }
        Ok(Self(fpz))
    }

    fn fpz(&self) -> *mut Fpz {
        self.0
    }
}

impl Drop for ReadStream {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a non-null FPZ from fpzip_read_from_buffer, not
        // yet closed (ownership is unique and Drop runs once).
        unsafe { fpzip_read_close(self.0) };
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SIZEOF_F32: usize = std::mem::size_of::<f32>();

/// Native-endian decode of `bytes` into owned floats — the aligned fallback
/// for buffers we may not reinterpret in place.
fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(SIZEOF_F32)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Whether `p` may be reinterpreted as `*const f32` without an unaligned access.
fn is_f32_aligned(p: *const u8) -> bool {
    p.cast::<f32>().is_aligned()
}

/// Size of the zero-padded scratch `decompress` copies its input into.
///
/// `fpzip_read_from_buffer` has no length argument, so fpzip reads wherever
/// the stream tells it to. This bounds the damage: it is an over-estimate of
/// the largest stream that could legitimately decode into `output_len` bytes
/// (fpzip does not meaningfully expand data — worst case is a shade over the
/// raw size — plus generous slack for the header and any lying stream), so a
/// corrupt header runs into our own zeroed padding instead of off the end of
/// the allocation.
fn read_scratch_len(input_len: usize, output_len: usize) -> usize {
    let worst_case = output_len
        .saturating_add(output_len / 4)
        .saturating_add(4096);
    input_len.max(worst_case).saturating_add(4096)
}

/// `prec` for a preset — mirrors C++ `LibPressioWithModes::ConfigureFPZIP`.
const fn precision_for(preset: Preset) -> c_int {
    match preset {
        Preset::Fast => 12,
        Preset::Balanced => 18,
        Preset::Best => 21,
    }
}

// ---------------------------------------------------------------------------
// FpzipCompressor
// ---------------------------------------------------------------------------

/// fpzip codec over 1-D `f32` arrays — the Rust face of C++ `ctp::Fpzip`.
///
/// Construct with [`FpzipCompressor::new`] (preset-driven, lossy) or
/// [`FpzipCompressor::with_precision`] (explicit `prec`; `0` = lossless).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FpzipCompressor {
    /// fpzip `FPZ::prec`: 0 = lossless, 2..=32 = lossy (float range).
    prec: c_int,
}

impl FpzipCompressor {
    /// Codec at `preset` (C++ `CompressFactory::Get("fpzip", preset)`).
    ///
    /// All presets are lossy; see the module docs for the `prec` table.
    pub fn new(preset: Preset) -> Self {
        Self {
            prec: precision_for(preset),
        }
    }

    /// Codec with an explicit fpzip precision (C++ `Fpzip::SetPrecision`).
    ///
    /// `0` selects lossless mode — the standalone C++ class's default — which
    /// round-trips bit-exactly. For floats fpzip accepts `2..=32`; `32`
    /// retains every bit and is likewise lossless. Other values are rejected
    /// here rather than deep inside the library.
    pub fn with_precision(prec: i32) -> Result<Self, CompressError> {
        if prec == 0 || (2..=32).contains(&prec) {
            Ok(Self { prec })
        } else {
            Err(CompressError(format!(
                "fpzip: precision {prec} out of range (0 = lossless, or 2..=32 for f32)"
            )))
        }
    }

    /// Current fpzip precision (C++ `Fpzip::GetPrecision`).
    pub fn precision(&self) -> i32 {
        self.prec
    }

    /// Whether this configuration is bit-exact (C++ `Fpzip::IsLossless`).
    ///
    /// Note C++ tests only `precision == 0`; `prec == 32` retains all 32 bits
    /// of an `f32` and is lossless too, so it counts here.
    pub fn is_lossless(&self) -> bool {
        self.prec == 0 || self.prec == 32
    }
}

impl Compressor for FpzipCompressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        // Divergence 3: the empty encoding is empty (C++ would pass nx = 0).
        if input.is_empty() {
            return Ok(0);
        }
        if !input.len().is_multiple_of(SIZEOF_F32) {
            return Err(CompressError(format!(
                "fpzip: input size {} is not a multiple of sizeof(f32)",
                input.len()
            )));
        }
        let n = input.len() / SIZEOF_F32;
        // FPZ::nx is an int; refuse rather than truncate the dimension.
        if n > c_int::MAX as usize {
            return Err(CompressError(format!(
                "fpzip: {n} floats exceeds the int range of FPZ::nx"
            )));
        }
        if output.is_empty() {
            return Err(CompressError(
                "fpzip: output buffer is empty (needs room for at least a header)".into(),
            ));
        }

        // Divergence 4: fpzip dereferences `data` as `const float*`, so it
        // must be 4-byte aligned. Reinterpret in place when we can, else copy.
        let owned_in: Vec<f32>;
        let src: &[f32] = if is_f32_aligned(input.as_ptr()) {
            // SAFETY: `input.len()` is a multiple of 4 (checked) and the
            // pointer is 4-byte aligned (checked), so the same bytes form `n`
            // f32 values. Every 32-bit pattern is a valid f32 (NaNs included),
            // so there is no invalid-value UB, and the borrow of `input` keeps
            // the memory live and immutable for the slice's lifetime.
            unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<f32>(), n) }
        } else {
            owned_in = decode_f32(input);
            &owned_in
        };

        // SAFETY: `output` is exclusively borrowed and valid for writes of
        // `output.len()` bytes, and the stream (with it) is dropped before
        // this function returns. fpzip tracks the limit internally and fails
        // with fpzipErrorBufferOverflow rather than overrunning — asserted by
        // `compress_into_too_small_output_errs_without_corrupting_memory`.
        let stream = unsafe { WriteStream::to_buffer(output.as_mut_ptr().cast(), output.len()) }?;

        // SAFETY: `fpz` is fpzip's live, non-null FPZ, owned by `stream` for
        // this scope; setting its public fields is exactly how fpzip's API is
        // driven (cf. C++ `ctp::Fpzip::Compress`). Framing matches the C++:
        // 1-D float array, one field.
        unsafe {
            let fpz = stream.fpz();
            (*fpz).type_ = FPZIP_TYPE_FLOAT;
            (*fpz).prec = self.prec;
            (*fpz).nx = n as c_int;
            (*fpz).ny = 1;
            (*fpz).nz = 1;
            (*fpz).nf = 1;
        }

        // SAFETY: live FPZ owned by `stream`; writes go to `output`.
        if unsafe { fpzip_write_header(stream.fpz()) } == 0 {
            return Err(last_error("fpzip_write_header"));
        }

        // SAFETY: live FPZ owned by `stream`; `src` is 4-byte aligned and
        // holds exactly nx*ny*nz*nf = n floats, which is what fpzip reads.
        let written = unsafe { fpzip_write(stream.fpz(), src.as_ptr().cast()) };
        if written == 0 {
            return Err(last_error("fpzip_write"));
        }
        // Defensive: a `written > capacity` would mean fpzip had already
        // overrun `output`. Cannot happen (fpzip bounds-checks); if the ABI
        // ever changed under us, fail loudly instead of reporting a bogus len.
        if written > output.len() {
            return Err(CompressError(format!(
                "fpzip: reported {written} bytes written into a {}-byte buffer",
                output.len()
            )));
        }
        Ok(written)
    }

    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        // Pairs with `compress`'s empty encoding (divergence 3).
        if input.is_empty() {
            return Ok(0);
        }
        if !output.len().is_multiple_of(SIZEOF_F32) {
            return Err(CompressError(format!(
                "fpzip: output size {} is not a multiple of sizeof(f32)",
                output.len()
            )));
        }
        let cap_floats = output.len() / SIZEOF_F32;

        // Divergence 5: fpzip_read_from_buffer takes no length, so give it a
        // zero-padded copy — a lying stream over-reads into our padding.
        let mut scratch = vec![0u8; read_scratch_len(input.len(), output.len())];
        scratch[..input.len()].copy_from_slice(input);

        // SAFETY: `scratch` is live for this scope and readable for
        // `read_scratch_len(..)` bytes — an over-estimate of any stream that
        // could decode into `output` — so fpzip's unbounded reads stay inside
        // the allocation. `stream` is dropped before `scratch`.
        let stream = unsafe { ReadStream::from_buffer(scratch.as_ptr().cast()) }?;

        // SAFETY: live FPZ owned by `stream`.
        if unsafe { fpzip_read_header(stream.fpz()) } == 0 {
            return Err(last_error("fpzip_read_header"));
        }

        // SAFETY: live FPZ owned by `stream`; fpzip_read_header populated it.
        let (ty, prec, nx, ny, nz, nf) = unsafe {
            let f = stream.fpz();
            ((*f).type_, (*f).prec, (*f).nx, (*f).ny, (*f).nz, (*f).nf)
        };

        if ty != FPZIP_TYPE_FLOAT {
            return Err(CompressError(format!(
                "fpzip: stream holds doubles (type {ty}); CTP frames f32 only"
            )));
        }
        // Divergence 1: derive the count from the header and bound it against
        // the caller's capacity BEFORE fpzip writes anything into `output`.
        let mut n: usize = 1;
        for d in [nx, ny, nz, nf] {
            let d = usize::try_from(d)
                .ok()
                .filter(|d| *d >= 1)
                .ok_or_else(|| CompressError(format!("fpzip: bad header dimension {d}")))?;
            n = n
                .checked_mul(d)
                .ok_or_else(|| CompressError("fpzip: header element count overflows".into()))?;
        }
        if n > cap_floats {
            return Err(CompressError(format!(
                "fpzip: stream holds {n} floats ({} bytes) but output capacity is {} bytes",
                n * SIZEOF_F32,
                output.len()
            )));
        }
        // fpzip returns truncated bits as zeros, so a prec mismatch is not
        // fatal — but the stream's own prec governs decoding, and a value out
        // of range means the header is not one of ours.
        if prec != 0 && !(2..=32).contains(&prec) {
            return Err(CompressError(format!(
                "fpzip: header precision {prec} out of range"
            )));
        }

        // Divergence 4 again: fpzip writes through a `float*`, so decode
        // straight into `output` when it is aligned and via a scratch when
        // not. In both arms fpzip writes exactly nx*ny*nz*nf = n floats — the
        // header values validated above — into room for at least that many.
        let n_bytes = n * SIZEOF_F32;
        if is_f32_aligned(output.as_ptr()) {
            // SAFETY: `output` is 4-byte aligned (checked) and valid for
            // writes of n f32, since n <= cap_floats == output.len() / 4.
            let consumed = unsafe { fpzip_read(stream.fpz(), output.as_mut_ptr().cast()) };
            if consumed == 0 {
                return Err(last_error("fpzip_read"));
            }
        } else {
            let mut owned_out = vec![0.0f32; n];
            // SAFETY: `owned_out` is a Vec<f32> — aligned by construction —
            // holding exactly n elements, which is what fpzip writes.
            let consumed = unsafe { fpzip_read(stream.fpz(), owned_out.as_mut_ptr().cast()) };
            if consumed == 0 {
                return Err(last_error("fpzip_read"));
            }
            for (chunk, f) in output[..n_bytes].chunks_exact_mut(SIZEOF_F32).zip(&owned_out) {
                chunk.copy_from_slice(&f.to_ne_bytes());
            }
        }
        Ok(n_bytes)
    }

    fn library(&self) -> LibraryId {
        LibraryId::Fpzip
    }

    /// fpzip does not meaningfully expand data, but the bound must cover the
    /// pathological case (incompressible random floats) plus the header, so
    /// `compress` never fails for want of room. Slightly roomier than the
    /// trait's default; `compress_bound_covers_random_floats` pins it down.
    fn max_compressed_size(&self, input_size: usize) -> usize {
        input_size + input_size / 4 + 1024
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic, smooth-ish float field — the kind of data fpzip is for.
    fn realistic_floats(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = i as f32 * 0.01;
                100.0 + 25.0 * (x * 0.5).sin() + 0.5 * (x * 3.0).cos()
            })
            .collect()
    }

    fn as_bytes(v: &[f32]) -> &[u8] {
        // SAFETY: f32 has no padding; reinterpreting n floats as n*4 bytes is
        // sound and matches the native-endian framing the codec expects.
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
    }

    fn round_trip(c: &FpzipCompressor, data: &[f32]) -> Vec<f32> {
        let input = as_bytes(data);
        let mut buf = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut buf, input).expect("compress");
        let mut out = vec![0u8; input.len()];
        let got = c.decompress(&mut out, &buf[..n]).expect("decompress");
        assert_eq!(got, input.len(), "decompressed byte count");
        decode_f32(&out)
    }

    /// The linked libfpzip must be the 1.x ABI these decls were written for.
    #[test]
    fn linked_library_is_fpzip_1x() {
        let (major, minor, patch) = library_version();
        assert_eq!(major, 1, "fpzip major version (ABI) changed: {major}.{minor}.{patch}");
    }

    /// (1) Round-trip over realistic data, within each preset's error bound.
    #[test]
    fn lossy_round_trip_within_preset_tolerance() {
        let data = realistic_floats(4096);
        // Max relative error per preset (see the module's `prec` table).
        for (preset, tol) in [
            (Preset::Fast, 1.5e-1_f32),
            (Preset::Balanced, 5e-3),
            (Preset::Best, 1e-3),
        ] {
            let c = FpzipCompressor::new(preset);
            let out = round_trip(&c, &data);
            assert_eq!(out.len(), data.len());
            let worst = data
                .iter()
                .zip(&out)
                .map(|(a, b)| ((a - b) / a).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst <= tol,
                "{}: relative error {worst:e} exceeds {tol:e}",
                preset.name()
            );
        }
    }

    /// The lossless configuration (C++ `ctp::Fpzip`'s default) is bit-exact —
    /// including for values a tolerance test would wave through (NaN, ±inf).
    #[test]
    fn lossless_precision_round_trips_bit_exactly() {
        let mut data = realistic_floats(1024);
        data.extend_from_slice(&[0.0, -0.0, f32::MIN_POSITIVE, f32::MAX, f32::INFINITY]);
        for prec in [0, 32] {
            let c = FpzipCompressor::with_precision(prec).unwrap();
            assert!(c.is_lossless());
            let out = round_trip(&c, &data);
            assert_eq!(
                out.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                data.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "prec {prec} must be bit-exact"
            );
        }
    }

    /// (2) Highly-compressible data actually shrinks.
    #[test]
    fn compressible_data_shrinks() {
        let c = FpzipCompressor::new(Preset::Balanced);
        // A smooth ramp: fpzip's predictor should nail this.
        let data: Vec<f32> = (0..8192).map(|i| i as f32 * 0.25).collect();
        let input = as_bytes(&data);
        let mut buf = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut buf, input).unwrap();
        assert!(
            n < input.len() / 2,
            "expected >2x on a ramp, got {n} from {}",
            input.len()
        );
        // …and it must still decode back within tolerance.
        let out = round_trip(&c, &data);
        for (a, b) in data.iter().zip(&out) {
            assert!((a - b).abs() <= a.abs() * 5e-3 + 1e-3, "{a} vs {b}");
        }
    }

    /// (3) A too-small output buffer errors — and does not scribble past the
    /// capacity it was handed. The canary is the real assertion here: fpzip
    /// bounds-checks (fpzipErrorBufferOverflow) rather than overrunning, and
    /// `compress`'s soundness argument depends on exactly that.
    #[test]
    fn compress_into_too_small_output_errs_without_corrupting_memory() {
        let c = FpzipCompressor::new(Preset::Balanced);
        let data = realistic_floats(4096);
        let input = as_bytes(&data);

        const CANARY: u8 = 0xAB;
        let mut arena = vec![CANARY; 8192];
        for cap in [1usize, 4, 16, 64, 512] {
            let err = c.compress(&mut arena[..cap], input).unwrap_err();
            assert!(
                err.0.contains("fpzip"),
                "expected an fpzip error, got {err}"
            );
            assert!(
                arena[cap..].iter().all(|&b| b == CANARY),
                "fpzip wrote past the {cap}-byte capacity it was given"
            );
            arena[..cap].fill(CANARY);
        }
    }

    /// (3b) The decompress side: a stream that decodes to more than the
    /// caller's capacity is rejected before fpzip writes anything.
    #[test]
    fn decompress_into_too_small_output_errs_without_corrupting_memory() {
        let c = FpzipCompressor::new(Preset::Best);
        let data = realistic_floats(2048);
        let input = as_bytes(&data);
        let mut buf = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut buf, input).unwrap();

        const CANARY: u8 = 0xCD;
        let mut arena = vec![CANARY; 4096];
        let cap = 256; // 64 floats << 2048
        let err = c.decompress(&mut arena[..cap], &buf[..n]).unwrap_err();
        assert!(err.0.contains("output capacity"), "got {err}");
        assert!(
            arena.iter().all(|&b| b == CANARY),
            "decompress wrote into a buffer it should have rejected"
        );
    }

    /// (4) Empty input round-trips as empty, and is not an error.
    #[test]
    fn empty_input_round_trips() {
        let c = FpzipCompressor::new(Preset::Balanced);
        let mut buf = vec![0u8; 64];
        assert_eq!(c.compress(&mut buf, &[]).unwrap(), 0);
        let mut out = vec![0u8; 64];
        assert_eq!(c.decompress(&mut out, &[]).unwrap(), 0);
        // A zero-capacity output is fine when there is nothing to write.
        assert_eq!(c.compress(&mut [], &[]).unwrap(), 0);
        assert_eq!(c.decompress(&mut [], &[]).unwrap(), 0);
    }

    /// Non-float-sized input is rejected (as in C++), not silently truncated.
    #[test]
    fn ragged_input_is_rejected() {
        let c = FpzipCompressor::new(Preset::Balanced);
        let mut buf = vec![0u8; 1024];
        let err = c.compress(&mut buf, &[1u8, 2, 3]).unwrap_err();
        assert!(err.0.contains("multiple of sizeof(f32)"), "got {err}");
    }

    /// Unaligned buffers must round-trip: the C++ `reinterpret_cast` would do
    /// an unaligned float access; we copy through an aligned scratch instead.
    #[test]
    fn unaligned_buffers_round_trip() {
        let c = FpzipCompressor::new(Preset::Best);
        let data = realistic_floats(512);
        let bytes = as_bytes(&data);

        // Offset by 1 to guarantee a non-4-byte-aligned start.
        let mut in_arena = vec![0u8; bytes.len() + 1];
        in_arena[1..].copy_from_slice(bytes);
        let unaligned_in = &in_arena[1..];
        assert!(!is_f32_aligned(unaligned_in.as_ptr()));

        let mut buf = vec![0u8; c.max_compressed_size(bytes.len())];
        let n = c.compress(&mut buf, unaligned_in).unwrap();

        let mut out_arena = vec![0u8; bytes.len() + 1];
        let got = c.decompress(&mut out_arena[1..], &buf[..n]).unwrap();
        assert_eq!(got, bytes.len());
        let out = decode_f32(&out_arena[1..]);
        for (a, b) in data.iter().zip(&out) {
            assert!((a - b).abs() <= a.abs() * 1e-3, "{a} vs {b}");
        }
    }

    /// Garbage must be rejected or produce garbage — never UB or a write past
    /// the caller's buffer. Exercises the padded-scratch reasoning.
    #[test]
    fn corrupt_input_does_not_corrupt_memory() {
        let c = FpzipCompressor::new(Preset::Balanced);
        let data = realistic_floats(1024);
        let mut buf = vec![0u8; c.max_compressed_size(data.len() * 4)];
        let n = c.compress(&mut buf, as_bytes(&data)).unwrap();

        const CANARY: u8 = 0x5A;
        // Truncated streams, byte-flipped headers, and pure noise.
        let mut cases: Vec<Vec<u8>> = vec![
            buf[..n / 2].to_vec(),
            buf[..8.min(n)].to_vec(),
            vec![0xFF; 32],
            vec![0x00; 32],
        ];
        for i in 0..n.min(16) {
            let mut bad = buf[..n].to_vec();
            bad[i] ^= 0xFF;
            cases.push(bad);
        }

        for case in cases {
            let mut arena = vec![CANARY; 4096 + 1024];
            let cap = 4096; // 1024 floats — the true size
            // Either an Err or some number of garbage floats is acceptable;
            // writing beyond `cap` is not.
            let _ = c.decompress(&mut arena[..cap], &case);
            assert!(
                arena[cap..].iter().all(|&b| b == CANARY),
                "decompress wrote past the caller's buffer on corrupt input"
            );
        }
    }

    /// `max_compressed_size` must cover fpzip's worst case: random floats,
    /// which the predictor cannot help with, in lossless mode.
    #[test]
    fn compress_bound_covers_random_floats() {
        let c = FpzipCompressor::with_precision(0).unwrap();
        // xorshift — deterministic, no dev-dependency.
        let mut s = 0x12345678u32;
        let data: Vec<f32> = (0..4096)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                // Full-range exponents make this maximally hostile.
                f32::from_bits(s)
            })
            .filter(|f| f.is_finite())
            .collect();
        let input = as_bytes(&data);
        let mut buf = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut buf, input).expect("bound must suffice");
        assert!(n <= c.max_compressed_size(input.len()));
        // Lossless, even on noise.
        let out = round_trip(&c, &data);
        assert_eq!(
            out.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            data.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );
    }

    /// Precision validation mirrors fpzip's documented float range.
    #[test]
    fn precision_range_is_validated() {
        assert!(FpzipCompressor::with_precision(0).is_ok());
        assert!(FpzipCompressor::with_precision(2).is_ok());
        assert!(FpzipCompressor::with_precision(32).is_ok());
        assert!(FpzipCompressor::with_precision(1).is_err());
        assert!(FpzipCompressor::with_precision(33).is_err());
        assert!(FpzipCompressor::with_precision(-1).is_err());
    }

    /// Preset → prec matches the C++ `ConfigureFPZIP` code, and the factory
    /// wires this type up under the right id.
    #[test]
    fn presets_match_cpp_and_factory_builds_us() {
        assert_eq!(FpzipCompressor::new(Preset::Fast).precision(), 12);
        assert_eq!(FpzipCompressor::new(Preset::Balanced).precision(), 18);
        assert_eq!(FpzipCompressor::new(Preset::Best).precision(), 21);
        // Every preset is lossy, per LibraryId::Fpzip.is_lossy().
        for p in [Preset::Fast, Preset::Balanced, Preset::Best] {
            assert!(!FpzipCompressor::new(p).is_lossless());
        }
        let boxed = crate::compressor_by_name("fpzip", Preset::Balanced).expect("factory");
        assert_eq!(boxed.library(), LibraryId::Fpzip);
        // Drive it through the trait object to prove the vtable is wired.
        let data = realistic_floats(256);
        let input = as_bytes(&data);
        let mut buf = vec![0u8; boxed.max_compressed_size(input.len())];
        let n = boxed.compress(&mut buf, input).unwrap();
        let mut out = vec![0u8; input.len()];
        assert_eq!(boxed.decompress(&mut out, &buf[..n]).unwrap(), input.len());
    }
}
