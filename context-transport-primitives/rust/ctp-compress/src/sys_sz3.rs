// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! FFI wrapper for **SZ3** — the error-bounded lossy compressor for
//! floating-point scientific data (<https://github.com/szcompressor/SZ3>).
//!
//! # What is wrapped
//!
//! The `SZ3c` C shim (`libSZ3c`, header `SZ3c/sz3c.h`), *not* the C++
//! template API (`SZ3/api/sz.hpp`) that the C++ CTP wrapper uses. The C
//! shim is the only stable, `extern "C"` entry point SZ3 ships, so it is
//! the only one callable without a C++ shim TU. Its whole surface is three
//! functions:
//!
//! ```text
//! unsigned char *SZ_compress_args(int dataType, void *data, size_t *outSize,
//!                                 int errBoundMode, double absErrBound,
//!                                 double relBoundRatio, double pwrBoundRatio,
//!                                 size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);
//! void          *SZ_decompress(int dataType, unsigned char *bytes, size_t byteLength,
//!                              size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);
//! void           free_buf(void *p);
//! ```
//!
//! # Version / ABI assumptions
//!
//! * Verified against **SZ3 3.3.2** (`SZ3_VER "3.3.2"`, the devcontainer's
//!   `/usr/local/lib/libSZ3c.so`). The C shim's signatures have been stable
//!   across the 3.x line; it is deliberately SZ2-source-compatible, so the
//!   `dataType` / `errBoundMode` integer codes below are frozen by SZ2's
//!   `defines.h` and are not expected to move.
//! * `libSZ3c` is C++ underneath: the build script links `stdc++` alongside
//!   it. Buffers returned by `SZ_compress_args` / `SZ_decompress` are owned
//!   by SZ3 and **must** be released with `free_buf` (they are not
//!   `malloc`-allocated — do not `libc::free` them). [`Sz3Buffer`] is the
//!   RAII guard that guarantees this on every path.
//! * Dimensions are passed as `r5..r1` with unused dimensions set to 0; CTP
//!   treats payloads as flat 1-D `f32` arrays, so only `r1` is ever nonzero
//!   (matching the C++ wrapper, which also compresses 1-D).
//! * SZ3 is **not** thread-affine and holds no global state across these
//!   calls, so [`Sz3Compressor`] is `Send + Sync` (an empty-ish value type).
//!
//! # Divergence from the C++ wrapper
//!
//! The C++ side has two disagreeing SZ3 paths; neither is copied verbatim,
//! and the reasons matter:
//!
//! 1. **`clio_ctp/compress/sz3.h` (`ctp::Sz3`) is dead code.** Nothing in
//!    the tree includes it, and it cannot compile against SZ3 3.x: it names
//!    `SZ::Config` / `SZ::EB_ABS` (3.x moved to namespace `SZ3`) and calls
//!    `SZ_decompress<float>(conf, ptr, len, num_floats)`, which matches no
//!    3.x overload (the 4-arg form takes `T *&decData`, not a count). Its
//!    *parameter choices* are still the model followed here: 1-D `f32`,
//!    absolute error bound (`ABS`), default bound `1e-3`.
//! 2. **`libpressio_modes.h::ConfigureSZ3` is the live factory path** and
//!    uses REL mode at 0.05 / 0.01 / 0.005 — while the doc comment above it
//!    still claims 1e-2 / 1e-3 / 1e-4. That path routes through LibPressio,
//!    which this crate does not link.
//!
//!    This module follows the *direct* SZ3 wrapper's model — `ABS` mode —
//!    with bounds FAST `1e-2`, BALANCED `1e-3`, BEST `1e-4`, so BALANCED
//!    (the [`Preset`] default) reproduces `ctp::Sz3`'s default constructor
//!    exactly. ABS is also the only mode whose guarantee is checkable
//!    without knowing the data's value range, which is what the round-trip
//!    tests assert.
//!
//! 3. **Framing: an 8-byte little-endian length header precedes the SZ3
//!    stream.** The C++ wrapper emits the raw SZ3 stream and requires the
//!    *caller* to already know the original element count, which it feeds
//!    to the decompressor. That contract cannot be made memory-safe here:
//!    the C shim's `SZ_decompress` sizes its output allocation from the
//!    `r1..r5` the caller passes while the stream itself dictates how many
//!    elements get written, so a caller who guesses low is a heap overflow
//!    (confirmed under ASan against 3.3.2). [`Compressor::decompress`]
//!    takes only an output *capacity*, so the true length must travel with
//!    the payload. The header makes the stream self-describing, turns
//!    "output buffer too small" into a clean [`CompressError`] instead of
//!    silent truncation, and gives empty input a valid encoding.
//!
//!    Consequence: SZ3 blobs written by this module are **not** byte-wise
//!    interchangeable with a raw SZ3 stream — strip/prepend 8 bytes to
//!    convert. Since the C++ SZ3 path never ran (dead code / LibPressio
//!    framing differs anyway), no existing blob format is broken.
//!
//! 4. **Errors are returned, not printed.** The C++ wrapper logs to
//!    `std::cerr` and returns `bool`; this returns [`CompressError`].

use std::ffi::{c_int, c_void};

use crate::{CompressError, Compressor, LibraryId, Preset};

// --- SZ3 C shim constants (SZ3c/sz3c.h, frozen from SZ2's defines.h) ------

/// `dataType`: IEEE-754 binary32. CTP payloads are `f32` arrays.
const SZ_FLOAT: c_int = 0;

/// `errBoundMode`: absolute error bound (`|decompressed - original| <= abs`).
const EB_ABS: c_int = 0;

/// Bytes of framing this module prepends to the SZ3 stream (see module docs).
const HEADER_LEN: usize = 8;

extern "C" {
    /// Returns an SZ3-owned buffer of `*out_size` bytes, or null on failure.
    /// `data` is read-only in practice but declared `void *` by the shim.
    /// Unused dimensions are passed as 0.
    fn SZ_compress_args(
        data_type: c_int,
        data: *mut c_void,
        out_size: *mut usize,
        err_bound_mode: c_int,
        abs_err_bound: f64,
        rel_bound_ratio: f64,
        pwr_bound_ratio: f64,
        r5: usize,
        r4: usize,
        r3: usize,
        r2: usize,
        r1: usize,
    ) -> *mut u8;

    /// Returns an SZ3-owned buffer holding the decompressed array, or null.
    /// The allocation is sized from `r5..r1`, so they MUST equal the count
    /// the stream actually encodes (see the module docs' framing note).
    fn SZ_decompress(
        data_type: c_int,
        bytes: *mut u8,
        byte_length: usize,
        r5: usize,
        r4: usize,
        r3: usize,
        r2: usize,
        r1: usize,
    ) -> *mut c_void;

    /// Releases a buffer returned by `SZ_compress_args` / `SZ_decompress`.
    fn free_buf(p: *mut c_void);
}

// --- RAII guard over SZ3-owned memory ------------------------------------

/// Owns a buffer returned by the SZ3 C shim and releases it with `free_buf`.
///
/// Every early return between the SZ3 call and the copy-out (buffer too
/// small, bad length, …) would otherwise leak, so ownership is taken the
/// instant a non-null pointer comes back.
struct Sz3Buffer {
    ptr: *mut c_void,
    len: usize,
}

impl Sz3Buffer {
    /// Take ownership of `ptr`/`len`, or produce an error if SZ3 failed.
    ///
    /// # Safety
    /// `ptr` must be null or an SZ3-owned buffer of at least `len` bytes
    /// that is valid to release with `free_buf`.
    unsafe fn adopt(ptr: *mut c_void, len: usize, what: &str) -> Result<Self, CompressError> {
        if ptr.is_null() {
            return Err(CompressError(format!("SZ3: {what} failed (null buffer)")));
        }
        if len == 0 {
            // SAFETY: non-null SZ3 buffer; releasing it before we bail out.
            unsafe { free_buf(ptr) };
            return Err(CompressError(format!("SZ3: {what} produced 0 bytes")));
        }
        Ok(Self { ptr, len })
    }

    /// The buffer's contents as bytes.
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `adopt` established a non-null SZ3 buffer of `len` bytes,
        // alive for `self`'s lifetime (Drop is the only release), and no
        // `&mut` to it exists. `u8` has no alignment requirement.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Drop for Sz3Buffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from SZ_compress_args/SZ_decompress and is
        // released exactly once — nothing else holds it.
        unsafe { free_buf(self.ptr) };
    }
}

// --- Safe compressor ------------------------------------------------------

/// SZ3 error-bounded lossy compressor for `f32` payloads.
///
/// **Lossy** (`LibraryId::Sz3.is_lossy()`): a round trip guarantees only
/// `|out[i] - in[i]| <= error_bound()`, never byte equality. Never route
/// metadata or non-`f32` payloads through it.
///
/// Input length must be a multiple of `size_of::<f32>()` — the payload is
/// interpreted as a flat 1-D `f32` array, exactly as the C++ wrapper does.
#[derive(Debug, Clone, Copy)]
pub struct Sz3Compressor {
    /// Absolute error bound handed to SZ3's `ABS` mode.
    error_bound: f64,
}

impl Sz3Compressor {
    /// Construct at `preset` (C++ `CompressFactory::MakeSz3`).
    ///
    /// Preset → absolute error bound: FAST `1e-2`, BALANCED `1e-3` (the C++
    /// `ctp::Sz3` default), BEST `1e-4`. See the module docs for why these
    /// are ABS rather than the live LibPressio path's REL bounds.
    pub fn new(preset: Preset) -> Self {
        let error_bound = match preset {
            Preset::Fast => 1e-2,
            Preset::Balanced => 1e-3,
            Preset::Best => 1e-4,
        };
        Self { error_bound }
    }

    /// Construct with an explicit absolute error bound (C++
    /// `Sz3::SetErrorBound`). Non-finite or negative bounds are rejected by
    /// [`Compressor::compress`] rather than handed to SZ3.
    pub fn with_error_bound(error_bound: f64) -> Self {
        Self { error_bound }
    }

    /// The configured absolute error bound (C++ `Sz3::GetErrorBound`).
    pub fn error_bound(&self) -> f64 {
        self.error_bound
    }
}

impl Compressor for Sz3Compressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        if !self.error_bound.is_finite() || self.error_bound < 0.0 {
            return Err(CompressError(format!(
                "SZ3: invalid error bound {}",
                self.error_bound
            )));
        }
        if input.len() % std::mem::size_of::<f32>() != 0 {
            return Err(CompressError(format!(
                "SZ3: input size {} is not a multiple of sizeof(f32)",
                input.len()
            )));
        }
        if output.len() < HEADER_LEN {
            return Err(CompressError(format!(
                "SZ3: output buffer too small ({} bytes; header alone needs {HEADER_LEN})",
                output.len()
            )));
        }
        output[..HEADER_LEN].copy_from_slice(&(input.len() as u64).to_le_bytes());
        if input.is_empty() {
            // SZ3 does emit a ~70-byte stream for a 0-element array, but a
            // header-only frame is cheaper and decodes to the same thing.
            return Ok(HEADER_LEN);
        }

        let num_floats = input.len() / std::mem::size_of::<f32>();

        // SZ3 reads `data` as `float*`. A `&[u8]` need not be 4-byte
        // aligned, and an unaligned float load is UB (and can fault under
        // SZ3's vectorized paths), so realign through an owned Vec<f32>
        // when needed. The common case (allocator-backed buffers) is
        // already aligned and copies nothing.
        let mut aligned: Vec<f32>;
        let data_ptr: *mut c_void = if input.as_ptr() as usize % std::mem::align_of::<f32>() == 0 {
            // SZ3 does not write through `data` (the shim's `void *` is
            // simply un-const'd C); casting away shared-ness is sound as
            // long as it stays read-only, which the C++ wrapper relies on
            // too.
            input.as_ptr() as *mut c_void
        } else {
            aligned = vec![0f32; num_floats];
            // SAFETY: both regions are valid for `input.len()` bytes, do
            // not overlap (fresh allocation), and `f32` is a POD whose
            // byte-image is exactly the source bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    aligned.as_mut_ptr() as *mut u8,
                    input.len(),
                );
            }
            aligned.as_mut_ptr() as *mut c_void
        };

        let mut out_size: usize = 0;
        // SAFETY: `data_ptr` is a readable, 4-byte-aligned array of
        // `num_floats` f32s (checked/realigned above) that outlives the
        // call; `out_size` is a valid out-pointer; dims are 1-D with the
        // unused r5..r2 zeroed per the shim's contract. The returned
        // buffer is adopted immediately so it cannot leak.
        let buf = unsafe {
            let ptr = SZ_compress_args(
                SZ_FLOAT,
                data_ptr,
                &mut out_size,
                EB_ABS,
                self.error_bound,
                // Mirrors the C++ wrapper, which sets abs and rel alike.
                // Unused in ABS mode; kept so switching modes is a
                // one-line change.
                self.error_bound,
                0.0,
                0,
                0,
                0,
                0,
                num_floats,
            );
            Sz3Buffer::adopt(ptr as *mut c_void, out_size, "compression")?
        };

        let total = HEADER_LEN + buf.len;
        if total > output.len() {
            // `buf` frees on return — this is the path the C++ wrapper had
            // to hand-`delete[]`.
            return Err(CompressError(format!(
                "SZ3: output buffer too small ({} bytes; need {total})",
                output.len()
            )));
        }
        output[HEADER_LEN..total].copy_from_slice(buf.as_bytes());
        Ok(total)
    }

    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        if input.len() < HEADER_LEN {
            return Err(CompressError(format!(
                "SZ3: truncated stream ({} bytes; need at least {HEADER_LEN})",
                input.len()
            )));
        }
        // Infallible: the slice is exactly 8 bytes wide.
        let orig_len = u64::from_le_bytes(input[..HEADER_LEN].try_into().unwrap());
        let orig_len = usize::try_from(orig_len).map_err(|_| {
            CompressError(format!(
                "SZ3: stream declares {orig_len} bytes, too large for this platform"
            ))
        })?;
        if orig_len % std::mem::size_of::<f32>() != 0 {
            return Err(CompressError(format!(
                "SZ3: corrupt stream (declared size {orig_len} is not a multiple of sizeof(f32))"
            )));
        }
        // Checked BEFORE calling SZ3: `SZ_decompress` sizes its allocation
        // from the count we pass, so a short output must fail here rather
        // than let us hand SZ3 a count that does not match the stream.
        if orig_len > output.len() {
            return Err(CompressError(format!(
                "SZ3: output buffer too small ({} bytes; need {orig_len})",
                output.len()
            )));
        }
        let payload = &input[HEADER_LEN..];
        if orig_len == 0 {
            return Ok(0);
        }
        if payload.is_empty() {
            return Err(CompressError(format!(
                "SZ3: corrupt stream (declares {orig_len} bytes but carries no payload)"
            )));
        }

        let num_floats = orig_len / std::mem::size_of::<f32>();
        // SAFETY: `payload` is readable for `payload.len()` bytes; the shim
        // does not write through `bytes` (un-const'd C again). `num_floats`
        // is the count this frame recorded at compress time, so SZ3's
        // output allocation matches what the stream decodes to. The result
        // is adopted immediately, so it frees on every path below.
        let buf = unsafe {
            let ptr = SZ_decompress(
                SZ_FLOAT,
                payload.as_ptr() as *mut u8,
                payload.len(),
                0,
                0,
                0,
                0,
                num_floats,
            );
            Sz3Buffer::adopt(ptr, orig_len, "decompression")?
        };

        output[..orig_len].copy_from_slice(buf.as_bytes());
        Ok(orig_len)
    }

    fn library(&self) -> LibraryId {
        LibraryId::Sz3
    }

    fn max_compressed_size(&self, input_size: usize) -> usize {
        // SZ3 exposes no bound API. Worst case is an incompressible array
        // stored near-verbatim plus SZ3's metadata block (~70 bytes for a
        // 1-element array on 3.3.2); keep the default's generous headroom
        // and add this module's frame header.
        input_size + input_size / 8 + 1024 + HEADER_LEN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth-ish scientific signal: what SZ3's prediction model targets.
    fn sample_field(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = i as f64 * 0.01;
                (x.sin() * 100.0 + x * 0.5 + (x * 0.1).cos() * 10.0) as f32
            })
            .collect()
    }

    fn as_bytes(v: &[f32]) -> &[u8] {
        // SAFETY: `f32` is POD; reading its 4*len byte-image aliases no
        // padding and stays inside the allocation.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    fn to_floats(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// (1) Round trip over realistic data stays inside the preset's bound —
    /// the only guarantee a lossy codec makes.
    #[test]
    fn roundtrip_within_error_bound() {
        let data = sample_field(8192);
        let raw = as_bytes(&data);

        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let c = Sz3Compressor::new(preset);
            let mut cbuf = vec![0u8; c.max_compressed_size(raw.len())];
            let n = c.compress(&mut cbuf, raw).unwrap();

            let mut dbuf = vec![0u8; raw.len()];
            let m = c.decompress(&mut dbuf, &cbuf[..n]).unwrap();
            assert_eq!(m, raw.len(), "{preset:?}: decompressed length");

            let back = to_floats(&dbuf[..m]);
            assert_eq!(back.len(), data.len());
            let max_err = data
                .iter()
                .zip(&back)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0.0f64, f64::max);
            // SZ3's ABS guarantee, with a hair of slack for the f32 store.
            let bound = c.error_bound() * 1.000_001 + f32::EPSILON as f64;
            assert!(
                max_err <= bound,
                "{preset:?}: max error {max_err:e} exceeds bound {:e}",
                c.error_bound()
            );
            // Lossy: it must not be silently returning the input verbatim.
            assert!(n < raw.len(), "{preset:?}: {n} vs {} raw", raw.len());
            eprintln!(
                "sz3 {preset:?}: {} -> {n} bytes ({:.1}x), max_err={max_err:e}",
                raw.len(),
                raw.len() as f64 / n as f64
            );
        }
    }

    /// Tighter presets must not be looser than coarser ones.
    #[test]
    fn presets_are_ordered() {
        assert!(
            Sz3Compressor::new(Preset::Best).error_bound()
                < Sz3Compressor::new(Preset::Balanced).error_bound()
        );
        assert!(
            Sz3Compressor::new(Preset::Balanced).error_bound()
                < Sz3Compressor::new(Preset::Fast).error_bound()
        );
        // BALANCED reproduces the C++ ctp::Sz3 default constructor.
        assert_eq!(Sz3Compressor::new(Preset::Balanced).error_bound(), 1e-3);
        assert_eq!(Sz3Compressor::new(Preset::default()).error_bound(), 1e-3);
        assert_eq!(Sz3Compressor::new(Preset::Fast).library(), LibraryId::Sz3);
    }

    /// (2) Highly-compressible data actually shrinks.
    #[test]
    fn compressible_data_shrinks() {
        // A constant field is the easy case; a linear ramp exercises the
        // Lorenzo/interpolation predictor rather than pure RLE.
        for data in [vec![3.25f32; 16384], (0..16384).map(|i| i as f32 * 0.5).collect()] {
            let raw = as_bytes(&data);
            let c = Sz3Compressor::new(Preset::Balanced);
            let mut cbuf = vec![0u8; c.max_compressed_size(raw.len())];
            let n = c.compress(&mut cbuf, raw).unwrap();
            assert!(
                n * 4 < raw.len(),
                "expected >4x on compressible data, got {} -> {n}",
                raw.len()
            );
        }
    }

    /// (3) Too-small output buffers return Err rather than corrupting
    /// memory — on both sides, and at both the header and payload stages.
    #[test]
    fn output_buffer_too_small_errors() {
        let data = sample_field(4096);
        let raw = as_bytes(&data);
        let c = Sz3Compressor::new(Preset::Balanced);

        // Smaller than the frame header.
        assert!(c.compress(&mut [0u8; 4], raw).is_err());
        // Room for the header, nowhere near the payload.
        assert!(c.compress(&mut [0u8; 64], raw).is_err());

        let mut cbuf = vec![0u8; c.max_compressed_size(raw.len())];
        let n = c.compress(&mut cbuf, raw).unwrap();

        // Decompress into a buffer one float short: must Err, not truncate
        // and not hand SZ3 a mismatched element count.
        let mut short = vec![0xAAu8; raw.len() - 4];
        assert!(c.decompress(&mut short, &cbuf[..n]).is_err());
        assert!(short.iter().all(|b| *b == 0xAA), "buffer was written to");

        // An exactly-sized buffer still succeeds.
        let mut exact = vec![0u8; raw.len()];
        assert_eq!(c.decompress(&mut exact, &cbuf[..n]).unwrap(), raw.len());
    }

    /// (4) Empty input round-trips to empty.
    #[test]
    fn empty_input_roundtrips() {
        let c = Sz3Compressor::new(Preset::Balanced);
        let mut cbuf = vec![0u8; c.max_compressed_size(0)];
        let n = c.compress(&mut cbuf, &[]).unwrap();
        assert_eq!(n, HEADER_LEN);

        let mut out = [0u8; 8];
        assert_eq!(c.decompress(&mut out, &cbuf[..n]).unwrap(), 0);
        // A zero-capacity output is fine for an empty payload.
        assert_eq!(c.decompress(&mut [], &cbuf[..n]).unwrap(), 0);
    }

    /// Malformed inputs are rejected, never fed to SZ3.
    #[test]
    fn invalid_inputs_error() {
        let c = Sz3Compressor::new(Preset::Balanced);
        let mut buf = vec![0u8; 4096];

        // Not a whole number of f32s.
        assert!(c.compress(&mut buf, &[1, 2, 3]).is_err());
        // Truncated / corrupt frames.
        assert!(c.decompress(&mut buf, &[]).is_err());
        assert!(c.decompress(&mut buf, &[0; 4]).is_err());
        // Declares 12 bytes but carries no SZ3 payload.
        let mut frame = 12u64.to_le_bytes().to_vec();
        assert!(c.decompress(&mut buf, &frame).is_err());
        // Declared size is not a multiple of sizeof(f32).
        frame = 7u64.to_le_bytes().to_vec();
        frame.extend_from_slice(&[0u8; 16]);
        assert!(c.decompress(&mut buf, &frame).is_err());
        // A garbage error bound fails before reaching the library.
        let bad = Sz3Compressor::with_error_bound(f64::NAN);
        assert!(bad.compress(&mut buf, as_bytes(&[1.0f32, 2.0])).is_err());
    }

    /// Unaligned input is realigned rather than fed to SZ3 as a misaligned
    /// `float*`.
    #[test]
    fn unaligned_input_roundtrips() {
        let data = sample_field(1024);
        let src = as_bytes(&data);
        // Offset by 1 byte inside a larger buffer to break f32 alignment.
        let mut skewed = vec![0u8; src.len() + 1];
        skewed[1..].copy_from_slice(src);
        let input = &skewed[1..];
        assert_ne!(input.as_ptr() as usize % 4, 0, "test setup: still aligned");

        let c = Sz3Compressor::new(Preset::Balanced);
        let mut cbuf = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut cbuf, input).unwrap();
        let mut dbuf = vec![0u8; input.len()];
        let m = c.decompress(&mut dbuf, &cbuf[..n]).unwrap();
        assert_eq!(m, input.len());

        let back = to_floats(&dbuf);
        let max_err = data
            .iter()
            .zip(&back)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err <= c.error_bound() * 1.000_001 + f32::EPSILON as f64);
    }

    /// The factory wires this type up (guards the name/ctor contract).
    #[test]
    fn factory_constructs_sz3() {
        let c = crate::compressor_by_name("sz3", Preset::Best).expect("sz3 feature is on");
        assert_eq!(c.library(), LibraryId::Sz3);
        assert!(c.library().is_lossy());
    }
}
