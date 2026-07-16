// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! FFI wrapper for **zfp** — LLNL's lossy fixed-rate/fixed-accuracy compressor
//! for floating-point arrays (<https://github.com/LLNL/zfp>).
//!
//! Wraps the C API of `libzfp` (`zfp_stream_*`, `zfp_field_*`, `zfp_compress` /
//! `zfp_decompress`) plus `libzfp`'s bitstream API (`stream_open` /
//! `stream_close`). This is the Rust face of C++ `ctp::Zfp`
//! (`clio_ctp/compress/zfp.h`) with the preset table of
//! `clio_ctp/compress/libpressio_modes.h::ConfigureZFP`.
//!
//! # Version / ABI assumptions
//!
//! Verified against **zfp 1.0.1** (`libzfp.so.1`, the devcontainer's
//! `libzfp-dev 1.0.1-3.1build1`). The declarations below are hand-rolled and
//! assume:
//!
//! * `zfp_stream`, `zfp_field` and `bitstream` are **opaque** — only ever held
//!   behind a pointer, never dereferenced here, so their layout is irrelevant.
//! * `zfp_type` is a C enum passed as a 4-byte value (`zfp_type_float = 3`);
//!   `zfp_bool` is `typedef int` (zfp.h:65). Both are register-passed
//!   identically as `c_uint`/`c_int` under the SysV and MS x64 ABIs.
//! * `zfp_compress` / `zfp_decompress` return `size_t` (bytes of compressed
//!   storage, **0 on failure**). zfp ≥ 1.0 kept the 0.5.x return type here.
//! * `stream_word_bits` (exported `const size_t`) is the bitstream granularity.
//!   Checked at runtime to be ≤ 64 — the whole alignment argument below rests
//!   on it, so a wider-word build of zfp is rejected instead of trusted.
//!
//! Every symbol used is exported from `libzfp.so.1` (they are not `inline_`
//! builds of the bitstream API); `build.rs` links `-lzfp` under the `zfp`
//! feature.
//!
//! # Framing (compatible with C++ `ctp::Zfp`)
//!
//! The stream is **headerless** (no `zfp_write_header`), exactly like the C++
//! wrapper: a 1D `zfp_field` of `zfp_type_float` over `input.len() / 4` values,
//! native-endian, compressed with the preset's mode. Because nothing is
//! self-describing, [`Compressor::decompress`] cannot discover the original
//! length: it treats `output.len()` as the **exact** decompressed size (the
//! C++ INOUT `output_size` is likewise an IN parameter on that path). CTP
//! callers have the original size in blob metadata, which is how the C++ side
//! works too.
//!
//! # Divergences from the C++ wrapper — deliberate, all safety-driven
//!
//! 1. **Presets, not a fixed tolerance.** C++ `ctp::Zfp` hardcodes accuracy
//!    mode at 1e-3. The factory path (`CompressFactory::Get("zfp", preset)` →
//!    `CreateLossy` → LibPressio `ConfigureZFP`) is preset-driven, and this
//!    crate's factory is preset-driven, so [`ZfpCompressor::new`] follows
//!    `ConfigureZFP`: FAST → rate 8.0, BALANCED → rate 16.0, BEST → accuracy
//!    1e-3. `Preset::Best` therefore reproduces C++ `ctp::Zfp`'s default byte
//!    for byte. (Note: **not** rate 24.0 for BEST — the C++ switches modes.)
//! 2. **Aligned scratch buffers.** C++ hands zfp the caller's `void*` directly.
//!    zfp casts bitstream buffers to `uint64*` and field pointers to `float*`,
//!    so a misaligned buffer is UB in C — and a `&[u8]` from safe Rust carries
//!    no alignment guarantee. Every call therefore stages through an internal
//!    `Vec<u64>` / `Vec<f32>` and copies at the boundary. Costs one memcpy of
//!    already-O(n) work; buys "no UB reachable from safe code".
//! 3. **Bounded reads on corrupt input.** zfp does no bounds checking. The
//!    decompress scratch is sized to `max(input.len(), zfp_stream_maximum_size)`
//!    so a truncated or corrupt stream reads zero padding instead of running off
//!    the allocation. Such a stream yields garbage floats, not an error — same
//!    observable result as C++, minus the out-of-bounds read.
//! 4. **Output-capacity check is on the actual size.** C++ rejects when
//!    `output_size < zfp_stream_maximum_size` (worst case). Since we compress
//!    into scratch anyway, we reject only when the *actual* compressed bytes do
//!    not fit — strictly more permissive, never less safe. Allocate
//!    [`Compressor::max_compressed_size`] and this never triggers.
//! 5. **Errors, not `std::cerr` + `bool`.** Returns [`CompressError`]; nothing
//!    panics on a library failure.
//!
//! # Lossiness
//!
//! zfp is lossy ([`LibraryId::is_lossy`]). Round-trips are accurate only within
//! the mode's implied tolerance — `Preset::Best` (accuracy mode) bounds the
//! absolute error at 1e-3; the fixed-rate presets bound the *bit budget*
//! (8 or 16 bits/value), not the error, so their accuracy is data-dependent.
//! Never route metadata or non-`f32` payloads through this codec.

use std::ffi::{c_int, c_uint, c_void};
use std::marker::PhantomData;

use crate::{CompressError, Compressor, LibraryId, Preset};

// ---------------------------------------------------------------------------
// Raw FFI declarations (hand-rolled against zfp 1.0.1's zfp.h / bitstream.h)
// ---------------------------------------------------------------------------

/// `zfp_stream*` — opaque.
type ZfpStreamPtr = *mut c_void;
/// `zfp_field*` — opaque.
type ZfpFieldPtr = *mut c_void;
/// `bitstream*` — opaque.
type BitStreamPtr = *mut c_void;
/// `zfp_type` (C enum, 4-byte).
type ZfpType = c_uint;
/// `zfp_bool` (`typedef int`, zfp.h:65).
type ZfpBool = c_int;

/// `zfp_type_float` — single precision (zfp.h:126).
const ZFP_TYPE_FLOAT: ZfpType = 3;
/// `align = zfp_false`: no word-aligned blocks (LibPressio's zfp plugin and
/// zfp's own `zfp_stream_set_rate` callers pass false for sequential streams).
const ZFP_ALIGN_FALSE: ZfpBool = 0;
/// 1D field — `dims` argument of `zfp_stream_set_rate`.
const ZFP_DIMS_1D: c_uint = 1;

extern "C" {
    // -- compressed stream construction / destruction -----------------------
    fn zfp_stream_open(stream: BitStreamPtr) -> ZfpStreamPtr;
    fn zfp_stream_close(stream: ZfpStreamPtr);

    // -- compression mode ---------------------------------------------------
    fn zfp_stream_set_rate(
        stream: ZfpStreamPtr,
        rate: f64,
        type_: ZfpType,
        dims: c_uint,
        align: ZfpBool,
    ) -> f64;
    fn zfp_stream_set_accuracy(stream: ZfpStreamPtr, tolerance: f64) -> f64;

    // -- stream inspectors / setup ------------------------------------------
    fn zfp_stream_maximum_size(stream: ZfpStreamPtr, field: ZfpFieldPtr) -> usize;
    fn zfp_stream_set_bit_stream(stream: ZfpStreamPtr, bs: BitStreamPtr);
    fn zfp_stream_rewind(stream: ZfpStreamPtr);

    // -- field metadata ------------------------------------------------------
    fn zfp_field_1d(pointer: *mut c_void, type_: ZfpType, nx: usize) -> ZfpFieldPtr;
    fn zfp_field_free(field: ZfpFieldPtr);

    // -- compression / decompression (return 0 on failure) -------------------
    fn zfp_compress(stream: ZfpStreamPtr, field: ZfpFieldPtr) -> usize;
    fn zfp_decompress(stream: ZfpStreamPtr, field: ZfpFieldPtr) -> usize;

    // -- bitstream (zfp/bitstream.h) -----------------------------------------
    fn stream_open(buffer: *mut c_void, bytes: usize) -> BitStreamPtr;
    fn stream_close(stream: BitStreamPtr);

    /// Exported `const size_t stream_word_bits` — bitstream granularity.
    static stream_word_bits: usize;
}

/// Bytes per `f32` scalar — the field's element size.
const SCALAR: usize = std::mem::size_of::<f32>();
/// Scratch element width in bytes (`Vec<u64>` ⇒ 8-byte aligned storage).
const WORD: usize = std::mem::size_of::<u64>();

/// Reject zfp builds whose bitstream word is wider than our `Vec<u64>` scratch,
/// since the no-UB argument depends on the scratch being word-aligned.
fn check_word_size() -> Result<(), CompressError> {
    // SAFETY: reading an immutable `const size_t` exported by libzfp. It is
    // initialized before `main` (it lives in .rodata) and never written.
    let bits = unsafe { stream_word_bits };
    if bits > WORD * 8 {
        return Err(CompressError(format!(
            "zfp built with {bits}-bit bitstream words; this wrapper aligns \
             scratch to {}-bit words",
            WORD * 8
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// RAII wrappers over the opaque handles
// ---------------------------------------------------------------------------

/// Owns a `zfp_stream*`.
struct ZfpStream {
    ptr: ZfpStreamPtr,
}

impl ZfpStream {
    /// `zfp_stream_open(NULL)` — no bitstream attached yet.
    fn open() -> Result<Self, CompressError> {
        // SAFETY: a NULL bitstream is explicitly allowed ("may be NULL",
        // zfp.h:160); the call only allocates.
        let ptr = unsafe { zfp_stream_open(std::ptr::null_mut()) };
        if ptr.is_null() {
            return Err(CompressError("zfp_stream_open: allocation failed".into()));
        }
        Ok(Self { ptr })
    }

    /// Apply the compressor's mode. Mirrors `ConfigureZFP` in
    /// `libpressio_modes.h`.
    fn configure(&self, mode: Mode) -> Result<(), CompressError> {
        match mode {
            Mode::Rate(rate) => {
                // SAFETY: `self.ptr` is a live zfp_stream from open(); the
                // enum/dims/align arguments are the constants above.
                let actual = unsafe {
                    zfp_stream_set_rate(
                        self.ptr,
                        rate,
                        ZFP_TYPE_FLOAT,
                        ZFP_DIMS_1D,
                        ZFP_ALIGN_FALSE,
                    )
                };
                // NaN or non-positive ⇒ zfp could not honour the request.
                if actual.is_nan() || actual <= 0.0 {
                    return Err(CompressError(format!(
                        "zfp_stream_set_rate({rate}) rejected: actual rate {actual}"
                    )));
                }
            }
            Mode::Accuracy(tolerance) => {
                // SAFETY: `self.ptr` is a live zfp_stream from open().
                let actual = unsafe { zfp_stream_set_accuracy(self.ptr, tolerance) };
                // NaN or non-positive ⇒ zfp could not honour the request.
                if actual.is_nan() || actual <= 0.0 {
                    return Err(CompressError(format!(
                        "zfp_stream_set_accuracy({tolerance}) rejected: actual tolerance {actual}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Attach `bs` and rewind, ready for compress/decompress.
    fn bind(&self, bs: &BitStream<'_>) {
        // SAFETY: both handles are live and were produced by their respective
        // open() calls; `bs` outlives this call (borrowed).
        unsafe {
            zfp_stream_set_bit_stream(self.ptr, bs.ptr);
            zfp_stream_rewind(self.ptr);
        }
    }

    /// `zfp_stream_maximum_size` — conservative compressed-size bound.
    fn maximum_size(&self, field: &ZfpField<'_>) -> usize {
        // SAFETY: both handles are live; the call only reads metadata (it does
        // not touch the field's data pointer, which may be NULL).
        unsafe { zfp_stream_maximum_size(self.ptr, field.ptr) }
    }
}

impl Drop for ZfpStream {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from zfp_stream_open, is non-null, and this runs
        // exactly once (no Copy/Clone). Does not free the bitstream.
        unsafe { zfp_stream_close(self.ptr) };
    }
}

/// Owns a `zfp_field*`. The lifetime ties it to the scalar buffer it describes,
/// so the field can never outlive the storage zfp will read/write through it.
struct ZfpField<'a> {
    ptr: ZfpFieldPtr,
    _data: PhantomData<&'a mut [f32]>,
}

impl<'a> ZfpField<'a> {
    /// `zfp_field_1d(data, zfp_type_float, data.len())`.
    fn new_1d(data: &'a mut [f32]) -> Result<Self, CompressError> {
        // SAFETY: `data` is a live, properly aligned (`Vec<f32>`-backed) slice
        // of exactly `data.len()` f32 scalars; zfp stores the pointer and the
        // count, and the PhantomData borrow keeps `data` alive and exclusively
        // borrowed for as long as this field exists.
        let ptr = unsafe {
            zfp_field_1d(
                data.as_mut_ptr().cast::<c_void>(),
                ZFP_TYPE_FLOAT,
                data.len(),
            )
        };
        if ptr.is_null() {
            return Err(CompressError("zfp_field_1d: allocation failed".into()));
        }
        Ok(Self {
            ptr,
            _data: PhantomData,
        })
    }

    /// A field with no backing storage — metadata only, for size queries.
    /// zfp documents the pointer as "may be NULL" (zfp.h:381).
    fn metadata_1d(nx: usize) -> Result<Self, CompressError> {
        // SAFETY: NULL storage is explicitly allowed for metadata-only fields;
        // we only pass such a field to zfp_stream_maximum_size, which never
        // dereferences the data pointer.
        let ptr = unsafe { zfp_field_1d(std::ptr::null_mut(), ZFP_TYPE_FLOAT, nx) };
        if ptr.is_null() {
            return Err(CompressError("zfp_field_1d: allocation failed".into()));
        }
        Ok(Self {
            ptr,
            _data: PhantomData,
        })
    }
}

impl Drop for ZfpField<'_> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from zfp_field_1d and is freed exactly once. This
        // frees only the metadata, never the caller's scalars.
        unsafe { zfp_field_free(self.ptr) };
    }
}

/// Owns a `bitstream*`. The lifetime ties it to its word-aligned buffer.
struct BitStream<'a> {
    ptr: BitStreamPtr,
    _buf: PhantomData<&'a mut [u64]>,
}

impl<'a> BitStream<'a> {
    /// `stream_open(buf, buf.len() * 8)`.
    fn open(buf: &'a mut [u64]) -> Result<Self, CompressError> {
        // SAFETY: `buf` is `Vec<u64>`-backed, hence aligned to 8 bytes (≥ the
        // bitstream word, checked by check_word_size) and exactly
        // `buf.len() * 8` bytes long, which is the capacity we declare. The
        // PhantomData borrow keeps it alive and exclusively borrowed.
        let ptr = unsafe {
            stream_open(
                buf.as_mut_ptr().cast::<c_void>(),
                buf.len().saturating_mul(WORD),
            )
        };
        if ptr.is_null() {
            return Err(CompressError("stream_open: allocation failed".into()));
        }
        Ok(Self {
            ptr,
            _buf: PhantomData,
        })
    }
}

impl Drop for BitStream<'_> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from stream_open and is closed exactly once. This
        // frees the bitstream object, not the buffer it wraps.
        unsafe { stream_close(self.ptr) };
    }
}

// ---------------------------------------------------------------------------
// Buffer staging (see divergence #2)
// ---------------------------------------------------------------------------

/// Reinterpret `input` as native-endian `f32` scalars, copying into aligned
/// storage. Mirrors the C++ `reinterpret_cast<float*>(input)` — including its
/// native-endian assumption — but without the alignment UB.
fn floats_from_bytes(input: &[u8]) -> Vec<f32> {
    input
        .chunks_exact(SCALAR)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Copy `src` into a zero-padded, 8-byte-aligned word buffer of at least
/// `min_bytes` capacity.
fn words_from_bytes(src: &[u8], min_bytes: usize) -> Vec<u64> {
    let bytes = src.len().max(min_bytes).max(WORD);
    let mut words = vec![0u64; bytes.div_ceil(WORD)];
    // SAFETY: `words` holds `words.len() * 8 >= src.len()` bytes of writable
    // storage; the regions are distinct allocations, so they cannot overlap.
    // u64 has no padding or invalid bit patterns, so writing arbitrary bytes
    // into it is well defined.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), words.as_mut_ptr().cast::<u8>(), src.len());
    }
    words
}

/// Copy the first `n` bytes of `words` into `dst`. `n` must be ≤ both sides.
fn bytes_from_words(dst: &mut [u8], words: &[u64], n: usize) {
    debug_assert!(n <= dst.len() && n <= words.len() * WORD);
    // SAFETY: `n` bytes are in bounds of both `words` (n ≤ len*8) and `dst`
    // (n ≤ dst.len(), checked by the caller before this runs); distinct
    // allocations, so non-overlapping. Reading a u64 buffer as bytes is
    // well defined (no padding, alignment 8 ≥ 1).
    unsafe {
        std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), dst.as_mut_ptr(), n);
    }
}

// ---------------------------------------------------------------------------
// ZfpCompressor
// ---------------------------------------------------------------------------

/// zfp's compression mode for a preset — a faithful copy of the C++
/// `ConfigureZFP` switch (`libpressio_modes.h:157`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    /// Fixed-rate mode: N compressed bits per scalar (`zfp:mode = 1`).
    Rate(f64),
    /// Fixed-accuracy mode: absolute error tolerance (`zfp:mode = 3`).
    Accuracy(f64),
}

/// zfp lossy compressor for 1D `f32` arrays (C++ `ctp::Zfp` + the factory's
/// preset table).
///
/// Only `Preset` picks the mode; see the module docs for the preset → mode map
/// and for why `decompress` needs an exactly-sized output buffer.
#[derive(Debug, Clone, Copy)]
pub struct ZfpCompressor {
    mode: Mode,
}

impl ZfpCompressor {
    /// Construct for `preset`, following C++ `ConfigureZFP`:
    ///
    /// | Preset | C++ zfp options | Mode here |
    /// |---|---|---|
    /// | `Fast` | `zfp:mode=1`, `zfp:rate=8.0` | rate 8 bits/value |
    /// | `Balanced` | `zfp:mode=1`, `zfp:rate=16.0` | rate 16 bits/value |
    /// | `Best` | `zfp:mode=3`, `zfp:accuracy=1e-3` | accuracy 1e-3 |
    ///
    /// `Best` matches C++ `ctp::Zfp`'s hardcoded default tolerance of 1e-3.
    pub fn new(preset: Preset) -> Self {
        let mode = match preset {
            Preset::Fast => Mode::Rate(8.0),
            Preset::Balanced => Mode::Rate(16.0),
            Preset::Best => Mode::Accuracy(1e-3),
        };
        Self { mode }
    }

    /// A configured stream — every entry point starts here.
    fn stream(&self) -> Result<ZfpStream, CompressError> {
        check_word_size()?;
        let zfp = ZfpStream::open()?;
        zfp.configure(self.mode)?;
        Ok(zfp)
    }
}

impl Compressor for ZfpCompressor {
    /// Compress `input` (native-endian `f32`) into `output`; returns bytes
    /// written.
    ///
    /// `input.len()` must be a multiple of 4 (C++ rejects the same way). Empty
    /// input produces an empty stream: `Ok(0)`.
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        if input.is_empty() {
            // Nothing to encode; zfp is never asked for a zero-scalar field.
            return Ok(0);
        }
        if !input.len().is_multiple_of(SCALAR) {
            return Err(CompressError(format!(
                "zfp: input size {} must be a multiple of sizeof(float)={SCALAR}",
                input.len()
            )));
        }

        let zfp = self.stream()?;
        let mut scalars = floats_from_bytes(input);
        let field = ZfpField::new_1d(&mut scalars)?;

        let bufsize = zfp.maximum_size(&field);
        if bufsize == 0 {
            return Err(CompressError("zfp_stream_maximum_size returned 0".into()));
        }

        // zfp does not bounds-check its writes, so the scratch must hold the
        // worst case (`bufsize`), rounded up to whole bitstream words.
        let mut scratch = words_from_bytes(&[], bufsize);
        let n = {
            let bs = BitStream::open(&mut scratch)?;
            zfp.bind(&bs);
            // SAFETY: the stream is configured and bound to `bs`, whose buffer
            // holds at least the `zfp_stream_maximum_size` bytes zfp may write
            // for `field`; `field` describes `scalars`, which is live and
            // exclusively borrowed here.
            unsafe { zfp_compress(zfp.ptr, field.ptr) }
        };
        if n == 0 {
            return Err(CompressError("zfp_compress failed".into()));
        }
        if n > output.len() {
            return Err(CompressError(format!(
                "zfp: output buffer too small: need {n} bytes, have {}",
                output.len()
            )));
        }
        bytes_from_words(output, &scratch, n);
        Ok(n)
    }

    /// Decompress `input` into `output`; returns bytes written.
    ///
    /// The stream is headerless, so **`output.len()` is the exact expected
    /// decompressed size**, not a capacity — it must equal the original input
    /// length and be a multiple of 4. Empty `output` yields `Ok(0)`.
    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        if output.is_empty() {
            // Caller expects no scalars back; matches compress's empty stream.
            return Ok(0);
        }
        if !output.len().is_multiple_of(SCALAR) {
            return Err(CompressError(format!(
                "zfp: output size {} must be a multiple of sizeof(float)={SCALAR}",
                output.len()
            )));
        }
        if input.is_empty() {
            return Err(CompressError(
                "zfp: empty compressed input cannot fill a non-empty output".into(),
            ));
        }

        let zfp = self.stream()?;
        let mut scalars = vec![0f32; output.len() / SCALAR];
        // Scoped so `field`'s exclusive borrow of `scalars` (and with it zfp's
        // write access) ends before we read the decoded values back out.
        let n = {
            let field = ZfpField::new_1d(&mut scalars)?;
            // Size the scratch to the larger of the stream and zfp's own
            // worst-case read for this field, so a truncated/corrupt stream
            // reads zero padding rather than off the end of the allocation
            // (see divergence #3).
            let mut scratch = words_from_bytes(input, zfp.maximum_size(&field));
            let bs = BitStream::open(&mut scratch)?;
            zfp.bind(&bs);
            // SAFETY: the stream is configured identically to compress() and
            // bound to `bs`, whose buffer holds the whole input plus zero
            // padding up to zfp's maximum read for `field`; `field` describes
            // `scalars`, live and exclusively borrowed, sized to hold exactly
            // the scalars zfp will write.
            unsafe { zfp_decompress(zfp.ptr, field.ptr) }
        };
        if n == 0 {
            return Err(CompressError("zfp_decompress failed".into()));
        }

        for (dst, src) in output.chunks_exact_mut(SCALAR).zip(scalars.iter()) {
            dst.copy_from_slice(&src.to_ne_bytes());
        }
        Ok(output.len())
    }

    fn library(&self) -> LibraryId {
        LibraryId::Zfp
    }

    /// `zfp_stream_maximum_size` for a 1D float field of `input_size` bytes —
    /// the exact bound zfp itself uses, so `compress` into a buffer this large
    /// never reports "output buffer too small".
    fn max_compressed_size(&self, input_size: usize) -> usize {
        let fallback = input_size + input_size / 8 + 1024;
        let Ok(zfp) = self.stream() else {
            return fallback;
        };
        let Ok(field) = ZfpField::metadata_1d(input_size.div_ceil(SCALAR)) else {
            return fallback;
        };
        match zfp.maximum_size(&field) {
            0 => fallback,
            n => n,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor_for;

    /// Realistic scientific payload: a smooth, mildly noisy 1D signal — the
    /// shape zfp is designed for. 4096 f32 = 16 KiB.
    fn signal() -> Vec<f32> {
        (0..4096)
            .map(|i| {
                let x = i as f32 * 0.01;
                x.sin() * 10.0 + (x * 0.25).cos() * 2.5
            })
            .collect()
    }

    /// Incompressible counterweight: white noise over the same magnitude band
    /// as `signal()`, so a size comparison isolates compressibility rather than
    /// dynamic range. Deterministic LCG — no rand dependency.
    fn noise() -> Vec<f32> {
        let mut s: u64 = 0x1234_5678;
        (0..4096)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u32 << 31) as f32) * 6.5 - 3.25
            })
            .collect()
    }

    fn to_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_ne_bytes()).collect()
    }

    fn from_bytes(b: &[u8]) -> Vec<f32> {
        floats_from_bytes(b)
    }

    /// Round-trip helper: returns (compressed_len, decoded scalars).
    fn round_trip(preset: Preset, data: &[f32]) -> (usize, Vec<f32>) {
        let c = ZfpCompressor::new(preset);
        let input = to_bytes(data);
        let mut comp = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut comp, &input).expect("compress");

        let mut out = vec![0u8; input.len()];
        let m = c.decompress(&mut out, &comp[..n]).expect("decompress");
        assert_eq!(m, input.len(), "decompress fills the exact output size");
        (n, from_bytes(&out))
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn best_preset_holds_the_accuracy_tolerance() {
        // BEST is accuracy mode at 1e-3: zfp bounds the ABSOLUTE error, so this
        // is a real guarantee, not a heuristic. (2x slack for the f32 rounding
        // in our own byte staging.)
        let data = signal();
        let (n, decoded) = round_trip(Preset::Best, &data);
        let err = max_abs_err(&data, &decoded);
        assert!(
            err <= 2e-3,
            "accuracy-mode error {err} exceeds the 1e-3 tolerance"
        );
        assert!(
            err > 0.0,
            "lossy codec should not be bit-exact on this data"
        );
        assert!(
            n < data.len() * SCALAR,
            "compressed {n} vs raw {}",
            data.len() * SCALAR
        );
    }

    #[test]
    fn fixed_rate_presets_hit_their_bit_budget() {
        // Fixed-rate mode's guarantee is the SIZE, not the error: 8 bits/value
        // on 32-bit floats is exactly 4x, 16 bits/value exactly 2x. Checking
        // the ratio is what pins the C++ preset table (rate 8.0 / 16.0).
        let data = signal();
        let raw = data.len() * SCALAR;

        let (fast_n, fast) = round_trip(Preset::Fast, &data);
        let (bal_n, bal) = round_trip(Preset::Balanced, &data);

        // Allow a few words of stream padding on top of the exact bit budget.
        assert!(
            (fast_n as f64) < raw as f64 / 3.9 && fast_n >= raw / 5,
            "FAST (8 bits/value) should be ~4x: {fast_n} vs {raw}"
        );
        assert!(
            (bal_n as f64) < raw as f64 / 1.95 && bal_n >= raw / 3,
            "BALANCED (16 bits/value) should be ~2x: {bal_n} vs {raw}"
        );
        assert!(bal_n > fast_n, "more bits/value must cost more bytes");

        // More bits/value ⇒ strictly better fidelity on the same data.
        let fast_err = max_abs_err(&data, &fast);
        let bal_err = max_abs_err(&data, &bal);
        assert!(
            bal_err < fast_err,
            "BALANCED err {bal_err} should beat FAST err {fast_err}"
        );
        // Rate-implied tolerance: the signal spans ~±12.5, and 16 bits/value
        // keeps it well under 1% of that range.
        assert!(
            bal_err < 0.125,
            "BALANCED err {bal_err} too large for 16 bits/value"
        );
    }

    #[test]
    fn highly_compressible_data_shrinks() {
        let flat = vec![3.25f32; 4096];
        let raw = flat.len() * SCALAR;

        // Every preset shrinks a constant field, and reproduces it well within
        // tolerance (zfp is in fact bit-exact on it in all three modes, but
        // that is not a documented guarantee, so we only assert the bound).
        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let (n, decoded) = round_trip(preset, &flat);
            assert!(n < raw, "{preset:?}: {n} did not shrink {raw}");
            assert!(
                max_abs_err(&flat, &decoded) <= 2e-3,
                "{preset:?} exceeded tolerance"
            );
        }

        // Only accuracy mode's SIZE responds to compressibility — the fixed-rate
        // presets spend the same bit budget on any data whatsoever. So the real
        // "compressible beats incompressible" claim is a BEST-preset claim.
        let flat_n = round_trip(Preset::Best, &flat).0;
        let noise_n = round_trip(Preset::Best, &noise()).0;
        assert!(
            flat_n < raw / 2,
            "constant field should shrink >2x under accuracy mode: {flat_n} vs {raw}"
        );
        assert!(
            flat_n * 4 < noise_n * 3,
            "constant field ({flat_n}) should beat same-range noise ({noise_n}) by >25%"
        );

        // ...and the converse, which pins FAST/BALANCED to rate mode rather than
        // accuracy mode: identical size for constant and for noise.
        assert_eq!(
            round_trip(Preset::Balanced, &flat).0,
            round_trip(Preset::Balanced, &noise()).0,
            "fixed-rate output size must not depend on the data"
        );
    }

    #[test]
    fn output_buffer_too_small_returns_err() {
        // Must be a clean Err, not a buffer overrun: zfp itself does no bounds
        // checking, so this is the guard that keeps compress() safe.
        let c = ZfpCompressor::new(Preset::Balanced);
        let data = signal();
        let input = to_bytes(&data);

        for cap in [0usize, 1, 8, 64, input.len() / 8] {
            let mut tiny = vec![0xAAu8; cap];
            let err = c
                .compress(&mut tiny, &input)
                .expect_err("undersized output must fail");
            assert!(err.0.contains("too small"), "unexpected error: {err}");
            // Nothing was written into the caller's buffer.
            assert!(tiny.iter().all(|&b| b == 0xAA), "buffer was scribbled on");
        }

        // One byte short of the real size still fails; the exact size succeeds.
        let mut exact = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut exact, &input).unwrap();
        assert!(c.compress(&mut vec![0u8; n - 1], &input).is_err());
        assert!(c.compress(&mut vec![0u8; n], &input).is_ok());
    }

    #[test]
    fn empty_input_and_empty_output_are_handled() {
        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let c = ZfpCompressor::new(preset);

            // Empty input ⇒ empty stream, even with a zero-length output.
            assert_eq!(c.compress(&mut [], &[]).unwrap(), 0);
            assert_eq!(c.compress(&mut [0u8; 16], &[]).unwrap(), 0);

            // Empty output ⇒ nothing to write.
            assert_eq!(c.decompress(&mut [], &[]).unwrap(), 0);
            assert_eq!(c.decompress(&mut [], &[1, 2, 3, 4]).unwrap(), 0);

            // Full empty round-trip.
            let n = c.compress(&mut [0u8; 16], &[]).unwrap();
            assert_eq!(c.decompress(&mut [], &[0u8; 16][..n]).unwrap(), 0);

            // A non-empty output cannot be filled from an empty stream.
            assert!(c.decompress(&mut [0u8; 4], &[]).is_err());
        }
    }

    #[test]
    fn non_float_sized_buffers_are_rejected() {
        // Mirrors the C++ "must be a multiple of sizeof(float)" checks.
        let c = ZfpCompressor::new(Preset::Balanced);
        for bad in [1usize, 2, 3, 5, 7] {
            let err = c
                .compress(&mut [0u8; 256], &vec![0u8; bad])
                .expect_err("ragged input must fail");
            assert!(err.0.contains("multiple of sizeof(float)"), "{err}");

            let err = c
                .decompress(&mut vec![0u8; bad], &[0u8; 256])
                .expect_err("ragged output must fail");
            assert!(err.0.contains("multiple of sizeof(float)"), "{err}");
        }
    }

    #[test]
    fn unaligned_buffers_round_trip() {
        // The point of the scratch staging (divergence #2): callers hand us
        // &[u8] with no alignment guarantee. Offsetting by 1 byte makes both
        // sides misaligned for f32 and for the 64-bit bitstream word.
        let data = signal();
        let bytes = to_bytes(&data);
        let c = ZfpCompressor::new(Preset::Best);

        let mut src = vec![0u8; bytes.len() + 1];
        src[1..].copy_from_slice(&bytes);
        let mut comp = vec![0u8; c.max_compressed_size(bytes.len()) + 1];
        let n = c.compress(&mut comp[1..], &src[1..]).unwrap();

        let mut out = vec![0u8; bytes.len() + 1];
        c.decompress(&mut out[1..], &comp[1..1 + n]).unwrap();
        assert!(max_abs_err(&data, &from_bytes(&out[1..])) <= 2e-3);
    }

    #[test]
    fn max_compressed_size_bounds_every_preset() {
        let data = signal();
        let input = to_bytes(&data);
        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let c = ZfpCompressor::new(preset);
            let bound = c.max_compressed_size(input.len());
            let mut buf = vec![0u8; bound];
            let n = c.compress(&mut buf, &input).unwrap();
            assert!(n <= bound, "{preset:?}: produced {n} > bound {bound}");
            // The bound must survive a ragged (non-multiple-of-4) size query.
            assert!(c.max_compressed_size(0) > 0);
            assert!(c.max_compressed_size(7) > 0);
        }
    }

    #[test]
    fn factory_builds_this_type() {
        // The factory arm in lib.rs must resolve to ZfpCompressor.
        let c = compressor_for(LibraryId::Zfp, Preset::Best).expect("zfp feature is on");
        assert_eq!(c.library(), LibraryId::Zfp);
        assert!(c.library().is_lossy());

        let data = signal();
        let input = to_bytes(&data);
        let mut comp = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut comp, &input).unwrap();
        let mut out = vec![0u8; input.len()];
        c.decompress(&mut out, &comp[..n]).unwrap();
        assert!(max_abs_err(&data, &from_bytes(&out)) <= 2e-3);
    }

    #[test]
    fn compressor_is_send_and_sync() {
        // The trait demands Send + Sync; each call owns all of its zfp state.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ZfpCompressor>();
    }
}
