// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! FFI wrapper for **zstd** (Facebook/Meta Zstandard, `libzstd`).
//!
//! Wraps the *simple* one-shot C API — `ZSTD_compress` / `ZSTD_decompress`
//! — mirroring C++ `ctp::ZstdWithModes` (`clio_ctp/compress/lossless_modes.h`).
//! Bindings are hand-rolled; no bindgen, no `zstd-sys`.
//!
//! # Version / ABI assumptions
//!
//! * Linked against the system `libzstd` (SONAME `libzstd.so.1`; the
//!   devcontainer ships 1.5.5). Only the **stable** section of `zstd.h` is
//!   used — every function bound here has been ABI-stable since 1.0.0, so
//!   any `libzstd.so.1` satisfies this module. No experimental/staging
//!   symbols (`ZSTD_c_*`, `ZSTD_CCtx_setParameter`, …) are referenced, so
//!   `ZSTD_STATIC_LINKING_ONLY` is irrelevant here.
//! * `size_t` is assumed to be `usize` and `unsigned` to be `c_uint`, which
//!   holds on every platform CTP builds for (LP64 Linux, LLP64 Windows).
//! * Return values are `size_t` where an *error* is encoded as a huge
//!   sentinel value, NOT as a small/zero size: errors are detectable only
//!   via `ZSTD_isError`, never by comparing against 0.
//! * The simple API is stateless and re-entrant (each call builds its own
//!   internal context), so [`ZstdCompressor`] is `Send + Sync` — it holds
//!   nothing but an `i32` level.
//!
//! # Framing
//!
//! Output is a standard zstd frame (magic `0xFD2FB528`) with the content
//! size written in the header, byte-identical to what the C++ wrapper
//! produces at the same level — blobs written by either stack are readable
//! by the other, and by the `zstd(1)` CLI. No CTP-specific framing is added.
//!
//! # Preset → level (matches C++ `ZstdWithModes`)
//!
//! | [`Preset`] | level | C++ rationale |
//! |---|---|---|
//! | `Fast` | 1 | fast compression |
//! | `Balanced` | 3 | zstd's own default |
//! | `Best` | 19 | high, but not 22 — avoids extreme slowdown |
//!
//! # Divergence from the C++ wrapper
//!
//! 1. **Error detection.** C++ reports success as `output_size != 0` after
//!    assigning the raw `ZSTD_compress` return. Because zstd encodes errors
//!    as enormous `size_t` sentinels, that test treats *every* error as
//!    success and hands back a bogus, huge `output_size`. This module
//!    checks `ZSTD_isError` and returns [`CompressError`] instead.
//! 2. **Tight output buffers are allowed.** C++ `ZstdWithModes::Compress`
//!    refuses (`return false`) whenever `output_size < ZSTD_compressBound()`,
//!    even when the data would comfortably fit. Here the buffer is passed to
//!    zstd as-is: a genuinely too-small buffer surfaces as zstd's own
//!    `dstSize_tooSmall` error (`Err`, never a partial/corrupt write), while
//!    a tight-but-sufficient buffer succeeds. Strictly more permissive; the
//!    safety property (no overrun) is enforced by zstd via `dstCapacity`.
//! 3. **Empty payloads round-trip.** Decompressing the 9-byte empty frame
//!    yields 0 bytes, which C++ misreports as failure (`output_size != 0`);
//!    this module returns `Ok(0)`.
//! 4. The legacy `ctp::Zstd` (`compress/zstd.h`) ignores presets entirely and
//!    always uses `ZSTD_maxCLevel()` (22). This module follows the
//!    preset-aware `ZstdWithModes` instead, per the crate's [`Preset`] API.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

use crate::{CompressError, Compressor, LibraryId, Preset};

// ---------------------------------------------------------------------------
// Raw bindings — stable section of <zstd.h>
// ---------------------------------------------------------------------------

extern "C" {
    /// `size_t ZSTD_compress(void* dst, size_t dstCapacity, const void* src,
    ///                       size_t srcSize, int compressionLevel);`
    /// Returns compressed size, or an error code (test `ZSTD_isError`).
    fn ZSTD_compress(
        dst: *mut c_void,
        dst_capacity: usize,
        src: *const c_void,
        src_size: usize,
        compression_level: c_int,
    ) -> usize;

    /// `size_t ZSTD_decompress(void* dst, size_t dstCapacity, const void* src,
    ///                         size_t compressedSize);`
    /// Returns decompressed size, or an error code (test `ZSTD_isError`).
    fn ZSTD_decompress(
        dst: *mut c_void,
        dst_capacity: usize,
        src: *const c_void,
        compressed_size: usize,
    ) -> usize;

    /// `size_t ZSTD_compressBound(size_t srcSize);` — worst-case bound;
    /// returns 0 when `srcSize` exceeds zstd's max input size.
    fn ZSTD_compressBound(src_size: usize) -> usize;

    /// `unsigned ZSTD_isError(size_t code);` — nonzero when `code` is an error.
    fn ZSTD_isError(code: usize) -> c_uint;

    /// `const char* ZSTD_getErrorName(size_t code);` — static NUL-terminated
    /// string, valid for any `code` (returns "No error detected" if not one).
    fn ZSTD_getErrorName(code: usize) -> *const c_char;

    /// `int ZSTD_maxCLevel(void);` — highest level this build accepts (22).
    fn ZSTD_maxCLevel() -> c_int;

    /// `unsigned ZSTD_versionNumber(void);` — MAJOR*100*100 + MINOR*100 + PATCH.
    fn ZSTD_versionNumber() -> c_uint;
}

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

/// Convert a zstd `size_t` return into a `Result`, since zstd signals errors
/// with sentinel sizes rather than a separate status.
fn zstd_check(rc: usize, what: &str) -> Result<usize, CompressError> {
    // SAFETY: `ZSTD_isError` is a pure predicate over an integer; it
    // dereferences nothing and has no preconditions.
    if unsafe { ZSTD_isError(rc) } == 0 {
        return Ok(rc);
    }
    // SAFETY: `ZSTD_getErrorName` accepts any code and returns a pointer to a
    // static, NUL-terminated string literal inside libzstd — never null, and
    // valid for the process lifetime, so building a CStr over it is sound.
    let name = unsafe { CStr::from_ptr(ZSTD_getErrorName(rc)) }.to_string_lossy();
    Err(CompressError(format!("{what}: {name}")))
}

/// Runtime `libzstd` version as `(major, minor, patch)`.
///
/// Useful for logging which ABI actually got loaded, since the wrapper only
/// requires "some `libzstd.so.1`".
pub fn version() -> (u32, u32, u32) {
    // SAFETY: no arguments, no preconditions; returns a plain integer.
    let v = unsafe { ZSTD_versionNumber() } as u32;
    (v / 10_000, (v / 100) % 100, v % 100)
}

// ---------------------------------------------------------------------------
// ZstdCompressor
// ---------------------------------------------------------------------------

/// Zstandard codec — the Rust face of C++ `ctp::ZstdWithModes`.
///
/// Stateless and cheap to construct; shareable across threads.
#[derive(Debug, Clone, Copy)]
pub struct ZstdCompressor {
    level: c_int,
}

impl ZstdCompressor {
    /// Build a compressor for `preset` (levels per the module table).
    ///
    /// The level is clamped to `ZSTD_maxCLevel()` so a `libzstd` built with a
    /// reduced level ceiling degrades instead of failing every call.
    pub fn new(preset: Preset) -> Self {
        let want: c_int = match preset {
            Preset::Fast => 1,     // C++: fast compression
            Preset::Balanced => 3, // C++: zstd's default level
            Preset::Best => 19,    // C++: high, not max, to avoid slowdown
        };
        // SAFETY: no arguments, no preconditions; returns a plain integer.
        let max = unsafe { ZSTD_maxCLevel() };
        Self {
            level: want.min(max),
        }
    }

    /// The zstd compression level this instance uses.
    pub fn level(&self) -> c_int {
        self.level
    }
}

impl Compressor for ZstdCompressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        // SAFETY: `output`/`input` are Rust slices, so each pointer is valid
        // for exactly the length passed alongside it (and well-aligned — u8
        // has alignment 1). zstd writes at most `dst_capacity` bytes and
        // reads at most `src_size`; it never retains either pointer past the
        // call. The regions cannot overlap: `output` is a unique borrow.
        // Empty slices yield dangling-but-nonnull pointers, which zstd never
        // dereferences because the matching length is 0.
        let rc = unsafe {
            ZSTD_compress(
                output.as_mut_ptr() as *mut c_void,
                output.len(),
                input.as_ptr() as *const c_void,
                input.len(),
                self.level,
            )
        };
        // A too-small `output` comes back as dstSize_tooSmall → Err, with
        // nothing written past `output.len()`.
        zstd_check(rc, "ZSTD_compress")
    }

    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        // SAFETY: same contract as `compress` above — pointer/length pairs
        // derive from live slices, zstd honours `dst_capacity` and does not
        // escape the pointers. Malformed or truncated `input` is rejected by
        // zstd's frame checks and surfaces as an error code, not UB.
        let rc = unsafe {
            ZSTD_decompress(
                output.as_mut_ptr() as *mut c_void,
                output.len(),
                input.as_ptr() as *const c_void,
                input.len(),
            )
        };
        zstd_check(rc, "ZSTD_decompress")
    }

    fn library(&self) -> LibraryId {
        LibraryId::Zstd
    }

    fn max_compressed_size(&self, input_size: usize) -> usize {
        // SAFETY: pure arithmetic over an integer; no preconditions.
        let bound = unsafe { ZSTD_compressBound(input_size) };
        if bound == 0 {
            // `input_size` exceeds zstd's max input; no bound exists. Fall
            // back to the trait's conservative estimate rather than return a
            // 0 that callers would read as "allocate nothing".
            return input_size + input_size / 8 + 1024;
        }
        bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compressor_by_name, compressor_for};

    /// Realistic-ish payload: structured log lines (compressible) spliced
    /// with a pseudo-random binary tail (incompressible), so the round-trip
    /// exercises both literal and match paths.
    fn realistic_data() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..4096u32 {
            v.extend_from_slice(
                format!(
                    "[2026-07-16T12:00:{:02}Z] INFO worker={} blob_id={} bytes={} ok\n",
                    i % 60,
                    i % 8,
                    i,
                    (i * 137) % 65536
                )
                .as_bytes(),
            );
        }
        // xorshift tail — defeats the entropy coder, keeps the test honest.
        let mut x: u32 = 0x1234_5678;
        for _ in 0..16384 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            v.push((x & 0xff) as u8);
        }
        v
    }

    #[test]
    fn round_trip_is_byte_exact_for_every_preset() {
        let data = realistic_data();
        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let c = ZstdCompressor::new(preset);
            let mut buf = vec![0u8; c.max_compressed_size(data.len())];
            let n = c.compress(&mut buf, &data).unwrap();

            let mut back = vec![0u8; data.len()];
            let m = c.decompress(&mut back, &buf[..n]).unwrap();

            assert_eq!(m, data.len(), "{preset:?}: decompressed length");
            assert_eq!(back, data, "{preset:?}: zstd is lossless — must be exact");
        }
    }

    #[test]
    fn preset_levels_match_the_cpp_wrapper() {
        assert_eq!(ZstdCompressor::new(Preset::Fast).level(), 1);
        assert_eq!(ZstdCompressor::new(Preset::Balanced).level(), 3);
        assert_eq!(ZstdCompressor::new(Preset::Best).level(), 19);
    }

    #[test]
    fn highly_compressible_data_actually_shrinks() {
        let data = vec![b'A'; 1 << 20]; // 1 MiB of one byte
        let c = ZstdCompressor::new(Preset::Balanced);
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();

        assert!(
            n < data.len() / 100,
            "1 MiB of 'A' should collapse, got {n} bytes"
        );

        let mut back = vec![0u8; data.len()];
        let m = c.decompress(&mut back, &buf[..n]).unwrap();
        assert_eq!(m, data.len());
        assert_eq!(back, data);
    }

    #[test]
    fn output_buffer_too_small_returns_err_without_corruption() {
        let data = realistic_data();
        let c = ZstdCompressor::new(Preset::Balanced);

        // Compression into a buffer that cannot possibly hold the frame.
        // Guard bytes flank the 16-byte window; zstd must not touch them.
        let mut guarded = [0xAAu8; 48];
        {
            let window = &mut guarded[16..32];
            let err = c.compress(window, &data).unwrap_err();
            assert!(
                err.0.contains("ZSTD_compress"),
                "expected a compress error, got {err}"
            );
        }
        assert!(
            guarded[..16].iter().all(|&b| b == 0xAA) && guarded[32..].iter().all(|&b| b == 0xAA),
            "zstd wrote outside the destination capacity"
        );

        // Decompression into a too-small buffer: valid frame, short output.
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();
        let mut tiny = vec![0u8; data.len() / 2];
        let err = c.decompress(&mut tiny, &buf[..n]).unwrap_err();
        assert!(
            err.0.contains("ZSTD_decompress"),
            "expected a decompress error, got {err}"
        );
    }

    #[test]
    fn corrupt_input_is_an_error_not_a_panic() {
        let c = ZstdCompressor::new(Preset::Balanced);
        let mut out = vec![0u8; 1024];
        // Not a zstd frame at all.
        assert!(c
            .decompress(&mut out, b"definitely not a zstd frame")
            .is_err());

        // Valid frame, truncated mid-way.
        let data = realistic_data();
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();
        let mut back = vec![0u8; data.len()];
        assert!(c.decompress(&mut back, &buf[..n / 2]).is_err());
    }

    #[test]
    fn empty_input_round_trips() {
        let c = ZstdCompressor::new(Preset::Balanced);
        let mut buf = vec![0u8; c.max_compressed_size(0)];
        let n = c.compress(&mut buf, &[]).unwrap();
        // An empty frame is still a frame: header + magic, never zero bytes.
        assert!(n > 0, "empty input must still emit a valid frame");

        // Decompress into a zero-length buffer — 0 bytes out, and Ok(0),
        // where the C++ wrapper's `output_size != 0` test reports failure.
        let m = c.decompress(&mut [], &buf[..n]).unwrap();
        assert_eq!(m, 0);

        // A larger destination must also work and consume no capacity.
        let mut roomy = vec![0xCDu8; 8];
        assert_eq!(c.decompress(&mut roomy, &buf[..n]).unwrap(), 0);
        assert_eq!(roomy, vec![0xCDu8; 8], "nothing should have been written");
    }

    #[test]
    fn compress_bound_covers_incompressible_input() {
        let c = ZstdCompressor::new(Preset::Balanced);
        // Incompressible data expands slightly; the bound must still hold.
        let mut x: u32 = 0xDEAD_BEEF;
        let data: Vec<u8> = (0..8192)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x & 0xff) as u8
            })
            .collect();
        let bound = c.max_compressed_size(data.len());
        assert!(bound >= data.len());
        let mut buf = vec![0u8; bound];
        let n = c.compress(&mut buf, &data).unwrap();
        assert!(n <= bound);
    }

    #[test]
    fn factory_builds_zstd_and_reports_its_library() {
        let c = compressor_for(LibraryId::Zstd, Preset::Fast).expect("zstd feature is on");
        assert_eq!(c.library(), LibraryId::Zstd);
        assert!(!c.library().is_lossy());

        let by_name = compressor_by_name("zstd", Preset::Best).expect("named lookup");
        assert_eq!(by_name.library(), LibraryId::Zstd);

        // Round-trip through the boxed trait object, as callers use it.
        let data = b"factory-built codec must actually work".repeat(32);
        let mut buf = vec![0u8; by_name.max_compressed_size(data.len())];
        let n = by_name.compress(&mut buf, &data).unwrap();
        let mut back = vec![0u8; data.len()];
        let m = by_name.decompress(&mut back, &buf[..n]).unwrap();
        assert_eq!(&back[..m], &data[..]);
    }

    #[test]
    fn frames_are_interoperable_across_presets() {
        // Level is an encoder-side choice only: any preset decodes any frame,
        // which is what makes `BlobInfo::compress_lib_` sufficient metadata.
        let data = realistic_data();
        let enc = ZstdCompressor::new(Preset::Best);
        let mut buf = vec![0u8; enc.max_compressed_size(data.len())];
        let n = enc.compress(&mut buf, &data).unwrap();

        let dec = ZstdCompressor::new(Preset::Fast);
        let mut back = vec![0u8; data.len()];
        let m = dec.decompress(&mut back, &buf[..n]).unwrap();
        assert_eq!(&back[..m], &data[..]);

        // And the frame really is standard zstd: magic 0xFD2FB528, LE.
        assert_eq!(&buf[..4], &[0x28, 0xB5, 0x2F, 0xFD]);
    }

    #[test]
    fn links_against_a_libzstd_1_x() {
        let (major, _minor, _patch) = version();
        assert_eq!(major, 1, "wrapper targets the libzstd.so.1 ABI");
    }
}
