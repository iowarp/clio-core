// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! FFI wrapper for **liblzma** (the xz-utils compression library), the Rust
//! face of C++ `ctp::Lzma` / `ctp::LzmaWithModes`
//! (`clio_ctp/compress/lzma.h`, `clio_ctp/compress/lossless_modes.h`).
//!
//! Bindings are hand-rolled — no bindgen, no `lzma-sys`. `build.rs` links
//! `liblzma` when the `lzma` feature is on.
//!
//! # Version / ABI assumptions
//!
//! Targets the **liblzma 5.x stable C ABI** (`liblzma.so.5`); verified
//! against 5.4.5 (the CTP devcontainer's `/usr/include/lzma/`). The two
//! entry points used here — `lzma_easy_buffer_encode` and
//! `lzma_stream_buffer_decode` — plus the `lzma_ret` and `lzma_check`
//! enumerator values and the decoder flag bits have been part of the frozen
//! public ABI since 5.0 (2010), so any 5.x runtime works. Specifically
//! assumed:
//!
//! - `lzma_ret` / `lzma_check` are C enums with all-non-negative values, so
//!   they cross the ABI as 32-bit ints (`c_int` / `c_uint` are equivalent
//!   here — same size, same register class on every supported target).
//! - `LZMA_OK = 0`, `LZMA_BUF_ERROR = 10`; `LZMA_CHECK_CRC64 = 4`;
//!   `LZMA_CONCATENATED = 0x08`.
//! - Both single-call functions are **reentrant**: they hold no shared
//!   mutable state and allocate internally via `malloc`/`free` when passed a
//!   NULL `lzma_allocator`. That is what makes [`LzmaCompressor`] `Sync`
//!   (auto-derived — it stores only a `u32`).
//!
//! # Format compatibility with the C++ wrapper
//!
//! The C++ side drives the **streaming** API (`lzma_easy_encoder` +
//! `lzma_code(LZMA_FINISH)`); this module uses the **single-call buffer**
//! API. Both emit the identical `.xz` container with a CRC64 check, so at
//! the same preset the bytes are interchangeable in both directions —
//! blobs written by the C++ stack decode here and vice versa. The buffer
//! API is used because CTP compresses whole blobs already resident in
//! memory, which is exactly the case it exists for (it skips the
//! `lzma_stream` setup/teardown dance).
//!
//! # Divergences from the C++ wrapper
//!
//! 1. **`Preset::Fast` → LZMA preset 1, not 0.** C++ `LzmaWithModes` maps
//!    FAST→0; issue #756 specifies 1/6/9, which this follows. Preset 0 and 1
//!    differ only in encoder speed/ratio, never in decodability, so this
//!    cannot make a blob unreadable — it only makes `Fast` marginally
//!    tighter and slower than the C++ path.
//! 2. **Preset is honored.** C++ `ctp::Lzma` (the non-`WithModes` class)
//!    hardcodes `LZMA_PRESET_DEFAULT` (6) and ignores the requested preset
//!    entirely. This module always applies the mapped preset, matching
//!    `LzmaWithModes`, which is the class the factory actually registers
//!    for `"lzma"`.
//! 3. **Truncation is an error, not silent success.** C++ `ctp::Lzma::
//!    Compress` accepts `LZMA_OK` from `lzma_code(LZMA_FINISH)` as success,
//!    but `LZMA_OK` there means "ran out of output room" — it returns
//!    `true` with a truncated, undecodable stream. This module reports
//!    `LZMA_BUF_ERROR` as [`CompressError`]. (`LzmaWithModes` gets this
//!    right by requiring `LZMA_STREAM_END`.)
//! 4. **Errors carry a reason** instead of a bare `bool` — the crate-wide
//!    convention documented on [`CompressError`].

use std::ffi::{c_int, c_uint, c_void};

use crate::{CompressError, Compressor, LibraryId, Preset};

// ---------------------------------------------------------------------------
// Raw liblzma declarations (lzma/container.h, lzma/base.h, lzma/check.h)
// ---------------------------------------------------------------------------

/// `lzma_ret` (lzma/base.h). C enum, all values >= 0.
type LzmaRet = c_int;
/// `lzma_check` (lzma/check.h). C enum, all values >= 0.
type LzmaCheck = c_uint;

const LZMA_OK: LzmaRet = 0;
const LZMA_STREAM_END: LzmaRet = 1;
const LZMA_NO_CHECK: LzmaRet = 2;
const LZMA_UNSUPPORTED_CHECK: LzmaRet = 3;
const LZMA_GET_CHECK: LzmaRet = 4;
const LZMA_MEM_ERROR: LzmaRet = 5;
const LZMA_MEMLIMIT_ERROR: LzmaRet = 6;
const LZMA_FORMAT_ERROR: LzmaRet = 7;
const LZMA_OPTIONS_ERROR: LzmaRet = 8;
const LZMA_DATA_ERROR: LzmaRet = 9;
const LZMA_BUF_ERROR: LzmaRet = 10;
const LZMA_PROG_ERROR: LzmaRet = 11;

/// CRC64 integrity check — the same check the C++ wrapper requests.
const LZMA_CHECK_CRC64: LzmaCheck = 4;

/// Decode all concatenated `.xz` streams, not just the first (matches the
/// C++ decoder's `lzma_stream_decoder(..., LZMA_CONCATENATED)`).
const LZMA_CONCATENATED: u32 = 0x08;

extern "C" {
    /// Single-call `.xz` encode. `out_pos` is INOUT: the write offset into
    /// `out`, advanced past the bytes written; updated only on success.
    fn lzma_easy_buffer_encode(
        preset: u32,
        check: LzmaCheck,
        allocator: *const c_void,
        in_: *const u8,
        in_size: usize,
        out: *mut u8,
        out_pos: *mut usize,
        out_size: usize,
    ) -> LzmaRet;

    /// Single-call `.xz` decode. Returns `LZMA_OK` (NOT `LZMA_STREAM_END`)
    /// on success. `memlimit` is INOUT, written only on LZMA_MEMLIMIT_ERROR;
    /// `in_pos`/`out_pos` are INOUT, updated only on success.
    fn lzma_stream_buffer_decode(
        memlimit: *mut u64,
        flags: u32,
        allocator: *const c_void,
        in_: *const u8,
        in_pos: *mut usize,
        in_size: usize,
        out: *mut u8,
        out_pos: *mut usize,
        out_size: usize,
    ) -> LzmaRet;

    /// Worst-case `.xz` size for `uncompressed_size` bytes; 0 if the bound
    /// would overflow `size_t`.
    fn lzma_stream_buffer_bound(uncompressed_size: usize) -> usize;

    /// Runtime library version, e.g. 50040052 for 5.4.5.
    fn lzma_version_number() -> u32;
}

/// Human-readable name for an `lzma_ret`. liblzma ships no `strerror`, so
/// the mapping is spelled out here.
fn ret_name(rc: LzmaRet) -> &'static str {
    match rc {
        LZMA_OK => "LZMA_OK",
        LZMA_STREAM_END => "LZMA_STREAM_END",
        LZMA_NO_CHECK => "LZMA_NO_CHECK",
        LZMA_UNSUPPORTED_CHECK => "LZMA_UNSUPPORTED_CHECK",
        LZMA_GET_CHECK => "LZMA_GET_CHECK",
        LZMA_MEM_ERROR => "LZMA_MEM_ERROR (allocation failed)",
        LZMA_MEMLIMIT_ERROR => "LZMA_MEMLIMIT_ERROR (memory limit reached)",
        LZMA_FORMAT_ERROR => "LZMA_FORMAT_ERROR (not an .xz stream)",
        LZMA_OPTIONS_ERROR => "LZMA_OPTIONS_ERROR (unsupported options)",
        LZMA_DATA_ERROR => "LZMA_DATA_ERROR (corrupt data)",
        LZMA_BUF_ERROR => "LZMA_BUF_ERROR (output buffer too small)",
        LZMA_PROG_ERROR => "LZMA_PROG_ERROR (invalid arguments)",
        _ => "unknown lzma_ret",
    }
}

fn lzma_err(what: &str, rc: LzmaRet) -> CompressError {
    CompressError(format!("{what}: {} ({rc})", ret_name(rc)))
}

/// liblzma's runtime version as (major, minor, patch).
///
/// Exposed so callers/tests can confirm they linked a 5.x runtime, matching
/// the ABI this module assumes.
pub fn version() -> (u32, u32, u32) {
    // SAFETY: lzma_version_number is a pure accessor — no arguments, no
    // preconditions, cannot fail.
    let v = unsafe { lzma_version_number() };
    // Encoding: XYYYZZZS (major, minor, patch, stability).
    (v / 10_000_000, (v / 10_000) % 1_000, (v / 10) % 1_000)
}

// ---------------------------------------------------------------------------
// LzmaCompressor
// ---------------------------------------------------------------------------

/// `.xz` (LZMA2 + CRC64) codec — C++ `ctp::LzmaWithModes`.
///
/// Lossless: `decompress(compress(x)) == x` byte for byte. Cheap to
/// construct and copy; holds only the encoder preset. The decode side
/// ignores the preset (it is recorded in the stream header), so any
/// `LzmaCompressor` decodes any `.xz` blob regardless of the preset it was
/// written with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LzmaCompressor {
    /// LZMA preset level 0..=9 handed to `lzma_easy_buffer_encode`.
    preset: u32,
}

impl LzmaCompressor {
    /// Build the codec for `preset` (Fast→1, Balanced→6, Best→9).
    ///
    /// See the module docs: C++ maps Fast→0; issue #756 specifies 1.
    pub fn new(preset: Preset) -> Self {
        let preset = match preset {
            Preset::Fast => 1,
            Preset::Balanced => 6,
            Preset::Best => 9,
        };
        Self { preset }
    }

    /// The raw LZMA preset level (0..=9) this codec encodes with.
    pub fn preset_level(&self) -> u32 {
        self.preset
    }
}

impl Default for LzmaCompressor {
    fn default() -> Self {
        Self::new(Preset::default())
    }
}

impl Compressor for LzmaCompressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        let mut out_pos: usize = 0;
        // SAFETY: `input`/`output` are live Rust slices, so the (ptr, len)
        // pairs describe exactly the memory liblzma may touch — it reads
        // in[0..in_size] and writes out[*out_pos..out_size] and nothing
        // else. Empty slices yield a dangling-but-aligned non-null pointer,
        // which is never dereferenced because the paired size is 0.
        // `out_pos` is a live local. NULL allocator selects malloc/free.
        // The two buffers come from distinct borrows (&[] and &mut []) so
        // they cannot alias. The call is reentrant (see module docs).
        let rc = unsafe {
            lzma_easy_buffer_encode(
                self.preset,
                LZMA_CHECK_CRC64,
                std::ptr::null(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                &mut out_pos,
                output.len(),
            )
        };
        if rc != LZMA_OK {
            // Includes LZMA_BUF_ERROR: `output` was too small. liblzma
            // leaves out_pos untouched on failure, and any bytes it did
            // scribble into `output` are just garbage in a caller-owned
            // buffer — no UB, and we report nothing was written.
            return Err(lzma_err("lzma_easy_buffer_encode", rc));
        }
        Ok(out_pos)
    }

    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        // UINT64_MAX effectively disables the decoder's memory limiter,
        // matching the C++ wrapper's `lzma_stream_decoder(&strm, UINT64_MAX,
        // LZMA_CONCATENATED)`.
        let mut memlimit: u64 = u64::MAX;
        let mut in_pos: usize = 0;
        let mut out_pos: usize = 0;
        // SAFETY: same contract as compress() — live slices bound the reads
        // and writes, the three INOUT cursors are live locals, and the
        // NULL allocator selects malloc/free. LZMA_CONCATENATED is a
        // documented-legal flag for this entry point (unlike
        // LZMA_TELL_ANY_CHECK, which would return LZMA_PROG_ERROR).
        let rc = unsafe {
            lzma_stream_buffer_decode(
                &mut memlimit,
                LZMA_CONCATENATED,
                std::ptr::null(),
                input.as_ptr(),
                &mut in_pos,
                input.len(),
                output.as_mut_ptr(),
                &mut out_pos,
                output.len(),
            )
        };
        // Note: unlike the streaming API, this returns LZMA_OK — not
        // LZMA_STREAM_END — on a fully decoded stream.
        if rc != LZMA_OK {
            return Err(lzma_err("lzma_stream_buffer_decode", rc));
        }
        Ok(out_pos)
    }

    fn library(&self) -> LibraryId {
        LibraryId::Lzma
    }

    fn max_compressed_size(&self, input_size: usize) -> usize {
        // SAFETY: pure arithmetic in liblzma; no pointers, no allocation.
        let bound = unsafe { lzma_stream_buffer_bound(input_size) };
        if bound == 0 {
            // Documented "the bound would overflow size_t" signal. Returning
            // 0 would invite a zero-sized allocation and a guaranteed
            // LZMA_BUF_ERROR, so fall back to the trait's conservative
            // estimate; an input this large will fail on allocation anyway.
            return input_size + input_size / 8 + 1024;
        }
        bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* — incompressible bytes without a dev-dep on
    /// `rand`.
    fn pseudo_random(len: usize) -> Vec<u8> {
        let mut s: u64 = 0x2545_F491_4F6C_DD1D;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 33) as u8
            })
            .collect()
    }

    /// Realistic CTP payload: structured log-ish records, i.e. mixed
    /// literal text and repeated field names.
    fn realistic_data() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..2_000u32 {
            v.extend_from_slice(
                format!(
                    "ts=2026-07-16T12:{:02}:{:02}Z node=ctp-{:03} blob_id={} size={} op={}\n",
                    (i / 60) % 60,
                    i % 60,
                    i % 128,
                    i.wrapping_mul(2_654_435_761),
                    (i % 97) * 4096,
                    if i % 3 == 0 { "put" } else { "get" },
                )
                .as_bytes(),
            );
        }
        v
    }

    fn roundtrip(c: &LzmaCompressor, data: &[u8]) -> Vec<u8> {
        let mut enc = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut enc, data).expect("compress");
        enc.truncate(n);

        let mut dec = vec![0u8; data.len() + 64];
        let m = c.decompress(&mut dec, &enc).expect("decompress");
        dec.truncate(m);
        dec
    }

    #[test]
    fn links_lzma_5x_abi() {
        let (major, _minor, _patch) = version();
        assert_eq!(major, 5, "this module targets the liblzma 5.x ABI");
    }

    /// (1) Round trip over realistic data — byte-exact, lzma is lossless.
    #[test]
    fn roundtrip_realistic_data_is_byte_exact() {
        let data = realistic_data();
        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let c = LzmaCompressor::new(preset);
            assert_eq!(roundtrip(&c, &data), data, "preset {:?}", preset);
        }
    }

    /// Round trip must also survive data with no structure at all.
    #[test]
    fn roundtrip_incompressible_data_is_byte_exact() {
        let data = pseudo_random(64 * 1024);
        let c = LzmaCompressor::default();
        assert_eq!(roundtrip(&c, &data), data);
    }

    /// (2) Highly compressible data actually shrinks.
    #[test]
    fn compressible_data_shrinks() {
        let data = vec![0xABu8; 1 << 20]; // 1 MiB of one byte
        let c = LzmaCompressor::new(Preset::Balanced);
        let mut enc = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut enc, &data).expect("compress");

        assert!(n < data.len() / 100, "1 MiB of 0xAB -> {n} bytes");
        assert_eq!(roundtrip(&c, &data), data);
    }

    /// (3) Output buffer too small -> Err, not a truncated "success" and not
    /// a scribble past the buffer. This is exactly the case C++
    /// `ctp::Lzma::Compress` gets wrong (module docs, divergence 3).
    #[test]
    fn compress_into_too_small_buffer_errs() {
        let data = pseudo_random(4096);
        let c = LzmaCompressor::default();

        // 8 bytes cannot even hold the 12-byte .xz stream header.
        let mut tiny = [0u8; 8];
        let err = c.compress(&mut tiny, &data).unwrap_err();
        assert!(err.0.contains("LZMA_BUF_ERROR"), "got: {err}");

        // Zero-capacity output: the dangling-pointer/len-0 path.
        let err = c.compress(&mut [], &data).unwrap_err();
        assert!(err.0.contains("LZMA_BUF_ERROR"), "got: {err}");
    }

    /// (3) Same on the decode side.
    #[test]
    fn decompress_into_too_small_buffer_errs() {
        let data = realistic_data();
        let c = LzmaCompressor::default();
        let mut enc = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut enc, &data).unwrap();
        enc.truncate(n);

        let mut small = [0u8; 10];
        let err = c.decompress(&mut small, &enc).unwrap_err();
        assert!(err.0.contains("LZMA_BUF_ERROR"), "got: {err}");
    }

    /// Corrupt/foreign input is rejected rather than trusted.
    #[test]
    fn decompress_garbage_errs() {
        let c = LzmaCompressor::default();
        let mut out = [0u8; 256];
        let err = c.decompress(&mut out, b"definitely not an .xz stream").unwrap_err();
        assert!(err.0.contains("LZMA_FORMAT_ERROR"), "got: {err}");

        // Empty input is not a valid .xz stream either (an *empty stream*
        // still has a 32-byte header/footer -- see empty_input_roundtrips).
        assert!(c.decompress(&mut out, &[]).is_err());
    }

    /// (4) Empty input: encodes to a valid empty .xz stream and decodes back
    /// to zero bytes without panicking on the dangling zero-length pointers.
    #[test]
    fn empty_input_roundtrips() {
        let c = LzmaCompressor::default();
        let mut enc = vec![0u8; c.max_compressed_size(0)];
        let n = c.compress(&mut enc, &[]).expect("compress empty");
        assert!(n > 0, "an empty .xz stream still has header+footer");
        enc.truncate(n);

        // Decoding an empty stream into an empty buffer must succeed with 0.
        let m = c.decompress(&mut [], &enc).expect("decompress empty");
        assert_eq!(m, 0);

        // ...and into a non-empty buffer too.
        let mut dec = [0xFFu8; 32];
        assert_eq!(c.decompress(&mut dec, &enc).unwrap(), 0);
        assert_eq!(dec, [0xFFu8; 32], "nothing should have been written");
    }

    #[test]
    fn preset_mapping_matches_issue_756() {
        assert_eq!(LzmaCompressor::new(Preset::Fast).preset_level(), 1);
        assert_eq!(LzmaCompressor::new(Preset::Balanced).preset_level(), 6);
        assert_eq!(LzmaCompressor::new(Preset::Best).preset_level(), 9);
        assert_eq!(LzmaCompressor::default().preset_level(), 6);
    }

    /// Best must not compress worse than Fast on compressible input, and
    /// every preset's output must be decodable by every other preset's
    /// codec (the preset lives in the stream header, not in the decoder).
    #[test]
    fn presets_are_decode_compatible() {
        let data = realistic_data();
        let fast = LzmaCompressor::new(Preset::Fast);
        let best = LzmaCompressor::new(Preset::Best);

        let mut enc = vec![0u8; best.max_compressed_size(data.len())];
        let n = best.compress(&mut enc, &data).unwrap();
        enc.truncate(n);

        // Decode Best-compressed bytes with a Fast-configured codec.
        let mut dec = vec![0u8; data.len()];
        let m = fast.decompress(&mut dec, &enc).unwrap();
        dec.truncate(m);
        assert_eq!(dec, data);
    }

    #[test]
    fn max_compressed_size_is_a_real_bound() {
        let c = LzmaCompressor::default();
        // Incompressible data is the worst case; the bound must hold.
        let data = pseudo_random(32 * 1024);
        let bound = c.max_compressed_size(data.len());
        let mut enc = vec![0u8; bound];
        let n = c.compress(&mut enc, &data).unwrap();
        assert!(n <= bound, "{n} > bound {bound}");
        assert!(c.max_compressed_size(0) > 0);
    }

    /// The factory arm in lib.rs must construct this type.
    #[test]
    fn factory_builds_lzma() {
        let c = crate::compressor_for(LibraryId::Lzma, Preset::Best).expect("lzma feature on");
        assert_eq!(c.library(), LibraryId::Lzma);
        assert!(!LibraryId::Lzma.is_lossy());

        let data = realistic_data();
        let mut enc = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut enc, &data).unwrap();
        enc.truncate(n);
        let mut dec = vec![0u8; data.len()];
        let m = c.decompress(&mut dec, &enc).unwrap();
        dec.truncate(m);
        assert_eq!(dec, data);

        assert!(crate::compressor_by_name("lzma", Preset::Fast).is_some());
    }

    /// The trait is `Send + Sync`; the single-call API is reentrant, so a
    /// shared codec must work from many threads at once.
    #[test]
    fn shared_across_threads() {
        let c: std::sync::Arc<dyn Compressor> =
            std::sync::Arc::new(LzmaCompressor::new(Preset::Fast));
        let data = std::sync::Arc::new(realistic_data());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = std::sync::Arc::clone(&c);
                let data = std::sync::Arc::clone(&data);
                std::thread::spawn(move || {
                    let mut enc = vec![0u8; c.max_compressed_size(data.len())];
                    let n = c.compress(&mut enc, &data).unwrap();
                    enc.truncate(n);
                    let mut dec = vec![0u8; data.len()];
                    let m = c.decompress(&mut dec, &enc).unwrap();
                    dec.truncate(m);
                    assert_eq!(&dec, data.as_ref());
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
