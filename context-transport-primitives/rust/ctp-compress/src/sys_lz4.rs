// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! FFI wrapper for **LZ4** (`liblz4`) — the C reference implementation from
//! <https://github.com/lz4/lz4>, linked by `build.rs` under the `lz4` feature.
//!
//! # Version / ABI assumptions
//!
//! Targets the stable `liblz4` v1.x C ABI (`liblz4.so.1`); developed and
//! tested against **1.9.4**, the devcontainer's system package. Only the
//! long-frozen block-format entry points are bound — `LZ4_compress_default`,
//! `LZ4_compress_HC`, `LZ4_decompress_safe`, `LZ4_compressBound` — all of
//! which have been ABI-stable since 1.7 and carry no struct layouts across
//! the boundary (every argument is a pointer or `int`). No streaming state
//! (`LZ4_stream_t`) is touched, so there is nothing whose size could drift
//! between the build-time and runtime `liblz4`. `LZ4_compress_HC` lives in
//! `lz4hc.h` but ships inside the same `liblz4` shared object.
//!
//! # Framing: none — raw LZ4 block format (matches C++)
//!
//! The LZ4 *block* format does not record the uncompressed length, so
//! decompression needs it from somewhere. The C++ wrappers
//! (`clio_ctp/compress/lz4.h`, `lossless_modes.h`) solve this by **not**
//! solving it: they pass the caller's `output_size` straight to
//! `LZ4_decompress_safe` as `dstCapacity` and let the caller remember the
//! original size. This module deliberately **matches that** and adds **no
//! length prefix**, because the bytes it emits must stay readable by the C++
//! `Lz4`/`Lz4WithModes` classes (and vice versa) — blobs already on disk
//! carry `LibraryId::Lz4` with no framing, and an 8-byte prefix would make
//! every one of them undecodable. The caller's contract is therefore the
//! same as in C++: **`decompress`'s `output` slice must be at least the
//! original byte length**. A slice longer than the original is fine (the
//! true length is returned); a shorter one returns `Err`.
//!
//! # Preset mapping (from C++ `Lz4WithModes`)
//!
//! | [`Preset`] | C++ `LosslessMode` | call |
//! |---|---|---|
//! | `Fast` | `FAST` | `LZ4_compress_default` |
//! | `Balanced` | `BALANCED` | `LZ4_compress_HC`, level 6 |
//! | `Best` | `BEST` | `LZ4_compress_HC`, level 12 (`LZ4HC_CLEVEL_MAX`) |
//!
//! # Divergences from the C++ wrapper
//!
//! All three are bug fixes or safety fixes; none changes the bytes emitted,
//! so format compatibility is unaffected.
//!
//! 1. **Decompression errors are detected.** `LZ4_decompress_safe` reports
//!    failure with a *negative* return, but both C++ wrappers test
//!    `output_size != 0` — so a negative result is reported as **success**
//!    and assigned into a `size_t &`, handing the caller a ~2^64 "length".
//!    This module tests `< 0` and returns [`CompressError`].
//! 2. **No `LZ4_compressBound` pre-rejection.** `Lz4WithModes::Compress`
//!    returns false whenever `output_size < LZ4_compressBound(input_size)`,
//!    refusing buffers that would in fact have succeeded (the bound is
//!    worst-case-incompressible; compressible data needs far less). Passing
//!    the real capacity to LZ4 is already safe — it never writes past
//!    `dstCapacity` and returns 0 if it would — so this module passes the
//!    true capacity through and lets LZ4 decide. Strictly more permissive;
//!    identical output when both accept.
//! 3. **Sizes are range-checked, not `int`-cast.** The C++ casts `size_t` to
//!    `int` unchecked, which is UB/silent truncation past 2 GiB (a >2 GiB
//!    buffer can cast negative and corrupt memory). Oversized inputs return
//!    `Err` here; output capacities are clamped to `INT_MAX`, which is safe
//!    because under-reporting capacity only makes LZ4 more conservative.
//! 4. **Empty buffers cannot reach LZ4's low-address underflow.** Pointers
//!    handed across the FFI are never dangling (see [`ffi_ptr`]) — without
//!    that, `Preset::Best` + empty input **segfaults** inside liblz4 1.9.4.
//!    The C++ has the same latent bug: `Lz4WithModes::Compress(out, n,
//!    nullptr, 0)` at BEST crashes for the identical reason (`nullptr - 12`
//!    underflows the loop guard), it is simply harder to trigger from C++
//!    because callers there pass real allocations.
//!
//! Empty input is accepted and compresses to LZ4's 1-byte empty block
//! (round-trips back to an empty slice), as in C++ — except that C++'s
//! `!= 0` check misreports that round-trip as a failure (divergence 1).

use std::ffi::{c_char, c_int};

use crate::{CompressError, Compressor, LibraryId, Preset};

// ---------------------------------------------------------------------------
// Raw FFI — hand-rolled from the documented liblz4 v1.x C API.
// Signatures verified against lz4.h / lz4hc.h (1.9.4):
//   int LZ4_compress_default(const char* src, char* dst, int srcSize, int dstCapacity);
//   int LZ4_compress_HC     (const char* src, char* dst, int srcSize, int dstCapacity, int compressionLevel);
//   int LZ4_decompress_safe (const char* src, char* dst, int compressedSize, int dstCapacity);
//   int LZ4_compressBound   (int inputSize);
//   int LZ4_versionNumber   (void);
// ---------------------------------------------------------------------------

extern "C" {
    /// Returns bytes written, or **0** if compression fails (e.g. `dst` too
    /// small). Never writes beyond `dst_capacity`.
    fn LZ4_compress_default(
        src: *const c_char,
        dst: *mut c_char,
        src_size: c_int,
        dst_capacity: c_int,
    ) -> c_int;

    /// High-compression variant (`lz4hc.h`, same shared object). Returns
    /// bytes written, or **0** on failure. Levels above `LZ4HC_CLEVEL_MAX`
    /// behave as `LZ4HC_CLEVEL_MAX`.
    fn LZ4_compress_HC(
        src: *const c_char,
        dst: *mut c_char,
        src_size: c_int,
        dst_capacity: c_int,
        compression_level: c_int,
    ) -> c_int;

    /// Returns bytes written into `dst`, or a **negative** value if the
    /// input is malformed or `dst_capacity` is too small. Guaranteed not to
    /// read outside `src[..compressed_size]` nor write outside
    /// `dst[..dst_capacity]` even on hostile input — this is the "safe"
    /// decoder, which is why it is the only one bound here.
    fn LZ4_decompress_safe(
        src: *const c_char,
        dst: *mut c_char,
        compressed_size: c_int,
        dst_capacity: c_int,
    ) -> c_int;

    /// Worst-case compressed size, or 0 if `input_size > LZ4_MAX_INPUT_SIZE`.
    fn LZ4_compressBound(input_size: c_int) -> c_int;

    /// Runtime library version, e.g. 10904 for 1.9.4.
    fn LZ4_versionNumber() -> c_int;
}

/// `LZ4_MAX_INPUT_SIZE` from lz4.h — 0x7E000000 (2,113,929,216 bytes).
/// Inputs above this cannot be represented in the block format at all.
const LZ4_MAX_INPUT_SIZE: usize = 0x7E00_0000;

/// `LZ4HC_CLEVEL_MAX` from lz4hc.h.
const LZ4HC_CLEVEL_MAX: c_int = 12;

/// HC level for [`Preset::Balanced`] — C++ `Lz4WithModes` BALANCED arm.
const HC_LEVEL_BALANCED: c_int = 6;

/// HC level for [`Preset::Best`] — C++ `Lz4WithModes` BEST arm.
const HC_LEVEL_BEST: c_int = LZ4HC_CLEVEL_MAX;

/// Runtime `liblz4` version as (major, minor, release).
///
/// Useful for asserting the loaded shared object matches ABI expectations.
pub fn version() -> (u32, u32, u32) {
    // SAFETY: LZ4_versionNumber takes no arguments, touches no memory, and
    // has no preconditions.
    let v = unsafe { LZ4_versionNumber() } as u32;
    (v / 10000, (v / 100) % 100, v % 100)
}

/// Clamp an output capacity to `INT_MAX` for the C API.
///
/// Under-reporting capacity is always sound: LZ4 simply refuses to write
/// past what it was told it has. (The C++ casts `size_t`→`int` unchecked,
/// which can go negative — see the module docs.)
fn clamp_capacity(cap: usize) -> c_int {
    cap.min(c_int::MAX as usize) as c_int
}

/// Anchor byte for empty slices — see [`ffi_ptr`]. Its address is a real
/// object's, which is all LZ4's bounds arithmetic needs.
static EMPTY_ANCHOR: u8 = 0;

/// Return a pointer for `slice` that is **never dangling**.
///
/// # Why this exists (a real liblz4 1.9.4 landmine, verified by probe)
///
/// Rust gives every empty slice the address `0x1` (`align_of::<u8>()`), not
/// a real address. `LZ4HC_compress_optimal` — the parser used for levels
/// >= `LZ4HC_CLEVEL_OPT_MIN` (10), i.e. our [`Preset::Best`] — does:
///
/// ```c
/// const BYTE* const iend = ip + *srcSizePtr;
/// const BYTE* const mflimit = iend - MFLIMIT;   /* MFLIMIT == 12 */
/// while (ip <= mflimit) { /* dereferences ip */ }
/// ```
///
/// With `ip == 0x1` and `srcSize == 0`, `iend - MFLIMIT` **underflows** to
/// ~2^64, so `ip <= mflimit` wrongly passes and LZ4 dereferences address
/// `0x1` → SIGSEGV, despite the length being 0. `LZ4_compress_default` and
/// HC levels <= 9 escape only because they test `srcSize < LZ4_minLength`
/// *before* that arithmetic. The same `iend - N` underflow pattern guards
/// reads in `LZ4_decompress_safe`, so empty inputs are anchored there too.
///
/// Handing LZ4 the address of a real object makes `iend - MFLIMIT` land
/// below `ip` as intended, so the loop is never entered and the anchor is
/// never actually read. Paired with a length of 0, semantics are unchanged.
///
/// Probe results on liblz4 1.9.4 (`srcSize=0`, level 12):
/// real pointer → returns 1 (the empty block); pointer `0x1` → SIGSEGV.
fn ffi_ptr(slice: &[u8]) -> *const u8 {
    if slice.is_empty() {
        &EMPTY_ANCHOR as *const u8
    } else {
        slice.as_ptr()
    }
}

// ---------------------------------------------------------------------------
// Lz4Compressor
// ---------------------------------------------------------------------------

/// LZ4 codec — the Rust face of C++ `ctp::Lz4WithModes`.
///
/// Emits and consumes **raw LZ4 block format with no framing**, byte-for-byte
/// interchangeable with the C++ wrapper (see the module docs). Stateless, so
/// one instance may be shared across threads.
#[derive(Debug, Clone, Copy)]
pub struct Lz4Compressor {
    preset: Preset,
}

impl Lz4Compressor {
    /// Construct at `preset` (C++ `CompressFactory::Get("lz4", preset)`).
    pub fn new(preset: Preset) -> Self {
        Self { preset }
    }

    /// The configured preset.
    pub fn preset(&self) -> Preset {
        self.preset
    }
}

impl Default for Lz4Compressor {
    fn default() -> Self {
        Self::new(Preset::default())
    }
}

impl Compressor for Lz4Compressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        if input.len() > LZ4_MAX_INPUT_SIZE {
            return Err(CompressError(format!(
                "lz4: input of {} bytes exceeds LZ4_MAX_INPUT_SIZE ({LZ4_MAX_INPUT_SIZE})",
                input.len()
            )));
        }
        // Checked directly above, so this cannot truncate.
        let src_size = input.len() as c_int;
        let dst_capacity = clamp_capacity(output.len());

        // Never hand LZ4 a dangling pointer — see `ffi_ptr`. `dst` needs its
        // own anchor because it must be writable; LZ4 cannot write through
        // it since the paired capacity is 0.
        let mut dst_anchor = 0u8;
        let src = ffi_ptr(input);
        let dst = if output.is_empty() {
            &mut dst_anchor as *mut u8
        } else {
            output.as_mut_ptr()
        };

        // SAFETY: `input`/`output` are live Rust slices, so their pointers
        // are valid and non-overlapping (`&[u8]` vs `&mut [u8]` cannot
        // alias); `ffi_ptr`/`dst_anchor` substitute the address of a live
        // object when a slice is empty, and both anchors outlive this call.
        // We pass each slice's true length: `src_size` is exact, and
        // `dst_capacity` never exceeds `output.len()` (clamp only lowers
        // it), so LZ4 — which is documented never to write past
        // `dst_capacity` — stays in bounds. u8 and c_char have identical
        // size and alignment. Failure is reported as a 0 return, not a trap.
        let written = unsafe {
            let src = src.cast::<c_char>();
            let dst = dst.cast::<c_char>();
            match self.preset {
                Preset::Fast => LZ4_compress_default(src, dst, src_size, dst_capacity),
                Preset::Balanced => LZ4_compress_HC(src, dst, src_size, dst_capacity, HC_LEVEL_BALANCED),
                Preset::Best => LZ4_compress_HC(src, dst, src_size, dst_capacity, HC_LEVEL_BEST),
            }
        };

        // LZ4 signals compression failure with 0 and never returns negative.
        if written <= 0 {
            return Err(CompressError(format!(
                "lz4: compression failed (preset {}, {} input bytes into {} bytes of output \
                 capacity — buffer likely too small; max_compressed_size() reports {})",
                self.preset.name(),
                input.len(),
                output.len(),
                self.max_compressed_size(input.len()),
            )));
        }
        Ok(written as usize)
    }

    /// Decompress a raw LZ4 block.
    ///
    /// `output.len()` must be **at least the original uncompressed length** —
    /// the block format does not carry it and this codec adds no framing, so
    /// the caller must know it (identical to the C++ contract; see the module
    /// docs). Too small an `output` yields `Err`, never a partial write the
    /// caller could mistake for success.
    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        if input.len() > c_int::MAX as usize {
            return Err(CompressError(format!(
                "lz4: compressed input of {} bytes exceeds INT_MAX",
                input.len()
            )));
        }
        let compressed_size = input.len() as c_int;
        let dst_capacity = clamp_capacity(output.len());

        // Never hand LZ4 a dangling pointer — see `ffi_ptr`. The decoder
        // guards its reads with the same `iend - N` arithmetic that
        // underflows on an empty slice's 0x1 address.
        let mut dst_anchor = 0u8;
        let src = ffi_ptr(input);
        let dst = if output.is_empty() {
            &mut dst_anchor as *mut u8
        } else {
            output.as_mut_ptr()
        };

        // SAFETY: same slice-validity and anchoring argument as `compress`.
        // LZ4_decompress_safe is the bounds-checked decoder: it reads only
        // within `src[..compressed_size]` and writes only within
        // `dst[..dst_capacity]` even for malformed/hostile input, reporting
        // any problem with a negative return rather than overrunning.
        let written = unsafe {
            LZ4_decompress_safe(
                src.cast::<c_char>(),
                dst.cast::<c_char>(),
                compressed_size,
                dst_capacity,
            )
        };

        // Negative is the error signal; 0 is a legitimate empty payload.
        // (The C++ `output_size != 0` test gets both of these wrong.)
        if written < 0 {
            return Err(CompressError(format!(
                "lz4: decompression failed (code {written}; {} compressed bytes into {} bytes \
                 of output capacity — malformed block, or buffer smaller than the original?)",
                input.len(),
                output.len(),
            )));
        }
        Ok(written as usize)
    }

    fn library(&self) -> LibraryId {
        LibraryId::Lz4
    }

    /// Exact `LZ4_compressBound` — tighter than the trait's generic default.
    ///
    /// Returns 0 for inputs above `LZ4_MAX_INPUT_SIZE`, which LZ4 cannot
    /// compress at any size; `compress` returns `Err` for those.
    fn max_compressed_size(&self, input_size: usize) -> usize {
        if input_size > LZ4_MAX_INPUT_SIZE {
            return 0;
        }
        // SAFETY: pure arithmetic on an `int`, no memory touched. The check
        // above guarantees `input_size` fits in c_int.
        let bound = unsafe { LZ4_compressBound(input_size as c_int) };
        if bound <= 0 {
            0
        } else {
            bound as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compressor_for;

    const PRESETS: [Preset; 3] = [Preset::Fast, Preset::Balanced, Preset::Best];

    /// Realistic mixed payload: structured log text (highly compressible) plus
    /// little-endian f64 telemetry (poorly compressible) — the shape of data
    /// CTP actually moves through blobs.
    fn realistic_data() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..2_000u32 {
            v.extend_from_slice(
                format!("[2026-07-16T12:00:{:02}Z] node=worker-{} op=put blob=b{:06} rc=0\n",
                    i % 60, i % 8, i)
                .as_bytes(),
            );
            let x = (i as f64) * std::f64::consts::FRAC_1_PI;
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    #[test]
    fn links_against_expected_abi() {
        let (major, minor, _rel) = version();
        // Only the frozen v1.x block API is bound; see module docs.
        assert_eq!(major, 1, "liblz4 major ABI must be 1, got {major}.{minor}");
        eprintln!("ctp-compress: liblz4 {major}.{minor}.{_rel}");
    }

    /// (1) Round-trip must be byte-exact — LZ4 is lossless.
    #[test]
    fn roundtrip_is_byte_exact_for_every_preset() {
        let data = realistic_data();
        for preset in PRESETS {
            let c = Lz4Compressor::new(preset);
            let mut buf = vec![0u8; c.max_compressed_size(data.len())];
            let n = c.compress(&mut buf, &data).unwrap();

            // No framing: the caller supplies the original size (C++ contract).
            let mut back = vec![0u8; data.len()];
            let m = c.decompress(&mut back, &buf[..n]).unwrap();

            assert_eq!(m, data.len(), "{preset:?}: wrong decompressed length");
            assert_eq!(back, data, "{preset:?}: round-trip corrupted data");
        }
    }

    /// A decompression buffer *larger* than the original is fine — the real
    /// length comes back. Locks the "caller may over-allocate" half of the
    /// no-framing contract.
    #[test]
    fn decompress_into_oversized_buffer_returns_true_length() {
        let data = realistic_data();
        let c = Lz4Compressor::new(Preset::Balanced);
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();

        let mut back = vec![0u8; data.len() + 4096];
        let m = c.decompress(&mut back, &buf[..n]).unwrap();
        assert_eq!(m, data.len());
        assert_eq!(&back[..m], &data[..]);
    }

    /// (2) Highly-compressible data must actually shrink.
    #[test]
    fn compressible_data_shrinks() {
        let data = vec![0xABu8; 256 * 1024];
        for preset in PRESETS {
            let c = Lz4Compressor::new(preset);
            let mut buf = vec![0u8; c.max_compressed_size(data.len())];
            let n = c.compress(&mut buf, &data).unwrap();
            assert!(
                n < data.len() / 100,
                "{preset:?}: 256 KiB of one repeated byte should compress >100x, got {n} bytes"
            );

            let mut back = vec![0u8; data.len()];
            assert_eq!(c.decompress(&mut back, &buf[..n]).unwrap(), data.len());
            assert_eq!(back, data);
        }
        // Real-world payload should shrink too (it is mostly log text).
        let real = realistic_data();
        let c = Lz4Compressor::new(Preset::Best);
        let mut buf = vec![0u8; c.max_compressed_size(real.len())];
        let n = c.compress(&mut buf, &real).unwrap();
        assert!(n < real.len(), "realistic payload should shrink: {n} vs {}", real.len());
    }

    /// (3) Too-small output buffers must return Err — never corrupt memory.
    /// Guard bytes around the destination prove LZ4 respected the capacity.
    #[test]
    fn output_buffer_too_small_errors_without_corrupting_memory() {
        let data = realistic_data();
        let c = Lz4Compressor::new(Preset::Fast);

        // Sandwich a tiny dst between guard regions; hand LZ4 only the middle.
        const GUARD: usize = 64;
        const DST: usize = 16;
        let mut arena = [0x5Au8; GUARD + DST + GUARD];
        let err = c.compress(&mut arena[GUARD..GUARD + DST], &data).unwrap_err();
        assert!(err.0.contains("lz4"), "unexpected error text: {}", err.0);
        assert!(arena[..GUARD].iter().all(|&b| b == 0x5A), "underflow past dst");
        assert!(arena[GUARD + DST..].iter().all(|&b| b == 0x5A), "overflow past dst");

        // Zero-length output is rejected rather than "succeeding" with 0.
        assert!(c.compress(&mut [], &data).is_err());

        // Decompression into too small a buffer must Err, not partially fill.
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();
        let mut short = vec![0u8; data.len() / 2];
        assert!(
            c.decompress(&mut short, &buf[..n]).is_err(),
            "decompress into an undersized buffer must fail"
        );
    }

    /// Malformed input must Err. Regression guard for the C++ `!= 0` bug:
    /// LZ4_decompress_safe's negative error code passes `!= 0`, so the C++
    /// reports success and yields a ~2^64 length. We must return Err.
    #[test]
    fn garbage_input_is_rejected_not_reported_as_huge_length() {
        let c = Lz4Compressor::new(Preset::Fast);
        let mut back = vec![0u8; 4096];
        for junk in [
            vec![0xFFu8; 128],
            vec![0x00u8; 1],
            b"definitely not an lz4 block".to_vec(),
        ] {
            match c.decompress(&mut back, &junk) {
                Err(_) => {}
                Ok(n) => assert!(
                    n <= back.len(),
                    "decompress reported {n} bytes into a {}-byte buffer",
                    back.len()
                ),
            }
        }
        // Empty input is not a valid block (needs at least a token byte).
        assert!(c.decompress(&mut back, &[]).is_err());
    }

    /// (4) Empty input round-trips to empty.
    #[test]
    fn empty_input_roundtrips() {
        for preset in PRESETS {
            let c = Lz4Compressor::new(preset);
            assert_eq!(c.max_compressed_size(0), 16, "LZ4_compressBound(0)");

            let mut buf = vec![0u8; c.max_compressed_size(0)];
            let n = c.compress(&mut buf, &[]).unwrap();

            // The LZ4 empty block is a single zero token — this also proves
            // we prepend no length prefix (an 8-byte header would show here).
            assert_eq!(n, 1, "{preset:?}: empty input should emit LZ4's 1-byte block");

            let mut back = vec![0u8; 0];
            let m = c.decompress(&mut back, &buf[..n]).unwrap();
            assert_eq!(m, 0, "{preset:?}: empty payload must decompress to 0 bytes");
        }
    }

    /// Regression: every combination of empty/non-empty input and output
    /// must return normally rather than segfault.
    ///
    /// `Preset::Best` (HC level 12) + empty input crashed liblz4 1.9.4 before
    /// `ffi_ptr` anchored the pointers: `LZ4HC_compress_optimal` computes
    /// `iend - MFLIMIT`, which underflows on the 0x1 address Rust gives every
    /// empty slice, so the loop guard passes and LZ4 reads address 0x1.
    /// A safe fn must never do that — hence this matrix.
    #[test]
    fn empty_slices_never_reach_lz4_as_dangling_pointers() {
        let nonempty = b"payload".to_vec();
        for preset in PRESETS {
            let c = Lz4Compressor::new(preset);
            for out_len in [0usize, 1, 64] {
                for input in [Vec::new(), nonempty.clone()] {
                    // Must return Ok or Err — the point is that it returns.
                    let mut out = vec![0u8; out_len];
                    let _ = c.compress(&mut out, &input);
                    let _ = c.decompress(&mut out, &input);
                }
            }
        }
    }

    /// Framing lock: output must be a *raw* block with no header, so a
    /// buffer produced here decodes with a bare LZ4_decompress_safe call —
    /// exactly what C++ `Lz4::Decompress` does. If anyone adds a length
    /// prefix, this fails and every C++-written blob would have broken.
    #[test]
    fn output_is_raw_block_format_no_header() {
        let data = b"iowarp ctp lz4 framing check; no length prefix may appear here".repeat(16);
        let c = Lz4Compressor::new(Preset::Fast);
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();

        // Decode via the raw C entry point, bypassing our wrapper entirely.
        let mut back = vec![0u8; data.len()];
        // SAFETY: valid slices; sizes are the true lengths and fit in c_int.
        let m = unsafe {
            LZ4_decompress_safe(
                buf.as_ptr().cast::<c_char>(),
                back.as_mut_ptr().cast::<c_char>(),
                n as c_int,
                back.len() as c_int,
            )
        };
        assert_eq!(m, data.len() as c_int, "raw LZ4 could not decode our output");
        assert_eq!(back, data);
    }

    /// The factory arm in lib.rs must build this type and report Lz4.
    #[test]
    fn factory_builds_lz4_and_roundtrips_through_trait_object() {
        let c = compressor_for(LibraryId::Lz4, Preset::Balanced).expect("lz4 feature is on");
        assert_eq!(c.library(), LibraryId::Lz4);
        assert!(!c.library().is_lossy());

        let data = realistic_data();
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();
        let mut back = vec![0u8; data.len()];
        let m = c.decompress(&mut back, &buf[..n]).unwrap();
        assert_eq!(&back[..m], &data[..]);
    }

    /// Presets must agree with C++ `Lz4WithModes`: FAST is the plain codec,
    /// BEST is HC-12. Every preset's output stays mutually decodable, since
    /// they all emit the same block format.
    #[test]
    fn presets_are_interchangeable_on_the_wire() {
        let data = realistic_data();
        let mut sizes = Vec::new();
        for preset in PRESETS {
            let enc = Lz4Compressor::new(preset);
            let mut buf = vec![0u8; enc.max_compressed_size(data.len())];
            let n = enc.compress(&mut buf, &data).unwrap();
            sizes.push(n);

            // Any preset's decoder reads any preset's bytes (format is one).
            for dec_preset in PRESETS {
                let dec = Lz4Compressor::new(dec_preset);
                let mut back = vec![0u8; data.len()];
                assert_eq!(dec.decompress(&mut back, &buf[..n]).unwrap(), data.len());
                assert_eq!(back, data, "{preset:?} -> {dec_preset:?} mismatch");
            }
        }
        // HC (Balanced/Best) must beat the fast codec on real data.
        assert!(
            sizes[1] <= sizes[0] && sizes[2] <= sizes[1],
            "expected Fast >= Balanced >= Best sizes, got {sizes:?}"
        );
    }

    /// `max_compressed_size` must match LZ4's own bound and behave at the
    /// LZ4_MAX_INPUT_SIZE cliff without panicking or overflowing.
    #[test]
    fn compress_bound_matches_lz4_and_rejects_oversized_input() {
        let c = Lz4Compressor::new(Preset::Fast);
        for n in [0usize, 1, 1024, 1 << 20] {
            // LZ4_COMPRESSBOUND(isize) = isize + isize/255 + 16
            assert_eq!(c.max_compressed_size(n), n + n / 255 + 16);
        }
        assert_eq!(c.max_compressed_size(LZ4_MAX_INPUT_SIZE + 1), 0);
        assert_eq!(c.max_compressed_size(usize::MAX), 0);
    }
}
