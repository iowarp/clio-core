// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Brotli bindings (hand-rolled, dependency-free) and the safe
//! [`Compressor`] implementation over them.
//!
//! # What is wrapped
//!
//! Google Brotli's **one-shot C API** (`libbrotlienc` / `libbrotlidec` /
//! `libbrotlicommon`), mirroring C++ `ctp::Brotli`
//! (`clio_ctp/compress/brotli.h`):
//!
//! | C | Rust |
//! |---|---|
//! | `BrotliEncoderCompress` | [`BrotliCompressor::compress`] |
//! | `BrotliDecoderDecompress` | [`BrotliCompressor::decompress`] |
//! | `BrotliEncoderMaxCompressedSize` | [`BrotliCompressor::max_compressed_size`] |
//!
//! # Version / ABI assumptions
//!
//! Verified against **brotli 1.1.0** headers (`brotli/encode.h`,
//! `brotli/decode.h`), the version in the CTP devcontainer. The one-shot
//! entry points have been ABI-stable since 1.0 and are the library's
//! documented stable C surface. Assumptions baked into the decls below:
//!
//! * `BROTLI_BOOL` is `int` (encode.h documents it as "actually it is
//!   @c int"); `BROTLI_TRUE` = 1, `BROTLI_FALSE` = 0. Per brotli's own
//!   docs it is tested for truthiness, never equality with `BROTLI_TRUE`.
//! * `BrotliEncoderMode` and `BrotliDecoderResult` are plain C enums with
//!   small non-negative values, hence `int`-sized under the SysV/MSVC ABIs.
//! * `size_t` == [`usize`] on every target CTP builds for.
//! * These are pure functions over caller memory — no global state, no
//!   allocator handles (we pass the default `NULL` allocators implicitly by
//!   using the one-shot API), so the codec is `Send + Sync`.
//!
//! The stateful `BrotliEncoderState` / `BrotliDecoderState` handles are
//! deliberately **not** declared: the one-shot API does not use them (see
//! the divergences below).
//!
//! # Divergence from the C++ wrapper
//!
//! Byte-format compatibility is unaffected by all of these — brotli streams
//! are self-describing (the window size is encoded in the stream header),
//! so blobs written by C++ `ctp::Brotli` decode here and vice versa,
//! regardless of the encode parameters. Only ratio/speed differ.
//!
//! 1. **Encode parameters (the important one).** C++ passes
//!    `BrotliEncoderCompress(BROTLI_PARAM_QUALITY, BROTLI_OPERATION_FINISH,
//!    BROTLI_DEFAULT_MODE, ...)`. Those first two arguments are *enum ids
//!    from unrelated enums*, not parameter values: `BROTLI_PARAM_QUALITY`
//!    is `BrotliEncoderParameter` = 1 and `BROTLI_OPERATION_FINISH` is
//!    `BrotliEncoderOperation` = 2. So C++ always encodes at `quality=1`
//!    and `lgwin=2` — the latter below `BROTLI_MIN_WINDOW_BITS` (10) and
//!    silently clamped up to it by brotli's internal `SanitizeParams`. The
//!    net effect is that the C++ wrapper ignores its preset and always
//!    compresses at the near-fastest setting. This wrapper instead passes
//!    real values: quality from the [`Preset`] (Fast=1, Balanced=5,
//!    Best=11) and `lgwin` = 22 (`BROTLI_DEFAULT_WINDOW`). Fast therefore
//!    matches C++'s effective quality, while Balanced/Best compress harder.
//! 2. **Undersized output is an error, not a log line.** C++ compares
//!    against `BrotliEncoderMaxCompressedSize` and only `HLOG(kError)`s,
//!    then compresses anyway and reports brotli's own status. We do not
//!    pre-reject a buffer smaller than the worst-case bound (compressed
//!    output is normally far below it, so that would be needlessly
//!    strict) — we pass the true capacity to brotli, which fails cleanly
//!    when the data does not fit, and surface that as [`CompressError`].
//! 3. **No unused state objects.** C++ creates and destroys a
//!    `BrotliEncoderState` / `BrotliDecoderState` around each call even
//!    though the one-shot API allocates its own internally; the handles are
//!    never passed anywhere. We skip that dead allocation.
//! 4. **Errors carry a reason** ([`CompressError`]) rather than a bare
//!    `bool` — the crate-wide convention from `lib.rs`.
//!
//! On failure, bytes already written into the caller's output slice are
//! unspecified (brotli may have emitted a partial stream). This is not
//! unsound — the slice stays initialized `u8` — but the contents are
//! meaningless and the returned `Err` carries no length.

use std::ffi::c_int;

use crate::{CompressError, Compressor, LibraryId, Preset};

// ---------------------------------------------------------------------------
// Raw FFI: brotli one-shot C API (encode.h / decode.h, v1.1.0)
// ---------------------------------------------------------------------------

/// `BROTLI_BOOL` — documented as `int`; truthy means success.
type BrotliBool = c_int;
/// `BrotliEncoderMode` — C enum, `int`-sized.
type BrotliEncoderMode = c_int;
/// `BrotliDecoderResult` — C enum, `int`-sized.
type BrotliDecoderResult = c_int;

/// `BROTLI_MODE_GENERIC` (= `BROTLI_DEFAULT_MODE`); what C++ passes too.
const BROTLI_MODE_GENERIC: BrotliEncoderMode = 0;

/// `BROTLI_DECODER_RESULT_*`.
const BROTLI_DECODER_RESULT_ERROR: BrotliDecoderResult = 0;
const BROTLI_DECODER_RESULT_SUCCESS: BrotliDecoderResult = 1;
const BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT: BrotliDecoderResult = 2;
const BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT: BrotliDecoderResult = 3;

/// `BROTLI_MIN_QUALITY` / `BROTLI_MAX_QUALITY`.
const BROTLI_MIN_QUALITY: c_int = 0;
const BROTLI_MAX_QUALITY: c_int = 11;
/// `BROTLI_DEFAULT_WINDOW`; within [`BROTLI_MIN_WINDOW_BITS` = 10, 24].
const BROTLI_DEFAULT_WINDOW: c_int = 22;

extern "C" {
    /// `BROTLI_ENC_API BROTLI_BOOL BrotliEncoderCompress(int quality, int lgwin,
    ///  BrotliEncoderMode mode, size_t input_size, const uint8_t input_buffer[],
    ///  size_t* encoded_size, uint8_t encoded_buffer[]);`
    ///
    /// `*encoded_size` is INOUT: capacity in, bytes written out.
    fn BrotliEncoderCompress(
        quality: c_int,
        lgwin: c_int,
        mode: BrotliEncoderMode,
        input_size: usize,
        input_buffer: *const u8,
        encoded_size: *mut usize,
        encoded_buffer: *mut u8,
    ) -> BrotliBool;

    /// `BROTLI_ENC_API size_t BrotliEncoderMaxCompressedSize(size_t input_size);`
    ///
    /// Returns 0 when the bound does not fit in `size_t`.
    fn BrotliEncoderMaxCompressedSize(input_size: usize) -> usize;

    /// `BROTLI_DEC_API BrotliDecoderResult BrotliDecoderDecompress(
    ///  size_t encoded_size, const uint8_t encoded_buffer[],
    ///  size_t* decoded_size, uint8_t decoded_buffer[]);`
    ///
    /// `*decoded_size` is INOUT: capacity in, bytes written out.
    fn BrotliDecoderDecompress(
        encoded_size: usize,
        encoded_buffer: *const u8,
        decoded_size: *mut usize,
        decoded_buffer: *mut u8,
    ) -> BrotliDecoderResult;
}

/// Human-readable reason for a `BrotliDecoderResult`.
///
/// The one-shot decoder collapses everything non-success to `ERROR`, but we
/// name the streaming codes too in case a future brotli lets them through.
fn decoder_result_str(rc: BrotliDecoderResult) -> &'static str {
    match rc {
        BROTLI_DECODER_RESULT_ERROR => "corrupt input, output too small, or allocation failure",
        BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT => "truncated input",
        BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT => "output buffer too small",
        _ => "unknown decoder result",
    }
}

// ---------------------------------------------------------------------------
// BrotliCompressor
// ---------------------------------------------------------------------------

/// Brotli codec — the Rust face of C++ `ctp::Brotli`.
///
/// Stateless and cheap to construct; every call is a one-shot into brotli.
#[derive(Debug, Clone, Copy)]
pub struct BrotliCompressor {
    /// Encode quality, clamped to [`BROTLI_MIN_QUALITY`, `BROTLI_MAX_QUALITY`].
    quality: c_int,
    /// Sliding-window bits (`BROTLI_DEFAULT_WINDOW`).
    lgwin: c_int,
}

impl BrotliCompressor {
    /// Build the codec for `preset`.
    ///
    /// Quality mapping: `Fast` → 1, `Balanced` → 5, `Best` → 11
    /// (`BROTLI_MAX_QUALITY`). See the module docs: the C++ wrapper's
    /// preset is inert, this one is not.
    pub fn new(preset: Preset) -> Self {
        let quality = match preset {
            Preset::Fast => 1,
            Preset::Balanced => 5,
            Preset::Best => BROTLI_MAX_QUALITY,
        };
        debug_assert!((BROTLI_MIN_QUALITY..=BROTLI_MAX_QUALITY).contains(&quality));
        Self {
            quality,
            lgwin: BROTLI_DEFAULT_WINDOW,
        }
    }

    /// The quality actually handed to brotli (test/introspection aid).
    pub fn quality(&self) -> i32 {
        self.quality
    }
}

impl Default for BrotliCompressor {
    fn default() -> Self {
        Self::new(Preset::default())
    }
}

impl Compressor for BrotliCompressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        let mut encoded_size = output.len();
        // SAFETY: `input`/`output` are live slices, so their pointers are
        // valid for `input.len()` reads and `output.len()` writes and stay
        // so for the call (brotli does not retain them). Empty slices yield
        // dangling-but-aligned non-null pointers, which brotli never
        // dereferences because the paired length is 0. `encoded_size` is a
        // live local: in it carries the true capacity, so brotli cannot
        // write past `output`; out it carries the bytes written. The decls
        // above are checked against brotli 1.1.0's headers.
        let ok = unsafe {
            BrotliEncoderCompress(
                self.quality,
                self.lgwin,
                BROTLI_MODE_GENERIC,
                input.len(),
                input.as_ptr(),
                &mut encoded_size,
                output.as_mut_ptr(),
            )
        };
        if ok == 0 {
            // Brotli does not report *why*; by far the usual cause is an
            // output buffer that cannot hold the stream.
            return Err(CompressError(format!(
                "BrotliEncoderCompress failed (quality={}, lgwin={}, input={} B, \
                 output capacity={} B; worst case needs {} B)",
                self.quality,
                self.lgwin,
                input.len(),
                output.len(),
                self.max_compressed_size(input.len()),
            )));
        }
        // Defensive: a compliant brotli never reports more than capacity.
        if encoded_size > output.len() {
            return Err(CompressError(format!(
                "BrotliEncoderCompress reported {encoded_size} B written into a {} B buffer",
                output.len()
            )));
        }
        Ok(encoded_size)
    }

    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        let mut decoded_size = output.len();
        // SAFETY: same contract as compress() — valid pointer/length pairs
        // for the duration of the call, `decoded_size` passed in as the true
        // output capacity so brotli cannot overrun `output`, and read back
        // as the number of bytes written.
        let rc = unsafe {
            BrotliDecoderDecompress(
                input.len(),
                input.as_ptr(),
                &mut decoded_size,
                output.as_mut_ptr(),
            )
        };
        if rc != BROTLI_DECODER_RESULT_SUCCESS {
            return Err(CompressError(format!(
                "BrotliDecoderDecompress failed: {} (input={} B, output capacity={} B)",
                decoder_result_str(rc),
                input.len(),
                output.len(),
            )));
        }
        // Defensive: a compliant brotli never reports more than capacity.
        if decoded_size > output.len() {
            return Err(CompressError(format!(
                "BrotliDecoderDecompress reported {decoded_size} B written into a {} B buffer",
                output.len()
            )));
        }
        Ok(decoded_size)
    }

    fn library(&self) -> LibraryId {
        LibraryId::Brotli
    }

    fn max_compressed_size(&self, input_size: usize) -> usize {
        // SAFETY: a pure arithmetic function over a scalar; no pointers.
        let n = unsafe { BrotliEncoderMaxCompressedSize(input_size) };
        // Brotli returns 0 when the bound overflows `size_t` (only at
        // absurd input sizes). Never hand back 0 — a caller would allocate
        // nothing and then "safely" fail every compress. usize::MAX is the
        // honest answer: no allocation can satisfy this input.
        if n == 0 {
            usize::MAX
        } else {
            n
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes (LCG) — stands in for
    /// incompressible payloads without pulling in a rand dependency.
    fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    /// Realistic CTP-ish payload: structured log/metadata text, which is
    /// what blob metadata and lightbeam frames actually look like.
    fn realistic_data() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..2000 {
            v.extend_from_slice(
                format!(
                    "[2026-07-16T12:{:02}:{:02}Z] blob_id={} tag=dataset/run{} \
                     size={} node=ctp-node-{} op=put status=ok\n",
                    i % 60,
                    (i * 7) % 60,
                    i,
                    i % 13,
                    4096 + (i % 512),
                    i % 8
                )
                .as_bytes(),
            );
        }
        v
    }

    /// (1) Round-trip must be byte-exact — brotli is lossless.
    #[test]
    fn round_trip_is_byte_exact() {
        let input = realistic_data();
        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let c = BrotliCompressor::new(preset);
            let mut enc = vec![0u8; c.max_compressed_size(input.len())];
            let n = c.compress(&mut enc, &input).unwrap();

            let mut dec = vec![0u8; input.len()];
            let m = c.decompress(&mut dec, &enc[..n]).unwrap();

            assert_eq!(m, input.len(), "preset {:?}: length mismatch", preset);
            assert_eq!(dec, input, "preset {:?}: payload mismatch", preset);
        }
    }

    /// Round-trip of incompressible data must also be byte-exact (the
    /// encoder falls back to stored blocks; output may exceed the input).
    #[test]
    fn round_trip_incompressible_is_byte_exact() {
        let input = pseudo_random(64 * 1024, 0xC0FFEE);
        let c = BrotliCompressor::new(Preset::Balanced);
        let mut enc = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut enc, &input).unwrap();
        let mut dec = vec![0u8; input.len()];
        let m = c.decompress(&mut dec, &enc[..n]).unwrap();
        assert_eq!(&dec[..m], &input[..]);
    }

    /// (2) Highly-compressible data must actually shrink.
    #[test]
    fn compressible_data_shrinks() {
        let input = vec![b'A'; 256 * 1024];
        let c = BrotliCompressor::new(Preset::Balanced);
        let mut enc = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut enc, &input).unwrap();
        assert!(
            n < input.len() / 100,
            "256 KiB of one byte should collapse; got {n} B"
        );

        // Realistic text should compress substantially too.
        let text = realistic_data();
        let mut enc2 = vec![0u8; c.max_compressed_size(text.len())];
        let n2 = c.compress(&mut enc2, &text).unwrap();
        assert!(n2 < text.len() / 2, "text: {n2} B vs {} B", text.len());

        // Best should be no worse than Fast on this payload.
        let fast = BrotliCompressor::new(Preset::Fast);
        let best = BrotliCompressor::new(Preset::Best);
        let mut ef = vec![0u8; fast.max_compressed_size(text.len())];
        let mut eb = vec![0u8; best.max_compressed_size(text.len())];
        let nf = fast.compress(&mut ef, &text).unwrap();
        let nb = best.compress(&mut eb, &text).unwrap();
        assert!(nb <= nf, "best ({nb} B) should not exceed fast ({nf} B)");
    }

    /// (3) Output buffer too small → Err, no memory corruption. The
    /// canary bytes around the window prove brotli respected the capacity.
    #[test]
    fn output_too_small_errors_without_corruption() {
        let input = pseudo_random(64 * 1024, 42);
        let c = BrotliCompressor::new(Preset::Balanced);

        // Compress: 8 bytes cannot hold 64 KiB of random data.
        let mut enc = [0xAAu8; 64];
        let (window, canary) = enc.split_at_mut(8);
        assert!(c.compress(window, &input).is_err());
        assert!(canary.iter().all(|&b| b == 0xAA), "wrote past the output");

        // Decompress into a buffer smaller than the payload.
        let mut good = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut good, &input).unwrap();
        let mut dec = vec![0xBBu8; 1024];
        let (window, canary) = dec.split_at_mut(512);
        assert!(c.decompress(window, &good[..n]).is_err());
        assert!(canary.iter().all(|&b| b == 0xBB), "wrote past the output");

        // Zero-length output for non-empty input must fail, not panic.
        assert!(c.compress(&mut [], &input).is_err());
    }

    /// (4) Empty input round-trips.
    #[test]
    fn empty_input_round_trips() {
        let c = BrotliCompressor::new(Preset::Balanced);

        // Brotli's bound for 0 bytes is 2; an empty stream is not 0 bytes.
        let bound = c.max_compressed_size(0);
        assert!(bound > 0, "max_compressed_size(0) must leave room");

        let mut enc = vec![0u8; bound];
        let n = c.compress(&mut enc, &[]).unwrap();
        assert!(n <= bound);

        // Decode back into an empty buffer: zero bytes out, still success.
        let mut dec = [0u8; 0];
        let m = c.decompress(&mut dec, &enc[..n]).unwrap();
        assert_eq!(m, 0);

        // And into a roomy buffer, for callers that over-allocate.
        let mut dec2 = vec![0xCCu8; 16];
        let m2 = c.decompress(&mut dec2, &enc[..n]).unwrap();
        assert_eq!(m2, 0);
        assert!(dec2.iter().all(|&b| b == 0xCC), "wrote into empty payload");
    }

    /// Corrupt/garbage input must return Err, never panic or hang.
    #[test]
    fn corrupt_input_errors() {
        let c = BrotliCompressor::new(Preset::Balanced);
        let garbage = pseudo_random(512, 7);
        let mut out = vec![0u8; 4096];
        // Not a valid stream (a stray success would mean brotli decoded
        // random bytes, which is fine too — assert only on no-panic).
        let _ = c.decompress(&mut out, &garbage);

        // Truncating a real stream must be detected.
        let input = realistic_data();
        let mut enc = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut enc, &input).unwrap();
        let mut dec = vec![0u8; input.len()];
        assert!(
            c.decompress(&mut dec, &enc[..n / 2]).is_err(),
            "truncated stream must not decode as success"
        );
    }

    /// The bound must actually bound: worst case (incompressible) fits.
    #[test]
    fn max_compressed_size_is_a_real_bound() {
        let c = BrotliCompressor::new(Preset::Best);
        for len in [0usize, 1, 17, 4096, 100_000] {
            let input = pseudo_random(len, len as u64 + 1);
            let bound = c.max_compressed_size(len);
            let mut enc = vec![0u8; bound];
            let n = c
                .compress(&mut enc, &input)
                .unwrap_or_else(|e| panic!("len {len} within bound {bound} failed: {e}"));
            assert!(n <= bound, "len {len}: {n} B exceeds bound {bound}");
        }
        // Overflow sentinel rather than a 0-byte "allocate nothing".
        assert_eq!(c.max_compressed_size(usize::MAX), usize::MAX);
    }

    /// Preset → quality mapping, and the factory wiring in lib.rs.
    #[test]
    fn preset_maps_to_quality_and_factory_builds() {
        assert_eq!(BrotliCompressor::new(Preset::Fast).quality(), 1);
        assert_eq!(BrotliCompressor::new(Preset::Balanced).quality(), 5);
        assert_eq!(BrotliCompressor::new(Preset::Best).quality(), 11);
        assert_eq!(
            BrotliCompressor::new(Preset::Balanced).library(),
            LibraryId::Brotli
        );

        let boxed = crate::compressor_by_name("brotli", Preset::Fast).expect("factory arm");
        assert_eq!(boxed.library(), LibraryId::Brotli);
        let input = realistic_data();
        let mut enc = vec![0u8; boxed.max_compressed_size(input.len())];
        let n = boxed.compress(&mut enc, &input).unwrap();
        let mut dec = vec![0u8; input.len()];
        let m = boxed.decompress(&mut dec, &enc[..n]).unwrap();
        assert_eq!(&dec[..m], &input[..]);
    }

    /// Cross-preset streams must interoperate — the format is
    /// self-describing, so quality/lgwin are encode-side only. This is what
    /// keeps C++-written blobs (effective quality 1) readable here.
    #[test]
    fn streams_are_preset_independent() {
        let input = realistic_data();
        let enc_c = BrotliCompressor::new(Preset::Best);
        let mut enc = vec![0u8; enc_c.max_compressed_size(input.len())];
        let n = enc_c.compress(&mut enc, &input).unwrap();

        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let dec_c = BrotliCompressor::new(preset);
            let mut dec = vec![0u8; input.len()];
            let m = dec_c.decompress(&mut dec, &enc[..n]).unwrap();
            assert_eq!(&dec[..m], &input[..], "decoder preset {:?}", preset);
        }
    }

    /// The trait object is used across worker threads; prove the bounds.
    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BrotliCompressor>();
    }
}
