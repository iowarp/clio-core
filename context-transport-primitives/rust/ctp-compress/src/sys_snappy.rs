// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! FFI wrapper for **Snappy** (google/snappy), a fast byte-oriented lossless
//! compressor. Bindings are hand-rolled against Snappy's stable C ABI
//! (`snappy-c.h`, linked from `libsnappy`); no bindgen, no `snappy` crate.
//!
//! # Version / ABI assumptions
//!
//! * Snappy's C API has been ABI-stable since 1.0 and is unversioned — the
//!   five entry points below have never changed signature. Verified against
//!   the devcontainer's `libsnappy1v5` / `/usr/include/snappy-c.h`.
//! * `snappy_status` is a plain C enum, passed as `int` (`c_int`) under the
//!   System V and MSVC ABIs alike. Only 0/1/2 are defined; unknown values are
//!   reported rather than assumed impossible.
//! * All lengths are `size_t` (`usize`). Buffers are `char*`, which is
//!   `u8`-compatible on every platform CTP targets.
//! * The **block format** is a stable, documented wire format. Bytes produced
//!   here decompress with the C++ `ctp::Snappy` wrapper and vice-versa: this
//!   module adds no framing, length prefix, or CTP header of its own, exactly
//!   like the C++ side. (This is the raw block format, *not* the
//!   stream/framing format used by `snappy` CLI tools — they are not
//!   interchangeable.)
//!
//! # Presets
//!
//! **Snappy has no compression levels, and the preset is ignored.** Snappy
//! deliberately exposes a single operating point (speed over ratio); there is
//! no level/effort knob anywhere in its API. [`SnappyCompressor::new`] accepts
//! a [`Preset`] only to satisfy the uniform factory signature, and stores it
//! purely so [`SnappyCompressor::preset`] can report what was requested.
//! `Fast`, `Balanced`, and `Best` all produce byte-identical output (asserted
//! by `preset_is_ignored`). The C++ `ctp::Snappy` wrapper likewise takes no
//! preset.
//!
//! # Divergence from the C++ wrapper (`clio_ctp/compress/snappy.h`)
//!
//! The C++ wrapper calls the **C++** API and is memory-unsafe on undersized
//! buffers; this module calls the **C** API, which bounds-checks. The output
//! bytes are identical — only the error/overflow behavior differs:
//!
//! 1. **Compress overflow.** C++ calls `snappy::RawCompress`, which *assumes*
//!    the output buffer holds at least `MaxCompressedLength(input_size)` and
//!    writes past the end otherwise (it checks `output_size` only as an
//!    out-param). It then calls `IsValidCompressedBuffer` on the *already
//!    overflowed* buffer — the corruption has happened by then. Here,
//!    `snappy_compress` takes the capacity as an INOUT and returns
//!    `SNAPPY_BUFFER_TOO_SMALL` instead, which we map to [`CompressError`].
//!    No overflow is reachable from safe code.
//! 2. **Decompress overflow.** C++ uses `UncompressAsMuchAsPossible` with an
//!    `UncheckedByteArraySink` — "unchecked" is literal: the sink has no idea
//!    how large `output` is and will happily run off the end. `snappy_uncompress`
//!    bounds-checks against the capacity we pass and reports
//!    `SNAPPY_BUFFER_TOO_SMALL`.
//! 3. **Empty payload.** C++ `Decompress` returns `output_size != 0`, so
//!    decompressing the (valid) encoding of an empty input reports *failure*.
//!    Here an empty payload is `Ok(0)` — a valid round-trip is not an error.
//! 4. **Corrupt input.** C++ `UncompressAsMuchAsPossible` returns however many
//!    bytes it managed to emit, so partial garbage can look like success. Here
//!    `SNAPPY_INVALID_INPUT` is an `Err`, and a truncated/corrupt frame never
//!    yields a partial buffer.
//! 5. **Capacity granularity (over-strict, by design).** `snappy_compress`
//!    rejects any output buffer smaller than
//!    `snappy_max_compressed_length(input.len())`, even when the data would in
//!    fact fit — the check is against the worst case, not the actual result.
//!    Callers must size with [`Compressor::max_compressed_size`], which this
//!    type overrides to return exactly that bound (the trait's generic default
//!    is *not* a safe size for Snappy).

use std::ffi::{c_char, c_int};

use crate::{CompressError, Compressor, LibraryId, Preset};

// ---------------------------------------------------------------------------
// Raw FFI: snappy-c.h
// ---------------------------------------------------------------------------

/// `snappy_status` — a C enum, i.e. `int` in the ABI.
type SnappyStatus = c_int;

const SNAPPY_OK: SnappyStatus = 0;
const SNAPPY_INVALID_INPUT: SnappyStatus = 1;
const SNAPPY_BUFFER_TOO_SMALL: SnappyStatus = 2;

extern "C" {
    /// `compressed_length` is INOUT: capacity in, bytes written out.
    fn snappy_compress(
        input: *const c_char,
        input_length: usize,
        compressed: *mut c_char,
        compressed_length: *mut usize,
    ) -> SnappyStatus;

    /// `uncompressed_length` is INOUT: capacity in, bytes written out.
    fn snappy_uncompress(
        compressed: *const c_char,
        compressed_length: usize,
        uncompressed: *mut c_char,
        uncompressed_length: *mut usize,
    ) -> SnappyStatus;

    fn snappy_max_compressed_length(source_length: usize) -> usize;

    fn snappy_uncompressed_length(
        compressed: *const c_char,
        compressed_length: usize,
        result: *mut usize,
    ) -> SnappyStatus;

    fn snappy_validate_compressed_buffer(
        compressed: *const c_char,
        compressed_length: usize,
    ) -> SnappyStatus;
}

/// Map a non-OK `snappy_status` to a [`CompressError`]. Snappy's C API has no
/// `strerror` equivalent, so the text is ours.
fn status_err(what: &str, rc: SnappyStatus) -> CompressError {
    let reason = match rc {
        SNAPPY_INVALID_INPUT => "SNAPPY_INVALID_INPUT (corrupt or truncated input)".to_string(),
        SNAPPY_BUFFER_TOO_SMALL => "SNAPPY_BUFFER_TOO_SMALL (output buffer too small)".to_string(),
        other => format!("unknown snappy_status {other}"),
    };
    CompressError(format!("snappy: {what}: {reason}"))
}

// ---------------------------------------------------------------------------
// Safe free functions
// ---------------------------------------------------------------------------

/// Worst-case compressed size of `input_size` bytes (`snappy_max_compressed_length`).
///
/// This is also the *minimum* output capacity [`compress`] accepts.
pub fn max_compressed_length(input_size: usize) -> usize {
    // SAFETY: a pure arithmetic function on a scalar — no pointers, no state,
    // no failure mode.
    unsafe { snappy_max_compressed_length(input_size) }
}

/// The size `compressed` will decompress to, read from its block header.
///
/// Lets callers size a decompression buffer without trial-and-error. Errors if
/// `compressed` is not a valid Snappy block.
pub fn uncompressed_length(compressed: &[u8]) -> Result<usize, CompressError> {
    let mut out: usize = 0;
    // SAFETY: `compressed` is a live slice, so its pointer is valid for
    // `compressed.len()` readable bytes (Snappy only reads the leading varint
    // and never writes through it — hence `*const`). `&mut out` is a valid,
    // uniquely-borrowed out-pointer. Empty slices yield a dangling-but-aligned
    // pointer, which Snappy will not dereference because the length is 0.
    let rc = unsafe {
        snappy_uncompressed_length(compressed.as_ptr() as *const c_char, compressed.len(), &mut out)
    };
    if rc != SNAPPY_OK {
        return Err(status_err("snappy_uncompressed_length", rc));
    }
    Ok(out)
}

/// True if `compressed` is a well-formed Snappy block
/// (`snappy_validate_compressed_buffer`).
pub fn is_valid_compressed(compressed: &[u8]) -> bool {
    // SAFETY: same contract as `uncompressed_length` — read-only access to a
    // live slice's bytes, bounded by the length we pass alongside it.
    let rc = unsafe {
        snappy_validate_compressed_buffer(compressed.as_ptr() as *const c_char, compressed.len())
    };
    rc == SNAPPY_OK
}

/// Compress `input` into `output`; returns bytes written.
///
/// `output` must have capacity of at least [`max_compressed_length`]`(input.len())`
/// or this returns `Err` (Snappy checks the worst case, not the actual size).
pub fn compress(output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
    // INOUT: capacity in, bytes written out.
    let mut out_len: usize = output.len();
    // SAFETY: `input` is valid for `input.len()` reads and `output` is valid
    // for `output.len()` writes; both lengths are passed to Snappy alongside
    // the pointers, and `out_len` starts as the true capacity, so Snappy
    // writes at most `output.len()` bytes (it returns BUFFER_TOO_SMALL rather
    // than overrun). The slices are disjoint: `output` is `&mut`, so Rust's
    // aliasing rules guarantee `input` cannot overlap it. `&mut out_len` is a
    // valid out-pointer. Dangling pointers from empty slices are never
    // dereferenced because their length is 0.
    let rc = unsafe {
        snappy_compress(
            input.as_ptr() as *const c_char,
            input.len(),
            output.as_mut_ptr() as *mut c_char,
            &mut out_len,
        )
    };
    if rc != SNAPPY_OK {
        return Err(status_err("snappy_compress", rc));
    }
    // Defense in depth: a library that wrote past the capacity would already
    // have corrupted memory, but a bogus out-length must never reach callers
    // as a slice bound.
    if out_len > output.len() {
        return Err(CompressError(format!(
            "snappy: snappy_compress reported {out_len} bytes written into a \
             {}-byte buffer",
            output.len()
        )));
    }
    Ok(out_len)
}

/// Decompress `input` into `output`; returns bytes written.
///
/// Errors if `input` is not a valid Snappy block or `output` is too small to
/// hold the full payload (see [`uncompressed_length`] to size it exactly).
pub fn decompress(output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
    // INOUT: capacity in, bytes written out.
    let mut out_len: usize = output.len();
    // SAFETY: identical contract to `compress` — pointers are paired with the
    // true lengths of live slices, `out_len` carries the real capacity in, and
    // `output: &mut` cannot alias `input`. Snappy bounds-checks against
    // `out_len` and returns BUFFER_TOO_SMALL instead of overrunning.
    let rc = unsafe {
        snappy_uncompress(
            input.as_ptr() as *const c_char,
            input.len(),
            output.as_mut_ptr() as *mut c_char,
            &mut out_len,
        )
    };
    if rc != SNAPPY_OK {
        return Err(status_err("snappy_uncompress", rc));
    }
    if out_len > output.len() {
        return Err(CompressError(format!(
            "snappy: snappy_uncompress reported {out_len} bytes written into a \
             {}-byte buffer",
            output.len()
        )));
    }
    Ok(out_len)
}

// ---------------------------------------------------------------------------
// Compressor impl
// ---------------------------------------------------------------------------

/// Snappy codec — the Rust face of C++ `ctp::Snappy`.
///
/// Stateless: Snappy's C API keeps no context, so this is `Send + Sync` and a
/// single instance can be shared across workers.
#[derive(Debug, Clone, Copy, Default)]
pub struct SnappyCompressor {
    /// Requested preset. **Ignored** — Snappy has no levels; kept only so
    /// [`Self::preset`] can report what the factory asked for. See module docs.
    preset: Preset,
}

impl SnappyCompressor {
    /// Construct the codec. `preset` is accepted for factory uniformity and
    /// has **no effect** on the output — Snappy has no compression levels.
    pub fn new(preset: Preset) -> Self {
        Self { preset }
    }

    /// The preset this was constructed with. Informational only.
    pub fn preset(&self) -> Preset {
        self.preset
    }
}

impl Compressor for SnappyCompressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        compress(output, input)
    }

    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        decompress(output, input)
    }

    fn library(&self) -> LibraryId {
        LibraryId::Snappy
    }

    /// Snappy's exact worst case. Overriding the trait default is **required**,
    /// not an optimization: `snappy_compress` rejects any buffer smaller than
    /// this bound, and the generic default (`n + n/8 + 1024`) is below it for
    /// large inputs — Snappy's bound is roughly `32 + n + n/6`.
    fn max_compressed_size(&self, input_size: usize) -> usize {
        max_compressed_length(input_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compressor_by_name, compressor_for};

    /// Log-like records plus a binary tail: compressible, but not trivially so.
    fn realistic_data() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..2000 {
            v.extend_from_slice(
                format!(
                    "[2026-07-16T12:{:02}:{:02}Z] worker={} blob=blob_{:06} \
                     op=put size={} status=ok\n",
                    i % 60,
                    (i * 7) % 60,
                    i % 16,
                    i,
                    1024 + (i % 977),
                )
                .as_bytes(),
            );
        }
        // Pseudo-random tail so the input is not uniformly compressible.
        let mut x: u32 = 0x1234_5678;
        for _ in 0..8192 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            v.push((x >> 24) as u8);
        }
        v
    }

    #[test]
    fn round_trip_is_byte_exact() {
        let c = SnappyCompressor::new(Preset::Balanced);
        let input = realistic_data();

        let mut buf = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut buf, &input).unwrap();
        let compressed = &buf[..n];

        // The block must be well-formed and self-describing.
        assert!(is_valid_compressed(compressed));
        assert_eq!(uncompressed_length(compressed).unwrap(), input.len());

        let mut out = vec![0u8; input.len()];
        let m = c.decompress(&mut out, compressed).unwrap();
        assert_eq!(m, input.len());
        // Snappy is lossless: byte-exact, not approximate.
        assert_eq!(out, input);
    }

    #[test]
    fn compressible_data_actually_shrinks() {
        let c = SnappyCompressor::new(Preset::Best);
        let input = vec![b'A'; 256 * 1024];

        let mut buf = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut buf, &input).unwrap();
        assert!(
            n < input.len() / 10,
            "256KiB of 'A' should compress hard, got {n} bytes"
        );

        let mut out = vec![0u8; input.len()];
        assert_eq!(c.decompress(&mut out, &buf[..n]).unwrap(), input.len());
        assert_eq!(out, input);

        // Realistic text should shrink too, if less dramatically.
        let text = realistic_data();
        let mut buf2 = vec![0u8; c.max_compressed_size(text.len())];
        let n2 = c.compress(&mut buf2, &text).unwrap();
        assert!(n2 < text.len(), "log text should shrink: {n2} vs {}", text.len());
    }

    #[test]
    fn output_buffer_too_small_errors_without_corrupting_memory() {
        let c = SnappyCompressor::new(Preset::Fast);
        let input = realistic_data();

        // (1) compress into an undersized buffer. The C++ wrapper's
        // RawCompress would overrun here; the C API must return Err.
        // Guard bytes flank the buffer so any write past the end is caught.
        let mut guarded = vec![0xAAu8; 1024 + 64];
        let (region, guard) = guarded.split_at_mut(1024);
        region.fill(0);
        assert!(c.compress(region, &input).is_err());
        assert!(guard.iter().all(|&b| b == 0xAA), "compress overran its buffer");

        // A zero-length output buffer is also just an error, not a crash.
        assert!(c.compress(&mut [], &input).is_err());

        // (2) decompress into an undersized buffer.
        let mut buf = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut buf, &input).unwrap();
        let mut small = vec![0u8; input.len() / 2];
        assert!(c.decompress(&mut small, &buf[..n]).is_err());

        // Exactly-sized succeeds — the bound is not off-by-one.
        let mut exact = vec![0u8; input.len()];
        assert_eq!(c.decompress(&mut exact, &buf[..n]).unwrap(), input.len());

        // (3) corrupt input is an Err, not a partial buffer (diverges from
        // C++ UncompressAsMuchAsPossible, which returns partial output).
        let mut corrupt = buf[..n].to_vec();
        corrupt.truncate(n / 2);
        let mut out = vec![0u8; input.len()];
        assert!(c.decompress(&mut out, &corrupt).is_err());
        assert!(c.decompress(&mut out, b"not snappy at all").is_err());
    }

    #[test]
    fn empty_input_round_trips() {
        let c = SnappyCompressor::new(Preset::Balanced);

        let mut buf = vec![0u8; c.max_compressed_size(0)];
        let n = c.compress(&mut buf, &[]).unwrap();
        // The empty block is still a real (tiny, non-zero) Snappy frame.
        assert!(n > 0 && n <= buf.len());
        assert!(is_valid_compressed(&buf[..n]));
        assert_eq!(uncompressed_length(&buf[..n]).unwrap(), 0);

        // Zero-length output for a zero-length payload is Ok(0) — the C++
        // wrapper reports this as failure (`output_size != 0`); see module docs.
        assert_eq!(c.decompress(&mut [], &buf[..n]).unwrap(), 0);

        let mut out = vec![0u8; 16];
        assert_eq!(c.decompress(&mut out, &buf[..n]).unwrap(), 0);
    }

    #[test]
    fn preset_is_ignored() {
        // Snappy has no levels: every preset must produce identical bytes.
        let input = realistic_data();
        let mut outs = Vec::new();
        for p in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let c = SnappyCompressor::new(p);
            assert_eq!(c.preset(), p);
            let mut buf = vec![0u8; c.max_compressed_size(input.len())];
            let n = c.compress(&mut buf, &input).unwrap();
            buf.truncate(n);
            outs.push(buf);
        }
        assert_eq!(outs[0], outs[1]);
        assert_eq!(outs[1], outs[2]);
    }

    #[test]
    fn max_compressed_size_bounds_the_worst_case() {
        // Incompressible data must still fit in the advertised bound — this is
        // the contract callers allocate against.
        let c = SnappyCompressor::new(Preset::Balanced);
        let mut x: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let input: Vec<u8> = (0..64 * 1024)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 32) as u8
            })
            .collect();

        let bound = c.max_compressed_size(input.len());
        assert!(bound >= input.len());
        let mut buf = vec![0u8; bound];
        let n = c.compress(&mut buf, &input).unwrap();
        assert!(n <= bound);

        let mut out = vec![0u8; input.len()];
        assert_eq!(c.decompress(&mut out, &buf[..n]).unwrap(), input.len());
        assert_eq!(out, input);
    }

    #[test]
    fn factory_builds_snappy() {
        let c = compressor_for(LibraryId::Snappy, Preset::Balanced).expect("snappy feature on");
        assert_eq!(c.library(), LibraryId::Snappy);
        assert!(!c.library().is_lossy());

        let c = compressor_by_name("SNAPPY", Preset::Fast).expect("name lookup");
        assert_eq!(c.library(), LibraryId::Snappy);

        // Round-trip through the boxed trait object, as the factory's callers do.
        let input = realistic_data();
        let mut buf = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut buf, &input).unwrap();
        let mut out = vec![0u8; input.len()];
        assert_eq!(c.decompress(&mut out, &buf[..n]).unwrap(), input.len());
        assert_eq!(out, input);
    }
}
