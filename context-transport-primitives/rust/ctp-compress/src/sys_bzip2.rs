// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! bzip2 binding — hand-rolled FFI over **libbz2**, plus the safe
//! [`Compressor`] implementation the factory hands out for
//! [`LibraryId::Bzip2`].
//!
//! # What is wrapped
//!
//! Julian Seward's `libbz2` (Burrows-Wheeler block-sorting compressor), via
//! its two one-shot buffer-to-buffer entry points:
//! `BZ2_bzBuffToBuffCompress` / `BZ2_bzBuffToBuffDecompress`. These are the
//! exact entry points the C++ wrapper uses (`clio_ctp/compress/bzip2.h`,
//! `ctp::Bzip2`; and `clio_ctp/compress/lossless_modes.h`,
//! `ctp::Bzip2WithModes` — the class the C++ factory actually constructs for
//! the name `"bzip2"`), so the bitstreams the two stacks produce are
//! interchangeable.
//!
//! # Version / ABI assumptions
//!
//! * Targets the **bzip2 1.0.x** C ABI (validated against 1.0.8 —
//!   `libbz2.so.1.0`, the devcontainer's Debian `libbz2-dev`). The
//!   buffer-to-buffer signatures and the `BZ_*` return codes have been
//!   stable across all of 1.0.x; the `.so.1.0` soname has not changed since
//!   2002. [`version`] reports what is actually linked at runtime.
//! * Plain C calling convention: on non-Windows, bzlib.h's `BZ_API(f)`
//!   expands to bare `f` with no decoration, so `extern "C"` matches. A
//!   Windows DLL build would use `WINAPI` (stdcall) on 32-bit x86; that
//!   configuration is out of scope here (CTP builds bzip2 on Linux/macOS,
//!   and x86-64 has one calling convention regardless).
//! * `destLen`/`sourceLen` are C `unsigned int` (32-bit on every platform
//!   CTP targets), which bounds a single call to 4 GiB — see below.
//! * The `BuffToBuff` entry points are stateless and reentrant: each call
//!   builds its own `bz_stream` on the stack and uses the default
//!   `malloc`/`free`. There is no global library state to initialize and no
//!   lock needed, which is why [`Bzip2Compressor`] is trivially `Send +
//!   Sync` and takes `&self`.
//!
//! # Preset mapping
//!
//! `blockSize100k`, matching `Bzip2WithModes`'s switch exactly:
//! [`Preset::Fast`] → 1, [`Preset::Balanced`] → 6, [`Preset::Best`] → 9.
//! `verbosity = 0` and `workFactor = 30` are likewise carried over from the
//! C++ wrapper's member initializers, and `small = 0` (the faster,
//! more-memory decompression path) from its `Decompress`.
//!
//! # Divergence from the C++ wrapper
//!
//! 1. **Oversized buffers are rejected, not silently truncated.** The C++
//!    wrapper does `unsigned int output_size_int = output_size;` on a
//!    `size_t` — for a buffer ≥ 4 GiB that wraps, and libbz2 is then told a
//!    small buffer is available (or, worse on the input side, that a huge
//!    input is short), so it quietly compresses the wrong length. Here a
//!    length that does not fit `unsigned int` is a [`CompressError`].
//! 2. **Errors carry a reason.** C++ returns `ret == BZ_OK` as a bare
//!    `bool`, discarding *which* `BZ_*` code came back. This maps every code
//!    to a message (see [`bz_err_str`]) — the crate-wide improvement
//!    documented on [`CompressError`]. Callers wanting the old bool use
//!    `.is_ok()`.
//! 3. **`max_compressed_size` is bzip2's documented bound** (1% + 600 bytes)
//!    rather than the trait's generic default; the C++ side has no such
//!    method at all.
//!
//! Compressed output is byte-identical to the C++ wrapper's for the same
//! preset and input: same entry point, same parameters.

use std::ffi::{c_char, c_int, c_uint, CStr};

use crate::{CompressError, Compressor, LibraryId, Preset};

// ---------------------------------------------------------------------------
// Raw FFI — bzlib.h (bzip2 1.0.x)
// ---------------------------------------------------------------------------

// Return codes (bzlib.h). Only the ones these two entry points can produce
// are named; `bz_err_str` still handles the rest defensively.
const BZ_OK: c_int = 0;
const BZ_SEQUENCE_ERROR: c_int = -1;
const BZ_PARAM_ERROR: c_int = -2;
const BZ_MEM_ERROR: c_int = -3;
const BZ_DATA_ERROR: c_int = -4;
const BZ_DATA_ERROR_MAGIC: c_int = -5;
const BZ_IO_ERROR: c_int = -6;
const BZ_UNEXPECTED_EOF: c_int = -7;
const BZ_OUTBUFF_FULL: c_int = -8;
const BZ_CONFIG_ERROR: c_int = -9;

extern "C" {
    /// `dest`/`source` are `char*` (not `const char*`) in bzlib.h — bzip2
    /// predates widespread const-correctness. `source` is only ever read.
    fn BZ2_bzBuffToBuffCompress(
        dest: *mut c_char,
        dest_len: *mut c_uint,
        source: *mut c_char,
        source_len: c_uint,
        block_size_100k: c_int,
        verbosity: c_int,
        work_factor: c_int,
    ) -> c_int;

    fn BZ2_bzBuffToBuffDecompress(
        dest: *mut c_char,
        dest_len: *mut c_uint,
        source: *mut c_char,
        source_len: c_uint,
        small: c_int,
        verbosity: c_int,
    ) -> c_int;

    fn BZ2_bzlibVersion() -> *const c_char;
}

/// Human-readable name for a `BZ_*` return code.
fn bz_err_str(code: c_int) -> &'static str {
    match code {
        BZ_OK => "BZ_OK",
        BZ_SEQUENCE_ERROR => "BZ_SEQUENCE_ERROR (bad call order)",
        BZ_PARAM_ERROR => "BZ_PARAM_ERROR (invalid argument)",
        BZ_MEM_ERROR => "BZ_MEM_ERROR (out of memory)",
        BZ_DATA_ERROR => "BZ_DATA_ERROR (corrupt input)",
        BZ_DATA_ERROR_MAGIC => "BZ_DATA_ERROR_MAGIC (not a bzip2 stream)",
        BZ_IO_ERROR => "BZ_IO_ERROR",
        BZ_UNEXPECTED_EOF => "BZ_UNEXPECTED_EOF (truncated input)",
        BZ_OUTBUFF_FULL => "BZ_OUTBUFF_FULL (output buffer too small)",
        BZ_CONFIG_ERROR => "BZ_CONFIG_ERROR (miscompiled libbz2)",
        _ => "unknown BZ_* code",
    }
}

/// The linked libbz2's version string, e.g. `"1.0.8, 13-Jul-2019"`.
pub fn version() -> String {
    // SAFETY: BZ2_bzlibVersion takes no arguments, touches no library state,
    // and cannot fail; it just returns a pointer to a static NUL-terminated
    // literal, valid for the process lifetime and never written.
    let p = unsafe { BZ2_bzlibVersion() };
    if p.is_null() {
        return String::new();
    }
    // SAFETY: non-null, static, NUL-terminated per the above.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Narrow a buffer length to bzip2's `unsigned int`, rather than truncating
/// the way the C++ wrapper's implicit `size_t` → `unsigned int` does.
fn as_c_uint(len: usize, what: &str) -> Result<c_uint, CompressError> {
    c_uint::try_from(len).map_err(|_| {
        CompressError(format!(
            "bzip2: {what} is {len} bytes; libbz2's buffer-to-buffer API takes \
             an unsigned int length (max {})",
            c_uint::MAX
        ))
    })
}

// ---------------------------------------------------------------------------
// Safe wrapper
// ---------------------------------------------------------------------------

/// bzip2 codec — the Rust face of C++ `ctp::Bzip2WithModes`.
///
/// Cheap to construct and to keep: it holds only libbz2's three tuning
/// integers, and every call is a self-contained one-shot into libbz2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bzip2Compressor {
    /// 1–9; block size in units of 100k. Also picks the compression ratio.
    block_size_100k: c_int,
    /// 0 = silent. The C++ wrapper never raises this, so neither do we:
    /// libbz2 logs to stderr, which a library has no business doing.
    verbosity: c_int,
    /// 0–250; effort spent before falling back to the fallback sort.
    /// 30 is libbz2's default and the C++ wrapper's value.
    work_factor: c_int,
}

impl Bzip2Compressor {
    /// Build the codec for `preset` (C++ `Bzip2WithModes(LosslessMode)`).
    pub fn new(preset: Preset) -> Self {
        let block_size_100k = match preset {
            Preset::Fast => 1,
            Preset::Balanced => 6,
            Preset::Best => 9,
        };
        Self {
            block_size_100k,
            verbosity: 0,
            work_factor: 30,
        }
    }

    /// The `blockSize100k` handed to libbz2 (1–9).
    pub fn block_size_100k(&self) -> i32 {
        self.block_size_100k
    }
}

impl Default for Bzip2Compressor {
    fn default() -> Self {
        Self::new(Preset::default())
    }
}

impl Compressor for Bzip2Compressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        // libbz2 rejects a NULL dest with BZ_PARAM_ERROR, and an empty slice
        // yields a dangling (not NULL) pointer, so catch this here and say
        // something useful instead. A 0-byte output can never hold a bzip2
        // stream anyway — the header alone is 14 bytes.
        if output.is_empty() {
            return Err(CompressError(
                "bzip2: output buffer is empty (a bzip2 stream is >= 14 bytes)".into(),
            ));
        }
        let source_len = as_c_uint(input.len(), "input")?;
        let mut dest_len = as_c_uint(output.len(), "output buffer")?;

        // SAFETY:
        // * `dest` is `output`'s pointer and `dest_len` starts as exactly
        //   `output.len()`; libbz2 treats it as INOUT — it writes at most
        //   `dest_len` bytes and lowers it to the count written, so writes
        //   stay in bounds of the `&mut` slice we own for this call.
        // * `source` is cast away from `*const` only because bzlib.h omits
        //   `const`; `BZ2_bzBuffToBuffCompress` only reads it (it feeds
        //   `strm.next_in`), so no write ever occurs through a pointer
        //   derived from our shared `&[u8]`.
        // * Both pointers may be dangling-but-nonnull for a 0-length input:
        //   libbz2 will not dereference `source` when `sourceLen == 0`, and
        //   `output` is non-empty per the check above.
        // * Lengths are validated to fit `c_uint` above, so libbz2's view of
        //   both buffers matches their real size.
        // * The call is self-contained (its own stack `bz_stream`, default
        //   allocator) and keeps no pointer past return.
        let rc = unsafe {
            BZ2_bzBuffToBuffCompress(
                output.as_mut_ptr() as *mut c_char,
                &mut dest_len,
                input.as_ptr() as *mut c_char,
                source_len,
                self.block_size_100k,
                self.verbosity,
                self.work_factor,
            )
        };
        if rc != BZ_OK {
            return Err(CompressError(format!(
                "bzip2: BZ2_bzBuffToBuffCompress failed: {} (code {rc})",
                bz_err_str(rc)
            )));
        }
        Ok(dest_len as usize)
    }

    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        if output.is_empty() {
            return Err(CompressError("bzip2: output buffer is empty".into()));
        }
        let source_len = as_c_uint(input.len(), "input")?;
        let mut dest_len = as_c_uint(output.len(), "output buffer")?;

        // SAFETY: identical contract to `compress` above — `dest_len` is the
        // true capacity of the `&mut` slice and bounds every write; `source`
        // is only read despite bzlib.h's missing `const`; lengths fit
        // `c_uint`; `small = 0` and `verbosity = 0` are in libbz2's accepted
        // ranges (0/1 and 0..=4), so no BZ_PARAM_ERROR from those. Corrupt
        // or truncated input is reported as BZ_DATA_ERROR* /
        // BZ_UNEXPECTED_EOF, not by overrunning the buffer.
        let rc = unsafe {
            BZ2_bzBuffToBuffDecompress(
                output.as_mut_ptr() as *mut c_char,
                &mut dest_len,
                input.as_ptr() as *mut c_char,
                source_len,
                0, // small: 0 = faster path, matching the C++ wrapper
                self.verbosity,
            )
        };
        if rc != BZ_OK {
            return Err(CompressError(format!(
                "bzip2: BZ2_bzBuffToBuffDecompress failed: {} (code {rc})",
                bz_err_str(rc)
            )));
        }
        Ok(dest_len as usize)
    }

    fn library(&self) -> LibraryId {
        LibraryId::Bzip2
    }

    /// bzip2's documented worst case for `BZ2_bzBuffToBuffCompress`: "1%
    /// larger than the uncompressed data, plus six hundred extra bytes".
    fn max_compressed_size(&self, input_size: usize) -> usize {
        input_size
            .saturating_add(input_size / 100)
            .saturating_add(600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRESETS: [Preset; 3] = [Preset::Fast, Preset::Balanced, Preset::Best];

    /// Deterministic xorshift — incompressible bytes without a dev-dependency.
    fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    /// Realistic payload: structured log text (compressible) followed by a
    /// packed binary record array (less so) — a plausible CTP blob.
    fn realistic_data() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..2000 {
            v.extend_from_slice(
                format!(
                    "[2026-07-15T12:{:02}:{:02}Z] node={} rank={} op=put blob=blob_{} bytes={}\n",
                    (i / 60) % 60,
                    i % 60,
                    i % 8,
                    i % 32,
                    i,
                    4096 + (i * 17) % 900_000
                )
                .as_bytes(),
            );
        }
        for i in 0..4000u64 {
            v.extend_from_slice(&i.to_le_bytes());
            v.extend_from_slice(&(i as f64 * 0.5).to_le_bytes());
        }
        v.extend_from_slice(&pseudo_random(8192, 0xC0FFEE));
        v
    }

    #[test]
    fn round_trip_is_byte_exact_at_every_preset() {
        let data = realistic_data();
        for preset in PRESETS {
            let c = Bzip2Compressor::new(preset);
            let mut buf = vec![0u8; c.max_compressed_size(data.len())];
            let n = c.compress(&mut buf, &data).unwrap();

            let mut back = vec![0u8; data.len() + 64];
            let m = c.decompress(&mut back, &buf[..n]).unwrap();

            // Lossless: byte-exact, exact length.
            assert_eq!(m, data.len(), "preset {:?}: length changed", preset);
            assert_eq!(&back[..m], &data[..], "preset {:?}: bytes changed", preset);
        }
    }

    #[test]
    fn max_compressed_size_holds_for_incompressible_data() {
        // The documented 1%+600 bound must survive data bzip2 cannot shrink.
        let data = pseudo_random(256 * 1024, 0xDEADBEEF);
        let c = Bzip2Compressor::new(Preset::Best);
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).expect("bound must not overflow");
        assert!(n <= c.max_compressed_size(data.len()));
    }

    #[test]
    fn highly_compressible_data_shrinks() {
        let data = vec![0x5Au8; 512 * 1024];
        let c = Bzip2Compressor::new(Preset::Balanced);
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();

        assert!(
            n < data.len() / 100,
            "512 KiB of one repeated byte should collapse to well under 1%, got {n}"
        );

        let mut back = vec![0u8; data.len()];
        let m = c.decompress(&mut back, &buf[..n]).unwrap();
        assert_eq!(&back[..m], &data[..]);
    }

    #[test]
    fn output_buffer_too_small_is_err_and_stays_in_bounds() {
        let data = realistic_data();
        let c = Bzip2Compressor::new(Preset::Balanced);

        // Give compress a deliberately tiny window inside a poisoned arena;
        // the tail proves libbz2 honoured destLen instead of running past it.
        let mut arena = vec![0xAAu8; 64 * 1024];
        let (window, tail) = arena.split_at_mut(64);
        let err = c.compress(window, &data).unwrap_err();
        assert!(
            err.0.contains("BZ_OUTBUFF_FULL"),
            "expected an overflow error, got: {}",
            err.0
        );
        assert!(
            tail.iter().all(|&b| b == 0xAA),
            "libbz2 wrote past the output slice"
        );

        // Same on the decompress side: a valid stream, too small a sink.
        let mut good = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut good, &data).unwrap();

        let mut arena = vec![0xAAu8; 64 * 1024];
        let (window, tail) = arena.split_at_mut(128);
        let err = c.decompress(window, &good[..n]).unwrap_err();
        assert!(
            err.0.contains("BZ_OUTBUFF_FULL"),
            "expected an overflow error, got: {}",
            err.0
        );
        assert!(
            tail.iter().all(|&b| b == 0xAA),
            "libbz2 wrote past the output slice"
        );

        // A zero-length sink is rejected before libbz2 sees a dangling ptr.
        assert!(c.compress(&mut [], &data).is_err());
        assert!(c.decompress(&mut [], &good[..n]).is_err());
    }

    #[test]
    fn empty_input_round_trips() {
        for preset in PRESETS {
            let c = Bzip2Compressor::new(preset);
            let mut buf = vec![0u8; c.max_compressed_size(0)];
            // bzip2 has no zero-length encoding: empty input still emits a
            // complete stream (header + EOS + CRC).
            let n = c.compress(&mut buf, &[]).unwrap();
            assert!(n > 0, "preset {:?}: expected a real stream", preset);

            let mut back = vec![0u8; 32];
            let m = c.decompress(&mut back, &buf[..n]).unwrap();
            assert_eq!(m, 0, "preset {:?}: empty in, empty out", preset);
        }
    }

    #[test]
    fn corrupt_input_is_err_not_panic() {
        let c = Bzip2Compressor::new(Preset::Balanced);
        let mut back = vec![0u8; 4096];
        // Not a bzip2 stream at all.
        assert!(c.decompress(&mut back, b"definitely not bzip2 data").is_err());

        // A real stream with a mangled tail.
        let data = realistic_data();
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();
        buf[n / 2] ^= 0xFF;
        let mut back = vec![0u8; data.len() + 64];
        // Either a data error, or (improbably) a benign bit — must not panic
        // and must not silently return the original bytes.
        if let Ok(m) = c.decompress(&mut back, &buf[..n]) {
            assert_ne!(&back[..m], &data[..], "corruption must not be invisible");
        }
    }

    #[test]
    fn preset_maps_to_cxx_block_sizes() {
        // Must match Bzip2WithModes(LosslessMode) in lossless_modes.h — the
        // block size is encoded in the stream header, so a mismatch would
        // change the bytes CTP writes.
        assert_eq!(Bzip2Compressor::new(Preset::Fast).block_size_100k(), 1);
        assert_eq!(Bzip2Compressor::new(Preset::Balanced).block_size_100k(), 6);
        assert_eq!(Bzip2Compressor::new(Preset::Best).block_size_100k(), 9);
        assert_eq!(
            Bzip2Compressor::new(Preset::Balanced).library(),
            LibraryId::Bzip2
        );
    }

    #[test]
    fn stream_header_records_the_preset() {
        // "BZh" + ASCII block-size digit: proof the preset reached libbz2 and
        // that our streams are ordinary bzip2 (`bunzip2`-readable) files.
        for (preset, digit) in [
            (Preset::Fast, b'1'),
            (Preset::Balanced, b'6'),
            (Preset::Best, b'9'),
        ] {
            let c = Bzip2Compressor::new(preset);
            let mut buf = vec![0u8; 1024];
            let n = c.compress(&mut buf, b"iowarp").unwrap();
            assert!(n > 4);
            assert_eq!(&buf[..3], b"BZh");
            assert_eq!(buf[3], digit, "preset {:?} header", preset);
        }
    }

    #[test]
    fn streams_are_readable_by_the_bunzip2_cli() {
        // Proves the framing is ordinary bzip2 — interchangeable with what
        // the C++ wrapper writes — rather than merely self-consistent.
        // Skips when the CLI is absent, matching ctp-gpu's convention of
        // skipping on a missing external dependency instead of failing.
        let data = realistic_data();
        let c = Bzip2Compressor::new(Preset::Best);
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();

        let path = std::env::temp_dir().join(format!("ctp_bzip2_{}.bz2", std::process::id()));
        if std::fs::write(&path, &buf[..n]).is_err() {
            eprintln!("ctp-compress: temp dir not writable; skipping");
            return;
        }
        let out = match std::process::Command::new("bunzip2").arg("-c").arg(&path).output() {
            Ok(o) => o,
            Err(_) => {
                eprintln!("ctp-compress: bunzip2 not on PATH; skipping");
                let _ = std::fs::remove_file(&path);
                return;
            }
        };
        let _ = std::fs::remove_file(&path);
        assert!(
            out.status.success(),
            "bunzip2 rejected our stream: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(out.stdout, data, "bunzip2 decoded different bytes");
    }

    #[test]
    fn links_a_1_0_x_libbz2() {
        // Guards the ABI assumption in the module docs.
        let v = version();
        assert!(v.starts_with("1.0."), "unexpected libbz2 version: {v:?}");
    }

    #[test]
    fn factory_builds_this_codec() {
        let c = crate::compressor_for(LibraryId::Bzip2, Preset::Best).expect("feature is on");
        assert_eq!(c.library(), LibraryId::Bzip2);
        let data = realistic_data();
        let mut buf = vec![0u8; c.max_compressed_size(data.len())];
        let n = c.compress(&mut buf, &data).unwrap();
        let mut back = vec![0u8; data.len()];
        let m = c.decompress(&mut back, &buf[..n]).unwrap();
        assert_eq!(&back[..m], &data[..]);
    }
}
