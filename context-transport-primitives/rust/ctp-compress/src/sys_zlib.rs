// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! zlib bindings (hand-rolled, dependency-free) and the safe [`Compressor`]
//! over them.
//!
//! # What is wrapped
//!
//! The C library **zlib** (<https://zlib.net>), linked as `libz` by
//! `build.rs` under the `zlib` feature. This is the same library the C++
//! stack uses via `clio_ctp/compress/zlib.h` (`ctp::Zlib`) and
//! `lossless_modes.h` (`ctp::ZlibWithModes`) — the codec is wrapped, not
//! ported, so blobs stay readable across both stacks (rust/MIGRATION.md).
//!
//! # Version / ABI assumptions
//!
//! * zlib **1.2.x or 1.3.x** (devcontainer ships 1.3). The three entry
//!   points used here are ABI-frozen: `compress2` and `uncompress` date to
//!   zlib 1.0/1.1, `compressBound` to 1.2.0. [`version`] reports what is
//!   actually loaded.
//! * `uLong`/`uLongf` are C `unsigned long`, so they are declared as
//!   [`c_ulong`] — 64-bit on LP64 Linux, **32-bit** on LLP64 Windows.
//!   Buffer lengths that do not fit are rejected as errors rather than
//!   truncated (see [`ZlibCompressor::compress`]).
//! * zlib >= 1.2.2.3 is assumed for large buffers: `compress2`/`uncompress`
//!   internally chunk `uLong` lengths into 32-bit `uInt` windows, so inputs
//!   above 4 GiB are handled correctly on LP64. Older zlib truncated.
//! * The build is assumed to be a stock `ZLIB_WINAPI`-free one (plain cdecl
//!   `ZEXPORT`), which is what every Unix build and vcpkg's zlib produce.
//!
//! # Framing
//!
//! Output is the zlib format (RFC 1950): a 2-byte CMF/FLG header, a raw
//! DEFLATE stream, and a trailing Adler-32. This is byte-identical framing
//! to the C++ wrapper, which calls `deflateInit()` — i.e. default
//! `windowBits = 15`, which selects the same zlib wrapper that `compress2`
//! uses internally. Streams therefore interchange in both directions. No
//! CTP-side length prefix is added (matching C++): the caller is expected to
//! know the decompressed size out-of-band, exactly as `BlobInfo` does.
//!
//! # Divergence from the C++ wrapper
//!
//! 1. **One-shot instead of streaming.** C++ drives
//!    `deflateInit`/`deflate(Z_FINISH)`/`deflateEnd` by hand; this module
//!    calls `compress2`/`uncompress`, which perform that exact sequence
//!    internally with the same defaults. The bytes on the wire are the same;
//!    the C++ code additionally leaks nothing here because there is no
//!    `z_stream` to forget to end on an error path.
//! 2. **Preset actually reaches zlib.** `ctp::Zlib` (compress.h) hardcodes
//!    `Z_DEFAULT_COMPRESSION` and ignores the preset entirely. This module
//!    follows `ctp::ZlibWithModes`, the class the factory arm for `"zlib"`
//!    constructs, and maps FAST/BALANCED/BEST → level 1/6/9. Balanced (6) is
//!    what `Z_DEFAULT_COMPRESSION` resolves to, so the default is unchanged.
//! 3. **Errors carry a reason.** C++ returns `bool` and logs; this returns
//!    [`CompressError`] with zlib's own `zError` text. No panics.
//! 4. **`max_compressed_size` is exact.** C++ callers size buffers by hand;
//!    this forwards to `compressBound`, so incompressible input never
//!    spuriously fails with `Z_BUF_ERROR`.

use std::ffi::{c_char, c_int, c_ulong, CStr};

use crate::{CompressError, Compressor, LibraryId, Preset};

// zlib's <zconf.h> types: `Byte` is `unsigned char`, `uLong` is
// `unsigned long`, and the `f` suffix is the historical FAR marker (a no-op
// on every target we build for).
type Bytef = u8;
type ULongf = c_ulong;

extern "C" {
    /// `int compress2(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level)`
    fn compress2(
        dest: *mut Bytef,
        dest_len: *mut ULongf,
        source: *const Bytef,
        source_len: c_ulong,
        level: c_int,
    ) -> c_int;

    /// `int uncompress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen)`
    fn uncompress(
        dest: *mut Bytef,
        dest_len: *mut ULongf,
        source: *const Bytef,
        source_len: c_ulong,
    ) -> c_int;

    /// `uLong compressBound(uLong sourceLen)`
    fn compressBound(source_len: c_ulong) -> c_ulong;

    /// `const char *zlibVersion(void)`
    fn zlibVersion() -> *const c_char;

    /// `const char *zError(int)` — static message for a zlib return code.
    fn zError(err: c_int) -> *const c_char;
}

const Z_OK: c_int = 0;
const Z_NEED_DICT: c_int = 2;
const Z_VERSION_ERROR: c_int = -6;

/// Runtime zlib version string, e.g. `"1.3"` (C++ `ZLIB_VERSION` analog).
pub fn version() -> String {
    // SAFETY: zlibVersion takes no arguments and returns a pointer to a
    // static NUL-terminated string owned by the library; it is never freed.
    unsafe { CStr::from_ptr(zlibVersion()) }
        .to_string_lossy()
        .into_owned()
}

/// Turn a zlib return code into a [`CompressError`].
fn z_err(rc: c_int, what: &str) -> CompressError {
    // zError indexes a fixed table by `Z_NEED_DICT - err`, which is only in
    // bounds for Z_VERSION_ERROR..=Z_NEED_DICT. compress2/uncompress only
    // return codes from that range, but bound it anyway so a surprising code
    // can never turn into an out-of-bounds read.
    if (Z_VERSION_ERROR..=Z_NEED_DICT).contains(&rc) {
        // SAFETY: `rc` is inside zError's documented domain (checked just
        // above), so it returns a pointer to a static NUL-terminated string.
        let msg = unsafe { CStr::from_ptr(zError(rc)) };
        CompressError(format!(
            "zlib {what} failed: {} (code {rc})",
            msg.to_string_lossy()
        ))
    } else {
        CompressError(format!("zlib {what} failed: unknown code {rc}"))
    }
}

/// Narrow a Rust length to zlib's `uLong`, erroring instead of truncating.
/// Only ever fails on LLP64 (Windows), where `uLong` is 32-bit.
fn to_ulong(n: usize, what: &str) -> Result<c_ulong, CompressError> {
    c_ulong::try_from(n).map_err(|_| {
        CompressError(format!(
            "zlib {what}: {n} bytes exceeds zlib's uLong range on this target"
        ))
    })
}

/// zlib codec — the Rust face of C++ `ctp::ZlibWithModes`.
///
/// Holds only the compression level, so it is trivially `Send + Sync`:
/// `compress2`/`uncompress` allocate their own `z_stream` per call and share
/// no mutable state between them.
#[derive(Debug, Clone, Copy)]
pub struct ZlibCompressor {
    /// zlib level 1..=9 (see [`Self::new`]).
    level: c_int,
}

impl ZlibCompressor {
    /// Build a codec for `preset`, mapping it to a zlib level exactly as
    /// C++ `ZlibWithModes` does: FAST→1, BALANCED→6, BEST→9.
    pub fn new(preset: Preset) -> Self {
        let level = match preset {
            Preset::Fast => 1,
            Preset::Balanced => 6,
            Preset::Best => 9,
        };
        Self { level }
    }

    /// The zlib compression level this codec deflates at (1..=9).
    pub fn level(&self) -> i32 {
        self.level
    }
}

impl Compressor for ZlibCompressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        let source_len = to_ulong(input.len(), "compress input")?;
        // INOUT: capacity in, bytes written out (C++ `output_size &`).
        let mut dest_len = to_ulong(output.len(), "compress output")?;

        // SAFETY: `output` and `input` are live slices, so their pointers are
        // non-null and aligned, and each is dereferenceable for exactly the
        // length handed to zlib alongside it. (An empty slice's pointer is
        // dangling-but-non-null; zlib checks for NULL and never dereferences
        // a buffer whose length is 0.) `dest_len` is a live local. compress2
        // writes at most `dest_len` bytes, reads at most `source_len`, and
        // retains no pointer past the call.
        let rc = unsafe {
            compress2(
                output.as_mut_ptr(),
                &mut dest_len,
                input.as_ptr(),
                source_len,
                self.level,
            )
        };
        if rc != Z_OK {
            // Z_BUF_ERROR here means `output` was too small — the C++
            // wrapper's `deflate() != Z_STREAM_END` case.
            return Err(z_err(rc, "compress2"));
        }
        // zlib guarantees dest_len <= capacity; clamp anyway so the value we
        // hand back can always be used to index `output` from safe code.
        Ok((dest_len as usize).min(output.len()))
    }

    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        let source_len = to_ulong(input.len(), "decompress input")?;
        let mut dest_len = to_ulong(output.len(), "decompress output")?;

        // SAFETY: same contract as `compress` above — valid slices, matching
        // lengths, live out-param, no pointer retained. `uncompress` validates
        // the stream itself (header, Adler-32) and returns Z_DATA_ERROR on
        // corrupt input rather than over-reading, and Z_BUF_ERROR rather than
        // writing past `dest_len`.
        let rc = unsafe {
            uncompress(
                output.as_mut_ptr(),
                &mut dest_len,
                input.as_ptr(),
                source_len,
            )
        };
        if rc != Z_OK {
            return Err(z_err(rc, "uncompress"));
        }
        Ok((dest_len as usize).min(output.len()))
    }

    fn library(&self) -> LibraryId {
        LibraryId::Zlib
    }

    fn max_compressed_size(&self, input_size: usize) -> usize {
        match c_ulong::try_from(input_size) {
            // SAFETY: compressBound is pure arithmetic over a scalar — no
            // pointers, no state, no failure mode.
            Ok(n) => (unsafe { compressBound(n) }) as usize,
            // Only reachable on LLP64 with a >4 GiB input, where we cannot
            // ask zlib. Mirror compressBound's own formula (deflate.c) so the
            // bound stays correct; `compress` will reject the length anyway.
            Err(_) => {
                input_size + (input_size >> 12) + (input_size >> 14) + (input_size >> 25) + 13
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64* — deterministic pseudo-random bytes without a dev-dep.
    fn next_rand(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Realistic CTP-ish payload: structured text records interleaved with
    /// float telemetry. Compressible, but not trivially so.
    fn realistic_blob() -> Vec<u8> {
        let mut v = Vec::with_capacity(256 * 1024);
        let mut rng = 0x2545_f491_4f6c_dd1d_u64;
        for i in 0..4096u32 {
            v.extend_from_slice(
                format!(
                    "[ctp] blob={i:06} node=n{:02} tag=periodic_checkpoint bytes=",
                    i % 17
                )
                .as_bytes(),
            );
            for _ in 0..8 {
                let x = next_rand(&mut rng) as f32 / 1e9;
                v.extend_from_slice(&x.to_le_bytes());
            }
            v.push(b'\n');
        }
        v
    }

    /// Incompressible payload (high-entropy bytes).
    fn random_blob(len: usize) -> Vec<u8> {
        let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
        (0..len).map(|_| next_rand(&mut rng) as u8).collect()
    }

    #[test]
    fn links_a_supported_zlib() {
        let v = version();
        assert!(
            v.starts_with("1."),
            "wrapper assumes the zlib 1.x ABI, got {v:?}"
        );
    }

    /// (1) Round trip over realistic data — zlib is lossless, so byte-exact.
    #[test]
    fn roundtrip_is_byte_exact_at_every_preset() {
        let input = realistic_blob();
        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let c = ZlibCompressor::new(preset);
            let mut packed = vec![0u8; c.max_compressed_size(input.len())];
            let n = c.compress(&mut packed, &input).unwrap();
            assert!(n <= packed.len());

            let mut back = vec![0u8; input.len()];
            let m = c.decompress(&mut back, &packed[..n]).unwrap();
            assert_eq!(m, input.len(), "preset {}", preset.name());
            assert_eq!(back, input, "preset {} corrupted data", preset.name());
        }
    }

    /// (2) Highly-compressible data must actually shrink.
    #[test]
    fn highly_compressible_data_shrinks() {
        let input = vec![b'A'; 64 * 1024];
        let c = ZlibCompressor::new(Preset::Balanced);
        let mut packed = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut packed, &input).unwrap();
        assert!(
            n < input.len() / 100,
            "64 KiB of one byte should collapse; got {n} bytes"
        );

        let mut back = vec![0u8; input.len()];
        assert_eq!(c.decompress(&mut back, &packed[..n]).unwrap(), input.len());
        assert_eq!(back, input);
    }

    /// (3) Output too small → Err, and zlib stays inside the slice we gave it.
    #[test]
    fn output_too_small_errs_without_writing_past_the_slice() {
        let input = realistic_blob();
        let c = ZlibCompressor::new(Preset::Balanced);

        // 64-byte output window inside a 4 KiB canary region.
        let mut region = vec![0xAAu8; 4096];
        let err = c.compress(&mut region[..64], &input).unwrap_err();
        assert!(err.0.contains("compress2"), "{err}");
        assert!(
            region[64..].iter().all(|&b| b == 0xAA),
            "compress2 wrote past the end of the output slice"
        );

        // Same for the decompress direction.
        let mut packed = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut packed, &input).unwrap();
        let mut region = vec![0xAAu8; 4096];
        let err = c.decompress(&mut region[..64], &packed[..n]).unwrap_err();
        assert!(err.0.contains("uncompress"), "{err}");
        assert!(
            region[64..].iter().all(|&b| b == 0xAA),
            "uncompress wrote past the end of the output slice"
        );
    }

    /// (4) Empty input: compresses to a bare zlib frame, decompresses to 0.
    #[test]
    fn empty_input_roundtrips() {
        let c = ZlibCompressor::new(Preset::Balanced);
        let mut packed = vec![0u8; c.max_compressed_size(0)];
        let n = c.compress(&mut packed, &[]).unwrap();
        assert!(n > 0, "zlib still emits a header + Adler-32 for empty input");

        let mut back = vec![0u8; 16];
        assert_eq!(c.decompress(&mut back, &packed[..n]).unwrap(), 0);
    }

    /// compressBound must cover even data that grows under DEFLATE.
    #[test]
    fn max_compressed_size_bounds_incompressible_data() {
        let input = random_blob(128 * 1024);
        let c = ZlibCompressor::new(Preset::Best);
        let mut packed = vec![0u8; c.max_compressed_size(input.len())];
        // Would be Z_BUF_ERROR if compressBound were bound incorrectly.
        let n = c.compress(&mut packed, &input).unwrap();
        assert!(n <= packed.len());

        let mut back = vec![0u8; input.len()];
        assert_eq!(c.decompress(&mut back, &packed[..n]).unwrap(), input.len());
        assert_eq!(back, input);
    }

    /// Framing check: the C++ wrapper's `deflateInit()` emits the RFC 1950
    /// zlib wrapper; `compress2` must emit the same or blobs written by the
    /// C++ stack would not decode here (and vice versa).
    #[test]
    fn emits_rfc1950_framing_like_the_cpp_wrapper() {
        let c = ZlibCompressor::new(Preset::Balanced);
        let mut packed = vec![0u8; c.max_compressed_size(1024)];
        let n = c.compress(&mut packed, &vec![b'z'; 1024]).unwrap();
        assert!(n >= 2);
        let (cmf, flg) = (packed[0], packed[1]);
        assert_eq!(cmf & 0x0f, 8, "CM must be DEFLATE (zlib wrapper, not raw)");
        assert_eq!(cmf >> 4, 7, "CINFO must be 7 (32 KiB window, windowBits=15)");
        assert_eq!(flg & 0x20, 0, "no preset dictionary");
        assert_eq!(
            u16::from_be_bytes([cmf, flg]) % 31,
            0,
            "FCHECK must make the header divisible by 31"
        );
    }

    /// Level is a compress-side knob only: any preset decodes any stream.
    #[test]
    fn streams_interoperate_across_presets() {
        let input = realistic_blob();
        let best = ZlibCompressor::new(Preset::Best);
        let fast = ZlibCompressor::new(Preset::Fast);

        let mut packed = vec![0u8; best.max_compressed_size(input.len())];
        let n = best.compress(&mut packed, &input).unwrap();
        let mut back = vec![0u8; input.len()];
        assert_eq!(fast.decompress(&mut back, &packed[..n]).unwrap(), input.len());
        assert_eq!(back, input);
    }

    /// Presets reach zlib (unlike C++ `ctp::Zlib`, which drops them).
    #[test]
    fn presets_map_to_cpp_levels_and_change_the_ratio() {
        assert_eq!(ZlibCompressor::new(Preset::Fast).level(), 1);
        assert_eq!(ZlibCompressor::new(Preset::Balanced).level(), 6);
        assert_eq!(ZlibCompressor::new(Preset::Best).level(), 9);

        let input = realistic_blob();
        let size_at = |p| {
            let c = ZlibCompressor::new(p);
            let mut packed = vec![0u8; c.max_compressed_size(input.len())];
            c.compress(&mut packed, &input).unwrap()
        };
        assert!(
            size_at(Preset::Best) <= size_at(Preset::Fast),
            "level 9 should not compress worse than level 1"
        );
    }

    /// Corrupt input must return Err, never panic or over-read.
    #[test]
    fn corrupt_input_errs() {
        let c = ZlibCompressor::new(Preset::Balanced);
        let mut back = vec![0u8; 4096];

        assert!(c.decompress(&mut back, b"not a zlib stream").is_err());
        assert!(c.decompress(&mut back, &[]).is_err());

        // Valid header, truncated body.
        let input = realistic_blob();
        let mut packed = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut packed, &input).unwrap();
        assert!(c.decompress(&mut back, &packed[..n / 2]).is_err());
    }

    /// The factory arm in lib.rs must build this type and round-trip.
    #[test]
    fn factory_builds_a_working_zlib_codec() {
        let c = crate::compressor_for(LibraryId::Zlib, Preset::Fast)
            .expect("zlib feature is enabled for this test run");
        assert_eq!(c.library(), LibraryId::Zlib);

        let input = b"iowarp ctp zlib factory roundtrip".repeat(64);
        let mut packed = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut packed, &input).unwrap();
        let mut back = vec![0u8; input.len()];
        assert_eq!(c.decompress(&mut back, &packed[..n]).unwrap(), input.len());
        assert_eq!(back, input);

        assert!(crate::compressor_by_name("zlib", Preset::Best).is_some());
    }
}
