// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! FFI wrapper for **c-blosc2** (`libblosc2`), the blocking/shuffling
//! meta-compressor. Mirrors C++ `ctp::Blosc`
//! (`clio_ctp/compress/blosc.h`).
//!
//! # Version / ABI assumptions
//!
//! Written against the **c-blosc2 2.x** C ABI (verified against 2.13.1, the
//! `libblosc2-dev` in the CTP devcontainer). Only the stable, flat C entry
//! points are bound — `blosc2_init`, `blosc2_compress`, `blosc2_decompress`,
//! `blosc1_cbuffer_validate` — all of which take scalars only. **No blosc2
//! struct layout is reproduced in Rust** (notably not `blosc2_cparams`),
//! so this binding cannot be broken by field additions to blosc2's structs
//! across 2.x releases; that is the main reason it uses the global-context
//! API rather than the `*_ctx` API the C++ wrapper uses.
//!
//! The constants below are compile-time-frozen copies of blosc2's; they are
//! format constants (`BLOSC2_MAX_OVERHEAD` = extended header length = 32,
//! `BLOSC_MIN_HEADER_LENGTH` = 16, `BLOSC_SHUFFLE` = 1) and are stable
//! across 2.x. `max_compressed_size` includes the mandated
//! `BLOSC2_MAX_OVERHEAD` headroom, which is what makes compression of
//! incompressible data succeed rather than return 0.
//!
//! # Codec parameters — matched to the C++ wrapper
//!
//! C++ compresses with `BLOSC2_CPARAMS_DEFAULTS`, i.e. codec BLOSCLZ,
//! `clevel` 5, `typesize` **8**, filter `BLOSC_SHUFFLE`, 1 thread. This
//! wrapper passes exactly those to `blosc2_compress` (typesize 8 = the
//! `double`/`int64` element width scientific payloads use; the byte shuffle
//! is what makes blosc pay off on such arrays). Output is therefore the
//! same on-disk chunk format the C++ stack reads and writes.
//!
//! # Divergences from the C++ wrapper (all deliberate)
//!
//! * **Errors are reported.** C++ `Blosc::Compress` returns `true`
//!   unconditionally and assigns blosc's `int` return straight into
//!   `size_t &output_size` — a negative error code silently becomes a huge
//!   size, and a 0 ("does not fit") becomes an empty-but-successful result.
//!   Here every negative/zero return becomes [`CompressError`].
//! * **`preset` is honored.** C++ ignores the preset and always compresses
//!   at the default `clevel` 5. This maps Fast/Balanced/Best → clevel
//!   1/5/9; `Balanced` reproduces the C++ bytes exactly.
//! * **Global API, not `*_ctx`.** See ABI note above. Consequence: blosc2's
//!   `BLOSC_CLEVEL` / `BLOSC_SHUFFLE` / `BLOSC_TYPESIZE` / `BLOSC_NTHREADS`
//!   environment variables are honored by `blosc2_compress` and are *not*
//!   honored by the C++ `*_ctx` path. Both remain round-trip-correct (the
//!   parameters are recorded in the chunk header); only the compressed
//!   bytes can differ when those vars are set.
//! * **`blosc2_destroy` is never called.** C++ ties it to a `Singleton`
//!   destructor. A Rust `static` has no destructor, and tearing down
//!   blosc2's global thread pool while another thread is inside
//!   `blosc2_compress` would be a use-after-free; leaking it until process
//!   exit is both correct and cheaper.
//! * **Empty input is framed as empty output.** blosc2 has no defined
//!   encoding for a zero-length buffer (`blosc2_compress` rejects
//!   `srcsize == 0`), so 0 bytes compress to 0 bytes and 0 bytes
//!   decompress back to 0 bytes rather than erroring.
//! * **Decompression pre-validates.** `blosc1_cbuffer_validate` is called
//!   before `blosc2_decompress`, so a truncated/corrupt chunk and a
//!   too-small output buffer are rejected up front instead of relying on
//!   the codec's internal bounds checks alone.

use std::ffi::{c_int, c_void};
use std::sync::Once;

use crate::{CompressError, Compressor, LibraryId, Preset};

// ---------------------------------------------------------------------------
// Raw C ABI (hand-rolled; see blosc2.h of c-blosc2 2.x)
// ---------------------------------------------------------------------------

extern "C" {
    /// `void blosc2_init(void)` — initialize the global context/thread pool.
    fn blosc2_init();

    /// `int blosc2_compress(int clevel, int doshuffle, int32_t typesize,
    ///                      const void* src, int32_t srcsize,
    ///                      void* dest, int32_t destsize)`
    ///
    /// Returns bytes written (> 0), 0 if the data does not fit in `destsize`,
    /// or a negative blosc2 error code.
    fn blosc2_compress(
        clevel: c_int,
        doshuffle: c_int,
        typesize: i32,
        src: *const c_void,
        srcsize: i32,
        dest: *mut c_void,
        destsize: i32,
    ) -> c_int;

    /// `int blosc2_decompress(const void* src, int32_t srcsize,
    ///                        void* dest, int32_t destsize)`
    ///
    /// Returns bytes written, or negative on corrupt input / short output.
    /// Guaranteed by blosc2 never to write more than `destsize` bytes.
    fn blosc2_decompress(
        src: *const c_void,
        srcsize: i32,
        dest: *mut c_void,
        destsize: i32,
    ) -> c_int;

    /// `int blosc1_cbuffer_validate(const void* cbuffer, size_t cbytes,
    ///                             size_t* nbytes)`
    ///
    /// Returns 0 and sets `nbytes` (the uncompressed size) when `cbuffer` is
    /// safe to hand to the decompressor; negative otherwise.
    fn blosc1_cbuffer_validate(cbuffer: *const c_void, cbytes: usize, nbytes: *mut usize) -> c_int;
}

// --- Frozen blosc2 format constants (see module docs) ----------------------

/// `BLOSC2_MAX_OVERHEAD` — max header/trailer bytes a chunk can add.
const BLOSC2_MAX_OVERHEAD: usize = 32;
/// `BLOSC_MIN_HEADER_LENGTH` — smallest legal chunk.
const BLOSC_MIN_HEADER_LENGTH: usize = 16;
/// `BLOSC2_MAX_BUFFERSIZE` — largest buffer blosc2 will compress.
const BLOSC2_MAX_BUFFERSIZE: usize = (i32::MAX as usize) - BLOSC2_MAX_OVERHEAD;
/// `BLOSC_SHUFFLE` — byte-wise shuffle filter (the C++ wrapper's default).
const BLOSC_SHUFFLE: c_int = 1;
/// `BLOSC2_CPARAMS_DEFAULTS.typesize` — 8 bytes (double / int64 elements).
const TYPESIZE: i32 = 8;

/// `blosc2_init` exactly once per process (C++ `Singleton<BloscInit>`).
/// Never torn down — see the module docs.
fn ensure_init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: blosc2_init takes no arguments and has no preconditions;
        // `Once` guarantees it runs exactly once, before any compress or
        // decompress call in this module.
        unsafe { blosc2_init() };
    });
}

/// Clamp a Rust length to what blosc2's `int32_t` size parameters accept.
/// Under-reporting capacity is always sound: blosc2 simply uses less of the
/// buffer than we own.
fn clamp_i32(len: usize) -> i32 {
    len.min(i32::MAX as usize) as i32
}

// ---------------------------------------------------------------------------
// Blosc2Compressor
// ---------------------------------------------------------------------------

/// Blosc2 codec — the Rust face of C++ `ctp::Blosc`.
#[derive(Debug, Clone, Copy)]
pub struct Blosc2Compressor {
    clevel: c_int,
}

impl Blosc2Compressor {
    /// Build a compressor at `preset`.
    ///
    /// `Preset::Balanced` is blosc2's own default `clevel` (5), i.e. the
    /// level the C++ wrapper hardcodes via `BLOSC2_CPARAMS_DEFAULTS`.
    pub fn new(preset: Preset) -> Self {
        let clevel = match preset {
            Preset::Fast => 1,
            Preset::Balanced => 5,
            Preset::Best => 9,
        };
        Self { clevel }
    }
}

impl Default for Blosc2Compressor {
    fn default() -> Self {
        Self::new(Preset::default())
    }
}

impl Compressor for Blosc2Compressor {
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        // blosc2 has no encoding for a zero-length buffer; frame it as empty.
        if input.is_empty() {
            return Ok(0);
        }
        if input.len() > BLOSC2_MAX_BUFFERSIZE {
            return Err(CompressError(format!(
                "blosc2: input of {} bytes exceeds BLOSC2_MAX_BUFFERSIZE ({BLOSC2_MAX_BUFFERSIZE})",
                input.len()
            )));
        }
        if output.is_empty() {
            return Err(CompressError(
                "blosc2: output buffer is empty; need at least max_compressed_size bytes".into(),
            ));
        }

        // SAFETY: `src`/`dest` are non-overlapping (distinct `&[u8]` and
        // `&mut [u8]`), each valid for the length passed alongside it —
        // lengths are clamped to i32 so the counts blosc2 sees never exceed
        // the real allocations. blosc2 writes at most `destsize` bytes into
        // `dest`. The library is initialized above.
        let rc = unsafe {
            ensure_init();
            blosc2_compress(
                self.clevel,
                BLOSC_SHUFFLE,
                TYPESIZE,
                input.as_ptr() as *const c_void,
                clamp_i32(input.len()),
                output.as_mut_ptr() as *mut c_void,
                clamp_i32(output.len()),
            )
        };

        match rc {
            // 0 = "compressed data does not fit in destsize". Unlike C++,
            // which would report success with output_size == 0.
            0 => Err(CompressError(format!(
                "blosc2: output buffer too small ({} bytes) for {} bytes of input; \
                 max_compressed_size() reports {}",
                output.len(),
                input.len(),
                self.max_compressed_size(input.len())
            ))),
            rc if rc < 0 => Err(CompressError(format!("blosc2_compress failed: error {rc}"))),
            rc => Ok(rc as usize),
        }
    }

    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError> {
        // Mirror compress()'s empty framing.
        if input.is_empty() {
            return Ok(0);
        }
        if input.len() < BLOSC_MIN_HEADER_LENGTH {
            return Err(CompressError(format!(
                "blosc2: input of {} bytes is shorter than a blosc header ({BLOSC_MIN_HEADER_LENGTH} bytes)",
                input.len()
            )));
        }

        // Validate the chunk before decoding it: this is what makes a
        // truncated/corrupt buffer a clean Err, and it yields the
        // uncompressed size so a short output buffer is caught up front.
        let mut nbytes: usize = 0;
        // SAFETY: `input` is valid for `input.len()` bytes (>= the 16-byte
        // header this reads), and `nbytes` is a live local out-pointer.
        // blosc1_cbuffer_validate only reads, and bounds its reads by
        // `cbytes`.
        let rc = unsafe {
            ensure_init();
            blosc1_cbuffer_validate(input.as_ptr() as *const c_void, input.len(), &mut nbytes)
        };
        if rc < 0 {
            return Err(CompressError(format!(
                "blosc2: input is not a valid blosc chunk (validate error {rc})"
            )));
        }
        if nbytes > output.len() {
            return Err(CompressError(format!(
                "blosc2: output buffer too small: {} bytes needed, {} available",
                nbytes,
                output.len()
            )));
        }
        if nbytes == 0 {
            return Ok(0);
        }

        // SAFETY: `src`/`dest` are non-overlapping and each valid for the
        // clamped length passed with it. The chunk was validated above, and
        // blosc2_decompress additionally guarantees it writes no more than
        // `destsize` bytes.
        let rc = unsafe {
            blosc2_decompress(
                input.as_ptr() as *const c_void,
                clamp_i32(input.len()),
                output.as_mut_ptr() as *mut c_void,
                clamp_i32(output.len()),
            )
        };
        if rc < 0 {
            return Err(CompressError(format!("blosc2_decompress failed: error {rc}")));
        }
        Ok(rc as usize)
    }

    fn library(&self) -> LibraryId {
        LibraryId::Blosc2
    }

    /// blosc2 guarantees compression succeeds when the destination holds
    /// `nbytes + BLOSC2_MAX_OVERHEAD` — that headroom is the whole reason
    /// incompressible input can still be stored (blosc falls back to a raw
    /// copy plus header).
    fn max_compressed_size(&self, input_size: usize) -> usize {
        input_size.saturating_add(BLOSC2_MAX_OVERHEAD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Realistic scientific payload: f64 samples, which is exactly what
    /// typesize=8 + byte shuffle is tuned for.
    fn sensor_doubles(n: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(n * 8);
        for i in 0..n {
            let t = i as f64 * 0.001;
            let v = 20.0 + 5.0 * (t * 0.5).sin() + 0.25 * (t * 7.0).cos();
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn round_trip_is_byte_exact() {
        // Lossless codec: every preset must reproduce the input exactly.
        for preset in [Preset::Fast, Preset::Balanced, Preset::Best] {
            let c = Blosc2Compressor::new(preset);
            let input = sensor_doubles(16 * 1024);

            let mut packed = vec![0u8; c.max_compressed_size(input.len())];
            let n = c.compress(&mut packed, &input).unwrap();
            assert!(n > 0 && n <= packed.len());

            let mut back = vec![0u8; input.len()];
            let m = c.decompress(&mut back, &packed[..n]).unwrap();
            assert_eq!(m, input.len(), "preset {:?}", preset);
            assert_eq!(back, input, "preset {:?} was not byte-exact", preset);
        }
    }

    #[test]
    fn round_trip_of_incompressible_data_fits_in_max_compressed_size() {
        // The BLOSC2_MAX_OVERHEAD headroom must make even pseudo-random
        // (uncompressible) input succeed rather than return 0.
        let c = Blosc2Compressor::default();
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let input: Vec<u8> = (0..8192)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();

        let mut packed = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut packed, &input).unwrap();
        let mut back = vec![0u8; input.len()];
        assert_eq!(c.decompress(&mut back, &packed[..n]).unwrap(), input.len());
        assert_eq!(back, input);
    }

    #[test]
    fn compressible_data_actually_shrinks() {
        let c = Blosc2Compressor::default();
        // Slowly-varying f64 ramp: highly compressible after the shuffle.
        let mut input = Vec::with_capacity(64 * 1024);
        for i in 0..8192u64 {
            input.extend_from_slice(&(i as f64).to_le_bytes());
        }

        let mut packed = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut packed, &input).unwrap();
        assert!(
            n < input.len() / 2,
            "expected >2x compression, got {n} from {} bytes",
            input.len()
        );

        let mut back = vec![0u8; input.len()];
        assert_eq!(c.decompress(&mut back, &packed[..n]).unwrap(), input.len());
        assert_eq!(back, input);
    }

    #[test]
    fn compress_into_too_small_buffer_errors() {
        let c = Blosc2Compressor::default();
        let input = sensor_doubles(4096);

        // Smaller than a blosc header: cannot possibly hold a chunk.
        let mut tiny = [0u8; 8];
        assert!(c.compress(&mut tiny, &input).is_err());
        // Untouched — blosc2 respects destsize rather than corrupting memory.
        assert_eq!(tiny, [0u8; 8]);

        // Zero-capacity output is rejected too.
        assert!(c.compress(&mut [], &input).is_err());
    }

    #[test]
    fn decompress_into_too_small_buffer_errors() {
        let c = Blosc2Compressor::default();
        let input = sensor_doubles(4096);
        let mut packed = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut packed, &input).unwrap();

        let mut short = vec![0xAAu8; input.len() - 1];
        assert!(c.decompress(&mut short, &packed[..n]).is_err());
        // Nothing was written into the undersized buffer.
        assert!(short.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn corrupt_and_truncated_input_error_rather_than_panic() {
        let c = Blosc2Compressor::default();
        let input = sensor_doubles(4096);
        let mut packed = vec![0u8; c.max_compressed_size(input.len())];
        let n = c.compress(&mut packed, &input).unwrap();
        let mut out = vec![0u8; input.len()];

        // Truncated below the header length.
        assert!(c.decompress(&mut out, &packed[..4]).is_err());
        // Header-length garbage is not a blosc chunk.
        assert!(c.decompress(&mut out, &[0xFFu8; 24]).is_err());
        // Truncated mid-chunk.
        assert!(c.decompress(&mut out, &packed[..n / 2]).is_err());
    }

    #[test]
    fn empty_input_round_trips() {
        let c = Blosc2Compressor::default();
        let mut packed = vec![0u8; c.max_compressed_size(0)];
        let n = c.compress(&mut packed, &[]).unwrap();
        assert_eq!(n, 0);

        let mut out = vec![0u8; 16];
        assert_eq!(c.decompress(&mut out, &packed[..n]).unwrap(), 0);
    }

    #[test]
    fn max_compressed_size_includes_blosc_overhead() {
        let c = Blosc2Compressor::default();
        assert_eq!(c.max_compressed_size(0), BLOSC2_MAX_OVERHEAD);
        assert_eq!(c.max_compressed_size(1024), 1024 + BLOSC2_MAX_OVERHEAD);
        // Must not overflow on absurd sizes.
        assert_eq!(c.max_compressed_size(usize::MAX), usize::MAX);
    }

    #[test]
    fn reports_its_library_and_is_reachable_through_the_factory() {
        assert_eq!(Blosc2Compressor::default().library(), LibraryId::Blosc2);
        assert!(!LibraryId::Blosc2.is_lossy());

        let boxed = crate::compressor_for(LibraryId::Blosc2, Preset::Balanced)
            .expect("blosc2 feature is on, factory must build it");
        assert_eq!(boxed.library(), LibraryId::Blosc2);

        let input = sensor_doubles(1024);
        let mut packed = vec![0u8; boxed.max_compressed_size(input.len())];
        let n = boxed.compress(&mut packed, &input).unwrap();
        let mut back = vec![0u8; input.len()];
        assert_eq!(boxed.decompress(&mut back, &packed[..n]).unwrap(), input.len());
        assert_eq!(back, input);
    }

    #[test]
    fn concurrent_use_is_sound() {
        // The trait requires Send + Sync; blosc2's global API takes its own
        // lock, so hammer it from several threads.
        let c = Blosc2Compressor::default();
        std::thread::scope(|s| {
            for t in 0..4 {
                let c = &c;
                s.spawn(move || {
                    let input = sensor_doubles(2048 + t * 32);
                    let mut packed = vec![0u8; c.max_compressed_size(input.len())];
                    let n = c.compress(&mut packed, &input).unwrap();
                    let mut back = vec![0u8; input.len()];
                    assert_eq!(c.decompress(&mut back, &packed[..n]).unwrap(), input.len());
                    assert_eq!(back, input);
                });
            }
        });
    }
}
