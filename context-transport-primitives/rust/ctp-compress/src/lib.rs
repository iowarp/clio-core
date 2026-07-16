// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! FFI wrappers for the C/C++ compression libraries CTP uses.
//!
//! **These libraries are wrapped, not ported** (rust/MIGRATION.md policy):
//! the codecs are mature C/C++ implementations (zstd, SZ3, ZFP, …); a Rust
//! rewrite would be a different project and would break format
//! compatibility with data already written by the C++ stack. Each
//! `sys_<lib>` module is a thin, feature-gated FFI binding plus a safe
//! [`Compressor`] implementation.
//!
//! # C++ → Rust mapping
//!
//! | C++ (`clio_ctp/compress/`) | Rust |
//! |---|---|
//! | `ctp::Compressor` (virtual Compress/Decompress) | [`Compressor`] trait |
//! | `CompressFactory::Get(library_name, preset)` | [`compressor_for`] |
//! | library id ints (BROTLI=0 … ZSTD=10) | [`LibraryId`] (same wire values) |
//! | `CompressionPreset` (FAST/BALANCED/BEST/DEFAULT) | [`Preset`] |
//!
//! # Wire-value stability
//!
//! [`LibraryId`] values are **append-only**: they are stored in blob
//! metadata (`BlobInfo::compress_lib_`), so renumbering an existing entry
//! makes previously-compressed blobs unreadable. Never reorder.

#![deny(unsafe_op_in_unsafe_fn)]

use std::fmt;

#[cfg(feature = "brotli")]
pub mod sys_brotli;
#[cfg(feature = "blosc2")]
pub mod sys_blosc2;
#[cfg(feature = "bzip2")]
pub mod sys_bzip2;
#[cfg(feature = "fpzip")]
pub mod sys_fpzip;
#[cfg(feature = "lz4")]
pub mod sys_lz4;
#[cfg(feature = "lzma")]
pub mod sys_lzma;
#[cfg(feature = "snappy")]
pub mod sys_snappy;
#[cfg(feature = "sz3")]
pub mod sys_sz3;
#[cfg(feature = "zfp")]
pub mod sys_zfp;
#[cfg(feature = "zlib")]
pub mod sys_zlib;
#[cfg(feature = "zstd")]
pub mod sys_zstd;

/// Failure of a compress/decompress call.
///
/// The C++ surface returns a bare `bool`; this carries the reason, which is
/// a deliberate (documented) improvement — callers that only need the bool
/// use `.is_ok()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressError(pub String);

impl fmt::Display for CompressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompressError: {}", self.0)
    }
}
impl std::error::Error for CompressError {}

/// Compression library ids. **Append-only** — these values are persisted in
/// blob metadata (see the module docs).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LibraryId {
    Brotli = 0,
    Bzip2 = 1,
    Blosc2 = 2,
    Fpzip = 3,
    Lz4 = 4,
    Lzma = 5,
    Snappy = 6,
    Sz3 = 7,
    Zfp = 8,
    Zlib = 9,
    Zstd = 10,
}

impl LibraryId {
    /// Canonical lowercase name (matches the C++ factory's name table).
    pub const fn name(self) -> &'static str {
        match self {
            LibraryId::Brotli => "brotli",
            LibraryId::Bzip2 => "bzip2",
            LibraryId::Blosc2 => "blosc2",
            LibraryId::Fpzip => "fpzip",
            LibraryId::Lz4 => "lz4",
            LibraryId::Lzma => "lzma",
            LibraryId::Snappy => "snappy",
            LibraryId::Sz3 => "sz3",
            LibraryId::Zfp => "zfp",
            LibraryId::Zlib => "zlib",
            LibraryId::Zstd => "zstd",
        }
    }

    /// Parse a library id from its canonical name (case-insensitive).
    pub fn from_name(s: &str) -> Option<LibraryId> {
        const ALL: [LibraryId; 11] = [
            LibraryId::Brotli,
            LibraryId::Bzip2,
            LibraryId::Blosc2,
            LibraryId::Fpzip,
            LibraryId::Lz4,
            LibraryId::Lzma,
            LibraryId::Snappy,
            LibraryId::Sz3,
            LibraryId::Zfp,
            LibraryId::Zlib,
            LibraryId::Zstd,
        ];
        ALL.into_iter().find(|l| l.name().eq_ignore_ascii_case(s))
    }

    /// True when the codec is lossy (error-bounded scientific compressors).
    /// Lossy codecs must not be used for metadata or non-float payloads.
    pub const fn is_lossy(self) -> bool {
        matches!(self, LibraryId::Fpzip | LibraryId::Sz3 | LibraryId::Zfp)
    }
}

/// Compression presets (C++ `CompressionPreset`; wire values 1/2/3).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preset {
    Fast = 1,
    #[default]
    Balanced = 2,
    Best = 3,
}

impl Preset {
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// C++ `GetPresetString`.
    pub const fn name(self) -> &'static str {
        match self {
            Preset::Fast => "fast",
            Preset::Balanced => "balanced",
            Preset::Best => "best",
        }
    }
}

/// A compression codec — the Rust face of C++ `ctp::Compressor`.
///
/// Both methods mirror the C++ INOUT `output_size` convention: the output
/// slice's length is the available CAPACITY, and the returned `usize` is
/// the number of bytes actually written.
pub trait Compressor: Send + Sync {
    /// Compress `input` into `output`; returns bytes written.
    fn compress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError>;

    /// Decompress `input` into `output`; returns bytes written.
    fn decompress(&self, output: &mut [u8], input: &[u8]) -> Result<usize, CompressError>;

    /// Which library this is.
    fn library(&self) -> LibraryId;

    /// A safe upper bound on the compressed size of `input_size` bytes —
    /// the allocation size callers should use for `compress`'s output.
    fn max_compressed_size(&self, input_size: usize) -> usize {
        // Conservative default: incompressible data plus generous headroom.
        input_size + input_size / 8 + 1024
    }
}

/// Construct the codec for `id` at `preset` (C++ `CompressFactory::Get`).
///
/// Returns `None` when the codec's feature is not compiled in, so callers
/// can degrade gracefully rather than link every library unconditionally.
pub fn compressor_for(id: LibraryId, preset: Preset) -> Option<Box<dyn Compressor>> {
    match id {
        #[cfg(feature = "brotli")]
        LibraryId::Brotli => Some(Box::new(sys_brotli::BrotliCompressor::new(preset))),
        #[cfg(feature = "bzip2")]
        LibraryId::Bzip2 => Some(Box::new(sys_bzip2::Bzip2Compressor::new(preset))),
        #[cfg(feature = "blosc2")]
        LibraryId::Blosc2 => Some(Box::new(sys_blosc2::Blosc2Compressor::new(preset))),
        #[cfg(feature = "fpzip")]
        LibraryId::Fpzip => Some(Box::new(sys_fpzip::FpzipCompressor::new(preset))),
        #[cfg(feature = "lz4")]
        LibraryId::Lz4 => Some(Box::new(sys_lz4::Lz4Compressor::new(preset))),
        #[cfg(feature = "lzma")]
        LibraryId::Lzma => Some(Box::new(sys_lzma::LzmaCompressor::new(preset))),
        #[cfg(feature = "snappy")]
        LibraryId::Snappy => Some(Box::new(sys_snappy::SnappyCompressor::new(preset))),
        #[cfg(feature = "sz3")]
        LibraryId::Sz3 => Some(Box::new(sys_sz3::Sz3Compressor::new(preset))),
        #[cfg(feature = "zfp")]
        LibraryId::Zfp => Some(Box::new(sys_zfp::ZfpCompressor::new(preset))),
        #[cfg(feature = "zlib")]
        LibraryId::Zlib => Some(Box::new(sys_zlib::ZlibCompressor::new(preset))),
        #[cfg(feature = "zstd")]
        LibraryId::Zstd => Some(Box::new(sys_zstd::ZstdCompressor::new(preset))),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

/// Construct by name (C++ `CompressFactory::Get(library_name, preset)`).
pub fn compressor_by_name(name: &str, preset: Preset) -> Option<Box<dyn Compressor>> {
    compressor_for(LibraryId::from_name(name)?, preset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_ids_are_stable_wire_values() {
        // Renumbering these makes already-compressed blobs unreadable.
        assert_eq!(LibraryId::Brotli as u32, 0);
        assert_eq!(LibraryId::Bzip2 as u32, 1);
        assert_eq!(LibraryId::Blosc2 as u32, 2);
        assert_eq!(LibraryId::Fpzip as u32, 3);
        assert_eq!(LibraryId::Lz4 as u32, 4);
        assert_eq!(LibraryId::Lzma as u32, 5);
        assert_eq!(LibraryId::Snappy as u32, 6);
        assert_eq!(LibraryId::Sz3 as u32, 7);
        assert_eq!(LibraryId::Zfp as u32, 8);
        assert_eq!(LibraryId::Zlib as u32, 9);
        assert_eq!(LibraryId::Zstd as u32, 10);
        assert_eq!(Preset::Fast.as_u32(), 1);
        assert_eq!(Preset::Balanced.as_u32(), 2);
        assert_eq!(Preset::Best.as_u32(), 3);
    }

    #[test]
    fn name_roundtrip_and_lossy_classification() {
        for id in [
            LibraryId::Brotli,
            LibraryId::Bzip2,
            LibraryId::Blosc2,
            LibraryId::Fpzip,
            LibraryId::Lz4,
            LibraryId::Lzma,
            LibraryId::Snappy,
            LibraryId::Sz3,
            LibraryId::Zfp,
            LibraryId::Zlib,
            LibraryId::Zstd,
        ] {
            assert_eq!(LibraryId::from_name(id.name()), Some(id));
            assert_eq!(LibraryId::from_name(&id.name().to_uppercase()), Some(id));
        }
        assert_eq!(LibraryId::from_name("nope"), None);
        assert!(LibraryId::Sz3.is_lossy() && LibraryId::Zfp.is_lossy() && LibraryId::Fpzip.is_lossy());
        assert!(!LibraryId::Zstd.is_lossy() && !LibraryId::Lz4.is_lossy());
    }
}
