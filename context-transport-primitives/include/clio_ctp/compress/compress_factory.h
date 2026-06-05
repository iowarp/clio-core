/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_UTIL_COMPRESS_COMPRESS_FACTORY_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_UTIL_COMPRESS_COMPRESS_FACTORY_H_

#if CTP_ENABLE_COMPRESS

#include <array>
#include <cctype>
#include <memory>
#include <string>
#include "compress.h"
#include "lossless_modes.h"
#include "snappy.h"
#include "blosc.h"

#if CTP_ENABLE_LIBPRESSIO
#include "libpressio_modes.h"
#endif

#if CTP_ENABLE_NVCOMP
#include "nvcomp.h"
#endif

namespace ctp {

/**
 * Compression preset levels for configurable compressors.
 * Maps to compression levels (lossless) or error bounds (lossy).
 */
enum class CompressionPreset {
  FAST,      // Fast compression, lower ratio/quality
  BALANCED,  // Balanced speed and ratio/quality
  BEST,      // Best ratio/quality, slower
  DEFAULT    // Default configuration (for compressors without levels)
};

/**
 * Factory for creating configured compression instances.
 *
 * Provides a unified interface for creating compressors with different
 * configuration presets (FAST/BALANCED/BEST).
 */
class CompressionFactory {
 public:
  /**
   * Get a compressor instance with the specified preset.
   *
   * @param library_name Name of compression library (case-insensitive)
   *                     Supported: "bzip2", "zstd", "lz4", "zlib", "lzma",
   *                                "brotli", "snappy", "blosc2"
   *                                "zfp", "sz3", "fpzip" (if LibPressio enabled)
   *                                "nvcomp-lz4", "nvcomp-snappy", "nvcomp-zstd",
   *                                "nvcomp-gdeflate", "nvcomp-deflate",
   *                                "nvcomp-ans" (if nvcomp enabled)
   * @param preset Compression preset level (FAST/BALANCED/BEST/DEFAULT)
   * @return Unique pointer to configured compressor instance,
   *         or nullptr if library not found
   *
   * Example:
   *   auto compressor = CompressionFactory::GetPreset("bzip2", CompressionPreset::FAST);
   *   if (compressor) {
   *     // Use compressor for compression
   *   }
   */
  static std::unique_ptr<Compressor> GetPreset(
      const std::string& library_name,
      CompressionPreset preset = CompressionPreset::BALANCED) {
    const CompressorInfo* info = FindByName(library_name);
    // info->make is null when the backend is disabled at build time (LibPressio
    // for lossy, nvcomp for GPU); GetPreset then returns nullptr just as the
    // old per-name if-chain fell through to "return nullptr".
    if (info == nullptr || info->make == nullptr) {
      return nullptr;
    }
    return info->make(preset);
  }

  /**
   * Get library ID for a given library name and preset.
   * This ID can be used for model training and runtime compression selection.
   *
   * @param library_name Name of compression library
   * @param preset Compression preset level
   * @return Integer ID encoding both library and preset, or 0 if unknown
   *
   * ID encoding scheme:
   * - Lossless: base_id * 10 + preset (e.g., BZIP2_FAST=11, BZIP2_BALANCED=12, BZIP2_BEST=13)
   * - Lossy: base_id * 10 + preset (e.g., ZFP_FAST=101, ZFP_BALANCED=102, ZFP_BEST=103)
   */
  static int GetLibraryId(const std::string& library_name, CompressionPreset preset) {
    const CompressorInfo* info = FindByName(library_name);
    if (info == nullptr) return 0;  // Unknown library

    // Single-mode libraries (snappy, blosc2, nvcomp) always encode preset 2.
    if (info->single_mode) {
      return info->base_id * 10 + 2;
    }

    // Encode preset: FAST=1, BALANCED=2, BEST=3, DEFAULT=2
    int preset_id = 2;  // Default to BALANCED
    switch (preset) {
      case CompressionPreset::FAST: preset_id = 1; break;
      case CompressionPreset::BALANCED: preset_id = 2; break;
      case CompressionPreset::BEST: preset_id = 3; break;
      case CompressionPreset::DEFAULT: preset_id = 2; break;
    }

    return info->base_id * 10 + preset_id;
  }

  /**
   * Get library name and preset from library ID.
   * Reverse of GetLibraryId().
   *
   * @param library_id Integer ID encoding library and preset
   * @return Pair of (library_name, preset), or ("unknown", DEFAULT) if invalid
   */
  static std::pair<std::string, CompressionPreset> GetLibraryInfo(int library_id) {
    int base_id = library_id / 10;
    int preset_id = library_id % 10;

    const CompressorInfo* info = FindByBaseId(base_id);
    std::string library_name = info ? info->name : "unknown";

    CompressionPreset preset = CompressionPreset::BALANCED;
    if (preset_id == 1) preset = CompressionPreset::FAST;
    else if (preset_id == 2) preset = CompressionPreset::BALANCED;
    else if (preset_id == 3) preset = CompressionPreset::BEST;

    return {library_name, preset};
  }

  /**
   * Map a CTE wire ID to its canonical library name.
   *
   * The wire ID is the raw integer stored in CompressionHeader.compress_lib_
   * (an array index, the on-disk/runtime protocol). It is a SEPARATE, frozen
   * namespace from GetLibraryId's ML scheme (base_id*10 + preset). Out-of-range
   * IDs fall back to "zstd" (the historical CTE default). Append-only: never
   * renumber existing IDs or old compressed blobs become unreadable.
   *
   * @param wire_id Integer wire ID (compress_lib_)
   * @return Canonical library name, or "zstd" if out of range
   */
  static std::string NameForWireId(int wire_id) {
    const CompressorInfo* info = FindByWireId(wire_id);
    return info ? info->name : "zstd";
  }

  /**
   * Get string name for preset.
   *
   * @param preset Compression preset
   * @return String representation ("fast", "balanced", "best", "default")
   */
  static std::string GetPresetName(CompressionPreset preset) {
    switch (preset) {
      case CompressionPreset::FAST: return "fast";
      case CompressionPreset::BALANCED: return "balanced";
      case CompressionPreset::BEST: return "best";
      case CompressionPreset::DEFAULT: return "default";
      default: return "unknown";
    }
  }

 private:
  /** Signature of a function that constructs a compressor for a given preset. */
  using MakeFn = std::unique_ptr<Compressor> (*)(CompressionPreset);

  /**
   * One row of the compressor registry -- the single source of truth tying a
   * compressor's canonical name to its two frozen, INDEPENDENT integer
   * namespaces:
   *   - wire_id: raw index stored in CompressionHeader.compress_lib_ (the CTE
   *              on-disk/runtime protocol; resolved by NameForWireId).
   *   - base_id: ML scheme, encoded as base_id*10 + preset (see GetLibraryId).
   * Both are append-only: NEVER renumber an existing entry or stored blobs /
   * trained models break. `make` is nullptr when the backend is unavailable in
   * this build (LibPressio off for lossy, nvcomp off for GPU); the name and ids
   * still resolve, but GetPreset then returns nullptr.
   *
   * To add a compressor, add ONE row here (plus, for a GPU algorithm, one case
   * in NvComp::MakeManager). No other edits in this file are needed.
   */
  struct CompressorInfo {
    const char* name;  // canonical lowercase name
    int wire_id;       // CTE wire/on-disk id (frozen, append-only)
    int base_id;       // ML scheme base id (frozen, append-only)
    bool single_mode;  // preset ignored; id always uses preset slot 2
    MakeFn make;       // constructor, or nullptr if backend disabled
  };

  // Construction helpers for single-mode and backend-guarded compressors.
  // (Multi-mode lossless compressors use CreateLossless<T> directly.)
  static std::unique_ptr<Compressor> MakeSnappy(CompressionPreset) {
    return std::make_unique<Snappy>();
  }
  static std::unique_ptr<Compressor> MakeBlosc(CompressionPreset) {
    return std::make_unique<Blosc>();
  }
  static std::unique_ptr<Compressor> MakeZfp(CompressionPreset preset) {
#if CTP_ENABLE_LIBPRESSIO
    return CreateLossy("zfp", preset);
#else
    (void)preset;
    return nullptr;
#endif
  }
  static std::unique_ptr<Compressor> MakeSz3(CompressionPreset preset) {
#if CTP_ENABLE_LIBPRESSIO
    return CreateLossy("sz3", preset);
#else
    (void)preset;
    return nullptr;
#endif
  }
  static std::unique_ptr<Compressor> MakeFpzip(CompressionPreset preset) {
#if CTP_ENABLE_LIBPRESSIO
    return CreateLossy("fpzip", preset);
#else
    (void)preset;
    return nullptr;
#endif
  }
  // GPU compressors (nvcomp). Each helper is always defined so its registry row
  // is valid in every build; it returns nullptr when nvcomp is disabled (the
  // name/ids still resolve, but GetPreset yields nullptr). All are single-mode.
  static std::unique_ptr<Compressor> MakeNvCompLz4(CompressionPreset) {
#if CTP_ENABLE_NVCOMP
    return std::make_unique<NvComp>(NvCompAlgo::LZ4);
#else
    return nullptr;
#endif
  }
  static std::unique_ptr<Compressor> MakeNvCompSnappy(CompressionPreset) {
#if CTP_ENABLE_NVCOMP
    return std::make_unique<NvComp>(NvCompAlgo::SNAPPY);
#else
    return nullptr;
#endif
  }
  static std::unique_ptr<Compressor> MakeNvCompZstd(CompressionPreset) {
#if CTP_ENABLE_NVCOMP
    return std::make_unique<NvComp>(NvCompAlgo::ZSTD);
#else
    return nullptr;
#endif
  }
  static std::unique_ptr<Compressor> MakeNvCompGdeflate(CompressionPreset) {
#if CTP_ENABLE_NVCOMP
    return std::make_unique<NvComp>(NvCompAlgo::GDEFLATE);
#else
    return nullptr;
#endif
  }
  static std::unique_ptr<Compressor> MakeNvCompDeflate(CompressionPreset) {
#if CTP_ENABLE_NVCOMP
    return std::make_unique<NvComp>(NvCompAlgo::DEFLATE);
#else
    return nullptr;
#endif
  }
  static std::unique_ptr<Compressor> MakeNvCompAns(CompressionPreset) {
#if CTP_ENABLE_NVCOMP
    return std::make_unique<NvComp>(NvCompAlgo::ANS);
#else
    return nullptr;
#endif
  }

  /**
   * The compressor registry: the single source of truth (see CompressorInfo).
   * constexpr std::array -> constant-initialized static data: no heap allocation
   * and no thread-safe-init guard. The size is deduced (CTAD), so adding a row
   * stays a one-line change. Iteration over 12 entries is trivial and happens
   * only once per blob (in the factory setup path), never in the compress loop.
   */
  static const auto& Registry() {
    //                                          name  wire base single  make
    static constexpr std::array kRegistry = {
        CompressorInfo{"brotli",      0,  6, false, &CreateLossless<BrotliWithModes>},
        CompressorInfo{"bzip2",       1,  1, false, &CreateLossless<Bzip2WithModes>},
        CompressorInfo{"blosc2",      2,  8, true,  &MakeBlosc},
        CompressorInfo{"fpzip",       3, 12, false, &MakeFpzip},
        CompressorInfo{"lz4",         4,  3, false, &CreateLossless<Lz4WithModes>},
        CompressorInfo{"lzma",        5,  5, false, &CreateLossless<LzmaWithModes>},
        CompressorInfo{"snappy",      6,  7, true,  &MakeSnappy},
        CompressorInfo{"sz3",         7, 11, false, &MakeSz3},
        CompressorInfo{"zfp",         8, 10, false, &MakeZfp},
        CompressorInfo{"zlib",        9,  4, false, &CreateLossless<ZlibWithModes>},
        CompressorInfo{"zstd",       10,  2, false, &CreateLossless<ZstdWithModes>},
        CompressorInfo{"nvcomp-lz4",      11, 13, true, &MakeNvCompLz4},
        CompressorInfo{"nvcomp-snappy",   12, 14, true, &MakeNvCompSnappy},
        CompressorInfo{"nvcomp-zstd",     13, 15, true, &MakeNvCompZstd},
        CompressorInfo{"nvcomp-gdeflate", 14, 16, true, &MakeNvCompGdeflate},
        CompressorInfo{"nvcomp-deflate",  15, 17, true, &MakeNvCompDeflate},
        CompressorInfo{"nvcomp-ans",      16, 18, true, &MakeNvCompAns},
    };
    return kRegistry;
  }

  /**
   * Look up a registry entry by (case-insensitive) name; nullptr if unknown.
   * Accepts the historical "blosc" alias for "blosc2".
   */
  static const CompressorInfo* FindByName(const std::string& name) {
    std::string n = name;
    for (auto& c : n) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (n == "blosc") n = "blosc2";  // historical alias
    for (const auto& e : Registry()) {
      if (n == e.name) return &e;
    }
    return nullptr;
  }

  /** Look up a registry entry by CTE wire id; nullptr if none. */
  static const CompressorInfo* FindByWireId(int wire_id) {
    for (const auto& e : Registry()) {
      if (e.wire_id == wire_id) return &e;
    }
    return nullptr;
  }

  /** Look up a registry entry by ML base id; nullptr if none. */
  static const CompressorInfo* FindByBaseId(int base_id) {
    for (const auto& e : Registry()) {
      if (e.base_id == base_id) return &e;
    }
    return nullptr;
  }

  template<typename CompressorClass>
  static std::unique_ptr<Compressor> CreateLossless(CompressionPreset preset) {
    LosslessMode mode;
    switch (preset) {
      case CompressionPreset::FAST:
        mode = LosslessMode::FAST;
        break;
      case CompressionPreset::BALANCED:
      case CompressionPreset::DEFAULT:
        mode = LosslessMode::BALANCED;
        break;
      case CompressionPreset::BEST:
        mode = LosslessMode::BEST;
        break;
    }
    return std::make_unique<CompressorClass>(mode);
  }

#if CTP_ENABLE_LIBPRESSIO
  static std::unique_ptr<Compressor> CreateLossy(
      const std::string& compressor_id, CompressionPreset preset) {
    CompressionMode mode;
    switch (preset) {
      case CompressionPreset::FAST:
        mode = CompressionMode::FAST;
        break;
      case CompressionPreset::BALANCED:
      case CompressionPreset::DEFAULT:
        mode = CompressionMode::BALANCED;
        break;
      case CompressionPreset::BEST:
        mode = CompressionMode::BEST;
        break;
    }
    return std::make_unique<LibPressioWithModes>(compressor_id, mode);
  }
#endif
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_UTIL_COMPRESS_COMPRESS_FACTORY_H_
