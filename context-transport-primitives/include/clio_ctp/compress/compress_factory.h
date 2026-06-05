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

    std::string lib_lower = library_name;
    for (auto& c : lib_lower) c = std::tolower(c);

    // Lossless compressors with compression level support
    if (lib_lower == "bzip2") {
      return CreateLossless<Bzip2WithModes>(preset);
    }
    if (lib_lower == "zstd") {
      return CreateLossless<ZstdWithModes>(preset);
    }
    if (lib_lower == "lz4") {
      return CreateLossless<Lz4WithModes>(preset);
    }
    if (lib_lower == "zlib") {
      return CreateLossless<ZlibWithModes>(preset);
    }
    if (lib_lower == "lzma") {
      return CreateLossless<LzmaWithModes>(preset);
    }
    if (lib_lower == "brotli") {
      return CreateLossless<BrotliWithModes>(preset);
    }

    // Lossless compressors without level support (default only)
    if (lib_lower == "snappy") {
      return std::make_unique<Snappy>();
    }
    if (lib_lower == "blosc" || lib_lower == "blosc2") {
      return std::make_unique<Blosc>();
    }

#if CTP_ENABLE_NVCOMP
    // GPU compressors (nvcomp). Single-mode: preset is ignored (LZ4 has no level).
    if (lib_lower == "nvcomp-lz4") {
      return std::make_unique<NvComp>(NvCompAlgo::LZ4);
    }
#endif

#if CTP_ENABLE_LIBPRESSIO
    // Lossy compressors with LibPressio
    if (lib_lower == "zfp") {
      return CreateLossy("zfp", preset);
    }
    if (lib_lower == "sz3") {
      return CreateLossy("sz3", preset);
    }
    if (lib_lower == "fpzip") {
      return CreateLossy("fpzip", preset);
    }
#endif

    // Unknown library
    return nullptr;
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
    std::string lib_lower = library_name;
    for (auto& c : lib_lower) c = std::tolower(c);

    int base_id = 0;

    // Lossless base IDs (1-8)
    if (lib_lower == "bzip2") base_id = 1;
    else if (lib_lower == "zstd") base_id = 2;
    else if (lib_lower == "lz4") base_id = 3;
    else if (lib_lower == "zlib") base_id = 4;
    else if (lib_lower == "lzma") base_id = 5;
    else if (lib_lower == "brotli") base_id = 6;
    else if (lib_lower == "snappy") base_id = 7;
    else if (lib_lower == "blosc" || lib_lower == "blosc2") base_id = 8;
    // Lossy base IDs (10+)
    else if (lib_lower == "zfp") base_id = 10;
    else if (lib_lower == "sz3") base_id = 11;
    else if (lib_lower == "fpzip") base_id = 12;
    // GPU compressors (nvcomp) base IDs (13+)
#if CTP_ENABLE_NVCOMP
    else if (lib_lower == "nvcomp-lz4") base_id = 13;
#endif

    if (base_id == 0) return 0;  // Unknown library

    // For single-mode libraries (SNAPPY, Blosc2, nvcomp), always use preset 2
    if (base_id == 7 || base_id == 8 || base_id == 13) {
      return base_id * 10 + 2;
    }

    // Encode preset: FAST=1, BALANCED=2, BEST=3, DEFAULT=2
    int preset_id = 2;  // Default to BALANCED
    switch (preset) {
      case CompressionPreset::FAST: preset_id = 1; break;
      case CompressionPreset::BALANCED: preset_id = 2; break;
      case CompressionPreset::BEST: preset_id = 3; break;
      case CompressionPreset::DEFAULT: preset_id = 2; break;
    }

    return base_id * 10 + preset_id;
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

    std::string library_name = "unknown";

    // Lossless libraries
    if (base_id == 1) library_name = "bzip2";
    else if (base_id == 2) library_name = "zstd";
    else if (base_id == 3) library_name = "lz4";
    else if (base_id == 4) library_name = "zlib";
    else if (base_id == 5) library_name = "lzma";
    else if (base_id == 6) library_name = "brotli";
    else if (base_id == 7) library_name = "snappy";
    else if (base_id == 8) library_name = "blosc2";
    // Lossy libraries
    else if (base_id == 10) library_name = "zfp";
    else if (base_id == 11) library_name = "sz3";
    else if (base_id == 12) library_name = "fpzip";
    // GPU compressors (nvcomp)
    else if (base_id == 13) library_name = "nvcomp-lz4";

    CompressionPreset preset = CompressionPreset::BALANCED;
    if (preset_id == 1) preset = CompressionPreset::FAST;
    else if (preset_id == 2) preset = CompressionPreset::BALANCED;
    else if (preset_id == 3) preset = CompressionPreset::BEST;

    return {library_name, preset};
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
