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

#ifndef CLIO_CTE_COMPRESSOR_COMPRESSION_HEADER_H_
#define CLIO_CTE_COMPRESSOR_COMPRESSION_HEADER_H_

#include <cstdint>

namespace clio::cte::compressor {

/**
 * Compression preset wire format. Must match CTP factory encoding:
 * FAST=1, BALANCED=2, BEST=3
 */
enum class CompressPreset : uint32_t {
  kFast = 1,      ///< Fast compression, lower ratio
  kBalanced = 2,  ///< Balanced speed and ratio (default)
  kBest = 3,      ///< Best ratio, slower
};

/**
 * Compression header prepended to compressed data for self-describing format.
 * This allows decompression without external metadata.
 *
 * Interoperable between compressor and dtschedule modules. Magic word is
 * "CTEC" (0x43544543). Header must be exactly 32 bytes for aligned placement.
 */
struct CompressionHeader {
  static constexpr uint32_t kMagic = 0x43544543;  // "CTEC" in ASCII

  uint32_t magic_;            ///< Magic number to identify compressed data
  uint32_t compress_lib_;     ///< Compression library ID
  uint32_t compress_preset_;  ///< Compression preset (CompressPreset wire format)
  uint64_t original_size_;    ///< Original uncompressed size
  /**
   * EXACT compressed payload size (bytes following this header).
   *
   * Frame-exact codecs need it. A reader does not know the compressed length
   * a priori, so it over-allocates its fetch and the trailing bytes are
   * garbage; passing that over-estimate to LZ4_decompress_safe (or zstd) makes
   * decompression FAIL. Without this field the reader could only guess
   * "request size minus header", which is exactly that over-estimate.
   *
   * 0 means "written before this field existed" -- readers fall back to the
   * derived estimate for those blobs.
   */
  uint64_t compressed_size_;

  /**
   * Default constructor initializing all fields to zero/default values.
   */
  CompressionHeader()
      : magic_(kMagic),
        compress_lib_(0),
        compress_preset_(0),
        original_size_(0),
        compressed_size_(0) {}

  /**
   * Constructor with specified compression parameters.
   *
   * @param lib Compression library ID
   * @param preset Compression preset (CompressPreset wire format)
   * @param orig_size Original uncompressed size in bytes
   * @param comp_size Compressed size in bytes (0 = unknown, reader estimates)
   */
  CompressionHeader(uint32_t lib, uint32_t preset, uint64_t orig_size,
                    uint64_t comp_size = 0)
      : magic_(kMagic),
        compress_lib_(lib),
        compress_preset_(preset),
        original_size_(orig_size),
        compressed_size_(comp_size) {}

  /**
   * Check if this header is valid (has correct magic number).
   *
   * @return True if magic matches kMagic, false otherwise
   */
  bool IsValid() const { return magic_ == kMagic; }
};

static_assert(sizeof(CompressionHeader) == 32,
              "CompressionHeader must be exactly 32 bytes for alignment");

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_COMPRESSION_HEADER_H_
