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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_FPZIP_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_FPZIP_H_

#if CTP_ENABLE_COMPRESS

#include <fpzip.h>
#include <cstring>
#include <iostream>

#include "compress.h"

namespace ctp {

/**
 * FPZIP fast floating-point compressor wrapper
 * Optimized for speed with moderate compression ratios
 * Supports both lossless and lossy compression modes
 */
class Fpzip : public Compressor {
 private:
  int precision_;   /**< Number of bits of precision (0 = lossless) */
  bool lossless_;   /**< True for lossless mode, false for lossy */

 public:
  /**
   * Constructor
   * @param precision Number of bits of precision (0 = lossless, 1-32 = lossy)
   *                  Higher values = better quality, lower compression
   */
  explicit Fpzip(int precision = 0)
      : precision_(precision), lossless_(precision == 0) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    // Assume input is float array
    if (input_size % sizeof(float) != 0) {
      std::cerr << "FPZIP: Input size must be multiple of sizeof(float)" << std::endl;
      return false;
    }

    size_t num_floats = input_size / sizeof(float);
    float* input_data = reinterpret_cast<float*>(input);

    // Create FPZIP stream
    FPZ* fpz = fpzip_write_to_buffer(output, output_size);
    if (!fpz) {
      std::cerr << "FPZIP: Failed to create compression stream" << std::endl;
      return false;
    }

    // Set compression parameters
    fpz->type = FPZIP_TYPE_FLOAT;
    fpz->prec = precision_;  // 0 = lossless, >0 = lossy with specified precision
    fpz->nx = num_floats;     // 1D array size
    fpz->ny = 1;
    fpz->nz = 1;
    fpz->nf = 1;

    // Write header (contains metadata for decompression)
    size_t header_size = fpzip_write_header(fpz);
    if (header_size == 0) {
      std::cerr << "FPZIP: Failed to write header" << std::endl;
      fpzip_write_close(fpz);
      return false;
    }

    // Compress the data
    size_t compressed_size = fpzip_write(fpz, input_data);
    fpzip_write_close(fpz);

    if (compressed_size == 0) {
      std::cerr << "FPZIP: Compression failed" << std::endl;
      return false;
    }

    output_size = compressed_size;
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    // Assume output is float array
    if (output_size % sizeof(float) != 0) {
      std::cerr << "FPZIP: Output size must be multiple of sizeof(float)" << std::endl;
      return false;
    }

    size_t num_floats = output_size / sizeof(float);
    float* output_data = reinterpret_cast<float*>(output);

    // Create FPZIP stream for reading
    FPZ* fpz = fpzip_read_from_buffer(input);
    if (!fpz) {
      std::cerr << "FPZIP: Failed to create decompression stream" << std::endl;
      return false;
    }

    // Read header to get compression parameters
    if (!fpzip_read_header(fpz)) {
      std::cerr << "FPZIP: Failed to read header" << std::endl;
      fpzip_read_close(fpz);
      return false;
    }

    // Decompress the data
    // Note: fpzip_read() returns compressed bytes consumed, not decompressed size
    size_t compressed_bytes_read = fpzip_read(fpz, output_data);
    fpzip_read_close(fpz);

    if (compressed_bytes_read == 0) {
      std::cerr << "FPZIP: Decompression failed" << std::endl;
      return false;
    }

    // output_size is already set correctly (num_floats * sizeof(float))
    // No need to update it
    return true;
  }

  /**
   * Set precision for lossy compression
   * @param precision Number of bits (0 = lossless, 1-32 = lossy)
   */
  void SetPrecision(int precision) {
    precision_ = precision;
    lossless_ = (precision == 0);
  }

  /**
   * Get current precision setting
   * @return Number of bits of precision (0 = lossless)
   */
  int GetPrecision() const {
    return precision_;
  }

  /**
   * Check if in lossless mode
   * @return True if lossless, false if lossy
   */
  bool IsLossless() const {
    return lossless_;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_FPZIP_H_
