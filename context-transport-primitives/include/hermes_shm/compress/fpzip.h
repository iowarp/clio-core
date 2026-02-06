/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_FPZIP_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_FPZIP_H_

#if HSHM_ENABLE_COMPRESS

#include <fpzip.h>
#include <cstring>
#include <iostream>

#include "compress.h"

namespace hshm {

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

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_FPZIP_H_
