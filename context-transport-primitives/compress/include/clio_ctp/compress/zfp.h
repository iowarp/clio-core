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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_ZFP_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_ZFP_H_

#if CTP_ENABLE_COMPRESS

#include <zfp.h>
#include <cstring>
#include <iostream>

#include "compress.h"

namespace ctp {

/**
 * ZFP lossy floating-point compressor wrapper
 * Supports configurable error bounds for lossy compression
 */
class Zfp : public Compressor {
 private:
  double tolerance_;  /**< Absolute error tolerance for compression */

 public:
  /**
   * Constructor
   * @param tolerance Absolute error tolerance (default: 1e-3)
   */
  explicit Zfp(double tolerance = 1e-3) : tolerance_(tolerance) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    // Assume input is float array
    if (input_size % sizeof(float) != 0) {
      std::cerr << "ZFP: Input size must be multiple of sizeof(float)" << std::endl;
      return false;
    }

    size_t num_floats = input_size / sizeof(float);
    float* input_data = reinterpret_cast<float*>(input);

    // Create ZFP field for 1D float array
    zfp_type type = zfp_type_float;
    zfp_field* field = zfp_field_1d(input_data, type, num_floats);

    // Create ZFP stream
    zfp_stream* zfp = zfp_stream_open(NULL);

    // Set accuracy mode with tolerance
    zfp_stream_set_accuracy(zfp, tolerance_);

    // Allocate buffer for compressed stream
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    if (bufsize > output_size) {
      std::cerr << "ZFP: Output buffer too small" << std::endl;
      zfp_field_free(field);
      zfp_stream_close(zfp);
      return false;
    }

    // Associate bit stream with allocated buffer
    bitstream* stream = stream_open(output, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // Compress entire array
    size_t zfpsize = zfp_compress(zfp, field);
    if (zfpsize == 0) {
      std::cerr << "ZFP: Compression failed" << std::endl;
      zfp_field_free(field);
      zfp_stream_close(zfp);
      stream_close(stream);
      return false;
    }

    output_size = zfpsize;

    // Cleanup
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    // Assume output is float array
    if (output_size % sizeof(float) != 0) {
      std::cerr << "ZFP: Output size must be multiple of sizeof(float)" << std::endl;
      return false;
    }

    size_t num_floats = output_size / sizeof(float);
    float* output_data = reinterpret_cast<float*>(output);

    // Create ZFP field for 1D float array
    zfp_type type = zfp_type_float;
    zfp_field* field = zfp_field_1d(output_data, type, num_floats);

    // Create ZFP stream
    zfp_stream* zfp = zfp_stream_open(NULL);

    // Set the same accuracy mode used during compression
    zfp_stream_set_accuracy(zfp, tolerance_);

    // Associate bit stream with compressed data
    bitstream* stream = stream_open(input, input_size);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    // Decompress entire array
    if (!zfp_decompress(zfp, field)) {
      std::cerr << "ZFP: Decompression failed" << std::endl;
      zfp_field_free(field);
      zfp_stream_close(zfp);
      stream_close(stream);
      return false;
    }

    // Cleanup
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    return true;
  }

  /**
   * Set compression tolerance
   * @param tolerance Absolute error tolerance
   */
  void SetTolerance(double tolerance) {
    tolerance_ = tolerance;
  }

  /**
   * Get current tolerance
   * @return Current absolute error tolerance
   */
  double GetTolerance() const {
    return tolerance_;
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_ZFP_H_
