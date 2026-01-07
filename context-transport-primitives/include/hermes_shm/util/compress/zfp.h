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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_ZFP_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_ZFP_H_

#if HSHM_ENABLE_COMPRESS

#include <zfp.h>
#include <cstring>
#include <iostream>

#include "compress.h"

namespace hshm {

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

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_ZFP_H_
