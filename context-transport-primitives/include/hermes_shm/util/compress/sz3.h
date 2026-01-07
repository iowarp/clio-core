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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_SZ3_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_SZ3_H_

#if HSHM_ENABLE_COMPRESS

#include <SZ3/api/sz.hpp>
#include <cstring>
#include <iostream>
#include <vector>

#include "compress.h"

namespace hshm {

/**
 * SZ3 error-bounded lossy compressor wrapper
 * Supports multiple error modes: absolute, relative, PSNR
 */
class Sz3 : public Compressor {
 public:
  enum class ErrorMode {
    ABS,      /**< Absolute error bound */
    REL,      /**< Relative error bound */
    PSNR,     /**< Peak Signal-to-Noise Ratio */
    ABS_AND_REL  /**< Both absolute and relative (use minimum) */
  };

 private:
  ErrorMode mode_;      /**< Error bound mode */
  double error_bound_;  /**< Error bound value */

 public:
  /**
   * Constructor
   * @param error_bound Error bound value (default: 1e-3)
   * @param mode Error mode (default: ABS)
   */
  explicit Sz3(double error_bound = 1e-3, ErrorMode mode = ErrorMode::ABS)
      : mode_(mode), error_bound_(error_bound) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    // Assume input is float array
    if (input_size % sizeof(float) != 0) {
      std::cerr << "SZ3: Input size must be multiple of sizeof(float)" << std::endl;
      return false;
    }

    size_t num_floats = input_size / sizeof(float);
    float* input_data = reinterpret_cast<float*>(input);

    // Create SZ3 configuration
    SZ::Config conf;
    conf.cmprAlgo = SZ::ALGO_INTERP_LORENZO;  // Default algorithm
    conf.errorBoundMode = GetSZ3ErrorMode();
    conf.absErrorBound = error_bound_;
    conf.relErrorBound = error_bound_;

    // For PSNR mode
    if (mode_ == ErrorMode::PSNR) {
      conf.psnr = error_bound_;
    }

    try {
      // Compress using SZ3
      size_t compressed_size = 0;
      char* compressed_data = SZ_compress<float>(
          conf,
          input_data,
          compressed_size,
          num_floats  // 1D array size
      );

      if (compressed_data == nullptr || compressed_size == 0) {
        std::cerr << "SZ3: Compression failed" << std::endl;
        return false;
      }

      if (compressed_size > output_size) {
        std::cerr << "SZ3: Output buffer too small" << std::endl;
        delete[] compressed_data;
        return false;
      }

      // Copy to output buffer
      std::memcpy(output, compressed_data, compressed_size);
      output_size = compressed_size;

      // Free SZ3-allocated memory
      delete[] compressed_data;

      return true;
    } catch (const std::exception& e) {
      std::cerr << "SZ3: Compression exception: " << e.what() << std::endl;
      return false;
    }
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    // Assume output is float array
    if (output_size % sizeof(float) != 0) {
      std::cerr << "SZ3: Output size must be multiple of sizeof(float)" << std::endl;
      return false;
    }

    size_t num_floats = output_size / sizeof(float);
    float* output_data = reinterpret_cast<float*>(output);

    // Create SZ3 configuration (needed for decompression)
    SZ::Config conf;

    try {
      // Decompress using SZ3
      float* decompressed_data = SZ_decompress<float>(
          conf,
          reinterpret_cast<char*>(input),
          input_size,
          num_floats  // Expected output size
      );

      if (decompressed_data == nullptr) {
        std::cerr << "SZ3: Decompression failed" << std::endl;
        return false;
      }

      // Copy to output buffer
      std::memcpy(output_data, decompressed_data, num_floats * sizeof(float));

      // Free SZ3-allocated memory
      delete[] decompressed_data;

      return true;
    } catch (const std::exception& e) {
      std::cerr << "SZ3: Decompression exception: " << e.what() << std::endl;
      return false;
    }
  }

  /**
   * Set error bound and mode
   * @param error_bound New error bound value
   * @param mode New error mode (optional, keeps current if not specified)
   */
  void SetErrorBound(double error_bound, ErrorMode mode) {
    error_bound_ = error_bound;
    mode_ = mode;
  }

  /**
   * Set error bound (keeps current mode)
   * @param error_bound New error bound value
   */
  void SetErrorBound(double error_bound) {
    error_bound_ = error_bound;
  }

  /**
   * Get current error bound
   * @return Current error bound value
   */
  double GetErrorBound() const {
    return error_bound_;
  }

  /**
   * Get current error mode
   * @return Current error mode
   */
  ErrorMode GetErrorMode() const {
    return mode_;
  }

 private:
  /**
   * Convert our ErrorMode to SZ3's error bound mode
   * @return SZ3 error bound mode enum
   */
  SZ::EB GetSZ3ErrorMode() const {
    switch (mode_) {
      case ErrorMode::ABS:
        return SZ::EB_ABS;
      case ErrorMode::REL:
        return SZ::EB_REL;
      case ErrorMode::PSNR:
        return SZ::EB_PSNR;
      case ErrorMode::ABS_AND_REL:
        return SZ::EB_ABS_AND_REL;
      default:
        return SZ::EB_ABS;
    }
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_SZ3_H_
