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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SZ_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SZ_H_

#if CTP_ENABLE_COMPRESS

#include <sz/sz.h>
#include <cstring>
#include <iostream>

#include "compress.h"

namespace ctp {

/**
 * SZ (classic) error-bounded lossy compressor wrapper
 * Original SZ compressor (predecessor to SZ3)
 * Supports absolute and relative error bounds
 */
class Sz : public Compressor {
 public:
  enum class ErrorMode {
    ABS,      /**< Absolute error bound */
    REL,      /**< Relative error bound */
    PSNR,     /**< Peak Signal-to-Noise Ratio */
    NORM      /**< Norm error bound */
  };

 private:
  ErrorMode mode_;      /**< Error bound mode */
  double error_bound_;  /**< Error bound value */
  bool initialized_;    /**< SZ initialization flag */

 public:
  /**
   * Constructor
   * @param error_bound Error bound value (default: 1e-3)
   * @param mode Error mode (default: ABS)
   */
  explicit Sz(double error_bound = 1e-3, ErrorMode mode = ErrorMode::ABS)
      : mode_(mode), error_bound_(error_bound), initialized_(false) {
    InitializeSZ();
  }

  ~Sz() {
    if (initialized_) {
      SZ_Finalize();
    }
  }

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (!initialized_) {
      std::cerr << "SZ: Not initialized" << std::endl;
      return false;
    }

    // Assume input is float array
    if (input_size % sizeof(float) != 0) {
      std::cerr << "SZ: Input size must be multiple of sizeof(float)" << std::endl;
      return false;
    }

    size_t num_floats = input_size / sizeof(float);
    float* input_data = reinterpret_cast<float*>(input);

    // Set error bound mode
    ConfigureSZ();

    // Compress using SZ
    size_t compressed_size = 0;
    unsigned char* compressed_data = SZ_compress_args(
        SZ_FLOAT,
        input_data,
        &compressed_size,
        ABS,  // Will be overridden by confparams
        error_bound_,
        error_bound_,  // Used for relative bound
        0,  // pw_rel_err
        0,  // r1
        0,  // r2
        0,  // r3
        0,  // r4
        0   // r5
    );

    if (compressed_data == nullptr || compressed_size == 0) {
      std::cerr << "SZ: Compression failed" << std::endl;
      return false;
    }

    if (compressed_size > output_size) {
      std::cerr << "SZ: Output buffer too small" << std::endl;
      free(compressed_data);
      return false;
    }

    // Copy to output buffer
    std::memcpy(output, compressed_data, compressed_size);
    output_size = compressed_size;

    // Free SZ-allocated memory
    free(compressed_data);

    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (!initialized_) {
      std::cerr << "SZ: Not initialized" << std::endl;
      return false;
    }

    // Assume output is float array
    if (output_size % sizeof(float) != 0) {
      std::cerr << "SZ: Output size must be multiple of sizeof(float)" << std::endl;
      return false;
    }

    size_t num_floats = output_size / sizeof(float);

    // Decompress using SZ
    float* decompressed_data = (float*)SZ_decompress(
        SZ_FLOAT,
        reinterpret_cast<unsigned char*>(input),
        input_size,
        0,  // r1
        0,  // r2
        0,  // r3
        0,  // r4
        0   // r5
    );

    if (decompressed_data == nullptr) {
      std::cerr << "SZ: Decompression failed" << std::endl;
      return false;
    }

    // Copy to output buffer
    std::memcpy(output, decompressed_data, num_floats * sizeof(float));

    // Free SZ-allocated memory
    free(decompressed_data);

    return true;
  }

  /**
   * Set error bound and mode
   * @param error_bound New error bound value
   * @param mode New error mode
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
   * Initialize SZ library
   */
  void InitializeSZ() {
    if (!initialized_) {
      SZ_Init(nullptr);  // Use default config file (or nullptr)
      initialized_ = true;
    }
  }

  /**
   * Configure SZ with current error bound settings
   */
  void ConfigureSZ() {
    // Set error bound mode in global configuration
    switch (mode_) {
      case ErrorMode::ABS:
        confparams_cpr->errorBoundMode = ABS;
        confparams_cpr->absErrBound = error_bound_;
        break;
      case ErrorMode::REL:
        confparams_cpr->errorBoundMode = REL;
        confparams_cpr->relBoundRatio = error_bound_;
        break;
      case ErrorMode::PSNR:
        confparams_cpr->errorBoundMode = PSNR;
        confparams_cpr->psnr = error_bound_;
        break;
      case ErrorMode::NORM:
        confparams_cpr->errorBoundMode = NORM;
        confparams_cpr->absErrBound = error_bound_;
        break;
      default:
        confparams_cpr->errorBoundMode = ABS;
        confparams_cpr->absErrBound = error_bound_;
        break;
    }
  }
};

}  // namespace ctp

#endif  // CTP_ENABLE_COMPRESS

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_COMPRESS_SZ_H_
