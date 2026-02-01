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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_BITGROOMING_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_BITGROOMING_H_

#if HSHM_ENABLE_COMPRESS

#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "compress.h"

namespace hshm {

/**
 * BitGrooming lossy compressor for floating-point data
 * Preserves a specified number of significant digits
 * NOTE: BitGrooming alone doesn't compress - it reduces precision
 * Should be paired with a lossless compressor for actual compression
 */
class BitGrooming : public Compressor {
 private:
  int nsd_;  /**< Number of significant digits to preserve */

 public:
  /**
   * Constructor
   * @param nsd Number of significant digits to preserve (default: 3)
   */
  explicit BitGrooming(int nsd = 3) : nsd_(nsd) {}

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    // Assume input is float array
    if (input_size % sizeof(float) != 0) {
      std::cerr << "BitGrooming: Input size must be multiple of sizeof(float)" << std::endl;
      return false;
    }

    size_t num_floats = input_size / sizeof(float);
    float* input_data = reinterpret_cast<float*>(input);
    float* output_data = reinterpret_cast<float*>(output);

    if (output_size < input_size) {
      std::cerr << "BitGrooming: Output buffer too small" << std::endl;
      return false;
    }

    // Apply bit grooming to each float
    for (size_t i = 0; i < num_floats; ++i) {
      output_data[i] = ApplyBitGrooming(input_data[i], nsd_);
    }

    output_size = input_size;
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    // BitGrooming is a lossy transformation - decompression is just a copy
    if (output_size < input_size) {
      std::cerr << "BitGrooming: Output buffer too small" << std::endl;
      return false;
    }

    std::memcpy(output, input, input_size);
    output_size = input_size;
    return true;
  }

  /**
   * Set number of significant digits
   * @param nsd Number of significant digits to preserve
   */
  void SetNSD(int nsd) {
    nsd_ = nsd;
  }

  /**
   * Get number of significant digits
   * @return Number of significant digits preserved
   */
  int GetNSD() const {
    return nsd_;
  }

 private:
  /**
   * Apply bit grooming to a single float value
   * @param value Input value
   * @param nsd Number of significant digits to preserve
   * @return Groomed value
   */
  float ApplyBitGrooming(float value, int nsd) {
    if (!std::isfinite(value) || value == 0.0f) {
      return value;
    }

    // Get exponent
    int exponent;
    std::frexp(value, &exponent);

    // Calculate number of bits to keep
    // IEEE 754 single precision has 23 mantissa bits
    // We want to keep nsd significant decimal digits
    // log2(10) ≈ 3.32, so nsd decimal digits ≈ nsd * 3.32 bits
    int bits_to_keep = static_cast<int>(nsd * 3.32192809489);
    bits_to_keep = std::min(bits_to_keep, 23);  // Max 23 mantissa bits
    bits_to_keep = std::max(bits_to_keep, 1);   // Min 1 bit

    // Number of bits to zero out
    int bits_to_zero = 23 - bits_to_keep;

    // Get the bit representation
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(float));

    // Create mask to zero out the least significant bits
    uint32_t mask = ~((1U << bits_to_zero) - 1);

    // Apply mask
    bits &= mask;

    // Convert back to float
    float result;
    std::memcpy(&result, &bits, sizeof(float));

    return result;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_BITGROOMING_H_
