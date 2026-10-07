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

#ifndef CLIO_CTE_DTSCHEDULE_CCM_DATA_STATS_H_
#define CLIO_CTE_DTSCHEDULE_CCM_DATA_STATS_H_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>

namespace clio::cte::dtschedule {

/**
 * Data type enumeration for dtschedule codec selection.
 */
enum class DataType : u_int8_t {
  kChar = 0,      ///< Byte data (default)
  kFloat32 = 1,   ///< 32-bit floating point
};

/**
 * Data statistics result structure.
 *
 * Contains the computed statistics for a blob used by the codec selection model.
 */
struct DataStatsResult {
  double entropy;      ///< Shannon entropy (bits, 0-8 for byte data)
  double mad;          ///< Mean absolute deviation from mean
  double d2;           ///< Mean absolute second derivative
  DataType dtype;      ///< Detected data type (char or float32)
};

/**
 * DataStats: Computes statistics for codec selection in dtschedule.
 *
 * This class extracts features from blobs (or samples of them) for the
 * Q-table codec selection model. It performs stride sampling to limit the
 * amount of data read (≤64 KiB) and computes Shannon entropy, mean absolute
 * deviation, and second-derivative statistics used by the trained model to
 * rank codecs.
 *
 * The class is a singleton to reuse computation state across calls.
 */
class DataStats {
 public:
  /**
   * Get the singleton instance of DataStats.
   *
   * @return Pointer to the DataStats singleton instance.
   */
  static DataStats* GetInstance();

  /**
   * Calculate statistics from blob data with stride sampling.
   *
   * Samples at most 64 KiB of the blob, using a stride to skip over large
   * blobs. Computes Shannon entropy (bytewise, log2), mean absolute deviation
   * from the sample mean, and mean absolute second derivative. Detects data
   * type as float32 (if size % 4 == 0 and floats are finite with bounded
   * exponent) or char.
   *
   * @param data Pointer to the blob data buffer.
   * @param size Total size of the blob in bytes.
   * @return DataStatsResult with computed statistics and detected type.
   */
  DataStatsResult Calculate(const uint8_t* data, size_t size);

 private:
  /**
   * Compute Shannon entropy from sampled data.
   *
   * Builds a bytewise histogram of the sampled data and calculates the
   * Shannon entropy using log2. Entropy ranges from 0 (all same byte) to
   * 8 (uniform random) for byte data.
   *
   * @param sample Sampled data bytes.
   * @return Shannon entropy in bits.
   */
  double ComputeEntropy(const std::vector<uint8_t>& sample);

  /**
   * Compute mean absolute deviation.
   *
   * Calculates the mean of the data, then returns the average absolute
   * distance from that mean.
   *
   * @param sample Sampled data bytes.
   * @return Mean absolute deviation.
   */
  double ComputeMAD(const std::vector<uint8_t>& sample);

  /**
   * Compute mean absolute second derivative.
   *
   * Calculates the second derivative at each point using the formula:
   * d2[i] = data[i+1] - 2*data[i] + data[i-1], then returns the mean
   * absolute value of these second derivatives.
   *
   * @param sample Sampled data bytes.
   * @return Mean absolute second derivative.
   */
  double ComputeSecondDerivative(const std::vector<uint8_t>& sample);

  /**
   * Detect the data type of the blob.
   *
   * Returns float32 if (size % 4 == 0) and the sampled floats are finite
   * with bounded exponent; otherwise returns char.
   *
   * @param data Pointer to the blob data buffer.
   * @param size Total size of the blob in bytes.
   * @param sample Sampled data bytes (must be already populated).
   * @return Detected DataType.
   */
  DataType DetectType(const uint8_t* data, size_t size,
                      const std::vector<uint8_t>& sample);

  /**
   * Check if a float32 value is valid (finite with bounded exponent).
   *
   * @param f The float32 value to check.
   * @return True if the float is finite and has a reasonable exponent.
   */
  bool IsValidFloat32(float f);

  DataStats() = default;
  ~DataStats() = default;
};

}  // namespace clio::cte::dtschedule

#endif  // CLIO_CTE_DTSCHEDULE_CCM_DATA_STATS_H_
