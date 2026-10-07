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

#include <clio_cte/dtschedule/ccm/data_stats.h>

#include <cstring>
#include <cmath>

namespace clio::cte::dtschedule {

DataStats* DataStats::GetInstance() {
  static DataStats instance;
  return &instance;
}

DataStatsResult DataStats::Calculate(const uint8_t* data, size_t size) {
  if (size == 0 || data == nullptr) {
    return {0.0, 0.0, 0.0, DataType::kChar};
  }

  // Stride sampling: max 64 KiB sample
  const size_t kMaxSampleBytes = 65536;
  size_t stride = (size > kMaxSampleBytes) ? (size / kMaxSampleBytes) : 1;

  std::vector<uint8_t> sample;
  sample.reserve((size + stride - 1) / stride);
  for (size_t i = 0; i < size; i += stride) {
    sample.push_back(data[i]);
  }

  // Compute statistics
  double entropy = ComputeEntropy(sample);
  double mad = ComputeMAD(sample);
  double d2 = ComputeSecondDerivative(sample);
  DataType dtype = DetectType(data, size, sample);

  return {entropy, mad, d2, dtype};
}

double DataStats::ComputeEntropy(const std::vector<uint8_t>& sample) {
  if (sample.empty()) {
    return 0.0;
  }

  // Build bytewise histogram
  size_t histogram[256] = {0};
  for (uint8_t byte : sample) {
    histogram[byte]++;
  }

  // Calculate entropy: sum(-p_i * log2(p_i))
  double entropy = 0.0;
  size_t num_bytes = sample.size();

  for (int i = 0; i < 256; ++i) {
    if (histogram[i] > 0) {
      double p_i = static_cast<double>(histogram[i]) / static_cast<double>(num_bytes);
      entropy += -p_i * std::log2(p_i);
    }
  }

  return entropy;
}

double DataStats::ComputeMAD(const std::vector<uint8_t>& sample) {
  if (sample.empty()) {
    return 0.0;
  }

  // Calculate mean
  double sum = 0.0;
  for (uint8_t byte : sample) {
    sum += static_cast<double>(byte);
  }
  double mean = sum / static_cast<double>(sample.size());

  // Calculate mean absolute deviation
  double mad_sum = 0.0;
  for (uint8_t byte : sample) {
    mad_sum += std::abs(static_cast<double>(byte) - mean);
  }

  return mad_sum / static_cast<double>(sample.size());
}

double DataStats::ComputeSecondDerivative(const std::vector<uint8_t>& sample) {
  if (sample.size() < 3) {
    return 0.0;
  }

  // Calculate mean |second derivative|
  // d2[i] = data[i+1] - 2*data[i] + data[i-1]
  double d2_sum = 0.0;
  size_t num_derivatives = sample.size() - 2;

  for (size_t i = 1; i < sample.size() - 1; ++i) {
    double second_diff = static_cast<double>(sample[i + 1])
                         - 2.0 * static_cast<double>(sample[i])
                         + static_cast<double>(sample[i - 1]);
    d2_sum += std::abs(second_diff);
  }

  return d2_sum / static_cast<double>(num_derivatives);
}

DataType DataStats::DetectType(const uint8_t* data, size_t size,
                                const std::vector<uint8_t>& sample) {
  // Float32 requires: (size % 4 == 0) and floats are finite with bounded
  // exponent
  if (size % 4 != 0) {
    return DataType::kChar;
  }

  // Stride used during sampling
  size_t stride = (size > 65536) ? (size / 65536) : 1;

  // Check sampled floats for validity
  // Each sample byte represents one byte of the float stream
  // To get complete floats, reconstruct from the original data
  for (size_t i = 0; i + 3 < size; i += stride * 4) {
    float f;
    std::memcpy(&f, data + i, sizeof(float));
    if (!IsValidFloat32(f)) {
      return DataType::kChar;
    }
  }

  return DataType::kFloat32;
}

bool DataStats::IsValidFloat32(float f) {
  // Check if finite
  if (!std::isfinite(f)) {
    return false;
  }

  // Check for reasonable exponent range
  // Filter out very small subnormal numbers and very large exponents
  int exponent;
  (void)std::frexp(f, &exponent);

  // Allow exponents from -126 to 127 (standard float32 range)
  // Subnormals and special cases return false
  if (exponent < -126 || exponent > 128) {
    return false;
  }

  return true;
}

}  // namespace clio::cte::dtschedule
