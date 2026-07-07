/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file quantization.h
 * @brief Error-bounded quantization preprocessing for CPU.
 *
 * Provides header-only CPU quantization as a preprocessing step to improve
 * compression ratios. Uses linear error-bounded quantization:
 *   q = round(value / (2 * error_bound))
 *
 * Automatically selects output precision (int8/16/32) based on data range
 * and error bound. Includes metadata for reversible dequantization.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_QUANTIZATION_H_
#define CLIO_CTP_COMPRESS_PREPROCESS_QUANTIZATION_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>

namespace ctp::compress::preprocess {

/**
 * @brief Result of quantization with metadata for dequantization.
 *
 * Stores quantized data and all parameters needed to reverse the
 * quantization with error bound guarantee:
 * |original - dequantized| <= error_bound
 */
struct QuantizationResult {
  std::vector<int32_t> quantized_;      /**< Quantized integer values */
  double error_bound_;                  /**< Absolute error bound */
  double data_min_;                     /**< Minimum of original data */
  double data_max_;                     /**< Maximum of original data */
  double offset_;                       /**< Offset for requantization */
  int precision_;                       /**< Bits used (8, 16, or 32) */
  size_t num_elements_;                 /**< Number of elements */

  QuantizationResult() = default;
};

/**
 * @brief Compute required precision for given range and error bound.
 *
 * Determines minimum bits needed to represent all quantized values.
 * Range is divided into bins of width 2*error_bound; we compute how many
 * bins fit and select int8/16/32 accordingly.
 *
 * @param data_range    max - min of original data
 * @param error_bound   absolute error tolerance
 * @return 8, 16, or 32 bits
 */
inline int ComputeRequiredPrecision(double data_range,
                                     double error_bound) {
  if (error_bound <= 0.0) return 32;
  double num_bins = data_range / (2.0 * error_bound);
  num_bins = num_bins * 1.1;  // 10% safety margin
  if (num_bins <= 127.0) return 8;
  if (num_bins <= 32767.0) return 16;
  return 32;
}

/**
 * @brief Quantize floating-point data with error bounds.
 *
 * Applies linear quantization: q = round(value / (2 * error_bound)).
 * Automatically selects output precision based on data range.
 *
 * Template parameter T: float or double (input type)
 *
 * @param data           Pointer to input data (float or double)
 * @param num_elements   Number of elements
 * @param error_bound    Absolute error tolerance (must be > 0)
 * @return QuantizationResult with quantized data and metadata
 */
template <typename T>
QuantizationResult Quantize(const T* data,
                            size_t num_elements,
                            double error_bound) {
  QuantizationResult result;

  if (!data || num_elements == 0 || error_bound <= 0.0) {
    return result;
  }

  // Compute data range
  T min_val = data[0];
  T max_val = data[0];
  for (size_t i = 1; i < num_elements; ++i) {
    if (data[i] < min_val) min_val = data[i];
    if (data[i] > max_val) max_val = data[i];
  }

  double data_min = static_cast<double>(min_val);
  double data_max = static_cast<double>(max_val);
  double data_range = data_max - data_min;

  // Select precision and compute scale
  int precision = ComputeRequiredPrecision(data_range, error_bound);
  double scale = 1.0 / (2.0 * error_bound);

  // Quantize: q = round((value - min) * scale)
  result.quantized_.reserve(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    double normalized = (static_cast<double>(data[i]) - data_min) * scale;
    int32_t q = static_cast<int32_t>(std::round(normalized));
    result.quantized_.push_back(q);
  }

  result.error_bound_ = error_bound;
  result.data_min_ = data_min;
  result.data_max_ = data_max;
  result.offset_ = scale;
  result.precision_ = precision;
  result.num_elements_ = num_elements;

  return result;
}

/**
 * @brief Dequantize back to floating-point with error bound guarantee.
 *
 * Reverses quantization using metadata from QuantizationResult.
 * Reconstructed value = (q / scale) + min, with error <= error_bound.
 *
 * Template parameter T: float or double (output type)
 *
 * @param result   QuantizationResult from Quantize()
 * @return Vector of reconstructed floating-point values
 */
template <typename T>
std::vector<T> Dequantize(const QuantizationResult& result) {
  std::vector<T> output;

  if (result.quantized_.empty() || result.offset_ <= 0.0) {
    return output;
  }

  output.reserve(result.num_elements_);
  double scale_inv = 1.0 / result.offset_;

  for (size_t i = 0; i < result.quantized_.size(); ++i) {
    double normalized =
        static_cast<double>(result.quantized_[i]) * scale_inv;
    double reconstructed = normalized + result.data_min_;
    output.push_back(static_cast<T>(reconstructed));
  }

  return output;
}

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_QUANTIZATION_H_
