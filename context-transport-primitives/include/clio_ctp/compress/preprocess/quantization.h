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
#include <cstring>
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
  /**
   * Quantized values, PACKED at the selected width (int8/int16/int32).
   *
   * Held as raw bytes rather than int32 because narrowing is the entire
   * point: NeuroPress sizes its output `num_elements * precision_to_bytes
   * (precision)` (quantization_kernels.cu:488), which is what turns float32
   * into a 4:1 or 2:1 smaller buffer BEFORE the codec runs. Storing every
   * value as int32 made the "quantizer" produce no size reduction at all
   * for float32 input, which is the only input the selection path has.
   */
  std::vector<uint8_t> quantized_;
  double error_bound_ = 0.0;            /**< Absolute error bound requested */
  /**
   * Bound actually used for the scale, after reserving float32
   * representation error and safety margin -- upstream's effective_eb
   * (quantization_kernels.cu:418-473). Always <= error_bound_.
   */
  double effective_error_bound_ = 0.0;
  /**
   * False when error_bound_ was below what float32 can represent over this
   * data range, so the round trip may EXCEED it.
   *
   * This regime is upstream's too -- it falls back to the tightest safe
   * bound and prints "Using maximum precision quantization (error may exceed
   * bound)" (quantization_kernels.cu:444-456). A header-only utility has
   * nowhere to warn, so the condition is reported here instead: silently
   * violating the guarantee this header documents would be worse than
   * either.
   */
  bool bound_achievable_ = true;
  double data_min_ = 0.0;               /**< Minimum of original data */
  double data_max_ = 0.0;               /**< Maximum of original data */
  double scale_ = 0.0;                  /**< 1 / (2 * effective_error_bound_) */
  int precision_ = 0;                   /**< Bits used (8, 16, or 32) */
  size_t num_elements_ = 0;             /**< Number of elements */

  QuantizationResult() = default;

  /** @brief Packed output size in bytes -- what the codec would see. */
  size_t SizeBytes() const { return quantized_.size(); }
};

/** @brief Bytes per quantized value, mirroring precision_to_bytes(). */
inline size_t PrecisionToBytes(int precision) {
  switch (precision) {
    case 8: return 1;
    case 16: return 2;
    default: return 4;
  }
}

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

  const double data_min = static_cast<double>(min_val);
  const double data_max = static_cast<double>(max_val);
  // Constant data gets upstream's substituted range, not the degenerate 0
  // (quantization_kernels.cu:408-411) -- see the same clamp, and why it
  // changes the selected precision, in QuantizeDevice
  // (data_stats_gpu_kernels.cu). This path has to agree with that one about
  // how a blob is encoded, so the derivation stays shared.
  double data_range = data_max - data_min;
  if (data_range <= 0.0) data_range = 1.0;

  // ---- Effective error bound (quantization_kernels.cu:418-473) ----
  // The user's bound has to cover BOTH the quantization step and the error
  // of representing the reconstructed value back in float. Spending all of
  // it on quantization means the round trip can exceed the bound this
  // header promises. Upstream reserves the float32 representation error and
  // a 5% margin, then floors the result so the quantized values cannot
  // overflow int32.
  const double max_abs_value = std::max(std::fabs(data_min), std::fabs(data_max));
  const double float_repr_error = max_abs_value * 2.4e-7;
  const double safety_margin = error_bound * 0.05;
  const double available_for_quant = error_bound - float_repr_error - safety_margin;
  const double min_eb_for_int32 = data_range / 4.0e9;

  double effective_eb;
  if (available_for_quant <= 0.0) {
    // Bound is below what float32 can represent over this range: fall back
    // to the tightest still-safe value rather than producing garbage.
    effective_eb = std::max(min_eb_for_int32, float_repr_error * 0.1);
    result.bound_achievable_ = false;
  } else {
    effective_eb = available_for_quant;
  }
  effective_eb = std::max(effective_eb, min_eb_for_int32);
  if (effective_eb <= 0.0) return QuantizationResult();

  // Precision is chosen from the EFFECTIVE bound, so the width can actually
  // hold every quantized value.
  const int precision = ComputeRequiredPrecision(data_range, effective_eb);
  const double scale = 1.0 / (2.0 * effective_eb);
  const size_t width = PrecisionToBytes(precision);

  // Quantize: q = round((value - min) * scale), clamped to the width.
  // Upstream clamps in-kernel (quantization_kernels.cu:70-77); without it a
  // value outside the type's range is an out-of-range float->int conversion,
  // which is undefined behavior rather than a saturated result.
  result.quantized_.resize(num_elements * width);
  for (size_t i = 0; i < num_elements; ++i) {
    const double centered = static_cast<double>(data[i]) - data_min;
    double q = std::round(centered * scale);
    uint8_t* slot = result.quantized_.data() + i * width;
    if (width == 1) {
      q = std::max(-128.0, std::min(127.0, q));
      const int8_t v = static_cast<int8_t>(q);
      std::memcpy(slot, &v, sizeof(v));
    } else if (width == 2) {
      q = std::max(-32768.0, std::min(32767.0, q));
      const int16_t v = static_cast<int16_t>(q);
      std::memcpy(slot, &v, sizeof(v));
    } else {
      q = std::max(-2147483648.0, std::min(2147483647.0, q));
      const int32_t v = static_cast<int32_t>(q);
      std::memcpy(slot, &v, sizeof(v));
    }
  }

  result.error_bound_ = error_bound;
  result.effective_error_bound_ = effective_eb;
  result.data_min_ = data_min;
  result.data_max_ = data_max;
  result.scale_ = scale;
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

  if (result.quantized_.empty() || result.scale_ <= 0.0 ||
      result.num_elements_ == 0) {
    return output;
  }

  // restored = q * (2 * effective_eb) + data_min -- the exact inverse of the
  // forward pass, and identical to upstream's dequantize_linear_kernel
  // (quantization_kernels.cu:83-99), which takes inv_scale = 2*effective_eb
  // and offset = data_min.
  const double inv_scale = 1.0 / result.scale_;
  const size_t width = PrecisionToBytes(result.precision_);
  if (result.quantized_.size() < result.num_elements_ * width) {
    return output;
  }

  output.reserve(result.num_elements_);
  for (size_t i = 0; i < result.num_elements_; ++i) {
    const uint8_t* slot = result.quantized_.data() + i * width;
    double q = 0.0;
    if (width == 1) {
      int8_t v = 0;
      std::memcpy(&v, slot, sizeof(v));
      q = static_cast<double>(v);
    } else if (width == 2) {
      int16_t v = 0;
      std::memcpy(&v, slot, sizeof(v));
      q = static_cast<double>(v);
    } else {
      int32_t v = 0;
      std::memcpy(&v, slot, sizeof(v));
      q = static_cast<double>(v);
    }
    output.push_back(static_cast<T>(q * inv_scale + result.data_min_));
  }

  return output;
}

/**
 * @brief Parameters a reader needs to invert a device quantization.
 *
 * Mirrors the fields NeuroPress stores in its own header
 * (compression_header.h:63-66: quant_error_bound, quant_scale, data_min,
 * data_max) plus the precision, which upstream keeps in quant_flags.
 * Everything here must survive to the read side or the data is unrecoverable.
 */
struct DeviceQuantizeParams {
  double error_bound = 0.0;      /**< Bound requested by the caller */
  double effective_error_bound = 0.0;  /**< Bound actually used for the scale */
  double scale = 0.0;            /**< 1 / (2 * effective_error_bound) */
  double data_min = 0.0;         /**< Minimum of the original data */
  double data_max = 0.0;         /**< Maximum of the original data */
  int precision = 0;             /**< 8, 16 or 32 bits per value */
  bool bound_achievable = true;  /**< False if eb was below float32 precision */
};

/**
 * @brief Quantize a device float32 buffer in place of a device output.
 *
 * Device-resident counterpart of Quantize() above, and the one the selection
 * path can actually use: the data being compressed lives on the GPU, so
 * quantizing on the host would mean a D2H/H2D round trip per chunk.
 * Reproduces quantize_simple()'s pipeline (quantization_kernels.cu): CUB
 * min/max reduction, effective error bound, precision from that bound, then
 * a clamped linear quantization written at the selected width.
 *
 * @param device_in   Device float32 buffer.
 * @param num_elements Element count (NOT bytes).
 * @param error_bound Requested absolute bound; must be > 0.
 * @param device_out  Device output, at least num_elements * 4 bytes.
 * @param out_bytes   Receives the packed output size.
 * @param out_params  Receives everything the read side needs to invert this.
 * @return false if unsupported, if CUDA failed, or if CUDA is not compiled in.
 */
bool QuantizeDevice(const void *device_in, size_t num_elements,
                    double error_bound, void *device_out, size_t *out_bytes,
                    DeviceQuantizeParams *out_params);

/**
 * @brief Inverse of QuantizeDevice(): packed integers back to float32.
 *
 * @param device_in  Packed quantized values, as QuantizeDevice wrote them.
 * @param num_elements Element count.
 * @param params     The parameters QuantizeDevice returned.
 * @param device_out Device float32 output, at least num_elements * 4 bytes.
 */
bool DequantizeDevice(const void *device_in, size_t num_elements,
                      const DeviceQuantizeParams &params, void *device_out);

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_QUANTIZATION_H_
