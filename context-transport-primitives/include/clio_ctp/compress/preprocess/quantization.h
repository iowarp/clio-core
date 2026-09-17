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
    case 64: return 8;
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

/* Quantize<T>() / Dequantize<T>() -- the CPU implementations -- were REMOVED,
 * for the reason given in byte_shuffle.h: NeuroPress preprocessing is
 * CUDA-only, and a working CPU mirror turned "the GPU path" into something a
 * host-resident chunk silently opted out of.
 *
 * Use QuantizeDevice() / DequantizeDevice() below. QuantizationResult is kept
 * because it still describes what those write into the blob header.
 */




/**
 * @brief Why QuantizeDevice() returned false. Never the data: any chunk of
 *        float32 or float64 quantizes under any positive bound.
 */
enum class QuantizeRefusal : int {
  kNone = 0,
  kInvalidArgument, /**< null buffer, no elements, element size or bound */
  kDeviceError,     /**< CUDA failed */
  kNoCudaSupport,
};

/** @brief Prose form, for logs. */
inline const char *QuantizeRefusalName(QuantizeRefusal reason) {
  switch (reason) {
    case QuantizeRefusal::kNone:
      return "none";
    case QuantizeRefusal::kInvalidArgument:
      return "invalid argument";
    case QuantizeRefusal::kDeviceError:
      return "CUDA error";
    case QuantizeRefusal::kNoCudaSupport:
      return "built without CUDA support";
  }
  return "unknown";
}

/** @brief Token form, for the explore.csv column: no spaces or commas. */
inline const char *QuantizeRefusalToken(QuantizeRefusal reason) {
  switch (reason) {
    case QuantizeRefusal::kNone:            return "none";
    case QuantizeRefusal::kInvalidArgument: return "invalid_argument";
    case QuantizeRefusal::kDeviceError:     return "device_error";
    case QuantizeRefusal::kNoCudaSupport:   return "no_cuda_support";
  }
  return "unknown";
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
  double data_min = 0.0;         /**< Grid offset: min of gridded elements */
  double data_max = 0.0;         /**< Max of gridded elements */
  int precision = 0;             /**< 8, 16, 32 or 64 bits per slot */
  int elem_bytes = 4;            /**< 4 = float32, 8 = float64 */
  /** Slots are elem_bytes wide and a bitmap (bit i of byte i/8) follows
      them; bit i%8 of byte i/8 set means slot i holds element i's own bits. */
  bool escapes = false;
  uint64_t escape_count = 0;     /**< Set bits in that bitmap */
  bool bound_achievable = true;  /**< Always true on success */
  QuantizeRefusal refusal = QuantizeRefusal::kNone; /**< Set on every false
                                      return */
};

/** @brief Bytes QuantizeDevice wrote for these parameters: slots, bitmap,
 *  and zeros up to a multiple of 8, since nvcomp-bitcomp reads 8-byte words
 *  and mangles a partial one. */
inline size_t QuantizedBytes(size_t num_elements,
                             const DeviceQuantizeParams &p) {
  const size_t b = num_elements * PrecisionToBytes(p.precision) +
                   (p.escapes ? (num_elements + 7) / 8 : 0);
  return (b + 7) & ~size_t{7};
}

/** @brief Output QuantizeDevice may need: escape slots, bitmap, padding. */
inline size_t QuantizeCapacity(size_t num_elements, size_t elem_bytes) {
  return (num_elements * elem_bytes + (num_elements + 7) / 8 + 7) &
         ~size_t{7};
}

/**
 * @brief Error-bounded linear quantization of a device float32/float64 buffer.
 *
 * Indices are round((x - data_min) * scale) at the narrowest width the chunk
 * needs, and every element is decoded with DequantizeDevice's arithmetic and
 * checked against error_bound before the chunk is accepted. When some element
 * cannot be held that way -- NaN or Inf, a spread wider than the widest index,
 * or a failed check -- the output switches to escape slots: elem_bytes-wide
 * slots, where an escaped element keeps its own bits, and a bitmap after them.
 * So every element decodes within error_bound or bit-exact, and the call never
 * declines a chunk.
 *
 * @param device_in    Device buffer of num_elements float32 or float64.
 * @param num_elements Element count (NOT bytes).
 * @param elem_bytes   4 (float32) or 8 (float64).
 * @param error_bound  Absolute bound; any value > 0, infinity included.
 * @param device_out   Device output of QuantizeCapacity() bytes.
 * @param out_bytes    Receives QuantizedBytes().
 * @param out_params   Receives everything the read side needs to invert this.
 * @param stream       cudaStream_t to launch on, as an opaque pointer, or
 *                     nullptr for the shared per-thread stream. The call
 *                     synchronizes only that stream.
 * @return false only for invalid arguments or a CUDA failure
 *         (out_params->refusal says which).
 */
bool QuantizeDevice(const void *device_in, size_t num_elements,
                    size_t elem_bytes, double error_bound, void *device_out,
                    size_t *out_bytes, DeviceQuantizeParams *out_params,
                    void *stream = nullptr);

/**
 * @brief Inverse of QuantizeDevice().
 *
 * @param device_in  QuantizedBytes() bytes, as QuantizeDevice wrote them.
 * @param num_elements Element count.
 * @param params     The parameters QuantizeDevice returned.
 * @param device_out Device output, num_elements * params.elem_bytes bytes.
 */
bool DequantizeDevice(const void *device_in, size_t num_elements,
                      const DeviceQuantizeParams &params, void *device_out);

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_QUANTIZATION_H_
