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

/* Quantize<T>() / Dequantize<T>() -- the CPU implementations -- were REMOVED,
 * for the reason given in byte_shuffle.h: NeuroPress preprocessing is
 * CUDA-only, and a working CPU mirror turned "the GPU path" into something a
 * host-resident chunk silently opted out of.
 *
 * Use QuantizeDevice() / DequantizeDevice() below. QuantizationResult is kept
 * because it still describes what those write into the blob header.
 */




/**
 * @brief Why QuantizeDevice() declined to quantize a chunk.
 *
 * A refusal is a routine outcome, not an error: the caller stores the chunk
 * losslessly, which honours any bound at zero error. It still has to be
 * reportable, because otherwise the single symptom is a compression ratio
 * that quietly got worse, with nothing saying the bound was the cause.
 */
enum class QuantizeRefusal : int {
  kNone = 0,           /**< Quantized; the requested bound holds */
  kNonFiniteRange,     /**< min/max not finite, so no grid is definable */
  kStepNotPositive,    /**< No positive step exists (defensive; eb > 0 is
                            checked on entry, so this should be unreachable) */
  kIndexExceedsInt32,  /**< The bound needs a finer grid than int32 indexes */
  kIndexLeftWidth,     /**< An index left the chosen width -- a planner bug,
                            impossible by construction, reported not clamped */
  kElementMissedBound, /**< An element missed the bound through the decoder
                            this build actually ships */
  kDeviceError,        /**< CUDA allocation, launch or copy failed. A property
                            of the machine, not of the data or the bound */
  kNoCudaSupport,      /**< Built without CUDA; there is no device path */
};

/** @brief Human-readable form of a QuantizeRefusal, for logs. */
inline const char *QuantizeRefusalName(QuantizeRefusal reason) {
  switch (reason) {
    case QuantizeRefusal::kNone:
      return "none";
    case QuantizeRefusal::kNonFiniteRange:
      return "data range is not finite";
    case QuantizeRefusal::kStepNotPositive:
      return "no positive quantization step exists for this bound";
    case QuantizeRefusal::kIndexExceedsInt32:
      return "bound needs a finer grid than int32 can index";
    case QuantizeRefusal::kIndexLeftWidth:
      return "an index left the chosen width (planner bug)";
    case QuantizeRefusal::kElementMissedBound:
      return "an element missed the bound through its own decoder";
    case QuantizeRefusal::kDeviceError:
      return "CUDA error";
    case QuantizeRefusal::kNoCudaSupport:
      return "built without CUDA support";
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
  double data_min = 0.0;         /**< Minimum of the original data */
  double data_max = 0.0;         /**< Maximum of the original data */
  int precision = 0;             /**< 8, 16 or 32 bits per value */
  bool bound_achievable = true;  /**< Always true on success: QuantizeDevice
                                      no longer has a state in which it
                                      quantizes against a substituted bound */
  QuantizeRefusal refusal = QuantizeRefusal::kNone; /**< Set on every false
                                      return, so a caller can log WHY a chunk
                                      fell back to lossless */
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
 * @param stream      cudaStream_t to launch on, as an opaque pointer, or
 *                    nullptr for the shared per-thread stream. Supplying one
 *                    keeps this slot's work off every other slot's stream, so
 *                    a sweep quantizing K candidates overlaps them the way
 *                    upstream's quantize_simple(..., s.stream) does.
 *
 *                    Unlike ByteShuffleDevice, this still synchronizes the
 *                    given stream once, internally: the packed width is chosen
 *                    from the data's min/max, so those have to reach the host
 *                    before the quantize kernel can be launched or *out_bytes
 *                    computed. The wait is scoped to `stream`, so it blocks
 *                    only this candidate -- slots already launched keep
 *                    running.
 * @return true only if EVERY element round-trips through DequantizeDevice
 *         within error_bound -- the kernel checks each one rather than
 *         trusting the step arithmetic. False if the bound cannot be honored,
 *         if CUDA failed, or if CUDA is not compiled in; in every false case
 *         out_params->refusal says which, and the caller is expected to store
 *         the chunk losslessly, honouring the bound at zero error.
 */
bool QuantizeDevice(const void *device_in, size_t num_elements,
                    double error_bound, void *device_out, size_t *out_bytes,
                    DeviceQuantizeParams *out_params, void *stream = nullptr);

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
