/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_LIBPRESSIO_MODES_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_LIBPRESSIO_MODES_H_

#if HSHM_ENABLE_COMPRESS && HSHM_ENABLE_LIBPRESSIO

#include <libpressio/libpressio.h>
#include <cstring>
#include <string>

#include "compress.h"

namespace hshm {

/**
 * Compression mode for lossy compressors.
 * Trades off compression speed vs quality.
 */
enum class CompressionMode {
  FAST,      // Fast compression, lower quality, better ratio
  BALANCED,  // Balanced speed and quality
  BEST       // Best quality, slower, lower ratio
};

/**
 * Enhanced LibPressio wrapper with compression mode support.
 *
 * This wrapper extends the basic LibPressio wrapper to support different
 * compression modes (fast/balanced/best) with appropriate configurations
 * for each lossy compressor.
 *
 * Compression modes:
 * - FAST: Aggressive lossy compression for maximum speed and ratio
 *   - ZFP: rate mode with low bitrate (8 bits)
 *   - SZ3: rel_error_bound = 1e-2 (1% error)
 *   - FPZIP: prec = 8 (8-bit precision, ~50% mantissa loss)
 *
 * - BALANCED: Moderate lossy compression balancing quality and ratio
 *   - ZFP: rate mode with medium bitrate (16 bits)
 *   - SZ3: rel_error_bound = 1e-3 (0.1% error)
 *   - FPZIP: prec = 16 (16-bit precision, ~33% mantissa loss)
 *
 * - BEST: High quality lossy compression
 *   - ZFP: accuracy mode with tolerance 1e-3
 *   - SZ3: rel_error_bound = 1e-4 (0.01% error)
 *   - FPZIP: prec = 20 (20-bit precision, ~17% mantissa loss)
 */
class LibPressioWithModes : public Compressor {
 private:
  struct pressio* library_;
  struct pressio_compressor* compressor_;
  std::string compressor_id_;
  CompressionMode mode_;

 public:
  /**
   * Constructor with compressor ID and compression mode.
   * @param compressor_id Name of the compressor (e.g., "zfp", "sz3", "fpzip")
   * @param mode Compression mode (FAST, BALANCED, or BEST)
   */
  LibPressioWithModes(const std::string& compressor_id, CompressionMode mode)
    : library_(nullptr), compressor_(nullptr), compressor_id_(compressor_id), mode_(mode) {
    library_ = pressio_instance();
    if (library_ != nullptr) {
      compressor_ = pressio_get_compressor(library_, compressor_id_.c_str());

      if (compressor_ != nullptr) {
        ConfigureCompressor();

        // DEBUG: Verify configuration was applied
        struct pressio_options* check_opts = pressio_compressor_get_options(compressor_);
        if (compressor_id == "sz3") {
          double rel_bound = 0.0;
          int error_mode = -1;
          if (pressio_options_get_double(check_opts, "sz3:rel_error_bound", &rel_bound) == pressio_options_key_set) {
            // Debug: configuration was read
          }
          if (pressio_options_get_integer(check_opts, "sz3:error_bound_mode", &error_mode) == pressio_options_key_set) {
            // Debug: error mode was read
          }
        }
        pressio_options_free(check_opts);
      }
    }
  }

  ~LibPressioWithModes() {
    if (library_ != nullptr) {
      pressio_release(library_);
    }
  }

  const char* GetCompressorId() const {
    return compressor_id_.c_str();
  }

  const char* GetModeName() const {
    switch (mode_) {
      case CompressionMode::FAST: return "fast";
      case CompressionMode::BALANCED: return "balanced";
      case CompressionMode::BEST: return "best";
      default: return "unknown";
    }
  }

 private:
  void ConfigureCompressor() {
    // Create NEW empty options instead of getting defaults
    // This avoids inheriting pressio:lossless=1 which breaks FPZIP lossy mode
    struct pressio_options* options = pressio_options_new();
    if (options == nullptr) return;

    if (compressor_id_ == "zfp") {
      ConfigureZFP(options);
    } else if (compressor_id_ == "sz3") {
      ConfigureSZ3(options);
    } else if (compressor_id_ == "fpzip") {
      ConfigureFPZIP(options);
    }

    pressio_compressor_set_options(compressor_, options);
    pressio_options_free(options);
  }

  void ConfigureZFP(struct pressio_options* options) {
    switch (mode_) {
      case CompressionMode::FAST:
        // Fixed-rate mode: 8 bits per value (aggressive compression)
        pressio_options_set_integer(options, "zfp:execution", 0);  // Serial execution
        pressio_options_set_double(options, "zfp:rate", 8.0);      // 8 bits/value
        pressio_options_set_integer(options, "zfp:mode", 1);       // Rate mode
        break;

      case CompressionMode::BALANCED:
        // Fixed-rate mode: 16 bits per value (balanced)
        pressio_options_set_integer(options, "zfp:execution", 0);
        pressio_options_set_double(options, "zfp:rate", 16.0);     // 16 bits/value
        pressio_options_set_integer(options, "zfp:mode", 1);       // Rate mode
        break;

      case CompressionMode::BEST:
        // Accuracy mode: error tolerance 1e-3 (high quality)
        pressio_options_set_integer(options, "zfp:execution", 0);
        pressio_options_set_double(options, "zfp:accuracy", 1e-3); // Tolerance
        pressio_options_set_integer(options, "zfp:mode", 3);       // Accuracy mode
        break;
    }
  }

  void ConfigureSZ3(struct pressio_options* options) {
    // SZ3 error bound modes:
    // 0 = ABS (absolute error)
    // 1 = REL (relative error)
    // 4 = PSNR

    // Use RELATIVE error mode for better lossy behavior across different data ranges
    pressio_options_set_integer(options, "sz3:error_bound_mode", 1);  // REL mode

    // Set both pressio generic and SZ3-specific options
    switch (mode_) {
      case CompressionMode::FAST:
        // Relative error 5% (aggressive lossy, better compression)
        pressio_options_set_double(options, "sz3:rel_error_bound", 0.05);
        pressio_options_set_double(options, "pressio:rel", 0.05);
        break;

      case CompressionMode::BALANCED:
        // Relative error 1% (moderate lossy)
        pressio_options_set_double(options, "sz3:rel_error_bound", 0.01);
        pressio_options_set_double(options, "pressio:rel", 0.01);
        break;

      case CompressionMode::BEST:
        // Relative error 0.5% (high quality lossy, still achieves good compression)
        pressio_options_set_double(options, "sz3:rel_error_bound", 0.005);
        pressio_options_set_double(options, "pressio:rel", 0.005);
        break;
    }
  }

  void ConfigureFPZIP(struct pressio_options* options) {
    // FPZIP uses precision mode (bits of mantissa to retain)
    // Float mantissa is 23 bits for IEEE 754 single precision
    // Setting prec < 23 should cause lossy compression

    // CRITICAL: For FPZIP to be lossy, prec must be LESS than the mantissa bits
    switch (mode_) {
      case CompressionMode::FAST:
        // Very lossy - keep only 12 bits of mantissa (drop 11 bits)
        pressio_options_set_integer(options, "fpzip:prec", 12);
        break;

      case CompressionMode::BALANCED:
        // Moderately lossy - keep 18 bits of mantissa (drop 5 bits)
        pressio_options_set_integer(options, "fpzip:prec", 18);
        break;

      case CompressionMode::BEST:
        // Slightly lossy - keep 21 bits of mantissa (drop 2 bits)
        pressio_options_set_integer(options, "fpzip:prec", 21);
        break;
    }
  }

 public:
  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (library_ == nullptr || compressor_ == nullptr) {
      return false;
    }

    // Detect if this is float data
    bool is_float_array = (input_size % sizeof(float) == 0);

    struct pressio_data* input_data = nullptr;
    if (is_float_array) {
      size_t num_floats = input_size / sizeof(float);
      size_t dims[1] = {num_floats};
      input_data = pressio_data_new_nonowning(
          pressio_float_dtype, input, 1, dims);
    } else {
      size_t dims[1] = {input_size};
      input_data = pressio_data_new_nonowning(
          pressio_uint8_dtype, input, 1, dims);
    }

    if (input_data == nullptr) {
      return false;
    }

    struct pressio_data* output_data = pressio_data_new_empty(
        pressio_uint8_dtype, 0, nullptr);
    if (output_data == nullptr) {
      pressio_data_free(input_data);
      return false;
    }

    int ret = pressio_compressor_compress(compressor_, input_data, output_data);
    pressio_data_free(input_data);

    if (ret != 0) {
      pressio_data_free(output_data);
      return false;
    }

    size_t compressed_bytes = pressio_data_get_bytes(output_data);
    if (compressed_bytes > output_size) {
      pressio_data_free(output_data);
      return false;
    }

    const void* compressed_ptr = pressio_data_ptr(output_data, nullptr);
    std::memcpy(output, compressed_ptr, compressed_bytes);
    output_size = compressed_bytes;

    pressio_data_free(output_data);
    return true;
  }

  bool Decompress(void *output, size_t &output_size, void *input,
                  size_t input_size) override {
    if (library_ == nullptr || compressor_ == nullptr) {
      return false;
    }

    size_t dims[1] = {input_size};
    struct pressio_data* input_data = pressio_data_new_nonowning(
        pressio_uint8_dtype, input, 1, dims);
    if (input_data == nullptr) {
      return false;
    }

    bool is_float_array = (output_size % sizeof(float) == 0);

    struct pressio_data* output_data = nullptr;
    if (is_float_array) {
      size_t num_floats = output_size / sizeof(float);
      size_t out_dims[1] = {num_floats};
      output_data = pressio_data_new_owning(
          pressio_float_dtype, 1, out_dims);
    } else {
      size_t out_dims[1] = {output_size};
      output_data = pressio_data_new_owning(
          pressio_uint8_dtype, 1, out_dims);
    }

    if (output_data == nullptr) {
      pressio_data_free(input_data);
      return false;
    }

    int ret = pressio_compressor_decompress(compressor_, input_data, output_data);
    pressio_data_free(input_data);

    if (ret != 0) {
      pressio_data_free(output_data);
      return false;
    }

    size_t decompressed_bytes = pressio_data_get_bytes(output_data);
    if (decompressed_bytes > output_size) {
      decompressed_bytes = output_size;
    }

    const void* decompressed_ptr = pressio_data_ptr(output_data, nullptr);
    if (decompressed_ptr != nullptr && decompressed_bytes > 0) {
      std::memcpy(output, decompressed_ptr, decompressed_bytes);
      output_size = decompressed_bytes;
    } else {
      pressio_data_free(output_data);
      return false;
    }

    pressio_data_free(output_data);
    return true;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS && HSHM_ENABLE_LIBPRESSIO

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_LIBPRESSIO_MODES_H_
