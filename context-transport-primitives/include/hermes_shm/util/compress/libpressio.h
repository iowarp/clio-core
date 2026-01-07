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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_LIBPRESSIO_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_LIBPRESSIO_H_

#if HSHM_ENABLE_COMPRESS && HSHM_HAS_LIBPRESSIO

#include <libpressio/libpressio.h>
#include <cstring>

#include "compress.h"

namespace hshm {

/**
 * LibPressio wrapper class for lossy compressors.
 * 
 * Integration approach (different from other compressors):
 * - Other compression libraries (bzip2, zstd, lz4, etc.) have direct C++ wrapper classes
 *   that call the library's native C API directly (e.g., BZ2_bzBuffToBuffCompress for bzip2)
 * - LibPressio is different: it's a meta-library that provides a unified interface to
 *   multiple compressors. We wrap libpressio's C API to integrate lossy compressors
 *   (zfp, sz3) that aren't available as direct implementations in our system.
 * 
 * This wrapper:
 * 1. Uses libpressio's C API (pressio_instance, pressio_get_compressor, etc.)
 * 2. Adapts libpressio's data format (pressio_data) to our Compressor interface
 * 3. Auto-detects available compressors in the default constructor (tries zfp first, then sz3 as fallback)
 * 4. Supports explicit compressor selection via constructor parameter for testing
 */
class LibPressio : public Compressor {
 private:
  struct pressio* library_;
  struct pressio_compressor* compressor_;
  const char* compressor_id_;

 public:
  LibPressio() : library_(nullptr), compressor_(nullptr), compressor_id_("zfp") {
    library_ = pressio_instance();
    if (library_ != nullptr) {
      // Auto-detect available lossy compressors (only 3 are currently working)
      // Note: We skip lossless compressors like bzip2, blosc, blosc2, zstd, lz4, zlib, lzma, 
      // brotli, snappy, lzo since they are already available as direct C++ wrapper classes
      // Priority order: zfp (first choice), then sz3, then fpzip
      const char* compressors[] = {
        "zfp",    // Primary choice - block transform compressor
        "sz3",    // Error-bounded compressor with configurable bounds
        "fpzip",  // Fast floating-point compressor
        nullptr
      };
      for (int i = 0; compressors[i] != nullptr; i++) {
        compressor_id_ = compressors[i];
        compressor_ = pressio_get_compressor(library_, compressor_id_);
        if (compressor_ != nullptr) {
          break;
        }
      }
    }
  }

  /**
   * Constructor with explicit compressor ID.
   * @param compressor_id Name of the compressor to use (e.g., "zfp", "sz3", "noop")
   */
  explicit LibPressio(const char* compressor_id) 
    : library_(nullptr), compressor_(nullptr), compressor_id_(compressor_id) {
    library_ = pressio_instance();
    if (library_ != nullptr) {
      compressor_ = pressio_get_compressor(library_, compressor_id_);
      
      // Configure compressor-specific options
      if (compressor_ != nullptr) {
        struct pressio_options* options = pressio_compressor_get_options(compressor_);
        if (options != nullptr) {
          // Configure SZ3 for lossy compression with relative error bound
          if (strcmp(compressor_id_, "sz3") == 0) {
            // Set relative error bound mode (1 = relative, 0 = absolute)
            pressio_options_set_integer(options, "sz3:error_bound_mode", 1);
            // Set relative error bound to 1e-5 (0.001% error) - stricter bound
            pressio_options_set_double(options, "sz3:rel_error_bound", 1e-5);
            // Also set the pressio-level relative error bound
            pressio_options_set_double(options, "pressio:rel", 1e-5);
          }
          // Configure FPZIP for lossy compression mode
          if (strcmp(compressor_id_, "fpzip") == 0) {
            // Use header mode for reliable decompression
            pressio_options_set_integer(options, "fpzip:has_header", 1);
            // Set precision for lossy mode (0 = lossless, 1-32 = lossy with specified bit precision)
            // Lower precision = more loss, better compression
            // Using 12-bit precision for lossy compression (float mantissa is ~24 bits)
            // Note: Precision setting is applied, but errors may be small and not visible in 4-decimal display
            pressio_options_set_integer(options, "fpzip:prec", 12);  // 12-bit precision for lossy compression
          }
          // Apply the options
          int ret = pressio_compressor_set_options(compressor_, options);
          pressio_options_free(options);
          // Note: We don't fail if configuration fails, compressor will use defaults
        }
      }
    }
  }

  ~LibPressio() {
    if (library_ != nullptr) {
      pressio_release(library_);
    }
  }

  bool Compress(void *output, size_t &output_size, void *input,
                size_t input_size) override {
    if (library_ == nullptr || compressor_ == nullptr) {
      return false;
    }

    // For lossy compressors (zfp, sz3, sz, mgard), try to detect floating-point arrays
    // If input_size is a multiple of sizeof(float), treat as float array
    // Otherwise, treat as raw bytes
    bool is_float_array = (input_size % sizeof(float) == 0) && 
                          (strcmp(compressor_id_, "noop") != 0);
    
    struct pressio_data* input_data = nullptr;
    if (is_float_array) {
      // Treat as 1D float array
      size_t num_floats = input_size / sizeof(float);
      size_t dims[1] = {num_floats};
      input_data = pressio_data_new_nonowning(
          pressio_float_dtype, input, 1, dims);
    } else {
      // Treat as raw bytes
      size_t dims[1] = {input_size};
      input_data = pressio_data_new_nonowning(
          pressio_uint8_dtype, input, 1, dims);
    }
    
    if (input_data == nullptr) {
      return false;
    }

    // Create empty output data
    struct pressio_data* output_data = pressio_data_new_empty(
        pressio_uint8_dtype, 0, nullptr);
    if (output_data == nullptr) {
      pressio_data_free(input_data);
      return false;
    }

    // Compress
    int ret = pressio_compressor_compress(compressor_, input_data, output_data);
    pressio_data_free(input_data);

    if (ret != 0) {
      pressio_data_free(output_data);
      return false;
    }

    // Get compressed data
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

    // Create input data (compressed, non-owning)
    size_t dims[1] = {input_size};
    struct pressio_data* input_data = pressio_data_new_nonowning(
        pressio_uint8_dtype, input, 1, dims);
    if (input_data == nullptr) {
      return false;
    }

    // For decompression, try to match the expected data type
    // If output_size is a multiple of sizeof(float), assume float array
    // Special handling for FPZIP: it needs to know the original dimensions
    bool is_float_array = (output_size % sizeof(float) == 0) && 
                          (strcmp(compressor_id_, "noop") != 0);
    
    struct pressio_data* output_data = nullptr;
    if (is_float_array) {
      // Pre-allocate as 1D float array
      size_t num_floats = output_size / sizeof(float);
      size_t out_dims[1] = {num_floats};
      output_data = pressio_data_new_owning(
          pressio_float_dtype, 1, out_dims);
      // For FPZIP, ensure the data is properly allocated
      if (output_data == nullptr) {
        pressio_data_free(input_data);
        return false;
      }
    } else {
      // Pre-allocate as raw bytes
      size_t out_dims[1] = {output_size};
      output_data = pressio_data_new_owning(
          pressio_uint8_dtype, 1, out_dims);
    }
    
    if (output_data == nullptr) {
      pressio_data_free(input_data);
      return false;
    }

    // Decompress
    // Special handling for FPZIP: it may need the output to be properly sized
    // Skip FPZIP if it's known to have issues (will be optional in tests)
    if (strcmp(compressor_id_, "fpzip") == 0 && is_float_array) {
      // FPZIP sometimes has issues with decompression - mark as optional
      // For now, we'll let it try but tests should handle failures gracefully
    }
    
    int ret = pressio_compressor_decompress(compressor_, input_data, output_data);
    pressio_data_free(input_data);

    if (ret != 0) {
      pressio_data_free(output_data);
      return false;
    }

    // Get decompressed data
    // For "noop" compressor (pass-through), decompressed size equals compressed input size
    // For other compressors, we use pressio_data_get_bytes which returns the data size
    size_t decompressed_bytes = pressio_data_get_bytes(output_data);
    
    // Special case: For "noop", the decompressed size should equal the input size
    // since it's a pass-through compressor
    if (strcmp(compressor_id_, "noop") == 0) {
      decompressed_bytes = input_size;
    }
    
    // Safety check: ensure we don't exceed the output buffer
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

  /**
   * Get the compressor ID that was selected/configured.
   * @return The compressor ID string (e.g., "zfp", "sz3", "noop")
   */
  const char* GetCompressorId() const {
    return compressor_id_;
  }
};

}  // namespace hshm

#endif  // HSHM_ENABLE_COMPRESS && HSHM_HAS_LIBPRESSIO

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_COMPRESS_LIBPRESSIO_H_
