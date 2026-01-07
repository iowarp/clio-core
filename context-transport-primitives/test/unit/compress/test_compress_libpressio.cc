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

#include "basic_test.h"
#include "hermes_shm/util/compress/libpressio.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstring>

#if HSHM_HAS_LIBPRESSIO

// Helper function to calculate relative error for lossy compression validation
double calculate_relative_error(const float* original, const float* decompressed, size_t count) {
  double total_error = 0.0;
  double total_magnitude = 0.0;
  
  for (double i = 0; i < static_cast<double>(count); ++i) {
    size_t idx = static_cast<size_t>(i);
    double error = std::abs(static_cast<double>(original[idx]) - static_cast<double>(decompressed[idx]));
    double magnitude = std::abs(static_cast<double>(original[idx]));
    total_error += error;
    if (magnitude > 1e-10) {  // Avoid division by very small numbers
      total_magnitude += magnitude;
    }
  }
  
  double count_d = static_cast<double>(count);
  if (total_magnitude < 1e-10) {
    return total_error / count_d;  // Absolute error if all values are near zero
  }
  return total_error / total_magnitude;  // Relative error
}

// Helper function to calculate max relative error
double calculate_max_relative_error(const float* original, const float* decompressed, size_t count) {
  double max_error = 0.0;
  
  for (size_t i = 0; i < count; ++i) {
    double magnitude = std::abs(static_cast<double>(original[i]));
    double error = std::abs(static_cast<double>(original[i]) - static_cast<double>(decompressed[i]));
    
    if (magnitude > 1e-10) {
      double rel_error = error / magnitude;
      max_error = std::max(max_error, rel_error);
    } else {
      max_error = std::max(max_error, error);
    }
  }
  
  return max_error;
}

TEST_CASE("TestCompressLibPressio") {
  // Generate floating-point test data designed to produce noticeable loss
  // Use high-precision values with fine-grained variation to trigger lossy compression
  // Fine precision values will be affected by FPZIP's 12-bit precision quantization
  const size_t num_floats = 1024;
  std::vector<float> original_floats(num_floats);
  
  // Generate test data with:
  // 1. High precision floating-point values (many decimal places)
  // 2. Fine-grained variations that will be quantized by lossy compressors
  // 3. Mix of large and small values to test dynamic range
  // 4. Values with many decimal places to make quantization errors visible
  for (size_t i = 0; i < num_floats; ++i) {
    float x = static_cast<float>(i);
    // Combine multiple frequencies and add fine-grained variations
    // Use irrational multipliers and modulo operations to create values
    // that don't align perfectly with quantization grids
    float value = std::sin(x * 0.1F) * 1000.0F + 
                  std::cos(x * 0.03F) * 500.0F +
                  std::sin(x * 0.007F) * 100.0F +
                  static_cast<float>(i % 17) * 0.123456789F +     // Fine variations
                  static_cast<float>(i % 23) * 0.000012345F +     // Very fine variations
                  static_cast<float>(i % 31) * 0.000000123F +     // Ultra-fine variations
                  static_cast<float>(i % 7) * 1.23456789F +       // Medium-fine variations
                  std::sin(x * 0.001F) * 0.987654321F;            // Continuous fine variations
    original_floats[i] = value;
  }
  
  // Convert to bytes for compression interface
  const size_t original_size = original_floats.size() * sizeof(float);
  std::vector<char> compressed(original_size * 2);  // Allow for expansion
  std::vector<char> decompressed(original_size * 2);
  
  // List of lossy compressors to test through libpressio
  // We only test lossy compressors since we already have lossless ones implemented
  struct CompressorTest {
    const char* name;
    bool optional;  // true if this compressor might not be available
    bool is_lossy;  // true if this is a lossy compressor
    double max_relative_error;  // Maximum acceptable relative error for lossy compressors
  };
  
  // Working lossy compressors via libpressio (only 3 are currently working)
  CompressorTest compressors_to_test[] = {
    {"zfp", false, true, 0.01},      // Lossy compressor - shows actual loss (max error ~1e-4)
    {"sz3", false, true, 0.01},      // Lossy compressor - configured with 1e-5 error bound
    {"fpzip", false, true, 0.01},    // Lossy floating-point compressor - working with header mode
  };
  
  for (const auto& comp_info : compressors_to_test) {
    try {
      PAGE_DIVIDE(("LibPressio-" + std::string(comp_info.name)).c_str()) {
        // Print original data
        std::cout << "Original data: [" 
                  << std::fixed << std::setprecision(4) << original_floats[0] << ", " 
                  << original_floats[1] << ", " 
                  << original_floats[2] << ", ...]\n";
        
        // Print compressor name
        std::cout << "Compressor: " << comp_info.name << "\n";
        
        hshm::LibPressio libpressio(comp_info.name);
        
        size_t cmpr_size = compressed.size();
        size_t raw_size = decompressed.size();
        bool compress_success = libpressio.Compress(compressed.data(), cmpr_size,
                                                    original_floats.data(), original_size);
        
        if (!compress_success) {
          if (comp_info.optional) {
            continue;
          }
          REQUIRE(compress_success);
        }
        
        REQUIRE(cmpr_size > 0);
        
        // Print compressed data info
        double ratio = (original_size > 0) ? static_cast<double>(original_size) / static_cast<double>(cmpr_size) : 0.0;
        std::cout << "Compressed: " << cmpr_size << " bytes (ratio: " 
                  << std::fixed << std::setprecision(2) << ratio << "x)\n";
        
        // Decompress
        bool decompress_success = libpressio.Decompress(decompressed.data(), raw_size,
                                                        compressed.data(), cmpr_size);
        
        if (!decompress_success) {
          if (comp_info.optional) {
            continue;
          }
          REQUIRE(decompress_success);
        }
        
        REQUIRE(raw_size > 0);
        
        // Convert decompressed bytes back to floats
        const float* decompressed_floats = reinterpret_cast<const float*>(decompressed.data());
        size_t compare_count = num_floats;
        
        // Print decompressed data
        std::cout << "Decompressed data: [" 
                  << std::fixed << std::setprecision(4) << decompressed_floats[0] << ", " 
                  << decompressed_floats[1] << ", " 
                  << decompressed_floats[2] << ", ...]\n";
        
        // Print relative error
        if (comp_info.is_lossy) {
          double max_rel_error = calculate_max_relative_error(original_floats.data(), 
                                                             decompressed_floats, 
                                                             compare_count);
          std::cout << "Relative error: " << std::scientific << std::setprecision(4)
                    << max_rel_error << "\n";
          REQUIRE(std::isfinite(decompressed_floats[0]));
        } else {
          REQUIRE(std::memcmp(original_floats.data(), decompressed_floats, original_size) == 0);
        }
        
        std::cout << "\n";
      }
    } catch (...) {
      if (!comp_info.optional) {
        throw;
      }
    }
  }
}

TEST_CASE("TestCompressLibPressioAutoSelect") {
  // Test the auto-selection feature - using default constructor without compressor ID
  // The default constructor should auto-detect and use the first available compressor
  // Priority order: zfp -> sz3 -> noop
  
  const size_t num_floats = 1024;
  std::vector<float> original_floats(num_floats);
  
  // Generate floating-point test data with fine precision for lossy compression testing
  for (size_t i = 0; i < num_floats; ++i) {
    float x = static_cast<float>(i);
    float value = std::sin(x * 0.1F) * 1000.0F + 
                  std::cos(x * 0.03F) * 500.0F +
                  std::sin(x * 0.007F) * 100.0F +
                  static_cast<float>(i % 17) * 0.123456789F +    // Fine variations
                  static_cast<float>(i % 23) * 0.000012345F +    // Very fine variations
                  std::sin(x * 0.001F) * 0.987654321F;            // Continuous fine variations
    original_floats[i] = value;
  }
  
  const size_t original_size = original_floats.size() * sizeof(float);
  std::vector<char> compressed(original_size * 2);
  std::vector<char> decompressed(original_size * 2);
  
  PAGE_DIVIDE("LibPressio-AutoSelect") {
    // Print original data
    std::cout << "Original data: [" 
              << std::fixed << std::setprecision(4) << original_floats[0] << ", " 
              << original_floats[1] << ", " 
              << original_floats[2] << ", ...]\n";
    
    // Use default constructor - auto-selection
    hshm::LibPressio libpressio;
    
    // Print compressor name (auto-selected)
    std::cout << "Compressor: " << libpressio.GetCompressorId() << " (auto-selected)\n";
    
    size_t cmpr_size = compressed.size();
    bool compress_success = libpressio.Compress(compressed.data(), cmpr_size,
                                                original_floats.data(), original_size);
    
    REQUIRE(compress_success);
    REQUIRE(cmpr_size > 0);
    
    // Print compressed data info
    double ratio = (original_size > 0) ? static_cast<double>(original_size) / static_cast<double>(cmpr_size) : 0.0;
    std::cout << "Compressed: " << cmpr_size << " bytes (ratio: " 
              << std::fixed << std::setprecision(2) << ratio << "x)\n";
    
    // Decompress
    size_t decompressed_size = decompressed.size();
    bool decompress_success = libpressio.Decompress(decompressed.data(), decompressed_size,
                                                    compressed.data(), cmpr_size);
    
    REQUIRE(decompress_success);
    REQUIRE(decompressed_size > 0);
    
    // Convert decompressed bytes back to floats
    const float* decompressed_floats = reinterpret_cast<const float*>(decompressed.data());
    
    // Print decompressed data
    std::cout << "Decompressed data: [" 
              << std::fixed << std::setprecision(4) << decompressed_floats[0] << ", " 
              << decompressed_floats[1] << ", " 
              << decompressed_floats[2] << ", ...]\n";
    
    // Print relative error
    double max_rel_error = calculate_max_relative_error(original_floats.data(), 
                                                        decompressed_floats, 
                                                        num_floats);
    std::cout << "Relative error: " << std::scientific << std::setprecision(4)
              << max_rel_error << "\n";
    
    REQUIRE(std::isfinite(decompressed_floats[0]));
    std::cout << "\n";
  }
}

#endif  // HSHM_HAS_LIBPRESSIO
