/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
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

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/preprocess/byte_shuffle.h"
#include "clio_ctp/compress/preprocess/feature_extractor.h"

using namespace ctp;
using namespace ctp::compress::preprocess;
using namespace ctp::compress::model;

/**
 * Synthetic data generator for float32 data spanning multiple distributions.
 */
class SyntheticDataGenerator {
 public:
  /**
   * Initialize generator with seed.
   */
  explicit SyntheticDataGenerator(uint32_t seed) : rng_(seed) {}

  /**
   * Generate a chunk of synthetic data with the given distribution.
   * Returns float32 data and the distribution name.
   */
  std::pair<std::vector<float>, std::string> GenerateChunk(
      const std::string& distribution, size_t num_elements, int chunk_index) {
    std::vector<float> data(num_elements);
    // Create a local RNG seeded from the generator and chunk index.
    std::mt19937 local_rng(rng_() ^ static_cast<uint32_t>(chunk_index));

    if (distribution == "uniform") {
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (auto& v : data) v = dist(local_rng);
    } else if (distribution == "gaussian") {
      std::normal_distribution<float> dist(0.5f, 0.1f);
      for (auto& v : data) v = dist(local_rng);
    } else if (distribution == "exponential") {
      std::exponential_distribution<float> dist(2.0f);
      for (auto& v : data) v = dist(local_rng);
    } else if (distribution == "constant") {
      float val = 0.5f;
      for (auto& v : data) v = val;
    } else if (distribution == "sinusoidal") {
      for (size_t i = 0; i < num_elements; ++i) {
        data[i] = 0.5f + 0.5f * std::sin(2.0f * M_PI * i / num_elements);
      }
    } else if (distribution == "bimodal") {
      std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
      for (auto& v : data) {
        float u = uniform(local_rng);
        v = (u < 0.5f) ? 0.2f : 0.8f;
      }
    } else {
      // Default to uniform
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (auto& v : data) v = dist(local_rng);
    }

    return {data, distribution};
  }

 private:
  std::mt19937 rng_;
};

/**
 * Run a single compression trial with optional preprocessing.
 * Returns CSV row as string, or empty string if compression failed.
 */
static std::string RunCompressionTrial(
    const std::string& library_name,
    CompressionPreset preset,
    const std::string& distribution,
    const std::vector<float>& data,
    int quantize,
    int byte_shuffle,
    double error_bound) {
  // Extract features from original data.
  auto features = FeatureExtractor::ExtractFeatures(
      data.data(), data.size() * sizeof(float), DataKind::kFloat);

  // Get library ID and parse base_id, preset_id.
  int lib_id = CompressionFactory::GetLibraryId(library_name, preset);
  if (lib_id == 0) {
    return "";  // Unknown library
  }
  int base_id = lib_id / 10;
  int preset_id = lib_id % 10;

  // Set library config in features.
  FeatureExtractor::SetLibraryConfig(features, base_id, preset_id);

  // Create compressor.
  auto compressor = CompressionFactory::GetPreset(library_name, preset);
  if (!compressor) {
    return "";  // Backend disabled or not available
  }

  // Apply preprocessing if requested.
  std::vector<uint8_t> to_compress;
  std::vector<uint8_t> shuffled_copy;
  size_t input_size = data.size() * sizeof(float);
  const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(data.data());

  if (byte_shuffle) {
    shuffled_copy = ByteShuffleVector(data_ptr, input_size, sizeof(float));
    if (shuffled_copy.empty()) {
      return "";  // Shuffle failed
    }
    to_compress = shuffled_copy;
  } else {
    to_compress.assign(data_ptr, data_ptr + input_size);
  }

  // Allocate output buffer (2x input + 1KB safety margin).
  size_t output_buffer_size = 2 * input_size + 1024;
  std::vector<uint8_t> output_buffer(output_buffer_size);
  std::vector<uint8_t> decompressed(input_size);

  // Measure compression time.
  auto compress_start = std::chrono::high_resolution_clock::now();
  size_t compressed_size = output_buffer_size;
  bool compress_ok = compressor->Compress(
      output_buffer.data(), compressed_size,
      to_compress.data(), input_size);
  auto compress_end = std::chrono::high_resolution_clock::now();
  double compress_time_ms =
      std::chrono::duration<double, std::milli>(compress_end - compress_start)
          .count();

  if (!compress_ok || compressed_size == 0) {
    // Compression failed.
    return "";
  }

  // Measure decompression time.
  size_t decompressed_size = input_size;
  auto decompress_start = std::chrono::high_resolution_clock::now();
  bool decompress_ok =
      compressor->Decompress(decompressed.data(), decompressed_size,
                             output_buffer.data(), compressed_size);
  auto decompress_end = std::chrono::high_resolution_clock::now();
  double decompress_time_ms =
      std::chrono::duration<double, std::milli>(decompress_end -
                                                decompress_start)
          .count();

  if (!decompress_ok || decompressed_size != input_size) {
    return "";
  }

  // If byte-shuffled, unshuffle to validate round-trip.
  int success = 1;
  if (byte_shuffle) {
    std::vector<uint8_t> unshuffled = ByteUnshuffleVector(
        decompressed.data(), input_size, sizeof(float));
    if (unshuffled.empty()) {
      return "";  // Unshuffle failed
    }
    // Verify round-trip matches original
    if (std::memcmp(unshuffled.data(), data_ptr, input_size) != 0) {
      success = 0;  // Mismatch in round-trip
    }
  } else {
    // For non-shuffled, verify decompressed matches original
    if (std::memcmp(decompressed.data(), data_ptr, input_size) != 0) {
      success = 0;
    }
  }

  // Compute compression ratio.
  double compression_ratio =
      (compressed_size > 0) ? static_cast<double>(input_size) / compressed_size
                            : 1.0;

  // Get preset name.
  std::string preset_str = CompressionFactory::GetPresetName(preset);

  // Emit CSV row: library,preset,distribution,data_type,chunk_size_bytes,
  // shannon_entropy,mad,second_derivative_mean,library_config_id,
  // config_fast,config_balanced,config_best,data_type_char,data_type_float,
  // quantize,byte_shuffle,error_bound,original_bytes,compressed_bytes,
  // compression_ratio,compression_time_ms,decompression_time_ms,psnr_db,success
  std::string row;
  row += library_name + ",";
  row += preset_str + ",";
  row += distribution + ",";
  row += "float,";
  row += std::to_string(static_cast<int>(features.chunk_size_bytes)) + ",";
  row += std::to_string(features.shannon_entropy) + ",";
  row += std::to_string(features.mad) + ",";
  row += std::to_string(features.second_derivative_mean) + ",";
  row += std::to_string(static_cast<int>(features.library_config_id)) + ",";
  row += std::to_string(features.config_fast) + ",";
  row += std::to_string(features.config_balanced) + ",";
  row += std::to_string(features.config_best) + ",";
  row += std::to_string(features.data_type_char) + ",";
  row += std::to_string(features.data_type_float) + ",";
  row += std::to_string(quantize) + ",";
  row += std::to_string(byte_shuffle) + ",";
  row += std::to_string(error_bound) + ",";
  row += std::to_string(input_size) + ",";
  row += std::to_string(compressed_size) + ",";
  row += std::to_string(compression_ratio) + ",";
  row += std::to_string(compress_time_ms) + ",";
  row += std::to_string(decompress_time_ms) + ",";
  row += "0.0,";  // psnr_db (0 for lossless)
  row += std::to_string(success);

  return row;
}

/**
 * Print usage message.
 */
static void PrintUsage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " [--rows N] [--chunk-bytes B] [--out FILE] [--seed S] [--preprocessors]\n";
  std::cerr << "  --rows N:         Number of chunks per compressor/preset "
               "(default 50)\n";
  std::cerr << "  --chunk-bytes B:  Chunk size in bytes (default 1048576)\n";
  std::cerr << "  --out FILE:       Output CSV file (default stdout)\n";
  std::cerr << "  --seed S:         Random seed (default 42)\n";
  std::cerr << "  --preprocessors:  Sweep preprocessor variants (byte-shuffle)\n";
}

int main(int argc, char** argv) {
  int rows = 50;
  size_t chunk_bytes = 1 << 20;  // 1 MB
  std::string output_file = "";
  uint32_t seed = 42;
  bool preprocessors = false;

  // Parse command-line arguments.
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--rows" && i + 1 < argc) {
      rows = std::stoi(argv[++i]);
    } else if (arg == "--chunk-bytes" && i + 1 < argc) {
      chunk_bytes = std::stoul(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      output_file = argv[++i];
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoul(argv[++i]);
    } else if (arg == "--preprocessors") {
      preprocessors = true;
    } else {
      PrintUsage(argv[0]);
      return 1;
    }
  }

  // Open output stream.
  std::ofstream out_file;
  std::ostream* out = &std::cout;
  if (!output_file.empty()) {
    out_file.open(output_file);
    if (!out_file) {
      std::cerr << "Error: Cannot open output file " << output_file << "\n";
      return 1;
    }
    out = &out_file;
  }

  // Compute number of float elements per chunk.
  size_t num_elements = chunk_bytes / sizeof(float);

  // Initialize synthetic data generator.
  SyntheticDataGenerator generator(seed);

  // Emit CSV header.
  *out << "library,preset,distribution,data_type,chunk_size_bytes,"
       << "shannon_entropy,mad,second_derivative_mean,library_config_id,"
       << "config_fast,config_balanced,config_best,data_type_char,"
       << "data_type_float,quantize,byte_shuffle,error_bound,original_bytes,"
       << "compressed_bytes,compression_ratio,compression_time_ms,"
       << "decompression_time_ms,psnr_db,success\n";

  // List of compressor names to test.
  std::vector<std::string> compressor_names = {
      "zstd",    "lz4",     "zlib",    "brotli",   "bzip2",   "lzma",
      "snappy",  "blosc2",  "nvcomp-lz4",   "nvcomp-snappy", "nvcomp-zstd",
      "zfp",     "sz3",     "fpzip",   "zfp-sycl", "cusz",    "cuszp",
      "ndzip"};

  // List of distributions.
  std::vector<std::string> distributions = {
      "uniform", "gaussian", "exponential", "constant", "sinusoidal", "bimodal"};

  // List of presets.
  CompressionPreset presets[] = {
      CompressionPreset::FAST,
      CompressionPreset::BALANCED,
      CompressionPreset::BEST,
  };

  // Main benchmark loop.
  int chunk_index = 0;
  for (const auto& compressor_name : compressor_names) {
    for (const auto& preset : presets) {
      for (const auto& distribution : distributions) {
        for (int row = 0; row < rows; ++row) {
          // Generate synthetic chunk.
          auto [chunk_data, dist_name] =
              generator.GenerateChunk(distribution, num_elements, chunk_index++);

          // Emit base row (no preprocessors).
          std::string csv_row = RunCompressionTrial(
              compressor_name, preset, dist_name, chunk_data, 0, 0, 0.0);
          if (!csv_row.empty()) {
            *out << csv_row << "\n";
          }

          // If preprocessors flag, emit byte-shuffle variant.
          if (preprocessors) {
            csv_row = RunCompressionTrial(
                compressor_name, preset, dist_name, chunk_data, 0, 1, 0.0);
            if (!csv_row.empty()) {
              *out << csv_row << "\n";
            }
          }
        }
      }
    }
  }

  if (out_file) {
    out_file.close();
  }

  return 0;
}
