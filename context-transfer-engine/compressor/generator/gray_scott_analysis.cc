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

/**
 * Gray-Scott ADIOS2 Real Data Compression Analysis
 *
 * This tool reads real Gray-Scott reaction-diffusion simulation data from ADIOS2
 * BP5 format and performs comprehensive compression benchmarking on each chunk.
 *
 * Features:
 * - Reads ADIOS2 BP5 output (U and V fields per timestep)
 * - Tests ALL available compression libraries with multiple presets
 * - Measures: compression_ratio, compress_time, decompress_time, PSNR
 * - Calculates: shannon_entropy, MAD, second_derivative using data_stats.h
 * - Outputs CSV for ML training and analysis
 *
 * Output CSV Format:
 *   library,preset,data_size,shannon_entropy,mad,second_derivative,
 *   compress_time,compression_ratio,decompress_time,psnr
 *
 * Usage:
 *   gray_scott_analysis [input_bp_path] [output_csv_path]
 *
 * Defaults:
 *   input:  /workspace/context-transfer-engine/compressor/results/real_data/gs-output.bp/
 *   output: /workspace/context-transfer-engine/compressor/results/real_data/grayscott_compression_analysis.csv
 */

#include <adios2.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <memory>

#include <hermes_shm/util/logging.h>

// Compression headers (from hermes_shm)
#include "hermes_shm/compress/compress.h"
#include "hermes_shm/compress/lossless_modes.h"
#include "hermes_shm/compress/blosc.h"
#include "hermes_shm/compress/snappy.h"

#if HSHM_ENABLE_COMPRESS && HSHM_ENABLE_LIBPRESSIO
#include "hermes_shm/compress/libpressio_modes.h"
#endif

// Data statistics
#include "wrp_cte/compressor/models/data_stats.h"

// Library ID mapping (consistent with training data generator)
enum class CompressionLibrary : int {
  ZSTD_FAST = 0,
  ZSTD_BALANCED = 1,
  ZSTD_BEST = 2,
  LZ4_FAST = 3,
  LZ4_BALANCED = 4,
  LZ4_BEST = 5,
  BLOSC2 = 6,
  ZLIB_FAST = 7,
  ZLIB_BALANCED = 8,
  ZLIB_BEST = 9,
  BZIP2_FAST = 10,
  BZIP2_BALANCED = 11,
  BZIP2_BEST = 12,
  LZMA_FAST = 13,
  LZMA_BALANCED = 14,
  LZMA_BEST = 15,
  BROTLI_FAST = 16,
  BROTLI_BALANCED = 17,
  BROTLI_BEST = 18,
  SNAPPY = 19,
  // Lossy compressors (for float data only)
  ZFP_FAST = 20,
  ZFP_BALANCED = 21,
  ZFP_BEST = 22,
  SZ3_FAST = 23,
  SZ3_BALANCED = 24,
  SZ3_BEST = 25,
  FPZIP_FAST = 26,
  FPZIP_BALANCED = 27,
  FPZIP_BEST = 28
};

struct CompressionResult {
  std::string library;
  std::string preset;
  size_t data_size;
  double shannon_entropy;
  double mad;
  double second_derivative;
  double compress_time_ms;
  double compression_ratio;
  double decompress_time_ms;
  double psnr;
  bool success;
};

/**
 * Get CPU time in nanoseconds for accurate timing
 */
inline double GetCpuTimeNs() {
#ifdef __linux__
  struct timespec ts;
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == 0) {
    return static_cast<double>(ts.tv_sec) * 1e9 + static_cast<double>(ts.tv_nsec);
  }
#endif
  return 0.0;
}

/**
 * Benchmark a single compressor on the given data
 */
CompressionResult BenchmarkCompressor(
    hshm::Compressor* compressor,
    const std::string& library,
    const std::string& preset,
    const std::vector<double>& data,
    size_t data_size_bytes) {

  CompressionResult result;
  result.library = library;
  result.preset = preset;
  result.data_size = data_size_bytes;
  result.shannon_entropy = 0.0;
  result.mad = 0.0;
  result.second_derivative = 0.0;
  result.compress_time_ms = 0.0;
  result.compression_ratio = 0.0;
  result.decompress_time_ms = 0.0;
  result.psnr = 0.0;
  result.success = false;

  size_t num_elements = data.size();
  if (num_elements == 0) return result;

  // Calculate data statistics using data_stats.h
  result.shannon_entropy = wrp_cte::compressor::DataStatistics<double>::CalculateShannonEntropy(
      data.data(), num_elements);
  result.mad = wrp_cte::compressor::DataStatistics<double>::CalculateMAD(
      data.data(), num_elements);
  result.second_derivative = wrp_cte::compressor::DataStatistics<double>::CalculateSecondDerivative(
      data.data(), num_elements);

  // Allocate buffers
  std::vector<uint8_t> compressed_data(data_size_bytes * 2);  // Safety margin
  std::vector<double> decompressed_data(num_elements);

  const void* input_ptr = data.data();
  void* compressed_ptr = compressed_data.data();
  void* decompressed_ptr = decompressed_data.data();

  // Warmup compression
  size_t warmup_size = compressed_data.size();
  bool warmup_ok = compressor->Compress(compressed_ptr, warmup_size,
                                        const_cast<void*>(input_ptr), data_size_bytes);
  if (!warmup_ok || warmup_size == 0) {
    return result;
  }

  // Warmup decompression
  size_t warmup_decomp_size = data_size_bytes;
  warmup_ok = compressor->Decompress(decompressed_ptr, warmup_decomp_size,
                                      compressed_ptr, warmup_size);
  if (!warmup_ok || warmup_decomp_size != data_size_bytes) {
    return result;
  }

  // Measure compression time
  size_t cmpr_size = compressed_data.size();
  double cpu_start = GetCpuTimeNs();
  bool comp_ok = compressor->Compress(compressed_ptr, cmpr_size,
                                      const_cast<void*>(input_ptr), data_size_bytes);
  double cpu_end = GetCpuTimeNs();

  if (!comp_ok || cmpr_size == 0) {
    return result;
  }

  result.compress_time_ms = (cpu_end - cpu_start) / 1e6;  // ns to ms
  result.compression_ratio = static_cast<double>(data_size_bytes) / static_cast<double>(cmpr_size);

  // Measure decompression time
  size_t decmpr_size = data_size_bytes;
  cpu_start = GetCpuTimeNs();
  bool decomp_ok = compressor->Decompress(decompressed_ptr, decmpr_size,
                                          compressed_ptr, cmpr_size);
  cpu_end = GetCpuTimeNs();

  if (!decomp_ok || decmpr_size != data_size_bytes) {
    return result;
  }

  result.decompress_time_ms = (cpu_end - cpu_start) / 1e6;  // ns to ms

  // Calculate PSNR for quality assessment
  result.psnr = wrp_cte::compressor::DataStatistics<double>::CalculatePSNR(
      data.data(), decompressed_data.data(), num_elements);

  result.success = true;
  return result;
}

/**
 * Write CSV header
 */
void WriteCSVHeader(std::ofstream& out) {
  out << "library,preset,data_size,shannon_entropy,mad,second_derivative,"
      << "compress_time,compression_ratio,decompress_time,psnr\n";
}

/**
 * Write result as CSV row
 */
void WriteCSVRow(std::ofstream& out, const CompressionResult& result) {
  if (!result.success) return;

  out << result.library << ","
      << result.preset << ","
      << result.data_size << ","
      << std::fixed << std::setprecision(6) << result.shannon_entropy << ","
      << std::setprecision(6) << result.mad << ","
      << std::setprecision(6) << result.second_derivative << ","
      << std::setprecision(6) << result.compress_time_ms << ","
      << std::setprecision(6) << result.compression_ratio << ","
      << std::setprecision(6) << result.decompress_time_ms << ","
      << std::setprecision(6) << result.psnr << "\n";
}

int main(int argc, char* argv[]) {
  // Parse command line arguments
  std::string input_path = "/workspace/context-transfer-engine/compressor/results/real_data/gs-output.bp/";
  std::string output_path = "/workspace/context-transfer-engine/compressor/results/real_data/grayscott_compression_analysis.csv";

  if (argc >= 2) {
    input_path = argv[1];
  }
  if (argc >= 3) {
    output_path = argv[2];
  }

  HLOG(kInfo, "Gray-Scott ADIOS2 Compression Analysis");
  HLOG(kInfo, "======================================");
  HLOG(kInfo, "Input:  {}", input_path);
  HLOG(kInfo, "Output: {}", output_path);

  // Open output CSV file
  std::ofstream csv_file(output_path);
  if (!csv_file.is_open()) {
    HLOG(kError, "Could not open output file: {}", output_path);
    return 1;
  }
  WriteCSVHeader(csv_file);

  // Initialize ADIOS2
  adios2::ADIOS adios;
  adios2::IO io = adios.DeclareIO("GrayScottAnalysis");
  io.SetEngine("BP5");

  adios2::Engine reader = io.Open(input_path, adios2::Mode::Read);

  // Need to BeginStep first before InquireVariable
  if (reader.BeginStep() != adios2::StepStatus::OK) {
    HLOG(kError, "Could not begin first step");
    return 1;
  }

  // Get available variables
  auto variables = io.AvailableVariables();

  HLOG(kInfo, "Available variables:");
  for (const auto& var : variables) {
    HLOG(kInfo, "  - {}", var.first);
  }

  // Get U and V variables
  adios2::Variable<double> varU = io.InquireVariable<double>("U");
  adios2::Variable<double> varV = io.InquireVariable<double>("V");

  if (!varU || !varV) {
    HLOG(kError, "Could not find U and V variables in ADIOS2 file");
    return 1;
  }

  // End the first step before entering the main loop
  reader.EndStep();

  // Initialize compression libraries
  struct CompressorConfig {
    std::string library;
    std::string preset;
    std::unique_ptr<hshm::Compressor> compressor;
  };

  std::vector<CompressorConfig> compressors;

  // Lossless compressors
  HLOG(kInfo, "Initializing compression libraries...");

  // ZSTD (3 modes)
  compressors.push_back({"zstd", "fast",
                         std::make_unique<hshm::ZstdWithModes>(hshm::LosslessMode::FAST)});
  compressors.push_back({"zstd", "balanced",
                         std::make_unique<hshm::ZstdWithModes>(hshm::LosslessMode::BALANCED)});
  compressors.push_back({"zstd", "best",
                         std::make_unique<hshm::ZstdWithModes>(hshm::LosslessMode::BEST)});

  // LZ4 (3 modes)
  compressors.push_back({"lz4", "fast",
                         std::make_unique<hshm::Lz4WithModes>(hshm::LosslessMode::FAST)});
  compressors.push_back({"lz4", "balanced",
                         std::make_unique<hshm::Lz4WithModes>(hshm::LosslessMode::BALANCED)});
  compressors.push_back({"lz4", "best",
                         std::make_unique<hshm::Lz4WithModes>(hshm::LosslessMode::BEST)});

  // Blosc2 (1 mode)
  compressors.push_back({"blosc2", "default",
                         std::make_unique<hshm::Blosc>()});

  // ZLIB (3 modes)
  compressors.push_back({"zlib", "fast",
                         std::make_unique<hshm::ZlibWithModes>(hshm::LosslessMode::FAST)});
  compressors.push_back({"zlib", "balanced",
                         std::make_unique<hshm::ZlibWithModes>(hshm::LosslessMode::BALANCED)});
  compressors.push_back({"zlib", "best",
                         std::make_unique<hshm::ZlibWithModes>(hshm::LosslessMode::BEST)});

  // BZIP2 (3 modes)
  compressors.push_back({"bzip2", "fast",
                         std::make_unique<hshm::Bzip2WithModes>(hshm::LosslessMode::FAST)});
  compressors.push_back({"bzip2", "balanced",
                         std::make_unique<hshm::Bzip2WithModes>(hshm::LosslessMode::BALANCED)});
  compressors.push_back({"bzip2", "best",
                         std::make_unique<hshm::Bzip2WithModes>(hshm::LosslessMode::BEST)});

  // LZMA (3 modes)
  compressors.push_back({"lzma", "fast",
                         std::make_unique<hshm::LzmaWithModes>(hshm::LosslessMode::FAST)});
  compressors.push_back({"lzma", "balanced",
                         std::make_unique<hshm::LzmaWithModes>(hshm::LosslessMode::BALANCED)});
  compressors.push_back({"lzma", "best",
                         std::make_unique<hshm::LzmaWithModes>(hshm::LosslessMode::BEST)});

  // BROTLI (3 modes)
  compressors.push_back({"brotli", "fast",
                         std::make_unique<hshm::BrotliWithModes>(hshm::LosslessMode::FAST)});
  compressors.push_back({"brotli", "balanced",
                         std::make_unique<hshm::BrotliWithModes>(hshm::LosslessMode::BALANCED)});
  compressors.push_back({"brotli", "best",
                         std::make_unique<hshm::BrotliWithModes>(hshm::LosslessMode::BEST)});

  // SNAPPY (1 mode)
  compressors.push_back({"snappy", "default",
                         std::make_unique<hshm::Snappy>()});

#if HSHM_ENABLE_COMPRESS && HSHM_ENABLE_LIBPRESSIO
  // Lossy compressors (LibPressio)
  HLOG(kInfo, "Adding lossy compressors (LibPressio enabled)...");

  // ZFP (3 modes)
  compressors.push_back({"zfp", "fast",
                         std::make_unique<hshm::LibPressioWithModes>("zfp", hshm::CompressionMode::FAST)});
  compressors.push_back({"zfp", "balanced",
                         std::make_unique<hshm::LibPressioWithModes>("zfp", hshm::CompressionMode::BALANCED)});
  compressors.push_back({"zfp", "best",
                         std::make_unique<hshm::LibPressioWithModes>("zfp", hshm::CompressionMode::BEST)});

  // SZ3 (3 modes)
  compressors.push_back({"sz3", "fast",
                         std::make_unique<hshm::LibPressioWithModes>("sz3", hshm::CompressionMode::FAST)});
  compressors.push_back({"sz3", "balanced",
                         std::make_unique<hshm::LibPressioWithModes>("sz3", hshm::CompressionMode::BALANCED)});
  compressors.push_back({"sz3", "best",
                         std::make_unique<hshm::LibPressioWithModes>("sz3", hshm::CompressionMode::BEST)});

  // FPZIP (3 modes)
  compressors.push_back({"fpzip", "fast",
                         std::make_unique<hshm::LibPressioWithModes>("fpzip", hshm::CompressionMode::FAST)});
  compressors.push_back({"fpzip", "balanced",
                         std::make_unique<hshm::LibPressioWithModes>("fpzip", hshm::CompressionMode::BALANCED)});
  compressors.push_back({"fpzip", "best",
                         std::make_unique<hshm::LibPressioWithModes>("fpzip", hshm::CompressionMode::BEST)});
#else
  HLOG(kInfo, "Lossy compressors disabled (LibPressio not available)");
#endif

  HLOG(kInfo, "Total compressors: {}", compressors.size());

  // Process each timestep
  size_t total_chunks = 0;
  size_t total_results = 0;

  while (reader.BeginStep() == adios2::StepStatus::OK) {
    size_t current_step = reader.CurrentStep();
    HLOG(kInfo, "Processing step {}...", current_step);

    // Read U field
    std::vector<double> dataU;
    reader.Get(varU, dataU, adios2::Mode::Sync);
    size_t sizeU_bytes = dataU.size() * sizeof(double);

    HLOG(kInfo, "  U field: {} elements ({} bytes)", dataU.size(), sizeU_bytes);

    // Benchmark all compressors on U
    for (const auto& config : compressors) {
      auto result = BenchmarkCompressor(config.compressor.get(), config.library, config.preset,
                                        dataU, sizeU_bytes);
      if (result.success) {
        WriteCSVRow(csv_file, result);
        total_results++;
      }
    }
    total_chunks++;

    // Read V field
    std::vector<double> dataV;
    reader.Get(varV, dataV, adios2::Mode::Sync);
    size_t sizeV_bytes = dataV.size() * sizeof(double);

    HLOG(kInfo, "  V field: {} elements ({} bytes)", dataV.size(), sizeV_bytes);

    // Benchmark all compressors on V
    for (const auto& config : compressors) {
      auto result = BenchmarkCompressor(config.compressor.get(), config.library, config.preset,
                                        dataV, sizeV_bytes);
      if (result.success) {
        WriteCSVRow(csv_file, result);
        total_results++;
      }
    }
    total_chunks++;

    csv_file.flush();  // Flush after each step for safety
    reader.EndStep();
  }

  reader.Close();
  csv_file.close();

  HLOG(kInfo, "======================================");
  HLOG(kInfo, "Analysis complete!");
  HLOG(kInfo, "Total chunks processed: {}", total_chunks);
  HLOG(kInfo, "Total results written: {}", total_results);
  HLOG(kInfo, "Output: {}", output_path);

  return 0;
}
