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
 * Unified Compression Benchmark
 *
 * This comprehensive benchmark tests all compression libraries with multiple
 * data distributions, sizes, and types. It consolidates three separate benchmarks
 * into a single unified test for easier analysis and comparison.
 *
 * Tests:
 * - Lossless libraries: BZIP2, ZSTD, LZ4, ZLIB, LZMA, BROTLI, SNAPPY, Blosc2
 * - Lossy libraries (requires LibPressio): ZFP, SZ3, FPZIP
 * - Data distributions (16 variants with different compressibility levels):
 *   - Uniform: high/medium/low entropy (full byte range to 4-bit values)
 *   - Gaussian: high/medium/low spread (wide to tight clustering)
 *   - Constant: maximally compressible
 *   - Linear gradient: smooth/coarse/stepped (different step sizes)
 *   - Sinusoidal: high/medium/low frequency (different periods)
 *   - Repeating: short/medium/long patterns (4 to 128 bytes)
 * - Data sizes: 64 logarithmically-spaced sizes from 1KB to 16MB
 * - Data types: char (uint8_t), int (int32_t), float (float)
 *
 * Parallelization:
 * - Set NUM_THREADS environment variable to enable parallel execution
 * - Uses work-stealing algorithm for optimal load balancing
 * - Threads pull compressor configs from a shared queue when ready
 * - Example: NUM_THREADS=8 ./benchmark_compression_unified_exec
 *
 * LibPressio Support:
 * - If LibPressio is not installed, only lossless compressors will be tested
 * - To enable lossy compression testing, install LibPressio and rebuild:
 *   - Ubuntu/Debian: sudo apt-get install libpressio-dev
 *   - Or build from source: https://github.com/robertu94/libpressio
 * - After installation, reconfigure with: cmake .. && make
 *
 * Output:
 * - Clean CSV file written to: results/compression_benchmark_results.csv
 * - Console output includes progress messages (stderr) and CSV data (stdout)
 * - CSV format with comprehensive metrics including:
 * - library_name, data_type, data_size, distribution_name
 * - shannon_entropy, mean_absolute_deviation
 * - compression_ratio, compress_time_ms, decompress_time_ms
 * - psnr_db (for lossy, 0.0 for lossless)
 */

#include "basic_test.h"
#include "hermes_shm/compress/compress_factory.h"
#include "hermes_shm/compress/data_stats.h"
#include "wrp_cte/compressor/models/distribution_classifier.h"
#include "hermes_shm/util/config_parse.h"
#include "hermes_shm/util/logging.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#include <ctime>
#endif

/**
 * @brief Get CPU time in nanoseconds using CLOCK_PROCESS_CPUTIME_ID.
 * @return CPU time in nanoseconds, or 0 if not available.
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

#if HSHM_ENABLE_COMPRESS
#include <lzo/lzo1x.h>
#endif

#if HSHM_ENABLE_COMPRESS
#include "hermes_shm/compress/lossless_modes.h"
#endif

#if HSHM_ENABLE_COMPRESS && HSHM_ENABLE_LIBPRESSIO
#include "hermes_shm/compress/libpressio.h"
#include "hermes_shm/compress/libpressio_modes.h"
#endif

// Bin configuration for binned distribution generator
struct BinConfig {
  double percentage;  // Fraction of data in this bin (0.0-1.0)
  double lo;          // Lower bound of bin range
  double hi;          // Upper bound of bin range
};

// Distribution configuration with parameters
struct DistributionConfig {
  std::string name;  // Full descriptive name (e.g., "uniform_high", "gaussian_low")
  std::string base_type;  // Base distribution type

  // Parameters for different distribution types
  int uniform_max_value = 256;  // For uniform: max random value
  double gaussian_stddev = 30.0;  // For gaussian: standard deviation
  int gradient_step_divisor = 1;  // For linear_gradient: step size
  double sinusoidal_period = 256.0;  // For sinusoidal: period length
  int repeating_pattern_len = 4;  // For repeating: pattern length

  // Parameters for binned distribution
  std::vector<BinConfig> bins;  // Bin configurations
  double perturbation = 0.0;    // Probability of repeating same bin (spatial correlation)
};

struct BenchmarkResult {
  std::string library;
  std::string configuration;  // Compression mode or config (e.g., "fast", "balanced", "best", "default")
  std::string data_type;
  std::string distribution;
  size_t data_size;
  double shannon_entropy;
  double mean_absolute_deviation;
  double first_order_derivative;   // Mean absolute first difference
  double second_order_derivative;  // Mean absolute second difference (curvature)
  double compression_ratio;
  double compress_time_ms;
  double decompress_time_ms;
  double psnr_db;  // 0.0 for lossless/perfect match, >0.0 for lossy quality in dB
  bool has_error;  // true if decompressed != input (expected for lossy, unexpected for lossless)
  double max_abs_error;  // Maximum absolute error (for diagnostics)
  size_t num_mismatches;  // Number of elements that differ

  // Block sampling features (for distribution classification)
  double block_entropy_mean;

  // Distribution classification results (mathematical approach)
  double classified_skewness;
  double classified_kurtosis;
};

// Data distribution generators with parameterization support
class DataGenerator {
 public:
  // Uniform random with configurable range (smaller range = more compressible)
  static void GenerateUniformRandom(void* data, size_t num_elements, hshm::DataType dtype, int max_value = 256) {
    std::random_device rd;
    std::mt19937 gen(rd());

    switch (dtype) {
      case hshm::DataType::UINT8: {
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          bytes[i] = gen() % max_value;
        }
        break;
      }
      case hshm::DataType::INT32: {
        int32_t* ints = static_cast<int32_t*>(data);
        std::uniform_int_distribution<int32_t> dist(-max_value * 1000, max_value * 1000);
        for (size_t i = 0; i < num_elements; i++) {
          ints[i] = dist(gen);
        }
        break;
      }
      case hshm::DataType::FLOAT32: {
        float* floats = static_cast<float*>(data);
        std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(max_value));
        for (size_t i = 0; i < num_elements; i++) {
          floats[i] = dist(gen);
        }
        break;
      }
      default:
        break;
    }
  }

  // Gaussian with configurable stddev (smaller stddev = more compressible)
  static void GenerateGaussian(void* data, size_t num_elements, hshm::DataType dtype, double stddev = 30.0) {
    std::random_device rd;
    std::mt19937 gen(rd());

    switch (dtype) {
      case hshm::DataType::UINT8: {
        uint8_t* bytes = static_cast<uint8_t*>(data);
        std::normal_distribution<> dist(128.0, stddev);
        for (size_t i = 0; i < num_elements; i++) {
          bytes[i] = static_cast<uint8_t>(std::clamp(dist(gen), 0.0, 255.0));
        }
        break;
      }
      case hshm::DataType::INT32: {
        int32_t* ints = static_cast<int32_t*>(data);
        std::normal_distribution<double> dist(0.0, stddev * 100.0);
        for (size_t i = 0; i < num_elements; i++) {
          ints[i] = static_cast<int32_t>(std::clamp(dist(gen), -2e9, 2e9));
        }
        break;
      }
      case hshm::DataType::FLOAT32: {
        float* floats = static_cast<float*>(data);
        std::normal_distribution<float> dist(0.0f, static_cast<float>(stddev));
        for (size_t i = 0; i < num_elements; i++) {
          floats[i] = dist(gen);
        }
        break;
      }
      default:
        break;
    }
  }

  // Constant (always maximally compressible)
  static void GenerateConstant(void* data, size_t num_elements, hshm::DataType dtype) {
    switch (dtype) {
      case hshm::DataType::UINT8: {
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          bytes[i] = 0x42;
        }
        break;
      }
      case hshm::DataType::INT32: {
        int32_t* ints = static_cast<int32_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          ints[i] = 42424242;
        }
        break;
      }
      case hshm::DataType::FLOAT32: {
        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          floats[i] = 42.42f;
        }
        break;
      }
      default:
        break;
    }
  }

  // Linear gradient with configurable step size (larger steps = more compressible)
  static void GenerateLinearGradient(void* data, size_t num_elements, hshm::DataType dtype, int step_divisor = 1) {
    switch (dtype) {
      case hshm::DataType::UINT8: {
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          bytes[i] = static_cast<uint8_t>(((i / step_divisor) * 256) / num_elements);
        }
        break;
      }
      case hshm::DataType::INT32: {
        int32_t* ints = static_cast<int32_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          int64_t val = static_cast<int64_t>(i / step_divisor) * 4000000000LL / num_elements - 2000000000LL;
          ints[i] = static_cast<int32_t>(val);
        }
        break;
      }
      case hshm::DataType::FLOAT32: {
        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          floats[i] = static_cast<float>(i / step_divisor) / num_elements * 1000.0f;
        }
        break;
      }
      default:
        break;
    }
  }

  // Sinusoidal with configurable period (longer period = more compressible)
  static void GenerateSinusoidal(void* data, size_t num_elements, hshm::DataType dtype, double period = 256.0) {
    switch (dtype) {
      case hshm::DataType::UINT8: {
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          double val = 128.0 + 127.0 * std::sin(2.0 * M_PI * i / period);
          bytes[i] = static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
        }
        break;
      }
      case hshm::DataType::INT32: {
        int32_t* ints = static_cast<int32_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          ints[i] = static_cast<int32_t>(1000000.0 * std::sin(2.0 * M_PI * i / period));
        }
        break;
      }
      case hshm::DataType::FLOAT32: {
        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          floats[i] = 1000.0f * std::sin(2.0f * M_PI * i / static_cast<float>(period));
        }
        break;
      }
      default:
        break;
    }
  }

  // Repeating pattern with configurable pattern length (shorter = more compressible)
  static void GenerateRepeating(void* data, size_t num_elements, hshm::DataType dtype, int pattern_len = 4) {
    switch (dtype) {
      case hshm::DataType::UINT8: {
        std::vector<uint8_t> pattern(pattern_len);
        for (int i = 0; i < pattern_len; i++) {
          pattern[i] = (i % 2 == 0) ? 0xAA : 0x55;
        }
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          bytes[i] = pattern[i % pattern_len];
        }
        break;
      }
      case hshm::DataType::INT32: {
        std::vector<int32_t> pattern(pattern_len);
        for (int i = 0; i < pattern_len; i++) {
          pattern[i] = (i % 2 == 0) ? 0xAAAAAAAA : 0x55555555;
        }
        int32_t* ints = static_cast<int32_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          ints[i] = pattern[i % pattern_len];
        }
        break;
      }
      case hshm::DataType::FLOAT32: {
        std::vector<float> pattern(pattern_len);
        for (int i = 0; i < pattern_len; i++) {
          pattern[i] = (i % 2 == 0) ? 123.456f : -789.012f;
        }
        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < num_elements; i++) {
          floats[i] = pattern[i % pattern_len];
        }
        break;
      }
      default:
        break;
    }
  }

  // Binned distribution with perturbation for spatial correlation control
  // Each bin has a percentage of data and a value range.
  // Perturbation controls how much values are interleaved:
  //   - perturbation=0: sample entire bin quota at once (long runs, low entropy)
  //   - perturbation=1: sample 1 value at a time (maximum interleaving, high entropy)
  // When a bin is chosen, we sample burst_size = max(1, bin_quota * (1 - perturbation)) values consecutively
  static void GenerateBinnedDistribution(void* data, size_t num_elements, hshm::DataType dtype,
                                         const std::vector<BinConfig>& bins, double perturbation = 0.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    size_t n_bins = bins.size();
    if (n_bins == 0) return;

    // Calculate target count for each bin
    std::vector<size_t> bin_targets(n_bins);
    std::vector<size_t> bin_counts(n_bins, 0);
    size_t total_assigned = 0;

    for (size_t i = 0; i < n_bins; ++i) {
      bin_targets[i] = static_cast<size_t>(bins[i].percentage * num_elements);
      total_assigned += bin_targets[i];
    }
    // Adjust last bin for rounding errors
    if (total_assigned < num_elements && n_bins > 0) {
      bin_targets[n_bins - 1] += num_elements - total_assigned;
    }

    // Track active bins (bins that haven't reached their quota)
    std::vector<size_t> active_bins;
    for (size_t i = 0; i < n_bins; ++i) {
      if (bin_targets[i] > 0) {
        active_bins.push_back(i);
      }
    }

    if (active_bins.empty()) return;

    // Helper to remove bin from active list
    auto remove_bin = [&active_bins](size_t bin_idx) {
      active_bins.erase(std::remove(active_bins.begin(), active_bins.end(), bin_idx), active_bins.end());
    };

    // Helper to choose a random active bin
    auto choose_random_bin = [&]() -> size_t {
      return active_bins[gen() % active_bins.size()];
    };

    // Generate values with burst sampling
    size_t i = 0;

    switch (dtype) {
      case hshm::DataType::UINT8: {
        uint8_t* bytes = static_cast<uint8_t*>(data);
        while (i < num_elements && !active_bins.empty()) {
          // Choose a random active bin
          size_t current_bin = choose_random_bin();
          size_t remaining = bin_targets[current_bin] - bin_counts[current_bin];

          // Calculate burst size: how many consecutive samples from this bin
          // burst_size = remaining * (1 - perturbation), at least 1
          size_t burst_size = std::max(static_cast<size_t>(1),
                                       static_cast<size_t>(remaining * (1.0 - perturbation)));
          burst_size = std::min(burst_size, remaining);  // Don't exceed quota
          burst_size = std::min(burst_size, num_elements - i);  // Don't exceed array

          // Generate burst_size consecutive values from this bin
          double lo = bins[current_bin].lo;
          double hi = bins[current_bin].hi;
          for (size_t j = 0; j < burst_size; ++j) {
            double val = (lo == hi) ? lo : lo + prob_dist(gen) * (hi - lo);
            bytes[i++] = static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
          }

          bin_counts[current_bin] += burst_size;

          // Check if bin reached its quota
          if (bin_counts[current_bin] >= bin_targets[current_bin]) {
            remove_bin(current_bin);
          }
        }
        break;
      }
      case hshm::DataType::INT32: {
        int32_t* ints = static_cast<int32_t*>(data);
        while (i < num_elements && !active_bins.empty()) {
          size_t current_bin = choose_random_bin();
          size_t remaining = bin_targets[current_bin] - bin_counts[current_bin];
          size_t burst_size = std::max(static_cast<size_t>(1),
                                       static_cast<size_t>(remaining * (1.0 - perturbation)));
          burst_size = std::min(burst_size, remaining);
          burst_size = std::min(burst_size, num_elements - i);

          double lo = bins[current_bin].lo;
          double hi = bins[current_bin].hi;
          for (size_t j = 0; j < burst_size; ++j) {
            double val = (lo == hi) ? lo : lo + prob_dist(gen) * (hi - lo);
            ints[i++] = static_cast<int32_t>(std::clamp(val, -2e9, 2e9));
          }

          bin_counts[current_bin] += burst_size;
          if (bin_counts[current_bin] >= bin_targets[current_bin]) {
            remove_bin(current_bin);
          }
        }
        break;
      }
      case hshm::DataType::FLOAT32: {
        float* floats = static_cast<float*>(data);
        while (i < num_elements && !active_bins.empty()) {
          size_t current_bin = choose_random_bin();
          size_t remaining = bin_targets[current_bin] - bin_counts[current_bin];
          size_t burst_size = std::max(static_cast<size_t>(1),
                                       static_cast<size_t>(remaining * (1.0 - perturbation)));
          burst_size = std::min(burst_size, remaining);
          burst_size = std::min(burst_size, num_elements - i);

          double lo = bins[current_bin].lo;
          double hi = bins[current_bin].hi;
          for (size_t j = 0; j < burst_size; ++j) {
            double val = (lo == hi) ? lo : lo + prob_dist(gen) * (hi - lo);
            floats[i++] = static_cast<float>(val);
          }

          bin_counts[current_bin] += burst_size;
          if (bin_counts[current_bin] >= bin_targets[current_bin]) {
            remove_bin(current_bin);
          }
        }
        break;
      }
      default:
        break;
    }
  }

  // Helper: Create uniform bin percentages (each bin gets equal share)
  static std::vector<double> CreateUniformPercentages(size_t n_bins) {
    std::vector<double> pct(n_bins, 1.0 / n_bins);
    return pct;
  }

  // Helper: Create normal-distributed bin percentages (centered on middle bins)
  static std::vector<double> CreateNormalPercentages(size_t n_bins, double std_bins = 2.0) {
    std::vector<double> pct(n_bins);
    double sum = 0.0;
    for (size_t i = 0; i < n_bins; ++i) {
      double x = -3.0 + 6.0 * i / (n_bins - 1);  // x in [-3, 3]
      double sigma = std_bins / 3.0 * 2.0;
      pct[i] = std::exp(-0.5 * (x / sigma) * (x / sigma));
      sum += pct[i];
    }
    for (size_t i = 0; i < n_bins; ++i) {
      pct[i] /= sum;
    }
    return pct;
  }

  // Helper: Create gamma-distributed bin percentages (skewed toward lower bins)
  static std::vector<double> CreateGammaPercentages(size_t n_bins, double shape = 2.0) {
    std::vector<double> pct(n_bins);
    double sum = 0.0;
    for (size_t i = 0; i < n_bins; ++i) {
      double x = 0.1 + 5.9 * i / (n_bins - 1);  // x in [0.1, 6.0]
      // Gamma PDF: x^(k-1) * e^(-x) / Gamma(k)
      pct[i] = std::pow(x, shape - 1.0) * std::exp(-x);
      sum += pct[i];
    }
    for (size_t i = 0; i < n_bins; ++i) {
      pct[i] /= sum;
    }
    return pct;
  }

  // Helper: Create exponential-distributed bin percentages (~90% in first bin, rest spread)
  // This creates data similar to Gray-Scott where most values are concentrated
  static std::vector<double> CreateExponentialPercentages(size_t n_bins, double concentration = 0.9) {
    std::vector<double> pct(n_bins);
    // First bin gets 'concentration' fraction (e.g., 90%)
    pct[0] = concentration;
    // Remaining bins share the rest exponentially decreasing
    double remaining = 1.0 - concentration;
    double decay = 0.5;  // Each subsequent bin gets half of previous
    double sum = 0.0;
    for (size_t i = 1; i < n_bins; ++i) {
      pct[i] = std::pow(decay, static_cast<double>(i));
      sum += pct[i];
    }
    // Normalize the remaining bins to sum to 'remaining'
    if (sum > 0) {
      for (size_t i = 1; i < n_bins; ++i) {
        pct[i] = pct[i] / sum * remaining;
      }
    }
    return pct;
  }

  // Helper: Create bimodal-distributed bin percentages (two peaks at extremes)
  // This represents data like Gray-Scott U/V concentrations with two dominant states
  static std::vector<double> CreateBimodalPercentages(size_t n_bins, double peak_ratio = 0.4) {
    std::vector<double> pct(n_bins, 0.0);
    if (n_bins < 4) {
      // Fallback to uniform for very small bin counts
      return CreateUniformPercentages(n_bins);
    }

    // Two peaks: one at low end (bins 0-1), one at high end (bins n-2 to n-1)
    // peak_ratio controls how much is in peaks vs transition region
    double peak_weight = peak_ratio;  // Total weight for each peak
    double transition_weight = 1.0 - 2 * peak_weight;  // Weight for middle region

    // Low peak (first 2 bins)
    pct[0] = peak_weight * 0.6;
    pct[1] = peak_weight * 0.4;

    // High peak (last 2 bins)
    pct[n_bins - 2] = peak_weight * 0.4;
    pct[n_bins - 1] = peak_weight * 0.6;

    // Transition region (middle bins) - small but non-zero for gradients
    size_t middle_bins = n_bins - 4;
    if (middle_bins > 0 && transition_weight > 0) {
      double per_middle = transition_weight / middle_bins;
      for (size_t i = 2; i < n_bins - 2; ++i) {
        pct[i] = per_middle;
      }
    }

    return pct;
  }

  // Helper: Create grayscott-like distribution (bimodal with smooth transitions)
  // Models reaction-diffusion patterns with background/spot regions
  static std::vector<double> CreateGrayscottPercentages(size_t n_bins, double background_ratio = 0.7) {
    std::vector<double> pct(n_bins, 0.0);
    if (n_bins < 4) {
      return CreateUniformPercentages(n_bins);
    }

    // Gray-Scott has:
    // - Large background region (low concentration, ~70%)
    // - Smaller spot/pattern regions (high concentration, ~20%)
    // - Transition edges (~10%)

    double background = background_ratio;  // Most data is background
    double spots = 0.2;  // Spot regions
    double edges = 1.0 - background - spots;  // Transition edges

    // Background: concentrated in first few bins (low values)
    size_t bg_bins = std::max<size_t>(2, n_bins / 4);
    for (size_t i = 0; i < bg_bins; ++i) {
      // Gaussian-like falloff from first bin
      double x = static_cast<double>(i) / bg_bins;
      pct[i] = background * std::exp(-2.0 * x * x);
    }

    // Spots: concentrated in last few bins (high values)
    size_t spot_bins = std::max<size_t>(2, n_bins / 4);
    for (size_t i = 0; i < spot_bins; ++i) {
      size_t idx = n_bins - 1 - i;
      double x = static_cast<double>(i) / spot_bins;
      pct[idx] = spots * std::exp(-2.0 * x * x);
    }

    // Edges: spread across middle bins
    size_t edge_start = bg_bins;
    size_t edge_end = n_bins - spot_bins;
    if (edge_end > edge_start) {
      double per_edge = edges / (edge_end - edge_start);
      for (size_t i = edge_start; i < edge_end; ++i) {
        pct[i] += per_edge;
      }
    }

    // Normalize to sum to 1.0
    double sum = 0.0;
    for (size_t i = 0; i < n_bins; ++i) sum += pct[i];
    if (sum > 0) {
      for (size_t i = 0; i < n_bins; ++i) pct[i] /= sum;
    }

    return pct;
  }

  // Helper: Create bin ranges given n_bins, bin_width, and value range
  static std::vector<BinConfig> CreateBinConfigs(size_t n_bins, double bin_width,
                                                  const std::vector<double>& percentages,
                                                  double value_lo = 0.0, double value_hi = 255.0) {
    std::vector<BinConfig> bins(n_bins);
    double total_range = value_hi - value_lo;

    if (bin_width * n_bins <= total_range) {
      // Bins fit within range - space them out
      double gap = (n_bins > 1) ? (total_range - bin_width * n_bins) / (n_bins - 1) : 0;
      double current = value_lo;
      for (size_t i = 0; i < n_bins; ++i) {
        bins[i].percentage = percentages[i];
        bins[i].lo = current;
        bins[i].hi = std::min(current + bin_width, value_hi);
        current = bins[i].hi + gap;
      }
    } else {
      // Bins overlap - distribute evenly
      double step = total_range / n_bins;
      for (size_t i = 0; i < n_bins; ++i) {
        bins[i].percentage = percentages[i];
        bins[i].lo = value_lo + i * step;
        bins[i].hi = std::min(bins[i].lo + bin_width, value_hi);
      }
    }

    return bins;
  }
};

// Benchmark a single compression library
BenchmarkResult BenchmarkCompressor(
    hshm::Compressor* compressor, const std::string& lib_name,
    const std::string& configuration,
    const std::string& data_type, const DistributionConfig& dist_config,
    size_t data_size, hshm::DataType dtype) {
  BenchmarkResult result;
  result.library = lib_name;
  result.configuration = configuration;
  result.data_type = data_type;
  result.distribution = dist_config.name;
  result.data_size = data_size;
  result.shannon_entropy = 0.0;
  result.mean_absolute_deviation = 0.0;
  result.first_order_derivative = 0.0;
  result.second_order_derivative = 0.0;
  result.compression_ratio = 0.0;
  result.compress_time_ms = 0.0;
  result.decompress_time_ms = 0.0;
  result.psnr_db = 0.0;  // Will be calculated after decompression
  result.has_error = false;
  result.max_abs_error = 0.0;
  result.num_mismatches = 0;

  // Block sampling features
  result.block_entropy_mean = 0.0;

  // Distribution classification
  result.classified_skewness = 0.0;
  result.classified_kurtosis = 0.0;

  size_t type_size = hshm::DataStatisticsFactory::GetTypeSize(dtype);
  size_t num_elements = data_size / type_size;

  // Allocate buffers
  std::vector<uint8_t> input_data(data_size);
  std::vector<uint8_t> compressed_data(data_size * 2);  // Safety margin
  std::vector<uint8_t> decompressed_data(data_size);

  // Generate test data with parameterization
  if (dist_config.base_type == "uniform") {
    DataGenerator::GenerateUniformRandom(input_data.data(),
                                         num_elements, dtype,
                                         dist_config.uniform_max_value);
  } else if (dist_config.base_type == "gaussian") {
    DataGenerator::GenerateGaussian(input_data.data(), num_elements,
                                    dtype, dist_config.gaussian_stddev);
  } else if (dist_config.base_type == "constant") {
    DataGenerator::GenerateConstant(input_data.data(), num_elements,
                                    dtype);
  } else if (dist_config.base_type == "linear_gradient") {
    DataGenerator::GenerateLinearGradient(input_data.data(),
                                          num_elements, dtype,
                                          dist_config.gradient_step_divisor);
  } else if (dist_config.base_type == "sinusoidal") {
    DataGenerator::GenerateSinusoidal(input_data.data(), num_elements,
                                      dtype, dist_config.sinusoidal_period);
  } else if (dist_config.base_type == "repeating") {
    DataGenerator::GenerateRepeating(input_data.data(), num_elements,
                                     dtype, dist_config.repeating_pattern_len);
  } else if (dist_config.base_type == "binned") {
    DataGenerator::GenerateBinnedDistribution(input_data.data(), num_elements,
                                              dtype, dist_config.bins,
                                              dist_config.perturbation);
  }

  // Calculate statistics using DataStatisticsFactory
  result.shannon_entropy =
      hshm::DataStatisticsFactory::CalculateShannonEntropy(input_data.data(), num_elements, dtype);
  result.mean_absolute_deviation =
      hshm::DataStatisticsFactory::CalculateMAD(input_data.data(), num_elements, dtype);
  result.first_order_derivative =
      hshm::DataStatisticsFactory::CalculateFirstDerivative(input_data.data(), num_elements, dtype);
  result.second_order_derivative =
      hshm::DataStatisticsFactory::CalculateSecondDerivative(input_data.data(), num_elements, dtype);

  // Block sampling for distribution classification
  // Use 16 blocks of 64 elements each for fast sampling
  size_t block_sample_size = 64;
  size_t num_sample_blocks = 16;
  // Adjust for small data: ensure we don't oversample
  if (num_elements < block_sample_size * num_sample_blocks) {
    block_sample_size = std::max<size_t>(8, num_elements / num_sample_blocks);
  }
  if (num_elements < block_sample_size) {
    block_sample_size = num_elements;
    num_sample_blocks = 1;
  }

  auto block_stats = hshm::BlockSamplerFactory::Sample(
      input_data.data(), num_elements, dtype,
      num_sample_blocks, block_sample_size, 42);  // Fixed seed for reproducibility

  result.block_entropy_mean = block_stats.entropy_mean;

  // Classify distribution using mathematical approach
  auto dist_result = wrp_cte::compressor::DistributionClassifierFactory::Classify(
      input_data.data(), num_elements, static_cast<wrp_cte::compressor::DataType>(dtype));
  result.classified_skewness = dist_result.skewness;
  result.classified_kurtosis = dist_result.kurtosis;

  // Warmup compression
  size_t warmup_cmpr_size = compressed_data.size();
  bool warmup_ok = compressor->Compress(
      compressed_data.data(), warmup_cmpr_size, input_data.data(), data_size);
  if (!warmup_ok || warmup_cmpr_size == 0) {
    return result;
  }

  // Warmup decompression
  size_t warmup_decomp_size = decompressed_data.size();
  warmup_ok = compressor->Decompress(
      decompressed_data.data(), warmup_decomp_size, compressed_data.data(),
      warmup_cmpr_size);
  if (!warmup_ok || warmup_decomp_size != data_size) {
    return result;
  }

  // Measure compression (CPU time)
  size_t cmpr_size = compressed_data.size();
  double cpu_start = GetCpuTimeNs();

  bool comp_ok = compressor->Compress(compressed_data.data(), cmpr_size,
                                       input_data.data(), data_size);

  double cpu_end = GetCpuTimeNs();

  if (!comp_ok || cmpr_size == 0) {
    return result;
  }

  // Convert nanoseconds to milliseconds
  result.compress_time_ms = (cpu_end - cpu_start) / 1e6;

  // Compression ratio
  if (cmpr_size > 0) {
    result.compression_ratio = static_cast<double>(data_size) / cmpr_size;
  } else {
    result.compression_ratio = 0.0;
  }

  // Measure decompression (CPU time)
  size_t decmpr_size = decompressed_data.size();
  cpu_start = GetCpuTimeNs();

  bool decomp_ok = compressor->Decompress(decompressed_data.data(), decmpr_size,
                                           compressed_data.data(), cmpr_size);

  cpu_end = GetCpuTimeNs();

  if (!decomp_ok || decmpr_size != data_size) {
    return result;
  }

  // Convert nanoseconds to milliseconds
  result.decompress_time_ms = (cpu_end - cpu_start) / 1e6;

  // Check for differences between input and decompressed data
  // This is critical for identifying whether lossy compression is actually working
  bool has_differences = false;
  size_t num_different_bytes = 0;
  size_t num_different_elements = 0;
  double max_element_error = 0.0;

  // Byte-level comparison
  for (size_t i = 0; i < data_size; i++) {
    if (input_data[i] != decompressed_data[i]) {
      has_differences = true;
      num_different_bytes++;
    }
  }

  // Element-level comparison (for typed data)
  if (dtype == hshm::DataType::FLOAT32) {
    const float* input_floats = reinterpret_cast<const float*>(input_data.data());
    const float* output_floats = reinterpret_cast<const float*>(decompressed_data.data());
    for (size_t i = 0; i < num_elements; i++) {
      if (input_floats[i] != output_floats[i]) {
        num_different_elements++;
        double abs_error = std::abs(input_floats[i] - output_floats[i]);
        max_element_error = std::max(max_element_error, abs_error);
      }
    }
  } else if (dtype == hshm::DataType::INT32) {
    const int32_t* input_ints = reinterpret_cast<const int32_t*>(input_data.data());
    const int32_t* output_ints = reinterpret_cast<const int32_t*>(decompressed_data.data());
    for (size_t i = 0; i < num_elements; i++) {
      if (input_ints[i] != output_ints[i]) {
        num_different_elements++;
        double abs_error = std::abs(input_ints[i] - output_ints[i]);
        max_element_error = std::max(max_element_error, abs_error);
      }
    }
  } else if (dtype == hshm::DataType::UINT8) {
    const uint8_t* input_bytes = input_data.data();
    const uint8_t* output_bytes = decompressed_data.data();
    for (size_t i = 0; i < num_elements; i++) {
      if (input_bytes[i] != output_bytes[i]) {
        num_different_elements++;
        double abs_error = std::abs(static_cast<int>(input_bytes[i]) - static_cast<int>(output_bytes[i]));
        max_element_error = std::max(max_element_error, abs_error);
      }
    }
  }

  result.has_error = has_differences;
  result.max_abs_error = max_element_error;
  result.num_mismatches = num_different_elements;

  // Diagnostic output for unexpected behavior
  bool is_lossy_compressor = (lib_name == "ZFP" || lib_name == "SZ3" || lib_name == "FPZIP");
  bool is_lossless_compressor = !is_lossy_compressor;

  if (is_lossy_compressor && !has_differences) {
    HLOG(kWarning, "Lossy compressor {} [{}] produced EXACT reconstruction (expected some error)!",
         lib_name, configuration);
    HLOG(kWarning, "    Data type: {}, Size: {}", data_type, data_size);
  }

  if (is_lossless_compressor && has_differences) {
    HLOG(kError, "Lossless compressor {} [{}] produced LOSSY reconstruction (data corruption)!",
         lib_name, configuration);
    HLOG(kError, "    Mismatched bytes: {}/{}", num_different_bytes, data_size);
    HLOG(kError, "    Mismatched elements: {}/{}", num_different_elements, num_elements);
  }

  // Calculate PSNR (quality metric for both lossless and lossy compression)
  // PSNR = 0.0 for lossless/perfect match (represents infinity)
  // PSNR > 0.0 for lossy compression (quality in dB)
  result.psnr_db = hshm::DataStatisticsFactory::CalculatePSNR(
      input_data.data(), decompressed_data.data(), num_elements, dtype);

  // DEBUG: For lossy compressors on float data with PSNR=0.0, print diagnostics
  if (result.psnr_db == 0.0 && dtype == hshm::DataType::FLOAT32 &&
      (lib_name == "SZ3" || lib_name == "FPZIP" || lib_name == "ZFP")) {
    const float* input_floats = reinterpret_cast<const float*>(input_data.data());
    const float* output_floats = reinterpret_cast<const float*>(decompressed_data.data());

    double mse = hshm::DataStatisticsFactory::CalculateMSE(
        input_data.data(), decompressed_data.data(), num_elements, dtype);

    HLOG(kInfo, "  [DIAGNOSTIC] {} {} on float data has PSNR=0.0!", lib_name, configuration);
    HLOG(kInfo, "    MSE: {}", mse);
    HIPRINT("    First 5 input:  ");
    for (int i = 0; i < std::min(5, (int)num_elements); i++) {
      HIPRINT("{} ", input_floats[i]);
    }
    HIPRINT("\n");
    HIPRINT("    First 5 output: ");
    for (int i = 0; i < std::min(5, (int)num_elements); i++) {
      HIPRINT("{} ", output_floats[i]);
    }
    HIPRINT("\n");
    HIPRINT("    First 5 diffs: ");
    for (int i = 0; i < std::min(5, (int)num_elements); i++) {
      HIPRINT("{} ", input_floats[i] - output_floats[i]);
    }
    HIPRINT("\n");
  }

  return result;
}

// Print CSV table header
void PrintCSVHeader(std::ostream& os) {
  os << "library_name,configuration,data_type,data_size,distribution_name,"
     << "classified_skewness,classified_kurtosis,"
     << "shannon_entropy,mean_absolute_deviation,first_order_derivative,second_order_derivative,"
     << "block_entropy_mean,"
     << "compression_ratio,compress_time_ms,decompress_time_ms,"
     << "psnr_db,has_error,max_abs_error,num_mismatches\n";
}

// Print result as CSV
void PrintResultCSV(const BenchmarkResult& result, std::ostream& os) {
  os << result.library << "," << result.configuration << ","
     << result.data_type << "," << result.data_size
     << "," << result.distribution << ","
     // Distribution classification
     << std::fixed << std::setprecision(4) << result.classified_skewness << ","
     << std::setprecision(4) << result.classified_kurtosis << ","
     // Full-data statistics
     << std::setprecision(4) << result.shannon_entropy << ","
     << std::setprecision(2) << result.mean_absolute_deviation << ","
     << std::setprecision(4) << result.first_order_derivative << ","
     << std::setprecision(4) << result.second_order_derivative << ","
     // Block sampling features
     << std::setprecision(4) << result.block_entropy_mean << ","
     // Compression results
     << std::setprecision(4) << result.compression_ratio << ","
     << std::setprecision(3) << result.compress_time_ms << ","
     << result.decompress_time_ms << "," << std::setprecision(2)
     << result.psnr_db << ","
     << (result.has_error ? "true" : "false") << ","
     << std::setprecision(6) << result.max_abs_error << ","
     << result.num_mismatches << "\n";
}

TEST_CASE("Unified Compression Benchmark") {
  // Parse command-line arguments from environment variables
  // MAX_SIZE: Maximum data size (default: 128KB) - supports semantic sizes like "128K", "16MB"
  // NUM_BINS: Number of logarithmic size bins (default: 12)
  size_t max_size = hshm::Unit<size_t>::Kilobytes(128);  // Default 128KB
  size_t num_bins = 12;  // Default 12 bins

  const char* max_size_env = std::getenv("MAX_SIZE");
  if (max_size_env) {
    try {
      max_size = hshm::ConfigParse::ParseSize(max_size_env);
      HLOG(kInfo, "Using MAX_SIZE={} ({} bytes)", max_size_env, max_size);
    } catch (...) {
      HLOG(kWarning, "Invalid MAX_SIZE value '{}', using default 16MB", max_size_env);
    }
  }

  const char* num_bins_env = std::getenv("NUM_BINS");
  if (num_bins_env) {
    try {
      num_bins = std::stoul(num_bins_env);
      if (num_bins < 1) num_bins = 1;
      if (num_bins > 256) num_bins = 256;
      HLOG(kInfo, "Using NUM_BINS={}", num_bins);
    } catch (...) {
      HLOG(kWarning, "Invalid NUM_BINS value, using default 64");
    }
  }

  // Generate logarithmically-spaced sizes from 1KB to max_size
  std::vector<size_t> data_sizes;
  const size_t min_size = 1024;  // 1KB minimum

  if (num_bins == 1) {
    data_sizes.push_back(max_size);
  } else {
    double log_min = std::log2(static_cast<double>(min_size));
    double log_max = std::log2(static_cast<double>(max_size));

    for (size_t i = 0; i < num_bins; i++) {
      double exponent = log_min + (i * (log_max - log_min) / (num_bins - 1));
      size_t size_bytes = static_cast<size_t>(std::pow(2.0, exponent));
      // Ensure we don't add duplicates
      if (data_sizes.empty() || data_sizes.back() != size_bytes) {
        data_sizes.push_back(size_bytes);
      }
    }
  }

  HLOG(kInfo, "Data sizes: {} bins from {} to {} bytes", data_sizes.size(),
       data_sizes.front(), data_sizes.back());

  // Generate binned distribution configurations
  // Environment variables to control distribution generation:
  // DIST_BINS: Number of value bins (default: 16)
  // DIST_WIDTHS: Comma-separated bin widths (default: "1,16,64,128,256")
  // DIST_PERTURBATIONS: Comma-separated perturbation values (default: "0,0.05,0.1,0.25,0.325,0.5")
  // DIST_TYPES: Comma-separated distribution types (default: "uniform,normal,gamma")

  size_t n_dist_bins = 48;
  std::vector<double> bin_widths = {0.0, 1.0, 4.0, 8.0, 16.0};  // 0 = constant per bin
  std::vector<double> perturbations = {0.0, 0.05, 0.1, 0.25, 0.325, 0.5};
  std::vector<std::string> dist_type_names = {"uniform", "normal", "gamma", "exponential", "bimodal", "grayscott"};  // All six by default

  // Parse DIST_BINS
  const char* dist_bins_env = std::getenv("DIST_BINS");
  if (dist_bins_env) {
    try {
      n_dist_bins = std::stoul(dist_bins_env);
      if (n_dist_bins < 2) n_dist_bins = 2;
      HLOG(kInfo, "Using DIST_BINS={}", n_dist_bins);
    } catch (...) {
      HLOG(kWarning, "Invalid DIST_BINS, using default 16");
    }
  }

  // Parse DIST_WIDTHS (comma-separated)
  const char* dist_widths_env = std::getenv("DIST_WIDTHS");
  if (dist_widths_env) {
    bin_widths.clear();
    std::stringstream ss(dist_widths_env);
    std::string item;
    while (std::getline(ss, item, ',')) {
      try {
        bin_widths.push_back(std::stod(item));
      } catch (...) {}
    }
    if (bin_widths.empty()) bin_widths = {1.0, 64.0, 256.0};
    HLOG(kInfo, "Using DIST_WIDTHS with {} values", bin_widths.size());
  }

  // Parse DIST_PERTURBATIONS (comma-separated)
  const char* dist_perturb_env = std::getenv("DIST_PERTURBATIONS");
  if (dist_perturb_env) {
    perturbations.clear();
    std::stringstream ss(dist_perturb_env);
    std::string item;
    while (std::getline(ss, item, ',')) {
      try {
        perturbations.push_back(std::stod(item));
      } catch (...) {}
    }
    if (perturbations.empty()) perturbations = {0.0, 0.25, 0.5};
    HLOG(kInfo, "Using DIST_PERTURBATIONS with {} values", perturbations.size());
  }

  // Parse DIST_TYPES (comma-separated)
  const char* dist_types_env = std::getenv("DIST_TYPES");
  if (dist_types_env) {
    dist_type_names.clear();
    std::stringstream ss(dist_types_env);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (item == "uniform" || item == "normal" || item == "gamma" || item == "exponential" ||
          item == "bimodal" || item == "grayscott") {
        dist_type_names.push_back(item);
      }
    }
    if (dist_type_names.empty()) dist_type_names = {"uniform", "normal", "gamma", "exponential", "bimodal", "grayscott"};
    HLOG(kInfo, "Using DIST_TYPES with {} types", dist_type_names.size());
  }

  // Pre-compute percentage distributions
  auto uniform_pct = DataGenerator::CreateUniformPercentages(n_dist_bins);
  auto normal_pct = DataGenerator::CreateNormalPercentages(n_dist_bins);
  auto gamma_pct = DataGenerator::CreateGammaPercentages(n_dist_bins);
  auto exponential_pct = DataGenerator::CreateExponentialPercentages(n_dist_bins);
  auto bimodal_pct = DataGenerator::CreateBimodalPercentages(n_dist_bins);
  auto grayscott_pct = DataGenerator::CreateGrayscottPercentages(n_dist_bins);

  std::vector<DistributionConfig> distributions;

  // Build distribution type map
  std::vector<std::pair<std::string, std::vector<double>>> dist_types;
  for (const auto& type_name : dist_type_names) {
    if (type_name == "uniform") {
      dist_types.push_back({"uniform", uniform_pct});
    } else if (type_name == "bimodal") {
      dist_types.push_back({"bimodal", bimodal_pct});
    } else if (type_name == "grayscott") {
      dist_types.push_back({"grayscott", grayscott_pct});
    } else if (type_name == "normal") {
      dist_types.push_back({"normal", normal_pct});
    } else if (type_name == "gamma") {
      dist_types.push_back({"gamma", gamma_pct});
    } else if (type_name == "exponential") {
      dist_types.push_back({"exponential", exponential_pct});
    }
  }

  // Generate all combinations
  for (const auto& [dist_name, percentages] : dist_types) {
    for (double bin_width : bin_widths) {
      for (double perturbation : perturbations) {
        DistributionConfig config;
        config.name = "binned_" + dist_name + "_w" + std::to_string(static_cast<int>(bin_width))
                    + "_p" + std::to_string(static_cast<int>(perturbation * 1000));
        config.base_type = "binned";
        config.bins = DataGenerator::CreateBinConfigs(n_dist_bins, bin_width, percentages, 0.0, 255.0);
        config.perturbation = perturbation;
        distributions.push_back(config);
      }
    }
  }

  HLOG(kInfo,
       "Generated {} binned distribution configurations ({} types × {} widths × {} perturbations)",
       distributions.size(), dist_types.size(), bin_widths.size(), perturbations.size());

  const std::vector<std::pair<std::string, hshm::DataType>> data_types = {
      {"char", hshm::DataType::UINT8},
      {"int", hshm::DataType::INT32},
      {"float", hshm::DataType::FLOAT32}};

  // Get number of threads from environment
  size_t num_threads = 1;  // Default to sequential
  const char* threads_env = std::getenv("NUM_THREADS");
  if (threads_env) {
    try {
      size_t requested = std::stoul(threads_env);
      if (requested > 0 && requested <= 256) {
        num_threads = requested;
      }
    } catch (...) {
      HLOG(kWarning, "Invalid NUM_THREADS value, defaulting to 1 thread");
    }
  }

  // Open output file in results directory
  // CSV output goes ONLY to file (not stdout) to avoid corruption when redirecting
  std::string output_path = "results/compression_benchmark_results.csv";
  const char* output_env = std::getenv("OUTPUT_FILE");
  if (output_env) {
    output_path = output_env;
    HLOG(kInfo, "Using OUTPUT_FILE={}", output_path);
  }

  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    HLOG(kError, "Could not open output file: {}", output_path);
    HLOG(kError, "Note: Make sure the directory exists.");
    return;
  }
  HLOG(kInfo, "Writing results to: {}", output_path);

  // Print header to file only
  PrintCSVHeader(outfile);

  // Initialize LZO library
#if HSHM_ENABLE_COMPRESS
  lzo_init();
#endif

  // Test each compression library with different configurations
  struct CompressorTest {
    std::string name;
    std::string configuration;
    std::unique_ptr<hshm::Compressor> compressor;
  };

  std::vector<CompressorTest> compressors;

  // Check for lossy-only mode via environment variable
  // Usage: LOSSY_ONLY=1 ./benchmark_compression_unified_exec
  const char* lossy_only_env = std::getenv("LOSSY_ONLY");
  bool lossy_only = lossy_only_env && (std::string(lossy_only_env) == "1" || std::string(lossy_only_env) == "true");

  // Add lossless compressors (unless in lossy-only mode)
  HLOG(kInfo, "=== Compression Benchmark Configuration ===");
  if (lossy_only) {
    HLOG(kInfo, "Mode: LOSSY ONLY (lossless libraries disabled)");
  } else {
    HLOG(kInfo, "Lossless libraries: BZIP2, ZSTD, LZ4, ZLIB, LZMA, BROTLI (with compression levels)");
    HLOG(kInfo, "Lossless libraries (single mode): SNAPPY, Blosc2");
    HLOG(kInfo, "Compression modes: FAST, BALANCED, BEST");

    // BZIP2 - compression levels 1, 6, 9
    compressors.push_back({"BZIP2", "fast", std::make_unique<hshm::Bzip2WithModes>(hshm::LosslessMode::FAST)});
    compressors.push_back({"BZIP2", "balanced", std::make_unique<hshm::Bzip2WithModes>(hshm::LosslessMode::BALANCED)});
    compressors.push_back({"BZIP2", "best", std::make_unique<hshm::Bzip2WithModes>(hshm::LosslessMode::BEST)});

    // ZSTD - compression levels 1, 3, 19
    compressors.push_back({"ZSTD", "fast", std::make_unique<hshm::ZstdWithModes>(hshm::LosslessMode::FAST)});
    compressors.push_back({"ZSTD", "balanced", std::make_unique<hshm::ZstdWithModes>(hshm::LosslessMode::BALANCED)});
    compressors.push_back({"ZSTD", "best", std::make_unique<hshm::ZstdWithModes>(hshm::LosslessMode::BEST)});

    // LZ4 - default vs HC mode
    compressors.push_back({"LZ4", "fast", std::make_unique<hshm::Lz4WithModes>(hshm::LosslessMode::FAST)});
    compressors.push_back({"LZ4", "balanced", std::make_unique<hshm::Lz4WithModes>(hshm::LosslessMode::BALANCED)});
    compressors.push_back({"LZ4", "best", std::make_unique<hshm::Lz4WithModes>(hshm::LosslessMode::BEST)});

    // ZLIB - compression levels 1, 6, 9
    compressors.push_back({"ZLIB", "fast", std::make_unique<hshm::ZlibWithModes>(hshm::LosslessMode::FAST)});
    compressors.push_back({"ZLIB", "balanced", std::make_unique<hshm::ZlibWithModes>(hshm::LosslessMode::BALANCED)});
    compressors.push_back({"ZLIB", "best", std::make_unique<hshm::ZlibWithModes>(hshm::LosslessMode::BEST)});

    // LZMA - compression levels 0, 6, 9
    compressors.push_back({"LZMA", "fast", std::make_unique<hshm::LzmaWithModes>(hshm::LosslessMode::FAST)});
    compressors.push_back({"LZMA", "balanced", std::make_unique<hshm::LzmaWithModes>(hshm::LosslessMode::BALANCED)});
    compressors.push_back({"LZMA", "best", std::make_unique<hshm::LzmaWithModes>(hshm::LosslessMode::BEST)});

    // BROTLI - quality levels 1, 6, 11
    compressors.push_back({"BROTLI", "fast", std::make_unique<hshm::BrotliWithModes>(hshm::LosslessMode::FAST)});
    compressors.push_back({"BROTLI", "balanced", std::make_unique<hshm::BrotliWithModes>(hshm::LosslessMode::BALANCED)});
    compressors.push_back({"BROTLI", "best", std::make_unique<hshm::BrotliWithModes>(hshm::LosslessMode::BEST)});

    // SNAPPY and Blosc2 - no compression level support, use default only
    compressors.push_back({"SNAPPY", "default", std::make_unique<hshm::Snappy>()});
    compressors.push_back({"Blosc2", "default", std::make_unique<hshm::Blosc>()});
  }
  HLOG(kInfo, "Distribution variants: {} (varying compressibility)", distributions.size());
  HLOG(kInfo, "Data sizes: {} bins", data_sizes.size());

#if HSHM_ENABLE_COMPRESS && HSHM_ENABLE_LIBPRESSIO
  // Add lossy compressors via LibPressio with multiple compression modes
  HLOG(kInfo, "Lossy libraries: ZFP, SZ3, FPZIP (LibPressio enabled)");
  HLOG(kInfo, "Compression modes: FAST, BALANCED, BEST");

  // ZFP - block transform compressor (best for floating-point data)
  compressors.push_back({"ZFP", "fast", std::make_unique<hshm::LibPressioWithModes>("zfp", hshm::CompressionMode::FAST)});
  compressors.push_back({"ZFP", "balanced", std::make_unique<hshm::LibPressioWithModes>("zfp", hshm::CompressionMode::BALANCED)});
  compressors.push_back({"ZFP", "best", std::make_unique<hshm::LibPressioWithModes>("zfp", hshm::CompressionMode::BEST)});

  // SZ3 - Lossy compressor for floating-point data only
  compressors.push_back({"SZ3", "fast", std::make_unique<hshm::LibPressioWithModes>("sz3", hshm::CompressionMode::FAST)});
  compressors.push_back({"SZ3", "balanced", std::make_unique<hshm::LibPressioWithModes>("sz3", hshm::CompressionMode::BALANCED)});
  compressors.push_back({"SZ3", "best", std::make_unique<hshm::LibPressioWithModes>("sz3", hshm::CompressionMode::BEST)});

  // FPZIP - Lossy compressor for floating-point data only
  compressors.push_back({"FPZIP", "fast", std::make_unique<hshm::LibPressioWithModes>("fpzip", hshm::CompressionMode::FAST)});
  compressors.push_back({"FPZIP", "balanced", std::make_unique<hshm::LibPressioWithModes>("fpzip", hshm::CompressionMode::BALANCED)});
  compressors.push_back({"FPZIP", "best", std::make_unique<hshm::LibPressioWithModes>("fpzip", hshm::CompressionMode::BEST)});
#else
  HLOG(kInfo, "Lossy libraries: DISABLED (LibPressio not available)");
  HLOG(kInfo, "  To enable lossy compression: install LibPressio and rebuild");
#endif

  HLOG(kInfo, "Total compressor configurations: {}", compressors.size());
  if (num_threads > 1) {
    HLOG(kInfo, "Parallel execution: {} threads", num_threads);
  }
  HLOG(kInfo, "========================================");

  // Thread-safe output and progress tracking
  std::mutex csv_mutex;
  std::atomic<size_t> samples_completed{0};
  auto start_time = std::chrono::steady_clock::now();

  // Calculate total samples (pre-computed for progress reporting)
  size_t total_samples = 0;
  for (const auto& test : compressors) {
    bool is_lossy = (test.name == "ZFP" || test.name == "SZ3" || test.name == "FPZIP");
    size_t num_types = is_lossy ? 1 : 3;  // Lossy only works on float
    total_samples += num_types * distributions.size() * data_sizes.size();
  }
  HLOG(kInfo, "Total samples to process: {}", total_samples);

  // Process compressors (in parallel if num_threads > 1)
  if (num_threads == 1) {
    // Sequential execution with progress reporting
    size_t seq_completed = 0;
    auto seq_start_time = std::chrono::steady_clock::now();

    for (const auto& test : compressors) {
      const std::string& lib_name = test.name;
      const std::string& config_name = test.configuration;
      hshm::Compressor* compressor = test.compressor.get();

      try {
        for (const auto& [data_type, dtype] : data_types) {
          // Skip lossy compressors for non-float data
          // ZFP, SZ3, and FPZIP are designed only for floating-point compression
          bool is_lossy = (lib_name == "ZFP" || lib_name == "SZ3" || lib_name == "FPZIP");
          bool is_float_type = (dtype == hshm::DataType::FLOAT32);

          if (is_lossy && !is_float_type) {
            continue;
          }

          size_t type_size = hshm::DataStatisticsFactory::GetTypeSize(dtype);
          for (const auto& dist_config : distributions) {
            for (size_t data_size : data_sizes) {
              // Skip if data size is not aligned with data type
              if (data_size % type_size != 0) {
                continue;
              }

              auto result = BenchmarkCompressor(
                  compressor, lib_name, config_name, data_type, dist_config, data_size,
                  dtype);

              // Write to file
              PrintResultCSV(result, outfile);
              outfile.flush();

              // Progress reporting
              ++seq_completed;
              if (seq_completed % 100 == 0 || seq_completed == total_samples) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - seq_start_time).count();
                double percent = (seq_completed * 100.0) / total_samples;
                double rate = seq_completed / (elapsed / 60.0 + 0.001);
                double eta_min = (total_samples - seq_completed) / rate;

                HIPRINT("\r[Progress] {} / {} ({}%) | Rate: {} samples/min | ETA: {} min   ",
                        seq_completed, total_samples, static_cast<int>(percent),
                        static_cast<int>(rate), static_cast<int>(eta_min));

                if (seq_completed == total_samples) {
                  HIPRINT("\n");
                }
              }
            }
          }
        }
      } catch (const std::exception& e) {
        HLOG(kError, "\nError benchmarking {} [{}]: {}", lib_name, config_name, e.what());
        continue;
      } catch (...) {
        HLOG(kError, "\nUnknown error benchmarking {} [{}]", lib_name, config_name);
        continue;
      }
    }
    HLOG(kInfo, "Sequential execution completed!");
  } else {
    // Parallel execution with work stealing
    std::vector<std::thread> workers;

    // Work queue containing indices of compressors to process
    std::queue<size_t> work_queue;
    std::mutex queue_mutex;

    // Populate work queue with all compressor indices
    for (size_t i = 0; i < compressors.size(); ++i) {
      work_queue.push(i);
    }

    HLOG(kInfo, "Work queue initialized with {} compressor configs", compressors.size());

    // Spawn worker threads
    for (size_t t = 0; t < num_threads; ++t) {
      workers.emplace_back([&]() {
        while (true) {
          // Get next work item (compressor index)
          size_t compressor_idx;
          {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (work_queue.empty()) {
              break;  // No more work
            }
            compressor_idx = work_queue.front();
            work_queue.pop();
          }

          // Process this compressor
          const auto& test = compressors[compressor_idx];
          const std::string& lib_name = test.name;
          const std::string& config_name = test.configuration;
          hshm::Compressor* compressor = test.compressor.get();

          try {
            for (const auto& [data_type, dtype] : data_types) {
              // Skip lossy compressors for non-float data
              bool is_lossy = (lib_name == "ZFP" || lib_name == "SZ3" || lib_name == "FPZIP");
              bool is_float_type = (dtype == hshm::DataType::FLOAT32);

              if (is_lossy && !is_float_type) {
                continue;
              }

              size_t type_size = hshm::DataStatisticsFactory::GetTypeSize(dtype);
              for (const auto& dist_config : distributions) {
                for (size_t data_size : data_sizes) {
                  if (data_size % type_size != 0) {
                    continue;
                  }

                  auto result = BenchmarkCompressor(
                      compressor, lib_name, config_name, data_type, dist_config, data_size, dtype);

                  // Thread-safe write to file only
                  {
                    std::lock_guard<std::mutex> lock(csv_mutex);
                    PrintResultCSV(result, outfile);
                    outfile.flush();

                    // Progress reporting
                    size_t completed = ++samples_completed;
                    if (completed % 100 == 0 || completed == total_samples) {
                      auto now = std::chrono::steady_clock::now();
                      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                      double percent = (completed * 100.0) / total_samples;
                      double rate = completed / (elapsed / 60.0 + 0.001);
                      double eta_min = (total_samples - completed) / rate;

                      HIPRINT("\r[Progress] {} / {} ({}%) | Rate: {} samples/min | ETA: {} min   ",
                              completed, total_samples, static_cast<int>(percent),
                              static_cast<int>(rate), static_cast<int>(eta_min));

                      if (completed == total_samples) {
                        HIPRINT("\n");
                      }
                    }
                  }
                }
              }
            }
          } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(csv_mutex);
            HLOG(kError, "\nError benchmarking {} [{}]: {}", lib_name, config_name, e.what());
          } catch (...) {
            std::lock_guard<std::mutex> lock(csv_mutex);
            HLOG(kError, "\nUnknown error benchmarking {} [{}]", lib_name, config_name);
          }
        }
      });
    }

    // Wait for all threads to complete
    for (auto& worker : workers) {
      worker.join();
    }

    HLOG(kInfo, "All threads completed!");
  }

  if (outfile.is_open()) {
    outfile.close();
    HLOG(kInfo, "Results saved to: results/compression_benchmark_results.csv");
  }
}
