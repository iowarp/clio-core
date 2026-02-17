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

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Comprehensive Multi-Threaded Compression Benchmark System
 * Combines palette-based synthetic data generation with compression testing
 * Generates high-quality training data for ML model training
 *
 * Key Features:
 * - 32-bin palette system with configurable weights and perturbation
 * - 16 distribution palettes (uniform, normal, gamma, exponential, bimodal, grayscott)
 * - All data types: char (uint8_t), int (int32_t), float (float)
 * - Multi-threaded work-stealing for optimal performance
 * - Comprehensive feature statistics (Shannon, MAD, 1st/2nd derivatives)
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <getopt.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <random>
#include <algorithm>

#include "wrp_cte/compressor/models/data_stats.h"
#include <hermes_shm/util/logging.h>

// Compression libraries
#ifdef HSHM_ENABLE_COMPRESS
#include "hermes_shm/compress/compress_factory.h"
#include "hermes_shm/compress/lossless_modes.h"
#ifdef HSHM_ENABLE_LIBPRESSIO
#include "hermes_shm/compress/libpressio.h"
#include "hermes_shm/compress/libpressio_modes.h"
#endif
#endif

using namespace wrp_cte::compressor;

// Bin fill mode: how to generate values within each bin
enum class BinFillMode {
  CONSTANT,      // All values in burst are identical (bin midpoint)
  LINEAR,        // Linear interpolation across burst
  QUADRATIC,     // Quadratic curve across burst
  SINUSOIDAL,    // Sinusoidal wave across burst
  UNIFORM_RANDOM // Random values within bin range (default)
};

// Bin configuration for palette generator
struct BinConfig {
  double percentage;  // Fraction of data in this bin (0.0-1.0)
  double lo;          // Lower bound of bin range
  double hi;          // Upper bound of bin range
};

// Distribution configuration with parameters
struct DistributionConfig {
  std::string name;         // Full descriptive name
  std::string palette;      // Palette type (uniform, exponential, high_entropy, etc.)
  std::vector<BinConfig> bins;  // Bin configurations (32 bins)
  double perturbation;      // Probability of interleaving (0.0=runs, 1.0=random)
  BinFillMode fill_mode;    // How to fill values within each bin
};

struct CompressorConfig {
  std::string library_name;
  std::string preset_name;
#ifdef HSHM_ENABLE_COMPRESS
  std::unique_ptr<hshm::Compressor> compressor;
  std::unique_ptr<std::mutex> compressor_mutex;  // Thread-safe access to compressor
#endif
};

struct BenchmarkResult {
  std::string library;
  std::string preset;
  std::string data_type;
  std::string distribution;
  size_t data_size;
  double shannon_entropy;
  double mad;
  double first_derivative;
  double second_derivative;
  double compression_ratio;
  double compress_time_ms;
  double decompress_time_ms;
  double psnr;
};

struct Config {
  int num_threads = 1;
  int samples_per_config = 50;
  std::string output_path = "/workspace/context-transfer-engine/compressor/results/compression_results.csv";
  bool skip_compression = false;  // Skip compression, only calculate statistics
  bool verbose = false;
};

void PrintUsage(const char* prog) {
  HLOG(kError, "Usage: {} [options]", prog);
  HLOG(kError, "Options:");
  HLOG(kError, "  --threads <n>        Number of threads [default: 1]");
  HLOG(kError, "  --samples <n>        Samples per compressor config [default: 50]");
  HLOG(kError, "  --output <path>      Output CSV path");
  HLOG(kError, "  --skip-compression   Skip compression tests, only calculate statistics");
  HLOG(kError, "  --verbose            Print progress");
}

Config ParseArgs(int argc, char** argv) {
  Config config;
  static struct option long_options[] = {
      {"threads", required_argument, 0, 't'},
      {"samples", required_argument, 0, 's'},
      {"output", required_argument, 0, 'o'},
      {"skip-compression", no_argument, 0, 'k'},
      {"verbose", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}
  };

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, "t:s:o:kvh", long_options, &option_index)) != -1) {
    switch (opt) {
      case 't': config.num_threads = std::stoi(optarg); break;
      case 's': config.samples_per_config = std::stoi(optarg); break;
      case 'o': config.output_path = optarg; break;
      case 'k': config.skip_compression = true; break;
      case 'v': config.verbose = true; break;
      case 'h': PrintUsage(argv[0]); exit(0);
      default: break;
    }
  }
  return config;
}

// ============================================================================
// Palette Distribution Generator (32-bin system)
// ============================================================================

class PaletteGenerator {
 public:
  // Helper: Create uniform bin percentages (each bin gets equal share)
  static std::vector<double> CreateUniformPercentages(size_t n_bins) {
    return std::vector<double>(n_bins, 1.0 / n_bins);
  }

  // Helper: Create normal-distributed bin percentages (centered on middle bins)
  static std::vector<double> CreateNormalPercentages(size_t n_bins) {
    std::vector<double> pct(n_bins);
    double sum = 0.0;
    for (size_t i = 0; i < n_bins; ++i) {
      double x = -3.0 + 6.0 * i / (n_bins - 1);  // x in [-3, 3]
      pct[i] = std::exp(-0.5 * x * x);  // Gaussian
      sum += pct[i];
    }
    for (size_t i = 0; i < n_bins; ++i) {
      pct[i] /= sum;
    }
    return pct;
  }

  // Helper: Create gamma-distributed bin percentages (skewed toward lower bins)
  static std::vector<double> CreateGammaPercentages(size_t n_bins) {
    std::vector<double> pct(n_bins);
    double sum = 0.0;
    double shape = 2.0;
    for (size_t i = 0; i < n_bins; ++i) {
      double x = 0.1 + 5.9 * i / (n_bins - 1);  // x in [0.1, 6.0]
      pct[i] = std::pow(x, shape - 1.0) * std::exp(-x);
      sum += pct[i];
    }
    for (size_t i = 0; i < n_bins; ++i) {
      pct[i] /= sum;
    }
    return pct;
  }

  // Helper: Create exponential-distributed bin percentages
  static std::vector<double> CreateExponentialPercentages(size_t n_bins) {
    std::vector<double> pct(n_bins);
    pct[0] = 0.9999;  // 99.99% in first bin (ultra-extreme concentration for lowest entropy)
    double remaining = 0.0001;
    double decay = 0.5;
    double sum = 0.0;
    for (size_t i = 1; i < n_bins; ++i) {
      pct[i] = std::pow(decay, static_cast<double>(i));
      sum += pct[i];
    }
    if (sum > 0) {
      for (size_t i = 1; i < n_bins; ++i) {
        pct[i] = pct[i] / sum * remaining;
      }
    }
    return pct;
  }

  // Helper: Create bimodal-distributed bin percentages (two peaks)
  static std::vector<double> CreateBimodalPercentages(size_t n_bins) {
    std::vector<double> pct(n_bins, 0.0);
    if (n_bins < 4) {
      return CreateUniformPercentages(n_bins);
    }

    double peak_weight = 0.4;  // 40% per peak
    pct[0] = peak_weight * 0.6;
    pct[1] = peak_weight * 0.4;
    pct[n_bins - 2] = peak_weight * 0.4;
    pct[n_bins - 1] = peak_weight * 0.6;

    // Middle bins get remaining 20%
    size_t middle_bins = n_bins - 4;
    if (middle_bins > 0) {
      double per_middle = 0.2 / middle_bins;
      for (size_t i = 2; i < n_bins - 2; ++i) {
        pct[i] = per_middle;
      }
    }
    return pct;
  }

  // Helper: Create high-entropy distribution (uniform across all bins)
  static std::vector<double> CreateHighEntropyPercentages(size_t n_bins) {
    return CreateUniformPercentages(n_bins);
  }

  // Helper: Create grayscott-like distribution (bimodal with smooth transitions)
  static std::vector<double> CreateGrayscottPercentages(size_t n_bins) {
    std::vector<double> pct(n_bins, 0.0);
    if (n_bins < 4) {
      return CreateUniformPercentages(n_bins);
    }

    double background = 0.7;  // 70% background
    double spots = 0.2;       // 20% spots
    double edges = 0.1;       // 10% edges

    // Background: concentrated in first few bins
    size_t bg_bins = std::max<size_t>(2, n_bins / 4);
    for (size_t i = 0; i < bg_bins; ++i) {
      double x = static_cast<double>(i) / bg_bins;
      pct[i] = background * std::exp(-2.0 * x * x);
    }

    // Spots: concentrated in last few bins
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

    // Normalize
    double sum = 0.0;
    for (size_t i = 0; i < n_bins; ++i) sum += pct[i];
    if (sum > 0) {
      for (size_t i = 0; i < n_bins; ++i) pct[i] /= sum;
    }
    return pct;
  }

  // Helper: Create high-entropy bins for floats with non-linear spacing
  // Spreads bins across orders of magnitude for maximum byte diversity
  static std::vector<BinConfig> CreateHighEntropyFloatBins(size_t n_bins,
                                                            const std::vector<double>& percentages) {
    std::vector<BinConfig> bins(n_bins);

    // Create bins at exponentially-spaced points for diverse byte patterns
    // Float byte patterns change significantly across orders of magnitude
    for (size_t i = 0; i < n_bins; ++i) {
      bins[i].percentage = percentages[i];

      // Exponential spacing from 1e-6 to 1.0
      double t = static_cast<double>(i) / (n_bins - 1);
      double center = std::pow(10.0, -6.0 + t * 6.0);  // 1e-6 to 1e0
      double width = center * 0.1;  // 10% width around center

      bins[i].lo = std::max(0.0, center - width / 2.0);
      bins[i].hi = std::min(1.0, center + width / 2.0);
    }

    return bins;
  }

  // Helper: Create bin ranges
  static std::vector<BinConfig> CreateBinConfigs(size_t n_bins, double bin_width,
                                                  const std::vector<double>& percentages,
                                                  double value_lo, double value_hi) {
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

  // Generate data using binned distribution with perturbation and fill mode
  template<typename T>
  static void GenerateBinnedData(T* data, size_t num_elements,
                                  const std::vector<BinConfig>& bins, double perturbation,
                                  BinFillMode fill_mode, std::mt19937& gen) {
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
    if (total_assigned < num_elements && n_bins > 0) {
      bin_targets[n_bins - 1] += num_elements - total_assigned;
    }

    // Track active bins
    std::vector<size_t> active_bins;
    for (size_t i = 0; i < n_bins; ++i) {
      if (bin_targets[i] > 0) {
        active_bins.push_back(i);
      }
    }

    if (active_bins.empty()) return;

    auto remove_bin = [&active_bins](size_t bin_idx) {
      active_bins.erase(std::remove(active_bins.begin(), active_bins.end(), bin_idx),
                       active_bins.end());
    };

    auto choose_random_bin = [&]() -> size_t {
      return active_bins[gen() % active_bins.size()];
    };

    // Generate values with burst sampling
    size_t i = 0;
    while (i < num_elements && !active_bins.empty()) {
      size_t current_bin = choose_random_bin();
      size_t remaining = bin_targets[current_bin] - bin_counts[current_bin];

      // Burst size: remaining * (1 - perturbation), at least 1
      size_t burst_size = std::max(static_cast<size_t>(1),
                                   static_cast<size_t>(remaining * (1.0 - perturbation)));
      burst_size = std::min(burst_size, remaining);
      burst_size = std::min(burst_size, num_elements - i);

      // Generate burst_size consecutive values from this bin using fill_mode
      double lo = bins[current_bin].lo;
      double hi = bins[current_bin].hi;
      double mid = (lo + hi) / 2.0;

      for (size_t j = 0; j < burst_size; ++j) {
        double val;

        switch (fill_mode) {
          case BinFillMode::CONSTANT:
            // Use lower bound for maximum value repetition and low entropy
            val = lo;
            break;

          case BinFillMode::LINEAR:
            // Linear interpolation from lo to hi across the burst
            if (burst_size > 1) {
              double t = static_cast<double>(j) / (burst_size - 1);
              val = lo + t * (hi - lo);
            } else {
              val = mid;
            }
            break;

          case BinFillMode::QUADRATIC:
            // Quadratic curve (parabola) across the burst
            if (burst_size > 1) {
              double t = static_cast<double>(j) / (burst_size - 1);
              // Parabola: starts at lo, peaks at mid+(hi-lo)/4, ends at hi
              double a = 4.0 * ((hi - lo) / 4.0) / (0.5 * 0.5);  // Quadratic coefficient
              double phase = t - 0.5;  // Center at t=0.5
              val = mid + a * phase * phase - (hi - lo) / 4.0;
              val = std::clamp(val, std::min(lo, hi), std::max(lo, hi));
            } else {
              val = mid;
            }
            break;

          case BinFillMode::SINUSOIDAL:
            // Sinusoidal wave across the burst (1 complete cycle)
            if (burst_size > 1) {
              double t = static_cast<double>(j) / burst_size;
              val = mid + (hi - lo) / 2.0 * std::sin(2.0 * M_PI * t);
            } else {
              val = mid;
            }
            break;

          case BinFillMode::UNIFORM_RANDOM:
          default:
            // Random value within bin range (original behavior)
            val = (lo == hi) ? lo : lo + prob_dist(gen) * (hi - lo);
            break;
        }

        // Convert to target type
        if constexpr (std::is_same_v<T, uint8_t>) {
          data[i++] = static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
        } else if constexpr (std::is_same_v<T, int32_t>) {
          data[i++] = static_cast<int32_t>(std::clamp(val, -2e9, 2e9));
        } else if constexpr (std::is_same_v<T, float>) {
          data[i++] = static_cast<float>(val);
        }
      }

      bin_counts[current_bin] += burst_size;
      if (bin_counts[current_bin] >= bin_targets[current_bin]) {
        remove_bin(current_bin);
      }
    }
  }
};

// Generate size bins: 32 logarithmic bins (1KB-128KB) + 1MB + 4MB
std::vector<size_t> GenerateSizeBins(bool statistics_only = false) {
  std::vector<size_t> sizes;

  if (statistics_only) {
    // For statistics analysis: test representative sizes
    sizes.push_back(128 * 1024);      // 128KB
    sizes.push_back(1 * 1024 * 1024); // 1MB
    return sizes;
  }

  // For compression benchmarking: test full size range
  size_t min_size = 1024;
  size_t max_size = 128 * 1024;
  int num_bins = 32;

  double log_min = std::log2(static_cast<double>(min_size));
  double log_max = std::log2(static_cast<double>(max_size));
  double log_step = (log_max - log_min) / (num_bins - 1);

  for (int i = 0; i < num_bins; i++) {
    size_t size_bytes = static_cast<size_t>(std::pow(2.0, log_min + i * log_step));
    sizes.push_back(((size_bytes + 255) / 256) * 256);  // Align to 256 bytes
  }

  sizes.push_back(1024 * 1024);      // 1MB
  sizes.push_back(4 * 1024 * 1024);  // 4MB

  return sizes;
}

// Helper: Convert fill mode to string
std::string FillModeToString(BinFillMode mode) {
  switch (mode) {
    case BinFillMode::CONSTANT: return "const";
    case BinFillMode::LINEAR: return "linear";
    case BinFillMode::QUADRATIC: return "quad";
    case BinFillMode::SINUSOIDAL: return "sin";
    case BinFillMode::UNIFORM_RANDOM: return "rand";
    default: return "unknown";
  }
}

// Generate comprehensive distribution configurations
// For statistics-only mode: generate exhaustive combinations
// For compression mode: generate limited set
std::vector<DistributionConfig> GenerateDistributions(bool comprehensive = false) {
  std::vector<DistributionConfig> distributions;

  const size_t n_bins = 32;  // 32-bin palette system

  // Pre-compute percentage distributions
  auto uniform_pct = PaletteGenerator::CreateUniformPercentages(n_bins);
  auto normal_pct = PaletteGenerator::CreateNormalPercentages(n_bins);
  auto gamma_pct = PaletteGenerator::CreateGammaPercentages(n_bins);
  auto exponential_pct = PaletteGenerator::CreateExponentialPercentages(n_bins);
  auto bimodal_pct = PaletteGenerator::CreateBimodalPercentages(n_bins);
  auto grayscott_pct = PaletteGenerator::CreateGrayscottPercentages(n_bins);
  auto high_entropy_pct = PaletteGenerator::CreateHighEntropyPercentages(n_bins);

  std::vector<std::pair<std::string, std::vector<double>>> palette_types = {
      {"uniform", uniform_pct},
      {"normal", normal_pct},
      {"gamma", gamma_pct},
      {"exponential", exponential_pct},
      {"bimodal", bimodal_pct},
      {"grayscott", grayscott_pct},
      {"high_entropy", high_entropy_pct}
  };

  std::vector<double> bin_widths;
  std::vector<double> perturbations;
  std::vector<BinFillMode> fill_modes;

  if (comprehensive) {
    // Comprehensive mode: test all combinations for analysis
    // Smaller bin widths (0.1-0.5) generate low entropy/MAD for Gray-Scott coverage
    bin_widths = {0.1, 0.25, 0.5, 1.0, 4.0, 16.0, 64.0};
    // Extended perturbations to reach higher entropy (up to 0.9 for maximum randomness)
    perturbations = {0.0, 0.05, 0.1, 0.2, 0.325, 0.5, 0.75, 0.9};
    fill_modes = {BinFillMode::CONSTANT, BinFillMode::LINEAR, BinFillMode::QUADRATIC,
                  BinFillMode::SINUSOIDAL, BinFillMode::UNIFORM_RANDOM};
  } else {
    // Compression mode: limited set for training
    bin_widths = {16.0};
    perturbations = {0.0, 0.325};
    fill_modes = {BinFillMode::UNIFORM_RANDOM};
  }

  // Generate all combinations (bins will be scaled per-type later)
  for (const auto& [palette_name, percentages] : palette_types) {
    for (double bin_width : bin_widths) {
      for (double perturbation : perturbations) {
        for (BinFillMode fill_mode : fill_modes) {
          DistributionConfig config;
          config.palette = palette_name;
          // Encode bin_width as int after multiplying by 10 to preserve decimal (0.1→1, 0.25→2, 1.0→10, etc.)
          config.name = palette_name + "_w" + std::to_string(static_cast<int>(bin_width * 10))
                      + "_p" + std::to_string(static_cast<int>(perturbation * 1000))
                      + "_" + FillModeToString(fill_mode);
          // Store normalized bins [0,1] - will be scaled to type-specific ranges
          // Exception: high_entropy palette uses special non-linear float bins
          if (palette_name == "high_entropy") {
            config.bins = PaletteGenerator::CreateHighEntropyFloatBins(n_bins, percentages);
          } else {
            config.bins = PaletteGenerator::CreateBinConfigs(n_bins, bin_width, percentages, 0.0, 1.0);
          }
          config.perturbation = perturbation;
          config.fill_mode = fill_mode;
          distributions.push_back(config);
        }
      }
    }
  }

  return distributions;
}

// Scale bins to type-specific ranges
template<typename T>
std::vector<BinConfig> ScaleBinsForType(const std::vector<BinConfig>& normalized_bins,
                                        const std::string& palette) {
  std::vector<BinConfig> type_bins = normalized_bins;

  // high_entropy palette already has type-specific bins, no scaling needed
  if (palette == "high_entropy" && std::is_same_v<T, float>) {
    return type_bins;
  }

  if constexpr (std::is_same_v<T, uint8_t>) {
    // uint8_t: [0, 255]
    for (auto& bin : type_bins) {
      bin.lo = bin.lo * 255.0;
      bin.hi = bin.hi * 255.0;
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    // int32_t: [-1000000, 1000000]
    for (auto& bin : type_bins) {
      bin.lo = bin.lo * 2e6 - 1e6;
      bin.hi = bin.hi * 2e6 - 1e6;
    }
  } else if constexpr (std::is_same_v<T, float>) {
    // float: [0.0, 1.0] (matches Gray-Scott normalized data)
    // No scaling needed - already in [0,1]
  }

  return type_bins;
}

// Initialize compressor configurations
std::vector<CompressorConfig> InitializeCompressors() {
  std::vector<CompressorConfig> configs;

#ifdef HSHM_ENABLE_COMPRESS
  // ZSTD: 3 levels (fast, balanced, best)
  configs.push_back({"zstd", "fast",
                     std::make_unique<hshm::ZstdWithModes>(hshm::LosslessMode::FAST),
                     std::make_unique<std::mutex>()});
  configs.push_back({"zstd", "balanced",
                     std::make_unique<hshm::ZstdWithModes>(hshm::LosslessMode::BALANCED)});
  configs.push_back({"zstd", "best",
                     std::make_unique<hshm::ZstdWithModes>(hshm::LosslessMode::BEST)});

  // LZ4: 3 levels
  configs.push_back({"lz4", "fast",
                     std::make_unique<hshm::Lz4WithModes>(hshm::LosslessMode::FAST)});
  configs.push_back({"lz4", "balanced",
                     std::make_unique<hshm::Lz4WithModes>(hshm::LosslessMode::BALANCED)});
  configs.push_back({"lz4", "best",
                     std::make_unique<hshm::Lz4WithModes>(hshm::LosslessMode::BEST)});

  // ZLIB: 3 levels
  configs.push_back({"zlib", "fast",
                     std::make_unique<hshm::ZlibWithModes>(hshm::LosslessMode::FAST)});
  configs.push_back({"zlib", "balanced",
                     std::make_unique<hshm::ZlibWithModes>(hshm::LosslessMode::BALANCED)});
  configs.push_back({"zlib", "best",
                     std::make_unique<hshm::ZlibWithModes>(hshm::LosslessMode::BEST)});

  // BZIP2: 3 levels
  configs.push_back({"bzip2", "fast",
                     std::make_unique<hshm::Bzip2WithModes>(hshm::LosslessMode::FAST)});
  configs.push_back({"bzip2", "balanced",
                     std::make_unique<hshm::Bzip2WithModes>(hshm::LosslessMode::BALANCED)});
  configs.push_back({"bzip2", "best",
                     std::make_unique<hshm::Bzip2WithModes>(hshm::LosslessMode::BEST)});

  // LZMA: 3 levels
  configs.push_back({"lzma", "fast",
                     std::make_unique<hshm::LzmaWithModes>(hshm::LosslessMode::FAST)});
  configs.push_back({"lzma", "balanced",
                     std::make_unique<hshm::LzmaWithModes>(hshm::LosslessMode::BALANCED)});
  configs.push_back({"lzma", "best",
                     std::make_unique<hshm::LzmaWithModes>(hshm::LosslessMode::BEST)});

  // BROTLI: 3 levels
  configs.push_back({"brotli", "fast",
                     std::make_unique<hshm::BrotliWithModes>(hshm::LosslessMode::FAST)});
  configs.push_back({"brotli", "balanced",
                     std::make_unique<hshm::BrotliWithModes>(hshm::LosslessMode::BALANCED)});
  configs.push_back({"brotli", "best",
                     std::make_unique<hshm::BrotliWithModes>(hshm::LosslessMode::BEST)});

  // SNAPPY: 1 default config
  configs.push_back({"snappy", "default", std::make_unique<hshm::Snappy>(),
                     std::make_unique<std::mutex>()});

  // Blosc2: 1 default config
  configs.push_back({"blosc2", "default", std::make_unique<hshm::Blosc>(),
                     std::make_unique<std::mutex>()});

#ifdef HSHM_ENABLE_LIBPRESSIO
  // ZFP: 3 modes (fast, balanced, best)
  configs.push_back({"zfp", "fast",
                     std::make_unique<hshm::LibPressioWithModes>("zfp", hshm::CompressionMode::FAST)});
  configs.push_back({"zfp", "balanced",
                     std::make_unique<hshm::LibPressioWithModes>("zfp", hshm::CompressionMode::BALANCED)});
  configs.push_back({"zfp", "best",
                     std::make_unique<hshm::LibPressioWithModes>("zfp", hshm::CompressionMode::BEST)});

  // SZ3: 3 modes
  configs.push_back({"sz3", "fast",
                     std::make_unique<hshm::LibPressioWithModes>("sz3", hshm::CompressionMode::FAST)});
  configs.push_back({"sz3", "balanced",
                     std::make_unique<hshm::LibPressioWithModes>("sz3", hshm::CompressionMode::BALANCED)});
  configs.push_back({"sz3", "best",
                     std::make_unique<hshm::LibPressioWithModes>("sz3", hshm::CompressionMode::BEST)});

  // FPZIP: 3 modes
  configs.push_back({"fpzip", "fast",
                     std::make_unique<hshm::LibPressioWithModes>("fpzip", hshm::CompressionMode::FAST)});
  configs.push_back({"fpzip", "balanced",
                     std::make_unique<hshm::LibPressioWithModes>("fpzip", hshm::CompressionMode::BALANCED)});
  configs.push_back({"fpzip", "best",
                     std::make_unique<hshm::LibPressioWithModes>("fpzip", hshm::CompressionMode::BEST)});
#endif
#endif

  return configs;
}

// Generate data and calculate statistics (shared across all compressors)
template<typename T>
struct GeneratedData {
  std::vector<T> data;
  double shannon_entropy;
  double mad;
  double first_derivative;
  double second_derivative;
};

template<typename T>
GeneratedData<T> GenerateDataWithStats(size_t data_size, const DistributionConfig& dist_config,
                                       std::mt19937& rng) {
  GeneratedData<T> result;
  size_t num_elements = data_size / sizeof(T);
  result.data.resize(num_elements);

  // Scale bins to type-specific ranges
  std::vector<BinConfig> type_bins = ScaleBinsForType<T>(dist_config.bins, dist_config.palette);

  // Generate data
  PaletteGenerator::GenerateBinnedData<T>(result.data.data(), num_elements, type_bins,
                                          dist_config.perturbation, dist_config.fill_mode, rng);

  // Calculate statistics
  typename DataStatistics<T>::StatsTiming timing;
  auto stats = DataStatistics<T>::CalculateAllStatistics(result.data.data(), num_elements, &timing);
  result.shannon_entropy = stats[0];
  result.mad = stats[1];
  result.first_derivative = stats[2];
  result.second_derivative = stats[3];

  return result;
}

// Compress pre-generated data (reuse data across multiple compressors)
template<typename T>
BenchmarkResult CompressData(const std::string& type_name, const GeneratedData<T>& gen_data,
                             const DistributionConfig& dist_config, CompressorConfig& config) {
  BenchmarkResult result;
  result.library = config.library_name;
  result.preset = config.preset_name;
  result.data_type = type_name;
  result.distribution = dist_config.name;
  result.data_size = gen_data.data.size() * sizeof(T);
  result.shannon_entropy = gen_data.shannon_entropy;
  result.mad = gen_data.mad;
  result.first_derivative = gen_data.first_derivative;
  result.second_derivative = gen_data.second_derivative;
  result.compression_ratio = 1.0;
  result.compress_time_ms = 0.0;
  result.decompress_time_ms = 0.0;
  result.psnr = 0.0;

#ifdef HSHM_ENABLE_COMPRESS
  size_t data_size = gen_data.data.size() * sizeof(T);

  // Prepare buffers
  std::vector<uint8_t> compressed(data_size * 2);
  std::vector<uint8_t> decompressed(data_size);

  // Lock compressor for thread-safe access
  std::lock_guard<std::mutex> comp_lock(*config.compressor_mutex);

  // Warmup
  size_t warmup_size = compressed.size();
  config.compressor->Compress(compressed.data(), warmup_size,
                               const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(gen_data.data.data())), data_size);

  // Measure compression
  size_t cmpr_size = compressed.size();
  auto t0 = std::chrono::high_resolution_clock::now();
  bool comp_ok = config.compressor->Compress(compressed.data(), cmpr_size,
                                              const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(gen_data.data.data())), data_size);
  auto t1 = std::chrono::high_resolution_clock::now();

  if (!comp_ok || cmpr_size == 0) {
    return result;
  }

  result.compress_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  result.compression_ratio = static_cast<double>(data_size) / cmpr_size;

  // Measure decompression
  size_t decmpr_size = decompressed.size();
  auto t2 = std::chrono::high_resolution_clock::now();
  bool decomp_ok = config.compressor->Decompress(decompressed.data(), decmpr_size,
                                                  compressed.data(), cmpr_size);
  auto t3 = std::chrono::high_resolution_clock::now();

  if (!decomp_ok || decmpr_size != data_size) {
    return result;
  }

  result.decompress_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  // Calculate PSNR
  result.psnr = DataStatistics<T>::CalculatePSNR(
      gen_data.data.data(), reinterpret_cast<const T*>(decompressed.data()), gen_data.data.size());
#endif

  return result;
}

// Template helper for type-specific benchmarking (used in statistics-only mode)
template<typename T>
BenchmarkResult BenchmarkSampleTyped(const std::string& type_name, size_t data_size,
                                     const DistributionConfig& dist_config,
                                     CompressorConfig& config, std::mt19937& rng) {
  BenchmarkResult result;
  result.library = config.library_name;
  result.preset = config.preset_name;
  result.data_type = type_name;
  result.distribution = dist_config.name;
  result.data_size = data_size;
  result.shannon_entropy = 0.0;
  result.mad = 0.0;
  result.first_derivative = 0.0;
  result.second_derivative = 0.0;
  result.compression_ratio = 1.0;
  result.compress_time_ms = 0.0;
  result.decompress_time_ms = 0.0;
  result.psnr = 0.0;

  size_t num_elements = data_size / sizeof(T);

  // Generate data using palette system
  std::vector<T> data(num_elements);

  // Scale bins to type-specific ranges
  std::vector<BinConfig> type_bins = ScaleBinsForType<T>(dist_config.bins, dist_config.palette);

  PaletteGenerator::GenerateBinnedData<T>(data.data(), num_elements, type_bins,
                                          dist_config.perturbation, dist_config.fill_mode, rng);

  // Calculate all features in minimal passes for performance
  typename DataStatistics<T>::StatsTiming timing;
  auto stats = DataStatistics<T>::CalculateAllStatistics(data.data(), num_elements, &timing);
  result.shannon_entropy = stats[0];
  result.mad = stats[1];
  result.first_derivative = stats[2];
  result.second_derivative = stats[3];

#ifdef HSHM_ENABLE_COMPRESS
  // Prepare buffers
  std::vector<uint8_t> compressed(data_size * 2);
  std::vector<uint8_t> decompressed(data_size);

  // Warmup
  size_t warmup_size = compressed.size();
  config.compressor->Compress(compressed.data(), warmup_size,
                               reinterpret_cast<uint8_t*>(data.data()), data_size);

  // Measure compression
  size_t cmpr_size = compressed.size();
  auto t0 = std::chrono::high_resolution_clock::now();
  bool comp_ok = config.compressor->Compress(compressed.data(), cmpr_size,
                                              reinterpret_cast<uint8_t*>(data.data()), data_size);
  auto t1 = std::chrono::high_resolution_clock::now();

  if (!comp_ok || cmpr_size == 0) {
    return result;
  }

  result.compress_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  result.compression_ratio = static_cast<double>(data_size) / cmpr_size;

  // Measure decompression
  size_t decmpr_size = decompressed.size();
  auto t2 = std::chrono::high_resolution_clock::now();
  bool decomp_ok = config.compressor->Decompress(decompressed.data(), decmpr_size,
                                                  compressed.data(), cmpr_size);
  auto t3 = std::chrono::high_resolution_clock::now();

  if (!decomp_ok || decmpr_size != data_size) {
    return result;
  }

  result.decompress_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  // Calculate PSNR
  result.psnr = DataStatistics<T>::CalculatePSNR(
      data.data(), reinterpret_cast<const T*>(decompressed.data()), num_elements);
#endif

  return result;
}

int main(int argc, char** argv) {
  Config config = ParseArgs(argc, argv);

  HLOG(kInfo, "=== Comprehensive Compression Benchmark ===");
  HLOG(kInfo, "Threads: {}", config.num_threads);
  HLOG(kInfo, "Samples per config: {}", config.samples_per_config);
  HLOG(kInfo, "Output: {}", config.output_path);
  HLOG(kInfo, "Mode: {}", config.skip_compression ? "Statistics Only" : "Full Compression");

  // Initialize
  auto compressor_configs = InitializeCompressors();
  auto size_bins = GenerateSizeBins(config.skip_compression);  // Fewer sizes for statistics mode
  auto distributions = GenerateDistributions(config.skip_compression);  // Comprehensive if skip_compression

  std::vector<std::string> data_types = {"char", "int", "float"};

  if (config.skip_compression) {
    HLOG(kInfo, "Distributions: {} (7 palettes × 7 widths × 8 perturbations × 5 fill modes)",
         distributions.size());
    HLOG(kInfo, "Size bins: {} (128KB + 1MB)", size_bins.size());
  } else {
    HLOG(kInfo, "Compressor configs: {}", compressor_configs.size());
    HLOG(kInfo, "Distributions: {} palettes", distributions.size());
    HLOG(kInfo, "Size bins: {} (1KB-128KB + 1MB + 4MB)", size_bins.size());
  }
  HLOG(kInfo, "Data types: {} (char, int, float)", data_types.size());

  size_t total_samples;
  if (config.skip_compression) {
    // In statistics mode: sample each distribution × data_type × size combination
    total_samples = distributions.size() * data_types.size() * size_bins.size();
  } else {
    // In compression mode: compressor_configs × samples_per_config
    total_samples = compressor_configs.size() * config.samples_per_config;
  }
  HLOG(kInfo, "Total samples: {}", total_samples);
  HLOG(kInfo, "===========================================");

  // Open output file
  std::ofstream out(config.output_path);
  if (!out) {
      HLOG(kError, "Cannot open {}", config.output_path);
    return 1;
  }

  // Write CSV header
  if (config.skip_compression) {
    out << "data_type,data_size,palette,bin_width,perturbation,fill_mode,"
        << "shannon_entropy,mad,first_derivative,second_derivative\n";
  } else {
    out << "library,preset,data_type,data_size,distribution,"
        << "shannon_entropy,mad,first_derivative,second_derivative,"
        << "compression_ratio,compress_time_ms,decompress_time_ms,psnr\n";
  }

  // Thread-safe output and progress tracking
  std::mutex out_mutex;
  std::atomic<size_t> completed{0};
  auto start_time = std::chrono::steady_clock::now();

  // Work queue
  std::queue<size_t> work_queue;
  std::mutex queue_mutex;

  if (config.skip_compression) {
    // Statistics mode: one work item per (distribution, type, size) combination
    for (size_t i = 0; i < total_samples; ++i) {
      work_queue.push(i);
    }
  } else {
    // Compression mode: one work item per compressor
    for (size_t i = 0; i < compressor_configs.size(); ++i) {
      work_queue.push(i);
    }
  }

  // Worker function - handles both statistics-only and compression modes
  auto worker = [&]() {
    std::mt19937 rng(std::random_device{}());

    if (config.skip_compression) {
      // Statistics-only mode: exhaustively test all combinations
      while (true) {
        size_t work_idx;
        {
          std::lock_guard<std::mutex> lock(queue_mutex);
          if (work_queue.empty()) break;
          work_idx = work_queue.front();
          work_queue.pop();
        }

        // Decode work_idx into (dist_idx, type_idx, size_idx)
        size_t num_sizes = size_bins.size();
        size_t num_types = data_types.size();
        size_t dist_idx = work_idx / (num_sizes * num_types);
        size_t rem = work_idx % (num_sizes * num_types);
        size_t type_idx = rem / num_sizes;
        size_t size_idx = rem % num_sizes;

        const auto& dist_config = distributions[dist_idx];
        const std::string& data_type = data_types[type_idx];
        size_t data_size = size_bins[size_idx];

        // Generate data and calculate statistics only (no compression)
        BenchmarkResult result;
        if (data_type == "char") {
          result = BenchmarkSampleTyped<uint8_t>(data_type, data_size, dist_config,
                                                 compressor_configs[0], rng);
        } else if (data_type == "int") {
          result = BenchmarkSampleTyped<int32_t>(data_type, data_size, dist_config,
                                                 compressor_configs[0], rng);
        } else {  // float
          result = BenchmarkSampleTyped<float>(data_type, data_size, dist_config,
                                               compressor_configs[0], rng);
        }

        // Parse distribution name to extract components
        std::string palette = dist_config.name.substr(0, dist_config.name.find("_w"));
        size_t w_pos = dist_config.name.find("_w") + 2;
        size_t p_pos = dist_config.name.find("_p");
        size_t fill_pos = dist_config.name.rfind("_");
        std::string bin_width_str = dist_config.name.substr(w_pos, p_pos - w_pos);
        std::string perturbation_str = dist_config.name.substr(p_pos + 2, fill_pos - p_pos - 2);
        std::string fill_mode = dist_config.name.substr(fill_pos + 1);

        // Convert back to actual values (bin_width was multiplied by 10, perturbation by 1000)
        double bin_width_val = std::stod(bin_width_str) / 10.0;
        double perturbation_val = std::stod(perturbation_str) / 1000.0;

        // Write statistics result
        {
          std::lock_guard<std::mutex> lock(out_mutex);
          out << data_type << ","
              << data_size << ","
              << palette << ","
              << std::fixed << std::setprecision(2) << bin_width_val << ","
              << std::setprecision(3) << perturbation_val << ","
              << fill_mode << ","
              << std::fixed << std::setprecision(6) << result.shannon_entropy << ","
              << std::setprecision(6) << result.mad << ","
              << std::setprecision(6) << result.first_derivative << ","
              << std::setprecision(6) << result.second_derivative << "\n";

          // Progress
          size_t count = ++completed;
          if (count % 100 == 0 || count == total_samples) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            double percent = (count * 100.0) / total_samples;
            double rate = count / (elapsed / 60.0 + 0.001);
            double eta_min = (total_samples - count) / rate;

            HIPRINT("\r[Progress] {}/{} ({}%) | Rate: {} samples/min | ETA: {} min   ",
                    count, total_samples, static_cast<int>(percent),
                    static_cast<int>(rate), static_cast<int>(eta_min));

            if (count == total_samples) {
              HIPRINT("\n");
            }
          }
        }
      }
    } else {
      // Compression mode: random sampling
      while (true) {
        size_t compressor_idx;
        {
          std::lock_guard<std::mutex> lock(queue_mutex);
          if (work_queue.empty()) break;
          compressor_idx = work_queue.front();
          work_queue.pop();
        }

        auto& comp_config = compressor_configs[compressor_idx];

        // Check if lossy compressor (only works with float)
        bool is_lossy = (comp_config.library_name == "zfp" ||
                        comp_config.library_name == "sz3" ||
                        comp_config.library_name == "fpzip");

        // Generate samples for this compressor
        for (int s = 0; s < config.samples_per_config; ++s) {
          // Select random data spec
          size_t data_size = size_bins[rng() % size_bins.size()];
          const auto& dist_config = distributions[rng() % distributions.size()];
          std::string data_type = data_types[rng() % data_types.size()];

          // Lossy compressors only work with float
          if (is_lossy) {
            data_type = "float";
          }

          BenchmarkResult result;

          // Dispatch to correct type
          if (data_type == "char") {
            result = BenchmarkSampleTyped<uint8_t>(data_type, data_size, dist_config,
                                                   comp_config, rng);
          } else if (data_type == "int") {
            result = BenchmarkSampleTyped<int32_t>(data_type, data_size, dist_config,
                                                   comp_config, rng);
          } else {  // float
            result = BenchmarkSampleTyped<float>(data_type, data_size, dist_config,
                                                 comp_config, rng);
          }

          // Write result
          {
            std::lock_guard<std::mutex> lock(out_mutex);
            out << result.library << ","
                << result.preset << ","
                << result.data_type << ","
                << result.data_size << ","
                << result.distribution << ","
                << std::fixed << std::setprecision(6) << result.shannon_entropy << ","
                << std::setprecision(6) << result.mad << ","
                << std::setprecision(6) << result.first_derivative << ","
                << std::setprecision(6) << result.second_derivative << ","
                << std::setprecision(4) << result.compression_ratio << ","
                << std::setprecision(3) << result.compress_time_ms << ","
                << std::setprecision(3) << result.decompress_time_ms << ","
                << std::setprecision(2) << result.psnr << "\n";

            // Progress
            size_t count = ++completed;
            if (count % 100 == 0 || count == total_samples) {
              auto now = std::chrono::steady_clock::now();
              auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
              double percent = (count * 100.0) / total_samples;
              double rate = count / (elapsed / 60.0 + 0.001);
              double eta_min = (total_samples - count) / rate;

              HIPRINT("\r[Progress] {}/{} ({}%) | Rate: {} samples/min | ETA: {} min   ",
                      count, total_samples, static_cast<int>(percent),
                      static_cast<int>(rate), static_cast<int>(eta_min));

              if (count == total_samples) {
                HIPRINT("\n");
              }
            }
          }
        }
      }
    }
  };

  // Launch workers
  std::vector<std::thread> workers;
  for (int t = 0; t < config.num_threads; ++t) {
    workers.emplace_back(worker);
  }

  // Wait for completion
  for (auto& w : workers) {
    w.join();
  }

  out.close();

  auto end_time = std::chrono::steady_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

  HLOG(kInfo, "");
  HLOG(kInfo, "=== Complete ===");
  HLOG(kInfo, "Total samples: {}", completed);
  HLOG(kInfo, "Time: {}s", total_time);
  HLOG(kInfo, "Saved to: {}", config.output_path);

  return 0;
}
