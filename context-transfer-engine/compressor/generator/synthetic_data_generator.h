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

#ifndef CLIO_CTE_SYNTHETIC_DATA_GENERATOR_H_
#define CLIO_CTE_SYNTHETIC_DATA_GENERATOR_H_

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <map>
#include <sstream>

namespace clio::cte {

/**
 * Data pattern types for synthetic workload generation
 */
enum class PatternType {
  kUniform,      ///< Uniform random distribution
  kGaussian,     ///< Normal/Gaussian distribution
  kConstant,     ///< All same values (maximally compressible)
  kGradient,     ///< Linear gradient
  kSinusoidal,   ///< Sinusoidal wave pattern
  kRepeating,    ///< Repeating pattern
  kGrayscott,    ///< Gray-Scott simulation-like bimodal distribution
  kBimodal,      ///< Generic bimodal distribution
  kExponential   ///< Exponential distribution
};

/**
 * Base distribution types for feature-targeted generation
 */
enum class BaseDistribution {
  kNormal,       ///< Normal/Gaussian distribution
  kGamma,        ///< Gamma distribution
  kExponential,  ///< Exponential distribution
  kUniform       ///< Uniform distribution
};

/**
 * Feature target ranges for synthetic data generation
 */
struct FeatureTargets {
  double shannon_entropy_min;      ///< Target minimum Shannon entropy (bits)
  double shannon_entropy_max;      ///< Target maximum Shannon entropy (bits)
  double mad_min;                  ///< Target minimum Mean Absolute Deviation
  double mad_max;                  ///< Target maximum Mean Absolute Deviation
  double second_deriv_min;         ///< Target minimum 2nd derivative
  double second_deriv_max;         ///< Target maximum 2nd derivative

  FeatureTargets()
      : shannon_entropy_min(6.5), shannon_entropy_max(7.6),
        mad_min(0.07), mad_max(0.16),
        second_deriv_min(0.05), second_deriv_max(0.32) {}

  FeatureTargets(double entropy_min, double entropy_max,
                 double mad_min_, double mad_max_,
                 double deriv_min, double deriv_max)
      : shannon_entropy_min(entropy_min), shannon_entropy_max(entropy_max),
        mad_min(mad_min_), mad_max(mad_max_),
        second_deriv_min(deriv_min), second_deriv_max(deriv_max) {}

  // Gray-Scott pattern-influenced targets with wider margins for 80% coverage
  // These are wider than exact pattern ranges but narrower than generic targets
  static FeatureTargets GrayscottLowEntropy() {
    // Covers Spots pattern area with margin
    return FeatureTargets(6.50, 7.20, 0.065, 0.090, 0.075, 0.130);
  }

  static FeatureTargets GrayscottHighEntropySmooth() {
    // Covers Stripes pattern area with margin
    return FeatureTargets(7.20, 7.70, 0.075, 0.100, 0.050, 0.080);
  }

  static FeatureTargets GrayscottMediumEntropyRough() {
    // Covers Coral pattern area with margin
    return FeatureTargets(6.70, 7.40, 0.105, 0.145, 0.070, 0.100);
  }

  static FeatureTargets GrayscottHighEntropySharp() {
    // Covers Mitosis pattern area with margin
    return FeatureTargets(7.30, 7.70, 0.130, 0.175, 0.250, 0.350);
  }

  // Comprehensive Gray-Scott target covering all patterns
  static FeatureTargets GrayscottComprehensive() {
    return FeatureTargets(6.60, 7.65, 0.065, 0.170, 0.050, 0.320);
  }

  // Tight Gray-Scott target for 80% coverage (tighter Shannon range)
  static FeatureTargets GrayscottTight() {
    return FeatureTargets(6.70, 7.60, 0.065, 0.165, 0.055, 0.320);
  }
};

/**
 * Pattern specification with percentage
 */
struct PatternSpec {
  PatternType type;
  double percentage;  ///< 0.0 - 1.0, fraction of data with this pattern
};

/**
 * Feature-targeted data generator
 *
 * Generates data from base distributions (normal, gamma, exponential)
 * and transforms it to achieve target Shannon entropy, MAD, and 2nd derivative ranges.
 */
class FeatureTargetedGenerator {
 public:
  /**
   * Generate data targeting specific feature ranges
   * @param data Output buffer (float array)
   * @param num_elements Number of elements to generate
   * @param base_dist Base distribution type (normal, gamma, exponential, uniform)
   * @param targets Target feature ranges
   * @param seed Random seed
   * @param max_iterations Maximum refinement iterations (default: 5)
   * @return True if targets were achieved within tolerance
   */
  static bool GenerateWithTargets(float* data, size_t num_elements,
                                   BaseDistribution base_dist,
                                   const FeatureTargets& targets,
                                   unsigned int seed = 42,
                                   int max_iterations = 5) {
    std::mt19937 gen(seed);

    // Step 1: Generate base distribution
    GenerateBaseDistribution(data, num_elements, base_dist, gen);

    // Step 2: Target MAD (most critical for Gray-Scott match)
    double target_mad = targets.mad_min +
        (targets.mad_max - targets.mad_min) * UniformRandom(gen);
    AdjustMAD(data, num_elements, target_mad);

    // Step 3: Target Shannon entropy
    double target_entropy = targets.shannon_entropy_min +
        (targets.shannon_entropy_max - targets.shannon_entropy_min) * UniformRandom(gen);
    AdjustEntropy(data, num_elements, target_entropy, gen, max_iterations);

    // Step 4: Target 2nd derivative
    double target_deriv = targets.second_deriv_min +
        (targets.second_deriv_max - targets.second_deriv_min) * UniformRandom(gen);
    AdjustSecondDerivative(data, num_elements, target_deriv, gen);

    // Verify targets were achieved (within 20% tolerance)
    double actual_mad = ComputeMAD(data, num_elements);
    double actual_entropy = ComputeEntropy(data, num_elements);
    double actual_deriv = ComputeSecondDerivative(data, num_elements);

    bool mad_ok = (actual_mad >= targets.mad_min * 0.8) &&
                  (actual_mad <= targets.mad_max * 1.2);
    bool entropy_ok = (actual_entropy >= targets.shannon_entropy_min * 0.9) &&
                      (actual_entropy <= targets.shannon_entropy_max * 1.1);
    bool deriv_ok = (actual_deriv >= targets.second_deriv_min * 0.8) &&
                    (actual_deriv <= targets.second_deriv_max * 1.2);

    return mad_ok && entropy_ok && deriv_ok;
  }

 private:
  static double UniformRandom(std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
  }

  static void GenerateBaseDistribution(float* data, size_t n,
                                        BaseDistribution dist_type,
                                        std::mt19937& gen) {
    switch (dist_type) {
      case BaseDistribution::kNormal: {
        std::normal_distribution<float> dist(0.5f, 0.2f);
        for (size_t i = 0; i < n; i++) {
          data[i] = std::clamp(dist(gen), 0.0f, 1.0f);
        }
        break;
      }
      case BaseDistribution::kGamma: {
        std::gamma_distribution<float> dist(2.0f, 0.3f);
        for (size_t i = 0; i < n; i++) {
          data[i] = std::clamp(dist(gen), 0.0f, 1.0f);
        }
        break;
      }
      case BaseDistribution::kExponential: {
        std::exponential_distribution<float> dist(3.0f);
        for (size_t i = 0; i < n; i++) {
          data[i] = std::clamp(dist(gen), 0.0f, 1.0f);
        }
        break;
      }
      case BaseDistribution::kUniform: {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < n; i++) {
          data[i] = dist(gen);
        }
        break;
      }
    }
  }

  static double ComputeMean(const float* data, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
      sum += data[i];
    }
    return sum / n;
  }

  static double ComputeMAD(const float* data, size_t n) {
    double mean = ComputeMean(data, n);
    double mad = 0.0;
    for (size_t i = 0; i < n; i++) {
      mad += std::abs(data[i] - mean);
    }
    return mad / n;
  }

  static double ComputeEntropy(const float* data, size_t n, int bins = 256) {
    // Build histogram
    std::vector<int> hist(bins, 0);
    for (size_t i = 0; i < n; i++) {
      int bin = static_cast<int>(data[i] * (bins - 1));
      bin = std::clamp(bin, 0, bins - 1);
      hist[bin]++;
    }

    // Compute entropy in bits
    double entropy = 0.0;
    for (int count : hist) {
      if (count > 0) {
        double prob = static_cast<double>(count) / n;
        entropy -= prob * std::log2(prob);
      }
    }
    return entropy;
  }

  static double ComputeSecondDerivative(const float* data, size_t n) {
    if (n < 3) return 0.0;

    double sum = 0.0;
    for (size_t i = 1; i < n - 1; i++) {
      double second_deriv = std::abs(data[i+1] - 2*data[i] + data[i-1]);
      sum += second_deriv;
    }
    return sum / (n - 2);
  }

  static void AdjustMAD(float* data, size_t n, double target_mad) {
    double current_mean = ComputeMean(data, n);
    double current_mad = ComputeMAD(data, n);

    if (current_mad < 1e-6) {
      // Data is too uniform, can't scale meaningfully
      return;
    }

    // Scale factor to achieve target MAD
    double scale = target_mad / current_mad;

    // Center data around mean, scale, and shift back
    for (size_t i = 0; i < n; i++) {
      data[i] = current_mean + (data[i] - current_mean) * scale;
      data[i] = std::clamp(data[i], 0.0f, 1.0f);
    }
  }

  static void AdjustEntropy(float* data, size_t n, double target_entropy,
                            std::mt19937& gen, int max_iterations) {
    // Entropy adjustment through noise addition or smoothing
    double current_entropy = ComputeEntropy(data, n);

    for (int iter = 0; iter < max_iterations; iter++) {
      if (std::abs(current_entropy - target_entropy) < 0.3) {
        break;  // Close enough
      }

      if (current_entropy < target_entropy) {
        // Add noise to increase entropy
        std::normal_distribution<float> noise(0.0f, 0.03f);
        for (size_t i = 0; i < n; i++) {
          data[i] = std::clamp(data[i] + noise(gen), 0.0f, 1.0f);
        }
      } else {
        // Apply slight smoothing to reduce entropy
        std::vector<float> temp(n);
        for (size_t i = 1; i < n - 1; i++) {
          temp[i] = 0.25f * data[i-1] + 0.5f * data[i] + 0.25f * data[i+1];
        }
        temp[0] = data[0];
        temp[n-1] = data[n-1];
        for (size_t i = 0; i < n; i++) {
          data[i] = temp[i];
        }
      }

      current_entropy = ComputeEntropy(data, n);
    }
  }

  static void AdjustSecondDerivative(float* data, size_t n, double target_deriv,
                                     std::mt19937& gen) {
    double current_deriv = ComputeSecondDerivative(data, n);

    if (current_deriv < target_deriv) {
      // Add high-frequency noise to increase curvature
      std::normal_distribution<float> noise(0.0f, 0.05f);
      for (size_t i = 0; i < n; i++) {
        data[i] = std::clamp(data[i] + noise(gen), 0.0f, 1.0f);
      }
    } else if (current_deriv > target_deriv * 1.5) {
      // Apply smoothing to reduce curvature
      std::vector<float> temp(n);
      for (size_t i = 1; i < n - 1; i++) {
        temp[i] = 0.25f * data[i-1] + 0.5f * data[i] + 0.25f * data[i+1];
      }
      temp[0] = data[0];
      temp[n-1] = data[n-1];
      for (size_t i = 0; i < n; i++) {
        data[i] = temp[i];
      }
    }
  }
};

/**
 * Synthetic data generator for compression benchmarking
 *
 * Generates data with various patterns that simulate scientific simulation output.
 * Patterns can be mixed with specified percentages to create realistic workloads.
 *
 * Usage:
 * @code
 *   std::vector<PatternSpec> patterns = {
 *       {PatternType::kGrayscott, 0.7},
 *       {PatternType::kGaussian, 0.2},
 *       {PatternType::kUniform, 0.1}
 *   };
 *   std::vector<float> data(1024 * 1024);
 *   SyntheticDataGenerator::GenerateMixedData(data.data(), data.size(), patterns, 0, 0);
 * @endcode
 */
class SyntheticDataGenerator {
 public:
  /**
   * Generate data with mixed patterns according to specifications
   * @param data Output buffer (float array)
   * @param num_elements Number of float elements to generate
   * @param patterns Vector of pattern specifications with percentages
   * @param seed_offset Offset for random seed (e.g., MPI rank)
   * @param iteration Iteration number for varying random state
   */
  static void GenerateMixedData(void* data, size_t num_elements,
                                 const std::vector<PatternSpec>& patterns,
                                 int seed_offset = 0, int iteration = 0) {
    float* floats = static_cast<float*>(data);
    std::mt19937 gen(seed_offset * 1000 + iteration);

    size_t offset = 0;
    for (const auto& spec : patterns) {
      size_t count = static_cast<size_t>(num_elements * spec.percentage);
      if (offset + count > num_elements) {
        count = num_elements - offset;
      }

      switch (spec.type) {
        case PatternType::kUniform:
          GenerateUniform(floats + offset, count, gen);
          break;
        case PatternType::kGaussian:
          GenerateGaussian(floats + offset, count, gen);
          break;
        case PatternType::kConstant:
          GenerateConstant(floats + offset, count);
          break;
        case PatternType::kGradient:
          GenerateGradient(floats + offset, count);
          break;
        case PatternType::kSinusoidal:
          GenerateSinusoidal(floats + offset, count);
          break;
        case PatternType::kRepeating:
          GenerateRepeating(floats + offset, count);
          break;
        case PatternType::kGrayscott:
          GenerateGrayscott(floats + offset, count, gen);
          break;
        case PatternType::kBimodal:
          GenerateBimodal(floats + offset, count, gen);
          break;
        case PatternType::kExponential:
          GenerateExponential(floats + offset, count, gen);
          break;
      }
      offset += count;
    }

    // Fill remaining with zeros if any
    while (offset < num_elements) {
      floats[offset++] = 0.0f;
    }
  }

  /**
   * Generate single-pattern data
   * @param data Output buffer
   * @param num_elements Number of elements
   * @param type Pattern type
   * @param seed Random seed
   */
  static void GenerateSinglePattern(void* data, size_t num_elements,
                                     PatternType type, unsigned int seed = 42) {
    std::vector<PatternSpec> patterns = {{type, 1.0}};
    GenerateMixedData(data, num_elements, patterns, seed, 0);
  }

  /**
   * Parse pattern specification string
   * Format: "<pattern1>:<percent1>,<pattern2>:<percent2>,..."
   * Example: "grayscott:70,gaussian:20,uniform:10"
   * @param spec Pattern specification string
   * @return Vector of pattern specifications
   */
  static std::vector<PatternSpec> ParsePatternSpec(const std::string& spec) {
    std::vector<PatternSpec> patterns;
    static const std::map<std::string, PatternType> pattern_map = {
        {"uniform", PatternType::kUniform},
        {"gaussian", PatternType::kGaussian},
        {"constant", PatternType::kConstant},
        {"gradient", PatternType::kGradient},
        {"sinusoidal", PatternType::kSinusoidal},
        {"repeating", PatternType::kRepeating},
        {"grayscott", PatternType::kGrayscott},
        {"bimodal", PatternType::kBimodal},
        {"exponential", PatternType::kExponential}
    };

    std::stringstream ss(spec);
    std::string item;
    double total_percent = 0.0;

    while (std::getline(ss, item, ',')) {
      size_t colon_pos = item.find(':');
      if (colon_pos == std::string::npos) continue;

      std::string name = item.substr(0, colon_pos);
      double percent = std::stod(item.substr(colon_pos + 1)) / 100.0;

      auto it = pattern_map.find(name);
      if (it != pattern_map.end()) {
        patterns.push_back({it->second, percent});
        total_percent += percent;
      }
    }

    // Normalize percentages if they don't sum to 1.0
    if (total_percent > 0 && std::abs(total_percent - 1.0) > 0.01) {
      for (auto& p : patterns) {
        p.percentage /= total_percent;
      }
    }

    return patterns;
  }

  /**
   * Get pattern type from string name
   * @param name Pattern name
   * @return Pattern type (kUniform if not found)
   */
  static PatternType GetPatternType(const std::string& name) {
    static const std::map<std::string, PatternType> pattern_map = {
        {"uniform", PatternType::kUniform},
        {"gaussian", PatternType::kGaussian},
        {"constant", PatternType::kConstant},
        {"gradient", PatternType::kGradient},
        {"sinusoidal", PatternType::kSinusoidal},
        {"repeating", PatternType::kRepeating},
        {"grayscott", PatternType::kGrayscott},
        {"bimodal", PatternType::kBimodal},
        {"exponential", PatternType::kExponential}
    };

    auto it = pattern_map.find(name);
    return (it != pattern_map.end()) ? it->second : PatternType::kUniform;
  }

  /**
   * Get pattern name from type
   * @param type Pattern type
   * @return Pattern name string
   */
  static std::string GetPatternName(PatternType type) {
    static const std::map<PatternType, std::string> name_map = {
        {PatternType::kUniform, "uniform"},
        {PatternType::kGaussian, "gaussian"},
        {PatternType::kConstant, "constant"},
        {PatternType::kGradient, "gradient"},
        {PatternType::kSinusoidal, "sinusoidal"},
        {PatternType::kRepeating, "repeating"},
        {PatternType::kGrayscott, "grayscott"},
        {PatternType::kBimodal, "bimodal"},
        {PatternType::kExponential, "exponential"}
    };

    auto it = name_map.find(type);
    return (it != name_map.end()) ? it->second : "unknown";
  }

 private:
  static void GenerateUniform(float* data, size_t count, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count; i++) {
      data[i] = dist(gen);
    }
  }

  static void GenerateGaussian(float* data, size_t count, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.5f, 0.15f);
    for (size_t i = 0; i < count; i++) {
      data[i] = std::clamp(dist(gen), 0.0f, 1.0f);
    }
  }

  static void GenerateConstant(float* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
      data[i] = 0.5f;
    }
  }

  static void GenerateGradient(float* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
      data[i] = static_cast<float>(i) / count;
    }
  }

  static void GenerateSinusoidal(float* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
      data[i] = 0.5f + 0.5f * std::sin(2.0f * M_PI * i / 256.0f);
    }
  }

  static void GenerateRepeating(float* data, size_t count) {
    const float pattern[] = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.7f, 0.5f, 0.3f};
    const size_t pattern_len = sizeof(pattern) / sizeof(pattern[0]);
    for (size_t i = 0; i < count; i++) {
      data[i] = pattern[i % pattern_len];
    }
  }

  /**
   * Gray-Scott like distribution
   * Models reaction-diffusion patterns:
   * - ~70% background (low concentration values)
   * - ~20% spots (high concentration values)
   * - ~10% edges/transitions
   */
  static void GenerateGrayscott(float* data, size_t count, std::mt19937& gen) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::normal_distribution<float> background(0.1f, 0.02f);
    std::normal_distribution<float> spots(0.9f, 0.03f);
    std::uniform_real_distribution<float> edges(0.3f, 0.7f);

    for (size_t i = 0; i < count; i++) {
      float p = prob(gen);
      if (p < 0.70f) {
        // Background region (low values)
        data[i] = std::clamp(background(gen), 0.0f, 1.0f);
      } else if (p < 0.90f) {
        // Spot region (high values)
        data[i] = std::clamp(spots(gen), 0.0f, 1.0f);
      } else {
        // Edge/transition region
        data[i] = edges(gen);
      }
    }
  }

  static void GenerateBimodal(float* data, size_t count, std::mt19937& gen) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::normal_distribution<float> low(0.2f, 0.05f);
    std::normal_distribution<float> high(0.8f, 0.05f);

    for (size_t i = 0; i < count; i++) {
      if (prob(gen) < 0.5f) {
        data[i] = std::clamp(low(gen), 0.0f, 1.0f);
      } else {
        data[i] = std::clamp(high(gen), 0.0f, 1.0f);
      }
    }
  }

  static void GenerateExponential(float* data, size_t count, std::mt19937& gen) {
    std::exponential_distribution<float> dist(3.0f);
    for (size_t i = 0; i < count; i++) {
      data[i] = std::clamp(dist(gen), 0.0f, 1.0f);
    }
  }
};

}  // namespace clio_cte

#endif  // CLIO_CTE_SYNTHETIC_DATA_GENERATOR_H_
