/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef WRP_CTE_SYNTHETIC_DATA_GENERATOR_H_
#define WRP_CTE_SYNTHETIC_DATA_GENERATOR_H_

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <map>
#include <sstream>

namespace wrp_cte {

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
 * Pattern specification with percentage
 */
struct PatternSpec {
  PatternType type;
  double percentage;  ///< 0.0 - 1.0, fraction of data with this pattern
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

}  // namespace wrp_cte

#endif  // WRP_CTE_SYNTHETIC_DATA_GENERATOR_H_
