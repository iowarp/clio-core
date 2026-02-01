/**
 * @file distribution_classifier.h
 * @brief Mathematical distribution classifier using statistical tests
 *
 * Classifies data into distribution types (normal, uniform, gamma, exponential)
 * using moment analysis and goodness-of-fit metrics without ML.
 */

#ifndef WRP_CTE_COMPRESSOR_MODELS_DISTRIBUTION_CLASSIFIER_H_
#define WRP_CTE_COMPRESSOR_MODELS_DISTRIBUTION_CLASSIFIER_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>
#include "data_stats.h"  // For DataType enum

namespace wrp_cte::compressor {

/**
 * Distribution type enumeration
 */
enum class DistributionType {
  UNKNOWN = 0,
  UNIFORM,      // Flat distribution
  NORMAL,       // Bell curve (Gaussian)
  GAMMA,        // Right-skewed, positive values
  EXPONENTIAL,  // Strongly right-skewed, memoryless
  CONSTANT      // All same value
};

/**
 * Convert DistributionType to string
 */
inline std::string DistributionTypeToString(DistributionType type) {
  switch (type) {
    case DistributionType::UNIFORM: return "uniform";
    case DistributionType::NORMAL: return "normal";
    case DistributionType::GAMMA: return "gamma";
    case DistributionType::EXPONENTIAL: return "exponential";
    case DistributionType::CONSTANT: return "constant";
    default: return "unknown";
  }
}

/**
 * Distribution classification result with confidence scores
 */
struct DistributionResult {
  DistributionType type = DistributionType::UNKNOWN;
  double confidence = 0.0;      // 0.0 to 1.0 confidence in classification

  // Computed moments
  double mean = 0.0;
  double variance = 0.0;
  double skewness = 0.0;        // 3rd standardized moment
  double kurtosis = 0.0;        // 4th standardized moment (excess)

  // Goodness-of-fit scores (lower = better fit)
  double uniform_score = 0.0;
  double normal_score = 0.0;
  double gamma_score = 0.0;
  double exponential_score = 0.0;

  std::string ToString() const {
    return DistributionTypeToString(type);
  }
};

/**
 * DistributionClassifier: Statistical distribution identification
 *
 * Uses moment analysis and histogram-based tests to classify data
 * into one of the standard distribution families.
 *
 * Algorithm:
 * 1. Compute sample moments (mean, variance, skewness, kurtosis)
 * 2. Build normalized histogram
 * 3. Compare to theoretical distributions using chi-squared-like metric
 * 4. Return best-matching distribution with confidence
 */
template<typename T>
class DistributionClassifier {
 public:
  /**
   * Classify the distribution of data
   *
   * @param data Pointer to data buffer
   * @param num_elements Number of elements
   * @param num_bins Number of histogram bins (default: 64)
   * @return DistributionResult with classification and scores
   */
  static DistributionResult Classify(const T* data, size_t num_elements,
                                      size_t num_bins = 64) {
    DistributionResult result;

    if (num_elements < 10) {
      result.type = DistributionType::UNKNOWN;
      return result;
    }

    // Step 1: Compute moments
    ComputeMoments(data, num_elements, result);

    // Check for constant data
    if (result.variance < 1e-10) {
      result.type = DistributionType::CONSTANT;
      result.confidence = 1.0;
      return result;
    }

    // Step 2: Build histogram
    std::vector<double> histogram(num_bins, 0.0);
    double data_min, data_max;
    BuildHistogram(data, num_elements, histogram, data_min, data_max);

    // Normalize histogram to probability distribution
    double total = std::accumulate(histogram.begin(), histogram.end(), 0.0);
    if (total > 0) {
      for (auto& h : histogram) h /= total;
    }

    // Step 3: Compute goodness-of-fit for each distribution
    result.uniform_score = ComputeUniformScore(histogram);
    result.normal_score = ComputeNormalScore(histogram, result.mean,
                                              std::sqrt(result.variance),
                                              data_min, data_max);
    result.gamma_score = ComputeGammaScore(histogram, result.mean,
                                            result.variance, result.skewness,
                                            data_min, data_max);
    result.exponential_score = ComputeExponentialScore(histogram, result.mean,
                                                        data_min, data_max);

    // Step 4: Also use moment-based decision rules
    double moment_score = ClassifyByMoments(result.skewness, result.kurtosis);

    // Step 5: Combine histogram and moment scores to make decision
    SelectBestDistribution(result, moment_score);

    return result;
  }

 private:
  /**
   * Compute sample moments (mean, variance, skewness, kurtosis)
   */
  static void ComputeMoments(const T* data, size_t n, DistributionResult& result) {
    // Mean
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      sum += static_cast<double>(data[i]);
    }
    result.mean = sum / n;

    // Variance and higher moments
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double diff = static_cast<double>(data[i]) - result.mean;
      double diff2 = diff * diff;
      m2 += diff2;
      m3 += diff2 * diff;
      m4 += diff2 * diff2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    result.variance = m2;

    // Skewness: E[(X-μ)³] / σ³
    if (m2 > 1e-10) {
      double sigma = std::sqrt(m2);
      result.skewness = m3 / (sigma * sigma * sigma);
      // Excess kurtosis: E[(X-μ)⁴] / σ⁴ - 3
      result.kurtosis = (m4 / (m2 * m2)) - 3.0;
    }
  }

  /**
   * Build normalized histogram
   */
  static void BuildHistogram(const T* data, size_t n,
                              std::vector<double>& histogram,
                              double& data_min, double& data_max) {
    // Find range
    data_min = static_cast<double>(data[0]);
    data_max = static_cast<double>(data[0]);
    for (size_t i = 1; i < n; ++i) {
      double val = static_cast<double>(data[i]);
      if (val < data_min) data_min = val;
      if (val > data_max) data_max = val;
    }

    // Handle edge case
    if (data_max <= data_min) {
      data_max = data_min + 1.0;
    }

    // Build histogram
    size_t num_bins = histogram.size();
    double bin_width = (data_max - data_min) / num_bins;

    for (size_t i = 0; i < n; ++i) {
      double val = static_cast<double>(data[i]);
      size_t bin = static_cast<size_t>((val - data_min) / bin_width);
      if (bin >= num_bins) bin = num_bins - 1;
      histogram[bin] += 1.0;
    }
  }

  /**
   * Compute chi-squared-like score for uniform distribution
   * Uniform: all bins should have equal probability
   */
  static double ComputeUniformScore(const std::vector<double>& histogram) {
    size_t n = histogram.size();
    double expected = 1.0 / n;
    double chi_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double diff = histogram[i] - expected;
      chi_sq += (diff * diff) / expected;
    }

    return chi_sq;
  }

  /**
   * Compute chi-squared-like score for normal distribution
   */
  static double ComputeNormalScore(const std::vector<double>& histogram,
                                    double mean, double sigma,
                                    double data_min, double data_max) {
    size_t n = histogram.size();
    double bin_width = (data_max - data_min) / n;
    double chi_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double bin_center = data_min + (i + 0.5) * bin_width;
      double z = (bin_center - mean) / sigma;
      // Normal PDF: 1/(σ√2π) * exp(-z²/2)
      double expected = std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * M_PI));
      expected *= bin_width;  // Convert PDF to probability in bin

      if (expected < 1e-10) expected = 1e-10;
      double diff = histogram[i] - expected;
      chi_sq += (diff * diff) / expected;
    }

    return chi_sq;
  }

  /**
   * Compute chi-squared-like score for gamma distribution
   * Uses method of moments to estimate shape (k) and scale (θ)
   */
  static double ComputeGammaScore(const std::vector<double>& histogram,
                                   double mean, double variance, double skewness,
                                   double data_min, double data_max) {
    // Gamma only defined for positive values
    if (data_min < 0 && std::abs(data_min) > 0.1 * (data_max - data_min)) {
      return 1e10;  // Poor fit for negative data
    }

    // Estimate gamma parameters from moments
    // mean = k*θ, variance = k*θ², so k = mean²/variance, θ = variance/mean
    if (mean <= 0 || variance <= 0) return 1e10;

    double k = (mean * mean) / variance;  // Shape
    double theta = variance / mean;        // Scale

    if (k < 0.1 || k > 100) return 1e10;  // Invalid shape

    size_t n = histogram.size();
    double bin_width = (data_max - data_min) / n;
    double chi_sq = 0.0;

    // Precompute log(Gamma(k)) using Stirling approximation for large k
    double log_gamma_k = (k > 10) ?
        (k - 0.5) * std::log(k) - k + 0.5 * std::log(2 * M_PI) :
        std::lgamma(k);

    for (size_t i = 0; i < n; ++i) {
      double bin_center = data_min + (i + 0.5) * bin_width;
      double x = bin_center - data_min;  // Shift to make minimum = 0

      if (x <= 0) continue;

      // Gamma PDF: x^(k-1) * exp(-x/θ) / (θ^k * Gamma(k))
      double log_pdf = (k - 1) * std::log(x) - x / theta - k * std::log(theta) - log_gamma_k;
      double expected = std::exp(log_pdf) * bin_width;

      if (expected < 1e-10) expected = 1e-10;
      double diff = histogram[i] - expected;
      chi_sq += (diff * diff) / expected;
    }

    return chi_sq;
  }

  /**
   * Compute chi-squared-like score for exponential distribution
   */
  static double ComputeExponentialScore(const std::vector<double>& histogram,
                                         double mean,
                                         double data_min, double data_max) {
    // Exponential only defined for positive values
    if (data_min < 0 && std::abs(data_min) > 0.1 * (data_max - data_min)) {
      return 1e10;
    }

    if (mean <= 0) return 1e10;

    // Exponential rate λ = 1/mean
    double lambda = 1.0 / mean;

    size_t n = histogram.size();
    double bin_width = (data_max - data_min) / n;
    double chi_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {
      double bin_center = data_min + (i + 0.5) * bin_width;
      double x = bin_center - data_min;  // Shift to make minimum = 0

      // Exponential PDF: λ * exp(-λx)
      double expected = lambda * std::exp(-lambda * x) * bin_width;

      if (expected < 1e-10) expected = 1e-10;
      double diff = histogram[i] - expected;
      chi_sq += (diff * diff) / expected;
    }

    return chi_sq;
  }

  /**
   * Classify using moment-based decision rules
   * Returns: 0=uniform, 1=normal, 2=gamma, 3=exponential
   *
   * Thresholds calibrated on synthetic binned distributions:
   *   Observed moments:
   *   - Uniform:     skew≈0.1,  kurt≈-0.75
   *   - Normal:      skew≈0.17, kurt≈-0.30
   *   - Gamma:       skew≈0.66, kurt≈-0.02
   *   - Exponential: skew≈2.9,  kurt≈8.2
   */
  static double ClassifyByMoments(double skewness, double kurtosis) {
    // Exponential: strongly right-skewed with high kurtosis
    if (skewness > 1.5) {
      return 3.0;  // Exponential
    }

    // Gamma: moderately right-skewed
    if (skewness > 0.4) {
      return 2.0;  // Gamma
    }

    // Uniform vs Normal: both have near-zero skewness
    // Uniform has more negative kurtosis
    if (kurtosis < -0.5) {
      return 0.0;  // Uniform
    }

    return 1.0;  // Normal (default for symmetric, mesokurtic)
  }

  /**
   * Select best distribution based on combined scores
   */
  static void SelectBestDistribution(DistributionResult& result, double moment_hint) {
    // Normalize scores (lower = better)
    double scores[4] = {
      result.uniform_score,
      result.normal_score,
      result.gamma_score,
      result.exponential_score
    };

    // Find minimum score
    double min_score = scores[0];
    int best_idx = 0;
    for (int i = 1; i < 4; ++i) {
      if (scores[i] < min_score) {
        min_score = scores[i];
        best_idx = i;
      }
    }

    // Apply moment-based weighting
    // If moment analysis strongly suggests a type, boost its score
    int moment_idx = static_cast<int>(moment_hint);
    if (moment_idx >= 0 && moment_idx < 4) {
      // If histogram and moments agree, increase confidence
      if (moment_idx == best_idx) {
        result.confidence = 0.9;
      } else {
        // Slight disagreement
        result.confidence = 0.6;
        // Use moment hint if histogram score is close
        if (scores[moment_idx] < min_score * 2.0) {
          best_idx = moment_idx;
        }
      }
    }

    // Set distribution type
    switch (best_idx) {
      case 0: result.type = DistributionType::UNIFORM; break;
      case 1: result.type = DistributionType::NORMAL; break;
      case 2: result.type = DistributionType::GAMMA; break;
      case 3: result.type = DistributionType::EXPONENTIAL; break;
      default: result.type = DistributionType::UNKNOWN; break;
    }

    // Compute confidence based on score separation
    double second_best = 1e10;
    for (int i = 0; i < 4; ++i) {
      if (i != best_idx && scores[i] < second_best) {
        second_best = scores[i];
      }
    }

    if (min_score > 0 && second_best < 1e9) {
      double ratio = second_best / min_score;
      // ratio > 2 means clear winner, ratio ~ 1 means uncertain
      result.confidence = std::min(0.95, 0.5 + 0.25 * std::log(ratio));
    }
  }
};

/**
 * Type-erased factory for DistributionClassifier
 */
class DistributionClassifierFactory {
 public:
  static DistributionResult Classify(const void* data, size_t num_elements,
                                      DataType type, size_t num_bins = 64) {
    switch (type) {
      case DataType::UINT8:
        return DistributionClassifier<uint8_t>::Classify(
            static_cast<const uint8_t*>(data), num_elements, num_bins);
      case DataType::INT32:
        return DistributionClassifier<int32_t>::Classify(
            static_cast<const int32_t*>(data), num_elements, num_bins);
      case DataType::FLOAT32:
        return DistributionClassifier<float>::Classify(
            static_cast<const float*>(data), num_elements, num_bins);
      case DataType::DOUBLE64:
        return DistributionClassifier<double>::Classify(
            static_cast<const double*>(data), num_elements, num_bins);
      default:
        return DistributionResult();
    }
  }
};

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_MODELS_DISTRIBUTION_CLASSIFIER_H_
