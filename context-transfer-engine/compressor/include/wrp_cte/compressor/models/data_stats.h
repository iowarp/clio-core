//
// Created by lukemartinlogan on 1/27/26.
//

#ifndef WRP_CTE_COMPRESSOR_MODELS_DATA_STATS_H_
#define WRP_CTE_COMPRESSOR_MODELS_DATA_STATS_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>
#include <type_traits>
#include <random>

namespace wrp_cte::compressor {

/**
 * Data type enumeration for statistics calculations
 */
enum class DataType {
  UINT8,
  INT32,
  FLOAT32,
  DOUBLE64
};

/**
 * DataStatistics: Templated statistics calculator for compression prediction
 *
 * This class provides statistical measures used for:
 * 1. Compression benchmark analysis
 * 2. Runtime compression library selection (CTE)
 * 3. ML model feature extraction
 *
 * Template parameter T: The actual data type (uint8_t, int32_t, float, double)
 *
 * Statistics included:
 * - Shannon Entropy: Information entropy (0-8 bits for byte data)
 * - MAD (Mean Absolute Deviation): Data variability measure
 * - MSE (Mean Squared Error): Reconstruction error measure
 * - PSNR (Peak Signal-to-Noise Ratio): Quality metric for lossy compression
 * - Derivative Stats: Rate of change measures
 */
template<typename T>
class DataStatistics {
 public:
  /**
   * Calculate Shannon Entropy
   *
   * Measures the information content / randomness of data
   * Range: 0.0 (all same value) to 8.0 (uniform random) for byte data
   *
   * @param data Pointer to data buffer
   * @param num_elements Number of elements (NOT bytes)
   * @return Shannon entropy in bits
   */
  static double CalculateShannonEntropy(const T* data, size_t num_elements) {
    if (num_elements == 0) return 0.0;

    // Convert to bytes for histogram
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    size_t num_bytes = num_elements * sizeof(T);

    // Build histogram
    std::vector<size_t> histogram(256, 0);
    for (size_t i = 0; i < num_bytes; i++) {
      histogram[bytes[i]]++;
    }

    // Calculate entropy
    double entropy = 0.0;
    for (size_t i = 0; i < 256; i++) {
      if (histogram[i] > 0) {
        double p_i = static_cast<double>(histogram[i]) / static_cast<double>(num_bytes);
        entropy += -p_i * std::log2(p_i);
      }
    }
    return entropy;
  }

  /**
   * Calculate Mean Absolute Deviation (MAD)
   *
   * Measures the average distance of data points from the mean
   * Lower MAD = more clustered data (potentially more compressible)
   *
   * @param data Pointer to data buffer
   * @param num_elements Number of elements
   * @return Mean absolute deviation
   */
  static double CalculateMAD(const T* data, size_t num_elements) {
    if (num_elements == 0) return 0.0;

    // Calculate mean
    double mean = 0.0;
    for (size_t i = 0; i < num_elements; i++) {
      mean += static_cast<double>(data[i]);
    }
    mean /= static_cast<double>(num_elements);

    // Calculate MAD
    double sum_abs_dev = 0.0;
    for (size_t i = 0; i < num_elements; i++) {
      double diff = std::abs(static_cast<double>(data[i]) - mean);
      sum_abs_dev += diff;
    }
    return sum_abs_dev / static_cast<double>(num_elements);
  }

  /**
   * Calculate Mean Squared Error (MSE)
   *
   * Measures the average squared difference between original and reconstructed data
   * Used for quality assessment and PSNR calculation
   *
   * @param original Pointer to original data buffer
   * @param reconstructed Pointer to reconstructed data buffer
   * @param num_elements Number of elements
   * @return Mean squared error
   */
  static double CalculateMSE(const T* original, const T* reconstructed, size_t num_elements) {
    if (num_elements == 0) return 0.0;

    double mse = 0.0;
    for (size_t i = 0; i < num_elements; i++) {
      double diff = static_cast<double>(original[i]) - static_cast<double>(reconstructed[i]);
      mse += diff * diff;
    }
    mse /= static_cast<double>(num_elements);

    return mse;
  }

  /**
   * Calculate PSNR (Peak Signal-to-Noise Ratio)
   *
   * Measures reconstruction quality for lossy compression
   * Convention: 0.0 = perfect match (lossless, represents infinity)
   *            >0.0 = lossy compression quality in dB
   *
   * Higher PSNR = better quality (less distortion)
   * Typical values: 30-50 dB = good quality, >50 dB = excellent
   *
   * @param original Pointer to original data buffer
   * @param reconstructed Pointer to reconstructed data buffer
   * @param num_elements Number of elements
   * @return PSNR in dB (0.0 for perfect match/lossless)
   */
  static double CalculatePSNR(const T* original, const T* reconstructed, size_t num_elements) {
    if (num_elements == 0) return 0.0;

    double mse = CalculateMSE(original, reconstructed, num_elements);

    // Perfect match (lossless compression)
    if (mse < 1e-10) {
      return 0.0;  // Represents infinity/perfect quality
    }

    // Calculate data range for MAX value
    T data_min = original[0];
    T data_max = original[0];
    for (size_t i = 1; i < num_elements; i++) {
      if (original[i] < data_min) data_min = original[i];
      if (original[i] > data_max) data_max = original[i];
    }

    double range = static_cast<double>(data_max) - static_cast<double>(data_min);

    // For integer types, use type-specific MAX
    if constexpr (std::is_same_v<T, uint8_t>) {
      // For uint8_t, MAX = 255
      double psnr = 10.0 * std::log10(255.0 * 255.0 / mse);
      return (psnr > 0.0) ? psnr : 0.0;
    } else {
      // For other types, use data range
      if (range < 1e-10) range = 1.0;  // Avoid division by zero
      double psnr = 10.0 * std::log10(range * range / mse);
      return (psnr > 0.0) ? psnr : 0.0;
    }
  }

  /**
   * Calculate first-order derivative statistics
   *
   * Measures the rate of change / smoothness of data
   * Lower values = smoother data (potentially more compressible)
   *
   * @param data Pointer to data buffer
   * @param num_elements Number of elements
   * @return Mean absolute first derivative
   */
  static double CalculateFirstDerivative(const T* data, size_t num_elements) {
    if (num_elements < 2) return 0.0;

    double sum_abs_diff = 0.0;
    for (size_t i = 1; i < num_elements; i++) {
      double diff = std::abs(static_cast<double>(data[i]) - static_cast<double>(data[i-1]));
      sum_abs_diff += diff;
    }
    return sum_abs_diff / static_cast<double>(num_elements - 1);
  }

  /**
   * Calculate second-order derivative statistics
   *
   * Measures the curvature / acceleration of data changes
   * Second derivative at i = data[i+1] - 2*data[i] + data[i-1]
   * Lower values = smoother transitions (more predictable, better for delta encoding)
   *
   * @param data Pointer to data buffer
   * @param num_elements Number of elements
   * @return Mean absolute second derivative
   */
  static double CalculateSecondDerivative(const T* data, size_t num_elements) {
    if (num_elements < 3) return 0.0;

    double sum_abs_second_diff = 0.0;
    for (size_t i = 1; i < num_elements - 1; i++) {
      // Second derivative: d2[i] = data[i+1] - 2*data[i] + data[i-1]
      double second_diff = static_cast<double>(data[i+1])
                         - 2.0 * static_cast<double>(data[i])
                         + static_cast<double>(data[i-1]);
      sum_abs_second_diff += std::abs(second_diff);
    }
    return sum_abs_second_diff / static_cast<double>(num_elements - 2);
  }

  /**
   * Calculate byte frequency variance
   *
   * Measures how uniform the byte distribution is
   * Lower values = more uniform (less compressible)
   * Higher values = clustered distribution (more compressible)
   *
   * @param data Pointer to data buffer
   * @param num_elements Number of elements
   * @return Variance of byte frequencies
   */
  static double CalculateByteFrequencyVariance(const T* data, size_t num_elements) {
    if (num_elements == 0) return 0.0;

    // Convert to bytes
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    size_t num_bytes = num_elements * sizeof(T);

    // Build histogram
    std::vector<size_t> histogram(256, 0);
    for (size_t i = 0; i < num_bytes; i++) {
      histogram[bytes[i]]++;
    }

    // Calculate mean frequency
    double mean_freq = static_cast<double>(num_bytes) / 256.0;

    // Calculate variance
    double variance = 0.0;
    for (size_t i = 0; i < 256; i++) {
      double diff = static_cast<double>(histogram[i]) - mean_freq;
      variance += diff * diff;
    }
    variance /= 256.0;

    return variance;
  }

  /**
   * Calculate run-length potential
   *
   * Measures the maximum length of consecutive repeated values
   * Higher values = more repetition (better for RLE-style compression)
   *
   * @param data Pointer to data buffer
   * @param num_elements Number of elements
   * @return Maximum run length found
   */
  static size_t CalculateMaxRunLength(const T* data, size_t num_elements) {
    if (num_elements == 0) return 0;

    size_t max_run = 1;
    size_t current_run = 1;
    T last_value = data[0];

    for (size_t i = 1; i < num_elements; i++) {
      if (data[i] == last_value) {
        current_run++;
        if (current_run > max_run) {
          max_run = current_run;
        }
      } else {
        current_run = 1;
        last_value = data[i];
      }
    }

    return max_run;
  }

  /**
   * Calculate zero ratio
   *
   * Measures the fraction of zero bytes in the data
   * Higher values = more zeros (sparse data, highly compressible)
   *
   * @param data Pointer to data buffer
   * @param num_elements Number of elements
   * @return Ratio of zero bytes (0.0 to 1.0)
   */
  static double CalculateZeroRatio(const T* data, size_t num_elements) {
    if (num_elements == 0) return 0.0;

    // Convert to bytes
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    size_t num_bytes = num_elements * sizeof(T);

    size_t zero_count = 0;
    for (size_t i = 0; i < num_bytes; i++) {
      if (bytes[i] == 0) {
        zero_count++;
      }
    }

    return static_cast<double>(zero_count) / static_cast<double>(num_bytes);
  }

  /**
   * Bundle: Calculate all basic features for ML prediction
   *
   * Returns the core features needed for compression prediction models
   * This is the standard feature set used by CTE runtime and benchmarks
   *
   * @param data Pointer to data buffer
   * @param num_elements Number of elements
   * @return Vector of [shannon_entropy, MAD, zero_ratio, first_derivative]
   */
  static std::vector<double> CalculateCompressionFeatures(const T* data, size_t num_elements) {
    std::vector<double> features;
    features.reserve(4);

    features.push_back(CalculateShannonEntropy(data, num_elements));
    features.push_back(CalculateMAD(data, num_elements));
    features.push_back(CalculateZeroRatio(data, num_elements));
    features.push_back(CalculateFirstDerivative(data, num_elements));

    return features;
  }
};

/**
 * DataStatisticsFactory: Type-erased interface for DataStatistics
 *
 * Allows calling DataStatistics methods without knowing the type at compile time.
 * Uses DataType enumeration to dispatch to the appropriate templated implementation.
 */
class DataStatisticsFactory {
 public:
  /**
   * Calculate Shannon Entropy (type-erased)
   */
  static double CalculateShannonEntropy(const void* data, size_t num_elements, DataType type) {
    switch (type) {
      case DataType::UINT8:
        return DataStatistics<uint8_t>::CalculateShannonEntropy(
            static_cast<const uint8_t*>(data), num_elements);
      case DataType::INT32:
        return DataStatistics<int32_t>::CalculateShannonEntropy(
            static_cast<const int32_t*>(data), num_elements);
      case DataType::FLOAT32:
        return DataStatistics<float>::CalculateShannonEntropy(
            static_cast<const float*>(data), num_elements);
      case DataType::DOUBLE64:
        return DataStatistics<double>::CalculateShannonEntropy(
            static_cast<const double*>(data), num_elements);
      default:
        return 0.0;
    }
  }

  /**
   * Calculate MAD (type-erased)
   */
  static double CalculateMAD(const void* data, size_t num_elements, DataType type) {
    switch (type) {
      case DataType::UINT8:
        return DataStatistics<uint8_t>::CalculateMAD(
            static_cast<const uint8_t*>(data), num_elements);
      case DataType::INT32:
        return DataStatistics<int32_t>::CalculateMAD(
            static_cast<const int32_t*>(data), num_elements);
      case DataType::FLOAT32:
        return DataStatistics<float>::CalculateMAD(
            static_cast<const float*>(data), num_elements);
      case DataType::DOUBLE64:
        return DataStatistics<double>::CalculateMAD(
            static_cast<const double*>(data), num_elements);
      default:
        return 0.0;
    }
  }

  /**
   * Calculate MSE (type-erased)
   */
  static double CalculateMSE(const void* original, const void* reconstructed,
                              size_t num_elements, DataType type) {
    switch (type) {
      case DataType::UINT8:
        return DataStatistics<uint8_t>::CalculateMSE(
            static_cast<const uint8_t*>(original),
            static_cast<const uint8_t*>(reconstructed), num_elements);
      case DataType::INT32:
        return DataStatistics<int32_t>::CalculateMSE(
            static_cast<const int32_t*>(original),
            static_cast<const int32_t*>(reconstructed), num_elements);
      case DataType::FLOAT32:
        return DataStatistics<float>::CalculateMSE(
            static_cast<const float*>(original),
            static_cast<const float*>(reconstructed), num_elements);
      case DataType::DOUBLE64:
        return DataStatistics<double>::CalculateMSE(
            static_cast<const double*>(original),
            static_cast<const double*>(reconstructed), num_elements);
      default:
        return 0.0;
    }
  }

  /**
   * Calculate PSNR (type-erased)
   */
  static double CalculatePSNR(const void* original, const void* reconstructed,
                               size_t num_elements, DataType type) {
    switch (type) {
      case DataType::UINT8:
        return DataStatistics<uint8_t>::CalculatePSNR(
            static_cast<const uint8_t*>(original),
            static_cast<const uint8_t*>(reconstructed), num_elements);
      case DataType::INT32:
        return DataStatistics<int32_t>::CalculatePSNR(
            static_cast<const int32_t*>(original),
            static_cast<const int32_t*>(reconstructed), num_elements);
      case DataType::FLOAT32:
        return DataStatistics<float>::CalculatePSNR(
            static_cast<const float*>(original),
            static_cast<const float*>(reconstructed), num_elements);
      case DataType::DOUBLE64:
        return DataStatistics<double>::CalculatePSNR(
            static_cast<const double*>(original),
            static_cast<const double*>(reconstructed), num_elements);
      default:
        return 0.0;
    }
  }

  /**
   * Calculate first derivative (type-erased)
   */
  static double CalculateFirstDerivative(const void* data, size_t num_elements, DataType type) {
    switch (type) {
      case DataType::UINT8:
        return DataStatistics<uint8_t>::CalculateFirstDerivative(
            static_cast<const uint8_t*>(data), num_elements);
      case DataType::INT32:
        return DataStatistics<int32_t>::CalculateFirstDerivative(
            static_cast<const int32_t*>(data), num_elements);
      case DataType::FLOAT32:
        return DataStatistics<float>::CalculateFirstDerivative(
            static_cast<const float*>(data), num_elements);
      case DataType::DOUBLE64:
        return DataStatistics<double>::CalculateFirstDerivative(
            static_cast<const double*>(data), num_elements);
      default:
        return 0.0;
    }
  }

  /**
   * Calculate second derivative (type-erased)
   */
  static double CalculateSecondDerivative(const void* data, size_t num_elements, DataType type) {
    switch (type) {
      case DataType::UINT8:
        return DataStatistics<uint8_t>::CalculateSecondDerivative(
            static_cast<const uint8_t*>(data), num_elements);
      case DataType::INT32:
        return DataStatistics<int32_t>::CalculateSecondDerivative(
            static_cast<const int32_t*>(data), num_elements);
      case DataType::FLOAT32:
        return DataStatistics<float>::CalculateSecondDerivative(
            static_cast<const float*>(data), num_elements);
      case DataType::DOUBLE64:
        return DataStatistics<double>::CalculateSecondDerivative(
            static_cast<const double*>(data), num_elements);
      default:
        return 0.0;
    }
  }

  /**
   * Calculate compression features bundle (type-erased)
   */
  static std::vector<double> CalculateCompressionFeatures(const void* data,
                                                            size_t num_elements,
                                                            DataType type) {
    switch (type) {
      case DataType::UINT8:
        return DataStatistics<uint8_t>::CalculateCompressionFeatures(
            static_cast<const uint8_t*>(data), num_elements);
      case DataType::INT32:
        return DataStatistics<int32_t>::CalculateCompressionFeatures(
            static_cast<const int32_t*>(data), num_elements);
      case DataType::FLOAT32:
        return DataStatistics<float>::CalculateCompressionFeatures(
            static_cast<const float*>(data), num_elements);
      case DataType::DOUBLE64:
        return DataStatistics<double>::CalculateCompressionFeatures(
            static_cast<const double*>(data), num_elements);
      default:
        return std::vector<double>();
    }
  }

  /**
   * Get DataType from string
   */
  static DataType GetDataType(const std::string& type_name) {
    if (type_name == "char" || type_name == "uint8" || type_name == "byte") {
      return DataType::UINT8;
    } else if (type_name == "int" || type_name == "int32") {
      return DataType::INT32;
    } else if (type_name == "float" || type_name == "float32") {
      return DataType::FLOAT32;
    } else if (type_name == "double" || type_name == "float64") {
      return DataType::DOUBLE64;
    }
    return DataType::UINT8;  // Default
  }

  /**
   * Get type size in bytes
   */
  static size_t GetTypeSize(DataType type) {
    switch (type) {
      case DataType::UINT8: return 1;
      case DataType::INT32: return 4;
      case DataType::FLOAT32: return 4;
      case DataType::DOUBLE64: return 8;
      default: return 1;
    }
  }
};

/**
 * BlockSamplingStats: Features computed from random block sampling
 *
 * Used for fast distribution classification without scanning entire data.
 * Samples N random blocks and aggregates statistics.
 */
struct BlockSamplingStats {
  // Entropy statistics across blocks
  double entropy_mean = 0.0;
  double entropy_std = 0.0;
  double entropy_min = 0.0;
  double entropy_max = 0.0;

  // MAD statistics across blocks
  double mad_mean = 0.0;
  double mad_std = 0.0;

  // Higher-order moments (computed from block means)
  double skewness = 0.0;    // Asymmetry of distribution
  double kurtosis = 0.0;    // Tail heaviness

  // Block-level derivatives
  double deriv1_mean = 0.0;  // Mean first derivative
  double deriv1_std = 0.0;   // Std of first derivative

  // Value range statistics
  double value_range = 0.0;     // max - min across all blocks
  double value_concentration = 0.0;  // Fraction of values in densest bin

  // Sampling metadata
  size_t num_blocks = 0;
  size_t block_size = 0;
  size_t total_samples = 0;
};

/**
 * BlockSampler: Random block sampling for fast distribution classification
 *
 * Samples N random blocks from data and computes aggregate statistics.
 * This enables fast runtime distribution detection with minimal overhead.
 *
 * Usage:
 *   BlockSampler<float> sampler;
 *   auto stats = sampler.Sample(data, num_elements, num_blocks, block_size);
 */
template<typename T>
class BlockSampler {
 public:
  /**
   * Sample random blocks and compute aggregate statistics
   *
   * @param data Pointer to data buffer
   * @param num_elements Total number of elements
   * @param num_blocks Number of blocks to sample (default: 16)
   * @param block_size Elements per block (default: 64)
   * @param seed Random seed (0 = use random device)
   * @return BlockSamplingStats with computed features
   */
  static BlockSamplingStats Sample(const T* data, size_t num_elements,
                                    size_t num_blocks = 16,
                                    size_t block_size = 64,
                                    uint32_t seed = 0) {
    BlockSamplingStats stats;

    if (num_elements == 0 || num_blocks == 0 || block_size == 0) {
      return stats;
    }

    // Ensure we don't sample more than available
    if (block_size > num_elements) {
      block_size = num_elements;
    }

    // Maximum valid start position for a block
    size_t max_start = (num_elements > block_size) ? (num_elements - block_size) : 0;

    // Initialize RNG
    std::mt19937 gen;
    if (seed == 0) {
      std::random_device rd;
      gen.seed(rd());
    } else {
      gen.seed(seed);
    }
    std::uniform_int_distribution<size_t> dist(0, max_start);

    // Storage for per-block statistics
    std::vector<double> block_entropies;
    std::vector<double> block_mads;
    std::vector<double> block_means;
    std::vector<double> block_derivs;
    block_entropies.reserve(num_blocks);
    block_mads.reserve(num_blocks);
    block_means.reserve(num_blocks);
    block_derivs.reserve(num_blocks);

    // Track global min/max
    T global_min = data[0];
    T global_max = data[0];

    // Build value histogram for concentration measure
    std::vector<size_t> value_histogram(256, 0);
    size_t total_byte_samples = 0;

    // Sample blocks
    for (size_t b = 0; b < num_blocks; ++b) {
      size_t start = dist(gen);
      const T* block = data + start;

      // Compute block entropy
      double entropy = DataStatistics<T>::CalculateShannonEntropy(block, block_size);
      block_entropies.push_back(entropy);

      // Compute block MAD
      double mad = DataStatistics<T>::CalculateMAD(block, block_size);
      block_mads.push_back(mad);

      // Compute block mean
      double mean = 0.0;
      for (size_t i = 0; i < block_size; ++i) {
        mean += static_cast<double>(block[i]);
        // Track global min/max
        if (block[i] < global_min) global_min = block[i];
        if (block[i] > global_max) global_max = block[i];
      }
      mean /= static_cast<double>(block_size);
      block_means.push_back(mean);

      // Compute block first derivative mean
      double deriv = DataStatistics<T>::CalculateFirstDerivative(block, block_size);
      block_derivs.push_back(deriv);

      // Add to byte histogram for concentration
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(block);
      size_t num_bytes = block_size * sizeof(T);
      for (size_t i = 0; i < num_bytes; ++i) {
        value_histogram[bytes[i]]++;
      }
      total_byte_samples += num_bytes;
    }

    stats.num_blocks = num_blocks;
    stats.block_size = block_size;
    stats.total_samples = num_blocks * block_size;

    // Compute entropy statistics
    stats.entropy_mean = ComputeMean(block_entropies);
    stats.entropy_std = ComputeStd(block_entropies, stats.entropy_mean);
    stats.entropy_min = *std::min_element(block_entropies.begin(), block_entropies.end());
    stats.entropy_max = *std::max_element(block_entropies.begin(), block_entropies.end());

    // Compute MAD statistics
    stats.mad_mean = ComputeMean(block_mads);
    stats.mad_std = ComputeStd(block_mads, stats.mad_mean);

    // Compute derivative statistics
    stats.deriv1_mean = ComputeMean(block_derivs);
    stats.deriv1_std = ComputeStd(block_derivs, stats.deriv1_mean);

    // Compute skewness and kurtosis from block means
    stats.skewness = ComputeSkewness(block_means);
    stats.kurtosis = ComputeKurtosis(block_means);

    // Value range
    stats.value_range = static_cast<double>(global_max) - static_cast<double>(global_min);

    // Value concentration: fraction in most common byte value
    if (total_byte_samples > 0) {
      size_t max_count = *std::max_element(value_histogram.begin(), value_histogram.end());
      stats.value_concentration = static_cast<double>(max_count) / static_cast<double>(total_byte_samples);
    }

    return stats;
  }

 private:
  static double ComputeMean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / static_cast<double>(values.size());
  }

  static double ComputeStd(const std::vector<double>& values, double mean) {
    if (values.size() < 2) return 0.0;
    double sum_sq = 0.0;
    for (double v : values) {
      double diff = v - mean;
      sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / static_cast<double>(values.size() - 1));
  }

  static double ComputeSkewness(const std::vector<double>& values) {
    if (values.size() < 3) return 0.0;

    double mean = ComputeMean(values);
    double n = static_cast<double>(values.size());

    double m2 = 0.0, m3 = 0.0;
    for (double v : values) {
      double diff = v - mean;
      m2 += diff * diff;
      m3 += diff * diff * diff;
    }
    m2 /= n;
    m3 /= n;

    if (m2 < 1e-10) return 0.0;
    return m3 / std::pow(m2, 1.5);
  }

  static double ComputeKurtosis(const std::vector<double>& values) {
    if (values.size() < 4) return 0.0;

    double mean = ComputeMean(values);
    double n = static_cast<double>(values.size());

    double m2 = 0.0, m4 = 0.0;
    for (double v : values) {
      double diff = v - mean;
      double diff2 = diff * diff;
      m2 += diff2;
      m4 += diff2 * diff2;
    }
    m2 /= n;
    m4 /= n;

    if (m2 < 1e-10) return 0.0;
    // Excess kurtosis (normal distribution = 0)
    return (m4 / (m2 * m2)) - 3.0;
  }
};

/**
 * BlockSamplerFactory: Type-erased interface for BlockSampler
 */
class BlockSamplerFactory {
 public:
  static BlockSamplingStats Sample(const void* data, size_t num_elements,
                                    DataType type,
                                    size_t num_blocks = 16,
                                    size_t block_size = 64,
                                    uint32_t seed = 0) {
    switch (type) {
      case DataType::UINT8:
        return BlockSampler<uint8_t>::Sample(
            static_cast<const uint8_t*>(data), num_elements,
            num_blocks, block_size, seed);
      case DataType::INT32:
        return BlockSampler<int32_t>::Sample(
            static_cast<const int32_t*>(data), num_elements,
            num_blocks, block_size, seed);
      case DataType::FLOAT32:
        return BlockSampler<float>::Sample(
            static_cast<const float*>(data), num_elements,
            num_blocks, block_size, seed);
      case DataType::DOUBLE64:
        return BlockSampler<double>::Sample(
            static_cast<const double*>(data), num_elements,
            num_blocks, block_size, seed);
      default:
        return BlockSamplingStats();
    }
  }
};

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_MODELS_DATA_STATS_H_
