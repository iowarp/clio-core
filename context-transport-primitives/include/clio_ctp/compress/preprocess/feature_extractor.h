/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file feature_extractor.h
 * @brief Unified preprocessing feature extraction for compression selection.
 *
 * Computes statistical features (shannon entropy, MAD, curvature) from raw
 * data buffers and populates CompressionFeatures for predictor models.
 * Delegates all statistics computation to DataStatistics<T>.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_FEATURE_EXTRACTOR_H_
#define CLIO_CTP_COMPRESS_PREPROCESS_FEATURE_EXTRACTOR_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "clio_ctp/compress/preprocess/data_stats.h"
#include "clio_ctp/compress/model/predictor.h"

namespace ctp::compress::preprocess {

/**
 * @brief Enumeration of supported data kinds.
 */
enum class DataKind : uint8_t {
  kChar = 0,    /**< uint8_t or char data */
  kInt = 1,     /**< int32_t data */
  kFloat = 2,   /**< float (32-bit) data */
  kDouble = 3   /**< double (64-bit) data */
};

/**
 * @brief Feature extraction from raw data buffers.
 *
 * Computes statistical features (shannon entropy, MAD, second derivative)
 * from a raw data buffer and populates a CompressionFeatures struct
 * for use by compression prediction models.
 */
class FeatureExtractor {
 public:
  /**
   * @brief Extract statistics and populate features for a data buffer.
   *
   * Determines element count from buffer size and data kind, computes
   * statistics via DataStatistics<T>, and returns populated
   * CompressionFeatures. The returned struct has chunk_size_bytes and
   * data-type one-hots filled in; caller must set library_config_id
   * and target_cpu_util, then optionally call SetLibraryConfig().
   *
   * @param data           Raw data buffer (any type)
   * @param size_bytes     Total buffer size in bytes
   * @param kind           Data kind (kChar, kInt, kFloat, kDouble)
   * @return CompressionFeatures with statistics populated.
   *
   * Example:
   *   float buffer[1000] = {...};
   *   auto features = FeatureExtractor::ExtractFeatures(
   *       buffer, sizeof(buffer), DataKind::kFloat);
   *   SetLibraryConfig(features, 1, 2);  // BZIP2_BALANCED
   */
  static ctp::compress::model::CompressionFeatures ExtractFeatures(
      const void* data,
      size_t size_bytes,
      DataKind kind) {
    ctp::compress::model::CompressionFeatures features;
    features.chunk_size_bytes = static_cast<double>(size_bytes);

    if (!data || size_bytes == 0) {
      return features;
    }

    size_t num_elements = 0;
    switch (kind) {
      case DataKind::kChar:
        num_elements = size_bytes;
        features.data_type_char = 1.0;
        {
          auto* typed_data = static_cast<const uint8_t*>(data);
          features.shannon_entropy =
              ctp::DataStatistics<uint8_t>::CalculateShannonEntropy(
                  typed_data, num_elements);
          features.mad =
              ctp::DataStatistics<uint8_t>::CalculateMAD(
                  typed_data, num_elements);
          features.second_derivative_mean =
              ctp::DataStatistics<uint8_t>::CalculateSecondDerivative(
                  typed_data, num_elements);
        }
        break;

      case DataKind::kInt:
        num_elements = size_bytes / sizeof(int32_t);
        features.data_type_char = 1.0;
        {
          auto* typed_data = static_cast<const int32_t*>(data);
          features.shannon_entropy =
              ctp::DataStatistics<int32_t>::CalculateShannonEntropy(
                  typed_data, num_elements);
          features.mad =
              ctp::DataStatistics<int32_t>::CalculateMAD(
                  typed_data, num_elements);
          features.second_derivative_mean =
              ctp::DataStatistics<int32_t>::CalculateSecondDerivative(
                  typed_data, num_elements);
        }
        break;

      case DataKind::kFloat:
        num_elements = size_bytes / sizeof(float);
        features.data_type_float = 1.0;
        {
          auto* typed_data = static_cast<const float*>(data);
          features.shannon_entropy =
              ctp::DataStatistics<float>::CalculateShannonEntropy(
                  typed_data, num_elements);
          features.mad =
              ctp::DataStatistics<float>::CalculateMAD(
                  typed_data, num_elements);
          features.second_derivative_mean =
              ctp::DataStatistics<float>::CalculateSecondDerivative(
                  typed_data, num_elements);
        }
        break;

      case DataKind::kDouble:
        num_elements = size_bytes / sizeof(double);
        features.data_type_float = 1.0;
        {
          auto* typed_data = static_cast<const double*>(data);
          features.shannon_entropy =
              ctp::DataStatistics<double>::CalculateShannonEntropy(
                  typed_data, num_elements);
          features.mad =
              ctp::DataStatistics<double>::CalculateMAD(
                  typed_data, num_elements);
          features.second_derivative_mean =
              ctp::DataStatistics<double>::CalculateSecondDerivative(
                  typed_data, num_elements);
        }
        break;
    }

    return features;
  }

  /**
   * @brief Extract the data-intrinsic features for bulk ranking.
   *
   * Returns a DataFeatures (statistics + data-type one-hots, no compressor
   * identity) suitable as the fixed input to CompressionPredictor::Rank().
   *
   * @param data       Raw data buffer (any type)
   * @param size_bytes Total buffer size in bytes
   * @param kind       Data kind (kChar, kInt, kFloat, kDouble)
   * @return DataFeatures with statistics populated.
   */
  static ctp::compress::model::DataFeatures ExtractDataFeatures(
      const void* data, size_t size_bytes, DataKind kind) {
    ctp::compress::model::CompressionFeatures f =
        ExtractFeatures(data, size_bytes, kind);
    ctp::compress::model::DataFeatures d;
    d.chunk_size_bytes = f.chunk_size_bytes;
    d.shannon_entropy = f.shannon_entropy;
    d.mad = f.mad;
    d.second_derivative_mean = f.second_derivative_mean;
    d.data_type_char = f.data_type_char;
    d.data_type_float = f.data_type_float;
    return d;
  }

  /**
   * @brief Set library_config_id and preset one-hots.
   *
   * Encodes library_config_id as base_id * 10 + preset_id, and sets the
   * config_fast/balanced/best one-hots based on preset_id:
   * - preset_id 1: config_fast = 1.0
   * - preset_id 2: config_balanced = 1.0
   * - preset_id 3: config_best = 1.0
   *
   * @param features  CompressionFeatures to populate.
   * @param base_id   Compressor library base id (e.g., 1 for BZIP2).
   * @param preset_id Compression preset (1=FAST, 2=BALANCED, 3=BEST).
   */
  static void SetLibraryConfig(
      ctp::compress::model::CompressionFeatures& features,
      int base_id,
      int preset_id) {
    features.library_config_id =
        static_cast<double>(base_id * 10 + preset_id);

    switch (preset_id) {
      case 1:
        features.config_fast = 1.0;
        break;
      case 2:
        features.config_balanced = 1.0;
        break;
      case 3:
        features.config_best = 1.0;
        break;
      default:
        // No preset one-hot set
        break;
    }
  }
};

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_FEATURE_EXTRACTOR_H_
