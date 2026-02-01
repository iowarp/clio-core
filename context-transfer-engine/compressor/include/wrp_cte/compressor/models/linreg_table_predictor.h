/**
 * @file linreg_table_predictor.h
 * @brief Linear regression table predictor for compression performance
 *
 * This header provides a compression predictor based on a table of linear
 * regressors, indexed by [library][configuration][data_type]. Each regressor
 * uses data_size as input to predict compress_time, decompress_time, and
 * compression_ratio.
 */

#ifndef WRP_CTE_COMPRESSOR_MODELS_LINREG_TABLE_PREDICTOR_H_
#define WRP_CTE_COMPRESSOR_MODELS_LINREG_TABLE_PREDICTOR_H_

#include "compression_features.h"
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <cmath>

namespace wrp_cte::compressor {

/**
 * @brief Linear regression coefficients for a single model
 *
 * y = slope * data_size + intercept
 * Supports multi-output: [compress_time_ms, decompress_time_ms, compress_ratio]
 */
struct LinearRegressionCoeffs {
  double slope_compress_time;     /**< Slope for compress time prediction */
  double intercept_compress_time; /**< Intercept for compress time prediction */
  double slope_decompress_time;   /**< Slope for decompress time prediction */
  double intercept_decompress_time; /**< Intercept for decompress time prediction */
  double slope_compress_ratio;    /**< Slope for compression ratio prediction */
  double intercept_compress_ratio; /**< Intercept for compression ratio prediction */
  size_t sample_count;            /**< Number of samples used to fit model */
  double r2_compress_time;        /**< R² score for compress time */
  double r2_decompress_time;      /**< R² score for decompress time */
  double r2_compress_ratio;       /**< R² score for compression ratio */

  LinearRegressionCoeffs()
      : slope_compress_time(0), intercept_compress_time(0),
        slope_decompress_time(0), intercept_decompress_time(0),
        slope_compress_ratio(0), intercept_compress_ratio(0),
        sample_count(0), r2_compress_time(0), r2_decompress_time(0),
        r2_compress_ratio(0) {}

  /**
   * @brief Predict compress time from data size
   * @param data_size Input data size in bytes
   * @return Predicted compression time in ms
   */
  double PredictCompressTime(double data_size) const {
    return std::max(0.0, slope_compress_time * data_size + intercept_compress_time);
  }

  /**
   * @brief Predict decompress time from data size
   * @param data_size Input data size in bytes
   * @return Predicted decompression time in ms
   */
  double PredictDecompressTime(double data_size) const {
    return std::max(0.0, slope_decompress_time * data_size + intercept_decompress_time);
  }

  /**
   * @brief Predict compression ratio from data size
   * @param data_size Input data size in bytes
   * @return Predicted compression ratio (>1 means compression)
   */
  double PredictCompressRatio(double data_size) const {
    return std::max(1.0, slope_compress_ratio * data_size + intercept_compress_ratio);
  }
};

/**
 * @brief Key for the linear regression lookup table
 *
 * Combines library name, configuration, data type, and distribution into a single key.
 */
struct LinRegTableKey {
  std::string library;      /**< Compression library name (e.g., "ZSTD") */
  std::string config;       /**< Configuration preset (e.g., "fast") */
  std::string data_type;    /**< Data type (e.g., "char", "float", "int") */
  std::string distribution; /**< Data distribution (e.g., "binned_normal_w0_p0") */

  LinRegTableKey() = default;
  LinRegTableKey(const std::string& lib, const std::string& cfg,
                 const std::string& dtype, const std::string& dist = "")
      : library(lib), config(cfg), data_type(dtype), distribution(dist) {}

  bool operator<(const LinRegTableKey& other) const {
    if (library != other.library) return library < other.library;
    if (config != other.config) return config < other.config;
    if (data_type != other.data_type) return data_type < other.data_type;
    return distribution < other.distribution;
  }

  bool operator==(const LinRegTableKey& other) const {
    return library == other.library && config == other.config &&
           data_type == other.data_type && distribution == other.distribution;
  }

  std::string ToString() const {
    if (distribution.empty()) {
      return library + "_" + config + "_" + data_type;
    }
    return library + "_" + config + "_" + data_type + "_" + distribution;
  }
};

/**
 * @brief Configuration for linear regression table predictor
 */
struct LinRegTableConfig {
  double min_samples;             /**< Minimum samples required for valid model */
  double fallback_compress_time;  /**< Fallback compress time if no model */
  double fallback_decompress_time; /**< Fallback decompress time if no model */
  double fallback_compress_ratio; /**< Fallback compression ratio if no model */

  LinRegTableConfig()
      : min_samples(10),
        fallback_compress_time(0.1),
        fallback_decompress_time(0.05),
        fallback_compress_ratio(1.5) {}
};

/**
 * @brief Linear regression table predictor for compression performance
 *
 * Uses a table of linear regressors indexed by [library][config][data_type].
 * Each regressor takes data_size as input and predicts:
 * - compress_time_ms
 * - decompress_time_ms
 * - compression_ratio
 *
 * Features:
 * - Simple linear regression: y = slope * data_size + intercept
 * - Fast O(1) lookup + O(1) prediction
 * - No external dependencies (pure C++)
 * - JSON model format for easy serialization
 * - Trained only on lossless algorithms
 *
 * Example usage:
 * @code
 * LinRegTablePredictor predictor;
 *
 * // Load pre-trained model
 * predictor.Load("/path/to/model_dir");
 *
 * // Predict using library, config, data_type, and data_size
 * auto result = predictor.PredictByKey("ZSTD", "fast", "char", 1048576);
 * std::cout << "Compress time: " << result.compression_time_ms << " ms\n";
 *
 * // Or use CompressionFeatures interface
 * CompressionFeatures features;
 * features.chunk_size_bytes = 1048576;
 * features.library_config_id = GetLibraryId("ZSTD", "fast");
 * features.data_type_char = 1.0;
 * auto result2 = predictor.Predict(features);
 * @endcode
 */
class LinRegTablePredictor : public CompressionPredictor {
 public:
  /**
   * @brief Default constructor
   */
  LinRegTablePredictor();

  /**
   * @brief Constructor with config
   * @param config Predictor configuration
   */
  explicit LinRegTablePredictor(const LinRegTableConfig& config);

  /**
   * @brief Destructor
   */
  ~LinRegTablePredictor() override;

  // Disable copy operations
  LinRegTablePredictor(const LinRegTablePredictor&) = delete;
  LinRegTablePredictor& operator=(const LinRegTablePredictor&) = delete;

  // Enable move operations
  LinRegTablePredictor(LinRegTablePredictor&& other) noexcept;
  LinRegTablePredictor& operator=(LinRegTablePredictor&& other) noexcept;

  /**
   * @brief Load model from a directory
   *
   * Expects the following files:
   * - linreg_table.json: Serialized linear regression table
   * - metadata.json: Model metadata (optional)
   *
   * @param model_dir Directory containing model files
   * @return true if loading succeeded
   */
  bool Load(const std::string& model_dir) override;

  /**
   * @brief Save model to a directory
   * @param model_dir Directory to save model files
   * @return true if saving succeeded
   */
  bool Save(const std::string& model_dir) override;

  /**
   * @brief Check if model is loaded and ready
   * @return true if ready
   */
  bool IsReady() const override;

  /**
   * @brief Predict compression metrics using CompressionFeatures
   *
   * Extracts library, config, data_type from features and performs lookup.
   *
   * @param features Input features (uses chunk_size_bytes, library_config_id)
   * @return Prediction result with all metrics
   */
  CompressionPrediction Predict(const CompressionFeatures& features) override;

  /**
   * @brief Predict compression metrics for a batch
   * @param batch Vector of input features
   * @return Vector of predictions
   */
  std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures>& batch) override;

  /**
   * @brief Direct prediction using library, config, data_type, distribution, and data_size
   *
   * More efficient than Predict() when you have the key components directly.
   *
   * @param library Compression library name (e.g., "ZSTD")
   * @param config Configuration preset (e.g., "fast", "balanced", "best")
   * @param data_type Data type (e.g., "char", "float", "int")
   * @param distribution Data distribution name (e.g., "binned_normal_w0_p0")
   * @param data_size Data size in bytes
   * @return Prediction result
   */
  CompressionPrediction PredictByKey(const std::string& library,
                                     const std::string& config,
                                     const std::string& data_type,
                                     const std::string& distribution,
                                     double data_size);

  /**
   * @brief Direct prediction without distribution (uses fallback)
   *
   * Convenience overload that doesn't require distribution.
   * Will try to find a matching model without distribution, or use fallback.
   *
   * @param library Compression library name
   * @param config Configuration preset
   * @param data_type Data type
   * @param data_size Data size in bytes
   * @return Prediction result
   */
  CompressionPrediction PredictByKey(const std::string& library,
                                     const std::string& config,
                                     const std::string& data_type,
                                     double data_size);

  /**
   * @brief Train linear regression models from data
   *
   * Fits a separate linear regressor for each unique (library, config, data_type)
   * combination. Uses ordinary least squares regression.
   *
   * @param features Vector of training features
   * @param labels Vector of training labels
   * @return true if training succeeded
   */
  bool Train(const std::vector<CompressionFeatures>& features,
             const std::vector<TrainingLabels>& labels) override;

  /**
   * @brief Train from raw CSV data arrays
   *
   * Alternative training interface that takes raw data arrays instead of
   * CompressionFeatures. This is more convenient for batch training from
   * CSV files.
   *
   * @param libraries Vector of library names
   * @param configs Vector of configuration names
   * @param data_types Vector of data type names
   * @param distributions Vector of distribution names
   * @param data_sizes Vector of data sizes
   * @param compress_times Vector of compress times in ms
   * @param decompress_times Vector of decompress times in ms
   * @param compress_ratios Vector of compression ratios
   * @return true if training succeeded
   */
  bool TrainFromRaw(const std::vector<std::string>& libraries,
                    const std::vector<std::string>& configs,
                    const std::vector<std::string>& data_types,
                    const std::vector<std::string>& distributions,
                    const std::vector<double>& data_sizes,
                    const std::vector<double>& compress_times,
                    const std::vector<double>& decompress_times,
                    const std::vector<double>& compress_ratios);

  /**
   * @brief Get the linear regression coefficients for a specific key
   * @param library Library name
   * @param config Configuration name
   * @param data_type Data type name
   * @param distribution Distribution name
   * @return Pointer to coefficients or nullptr if not found
   */
  const LinearRegressionCoeffs* GetCoeffs(const std::string& library,
                                          const std::string& config,
                                          const std::string& data_type,
                                          const std::string& distribution) const;

  /**
   * @brief Get list of supported distributions
   * @return Vector of distribution names
   */
  std::vector<std::string> GetDistributions() const;

  /**
   * @brief Get statistics about the loaded model
   * @return String with model statistics
   */
  std::string GetStatistics() const;

  /**
   * @brief Get number of models in the table
   * @return Number of (library, config, data_type) combinations
   */
  size_t GetNumModels() const { return table_.size(); }

  /**
   * @brief Get list of supported libraries
   * @return Vector of library names
   */
  std::vector<std::string> GetLibraries() const;

  /**
   * @brief Get list of supported configurations
   * @return Vector of configuration names
   */
  std::vector<std::string> GetConfigurations() const;

  /**
   * @brief Get list of supported data types
   * @return Vector of data type names
   */
  std::vector<std::string> GetDataTypes() const;

 private:
  /**
   * @brief Fit a single linear regression model using OLS
   * @param x Vector of input values (data_size)
   * @param y Vector of output values (target metric)
   * @param slope Output slope coefficient
   * @param intercept Output intercept coefficient
   * @param r2 Output R² score
   */
  void FitLinearRegression(const std::vector<double>& x,
                           const std::vector<double>& y,
                           double& slope, double& intercept, double& r2);

  /**
   * @brief Decode library_config_id to library and config names
   * @param library_config_id Encoded library config ID
   * @param library Output library name
   * @param config Output configuration name
   */
  void DecodeLibraryConfigId(int library_config_id,
                             std::string& library, std::string& config) const;

  /**
   * @brief Get data type from one-hot encoded features
   * @param features Input features
   * @return Data type string
   */
  std::string GetDataTypeFromFeatures(const CompressionFeatures& features) const;

  LinRegTableConfig config_;                        /**< Predictor configuration */
  std::map<LinRegTableKey, LinearRegressionCoeffs> table_; /**< Main lookup table */
  bool ready_;                                      /**< Whether model is ready */
  mutable std::mutex mutex_;                        /**< Mutex for thread safety */
  std::map<int, std::pair<std::string, std::string>> id_to_lib_config_; /**< ID decoder */
};

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_MODELS_LINREG_TABLE_PREDICTOR_H_
