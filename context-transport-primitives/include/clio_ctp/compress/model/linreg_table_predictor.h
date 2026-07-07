/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file linreg_table_predictor.h
 * @brief Linear regression table predictor for compression performance.
 *
 * Uses a table of linear regressors indexed by [library][config][data_type].
 * Each regressor predicts compress_time, decompress_time, and compression_ratio
 * from data_size. Implements the unified CompressionPredictor interface.
 */

#ifndef CLIO_CTP_COMPRESS_MODEL_LINREG_TABLE_PREDICTOR_H_
#define CLIO_CTP_COMPRESS_MODEL_LINREG_TABLE_PREDICTOR_H_

#include "clio_ctp/compress/model/predictor.h"
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ctp::compress::model {

/**
 * @brief Linear regression coefficients for a single model.
 *
 * Implements y = slope * data_size + intercept for three outputs:
 * compress_time_ms, decompress_time_ms, compression_ratio.
 */
struct LinearRegressionCoeffs {
  double slope_compress_time;
  double intercept_compress_time;
  double slope_decompress_time;
  double intercept_decompress_time;
  double slope_compress_ratio;
  double intercept_compress_ratio;
  size_t sample_count;
  double r2_compress_time;
  double r2_decompress_time;
  double r2_compress_ratio;

  LinearRegressionCoeffs()
    : slope_compress_time(0), intercept_compress_time(0),
      slope_decompress_time(0), intercept_decompress_time(0),
      slope_compress_ratio(0), intercept_compress_ratio(0),
      sample_count(0), r2_compress_time(0), r2_decompress_time(0),
      r2_compress_ratio(0) {}

  /// @brief Predict compress time from data size.
  double PredictCompressTime(double data_size) const {
    return std::max(0.0, slope_compress_time * data_size +
                            intercept_compress_time);
  }

  /// @brief Predict decompress time from data size.
  double PredictDecompressTime(double data_size) const {
    return std::max(0.0, slope_decompress_time * data_size +
                            intercept_decompress_time);
  }

  /// @brief Predict compression ratio from data size.
  double PredictCompressRatio(double data_size) const {
    return std::max(1.0, slope_compress_ratio * data_size +
                            intercept_compress_ratio);
  }
};

/**
 * @brief Key for linear regression lookup table.
 *
 * Combines library, config, data_type, and distribution into a single key.
 */
struct LinRegTableKey {
  std::string library;
  std::string config;
  std::string data_type;
  std::string distribution;

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
 * @brief Configuration for linear regression table predictor.
 */
struct LinRegTableConfig {
  double min_samples;
  double fallback_compress_time;
  double fallback_decompress_time;
  double fallback_compress_ratio;

  LinRegTableConfig()
    : min_samples(10),
      fallback_compress_time(0.1),
      fallback_decompress_time(0.05),
      fallback_compress_ratio(1.5) {}
};

/**
 * @brief Linear regression table predictor for compression performance.
 *
 * Implements CompressionPredictor interface. Uses OLS regression per
 * (library, config, data_type) to predict compression metrics from data_size.
 */
class LinRegTablePredictor : public CompressionPredictor {
 public:
  /// @brief Default constructor.
  LinRegTablePredictor();

  /// @brief Constructor with config.
  explicit LinRegTablePredictor(const LinRegTableConfig& config);

  /// @brief Destructor.
  ~LinRegTablePredictor() override;

  LinRegTablePredictor(const LinRegTablePredictor&) = delete;
  LinRegTablePredictor& operator=(const LinRegTablePredictor&) = delete;

  LinRegTablePredictor(LinRegTablePredictor&& other) noexcept;
  LinRegTablePredictor& operator=(LinRegTablePredictor&& other) noexcept;

  /// @brief Load model from directory (linreg_table.csv).
  bool Load(const std::string& model_dir) override;

  /// @brief Save model to directory.
  bool Save(const std::string& model_dir) override;

  /// @brief Check if model is loaded and ready.
  bool IsReady() const override;

  /// @brief Predict metrics using CompressionFeatures.
  CompressionPrediction Predict(const CompressionFeatures& features) override;

  /// @brief Predict metrics for batch.
  std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures>& batch) override;

  /// @brief Return model type.
  ModelType Type() const override { return ModelType::kLinRegTable; }

  /// @brief Direct prediction by key with distribution.
  CompressionPrediction PredictByKey(const std::string& library,
                                     const std::string& config,
                                     const std::string& data_type,
                                     const std::string& distribution,
                                     double data_size);

  /// @brief Direct prediction by key without distribution.
  CompressionPrediction PredictByKey(const std::string& library,
                                     const std::string& config,
                                     const std::string& data_type,
                                     double data_size);

  /// @brief Train from features and labels.
  bool Train(const std::vector<CompressionFeatures>& features,
             const std::vector<TrainingLabels>& labels) override;

  /// @brief Train from raw CSV data arrays.
  bool TrainFromRaw(const std::vector<std::string>& libraries,
                    const std::vector<std::string>& configs,
                    const std::vector<std::string>& data_types,
                    const std::vector<std::string>& distributions,
                    const std::vector<double>& data_sizes,
                    const std::vector<double>& compress_times,
                    const std::vector<double>& decompress_times,
                    const std::vector<double>& compress_ratios);

  /// @brief Get regression coefficients for a key.
  const LinearRegressionCoeffs* GetCoeffs(const std::string& library,
                                          const std::string& config,
                                          const std::string& data_type,
                                          const std::string& distribution)
      const;

  /// @brief Get list of supported distributions.
  std::vector<std::string> GetDistributions() const;

  /// @brief Get statistics about loaded model.
  std::string GetStatistics() const;

  /// @brief Get number of models in table.
  size_t GetNumModels() const { return table_.size(); }

  /// @brief Get list of supported libraries.
  std::vector<std::string> GetLibraries() const;

  /// @brief Get list of supported configurations.
  std::vector<std::string> GetConfigurations() const;

  /// @brief Get list of supported data types.
  std::vector<std::string> GetDataTypes() const;

 private:
  /// @brief Fit linear regression model using OLS.
  void FitLinearRegression(const std::vector<double>& x,
                           const std::vector<double>& y,
                           double& slope, double& intercept, double& r2);

  /// @brief Decode library_config_id to library and config names.
  void DecodeLibraryConfigId(int library_config_id, std::string& library,
                             std::string& config) const;

  /// @brief Get data type from one-hot encoded features.
  std::string GetDataTypeFromFeatures(const CompressionFeatures& features)
      const;

  LinRegTableConfig config_;
  std::map<LinRegTableKey, LinearRegressionCoeffs> table_;
  bool ready_;
  mutable std::mutex mutex_;
  std::map<int, std::pair<std::string, std::string>> id_to_lib_config_;
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_LINREG_TABLE_PREDICTOR_H_
