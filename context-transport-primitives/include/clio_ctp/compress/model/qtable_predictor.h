/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file qtable_predictor.h
 * @brief Q-Table compression predictor using discrete state binning.
 *
 * Uses feature discretization to create a lookup table mapping states to
 * predicted compression metrics. Very fast inference with no external
 * dependencies. Implements the unified CompressionPredictor interface.
 */

#ifndef CLIO_CTP_COMPRESS_MODEL_QTABLE_PREDICTOR_H_
#define CLIO_CTP_COMPRESS_MODEL_QTABLE_PREDICTOR_H_

#include "clio_ctp/compress/model/predictor.h"
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ctp::compress::model {

/**
 * @brief State representation for Q-table lookup.
 *
 * A state is a discretized version of input features where continuous
 * values are binned into discrete buckets using percentile-based binning.
 */
struct QState {
  std::array<int, 11> bins;

  QState() { bins.fill(0); }

  /// @brief Equality comparison for map lookup.
  bool operator==(const QState& other) const {
    return bins == other.bins;
  }

  /// @brief Less-than comparison for map ordering.
  bool operator<(const QState& other) const {
    return bins < other.bins;
  }

  /// @brief Convert state to string for debugging.
  std::string ToString() const;
};

/**
 * @brief Q-value entry storing predictions and statistics.
 */
struct QValue {
  float compression_ratio;
  float psnr_db;
  float compression_time_ms;
  size_t sample_count;

  QValue()
    : compression_ratio(0), psnr_db(0), compression_time_ms(0),
      sample_count(0) {}

  QValue(float ratio, float psnr, float time, size_t count = 1)
    : compression_ratio(ratio), psnr_db(psnr),
      compression_time_ms(time), sample_count(count) {}
};

/**
 * @brief Configuration for Q-table predictor.
 */
struct QTableConfig {
  int n_bins;
  bool use_nearest_neighbor;
  int nn_k;
  bool separate_tables;

  QTableConfig()
    : n_bins(15), use_nearest_neighbor(false), nn_k(5),
      separate_tables(false) {}
};

/**
 * @brief Q-Table compression predictor using discrete state binning.
 *
 * Implements CompressionPredictor interface. Fast, dependency-free
 * prediction via percentile-based feature discretization and lookup table.
 */
class QTablePredictor : public CompressionPredictor {
 public:
  /// @brief Default constructor.
  QTablePredictor();

  /// @brief Constructor with config.
  explicit QTablePredictor(const QTableConfig& config);

  /// @brief Destructor.
  ~QTablePredictor() override;

  QTablePredictor(const QTablePredictor&) = delete;
  QTablePredictor& operator=(const QTablePredictor&) = delete;

  QTablePredictor(QTablePredictor&& other) noexcept;
  QTablePredictor& operator=(QTablePredictor&& other) noexcept;

  /// @brief Load Q-table from directory (qtable.csv, binning.csv).
  bool Load(const std::string& model_dir) override;

  /// @brief Save Q-table to directory.
  bool Save(const std::string& model_dir) override;

  /// @brief Check if Q-table is loaded and ready.
  bool IsReady() const override;

  /// @brief Predict metrics for single input.
  CompressionPrediction Predict(const CompressionFeatures& features) override;

  /// @brief Predict metrics for batch of inputs.
  std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures>& batch) override;

  /// @brief Train Q-table from features and labels.
  bool Train(const std::vector<CompressionFeatures>& features,
             const std::vector<TrainingLabels>& labels) override;

  /// @brief Return model type.
  ModelType Type() const override { return ModelType::kQTable; }

  /// @brief Get statistics string.
  std::string GetStatistics() const;

  /// @brief Get number of states in Q-table.
  size_t GetNumStates() const { return qtable_.size(); }

  /// @brief Get unknown state count.
  size_t GetUnknownCount() const { return unknown_count_; }

 private:
  /// @brief Discretize features to create a state.
  QState DiscretizeFeatures(const CompressionFeatures& features) const;

  /// @brief Build binning edges from training data.
  void BuildBinningEdges(const std::vector<CompressionFeatures>& features);

  /// @brief Find nearest neighbors for unknown state.
  std::vector<QState> FindNearestNeighbors(const QState& state, int k)
      const;

  /// @brief Compute distance between two states.
  double ComputeDistance(const QState& s1, const QState& s2) const;

  /// @brief Get prediction for a state (with fallback handling).
  QValue GetPrediction(const QState& state);

  QTableConfig config_;
  std::map<QState, QValue> qtable_;
  std::vector<std::vector<float>> bin_edges_;
  QValue global_average_;
  bool table_ready_;
  mutable size_t unknown_count_;
  mutable std::mutex mutex_;
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_QTABLE_PREDICTOR_H_
