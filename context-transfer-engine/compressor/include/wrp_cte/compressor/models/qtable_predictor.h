/**
 * @file qtable_predictor.h
 * @brief Q-Table compression predictor using discrete state binning
 *
 * This header provides a C++ interface for Q-table (discrete binning) models
 * to predict compression metrics. Uses feature discretization to create states
 * and lookup tables for fast, dependency-free prediction.
 */

#ifndef WRP_CTE_COMPRESSOR_MODELS_QTABLE_PREDICTOR_H_
#define WRP_CTE_COMPRESSOR_MODELS_QTABLE_PREDICTOR_H_

#include "compression_features.h"
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <array>

namespace wrp_cte::compressor {

/**
 * @brief State representation for Q-table lookup
 *
 * A state is a discretized version of input features where continuous
 * values are binned into discrete buckets using percentile-based binning.
 */
struct QState {
  std::array<int, 11> bins;  /**< Discretized feature values */

  QState() { bins.fill(0); }

  /**
   * @brief Equality comparison for map lookup
   */
  bool operator==(const QState& other) const {
    return bins == other.bins;
  }

  /**
   * @brief Less-than comparison for map ordering
   */
  bool operator<(const QState& other) const {
    return bins < other.bins;
  }

  /**
   * @brief Convert state to string for debugging
   */
  std::string ToString() const;
};

/**
 * @brief Q-value entry storing predictions and statistics
 */
struct QValue {
  float compression_ratio;     /**< Average compression ratio */
  float psnr_db;               /**< Average PSNR (0 for lossless) */
  float compression_time_ms;   /**< Average compression time */
  size_t sample_count;         /**< Number of samples averaged */

  QValue() : compression_ratio(0), psnr_db(0), compression_time_ms(0), sample_count(0) {}

  QValue(float ratio, float psnr, float time, size_t count = 1)
      : compression_ratio(ratio), psnr_db(psnr),
        compression_time_ms(time), sample_count(count) {}
};

/**
 * @brief Configuration for Q-table predictor
 */
struct QTableConfig {
  int n_bins;                      /**< Number of bins for continuous features (default: 15) */
  bool use_nearest_neighbor;       /**< Use NN fallback for unknown states (default: false) */
  int nn_k;                        /**< Number of nearest neighbors (default: 5) */
  bool separate_tables;            /**< Use separate tables per output (default: false) */

  QTableConfig()
      : n_bins(15),
        use_nearest_neighbor(false),
        nn_k(5),
        separate_tables(false) {}
};

/**
 * @brief Q-Table compression predictor using discrete state binning
 *
 * Uses feature discretization to create a lookup table mapping states to
 * predicted compression metrics. Very fast inference with no dependencies.
 * Best for embedded systems or when interpretability is important.
 *
 * Features:
 * - Percentile-based binning for continuous features
 * - Optional nearest neighbor fallback for unknown states
 * - Configurable number of bins (15 recommended from experiments)
 * - No external dependencies
 *
 * Example usage:
 * @code
 * QTablePredictor predictor;
 *
 * // Train with data
 * std::vector<TrainingLabels> labels = {...};
 * predictor.Train(features, labels);
 *
 * // Predict
 * auto result = predictor.Predict(features);
 * std::cout << "Ratio: " << result.compression_ratio << std::endl;
 * @endcode
 */
class QTablePredictor : public CompressionPredictor {
 public:
  /**
   * @brief Default constructor
   */
  QTablePredictor();

  /**
   * @brief Constructor with config
   * @param config Q-table configuration
   */
  explicit QTablePredictor(const QTableConfig& config);

  /**
   * @brief Destructor
   */
  ~QTablePredictor() override;

  // Disable copy operations
  QTablePredictor(const QTablePredictor&) = delete;
  QTablePredictor& operator=(const QTablePredictor&) = delete;

  // Enable move operations
  QTablePredictor(QTablePredictor&& other) noexcept;
  QTablePredictor& operator=(QTablePredictor&& other) noexcept;

  /**
   * @brief Load Q-table from a directory
   *
   * Expects the following files:
   * - qtable.bin: Serialized Q-table
   * - binning_params.json: Binning edge parameters
   *
   * @param model_dir Directory containing model files
   * @return true if loading succeeded
   */
  bool Load(const std::string& model_dir) override;

  /**
   * @brief Save Q-table to a directory
   * @param model_dir Directory to save model files
   * @return true if saving succeeded
   */
  bool Save(const std::string& model_dir) override;

  /**
   * @brief Check if Q-table is loaded and ready
   * @return true if ready
   */
  bool IsReady() const override;

  /**
   * @brief Predict compression metrics for a single input
   * @param features Input features
   * @return Prediction result
   */
  CompressionPrediction Predict(const CompressionFeatures& features) override;

  /**
   * @brief Predict compression metrics for a batch of inputs
   * @param batch Vector of input features
   * @return Vector of predictions
   */
  std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures>& batch) override;

  /**
   * @brief Train the Q-table from data
   *
   * Builds binning edges from training data and populates Q-table
   * with average values for each state.
   *
   * @param features Vector of training features
   * @param labels Vector of training labels
   * @return true if training succeeded
   */
  bool Train(const std::vector<CompressionFeatures>& features,
             const std::vector<TrainingLabels>& labels) override;

  /**
   * @brief Get Q-table statistics
   * @return String with table size, unknown rate, etc.
   */
  std::string GetStatistics() const;

  /**
   * @brief Get number of states in Q-table
   * @return Number of unique states
   */
  size_t GetNumStates() const { return qtable_.size(); }

  /**
   * @brief Get unknown state count (for testing)
   * @return Number of unknown states encountered during last prediction batch
   */
  size_t GetUnknownCount() const { return unknown_count_; }

 private:
  /**
   * @brief Discretize features to create a state
   * @param features Input features
   * @return Discretized state
   */
  QState DiscretizeFeatures(const CompressionFeatures& features) const;

  /**
   * @brief Build binning edges from training data
   * @param features Training features
   */
  void BuildBinningEdges(const std::vector<CompressionFeatures>& features);

  /**
   * @brief Find nearest neighbors for unknown state
   * @param state Unknown state
   * @param k Number of neighbors
   * @return Vector of k nearest known states
   */
  std::vector<QState> FindNearestNeighbors(const QState& state, int k) const;

  /**
   * @brief Compute distance between two states
   * @param s1 First state
   * @param s2 Second state
   * @return Euclidean distance
   */
  double ComputeDistance(const QState& s1, const QState& s2) const;

  /**
   * @brief Get prediction for a state (with fallback handling)
   * @param state Input state
   * @return Q-value prediction
   */
  QValue GetPrediction(const QState& state);

  QTableConfig config_;                    /**< Q-table configuration */
  std::map<QState, QValue> qtable_;        /**< Main Q-table lookup */
  std::vector<std::vector<float>> bin_edges_; /**< Binning edges per feature */
  QValue global_average_;                  /**< Fallback for unknown states */
  bool table_ready_;                       /**< Whether table is ready */
  mutable size_t unknown_count_;           /**< Count of unknown states */
  mutable std::mutex mutex_;               /**< Mutex for thread safety */
};

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_MODELS_QTABLE_PREDICTOR_H_
