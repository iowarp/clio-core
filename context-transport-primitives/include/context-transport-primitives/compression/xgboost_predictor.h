/**
 * @file xgboost_predictor.h
 * @brief XGBoost-based compression prediction model
 *
 * This header provides a C++ interface for XGBoost gradient boosting models
 * to predict compression metrics. Supports both training and inference.
 */

#ifndef HSHM_XGBOOST_PREDICTOR_H
#define HSHM_XGBOOST_PREDICTOR_H

#include "compression_features.h"
#include <xgboost/c_api.h>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <random>

namespace hshm {
namespace compress {

/**
 * @brief XGBoost-based compression performance predictor
 *
 * Uses gradient boosting decision trees to predict compression metrics.
 * Predicts all outputs (compression ratio, PSNR, compression time) using
 * a unified interface with internally managed boosters.
 *
 * Example usage:
 * @code
 * XGBoostPredictor predictor;
 *
 * // Train with multi-output labels
 * std::vector<TrainingLabels> labels = {...};
 * predictor.Train(features, labels);
 *
 * // Predict all metrics at once
 * auto result = predictor.Predict(features);
 * std::cout << "Ratio: " << result.compression_ratio << std::endl;
 * std::cout << "PSNR: " << result.psnr_db << std::endl;
 * std::cout << "Time: " << result.compression_time_ms << std::endl;
 * @endcode
 */
class XGBoostPredictor : public CompressionPredictor {
 public:
  /**
   * @brief Default constructor
   */
  XGBoostPredictor();

  /**
   * @brief Destructor - frees XGBoost resources
   */
  ~XGBoostPredictor() override;

  // Disable copy operations (XGBoost handles are not copyable)
  XGBoostPredictor(const XGBoostPredictor&) = delete;
  XGBoostPredictor& operator=(const XGBoostPredictor&) = delete;

  // Enable move operations
  XGBoostPredictor(XGBoostPredictor&& other) noexcept;
  XGBoostPredictor& operator=(XGBoostPredictor&& other) noexcept;

  /**
   * @brief Load models from a directory
   *
   * Expects the following files in the directory:
   * - compression_ratio_model.json: XGBoost model for ratio prediction
   * - psnr_model.json: XGBoost model for PSNR prediction
   * - model_metadata.json: Feature names and model configuration
   *
   * @param model_dir Directory containing model files
   * @return true if both models loaded successfully
   */
  bool Load(const std::string& model_dir) override;

  /**
   * @brief Save models to a directory
   *
   * Saves models in JSON format for portability.
   *
   * @param model_dir Directory to save model files
   * @return true if saving succeeded
   */
  bool Save(const std::string& model_dir) override;

  /**
   * @brief Check if models are loaded and ready
   * @return true if both models are ready for prediction
   */
  bool IsReady() const override;

  /**
   * @brief Predict compression metrics for a single input
   *
   * @param features Input features for prediction
   * @return Prediction result with compression ratio and PSNR
   */
  CompressionPrediction Predict(const CompressionFeatures& features) override;

  /**
   * @brief Predict compression metrics for a batch of inputs
   *
   * More efficient than calling Predict() in a loop due to
   * batched DMatrix creation.
   *
   * @param batch Vector of input features
   * @return Vector of prediction results
   */
  std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures>& batch) override;

  /**
   * @brief Train the unified multi-output model (override from base class)
   *
   * Trains all three boosters (ratio, PSNR, compression time) from
   * a single training call with multi-output labels.
   *
   * @param features Vector of training features
   * @param labels Vector of training labels (ratio, psnr, time)
   * @return true if training succeeded
   */
  bool Train(const std::vector<CompressionFeatures>& features,
             const std::vector<TrainingLabels>& labels) override;

  /**
   * @brief Train the unified multi-output model with config
   *
   * @param features Vector of training features
   * @param labels Vector of training labels (ratio, psnr, time)
   * @param config Training configuration
   * @return true if training succeeded
   */
  bool TrainWithConfig(const std::vector<CompressionFeatures>& features,
                       const std::vector<TrainingLabels>& labels,
                       const XGBoostConfig& config);

  /**
   * @brief Train the compression ratio model from data (legacy)
   *
   * @param features Vector of training features
   * @param labels Vector of compression ratio labels
   * @param config Training configuration
   * @return true if training succeeded
   */
  bool TrainRatioModel(const std::vector<CompressionFeatures>& features,
                       const std::vector<float>& labels,
                       const XGBoostConfig& config = XGBoostConfig());

  /**
   * @brief Train the PSNR model from data (legacy)
   *
   * @param features Vector of training features (lossy compression only)
   * @param labels Vector of PSNR labels
   * @param config Training configuration
   * @return true if training succeeded
   */
  bool TrainPSNRModel(const std::vector<CompressionFeatures>& features,
                      const std::vector<float>& labels,
                      const XGBoostConfig& config = XGBoostConfig());

  /**
   * @brief Train the compression time model from data (legacy)
   *
   * @param features Vector of training features
   * @param labels Vector of compression time labels (ms)
   * @param config Training configuration
   * @return true if training succeeded
   */
  bool TrainTimeModel(const std::vector<CompressionFeatures>& features,
                      const std::vector<float>& labels,
                      const XGBoostConfig& config = XGBoostConfig());

  /**
   * @brief Get the last error message from XGBoost
   * @return Error message string
   */
  std::string GetLastError() const;

  // ============================================================================
  // Reinforcement Learning Methods
  // ============================================================================

  /**
   * @brief Record an experience for reinforcement learning
   * @param experience Experience tuple with features, prediction, and actual result
   */
  void RecordExperience(const RLExperience& experience) override;

  /**
   * @brief Perform an RL update step using recorded experiences
   * @param config RL configuration
   * @return true if update was performed
   */
  bool UpdateFromExperiences(const RLConfig& config = RLConfig()) override;

  /**
   * @brief Enable or disable exploration mode
   * @param enable Whether to enable exploration noise
   */
  void SetExplorationMode(bool enable) override;

  /**
   * @brief Get current exploration rate
   * @return Current epsilon value
   */
  double GetExplorationRate() const override;

  /**
   * @brief Get number of experiences in replay buffer
   * @return Number of stored experiences
   */
  size_t GetExperienceCount() const override;

  /**
   * @brief Clear the experience replay buffer
   */
  void ClearExperiences() override;

 private:
  /**
   * @brief Create a DMatrix from feature vectors
   *
   * @param features Vector of features
   * @param labels Optional labels for training (nullptr for inference)
   * @return DMatrix handle or nullptr on error
   */
  DMatrixHandle CreateDMatrix(const std::vector<CompressionFeatures>& features,
                              const std::vector<float>* labels = nullptr);

  /**
   * @brief Free a DMatrix handle
   * @param dmatrix Handle to free
   */
  void FreeDMatrix(DMatrixHandle dmatrix);

  /**
   * @brief Perform prediction with a booster
   *
   * @param booster Booster handle
   * @param dmatrix DMatrix with input data
   * @param out_predictions Output vector for predictions
   * @return true if prediction succeeded
   */
  bool PredictWithBooster(BoosterHandle booster,
                          DMatrixHandle dmatrix,
                          std::vector<float>& out_predictions);

  /**
   * @brief Train a booster with given data
   *
   * @param dmatrix Training data
   * @param config Training configuration
   * @param out_booster Output booster handle
   * @return true if training succeeded
   */
  bool TrainBooster(DMatrixHandle dmatrix,
                    const XGBoostConfig& config,
                    BoosterHandle& out_booster);

  /**
   * @brief Check if this is a lossy compression (has PSNR)
   * @param features Input features
   * @return true if lossy compression
   */
  bool IsLossyCompression(const CompressionFeatures& features) const;

  BoosterHandle ratio_booster_;  /**< Booster for compression ratio */
  BoosterHandle psnr_booster_;   /**< Booster for PSNR prediction */
  BoosterHandle time_booster_;   /**< Booster for compression time prediction */
  bool is_ready_;                /**< Whether models are loaded */
  mutable std::mutex mutex_;     /**< Mutex for thread safety */

  // Reinforcement learning members
  std::deque<RLExperience> experience_buffer_;  /**< Experience replay buffer */
  double epsilon_;                              /**< Current exploration rate */
  bool exploration_enabled_;                    /**< Whether exploration is enabled */
  mutable std::mt19937 rng_;                    /**< Random number generator */
  size_t experience_count_;                     /**< Total experiences recorded */
};

}  // namespace compress
}  // namespace hshm

#endif  // HSHM_XGBOOST_PREDICTOR_H
