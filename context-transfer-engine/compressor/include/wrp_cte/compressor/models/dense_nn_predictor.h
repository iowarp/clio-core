/**
 * @file dense_nn_predictor.h
 * @brief Dense Neural Network compression predictor using MiniDNN
 *
 * This header provides a C++ interface for dense neural network models
 * to predict compression metrics. Uses a unified multi-output architecture
 * that predicts all outputs (ratio, PSNR, time) in a single forward pass.
 * Uses MiniDNN (header-only library based on Eigen) for neural network operations.
 */

#ifndef WRP_CTE_COMPRESSOR_MODELS_DENSE_NN_PREDICTOR_H_
#define WRP_CTE_COMPRESSOR_MODELS_DENSE_NN_PREDICTOR_H_

#include "compression_features.h"
#include <Eigen/Core>
#include <MiniDNN.h>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <random>

namespace wrp_cte::compressor {

/**
 * @brief Standard scaler for feature normalization
 *
 * Normalizes features to zero mean and unit variance, matching
 * sklearn's StandardScaler behavior.
 */
class StandardScaler {
 public:
  StandardScaler() : fitted_(false) {}

  /**
   * @brief Fit the scaler to training data
   * @param data Training data (num_samples x num_features)
   */
  void Fit(const std::vector<std::vector<float>>& data);

  /**
   * @brief Transform data using fitted parameters
   * @param data Input data
   * @return Normalized data
   */
  std::vector<std::vector<float>> Transform(
      const std::vector<std::vector<float>>& data) const;

  /**
   * @brief Fit and transform in one step
   * @param data Training data
   * @return Normalized data
   */
  std::vector<std::vector<float>> FitTransform(
      const std::vector<std::vector<float>>& data);

  /**
   * @brief Save scaler parameters to file
   * @param path File path
   * @return true if successful
   */
  bool Save(const std::string& path) const;

  /**
   * @brief Load scaler parameters from file
   * @param path File path
   * @return true if successful
   */
  bool Load(const std::string& path);

  /**
   * @brief Check if scaler is fitted
   * @return true if fitted
   */
  bool IsFitted() const { return fitted_; }

  /**
   * @brief Get mean values
   * @return Mean vector
   */
  const std::vector<float>& GetMean() const { return mean_; }

  /**
   * @brief Get standard deviation values
   * @return Std vector
   */
  const std::vector<float>& GetStd() const { return std_; }

 private:
  std::vector<float> mean_;    /**< Feature means */
  std::vector<float> std_;     /**< Feature standard deviations */
  bool fitted_;                /**< Whether scaler is fitted */
};

/**
 * @brief Dense Neural Network compression predictor using MiniDNN
 *
 * Uses a multi-layer perceptron with multi-output architecture to predict
 * all compression metrics (ratio, PSNR, time) in a single forward pass.
 * MiniDNN is a header-only library based on Eigen, providing fast CPU inference.
 *
 * Example usage:
 * @code
 * DenseNNPredictor predictor;
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
class DenseNNPredictor : public CompressionPredictor {
 public:
  /**
   * @brief Default constructor
   */
  DenseNNPredictor();

  /**
   * @brief Destructor
   */
  ~DenseNNPredictor() override;

  // Disable copy operations
  DenseNNPredictor(const DenseNNPredictor&) = delete;
  DenseNNPredictor& operator=(const DenseNNPredictor&) = delete;

  // Enable move operations
  DenseNNPredictor(DenseNNPredictor&& other) noexcept;
  DenseNNPredictor& operator=(DenseNNPredictor&& other) noexcept;

  /**
   * @brief Load unified model from a directory
   *
   * Expects the following files:
   * - unified_model.bin: MiniDNN multi-output model
   * - input_scaler.json: Scaler parameters for input features
   * - output_scaler.json: Scaler parameters for output targets
   *
   * @param model_dir Directory containing model files
   * @return true if loading succeeded
   */
  bool Load(const std::string& model_dir) override;

  /**
   * @brief Load model weights from JSON file (Python format)
   *
   * Loads weights exported by dnn_hyperparameter_study.py.
   * JSON format includes architecture, weights, biases, and scaler parameters.
   *
   * @param json_path Path to JSON weights file
   * @return true if loading succeeded
   */
  bool LoadWeights(const std::string& json_path);

  /**
   * @brief Save models to a directory
   * @param model_dir Directory to save model files
   * @return true if saving succeeded
   */
  bool Save(const std::string& model_dir) override;

  /**
   * @brief Check if models are loaded and ready
   * @return true if models are ready
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
   * @brief Train the unified multi-output model (override from base class)
   *
   * Trains all three networks (ratio, PSNR, time) from a single call.
   *
   * @param features Vector of training features
   * @param labels Vector of training labels (ratio, psnr, time)
   * @return true if training succeeded
   */
  bool Train(const std::vector<CompressionFeatures>& features,
             const std::vector<TrainingLabels>& labels) override;

  /**
   * @brief Train the unified model with config
   *
   * @param features Vector of training features
   * @param labels Vector of training labels
   * @param config Training configuration
   * @return true if training succeeded
   */
  bool TrainWithConfig(const std::vector<CompressionFeatures>& features,
                       const std::vector<TrainingLabels>& labels,
                       const TrainingConfig& config);


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
   * @brief Convert features to Eigen matrix
   * @param features Input features
   * @param scaler Scaler to normalize features
   * @return Matrix ready for inference (features x samples)
   */
  Eigen::MatrixXd FeaturesToMatrix(
      const std::vector<CompressionFeatures>& features,
      const StandardScaler& scaler);

  /**
   * @brief Train a MiniDNN network
   *
   * @param features Training features
   * @param labels Training labels
   * @param config Training configuration
   * @param network Output network
   * @param scaler Output scaler
   * @return true if training succeeded
   */
  bool TrainNetwork(const std::vector<CompressionFeatures>& features,
                    const std::vector<float>& labels,
                    const TrainingConfig& config,
                    MiniDNN::Network& network,
                    StandardScaler& scaler);

  /**
   * @brief Check if this is lossy compression
   * @param features Input features
   * @return true if lossy compression
   */
  bool IsLossyCompression(const CompressionFeatures& features) const;

  /**
   * @brief Build network architecture
   * @param network Network to build
   * @param hidden_layers Hidden layer sizes
   * @param dropout_rate Dropout rate (unused in MiniDNN, kept for API compat)
   */
  void BuildNetwork(MiniDNN::Network& network,
                    const std::vector<int>& hidden_layers,
                    double dropout_rate);

  std::unique_ptr<MiniDNN::Network> unified_network_;  /**< Unified multi-output network */
  StandardScaler input_scaler_;                        /**< Scaler for input features */
  StandardScaler output_scaler_;                       /**< Scaler for output targets */
  bool model_ready_;                                   /**< Whether model is ready */
  mutable std::mutex mutex_;                           /**< Mutex for thread safety */

  // Architecture storage for save/load
  std::vector<int> hidden_layers_;                     /**< Hidden layer sizes */
  double dropout_rate_;                                /**< Dropout rate (kept for API compatibility) */

  // Reinforcement learning members
  std::deque<RLExperience> experience_buffer_;        /**< Experience replay buffer */
  double epsilon_;                                    /**< Current exploration rate */
  bool exploration_enabled_;                          /**< Whether exploration is enabled */
  mutable std::mt19937 rng_;                          /**< Random number generator */
  size_t experience_count_;                           /**< Total experiences recorded */
};

}  // namespace wrp_cte::compressor

#endif  // WRP_CTE_COMPRESSOR_MODELS_DENSE_NN_PREDICTOR_H_
