/**
 * @file compression_features.h
 * @brief Data structures for compression prediction models
 *
 * This header defines the common data structures used by both XGBoost and
 * Dense Neural Network predictors for compression performance prediction.
 */

#ifndef HSHM_COMPRESSION_FEATURES_H
#define HSHM_COMPRESSION_FEATURES_H

#include <vector>
#include <string>
#include <cstdint>

namespace hshm {
namespace compress {

/**
 * @brief Input features for compression prediction models
 *
 * Contains all features required by the trained models to predict
 * compression ratio and PSNR (for lossy compressors).
 *
 * Features are based on data statistics and compressor choice:
 * - Numerical: chunk_size, cpu_util, entropy, MAD, second_derivative
 * - One-hot encoded: library type, data type
 */
struct CompressionFeatures {
  // Numerical features
  double chunk_size_bytes;         /**< Size of data chunk in bytes */
  double target_cpu_util;          /**< Target CPU utilization (0-100%) */
  double shannon_entropy;          /**< Shannon entropy in bits/byte (0-8) */
  double mad;                      /**< Mean Absolute Deviation */
  double second_derivative_mean;   /**< Mean of second derivative (curvature) */

  // One-hot encoded library features
  double library_bzip2;            /**< 1 if BZIP2, 0 otherwise */
  double library_zfp_tol_0_01;     /**< 1 if ZFP with tolerance 0.01, 0 otherwise */
  double library_zfp_tol_0_1;      /**< 1 if ZFP with tolerance 0.1, 0 otherwise */

  // One-hot encoded data type features
  double data_type_char;           /**< 1 if char/int data, 0 otherwise */
  double data_type_float;          /**< 1 if float data, 0 otherwise */

  /**
   * @brief Default constructor initializes all features to zero
   */
  CompressionFeatures()
      : chunk_size_bytes(0), target_cpu_util(0), shannon_entropy(0),
        mad(0), second_derivative_mean(0), library_bzip2(0),
        library_zfp_tol_0_01(0), library_zfp_tol_0_1(0),
        data_type_char(0), data_type_float(0) {}

  /**
   * @brief Convert features to a flat vector for model input
   * @return Vector of 10 feature values in the expected order
   */
  std::vector<float> ToVector() const {
    return {
        static_cast<float>(chunk_size_bytes),
        static_cast<float>(target_cpu_util),
        static_cast<float>(shannon_entropy),
        static_cast<float>(mad),
        static_cast<float>(second_derivative_mean),
        static_cast<float>(library_bzip2),
        static_cast<float>(library_zfp_tol_0_01),
        static_cast<float>(library_zfp_tol_0_1),
        static_cast<float>(data_type_char),
        static_cast<float>(data_type_float)
    };
  }

  /**
   * @brief Get the number of features
   * @return Number of features (10)
   */
  static constexpr size_t NumFeatures() { return 10; }

  /**
   * @brief Get feature names in order
   * @return Vector of feature name strings
   */
  static std::vector<std::string> FeatureNames() {
    return {
        "Chunk Size (bytes)",
        "Target CPU Util (%)",
        "Shannon Entropy (bits/byte)",
        "MAD",
        "Second Derivative Mean",
        "Library_BZIP2",
        "Library_ZFP_tol_0.010000",
        "Library_ZFP_tol_0.100000",
        "Data Type_char",
        "Data Type_float"
    };
  }
};

/**
 * @brief Output of compression prediction models
 *
 * Contains predicted compression metrics and timing information.
 * The unified model predicts all outputs in a single forward pass.
 */
struct CompressionPrediction {
  double compression_ratio;     /**< Predicted compression ratio (>1 means smaller) */
  double psnr_db;               /**< Predicted PSNR in dB (0 for lossless) */
  double compression_time_ms;   /**< Predicted compression time in milliseconds */
  double inference_time_ms;     /**< Time taken for inference in milliseconds */

  /**
   * @brief Default constructor
   */
  CompressionPrediction()
      : compression_ratio(0), psnr_db(0), compression_time_ms(0), inference_time_ms(0) {}

  /**
   * @brief Constructor with values
   * @param ratio Compression ratio
   * @param psnr PSNR value in dB
   * @param compress_time Predicted compression time in ms
   * @param infer_time Inference time in milliseconds
   */
  CompressionPrediction(double ratio, double psnr, double compress_time, double infer_time)
      : compression_ratio(ratio), psnr_db(psnr),
        compression_time_ms(compress_time), inference_time_ms(infer_time) {}

  /**
   * @brief Get number of output features
   * @return Number of outputs (3: ratio, psnr, compression_time)
   */
  static constexpr size_t NumOutputs() { return 3; }
};

/**
 * @brief Training configuration for neural network models
 */
struct TrainingConfig {
  std::vector<int> hidden_layers;  /**< Sizes of hidden layers */
  double dropout_rate;             /**< Dropout rate for regularization */
  double learning_rate;            /**< Initial learning rate */
  double l2_reg;                   /**< L2 regularization strength */
  int epochs;                      /**< Maximum training epochs */
  int batch_size;                  /**< Mini-batch size */
  int patience;                    /**< Early stopping patience */
  bool verbose;                    /**< Print training progress */

  /**
   * @brief Default constructor with reasonable defaults
   */
  TrainingConfig()
      : hidden_layers({128, 64, 32, 16}),
        dropout_rate(0.2),
        learning_rate(0.01),
        l2_reg(0.0001),
        epochs(200),
        batch_size(16),
        patience(30),
        verbose(true) {}
};

/**
 * @brief Training configuration for XGBoost models
 */
struct XGBoostConfig {
  int max_depth;           /**< Maximum tree depth */
  double learning_rate;    /**< Learning rate (eta) */
  int n_estimators;        /**< Number of boosting rounds */
  double subsample;        /**< Row subsampling ratio */
  double colsample_bytree; /**< Column subsampling ratio */
  double reg_lambda;       /**< L2 regularization term */
  int min_child_weight;    /**< Minimum sum of instance weight in child */
  int seed;                /**< Random seed */
  bool verbose;            /**< Print training progress */

  /**
   * @brief Default constructor with reasonable defaults
   */
  XGBoostConfig()
      : max_depth(4),
        learning_rate(0.05),
        n_estimators(200),
        subsample(0.6),
        colsample_bytree(0.6),
        reg_lambda(1.0),
        min_child_weight(10),
        seed(42),
        verbose(true) {}
};

/**
 * @brief Experience tuple for reinforcement learning
 *
 * Stores a single experience from compression operation for online learning.
 */
struct RLExperience {
  CompressionFeatures features;     /**< Input features (state) */
  double predicted_ratio;           /**< Predicted compression ratio (action) */
  double actual_ratio;              /**< Actual compression ratio (reward signal) */
  double predicted_psnr;            /**< Predicted PSNR */
  double actual_psnr;               /**< Actual PSNR (for lossy) */
  double predicted_compress_time;   /**< Predicted compression time in ms */
  double actual_compress_time;      /**< Actual compression time in ms */

  /**
   * @brief Default constructor
   */
  RLExperience()
      : predicted_ratio(0), actual_ratio(0),
        predicted_psnr(0), actual_psnr(0),
        predicted_compress_time(0), actual_compress_time(0) {}

  /**
   * @brief Compute reward based on prediction error
   * @return Reward value (higher is better, negative for large errors)
   */
  double ComputeReward() const {
    // Reward based on prediction accuracy (negative squared error)
    double ratio_error = predicted_ratio - actual_ratio;
    double psnr_error = predicted_psnr - actual_psnr;
    double time_error = predicted_compress_time - actual_compress_time;
    // Weighted combination: ratio is most important, then time, then psnr
    return -(ratio_error * ratio_error +
             0.5 * time_error * time_error +
             0.1 * psnr_error * psnr_error);
  }
};

/**
 * @brief Configuration for reinforcement learning
 */
struct RLConfig {
  size_t replay_buffer_size;   /**< Maximum size of experience replay buffer */
  size_t batch_size;           /**< Mini-batch size for updates */
  size_t update_frequency;     /**< Update model every N experiences */
  double learning_rate;        /**< Learning rate for RL updates */
  double discount_factor;      /**< Discount factor (gamma) for future rewards */
  double epsilon;              /**< Exploration rate (epsilon-greedy) */
  double epsilon_decay;        /**< Epsilon decay rate per update */
  double min_epsilon;          /**< Minimum epsilon value */
  bool enable_exploration;     /**< Whether to add exploration noise */

  /**
   * @brief Default constructor with reasonable defaults
   */
  RLConfig()
      : replay_buffer_size(10000),
        batch_size(32),
        update_frequency(100),
        learning_rate(0.001),
        discount_factor(0.99),
        epsilon(0.1),
        epsilon_decay(0.995),
        min_epsilon(0.01),
        enable_exploration(true) {}
};

/**
 * @brief Training labels for multi-output model
 *
 * Contains all target labels for training the unified model.
 */
struct TrainingLabels {
  float compression_ratio;    /**< Target compression ratio */
  float psnr_db;              /**< Target PSNR in dB */
  float compression_time_ms;  /**< Target compression time in ms */

  /**
   * @brief Default constructor
   */
  TrainingLabels() : compression_ratio(0), psnr_db(0), compression_time_ms(0) {}

  /**
   * @brief Constructor with values
   */
  TrainingLabels(float ratio, float psnr, float time)
      : compression_ratio(ratio), psnr_db(psnr), compression_time_ms(time) {}
};

/**
 * @brief Abstract base class for compression predictors
 *
 * Defines the common interface for all compression prediction models.
 * Uses a unified multi-output model that predicts all metrics in one pass.
 * Supports training, inference, and reinforcement learning.
 */
class CompressionPredictor {
 public:
  virtual ~CompressionPredictor() = default;

  /**
   * @brief Load model from a directory
   * @param model_dir Directory containing model files
   * @return true if loading succeeded, false otherwise
   */
  virtual bool Load(const std::string& model_dir) = 0;

  /**
   * @brief Save model to a directory
   * @param model_dir Directory to save model files
   * @return true if saving succeeded, false otherwise
   */
  virtual bool Save(const std::string& model_dir) = 0;

  /**
   * @brief Check if the model is loaded and ready for prediction
   * @return true if model is ready, false otherwise
   */
  virtual bool IsReady() const = 0;

  /**
   * @brief Predict compression metrics for a single input
   * @param features Input features
   * @return Prediction result with all metrics (ratio, psnr, compress_time)
   */
  virtual CompressionPrediction Predict(const CompressionFeatures& features) = 0;

  /**
   * @brief Predict compression metrics for a batch of inputs
   * @param batch Vector of input features
   * @return Vector of prediction results
   */
  virtual std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures>& batch) = 0;

  /**
   * @brief Train the unified multi-output model
   *
   * Trains a single model that predicts all outputs (compression ratio,
   * PSNR, and compression time) in one forward pass.
   *
   * @param features Vector of training features
   * @param labels Vector of training labels (all outputs)
   * @return true if training succeeded
   */
  virtual bool Train(const std::vector<CompressionFeatures>& features,
                     const std::vector<TrainingLabels>& labels) {
    (void)features;
    (void)labels;
    return false;  // Default: not supported
  }

  // ============================================================================
  // Reinforcement Learning Interface
  // ============================================================================

  /**
   * @brief Record an experience for reinforcement learning
   *
   * Call this after compression to provide feedback on prediction accuracy.
   *
   * @param experience Experience tuple with features, prediction, and actual result
   */
  virtual void RecordExperience(const RLExperience& experience) {
    (void)experience;  // Default: no-op
  }

  /**
   * @brief Perform an RL update step using recorded experiences
   *
   * Uses experience replay to update the model incrementally.
   *
   * @param config RL configuration
   * @return true if update was performed, false if not enough experiences
   */
  virtual bool UpdateFromExperiences(const RLConfig& config = RLConfig()) {
    (void)config;
    return false;  // Default: not supported
  }

  /**
   * @brief Enable or disable exploration mode
   * @param enable Whether to enable exploration noise in predictions
   */
  virtual void SetExplorationMode(bool enable) {
    (void)enable;  // Default: no-op
  }

  /**
   * @brief Get current exploration rate (epsilon)
   * @return Current epsilon value
   */
  virtual double GetExplorationRate() const {
    return 0.0;  // Default: no exploration
  }

  /**
   * @brief Get number of experiences in replay buffer
   * @return Number of stored experiences
   */
  virtual size_t GetExperienceCount() const {
    return 0;  // Default: no buffer
  }

  /**
   * @brief Clear the experience replay buffer
   */
  virtual void ClearExperiences() {
    // Default: no-op
  }
};

}  // namespace compress
}  // namespace hshm

#endif  // HSHM_COMPRESSION_FEATURES_H
