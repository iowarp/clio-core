/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file predictor.h
 * @brief Unified compression-metric prediction interface (issue #693).
 *
 * Consolidates the feature/label data structures and the abstract predictor
 * interface that were previously duplicated across
 * context-transfer-engine/compressor and context-transport-primitives. All
 * compression stat models (XGBoost, Q-table, the NeuroPress neural net, and
 * the legacy dense-NN / linear-regression table) implement this one interface.
 *
 * Feature/label schema mirrors the training-CSV emitted by the unified
 * synthetic compressor benchmark (see test/unit/compress/model and the
 * benchmark driver) so a model trained offline in Python loads and predicts
 * here without re-deriving the feature order.
 */
#ifndef CLIO_CTP_COMPRESS_MODEL_PREDICTOR_H_
#define CLIO_CTP_COMPRESS_MODEL_PREDICTOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ctp::compress::model {

/**
 * @brief Input features for compression-metric prediction.
 *
 * Numerical data statistics (from clio_ctp/compress/preprocess) plus the
 * candidate compressor's identity/preset. library_config_id follows the
 * CompressionFactory encoding base_id * 10 + preset_id (FAST=1, BALANCED=2,
 * BEST=3), e.g. BZIP2_FAST=11, ZFP_BEST=103.
 */
struct CompressionFeatures {
  double chunk_size_bytes = 0;       /**< Size of the data chunk in bytes */
  double target_cpu_util = 0;        /**< Target CPU utilization (0-100%) */
  double shannon_entropy = 0;        /**< Shannon entropy in bits/byte (0-8) */
  double mad = 0;                    /**< Mean absolute deviation */
  double second_derivative_mean = 0; /**< Mean second derivative (curvature) */
  double library_config_id = 0;      /**< Encodes library + preset */
  double config_fast = 0;            /**< One-hot: 1 if FAST preset */
  double config_balanced = 0;        /**< One-hot: 1 if BALANCED preset */
  double config_best = 0;            /**< One-hot: 1 if BEST preset */
  double data_type_char = 0;         /**< One-hot: 1 if char/int data */
  double data_type_float = 0;        /**< One-hot: 1 if float data */

  /** @brief Flatten to the fixed 11-feature model-input order. */
  std::vector<float> ToVector() const {
    return {static_cast<float>(chunk_size_bytes),
            static_cast<float>(target_cpu_util),
            static_cast<float>(shannon_entropy),
            static_cast<float>(mad),
            static_cast<float>(second_derivative_mean),
            static_cast<float>(library_config_id),
            static_cast<float>(config_fast),
            static_cast<float>(config_balanced),
            static_cast<float>(config_best),
            static_cast<float>(data_type_char),
            static_cast<float>(data_type_float)};
  }

  /** @brief Number of features (11). */
  static constexpr size_t NumFeatures() { return 11; }

  /** @brief Feature column names, in ToVector() order. */
  static std::vector<std::string> FeatureNames() {
    return {"chunk_size_bytes", "target_cpu_util", "shannon_entropy",
            "mad",              "second_derivative_mean",
            "library_config_id", "config_fast", "config_balanced",
            "config_best",      "data_type_char", "data_type_float"};
  }
};

/**
 * @brief Predicted compression metrics (multi-output, one forward pass).
 */
struct CompressionPrediction {
  double compression_ratio = 0;    /**< Predicted ratio (>1 means smaller) */
  double psnr_db = 0;              /**< Predicted PSNR in dB (0 for lossless) */
  double compression_time_ms = 0; /**< Predicted compression time (ms) */
  double inference_time_ms = 0;   /**< Model inference latency (ms) */

  CompressionPrediction() = default;
  CompressionPrediction(double ratio, double psnr, double compress_time,
                        double infer_time)
      : compression_ratio(ratio),
        psnr_db(psnr),
        compression_time_ms(compress_time),
        inference_time_ms(infer_time) {}

  /** @brief Number of predicted outputs (ratio, psnr, time). */
  static constexpr size_t NumOutputs() { return 3; }
};

/**
 * @brief Training labels for the multi-output model.
 */
struct TrainingLabels {
  float compression_ratio = 0;
  float psnr_db = 0;
  float compression_time_ms = 0;

  TrainingLabels() = default;
  TrainingLabels(float ratio, float psnr, float time)
      : compression_ratio(ratio), psnr_db(psnr), compression_time_ms(time) {}
};

/** @brief Which concrete model backs a predictor. */
enum class ModelType {
  kQTable,       /**< Discrete Q-table, pure C++, no external deps */
  kLinRegTable,  /**< Per-config OLS regression table, pure C++ */
  kXGBoost,      /**< Gradient-boosted trees (optional xgboost C API) */
  kNeuroPressNN, /**< NeuroPress dense multi-output net (.nnwt weights) */
  kDenseNN,      /**< Legacy MiniDNN/Eigen dense net */
};

/**
 * @brief Abstract base for all compression-metric predictors.
 *
 * Load()/Save() persist a model to/from a directory; Predict() runs one
 * inference; PredictBatch() runs many. Train() and the reinforcement-learning
 * hooks are optional (default no-op) so pure-inference models need not
 * implement them.
 */
class CompressionPredictor {
 public:
  virtual ~CompressionPredictor() = default;

  /** @brief Load model artifacts from model_dir. */
  virtual bool Load(const std::string &model_dir) = 0;
  /** @brief Persist model artifacts to model_dir. */
  virtual bool Save(const std::string &model_dir) = 0;
  /** @brief True once the model can serve predictions. */
  virtual bool IsReady() const = 0;
  /** @brief Which concrete model this is. */
  virtual ModelType Type() const = 0;

  /** @brief Predict metrics for a single feature vector. */
  virtual CompressionPrediction Predict(
      const CompressionFeatures &features) = 0;

  /** @brief Predict metrics for a batch (default: loop over Predict). */
  virtual std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures> &batch) {
    std::vector<CompressionPrediction> out;
    out.reserve(batch.size());
    for (const auto &f : batch) out.push_back(Predict(f));
    return out;
  }

  /** @brief Optionally (re)train from features+labels. */
  virtual bool Train(const std::vector<CompressionFeatures> &features,
                     const std::vector<TrainingLabels> &labels) {
    (void)features;
    (void)labels;
    return false;
  }
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_PREDICTOR_H_
