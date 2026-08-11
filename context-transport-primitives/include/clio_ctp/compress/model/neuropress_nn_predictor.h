/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file neuropress_nn_predictor.h
 * @brief NeuroPress dense multi-output neural net predictor (pure C++ CPU).
 *
 * Implements CompressionPredictor using a 5-layer feedforward network
 * (8 → 64 → 64 → 64 → 64 → 8) with ReLU on hidden layers. Weights are loaded
 * from a .nnwt binary format (little-endian, version 2). Inference is CPU-only
 * using standard library (no CUDA, no external deps beyond stdlib).
 *
 * .nnwt Layout (little-endian):
 *   - Header (24 bytes): magic=0x4E4E5754, version=2, num_layers, input_dim,
 *     hidden_dim, output_dim.
 *   - Normalization (32 floats): x_means[8], x_stds[8], y_means[8], y_stds[8].
 *   - 5 layers: each [weights (row-major), biases].
 *     Layer 0: 8×64,   Layer 1-3: 64×64,  Layer 4: 64×8.
 *   - Feature bounds (16 floats): x_mins[8], x_maxs[8].
 */

#ifndef CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_PREDICTOR_H_
#define CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_PREDICTOR_H_

#include "clio_ctp/compress/model/predictor.h"
#include <cstdint>
#include <string>
#include <vector>

namespace ctp::compress::model {

/**
 * @brief NeuroPress dense multi-output neural network predictor.
 *
 * 5-layer feedforward: 8 → 64 → 64 → 64 → 64 → 8 with ReLU on hidden.
 * Maps compression features → normalized input → forward pass → outputs →
 * inverse-transform → CompressionPrediction.
 */
class NeuroPressNNPredictor : public CompressionPredictor {
 public:
  /** @brief Default constructor. */
  NeuroPressNNPredictor();

  /** @brief Destructor. */
  ~NeuroPressNNPredictor() override = default;

  // Allow copy and move.
  NeuroPressNNPredictor(const NeuroPressNNPredictor&) = default;
  NeuroPressNNPredictor& operator=(const NeuroPressNNPredictor&) = default;
  NeuroPressNNPredictor(NeuroPressNNPredictor&&) noexcept = default;
  NeuroPressNNPredictor& operator=(NeuroPressNNPredictor&&) noexcept =
      default;

  /**
   * @brief Load .nnwt weight file from model_dir/model.nnwt.
   *
   * Parses binary header, validates magic/version, loads normalization
   * parameters and weight matrices into host vectors.
   *
   * @param model_dir Directory containing model.nnwt.
   * @return true if load succeeded and weights are valid.
   */
  bool Load(const std::string& model_dir) override;

  /**
   * @brief Persist weights to model_dir/model.nnwt in .nnwt format.
   *
   * Writes header, normalization params, weights, biases, and bounds
   * in the same format as export_weights.py.
   *
   * @param model_dir Directory to write model.nnwt.
   * @return true if save succeeded.
   */
  bool Save(const std::string& model_dir) override;

  /**
   * @brief True if weights are loaded and model is ready.
   * @return true after a successful Load.
   */
  bool IsReady() const override;

  /**
   * @brief Return ModelType::kNeuroPressNN.
   */
  ModelType Type() const override {
    return ModelType::kNeuroPressNN;
  }

  /**
   * @brief Predict compression metrics for one feature vector.
   *
   * Maps features → standardized input → 5-layer forward → outputs →
   * inverse-transform → clamp → inference_time_ms from measurement.
   *
   * @param features Input features (11 total; NeuroPress uses 8).
   * @return CompressionPrediction with ratio, PSNR, compression_time_ms,
   *         and measured inference_time_ms.
   */
  CompressionPrediction Predict(
      const CompressionFeatures& features) override;

 private:
  /**
   * @brief Build the 8-input NeuroPress vector from CompressionFeatures.
   *
   * Maps: library_config_id → algo_id (0-7, exact for NeuroPress's 8 trained
   * nvcomp algorithms, best-effort fallback otherwise -- see
   * NeuroPressAlgoIdForBaseId() in the .cc); quant/shuffle/error_bound
   * passed through; data_size, entropy, mad, second_derivative as-is.
   *
   * @param features Input features.
   * @return 8-element vector in NeuroPress order.
   */
  std::vector<float> FeaturesTo8Input(
      const CompressionFeatures& features) const;

  /**
   * @brief Standardize input: (x - x_means) / max(x_stds, 1e-8).
   */
  std::vector<float> Standardize(const std::vector<float>& x) const;

  /**
   * @brief Forward pass through 5 layers (4 hidden with ReLU, 1 output).
   */
  std::vector<float> ForwardPass(
      const std::vector<float>& x_norm) const;

  /**
   * @brief Inverse-transform outputs: y = x * y_stds + y_means.
   */
  std::vector<float> InverseTransform(
      const std::vector<float>& y_norm) const;

  // Architecture parameters.
  static constexpr uint32_t kInputDim = 8;
  static constexpr uint32_t kHiddenDim = 64;
  static constexpr uint32_t kOutputDim = 8;
  static constexpr uint32_t kNumLayers = 5;  // 4 hidden + 1 output

  // Normalization parameters.
  std::vector<float> x_means_;  // [8]
  std::vector<float> x_stds_;   // [8]
  std::vector<float> y_means_;  // [8]
  std::vector<float> y_stds_;   // [8]

  // Feature bounds.
  std::vector<float> x_mins_;  // [8]
  std::vector<float> x_maxs_;  // [8]

  // Weight matrices (row-major) and biases for each layer.
  // Layer 0: weight [64,8] → bias [64]
  // Layer 1-3: weight [64,64] → bias [64]
  // Layer 4: weight [8,64] → bias [8]
  std::vector<float> weights_;  // All flattened row-major
  std::vector<float> biases_;   // All flattened

  // Offsets into weights_ and biases_ for each layer.
  std::vector<size_t> weight_offsets_;
  std::vector<size_t> bias_offsets_;

  bool is_ready_ = false;
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_PREDICTOR_H_
