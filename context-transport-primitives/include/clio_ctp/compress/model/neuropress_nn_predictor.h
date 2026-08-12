/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

/**
 * @file neuropress_nn_predictor.h
 * @brief NeuroPress dense multi-output neural net predictor.
 *
 * Implements CompressionPredictor using a 5-layer feedforward network
 * (8 → 64 → 64 → 64 → 64 → 8) with ReLU on hidden layers. Weights are loaded
 * from a .nnwt binary format (little-endian, version 2).
 *
 * When built with CUDA (CTP_ENABLE_NEUROPRESS_GPU=1), inference and online
 * SGD training run as real CUDA kernels operating on device-resident
 * weights -- matching the original NeuroPress project's own design
 * (src/nn/nn_gpu.cu), where the host never touches the decision data. When
 * CUDA is unavailable, or no device is present at runtime, this falls back
 * to an equivalent pure-C++ CPU implementation (same algorithm, same
 * constants) so the predictor still works, just without GPU acceleration.
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
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if CTP_ENABLE_NEUROPRESS_GPU
#include "clio_ctp/compress/model/neuropress_nn_gpu_kernels.h"
#endif

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

  /**
   * @brief Predict metrics for a whole candidate batch in one call.
   *
   * On the GPU backend this is the REAL batched path: one kernel launch
   * scores every candidate (mirrors NeuroPress's nnFusedInferenceKernel),
   * rather than the base class's default of looping Predict() once per
   * candidate. Rank() already calls PredictBatch() exactly once with the
   * full candidate set, so overriding this (not Predict()) is what makes
   * Rank() itself GPU-batched for free. On the CPU fallback this is
   * equivalent to looping Predict().
   */
  std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures>& batch) override;

  /**
   * @brief Online SGD update from real (predicted vs. actual) outcomes.
   *
   * Ports NeuroPress's nnSGDKernel (src/nn/nn_gpu.cu) to plain scalar CPU
   * code: per-sample forward pass, log1p-encoded clamped targets, noise
   * gating, uncertainty weighting (Kendall et al. 2018 log-variance),
   * per-output backward passes with PCGrad-lite gradient-conflict
   * projection and per-output gradient clipping, trust-region step sizing,
   * EMA-smoothed updates with anti-flip damping, and weight clamping.
   * State (log_var_, ema_weights_/ema_biases_, sgd_call_count_) persists
   * across calls, exactly as the original's device-resident NNWeightsGPU
   * does across kernel launches.
   *
   * @param features Per-sample candidate features (up to 8 per call,
   *   matching NeuroPress's NN_MAX_SGD_SAMPLES batch size -- extra
   *   samples beyond the first 8 are ignored, matching upstream's own
   *   truncation).
   * @param labels Per-sample real outcomes, same length as features.
   * @return true if the weight update was applied (false if not ready,
   *   inputs are malformed, or the computed gradient/step was non-finite
   *   and the update was safely skipped).
   */
  /**
   * @brief Set the online-SGD learning rate.
   *
   * Non-positive values are IGNORED, not applied -- upstream guards the
   * same way (`if (learning_rate > 0.0f) g_reinforce_lr = learning_rate;`,
   * gpucompress_learning.cpp:200), so a caller passing 0 leaves the default
   * intact rather than disabling learning.
   */
  void SetLearningRate(float learning_rate) {
    if (learning_rate > 0.0f) learning_rate_ = learning_rate;
  }

  /** @brief Current online-SGD learning rate. */
  float GetLearningRate() const { return learning_rate_; }

  bool Train(const std::vector<CompressionFeatures>& features,
            const std::vector<TrainingLabels>& labels) override;

  /**
   * @brief Deferred, head-only SGD for the DEcompression-time output.
   *
   * Port of NeuroPress's nnBatchedDecompSGDKernel (src/nn/nn_gpu.cu) and its
   * driver gpucompress_batched_decomp_sgd() (src/api/gpucompress_learning.cpp).
   *
   * Decompression time is the one label that cannot be known when the data is
   * compressed -- only a later read reveals it, possibly much later, possibly
   * never. So upstream splits it out: Train() above deliberately leaves output
   * 1's head weights alone (`if (out == 1) continue;`), and this batched pass
   * owns them, fed by real measured times joined back to the features the
   * prediction was originally made from.
   *
   * Updates ONLY w5 row 1 and b5[1]. The trunk (W1-W4) is read-only here --
   * a decompression-time miss must not perturb the shared representation the
   * other three heads depend on.
   *
   * Uses its own trust region, deliberately different from Train()'s:
   * step = clamp(0.15 * mean|err|, 1e-4, 0.05), weights clamped to +-5.
   *
   * @param features Per-sample features, as originally predicted from.
   * @param decompression_times_ms Real measured times, same length. Entries
   *   <= 0 are skipped (not measured).
   * @return true if a weight update was applied; false if not ready, inputs
   *   mismatch, or every sample was gated out as noise.
   */
  bool TrainDecompHead(
      const std::vector<CompressionFeatures>& features,
      const std::vector<double>& decompression_times_ms);

  /**
   * @brief Current weights/biases, in per-layer order (L0..L4).
   *
   * For differential testing against the upstream NeuroPress implementation
   * (see test/unit/compress/model/parity/) -- comparing two implementations
   * by reading their source is not the same as diffing their weights after
   * running both on identical inputs.
   *
   * Syncs from the device copy first when GPU inference is active, so the
   * caller always sees the weights inference would actually use.
   */
  const std::vector<float>& DebugWeights();
  const std::vector<float>& DebugBiases();

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

  /**
   * Online-SGD learning rate -- Clio's stand-in for NeuroPress's
   * g_reinforce_lr global (gpucompress_api.cpp:102), which upstream passes
   * to runNNSGD on every call and exposes via gpucompress_set_reinforcement().
   * Held per-predictor rather than globally so two predictors cannot fight
   * over it; set from CompressorConfig at load time.
   */
  float learning_rate_ = 0.01f;

  /**
   * Serializes every writer of the model state against every other.
   *
   * NeuroPress takes g_sgd_mutex around each SGD dispatch
   * (gpucompress_compress.cpp:719 and :1021, gpucompress_learning.cpp:100),
   * runs all SGD on a dedicated stream, and makes inference wait on a
   * completion event before reading the weights (nn_gpu.cu:1951-1958).
   *
   * Clio needs this for a sharper reason than upstream does. Upstream's
   * decompression-head update is a device kernel touching only w5[1] and
   * b5[1]; Clio's TrainDecompHead is a host-side read-modify-write that
   * downloads ALL parameters, edits two of them, and uploads everything
   * back. A Train() landing inside that window is silently reverted --
   * the upload writes a pre-Train snapshot of the entire trunk. Train()
   * and TrainDecompHead() run from different runtime tasks (DynamicSchedule
   * and Decompress), so the window is reachable.
   *
   * Held by shared_ptr, not by value: this class keeps its defaulted
   * copy/move constructors (a copy shares the device weight handle), and a
   * bare std::mutex member would delete them. Sharing the lock alongside
   * the state it guards is also the correct semantic.
   */
  std::shared_ptr<std::mutex> model_mutex_ = std::make_shared<std::mutex>();

#if CTP_ENABLE_NEUROPRESS_GPU
  // Device-resident weights + online-learning state (log_var, EMA gradient,
  // call count all live on the GPU too -- see neuropress_nn_gpu_kernels.cu).
  // shared_ptr with a custom deleter (not unique_ptr) so this class's
  // existing defaulted copy/move constructors stay correct: a copy shares
  // the device handle rather than double-freeing it. Null when CUDA is
  // compiled in but no device was available at Load() time -- Predict()/
  // Train() fall back to the CPU path below in that case.
  std::shared_ptr<gpu::NeuroPressGpuWeights> gpu_weights_;
#endif

  // ---- Online-learning (SGD) state -- ported from NeuroPress's
  // nnSGDKernel (src/nn/nn_gpu.cu). Persists across Train() calls, mirroring
  // the original's device-resident NNWeightsGPU fields. Sized/zeroed in
  // Load() once weights_/biases_ are known. Not part of the .nnwt file
  // format (matches upstream: log_var/EMA/call_count are runtime-only
  // learning state, never serialized to the weight file either).
  static constexpr int kMaxSGDSamples = 8;  // NeuroPress's NN_MAX_SGD_SAMPLES
  std::vector<float> log_var_;       // [kOutputDim] uncertainty log-variance
  std::vector<float> ema_weights_;   // same layout as weights_, grad EMA
  std::vector<float> ema_biases_;    // same layout as biases_, grad EMA
  int sgd_call_count_ = 0;

  /** @brief Cached forward-pass activations for one sample (SGD backprop). */
  struct SGDActivations {
    std::vector<float> x;                        // [kInputDim] standardized
    std::vector<float> z1, h1, z2, h2, z3, h3, z4, h4;  // [kHiddenDim] each
    std::vector<float> y;                         // [kOutputDim] raw output
  };

  /** @brief Forward pass that additionally caches per-layer activations. */
  SGDActivations ForwardWithCache(const std::vector<float>& x_norm) const;
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_PREDICTOR_H_
