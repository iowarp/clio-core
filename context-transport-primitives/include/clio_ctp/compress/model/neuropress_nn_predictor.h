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
 * @brief CompressionFactory base_id -> NeuroPress's trained algo index (0-7).
 *
 * The order is ALGORITHM_NAMES from the training code
 * (neural_net/core/configs.py): lz4, snappy, deflate, gdeflate, zstd, ans,
 * cascaded, bitcomp. Exposed because it is not merely an input encoding: it
 * is half of upstream's ACTION index, and therefore decides tie-breaks. A
 * candidate list built in any other order resolves ties differently from
 * upstream even when every prediction agrees.
 *
 * Anything outside the trained action space returns a deterministic in-range
 * value; such candidates are filtered out before ranking (see
 * neuropress_bridge.cc) and never reach the model.
 */
int NeuroPressAlgoIdForBaseId(int base_id);

/**
 * @brief The action index upstream would give this configuration.
 *
 * `action = algo + 8*quantize + 16*shuffle` -- decodeAction's encoding
 * (internal.hpp), which the inference kernel inverts to rebuild the
 * per-config inputs. Ordering a candidate list by this makes slot order and
 * action order the same thing, so "first enumerated wins" and upstream's
 * "lowest action index wins" become the same rule.
 */
inline int NeuroPressActionId(int base_id, bool quantize, bool byte_shuffle) {
  return NeuroPressAlgoIdForBaseId(base_id) + (quantize ? 8 : 0) +
         (byte_shuffle ? 16 : 0);
}

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
   * @brief True when inference and SGD actually run as CUDA kernels.
   *
   * On a GPU build this is EQUIVALENT to IsReady(), deliberately: Load()
   * refuses rather than returning a predictor backed by the host port, so there
   * is no ready-but-host-only state. That equivalence is the invariant
   * "NeuroPress never runs on the CPU" reduced to something a test can assert,
   * and ctp_neuropress_device_path_parity asserts exactly it.
   *
   * Not a guard to branch on -- IsReady() already carries the meaning, and
   * upstream likewise exposes a single predicate. The two differ only under
   * CTP_ENABLE_NEUROPRESS_GPU=0, a build upstream has no equivalent of.
   */
  bool GpuInferenceActive() const {
#if CTP_ENABLE_NEUROPRESS_GPU
    return gpu_weights_ != nullptr;
#else
    return false;
#endif
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
   * @brief PredictBatch, with the data features read from DEVICE memory.
   *
   * Not an override: the CompressionPredictor interface is shared with models
   * that have no device path, and widening it for one would put a CUDA concept
   * in everyone's way. PredictBatch takes a host-built [n][8] matrix, so the
   * chunk's statistics would have to be copied down first; upstream instead
   * keeps them on the device and reads them there, which this restores.
   *
   * Inputs 5-7 are taken from `device_stats`, NOT from `batch` -- the entropy,
   * MAD and second-derivative fields of the passed features are ignored.
   * Everything else (algorithm, the two preprocessor bits, the error bound, the
   * chunk size) is host knowledge and still comes from `batch`.
   *
   * @param device_stats Device pointer from ctp::ComputeDeviceStatsResident().
   * @param batch Candidates, as for PredictBatch. Must be non-empty; every
   *   entry's chunk_size_bytes must agree (they describe one chunk).
   * @param stream The stream the statistics were enqueued on --
   *   ctp::DeviceStatsStream(). Passing another breaks the GPU-side chaining.
   * @param weights,out_order Optional. When both are given and the cost model
   *   is on, the cost, the argmax and the ordering are computed on the GPU too
   *   -- upstream's own bitonic network -- and out_order comes back holding
   *   candidate slots best-first. Omit them for a pure inference call.
   * @return Predictions, or EMPTY on failure. Never a zero-filled vector.
   */
  std::vector<CompressionPrediction> PredictBatchDeviceStats(
      const void* device_stats,
      const std::vector<CompressionFeatures>& batch, void* stream,
      const RankingWeights* weights = nullptr,
      std::vector<int>* out_order = nullptr, double min_psnr = 0.0,
      std::vector<double>* out_scores = nullptr);

  /**
   * @brief Set the online-SGD learning rate.
   *
   * Non-positive values are IGNORED, not applied -- upstream guards the
   * same way (`if (learning_rate > 0.0f) g_reinforce_lr = learning_rate;`,
   * gpucompress_learning.cpp), so a caller passing 0 leaves the default
   * intact rather than disabling learning.
   */
  void SetLearningRate(float learning_rate) {
    if (learning_rate > 0.0f) learning_rate_ = learning_rate;
  }

  /** @brief Current online-SGD learning rate. */
  float GetLearningRate() const { return learning_rate_; }

  /**
   * @brief Online SGD update from real (predicted vs. actual) outcomes.
   *
   * Ports NeuroPress's nnSGDKernel: per-sample forward pass, log1p-encoded
   * clamped targets, noise gating, uncertainty weighting (Kendall et al. 2018
   * log-variance), per-output backward passes with PCGrad-lite gradient
   * projection and per-output clipping, trust-region step sizing, EMA-smoothed
   * updates with anti-flip damping, and weight clamping. That state lives in
   * device memory for the handle's lifetime, as upstream's NNWeightsGPU does.
   *
   * Leaves output 1's head weights alone -- see TrainDecompHead().
   *
   * @param features Per-sample candidate features. Up to NN_MAX_SGD_SAMPLES (8)
   *   per call; extras are ignored, matching upstream's own truncation.
   * @param labels Per-sample real outcomes, same length as features.
   * @return true if the update was applied; false if not ready, inputs are
   *   malformed, or the computed gradient was non-finite and safely skipped.
   */
  bool Train(const std::vector<CompressionFeatures>& features,
            const std::vector<TrainingLabels>& labels) override;

  /**
   * @brief Deferred, head-only SGD for the DEcompression-time output.
   *
   * Port of NeuroPress's nnBatchedDecompSGDKernel and its driver.
   *
   * Decompression time is the one label unknowable at compress time -- only a
   * later read reveals it, possibly never. Upstream therefore splits it out:
   * Train() leaves output 1's head weights alone, and this batched pass owns
   * them, fed by measured times joined back to the features the prediction was
   * originally made from.
   *
   * Updates ONLY w5 row 1 and b5[1]; the trunk is read-only here, so a
   * decompression-time miss cannot perturb the representation the other three
   * heads depend on. Uses its own trust region, deliberately unlike Train()'s:
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
   * @brief Build the 8-element NN input from CompressionFeatures.
   *
   * Maps library_config_id to an algo_id (0-7, exact for the 8 trained nvcomp
   * algorithms -- see NeuroPressAlgoIdForBaseId()); quantize, byte_shuffle and
   * error_bound pass through; data_size, entropy, mad and second_derivative are
   * used as-is.
   *
   * @param features Input features.
   * @param apply_lossless_sentinel Substitute 1e-7 for a lossless config's
   *   error bound. TRUE only for INFERENCE -- upstream applies the sentinel in
   *   its inference kernel and nowhere else; both SGD kernels feed the RAW
   *   bound, so every training path must pass false or it trains against an
   *   input upstream never builds.
   * @return 8-element vector in NeuroPress order.
   */
  std::vector<float> FeaturesTo8Input(
      const CompressionFeatures& features,
      bool apply_lossless_sentinel = true) const;

  // Architecture parameters.
  static constexpr uint32_t kInputDim = 8;
  static constexpr uint32_t kHiddenDim = 64;
  static constexpr uint32_t kOutputDim = 8;
  static constexpr uint32_t kNumLayers = 5;  // 4 hidden + 1 output
  /** Upstream's NN_MAX_SGD_SAMPLES: samples past the first 8 are dropped,
   *  matching its own truncation (nn_weights.h). */
  static constexpr int kMaxSGDSamples = 8;

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
   * g_reinforce_lr global (gpucompress_api.cpp), which upstream passes
   * to runNNSGD on every call and exposes via gpucompress_set_reinforcement().
   * Held per-predictor rather than globally so two predictors cannot fight
   * over it; set from CompressorConfig at load time.
   */
  float learning_rate_ = 0.01f;

  /**
   * Serializes every writer of the model state against every other.
   *
   * Upstream takes a mutex around each SGD dispatch, runs all SGD on a
   * dedicated stream, and makes inference wait on a completion event.
   *
   * Clio needs this more sharply: upstream's decompression-head update is a
   * device kernel touching only w5[1] and b5[1], while TrainDecompHead is a
   * host-side read-modify-write that downloads ALL parameters, edits two, and
   * uploads everything back. A Train() landing in that window is silently
   * reverted. The two run from different runtime tasks (DynamicSchedule and
   * Decompress), so the window is reachable.
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

};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_PREDICTOR_H_
