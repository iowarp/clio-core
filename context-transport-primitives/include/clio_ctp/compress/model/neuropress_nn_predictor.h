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
 * (internal.hpp:167-172), which the inference kernel inverts to rebuild the
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
   * On a GPU build this is EQUIVALENT to IsReady(), and deliberately so:
   * Load() refuses rather than returning a predictor backed by the host port,
   * so there is no ready-but-host-only state for the two to disagree about.
   * That equivalence is the invariant "NeuroPress never runs on the CPU"
   * reduced to something a test can assert, and
   * ctp_neuropress_device_path_parity asserts exactly it.
   *
   * It is therefore NOT a guard callers should branch on -- IsReady() already
   * carries the meaning, and upstream likewise exposes a single predicate
   * (gpucompress_nn_is_loaded, gpucompress.h:365, set only once the weights
   * reach the device at nn_gpu.cu:1790) rather than one for "loaded" and
   * another for "on the GPU". Adding a second gate here would be a second
   * gate upstream does not have and, on a GPU build, one that can never fire.
   *
   * The two differ only under CTP_ENABLE_NEUROPRESS_GPU=0, a build upstream
   * cannot be compiled for at all.
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
   * Not an override: the CompressionPredictor interface is shared with the
   * Q-table, linreg and XGBoost models, none of which has a device path, and
   * widening it for one model would put a CUDA concept in everyone's way.
   * This is the NeuroPress-specific entry the bridge reaches for when the
   * chunk is device-resident.
   *
   * Why it exists: PredictBatch takes a host-built [n][8] matrix, so the
   * chunk's entropy/MAD/second-derivative must already have been copied down
   * from the GPU -- and computing them meant a further host round trip in the
   * middle of the statistics pipeline. Upstream has neither: it keeps an
   * AutoStatsGPU on the device and its inference kernel reads it there
   * ("Stats remain on GPU", gpucompress_compress.cpp:281). This restores that.
   *
   * Inputs 5-7 are taken from `device_stats`, NOT from `batch` -- the entropy,
   * MAD and second-derivative fields of the passed features are ignored, and
   * callers need not populate them. Everything else (algorithm, the two
   * preprocessor bits, the error bound, the chunk size) is host knowledge and
   * still comes from `batch`.
   *
   * @param device_stats Device pointer from ctp::ComputeDeviceStatsResident().
   * @param batch Candidates, as for PredictBatch. Must be non-empty; every
   *   entry's chunk_size_bytes must agree (they describe one chunk).
   * @param stream The stream the statistics were enqueued on --
   *   ctp::DeviceStatsStream(). Passing another breaks the GPU-side chaining.
   * @return Predictions, or EMPTY on failure. Never a zero-filled vector.
   */
  /**
   * @param weights,out_order Optional. When both are given and the cost model
   *   is on, the cost, the argmax and the ORDERING are computed on the GPU too
   *   -- one warp, upstream's own bitonic network (nn_gpu.cu:499-532) -- and
   *   out_order comes back holding candidate slots best-first. Upstream ranks
   *   inside its inference kernel and never returns an unranked candidate set,
   *   so leaving this to the host was the last stage of the decision that
   *   crossed back. Omit them for a pure inference call; the ordering is then
   *   the caller's problem, as before.
   */
  std::vector<CompressionPrediction> PredictBatchDeviceStats(
      const void* device_stats,
      const std::vector<CompressionFeatures>& batch, void* stream,
      const RankingWeights* weights = nullptr,
      std::vector<int>* out_order = nullptr, double min_psnr = 0.0,
      std::vector<double>* out_scores = nullptr);

  /**
   * @brief Online SGD update from real (predicted vs. actual) outcomes.
   *
   * Ports NeuroPress's nnSGDKernel (src/nn/nn_gpu.cu) to plain scalar CPU
   * code: per-sample forward pass, log1p-encoded clamped targets, noise
   * gating, uncertainty weighting (Kendall et al. 2018 log-variance),
   * per-output backward passes with PCGrad-lite gradient-conflict
   * projection and per-output gradient clipping, trust-region step sizing,
   * EMA-smoothed updates with anti-flip damping, and weight clamping.
   * All of that state (log-variance, EMA gradient, call count) lives in
   * device memory for the handle's lifetime, exactly as the original's
   * NNWeightsGPU does across kernel launches.
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
  /**
   * @brief Build the 8-element NN input.
   *
   * @param apply_lossless_sentinel Substitute 1e-7 for a lossless config's
   *   error bound. TRUE only for INFERENCE: upstream applies the sentinel in
   *   nnFusedInferenceKernel (nn_gpu.cu:144, "training used 1e-7 sentinel for
   *   lossless") and NOWHERE else. Both SGD kernels feed the RAW bound --
   *   nnSGDKernel's `raw[3] = eb_enc` under the comment "Use raw values for
   *   error_bound and data_size (no log encoding)" (:85), and
   *   nnBatchedDecompSGDKernel's `raw[3] = samp.error_bound_enc` -- so every
   *   training path must pass false or it trains against an input the
   *   original never builds.
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
   *  matching its own truncation (nn_weights.h:17). */
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

};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_NEUROPRESS_NN_PREDICTOR_H_
