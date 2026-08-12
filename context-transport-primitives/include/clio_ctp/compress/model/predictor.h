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

#include <algorithm>
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

  // Preprocessor selection for this candidate. Modeled by the NeuroPress NN
  // (see FeaturesTo8Input); intentionally NOT part of ToVector()'s 11 features
  // so previously trained tree/table models remain valid.
  double quantize = 0;      /**< 1 if error-bounded quantization applied */
  double byte_shuffle = 0;  /**< 1 if byte-plane shuffle applied */
  double error_bound = 0;   /**< Error bound for lossy / quantized paths */

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
  /**
   * Predicted DEcompression time (ms). The NeuroPress NN emits this as its
   * own output (index 1, distinct from compression time at index 0) and
   * ranks with it -- see RankingWeights' cost model. 0 means "not
   * predicted": models that don't emit it (qtable/linreg/dense-NN) leave it
   * here, and the cost model treats 0 as "fall back to the compression-time
   * estimate" rather than as a real zero-cost decompression.
   */
  double decompression_time_ms = 0;
  double inference_time_ms = 0;   /**< Model inference latency (ms) */

  CompressionPrediction() = default;
  CompressionPrediction(double ratio, double psnr, double compress_time,
                        double infer_time)
      : compression_ratio(ratio),
        psnr_db(psnr),
        compression_time_ms(compress_time),
        inference_time_ms(infer_time) {}
  CompressionPrediction(double ratio, double psnr, double compress_time,
                        double decompress_time, double infer_time)
      : compression_ratio(ratio),
        psnr_db(psnr),
        compression_time_ms(compress_time),
        decompression_time_ms(decompress_time),
        inference_time_ms(infer_time) {}

  /** @brief Number of predicted outputs (ratio, psnr, time). */
  static constexpr size_t NumOutputs() { return 3; }
};

/**
 * @brief Data-intrinsic features (independent of the compressor choice).
 *
 * These come from preprocessing the actual data buffer (see
 * clio_ctp/compress/preprocess/FeatureExtractor). They are the fixed part of a
 * bulk ranking query: one DataFeatures is scored against many CandidateConfig.
 */
struct DataFeatures {
  double chunk_size_bytes = 0;
  double target_cpu_util = 0;
  double shannon_entropy = 0;
  double mad = 0;
  double second_derivative_mean = 0;
  double data_type_char = 0;   /**< One-hot: 1 if char/int data */
  double data_type_float = 0;  /**< One-hot: 1 if float data */
};

/**
 * @brief One compressor+preset+preprocessor combination to evaluate.
 *
 * base_id/preset_id follow the CompressionFactory encoding
 * (library_config_id = base_id * 10 + preset_id; preset FAST=1, BALANCED=2,
 * BEST=3). quantize/byte_shuffle select preprocessors applied before the
 * compressor.
 */
struct CandidateConfig {
  int base_id = 0;             /**< Compressor library base id (frozen) */
  int preset_id = 2;           /**< 1 FAST, 2 BALANCED, 3 BEST */
  bool quantize = false;       /**< Apply error-bounded quantization first */
  bool byte_shuffle = false;   /**< Apply byte-plane shuffle first */
  double error_bound = 0;      /**< Error bound for lossy / quantized paths */
  std::string library_name;    /**< Optional human-readable label */

  /** @brief CompressionFactory-style encoded id (base_id*10 + preset_id). */
  double LibraryConfigId() const {
    return static_cast<double>(base_id * 10 + preset_id);
  }
};

/**
 * @brief Scalar scoring of a prediction for ranking (higher = better).
 *
 * The default rewards compression ratio only. Penalize time or reward PSNR by
 * setting the corresponding weights; e.g. w_compress_time>0 to prefer faster
 * compressors, w_psnr>0 to value lossy reconstruction quality.
 */
struct RankingWeights {
  double w_ratio = 1.0;          /**< Reward per unit compression ratio */
  double w_compress_time = 0.0;  /**< Penalty per ms of compression time */
  double w_psnr = 0.0;           /**< Reward per dB of PSNR */

  /**
   * Cost-model ranking (NeuroPress's own policy, nn_gpu.cu):
   *
   *   cost = w0*comp_time + w1*decomp_time + w2*data_size/(ratio*bandwidth)
   *
   * and the best candidate is the one MINIMIZING it -- so Score() returns
   * -cost, keeping "higher score is better" for the shared sort. This
   * balances the three things that actually cost wall-clock time: producing
   * the compressed bytes, reading them back, and moving the (now smaller)
   * bytes to/from storage. Ranking on ratio alone -- the w_ratio default
   * above -- silently prefers a codec that compresses marginally better but
   * takes far longer, which is the opposite of what a storage tier wants.
   *
   * Off by default so existing callers/models keep the ratio-only behavior;
   * NeuroPressCandidateStats opts in. Weight and bandwidth defaults mirror
   * g_rank_w0/w1/w2 (all 1.0) and g_measured_bw_bytes_per_ms (5e6 bytes/ms
   * = 5 GB/s) from gpucompress_api.cpp.
   */
  bool use_cost_model = false;
  double w_cost_compress_time = 1.0;
  double w_cost_decompress_time = 1.0;
  double w_cost_io = 1.0;
  double bandwidth_bytes_per_ms = 5e6;

  double Score(const CompressionPrediction &p) const {
    return w_ratio * p.compression_ratio -
           w_compress_time * p.compression_time_ms + w_psnr * p.psnr_db;
  }

  /**
   * Cost-model score for one candidate. `data_size_bytes` is the chunk being
   * compressed (DataFeatures::chunk_size_bytes). Falls back to Score(p) when
   * the cost model is off.
   */
  double Score(const CompressionPrediction &p, double data_size_bytes) const {
    if (!use_cost_model) {
      return Score(p);
    }
    // Same policy clamps NeuroPress applies before ranking (nn_gpu.cu):
    // times floor at 1ms, ratio caps at 100x.
    double ct = std::max(1.0, p.compression_time_ms);
    // A model that doesn't predict decompression time reports 0; use the
    // compression-time estimate rather than scoring it as free.
    double dt = (p.decompression_time_ms > 0.0)
                    ? std::max(1.0, p.decompression_time_ms)
                    : ct;
    // Floor at 0.1 as well as capping at 100: nn_gpu.cu:221 clamps ratio to
    // [0.1, 1e5] before the policy cap, so io_cost can never be divided by a
    // vanishing or negative ratio. Callers whose predictor already floors
    // (NeuroPressNNPredictor does) are unaffected; this protects the ones
    // that do not.
    double ratio = std::max(0.1, std::min(100.0, p.compression_ratio));
    double bw = (bandwidth_bytes_per_ms > 0.0) ? bandwidth_bytes_per_ms : 1.0;
    double io_cost = (ratio > 0.0) ? (data_size_bytes / (ratio * bw)) : 1e30;
    double cost = w_cost_compress_time * ct + w_cost_decompress_time * dt +
                  w_cost_io * io_cost;
    return -cost;  // lower cost is better
  }
};

/**
 * @brief A candidate paired with its predicted metrics and ranking score.
 */
struct RankedPrediction {
  CandidateConfig candidate;
  CompressionPrediction prediction;
  double score = 0;  /**< Per RankingWeights; results are sorted by this. */
};

/**
 * @brief Compose the full model feature vector from data + a candidate.
 *
 * Bridges the bulk-ranking inputs (DataFeatures + CandidateConfig) to the
 * per-config CompressionFeatures the models consume.
 */
inline CompressionFeatures MakeCompressionFeatures(const DataFeatures &d,
                                                   const CandidateConfig &c) {
  CompressionFeatures f;
  f.chunk_size_bytes = d.chunk_size_bytes;
  f.target_cpu_util = d.target_cpu_util;
  f.shannon_entropy = d.shannon_entropy;
  f.mad = d.mad;
  f.second_derivative_mean = d.second_derivative_mean;
  f.data_type_char = d.data_type_char;
  f.data_type_float = d.data_type_float;
  f.library_config_id = c.LibraryConfigId();
  f.config_fast = (c.preset_id == 1) ? 1.0 : 0.0;
  f.config_balanced = (c.preset_id == 2) ? 1.0 : 0.0;
  f.config_best = (c.preset_id == 3) ? 1.0 : 0.0;
  f.quantize = c.quantize ? 1.0 : 0.0;
  f.byte_shuffle = c.byte_shuffle ? 1.0 : 0.0;
  f.error_bound = c.error_bound;
  return f;
}

/**
 * @brief Training labels for the multi-output model: real, observed outcomes
 * of an actual compress/decompress, paired with the CompressionFeatures that
 * were used to pick the candidate. Feed to Train() for online learning.
 *
 * Convention (matches NeuroPress's own SGDSample semantics exactly, so a
 * predictor porting its training rule doesn't need a second convention):
 *  - compression_time_ms / decompression_time_ms: 0 (default) means "not
 *    measured, skip this output's gradient" -- only a strictly positive
 *    measured duration trains that head.
 *  - psnr_db: negative means "not applicable, skip this output's gradient"
 *    (e.g. a lossless candidate has no PSNR); 0 means "measured and
 *    lossless" (treated as maximal quality); positive is a real measured
 *    PSNR in dB.
 */
struct TrainingLabels {
  float compression_ratio = 0;
  float psnr_db = 0;
  float compression_time_ms = 0;
  float decompression_time_ms = 0;

  TrainingLabels() = default;
  TrainingLabels(float ratio, float psnr, float time,
                 float decompress_time = 0)
      : compression_ratio(ratio),
        psnr_db(psnr),
        compression_time_ms(time),
        decompression_time_ms(decompress_time) {}
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
 * The PRIMARY entry point is Rank(): given the data statistics and a set of
 * candidate compressor/preprocessor configurations, it returns every candidate
 * scored and sorted best-first. Predict()/PredictBatch() are the per-config
 * primitives Rank() is built on. Train() and the reinforcement-learning hooks
 * are optional (default no-op) so pure-inference models need not implement them.
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

  /**
   * @brief PRIMARY API: bulk-rank candidate configurations for one data buffer.
   *
   * Scores each CandidateConfig (compressor + preset + preprocessor combo)
   * against the given DataFeatures and returns them sorted best-first by
   * RankingWeights::Score. Every model supports this via the default
   * implementation, which composes CompressionFeatures per candidate and calls
   * PredictBatch once; models with a native multi-config forward pass (e.g. the
   * NeuroPress NN) may override for efficiency without changing the semantics.
   *
   * @param data       Data-intrinsic statistics (fixed across candidates).
   * @param candidates Compressor/preprocessor combinations to evaluate.
   * @param weights    How to score a prediction into a rank (default: ratio).
   * @return Ranked results, highest score first.
   */
  virtual std::vector<RankedPrediction> Rank(
      const DataFeatures &data, const std::vector<CandidateConfig> &candidates,
      const RankingWeights &weights = RankingWeights()) {
    std::vector<CompressionFeatures> feats;
    feats.reserve(candidates.size());
    for (const auto &c : candidates) {
      feats.push_back(MakeCompressionFeatures(data, c));
    }
    std::vector<CompressionPrediction> preds = PredictBatch(feats);
    std::vector<RankedPrediction> ranked;
    ranked.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      ranked.push_back({candidates[i], preds[i],
                        weights.Score(preds[i], data.chunk_size_bytes)});
    }
    // stable_sort, not sort: ties are reachable because both sides clamp
    // hard (times floor at 1 ms, ratio caps at 100x), so a highly
    // compressible chunk can saturate several candidates to a bit-identical
    // cost. std::sort leaves their relative order unspecified -- it can
    // differ between container sizes and library versions -- whereas
    // NeuroPress's bitonic ranking network uses strict comparators and never
    // swaps equal keys, so it always returns the lowest action index. Stable
    // sorting reproduces that: first enumerated wins.
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const RankedPrediction &a, const RankedPrediction &b) {
                       return a.score > b.score;
                     });
    return ranked;
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
