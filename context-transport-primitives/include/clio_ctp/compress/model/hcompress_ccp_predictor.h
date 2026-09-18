/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/** @file hcompress_ccp_predictor.h
 *  @brief HCompress's Expected-Compression-Cost predictor, as a baseline.
 *
 *  A reimplementation of the cost model in HCompress (Devarajan et al.,
 *  "HCompress: Hierarchical Data Compression for Multi-Tiered Storage
 *  Environments", IPDPS 2020, Sec. IV-D), so our own per-chunk model can be
 *  scored against a published alternative on identical inputs.
 *
 *  WHAT THE PAPER SPECIFIES, AND WHAT THAT IMPLIES
 *
 *  - Form: a linear regression (the paper uses dlib), predicting the Expected
 *    Compression Cost of one compression library.
 *  - Inputs: data type, data format, compression library, data distribution.
 *    The distribution is a CLASS -- Normal, Gamma, Exponential or Uniform --
 *    "deduced by sub-sampling the buffer". There are NO per-chunk statistics:
 *    no entropy, no MAD, no derivative. All four inputs are categorical.
 *  - Outputs: compression speed (MB/s), decompression speed (MB/s) and
 *    compression ratio. NO quality metric -- so PSNR, SSIM and pointwise error
 *    have no prediction at all, and this class reports psnr_db < 0 ("not
 *    predicted") rather than 0, which in this codebase means "lossless, i.e.
 *    maximal quality" and would be a fabricated answer.
 *  - Seeding: an offline profiler benchmarks every library over sample data
 *    and writes a JSON seed; the model starts from that seed (Seed()/Save()/
 *    Load()).
 *  - Feedback: every n operations the measured cost of the EXECUTED choice is
 *    fed back and the model is updated (Observe() + ApplyFeedback()).
 *
 *  Two consequences of the input list are worth stating plainly, because they
 *  are properties of the design and not of this implementation:
 *
 *  1. SPEED, NOT TIME, is what the model regresses. Speed is size-independent,
 *     which is why the paper needs no size input; a time is then recovered as
 *     bytes / speed. This class therefore predicts a per-chunk TIME that
 *     varies with chunk size even though the regression itself does not see
 *     the size.
 *  2. With every input categorical, a linear regression over their one-hot
 *     encoding is an ADDITIVE per-cell model: within one (type, format,
 *     library, distribution) cell it predicts one number for every chunk. It
 *     cannot, by construction, distinguish two chunks that fall in the same
 *     cell. That is the substantive difference from a per-chunk model, and it
 *     is the thing the comparison is meant to expose -- so it must not be
 *     quietly repaired by feeding this class a statistic the paper does not
 *     give it.
 *
 *  The fit is recursive least squares, which is what makes one mechanism serve
 *  both phases: over the seed with a forget factor of 1 it is exactly ridge
 *  regression and independent of row order, and it then accepts incremental
 *  feedback updates without refitting. dlib's `rls` is the same algorithm.
 */
#ifndef CLIO_CTP_COMPRESS_MODEL_HCOMPRESS_CCP_PREDICTOR_H_
#define CLIO_CTP_COMPRESS_MODEL_HCOMPRESS_CCP_PREDICTOR_H_

#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "clio_ctp/compress/model/predictor.h"

namespace ctp::compress::model {

/** The four categorical inputs for ONE prediction.
 *
 *  `distribution` is deduced once per buffer, not per candidate -- so a caller
 *  ranking 32 candidates classifies once and varies only `library`.
 *
 *  An EMPTY string means "this input is unavailable here", which is different
 *  from an unrecognised value: both contribute no term, but the first is a
 *  property of the deployment (e.g. raw in-situ buffers carry no file format)
 *  and is reported as such, while the second is a gap in the seed. */
struct CcpInputs {
  std::string data_type;    /**< e.g. "float32" */
  std::string data_format;  /**< e.g. "hdf5", "raw"; empty = unavailable */
  std::string library;      /**< the configured compression option */
  std::string distribution; /**< "normal" | "gamma" | "exponential" | "uniform" */
};

/** One measured outcome, either a seed row or a feedback row.
 *
 *  A non-positive measurement means NOT MEASURED and trains nothing: the three
 *  outputs are fitted independently, so a row with a compression time but no
 *  decompression time still teaches the compression-speed head. Substituting a
 *  value would be indistinguishable, downstream, from having measured it. */
struct CcpObservation {
  CcpInputs inputs;
  double bytes = 0;
  double compress_time_ms = 0;
  double decompress_time_ms = 0;
  double compression_ratio = 0;
};

/** Configuration. The defaults are the paper's where it states one and dlib's
 *  `rls` defaults otherwise. */
struct CcpConfig {
  /** dlib rls regularization. Also the prior: P starts at (1/C)I, so a large C
   *  lets the seed dominate quickly. */
  double regularization = 1e-3;
  /** dlib rls forget factor, applied to FEEDBACK only. 1.0 (dlib's default)
   *  weights every observation equally, so late feedback moves a model already
   *  fitted on a large seed very little; below 1.0 the model tracks the current
   *  workload. Seeding always uses 1.0 -- see Seed(). */
  double forget_factor = 1.0;
  /** The paper's configurable n: feedback is applied every n observations. */
  size_t feedback_interval = 1;
  /** Speed floor, so a negative extrapolation cannot become a negative or
   *  absurd time. Reported, not silent: PredictCount/ClampCount expose how
   *  often it bound. */
  double min_speed_mbps = 1.0;
  /** Sanity bounds on the ratio head, matching the ones the NeuroPress kernel
   *  applies to its own ratio output before any policy cap. */
  double min_ratio = 0.1;
  double max_ratio = 1e5;
  /**
   * Clamp applied to a measured RATIO before it is used as a training target.
   * 0 disables it, which is the paper-faithful default.
   *
   * Our own online SGD clamps this target at 100
   * (CLIO_NEUROPRESS_SGD_RATIO_TARGET_CAP; upstream uses 10000), because a
   * measured ratio on these workloads reaches several thousand on a constant
   * chunk and a least-squares fit chases the outlier: 24% of VPIC's and 65% of
   * AI's executed chunks compress by more than 100x. Setting this to the same
   * 100 is therefore the like-for-like comparison -- the same robustness choice
   * on both sides -- and leaving it off is the paper's own. Both are reported,
   * because it changes the answer.
   */
  double ratio_target_cap = 0.0;
};

/** Recursive least squares for one scalar target. */
class CcpRls {
 public:
  void Reset(size_t dim, double regularization);
  /**
   * @brief Set the fit from accumulated normal equations: the SEED path.
   *
   * RLS at forget factor 1.0 converges to the ridge solution, so the seed can
   * be obtained either by streaming every row through Update() or by solving
   * (A + C I) w = b once, where A = sum(x x') and b = sum(x y). The closed
   * form is used because the streaming form does not survive the seed's size:
   * 401k rank-1 updates to a 41x41 inverse-correlation matrix accumulate
   * enough floating-point error to visibly corrupt the fit, and the corruption
   * shows up as an erratic FEEDBACK phase rather than as an obviously wrong
   * seed.
   *
   * P is set to (A + C I)^-1, which is exactly the RLS state after those rows
   * -- so feedback continues from here with correctly sized steps rather than
   * from a fresh prior, which would make the first feedback observation count
   * as much as the whole seed.
   */
  void InitFromRidge(const std::vector<double>& a, const std::vector<double>& b,
                     size_t samples, double regularization);
  /** One RLS step. `forget` of 1.0 makes the result order-independent and
   *  equal to the ridge solution. */
  void Update(const std::vector<double>& x, double y, double forget);
  double Predict(const std::vector<double>& x) const;
  size_t Samples() const { return samples_; }
  const std::vector<double>& Weights() const { return w_; }
  void SetWeights(std::vector<double> w, size_t samples) {
    w_ = std::move(w);
    samples_ = samples;
  }
  bool Ready() const { return samples_ > 0; }

 private:
  std::vector<double> w_;  /**< dim */
  std::vector<double> p_;  /**< dim x dim, row-major */
  size_t dim_ = 0;
  size_t samples_ = 0;
};

/**
 * @brief HCompress's Expected Compression Cost predictor.
 *
 * Usage mirrors the paper's flow:
 *
 * @code
 *   HCompressCcpPredictor ccp;
 *   ccp.Seed(profiler_rows);        // offline profiler -> JSON seed
 *   ccp.Save(dir);                 //   ... written once
 *   ccp.Load(dir);                 //   ... and reloaded at startup
 *
 *   ccp.SetInputs({"float32", "raw", "", "normal"});   // per BUFFER
 *   for (const auto& lib : libraries) {                // per CANDIDATE
 *     auto p = ccp.PredictFor(lib, bytes);
 *   }
 *   ccp.Observe(measured_outcome_of_the_executed_choice);
 *   ccp.ApplyFeedback();           // every feedback_interval observations
 * @endcode
 */
class HCompressCcpPredictor : public CompressionPredictor {
 public:
  HCompressCcpPredictor() = default;
  explicit HCompressCcpPredictor(const CcpConfig& config) : config_(config) {}
  ~HCompressCcpPredictor() override = default;

  HCompressCcpPredictor(const HCompressCcpPredictor&) = delete;
  HCompressCcpPredictor& operator=(const HCompressCcpPredictor&) = delete;

  // ---- CompressionPredictor ------------------------------------------------
  bool Load(const std::string& model_dir) override;
  bool Save(const std::string& model_dir) override;
  bool IsReady() const override;
  ModelType Type() const override { return ModelType::kHCompressCcp; }

  /** Predict for one candidate.
   *
   *  CompressionFeatures carries the library identity (library_config_id plus
   *  the quantize/byte_shuffle preprocessor bits) but has no field for a
   *  distribution class, a data type name or a format -- so those three come
   *  from the per-buffer context set by SetInputs(). The library key is built
   *  from the features by LibraryKey() so that one ranking call varies exactly
   *  what the paper's model varies.
   *
   *  The per-chunk data statistics in `features` (entropy, MAD, second
   *  derivative) are IGNORED, deliberately and by design. */
  CompressionPrediction Predict(const CompressionFeatures& features) override;
  std::vector<CompressionPrediction> PredictBatch(
      const std::vector<CompressionFeatures>& batch) override;

  /** Not supported: this model is seeded and then updated by feedback, which
   *  is Seed() and Observe(). Returning false rather than silently doing
   *  something else keeps a caller from thinking it trained this model. */
  bool Train(const std::vector<CompressionFeatures>&,
             const std::vector<TrainingLabels>&) override {
    return false;
  }

  // ---- HCompress's own flow ------------------------------------------------
  /** Fit from the offline profiler's rows. Always at forget factor 1.0, so the
   *  seed is the ridge solution and does not depend on row order -- a seed that
   *  changed with the order of a profiler's output would make every downstream
   *  number irreproducible. Extends the category vocabularies. */
  void Seed(const std::vector<CcpObservation>& rows);

  /** The per-buffer inputs. Set once per chunk, before ranking its candidates. */
  void SetInputs(const CcpInputs& inputs);
  const CcpInputs& Inputs() const { return inputs_; }

  /** Predict for one library under the current per-buffer inputs. */
  CompressionPrediction PredictFor(const std::string& library,
                                   double bytes) const;
  /** Predict for a fully specified input set, ignoring the stored context. */
  CompressionPrediction PredictWith(const CcpInputs& inputs,
                                    double bytes) const;

  /** Buffer one measured outcome. Applies the update once `feedback_interval`
   *  observations have accumulated; returns true when an update ran. */
  bool Observe(const CcpObservation& row);
  /** Apply whatever is buffered now, whatever the interval. Returns the number
   *  of observations consumed. */
  size_t ApplyFeedback();
  size_t PendingFeedback() const { return pending_.size(); }
  size_t FeedbackUpdates() const { return feedback_updates_; }

  /** The encoded dimension and the vocabularies, for reporting which inputs
   *  actually carried information in a given experiment. */
  size_t Dimension() const;
  std::vector<std::string> Vocabulary(const std::string& field) const;
  size_t SeedRows() const { return seed_rows_; }

  const CcpConfig& Config() const { return config_; }
  void SetConfig(const CcpConfig& c) { config_ = c; }

  /** The library key this model uses for a candidate: the library name plus
   *  the preprocessor bits, because a quantized or shuffled call to the same
   *  codec is a different configured compression option with a different
   *  measured cost. Keying on the bare codec instead would force one
   *  prediction to cover all four variants, which would understate the
   *  baseline. */
  static std::string LibraryKey(const CompressionFeatures& features);
  static std::string LibraryKey(const std::string& library_name, bool quantize,
                                bool byte_shuffle);

 private:
  /** Index of `value` in the vocabulary, appending it when `grow` is set;
   *  npos when absent and not growing. */
  size_t Index(std::vector<std::string>* vocab, const std::string& value,
               bool grow) const;
  std::vector<double> Encode(const CcpInputs& inputs) const;
  void Rebuild(size_t dim);

  CcpConfig config_;
  mutable std::mutex mutex_;

  // Vocabularies, in encoding order after the intercept.
  std::vector<std::string> types_, formats_, libraries_, distributions_;

  CcpRls comp_speed_, decomp_speed_, ratio_;
  CcpInputs inputs_;
  std::vector<CcpObservation> pending_;
  size_t seed_rows_ = 0;
  size_t feedback_updates_ = 0;
};

}  // namespace ctp::compress::model

#endif  // CLIO_CTP_COMPRESS_MODEL_HCOMPRESS_CCP_PREDICTOR_H_
