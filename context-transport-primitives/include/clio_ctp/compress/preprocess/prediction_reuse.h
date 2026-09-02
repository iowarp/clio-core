/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file prediction_reuse.h
 * @brief Does this chunk need a fresh model run, or will the last one do?
 *
 * Everything here is `__host__ __device__` and free of state beyond what is
 * passed in, so the same arithmetic runs inside the inference kernel and
 * inside a host unit test. That is deliberate: the rules this has to get
 * right -- which state is compared against what, and when each is written --
 * are invisible at runtime and cheap to assert on the host.
 *
 * WHY THE DIVERGENCE IS RELATIVE AND NOT SIGMA-NORMALISED.
 * The obvious metric is distance in the units the network normalises by, its
 * own x_stds. Measured on this benchmark that metric is unusable: WarpX E and
 * j fields carry MAD around 1e10 in physical units against an x_std of 0.106,
 * so consecutive timesteps sit ~1e11 "standard deviations" apart and every
 * threshold degenerates to always-run. The model is far outside its training
 * distribution on those fields, and a metric anchored to that distribution
 * inherits the problem. A relative measure does not: it is bounded in [0,1]
 * and reads the same on a field of order 1 and a field of order 1e10.
 *
 * WHY THE MEAN OF THREE AND NOT ONE OF THEM.
 * Measured against whether the SELECTION actually changed, over 190 WarpX and
 * 114 Nyx consecutive-timestep pairs: single features are unstable across
 * workloads (entropy alone is the best predictor on WarpX and the worst on
 * Nyx), while the three-feature combinations are consistently near-best on
 * both. The mean is the one taken; max and L2 measured within noise of it.
 *
 * WHAT THIS CANNOT SEE. Online SGD moves the weights between chunks, so an
 * unchanged signature can still produce a changed prediction. No function of
 * the input detects that; the model's SGD counter does (kModelChanged).
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_PREDICTION_REUSE_H_
#define CLIO_CTP_COMPRESS_PREPROCESS_PREDICTION_REUSE_H_

#include <cstdint>

#include "clio_ctp/compress/preprocess/prediction_reuse_state.h"

#if defined(__CUDACC__)
#define CTP_REUSE_HD __host__ __device__
#else
#define CTP_REUSE_HD
#endif

namespace ctp::compress::preprocess {

/**
 * Why the model ran -- or, for kReuseCachedPrediction, that it did not.
 *
 * A bit field, not an enum, because several reasons genuinely co-occur (a
 * sudden change on the same timestep a refresh comes due) and reporting only
 * the first would make the diagnostics lie about why work happened.
 *
 * Zero is REUSE. That makes "did anything ask for a run?" a test against
 * zero, so a reason added later cannot accidentally be missed by the check.
 */
namespace ReuseDecision {
constexpr uint32_t kReuseCachedPrediction = 0u;
constexpr uint32_t kFirstObservation         = 1u << 0;
constexpr uint32_t kNoPreviousState          = 1u << 1;
constexpr uint32_t kNoAnchorState            = 1u << 2;
constexpr uint32_t kInvalidPrediction        = 1u << 3;
constexpr uint32_t kStepDivergenceExceeded   = 1u << 4;
constexpr uint32_t kAnchorDivergenceExceeded = 1u << 5;
constexpr uint32_t kPeriodicRefresh          = 1u << 6;
constexpr uint32_t kTimestepNotAdvancing     = 1u << 7;
constexpr uint32_t kBlockLayoutChanged       = 1u << 8;
/** Not produced by the decision: exploration never consults it. Defined here
 *  so the runtime's per-chunk log has one vocabulary. */
constexpr uint32_t kExplorationMode          = 1u << 9;
/** The weights changed (an SGD step landed) since the cache was made. */
constexpr uint32_t kModelChanged             = 1u << 10;
}  // namespace ReuseDecision

/**
 * May a chunk be served a cached prediction at all?
 *
 * EXPLORATION IS EXEMPT UNCONDITIONALLY. An exploring run has to rank and
 * measure the full action space at every timestep, however little the data
 * moved between them -- that sweep is how the benchmark learns what each
 * action really costs, and a replayed ranking would both narrow what it
 * visits (the alternatives come from the ranked list) and distort whether it
 * runs at all (the gate gets its error from that same prediction).
 *
 * Note what this predicate does NOT take: a divergence, a threshold, or a
 * state. The exemption is not "reuse less while exploring", it is "never",
 * and a signature identical to the previous timestep does not weaken it.
 * Keeping the rule here rather than inline at the call site is what lets it
 * be asserted rather than reviewed.
 */
CTP_REUSE_HD inline bool PredictionReuseAllowed(bool reuse_enabled,
                                                bool exploration_enabled) {
  return reuse_enabled && !exploration_enabled;
}

/** Anything other than reuse means run the model. */
CTP_REUSE_HD inline bool MustRunModel(uint32_t flags) {
  return flags != ReuseDecision::kReuseCachedPrediction;
}

/** Thresholds and cadence. Defaults are the measured operating point; the
 *  runtime overrides them from the environment. */
struct PredictionReuseThresholds {
  /**
   * D_step above this runs the model. Strict >, so D == step REUSES.
   *
   * Not "a 5% change": the per-feature term is |b-a|/(|a|+|b|), so 0.05 is
   * about a 1.1x change either way and 0.20 about 1.5x. State thresholds in
   * this unit, never as a percentage of the feature.
   *
   * No denominator floor, deliberately. A floor of c*x_std was measured: it
   * removes the near-zero triggers, but some of those pairs change selection,
   * and at matched reuse rates it is never better. The near-zero sensitivity
   * is catching real decision-boundary crossings.
   */
  double step = 0.05;
  /** D_anchor above this runs the model. Strict >. */
  double anchor = 0.20;
  /** Run the model on the Nth OBSERVATION of a lineage since it last ran,
   *  whatever the divergences say -- observations, not simulation steps, so
   *  a workload that dumps every 10 steps is not refreshed on every dump.
   *  Covers SGD weight drift, which no input metric sees. */
  long long refresh_interval = 10;
};

namespace detail {

/** |b-a| / (|a|+|b|), which the triangle inequality bounds by 1. The epsilon
 *  turns 0/0 into 0 for a feature that was zero and stayed zero -- the
 *  all-zero initial condition every one of these workloads starts from. */
CTP_REUSE_HD inline double RelativeChange(double a, double b) {
  const double aa = a < 0.0 ? -a : a;
  const double bb = b < 0.0 ? -b : b;
  double diff = b - a;
  if (diff < 0.0) diff = -diff;
  return diff / (aa + bb + 1e-30);
}

CTP_REUSE_HD inline bool IsFinite(double v) {
  return v == v && v <= 1.7976931348623157e308 && v >= -1.7976931348623157e308;
}

}  // namespace detail

/**
 * Distance between two signatures, in [0, 1].
 *
 * Each term is bounded by the triangle inequality, |b-a| <= |a|+|b|, and
 * with epsilon in the denominator the bound is strict for finite inputs --
 * the exact maximum of 1.0 is reserved for the non-finite case below. The
 * epsilon also means the metric is scale-invariant only APPROXIMATELY: exact
 * for anything representable, perturbed by one part in 1e30/(|a|+|b|).
 *
 * A non-finite feature on either side returns the maximum. Reporting a small
 * divergence there would reuse a stale prediction for a chunk whose
 * statistics have gone bad, which is the one case where "nothing changed" is
 * both wrong and undetectable downstream.
 */
CTP_REUSE_HD inline double SignatureDivergence(const ChunkSignature &a,
                                                  const ChunkSignature &b) {
  if (!detail::IsFinite(a.entropy) || !detail::IsFinite(a.mad) ||
      !detail::IsFinite(a.second_derivative) || !detail::IsFinite(b.entropy) ||
      !detail::IsFinite(b.mad) || !detail::IsFinite(b.second_derivative)) {
    return 1.0;
  }
  return (detail::RelativeChange(a.entropy, b.entropy) +
          detail::RelativeChange(a.mad, b.mad) +
          detail::RelativeChange(a.second_derivative, b.second_derivative)) /
         3.0;
}

/**
 * Decide, and record the two divergences for diagnostics.
 *
 * DOES NOT ADVANCE THE STATE. `previous` and `last_nn` are what this measures
 * against, so writing either here would compare the current signature with
 * itself and report zero divergence forever. Advancing them is
 * CommitChunkObservation's job, called after the decision has been acted
 * on. The only fields written are step_divergence, anchor_divergence and
 * decision_flags, which are outputs of the measurement, not inputs to it --
 * so calling this twice gives the same answer.
 *
 * @param timestep the CURRENT simulation timestep
 * @return ReuseDecision bits; zero means reuse the cached prediction.
 */
CTP_REUSE_HD inline uint32_t DecidePredictionReuse(
    DevicePredictionReuseState *s, const ChunkSignature &now,
    long long timestep, const PredictionReuseThresholds &th,
    double chunk_bytes = -1.0, double error_bound = -1.0,
    /* w->sgd_call_count, or null to skip the check (inference-only). */
    const int *sgd_counter = nullptr) {
  uint32_t flags = ReuseDecision::kReuseCachedPrediction;

  const double d_step =
      s->has_previous ? SignatureDivergence(s->previous, now) : 1.0;
  const double d_anchor =
      s->has_anchor ? SignatureDivergence(s->last_nn, now) : 1.0;
  s->step_divergence = d_step;
  s->anchor_divergence = d_anchor;

  if (!s->has_previous) flags |= ReuseDecision::kFirstObservation |
                                 ReuseDecision::kNoPreviousState;
  if (!s->has_anchor) flags |= ReuseDecision::kNoAnchorState;
  if (!s->has_prediction) flags |= ReuseDecision::kInvalidPrediction;

  // A timestep that does not advance means this is not the series the state
  // describes -- a restart, a replay, or two blocks sharing a key.
  if (s->has_previous && timestep <= s->previous_timestep) {
    flags |= ReuseDecision::kTimestepNotAdvancing;
  }

  // Inputs 3 and 4 changed under a cache made for the old ones. The
  // signature cannot see this; see cached_chunk_bytes in the state header.
  if (s->has_prediction && chunk_bytes >= 0.0 &&
      s->cached_chunk_bytes != chunk_bytes) {
    flags |= ReuseDecision::kBlockLayoutChanged;
  }
  if (s->has_prediction && error_bound >= 0.0 &&
      s->cached_error_bound != error_bound) {
    flags |= ReuseDecision::kBlockLayoutChanged;
  }
  // Same data, different model. An SGD step since the cache was made means
  // the cached prediction came from weights that no longer exist; no input
  // metric can detect that, so it is read straight off the counter.
  if (s->has_prediction && sgd_counter != nullptr &&
      *sgd_counter != s->cached_sgd_calls) {
    flags |= ReuseDecision::kModelChanged;
  }

  // Strict >, so D == threshold reuses. One policy, applied to both.
  if (s->has_previous && d_step > th.step) {
    flags |= ReuseDecision::kStepDivergenceExceeded;
  }
  if (s->has_anchor && d_anchor > th.anchor) {
    flags |= ReuseDecision::kAnchorDivergenceExceeded;
  }
  // Observations since the model ran, NOT simulation time -- see the note on
  // observation_count in prediction_reuse_state.h. This observation's index
  // is observation_count; the interval counts from the run's index.
  if (s->has_anchor && th.refresh_interval > 0 &&
      (s->observation_count - s->last_nn_observation) >= th.refresh_interval) {
    flags |= ReuseDecision::kPeriodicRefresh;
  }

  s->decision_flags = flags;
  return flags;
}

/**
 * Advance the state after the chunk has been dealt with.
 *
 * The two updates are deliberately different, and conflating them is the bug
 * this separation exists to prevent: `previous` advances every timestep so that
 * D_step means "since last time", while `last_nn` advances ONLY when the
 * model actually ran, so that D_anchor means "since the cached prediction was
 * made". If last_nn were advanced on reuse, D_anchor would collapse onto
 * D_step and slow drift would never be detected.
 *
 * path_divergence accumulates here rather than in the evaluation so that it
 * counts each timestep once. It is diagnostic: an oscillation returns to the
 * anchor while the path keeps growing, which is precisely why summed step
 * divergence is not a substitute for the anchor.
 */
CTP_REUSE_HD inline void CommitChunkObservation(
    DevicePredictionReuseState *s, const ChunkSignature &now,
    long long timestep, bool model_ran, const int *sgd_counter = nullptr) {
  if (s->has_previous) {
    s->path_divergence += SignatureDivergence(s->previous, now);
  }
  s->previous = now;
  s->previous_timestep = timestep;
  s->has_previous = 1;
  if (model_ran) {
    if (sgd_counter != nullptr) s->cached_sgd_calls = *sgd_counter;
    s->last_nn = now;
    s->last_nn_timestep = timestep;
    s->last_nn_observation = s->observation_count;
    s->has_anchor = 1;
  }
  ++s->observation_count;
}

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_PREDICTION_REUSE_H_
