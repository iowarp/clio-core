/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file prediction_reuse_gpu.h
 * @brief Device residence for the reuse state, and the two kernels that
 * bracket an inference.
 *
 * The order on the stream is
 *
 *     stats -> DECISION -> inference -> ranking -> COMMIT -> one D2H
 *
 * and every arrow is a stream dependency, not a host handoff. The decision
 * kernel writes its verdict into device memory; the inference and ranking kernels
 * read it there and either compute or replay the cached block; the commit
 * kernel stores a fresh result and advances the state. The host learns what
 * happened only from the transfer that was already going to happen at the
 * end -- by which point the decision has been made and acted on.
 *
 * Two kernels rather than one: the state the divergence is
 * measured against must not be advanced until after it has been measured AND
 * the result used. Folding the commit into the decision would compare the
 * current signature against itself from the second timestep onward.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_PREDICTION_REUSE_GPU_H_
#define CLIO_CTP_COMPRESS_PREPROCESS_PREDICTION_REUSE_GPU_H_

#include <cstdint>

#include "clio_ctp/compress/preprocess/prediction_reuse_state.h"
#include "clio_ctp/compress/preprocess/prediction_reuse.h"

namespace ctp::compress::preprocess {

/**
 * Everything one chunk needs to consult the reuse state, passed as a unit
 * so the three layers between the runtime and the kernel gain one parameter
 * rather than six.
 *
 * A null `states`, or a slot of kNoLineageSlot, disables the whole mechanism
 * for that chunk: the kernels then behave exactly as they do with reuse
 * disabled, which is what the exploration path relies on.
 */
struct PredictionReuseContext {
  /** Device array of DevicePredictionReuseState, from ReuseStatesAlloc. */
  void *states = nullptr;
  uint32_t slot = kNoLineageSlot;
  long long timestep = -1;
  PredictionReuseThresholds thresholds;
  /** NN inputs 4 and 3. Carried so the decision can refuse a cache that was
   *  made for a different chunk size or bound -- the signature cannot see
   *  either. Negative means "not supplied", which disables the check. */
  double chunk_bytes = -1.0;
  double error_bound = -1.0;
};

/**
 * What the decision kernel decided, for logging. Rides the existing D2H.
 *
 * FIELD ORDER IS LOAD-BEARING: it mirrors the tail of
 * DevicePredictionReuseState so the outcome is one contiguous copy out of the
 * slot rather than four. prediction_reuse_gpu.cu static_asserts the match.
 */
struct PredictionReuseOutcome {
  double step_divergence = 0.0;
  double anchor_divergence = 0.0;
  double path_divergence = 0.0;
  uint32_t flags = 0;
};

/** Zero-initialised device store for `capacity` lineages, or nullptr. */
void *ReuseStatesAlloc(uint32_t capacity);
void ReuseStatesFree(void *states);

/** One slot to the host. Diagnostics and tests only -- NOT the decision
 *  path, which never reads state back. */
bool ReuseStatesRead(const void *states, uint32_t slot,
                     DevicePredictionReuseState *out);

/**
 * Evaluate the decision for one chunk, on `stream`, from statistics that are
 * already on the device. Writes flags and both divergences into the slot;
 * advances nothing.
 */
bool LaunchReuseDecision(const PredictionReuseContext &ctx,
                         const void *device_stats, void *stream,
                         /* &w->sgd_call_count, device-resident; null skips. */
                         const int *sgd_counter = nullptr);

/**
 * Store a freshly computed ranking into the slot and advance the state.
 *
 * `previous` advances whatever happened; the anchor and the cached block are
 * written ONLY when the model actually ran -- the kernel reads that from
 * the flags the decision left behind, so the host does not have to know.
 */
bool LaunchReuseCommit(const PredictionReuseContext &ctx,
                       const void *device_stats, const double *d_scores,
                       const int *d_order, const float *d_comp_time,
                       const float *d_decomp_time, const float *d_ratio,
                       const float *d_psnr, int num_candidates, void *stream,
                       const int *sgd_counter = nullptr);

/**
 * Enqueue the outcome copy on `stream`. Does NOT synchronize.
 *
 * The caller's existing end-of-inference synchronize covers it, which is what
 * keeps this off the synchronization budget entirely -- it adds a transfer,
 * not a stall. `host_out` must therefore stay alive until that synchronize,
 * and its fields are meaningless before it.
 */
bool EnqueueReuseOutcome(const PredictionReuseContext &ctx,
                         PredictionReuseOutcome *host_out, void *stream);

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_PREDICTION_REUSE_GPU_H_
