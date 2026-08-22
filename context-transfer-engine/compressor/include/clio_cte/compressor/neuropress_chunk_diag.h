/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_chunk_diag.h
 * @brief Per-chunk record of what the model was shown and what it decided.
 *
 * Clio's counterpart of gpucompress_chunk_diag_t and its reset/count/get
 * accessors (gpucompress.h). The feat_* group IS the model's input vector,
 * which neither the selection nor the explore log records. A subset of
 * upstream's 65 fields, under upstream's names so the two can be diffed.
 */

#ifndef CLIO_CTE_COMPRESSOR_NEUROPRESS_CHUNK_DIAG_H_
#define CLIO_CTE_COMPRESSOR_NEUROPRESS_CHUNK_DIAG_H_

#include <cstddef>

namespace clio::cte::compressor {

/** Ranked action ids retained per chunk, matching upstream's 32. */
constexpr int kNeuroPressRankingSlots = 32;

/** One chunk's NeuroPress inputs and the decision they produced. Names and
 *  semantics follow upstream's; values are pre-normalization, as upstream. */
struct NeuroPressChunkDiag {
  /* ---- Action identity ---- */
  int nn_action = -1;           /**< used; differs when exploration swapped */
  int nn_original_action = -1;  /**< the model's pick, before any swap */
  int exploration_triggered = 0;
  int sgd_fired = 0;

  /* ---- The model's input vector ---- */
  /** From nn_original_action, not nn_action (gpucompress_diagnostics.cpp). */
  int feat_action = -1;
  float feat_entropy = 0;  /**< Shannon entropy, bits/byte */
  float feat_mad = 0;      /**< normalized mean absolute deviation */
  float feat_deriv = 0;    /**< normalized mean second derivative */
  /** RAW configured bound: not the 1e-7 inference sentinel, and not zero
   *  for a lossless action. */
  float feat_eb_enc = 0;
  float feat_ds_enc = 0;   /**< raw chunk size in bytes */

  /* ---- Cost model ---- */
  float cost_model_error_pct = 0;
  float actual_cost = 0;     /**< post-clamp */
  float predicted_cost = 0;  /**< post-clamp */

  /* ---- Per-statistic MAPE, one per predicted metric, clamped first as
   *      upstream does: ratio capped at 100x, times floored at 1 ms
   *      (gpucompress_compress.cpp). ---- */
  float ratio_mape = 0;
  float comp_time_mape = 0;
  /** Decompression is not measured at write time, so upstream substitutes the
   *  prediction for the actual and this reads 0 until a later read fills it. */
  float decomp_time_mape = 0;

  /** (primary_cost - best_explored_cost) / best_explored_cost. -1 when
   *  exploration did not fire: upstream's sentinel, not a zero. */
  float regret = -1.0f;

  /* ---- Predicted vs measured, per metric. Upstream's names. ---- */
  float predicted_ratio = 0;
  float actual_ratio = 0;
  float predicted_comp_time = 0;   /**< ms, NN output 0 */
  float compression_ms = 0;        /**< ms, measured codec kernel time */
  float predicted_decomp_time = 0; /**< ms, NN output 1 */
  /** Not measured at write time -- a later read is the only place it becomes
   *  known, which is also why decomp_time_mape reads 0 here. */
  float decompression_ms = 0;

  /** The model's ranked order, best first. */
  int predicted_ranking[kNeuroPressRankingSlots] = {};
  int predicted_ranking_count = 0;
};

/** Upstream's gpucompress_reset_chunk_history(). */
void NeuroPressResetChunkHistory();

/** Upstream's gpucompress_get_chunk_history_count(). */
int NeuroPressChunkHistoryCount();

/** Upstream's gpucompress_get_chunk_diag(): 0, or -1 on a bad index/null. */
int NeuroPressGetChunkDiag(int idx, NeuroPressChunkDiag *out);

/**
 * Append a record; returns its slot index, or -1 if not stored. The index
 * enables the post-hoc update below, as upstream's recordChunkDiagnostic
 * returns one (internal.hpp). Past the cap, chunks are dropped, not evicted.
 */
int NeuroPressRecordChunkDiag(const NeuroPressChunkDiag &diag);

/**
 * Fill in what only the exploration block knows. The record is written where
 * the inputs are assembled, before exploration runs, so the slot is claimed
 * early and completed here -- upstream's split for decompression time.
 */
void NeuroPressUpdateChunkDiagExploration(int idx, int final_action,
                                          bool triggered, float regret);

/** Mark that this chunk produced a weight update. No-op for a bad index. */
void NeuroPressUpdateChunkDiagSgd(int idx, bool sgd_fired);

/** @brief Cap on retained records between resets. */
constexpr size_t kNeuroPressChunkDiagCap = 65536;

}  // namespace clio::cte::compressor

#endif  // CLIO_CTE_COMPRESSOR_NEUROPRESS_CHUNK_DIAG_H_
