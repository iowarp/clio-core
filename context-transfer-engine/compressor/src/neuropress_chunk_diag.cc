/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_chunk_diag.cc
 * @brief Per-chunk diagnostic history; upstream's g_chunk_history. Process-wide,
 * as upstream's is, so tests read it without holding a Runtime.
 */

#include "clio_cte/compressor/neuropress_chunk_diag.h"

#include <cstddef>
#include <mutex>
#include <vector>

namespace clio::cte::compressor {

namespace {
/** Per-chunk NN input history; upstream's g_chunk_history
 *  (gpucompress_diagnostics.cpp). */
struct ChunkDiagHistory {
  std::mutex mtx;
  std::vector<NeuroPressChunkDiag> rows;
};

/** Leaked on purpose: worker threads can reach this during static
 *  destruction, and tearing the mutex down under them crashes for no gain. */
ChunkDiagHistory *ChunkDiagHistoryInstance() {
  static ChunkDiagHistory *h = new ChunkDiagHistory();
  return h;
}

}  // namespace

/* Upstream's reset/count/get accessors (gpucompress.h), -1 on a bad index. */

void NeuroPressResetChunkHistory() {
  ChunkDiagHistory *h = ChunkDiagHistoryInstance();
  std::lock_guard<std::mutex> lk(h->mtx);
  h->rows.clear();
}

int NeuroPressChunkHistoryCount() {
  ChunkDiagHistory *h = ChunkDiagHistoryInstance();
  std::lock_guard<std::mutex> lk(h->mtx);
  return static_cast<int>(h->rows.size());
}

int NeuroPressGetChunkDiag(int idx, NeuroPressChunkDiag *out) {
  if (out == nullptr || idx < 0) return -1;
  ChunkDiagHistory *h = ChunkDiagHistoryInstance();
  std::lock_guard<std::mutex> lk(h->mtx);
  if (static_cast<size_t>(idx) >= h->rows.size()) return -1;
  *out = h->rows[static_cast<size_t>(idx)];
  return 0;
}

int NeuroPressRecordChunkDiag(const NeuroPressChunkDiag &diag) {
  ChunkDiagHistory *h = ChunkDiagHistoryInstance();
  std::lock_guard<std::mutex> lk(h->mtx);
  // Drop, not evict: a reader always sees a contiguous prefix.
  if (h->rows.size() >= kNeuroPressChunkDiagCap) return -1;
  h->rows.push_back(diag);
  return static_cast<int>(h->rows.size()) - 1;
}

void NeuroPressUpdateChunkDiagSgd(int idx, bool sgd_fired) {
  if (idx < 0) return;
  ChunkDiagHistory *h = ChunkDiagHistoryInstance();
  std::lock_guard<std::mutex> lk(h->mtx);
  if (static_cast<size_t>(idx) >= h->rows.size()) return;
  h->rows[static_cast<size_t>(idx)].sgd_fired = sgd_fired ? 1 : 0;
}

void NeuroPressUpdateChunkDiagExploration(int idx, int final_action,
                                          bool triggered, float regret) {
  if (idx < 0) return;
  ChunkDiagHistory *h = ChunkDiagHistoryInstance();
  std::lock_guard<std::mutex> lk(h->mtx);
  if (static_cast<size_t>(idx) >= h->rows.size()) return;
  NeuroPressChunkDiag &row = h->rows[static_cast<size_t>(idx)];
  row.nn_action = final_action;
  row.exploration_triggered = triggered ? 1 : 0;
  row.regret = regret;
  // feat_action untouched: upstream ties it to nn_original_action.
}
}  // namespace clio::cte::compressor
