/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file prediction_reuse_state.h
 * @brief Per-lineage prediction-reuse state, and the host index that
 * addresses it.
 *
 * WHAT IS RETAINED, AND WHY IT IS THIS AND NOT THE BLOCK.
 *
 * The NeuroPress selection is a deterministic function of eight inputs
 * (InferKernelDeviceStats, neuropress_nn_gpu_kernels.cu). Five of them -- the three action
 * bits, the error bound and the chunk size -- are constant for a given
 * lineage in a given run. Only entropy, MAD and the second derivative vary
 * from one timestep to the next. So the three numbers below are not a
 * SUMMARY of the chunk that the model might disagree with: they are the
 * model's entire view of it. Two timesteps with the same signature cannot
 * produce different predictions, because the network is never shown anything
 * that distinguishes them.
 *
 * That is what rules out retaining the block itself. block_evolution_gpu.h
 * does retain it, and correctly so -- it measures PHYSICAL evolution, which
 * is a different question and a legitimate one. For deciding whether to
 * re-run the model it would cost a full chunk copy per lineage plus an extra
 * memory-bound pass, to resolve detail the decision-maker cannot see. The
 * signature costs 24 bytes and no pass at all: the statistics are already
 * computed on the device, on the critical path, for every chunk.
 *
 * WHAT THE SIGNATURE CANNOT SEE. Online SGD changes the weights without
 * changing any input, so an unchanged signature can still yield a changed
 * prediction. That is detected from the model's own SGD counter
 * (cached_sgd_calls below), not inferred from the data.
 */
#ifndef CLIO_CTP_COMPRESS_PREPROCESS_PREDICTION_REUSE_STATE_H_
#define CLIO_CTP_COMPRESS_PREPROCESS_PREDICTION_REUSE_STATE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace ctp::compress::preprocess {

/** Widest candidate set the ranking warp can hold; mirrors kMaxCandidates in
 *  neuropress_nn_gpu_kernels.cu, which is upstream's NN_NUM_CONFIGS. */
constexpr int kReuseMaxCandidates = 32;

/** No slot: the lineage is unresolved, or the registry is full. The caller
 *  must run the model -- never reuse -- when it sees this. */
constexpr uint32_t kNoLineageSlot = 0xFFFFFFFFu;

/**
 * The model's whole view of one chunk at one timestep.
 *
 * Field-for-field the varying half of DeviceFeatureStats (data_stats_gpu.h);
 * kept as its own type so that "the signature" is a thing the trigger talks
 * about rather than a convention about which fields of a stats struct matter.
 */
struct ChunkSignature {
  double entropy = 0.0;
  double mad = 0.0;
  double second_derivative = 0.0;
};

/**
 * Everything retained for one lineage, in DEVICE memory.
 *
 * previous_* and last_nn_* are deliberately separate and are updated by
 * different rules: previous_* advances every timestep, last_nn_* only when
 * the model actually ran. Collapsing them would make the anchor divergence
 * identical to the step divergence and lose the slow-drift detection that is
 * the anchor's entire reason for existing.
 */
struct DevicePredictionReuseState {
  /** Signature at the previous timestep. Valid iff has_previous. */
  ChunkSignature previous;
  /** Signature at the timestep the model last actually ran on. Valid iff
   *  has_anchor. */
  ChunkSignature last_nn;

  /** The cached ranking, as the device produced it. Reuse replays exactly
   *  these bytes, so nothing downstream can distinguish a reused decision
   *  from a recomputed one except that the forward pass did not run. */
  double score[kReuseMaxCandidates];
  int order[kReuseMaxCandidates];
  float comp_time_ms[kReuseMaxCandidates];
  float decomp_time_ms[kReuseMaxCandidates];
  float ratio[kReuseMaxCandidates];
  float psnr_db[kReuseMaxCandidates];
  /** Candidates the cached block actually covers; 0 means no cache yet. */
  int cached_count;

  /**
   * The other two NN inputs the cached prediction was made for.
   *
   * The signature covers inputs 5-7. Inputs 3 and 4 -- the error bound and
   * the chunk size -- are constant for a lineage in almost every run, which
   * is exactly why they are dangerous: nothing would notice the exception.
   * Entropy, MAD and the second derivative are all INTENSIVE, so halving a
   * chunk leaves the signature untouched while input 4 halves, and the cache
   * would replay a prediction the model never made for this chunk.
   */
  double cached_chunk_bytes;
  double cached_error_bound;
  /**
   * The model's SGD call count when the cache was made (w->sgd_call_count,
   * device-resident beside the weights). If it has moved, the weights have,
   * and the cached prediction was produced by a model that no longer exists
   * -- whatever the data did. This is the exact signal the periodic refresh
   * only approximates. Inference-only runs never change it.
   */
  int cached_sgd_calls;

  long long previous_timestep;
  long long last_nn_timestep;

  /**
   * Observation counters for the periodic refresh.
   *
   * Counted in OBSERVATIONS of this lineage, not simulation time: dump
   * cadence is a property of the workload (WarpX writes every 10 steps, Nyx
   * every step), so an interval in timestep units would mean "every frame"
   * on one and "every tenth frame" on the other. observation_count is the
   * index of the observation being decided; last_nn_observation is the index
   * at which the model last ran.
   */
  long long observation_count;
  long long last_nn_observation;

  /** Divergence at the current timestep, in [0,1] (SignatureDivergence).
   *  Written by the decision, read by diagnostics. */
  double step_divergence;
  double anchor_divergence;
  /** Sum of every step divergence since the lineage was first seen.
   *  UNBOUNDED -- each step is in [0,1], their sum is not. DIAGNOSTIC ONLY:
   *  see prediction_reuse.h on why summed step divergence is not the anchor
   *  divergence (an oscillation returns to the anchor while this grows). */
  double path_divergence;

  /** Why the model ran, or that it did not; ReuseDecision bits. */
  uint32_t decision_flags;
  uint8_t has_previous;
  uint8_t has_anchor;
  uint8_t has_prediction;
  uint8_t reserved;
};

/**
 * Lineage key -> dense slot in the device state array.
 *
 * Host-side and deliberately so. This is a string lookup on the thread that
 * is already running the chunk; it involves no device memory, no transfer and
 * no synchronization -- it is not a host round trip. What crosses to the
 * device is one integer, passed as a kernel argument.
 *
 * FULL MEANS REFUSED, NOT EVICTED. An evicted lineage would hand its
 * successor a cached prediction belonging to a different block. Refusing
 * returns kNoLineageSlot, the caller runs the model, and the lineages
 * already tracked keep working.
 *
 * Thread-safety: none, matching BlockEvolutionTracker. See the runtime for
 * how the per-chunk path serialises access.
 */
class LineageSlotRegistry {
 public:
  explicit LineageSlotRegistry(uint32_t capacity) : capacity_(capacity) {
    slots_.reserve(capacity);
  }

  /** @return dense slot in [0, capacity), or kNoLineageSlot when the key is
   *  empty (unresolved lineage) or the registry is full. */
  uint32_t SlotFor(const std::string &key) {
    if (key.empty()) return kNoLineageSlot;
    auto it = slots_.find(key);
    if (it != slots_.end()) return it->second;
    if (slots_.size() >= capacity_) return kNoLineageSlot;
    const uint32_t slot = static_cast<uint32_t>(slots_.size());
    slots_.emplace(key, slot);
    return slot;
  }

  size_t size() const { return slots_.size(); }
  uint32_t capacity() const { return capacity_; }

 private:
  uint32_t capacity_;
  std::unordered_map<std::string, uint32_t> slots_;
};

}  // namespace ctp::compress::preprocess

#endif  // CLIO_CTP_COMPRESS_PREPROCESS_PREDICTION_REUSE_STATE_H_
