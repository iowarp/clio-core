/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file prediction_reuse_gpu.cu
 * @brief The two kernels that bracket an inference, and the device store they
 * read and write.
 *
 * The arithmetic is NOT reimplemented here. SignatureDivergence,
 * DecidePredictionReuse and CommitChunkObservation are the same
 * `__host__ __device__` functions the host unit tests call, so the truth
 * table asserted on the CPU is the truth table that runs on the GPU. A second
 * device-side copy would be free to drift, and the drift would be invisible.
 */
#include <cuda_runtime.h>

#include <cstddef>

#include "clio_ctp/compress/preprocess/data_stats_gpu.h"
#include "clio_ctp/compress/preprocess/prediction_reuse_gpu.h"

namespace ctp::compress::preprocess {

namespace {

/** The chunk's signature is the varying half of the stats the NN reads. */
__device__ inline ChunkSignature SignatureOf(
    const DeviceFeatureStats *stats) {
  ChunkSignature s;
  s.entropy = stats->entropy;
  s.mad = stats->mad;
  s.second_derivative = stats->second_derivative;
  return s;
}

__global__ void ReuseDecisionKernel(DevicePredictionReuseState *states,
                                    uint32_t slot,
                                    const DeviceFeatureStats *stats,
                                    long long timestep,
                                    PredictionReuseThresholds th,
                                    double chunk_bytes, double error_bound,
                                    const int *sgd_counter) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  DecidePredictionReuse(&states[slot], SignatureOf(stats), timestep, th,
                        chunk_bytes, error_bound, sgd_counter);
}

/**
 * Cache the ranking and advance the state.
 *
 * One warp: the candidate set is at most kReuseMaxCandidates, which is
 * kMaxCandidates next door, which is upstream's NN_NUM_CONFIGS. Lane c copies
 * candidate c; lane 0 then does the scalar bookkeeping.
 */
__global__ void ReuseCommitKernel(DevicePredictionReuseState *states,
                                  uint32_t slot,
                                  const DeviceFeatureStats *stats,
                                  long long timestep, const double *scores,
                                  const int *order, const float *comp_time,
                                  const float *decomp_time, const float *ratio,
                                  const float *psnr, int n, double chunk_bytes,
                                  double error_bound, const int *sgd_counter) {
  DevicePredictionReuseState *s = &states[slot];
  // Read the verdict the decision kernel left in device memory; the host has
  // not been told and does not need to be.
  const bool model_ran = MustRunModel(s->decision_flags);

  const int c = threadIdx.x;
  if (model_ran && c < n && c < kReuseMaxCandidates) {
    s->score[c] = scores[c];
    s->order[c] = order[c];
    s->comp_time_ms[c] = comp_time[c];
    s->decomp_time_ms[c] = decomp_time[c];
    s->ratio[c] = ratio[c];
    s->psnr_db[c] = psnr[c];
  }
  __syncthreads();

  if (c != 0) return;
  if (model_ran) {
    s->cached_count = (n < kReuseMaxCandidates) ? n : kReuseMaxCandidates;
    s->has_prediction = 1;
    // Record WHAT the cache is valid for, so the next timestep can refuse it
    // if either changed.
    s->cached_chunk_bytes = chunk_bytes;
    s->cached_error_bound = error_bound;
  }
  // previous always, anchor only when the model ran -- in the one function
  // the host tests assert against.
  CommitChunkObservation(s, SignatureOf(stats), timestep, model_ran,
                         sgd_counter);
}

}  // namespace

void *ReuseStatesAlloc(uint32_t capacity) {
  if (capacity == 0) return nullptr;
  void *p = nullptr;
  const size_t bytes =
      static_cast<size_t>(capacity) * sizeof(DevicePredictionReuseState);
  if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
  // Zero IS the initial state: has_previous/has_anchor/has_prediction all
  // false, which the decision reads as a first observation.
  if (cudaMemset(p, 0, bytes) != cudaSuccess) {
    cudaFree(p);
    return nullptr;
  }
  return p;
}

void ReuseStatesFree(void *states) {
  if (states != nullptr) cudaFree(states);
}

bool ReuseStatesRead(const void *states, uint32_t slot,
                     DevicePredictionReuseState *out) {
  if (states == nullptr || out == nullptr || slot == kNoLineageSlot) {
    return false;
  }
  const auto *base = static_cast<const DevicePredictionReuseState *>(states);
  return cudaMemcpy(out, base + slot, sizeof(DevicePredictionReuseState),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool LaunchReuseDecision(const PredictionReuseContext &ctx,
                         const void *device_stats, void *stream,
                         const int *sgd_counter) {
  if (ctx.states == nullptr || ctx.slot == kNoLineageSlot ||
      device_stats == nullptr) {
    return false;
  }
  ReuseDecisionKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<DevicePredictionReuseState *>(ctx.states), ctx.slot,
      static_cast<const DeviceFeatureStats *>(device_stats), ctx.timestep,
      ctx.thresholds, ctx.chunk_bytes, ctx.error_bound, sgd_counter);
  return cudaGetLastError() == cudaSuccess;
}

bool LaunchReuseCommit(const PredictionReuseContext &ctx,
                       const void *device_stats, const double *d_scores,
                       const int *d_order, const float *d_comp_time,
                       const float *d_decomp_time, const float *d_ratio,
                       const float *d_psnr, int num_candidates, void *stream,
                       const int *sgd_counter) {
  if (ctx.states == nullptr || ctx.slot == kNoLineageSlot ||
      device_stats == nullptr || num_candidates <= 0) {
    return false;
  }
  ReuseCommitKernel<<<1, kReuseMaxCandidates, 0,
                         static_cast<cudaStream_t>(stream)>>>(
      static_cast<DevicePredictionReuseState *>(ctx.states), ctx.slot,
      static_cast<const DeviceFeatureStats *>(device_stats), ctx.timestep,
      d_scores, d_order, d_comp_time, d_decomp_time, d_ratio, d_psnr,
      num_candidates, ctx.chunk_bytes, ctx.error_bound, sgd_counter);
  return cudaGetLastError() == cudaSuccess;
}

bool EnqueueReuseOutcome(const PredictionReuseContext &ctx,
                         PredictionReuseOutcome *host_out, void *stream) {
  if (ctx.states == nullptr || ctx.slot == kNoLineageSlot ||
      host_out == nullptr) {
    return false;
  }
  // The three divergences and the flags are contiguous in the slot and
  // PredictionReuseOutcome mirrors that layout, so this is ONE transfer.
  // Asynchronous on the caller's stream: it adds a copy, never a stall, and
  // the inference's own synchronize is what makes it readable.
  static_assert(offsetof(DevicePredictionReuseState, anchor_divergence) ==
                    offsetof(DevicePredictionReuseState, step_divergence) +
                        sizeof(double),
                "divergence fields must stay adjacent for the packed copy");
  static_assert(offsetof(DevicePredictionReuseState, path_divergence) ==
                    offsetof(DevicePredictionReuseState, anchor_divergence) +
                        sizeof(double),
                "divergence fields must stay adjacent for the packed copy");
  static_assert(offsetof(DevicePredictionReuseState, decision_flags) ==
                    offsetof(DevicePredictionReuseState, path_divergence) +
                        sizeof(double),
                "decision_flags must follow the divergences for the packed copy");
  static_assert(offsetof(PredictionReuseOutcome, anchor_divergence) ==
                    offsetof(PredictionReuseOutcome, step_divergence) +
                        sizeof(double) &&
                offsetof(PredictionReuseOutcome, path_divergence) ==
                    offsetof(PredictionReuseOutcome, anchor_divergence) +
                        sizeof(double) &&
                offsetof(PredictionReuseOutcome, flags) ==
                    offsetof(PredictionReuseOutcome, path_divergence) +
                        sizeof(double),
                "PredictionReuseOutcome must mirror the slot's tail layout");

  const auto *base = static_cast<const DevicePredictionReuseState *>(ctx.states);
  const DevicePredictionReuseState *s = base + ctx.slot;
  constexpr size_t kBytes = sizeof(double) * 3 + sizeof(uint32_t);
  return cudaMemcpyAsync(&host_out->step_divergence, &s->step_divergence,
                         kBytes, cudaMemcpyDeviceToHost,
                         static_cast<cudaStream_t>(stream)) == cudaSuccess;
}

}  // namespace ctp::compress::preprocess
