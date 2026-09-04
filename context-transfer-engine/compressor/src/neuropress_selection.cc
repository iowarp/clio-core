/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * @file neuropress_selection.cc
 * @brief Runtime::NeuroPressRankChunk -- everything that happens when
 * NeuroPress, rather than Clio's own heuristics, decides how a chunk is
 * compressed.
 *
 * This is the model's half of EstCompressionStats, which used to hold both
 * halves interleaved: one data_type expression, one features_ok expression and
 * one ranking gate each carrying a `neuropress_active ? ... : ...` branch. The
 * two halves do not share a model, a feature definition, or even a way of
 * reading the same bytes -- NeuroPress reads every chunk as float32 because
 * its statistics kernel is typed that way, while the legacy models use the
 * context's own type mapping -- so interleaving them meant every edit to one
 * had to be checked against the other.
 *
 * Separated, each half states its own preconditions. The caller decides which
 * one runs; an empty return here means "NeuroPress declined", and the legacy
 * heuristics take the chunk with the statistics this pass already computed.
 */

#include <clio_ctp/compress/compress_factory.h>
#include <clio_ctp/compress/preprocess/data_stats.h>
#include <clio_ctp/compress/preprocess/data_stats_gpu.h>
#include <clio_ctp/compress/preprocess/feature_extractor.h>

#include <chrono>
#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"
#include "clio_cte/compressor/models/neuropress_bridge.h"
#include "clio_cte/compressor/neuropress_path_trace.h"
#include "clio_cte/compressor/neuropress_telemetry.h"

namespace clio::cte::compressor {

std::vector<CompressionStats> Runtime::NeuroPressRankChunk(
    const void* chunk, clio::run::u64 chunk_size, const Context& context,
    double* entropy, double* mad, double* second_derivative_mean,
    double* out_entropy, double* out_mad, double* out_second_deriv,
    bool* out_neuropress_gpu_failed, const void** out_device_stats,
    const ctp::compress::preprocess::PredictionReuseContext* reuse,
    ctp::compress::preprocess::PredictionReuseOutcome* out_outcome) {
  /* Selection latency, parked for this chunk's row. Covers everything below:
     the statistics kernel, the reuse decision, and either a forward pass and
     ranking or the cached ranking. Reported per chunk so the reused and
     computed populations can be compared -- the difference between them is
     what reuse saves, and it is far too small to read off a run's wall time. */
  const auto sel_t0 = std::chrono::steady_clock::now();
  bool sel_reused = false;
  struct SelTimer {
    const std::chrono::steady_clock::time_point &t0;
    const bool &reused;
    /* Destructor, so an early return -- unusable statistics, a declined
       chunk -- still records what the attempt cost. */
    ~SelTimer() {
      RecordSelectionTiming(
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - t0).count(), reused);
    }
  } sel_timer{sel_t0, sel_reused};

  // float32 always: the stats kernel is typed `const float*` and model.nnwt
  // was normalized against float32. Reading as uint8 puts MAD hundreds of
  // sigma outside the training range.
  const ctp::DataType data_type = ctp::DataType::FLOAT32;
  size_t type_size = ctp::DataStatisticsFactory::GetTypeSize(data_type);

  // Whole chunk, not a prefix: the model was normalized against whole-chunk
  // statistics, and every candidate is scored from these three numbers.
  size_t num_elements = static_cast<size_t>(chunk_size / type_size);

  // A CUDA-IPC chunk is not host-readable, so features are computed on-device
  // rather than staging the buffer through host memory.

  // Statistics stay on the GPU. Deliberately unsynchronized: the ranking below
  // chains onto the SAME stream, so the decision is one async chain.
  const void *device_stats = nullptr;
  void *np_stream = nullptr;
  const bool np_device_path =
      num_elements > 0 && ctp::IsDevicePointer(chunk);
  if (np_device_path) {
    np_stream = ctp::DeviceStatsStream();
    // float64 is CONVERTED, not reinterpreted: reading a double as two float32
    // words makes the low half NaN about 1 time in 256.
    if (context.data_type_ == 2) {
      device_stats = ctp::ComputeDeviceStatsResidentF32From64(
          chunk, chunk_size / sizeof(double), np_stream);
    } else {
      device_stats = ctp::ComputeDeviceStatsResident(chunk, num_elements,
                                                     data_type, np_stream);
    }
    /* Reused below instead of re-measuring; upstream passes one d_stats_ptr
       to both inference and runNNSGDCtx. */
    if (out_device_stats) *out_device_stats = device_stats;
  }

  // A failed device path is reported and declined, never recomputed on the
  // host: upstream propagates the failure, and a silent host fallback would
  // look like a perfectly normal selection.

  // ComputeNeuroPressFeatures honours the element type. Reinterpreting float64
  // as float32 instead handed the model NaN for two of its three inputs.
  const bool features_ok =
      np_device_path ? device_stats != nullptr
                     : ctp::ComputeNeuroPressFeatures(
                           chunk, chunk_size, context.data_type_, entropy,
                           mad, second_derivative_mean);
  CLIO_PATH_TRACE(
      "1 stats    %s chunk=%llu B  %s",
      NpWhere(chunk), (unsigned long long)chunk_size,
      np_device_path
          ? "ComputeDeviceStatsResident -- CUDA kernel, stats stay on device"
          : "ComputeNeuroPressFeatures -- host code, CPU");
  if (np_device_path && device_stats == nullptr) {
    HLOG(kError,
         "EstCompressionStats: device-resident statistics failed for a "
         "NeuroPress chunk (size={}); NOT recomputing them on the host",
         chunk_size);
    if (out_neuropress_gpu_failed) *out_neuropress_gpu_failed = true;
  }
  if (features_ok && device_stats == nullptr) {
    // Report the statistics the ranking below is about to use, so a caller
    // can record what a selection was actually based on.
    if (out_entropy) *out_entropy = *entropy;
    if (out_mad) *out_mad = *mad;
    if (out_second_deriv) *out_second_deriv = *second_derivative_mean;
  }
  if (!features_ok) {
    // Ranking on the zeros left behind would look perfectly compressible.
    HLOG(kWarning,
         "EstCompressionStats: no usable statistics for this chunk "
         "(size={} elem_size={}); skipping NeuroPress ranking for it",
         chunk_size, type_size);
  }

  // Restricted to the trained 8-algorithm nvcomp space. All GPU-native, so a
  // device-resident chunk never forces a host round-trip in Compress().
  if (!features_ok) return {};

  {
    bool data_type_float = (context.data_type_ == 1);
    std::vector<CompressionStats> neuropress_stats;
    /* Set by the host-decided path when this chunk was served the cached
       ranking. Declared out here so the trace below can say so. */
    bool served_from_cache = false;
    if (device_stats != nullptr) {
      bool np_infer_failed = false;
      /* ---- host-decided prediction reuse --------------------------------
         The decision is a function of three doubles, and those three doubles
         are fetched to the host every chunk anyway (the ReadDeviceFeatureStats
         below, which used to be the only place they arrived). Asking BEFORE
         the forward pass rather than after it is what makes a hit cheap: it
         returns here, having launched nothing but the statistics kernel and
         waited on nothing but the 24-byte copy.

         DecidePredictionReuse and CommitChunkObservation below are the same
         `__host__ __device__` functions the decision and commit kernels call
         -- not a host reimplementation of them -- so the two paths cannot
         disagree about what reuse means. */
      NpHostReuseSlot *host_slot =
          (np_reuse_host_decide_ && reuse != nullptr &&
           reuse->slot != ctp::compress::preprocess::kNoLineageSlot)
              ? NpHostReuseSlotFor(reuse->slot)
              : nullptr;
      ctp::compress::preprocess::ChunkSignature sig{};
      /* Read once and used for both the decision and the commit: a Train()
         landing between them would otherwise cache the new epoch against a
         prediction made under the old weights. */
      const int sgd_epoch = np_sgd_epoch_.load(std::memory_order_relaxed);
      bool stats_read = false;
      if (host_slot != nullptr) {
        if (!ctp::ReadDeviceFeatureStats(device_stats, entropy, mad,
                                         second_derivative_mean, np_stream)) {
          *entropy = *mad = *second_derivative_mean = 0.0;
        }
        stats_read = true;
        sig.entropy = *entropy;
        sig.mad = *mad;
        sig.second_derivative = *second_derivative_mean;
        const uint32_t flags =
            ctp::compress::preprocess::DecidePredictionReuse(
                &host_slot->state, sig, reuse->timestep, reuse->thresholds,
                reuse->chunk_bytes, reuse->error_bound, &sgd_epoch);
        /* An empty cache is a miss whatever the divergence said. It is
           reachable: a lineage whose first inference FAILED has advanced its
           state without ever storing a ranking. */
        if (!ctp::compress::preprocess::MustRunModel(flags) &&
            !host_slot->ranked.empty()) {
          ctp::compress::preprocess::CommitChunkObservation(
              &host_slot->state, sig, reuse->timestep, /*model_ran=*/false,
              &sgd_epoch);
          if (out_outcome != nullptr) {
            out_outcome->step_divergence = host_slot->state.step_divergence;
            out_outcome->anchor_divergence =
                host_slot->state.anchor_divergence;
            out_outcome->path_divergence = host_slot->state.path_divergence;
            out_outcome->flags = flags;
          }
          if (out_entropy) *out_entropy = *entropy;
          if (out_mad) *out_mad = *mad;
          if (out_second_deriv) *out_second_deriv = *second_derivative_mean;
          /* Falls through to the shared tail rather than returning here: a
             reused chunk must reach the same "2 infer" and "3 rank" traces
             and the same return as a computed one, or the log stops
             describing every selection. */
          neuropress_stats = host_slot->ranked;
          served_from_cache = true;
          sel_reused = true;
        }
      }
      if (!served_from_cache) {
        neuropress_stats = NeuroPressCandidateStatsDevice(
            *neuropress_predictor_, chunk_size, device_stats, np_stream,
            data_type_float, context.error_bound_, context.target_psnr_,
            &np_infer_failed, config_.neuropress_best_mode_,
            /* The device keeps its own state only when the host is not
               deciding; running both would advance the anchor twice. */
            host_slot != nullptr ? nullptr : reuse,
            host_slot != nullptr ? nullptr : out_outcome);
      }
      if (np_infer_failed && out_neuropress_gpu_failed) {
        *out_neuropress_gpu_failed = true;
      }
      // 24-byte copy, no extra stall: the ranking already waited on the
      // stream. Ungated -- the per-chunk record needs these every chunk.
      // Skipped when the host path already fetched them above.
      if (!stats_read &&
          !ctp::ReadDeviceFeatureStats(device_stats, entropy, mad,
                                       second_derivative_mean, np_stream)) {
        *entropy = *mad = *second_derivative_mean = 0.0;
      }
      if (host_slot != nullptr && !served_from_cache) {
        /* Mirrors ReuseCommitKernel's model-ran half. has_prediction is set
           only on a ranking that exists: caching an empty one would make the
           next timestep's hit serve nothing. */
        if (!neuropress_stats.empty()) {
          host_slot->ranked = neuropress_stats;
          const size_t n = neuropress_stats.size();
          host_slot->state.cached_count = static_cast<int>(
              n < static_cast<size_t>(
                      ctp::compress::preprocess::kReuseMaxCandidates)
                  ? n
                  : static_cast<size_t>(
                        ctp::compress::preprocess::kReuseMaxCandidates));
          host_slot->state.has_prediction = 1;
          host_slot->state.cached_chunk_bytes = reuse->chunk_bytes;
          host_slot->state.cached_error_bound = reuse->error_bound;
        }
        ctp::compress::preprocess::CommitChunkObservation(
            &host_slot->state, sig, reuse->timestep, /*model_ran=*/true,
            &sgd_epoch);
        if (out_outcome != nullptr) {
          out_outcome->step_divergence = host_slot->state.step_divergence;
          out_outcome->anchor_divergence = host_slot->state.anchor_divergence;
          out_outcome->path_divergence = host_slot->state.path_divergence;
          out_outcome->flags = host_slot->state.decision_flags;
        }
      }
      if (out_entropy) *out_entropy = *entropy;
      if (out_mad) *out_mad = *mad;
      if (out_second_deriv) *out_second_deriv = *second_derivative_mean;
    } else {
      neuropress_stats = NeuroPressCandidateStats(
          *neuropress_predictor_, chunk_size, *entropy, *mad,
          *second_derivative_mean, data_type_float, context.error_bound_,
          config_.neuropress_best_mode_);
    }
    /* Whether the model ran, for the per-chunk record. The host path knows it
       from its own branch; the device path has to read the verdict back out of
       the outcome, because there the decision was made on the stream and the
       host was never told. Without this the column reads "ran" for all 500
       device rows while its trace reports 396 reuses -- the two disagreeing
       about the same run. */
    if (reuse != nullptr && out_outcome != nullptr && !served_from_cache &&
        !ctp::compress::preprocess::MustRunModel(out_outcome->flags)) {
      sel_reused = true;
    }
    /* The trace claims a forward pass per action. With prediction reuse that
       is no longer unconditional, and a log that says the model ran when it
       did not would make the NN-invocation counts unauditable. */
    if (served_from_cache) {
      /* Checked BEFORE out_outcome, and on a flag rather than on the flags:
         a caller that passed no outcome still reused, and the count of
         forward passes has to come out right in either case. */
      CLIO_PATH_TRACE(
          "2 infer    [HOST] REUSED the cached ranking -- no forward pass, "
          "no ranking kernel, no commit kernel, no prediction D2H "
          "(d_step=%.6f d_anchor=%.6f d_path=%.6f, decided before the "
          "stream was touched)",
          out_outcome != nullptr ? out_outcome->step_divergence : -1.0,
          out_outcome != nullptr ? out_outcome->anchor_divergence : -1.0,
          out_outcome != nullptr ? out_outcome->path_divergence : -1.0);
    } else if (reuse != nullptr && out_outcome != nullptr &&
        !ctp::compress::preprocess::MustRunModel(out_outcome->flags)) {
      CLIO_PATH_TRACE(
          "2 infer    [GPU] REUSED the cached prediction -- no forward pass "
          "(d_step=%.6f d_anchor=%.6f d_path=%.6f, decided on device)",
          out_outcome->step_divergence, out_outcome->anchor_divergence,
          out_outcome->path_divergence);
    } else if (reuse != nullptr && out_outcome != nullptr) {
      /* Ran with reuse armed. The divergences are logged on this branch too,
         so the full distribution -- not only the chunks that reused -- is
         recoverable from the log. */
      CLIO_PATH_TRACE(
          "2 infer    [GPU] %zu actions, ONE NN forward pass EACH "
          "(8->64->64->64->64->8) -- RAN (d_step=%.6f d_anchor=%.6f "
          "d_path=%.6f reason=0x%x)",
          neuropress_stats.size(), out_outcome->step_divergence,
          out_outcome->anchor_divergence, out_outcome->path_divergence,
          out_outcome->flags);
    } else
    CLIO_PATH_TRACE(
        "2 infer    %s %zu actions, ONE NN forward pass EACH "
        "(8->64->64->64->64->8) -- %s",
        device_stats != nullptr ? "[GPU]" : "[CPU]", neuropress_stats.size(),
        device_stats != nullptr
            ? "InferKernelDeviceStats<<<n_actions,64>>>, 1 block per action"
            : "host matrix path, looped per action");
    // HOST path only -- the device path masks inside RankKernel. Not
    // interchangeable: masking ranks last but stays selectable, removing here
    // can empty the list.
    if (device_stats == nullptr && context.target_psnr_ > 0) {
      std::vector<CompressionStats> filtered;
      filtered.reserve(neuropress_stats.size());
      for (const auto& stat : neuropress_stats) {
        // No `psnr_db_ > 0 &&` guard: upstream masks purely on psnr <
        // min_psnr, and 0.0 is what a failed inference leaves behind.
        if (stat.psnr_db_ < context.target_psnr_) continue;
        filtered.push_back(stat);
      }
      neuropress_stats = std::move(filtered);
    }
    if (!neuropress_stats.empty()) {
      CLIO_PATH_TRACE(
          "3 rank     %s %zu ranked by predicted COST; primary=%s  "
          "(entropy=%.4g mad=%.4g d2=%.4g)",
          device_stats != nullptr
              ? "[GPU] RankKernel -- cost model + 32-lane bitonic sort on device"
              : "[CPU] host sort",
          neuropress_stats.size(),
          ctp::CompressionFactory::NameForWireId(
              neuropress_stats.front().compress_lib_).c_str(),
          *entropy, *mad, *second_derivative_mean);
      HLOG(kDebug,
           "NeuroPress dynamic selection: chunk_size={} entropy={} mad={} "
           "-> top pick wire_id={} ({} candidates ranked)",
           chunk_size, *entropy, *mad, neuropress_stats.front().compress_lib_,
           neuropress_stats.size());
      return neuropress_stats;
    }
    // Empty means the caller's heuristics take over: a change of MODEL, which
    // is said out loud because downstream a legacy pick is indistinguishable
    // from a NeuroPress one.
    if (device_stats != nullptr) {
      HLOG(kError,
           "EstCompressionStats: NeuroPress produced no candidates for a "
           "device-resident chunk (size={}); NOT re-ranking it on the host",
           chunk_size);
    } else {
      // Warning, not error: nothing failed and the heuristics will produce a
      // valid selection. Only WHICH model chose it must stay visible.
      HLOG(kWarning,
           "EstCompressionStats: NeuroPress produced no candidates for a "
           "host-resident chunk (size={} entropy={} mad={}); falling back "
           "to Clio's legacy heuristics -- this selection is NOT "
           "NeuroPress's",
           chunk_size, *entropy, *mad);
    }
  }
  return {};
}

}  // namespace clio::cte::compressor
