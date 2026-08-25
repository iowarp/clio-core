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

#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"
#include "clio_cte/compressor/models/neuropress_bridge.h"

namespace clio::cte::compressor {

std::vector<CompressionStats> Runtime::NeuroPressRankChunk(
    const void* chunk, clio::run::u64 chunk_size, const Context& context,
    double* entropy, double* mad, double* second_derivative_mean,
    double* out_entropy, double* out_mad, double* out_second_deriv,
    bool* out_neuropress_gpu_failed, const void** out_device_stats) {
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
    if (device_stats != nullptr) {
      bool np_infer_failed = false;
      neuropress_stats = NeuroPressCandidateStatsDevice(
          *neuropress_predictor_, chunk_size, device_stats, np_stream,
          data_type_float, context.error_bound_, context.target_psnr_,
          &np_infer_failed, config_.neuropress_best_mode_);
      if (np_infer_failed && out_neuropress_gpu_failed) {
        *out_neuropress_gpu_failed = true;
      }
      // 24-byte copy, no extra stall: the ranking already waited on the
      // stream. Ungated -- the per-chunk record needs these every chunk.
      if (!ctp::ReadDeviceFeatureStats(device_stats, entropy, mad,
                                       second_derivative_mean, np_stream)) {
        *entropy = *mad = *second_derivative_mean = 0.0;
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
