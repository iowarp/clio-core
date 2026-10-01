/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file xgb_selection.cc
 * @brief The XGBoost baseline, deployed as the chunk's selector.
 *
 * Figure 9 compares three predictors choosing a chunk's codec: NeuroPress's
 * NN, HCompress's cost model, and XGBoost. All three rank the SAME candidate
 * set (NeuroPress's 32 nvcomp actions) under the SAME cost model
 * (NeuroPressCost, NeuroPressCandidateStats) with the same tie order, so the
 * bars differ only in the predictor.
 *
 * XGBoost was trained (train_xgb.py) on the NN's own inputs: the chunk's
 * entropy, MAD and second derivative, plus the candidate's algorithm, shuffle,
 * quantize, bound and size. So those statistics are computed here with the
 * very code NeuroPress's selection uses -- the device kernel for a
 * GPU-resident chunk, the host routine otherwise -- and read back to rank on
 * the host, where the model runs.
 */

#include <clio_ctp/compress/preprocess/data_stats.h>
#include <clio_ctp/compress/preprocess/data_stats_gpu.h>
#include <clio_ctp/compress/preprocess/feature_extractor.h>
#include <clio_ctp/util/gpu_api.h>

#include <chrono>
#include <cmath>
#include <mutex>
#include <vector>

#include "clio_cte/compressor/compressor_runtime.h"
#include "clio_cte/compressor/models/neuropress_bridge.h"
#include "clio_cte/compressor/neuropress_telemetry.h"

namespace clio::cte::compressor {

namespace {

/** Milliseconds since `t0`. */
double XgMsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::vector<CompressionStats> Runtime::XgbRankChunk(const void *chunk,
                                                    clio::run::u64 chunk_size,
                                                    const Context &context) {
  const auto t0 = std::chrono::steady_clock::now();
  double entropy = 0.0, mad = 0.0, second_deriv = 0.0;
  bool features_ok = false;

  // ---- 1. The NN's statistics, computed the way NeuroPressRankChunk does.
  // float32 always for the device kernel (it is typed that way, and the
  // corpus both models trained on is float32); float64 is converted, not
  // reinterpreted.
  const size_t num_elements = static_cast<size_t>(chunk_size / sizeof(float));
  if (num_elements > 0 && ctp::IsDevicePointer(chunk)) {
    void *stream = ctp::DeviceStatsStream();
    const void *device_stats =
        (context.data_type_ == 2)
            ? ctp::ComputeDeviceStatsResidentF32From64(
                  chunk, chunk_size / sizeof(double), stream)
            : ctp::ComputeDeviceStatsResident(chunk, num_elements,
                                              ctp::DataType::FLOAT32, stream);
    features_ok = device_stats != nullptr &&
                  ctp::ReadDeviceFeatureStats(device_stats, &entropy, &mad,
                                              &second_deriv, stream);
  } else {
    features_ok = ctp::ComputeNeuroPressFeatures(
        chunk, chunk_size, context.data_type_, &entropy, &mad, &second_deriv);
  }
  const double stats_ms = XgMsSince(t0);
  if (!features_ok || !std::isfinite(entropy) || !std::isfinite(mad) ||
      !std::isfinite(second_deriv)) {
    // Ranking on zeros or NaN would look like a perfectly compressible chunk.
    HLOG(kWarning,
         "XGBoost: no usable statistics for a chunk of {} bytes; declining it "
         "(stored uncompressed, NOT handed to another selector)",
         chunk_size);
    return {};
  }

  // ---- 2. Rank NeuroPress's candidates with XGBoost's predictions.
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<CompressionStats> stats = NeuroPressCandidateStats(
      *xgb_predictor_, chunk_size, entropy, mad, second_deriv,
      context.data_type_ == 1, context.error_bound_, /*ratio_only=*/false);
  const double rank_ms = XgMsSince(t1);

  // The model does predict PSNR, but the NN's device path is the only one
  // that filters on a target; saying so beats a silent difference.
  if (context.target_psnr_ > 0) {
    static std::once_flag warned;
    std::call_once(warned, [] {
      HLOG(kWarning,
           "XGBoost: a PSNR target is set; the host ranking path does not "
           "filter on it, so XGBoost ranks without it");
    });
  }

  // Same per-chunk records as NeuroPress and HCompress: stats = the feature
  // pass (kernel + read-back), nn = prediction + ranking.
  RecordSelectionTiming((stats_ms + rank_ms) * 1000.0, /*reused=*/false);
  if (PhaseLogEnabled()) {
    RecordSelectionPhases(stats_ms, rank_ms, /*choice_ms=*/0.0,
                          /*reused=*/false);
  }
  if (!stats.empty()) {
    HLOG(kDebug,
         "XGBoost selection: chunk_size={} entropy={} mad={} d2={} -> "
         "wire_id={} ({} candidates; stats {} ms, rank {} ms)",
         chunk_size, entropy, mad, second_deriv, stats.front().compress_lib_,
         stats.size(), stats_ms, rank_ms);
  }
  return stats;
}

}  // namespace clio::cte::compressor
