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

#include "clio_cte/compressor/models/neuropress_bridge.h"

#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/model/ranking.h"

namespace clio::cte::compressor {

std::vector<CompressionStats> NeuroPressCandidateStats(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, double entropy, double mad,
    double second_derivative_mean, bool data_type_float, bool include_gpu) {
  using ctp::compress::model::CandidateConfig;
  using ctp::compress::model::DataFeatures;
  using ctp::compress::model::DefaultCandidates;
  using ctp::compress::model::RankedPrediction;

  DataFeatures data;
  data.chunk_size_bytes = static_cast<double>(chunk_size);
  data.shannon_entropy = entropy;
  data.mad = mad;
  data.second_derivative_mean = second_derivative_mean;
  data.data_type_char = data_type_float ? 0.0 : 1.0;
  data.data_type_float = data_type_float ? 1.0 : 0.0;

  std::vector<CandidateConfig> candidates = DefaultCandidates(include_gpu);
  std::vector<RankedPrediction> ranked = predictor.Rank(data, candidates);

  std::vector<CompressionStats> results;
  results.reserve(ranked.size());
  for (const auto &r : ranked) {
    // candidate.base_id is the ML-scheme base id (CandidateConfig / ranking.h),
    // not the CTE wire id CompressionStats::compress_lib_ expects -- go
    // through GetLibraryInfo (base_id -> name) then WireIdForName (name ->
    // wire_id) to cross from one frozen namespace to the other.
    auto library_info = ctp::CompressionFactory::GetLibraryInfo(
        static_cast<int>(r.candidate.LibraryConfigId()));
    int wire_id = ctp::CompressionFactory::WireIdForName(library_info.first);
    results.emplace_back(wire_id, r.candidate.preset_id,
                          r.prediction.compression_ratio,
                          r.prediction.compression_time_ms,
                          r.prediction.compression_time_ms,
                          r.prediction.psnr_db);
  }
  return results;
}

}  // namespace clio::cte::compressor
