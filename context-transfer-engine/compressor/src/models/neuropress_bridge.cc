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

#include <algorithm>
#include <set>

#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/model/ranking.h"

namespace clio::cte::compressor {

std::vector<CompressionStats> NeuroPressCandidateStats(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, double entropy, double mad,
    double second_derivative_mean, bool data_type_float) {
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

  // Request the broadest possible set; the filter below is the sole gate
  // on what NeuroPress is actually allowed to rank.
  std::vector<CandidateConfig> candidates =
      DefaultCandidates(/*include_gpu=*/true, {1, 2, 3}, false, 1e-3,
                        /*include_cpu=*/true);

  // Restrict to NeuroPress's actual trained action space: the network's
  // action-decoding scheme (nn_gpu.cu's decodeAction, "algorithm index 0-7")
  // hard-codes exactly 8 GPU-lossless nvcomp algorithms -- LZ4/Snappy/
  // Deflate/GDeflate/Zstd/ANS/Cascaded/Bitcomp (base_ids 13-18, 23-24 in
  // ranking.h's KnownCompressors) -- and NOTHING else: no CPU library, no
  // zfp-sycl/cuSZ/nDzip/cuSZp. None of those were ever part of the trained
  // action space in the original project either -- CPU libraries and the
  // four extra GPU algorithms all fall outside decodeAction's 0-7 range and
  // are reachable only via explicit/static selection there, never as
  // something the trained network can output as its own choice. Without
  // this filter, NeuroPressNNPredictor::FeaturesTo8Input's own fallback
  // (base_id % 8) aliased any of them onto whichever real trained algorithm
  // happens to share that remainder and returned ITS prediction as if it
  // were a genuine, learned opinion about the untrained one -- not
  // something dynamic selection was ever supposed to reach.
  static const std::set<int> kNeuroPressTrainedGpuBaseIds = {
      13, 14, 15, 16, 17, 18, 23, 24};
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [](const CandidateConfig &c) {
                       return kNeuroPressTrainedGpuBaseIds.count(
                                 c.base_id) == 0;
                     }),
      candidates.end());

  // Rank by NeuroPress's own cost model rather than the library default
  // (ratio only): cost = w0*ct + w1*dt + w2*size/(ratio*bw), minimized.
  // Selecting on ratio alone picks a codec that squeezes marginally harder
  // even when it takes far longer to run -- the opposite of what this tier
  // wants, and not what NeuroPress does.
  ctp::compress::model::RankingWeights weights;
  weights.use_cost_model = true;
  std::vector<RankedPrediction> ranked =
      predictor.Rank(data, candidates, weights);

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
    // decompress_time_ms_ takes the NN's OWN decompression-time output, not
    // a copy of the compression time -- they are separate predictions and
    // the cost model ranks on both.
    results.emplace_back(wire_id, r.candidate.preset_id,
                          r.prediction.compression_ratio,
                          r.prediction.compression_time_ms,
                          r.prediction.decompression_time_ms,
                          r.prediction.psnr_db);
  }
  return results;
}

}  // namespace clio::cte::compressor
