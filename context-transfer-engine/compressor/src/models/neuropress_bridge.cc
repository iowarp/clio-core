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
#include <string>

#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/model/ranking.h"

namespace clio::cte::compressor {

std::vector<CompressionStats> NeuroPressCandidateStats(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, double entropy, double mad,
    double second_derivative_mean, bool data_type_float, double error_bound) {
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

  // ONE preset per algorithm, not three. NeuroPress has no preset concept at
  // all -- its config space is algorithm x quantize x byte-shuffle
  // (nn_weights.h's NN_NUM_CONFIGS = 32, internal.hpp's decodeAction) -- and
  // preset is not one of the 8 NN inputs (FeaturesTo8Input). Enumerating
  // {1,2,3} therefore produced three candidates with a bit-identical feature
  // vector, hence an identical prediction and an identical score; since
  // Rank() finishes with std::sort (not stable_sort), which of FAST/BALANCED/
  // BEST actually got applied to the data was decided by sort tie-breaking
  // rather than by the model -- and preset genuinely changes codec settings
  // downstream. Pin BALANCED so the choice is deterministic and the ranked
  // list has one entry per algorithm, matching the model's real resolution.
  std::vector<CandidateConfig> candidates =
      DefaultCandidates(/*include_gpu=*/true, {2}, false, 1e-3,
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

  // Drop anything this build cannot actually construct.
  //
  // NeuroPress's action space and its buildable algorithms are the same set
  // by construction -- nvcomp is a hard build dependency there, so all eight
  // always exist and decodeAction can never name one that is missing. Clio's
  // nvcomp support is optional (compress_factory.h's Make* helpers return
  // nullptr under #if !CTP_ENABLE_NVCOMP), so without this the selector
  // could rank, and pick, an algorithm that cannot be instantiated. That is
  // not a degraded write but a LOST one: Compress() logs "Failed to create
  // compressor", sets return_code_ 3 and returns WITHOUT a PutBlob, so on a
  // build without nvcomp every dynamic put would drop its blob. Upstream's
  // own primary path also refuses to store when the manager cannot be built
  // (gpucompress_compress.cpp:485-489 returns GPUCOMPRESS_ERROR_COMPRESSION),
  // so the fix is to preserve its invariant -- every action in the space is
  // constructible -- rather than to reproduce an error path it cannot reach.
  //
  // ranking.h states this is the caller's job: "callers that only want
  // available compressors should filter the returned list".
  //
  // Availability is fixed at build time, so it is resolved once rather than
  // per chunk -- GetPreset() constructs a real compressor object, which is
  // far too expensive to repeat for every candidate on every write.
  // Probed ONCE for the whole process: availability is decided at build time,
  // GetPreset() constructs a real compressor object (far too costly to repeat
  // per candidate per write), and a function-local static's initialization is
  // thread-safe, leaving the set read-only afterwards -- this runs from
  // concurrent runtime tasks, so a lazily-mutated cache would race.
  static const std::set<std::string> kAvailable = [] {
    std::set<std::string> avail;
    for (const auto &entry : ctp::compress::model::KnownCompressors()) {
      if (kNeuroPressTrainedGpuBaseIds.count(entry.base_id) == 0) continue;
      if (ctp::CompressionFactory::GetPreset(
              entry.name, ctp::CompressionPreset::BALANCED) != nullptr) {
        avail.insert(entry.name);
      }
    }
    return avail;
  }();
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [](const CandidateConfig &c) {
                       return kAvailable.count(c.library_name) == 0;
                     }),
      candidates.end());

  // Byte-shuffle variants. NeuroPress's action space is
  // algorithm x quantize x byte-shuffle (decodeAction, internal.hpp) and its
  // ranking masks every quantize action to -INFINITY when error_bound <= 0
  // (nn_gpu.cu), which is the lossless case Clio runs in. Byte-shuffle is
  // therefore the one remaining dimension upstream can reach and, without
  // this, Clio could not: the parity harness shows native repeatedly
  // selecting actions 20/21/23, all of which have the shuffle bit set.
  // Expand to the full action space IN UPSTREAM'S ORDER.
  //
  // decodeAction (internal.hpp:167-172) numbers an action
  //     algo + 8*quant + 16*shuffle
  // so the space enumerates as: 0-7 plain, 8-15 quantized, 16-23 shuffled,
  // 24-31 quantized+shuffled. Order matters because ties are common and
  // resolved by position -- ratio saturates at the 100x cap for many
  // candidates on compressible data, and upstream's bitonic ranking network
  // uses strict comparators, so the LOWEST action index survives a tie.
  // Building shuffle-major (as this did) made a lossless shuffled config
  // beat a quantized unshuffled one on equal cost, where upstream picks the
  // quantized one.
  //
  // Quantized variants exist only with a positive error bound: upstream's
  // ranking masks them to -INFINITY otherwise (nn_gpu.cu:238) and its
  // compress path will not run the quantizer (gpucompress_compress.cpp:434),
  // so a lossless run still ranks exactly the 16 configs it did before and
  // no existing selection can change.
  //
  // The bound rides on the candidate because it is NN input 3, not merely an
  // execution parameter -- the model was trained with the real bound for
  // quantized configs and a 1e-7 sentinel for lossless ones
  // (neural_net/core/configs.py:44).
  {
    const std::vector<CandidateConfig> plain = candidates;
    candidates.clear();
    candidates.reserve(plain.size() * (error_bound > 0.0 ? 4 : 2));
    for (int shuffle = 0; shuffle <= 1; ++shuffle) {
      for (int quant = 0; quant <= 1; ++quant) {
        if (quant == 1 && !(error_bound > 0.0)) continue;
        for (const auto &base : plain) {
          CandidateConfig c = base;
          c.byte_shuffle = (shuffle != 0);
          c.quantize = (quant != 0);
          c.error_bound = (quant != 0) ? error_bound : 0.0;
          candidates.push_back(c);
        }
      }
    }
  }

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
    //
    // Byte-shuffle rides in the free high bits of compress_preset_ (see
    // PackPreset in compressor_runtime.cc): CompressionStats has no field
    // for it, and this is the same encoding the on-disk header uses, so the
    // selection survives all the way to Compress() without a wire change.
    // Element size 4 matches NeuroPress's own shuffle_size (internal.hpp:
    // `shuffle_size > 0 ? 4 : 0`).
    // Same packed word compressor_runtime.cc's PackPreset/PackQuant build:
    // bits 0-7 preset, 8-15 shuffle element size, 16-23 format version,
    // 24 quantize-enabled. Only the ENABLED bit is set here -- the precision
    // is not known until the quantizer has seen the data's range, so
    // Compress() fills that in when it writes the header.
    uint32_t preset_bits = static_cast<uint32_t>(r.candidate.preset_id) & 0xFFu;
    if (r.candidate.byte_shuffle) preset_bits |= (4u << 8);
    if (r.candidate.quantize) preset_bits |= (1u << 24);
    const int preset_field = static_cast<int>(preset_bits);
    results.emplace_back(wire_id, preset_field,
                          r.prediction.compression_ratio,
                          r.prediction.compression_time_ms,
                          r.prediction.decompression_time_ms,
                          r.prediction.psnr_db);
  }
  return results;
}

}  // namespace clio::cte::compressor
