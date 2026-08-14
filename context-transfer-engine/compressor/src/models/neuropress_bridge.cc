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
#include <cstdlib>
#include <set>
#include <string>

#include "clio_ctp/compress/compress_factory.h"
#include "clio_ctp/compress/model/neuropress_nn_predictor.h"
#include "clio_ctp/compress/model/ranking.h"

namespace clio::cte::compressor {

namespace {

/**
 * Shared body of both entry points.
 *
 * The device path differs from the host one in exactly one respect -- where
 * the three data statistics come from -- so it shares this function rather
 * than copying it. Everything the two must agree on (which algorithms are in
 * the trained action space, which are buildable, and above all the action
 * ORDER that decides ties) is therefore written once.
 *
 * @param device_stats Device pointer from ComputeDeviceStatsResident(), or
 *   nullptr for the host path. When set, the entropy/mad/second-derivative
 *   arguments are used ONLY to populate DataFeatures for the cost model's
 *   size term and are never fed to the network -- the kernel reads the real
 *   values out of device memory.
 */
std::vector<CompressionStats> RankIntoStats(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, double entropy, double mad,
    double second_derivative_mean, bool data_type_float, double error_bound,
    const void *device_stats, void *stream, double min_psnr,
    bool *out_inference_failed) {
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
  // vector, hence an identical prediction and an identical score, leaving
  // which of FAST/BALANCED/BEST got applied to the data to be settled by
  // where they happened to sit in the candidate list rather than by the
  // model -- and preset genuinely changes codec settings downstream. Pin
  // BALANCED so the choice is deterministic and the ranked list has one
  // entry per algorithm, matching the model's real resolution.
  //
  // (Rank() sorts with std::stable_sort, so such a tie resolves to the
  // first enumerated rather than arbitrarily -- but "first enumerated" is
  // still an accident of list construction here, not a decision the model
  // made, so pinning remains the right fix.)
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
    // Order the algorithms by NeuroPress's OWN algo index before expanding.
    //
    // KnownCompressors() lists the eight nvcomp entries by base_id
    // (13,14,15,16,17,18,23,24), which maps to algo indices 0,1,4,3,2,5,6,7 --
    // zstd, gdeflate and deflate out of order. The group order below was
    // already upstream's, but WITHIN a group the slots did not follow the
    // action index, and the action index is what decides a tie: upstream's
    // ranking network keeps the lowest action, while first-enumerated-wins
    // kept whatever this list happened to put first. On a tie between zstd
    // (algo 4) and deflate (algo 2) upstream picks deflate and Clio picked
    // zstd. Ties are not rare -- ratio saturates at the 100x cap on
    // compressible data, and one parity run counts 32 of them.
    //
    // Sorting here makes slot order and action order the same thing, which is
    // also what lets the kernel decode a slot back into a configuration.
    // Sorted ONCE for the process, not per chunk. The list depends only on
    // which algorithms this build can construct, which is fixed at build time
    // and already probed once above -- and upstream assembles no per-chunk
    // configuration at all, its lanes simply ARE the actions. Re-sorting eight
    // entries on every write would be cheap but it would also be host work
    // upstream does not do.
    static const std::vector<CandidateConfig> kPlainInActionOrder = [&] {
      std::vector<CandidateConfig> p = candidates;
      std::stable_sort(p.begin(), p.end(),
                       [](const CandidateConfig &a, const CandidateConfig &b) {
                         return ctp::compress::model::NeuroPressAlgoIdForBaseId(
                                    a.base_id) <
                                ctp::compress::model::NeuroPressAlgoIdForBaseId(
                                    b.base_id);
                       });
      return p;
    }();
    const std::vector<CandidateConfig> &plain = kPlainInActionOrder;
    candidates.clear();
    candidates.reserve(plain.size() * 4);
    // Enumerate the quantize half only when something will actually MASK it.
    //
    // Upstream evaluates all 32 configs every time and masks the quantize ones
    // to -INFINITY when the bound is non-positive (nn_gpu.cu:238). RankKernel
    // now does the same -- but only the device path runs it. On the host path
    // there is no masker, so enumerating them there leaves them ranked on
    // their real scores, and they win: a lossless run then selects a quantize
    // action that Compress() will not execute (`want_quant` requires
    // error_bound > 0), throwing away the model's choice. That regression is
    // exactly what showed up on the first real run after the masks moved.
    //
    // Eliding and masking pick the same winner whenever an unmasked candidate
    // exists, which on a lossless run is always -- so the host path loses
    // nothing by not enumerating them.
    const bool kernel_will_mask = (device_stats != nullptr);
    for (int shuffle = 0; shuffle <= 1; ++shuffle) {
      for (int quant = 0; quant <= 1; ++quant) {
        if (quant == 1 && !(error_bound > 0.0) && !kernel_will_mask) continue;
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

  // Cost-model weights are overridable for experiments, and ONLY for
  // experiments: the defaults above are upstream's (all 1.0, bw 5e6) and are
  // what every normal write uses. Read once -- this runs per chunk on runtime
  // worker threads, so a getenv per call would be both a syscall and a data
  // race against anything that sets the environment.
  //
  // The motivating case is a "ratio cost model": CLIO_NEUROPRESS_COST_W_CT=0
  // CLIO_NEUROPRESS_COST_W_DT=0 leaves cost = w_io*size/(ratio*bw), i.e. pure
  // compression ratio, which is what the library default would have been.
  // That is a genuinely different selector, not a tuning knob -- with equal
  // weights the I/O term is ~0.4% of the cost at 4 MiB, so the shipped model
  // is effectively a latency model and the ratio one picks quite differently.
  struct CostWeightOverride {
    double ct, dt, io, bw;
    bool any;
  };
  static const CostWeightOverride kOverride = [] {
    auto read = [](const char *name, double fallback, bool *seen) {
      const char *v = std::getenv(name);
      if (v == nullptr || *v == '\0') return fallback;
      char *end = nullptr;
      const double parsed = std::strtod(v, &end);
      if (end == v) return fallback;
      *seen = true;
      return parsed;
    };
    bool seen = false;
    CostWeightOverride o{};
    o.ct = read("CLIO_NEUROPRESS_COST_W_CT", 1.0, &seen);
    o.dt = read("CLIO_NEUROPRESS_COST_W_DT", 1.0, &seen);
    o.io = read("CLIO_NEUROPRESS_COST_W_IO", 1.0, &seen);
    o.bw = read("CLIO_NEUROPRESS_COST_BW", 5e6, &seen);
    o.any = seen;
    return o;
  }();
  if (kOverride.any) {
    weights.w_cost_compress_time = kOverride.ct;
    weights.w_cost_decompress_time = kOverride.dt;
    weights.w_cost_io = kOverride.io;
    weights.bandwidth_bytes_per_ms = kOverride.bw;
  }

  std::vector<RankedPrediction> ranked;
  auto *np = dynamic_cast<ctp::compress::model::NeuroPressNNPredictor *>(
      &predictor);
  if (device_stats != nullptr && np != nullptr) {
    // Device path: the network reads entropy/MAD/second-derivative straight
    // out of GPU memory, so nothing about this chunk's statistics has to come
    // to the host to make the decision -- upstream's arrangement
    // (gpucompress_infer_gpu passes runNNFusedInferenceCtx an AutoStatsGPU*).
    // Scoring and sorting are then the SAME code the host path uses, so the
    // tie-breaking behaviour pinned by ctp_neuropress_tiebreak_parity is
    // unchanged.
    std::vector<ctp::compress::model::CompressionFeatures> feats;
    feats.reserve(candidates.size());
    for (const auto &c : candidates) {
      feats.push_back(
          ctp::compress::model::MakeCompressionFeatures(data, c));
    }
    std::vector<int> order;
    std::vector<double> scores;
    auto preds = np->PredictBatchDeviceStats(device_stats, feats, stream,
                                             &weights, &order, min_psnr,
                                             &scores);
    if (preds.empty() && !candidates.empty()) {
      // Inference FAILED, as distinct from there being nothing to rank. Only
      // the former is upstream's error case; an empty candidate set means this
      // build cannot construct any of the eight algorithms, which is a
      // configuration state and must not fail a write.
      if (out_inference_failed) *out_inference_failed = true;
    }
    if (preds.empty()) {
      // FAIL, do not substitute. Re-ranking on the host here would run
      // NeuroPress's own inference somewhere upstream never runs it, and
      // silently: the caller would receive a complete, plausible ranking with
      // no indication the GPU path had dropped out. Upstream refuses in the
      // same situation rather than degrading -- a failed inference gives
      // "ALGO_AUTO requested but NN inference failed" and
      // GPUCOMPRESS_ERROR_NN_NOT_LOADED (gpucompress_compress.cpp:208-212),
      // and there is no CPU implementation of the network in that project to
      // fall back TO. An empty result here makes the caller's own
      // no-selection path run, which is visible and logged.
      return {};
    }
    // Gather in the order the kernel produced, carrying the scores it also
    // produced. Nothing here re-derives the ranking: no comparison, no
    // Score() call, no sort. That is the point of the kernel having done it.
    if (order.size() != candidates.size() ||
        scores.size() != candidates.size()) {
      return {};
    }
    ranked.reserve(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
      const int slot = order[i];
      if (slot < 0 || static_cast<size_t>(slot) >= candidates.size()) {
        // A malformed permutation. Re-ranking on the host would be a silent
        // fallback for a stage that just ran on the GPU, so refuse instead --
        // upstream ends the call on an inference failure rather than
        // substituting anything (gpucompress_compress.cpp:208-212).
        return {};
      }
      ranked.push_back({candidates[slot], preds[slot], scores[i]});
    }
  } else {
    ranked = predictor.Rank(data, candidates, weights);
  }

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

}  // namespace

std::vector<CompressionStats> NeuroPressCandidateStats(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, double entropy, double mad,
    double second_derivative_mean, bool data_type_float, double error_bound) {
  return RankIntoStats(predictor, chunk_size, entropy, mad,
                       second_derivative_mean, data_type_float, error_bound,
                       /*device_stats=*/nullptr, /*stream=*/nullptr,
                       /*min_psnr=*/0.0, /*out_inference_failed=*/nullptr);
}

std::vector<CompressionStats> NeuroPressCandidateStatsDevice(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, const void *device_stats, void *stream,
    bool data_type_float, double error_bound, double min_psnr,
    bool *out_inference_failed) {
  // Zeros for the three data statistics: on this path they reach neither the
  // network (which reads device_stats) nor the score (which reads only the
  // prediction and the chunk size). See the header for why they are not
  // parameters -- fetching them here would mean synchronizing before the
  // inference rather than after it.
  return RankIntoStats(predictor, chunk_size, 0.0, 0.0, 0.0, data_type_float,
                       error_bound, device_stats, stream, min_psnr,
                       out_inference_failed);
}

}  // namespace clio::cte::compressor
