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
struct CostWeightOverride { double ct, dt, io, bw; bool any; };

/* Read once: this runs per chunk on runtime worker threads, where a getenv per
   call is both a syscall and a data race against anything setting the env. */
CostWeightOverride ResolveCostOverride() {
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
  }
}  // namespace

NeuroPressCostWeights NeuroPressResolvedCostWeights() {
  static const CostWeightOverride o = ResolveCostOverride();
  return NeuroPressCostWeights{o.ct, o.dt, o.io, o.bw};
}


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
    bool *out_inference_failed, bool ratio_only) {
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

  // ONE preset per algorithm, not three. NeuroPress has no preset concept --
  // its config space is algorithm x quantize x byte-shuffle, and preset is not
  // one of the 8 NN inputs. Enumerating {1,2,3} produces three candidates with
  // a bit-identical feature vector and therefore an identical score, leaving
  // which of FAST/BALANCED/BEST reaches the data to list order rather than to
  // the model -- and preset does change codec settings downstream. Pin BALANCED
  // so the ranked list has one entry per algorithm.
  std::vector<CandidateConfig> candidates =
      DefaultCandidates(/*include_gpu=*/true, {2}, false, 1e-3,
                        /*include_cpu=*/true);

  // Restrict to NeuroPress's trained action space: decodeAction encodes exactly
  // 8 GPU-lossless nvcomp algorithms -- LZ4/Snappy/Deflate/GDeflate/Zstd/ANS/
  // Cascaded/Bitcomp (base_ids 13-18, 23-24) -- and nothing else. CPU libraries
  // and the four extra GPU algorithms fall outside its 0-7 range and are
  // reachable only by explicit selection upstream. Without this filter,
  // FeaturesTo8Input's base_id % 8 fallback would alias one onto a trained
  // algorithm and return its prediction as if it were about the untrained one.
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
  // Upstream's action space and its buildable algorithms are the same set:
  // nvcomp is a hard build dependency there, so decodeAction can never name a
  // missing one. Clio's nvcomp support is optional, so without this the
  // selector could pick an algorithm that cannot be instantiated -- not a
  // degraded write but a lost one, since Compress() returns without a PutBlob.
  // ranking.h states the filtering is the caller's job.
  //
  // Probed ONCE per process: availability is fixed at build time, GetPreset()
  // constructs a real compressor object (far too costly per candidate per
  // write), and a function-local static initializes thread-safely, leaving the
  // set read-only -- this runs from concurrent runtime tasks.
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

  // Expand to the full action space, in upstream's order. decodeAction numbers
  // an action `algo + 8*quantize + 16*shuffle`, so the space enumerates as 0-7
  // plain, 8-15 quantized, 16-23 shuffled, 24-31 both. Order matters because
  // ties are common -- ratio saturates at the 100x cap on compressible data --
  // and upstream's ranking network uses strict comparators, so the LOWEST
  // action index survives a tie. Enumerating shuffle-major would let a lossless
  // shuffled config beat a quantized unshuffled one that upstream prefers.
  //
  // Quantized variants exist only with a positive error bound, since upstream
  // masks them otherwise and its compress path will not run the quantizer. The
  // bound rides on the candidate because it is NN input 3, not merely an
  // execution parameter: the model was trained with the real bound for
  // quantized configs and a 1e-7 sentinel for lossless ones.
  {
    // Order the algorithms by NeuroPress's own algo index before expanding.
    // KnownCompressors() lists the eight nvcomp entries by base_id, which maps
    // to algo indices 0,1,4,3,2,5,6,7 -- zstd, gdeflate and deflate out of
    // order. The action index is what decides a tie, so sorting here makes slot
    // order and action order the same thing, which is also what lets the kernel
    // decode a slot back into a configuration.
    //
    // Sorted ONCE per process: the list depends only on which algorithms this
    // build can construct, already probed above.
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
    // to -INFINITY when the bound is non-positive (nn_gpu.cu). RankKernel
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
  //
  // ORDER MATTERS, and it is upstream's order: the environment is resolved
  // FIRST, then best mode has the last word. Upstream reads its one env knob
  // (GPUCOMPRESS_BW_GBPS) inside gpucompress_init (gpucompress_api.cpp:229),
  // and gpucompress_set_best_mode runs later and overwrites g_rank_w0/w1 to
  // zero (gpucompress_learning.cpp:134-136), so nothing can land on top of
  // the ratio-only objective.
  static const CostWeightOverride kOverride = ResolveCostOverride();
  if (kOverride.any) {
    weights.w_cost_compress_time = kOverride.ct;
    weights.w_cost_decompress_time = kOverride.dt;
    weights.w_cost_io = kOverride.io;
    weights.bandwidth_bytes_per_ms = kOverride.bw;
  }

  // Best mode's ratio-only objective. Upstream reaches it by zeroing
  // g_rank_w0/w1 (gpucompress_learning.cpp), which leaves
  // cost = size/(ratio*bw) -- a monotone function of ratio alone, so the
  // ranking becomes "smallest output first". This has to apply to the NN
  // RANKING too, not just to the exploration winner: the ranking is what
  // decides which slots the exhaustive sweep visits.
  //
  // LAST, deliberately -- see the ordering note above. When this ran before
  // the override block, setting ANY one of the four env vars re-applied the
  // other three from their FALLBACKS, and two of those fallbacks (ct=1, dt=1)
  // are exactly what this zeroes. A bandwidth-only tweak therefore restored
  // the latency objective in the RANKING while DynamicSchedule's own cost
  // lambda kept `best_mode ? 0.0 : ...` -- the two halves of best mode then
  // ranked on different objectives, a state upstream cannot reach because it
  // keeps one copy of the weights that every reader shares.
  if (ratio_only) {
    weights.w_cost_compress_time = 0.0;
    weights.w_cost_decompress_time = 0.0;
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
      // GPUCOMPRESS_ERROR_NN_NOT_LOADED (gpucompress_compress.cpp),
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
        // substituting anything (gpucompress_compress.cpp).
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
    double second_derivative_mean, bool data_type_float, double error_bound,
    bool ratio_only) {
  return RankIntoStats(predictor, chunk_size, entropy, mad,
                       second_derivative_mean, data_type_float, error_bound,
                       /*device_stats=*/nullptr, /*stream=*/nullptr,
                       /*min_psnr=*/0.0, /*out_inference_failed=*/nullptr,
                       ratio_only);
}

std::vector<CompressionStats> NeuroPressCandidateStatsDevice(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, const void *device_stats, void *stream,
    bool data_type_float, double error_bound, double min_psnr,
    bool *out_inference_failed, bool ratio_only) {
  // Zeros for the three data statistics: on this path they reach neither the
  // network (which reads device_stats) nor the score (which reads only the
  // prediction and the chunk size). See the header for why they are not
  // parameters -- fetching them here would mean synchronizing before the
  // inference rather than after it.
  return RankIntoStats(predictor, chunk_size, 0.0, 0.0, 0.0, data_type_float,
                       error_bound, device_stats, stream, min_psnr,
                       out_inference_failed, ratio_only);
}

}  // namespace clio::cte::compressor
