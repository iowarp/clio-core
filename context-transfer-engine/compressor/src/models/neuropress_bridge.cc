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
struct CostWeightOverride { double ct, dt, io, bw, cap; bool any; };

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
    /* Ratio ceiling, upstream's RATIO_CAP of 100. An experiment knob that must
       reach SEVEN places at once; miss one and it silently re-imposes 100.
       Raising it makes MAPE rise -- the cap was hiding real prediction error. */
    /* Deliberately NOT folded into `seen`: the cap is not one of the four
       cost WEIGHTS, and letting it set that flag would make a cap-only
       experiment re-apply all four weights from their fallbacks. */
    bool cap_seen = false;
    o.cap = read("CLIO_NEUROPRESS_RATIO_CAP", 100.0, &cap_seen);
    if (!(o.cap > 0.0)) o.cap = 100.0;
    o.any = seen;
    return o;
  }
}  // namespace

NeuroPressCostWeights NeuroPressResolvedCostWeights() {
  static const CostWeightOverride o = ResolveCostOverride();
  return NeuroPressCostWeights{o.ct, o.dt, o.io, o.bw, o.cap};
}


namespace {

/** Shared body of both entry points; they differ only in where the statistics
 *  come from, so action space, buildability and tie ORDER are written once.
 *  With device_stats set, entropy/mad/deriv feed the cost model, not the NN. */
std::vector<CompressionStats> RankIntoStats(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, double entropy, double mad,
    double second_derivative_mean, bool data_type_float, double error_bound,
    const void *device_stats, void *stream, double min_psnr,
    bool *out_inference_failed, bool ratio_only,
    const ctp::compress::preprocess::PredictionReuseContext *reuse,
    ctp::compress::preprocess::PredictionReuseOutcome *out_outcome) {
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

  // ONE preset per algorithm: preset is not an NN input, so enumerating three
  // gives identical feature vectors and leaves the choice to list order.
  // Pin BALANCED.
  std::vector<CandidateConfig> candidates =
      DefaultCandidates(/*include_gpu=*/true, {2}, false, 1e-3,
                        /*include_cpu=*/true);

  // Trained action space only: decodeAction encodes 8 GPU-lossless nvcomp
  // algorithms (base_ids 13-18, 23-24). Without this filter the base_id % 8
  // fallback aliases an untrained algorithm onto a trained one's prediction.
  static const std::set<int> kNeuroPressTrainedGpuBaseIds = {
      13, 14, 15, 16, 17, 18, 23, 24};
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [](const CandidateConfig &c) {
                       return kNeuroPressTrainedGpuBaseIds.count(
                                 c.base_id) == 0;
                     }),
      candidates.end());

  // Drop what this build cannot construct: Clio's nvcomp is optional, and
  // picking an uninstantiable algorithm loses the write, not just degrades it.
  // Probed once per process -- GetPreset() builds a real compressor object.
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

  // Full action space in upstream's order: algo + 8*quantize + 16*shuffle.
  // Order decides ties (strict comparators, lowest index wins) and ties are
  // common because ratio saturates at the cap.

  // Quantized variants need a positive bound: it is NN input 3, trained with
  // the real bound for quantized configs and a 1e-7 sentinel for lossless.
  {
    // Sort by algo index: KnownCompressors() lists by base_id, which maps to
    // 0,1,4,3,2,5,6,7. Slot order must equal action order, both because ties
    // resolve on it and because the kernel decodes a slot back to a config.
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
    // Enumerate the quantize half only where something MASKS it. Only the
    // device path runs RankKernel; on the host these would rank on real scores
    // and win, selecting an action Compress() will not execute.
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

  // NeuroPress's cost model, not the library default: ratio alone picks a
  // codec that squeezes marginally harder however long it takes.
  ctp::compress::model::RankingWeights weights;
  weights.use_cost_model = true;
  // Experiment-only overrides; defaults are upstream's (all 1.0, bw 5e6).
  // Read once: this runs per chunk on worker threads, so a getenv per call is
  // a syscall and a race.

  // W_CT=0 W_DT=0 gives a pure ratio selector -- a different selector, not a
  // tuning knob: at 4 MiB the I/O term is ~0.4% of the cost, so the shipped
  // model is effectively a latency model.

  // ORDER MATTERS, and it is upstream's: environment first, best mode last.
  static const CostWeightOverride kOverride = ResolveCostOverride();
  if (kOverride.any) {
    weights.w_cost_compress_time = kOverride.ct;
    weights.w_cost_decompress_time = kOverride.dt;
    weights.w_cost_io = kOverride.io;
    weights.bandwidth_bytes_per_ms = kOverride.bw;
  }
  /* Independent of the weight override (see ResolveCostOverride): the cap
     reaches the RANKING and the inference kernel's own clamp, so the model's
     predictions and the cost model score on one scale. */
  weights.ratio_cap = kOverride.cap;

  // Best mode's ratio-only objective: zeroing ct/dt leaves a monotone function
  // of ratio. Applies to the RANKING too -- that decides which slots the sweep
  // visits.

  // LAST, deliberately: run before the override block, setting any one env var
  // re-applies the other three from fallbacks, two of which are what this
  // zeroes, splitting best mode across two objectives.
  if (ratio_only) {
    weights.w_cost_compress_time = 0.0;
    weights.w_cost_decompress_time = 0.0;
  }

  std::vector<RankedPrediction> ranked;
  auto *np = dynamic_cast<ctp::compress::model::NeuroPressNNPredictor *>(
      &predictor);
  if (device_stats != nullptr && np != nullptr) {
    // The network reads the statistics straight out of GPU memory. Scoring and
    // sorting are the SAME code as the host path, so the tie-breaking pinned by
    // ctp_neuropress_tiebreak_parity is unchanged.
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
                                             &scores, /*out_full=*/nullptr,
                                             reuse, out_outcome);
    if (preds.empty() && !candidates.empty()) {
      // Inference FAILED, as distinct from nothing to rank: an empty candidate
      // set is a build configuration, and must not fail a write.
      if (out_inference_failed) *out_inference_failed = true;
    }
    if (preds.empty()) {
      // FAIL, do not substitute: a host re-rank would run the inference where
      // upstream never does, and silently -- the caller would get a plausible
      // ranking with no sign the GPU path dropped out.
      return {};
    }
    // Gather in the order the kernel produced, carrying the scores it also
    // produced. Nothing here re-derives the ranking: no comparison, no
    // Score() call, no sort. That is the point of the kernel having done it.
    if (order.size() != candidates.size() ||
        scores.size() != candidates.size()) {
      // Say it FAILED. Without this the caller sees an empty list, skips its
      // rc=4, and quietly re-ranks on the legacy CPU candidates -- the exact
      // host fallback the kError below claims is not happening.
      if (out_inference_failed) *out_inference_failed = true;
      return {};
    }
    ranked.reserve(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
      const int slot = order[i];
      if (slot < 0 || static_cast<size_t>(slot) >= candidates.size()) {
        // Malformed permutation. Refuse rather than silently re-rank on the
        // host for a stage that just ran on the GPU.
        if (out_inference_failed) *out_inference_failed = true;
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
    // base_id is the ML-scheme id, not the CTE wire id compress_lib_ expects:
    // GetLibraryInfo (id -> name) then WireIdForName (name -> wire id).
    auto library_info = ctp::CompressionFactory::GetLibraryInfo(
        static_cast<int>(r.candidate.LibraryConfigId()));
    int wire_id = ctp::CompressionFactory::WireIdForName(library_info.first);
    // decompress_time_ms_ is the NN's OWN output, not a copy of the compress
    // time: separate predictions, and the cost model ranks on both.

    // Shuffle/quantize ride in the free high bits of compress_preset_ (bits
    // 0-7 preset, 8-15 shuffle elem size, 16-23 version, 24 quantize), the same
    // encoding the on-disk header uses. Element size 4 matches upstream.

    // Only the ENABLED bit is set: precision is unknown until the quantizer has
    // seen the data range, so Compress() fills it in at header-write time.
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
                       ratio_only, /*reuse=*/nullptr,
                       /*out_outcome=*/nullptr);
}

std::vector<CompressionStats> NeuroPressCandidateStatsDevice(
    ctp::compress::model::CompressionPredictor &predictor,
    clio::run::u64 chunk_size, const void *device_stats, void *stream,
    bool data_type_float, double error_bound, double min_psnr,
    bool *out_inference_failed, bool ratio_only,
    const ctp::compress::preprocess::PredictionReuseContext *reuse,
    ctp::compress::preprocess::PredictionReuseOutcome *out_outcome) {
  // Zeros: on this path the statistics reach neither the network (it reads
  // device_stats) nor the score. Fetching them here would mean synchronizing
  // before the inference rather than after.
  return RankIntoStats(predictor, chunk_size, 0.0, 0.0, 0.0, data_type_float,
                       error_bound, device_stats, stream, min_psnr,
                       out_inference_failed, ratio_only, reuse,
                       out_outcome);
}

}  // namespace clio::cte::compressor
