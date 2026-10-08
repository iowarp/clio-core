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

#include <clio_cte/dtschedule/ccm/ranker.h>
#include <cmath>
#include <limits>
#include <algorithm>

namespace clio::cte::dtschedule::ccm {

Decision Ranker::Rank(const std::vector<Candidate> &candidates,
                      const Features &features,
                      size_t size,
                      CcmPredictor *predictor,
                      const std::string &objective,
                      double tier_bw,
                      double load_mult,
                      double ratio_noise_sigma,
                      std::mt19937_64 *rng,
                      bool compare_raw,
                      const std::function<double(uint64_t)> *store_bw,
                      double cpu_parallelism) {
  Decision decision{};
  // Costs are per-chunk shares of the bottleneck, not latencies: codec work
  // runs on cpu_parallelism runtime workers at once, while the network and a
  // tier's device are shared by every chunk in flight. Load scales CPU only.
  const double par = std::max(cpu_parallelism, 1.0);
  decision.n_candidates_ = candidates.size();
  decision.chosen_tier_ = "default";  // tier selection arrives in phase 4
  if (candidates.empty()) {
    return decision;
  }
  std::normal_distribution<double> noise(0.0, ratio_noise_sigma);
  const double size_mb = static_cast<double>(size) / 1e6;
  double best_cost = std::numeric_limits<double>::max();
  double best_ratio = 0.0;
  int best = -1;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto &c = candidates[i];
    Prediction pred = predictor->Predict(features, c.lib_, c.preset_);
    if (ratio_noise_sigma > 0.0 && rng != nullptr) {
      pred.ratio_ *= std::exp(noise(*rng));  // independent draw per candidate
    }
    if (pred.ratio_ < 1.0) {
      pred.ratio_ = 1.0;
    }
    // Bandwidth of the tier this candidate's stored size lands in (raw may
    // spill to a slower tier while the compressed bytes still fit a fast one).
    const double z_bw =
        store_bw != nullptr
            ? (*store_bw)(static_cast<uint64_t>(size / pred.ratio_))
            : tier_bw;
    const double storage_ms = (size_mb / pred.ratio_) / z_bw;
    const double cost =
        (load_mult * pred.ctime_ms_ + pred.dtime_ms_) / par + storage_ms;
    const bool useful = pred.ratio_ >= kMinUsefulRatio;
    decision.candidates_.push_back(CandidateRecord{
        c.lib_, c.preset_, pred.ctime_ms_, pred.dtime_ms_, pred.ratio_,
        useful ? cost : 0.0, useful ? kReasonOk : kReasonSkipRatio});
    if (!useful) {
      continue;
    }
    bool better = (objective == "ratio")
                      ? (pred.ratio_ > best_ratio ||
                         (pred.ratio_ == best_ratio && cost < best_cost))
                      : (cost < best_cost);
    if (better) {
      best_cost = cost;
      best_ratio = pred.ratio_;
      best = static_cast<int>(i);
    }
  }
  // The alternative every codec competes with: store the bytes as they are.
  // Under the performance objective a codec only wins when compressing and
  // storing the smaller payload beats storing the raw payload (same load
  // multiplier, no decompression); under the ratio objective raw is the
  // fallback only when no candidate is useful.
  const double raw_bw =
      store_bw != nullptr ? (*store_bw)(static_cast<uint64_t>(size)) : tier_bw;
  const double raw_cost = size_mb / raw_bw;
  const bool raw_wins =
      best < 0 ||
      (compare_raw && objective != "ratio" && best_cost >= raw_cost);
  decision.candidates_.push_back(CandidateRecord{
      kRawLib, ctp::CompressionPreset::FAST, 0.0, 0.0, 1.0, raw_cost,
      raw_wins ? kReasonOk : kReasonRawCostlier});
  if (raw_wins) {
    return decision;  // nothing worth compressing: chosen_lib_ stays empty
  }
  const auto &rec = decision.candidates_[best];
  decision.chosen_lib_ = rec.lib_;
  decision.chosen_preset_ = rec.preset_;
  decision.pred_ctime_ms_ = rec.pred_ctime_ms_;
  decision.pred_dtime_ms_ = rec.pred_dtime_ms_;
  decision.pred_ratio_ = rec.pred_ratio_;
  return decision;
}

}  // namespace clio::cte::dtschedule::ccm
