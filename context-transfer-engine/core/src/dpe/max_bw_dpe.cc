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

#include <clio_cte/core/dpe/max_bw_dpe.h>

#include <algorithm>

#include "clio_ctp/util/logging.h"

namespace clio::cte::core {

MaxBwDpe::MaxBwDpe() {
}

std::vector<TargetInfo> MaxBwDpe::SelectTargets(const std::vector<TargetInfo>& targets,
                                               float blob_score,
                                               clio::run::u64 data_size) {
  std::vector<TargetInfo> result;

  if (targets.empty()) {
    return result;
  }

  // Filter targets with sufficient space
  std::vector<TargetInfo> available_targets;
  for (const auto& target : targets) {
    if (target.remaining_space_ >= data_size) {
      available_targets.push_back(target);
    }
  }

  if (available_targets.empty()) {
    return result;  // No targets have space
  }

  // Sort targets by performance metrics (bandwidth or latency)
  auto perf_comparator = [data_size](const TargetInfo& a, const TargetInfo& b) {
    if (data_size >= kLatencyThreshold) {
      // Sort by write bandwidth (descending)
      return a.perf_metrics_.write_bandwidth_mbps_ > b.perf_metrics_.write_bandwidth_mbps_;
    } else {
      // Sort by latency (ascending - lower is better)
      double avg_latency_a = (a.perf_metrics_.read_latency_us_ + a.perf_metrics_.write_latency_us_) / 2.0;
      double avg_latency_b = (b.perf_metrics_.read_latency_us_ + b.perf_metrics_.write_latency_us_) / 2.0;
      return avg_latency_a < avg_latency_b;
    }
  };

  // Partition targets into two groups based on score:
  // - low_score: targets with score <= blob_score (preferred, matching tier or below)
  // - high_score: targets with score > blob_score (fallback, higher tier)
  std::vector<TargetInfo> low_score_targets;
  std::vector<TargetInfo> high_score_targets;

  HLOG(kDebug, "MaxBwDpe::SelectTargets: blob_score={}, ranking {} targets",
       blob_score, available_targets.size());

  for (const auto& target : available_targets) {
    HLOG(kDebug, "  Target pool_id=({},{}), target_score_={}, remaining_space_={}",
         target.bdev_client_.pool_id_.major_, target.bdev_client_.pool_id_.minor_,
         target.target_score_, target.remaining_space_);
    if (target.target_score_ <= blob_score) {
      low_score_targets.push_back(target);
      HLOG(kDebug, "    -> LOW_SCORE (preferred): target_score_ {} <= blob_score {}",
           target.target_score_, blob_score);
    } else {
      high_score_targets.push_back(target);
      HLOG(kDebug, "    -> HIGH_SCORE (fallback): target_score_ {} > blob_score {}",
           target.target_score_, blob_score);
    }
  }

  // Sort low_score targets by performance (already correct order for placement)
  std::sort(low_score_targets.begin(), low_score_targets.end(), perf_comparator);

  // Sort high_score targets by performance in REVERSE order
  // (when falling back to higher tiers, prefer lower-performing ones first)
  std::sort(high_score_targets.begin(), high_score_targets.end(),
            [&perf_comparator](const TargetInfo& a, const TargetInfo& b) {
              return perf_comparator(b, a);  // Reverse by swapping arguments
            });

  // Build result: low_score targets first (preferred), then high_score (fallback)
  result.reserve(low_score_targets.size() + high_score_targets.size());
  for (const auto& target : low_score_targets) {
    result.push_back(target);
  }
  for (const auto& target : high_score_targets) {
    result.push_back(target);
  }

  HLOG(kDebug, "MaxBwDpe::SelectTargets: returning {} targets ({} preferred, {} fallback)",
       result.size(), low_score_targets.size(), high_score_targets.size());

  return result;
}

} // namespace clio::cte::core
