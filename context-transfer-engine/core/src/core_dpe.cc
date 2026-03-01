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

#include <wrp_cte/core/core_dpe.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "hermes_shm/util/logging.h"

namespace wrp_cte::core {

// Static member definition for round-robin counter
std::atomic<chi::u32> RoundRobinDpe::round_robin_counter_(0);

// DPE Type conversion functions
DpeType StringToDpeType(const std::string& dpe_str) {
  if (dpe_str == "random") {
    return DpeType::kRandom;
  } else if (dpe_str == "round_robin" || dpe_str == "roundrobin") {
    return DpeType::kRoundRobin;
  } else if (dpe_str == "max_bw" || dpe_str == "maxbw") {
    return DpeType::kMaxBW;
  } else {
    HLOG(kError, "Unknown DPE type: {}, defaulting to random", dpe_str);
    return DpeType::kRandom;
  }
}

std::string DpeTypeToString(DpeType dpe_type) {
  switch (dpe_type) {
    case DpeType::kRandom:
      return "random";
    case DpeType::kRoundRobin:
      return "round_robin";
    case DpeType::kMaxBW:
      return "max_bw";
    default:
      return "random";
  }
}

// RandomDpe Implementation
RandomDpe::RandomDpe() : rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

std::vector<TargetInfo> RandomDpe::SelectTargets(const std::vector<TargetInfo>& targets,
                                                float blob_score,
                                                chi::u64 data_size) {
  std::vector<TargetInfo> result;

  if (targets.empty()) {
    return result;
  }

  // Partition targets with sufficient space into two groups based on score
  std::vector<TargetInfo> low_score_targets;
  std::vector<TargetInfo> high_score_targets;

  for (const auto& target : targets) {
    if (target.remaining_space_ >= data_size) {
      if (target.target_score_ <= blob_score) {
        low_score_targets.push_back(target);
      } else {
        high_score_targets.push_back(target);
      }
    }
  }

  // Shuffle each group randomly
  std::shuffle(low_score_targets.begin(), low_score_targets.end(), rng_);
  std::shuffle(high_score_targets.begin(), high_score_targets.end(), rng_);

  // Build result: low_score targets first (preferred), then high_score (fallback)
  result.reserve(low_score_targets.size() + high_score_targets.size());
  for (const auto& target : low_score_targets) {
    result.push_back(target);
  }
  for (const auto& target : high_score_targets) {
    result.push_back(target);
  }

  return result;
}

// RoundRobinDpe Implementation  
RoundRobinDpe::RoundRobinDpe() {
}

std::vector<TargetInfo> RoundRobinDpe::SelectTargets(const std::vector<TargetInfo>& targets,
                                                    float blob_score,
                                                    chi::u64 data_size) {
  std::vector<TargetInfo> result;

  if (targets.empty()) {
    return result;
  }

  // Partition targets with sufficient space into two groups based on score
  std::vector<TargetInfo> low_score_targets;
  std::vector<TargetInfo> high_score_targets;

  for (const auto& target : targets) {
    if (target.remaining_space_ >= data_size) {
      if (target.target_score_ <= blob_score) {
        low_score_targets.push_back(target);
      } else {
        high_score_targets.push_back(target);
      }
    }
  }

  // Apply round-robin rotation to each group
  chi::u32 counter = round_robin_counter_.fetch_add(1);

  if (!low_score_targets.empty()) {
    size_t shift_amount = counter % low_score_targets.size();
    if (shift_amount > 0) {
      std::rotate(low_score_targets.begin(),
                  low_score_targets.begin() + shift_amount,
                  low_score_targets.end());
    }
  }

  if (!high_score_targets.empty()) {
    size_t shift_amount = counter % high_score_targets.size();
    if (shift_amount > 0) {
      std::rotate(high_score_targets.begin(),
                  high_score_targets.begin() + shift_amount,
                  high_score_targets.end());
    }
  }

  // Build result: low_score targets first (preferred), then high_score (fallback)
  result.reserve(low_score_targets.size() + high_score_targets.size());
  for (const auto& target : low_score_targets) {
    result.push_back(target);
  }
  for (const auto& target : high_score_targets) {
    result.push_back(target);
  }

  return result;
}

// MaxBwDpe Implementation
MaxBwDpe::MaxBwDpe() {
}

std::vector<TargetInfo> MaxBwDpe::SelectTargets(const std::vector<TargetInfo>& targets,
                                               float blob_score,
                                               chi::u64 data_size) {
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
  auto perf_comparator = [data_size, this](const TargetInfo& a, const TargetInfo& b) {
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

// DpeFactory Implementation
std::unique_ptr<DataPlacementEngine> DpeFactory::CreateDpe(DpeType dpe_type) {
  switch (dpe_type) {
    case DpeType::kRandom:
      return std::make_unique<RandomDpe>();
    case DpeType::kRoundRobin:
      return std::make_unique<RoundRobinDpe>();
    case DpeType::kMaxBW:
      return std::make_unique<MaxBwDpe>();
    default:
      HLOG(kError, "Unknown DPE type, defaulting to Random");
      return std::make_unique<RandomDpe>();
  }
}

std::unique_ptr<DataPlacementEngine> DpeFactory::CreateDpe(const std::string& dpe_str) {
  return CreateDpe(StringToDpeType(dpe_str));
}

} // namespace wrp_cte::core