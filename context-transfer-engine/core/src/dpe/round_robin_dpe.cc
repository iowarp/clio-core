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

#include <clio_cte/core/dpe/round_robin_dpe.h>

#include <algorithm>

namespace clio::cte::core {

// Static member definition for round-robin counter
std::atomic<clio::run::u32> RoundRobinDpe::round_robin_counter_(0);

RoundRobinDpe::RoundRobinDpe() {
}

std::vector<TargetInfo> RoundRobinDpe::SelectTargets(const std::vector<TargetInfo>& targets,
                                                    float blob_score,
                                                    clio::run::u64 data_size) {
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
  clio::run::u32 counter = round_robin_counter_.fetch_add(1);

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

} // namespace clio::cte::core
