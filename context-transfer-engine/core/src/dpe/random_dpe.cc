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

#include <clio_cte/core/dpe/random_dpe.h>

#include <algorithm>
#include <chrono>

namespace clio::cte::core {

RandomDpe::RandomDpe() : rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

std::vector<TargetInfo> RandomDpe::SelectTargets(const std::vector<TargetInfo>& targets,
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

} // namespace clio::cte::core
