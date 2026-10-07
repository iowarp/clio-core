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

#ifndef CLIO_CTE_DTSCHEDULE_CCM_RANKER_H_
#define CLIO_CTE_DTSCHEDULE_CCM_RANKER_H_

#include <clio_cte/dtschedule/ccm/predictor.h>
#include <clio_cte/dtschedule/ccm/candidates.h>
#include <random>
#include <functional>
#include <vector>
#include <string>

namespace clio::cte::dtschedule::ccm {


/**
 * Decision: the chosen codec and associated prediction.
 */
struct Decision {
  std::string chosen_lib_;               ///< Chosen library name
  ctp::CompressionPreset chosen_preset_; ///< Chosen preset
  std::string chosen_tier_;              ///< Tier for this decision (phase 3+)
  double pred_ctime_ms_;                 ///< Predicted compress time
  double pred_dtime_ms_;                 ///< Predicted decompress time
  double pred_ratio_;                    ///< Predicted compression ratio
  size_t n_candidates_;                  ///< Number of candidates considered
  int qos_stage_index_ = -1;             ///< Which QoS stage matched (-1 if none)
  std::vector<CandidateRecord> candidates_;  ///< Evaluated candidates (task 3)
};

/**
 * Candidate ranking: score each (lib, preset) pair and select the best.
 *
 * Scores based on:
 * - objective: "performance" → minimize cost, "ratio" → maximize ratio (time tie-breaker)
 * - cost = ctime + dtime + (size / ratio) / tier_bw
 * - decision_order: "joint" (DESIGN default), "codec_first", "tier_first"
 *
 * Skips compression if best predicted ratio >= 0.875 (cutoff for "not worth compressing").
 */
class Ranker {
 public:
  /**
   * Rank candidates and select the best.
   *
   * Scores each candidate using the predictor, then picks the best based on
   * objective and decision_order knobs. Returns decision or skip.
   *
   * @param candidates Filtered candidate list
   * @param features Data statistics
   * @param size Original blob size (bytes)
   * @param predictor Compression characteristic predictor
   * @param objective "performance" or "ratio"
   * @param tier_bw Chosen tier bandwidth (MB/ms) for cost model
   * @param load_mult Load multiplier for this decision
   * @return Decision with chosen codec and predictions
   */
  static Decision Rank(const std::vector<Candidate> &candidates,
                       const Features &features,
                       size_t size,
                       CcmPredictor *predictor,
                       const std::string &objective,
                       double tier_bw,
                       double load_mult = 1.0,
                       double ratio_noise_sigma = 0.0,
                       std::mt19937_64 *rng = nullptr,
                       bool compare_raw = true,
                       const std::function<double(uint64_t)> *store_bw =
                           nullptr);

};

}  // namespace clio::cte::dtschedule::ccm

#endif  // CLIO_CTE_DTSCHEDULE_CCM_RANKER_H_
