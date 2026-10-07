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

#ifndef CLIO_CTE_DTSCHEDULE_CCM_EMA_PREDICTOR_H_
#define CLIO_CTE_DTSCHEDULE_CCM_EMA_PREDICTOR_H_

#include <clio_cte/dtschedule/ccm/predictor.h>
#include <map>
#include <string>
#include <utility>

namespace clio::cte::dtschedule::ccm {

/**
 * EMA predictor: per-(lib, preset) exponential moving average.
 *
 * No data features; only keeps running EMA of observed cost per candidate.
 * Used as the baseline in E8 (predictor ablation).
 * Alpha = 0.2 (weights recent samples more heavily).
 */
class EmaPredictor : public CcmPredictor {
 public:
  EmaPredictor() = default;
  ~EmaPredictor() override = default;

  /**
   * Predict using per-(lib, preset) EMA.
   *
   * If no observation yet, returns a default (cost=1e6, ratio=1.0) to
   * encourage exploration of untested candidates.
   */
  Prediction Predict(const Features &features,
                     const std::string &lib,
                     ctp::CompressionPreset preset) override;

  /**
   * Update EMA with observed values.
   *
   * EMA_new = alpha * obs + (1 - alpha) * EMA_old, alpha = 0.2.
   * Ignores data features; only uses (lib, preset) key.
   */
  void Observe(const Features &features,
               const std::string &lib,
               ctp::CompressionPreset preset,
               double obs_ctime_ms,
               double obs_dtime_ms,
               double obs_ratio) override;

  /** True once the one-time probe has seeded every candidate. */
  bool IsSeeded() const { return seeded_; }
  /** Record that the one-time probe ran. */
  void MarkSeeded() { seeded_ = true; }

 private:
  bool seeded_ = false;  ///< Set after the cold-start probe
  /**
   * Per-candidate EMA state.
   */
  struct EmaState {
    double ema_ctime_ms_ = 0.0;    ///< EMA of compression time
    double ema_dtime_ms_ = 0.0;    ///< EMA of decompression time
    double ema_ratio_ = 1.0;       ///< EMA of compression ratio
    bool initialized_ = false;     ///< True if any observation received
  };

  static constexpr double kAlpha = 0.2;  ///< EMA smoothing factor

  std::map<std::pair<std::string, ctp::CompressionPreset>, EmaState>
      ema_states_;  ///< Per-(lib, preset) EMA
};

}  // namespace clio::cte::dtschedule::ccm

#endif  // CLIO_CTE_DTSCHEDULE_CCM_EMA_PREDICTOR_H_
