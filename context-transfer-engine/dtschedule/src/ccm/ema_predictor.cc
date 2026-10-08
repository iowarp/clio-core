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

#include <clio_cte/dtschedule/ccm/ema_predictor.h>

namespace clio::cte::dtschedule::ccm {

Prediction EmaPredictor::Predict(const Features &features,
                                  const std::string &lib,
                                  ctp::CompressionPreset preset) {
  (void)features;  // EMA doesn't use data features

  // Create key for this (lib, preset) pair
  auto key = std::make_pair(lib, preset);

  // Look up (lib, preset) in ema_states_
  auto it = ema_states_.find(key);
  if (it != ema_states_.end() && it->second.initialized_) {
    // Return the EMA values
    return {it->second.ema_ctime_ms_, it->second.ema_dtime_ms_,
            it->second.ema_ratio_};
  }

  // Not found or not initialized - return default (high cost to encourage
  // exploration)
  return NeutralPrior(features.size_);
}

void EmaPredictor::Observe(const Features &features,
                            const std::string &lib,
                            ctp::CompressionPreset preset,
                            double obs_ctime_ms,
                            double obs_dtime_ms,
                            double obs_ratio) {
  (void)features;  // EMA doesn't use data features

  // Create key for this (lib, preset) pair
  auto key = std::make_pair(lib, preset);

  // Get or create EmaState for this key
  auto &ema_state = ema_states_[key];

  if (!ema_state.initialized_) {
    // First observation - set EMA directly
    ema_state.ema_ctime_ms_ = obs_ctime_ms;
    ema_state.ema_dtime_ms_ = obs_dtime_ms;
    ema_state.ema_ratio_ = obs_ratio;
    ema_state.initialized_ = true;
  } else {
    // Update EMA: EMA_new = alpha * obs + (1 - alpha) * EMA_old
    ema_state.ema_ctime_ms_ = kAlpha * obs_ctime_ms +
                               (1.0 - kAlpha) * ema_state.ema_ctime_ms_;
    // obs_dtime_ms <= 0: no decompression observed; keep the estimate.
    if (obs_dtime_ms > 0.0) {
      ema_state.ema_dtime_ms_ = kAlpha * obs_dtime_ms +
                                (1.0 - kAlpha) * ema_state.ema_dtime_ms_;
    }
    ema_state.ema_ratio_ =
        kAlpha * obs_ratio + (1.0 - kAlpha) * ema_state.ema_ratio_;
  }
}

}  // namespace clio::cte::dtschedule::ccm
