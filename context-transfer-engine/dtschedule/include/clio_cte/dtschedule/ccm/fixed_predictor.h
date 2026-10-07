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

#ifndef CLIO_CTE_DTSCHEDULE_CCM_FIXED_PREDICTOR_H_
#define CLIO_CTE_DTSCHEDULE_CCM_FIXED_PREDICTOR_H_

#include <clio_cte/dtschedule/ccm/predictor.h>
#include <string>

namespace clio::cte::dtschedule::ccm {

/**
 * Fixed predictor: always returns the same candidate, cost 0.
 *
 * For testing and ablations. Returns a prediction where ctime and dtime
 * are both 0 (no selection cost), ratio is 1.0 (no compression assumed).
 */
class FixedPredictor : public CcmPredictor {
 public:
  /**
   * Construct a fixed predictor for a specific codec.
   *
   * @param lib Compression library name (ignored; fixed codec always wins)
   * @param preset Compression preset (ignored)
   */
  FixedPredictor(const std::string &lib, ctp::CompressionPreset preset)
      : lib_(lib), preset_(preset) {}

  ~FixedPredictor() override = default;

  /**
   * Always return zero-cost prediction.
   *
   * This causes the ranker to always choose lib_/preset_, and the "no
   * selection cost" assumption helps isolate selection overhead in ablations.
   */
  Prediction Predict(const Features &features,
                     const std::string &lib,
                     ctp::CompressionPreset preset) override;

  /**
   * No-op; fixed codec cannot learn from observations.
   */
  void Observe(const Features &features,
               const std::string &lib,
               ctp::CompressionPreset preset,
               double obs_ctime_ms,
               double obs_dtime_ms,
               double obs_ratio) override;

  /**
   * Get the fixed codec library name.
   */
  const std::string &GetLib() const { return lib_; }

  /**
   * Get the fixed codec preset.
   */
  ctp::CompressionPreset GetPreset() const { return preset_; }

 private:
  std::string lib_;                  ///< Fixed library name
  ctp::CompressionPreset preset_;   ///< Fixed preset
};

}  // namespace clio::cte::dtschedule::ccm

#endif  // CLIO_CTE_DTSCHEDULE_CCM_FIXED_PREDICTOR_H_
