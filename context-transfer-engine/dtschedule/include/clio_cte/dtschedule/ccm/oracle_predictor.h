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

#ifndef CLIO_CTE_DTSCHEDULE_CCM_ORACLE_PREDICTOR_H_
#define CLIO_CTE_DTSCHEDULE_CCM_ORACLE_PREDICTOR_H_

#include <clio_cte/dtschedule/ccm/predictor.h>
#include <map>
#include <utility>

namespace clio::cte::dtschedule::ccm {

/**
 * Oracle predictor: compress/decompress every candidate to get ground truth.
 *
 * For E8/E9 evaluation only. The Predict method is called before compression
 * actually happens, so Oracle::Predict is a no-op. The actual cost comes from
 * the decision stage running Oracle on all candidates (expensive).
 *
 * This predictor returns default predictions; cost accounting happens in the
 * ranker, which calls Oracle for every candidate explicitly.
 */
class OraclePredictor : public CcmPredictor {
 public:
  using Key = std::pair<std::string, ctp::CompressionPreset>;
  /** Install the measurements of the current chunk (from the probe). */
  void SetMeasurements(std::map<Key, Prediction> measurements) {
    measurements_ = std::move(measurements);
  }
  OraclePredictor() = default;
  ~OraclePredictor() override = default;

  /**
   * Return a placeholder (no real prediction).
   *
   * The actual measurements are made in the ranker/decision stage which
   * compresses every candidate in the oracle codepath.
   */
  Prediction Predict(const Features &features,
                     const std::string &lib,
                     ctp::CompressionPreset preset) override;

  /**
   * No-op; Oracle does not learn from observations (it measures directly).
   */
  void Observe(const Features &features,
               const std::string &lib,
               ctp::CompressionPreset preset,
               double obs_ctime_ms,
               double obs_dtime_ms,
               double obs_ratio) override;
 private:
  std::map<Key, Prediction> measurements_;  ///< Truth for the current chunk
};

}  // namespace clio::cte::dtschedule::ccm

#endif  // CLIO_CTE_DTSCHEDULE_CCM_ORACLE_PREDICTOR_H_
