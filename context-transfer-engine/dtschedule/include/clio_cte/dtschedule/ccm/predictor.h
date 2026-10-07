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

#ifndef CLIO_CTE_DTSCHEDULE_CCM_PREDICTOR_H_
#define CLIO_CTE_DTSCHEDULE_CCM_PREDICTOR_H_

#include <array>
#include <string>
#include <memory>
#include <map>
#include <clio_ctp/compress/compress_factory.h>

namespace clio::cte::dtschedule::ccm {

/**
 * Data statistics features for prediction.
 * Computed from a sample of the blob being compressed.
 */
struct Features {
  double entropy_;        ///< Bytewise Shannon entropy (0-8)
  double mad_;            ///< Mean absolute deviation
  double d2_;             ///< Mean |second derivative|
  size_t size_;           ///< Blob size in bytes
  int dtype_;             ///< 0=char, 1=float, 2=int
};

/**
 * Prediction result: estimated compress time, decompress time, and ratio.
 */
struct Prediction {
  double ctime_ms_;       ///< Predicted compression time (ms)
  double dtime_ms_;       ///< Predicted decompression time (ms)
  double ratio_;          ///< Predicted compression ratio
};

/**
 * Abstract predictor interface for estimating compression characteristics.
 *
 * Different CCM backends (Q-table, EMA, fixed, oracle) implement this
 * interface to provide codec-specific cost predictions.
 */
/**
 * Ratio convention for the whole module: original_size / compressed_size
 * (>= 1). A codec is only worth using when the stored bytes are below
 * 87.5 % of the original, i.e. when the ratio is at least this value.
 */
constexpr double kMinUsefulRatio = 1.0 / 0.875;

/**
 * Prior used for a candidate nothing is known about: 200 MB/s compression,
 * 600 MB/s decompression, 2x ratio.
 *
 * @param size blob size in bytes
 * @return the prior prediction
 */
inline Prediction NeutralPrior(size_t size) {
  const double mb = static_cast<double>(size) / 1e6;
  return Prediction{mb / 200.0 * 1000.0, mb / 600.0 * 1000.0, 2.0};
}

class CcmPredictor {
 public:
  virtual ~CcmPredictor() = default;

  /**
   * Predict compression characteristics for a candidate (lib, preset).
   *
   * @param features Data statistics (entropy, MAD, d2, size, dtype)
   * @param lib Compression library name (e.g., "zstd", "lz4")
   * @param preset Compression preset (kFast=1, kBalanced=2, kBest=3)
   * @return Prediction with estimated ctime_ms, dtime_ms, ratio
   */
  virtual Prediction Predict(const Features &features,
                             const std::string &lib,
                             ctp::CompressionPreset preset) = 0;

  /**
   * Observe actual compression performance and update model.
   *
   * Called after compression completes. The predictor may use this
   * observation to online-update its model (e.g., EMA) or trigger
   * resampling (Q-table with high error).
   *
   * @param features Data statistics (same blob)
   * @param lib Compression library
   * @param preset Compression preset
   * @param obs_ctime_ms Observed compression time
   * @param obs_dtime_ms Observed decompression time (0 if not measured)
   * @param obs_ratio Observed compression ratio (compressed_size / original)
   */
  virtual void Observe(const Features &features,
                       const std::string &lib,
                       ctp::CompressionPreset preset,
                       double obs_ctime_ms,
                       double obs_dtime_ms,
                       double obs_ratio) = 0;
};

}  // namespace clio::cte::dtschedule::ccm

#endif  // CLIO_CTE_DTSCHEDULE_CCM_PREDICTOR_H_
