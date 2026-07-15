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

#ifndef WRPCTE_CORE_DATA_ORGANIZER_FRECENCY_ORGANIZER_H_
#define WRPCTE_CORE_DATA_ORGANIZER_FRECENCY_ORGANIZER_H_

#include <clio_cte/core/data_organizer/data_organizer.h>

#include <string>

namespace clio::cte::core {

/**
 * Frecency-based data organizer.
 *
 * Rescores blobs with a frecency score — a blend of access recency and
 * access frequency, both normalized to [0, 1]:
 *
 *   recency   = 2^(-age / half_life)        age = now - last access
 *   frequency = n / (n + kFreqSaturation)   n   = data ops on the blob
 *   score     = kRecencyWeight * recency + (1 - kRecencyWeight) * frequency
 *
 * Recently- or frequently-accessed blobs score high (float toward fast
 * tiers); cold blobs decay toward 0 (sink toward slow tiers). The new score
 * is applied through Runtime::ReorganizeBlobInternal, which already skips
 * moves whose score delta is below the configured
 * score_difference_threshold — so a hot blob that stays hot is not
 * repeatedly rewritten.
 */
class FrecencyDataOrganizer : public DataOrganizer {
 public:
  // Half-life of the recency term: a blob untouched for this long loses half
  // its recency score.
  static constexpr double kRecencyHalfLifeSec = 600.0;
  // Access count at which the frequency term reaches 0.5 (saturates toward
  // 1.0 as the count grows).
  static constexpr double kFreqSaturation = 10.0;
  // Blend weight of recency vs frequency.
  static constexpr double kRecencyWeight = 0.5;

  clio::run::TaskResume Reorganize(Runtime *server,
                                   clio::run::u32 replica_id) override;

  std::string GetName() const override { return "frecency"; }

  /**
   * Compute the frecency score for one blob snapshot (pure function, exposed
   * for unit testing).
   * @param stat Blob metadata snapshot
   * @param now Current steady-clock timestamp (ns)
   * @return Score in [0, 1]
   */
  static float ComputeScore(const OrganizerBlobStat &stat, Timestamp now);
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_DATA_ORGANIZER_FRECENCY_ORGANIZER_H_
