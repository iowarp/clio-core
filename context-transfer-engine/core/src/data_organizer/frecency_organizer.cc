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

#include <clio_cte/core/data_organizer/frecency_organizer.h>
#include <clio_cte/core/core_runtime.h>

#include <algorithm>
#include <cmath>

namespace clio::cte::core {

float FrecencyDataOrganizer::ComputeScore(const OrganizerBlobStat &stat,
                                          Timestamp now) {
  // Age since the most recent data access (read or write). Timestamps are
  // steady-clock ns; guard against a stamp taken after `now` was sampled.
  Timestamp last_access = std::max(stat.last_read_, stat.last_modified_);
  double age_sec = 0.0;
  if (now > last_access) {
    age_sec = static_cast<double>(now - last_access) / 1e9;
  }

  // Recency: exponential decay with a half-life — 1.0 at the moment of
  // access, 0.5 after kRecencyHalfLifeSec, ~0 for long-cold blobs.
  double recency = std::exp2(-age_sec / kRecencyHalfLifeSec);

  // Frequency: saturating in [0, 1) — 0.5 at kFreqSaturation accesses.
  double n = static_cast<double>(stat.access_count_);
  double frequency = n / (n + kFreqSaturation);

  double score = kRecencyWeight * recency + (1.0 - kRecencyWeight) * frequency;
  return static_cast<float>(std::clamp(score, 0.0, 1.0));
}

clio::run::TaskResume FrecencyDataOrganizer::Reorganize(
    Runtime *server, clio::run::u32 replica_id) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  // Snapshot this replica's partition of the blob space. The snapshot (not
  // live map references) is what we iterate, so blob deletion racing this
  // loop is benign — ReorganizeBlobInternal returns "not found" and we move
  // on.
  std::vector<OrganizerBlobStat> stats;
  server->CollectOrganizerBlobStats(replica_id, stats);

  Timestamp now = GetCurrentTimeNs();
  for (const OrganizerBlobStat &stat : stats) {
    float new_score = ComputeScore(stat, now);
    // ReorganizeBlobInternal applies the score through the normal
    // reorganization path (read → free old placement → re-put at the new
    // score) and itself skips blobs whose score delta is under the
    // configured score_difference_threshold, so steady-state hot/cold blobs
    // cost one metadata check per round, not a data movement.
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(server->ReorganizeBlobInternal(stat.tag_id_,
                                                 stat.blob_name_, new_score,
                                                 rc));
    if (rc != 0) {
      HLOG(kDebug,
           "FrecencyDataOrganizer: rescore skipped/failed: blob={}, "
           "new_score={}, rc={}",
           stat.blob_name_, new_score, rc);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::core
