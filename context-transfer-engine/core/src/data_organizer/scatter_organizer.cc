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

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <clio_cte/core/data_organizer/scatter_organizer.h>
#include <clio_cte/core/core_runtime.h>

#include <cmath>
#include <vector>

namespace clio::cte::core {

bool ScatterDataOrganizer::ShouldPromote(const OrganizerBlobStat &stat,
                                         bool gather_phase) {
  if (gather_phase) return true;
  // A page that has been read at all is past its write-once life.
  return stat.last_read_ != 0;
}

clio::run::TaskResume ScatterDataOrganizer::Reorganize(
    Runtime *server, clio::run::u32 replica_id) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  std::vector<OrganizerBlobStat> stats;
  server->CollectOrganizerBlobStats(replica_id, stats);
  const clio::run::u64 fast_bytes = server->FastTierCapacityBytes();
  if (fast_bytes == 0 || stats.empty()) {
    CLIO_CO_RETURN;
  }
  const clio::run::i32 hint = server->OrganizerHint();
  const bool gather = hint == kPhaseGather;
  // CLIO_ORGANIZER_TRACE=1 reports each distinct phase hint this organizer
  // observes, once. The hint arrives by ReorganizeHint broadcast from the
  // application, and a hint that never lands is indistinguishable from a
  // policy that does not help -- both just look like "no effect" -- so the
  // arrival has to be observable on its own.
  {
    static std::atomic<clio::run::i32> last_seen{-12345};
    if (last_seen.exchange(hint) != hint) {
      const char *tr = getenv("CLIO_ORGANIZER_TRACE");
      if (tr != nullptr && tr[0] != '\0' && tr[0] != '0') {
        std::fprintf(stderr, "[organizer scatter] phase hint -> %d "
                     "(blobs=%zu fast_bytes=%llu)\n", (int)hint, stats.size(),
                     (unsigned long long)fast_bytes);
        std::fflush(stderr);
      }
    }
  }
  const clio::run::u64 budget =
      static_cast<clio::run::u64>(static_cast<double>(fast_bytes) * kFillFactor);
  for (const OrganizerBlobStat &stat : stats) {
    if (!ShouldPromote(stat, gather)) continue;
    if (std::fabs(kHotScore - stat.score_) < kEpsilon) continue;
    if (promoted_bytes_ + stat.size_ > budget) continue;
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(server->ReorganizeBlobInternal(stat.tag_id_, stat.blob_name_,
                                                 kHotScore, rc));
    if (rc == 0) promoted_bytes_ += stat.size_;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::core
