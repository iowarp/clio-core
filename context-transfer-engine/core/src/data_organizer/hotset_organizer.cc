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

#include <clio_cte/core/data_organizer/hotset_organizer.h>
#include <clio_cte/core/core_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace clio::cte::core {

clio::run::TaskResume HotSetDataOrganizer::Reorganize(
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
  // Most-reused first. Stable tie-break on name so the chosen set does not
  // churn between rounds when counts are equal -- churn means migrations.
  std::sort(stats.begin(), stats.end(),
            [](const OrganizerBlobStat &a, const OrganizerBlobStat &b) {
              if (a.access_count_ != b.access_count_) {
                return a.access_count_ > b.access_count_;
              }
              return a.blob_name_ < b.blob_name_;
            });
  const clio::run::u64 budget =
      static_cast<clio::run::u64>(static_cast<double>(fast_bytes) * kFillFactor);
  clio::run::u64 spent = 0;
  for (const OrganizerBlobStat &stat : stats) {
    if (spent + stat.size_ > budget) break;   // tier full; the rest stay put
    spent += stat.size_;
    if (std::fabs(kHotScore - stat.score_) < kEpsilon) continue;  // already hot
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(server->ReorganizeBlobInternal(stat.tag_id_, stat.blob_name_,
                                                 kHotScore, rc));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::core
