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

#include <clio_cte/core/data_organizer/cyclic_organizer.h>
#include <clio_cte/core/core_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace clio::cte::core {

bool CyclicDataOrganizer::PageOf(const std::string &blob_name,
                                 clio::run::u64 &out) {
  // WHAT A PAGE BLOB IS ACTUALLY NAMED. DeviceVector submits page names as a
  // raw little-endian u32 with Context::kBlobNameRawInt32, but the CTE calls
  // NormalizeBlobName() on the way in, which renders the name DECIMAL in
  // place and clears the flag. So by the time a name reaches a BlobInfo -- and
  // therefore an organizer -- it is the decimal text of the page number.
  //
  // Testing for the 4-byte raw form here instead is not merely wrong, it is
  // wrong in a way that looks like it works: exactly the pages numbered
  // 1000-1023 have 4-character decimal names, so a size==4 test matches 24
  // blobs out of 1024 and decodes them as garbage page numbers.
  if (blob_name.empty() || blob_name.size() > 10) return false;
  clio::run::u64 v = 0;
  for (char c : blob_name) {
    if (c < '0' || c > '9') return false;
    v = v * 10 + static_cast<clio::run::u64>(c - '0');
  }
  out = v;
  return true;
}

clio::run::u64 CyclicDataOrganizer::HotPageLimit(clio::run::u64 fast_tier_bytes,
                                                 clio::run::u64 page_bytes,
                                                 clio::run::u64 num_tags) {
  if (page_bytes == 0 || num_tags == 0) return 0;
  const double budget =
      static_cast<double>(fast_tier_bytes) * kFillFactor /
      static_cast<double>(num_tags);
  return static_cast<clio::run::u64>(budget / static_cast<double>(page_bytes));
}

clio::run::TaskResume CyclicDataOrganizer::Reorganize(
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
  // PROMOTE WHAT IS VISIBLE, UP TO A BUDGET -- not a fixed page prefix.
  //
  // The first version pinned pages [0, C/page_size). It never fired: the CTE's
  // blob metadata holds only the pages that currently exist, and under a
  // paged sweep that set churns continuously (276 blobs one round, 3 the
  // next), so a fixed prefix almost never intersects what a round can see.
  // Instead: promote any visible vector page that is not already hot, and
  // stop once this container has promoted a fast tier's worth. promoted_bytes_
  // is member state, so the budget holds ACROSS rounds and the chosen set
  // stays stable -- which is what makes the migration amortize.
  const clio::run::u64 budget =
      static_cast<clio::run::u64>(static_cast<double>(fast_bytes) * kFillFactor);
  clio::run::u64 n_promo = 0, n_fail = 0, n_hot = 0, n_full = 0;
  for (const OrganizerBlobStat &stat : stats) {
    clio::run::u64 pn = 0;
    if (!PageOf(stat.blob_name_, pn)) continue;   // not a vector page
    if (std::fabs(kHotScore - stat.score_) < kEpsilon) { ++n_hot; continue; }
    if (promoted_bytes_ + stat.size_ > budget) { ++n_full; continue; }
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(server->ReorganizeBlobInternal(stat.tag_id_, stat.blob_name_,
                                                 kHotScore, rc));
    if (rc != 0) { ++n_fail; continue; }
    promoted_bytes_ += stat.size_;
    ++n_promo;
  }
  if (n_promo != 0 || n_fail != 0) {
    HLOG(kDebug,
         "CyclicDataOrganizer: round: promoted={} failed={} already_hot={} "
         "budget_full={} promoted_bytes={} budget={}",
         n_promo, n_fail, n_hot, n_full, promoted_bytes_, budget);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::core
