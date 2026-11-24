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
#include <clio_cte/core/data_organizer/grayscott_organizer.h>
#include <clio_cte/core/core_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace clio::cte::core {

bool GrayScottDataOrganizer::PageOf(const std::string &blob_name,
                                    clio::run::u64 &out) {
  // WHAT A PAGE BLOB IS ACTUALLY NAMED. DeviceVector submits page names as a
  // raw little-endian u32 with Context::kBlobNameRawInt32, but the CTE calls
  // NormalizeBlobName() on the way in, which renders the name DECIMAL in
  // place and clears the flag. So by the time a name reaches a BlobInfo -- and
  // therefore an organizer -- it is the decimal text of the page number.
  //
  // Testing for the 4-byte raw form here instead is not merely wrong, it is
  // wrong in a way that looks like it works: on a 2048-page vector exactly the
  // pages numbered 1000-2047 have 4-character decimal names, so a size==4 test
  // matches half the tag and memcpy-decodes each to garbage ("1000" reads back
  // as 0x30303031 = 808464433). Rule 2 below therefore never saw a real page
  // number until this was fixed; rule 1 matches on the TAG name and is
  // unaffected.
  if (blob_name.empty() || blob_name.size() > 10) return false;
  clio::run::u64 v = 0;
  for (char c : blob_name) {
    if (c < '0' || c > '9') return false;
    v = v * 10 + static_cast<clio::run::u64>(c - '0');
  }
  out = v;
  return true;
}

float GrayScottDataOrganizer::TargetScore(const OrganizerBlobStat &stat,
                                          Timestamp now, clio::run::i32 hint,
                                          clio::run::u64 pages_in_tag) {
  // Rule 1: checkpoint history is dead after write. By tag name only -- see
  // the header for why "never read" cannot be trusted on paged data.
  (void)now;
  if (stat.tag_name_.rfind(kCheckpointPrefix, 0) == 0) return kColdScore;
  // Rule 2: the pair being overwritten this step, only with a hint and only
  // on a tag shaped like the four-region vector.
  if (hint <= 0 || pages_in_tag == 0 || pages_in_tag % kRegions != 0) {
    return -1.0f;
  }
  clio::run::u64 pn = 0;
  if (!PageOf(stat.blob_name_, pn)) return -1.0f;
  const clio::run::u64 nz = pages_in_tag / kRegions;
  const clio::run::u64 region = pn / nz;
  // Step s = hint - 1 reads regions {0,1} when s is even and {2,3} when odd;
  // it writes the other pair, whose previous contents are dead.
  const clio::run::u32 step = static_cast<clio::run::u32>(hint - 1);
  const bool writes_low_pair = (step % 2u) == 1u;
  const bool dead = writes_low_pair ? (region < 2) : (region >= 2);
  return dead ? kWarmScore : -1.0f;
}

clio::run::TaskResume GrayScottDataOrganizer::Reorganize(
    Runtime *server, clio::run::u32 replica_id) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  std::vector<OrganizerBlobStat> stats;
  server->CollectOrganizerBlobStats(replica_id, stats);
  // Page count per tag gives the region geometry. With several replicas each
  // sees a hash-partitioned slice, so scale the slice back up; the ratio is
  // what matters and 4*nz is large.
  std::unordered_map<std::string, clio::run::u64> pages_by_tag;
  for (const OrganizerBlobStat &stat : stats) pages_by_tag[stat.tag_name_]++;
  const clio::run::u64 replicas = server->OrganizerReplicas();
  const clio::run::i32 hint = server->OrganizerHint();
  const Timestamp now = GetCurrentTimeNs();
  for (const OrganizerBlobStat &stat : stats) {
    const clio::run::u64 pages = pages_by_tag[stat.tag_name_] * replicas;
    const float target = TargetScore(stat, now, hint, pages);
    if (target < 0.0f || std::fabs(target - stat.score_) < kEpsilon) continue;
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(server->ReorganizeBlobInternal(stat.tag_id_, stat.blob_name_,
                                                 target, rc));
    if (rc != 0) {
      HLOG(kDebug, "GrayScottDataOrganizer: rescore skipped/failed: tag={} blob={} target={} rc={}",
           stat.tag_name_, stat.blob_name_, target, rc);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::core
