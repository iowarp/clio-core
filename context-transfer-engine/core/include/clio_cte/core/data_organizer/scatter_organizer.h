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

#ifndef WRPCTE_CORE_DATA_ORGANIZER_SCATTER_ORGANIZER_H_
#define WRPCTE_CORE_DATA_ORGANIZER_SCATTER_ORGANIZER_H_

#include <clio_cte/core/data_organizer/data_organizer.h>

#include <string>

namespace clio::cte::core {

/**
 * Data organizer for SCATTER-THEN-GATHER workloads: a grid is written once,
 * each page by exactly one writer, and only afterwards read back. The gmx
 * benchmark is this shape -- PME charge spreading fills a K^3 mesh plane by
 * plane ("the block visits plane z once, accumulates every contribution,
 * publishes, and moves on"), and the force stage then gathers from it.
 *
 * WHY cyclic IS WRONG HERE. CyclicDataOrganizer promotes on sight, which
 * during the spread phase puts pages in the fast tier that are written once
 * and not touched again until gather. That is a migration bought for a page
 * with no reuse ahead of it -- the fast tier fills with finished work while
 * the pages the gather will want are still being produced.
 *
 * THE POLICY: PROMOTE ONLY WHAT HAS BEEN READ, OR ONLY ONCE THE HINT SAYS THE
 * PHASE TURNED. Two signals, either sufficient:
 *
 *   - access_count_ above the write: a page whose reads have begun is in the
 *     gather phase and will be read again, so it is worth the fast tier;
 *   - ReorganizeHint(kPhaseGather): the application announcing the turn, which
 *     lets the organizer promote before the first gather fault rather than
 *     after it. Without a hint the policy still works, one page-read late.
 *
 * MEASURED (RTX 4070 Laptop, single node, 3 reps each, file-backed slow tier:
 * fast tier at score 1.0 + a 4 GB file tier at 0.0, which is the regime where
 * placement can pay at all -- see the note on the RAM-backed regime below):
 *
 *   kmeans  --data-mb 256 --page-kb 256 --blocks 8 --iters 8, 64 MB fast tier
 *     none 1581.2 ms | frecency 1564.7 (-1.0%) | cyclic 1527.8 (-3.4%)
 *     hotset 1530.6 (-3.2%) | scatter 1539.3 (-2.7%)
 *   lbann   --blocks 4 --cap 24 --page-kb 128 --steps 24, 48 MB fast tier
 *     none 29.2 ms/step | frecency 29.2 (+0.1%) | cyclic 28.6 (-2.1%)
 *     hotset 28.4 (-2.9%) | scatter 28.1 (-3.7%); evictions 1540 -> 1416
 *   gmx     --page-kb 128 --atoms 400000 --blocks 8 --repeat 3, 48 MB tier
 *     none 45.0 ms | frecency 44.9 (-0.2%) | cyclic 51.2 (+13.8%)
 *     hotset 51.1 (+13.6%) | scatter 55.5 (+23.3%)   <-- ALL LOSE, see below
 *
 * THE RULE THAT PREDICTS WHICH WAY IT GOES. Filling the fast tier costs one
 * migration per page: (tier_bytes / page_bytes) x ~271 us. gmx's 48 MB tier at
 * 128 KB pages is ~384 migrations ~= 104 ms of work inside a benchmark that
 * runs 45 ms -- the budget exceeds the entire workload, so no hit rate can
 * repay it. kmeans spends ~62 ms of migration against a 1581 ms run (4%) and
 * comes out ahead. Before writing a promoting organizer, compare the cost of
 * filling the tier against the runtime it has to amortize over.
 *
 * AND IT ONLY MATTERS OFF-RAM. With the slow tier in host RAM instead of a
 * file, every arm collapses into noise (kmeans: none 1510.7, frecency 1500.6,
 * cyclic 1495.7, hotset 1497.0 -- all within +-1%). A page fault is ~110 us of
 * round trip against ~6 us of data movement, so tier placement touches about
 * 5% of a fault's cost; only a genuinely slow tier makes that 5% worth buying.
 *
 * FRECENCY IS A NO-OP ON ALL OF THEM (-1.0% to +0.1%), for a structural
 * reason: its scores top out near 0.85 and MaxBwDpe admits a tier only when
 * target_score <= blob_score, so a fast tier at 1.0 is unreachable to it. On a
 * cyclic sweep recency is also an ANTI-signal -- the most recently touched
 * page is the one furthest from its reuse. *
 * Promote-only and budgeted, for the reasons in CyclicDataOrganizer.
 */
class ScatterDataOrganizer : public DataOrganizer {
 public:
  /** Hint value meaning "the scatter phase is over; reads start now". */
  static constexpr clio::run::i32 kPhaseGather = 2;
  static constexpr float kHotScore = 1.0f;
  static constexpr double kFillFactor = 0.9;
  static constexpr float kEpsilon = 0.01f;

  clio::run::TaskResume Reorganize(Runtime *server,
                                   clio::run::u32 replica_id) override;
  std::string GetName() const override { return "scatter"; }

  /**
   * Is this page worth the fast tier yet?
   * @param stat The blob's stats
   * @param gather_phase True once ReorganizeHint announced the gather
   * @return true to promote
   */
  static bool ShouldPromote(const OrganizerBlobStat &stat, bool gather_phase);

 private:
  clio::run::u64 promoted_bytes_ = 0;
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_DATA_ORGANIZER_SCATTER_ORGANIZER_H_
