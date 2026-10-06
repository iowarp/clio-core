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

#ifndef WRPCTE_CORE_DATA_ORGANIZER_HOTSET_ORGANIZER_H_
#define WRPCTE_CORE_DATA_ORGANIZER_HOTSET_ORGANIZER_H_

#include <clio_cte/core/data_organizer/data_organizer.h>

#include <string>

namespace clio::cte::core {

/**
 * Data organizer for workloads whose vectors have DIFFERENT reuse: some
 * structure is re-read on every pass while bulk data streams past once.
 * lammps_md is the case -- the Verlet neighbour list is rebuilt rarely and
 * read by every force pass, while the atom position and velocity vectors are
 * swept once per step -- and lbann's biases behave the same way against its
 * weight matrices.
 *
 * WHY NOT cyclic. CyclicDataOrganizer promotes whatever it sees first until
 * the tier is full, which on a mixed workload fills the fast tier with
 * whichever vector happened to be written first. Reuse, not arrival order, is
 * what should win the tier.
 *
 * THE POLICY: RANK BY OBSERVED REUSE, KEEP THE TOP. access_count_ counts
 * put+get data ops per blob, so reads-per-byte is a direct measure of how
 * much a page repays residency. Each round the organizer sorts the visible
 * pages by access_count_ descending and promotes down that list until the
 * fast tier's budget is spent. A page that keeps being read stays promoted
 * (it is already hot, so it costs nothing to re-affirm); one that goes quiet
 * is simply not re-promoted and ages out through ordinary eviction.
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
 * Promote-only, for the reason GrayScottDataOrganizer measured: demoting by
 * rescore is a migration, and the eviction path reclaims the frame for less.
 */
class HotSetDataOrganizer : public DataOrganizer {
 public:
  static constexpr float kHotScore = 1.0f;
  static constexpr double kFillFactor = 0.9;
  static constexpr float kEpsilon = 0.01f;

  clio::run::TaskResume Reorganize(Runtime *server,
                                   clio::run::u32 replica_id) override;
  std::string GetName() const override { return "hotset"; }
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_DATA_ORGANIZER_HOTSET_ORGANIZER_H_
