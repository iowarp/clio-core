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

#ifndef WRPCTE_CORE_DATA_ORGANIZER_CYCLIC_ORGANIZER_H_
#define WRPCTE_CORE_DATA_ORGANIZER_CYCLIC_ORGANIZER_H_

#include <clio_cte/core/data_organizer/data_organizer.h>

#include <string>

namespace clio::cte::core {

/**
 * Data organizer for workloads that sweep a fixed dataset CYCLICALLY: the
 * whole thing is walked in page order, then walked again. kmeans (one Lloyd
 * iteration per pass), weights (the model re-read every pass) and lbann (read
 * then update every step) are all this shape.
 *
 * WHY THE DEFAULT IS PESSIMAL HERE, AND WHY FRECENCY MAKES IT WORSE. Under a
 * cyclic sweep with a cache smaller than the dataset, recency is an
 * ANTI-signal: the page touched most recently is the one that will not be
 * touched again until the whole rest of the dataset has gone by, and the page
 * about to be needed is the coldest one there is. An LRU-shaped policy
 * therefore evicts every page just before its reuse and converges on a 0% hit
 * rate -- the classic sequential-flooding result. FrecencyDataOrganizer
 * scores exactly that way (recency blended with frequency), so on this access
 * pattern it actively promotes the wrong pages.
 *
 * THE POLICY: PIN A STABLE PREFIX, AND ONLY EVER PROMOTE. The fastest tier
 * holds C bytes; the dataset needs far more. Any FIXED subset of C bytes
 * gives a hit rate of C/dataset every pass, which is the best a
 * capacity-bound cyclic sweep can do without lookahead. So this organizer
 * picks the subset by PAGE NUMBER -- pages [0, C/page_size) of each tag -- and
 * raises them to kHotScore.
 *
 * Two properties follow from choosing the set by page number rather than by
 * access:
 *
 *   1. It is STABLE. The same pages are chosen every round, so they migrate
 *      once and every later round finds them already at kHotScore and skips
 *      them. That is what makes the migration cost amortize instead of
 *      recurring -- the failure mode GrayScottDataOrganizer measured.
 *   2. It is PROMOTE-ONLY. Pages outside the set are left alone at their put
 *      score, never demoted. A demotion is a page read plus a page write for
 *      data that is about to be read anyway; the eviction path reclaims those
 *      frames for free when the fast tier fills.
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
 * It also fills a tier the workload cannot otherwise reach. The gpu_vector
 * writes pages at kVectorBlobScore = 0.5 and MaxBwDpe admits a tier only when
 * target_score <= blob_score, so with the benches' usual HBM 1.0 / host RAM
 * 0.2 configuration no live page EVER lands in HBM -- the benchmarks report
 * this as "nothing landed in the fastest tier". Promotion is what makes the
 * fast tier reachable at all.
 */
class CyclicDataOrganizer : public DataOrganizer {
 public:
  /** Score that makes the fastest tier reachable (>= its target score). */
  static constexpr float kHotScore = 1.0f;
  /** Fraction of the fast tier to claim; the rest is headroom for churn. */
  static constexpr double kFillFactor = 0.9;
  /** Below this delta a rescore is a no-op to the CTE and only costs a task. */
  static constexpr float kEpsilon = 0.01f;

  clio::run::TaskResume Reorganize(Runtime *server,
                                   clio::run::u32 replica_id) override;
  std::string GetName() const override { return "cyclic"; }

  /**
   * Decode the page number a gpu_vector blob is named with.
   * @param blob_name Raw little-endian u32, as DeviceVector writes it
   * @param out Page number on success
   * @return false if the name is not a 4-byte raw int (not a vector page)
   */
  static bool PageOf(const std::string &blob_name, clio::run::u64 &out);

  /**
   * How many leading pages of one tag fit in that tag's share of the fast
   * tier.
   * @param fast_tier_bytes Capacity of the fastest tier
   * @param page_bytes Size of one page (from a blob's stat)
   * @param num_tags Tags sharing the tier; the budget is split evenly
   * @return page count, 0 when nothing fits
   */
  static clio::run::u64 HotPageLimit(clio::run::u64 fast_tier_bytes,
                                     clio::run::u64 page_bytes,
                                     clio::run::u64 num_tags);

 private:
  /** Bytes this container has already promoted. Carried ACROSS rounds so the
   *  pinned set stays stable and the fast tier is not over-subscribed. */
  clio::run::u64 promoted_bytes_ = 0;
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_DATA_ORGANIZER_CYCLIC_ORGANIZER_H_
