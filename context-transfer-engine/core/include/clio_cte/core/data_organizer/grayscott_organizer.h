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
#ifndef WRPCTE_CORE_DATA_ORGANIZER_GRAYSCOTT_ORGANIZER_H_
#define WRPCTE_CORE_DATA_ORGANIZER_GRAYSCOTT_ORGANIZER_H_

#include <clio_cte/core/data_organizer/data_organizer.h>

#include <string>
#include <vector>

namespace clio::cte::core {

/**
 * Data organizer for the Gray-Scott paged stencil (and any workload shaped
 * like it: a fixed set of regions swept in a known order, plus write-once
 * checkpoints).
 *
 * WHAT THE HOST CAN SEE THAT THE DEVICE PREFETCHER CANNOT. The bench's own
 * `--prefetch gs` already places pages Belady-optimally INSIDE a step: the
 * sweep order is known, so it promotes the next window per block and demotes
 * behind it. A periodic host organizer cannot compete with that and does not
 * try. What it can do is act on things the stats and the phase hint expose
 * across steps and phases:
 *
 *   1. CHECKPOINTS ARE DEAD AFTER WRITE. Every `--ckpt-every` steps the bench
 *      materialises a whole new tag (gv_gs_ck<k>) whose pages land at blob
 *      score 1.0 and are never read again -- cold history occupying the
 *      fastest tier. Every blob in a tag named with the checkpoint prefix is
 *      rescored to kColdScore so it sinks to the slowest tier.
 *
 *      NOT "written but never read". A first version also demoted any blob
 *      whose last_read_ stayed 0 for a grace period, on the theory that it
 *      catches checkpoints under any name. It does not work for paged data:
 *      the GPU fault path (PodMultiGetBlob) never stamps last_read_, so every
 *      live vector page looks never-read forever and the whole dataset was
 *      rescored to cold every round. Measured cost with no checkpoints at
 *      all: +9.9% wall time (1142 -> 1254 ms) from the rescore tasks alone.
 *      Recognise checkpoints by name until the fault path stamps reads.
 *
 *   2. THE DEAD PAIR AT A SWAP (hint-driven, off without a hint). Step s reads
 *      one region pair and writes the other; after the swap the pair about to
 *      be overwritten holds bytes nobody will read. With ReorganizeHint(s+1)
 *      from the host, the pair being written this step is rescored to
 *      kWarmScore so it stops occupying the fast tier ahead of the writes.
 *      A demotion is itself a migration, so this is expected to be at best
 *      neutral and is measured, not assumed -- see the numbers below.
 *
 * MEASURED (RTX 4070 Laptop, single node, 16 cores, runtime num_threads: 8,
 * v_checksum identical in every run of every arm below).
 *
 * THE DECK DECIDES WHETHER THIS POLICY CAN PAY AT ALL, and two earlier
 * sweeps got it wrong in opposite directions. What the policy needs is DRAM
 * sized for the live set PLUS EXACTLY ONE CHECKPOINT, so that checkpoint k
 * squats in the fast tier through the following compute window and
 * checkpoint k+1 would have to evict the LIVE grid to land. Then demoting
 * gv_gs_ck* spills k to disk while the GPU computes and k+1 lands free.
 *   - Too small (512 MB live, 768 MB DRAM): no room for a checkpoint at all,
 *     so eviction already spills them synchronously and rule 1 has nothing
 *     to do. Measured +2.9%: pure overhead.
 *   - Too large: nothing competes, same result.
 * The GPU page cache must also not be the bottleneck. A plane is exactly one
 * 256 KB page and the stencil holds 6 in + 2 out per block, so --slots must
 * clear blocks*8 or the run is ~97% GPU fault round trips (at --slots 12,
 * 50k faults x ~97 us) and every tier effect is buried. At --slots 512 the
 * faults are cold-only (1024, evicts 0) and checkpoint stall rises to 38% of
 * runtime -- which is the headroom this organizer converts.
 *
 * Deck: --data-mb 512 --page-kb 256 --blocks 8 --steps 24 --ckpt-every 8
 * --slots 512 --ram-mb 1088 (=512 live + 512 one checkpoint + 64) @0.2,
 * HBM 8 MB @1.0, NVMe file 8 GB @0.0, NO --ckpt-drain (the application must
 * not drain; that is what is under test). 10 reps, sd in parentheses:
 *
 *     organizer none            4622 ms (1.1%)   ckpt 1745 ms  DRAM 1088/1088
 *     grayscott, rule 1 only    4445 ms (4.1%)   ckpt 1537 ms  DRAM  967   -3.8%
 *     grayscott, rules 1+2      4424 ms (1.6%)   ckpt 1511 ms  DRAM  987   -4.3%
 *     frecency                  4573 ms (3.8%)   ckpt 1752 ms  DRAM 1088   -0.4%
 *
 * VERDICT: rule 1 pays, and it is the whole effect. -4.3% total (t ~ 7 at
 * n=10) by cutting the checkpoint stall 1745 -> 1511 ms and holding DRAM ~100
 * MiB below its cap instead of pinned at it. frecency on the same deck is
 * -0.4% (t ~ 0.3) and leaves DRAM at 1088 and ckpt_ms unchanged -- it never
 * recognises the checkpoint tag, which is the control: the win is this
 * policy, not organizer machinery in general.
 *
 * BE GENTLE. organizer_tasks: 1 / organizer_period_ms: 200. The spill is not
 * limited by the length of the compute window -- it is limited by worker
 * threads, which the migration coroutines share with the fault path. Raising
 * the rate spills strictly more (stor 952 -> 1024 -> 1092 -> 1197 -> 1348 MiB)
 * and costs strictly more (+1.0%, +4.9%, +7.0% at t1p50, t4p50, t8p25),
 * because the extra faults it induces outweigh the extra stall it saves.
 *
 * RULE 2 IS WORTH NOTHING, now measured honestly. Its contribution above is
 * 4445 -> 4424 ms, 0.45% at t ~ 0.3. Every earlier number for rule 2 was
 * taken with PageOf() decoding page names as a raw u32, which on a 2048-page
 * tag matched only pages 1000-2047 and decoded each to garbage -- so rule 2
 * never saw a real page number and "never helped" was not evidence of
 * anything. With the decode fixed it still does not help, for the reason the
 * migration economics predict: each demoted page costs a read plus a write
 * and is overwritten milliseconds later regardless of which tier it is on.
 * Rule 1 does not use PageOf and was never affected.
 *
 * STILL TRUE, AND STILL THE BIGGER PRIZE: dead data should be DROPPED, not
 * moved. Rule 1 wins while paying a full writeback for bytes nobody will ever
 * read. An organizer verb that releases a blob's fast-tier blocks without
 * migrating them would take the same stall to zero instead of to 1511 ms.
 * The interface only has "rescore", which always preserves the bytes.
 *
 * REGION GEOMETRY IS DERIVED, NOT CONFIGURED. DeviceVector submits page names
 * as a raw little-endian u32 (kBlobNameRawInt32), but the CTE runs
 * NormalizeBlobName() on the way in, which renders the name DECIMAL in place
 * and clears the flag -- so what reaches an organizer is decimal text. See
 * PageOf(). A vector with four equal regions has 4*nz pages, so
 * nz = pages_in_tag / 4 and region = page / nz. Tags whose page count is not
 * a multiple of four are left alone by rule 2.
 *
 * SCORE VOCABULARY follows PrefetchPolicy and the bench's tier config:
 * HBM 1.0, host RAM 0.2, storage 0.0. Higher score = faster tier.
 */
class GrayScottDataOrganizer : public DataOrganizer {
 public:
  /** Tag-name prefix that marks a checkpoint tag (Vector::Copy target). */
  static constexpr const char *kCheckpointPrefix = "gv_gs_ck";
  /** Score for data nobody will read again: eligible only for the slowest tier. */
  static constexpr float kColdScore = 0.0f;
  /** Score for a region about to be overwritten: out of the fast tier, not to storage. */
  static constexpr float kWarmScore = 0.2f;
  /** Below this delta a rescore is a no-op to the CTE and only costs a task. */
  static constexpr float kEpsilon = 0.01f;
  /** Regions per vector for the pair-swap rule (u, v, u_next, v_next). */
  static constexpr clio::run::u32 kRegions = 4;

  clio::run::TaskResume Reorganize(Runtime *server,
                                   clio::run::u32 replica_id) override;
  std::string GetName() const override { return "grayscott"; }

  /**
   * Decode the page number a vector blob is named with.
   * @param blob_name Raw little-endian u32, as written by DeviceVector
   * @param out Page number on success
   * @return false if the name is not a 4-byte raw int
   */
  static bool PageOf(const std::string &blob_name, clio::run::u64 &out);

  /**
   * Target score for one blob under this policy, or a negative value for
   * "leave it alone".
   * @param stat The blob's collected stats
   * @param now Current steady-clock ns
   * @param hint The phase hint (0 = none); step index + 1 when present
   * @param pages_in_tag Blob count of the blob's tag, for region geometry
   */
  static float TargetScore(const OrganizerBlobStat &stat, Timestamp now,
                           clio::run::i32 hint, clio::run::u64 pages_in_tag);
};

}  // namespace clio::cte::core

#endif  // WRPCTE_CORE_DATA_ORGANIZER_GRAYSCOTT_ORGANIZER_H_
