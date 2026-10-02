/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * Prefetching for gpu_vector: workload-specific TIER hints, driven by the
 * yield state, delivered through the CTE's bulk rescore API.
 *
 * WHAT THIS IS NOT. It does not pull pages into the GPU page cache. The
 * vector already has that verb -- BeginFetch/AwaitFetch -- and only the
 * kernel can use it correctly, because only the kernel knows which frames it
 * still holds pins on. A host agent claiming frames behind a running kernel's
 * back would evict pages a block is still reading.
 *
 * WHAT IT IS. It moves a page UP THE TIER STACK before the demand fetch asks
 * for it. The fault still happens, still block-collectively, still under the
 * kernel's control -- it is simply served from HBM instead of from NVMe.
 * Nothing about frames, pins or eviction changes; what changes is which bdev
 * the CTE reads when the fault arrives.
 *
 * That is a blob score, and the CTE has a batched API for it:
 * PodMultiScoreTask -> PodMultiScore -> PodReorganizeBlob per record ->
 * ReorganizeBlobInternal, which re-runs the DPE and physically moves the
 * blob's blocks when the score moves by more than the configured
 * score_difference_threshold. So a prefetch IS a bulk rescore, and a
 * prefetcher is a function from "where is this block in its computation" to
 * "which pages should be hot, and which should stop being hot".
 *
 * A HINT IS ONLY EVER A HINT. A wrong one costs a wasted tier migration. It
 * can never cost a wrong answer, because the demand fetch remains the only
 * thing that makes a page resident. That property is what makes it safe to
 * hand this API to a workload author, and it must survive every change here.
 */
#ifndef CLIO_CTE_GPU_VECTOR_PREFETCH_H_
#define CLIO_CTE_GPU_VECTOR_PREFETCH_H_

#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace clio::cte::gpu_vector {

/**
 * The three-value score vocabulary a prefetcher speaks.
 *
 * Scores are clamped to [0,1] by ReorganizeBlobInternal, and MaxBwDpe splits
 * tiers on target_score <= blob_score, ranking the preferred group DESCENDING
 * -- so HIGHER SCORE MEANS FASTER TIER. That is the reverse of the GNN
 * trainer's hierarchy, and getting it backwards produces a prefetcher that
 * demotes everything it means to promote while reporting perfect health.
 *
 * Defaults match the Gray-Scott tier config (HBM 1.0, host RAM 0.2, NVMe
 * 0.0). They are constructor arguments because a run with no storage tier
 * wants cold == warm: there is nowhere lower to go, and a rescore that moves
 * no bytes still costs a task.
 */
struct PrefetchPolicy {
  float hot_ = 1.0f;    // read within the lookahead window -> fastest tier
  float warm_ = 0.2f;   // out of the window, read again soon -> host RAM
  float cold_ = 0.0f;   // done with for a long time -> let it spill
  /** How far ahead of the cursor to promote, in whatever unit the workload's
   *  cursor counts. Small on purpose: a migration has to COMPLETE inside the
   *  window to be worth anything, and everything promoted is occupying the
   *  fast tier until it is used. */
  clio::run::u64 lookahead_ = 4;
  /** Mirror of the CTE's own score_difference_threshold, used to drop
   *  no-op hints before they cost a task submission. Conservative: a value
   *  smaller than the CTE's only wastes submissions, a larger one silently
   *  swallows migrations the CTE would have performed. */
  float epsilon_ = 0.01f;
  /**
   * EMIT ONLY ONCE PER X OBSERVED STALLS. 0 disables the gate (emit whenever
   * the position advances, the unrated behaviour).
   *
   * A migration costs a full page read plus a full page write (~271 us for
   * 256 KB) and buys, at most, the difference between one slow-tier fault and
   * one fast-tier fault (~46 us measured). So a prefetcher that promotes
   * every plane in its window spends that cost on EVERY plane while only the
   * ones that were actually on a slow tier can repay it -- at a 30/30/40
   * split, 60% of promotions are no-ops in value terms and full price in
   * cost.
   *
   * A stall is the signal that separates them: a block parked on an
   * outstanding transfer waited for a tier, and the pages around it are the
   * ones worth moving. Gating on stalls is therefore SELECTION, not merely
   * rate limiting -- it spends the migration budget where a migration can pay.
   *
   * THE GATE MUST DROP, NOT DEFER. With an incremental emitter the number of
   * planes promoted per step is fixed by how far the cursor travels, so
   * batching emissions changes only WHEN they are sent, not HOW MANY. A
   * closed gate therefore lets the window slide past unhinted, and those
   * planes are never promoted at all. That is the whole point: 1/X of the
   * migrations, chosen by where the stalls were.
   */
  clio::run::u32 stall_every_ = 0;
};

/** One hint: "page `page_` should be scored `score_`." */
struct PrefetchHint {
  clio::run::u64 page_ = 0;
  float score_ = 0.0f;
};

/**
 * What a prefetcher writes into.
 *
 * BOUNDED, so a prefetcher with an arithmetic bug cannot flood the CTE with
 * rescores from inside the driver's round gap -- which is host time the next
 * kernel launch is waiting on. Overflow is counted and dropped, never
 * silently truncated: a prefetcher that is being clipped is misconfigured,
 * and the count is how anyone finds out.
 */
class PrefetchSink {
 public:
  static constexpr clio::run::u32 kMaxHintsPerRound = 4096;

  void Hint(clio::run::u64 page, float score) {
    if (hints_.size() >= kMaxHintsPerRound) {
      ++dropped_;
      return;
    }
    hints_.push_back(PrefetchHint{page, score});
  }

  /** Half-open page range [pg_lo, pg_hi). */
  void HintRange(clio::run::u64 pg_lo, clio::run::u64 pg_hi, float score) {
    for (clio::run::u64 p = pg_lo; p < pg_hi; ++p) Hint(p, score);
  }

  const std::vector<PrefetchHint> &hints() const { return hints_; }
  clio::run::u32 dropped() const { return dropped_; }
  void Clear() {
    hints_.clear();
    dropped_ = 0;
  }

 private:
  std::vector<PrefetchHint> hints_;
  clio::run::u32 dropped_ = 0;
};

/**
 * One block's yield state, as a prefetcher sees it.
 *
 * A flattened copy of YieldBlockState plus the block id and round, so a
 * prefetcher never needs the runtime's types or the driver itself.
 */
struct YieldEvent {
  /** LOGICAL block id -- what the kernel calls yv.Block(), never blockIdx.x. */
  clio::run::u32 block_ = 0;
  /** Driver round, monotone within one kernel run, 1 for the first. */
  clio::run::u32 round_ = 0;
  /** kYieldSuspended or kYieldDone. A DONE block is as interesting as a
   *  suspended one: it is exactly when the pages behind it become demotable. */
  clio::run::u32 status_ = 0;
  /** Macro mechanism: the __LINE__ of the yield. Coroutines: always 0. */
  clio::run::u32 resume_point_ = 0;
  /** The completion word this block parked on; 0 means "no token". */
  clio::run::u64 wait_tag_ = 0;
  /** What the kernel published with YieldPublishCursor. 0 means the workload
   *  said nothing, and a prefetcher must then emit nothing. */
  clio::run::u64 cursor_ = 0;
  clio::run::u64 cursor_aux_ = 0;

  bool Suspended() const {
    return status_ == clio::run::gpu::kYieldSuspended;
  }

  /**
   * THIS BLOCK IS WAITING ON A TRANSFER -- the stall signal.
   *
   * Suspended with a non-zero wait tag means the block parked on a completion
   * word that has not been written: it issued a fetch and the bytes are not
   * back. That is precisely "this access is being served by a tier, and the
   * block is idle until it finishes". A suspend with NO tag is a block that
   * yielded for some other reason and will be relaunched freely, which is not
   * a stall and must not be counted as one.
   */
  bool Stalled() const { return Suspended() && wait_tag_ != 0; }
};

/**
 * A prefetcher. Register one (or several) on a Vector; the vector fires them
 * from the driver's post-round observer.
 *
 * Called on the HOST, between rounds, with no kernel resident. Must not
 * block: anything slow here is on the next launch's critical path, which is
 * precisely what prefetching exists to keep things off.
 */
class Prefetcher {
 public:
  virtual ~Prefetcher() = default;

  /** Once per block per round. */
  virtual void OnYield(const YieldEvent &ev, PrefetchSink &out) = 0;

  /** Once per round, after every block's OnYield -- for prefetchers whose
   *  decision needs the whole grid's cursors rather than one block's. */
  virtual void OnRoundEnd(clio::run::u32 round, PrefetchSink &out) {
    (void) round;
    (void) out;
  }

  /** Reset any per-run history. Called when a kernel run begins. */
  virtual void OnRunBegin() {}

  /** For the report line. A run whose prefetcher is unnamed is a run nobody
   *  can reconstruct from its log. */
  virtual const char *Name() const = 0;
};

/** What the rescore engine actually did. The tiered evaluation reads these:
 *  timings on these workloads move tens of percent run to run, so a
 *  prefetcher that works and one that no-ops are indistinguishable in the
 *  wall clock and trivially distinguishable here. */
struct PrefetchStats {
  clio::run::u64 hints_ = 0;       // emitted by prefetchers
  clio::run::u64 deduped_ = 0;     // dropped: same score already sent
  clio::run::u64 dropped_ = 0;     // dropped: sink overflow (a bug signal)
  clio::run::u64 sent_ = 0;        // records actually submitted
  clio::run::u64 batches_ = 0;     // PodMultiScore tasks submitted
  /** Records the CTE refused for a reason that is NOT "no such blob". A
   *  nonzero value here is a real problem; not_found_ is not. */
  clio::run::u64 errors_ = 0;
  /** Records naming a page nobody has written yet. EXPECTED, not a fault:
   *  the lookahead necessarily runs past the written frontier, and a
   *  workload whose regions are written in phases (Gray-Scott's outputs do
   *  not exist until the first step flushes them) hints at them before they
   *  are there. Counted separately so a healthy run does not read as broken. */
  clio::run::u64 not_found_ = 0;
  clio::run::u64 promotions_ = 0;  // sent records that raised a page's score
  clio::run::u64 demotions_ = 0;   // sent records that lowered it
};

/**
 * THE GENERIC CONTROL, and the reason it exists.
 *
 * A workload-specific prefetcher that beats no-prefetch proves that
 * prefetching helps. It does NOT prove that workload-SPECIFIC prefetching
 * helps; for that the comparison has to be against a prefetcher that knows
 * nothing about the workload. This is that prefetcher.
 *
 * Per block it keeps the last two cursors, takes the difference as a stride,
 * and promotes cursor + k*stride for k in [1, lookahead] within ONE caller-
 * named region. It cannot know about a second region, and it cannot know that
 * anything has become cold -- those two gaps are exactly what a
 * workload-specific prefetcher is being measured on.
 *
 * A stride of 0 or a reversal emits nothing: two samples is not enough to
 * call a direction, and a wrong promotion under tier pressure evicts
 * something real.
 */
class StridePrefetcher : public Prefetcher {
 public:
  /**
   * @param page_base   page index the cursor's zero maps to
   * @param page_count  pages in the region; hints are clipped to it
   * @param policy      score vocabulary and lookahead
   */
  StridePrefetcher(clio::run::u64 page_base, clio::run::u64 page_count,
                   const PrefetchPolicy &policy)
      : page_base_(page_base), page_count_(page_count), policy_(policy) {}

  void OnRunBegin() override { last_.clear(); }

  void OnYield(const YieldEvent &ev, PrefetchSink &out) override {
    if (ev.cursor_ == 0 && !ev.Suspended()) return;
    auto it = last_.find(ev.block_);
    const clio::run::u64 prev = (it == last_.end()) ? ev.cursor_ : it->second;
    last_[ev.block_] = ev.cursor_;
    // ONE SAMPLE IS NOT A STRIDE. The first round of a run, and any round in
    // which a block yielded without advancing (which is most of them -- a
    // block parks several times inside one z-iteration), carries no new
    // direction and must not be guessed at.
    if (ev.cursor_ <= prev) return;
    const clio::run::u64 stride = ev.cursor_ - prev;
    for (clio::run::u64 k = 1; k <= policy_.lookahead_; ++k) {
      const clio::run::u64 pos = ev.cursor_ + k * stride;
      if (pos >= page_count_) break;
      out.Hint(page_base_ + pos, policy_.hot_);
    }
  }

  const char *Name() const override { return "stride"; }

 private:
  clio::run::u64 page_base_;
  clio::run::u64 page_count_;
  PrefetchPolicy policy_;
  std::unordered_map<clio::run::u32, clio::run::u64> last_;
};

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_PREFETCH_H_
