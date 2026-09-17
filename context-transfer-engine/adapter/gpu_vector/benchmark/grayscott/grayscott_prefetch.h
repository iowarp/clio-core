/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The Gray-Scott prefetcher: a sliding-window tier policy.
 *
 * HOST-ONLY. Nothing here runs on the device; it turns the cursor the kernel
 * publishes (the current z) into CTE bulk rescores, in the driver's gap
 * between rounds. See ../../include/clio_cte/gpu_vector/prefetch.h for what a
 * prefetch is here (a tier migration, not a cache fetch) and why the whole
 * thing is a hint that can never affect correctness.
 *
 * WHAT THE KERNEL ACTUALLY TOUCHES. StepCoro at z holds eight planes:
 * u{z-1,z,z+1}, v{z-1,z,z+1}, unext{z}, vnext{z}. Of those, u/v at z and z+1
 * are re-read at z+1 -- that is the sliding window, and it is why the cache
 * axis on this workload is inert above 8 planes per block. What is NOT
 * captured by any cache size is where those planes live when they are first
 * touched, and that is what this moves.
 *
 * GEOMETRY. One page is exactly one XY plane and every region base is a
 * multiple of `plane`, so page arithmetic is exact:
 *
 *     page(base, z) = base / plane + z
 */
#ifndef CLIO_GV_BENCH_GRAYSCOTT_PREFETCH_H_
#define CLIO_GV_BENCH_GRAYSCOTT_PREFETCH_H_

#include <clio_cte/gpu_vector/prefetch.h>
#include <clio_runtime/types.h>

#include <unordered_map>

namespace clio::gv_bench::grayscott {

namespace gvp = ::clio::cte::gpu_vector;
using ::clio::run::u32;
using ::clio::run::u64;

/**
 * @param plane  elements in one XY plane == elements in one page
 * @param nz     planes in the GLOBAL field (the fixed boundary is global)
 * @param zbase  first plane of THIS NODE's slab
 * @param zend   one past its last
 * @param zper   planes per CUDA block
 *
 * zbase/zend/zper must be the SAME values passed to LaunchStep -- block b's
 * range is derived here exactly as the kernel derives it
 * (z0 = zbase + b*zper, z1 = min(z0+zper, zend)). They are read from the same
 * locals at the same call site, which is what keeps them from drifting; the
 * kernel publishes only its position, not its bounds.
 */
class GrayScottPrefetcher : public gvp::Prefetcher {
 public:
  GrayScottPrefetcher(u64 plane, u64 nz, u64 zbase, u64 zend, u64 zper,
                      const gvp::PrefetchPolicy &policy, bool demote = true)
      : plane_(plane), nz_(nz), zbase_(zbase), zend_(zend), zper_(zper),
        policy_(policy), demote_(demote) {}

  /**
   * The four region bases, in ELEMENTS, for the step about to run.
   *
   * MUST be re-armed every step. The step loop swaps cu/nu and cv/nv, so a
   * page that was "the output plane, written once" becomes "the input plane,
   * read three times" -- and a prefetcher still hinting about last step's
   * assignment promotes exactly the wrong four regions while reporting
   * perfect health.
   *
   * Do NOT pair this with Vector::ResetPrefetchHistory(): the dedup map
   * models the score the CTE holds for a page, which a change of MEANING does
   * not invalidate, and clearing it re-sends every page its own current score.
   */
  void SetRegions(u64 cu, u64 cv, u64 nu, u64 nv) {
    cu_ = cu; cv_ = cv; nu_ = nu; nv_ = nv;
    // The window has to be re-stated against the new regions: the same page
    // numbers now mean different planes. Forgetting the per-block cursor is
    // what makes the next OnYield emit a full window rather than a delta.
    last_z_.clear();
  }

  void OnRunBegin() override { last_z_.clear(); stalls_.clear(); }

  const char *Name() const override { return "grayscott"; }

  void OnYield(const gvp::YieldEvent &ev, gvp::PrefetchSink &out) override {
    // 0 means the kernel published nothing (the seed and sum kernels do not,
    // and neither does any block before its first z). Guessing from silence
    // is how a prefetcher starts evicting live pages.
    if (ev.cursor_ == 0) return;
    // Published as z+1 so that z == 0 is distinguishable from "no cursor".
    const u64 z = ev.cursor_ - 1;
    const u64 z0 = zbase_ + static_cast<u64>(ev.block_) * zper_;
    const u64 z1 = (z0 + zper_ < zend_) ? (z0 + zper_) : zend_;
    if (z < z0 || z >= z1) return;   // a cursor from a previous kernel run

    // ---- THE STALL GATE -------------------------------------------------
    //
    // Count every round this block spent waiting on a transfer, and let a
    // hint through only once X of them have accumulated. See
    // PrefetchPolicy::stall_every_ for why this is selection rather than
    // rate limiting: a stall means the last access was served by a tier the
    // block had to wait for, and those are the only accesses a migration can
    // repay.
    //
    // Counted before the advance check, because the stalls a block suffers
    // while sitting at ONE plane are exactly the evidence that this
    // neighbourhood is slow. A block sailing through resident pages never
    // accumulates a budget and never asks for a migration -- which is the
    // behaviour we want, and it costs nothing to get.
    if (ev.Stalled()) ++stalls_[ev.block_];

    auto it = last_z_.find(ev.block_);
    const bool fresh = (it == last_z_.end());
    const u64 prev = fresh ? z : it->second;
    // ---- EMIT THE DELTA, NOT THE WINDOW ---------------------------------
    //
    // A BLOCK YIELDS MANY TIMES PER Z-ITERATION -- once per page it waits on,
    // about five rounds per z here -- with an unchanged cursor across all of
    // them. Re-stating the whole lookahead window every round costs O(L) hint
    // objects per block per round, all but the first set of which the
    // vector's dedup then discards, in the driver's gap between rounds --
    // host time the next kernel launch is waiting on. At L=74 that was 12.4
    // MILLION hints for 8192 block-z-iterations, and the slowdown tracked L
    // almost exactly: 4860 ms at L=8 rising to 8076 ms at L=148 against a
    // 4263 ms baseline. GENERATING hints was the cost, not migrating pages.
    //
    // So: remember where this block was, and when it advances by d, promote
    // only the d planes that ENTERED the window. Resulting placement is
    // identical; a unit advance costs 4 hints instead of 4L.
    if (!fresh && z == prev) return;         // parked, not advanced: nothing new
    // ADVANCE THE POSITION UNCONDITIONALLY, even when the gate is shut. This
    // is what makes the gate DROP rather than DEFER: planes that slid past
    // while the budget was short are never hinted at all, so the migration
    // count falls by roughly X instead of merely being sent in bursts.
    last_z_[ev.block_] = z;

    if (policy_.stall_every_ != 0) {
      clio::run::u32 &budget = stalls_[ev.block_];
      if (budget < policy_.stall_every_) return;   // gate shut: drop
      budget = 0;
    }

    // First sight of this block (or of these regions) has no window yet, so
    // state it whole; afterwards only the planes that just entered it.
    const u64 lo = fresh ? (z + 1) : (prev + policy_.lookahead_ + 1);
    const u64 hi = z + policy_.lookahead_;
    for (u64 zz = lo; zz <= hi; ++zz) {
      if (zz >= nz_) break;
      out.Hint(PageOf(cu_, zz), policy_.hot_);
      out.Hint(PageOf(cv_, zz), policy_.hot_);
      out.Hint(PageOf(nu_, zz), policy_.hot_);
      out.Hint(PageOf(nv_, zz), policy_.hot_);
    }

    if (!demote_) return;

    // ---- DEMOTE: what has left the window for good ----------------------
    //
    // z-1 is STILL IN THE WINDOW (it is `zm` for the current z, and the
    // guards uzm/vzm hold it). z-2 is the first plane the step will not touch
    // again -- not until the NEXT step's pass reaches it, which is up to
    // `zper` z-iterations away. That distance is the entire justification for
    // demoting it; at zper == 1 there is none and this is a pessimization.
    if (z < z0 + 2) return;
    const u64 zz = z - 2;
    // STRICTLY INSIDE THIS BLOCK'S OWN SLAB. Demoting below z0 would target a
    // NEIGHBOUR's plane, which that block may be reading right now -- the
    // window slides through the slab at a different rate per block, so "I am
    // done with it" is not transitive across a slab boundary.
    if (zz < z0) return;
    out.Hint(PageOf(cu_, zz), policy_.cold_);
    out.Hint(PageOf(cv_, zz), policy_.cold_);
    // OUTPUTS NEVER GO BELOW WARM. StepCoro fetches unext/vnext BEFORE
    // write-holding them, so a plane demoted to storage at the end of step s
    // is a plane faulted from storage at the start of step s+1 -- the regions
    // swap and last step's outputs are this step's inputs. This is the one
    // place where the obvious policy is wrong, and it is wrong in the
    // direction that looks like a win in the fault count and a loss in the
    // wall clock.
    out.Hint(PageOf(nu_, zz), policy_.warm_);
    out.Hint(PageOf(nv_, zz), policy_.warm_);
  }

 private:
  u64 PageOf(u64 base, u64 z) const { return base / plane_ + z; }

  u64 plane_, nz_, zbase_, zend_, zper_;
  gvp::PrefetchPolicy policy_;
  bool demote_;
  u64 cu_ = 0, cv_ = 0, nu_ = 0, nv_ = 0;
  /** Where each block was at its last ADVANCE, so a round in which it merely
   *  parked again emits nothing. See the note in OnYield. */
  std::unordered_map<u32, u64> last_z_;
  /** Stall budget per block, spent one emission at a time. See the stall
   *  gate in OnYield and PrefetchPolicy::stall_every_. */
  std::unordered_map<u32, clio::run::u32> stalls_;
};

}  // namespace clio::gv_bench::grayscott

#endif  // CLIO_GV_BENCH_GRAYSCOTT_PREFETCH_H_
