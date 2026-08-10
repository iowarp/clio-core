/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The device half of a gpu_vector.
 *
 * This object is passed to a kernel BY VALUE, so every thread gets its own
 * copy. That is what makes `last_page_` a per-thread pointer with no
 * __shared__ storage and no cross-thread coordination: each thread caches
 * the page it last touched, and when the page size is chosen so a warp's
 * accesses land in one page, all 32 lanes hold the identical pointer and
 * the common case costs one compare.
 *
 * Everything else lives in device global memory: the page table (one
 * contiguous array, partitioned per block) and the pages' bytes.
 */
#ifndef CLIO_CTE_GPU_VECTOR_DEVICE_VECTOR_H_
#define CLIO_CTE_GPU_VECTOR_DEVICE_VECTOR_H_

#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/page.h>

namespace clio::cte::gpu_vector {

/**
 * Device-side view of the vector. Constructed on the host by
 * Vector::GetDevice() and copied into the kernel as a plain argument.
 */
/**
 * Everything about a vector that is the SAME for every thread, kept in
 * GLOBAL memory next to the page table rather than inside the view.
 *
 * The view used to carry all of this by value. That is ~200 bytes, and it
 * only stays in constant memory while nothing writes to it -- the moment a
 * kernel assigns ipc_, or HoldPage updates last_page_, the compiler must give
 * every thread a private copy in LOCAL memory. Measured on the llama mat-vec
 * kernel: 1024 bytes of stack per thread and 64 registers, which pinned the
 * launch to one block per SM, and a single spill store (STL.64) carried 50%
 * of all warp stall samples.
 *
 * Exactly one field here is ever read on the hot path (the geometry); the
 * submit-only fields and the statistics counters are touched on a fault,
 * which already costs a round trip. So they cost nothing behind a pointer.
 */
struct VecHeader {
  // ---- layout, filled in by the host ---------------------------------
  clio::cte::core::TagId tag_id_;
  /** Page table for ALL blocks: nblocks_ * pages_per_block_ entries. */
  Page *pages_ = nullptr;
  clio::run::u32 nblocks_ = 0;
  clio::run::u32 pages_per_block_ = 0;
  /** Bytes per page, and the element count that implies. */
  clio::run::u64 page_bytes_ = 0;
  clio::run::u64 elems_per_page_ = 0;
  /** log2(elems_per_page_) and its mask; 0 shift means "not a power of two,
   *  fall back to division". See PageOf. */
  clio::run::u32 page_shift_ = 0;
  clio::run::u64 page_mask_ = 0;
  /** Logical element count of the whole vector. */
  clio::run::u64 size_ = 0;
  /** Pool the page tasks are addressed to (the CTE core). */
  clio::run::PoolId pool_id_;
  /** Allocator of the backend holding the task slots. */
  ctp::ipc::AllocatorId task_alloc_id_;
  /** Codec wire id stamped on page PUTs. 0 stores raw. The compressor pool
   *  reads this from the task context to decide whether to compress. */
  int compress_lib_ = 0;
  int compress_preset_ = 1;  // 1 FAST -- see gpu_vector.h
  /**
   * Device-global counter handing out strictly increasing task ids.
   *
   * CreateTaskId()'s device build is a stub returning one constant for every
   * call, so identity has to come from somewhere. Deriving it from (slot,
   * kind) alone was not enough -- every device task also carries pid_=tid_=0,
   * so ids must be distinct in the fields that actually vary.
   */
  unsigned long long *task_seq_ = nullptr;
  /**
   * Optional instrumentation, null unless Vector::EnableStats() is called.
   *
   * Final values alone cannot tell a correct pager from a broken one: a cache
   * that never evicts, one that re-faults every access, and one that honours
   * scores all return the SAME bytes. These counters make the policy itself
   * assertable -- a test can demand an exact number of faults for a known walk
   * and so pin down hit/miss behaviour, writeback of dirty pages, and which
   * page eviction actually chose.
   */
  /**
   * One spin lock per block, guarding that block's page table.
   *
   * Without it the fault path is only safe when a single lane runs it. With
   * all 32 lanes of a warp missing the same page, every lane independently
   * found the same free slot, claimed it, and fired its own get into the same
   * page's single task slot -- measured 7037 wrong elements out of 8192. The
   * lock makes the first lane in do the fault while the rest wait and then
   * find the page resident, which is the behaviour the "a warp sees one page"
   * contract always implied but never enforced.
   */
  /**
   * Batched writeback state, one MultiPutBatch array per block.
   *
   * A page flush costs one task SUBMISSION per page, and on the device path
   * that submission is the expense -- a context entry is a fixed cost whatever
   * rides on it. `multi_per_block_` is sized so a block's ENTIRE page cache
   * fits in that many batches (ceil(pages_per_block_ / kPodMultiMax)), which
   * is the case worth optimizing: 256 pages flush in 4 submissions, not 256.
   *
   * Null / zero when the host did not provision them, in which case
   * FlushBlockBatched falls back to the scalar path.
   */
  MultiBatch *multi_ = nullptr;
  clio::run::u32 multi_per_block_ = 0;
  int *block_locks_ = nullptr;
  unsigned long long *stat_faults_ = nullptr;   // page-ins  (SubmitGet)
  unsigned long long *stat_puts_ = nullptr;     // writebacks (SubmitPut)
  unsigned long long *stat_evicts_ = nullptr;   // slots reclaimed
  unsigned long long *stat_prefetches_ = nullptr;    // BeginFetch gets issued
  /** Faults that found the page already in flight -- prefetch paid off. */
  unsigned long long *stat_prefetch_hits_ = nullptr;
  /** Faults whose prefetch was issued but had NOT landed yet. */
  unsigned long long *stat_prefetch_late_ = nullptr;
  unsigned long long *stat_rescores_ = nullptr;      // reorganize hints sent
  /** Gets that came back with a NON-ZERO return code. SubmitGet never checked
   *  this, so a failed read silently left the slot's previous page in place. */
  unsigned long long *stat_get_errors_ = nullptr;
  unsigned long long *stat_pf_dropped_ = nullptr;  // async hints dropped
  // Optional per-page fault counter (CLIO_FAULT_HIST): distinguishes churn
  // (same pages refaulted) from coverage (more unique pages) between the
  // bistable attractors. Device array of one u32 per vector page.
  unsigned int *fault_hist_ = nullptr;
  // CLIO_MOE_VERIFY in-vivo counters: after computing an item the kernel
  // re-probes its page; lost = the held pointer no longer resolves to the
  // same slot (evicted/recycled mid-read).
  unsigned long long *stat_verify_ok_ = nullptr;
  unsigned long long *stat_verify_lost_ = nullptr;

};

template <typename T>
class DeviceVector {
 public:
  DeviceVector() = default;

  /** Everything shared: geometry, submit state, counters. */
  const VecHeader *h_ = nullptr;

#if CTP_IS_GPU_COMPILER
  // ---- device API -----------------------------------------------------

  /**
   * Element access. Resolves the page (faulting it in if needed), records
   * the access, and returns a reference into the page's bytes.
   *
   * The non-const form marks the page dirty: a reference handed out for
   * writing cannot be observed later, so the write must be assumed.
   */
  /**
   * Pin the page holding `off` and report how far you can walk from there.
   *
   * This is the indexing fast path. It resolves the page ONCE -- faulting it in
   * if needed -- caches it in last_page_, and returns how many elements can be
   * read or written sequentially from `off` before leaving that page. The
   * caller loops over that run using operator[]/at(), which then do no
   * resolution at all.
   *
   * Why this exists: resolving per element cost 64x a raw pointer on resident
   * data. About a fifth of that was the unavoidable bookkeeping (a local-memory
   * load of last_page_, then two dependent loads to reach the bytes); the rest
   * was the page-TRANSITION path, which takes the block lock -- atomicCAS plus
   * two __threadfence()s -- and linearly scans the page table, and which 256
   * threads hit simultaneously and serialise on. Hoisting the transition out of
   * the inner loop removes that entirely: one resolution per page instead of
   * one per element.
   *
   * @param off   first element to be accessed
   * @param count how many the caller would like to walk
   * @return elements accessible from `off` without another HoldPage; always
   *         >= 1 and <= count. Loop that many, then call again.
   */
  CTP_GPU_FUN clio::run::u64 HoldPage(clio::run::u64 off,
                                      clio::run::u64 count) {
    const clio::run::u64 pn = PageOf(off);
    Page *p = last_page_;
    if (p == nullptr || p->page_num != pn || p->fetching) {
      // Look the page up WITHOUT the block lock. Taking it here is what made
      // the fast path slower than demand faulting: at kernel entry every
      // thread misses, so a whole block serialises through atomicCAS and two
      // __threadfence()s each, once per vector per launch. A resident page
      // needs no exclusive access -- only claiming a slot or evicting does.
      //
      // The scan is safe against a concurrent fault because a slot publishes
      // `fetching` BEFORE its page_num (see ClaimSlot), so a page that looks
      // resident here really is: the two are ordered by a threadfence, and a
      // scanner that sees the new page_num also sees fetching set and falls
      // through to the locked path.
      p = nullptr;
      Page *tbl = BlockPages();
      // Rotated scan: start at the LAST HIT slot and wrap. Worst case visits
      // the same slots; the sequential-walk case hits on the first probe or
      // two, because FetchPagesBatchedLocked claims first-free slots in order
      // so consecutive pages sit in consecutive slots. Starting at 0 instead
      // measured as a ~115 us floor on every launch (each warp's first hold
      // scanning ~400 occupied slots), invariant to every inner-loop change
      // because it runs before any of them.
      const clio::run::u32 ppb = h_->pages_per_block_;
      // OPEN-ADDRESSED LOOKUP. Pages are placed at pn % ppb, or failing that
      // at the next free slot linearly (see the claim sites), so a lookup
      // probes from the direct slot and an EMPTY slot proves absence. At the
      // ~0.5 load factor residency produces, chains average under two probes.
      //
      // The previous scheme (direct probe, then a rotated full scan) measured
      // ~10 us per resolve for collided pages -- the scan walked hundreds of
      // slots, and half the pages had collided at warm time.
      //
      // Deletions (evictions) punch holes that can cut a chain short; the
      // empty-slot early-out then reports a false miss, which falls to the
      // LOCKED path whose full scan under the lock is authoritative. That is
      // correctness preserved at streaming-mode cost; when nothing evicts,
      // chains have no holes.
      {
        const clio::run::u32 d = (clio::run::u32) (pn % ppb);
        const clio::run::u32 w = ppb < kProbeWindow ? ppb : kProbeWindow;
        for (clio::run::u32 k = 0; k < w; ++k) {
          clio::run::u32 i = d + k;
          if (i >= ppb) i -= ppb;
          if (tbl[i].page_num == pn && !tbl[i].fetching) {
            p = &tbl[i];
            break;
          }
          if (tbl[i].page_num == kNoPage) break;   // absence proven
        }
      }
      if (p == nullptr) {
        LockBlock();          // a real miss: claiming/eviction needs the lock
        p = ResolveLocked(pn);
        UnlockBlock();
      }
      // Recency must be stamped on a HIT as well, not just in ResolveLocked.
      // Once the lock-free path started serving hits, ResolveLocked stopped
      // running for them, so re-touching a resident page no longer refreshed
      // it and LRU evicted the most recently used page instead of the oldest.
      // Still once per page TRANSITION, never per element.
      p->last_access = Now();
      p->score += 1.0f;
      last_page_ = p;
    }
    const clio::run::u64 within = IndexIn(off, p);
    const clio::run::u64 left = h_->elems_per_page_ - within;
    return (count < left) ? count : left;
  }

  /**
   * Raw pointer to the page most recently resolved by HoldPage on THIS
   * DeviceVector copy — page-relative access without per-element global
   * offset arithmetic:
   *
   *   v.HoldPage(pn * elems_per_page, elems_per_page);
   *   T *page = v.GetPagePtr();
   *   ... page[i] ...            // i is an index WITHIN the page
   *
   * Same lifetime rule as every raw hold: nothing pins the slot, so a
   * concurrent claim can recycle it mid-read. For long reads, capture the
   * claim generation (TryHoldRawConstG) and validate with HoldStillValid
   * after — see the kq kernel's seqlock retry.
   *
   * @return the held page's data, or nullptr if no page has been resolved.
   */
  CTP_GPU_FUN T *GetPagePtr() const {
    return last_page_ != nullptr ? static_cast<T *>(last_page_->data)
                                 : nullptr;
  }

  /**
   * Element access through the HELD page -- no resolution, no checks.
   *
   * Assumes HoldPage() covers `off`: it indexes last_page_ directly, which is
   * what makes it cost about the same as a pointer dereference. Accessing
   * outside the held run reads the wrong page's bytes, so the run length
   * HoldPage returned is a contract, not a hint: re-hold before stepping past
   * it. There is deliberately no resolve-per-access accessor -- that path cost
   * 12x a raw pointer even when it always hit, and hold-and-iterate is 1.44x.
   */
  CTP_GPU_FUN T &operator[](clio::run::u64 off) {
    Page *p = last_page_;
    p->dirty = 1u;
    return static_cast<T *>(p->data)[IndexIn(off, p)];
  }

  /** Read-only access through the held page. Does NOT dirty it. */
  CTP_GPU_FUN const T &at(clio::run::u64 off) const {
    const Page *p = last_page_;
    return static_cast<const T *>(p->data)[IndexIn(off, p)];
  }

  /**
   * Hold `count` elements from `off` and hand back a RAW pointer to them.
   *
   * Same contract as HoldPage, but returns the base pointer so a hot loop can
   * keep it in a register instead of re-reading last_page_ (which lives in the
   * by-value view's per-thread LOCAL memory) on every access. Use when the
   * inner loop is tight enough for that load to matter.
   */
  CTP_GPU_FUN T *HoldRaw(clio::run::u64 off, clio::run::u64 count,
                         clio::run::u64 *run) {
    *run = HoldPage(off, count);
    Page *p = last_page_;
    p->dirty = 1u;
    return static_cast<T *>(p->data) + IndexIn(off, p);
  }

  /**
   * PROBE-ONLY hold: the lock-free resident lookup and nothing else.
   *
   * Returns null on a miss instead of falling into the locked fault path, so
   * a caller with a BETTER recovery than a scalar fault (a batched fetch of
   * a whole expert, say) can take it. On a hit it is exactly HoldRawConst.
   * Exists because FetchPagesBatched holds the table lock across its whole
   * wait -- correct, but it means resident lookups that share a table with
   * an in-flight batch convoy behind ~ms of lock hold unless they can probe
   * without locking first.
   */
  CTP_GPU_FUN const T *TryHoldRawConst(clio::run::u64 off,
                                       clio::run::u64 count,
                                       clio::run::u64 *run) {
    const clio::run::u64 pn = PageOf(off);
    Page *p = last_page_;
    if (p == nullptr || p->page_num != pn || p->fetching) {
      p = nullptr;
      Page *tbl = BlockPages();
      const clio::run::u32 ppb = h_->pages_per_block_;
      const clio::run::u32 d = (clio::run::u32) (pn % ppb);
      const clio::run::u32 w = ppb < kProbeWindow ? ppb : kProbeWindow;
      for (clio::run::u32 k = 0; k < w; ++k) {
        clio::run::u32 i = d + k;
        if (i >= ppb) i -= ppb;
        if (tbl[i].page_num == pn) {
          // Re-read fetching AFTER the page_num match through volatile: the
          // claim stores fetching first (device fence between), so a probe
          // that saw the new page_num must also see fetching != 0.
          if (*(volatile clio::run::u32 *) &tbl[i].fetching == 0u) {
            p = &tbl[i];
            break;
          }
          return nullptr;  // claimed / in flight: miss
        }
        if (tbl[i].page_num == kNoPage) break;
      }
      if (p == nullptr) {
        return nullptr;
      }
      p->last_access = Now();
      p->score += 1.0f;   // frequency: what the window eviction ranks by.
      // The bistable-attractor caveat: on the MoE workload identical runs
      // land at 13k or 22-36k faults depending on first-burst timing. Both
      // a HIGHER fresh score (4.0) and a steeper reuse gradient (+2 touch,
      // 1.0 fresh) were measured WORSE than this policy (5-6/7 slow runs vs
      // 3/4 fast here) — do not re-tune blind; measure page-level churn
      // first.
      last_page_ = p;
    }
    const clio::run::u64 within = IndexIn(off, p);
    const clio::run::u64 left = h_->elems_per_page_ - within;
    *run = (count < left) ? count : left;
    return static_cast<const T *>(p->data) + within;
  }

  /** TryHoldRawConst that also captures the slot's claim generation, for
   *  seqlock validation after a long raw read (see Page::gen). */
  CTP_GPU_FUN const T *TryHoldRawConstG(clio::run::u64 off,
                                        clio::run::u64 count,
                                        clio::run::u64 *run,
                                        clio::run::u32 *gen_out,
                                        Page **slot_out) {
    const T *q = TryHoldRawConst(off, count, run);
    if (q != nullptr && last_page_ != nullptr) {
      *slot_out = last_page_;
      *gen_out = *(volatile clio::run::u32 *) &last_page_->gen;
    }
    return q;
  }

  /** @return true if the slot still holds the same claim generation (no
   *  recycle happened since the paired TryHoldRawConstG). */
  CTP_GPU_FUN bool HoldStillValid(Page *slot, clio::run::u32 gen) {
    __threadfence();
    return *(volatile clio::run::u32 *) &slot->gen == gen;
  }

  /** Read-only form of HoldRaw: does not dirty the page. */
  CTP_GPU_FUN const T *HoldRawConst(clio::run::u64 off, clio::run::u64 count,
                                    clio::run::u64 *run) {
    *run = HoldPage(off, count);
    const Page *p = last_page_;
    return static_cast<const T *>(p->data) + IndexIn(off, p);
  }

  /**
   * Which page an offset falls in.
   *
   * A GPU has NO hardware 64-bit integer divide, so `off / h_->elems_per_page_`
   * compiles to a software routine of hundreds of cycles -- on EVERY element
   * access. Measured: 683 ns per element, making a scan of already-resident
   * pages cost 350 us/page against a 15 us/page page fault. Reading the cache
   * was 20x more expensive than filling it.
   *
   * Page sizes are powers of two in every real use, so the host precomputes a
   * shift and mask and this becomes one instruction. The divide remains as a
   * fallback for a non-power-of-two page size.
   */
  CTP_GPU_FUN clio::run::u64 PageOf(clio::run::u64 off) const {
    return h_->page_shift_ ? (off >> h_->page_shift_) : (off / h_->elems_per_page_);
  }

  /** Offset of an element within its page. */
  CTP_GPU_FUN clio::run::u64 IndexIn(clio::run::u64 off, const Page *p) const {
    return h_->page_shift_ ? (off & h_->page_mask_)
                       : (off - p->page_num * h_->elems_per_page_);
  }

  /**
   * Evict `num_pages` resident pages, lowest score first and oldest access
   * breaking ties. A clean page is simply dropped; a dirty one is flushed
   * whole and waited on, because its bytes are the only copy.
   */
  CTP_GPU_FUN void EvictPages(clio::run::u32 num_pages) {
    LockBlock();
    EvictLocked(num_pages);
    UnlockBlock();
  }

  /**
   * Drop this block's whole cache: flush anything dirty, then clear every
   * slot AND its score.
   *
   * EvictPages alone is not enough to get back to a cold cache, because a
   * page left with a high score from a previous phase keeps steering eviction
   * afterwards. Benchmarks need this to make two modes start from identical
   * state; without it, whichever mode ran second inherits a warm cache and
   * the comparison is meaningless.
   */
  CTP_GPU_FUN void DropAll() {
    LockBlock();
    Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      tbl[i].score = 0.0f;
    }
    EvictLocked(h_->pages_per_block_);
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      tbl[i].last_access = 0;
    }
    UnlockBlock();
    last_page_ = nullptr;   // the per-thread cache now points at a free slot
  }

  /**
   * Claim a slot for `pn` within a BOUNDED probe window of its home slot.
   *
   * Open addressing degrades catastrophically at full load: with the table
   * 100% occupied (an out-of-core working set churning through LRU, the MoE
   * case), an unsuccessful probe scanned every slot (~743 dependent global
   * loads) and successful chains grew to hundreds -- measured as ~3 ms of
   * per-launch dwell that no compute optimization touched. Bounding the
   * probe to kProbeWindow slots and EVICTING THE WINDOW'S LRU when no slot
   * is free keeps every chain shorter than the window, permanently, at the
   * cost of approximating global LRU by window-local LRU.
   *
   * Caller must hold the block lock. Returns the claimed slot, its entry
   * reset and NOT yet marked fetching; ~0u if every window slot is pinned by
   * an in-flight transfer.
   */
  static constexpr clio::run::u32 kProbeWindow = 64;
  CTP_GPU_FUN clio::run::u32 ClaimSlotWindowLocked(clio::run::u64 pn) {
    clio::run::u32 s = ClaimSlotWindowOnceLocked(pn);
    if (s == ~0u) {
      // Every candidate is mid-transfer. If any are fetching=2 their batch
      // may long since have LANDED — nothing settles an async batch until a
      // toucher hits one of ITS pages, so a window can fill with landed-but-
      // unsettled pages and starve every claimer whose page is not among
      // them (observed: kernel spun forever, CPU idle). Settle and rescan.
      SettleBatchLocked();
      s = ClaimSlotWindowOnceLocked(pn);
    }
    return s;
  }

  CTP_GPU_FUN clio::run::u32 ClaimSlotWindowOnceLocked(clio::run::u64 pn) {
    Page *tbl = BlockPages();
    const clio::run::u32 ppb = h_->pages_per_block_;
    const clio::run::u32 d = (clio::run::u32) (pn % ppb);
    const clio::run::u32 w = ppb < kProbeWindow ? ppb : kProbeWindow;
    clio::run::u32 victim = ~0u;
    float best = 3.4e38f;
    for (clio::run::u32 k = 0; k < w; ++k) {
      clio::run::u32 i = d + k;
      if (i >= ppb) i -= ppb;
      Page &pgi = tbl[i];
      if (pgi.page_num == kNoPage) {
        return i;
      }
      // FREQUENCY, not recency. With the working set larger than the cache,
      // LRU converges to ~0% hit rate: every page's reuse distance exceeds
      // capacity, so everything is evicted before its next use. Routed-expert
      // traffic is SKEWED -- hot experts recur -- and a touch-count score
      // keeps them resident while cold ones churn.
      //
      // Aging must be LINEAR and slow. Halving per claim scan was ~0.5^23
      // per token (a table sees ~23 claims/token): every score decayed to
      // zero between touches, eviction degenerated to random, and a batch's
      // own just-fetched pages self-evicted at score ~0.004. At -0.02 per
      // scan, a page touched once per token sustains ~+0.5/token net while
      // an untouched one loses its history in ~50 claims.
      pgi.score = fmaxf(pgi.score - 0.02f, 0.0f);
      if (!pgi.fetching && !pgi.flushing && !pgi.rescoring &&
          pgi.score < best) {
        best = pgi.score;
        victim = i;
      }
    }
    if (victim == ~0u) {
      return ~0u;
    }
    // Weights are read-only here (dirty is never set on this path); dropping
    // the clean victim is a metadata write.
    tbl[victim].page_num = kNoPage;
    Bump(h_->stat_evicts_);
    return victim;
  }

  /** EvictPages' body, for callers that already hold the block lock. */
  CTP_GPU_FUN void EvictLocked(clio::run::u32 num_pages) {
    Page *tbl = BlockPages();
    for (clio::run::u32 k = 0; k < num_pages; ++k) {
      clio::run::u32 victim = h_->pages_per_block_;
      for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
        if (tbl[i].page_num == kNoPage) continue;
        // A page with a transfer in flight is not a candidate: its slot is
        // already promised to that transfer, and evicting it would let the
        // copy land in a slot that now belongs to a different page.
        if (tbl[i].fetching) continue;
        if (victim == h_->pages_per_block_) {
          victim = i;
          continue;
        }
        if (tbl[i].score < tbl[victim].score ||
            (tbl[i].score == tbl[victim].score &&
             tbl[i].last_access < tbl[victim].last_access)) {
          victim = i;
        }
      }
      if (victim == h_->pages_per_block_) return;   // nothing resident
      Page *p = &tbl[victim];
      if (p->dirty || p->flushing) {
        SubmitPut(p);
        AwaitPut(p);
      }
      p->page_num = kNoPage;
      p->dirty = 0u;
      p->flushing = 0u;
      Bump(h_->stat_evicts_);
    }
  }

  /**
   * Start writing back every page overlapping [off, off+count) elements.
   * Pages are marked clean immediately -- the put carries the bytes as they
   * are now, and a later write re-dirties the page for the next flush.
   */
  CTP_GPU_FUN void BeginFlush(clio::run::u64 off, clio::run::u64 count) {
    LockBlock();
    ForEachResident(off, count, [this](Page *p) {
      if (!p->dirty) return;
      SubmitPut(p);
    });
    UnlockBlock();
  }

  /**
   * Write back every dirty page in THIS block using batched puts.
   *
   * Synchronous by construction: all batches are submitted, then all are
   * waited on, with the block lock held the whole time. That is what makes it
   * safe to add without a new state machine -- no other thread can observe a
   * page mid-batch, so pages need no per-page `flushing` bookkeeping and the
   * existing scalar invariants are untouched. The overlap that matters is
   * already there: the batches are all in flight at once.
   *
   * A page is marked clean only if ITS OWN record succeeded. The scalar path
   * clears `dirty` at submission time; here the per-record return code is
   * available, so a failed page stays dirty and will be retried rather than
   * being silently dropped.
   *
   * @return the number of pages successfully written back.
   */
  CTP_GPU_FUN clio::run::u32 FlushBlockBatched() {
    LockBlock();
    const clio::run::u32 n = FlushBlockBatchedLocked();
    UnlockBlock();
    return n;
  }

  /**
   * Fault pages [first_page, first_page + n) into this block's cache with
   * BATCHED gets, and wait for them.
   *
   * This is the read-side counterpart to FlushBlockBatched, and it exists for
   * the same measured reason: a fault costs ~110us of round trip against ~6us
   * of actual 256 KB device-to-device copy, so the SUBMISSION is ~95% of a
   * read and cutting bytes (by compressing) cannot pay for itself while every
   * page costs one. n pages in one submission amortizes that, which is the
   * only way reduced bytes can turn into reduced time.
   *
   * Blocking by design: a streaming reader asks for its next chunk, gets it
   * all at once, and then walks it with no faults at all. `n` is clamped to
   * what the cache can actually hold and to kPodMultiMax.
   *
   * @return the number of pages now resident and valid.
   */
  CTP_GPU_FUN clio::run::u32 FetchPagesBatched(clio::run::u64 first_page,
                                               clio::run::u32 n) {
    LockBlock();
    const clio::run::u32 got = FetchPagesBatchedLocked(first_page, n);
    UnlockBlock();
    return got;
  }

  /**
   * Issue ONE batched get for the missing pages of [first_page, first_page+n)
   * WITHOUT waiting for it — the batched analogue of BeginFetch, and the
   * primitive that makes fetch/compute PIPELINING possible: the synchronous
   * FetchPagesBatched holds the table lock through the whole round trip
   * (~300 µs), so nothing it "prefetched" ever overlapped anything.
   *
   * Claimed pages are marked fetching=2; any later toucher (AwaitFetch via
   * Resolve/Hold, or the synchronous batch) settles the WHOLE batch. One
   * outstanding async batch per table: if one is already in flight and not
   * yet complete, this returns 0 and fetches nothing — the caller is
   * prefetching, so dropping the hint is always safe.
   *
   * @return pages actually claimed and issued (0 = all resident/in-flight,
   *         batch slot busy, or batching not provisioned).
   */
  CTP_GPU_FUN clio::run::u32 FetchPagesBatchedAsync(clio::run::u64 first_page,
                                                    clio::run::u32 n) {
    LockBlock();
    const clio::run::u32 got = FetchPagesBatchedAsyncLocked(first_page, n);
    UnlockBlock();
    return got;
  }

  /**
   * Start faulting `page_num` in WITHOUT waiting for it.
   *
   * This is what makes compute and I/O overlap: the kernel asks for the page
   * it will need next, keeps computing on the page it already has, and by the
   * time it dereferences the new one the bytes have arrived. Resolve waits on
   * an in-flight fetch rather than issuing a second get, so a prefetch that
   * has not landed yet degrades to exactly the synchronous cost and never to
   * a wrong result.
   *
   * @return false if no slot could be freed (every resident page is pinned by
   *         an in-flight transfer), in which case nothing was started.
   */
  CTP_GPU_FUN bool BeginFetch(clio::run::u64 page_num) {
    LockBlock();
    const bool ok = BeginFetchLocked(page_num);
    UnlockBlock();
    return ok;
  }

  /** Wait for a page started by BeginFetch, if it is still in flight. */
  CTP_GPU_FUN void WaitFetch(clio::run::u64 page_num) {
    LockBlock();
    Page *p = Find(page_num);
    if (p != nullptr) AwaitFetch(p);
    UnlockBlock();
  }

  /** Wait for the puts started by BeginFlush over the same range. */
  CTP_GPU_FUN void WaitFlush(clio::run::u64 off, clio::run::u64 count) {
    LockBlock();
    ForEachResident(off, count, [this](Page *p) {
      if (p->flushing) AwaitPut(p);
    });
    UnlockBlock();
  }

  /**
   * Set a page's score. If the page is resident the score is applied
   * locally (it steers eviction immediately, with no round trip); a
   * PodReorganizeBlobTask is also sent so the CTE's placement follows,
   * which is what makes this usable as a prefetch hint.
   */
  CTP_GPU_FUN void RescorePage(clio::run::u64 page_id, float score) {
    LockBlock();
    RescoreLocked(page_id, score);
    UnlockBlock();
  }

 private:
  CTP_GPU_FUN void RescoreLocked(clio::run::u64 page_id, float score) {
    Page *p = Find(page_id);
    if (p != nullptr) {
      p->score = score;
    }
    Page *slot = (p != nullptr) ? p : &BlockPages()[page_id % h_->pages_per_block_];
    if (slot->rescore == nullptr) return;
    // A rescore is fire-and-forget, so the previous one on this slot may still
    // be executing. Reusing the task in place would mutate a task the runtime
    // is reading. Rescoring is a hot operation when it is used as a prefetch
    // hint (twice per page: pin, then release), so this is reachable in normal
    // use, not just in theory.
    if (slot->rescoring) {
      slot->rescore_fut.Wait();
      slot->rescoring = 0u;
    }
    auto *t = slot->rescore;
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = NextTaskId(kKindRescore);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodReorganizeBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    char name[32];
    PageBlobName(page_id, name);
    t->blob_name_ = name;
    t->new_score_ = score;
    t->replica_ = 0;
    // Fire and forget: a hint that arrives late is still useful, and
    // waiting here would serialise the kernel behind placement.
    ClearRunCtx(t);
    Bump(h_->stat_rescores_);
    slot->rescore_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(slot->rescore));
    slot->rescoring = 1u;
  }

 public:
  /** Number of elements in the vector. */
  CTP_GPU_FUN clio::run::u64 size() const { return h_->size_; }


 private:
  /** Per-thread cache of the last page touched. NOT __shared__. */
  Page *last_page_ = nullptr;
  /**
   * Slot index where the lock-free scan STARTS: the last hit.
   *
   * The scan is O(pages_per_block) and residency sizing makes that large
   * (816 slots, ~424 occupied). From slot 0 it measured as a ~115 us floor on
   * every kernel launch -- each warp's first HoldPage walking ~400 slots --
   * invariant to every inner-loop optimization because it precedes them.
   * FetchPagesBatchedLocked claims first-free slots in order, so consecutive
   * pages sit in consecutive slots and a scan starting at the last hit finds
   * a sequential walk's next page in a probe or two.
   */
  clio::run::u32 last_slot_ = 0;

 public:
  /** BlockPages() uses this instead of blockIdx.x when set. */
  static constexpr clio::run::u32 kNoBlockOverride = ~0u;
  /**
   * Which page-table partition this thread addresses; kNoBlockOverride means
   * blockIdx.x, the default and the only behaviour before launch fusion.
   *
   * Fusion by grid partition gives a fused member blocks whose GLOBAL index
   * differs from the index its own scalar launch (and the warm-up, which
   * launches per weight) would have had -- so its resolves looked in tables
   * that never held its pages. Setting this to the member-local block index
   * restores the exact table its scalar launch would use. Two members' blocks
   * may then address one table concurrently; resident lookups are lock-free
   * reads and the fault path takes the table's lock, so both remain safe.
   */
  clio::run::u32 block_override_ = kNoBlockOverride;

 private:

  /**
   * Take this block's page-table lock.
   *
   * Safe to contend on from within a warp because independent thread
   * scheduling (Volta and later, which the runtime already assumes -- see the
   * __nanosleep in ipc_gpu2cpu_impl.h) lets the holder make progress while its
   * warp-mates spin. The critical section can be long: it covers a page fault's
   * whole round trip, which is exactly the point -- lanes that wanted the same
   * page are held until it is resident, then find it with no fault of their own.
   */
  CTP_GPU_FUN void LockBlock() {
    if (h_->block_locks_ == nullptr) return;
    int *lk = h_->block_locks_ + BlockIndex();
    while (atomicCAS(lk, 0, 1) != 0) {
      __nanosleep(32);
    }
    __threadfence();
  }

  CTP_GPU_FUN void UnlockBlock() {
    if (h_->block_locks_ == nullptr) return;
    // Publish every page-table write before the lock is observed free.
    __threadfence();
    atomicExch(h_->block_locks_ + BlockIndex(), 0);
  }

  /** Increment an instrumentation counter if the test enabled it. */
  CTP_GPU_FUN void Bump(unsigned long long *c) const {
    if (c != nullptr) atomicAdd(c, 1ull);
  }

  CTP_GPU_FUN clio::run::u64 Now() const {
    return static_cast<clio::run::u64>(clock64());
  }

  /** This block's batch slots -- same partitioning as the page table. */
  CTP_GPU_FUN MultiBatch *BlockBatches() const {
    return h_->multi_ +
           static_cast<clio::run::u64>(BlockIndex()) * h_->multi_per_block_;
  }

  /** This block's slice of the page table. */
  /**
   * THE block index -- the only definition. The page table, its lock, and
   * the batch slots must all be selected by the SAME index: when fused
   * launches introduced block_override_, only BlockPages() was updated, so a
   * fused member's blocks mutated one table while holding a DIFFERENT
   * table's lock. Two same-table blocks faulting concurrently then raced the
   * claim/evict path unserialized -- the same page slot claimed twice, its
   * get task submitted twice, and the second host staging clobbering the
   * first's RunContext ("null RunContext" at varying points, constant under
   * an eviction storm, rare but latent in ordinary fused decode).
   */
  CTP_GPU_FUN clio::run::u32 BlockIndex() const {
    const clio::run::u32 raw =
        block_override_ != kNoBlockOverride ? block_override_ : blockIdx.x;
    return raw % h_->nblocks_;
  }

  CTP_GPU_FUN Page *BlockPages() const {
    return h_->pages_ +
           static_cast<clio::run::u64>(BlockIndex()) * h_->pages_per_block_;
  }

  /** A globally unique id for the next task this vector submits. */
  CTP_GPU_FUN clio::run::TaskId NextTaskId(clio::run::u32 kind) const {
    unsigned long long n =
        (h_->task_seq_ != nullptr) ? atomicAdd(h_->task_seq_, 1ull) : 0ull;
    return DeviceTaskId(n, kind, static_cast<clio::run::u32>(n));
  }

  /** Index of `p` in the whole page table -- the basis of its task ids. */
  CTP_GPU_FUN clio::run::u64 SlotOf(const Page *p) const {
    return static_cast<clio::run::u64>(p - h_->pages_);
  }

  CTP_GPU_FUN Page *Find(clio::run::u64 page_num) const {
    Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].page_num == page_num) return &tbl[i];
    }
    return nullptr;
  }

  template <typename TaskT>
  CTP_GPU_FUN ctp::ipc::FullPtr<TaskT> SlotPtr(TaskT *task) const {
    ctp::ipc::FullPtr<TaskT> fp;
    fp.shm_.alloc_id_ = h_->task_alloc_id_;
    fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task);
    fp.ptr_ = task;
    return fp;
  }

  /**
   * Point a task's blob_data_ at raw device memory: a null allocator id
   * makes ToFullPtr treat off_ as the address itself, which is how the
   * runtime reaches page bytes that live in a device backend.
   */
  CTP_GPU_FUN ctp::ipc::ShmPtr<> RawPtr(void *addr) const {
    ctp::ipc::ShmPtr<> p;
    p.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    p.off_ = reinterpret_cast<clio::run::u64>(addr);
    return p;
  }

  /**
   * Clear a task's RunContext handle before RE-submitting the slot.
   *
   * A task slot is reused every time its page is re-faulted or re-flushed.
   * The first execution leaves run_ctx_ pointing at a RunContext that is
   * already marked STARTED, and SendOut copies that pointer back into the
   * device task. Resubmitting as-is makes the worker take the RESUME path,
   * find no coroutine handle, and drop the task -- the kernel then waits
   * forever on a completion that cannot come. Handing the task over with no
   * RunContext makes RecvIn's BeginRunContext allocate a fresh one, which is
   * what a first submission gets.
   *
   * It is three pointers of POD on both sides (see Task::run_ctx_), so
   * clearing the bytes is the device-side equivalent of ResetRunCtx().
   */
  template <typename TaskT>
  CTP_GPU_FUN void ClearRunCtx(TaskT *t) const {
    char *raw = reinterpret_cast<char *>(&t->run_ctx_);
    for (unsigned i = 0; i < sizeof(t->run_ctx_); ++i) raw[i] = 0;
  }

  /** Issue this page's PodPutBlobTask for the whole page. */
  CTP_GPU_FUN void SubmitPut(Page *p) {
    if (p->flushing) return;             // one outstanding put per page
    auto *t = p->put;
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = NextTaskId(kKindPut);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodPutBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    char name[32];
    PageBlobName(p->page_num, name);
    t->blob_name_ = name;
    t->offset_ = 0;
    t->size_ = h_->page_bytes_;
    t->blob_data_ = RawPtr(p->data);
    // An UNSCORED page must say "no opinion" (-1), not "score zero".
    //
    // The two numbers mean opposite things. Device-side, score 0 is this
    // page's eviction rank -- the default, meaning "evict me first". To the
    // CTE it is a placement request, and MaxBwDpe prefers targets whose
    // target_score_ is <= the blob's; with tiers at 1.0 (top) and 0.2 (spill),
    // a 0.0 blob matches NEITHER, so both became fallbacks and the ranking
    // deliberately picks the slowest of those first. Every page therefore
    // landed on the spill tier and the fast top tier went unused -- measured:
    // reads under a throttled spill tier were fully throttled, while the same
    // cap on the top tier changed nothing at all because nothing read from it.
    //
    // -1 asks the CTE for its default (1.0 for a new blob), i.e. the fastest
    // tier that fits, which is what a page cache writing back hot data wants.
    // An explicitly rescored page still sends its own score, so RescorePage
    // keeps steering placement.
    t->score_ = (p->score > 0.0f) ? p->score : -1.0f;
    t->flags_ = 0;
    t->context_ = clio::cte::core::Context();
    t->context_.compress_lib_ = h_->compress_lib_;
    t->context_.compress_preset_ = h_->compress_preset_;
    ClearRunCtx(t);
    Bump(h_->stat_puts_);
    p->put_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(p->put));
    // Clean as of THIS put: the bytes it carries are what the page held
    // when it was submitted. A later write dirties it again for the next.
    p->dirty = 0u;
    p->flushing = 1u;
  }

  /** Header fields of a batch task; the records are appended by the caller. */
  CTP_GPU_FUN void PrepareMultiPut(
      clio::cte::core::PodMultiPutBlobTask *t) const {
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = NextTaskId(kKindPut);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodMultiPutBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    t->count_ = 0;
    t->flags_ = 0;
    t->num_ok_ = 0;
    t->context_ = clio::cte::core::Context();
    t->context_.compress_lib_ = h_->compress_lib_;
    t->context_.compress_preset_ = h_->compress_preset_;
    ClearRunCtx(t);
  }

  /** Header fields of a batched get; records are appended by the caller. */
  CTP_GPU_FUN void PrepareMultiGet(
      clio::cte::core::PodMultiGetBlobTask *t) const {
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = NextTaskId(kKindGet);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodMultiGetBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    t->count_ = 0;
    t->flags_ = 0;
    t->num_ok_ = 0;
    // Same codec the PUT declared: a page written through the compressor is
    // STORED compressed, so a get with an empty context asks for the stored
    // bytes rather than the data.
    t->context_ = clio::cte::core::Context();
    t->context_.compress_lib_ = h_->compress_lib_;
    t->context_.compress_preset_ = h_->compress_preset_;
    ClearRunCtx(t);
  }

  /** FlushBlockBatched's body, for callers already holding the block lock. */
  CTP_GPU_FUN clio::run::u32 FlushBlockBatchedLocked() {
    Page *tbl = BlockPages();
    // Not provisioned (host did not allocate batch slots): do it scalar-wise
    // rather than silently skipping the flush.
    if (h_->multi_ == nullptr || h_->multi_per_block_ == 0) {
      clio::run::u32 done = 0;
      for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
        if (!tbl[i].dirty && !tbl[i].flushing) continue;
        SubmitPut(&tbl[i]);
        AwaitPut(&tbl[i]);
        ++done;
      }
      return done;
    }
    // A scalar put still in flight owns that page's bytes; batching it now
    // would put the same page on the wire twice.
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].flushing) AwaitPut(&tbl[i]);
    }

    MultiBatch *mb = BlockBatches();
    clio::run::u32 nb = 0;              // batches with records in them
    clio::run::u32 filled = 0;          // records in batch nb
    PrepareMultiPut(mb[0].put);
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      Page *p = &tbl[i];
      if (p->page_num == kNoPage || !p->dirty) continue;
      if (filled == clio::cte::core::kPodMultiMax) {
        // Full: submit and move on. h_->multi_per_block_ is sized so this can
        // only run out if the host provisioned fewer batches than pages.
        mb[nb].put_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(mb[nb].put));
        ++nb;
        filled = 0;
        if (nb == h_->multi_per_block_) break;
        PrepareMultiPut(mb[nb].put);
      }
      char name[32];
      PageBlobName(p->page_num, name);
      // Same placement contract as SubmitPut: an unscored page asks for the
      // CTE's default tier rather than requesting score zero.
      mb[nb].put->Add(name, 0, h_->page_bytes_, RawPtr(p->data),
                      (p->score > 0.0f) ? p->score : -1.0f);
      mb[nb].page_slot[filled] = i;
      ++filled;
      Bump(h_->stat_puts_);
    }
    clio::run::u32 nsub = nb;
    if (filled > 0) {
      mb[nb].put_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(mb[nb].put));
      nsub = nb + 1;
    }

    clio::run::u32 ok = 0;
    for (clio::run::u32 b = 0; b < nsub; ++b) {
      mb[b].put_fut.Wait();
      auto *t = mb[b].put;
      for (clio::run::u32 r = 0; r < t->count_; ++r) {
        Page *p = &tbl[mb[b].page_slot[r]];
        // Clean only on this record's own success -- a failed page keeps its
        // dirty bit and gets written back on the next flush or eviction.
        if (t->reqs_[r].rc_ == 0) {
          p->dirty = 0u;
          ++ok;
        }
      }
    }
    return ok;
  }

  /** FetchPagesBatched's body, for callers already holding the block lock. */
  /** Async batch slots: the LAST kAsyncBatchSlots of the table's
   *  multi_per_block_ MultiBatch entries. The synchronous batch path only
   *  ever uses mb[0], so async batches never make a gate fault wait behind
   *  a sibling prefetch on the same table; several slots because one layer's
   *  prefetch wave overlapping the previous layer's on a shared table was
   *  otherwise a dropped hint (= a sync fault two launches later). */
  // Measured: 3 beats 1 (dropped hints become sync faults) and beats 6
  // (more outstanding claims churned the eviction window; one K=6 run hit
  // 107 ms/tok vs the 75 baseline).
  static constexpr clio::run::u32 kAsyncBatchSlots = 3;

  CTP_GPU_FUN clio::run::u32 AsyncSlotCount() const {
    const clio::run::u32 mpb = h_->multi_per_block_;
    if (mpb < 2) return 0;
    const clio::run::u32 avail = mpb - 1;  // mb[0] is the sync slot
    return avail < kAsyncBatchSlots ? avail : kAsyncBatchSlots;
  }

  CTP_GPU_FUN MultiBatch *AsyncBatchSlot(clio::run::u32 k) const {
    return &BlockBatches()[h_->multi_per_block_ - 1 - k];
  }

  /** Non-blocking completion probe of an async batch future. */
  CTP_GPU_FUN bool MultiGetDone(MultiBatch *mb) {
    if (mb->get_fut.IsNull()) return true;
    volatile unsigned int *fp = reinterpret_cast<volatile unsigned int *>(
        &mb->get_fut.get()->fut_.is_complete_.x);
    return ((*fp) & 1u) != 0u;
  }

  /** Settle ONE async slot: wait for the multi, clear the fetching=2 marks,
   *  free slots whose record failed. Caller must hold the block lock. */
  CTP_GPU_FUN void SettleOneLocked(MultiBatch *mb) {
    if (mb->async_pending == 0u) return;
    mb->get_fut.Wait();
    auto *t = mb->get;
    Page *tbl = BlockPages();
    for (clio::run::u32 r = 0; r < mb->async_n; ++r) {
      Page *p = &tbl[mb->page_slot[r]];
      if (p->fetching == 2u) p->fetching = 0u;
      if (!(r < t->count_ && t->reqs_[r].rc_ == 0)) {
        // Same rule as the synchronous batch: a failed record must free its
        // slot, never serve its stale bytes as this page.
        p->page_num = kNoPage;
        Bump(h_->stat_get_errors_);
      }
    }
    mb->async_pending = 0u;
  }

  /**
   * Complete every outstanding async batch on this table. A fetching=2 page
   * does not record WHICH batch fetched it, so a toucher settles them all —
   * by touch time they are almost always complete anyway.
   * Caller must hold the block lock.
   */
  CTP_GPU_FUN void SettleBatchLocked() {
    if (h_->multi_ == nullptr) return;
    const clio::run::u32 cnt = AsyncSlotCount();
    for (clio::run::u32 k = 0; k < cnt; ++k) SettleOneLocked(AsyncBatchSlot(k));
  }

  /** Async batched fetch body; see FetchPagesBatchedAsync. Lock held. */
  CTP_GPU_FUN clio::run::u32 FetchPagesBatchedAsyncLocked(
      clio::run::u64 first_page, clio::run::u32 n) {
    // Needs slots of its own (see AsyncBatchSlot); with only mb[0] available
    // async fetching would contend with the synchronous path, so disable it.
    if (h_->multi_ == nullptr) return 0;
    const clio::run::u32 cnt = AsyncSlotCount();
    if (cnt == 0) return 0;
    MultiBatch *mb = nullptr;
    for (clio::run::u32 k = 0; k < cnt; ++k) {
      MultiBatch *cand = AsyncBatchSlot(k);
      if (cand->async_pending == 0u) {
        mb = cand;
        break;
      }
      if (MultiGetDone(cand)) {
        SettleOneLocked(cand);
        mb = cand;
        break;
      }
    }
    if (mb == nullptr) {
      // Every slot carries an unfinished batch. Drop the hint, never queue —
      // stacked prefetches serialize their callers, which is the exact
      // failure the async form exists to avoid.
      Bump(h_->stat_pf_dropped_);
      return 0;
    }
    if (n > clio::cte::core::kPodMultiMax) n = clio::cte::core::kPodMultiMax;
    if (n > h_->pages_per_block_) n = h_->pages_per_block_;
    Page *tbl = BlockPages();
    PrepareMultiGet(mb->get);
    // Prefetch, not demand: let the CTE service it off the critical path.
    mb->get->flags_ = clio::cte::core::kCtePrefetchHint;
    clio::run::u32 filled = 0;
    for (clio::run::u32 k = 0; k < n; ++k) {
      const clio::run::u64 pg = first_page + k;
      if (Find(pg) != nullptr) continue;  // resident or already coming
      const clio::run::u32 slot = ClaimSlotWindowLocked(pg);
      if (slot == ~0u) break;
      Page *np = &tbl[slot];
      np->gen += 1u;
      np->fetching = 2u;  // batch-async: settled via SettleBatchLocked
      // DEVICE-wide: the readers that must observe fetching before the new
      // page_num are lock-free probes in OTHER blocks. A block fence let
      // them see the claimed page_num with fetching still 0 and read the
      // EVICTED page's bytes for the whole async in-flight window — the
      // intermittent-garbage-output bug.
      __threadfence();
      np->page_num = pg;
      np->dirty = 0u;
      np->flushing = 0u;
      // Same as the sync claim. 4.0 (protect prefetches) and 1.0 (steep
      // reuse gradient) were both measured worse — see the eviction note.
      np->score = 2.0f;
      np->last_access = Now();
      char name[32];
      PageBlobName(pg, name);
      mb->get->Add(name, 0, h_->page_bytes_, RawPtr(np->data));
      mb->page_slot[filled] = slot;
      ++filled;
      Bump(h_->stat_faults_);
      Bump(h_->stat_prefetches_);
      if (h_->fault_hist_ != nullptr) atomicAdd(&h_->fault_hist_[pg], 1u);
    }
    if (filled == 0) return 0;
    mb->get_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(
        SlotPtr(mb->get));
    mb->async_pending = 1u;
    mb->async_n = filled;
    return filled;
  }

  CTP_GPU_FUN clio::run::u32 FetchPagesBatchedLocked(clio::run::u64 first_page,
                                                     clio::run::u32 n) {
    Page *tbl = BlockPages();
    if (h_->multi_ == nullptr || h_->multi_per_block_ == 0) {
      // Not provisioned: fault them one at a time rather than silently
      // returning nothing.
      clio::run::u32 got = 0;
      for (clio::run::u32 k = 0; k < n; ++k) {
        if (ResolveLocked(first_page + k) != nullptr) ++got;
      }
      return got;
    }
    if (n > clio::cte::core::kPodMultiMax) n = clio::cte::core::kPodMultiMax;
    // A batch cannot exceed the cache: claiming more slots than exist would
    // evict a page this same batch already claimed.
    if (n > h_->pages_per_block_) n = h_->pages_per_block_;

    MultiBatch *mb = BlockBatches();
    PrepareMultiGet(mb[0].get);
    clio::run::u32 filled = 0;
    clio::run::u32 resident = 0;
    for (clio::run::u32 k = 0; k < n; ++k) {
      const clio::run::u64 pg = first_page + k;
      Page *p = Find(pg);
      if (p != nullptr) {
        // Already here (or already in flight from a scalar prefetch): let the
        // existing path finish it rather than issuing a second get.
        if (p->fetching) AwaitFetch(p);
        ++resident;
        continue;
      }
      const clio::run::u32 slot = ClaimSlotWindowLocked(pg);
      if (slot == ~0u) break;   // whole window pinned by in-flight transfers
      Page *np = &tbl[slot];
      // fetching FIRST, then the page number -- same ordering the scalar claim
      // uses, and it is what keeps EvictLocked from choosing a slot this batch
      // has already claimed but not yet filled.
      np->gen += 1u;
      np->fetching = 1u;
      __threadfence();  // device-wide; see the async claim note
      np->page_num = pg;
      np->dirty = 0u;
      np->flushing = 0u;
      // A fresh page must not be the window's instant victim: claimed with
      // score 0 and stale last_access it was FIRST in line for eviction --
      // measured as more faults per token than routed pages.
      np->score = 2.0f;
      np->last_access = Now();
      char name[32];
      PageBlobName(pg, name);
      mb[0].get->Add(name, 0, h_->page_bytes_, RawPtr(np->data));
      mb[0].page_slot[filled] = slot;
      ++filled;
      Bump(h_->stat_faults_);
      Bump(h_->stat_prefetches_);
      if (h_->fault_hist_ != nullptr) atomicAdd(&h_->fault_hist_[pg], 1u);
    }
    if (filled == 0) return resident;

    mb[0].get_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(mb[0].get));
    mb[0].get_fut.Wait();
    auto *t = mb[0].get;
    clio::run::u32 ok = 0;
    for (clio::run::u32 r = 0; r < filled; ++r) {
      Page *p = &tbl[mb[0].page_slot[r]];
      p->fetching = 0u;
      if (r < t->count_ && t->reqs_[r].rc_ == 0) {
        ++ok;
      } else {
        // A failed get leaves the slot holding whatever it held before. Free
        // it rather than letting the caller read those bytes as this page --
        // serving stale data silently is the one outcome worse than a miss.
        p->page_num = kNoPage;
        Bump(h_->stat_get_errors_);
      }
    }
    return resident + ok;
  }

  CTP_GPU_FUN void AwaitPut(Page *p) {
    if (!p->flushing) return;
    p->put_fut.Wait();
    p->flushing = 0u;
  }

  /**
   * Claim a slot for `page_num` and start its get, without waiting.
   * Caller must hold the block lock.
   */
  CTP_GPU_FUN bool BeginFetchLocked(clio::run::u64 page_num) {
    if (Find(page_num) != nullptr) return true;   // resident or already coming
    Page *tbl = BlockPages();
    clio::run::u32 free_slot = h_->pages_per_block_;
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].page_num == kNoPage) {
        free_slot = i;
        break;
      }
    }
    if (free_slot == h_->pages_per_block_) {
      const clio::run::u32 wslot = ClaimSlotWindowLocked(page_num);
      // Everything in the window is mid-transfer: report failure rather than
      // stalling, so the caller simply falls back to a demand fault later.
      if (wslot == ~0u) return false;
      free_slot = wslot;
    }
    Page *p = &tbl[free_slot];
    // Same ordering as the synchronous claim: busy first, then the page
    // number. Reversed, HoldPage's lock-free scan can see a slot that looks
    // resident while its transfer is still in flight.
    p->fetching = 1u;
    __threadfence_block();
    p->gen += 1u;
    p->page_num = page_num;
    p->dirty = 0u;
    p->flushing = 0u;
    p->score = 0.0f;
    SubmitGetAsync(p, page_num);
    return true;
  }

  /** Issue this page's get and return immediately. */
  CTP_GPU_FUN void SubmitGetAsync(Page *p, clio::run::u64 page_num) {
    PrepareGet(p, page_num);
    Bump(h_->stat_faults_);
    Bump(h_->stat_prefetches_);
    p->fetching = 1u;                     // already set by the claim; keep it
    p->get_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(p->get));
    if (p->get_fut.IsNull()) {
      Bump(h_->stat_get_errors_);
    }
  }

  /** Wait for an in-flight asynchronous get on `p`. */
  CTP_GPU_FUN void AwaitFetch(Page *p) {
    if (!p->fetching) return;
    if (p->fetching == 2u) {
      // Claimed by an async batch: its bytes arrive with the batch's multi,
      // not a per-page get, so the page's own future carries nothing. Settle
      // the whole batch (this also clears every sibling page it fetched).
      SettleBatchLocked();
      return;
    }
    p->get_fut.Wait();
    if (p->get->GetReturnCode() != 0) Bump(h_->stat_get_errors_);
    p->fetching = 0u;
  }

  /** Fill in the get task's fields for `page_num`. */
  CTP_GPU_FUN void PrepareGet(Page *p, clio::run::u64 page_num) {
    auto *t = p->get;
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = NextTaskId(kKindGet);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodGetBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    char name[32];
    PageBlobName(page_num, name);
    t->blob_name_ = name;
    t->offset_ = 0;
    t->size_ = h_->page_bytes_;
    t->blob_data_ = RawPtr(p->data);
    t->flags_ = 0;
    // The GET must declare the same codec the PUT did. A page written through
    // the compressor is stored in its COMPRESSED form, so a reader that says
    // nothing about compression is asking for the stored bytes, not its data.
    // This was latent for as long as the compressor decompressed every device
    // read regardless of what the request asked for; the moment the no-codec
    // read became a straight pass-through to the core (the copy this avoids is
    // a whole extra staging round trip), a get with an empty context silently
    // returned compressed bytes -- and only for pages the codec had actually
    // shrunk, so incompressible data still looked correct.
    t->context_ = clio::cte::core::Context();
    t->context_.compress_lib_ = h_->compress_lib_;
    t->context_.compress_preset_ = h_->compress_preset_;
    ClearRunCtx(t);
  }

  /** Fault `page_num` into `p` with a SYNCHRONOUS get. */
  CTP_GPU_FUN void SubmitGet(Page *p, clio::run::u64 page_num) {
    PrepareGet(p, page_num);
    Bump(h_->stat_faults_);
    // Store the future in the PAGE, not a local. The claim path publishes
    // fetching=1 before this runs, so another thread can reach AwaitFetch for
    // this page -- and AwaitFetch waits on p->get_fut. With a local future
    // that field held a stale/empty value, the wait returned immediately and
    // the caller read a page whose bytes had not arrived. That is why the
    // demand-faulting read path returned a wrong checksum while the
    // prefetching path (which does set p->get_fut) returned the right one.
    p->get_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(p->get));
    // An empty future means the submission was DISCARDED (queue missing or a
    // refused send). Waiting on it returns immediately and the stale return
    // code reads 0, so without this check the slot's previous page is served
    // as this one -- silently. That exact failure shipped once already.
    if (p->get_fut.IsNull()) {
      Bump(h_->stat_get_errors_);
    }
    p->get_fut.Wait();                    // demand faults block by definition
    // CHECK IT. A failed get leaves this slot holding the page it held before,
    // and the caller reads that as if it were the page it asked for.
    if (p->get->GetReturnCode() != 0) Bump(h_->stat_get_errors_);
    p->fetching = 0u;
  }

  /**
   * Resolve `page_num` to a resident page, faulting it in on a miss.
   *
   * On a miss with a free slot the get is synchronous; with none free, one
   * page is evicted (which flushes it if dirty) and the fault is retried --
   * the retry is what the spec calls "reperform", and it terminates because
   * EvictPages always frees a slot when any page is resident.
   */
  CTP_GPU_FUN Page *Resolve(clio::run::u64 page_num) {
    Page *p = last_page_;
    // The fast path may only skip the lock for a page that is fully resident;
    // one still being prefetched has to go through the slow path to wait.
    if (p != nullptr && p->page_num == page_num && !p->fetching) return p;
    LockBlock();
    p = ResolveLocked(page_num);
    UnlockBlock();
    last_page_ = p;
    return p;
  }

  /** Resolve with the block lock ALREADY held. */
  CTP_GPU_FUN Page *ResolveLocked(clio::run::u64 page_num) {
    Page *p = nullptr;
    // Recency is stamped HERE, on a page transition, not per element access.
    // Doing it in operator[]/at() cost a clock64() and a global write on every
    // element: a single-threaded pass over a 64 KiB page spent milliseconds in
    // bookkeeping, which swamped the fault path and made storage tier and
    // codec differences unmeasurable. LRU only needs page granularity anyway.
    p = Find(page_num);
    if (p != nullptr) {
      // Already coming: a prefetch beat us here. Wait for that transfer
      // instead of issuing a second get into the same slot.
      if (p->fetching) {
        // Count whether the prefetch actually LANDED in time, not merely that
        // one was outstanding. `fetching` stays set until someone waits, so
        // testing the flag alone reports a hit even for a transfer that
        // finished long ago -- it measured nothing and read as 100% hits at
        // every prefetch depth.
        if (p->get->fut_.is_complete_.load() != 0) {
          Bump(h_->stat_prefetch_hits_);
        } else {
          Bump(h_->stat_prefetch_late_);
        }
        AwaitFetch(p);
      }
    }
    if (p == nullptr) {
      Page *tbl = BlockPages();
      for (;;) {
        clio::run::u32 free_slot = ClaimSlotWindowLocked(page_num);
        if (free_slot != ~0u) {
          p = &tbl[free_slot];
          // Order matters: mark the slot busy and publish that BEFORE the
          // page number, so HoldPage's lock-free scan cannot see a page that
          // looks resident while its bytes are still in flight.
          p->fetching = 1u;
          __threadfence_block();
          p->gen += 1u;
          p->page_num = page_num;
          p->dirty = 0u;
          p->flushing = 0u;
          p->score = 0.0f;
          SubmitGet(p, page_num);
          break;
        }
        // Whole window pinned by in-flight transfers: brief wait, retry.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
        continue;
        EvictLocked(1);   // already holding the lock
      }
    }
    if (p != nullptr) p->last_access = Now();
    return p;
  }

  /** Apply `fn` to each RESIDENT page overlapping the element range. */
  template <typename Fn>
  CTP_GPU_FUN void ForEachResident(clio::run::u64 off, clio::run::u64 count,
                                   Fn fn) {
    if (count == 0) return;
    const clio::run::u64 first = off / h_->elems_per_page_;
    const clio::run::u64 last = (off + count - 1) / h_->elems_per_page_;
    Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].page_num == kNoPage) continue;
      if (tbl[i].page_num < first || tbl[i].page_num > last) continue;
      fn(&tbl[i]);
    }
  }
#endif  // CTP_IS_GPU_COMPILER
};

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_DEVICE_VECTOR_H_
