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
#include <clio_runtime/gpu/yield_stack.h>
#include <clio_runtime/gpu/yield_coro.h>
#include <clio_runtime/gpu/yieldable.h>
#include <clio_runtime/types.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/page.h>


namespace clio::cte::gpu_vector {

/**
 * Device-side view of the vector. Constructed on the host by
 * Vector::GetDevice() and copied into the kernel as a plain argument.
 */
/**
 * One per block: what that block faulted on, and whether it has been served.
 *
 * THE POINT OF THIS STRUCT is that the kernel does not resolve its own fault.
 * It records the page it needs and parks; the host servicer claims a slot,
 * writes back a dirty victim, fetches, publishes the page, and only then
 * clears `pending`. The kernel resumes with the page GUARANTEED resident, so
 * there is no retry path on the device -- and, more importantly, none of the
 * task-submission machinery that made the in-kernel fault path 19,400 of a
 * hold's 20,000 instructions.
 *
 * Lives in pinned host memory: the device writes the request, the host reads
 * it and writes the acknowledgement. Single writer per side, so ordinary
 * volatile accesses with a system fence are sufficient -- no atomics, which
 * this GPU cannot do on host memory anyway.
 */
struct FaultReq {
  /** Page the block is waiting for. */
  clio::run::u64 page_num;
  /** Nonzero when the hold declared write intent. */
  clio::run::u32 write;
  /** 1 = the device needs this served; 0 = resident, resume. The wait tag a
   *  parked block publishes is the address of THIS field. */
  clio::run::u32 pending;
};

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
  /**
   * 1 when every page of this vector has a dedicated, already-placed slot
   * in every block table -- the RESIDENT regime. Set by Vector::Prefetch
   * when it places the whole vector (which the identity way makes
   * possible: page p always fits slot p % ppb when npages <= slots) and
   * cleared by ClearCache.
   *
   * It is a licence to SKIP WORK, not to change behaviour. Nothing can be
   * claimed or evicted here, because no page is ever missing and so no
   * fault path ever runs -- which makes three things in the hold fast path
   * provably dead: the eviction policy's LRU bookkeeping (every thread
   * writing last_access/score to the SAME address, the contended write
   * this exists to kill), and the __threadfence + re-read of the
   * hold-vs-claim handshake, whose whole purpose is to catch a concurrent
   * claim. The pin is still taken: it costs one atomic on thread 0 and it
   * keeps the invariant true even if a fault ever did occur.
   */
  clio::run::u32 no_evict_ = 0;
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
  /** Per-block fault mailbox, `nblocks_` entries in pinned host memory.
   *  Null when the vector is driven by the legacy in-kernel fault path. */
  FaultReq *faults_ = nullptr;
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
  /**
   * ADMISSION CONTROL against the hold-while-waiting livelock.
   *
   * A block that parks inside a multi-page hold keeps its slots pinned, so
   * with enough blocks in flight every slot in the table is spoken for and
   * no claim can ever succeed -- measured directly as "48 of 48 slots
   * pinned" at a failing claim, and confirmed by the fact that the same
   * 32-slot cache that wedges at 64 blocks runs fine at 32.
   *
   * `hold_admits_` counts slots RESERVED by blocks currently inside a hold
   * set; a block may not begin one until its reservation fits under
   * `hold_admit_cap_`. Total reservations therefore never exceed the slots
   * that exist, so every admitted block can complete its working set and
   * release -- progress is structural rather than a matter of luck.
   */
  unsigned int *hold_admits_ = nullptr;
  clio::run::u32 hold_admit_cap_ = 0;
  /**
   * Whether admission is in force. Set once at construction from
   * CLIO_GV_ADMIT; [1] is unused.
   *
   * AUTO-ARMING ON THE LIVELOCK SIGNATURE WAS TRIED AND DOES NOT WORK. By
   * the time claims are failing en masse every block is already INSIDE a
   * hold set, and a block only consults this flag before entering one -- so
   * the blocks that would need throttling are past the decision, and the
   * next chunk that would honour it is never reached. Measured: with the
   * watchdog in place a 32-slot run still wedged, exactly as it did
   * without. Admission has to be armed BEFORE the run, not during it.
   */
  unsigned int *admit_armed_ = nullptr;
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
  /** Gets that read COMPLETE immediately after Send (stale flag). */
  unsigned long long *stat_early_complete_ = nullptr;
  /** [0]=resolves landing outside the block's table, [1]=last site id. */
  unsigned long long *stat_bad_slot_ = nullptr;
  /** [0]=claims that found NO usable slot, [1]=peak slots pinned at such a
   *  moment. Sampled only on failure, so the hot path pays nothing. */
  unsigned long long *stat_claim_fail_ = nullptr;
  /** Writebacks that came back with a NON-ZERO return code. AwaitPut cleared
   *  `flushing` without ever reading it, so a put that failed -- a full tier
   *  being the obvious way -- marked the page CLEAN and dropped its contents
   *  on the floor. With no spill tier configured, 14 of 16 MB vanished this
   *  way and the run still reported success until the checksum disagreed. */
  unsigned long long *stat_put_errors_ = nullptr;
  unsigned long long *stat_pf_dropped_ = nullptr;  // async hints dropped
  /** CLIO_GV_TRACE_PUT_ERRORS: device-print each failed writeback's page and
   *  return code. Off by default; a failed put is otherwise only a counter. */
  unsigned int trace_put_errors_ = 0u;
  // Optional per-page fault counter (CLIO_FAULT_HIST): distinguishes churn
  // (same pages refaulted) from coverage (more unique pages) between the
  // bistable attractors. Device array of one u32 per vector page.
  unsigned int *fault_hist_ = nullptr;
  // CLIO_MOE_VERIFY in-vivo counters: after computing an item the kernel
  // re-probes its page; lost = the held pointer no longer resolves to the
  // same slot (evicted/recycled mid-read).
  unsigned long long *stat_verify_ok_ = nullptr;
  unsigned long long *stat_verify_lost_ = nullptr;
  // The zero-copy device-tier map that used to live here is GONE, by
  // decision, not by accident: it aliased SHARED CTE tier memory from GPU
  // kernels (reads in place, and writes to mapped dirty pages mutated tier
  // bytes directly), which put every tier mover -- organizer, target
  // evacuation, eviction -- in a silent race with resident kernels. The
  // vector's memory is private; bytes cross the CTE boundary only through
  // orchestrated transfers. The fits-in-VRAM fast path is a private cache
  // sized to the data (slots >= pages), seeded once -- not a pointer into
  // somebody else's tier.

  // ---- SPILLED encoded pages -------------------------------------------
  // A page the tier map does not cover (its blob is on a host tier) still
  // has to be decoded. Routing it through the compressor costs THREE task
  // round trips, a codec slot and a stream sync per page -- measured at
  // ~1.6ms against ~40us for a spilled raw page, which is why compression
  // LOSES whenever it spills even though it moves 8x fewer bytes.
  //
  // These let the fault fetch the STORED bytes with one raw get (addressed
  // at the core pool, so the compressor never sees it) into per-slot device
  // scratch, and decode in the faulting warp exactly like a mapped page.
  /** Stored (post-codec) size of every page, or ~0 when unknown. */
  const unsigned long long *stored_size_ = nullptr;
  /** CTE CORE pool -- bypasses the compressor interposer on the fault. */
  clio::run::PoolId core_pool_id_;
  /**
   * Per-BLOCK count of outstanding transfers (puts + gets + one per live
   * multi). Not a correctness structure: the authoritative answer is always
   * the slot scan. It exists so the scans can be SKIPPED when nothing is in
   * flight -- with the private cache sized to the data (slots == pages) the
   * steady state has zero transfers, and the O(slots) reap/vote scans per
   * page hold were the entire measured pass (1254 ms of a 1067 ms/215 ms
   * story; see HoldPageYield). Mutated only under the block lock (or by the
   * lock-holding thread), read lock-free by the votes.
   */
  unsigned int *xfer_cnt_ = nullptr;
  /** nslots * page_bytes_ of device scratch, one slab per cache slot, for
   *  the compressed image while it is being decoded. */
  char *cscratch_ = nullptr;
  /** CLIO_DECODE_TRACE: printf each in-kernel decode failure (diagnosis). */
  unsigned int decode_trace_ = 0;
};

/** One entry of a batched rescore: WHICH page, and how much it matters.
 *  score == 1.0f is the maximum preference and means "make it RESIDENT":
 *  the entry rides the batched fetch. Any other score is a CTE placement
 *  hint for the page's blob (tier steering), batched through one
 *  PodMultiScoreTask. */
struct PageScore {
  clio::run::u64 page;
  float score;
};

/** White-box accessor for the machinery tests and benches; production code
 *  must use the public tiers. */
struct DeviceVectorTestAccess;

#if CTP_IS_GPU_COMPILER
/**
 * A HELD RUN of one page: what `co_await vec.HoldPage(off, count, write)`
 * returns. The page stays pinned -- unevictable, its bytes stable -- for
 * exactly this object's lifetime; the destructor unholds. Access flows
 * through the guard, so "pointer valid only while held" is structural:
 *
 *   Held<float> h = co_await vec.HoldPage(off, count);
 *   h[off + i]                      // element access, GLOBAL offsets
 *   for (float &x : h) ...          // iteration over the held elements
 *   h.begin_off(), h.end_off()      // the global range [begin, end) covered
 *   h.run()                         // == end_off() - begin_off()
 *   h.ptr()                         // raw base pointer, same lifetime
 *   h.release()                     // leak the pin deliberately (persistent
 *                                   // published pages); pair with a later
 *                                   // UnholdPage(page_num)
 *
 * ADOPTS the pin the resolve took (it does not pin again); move-only, so
 * exactly one guard ever owns a given pin. Per-thread, like the hold.
 */
template <typename T>
class Held {
 public:
  Held() = default;
  Held(const Held &) = delete;
  Held &operator=(const Held &) = delete;
  CTP_GPU_FUN Held(Page *page, T *data, clio::run::u64 begin,
                   clio::run::u64 run, bool pinned = true)
      : page_(page), data_(data), begin_(begin), run_(run),
        pinned_(pinned) {}
  CTP_GPU_FUN Held(Held &&o) noexcept { Steal(o); }
  CTP_GPU_FUN Held &operator=(Held &&o) noexcept {
    if (this != &o) {
      Unhold();
      Steal(o);
    }
    return *this;
  }
  CTP_GPU_FUN ~Held() { Unhold(); }

  /** Global element offsets this hold covers: [begin_off, end_off). */
  CTP_GPU_FUN clio::run::u64 begin_off() const { return begin_; }
  CTP_GPU_FUN clio::run::u64 end_off() const { return begin_ + run_; }
  CTP_GPU_FUN clio::run::u64 run() const { return run_; }
  /** Iteration over the held elements. */
  CTP_GPU_FUN T *begin() const { return data_; }
  CTP_GPU_FUN T *end() const { return data_ + run_; }
  CTP_GPU_FUN T *ptr() const { return data_; }

  /** Diagnostic: the page this guard's SLOT currently holds. A guard whose
   *  slot does not hold the page the caller asked for is reading a frame
   *  that belongs to someone else. */
  /** Diagnostic: {pins, fetching} of this guard's slot, packed. */
  /** Diagnostic: the page of the last get SUBMITTED into this guard's own
   *  frame. Table-independent -- it reads the guard's slot directly, so it
   *  is valid no matter which block table the slot belongs to. */
  /** Diagnostic: this guard's slot object itself. */
  CTP_GPU_FUN const void *dbg_slot_ptr() const { return page_; }

  CTP_GPU_FUN clio::run::u64 dbg_slot_fetched() const {
    return page_ == nullptr
               ? ~static_cast<clio::run::u64>(0)
               : *reinterpret_cast<volatile clio::run::u64 *>(
                     &page_->dbg_get_page);
  }

  CTP_GPU_FUN clio::run::u32 dbg_slot_state() const {
    if (page_ == nullptr) return ~0u;
    const clio::run::u32 pins =
        *reinterpret_cast<volatile clio::run::u32 *>(&page_->pins);
    const clio::run::u32 f =
        *reinterpret_cast<volatile clio::run::u32 *>(&page_->fetching);
    return (pins << 8) | (f & 0xffu);
  }

  CTP_GPU_FUN clio::run::u64 dbg_slot_page() const {
    return page_ == nullptr
               ? ~static_cast<clio::run::u64>(0)
               : *reinterpret_cast<volatile clio::run::u64 *>(&page_->page_num);
  }
  /** Element access by GLOBAL offset; valid within [begin_off, end_off). */
  CTP_GPU_FUN T &operator[](clio::run::u64 off) const {
    return data_[off - begin_];
  }
  CTP_GPU_FUN explicit operator bool() const { return page_ != nullptr; }

  /** Keep the page pinned past this guard's death (persistent published
   *  pages). Pairs with exactly one later UnholdPage(page_num).
   *  @return the page number left pinned. */
  CTP_GPU_FUN clio::run::u64 release() {
    const clio::run::u64 pn = page_ != nullptr ? page_->page_num : kNoPage;
    page_ = nullptr;
    return pn;
  }

 private:
  CTP_GPU_FUN void Unhold() {
    if (page_ != nullptr && pinned_) {
      // Pins are PER-BLOCK (see AcquireHoldPin): every thread carries a
      // guard, but only thread 0's decrements the counter.
      if (threadIdx.x == 0) {
        atomicSub((clio::run::u32 *) &page_->pins, 1u);
      }
      page_ = nullptr;
    }
  }
  CTP_GPU_FUN void Steal(Held &o) {
    page_ = o.page_;
    data_ = o.data_;
    begin_ = o.begin_;
    run_ = o.run_;
    pinned_ = o.pinned_;
    o.page_ = nullptr;
  }

  Page *page_ = nullptr;
  T *data_ = nullptr;
  clio::run::u64 begin_ = 0;
  clio::run::u64 run_ = 0;
  /** False when the hold took no pin because none was needed -- the
   *  resident regime, where nothing can be claimed or evicted (see
   *  VecHeader::no_evict_). The guard is still the access; it simply has
   *  no reference count to give back. Pin atomics from every block onto
   *  the SAME shared Page serialize through L2 and were measured to be
   *  the dominant cost of a resident hold. */
  bool pinned_ = true;
};
#endif  // CTP_IS_GPU_COMPILER (Held)


/*
 * ============================== THE API ==================================
 * This is the WHOLE public surface. Four verbs plus the guard they hand out:
 *
 *   Held<T> h = co_await HoldPage(off, count, w)   access, faulting as
 *                                            needed; w=true declares stores.
 *                                            The guard IS the pin and the
 *                                            element access: h[global],
 *                                            begin()/end() iteration,
 *                                            begin_off()/end_off()/run(),
 *                                            ptr(); destructor unholds;
 *                                            release() leaks deliberately
 *                                            (pair with UnholdPage(pn)).
 *   FlushAsync(off, count)                   start writeback, no wait
 *   co_await AwaitFlush()                    all started flushes durable
 *   RescorePagesBatchedAsync(n, gen)         score==1 -> make resident
 *                                              (absent pages only -- never
 *                                              fetches over a dirty copy),
 *                                            score==0 -> drop from cache
 *                                              (clean pages only -- flush
 *                                              explicitly first; drops
 *                                              never write back),
 *                                            else -> batched CTE rescore
 *   co_await AwaitRescore()                  both batches settled
 *
 * (plus size(), and UnholdPage(page_num) for pins a release() left behind.)
 * Kernels are C++20 device COROUTINES -- there are no macro or blocking
 * transports: a resident page completes the co_await on the fast path with
 * no suspension, so the coro verb IS the fast path.
 *
 * There is NO explicit pinning and no manual dirty marking. THE HOLD IS THE
 * PIN: the held page cannot be evicted or re-tenanted while any guard on it
 * lives, so everything a guard exposes is valid for exactly the guard's
 * lifetime, and a multi-page working set is simply several live guards.
 * Residency beyond a hold is Rescore(1.0); invalidation is Rescore(0.0).
 * Everything else is machinery: private, reachable in tests only through
 * DeviceVectorTestAccess.
 * ========================================================================
 */template <typename T>
class DeviceVector {
  friend struct DeviceVectorTestAccess;

 public:
  DeviceVector() = default;

  /** Everything shared: geometry, submit state, counters. */
  const VecHeader *h_ = nullptr;

#if CTP_IS_GPU_COMPILER
  // ---- device API -----------------------------------------------------

 private:
  /**
   * Pin the page holding `off` and report how far you can walk from there.
   *
   * This is the indexing fast path. It resolves the page ONCE -- faulting it in
   * if needed -- caches it in last_page_, and returns how many elements can be
   * read or written sequentially from `off` before leaving that page. The
   * caller loops over that run using operator[], which then does no
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
  CTP_GPU_FUN clio::run::u64 ProbeHold(clio::run::u64 off,
                                       clio::run::u64 count,
                                       bool write = false) {
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
        // The d candidates, same set the claim uses. NOTE: there is no
        // empty-slot early-out any more. Under open addressing an empty slot
        // proved absence because a chain has no holes; with d-left hashing the
        // ways are INDEPENDENT, so an empty way says nothing about the others
        // and breaking there would report false misses.
        const clio::run::u32 nw = Ways();
        for (clio::run::u32 w = 0; w < nw; ++w) {
          const clio::run::u32 i = WaySlot(pn, w);
          if (tbl[i].page_num == pn && !tbl[i].fetching) {
            p = &tbl[i];
            break;
          }
        }
      }
      if (p == nullptr) {
        // A real miss. NEVER fetch synchronously from inside the kernel --
        // that is the resident-kernel deadlock (the host copy that would
        // service it cannot launch behind us). Report the miss; the coro
        // verb parks the block and retries once the page lands.
        return 0;
      }
      // Skipped entirely in the resident regime: nothing evicts, so the
      // policy that would read these never runs, and these two stores are
      // made by EVERY thread to one address (see VecHeader::no_evict_).
      if (h_->no_evict_ == 0u) {
      // Recency must be stamped on a HIT as well, not just in ResolveLocked.
      // Once the lock-free path started serving hits, ResolveLocked stopped
      // running for them, so re-touching a resident page no longer refreshed
      // it and LRU evicted the most recently used page instead of the oldest.
      // Still once per page TRANSITION, never per element.
      p->last_access = Now();
      p->score += 1.0f;
      }
      last_page_ = p;
      last_pn_ = pn;
    }
    DbgCheckInTable(p, 1);
    // EVERY successful hold takes one pin, which exactly one Held guard
    // adopts and releases; the fast path pins too, or a re-hold's guard
    // would release a pin nobody took.
    //
    // ... EXCEPT in the resident regime, where no claim or eviction can
    // ever run and the pin therefore protects nothing. It is not free:
    // every block holding a shared page does an atomicAdd and an
    // atomicSub on the SAME address, and those serialize through L2. With
    // the pair loop deleted the force pass of a 256k-atom MD step cost
    // 245 ms against 252 with it -- i.e. the arithmetic was 3% and this
    // was most of the rest.
    if (h_->no_evict_ == 0u) AcquireHoldPin(p);
    // The prober's half of the FreeVictimSlot handshake: publish the pin,
    // THEN re-read the slot. A claim that read pins == 0 before the pin
    // landed has published its takeover by now (its fence is between the
    // two), so one of us is guaranteed to see the other. A slot that
    // changed hands in the match->pin window is a MISS, never a hold --
    // thread 0's pin is the one that counts (AcquireHoldPin), and its miss
    // fails the caller's all-or-nothing vote for the whole block.
    // ... and skipped in the resident regime, where there is no claim to
    // race against (see VecHeader::no_evict_): a device-scope fence per
    // hold is not free.
    if (h_->no_evict_ == 0u) {
      __threadfence();
      if (*(volatile clio::run::u64 *)&p->page_num != pn ||
          *(volatile clio::run::u32 *)&p->fetching != 0u) {
        ReleaseHoldPin(p);
        last_page_ = nullptr;
        return 0;
      }
      // A WRITE must not proceed into a page whose writeback is already in
      // flight. SubmitPut clears `dirty` as it sends, so a store landing
      // after the put's DMA would leave the page marked CLEAN with the
      // update only in the frame -- and the next drop discards it. This is
      // the prober's half of the eviction handshake (StartEvictionAsync
      // has the other); the caller waits for the flush and retries.
      if (write && (*(volatile clio::run::u32 *)&p->flushing != 0u ||
                    *(volatile clio::run::u32 *)&p->evicting != 0u)) {
        ReleaseHoldPin(p);
        last_page_ = nullptr;
        return 0;
      }
    }
    // Write intent is declared HERE, at the hold, not inferred per element:
    // operator[] is side-effect-free for reads and writes alike, so a page
    // held without write=true is dropped clean at eviction no matter what
    // was stored through it.
    // One writer, not all of them: `dirty` is idempotent, and the vote
    // barrier below publishes thread 0's store to the whole block.
    if (write && threadIdx.x == 0) p->dirty = 1u;
    const clio::run::u64 within = IndexIn(off, p);
    const clio::run::u64 left = h_->elems_per_page_ - within;
    return (count < left) ? count : left;
  }

 public:

  /**
   * Release a hold on a SPECIFIC page of this block's table -- for holds
   * whose view did not survive to the release point (one coroutine holds and
   * publishes, a later one releases). Pairs with exactly one earlier
   * HoldPage of that page by whoever published it. No-op when the page is
   * not resident or carries no pin.
   */
  CTP_GPU_FUN void UnholdPage(clio::run::u64 page_num) {
    // Pins are PER-BLOCK (see AcquireHoldPin): one decrement releases the
    // block's hold, however many threads call this.
    if (threadIdx.x != 0) return;
    LockBlock();
    Page *p = Find(page_num);
    if (p != nullptr && p->pins != 0u) {
      atomicSub((clio::run::u32 *) &p->pins, 1u);
    }
    UnlockBlock();
  }

 private:
  /** THE HOLD IS THE PIN: every successful internal hold acquires exactly
   *  one pin, and exactly one Held guard adopts it -- the guard's destructor
   *  (or UnholdPage(page_num) after a release()) is the only release.
   *  Per-thread: block-collective holds put blockDim.x pins on the page and
   *  each thread's guard takes one back off. */
  /** PER-BLOCK, not per-thread: holds are block-collective, so one pin per
   *  block carries the same meaning as 256 -- and 256 threads atomically
   *  bumping ONE address per hold, across every block sharing the table,
   *  measured as most of the fast-path hold's cost (~58us/hold at 64
   *  blocks). Thread 0 pins; the thread-0 lane of the guards unpins. */
  CTP_GPU_FUN void AcquireHoldPin(Page *p) {
    if (threadIdx.x == 0) {
      atomicAdd((clio::run::u32 *) &p->pins, 1u);
    }
  }

  /** Undo one AcquireHoldPin. Used when a probe SUCCEEDED on this thread but
   *  the block's all-or-nothing vote abandoned the round: the pin was taken
   *  in ProbeHold and no Held guard will ever adopt it, so it must be given
   *  back here or the page is pinned forever. `last_page_` is the page that
   *  probe pinned -- ProbeHold updates it on every success. */
  CTP_GPU_FUN void ReleaseHoldPin(Page *p) {
    if (threadIdx.x == 0) {
      atomicSub((clio::run::u32 *) &p->pins, 1u);
    }
  }

 public:


 private:
  /**
   * PROBE-ONLY hold: the lock-free resident lookup and nothing else.
   *
   * Returns null on a miss instead of falling into the locked fault path, so
   * a caller with a BETTER recovery than a scalar fault (a batched fetch of
   * a whole expert, say) can take it.
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
      const clio::run::u32 nw = Ways();
      for (clio::run::u32 w = 0; w < nw; ++w) {
        const clio::run::u32 i = WaySlot(pn, w);
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
      last_pn_ = pn;
    }
    DbgCheckInTable(p, 2);
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
    if (slot == nullptr) return true;   // direct-mapped: immutable
    __threadfence();
    return *(volatile clio::run::u32 *) &slot->gen == gen;
  }


 public:
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
    // A PURE FUNCTION OF `off` -- never of the slot.
    //
    // The fallback arm used to read p->page_num, and that is shared mutable
    // state: when it disagreed with the page `off` belongs to, the returned
    // offset was wrong by (PageOf(off) - page_num) WHOLE PAGES, so the guard
    // handed the caller a pointer into a different frame entirely. The
    // reader then saw zeros (a frame nothing had fetched into) or another
    // page's bytes (one that was in use), with the fetch, the rc, the pin
    // and the slot all perfectly correct -- the out-of-core read corruption.
    //
    // The power-of-two arm was always immune because `off & page_mask_` does
    // not consult the slot, which is exactly why every gate (64KB pages)
    // passed while a 96KB-page run corrupted: 24576 elements is not a power
    // of two, so only that configuration took the racy arm.
    (void) p;
    return h_->page_shift_ ? (off & h_->page_mask_)
                           : (off - PageOf(off) * h_->elems_per_page_);
  }

 private:
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
      tbl[i].user_score = 0.0f;
      tbl[i].has_user = 0u;
    }
    EvictLocked(h_->pages_per_block_);
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      tbl[i].last_access = 0;
    }
    UnlockBlock();
    last_page_ = nullptr;   // the per-thread cache now points at a free slot
  }

  /**
   * D-LEFT HASH INDEX. A page lives in exactly one of `Ways()` candidate
   * slots, so both lookup and claim are O(d) -- INDEPENDENT of the cache size.
   *
   * This replaces a bounded-window open-addressing scheme. That scheme made a
   * HIT cheap (one probe at pn % ppb) but left a MISS scanning the whole
   * table, and in the out-of-core regime a miss is the common case: measured
   * 99.98% of accesses at 1/16 coverage. Since a Page is ~200 bytes, each
   * probed slot is its own cache line, so a miss cost ppb cache-line reads --
   * 12288 of them at the top of a coverage sweep. Time therefore grew with the
   * cache size while the cache bought almost nothing (128x more slots removed
   * 2.3% of faults, and runtime rose 165%).
   *
   * D-LEFT, not d independent hashes over the whole table: way `w` owns its
   * own disjoint segment, so the d candidates are ALWAYS distinct slots.
   * Independent hashes can collide with each other, silently reducing
   * associativity -- worst exactly where the table is small.
   *
   * Segments are [ppb*w/d, ppb*(w+1)/d), which tile [0, ppb) exactly and
   * differ in size by at most one, so every slot is reachable and none is
   * stranded.
   */
  static constexpr clio::run::u32 kMaxWays = 8;

  /**
   * Victim selection among the d candidates.
   *
   *   kEvictLru   least-recently-used. PATHOLOGICAL ON CYCLIC SCANS: a page
   *               must survive a whole pass to be reused, so it is always the
   *               least-recently-used one and is always the one evicted.
   *               MEASURED on weights (4096MB, cyclic re-read), hit rate at
   *               1/16, 1/4, 1/2, 3/4 coverage: 0.0%, 0.0%, 0.1%, 7.2% --
   *               against ~75% that 3/4 coverage should allow.
   *   kEvictFreq  touch-count score with linear aging. The original policy,
   *               chosen precisely because "with the working set larger than
   *               the cache, LRU converges to ~0% hit rate".
   *   kEvictRand  uniform among the candidates (reservoir sampling). The
   *               textbook answer for scans: expected hit rate tracks coverage
   *               rather than collapsing to zero.
   *
   * Selected at compile time so the comparison costs a rebuild, not a
   * re-plumb. `score` is aged under every policy so the frequency history
   * stays live and switching is a one-line change.
   */
  static constexpr int kEvictLru = 0;
  static constexpr int kEvictFreq = 1;
  static constexpr int kEvictRand = 2;
  /**
   * kEvictUser -- evict by the score the KERNEL set through RescorePage.
   *
   * The caller usually knows the reuse distance the cache cannot infer: a
   * stencil knows which planes its window still needs, a streaming pass knows
   * a page is dead the moment it has been consumed. This policy makes that
   * knowledge authoritative instead of advisory.
   *
   * Ranking, worst victim first:
   *   1. pages with NO user score          (nobody vouched for them)
   *   2. pages with a LOWER user score
   *   3. LRU as the tie-break within each group
   *
   * Un-hinted pages go first ON PURPOSE. The alternative -- treating "no
   * hint" as a middling score -- lets an un-hinted page outrank one the caller
   * explicitly released, which is the opposite of what a release means. It
   * does mean a workload that hints NOTHING degenerates to LRU, so this policy
   * is only meaningful where the benchmark actually calls RescorePage.
   */
  static constexpr int kEvictUser = 3;
#ifndef CLIO_GV_EVICT_POLICY
#define CLIO_GV_EVICT_POLICY 0
#endif
  static constexpr int kEvictPolicy = CLIO_GV_EVICT_POLICY;

  /** Cheap in-kernel LCG, for kEvictRand only. */
  CTP_GPU_FUN clio::run::u32 Rand32(clio::run::u64 seed) const {
    clio::run::u64 x = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    x ^= x >> 33;
    return (clio::run::u32) (x >> 16);
  }

  /**
   * Should candidate `cand` replace the current best victim `cur`?
   *
   * `k` is the candidate's index in the probe order, which kEvictRand needs
   * for reservoir sampling (pick each new candidate with probability 1/(k+1),
   * giving a uniform choice in a single pass).
   */
  CTP_GPU_FUN bool PreferVictim(const Page &cand, const Page &cur,
                                bool have_cur, clio::run::u32 k,
                                clio::run::u64 pn) const {
    if (!have_cur) return true;
    // A USER hint outranks every policy: it is the Rescore verb's explicit
    // intent, not an access-pattern heuristic, and it must mean the same
    // thing on every eviction path. Historically only the locked full-scan
    // honored the hinted score; this path ran pure LRU and evicted a
    // score-100 page as merely "oldest". Un-hinted victims first; the
    // policy orders within the class.
    if (cand.has_user != cur.has_user) return cand.has_user < cur.has_user;
    if (cand.has_user && cur.has_user && cand.user_score != cur.user_score) {
      return cand.user_score < cur.user_score;
    }
    if (kEvictPolicy == kEvictFreq) {
      return cand.score < cur.score ||
             (cand.score == cur.score && cand.last_access < cur.last_access);
    }
    if (kEvictPolicy == kEvictRand) {
      return (Rand32(pn + k * 0x9E3779B9ULL) % (k + 1u)) == 0u;
    }
    if (kEvictPolicy == kEvictUser) {
      // Un-hinted before hinted; then lower user score; then LRU.
      if (cand.has_user != cur.has_user) return cand.has_user < cur.has_user;
      if (cand.has_user && cand.user_score != cur.user_score) {
        return cand.user_score < cur.user_score;
      }
      return cand.last_access < cur.last_access;
    }
    return cand.last_access < cur.last_access;   // kEvictLru
  }

 public:
  /** Number of hash ways: min(kMaxWays, pages_per_block). CROSS and PUBLIC:
   *  the host Vector's Prefetch places pages with the SAME function the
   *  device looks up with; a host-side mirror would be a drift bug waiting
   *  to happen. */
  static CTP_CROSS_FUN clio::run::u32 Ways(clio::run::u32 ppb) {
    return ppb < kMaxWays ? ppb : kMaxWays;
  }
  CTP_GPU_FUN clio::run::u32 Ways() const {
    return Ways(h_->pages_per_block_);
  }

  /**
   * The slot page `pn` occupies in way `w`.
   *
   * MUST be a pure function of (pn, w, ppb): lookup, claim, AND the host's
   * Prefetch all derive the candidate set from it, and any disagreement
   * makes a resident page invisible -- which does not fail loudly, it
   * re-fetches the page into a second slot and leaves two slots holding it.
   */
  static CTP_CROSS_FUN clio::run::u32 WaySlot(clio::run::u64 pn,
                                              clio::run::u32 w,
                                              clio::run::u32 ppb32) {
    const clio::run::u64 ppb = (clio::run::u64) ppb32;
    // WAY 0 IS THE IDENTITY WAY: pn % ppb, over the whole table. This is
    // what makes the resident regime a THEOREM instead of a lottery: when
    // npages <= slots, every page's identity slot is distinct, so host
    // Prefetch (which tries ways in order into an empty table) places EVERY
    // page and the kernel cannot fault. Before this, placement was
    // best-effort d-left hashing, and a 96-page vector in a 98-slot table
    // silently left way-collided pages to demand-fault mid-kernel. Out of
    // core, mod is collision-free over any contiguous window of <= ppb
    // pages -- the working-set shape the range/stencil holds produce -- and
    // pathological strides fall back to the hashed ways below.
    if (w == 0) {
      return (clio::run::u32) (pn % ppb);
    }
    const clio::run::u64 d = (clio::run::u64) Ways(ppb32);
    const clio::run::u32 base = (clio::run::u32) ((ppb * w) / d);
    const clio::run::u32 end = (clio::run::u32) ((ppb * (w + 1)) / d);
    const clio::run::u32 span = end - base;   // >= 1, since d <= ppb
    // splitmix64 finalizer, seeded per way. The seed must differ per way or
    // every way maps to the same offset within its segment and the index
    // degenerates to 1-way.
    clio::run::u64 x = pn + 0x9E3779B97F4A7C15ULL * (clio::run::u64) (w + 1);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return base + (clio::run::u32) (x % (clio::run::u64) span);
  }
  CTP_GPU_FUN clio::run::u32 WaySlot(clio::run::u64 pn,
                                     clio::run::u32 w) const {
    return WaySlot(pn, w, h_->pages_per_block_);
  }

 private:

  /**
   * Claim one of `pn`'s d candidate slots.
   *
   * Caller must hold the block lock. Returns the claimed slot, its entry
   * reset and NOT yet marked fetching; ~0u if no candidate can be taken.
   */
  CTP_GPU_FUN clio::run::u32 ClaimSlotWindowLocked(clio::run::u64 pn) {
    clio::run::u32 s = ClaimSlotWindowOnceLocked(pn);
    if (s == ~0u) {
      // Every candidate is mid-transfer. If any are fetching=2 their batch
      // may long since have LANDED — nothing settles an async batch until a
      // toucher hits one of ITS pages, so the candidate set can fill with
      // landed-but-unsettled pages and starve every claimer whose page is not
      // among them (observed: kernel spun forever, CPU idle). Settle and
      // rescan.
      SettleBatchLocked();
      s = ClaimSlotWindowOnceLocked(pn);
    }
    return s;
  }

  /**
   * Free a victim slot with the DEKKER HANDSHAKE against the lock-free hold
   * fast path. ProbeHold pins WITHOUT the block lock -- that is what makes
   * it fast -- so a "pins == 0" read moments earlier proves nothing: a hold
   * can land in the window between that read and the takeover (the window
   * spans the caller's whole victim scan, and on the prober's side two
   * __syncthreads of its vote). Both sides therefore publish-then-check
   * with fences: the prober publishes its pin and re-reads page_num;
   * this side publishes the takeover and re-reads pins. The fences make
   * it impossible for BOTH to miss the other, so exactly one backs off.
   * (This was the intermittent reneighbour-step corruption: a shared-shard
   * claim re-tenanted a slot a sibling block had just fast-path held, and
   * the sibling walked the new tenant's bytes with the old page's ranges.)
   * @return true when the slot is now free; false when a holder raced in --
   *         the slot is restored untouched and must not be taken.
   */
  CTP_GPU_FUN bool FreeVictimSlot(Page *p) {
    const clio::run::u64 prev = p->page_num;
    p->page_num = kNoPage;
    __threadfence();
    if (*(volatile clio::run::u32 *)&p->pins != 0u) {
      p->page_num = prev;
      __threadfence();
      return false;
    }
    return true;
  }

  CTP_GPU_FUN clio::run::u32 ClaimSlotWindowOnceLocked(clio::run::u64 pn) {
    Page *tbl = BlockPages();
    const clio::run::u32 nw = Ways();
    clio::run::u32 victim = ~0u;
    for (clio::run::u32 w = 0; w < nw; ++w) {
      const clio::run::u32 i = WaySlot(pn, w);
      Page &pgi = tbl[i];
      // A pinned page is being read through a raw pointer right now. Recycling
      // it is silent corruption, so it is not a candidate at any score.
      if (pgi.pins != 0u) continue;
      if (pgi.page_num == kNoPage) {
        // A FREED slot can still have I/O IN FLIGHT targeting its frame
        // (e.g. a batch settle freeing a failed record whose sibling copies
        // have not retired, or any path that clears page_num early). The
        // victim branch below has always checked fetching/flushing; this
        // fast-path did not, and handing such a frame to a new page let the
        // old transfer land on top of the new page's content. Caught live:
        // page 33's freshly written frame containing page 57's bytes -- a
        // crossed copy, not stale data -- the intermittent rebuild
        // corruption (~25% of runs).
        if (pgi.fetching || pgi.flushing) continue;
        return i;
      }
      // Score is still AGED here so the frequency history stays live and the
      // policy can be switched back without re-plumbing. The victim choice
      // below is LRU by request; the previous policy was frequency-based, for
      // this reason, which is worth keeping in view:
      //
      //   "FREQUENCY, not recency. With the working set larger than the cache,
      //    LRU converges to ~0% hit rate: every page's reuse distance exceeds
      //    capacity, so everything is evicted before its next use.
      //    Routed-expert traffic is SKEWED -- hot experts recur -- and a
      //    touch-count score keeps them resident while cold ones churn."
      //
      // Aging must be LINEAR and slow. Halving per claim scan was ~0.5^23
      // per token (a table sees ~23 claims/token): every score decayed to
      // zero between touches, eviction degenerated to random, and a batch's
      // own just-fetched pages self-evicted at score ~0.004.
      pgi.score = fmaxf(pgi.score - 0.02f, 0.0f);
      // `dirty` belongs in this test. This path DROPS its victim outright --
      // it does not write it back -- on the stated assumption that "weights
      // are read-only here, dirty is never set on this path". That holds for
      // an inference table and for nothing else: the same claim serves the
      // prefetch and the batched fetch, both reachable from a read-WRITE
      // workload, and dropping a dirty page there discards the only copy of
      // its writes. Skipping dirty victims costs at worst a declined claim,
      // which the caller already handles; the alternative is silent loss.
      if (!pgi.fetching && !pgi.flushing && !pgi.rescoring && !pgi.dirty &&
          PreferVictim(pgi, tbl[victim == ~0u ? i : victim],
                       victim != ~0u, w, pn)) {
        victim = i;
      }
    }
    if (victim == ~0u) {
      // No CLEAN victim: DECLINE. Dropping a dirty page here would discard
      // the only copy of its writes.
      //
      // Deliberately does NOT start a writeback to free one up. That was
      // tried and it thrashes: this claim runs on every fault, so it evicts
      // pages the block is still working through, and the seed phase stopped
      // converging entirely (200000 rounds, cap reached, where it had taken
      // 43ms). Whoever needs room must arrange it with knowledge of what it
      // still needs -- HoldPageCoro's StartEvictionAsync picks one victim
      // per fault and waits for it. Declining here just means the caller
      // falls back to that demand path.
      return ~0u;
    }
    // The victim is CLEAN (guaranteed above), so dropping it is a metadata
    // write and loses nothing -- unless a lock-free hold raced in since the
    // pins check above, in which case the handshake declines the claim.
    if (!FreeVictimSlot(&tbl[victim])) return ~0u;
    Bump(h_->stat_evicts_);
    return victim;
  }

  /**
   * Free one of `page_num`'s d CANDIDATE SLOTS, for the blocking path.
   *
   * EvictLocked cannot serve that caller: it scans the whole table, so it
   * frees slots the claim will never probe and the claim/evict pair can miss
   * each other forever. Both must work on the SAME candidate set -- that
   * agreement is the invariant, and violating it is what deadlocked the
   * kernel before.
   *
   * Returns false only when every candidate is mid-FETCH and so cannot be
   * taken; the caller must then wait rather than retry immediately.
   */
  CTP_GPU_FUN bool EvictWindowLocked(clio::run::u64 page_num) {
    Page *tbl = BlockPages();
    const clio::run::u32 ppb = h_->pages_per_block_;
    const clio::run::u32 nw = Ways();
    clio::run::u32 victim = ppb;
    clio::run::u32 fetching_slot = ppb;
    for (clio::run::u32 w = 0; w < nw; ++w) {
      const clio::run::u32 i = WaySlot(page_num, w);
      if (tbl[i].pins != 0u) continue;   // in use through a raw pointer
      if (tbl[i].page_num == kNoPage) return true;   // room already
      // A page mid-fetch is promised to that transfer: taking its slot would
      // let the copy land under a different page.
      if (tbl[i].fetching) {
        if (fetching_slot == ppb) fetching_slot = i;
        continue;
      }
      // `rescoring` is NOT skipped, for the reason spelled out in
      // StartEvictionAsync: the flag is sticky, so skipping it here would make
      // rescored pages permanently unevictable and starve this path.
      if (PreferVictim(tbl[i], tbl[victim == ppb ? i : victim],
                       victim != ppb, w, page_num)) {
        victim = i;
      }
    }
    if (victim == ppb) {
      // Nothing but in-flight fetches. Waiting on one IN HERE is the only way
      // forward: the caller holds the block lock, so ReapFetched cannot run
      // and the flag would never clear on its own.
      if (fetching_slot != ppb) {
        AwaitFetch(&tbl[fetching_slot]);
        return true;                  // now resident, hence evictable
      }
      return false;
    }
    Page *p = &tbl[victim];
    if (p->dirty || p->flushing) {
      // Same handshake as StartEvictionAsync, and MORE important here
      // because AwaitPut BLOCKS: the victim was chosen with pins == 0, but
      // ProbeHold pins without the lock and SubmitPut clears `dirty` as it
      // sends, so a writer landing after the DMA would leave the page
      // marked CLEAN with its update only in the frame.
      p->evicting = 1u;
      __threadfence();
      if (*(volatile clio::run::u32 *)&p->pins != 0u) {
        p->evicting = 0u;
        return false;              // a writer owns it; caller must wait
      }
      SubmitPut(p);
      AwaitPut(p);
      p->evicting = 0u;
    }
    if (!FreeVictimSlot(p)) {
      // A lock-free hold landed during the scan or the writeback: the page
      // stays resident (and is clean now), and the caller must wait.
      return false;
    }
    p->dirty = 0u;
    p->flushing = 0u;
    Bump(h_->stat_evicts_);
    return true;
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
        if (tbl[i].pins != 0u) continue;   // held: the hold is the pin
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
        p->evicting = 1u;          // see EvictWindowLocked for why
        __threadfence();
        if (*(volatile clio::run::u32 *)&p->pins != 0u) {
          p->evicting = 0u;
          continue;                // a writer owns it; leave it alone
        }
        SubmitPut(p);
        AwaitPut(p);
        p->evicting = 0u;
      }
      if (!FreeVictimSlot(p)) continue;   // holder raced in; rescan skips it
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
  /**
   * Mark every RESIDENT page of [off, off+count) dirty, so a later
   * BeginFlush/FlushBlockBatched writes it back.
   *
   * Exists for callers that fill a page's bytes OUTSIDE this view -- the
   * two-phase pair kernel writes forces through raw pinned-page pointers in
   * a plain (non-coroutine) kernel, or written into by machinery that never
   * took a write-hold (batched fetches, decode scratch) -- since operator[]
   * no longer infers a write from a non-const access, writes are declared
   * at the hold (`write=true`) and this covers the pages no hold saw.
   * Marking is separated from flushing so the write and the flush can live
   * in different kernel launches.
   */
  CTP_GPU_FUN void MarkResidentDirty(clio::run::u64 off, clio::run::u64 count) {
    LockBlock();
    ForEachResident(off, count, [](Page *p) { p->dirty = 1u; });
    UnlockBlock();
  }

 public:
  /**
   * BASIC: block-collective. Starts writeback of the dirty pages covering
   * [off, off+count) as BATCHED multi-puts -- one task per 64 pages, not
   * one per page -- and returns without waiting for the data to land.
   * Pair with `co_await AwaitFlush()`. Every thread of the block must call
   * this together (it synchronizes internally); thread 0 does the work.
   */
  CTP_GPU_FUN void FlushAsync(clio::run::u64 off, clio::run::u64 count) {
    __syncthreads();
    if (threadIdx.x == 0) {
      LockBlock();
      FlushRangeBatchedAsyncLocked(off, count);
      UnlockBlock();
    }
    __syncthreads();
  }

 private:
  /** Batched, ASYNC form of the flush walk: fills multi-put batches for the
   *  dirty resident pages of [off, off+count) and SENDS them without
   *  waiting. Pages ride with flushing=3 (batch marker); AwaitFlush's reap
   *  settles them (SettlePutLocked) and clears dirty per record rc.
   *  Falls back to scalar SubmitPut when batching is not provisioned. */
  CTP_GPU_FUN void FlushRangeBatchedAsyncLocked(clio::run::u64 off,
                                                clio::run::u64 count) {
    Page *tbl = BlockPages();
    const clio::run::u64 p0 = PageOf(off);
    const clio::run::u64 p1 = PageOf(off + (count == 0 ? 0 : count - 1));
    if (h_->multi_ == nullptr || h_->multi_per_block_ == 0) {
      for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
        Page *p = &tbl[i];
        if (p->page_num == kNoPage || p->page_num < p0 || p->page_num > p1) {
          continue;
        }
        if (p->dirty && !p->flushing) SubmitPut(p);
      }
      return;
    }
    MultiBatch *mb = BlockBatches();
    clio::run::u32 nb = 0;
    clio::run::u32 filled = 0;
    bool open = false;
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      Page *p = &tbl[i];
      if (p->page_num == kNoPage || !p->dirty || p->flushing) continue;
      if (p->page_num < p0 || p->page_num > p1) continue;
      if (!open) {
        // POD-reuse rule: settle whatever this slot still carries before
        // rewriting its task (get batch, score batch, or a previous async
        // flush) -- a bounded wait on the PREVIOUS work's tail.
        if (nb == 0) {
          if (mb[0].async_pending != 0u) SettleOneLocked(&mb[0]);
          if (mb[0].score_pending != 0u) SettleScoreLocked(&mb[0]);
        }
        if (mb[nb].put_pending != 0u) SettlePutLocked(&mb[nb]);
        PrepareMultiPut(mb[nb].put);
        open = true;
      }
      char name[32];
      PageBlobName(p->page_num, name);
      // Constant blob score: a reput whose score differs from the previous put
      // relocates the blob to another tier, and gets can then return the stale
      // replica. Never forward the page's drifting eviction score here.
      mb[nb].put->Add(name, 0, h_->page_bytes_, RawPtr(p->data), 1.0f);
      mb[nb].page_slot[filled] = i;
      p->flushing = 3u;   // batch-flushed; settled via SettlePutLocked
      ++filled;
      Bump(h_->stat_puts_);
      if (filled == clio::cte::core::kPodMultiMax) {
        XferAdd(1);
        mb[nb].put_fut =
            clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(
                SlotPtr(mb[nb].put));
        mb[nb].put_pending = 1u;
        mb[nb].put_n = filled;
        ++nb;
        filled = 0;
        open = false;
        if (nb == h_->multi_per_block_) return;   // provisioning ceiling
      }
    }
    if (open && filled > 0) {
      XferAdd(1);
      mb[nb].put_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(
          SlotPtr(mb[nb].put));
      mb[nb].put_pending = 1u;
      mb[nb].put_n = filled;
    }
  }

  /** Non-blocking completion probe of an async flush batch. */
  CTP_GPU_FUN bool PutDone(const MultiBatch *mb) const {
    if (mb->put_fut.IsNull()) return true;
    volatile const unsigned int *fp =
        reinterpret_cast<volatile const unsigned int *>(
            &mb->put_fut.get()->fut_.is_complete_.x);
    return ((*fp) & 1u) != 0u;
  }

  /** Settle an async flush batch: wait its future, clear dirty on each
   *  successful record, release the flushing=3 marks. Lock held. */
  CTP_GPU_FUN void SettlePutLocked(MultiBatch *mb) {
    if (mb->put_pending == 0u) return;
    mb->put_fut.Wait();
    Page *tbl = BlockPages();
    auto *t = mb->put;
    for (clio::run::u32 r = 0; r < mb->put_n; ++r) {
      Page *p = &tbl[mb->page_slot[r]];
      if (p->flushing == 3u) p->flushing = 0u;
      if (r < t->count_ && t->reqs_[r].rc_ == 0) p->dirty = 0u;
    }
    mb->put_pending = 0u;
    mb->put_n = 0u;
    XferAdd(-1);
  }

  /** Reap step for AwaitFlush: settle async flush batches that landed. */
  CTP_GPU_FUN void ReapFlushBatches() {
    if (h_->multi_ == nullptr) return;
    LockBlock();
    MultiBatch *mb = BlockBatches();
    for (clio::run::u32 b = 0; b < h_->multi_per_block_; ++b) {
      if (mb[b].put_pending != 0u && PutDone(&mb[b])) SettlePutLocked(&mb[b]);
    }
    UnlockBlock();
  }

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

 private:
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

 public:

  /**
   * BASIC: batched rescore, up to kPodMultiMax (64) entries per call.
   *
   * `gen(i)` returns the i-th PageScore. score == 1.0f is the maximum
   * preference and means MAKE IT RESIDENT: the entry joins one batched
   * fetch (the page is claimed and its bytes requested; a resident page
   * no-ops). Any other score is a CTE placement hint: the page's local
   * eviction score is refreshed if resident, and the blob's tier
   * preference rides one batched PodMultiScore task.
   *
   * Fire-and-forget: ISSUES both batches and returns without waiting for
   * either. `co_await AwaitRescore()` when completion matters; otherwise
   * the machinery settles them lazily. BLOCK-COLLECTIVE: every thread of
   * the block calls it together (thread 0 does the work). A
   * still-pending previous batch is settled here first (a bounded wait on
   * the PREVIOUS call's tail, never on this call's own data). Entries
   * that cannot claim a slot are dropped -- this is a hint interface, and
   * a later true access simply faults.
   *
   * @return entries accepted (fetch-claimed + score-batched).
   */
  template <typename GenF>
  CTP_GPU_FUN clio::run::u32 RescorePagesBatchedAsync(clio::run::u32 n,
                                                      GenF &&gen) {
    __syncthreads();
    clio::run::u32 got = 0;
    if (threadIdx.x == 0) {
      got = RescorePagesBatchedAsyncT0(n, gen);
    }
    __syncthreads();
    return got;   // meaningful on thread 0; a hint count, never load-bearing
  }

 private:
  template <typename GenF>
  CTP_GPU_FUN clio::run::u32 RescorePagesBatchedAsyncT0(clio::run::u32 n,
                                                        GenF &&gen) {
    if (h_->multi_ == nullptr || h_->multi_per_block_ == 0) return 0;
    if (n > clio::cte::core::kPodMultiMax) n = clio::cte::core::kPodMultiMax;
    LockBlock();
    MultiBatch *mb = BlockBatches();
    if (mb[0].async_pending != 0u) SettleOneLocked(&mb[0]);
    if (mb[0].score_pending != 0u) SettleScoreLocked(&mb[0]);
    PrepareMultiGet(mb[0].get);
    mb[0].get->flags_ = clio::cte::core::kCtePrefetchHint;
    PrepareMultiScore(mb[0].score);
    Page *tbl = BlockPages();
    clio::run::u32 nf = 0, ns = 0, nd = 0;
    for (clio::run::u32 i = 0; i < n; ++i) {
      const PageScore e = gen(i);
      char name[32];
      PageBlobName(e.page, name);
      if (e.score == 0.0f) {
        // Zero preference means DROP -- and drops NEVER write back. Only a
        // clean page (explicitly flushed, or never written) is discarded;
        // a dirty, flushing, held, or in-flight page is left alone. The
        // one writeback path is FlushAsync/AwaitFlush, called explicitly:
        // flush first, then drop.
        nd += DropPageLocked(e.page) ? 1u : 0u;
        continue;
      }
      if (e.score == 1.0f) {
        // Prefetch NEVER touches a page that is already in the table: a
        // resident dirty page's bytes are the only valid copy, and fetching
        // the blob over it would resurrect stale data. Only absent pages
        // are claimed and fetched.
        if (Find(e.page) != nullptr) continue;   // resident or in flight
        const clio::run::u32 slot = ClaimSlotWindowLocked(e.page);
        if (slot == ~0u) continue;               // hint: drop, never stall
        Page *np = &tbl[slot];
        np->gen += 1u;
        np->pins = 0u;
        np->fetching = 2u;   // batch-async; settled via the reap
        __threadfence();
        np->page_num = e.page;
        np->dirty = 0u;
        np->flushing = 0u;
        np->score = 2.0f;
        np->user_score = 0.0f;
        np->has_user = 0u;
        np->last_access = Now();
        mb[0].get->Add(name, 0, h_->page_bytes_, RawPtr(np->data));
        np->dbg_get_page = e.page;
        mb[0].page_slot[nf++] = slot;
        Bump(h_->stat_faults_);
        Bump(h_->stat_prefetches_);
      } else {
        Page *p = Find(e.page);
        if (p != nullptr) {
          p->score = e.score;
          p->user_score = e.score;
          p->has_user = 1u;
        }
        const float clamped =
            e.score < 0.0f ? 0.0f : (e.score > 1.0f ? 1.0f : e.score);
        mb[0].score->Add(name, 0, 0, ctp::ipc::ShmPtr<>::GetNull(), clamped);
        ++ns;
      }
    }
    if (nf != 0u) {
      XferAdd(1);
      mb[0].get_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(
          SlotPtr(mb[0].get));
      mb[0].async_pending = 1u;
      mb[0].async_n = nf;
    }
    if (ns != 0u) {
      mb[0].score_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(
          SlotPtr(mb[0].score));
      mb[0].score_pending = 1u;
    }
    UnlockBlock();
    return nf + ns + nd;
  }

  /** Drop one page from this block's cache (the batched verb's score==0.0
   *  path). CLEAN pages only: a dirty or flushing page is refused, never
   *  written back -- the caller flushes explicitly first. Fetching or
   *  pinned pages are not touched either. Caller holds the block lock.
   *  @return true if the page was resident, clean, and dropped. */
  CTP_GPU_FUN bool DropPageLocked(clio::run::u64 page_id) {
    Page *p = Find(page_id);
    if (p == nullptr || p->fetching || p->pins != 0u) return false;
    if (p->dirty || p->flushing) return false;
    if (!FreeVictimSlot(p)) return false;   // lock-free hold raced in
    p->dirty = 0u;
    p->flushing = 0u;
    p->score = 0.0f;
    p->user_score = 0.0f;
    p->has_user = 0u;
    Bump(h_->stat_evicts_);
    return true;
  }

  CTP_GPU_FUN void PrepareMultiScore(
      clio::cte::core::PodMultiScoreTask *t) const {
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = NextTaskId(kKindRescore);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodMultiScore;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    t->count_ = 0;
    t->flags_ = 0;
    t->num_ok_ = 0;
    t->context_ = clio::cte::core::Context();
    ClearRunCtx(t);
  }

  /** Non-blocking completion probe of the score batch. */
  CTP_GPU_FUN bool ScoreDone(MultiBatch *mb) {
    if (mb->score_fut.IsNull()) return true;
    volatile unsigned int *fp = reinterpret_cast<volatile unsigned int *>(
        &mb->score_fut.get()->fut_.is_complete_.x);
    return ((*fp) & 1u) != 0u;
  }

  /** Settle the score batch: wait (bounded; the task has no data phase)
   *  and clear the pending mark. Caller holds the block lock. */
  CTP_GPU_FUN void SettleScoreLocked(MultiBatch *mb) {
    if (mb->score_pending == 0u) return;
    mb->score_fut.Wait();
    mb->score_pending = 0u;
  }

  /** Reap step for AwaitRescore: settle whichever batches have landed. */
  CTP_GPU_FUN void ReapRescore() {
    if (h_->multi_ == nullptr) return;
    LockBlock();
    MultiBatch *mb = BlockBatches();
    if (mb[0].async_pending != 0u && MultiGetDone(&mb[0])) {
      SettleOneLocked(&mb[0]);
    }
    if (mb[0].score_pending != 0u && ScoreDone(&mb[0])) {
      SettleScoreLocked(&mb[0]);
    }
    UnlockBlock();
  }

  /** Block-uniform: is anything from RescorePagesBatchedAsync unsettled? */
  CTP_GPU_FUN bool RescoreInFlight() const {
    if (h_->multi_ == nullptr) return false;
    const MultiBatch *mb = BlockBatches();
    return mb[0].async_pending != 0u || mb[0].score_pending != 0u;
  }

  /**
   * Pin the page THIS THREAD is holding, and hand back the slot so it can be
   * released exactly.
   *
   * Prefer this to PinRange when the page has just been held. PinRange finds
   * the page with the lock-free Find, which is racy by design and can MISS a
   * page that is present -- leaving it unpinned and evictable while a raw
   * pointer to it is already in use. That is not theoretical: it slipped one
   * window past pinning after ~80 steps, and the stale-page guard caught it.
   *
   * @return the pinned slot, or nullptr if nothing is held.
   */
  CTP_GPU_FUN Page *PinHeld() {
    Page *p = last_page_;
    if (p == nullptr) return nullptr;
    LockBlock();
    // RE-VALIDATE under the lock. In a SHARED (ro-shard) cache a sibling
    // block can evict and re-tenant this slot between the hold that cached
    // last_page_ and this pin; pinning blind then protects -- and lets the
    // caller PUBLISH -- the wrong page's frame. Caught live: page 33
    // published to a frame a sibling had re-claimed for page 57, whose
    // in-flight fetch then landed on top of page 33's freshly written
    // content (the intermittent rebuild corruption, ~25% of runs). A null
    // return means the caller must re-hold and try again.
    if (p->page_num != last_pn_ || p->fetching != 0u) {
      UnlockBlock();
      return nullptr;
    }
    p->pins += 1u;
    UnlockBlock();
    return p;
  }

  /** Release a pin taken by PinHeld, by slot rather than by lookup. */
  CTP_GPU_FUN void UnpinPage(Page *p) {
    if (p == nullptr) return;
    LockBlock();
    if (p->pins != 0u) p->pins -= 1u;
    UnlockBlock();
  }

  /**
   * Pin every resident page of [p0, p0+n) so a claim cannot recycle it.
   *
   * Call from ONE thread, pair with UnpinRange, and keep the pinned span
   * smaller than the table or claims will find no candidate. This is what
   * makes it safe to record raw page pointers and read through them, which is
   * the whole point of a page-pointer table.
   *
   * @return pages actually pinned.
   */
  CTP_GPU_FUN clio::run::u32 PinRange(clio::run::u64 p0, clio::run::u32 n) {
    LockBlock();
    clio::run::u32 got = 0;
    for (clio::run::u32 k = 0; k < n; ++k) {
      Page *p = Find(p0 + k);
      if (p != nullptr) {
        p->pins += 1u;
        ++got;
      }
    }
    UnlockBlock();
    return got;
  }

  /** Release pins taken by PinRange. */
  CTP_GPU_FUN void UnpinRange(clio::run::u64 p0, clio::run::u32 n) {
    LockBlock();
    for (clio::run::u32 k = 0; k < n; ++k) {
      Page *p = Find(p0 + k);
      if (p != nullptr && p->pins != 0u) p->pins -= 1u;
    }
    UnlockBlock();
  }

  /** True when `pg` is resident but its writeback is in flight, so a WRITE
   *  hold must wait rather than store into it (see ProbeHold). */
  CTP_GPU_FUN bool WriteBusy(clio::run::u64 pg) const {
    const Page *p = Find(pg);
    return p != nullptr && (p->flushing != 0u || p->evicting != 0u);
  }

  /** Lock-free residency probe. RACY BY DESIGN: a concurrent claim can
   *  make it misread one page — callers only use it to decide whether a
   *  batched fault phase is worth entering, where either error is safe. */
  CTP_GPU_FUN bool ProbeResident(clio::run::u64 pg) const {
    return Find(pg) != nullptr;
  }

  // The single-thread scalar LZ4 decoder that used to live here is gone too.
  //
  // It existed as the "airtight fallback" for sites where the warp decoder
  // could not run, and it is what a compressed run silently fell back to
  // whenever the device API was absent -- roughly two orders of magnitude
  // slower than nvcomp, while still being reported as nvcomp. With the CPU
  // launching the codec there is no site that needs it, and no way for a run
  // to land on it by accident.

  // NOTE: the in-kernel nvcomp decoder that used to live here is GONE.
  //
  // Decompression is launched by the CPU now (the compressor's batched
  // drain), for two reasons that outrank the round trip it costs: nvcomp's
  // device API is a separate artifact a stock install does not ship, and the
  // decoder was written against LZ4's device internals, which are not
  // guaranteed to exist on every GPU. The host batched entry points cover all
  // nine codecs using the ordinary library.
  //
  // What remains in this file is deliberate: page-cache bookkeeping, and
  // issuing reads. It knows nothing about compression.

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

#if defined(CLIO_YIELD_CORO) && defined(__clang__) && defined(__CUDA__)
 private:
  /**
   * Block-collective hold that CANNOT suspend: returns 0 if the page is not
   * already resident, so the caller can fall back to the coroutine.
   *
   * Exists because `co_await HoldPageCoro(...)` costs ~31,000 cycles even when
   * the page is resident and it returns without ever suspending. The work is
   * not the lookup -- that is a handful of probes -- it is CONSTRUCTING THE
   * COROUTINE: every one of the block's threads builds a frame in the 4 KB
   * per-thread yield stack, runs the promise machinery and tears it down, and
   * a LAMMPS step does 772 of these. That was ~16 ms of a 65 ms kernel.
   *
   * This is a plain device function, so a hit costs a residency probe and the
   * existing lock-free lookup, and nothing else.
   *
   * Block-uniform: IsResident depends only on the page number and this block's
   * table, so every thread returns the same answer and the caller's branch
   * cannot split the block across a barrier.
   */
  CTP_GPU_FUN clio::run::u64 TryHoldFast(clio::run::u64 off,
                                         clio::run::u64 count,
                                         bool write = false) {
    // The barrier is NOT optional. HoldPageCoro opens with one, and the block
    // relies on it: a thread must not inspect the page table while another is
    // still writing to the page the previous hold covered. Dropping it here
    // produced 8 force writebacks instead of 123 and an energy of -4.6731464
    // against the correct -4.7638693 -- wrong, and plausible enough to pass
    // an eyeball.
    __syncthreads();
    // VOTE THE OUTCOME, not just each thread's own probe. The scan is
    // lock-free, and on a SHARED table (eternia's read-only shards) another
    // block's eviction can flip a slot between two threads' reads: thread A
    // then succeeds where thread B misses. Returning those per-thread
    // answers splits the caller's `if (run == 0) co_await ...` across the
    // block -- half the threads suspend inside HoldPageCoro's barriers while
    // the rest run ahead, and sm_89 traps the mismatched BAR as an illegal
    // instruction (observed as the LAMMPS 256k-atom crash). The whole block
    // must take ONE path: all-fast only when EVERY thread held the page.
    const clio::run::u64 r = ProbeHold(off, count, write);
    if (__syncthreads_and(r != 0 ? 1 : 0)) {
      return r;   // every thread succeeded; each carries its pin
    }
    // At least one thread missed: the block falls to the coro path
    // TOGETHER. A thread that DID succeed gives its pin back.
    if (r != 0) {
      ReleaseHoldPin(last_page_);
    }
    __syncthreads();
    return 0;
  }

  __device__ clio::run::gpu::YCoroTask HoldPageCoro(clio::run::u64 off,
                                                    clio::run::u64 count,
                                                    clio::run::u64 *run_out,
                                                    bool write = false) {
    if (threadIdx.x == 0) {
      ReapFetched();
      ReapFlushed();
    }
    __syncthreads();
    // 1. WRITEBACK: submit the victim's put and leave (writeback step 1).
    if (threadIdx.x == 0) {
      StartEvictionAsync(PageOf(off));
#if defined(CLIO_YCORO_DEVTRACE)
      printf("[hpc] blk=%u pg=%llu evict-submitted inflight=%d tag=%llx\n",
             block_override_, (unsigned long long)PageOf(off),
             (int)AnyTransferInFlight(), (unsigned long long)FlushWaitTag());
#endif
    }
    CLIO_CO_YIELD_WHEN((ReapFlushed(), ReapFetched()), AnyTransferInFlight(),
                       FlushWaitTag());
    // Retire the writeback WITHIN this entry, before fetching -- and NOT only
    // inside the loop above. The loop's exit condition is the RAW completion
    // flag, so the put can complete in the window between an iteration's reap
    // and its vote: the loop then breaks with the victim still flushing and
    // its slot unfreed, BeginFetch below declines (the victim is mid-flush,
    // every other slot is dirty), and nothing ever re-issues it -- a livelock
    // the macro version cannot have, because its fall-through path reaps
    // unconditionally after the yield.
    if (threadIdx.x == 0) {
      ReapFlushed();
      ReapFetched();
    }
#if defined(CLIO_YCORO_DEVTRACE)
    if (threadIdx.x == 0) {
      printf("[hpc] blk=%u pg=%llu flushwait-done inflight=%d\n",
             block_override_, (unsigned long long)PageOf(off),
             (int)AnyTransferInFlight());
    }
#endif
    __syncthreads();
    // 2. FETCH: issue from one lane, then leave (fetch step 2 --
    //    encoded tags fault a RUN so the batched decode gets full launches).
    if (threadIdx.x == 0 && !IsResident(PageOf(off))) {
      if (h_->compress_lib_ != 0) {
        LockBlock();
        const clio::run::u32 got =
            BeginFetchRunLocked(PageOf(off), clio::cte::core::kPodMultiMax);
        UnlockBlock();
        if (got == 0 && !IsResident(PageOf(off))) {
          BeginFetch(PageOf(off), /*is_prefetch=*/false);
        }
      } else {
        const bool bf = BeginFetch(PageOf(off), /*is_prefetch=*/false);
#if defined(CLIO_YCORO_DEVTRACE)
        printf("[hpc] blk=%u pg=%llu beginfetch=%d resident=%d\n",
               block_override_, (unsigned long long)PageOf(off), (int)bf,
               (int)IsResident(PageOf(off)));
#else
        (void)bf;
#endif
      }
    }
    // RE-ISSUE THE FETCH ON EVERY RESUME IF IT WAS LOST. BeginFetch above
    // runs ONCE, and it FAILS -- returns false with nothing in flight --
    // whenever ClaimSlotWindowLocked finds every candidate slot mid-transfer
    // or pinned. The park below then waits on !IsResident with a null wait
    // tag, which nothing will ever satisfy: observed as four blocks spinning
    // a reneighbour step to the 2,000,000-round cap, each on the second page
    // of a chunk-straddling fetch whose candidate slots were pinned by its
    // neighbours. Retrying inside the yield's reap step turns a transient
    // claim failure into one extra round instead of a livelock.
    for (;;) {
      // ... and wait out an in-flight writeback for a WRITE hold, or the
      // loop would spin without yielding: the page IS resident, so the
      // residency condition alone never suspends, and `evicting` is only
      // cleared by the reap step that a suspension runs.
      CLIO_CO_YIELD_WHEN((ReapFlushed(), ReapFetched(),
                          RetryLostFetch(PageOf(off))),
                         !IsResident(PageOf(off)) ||
                             (write && WriteBusy(PageOf(off))),
                         IsResident(PageOf(off)) ? FlushWaitTag()
                                                 : FetchWaitTag(PageOf(off)));
      // 3. Resident for the whole block: the lock-free fast path. The probe
      // can still lose the slot to a concurrent claim between the vote and
      // the scan -- that is a MISS, and the answer is another park, never a
      // synchronous fetch from inside the kernel.
      //
      // AND-vote, not OR. On a shared table the probe can diverge across the
      // block (another block's eviction lands between two threads' scans);
      // breaking on ANY success let the LOSING threads exit with run == 0 and
      // a stale last_page_, and the caller then built their Held from the
      // wrong page -- observed as a few thousand silently skipped pair
      // entries per step under eviction pressure. Exit only when EVERY
      // thread holds the page; winners of a failed round release the pin
      // they just took, or the retry pins the page up forever.
      *run_out = ProbeHold(off, count, write);
      if (__syncthreads_and(*run_out != 0 ? 1 : 0)) {
        break;
      }
      if (*run_out != 0) {
        ReleaseHoldPin(last_page_);
        *run_out = 0;
      }
    }
  }

 public:
  /**
   * BASIC: `Held<T> h = co_await vec.HoldPage(off, count, write)` -- THE
   * access verb. Resident pages complete on the fast path with no
   * suspension; a miss faults the page in and suspends the block until it
   * lands. Block-collective: every thread of the block must co_await it
   * together (it synchronizes internally); each thread gets its own guard.
   *
   * The guard is the access AND the pin: h.run() elements from `off` are
   * reachable through it (see Held), and the page stays put until the guard
   * dies. Scope the guard to the work -- a guard alive across the NEXT hold
   * keeps its page unevictable, which is either exactly what a multi-page
   * working set wants or a slot leak, depending on whether you meant it.
   *
   * `write` declares intent for the WHOLE hold: pass true when anything will
   * be stored through the guard, and the page is marked dirty up front so
   * eviction and FlushAsync write it back. A page held without the flag is
   * dropped clean no matter what was stored.
   */
  /**
   * SERVICED HOLD -- the target fault path (design.md Part II).
   *
   * On a miss the block does not resolve anything. It records the page it
   * needs in its fault mailbox, parks, and re-probes on resume. The host
   * servicer claims the slot, writes back a dirty victim, fetches, publishes
   * the page and only then clears `pending`, so:
   *
   *   RULE 1: when this co_await returns, the page IS resident.
   *   RULE 2: the only wait on the device is the park. No future, no spin.
   *
   * The loop below should therefore execute its body exactly once. It is a
   * safety net against a servicer bug, not a retry protocol -- if it ever
   * iterates twice, rule 1 has been violated and the servicer is wrong.
   *
   * Falls back to the legacy in-kernel path when no mailbox is installed, so
   * both can coexist while this is proven out.
   */
  __device__ clio::run::gpu::YCoroTaskT<Held<T>> HoldPageServiced(
      clio::run::u64 off, clio::run::u64 count, bool write = false) {
    clio::run::u64 run = TryHoldFast(off, count, write);
    while (run == 0) {
      volatile FaultReq *f = &h_->faults_[BlockIndex()];
      if (threadIdx.x == 0) {
        f->page_num = PageOf(off);
        f->write = write ? 1u : 0u;
        __threadfence_system();       // request visible before the doorbell
        f->pending = 1u;
        __threadfence_system();
      }
      __syncthreads();
      co_await clio::run::gpu::YCoroSuspend{
          reinterpret_cast<clio::run::u64>(
              const_cast<clio::run::u32 *>(&f->pending))};
      run = TryHoldFast(off, count, write);
    }
    Page *p = last_page_;
    co_return Held<T>(p, static_cast<T *>(p->data) + IndexIn(off, p), off,
                      run, /*pinned=*/h_->no_evict_ == 0u);
  }

  __device__ clio::run::gpu::YCoroTaskT<Held<T>> HoldPage(
      clio::run::u64 off, clio::run::u64 count, bool write = false) {
    clio::run::u64 run = TryHoldFast(off, count, write);
    if (run == 0) {
      co_await HoldPageCoro(off, count, &run, write);
    }
    Page *p = last_page_;
    co_return Held<T>(p, static_cast<T *>(p->data) + IndexIn(off, p), off,
                      run, /*pinned=*/h_->no_evict_ == 0u);
  }

  /**
   * BASIC: `co_await vec.AwaitFlush()` -- suspends until every writeback
   * started by FlushAsync on this block's table has completed. Reaps
   * completed puts as it waits. Block-collective.
   */
  /**
   * Is admission control armed on this vector?
   *
   * Call sites MUST test this before `co_await EnterHoldSet(...)`. An
   * early-returning coroutine is not free: awaiting one still builds a frame
   * on the yield stack and costs a __syncthreads, and at three per row that
   * measured 110 ms/step against 43 with the awaits skipped -- i.e. the
   * disabled path cost 2.5x all by itself.
   */
  /**
   * Is admission in force RIGHT NOW? Call sites must latch this in a local
   * at Enter and reuse that local at Exit: it can flip mid-run (see
   * VecHeader::admit_armed_), and re-testing it at Exit would let a block
   * release a reservation it never took.
   */
  CTP_GPU_FUN bool AdmissionOn() const {
    return h_->hold_admits_ != nullptr && h_->admit_armed_ != nullptr &&
           *(volatile unsigned int *) h_->admit_armed_ != 0u;
  }

  /**
   * Do these two handles address the SAME page table (and so the same
   * admission counter)?
   *
   * Callers must aggregate their reservations per TABLE, not per handle: two
   * EnterHoldSet calls on one table by one block self-deadlock, because the
   * second waits for capacity the first is already holding. The gather pass
   * is launched as (src=dx, srcx=dx, dst=dx2) -- two of its three handles
   * are the same vector -- so this is not hypothetical.
   */
  CTP_GPU_FUN bool SameTableAs(const DeviceVector<T> &o) const {
    return h_->pages_ == o.h_->pages_;
  }

  /**
   * How many pages the range [off, off+count) touches -- i.e. exactly how
   * many guards holding it will take, since a hold is cut at each page
   * boundary.
   *
   * This is what lets a caller reserve its TRUE hold set instead of a
   * worst-case bound. It is pure arithmetic on the request, so it can run
   * before any hold is taken, which is the ordering admission requires.
   */
  CTP_GPU_FUN clio::run::u32 PagesSpanned(clio::run::u64 off,
                                          clio::run::u64 count) const {
    if (count == 0) return 0;
    return static_cast<clio::run::u32>(PageOf(off + count - 1) - PageOf(off)) +
           1u;
  }

  /** Reserve `n` slots for this block's hold set, or fail without waiting. */
  CTP_GPU_FUN bool TryAdmit(clio::run::u32 n) {
    unsigned int *a = h_->hold_admits_;
    clio::run::u32 cur = *(volatile unsigned int *) a;
    while (cur + n <= h_->hold_admit_cap_) {
      const clio::run::u32 old = atomicCAS(a, cur, cur + n);
      if (old == cur) return true;
      cur = old;
    }
    return false;
  }

  /**
   * BASIC: `co_await vec.EnterHoldSet(n)` -- declare that this block is
   * about to hold up to `n` pages AT ONCE, and wait until that many slots
   * can be reserved for it. Pair with ExitHoldSet(n) once the guards die.
   *
   * Waiting happens while holding NOTHING, which is the whole point: a
   * block that cannot be admitted parks without pinning anything, so it
   * cannot contribute to the exhaustion it is waiting on.
   *
   * Block-collective. `n` is clamped to the cap so a single block is always
   * admissible -- an unclamped request larger than the table would wait
   * forever for slots that cannot exist.
   */
  __device__ clio::run::gpu::YCoroTask EnterHoldSet(clio::run::u32 n) {
    if (h_->hold_admits_ == nullptr || n == 0) {
      __syncthreads();
      co_return;
    }
    if (n > h_->hold_admit_cap_) n = h_->hold_admit_cap_;
    // Only thread 0 reserves; the others vote false, so the OR-vote inside
    // the macro carries thread 0's answer to the whole block and every
    // thread leaves together.
    CLIO_CO_YIELD_WHEN((void) 0,
                       (threadIdx.x == 0) ? !TryAdmit(n) : false,
                       AdmitWaitTag());
  }

  /** Release a hold set's reservation. Pairs with one EnterHoldSet(n). */
  CTP_GPU_FUN void ExitHoldSet(clio::run::u32 n) {
    if (h_->hold_admits_ == nullptr || n == 0) return;
    if (n > h_->hold_admit_cap_) n = h_->hold_admit_cap_;
    __syncthreads();   // every guard in the set is dead before we give slots back
    if (threadIdx.x == 0) atomicSub(h_->hold_admits_, n);
  }

  __device__ clio::run::gpu::YCoroTask AwaitFlush() {
    // Reap BEFORE the first vote, not only after a resume. A put that
    // completed before this call makes AnyFlushInFlight() false on the
    // first evaluation, so the loop breaks having never settled the batch
    // -- and its pages stay flushing=3 forever, silently skipped by every
    // later FlushAsync (observed as force blobs frozen at their step-0
    // bytes). One locked scan per await is noise; this is per phase, not
    // per hold.
    if (threadIdx.x == 0) {
      ReapFlushed();
      ReapFlushBatches();
    }
    __syncthreads();
    CLIO_CO_YIELD_WHEN((ReapFlushed(), ReapFlushBatches()),
                       AnyFlushInFlight(), FlushWaitTag());
    // Reap AFTER the loop too. AnyFlushInFlight() answers "incomplete", not
    // "unsettled": a batch that lands between a reap and its vote breaks the
    // loop landed-but-unsettled, and its pages keep flushing=3 -- every later
    // FlushAsync then drops them (observed as force blobs going stale on
    // alternating steps). Every future is complete here, so this never waits.
    if (threadIdx.x == 0) {
      ReapFlushed();
      ReapFlushBatches();
    }
    __syncthreads();
  }

  /**
   * BASIC: `co_await vec.AwaitRescore()` -- suspends until both batches a
   * RescorePagesBatchedAsync call issued (the score-1 fetch batch and the
   * placement-score batch) have settled. Block-collective. Waiting is
   * OPTIONAL: an unawaited batch settles lazily through the machinery's
   * own reaps; await only when subsequent logic depends on residency or
   * on the placement having been applied.
   */
  __device__ clio::run::gpu::YCoroTask AwaitRescore() {
    // Same first-vote reap as AwaitFlush, same reason.
    if (threadIdx.x == 0) {
      ReapRescore();
    }
    __syncthreads();
    CLIO_CO_YIELD_WHEN((ReapRescore()), RescoreInFlight(), FlushWaitTag());
    // Same post-loop reap as AwaitFlush, same landed-but-unsettled race.
    if (threadIdx.x == 0) {
      ReapRescore();
    }
    __syncthreads();
  }

#endif  // CLIO_YIELD_CORO

 private:
  /** One lane re-issues a page's fetch if no slot carries it (see the
   *  HoldPageCoro comment: a failed claim must not become a silent park). */
  CTP_GPU_FUN void RetryLostFetch(clio::run::u64 page_num) {
    if (threadIdx.x == 0 && Find(page_num) == nullptr) {
      if (!BeginFetch(page_num, /*is_prefetch=*/false)) {
        // AND MAKE ROOM AGAIN. BeginFetch fails when the claim finds every
        // candidate slot pinned, dirty or mid-transfer, and retrying it
        // alone changes nothing: StartEvictionAsync runs ONCE, at the top
        // of HoldPageCoro, so if its victim was taken by a sibling block
        // before this block got there, no slot is ever freed again and the
        // fault spins until the driver's round cap -- which used to be
        // silent, so the kernel came back partially executed. Re-arming the
        // eviction here is what the fault path's own contract says
        // ("whoever needs room must arrange it"); it is a no-op when the
        // page is already present or coming.
        StartEvictionAsync(page_num);
      }
    }
  }

  /**
   * Device address of the completion flag for `page_num`'s in-flight get, or 0
   * if there is nothing specific to wait on (no slot claimed yet, so the block
   * should be relaunched to claim one).
   */
  CTP_GPU_FUN clio::run::u64 FetchWaitTag(clio::run::u64 page_num) const {
    const Page *p = Find(page_num);
    if (p == nullptr || !p->fetching) {
      return 0;
    }
    if (p->fetching == 2u) {
      // Run-fetched: the page's own future is stale; the thing to wait for
      // is the multi carrying it.
      MultiBatch *mb = BlockBatches();
      if (mb[0].async_pending != 0u && !mb[0].get_fut.IsNull()) {
        return reinterpret_cast<clio::run::u64>(
            &mb[0].get_fut.get()->fut_.is_complete_.x);
      }
      return 0;
    }
    return reinterpret_cast<clio::run::u64>(&p->get->fut_.is_complete_.x);
  }

  /** Same, for the first writeback still outstanding in this block. */
  /**
   * Admission waits with NO tag, i.e. tag 0 -- "always worth relaunching".
   *
   * A tag is polled by the host as `*tag != 0`, so the admission counter is
   * exactly the wrong thing to point at: it is non-zero while the table is
   * BUSY and zero when slots are free, so blocks were resumed while full
   * and left parked once space appeared. That inversion deadlocked 48- and
   * 40-slot runs that worked before admission existed. Tag 0 re-tries the
   * reservation each round, which also lets the driver service fetches in
   * between -- the reason a spin here would be fatal rather than merely
   * wasteful (the yield driver exits the kernel to service a fault, so a
   * block that never parks stops every fetch in the grid).
   */
  CTP_GPU_FUN clio::run::u64 AdmitWaitTag() const { return 0; }

  CTP_GPU_FUN clio::run::u64 FlushWaitTag() const {
    const Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      const Page *p = &tbl[i];
      if (p->flushing == 3u) continue;   // completion is the BATCH future's
      if (p->flushing && p->put->fut_.is_complete_.load() == 0) {
        return reinterpret_cast<clio::run::u64>(&p->put->fut_.is_complete_.x);
      }
    }
    // BATCH futures too, or a batched flush has no tag at all: pages ride
    // flushing=3 with no live scalar future, the tag stays 0, and the yield
    // driver RELAUNCHES the parked block immediately instead of waiting --
    // measured as ~37 relaunch rounds per AwaitFlush (98% of a flush
    // benchmark's wall was kernel-resident spin, ~1.7ms per round).
    if (h_->multi_ != nullptr) {
      const MultiBatch *mb = BlockBatches();
      for (clio::run::u32 b = 0; b < h_->multi_per_block_; ++b) {
        if (mb[b].put_pending != 0u && !mb[b].put_fut.IsNull() &&
            !PutDone(&mb[b])) {
          return reinterpret_cast<clio::run::u64>(
              &mb[b].put_fut.get()->fut_.is_complete_.x);
        }
        if (mb[b].async_pending != 0u && !mb[b].get_fut.IsNull() &&
            (mb[b].get_fut.get()->fut_.is_complete_.load() & 1u) == 0u) {
          return reinterpret_cast<clio::run::u64>(
              &mb[b].get_fut.get()->fut_.is_complete_.x);
        }
        if (mb[b].score_pending != 0u && !mb[b].score_fut.IsNull() &&
            (mb[b].score_fut.get()->fut_.is_complete_.load() & 1u) == 0u) {
          return reinterpret_cast<clio::run::u64>(
              &mb[b].score_fut.get()->fut_.is_complete_.x);
        }
      }
    }
    return 0;
  }

  /**
   * True while any of this block's pages still has a put or get outstanding.
   *
   * Deliberately lock-free and approximate: it is a yield predicate, re-run on
   * every resume, so a stale read costs one extra round rather than a wrong
   * answer. Taking the block lock here would have all 32 lanes serialise
   * through an atomicCAS on what is meant to be a cheap test.
   */
  CTP_GPU_FUN void XferAdd(int d) const {
    if (h_->xfer_cnt_ != nullptr) {
      atomicAdd(&h_->xfer_cnt_[BlockIndex()], (unsigned int) d);
    }
  }
  CTP_GPU_FUN bool XferIdle() const {
    return h_->xfer_cnt_ != nullptr && h_->xfer_cnt_[BlockIndex()] == 0u;
  }

  /** Flush-only variant of AnyTransferInFlight, for AwaitFlush. Covers the
   *  scalar puts AND the async flush batches (flushing == 3u pages have no
   *  live scalar future; their completion is the batch's). */
  CTP_GPU_FUN bool AnyFlushInFlight() const {
    if (XferIdle()) return false;
    const Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      const Page *p = &tbl[i];
      if (p->flushing == 3u) continue;   // completion is the BATCH future's
      if (p->flushing && p->put->fut_.is_complete_.load() == 0) return true;
    }
    if (h_->multi_ != nullptr) {
      const MultiBatch *mb = BlockBatches();
      for (clio::run::u32 b = 0; b < h_->multi_per_block_; ++b) {
        // In-flight only while the future is INCOMPLETE: an unsettled but
        // landed batch must not spin a waiter that never reaps.
        if (mb[b].put_pending != 0u && !PutDone(&mb[b])) return true;
      }
    }
    return false;
  }

  CTP_GPU_FUN bool AnyTransferInFlight() const {
    if (XferIdle()) return false;
    const Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      const Page *p = &tbl[i];
      if (p->flushing == 3u) continue;   // completion is the BATCH future's
      if (p->flushing && p->put->fut_.is_complete_.load() == 0) return true;
      if (p->fetching && p->get->fut_.is_complete_.load() == 0) return true;
    }
    if (h_->multi_ != nullptr) {
      const MultiBatch *mb = BlockBatches();
      for (clio::run::u32 b = 0; b < h_->multi_per_block_; ++b) {
        if (mb[b].put_pending != 0u && !PutDone(&mb[b])) return true;
      }
    }
    return false;
  }

  /**
   * Free a slot for `page_num` WITHOUT waiting: if the victim is dirty its
   * put is submitted and the slot is left in flight for the caller to yield
   * on. Clean victims are dropped immediately.
   *
   * SCANS THE SAME WINDOW ClaimSlotWindowOnceLocked WILL PROBE, and that is
   * load-bearing rather than an optimization. This is the demand path that
   * makes room when the claim declines, so the two must agree on WHICH slots
   * are candidates. They did not: the claim is bounded to kProbeWindow slots
   * from the page's home slot, while this scanned the whole table and freed a
   * globally-chosen victim.
   *
   * With pages_per_block <= kProbeWindow the window IS the whole table and the
   * disagreement is invisible. Above it the two can miss each other
   * permanently: the claim declines because its window is full, this frees a
   * slot outside that window (or, worse, sees a free slot outside it at the
   * old early-return and did nothing at all), and the next round probes the
   * same full window. The kernel spins forever with the CPU idle.
   *
   * MEASURED: Gray-Scott at 16 GB hung at 256 pages/block and passed at 64,
   * with the hang tracking pages_per_block and NOT total cache bytes -- 128 MB
   * at 256 slots/block hung while 256 MB at 16 and at 64 slots/block both
   * passed. 64 is exactly kProbeWindow.
   */
  CTP_GPU_FUN void StartEvictionAsync(clio::run::u64 page_num) {
    if (Find(page_num) != nullptr) return;          // already here or coming
    LockBlock();
    Page *tbl = BlockPages();
    const clio::run::u32 ppb = h_->pages_per_block_;
    const clio::run::u32 nw = Ways();
    clio::run::u32 victim = ppb;
    for (clio::run::u32 w = 0; w < nw; ++w) {
      const clio::run::u32 i = WaySlot(page_num, w);
      if (tbl[i].pins != 0u) continue;   // in use through a raw pointer
      if (tbl[i].page_num == kNoPage) { UnlockBlock(); return; }  // free slot
      // `rescoring` is deliberately NOT skipped here, though the claim does
      // skip it. That asymmetry is the escape hatch: `rescoring` is STICKY --
      // Rescore sets it and only a LATER rescore of the SAME slot clears it,
      // so a rescored page stays flagged indefinitely. If this path skipped
      // them too, every rescored page would become permanently unevictable,
      // no victim would ever be found, and the fault path would retry
      // forever. MEASURED: adding the skip here for "consistency" took a
      // weights cell that runs in ~70s to a 600s timeout at 8 slots/block.
      if (tbl[i].fetching || tbl[i].flushing) continue;
      if (PreferVictim(tbl[i], tbl[victim == ppb ? i : victim],
                       victim != ppb, w, page_num)) {
        victim = i;
      }
    }
    if (victim == ppb) {
      // Nothing evictable in the window -- every slot is mid-transfer. The
      // caller yields on AnyTransferInFlight and retries; the reaps then free
      // one of these, IN THIS WINDOW, so the next claim succeeds.
      UnlockBlock();
      return;
    }
    Page *p = &tbl[victim];
    if (p->dirty || p->flushing) {
      // THE SAME HANDSHAKE THE CLEAN PATH USES, and for the same reason.
      // The victim was chosen with pins == 0, but ProbeHold pins WITHOUT
      // the block lock, so a write-hold can land in the window between
      // that read and this writeback. SubmitPut clears `dirty` as it
      // sends, so a writer whose data lands after the put's DMA leaves the
      // page marked CLEAN with its update only in the frame -- and the
      // next drop discards it. Publish `evicting`, fence, then re-read
      // pins: a holder that raced in wins and the writeback is abandoned
      // (the page stays dirty and will be flushed later).
      p->evicting = 1u;        // tells ReapFlushed to release the slot
      __threadfence();
      if (*(volatile clio::run::u32 *)&p->pins != 0u) {
        p->evicting = 0u;      // someone is writing it; do not write back
      } else {
        SubmitPut(p);          // async; caller yields on AnyTransferInFlight
      }
    } else if (FreeVictimSlot(p)) {
      Bump(h_->stat_evicts_);
    }
    UnlockBlock();
  }

  /**
   * Clear `fetching` on gets that have landed.
   *
   * Nothing else does: AwaitFetch is what retires a fetch, and the blocking
   * path called it from inside the wait. Without this the page arrives but
   * stays flagged in flight, IsResident keeps reporting false, and the block
   * suspends forever -- which is exactly how the first yieldable run burned
   * 200000 rounds and then produced a wrong checksum.
   */
  CTP_GPU_FUN void ReapFetched() {
    if (XferIdle()) return;
    LockBlock();
    Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      Page *p = &tbl[i];
      if (p->fetching == 4u) continue;   // decode-pending; see AwaitFetch
      if (p->fetching == 2u) {
        // Run-fetched: the page's own future is STALE, so gating on it left
        // landed runs unsettled here -- the block then churned through ~40
        // resume/suspend rounds per multi until the stale flag happened to
        // read complete. Ask the multi itself, and only settle when it is
        // genuinely done (SettleOneLocked WAITS, which pre-yield must not).
        MultiBatch *mb = BlockBatches();
        if (mb[0].async_pending != 0u && MultiGetDone(&mb[0])) {
          SettleOneLocked(&mb[0]);
        }
        // ALSO the async slots, which live at the OTHER END of the array
        // (AsyncBatchSlot counts down from multi_per_block-1) and which this
        // loop used to ignore entirely. A page claimed by
        // FetchPagesBatchedAsync is marked fetching=2 exactly like a run-fetch,
        // but nothing here settled its batch, so on the yieldable path it
        // deadlocked: HoldPageCoro parks until the page is resident, only
        // AwaitFetch settles the batch, and AwaitFetch is reached only AFTER
        // the page is resident. Observed as a live-lock -- 2,000,000 rounds and
        // 32M holds for zero pairs -- the moment a cache was large enough for
        // AsyncSlotCount() to be non-zero, which is why it hid below 65 slots.
        const clio::run::u32 acnt = AsyncSlotCount();
        for (clio::run::u32 k = 0; k < acnt; ++k) {
          MultiBatch *am = AsyncBatchSlot(k);
          if (am->async_pending != 0u && MultiGetDone(am)) {
            SettleOneLocked(am);
          }
        }
        continue;
      }
      if (p->fetching && p->get->fut_.is_complete_.load() != 0) {
        // NOTE: a failed get is NOT freed here. First touch of a page that
        // does not exist yet fails by design, and the slot staying claimed is
        // what lets the write land (write-allocate). Freeing it instead made
        // seeding loop forever and store nothing.
        AwaitFetch(p);          // complete already, so this cannot block
      }
    }
    UnlockBlock();
  }

  /**
   * Retire slots whose writeback has landed. Called after the yield, when the
   * puts are known complete, so AwaitPut cannot block.
   */
  CTP_GPU_FUN void ReapFlushed() {
    if (XferIdle()) return;
    LockBlock();
    Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      Page *p = &tbl[i];
      if (p->flushing == 3u) continue;   // batch-flushed: settled below
      if (p->flushing && p->put->fut_.is_complete_.load() != 0) {
        AwaitPut(p);   // clears `flushing`; re-dirties the page if it FAILED
        if (p->evicting && !p->dirty) {
          // Eviction: the bytes are safely stored -- release the slot,
          // UNLESS a lock-free hold landed while the writeback ran (probes
          // do not reject flushing slots). The page then simply stays
          // resident, clean.
          if (FreeVictimSlot(p)) Bump(h_->stat_evicts_);
          p->evicting = 0u;
        } else if (p->evicting) {
          // The writeback failed, so the cache holds the only copy. Keep the
          // page and let the next attempt retry rather than dropping it.
          p->evicting = 0u;
        }
        // A plain BeginFlush writeback leaves the page RESIDENT and clean --
        // that is its contract ("a later write re-dirties the page"). Freeing
        // the slot here evicted every flushed page and counted it as an
        // eviction, which is where a no-op EvictPages(0) got 4 evicts from.
      }
    }
    if (h_->multi_ != nullptr) {
      MultiBatch *mb = BlockBatches();
      for (clio::run::u32 b = 0; b < h_->multi_per_block_; ++b) {
        if (mb[b].put_pending != 0u && PutDone(&mb[b])) SettlePutLocked(&mb[b]);
      }
    }
    UnlockBlock();
  }

  /**
   * Is `page_num` resident AND finished arriving? Never blocks, never faults.
   *
   * This is the query a yieldable kernel tests before suspending: a page that
   * is claimed but still `fetching` is NOT usable, so it must report false or
   * the caller would read a slot whose transfer has not landed. Lock-free on
   * purpose -- taking the block lock here would serialise every thread of the
   * block on what is meant to be a cheap predicate, and the claim protocol
   * publishes `fetching` before `page_num` precisely so a scanner cannot see a
   * half-claimed slot as ready.
   */
  CTP_GPU_FUN bool IsResident(clio::run::u64 page_num) const {
    const Page *p = Find(page_num);
    return p != nullptr && !p->fetching;
  }

  CTP_GPU_FUN bool BeginFetch(clio::run::u64 page_num,
                              bool is_prefetch = true) {
    LockBlock();
    const bool ok = BeginFetchLocked(page_num, is_prefetch);
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

  /** Wait for the puts started by FlushAsync over the same range. */
  CTP_GPU_FUN void WaitFlush(clio::run::u64 off, clio::run::u64 count) {
    LockBlock();
    ForEachResident(off, count, [this](Page *p) {
      if (p->flushing) AwaitPut(p);
    });
    UnlockBlock();
  }

 private:
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
      // Durable copy for kEvictUser. `score` above is left as-is so the
      // frequency policy still sees the hint the way it always did; this one
      // survives the touch bumps, the -0.02 aging and the refill resets that
      // erase `score` within ~50 claims.
      p->user_score = score;
      p->has_user = 1u;
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
    // CLAMPED into the CTE's [0,1], for the same reason SubmitPut clamps: the
    // device-side score is an unbounded eviction rank while the CTE's is a
    // tier preference, and PodReorganizeBlob REJECTS anything outside [0,1].
    //
    // Callers say "pin this page" as 1000.0f and "release it" as -1000.0f --
    // the flush and overlap benchmarks and the semantics test all do. Every
    // one of those hints was therefore refused by the CTE and did nothing,
    // silently, because a rescore is fire-and-forget and nobody reads its
    // return code. The prefetch-hint feature was not merely weak, it was
    // inert. Clamping preserves the intent exactly: >=1 means the fastest
    // tier, <=0 means the slowest.
    t->new_score_ = fminf(fmaxf(score, 0.0f), 1.0f);
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
  /** BlockIndex() memo -- see the comment there. `mutable` because the
   *  accessor is const and this is a pure cache of a pure function. */
  mutable clio::run::u32 bi_raw_ = 0;
  mutable clio::run::u32 bi_val_ = 0;
  mutable bool bi_valid_ = false;
  Page *last_page_ = nullptr;
  /** Page number last_page_ was holding when it was cached, so PinHeld can
   *  detect a sibling block re-tenanting the slot in a SHARED shard cache. */
  clio::run::u64 last_pn_ = kNoPage;
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
  /** Bounded spin backoff. __nanosleep is CUDA Volta+ only; ROCm (and
   *  pre-Volta) callers just spin -- the same convention as
   *  ipc_gpu2cpu_impl.h, and the reason no raw __nanosleep may appear in
   *  this header. */
  CTP_GPU_FUN static void Backoff(unsigned ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(ns);
#else
    (void)ns;
#endif
  }

  CTP_GPU_FUN void LockBlock() {
    if (h_->block_locks_ == nullptr) return;
    int *lk = h_->block_locks_ + BlockIndex();
    while (atomicCAS(lk, 0, 1) != 0) {
      Backoff(32);
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
  /**
   * DIAGNOSTIC: a resolved slot MUST live in this block's own table. A
   * pointer outside it means the lookup and the fetch are addressing
   * different tables, and the guard hands out a frame nothing fetched into.
   * `site` identifies which resolve produced it.
   */
  CTP_GPU_FUN void DbgCheckInTable(const Page *p, clio::run::u32 site) const {
    if (h_->stat_bad_slot_ == nullptr || p == nullptr) return;
    const Page *tbl = BlockPages();
    if (p < tbl || p >= tbl + h_->pages_per_block_) {
      atomicAdd(h_->stat_bad_slot_, 1ull);
      atomicMax(reinterpret_cast<unsigned long long *>(h_->stat_bad_slot_ + 1),
                static_cast<unsigned long long>(site));
    }
  }

  CTP_GPU_FUN clio::run::u32 BlockIndex() const {
    // CACHED. The modulo is the point: a GPU has no integer divide, so
    // `raw % nblocks_` is a software sequence, and this accessor is reached
    // from BlockPages(), Find(), Ways() and most other helpers -- measured at
    // ~350 invocations inside a SINGLE HoldPage, 700 PTX instructions of
    // nothing but re-deriving a value that cannot change within a block.
    //
    // Keyed on `raw`, not computed once, because `block_override_` is public
    // and kernels assign it AFTER construction (`dst.block_override_ = 0;`).
    // A cache that ignored that would silently serve another block's table --
    // the same class of bug as the stale-slot page corruption. Keying on the
    // input makes a changed override recompute by construction.
    const clio::run::u32 raw =
        block_override_ != kNoBlockOverride ? block_override_ : blockIdx.x;
    if (!bi_valid_ || raw != bi_raw_) {
      bi_raw_ = raw;
      bi_val_ = raw % h_->nblocks_;
      bi_valid_ = true;
    }
    return bi_val_;
  }

 public:
  /**
   * Diagnostic: which page is parked in slot `sl` of THIS block right now.
   * Exposed so a kernel can look for a page occupying two slots at the exact
   * moment it reads a bad value -- a transient duplicate is invisible to any
   * host-side check that runs once the kernel is already quiescent.
   */
  /** Diagnostic: the frame address slot `sl` is backed by. */
  /** Diagnostic: index of a slot within THIS block's table, or -1. */
  CTP_GPU_FUN long long DbgSlotIndexOf(const void *slot) const {
    if (slot == nullptr) return -1;
    const Page *tbl = BlockPages();
    const Page *p = static_cast<const Page *>(slot);
    if (p < tbl || p >= tbl + h_->pages_per_block_) return -1;
    return static_cast<long long>(p - tbl);
  }

  CTP_GPU_FUN const void *DbgFrameAt(clio::run::u32 sl) const {
    return BlockPages()[sl].data;
  }

  CTP_GPU_FUN clio::run::u64 DbgGetPageAt(clio::run::u32 sl) const {
    return *reinterpret_cast<volatile clio::run::u64 *>(
        &BlockPages()[sl].dbg_get_page);
  }

  CTP_GPU_FUN clio::run::u64 PageAt(clio::run::u32 sl) const {
    return *reinterpret_cast<volatile clio::run::u64 *>(
        &BlockPages()[sl].page_num);
  }

 private:
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
    // O(d), independent of the cache size. This was a hint probe plus a FULL
    // TABLE SCAN on miss: the hint made a hit one compare, but a miss -- the
    // common case out of core, 99.98% of accesses at 1/16 coverage -- read
    // every slot, and a Page is ~200 bytes so each is its own cache line.
    // That single scan is why runtime grew with the cache size.
    //
    // Correct ONLY while every placement goes through WaySlot(). Any path that
    // parks a page outside its d candidates makes it invisible here, and the
    // caller then fetches it AGAIN into a second slot -- two slots holding one
    // page, silently. BeginFetchLocked's first-free-slot scan did exactly that
    // and was removed.
    const clio::run::u32 nw = Ways();
    for (clio::run::u32 w = 0; w < nw; ++w) {
      Page *p = &tbl[WaySlot(page_num, w)];
      if (p->page_num == page_num) return p;
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
    XferAdd(1);
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
    // CLAMPED to 1.0. The two scores are different quantities that happen to
    // share a name: device-side `score` is an eviction RANK -- 2.0 on fault,
    // +1.0 per touch, unbounded -- while the CTE's is a tier preference in
    // [0,1] and PutBlob REJECTS anything above 1.0 with rc=5. So every
    // writeback of a page that had been touched more than once failed, and
    // failed silently: AwaitPut did not read the return code, so the page was
    // marked clean and its contents dropped. In the workloads test that lost
    // 7 of 8 pages -- the reader saw the seeded values, never the computed
    // ones. Clamping is the right direction as well as the safe one: a hot
    // page wants the FASTEST tier, which is what 1.0 asks for.
    t->score_ = (p->score > 0.0f) ? fminf(p->score, 1.0f) : -1.0f;
    t->flags_ = 0;
    t->context_ = clio::cte::core::Context();
    t->context_.compress_lib_ = h_->compress_lib_;
    t->context_.compress_preset_ = h_->compress_preset_;
    ClearRunCtx(t);
    Bump(h_->stat_puts_);
    // PUBLISH BEFORE ISSUING THE TRANSFER, not after. Send() queues the
    // put, so its DMA may begin immediately; clearing `dirty` afterwards
    // leaves a window in which a store is BOTH missed by the DMA and
    // stripped of its dirty flag -- silently lost, with the page then
    // looking clean and droppable. Done in this order the invariant holds:
    // everything written before this point is captured by the transfer,
    // and anything written after it re-dirties the page and survives.
    // Setting `flushing` first also means a concurrent write-hold sees it
    // and backs off (see ProbeHold) rather than racing the DMA.
    p->dirty = 0u;
    p->flushing = 1u;
    __threadfence();
    p->put_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(p->put));
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
      if (!(r < t->count_ && t->reqs_[r].rc_ == 0)) {
        // Same rule as the synchronous batch: a failed record must free its
        // slot, never serve its stale bytes as this page -- and it must be
        // freed BEFORE `fetching` clears, while probes still skip the slot.
        p->page_num = kNoPage;
        Bump(h_->stat_get_errors_);
        __threadfence();
      }
      if (p->fetching == 2u) p->fetching = 0u;
    }
    mb->async_pending = 0u;
    XferAdd(-1);
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
    // The run-fetch (BeginFetchRunLocked) parks its batch on the SYNC slot.
    if (h_->multi_ != nullptr) SettleOneLocked(&BlockBatches()[0]);
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
      // Encoded mapped pages must NEVER ride a CPU-serviced get (it can
      // channel-order behind the resident kernel). This is a prefetch HINT,
      // so dropping the page is always safe; a toucher decodes it in-kernel.
      const clio::run::u32 slot = ClaimSlotWindowLocked(pg);
      if (slot == ~0u) break;
      Page *np = &tbl[slot];
      np->gen += 1u;
      np->pins = 0u;
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
      np->user_score = 0.0f;
      np->has_user = 0u;   // hint belonged to the displaced page
      np->last_access = Now();
      char name[32];
      PageBlobName(pg, name);
      mb->get->Add(name, 0, h_->page_bytes_, RawPtr(np->data));
      np->dbg_get_page = pg;
      mb->page_slot[filled] = slot;
      ++filled;
      Bump(h_->stat_faults_);
      Bump(h_->stat_prefetches_);
      if (h_->fault_hist_ != nullptr) atomicAdd(&h_->fault_hist_[pg], 1u);
    }
    if (filled == 0) return 0;
    XferAdd(1);
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
    // Same POD-reuse rule as SubmitGetAsync/BeginFetchRunLocked: a still-
    // pending run-fetch on the sync slot must be settled before its task is
    // rewritten, or the new Send's completion-flag reset races the old
    // completion across PCIe.
    if (mb[0].async_pending != 0u) {
      SettleOneLocked(&mb[0]);
    }
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
      np->user_score = 0.0f;
      np->has_user = 0u;   // hint belonged to the displaced page
      np->last_access = Now();
      char name[32];
      PageBlobName(pg, name);
      mb[0].get->Add(name, 0, h_->page_bytes_, RawPtr(np->data));
      np->dbg_get_page = pg;
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
      if (r < t->count_ && t->reqs_[r].rc_ == 0) {
        p->fetching = 0u;
        ++ok;
      } else {
        // A failed get leaves the slot holding whatever it held before. Free
        // it rather than letting the caller read those bytes as this page --
        // serving stale data silently is the one outcome worse than a miss.
        // Freed BEFORE `fetching` clears, while probes still skip the slot.
        p->page_num = kNoPage;
        Bump(h_->stat_get_errors_);
        __threadfence();
        p->fetching = 0u;
      }
    }
    return resident + ok;
  }

  /**
   * Fault a RUN of pages with ONE multi-get, without waiting.
   *
   * This is what lets the CPU-side decoder batch. nvcomp's batched
   * decompress costs ~151us PER LAUNCH regardless of chunk count (2.4us per
   * chunk at 64), so single-page faults can never be competitive at scale:
   * the launch count is the cost. A run of pages per fault, multiplied by
   * the concurrent faulting blocks, is what fills 64-chunk launches.
   *
   * Uses the SYNC multi slot (mb[0]) exactly as FetchPagesBatched does, but
   * Sends without Waiting: pages are marked fetching=2 and the batch is
   * settled by the same SettleBatchLocked machinery that serves the async
   * prefetch slots. Declines (returns 0) when mb[0] is still carrying a
   * previous run. Caller holds the block lock.
   */
  CTP_GPU_FUN clio::run::u32 BeginFetchRunLocked(clio::run::u64 first_page,
                                                 clio::run::u32 n) {
    if (h_->multi_ == nullptr) return 0;
    MultiBatch *mb = BlockBatches();
    if (mb[0].async_pending != 0u) {
      if (!MultiGetDone(&mb[0])) return 0;   // previous run still in flight
      SettleOneLocked(&mb[0]);
    }
    if (n > clio::cte::core::kPodMultiMax) n = clio::cte::core::kPodMultiMax;
    if (n > h_->pages_per_block_) n = h_->pages_per_block_;
    const clio::run::u64 npages_total =
        (h_->size_ + h_->elems_per_page_ - 1) / h_->elems_per_page_;
    Page *tbl = BlockPages();
    PrepareMultiGet(mb[0].get);
    clio::run::u32 filled = 0;
    for (clio::run::u32 k = 0; k < n; ++k) {
      const clio::run::u64 pg = first_page + k;
      if (pg >= npages_total) break;
      // Existence proof required: no stored size means the page has never
      // been written (or the table is not published yet, i.e. seeding), and
      // a run of first-touch gets is how the seed livelocked.
      if (h_->stored_size_ == nullptr || h_->stored_size_[pg] == 0) break;
      if (Find(pg) != nullptr) continue;      // resident or already coming
      const clio::run::u32 slot = ClaimSlotWindowLocked(pg);
      if (slot == ~0u) break;                 // window pinned; run ends here
      Page *np = &tbl[slot];
      np->gen += 1u;
      np->fetching = 2u;                      // settled via SettleBatchLocked
      __threadfence();
      np->page_num = pg;
      np->dirty = 0u;
      np->flushing = 0u;
      np->evicting = 0u;
      np->score = 2.0f;
      np->user_score = 0.0f;
      np->has_user = 0u;   // hint belonged to the displaced page
      np->last_access = Now();
      char name[32];
      PageBlobName(pg, name);
      mb[0].get->Add(name, 0, h_->page_bytes_, RawPtr(np->data));
      np->dbg_get_page = pg;
      mb[0].page_slot[filled] = slot;
      ++filled;
      Bump(h_->stat_faults_);
      if (h_->fault_hist_ != nullptr) atomicAdd(&h_->fault_hist_[pg], 1u);
    }
    if (filled == 0) return 0;
    XferAdd(1);
    mb[0].async_n = filled;
    mb[0].async_pending = 1u;
    mb[0].get_fut =
        clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(mb[0].get));
    return filled;
  }

  CTP_GPU_FUN void AwaitPut(Page *p) {
    if (!p->flushing) return;
    p->put_fut.Wait();
    // A failed writeback must NOT leave the page clean. Clearing `flushing`
    // without reading the return code is how a page whose put was rejected
    // (full tier, no target with space) became indistinguishable from one
    // safely on storage: the slot was then free to be evicted and the only
    // copy of those bytes disappeared. Keep it dirty so the next flush
    // retries, and count it so the loss is visible rather than silent.
    if (p->put->GetReturnCode() != 0) {
      Bump(h_->stat_put_errors_);
      if (h_->trace_put_errors_ != 0u) {
        printf("[gv] page %llu writeback FAILED rc=%d\n",
               (unsigned long long) p->page_num,
               (int) p->put->GetReturnCode());
      }
      p->dirty = 1u;
    }
    p->flushing = 0u;
    XferAdd(-1);
  }

  /**
   * Claim a slot for `page_num` and start its get, without waiting.
   * Caller must hold the block lock.
   */
  CTP_GPU_FUN bool BeginFetchLocked(clio::run::u64 page_num,
                                    bool is_prefetch = true) {
    if (Find(page_num) != nullptr) return true;   // resident or already coming
    Page *tbl = BlockPages();
    // NO FIRST-FREE-SLOT SCAN. This used to take the lowest-indexed free slot
    // anywhere in the table, which broke the hash index two ways: the page
    // became invisible to Find (which only probes the d ways), so it would be
    // fetched AGAIN into a second slot; and in the steady state -- a full
    // cache, which is every out-of-core run -- the scan found nothing and fell
    // through here anyway, so it was a guaranteed-futile O(pages_per_block)
    // read of every slot on every fault.
    //
    // ClaimSlotWindowLocked already returns a free candidate when one exists.
    const clio::run::u32 free_slot = ClaimSlotWindowLocked(page_num);
    // Every candidate mid-transfer: report failure rather than stalling, so
    // the caller simply falls back to a demand fault later.
    if (free_slot == ~0u) {
      // DERIVING THE CACHE FLOOR, at the only moment it matters: a claim
      // that found nothing. Count how much of the table is PINNED right
      // now. If that approaches the slot count, every slot is spoken for by
      // a live hold and no block can assemble its working set while the
      // others hold theirs -- the wedge. Sampled on failure only, so the
      // hot path pays nothing and the measurement does not perturb the
      // timing of the thing being measured.
      if (h_->stat_claim_fail_ != nullptr && threadIdx.x == 0) {
        atomicAdd(h_->stat_claim_fail_, 1ull);
        unsigned long long pinned = 0;
        for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
          if (*(volatile clio::run::u32 *) &tbl[i].pins != 0u) ++pinned;
        }
        atomicMax(h_->stat_claim_fail_ + 1, pinned);
      }
      return false;
    }
    Page *p = &tbl[free_slot];
    // Same ordering as the synchronous claim: busy first, then the page
    // number. Reversed, HoldPage's lock-free scan can see a slot that looks
    // resident while its transfer is still in flight.
    p->fetching = 1u;
    __threadfence_block();
    p->gen += 1u;
    p->dbg_get_page = kNoPage;
    p->page_num = page_num;
    p->dirty = 0u;
    p->flushing = 0u;
    p->score = 0.0f;
    p->user_score = 0.0f;
    p->has_user = 0u;   // hint belonged to the displaced page
    SubmitGetAsync(p, page_num, is_prefetch);
    return true;
  }

  /** @return true when `pn` is an encoded page this build can fetch STORED
   *  and decode in the faulting warp. Only meaningful once
   *  PageNeedsHostFetch(pn) has already said the map does not cover it. */
  CTP_GPU_FUN bool SpilledEncodedPage(clio::run::u64 pn) const {
#if defined(CLIO_GV_NVCOMP_DEVICE)
    return h_->compress_lib_ != 0 && h_->cscratch_ != nullptr &&
           h_->stored_size_ != nullptr && h_->stored_size_[pn] != 0 &&
           h_->stored_size_[pn] <= h_->page_bytes_;
#else
    (void) pn;
    return false;
#endif
  }

  /** This slot's compressed-image scratch (one page's worth per slot). */
  CTP_GPU_FUN char *ScratchFor(const Page *p) const {
    const clio::run::u64 idx =
        static_cast<clio::run::u64>(p - BlockPages()) +
        static_cast<clio::run::u64>(BlockIndex()) * h_->pages_per_block_;
    return h_->cscratch_ + idx * h_->page_bytes_;
  }

  /**
   * Fetch a SPILLED encoded page's STORED bytes into scratch, without waiting.
   *
   * The routine fetch asks the compressor for the page's LOGICAL bytes, which
   * costs three task round trips, a codec slot and a stream sync per page --
   * measured ~1.6ms against ~40us for a spilled raw page. This asks for the
   * stored image instead: an EMPTY codec context makes the read a straight
   * pass-through to the core, so the compressor never decompresses it and the
   * faulting warp decodes it itself.
   *
   * Caller must hold the block lock. Marks the page fetching=4, meaning
   * "bytes in flight, NOT yet decoded" -- AwaitFetch and ReapFetched both
   * refuse to publish that state.
   */
  CTP_GPU_FUN bool BeginSpilledFetchLocked(clio::run::u64 page_num) {
    if (Find(page_num) != nullptr) return true;      // resident or coming
    Page *tbl = BlockPages();
    clio::run::u32 slot = h_->pages_per_block_;
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].page_num == kNoPage) { slot = i; break; }
    }
    if (slot == h_->pages_per_block_) {
      slot = ClaimSlotWindowLocked(page_num);
      if (slot == ~0u) return false;                 // no room; try later
    }
    Page *p = &tbl[slot];
    p->fetching = 4u;
    __threadfence_block();
    p->gen += 1u;
    p->dbg_get_page = kNoPage;
    p->page_num = page_num;
    p->dirty = 0u;
    p->flushing = 0u;
    p->evicting = 0u;
    p->score = 0.0f;
    p->user_score = 0.0f;
    p->has_user = 0u;   // hint belonged to the displaced page

    auto *t = p->get;
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = NextTaskId(kKindGet);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodGetBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    char nm[32];
    PageBlobName(page_num, nm);
    t->blob_name_ = nm;
    t->offset_ = 0;
    t->size_ = h_->stored_size_[page_num];
    t->blob_data_ = RawPtr(ScratchFor(p));
    t->flags_ = 0;
    // EMPTY context on purpose: that is what makes this a pass-through and
    // returns the stored bytes. PrepareGet sets the codec precisely to get
    // the opposite behaviour.
    t->context_ = clio::cte::core::Context();
    ClearRunCtx(t);
    Bump(h_->stat_faults_);
    p->get_fut =
        clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(p->get));
    if (p->get_fut.IsNull()) {
      Bump(h_->stat_get_errors_);
      p->page_num = kNoPage;
      __threadfence();   // freed before `fetching` clears; probes skip it
      p->fetching = 0u;
      return false;
    }
    return true;
  }

  /** @return true when `pn`'s stored bytes have landed and it is waiting to
   *  be decoded by the block. */
  CTP_GPU_FUN bool SpilledReady(clio::run::u64 pn) const {
    const Page *p = Find(pn);
    return p != nullptr && p->fetching == 4u &&
           p->get->fut_.is_complete_.load() != 0;
  }

  /**
   * Issue this page's get and return immediately.
   *
   * @param is_prefetch true when this get was asked for AHEAD of the access
   *        (an explicit BeginFetch hint), false when it is servicing a fault
   *        that has already happened. Both are faults; only the former is a
   *        prefetch, and conflating them made stat_prefetches_ meaningless
   *        the moment the demand path started routing through here.
   */
  CTP_GPU_FUN void SubmitGetAsync(Page *p, clio::run::u64 page_num,
                                  bool is_prefetch = true) {
    // NEVER reuse a task POD the runtime may still be reading -- the same
    // rule RescoreLocked has always enforced, adopted here after it was
    // caught red-handed: the new Send resets the POD's is_complete_ to 0,
    // and that device store RACES the PREVIOUS request's CPU-side
    // completion store of 1 across PCIe. When the old store wins, the new
    // fetch reads as already complete: the page is published and written
    // while the CPU worker is still queued to copy stale blob bytes over
    // it. Captured live (task trace + write-then-readback audit): three
    // gets through one POD, the last one's stale copy landing microseconds
    // after the page's fresh content -- the intermittent rebuild
    // stale-page corruption (~25% of runs). Waiting here makes the device
    // OBSERVE the previous completion before the reset, closing the race;
    // in the common case the flag is already 1 and this is a single read.
    if (!p->get_fut.IsNull()) p->get_fut.Wait();
    PrepareGet(p, page_num);
    Bump(h_->stat_faults_);
    if (is_prefetch) Bump(h_->stat_prefetches_);
    XferAdd(1);
    p->fetching = 1u;                     // already set by the claim; keep it
    p->dbg_get_page = page_num;
    p->get_fut = clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(p->get));
    // DIAGNOSTIC (publish-before-fill hunt): the POD's completion flag must
    // read NOT-COMPLETE immediately after a Send -- the Send resets it to 0.
    // If it reads 1 here, the reset lost the race against the PREVIOUS
    // request's CPU-side completion store, and this brand-new get already
    // looks finished: ReapFetched will clear `fetching` and publish the frame
    // before a single byte of the new page has landed.
    if (h_->stat_early_complete_ != nullptr &&
        p->get->fut_.is_complete_.load() != 0) {
      atomicAdd(h_->stat_early_complete_, 1ull);
    }
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
    if (p->fetching == 4u) {
      // Compressed bytes are in flight (or landed) in cscratch and the page
      // has NOT been decoded yet. Publishing it here would hand the caller
      // the COMPRESSED image as page data -- silent corruption. Do not touch
      // it: the decode is block-collective (the nvcomp warp decoder needs all
      // 32 lanes) and HoldPageCoro drives it once the whole block is present.
      // Spinning would deadlock instead: thread 0 reaches this from
      // HoldPageCoro's prologue while the rest of the block waits at the
      // barrier that would release the decode.
      return;
    }
    if (p->fetching == 3u) {
      // In-kernel decode (DecodePageWarp) in flight: the slot's get_fut is
      // STALE — waiting on it would return instantly and expose half-written
      // bytes. The decoder publishes fetching=0 without needing this block's
      // lock, so spinning here is deadlock-free even for lock-holders.
      volatile clio::run::u32 *f =
          reinterpret_cast<volatile clio::run::u32 *>(&p->fetching);
      while (*f == 3u) {
        Backoff(64);
      }
      return;
    }
    p->get_fut.Wait();
    if (p->get->GetReturnCode() != 0) Bump(h_->stat_get_errors_);
    p->fetching = 0u;
    XferAdd(-1);
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
          p->dbg_get_page = kNoPage;
          p->page_num = page_num;
          p->dirty = 0u;
          p->flushing = 0u;
          p->score = 0.0f;
    p->user_score = 0.0f;
    p->has_user = 0u;   // hint belonged to the displaced page
          SubmitGet(p, page_num);
          break;
        }
        // The claim declined: no CLEAN victim among the probe slots. Free one
        // HERE, in THAT SAME WINDOW, or this loop cannot terminate -- it runs
        // with the block lock held, so no reap and no other thread can free a
        // slot underneath it.
        //
        // This previously read `continue;` followed by an unreachable
        // EvictLocked(1). Nothing ever freed a slot, so the loop spun forever
        // holding the lock. EvictLocked would not have been correct either:
        // it scans the whole table and above kProbeWindow pages per block it
        // frees slots this claim never probes.
        if (!EvictWindowLocked(page_num)) {
          // Every window slot is mid-fetch and none could be waited on:
          // back off briefly and re-probe.
          Backoff(200);
        }
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

/** White-box access to DeviceVector machinery, for the tests and benches
 *  that measure it directly. Production kernels use the public tiers; if a
 *  kernel needs something from here, the public API is missing a verb --
 *  add the verb, do not widen this. */
struct DeviceVectorTestAccess {
  // ---- register/size attribution (benchmark/gpu_vector_regprobe.cc) -------
  // Forwarders so one compile-only kernel can exercise ONE internal and
  // `cuobjdump -res-usage` can price it. Templates: uninstantiated in every
  // build that does not use them, so they cost nothing.
  template <typename T>
  static CTP_GPU_FUN clio::run::u64 ProbeHold(DeviceVector<T> &v,
                                              clio::run::u64 off) {
    return v.ProbeHold(off, 1, false);
  }
  template <typename T>
  static CTP_GPU_FUN clio::run::u64 TryHoldFast(DeviceVector<T> &v,
                                                clio::run::u64 off) {
    return v.TryHoldFast(off, 1, false);
  }
  template <typename T>
  static CTP_GPU_FUN const void *Find(DeviceVector<T> &v, clio::run::u64 pn) {
    return v.Find(pn);
  }
  template <typename T>
  static CTP_GPU_FUN clio::run::u32 ClaimSlot(DeviceVector<T> &v,
                                              clio::run::u64 pn) {
    v.LockBlock();
    const clio::run::u32 r = v.ClaimSlotWindowLocked(pn);
    v.UnlockBlock();
    return r;
  }
  template <typename T>
  static CTP_GPU_FUN clio::run::u32 FetchBatchedLocked(DeviceVector<T> &v,
                                                       clio::run::u64 pn) {
    v.LockBlock();
    const clio::run::u32 r = v.FetchPagesBatchedLocked(pn, 8);
    v.UnlockBlock();
    return r;
  }
  template <typename T>
  static CTP_GPU_FUN clio::run::u32 FetchRunLocked(DeviceVector<T> &v,
                                                   clio::run::u64 pn) {
    v.LockBlock();
    const clio::run::u32 r = v.BeginFetchRunLocked(pn, 8);
    v.UnlockBlock();
    return r;
  }
  template <typename T>
  static CTP_GPU_FUN void StartEviction(DeviceVector<T> &v,
                                        clio::run::u64 pn) {
    v.StartEvictionAsync(pn);
  }
  template <typename T>
  static CTP_GPU_FUN void EvictPages(DeviceVector<T> &v, clio::run::u32 n) {
    v.EvictPages(n);
  }
  template <typename T>
  static CTP_GPU_FUN void FlushRangeLocked(DeviceVector<T> &v,
                                           clio::run::u64 off) {
    v.LockBlock();
    v.FlushRangeBatchedAsyncLocked(off, 4096);
    v.UnlockBlock();
  }
  template <typename T>
  static CTP_GPU_FUN void SettleBatch(DeviceVector<T> &v) {
    v.LockBlock();
    v.SettleBatchLocked();
    v.UnlockBlock();
  }
  template <typename T>
  static CTP_GPU_FUN void ReapFetched(DeviceVector<T> &v) { v.ReapFetched(); }
  template <typename T>
  static CTP_GPU_FUN void ReapFlushed(DeviceVector<T> &v) { v.ReapFlushed(); }

  template <typename T>
  static CTP_GPU_FUN clio::run::u32 FetchPagesBatched(DeviceVector<T> &v,
                                                      clio::run::u64 first,
                                                      clio::run::u32 n) {
    return v.FetchPagesBatched(first, n);
  }
  template <typename T>
  static CTP_GPU_FUN clio::run::u32 FetchPagesBatchedAsync(
      DeviceVector<T> &v, clio::run::u64 first, clio::run::u32 n) {
    return v.FetchPagesBatchedAsync(first, n);
  }
  // The pre-six-verb machinery, reachable for the white-box tests that probe
  // it. Production code uses the public verbs only.
  template <typename T>
  static CTP_GPU_FUN void DropAll(DeviceVector<T> &v) {
    v.DropAll();
  }
  template <typename T>
  static CTP_GPU_FUN bool IsResident(DeviceVector<T> &v, clio::run::u64 pn) {
    return v.IsResident(pn);
  }
  template <typename T>
  static CTP_GPU_FUN void RescorePage(DeviceVector<T> &v, clio::run::u64 pn,
                                      float score) {
    v.RescorePage(pn, score);
  }
  template <typename T>
  static CTP_GPU_FUN void BeginFlush(DeviceVector<T> &v, clio::run::u64 off,
                                     clio::run::u64 count) {
    v.BeginFlush(off, count);
  }
  template <typename T>
  static CTP_GPU_FUN void WaitFlush(DeviceVector<T> &v, clio::run::u64 off,
                                    clio::run::u64 count) {
    v.WaitFlush(off, count);
  }
  template <typename T>
  static CTP_GPU_FUN clio::run::u32 FlushBlockBatched(DeviceVector<T> &v) {
    return v.FlushBlockBatched();
  }
  template <typename T>
  static CTP_GPU_FUN const T *TryHoldRawConstG(DeviceVector<T> &v,
                                               clio::run::u64 off,
                                               clio::run::u64 count,
                                               clio::run::u64 *run,
                                               clio::run::u32 *gen_out,
                                               Page **slot_out) {
    return v.TryHoldRawConstG(off, count, run, gen_out, slot_out);
  }
  template <typename T>
  static CTP_GPU_FUN bool HoldStillValid(DeviceVector<T> &v, Page *slot,
                                         clio::run::u32 gen) {
    return v.HoldStillValid(slot, gen);
  }
  template <typename T>
  static CTP_GPU_FUN bool AnyTransferInFlight(DeviceVector<T> &v) {
    return v.AnyTransferInFlight();
  }
  template <typename T>
  static CTP_GPU_FUN void Reap(DeviceVector<T> &v) {
    v.ReapFlushed();
    v.ReapFetched();
  }
  template <typename T>
  static CTP_GPU_FUN clio::run::u64 FlushWaitTag(DeviceVector<T> &v) {
    return v.FlushWaitTag();
  }
  template <typename T>
  static CTP_GPU_FUN bool BeginFetch(DeviceVector<T> &v,
                                     clio::run::u64 page_num) {
    return v.BeginFetch(page_num);
  }
};

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_DEVICE_VECTOR_H_
