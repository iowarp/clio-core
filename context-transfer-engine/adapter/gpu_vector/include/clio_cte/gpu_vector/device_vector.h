/* Copyright 2024 IOWarp - BSD 3-Clause License */
/**
 * The device-side view of a gpu_vector: a per-block page cache.
 *
 * Every public function is BLOCK-COLLECTIVE -- all threads of the block enter
 * and leave together, and no co_await sits inside a thread-predicated branch.
 * Stores may be predicated; suspensions may not.
 *
 * A block owns its page table and its four tasks, so nothing here locks.
 */
#ifndef CLIO_CTE_GPU_VECTOR_DEVICE_VECTOR_H_
#define CLIO_CTE_GPU_VECTOR_DEVICE_VECTOR_H_

#include <clio_cte/core/core_tasks.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/gpu/yield_coro.h>
#include <clio_runtime/types.h>

#include "page.h"

namespace clio::cte::gpu_vector {

/** Eviction rank a freshly faulted page starts at. Higher means keep. */
constexpr float kDefaultScore = 1.0f;

/** table_ before Init binds it. */
constexpr clio::run::u32 kUnbound = ~0u;

/** Shared state, one per vector per GPU. */
struct VecHeader {
  Page *pages_;              // nblocks_ * pages_per_block_
  BlockTasks *tasks_;        // nblocks_
  clio::run::u64 page_bytes_;
  clio::run::u64 elems_per_page_;
  clio::run::u64 num_elems_;
  clio::run::u32 nblocks_;          // private: tables. shared: SETS.
  clio::run::u32 pages_per_block_;  // private: frames/table. shared: set_size.
  /** SHARED MODE. 0 = one private table per CUDA block (a page touched by N
   *  blocks is stored N times). 1 = one associative cache for the whole grid,
   *  where a page's home set is Hash(page_num) % nblocks_ -- a function of the
   *  PAGE, not of who is asking. That is the entire difference. */
  clio::run::u32 shared_;
  /** One spin lock per set, kLockStride ints apart so two sets never share a
   *  cache line. Null in private mode: a private table has exactly one writer.
   */
  int *set_locks_;
  /** Occupied frames, and the byte budget expressed in frames. Sets are
   *  OVERPROVISIONED (set_size = 2x budget/page/sets) so collisions have room,
   *  which means the frame array is larger than the budget -- this counter is
   *  what actually holds the cache to its size. */
  unsigned long long *cache_frames_;
  clio::run::u64 max_frames_;
  clio::cte::core::TagId tag_id_;
  clio::run::PoolId pool_id_;
  ctp::ipc::AllocatorId task_alloc_id_;
  int compress_lib_;
  int compress_preset_;
  unsigned long long *stat_faults_;
  unsigned long long *stat_puts_;
  unsigned long long *stat_evicts_;
  /** Transfers that came back non-zero. Load-bearing: a codec that quietly
   *  stored raw bytes shows up here and nowhere else. */
  unsigned long long *stat_get_errors_;
  unsigned long long *stat_put_errors_;
};

/**
 * A held page: the guard IS the pin. Elements [0, run()) are reachable from
 * ptr(); the frame cannot be evicted until the guard dies.
 */
template <typename T>
class Held {
 public:
  CTP_GPU_FUN Held() = default;
  CTP_GPU_FUN Held(Page *page, T *data, clio::run::u64 begin,
                   clio::run::u64 run, bool owns_pin = true)
      : page_(page), data_(data), begin_(begin), run_(run),
        owns_pin_(owns_pin) {}
  CTP_GPU_FUN Held(Held &&o) noexcept { Steal(o); }
  CTP_GPU_FUN Held &operator=(Held &&o) noexcept {
    if (this != &o) {
      Unpin();
      Steal(o);
    }
    return *this;
  }
  CTP_GPU_FUN ~Held() { Unpin(); }

  CTP_GPU_FUN T *ptr() const { return data_; }
  CTP_GPU_FUN clio::run::u64 run() const { return run_; }
  CTP_GPU_FUN explicit operator bool() const { return page_ != nullptr; }
  /** Indexed by ABSOLUTE element offset, like the offset passed to
   *  HoldPage -- not by position within the run. */
  CTP_GPU_FUN T &operator[](clio::run::u64 off) const {
    return data_[off - begin_];
  }
  CTP_GPU_FUN clio::run::u64 begin_off() const { return begin_; }


  /** Set this frame's eviction rank. Higher means keep. */
  CTP_GPU_FUN void Rescore(float rank) const {
    if (page_ != nullptr && threadIdx.x == 0) page_->score = rank;
  }

 private:
  CTP_GPU_FUN void Unpin() {
    // IN SHARED MODE THE HOLD IS NOT THE PIN. Fetch pinned these pages and
    // UnpinRange releases them, so a guard that also unpinned would drop a
    // reference it never took.
    if (page_ != nullptr && owns_pin_) atomicSub(&page_->pins, 1u);
    page_ = nullptr;
  }
  CTP_GPU_FUN void Steal(Held &o) {
    page_ = o.page_;
    data_ = o.data_;
    begin_ = o.begin_;
    run_ = o.run_;
    owns_pin_ = o.owns_pin_;
    o.page_ = nullptr;
  }

  Page *page_ = nullptr;
  T *data_ = nullptr;
  clio::run::u64 begin_ = 0;
  clio::run::u64 run_ = 0;
  bool owns_pin_ = true;
};

template <typename T>
class DeviceVector {
 public:
  // ======================= 1. Public variables =======================

  // (none: a kernel drives the vector entirely through the functions below)

 private:
  // ======================= 2. Private variables ======================

  VecHeader *h_ = nullptr;
  /** Which page table this block owns. Set by Init; kUnbound until then. */
  clio::run::u32 table_ = kUnbound;

 public:
  // ======================= 3. Public functions =======================

  CTP_CROSS_FUN DeviceVector() = default;
  CTP_CROSS_FUN explicit DeviceVector(VecHeader *h) : h_(h) {}

  /**
   * Bind this block to a page table. CALL ONCE, AT THE TOP OF THE KERNEL,
   * before any other verb.
   *
   * `block` is the LOGICAL block index -- yv.Block() under the yield driver,
   * not blockIdx.x. The driver compacts the grid between rounds, so a
   * coroutine resumes on a different CUDA block than it started on; binding
   * to blockIdx.x makes the table move underneath a block mid-hold.
   */
  CTP_GPU_FUN void Init(clio::run::u32 block) {
    if (block >= h_->nblocks_) {
      if (threadIdx.x == 0) {
        printf("[gpu_vector] FATAL Init(%u): only %u tables exist\n", block,
               h_->nblocks_);
      }
      __trap();
    }
    table_ = block;
  }

  CTP_GPU_FUN clio::run::u64 size() const { return h_->num_elems_; }
  CTP_GPU_FUN clio::run::u64 PageBytes() const { return h_->page_bytes_; }
  CTP_GPU_FUN clio::run::u64 ElemsPerPage() const { return h_->elems_per_page_; }
  CTP_GPU_FUN clio::run::u64 PageOf(clio::run::u64 off) const {
    return off / h_->elems_per_page_;
  }
  /** Start of the page containing `off`. Use with PageSpan to fetch WHOLE
   *  pages: a frame is only valid where it was fetched, so anything that
   *  will be flushed as a whole page must be fetched as a whole page. */
  CTP_GPU_FUN clio::run::u64 PageLo(clio::run::u64 off) const {
    return PageOf(off) * h_->elems_per_page_;
  }
  /** Element count from PageLo(off) covering whole pages through
   *  [off, off+count). Pairs with PageLo as a (start, count) argument. */
  CTP_GPU_FUN clio::run::u64 PageSpan(clio::run::u64 off,
                                      clio::run::u64 count) const {
    const clio::run::u64 last = off + (count != 0 ? count - 1 : 0);
    return (PageOf(last) + 1) * h_->elems_per_page_ - PageLo(off);
  }

  CTP_GPU_FUN clio::run::u32 NumTables() const { return h_->nblocks_; }
  /** This block's page table, for diagnostics that dump residency. */
  CTP_GPU_FUN const Page *TableForDebug() const { return Pages(); }
  /** The device header, so a host diagnostic can copy the table out. */
  CTP_CROSS_FUN const VecHeader *HeaderForDebug() const { return h_; }
  CTP_GPU_FUN clio::run::u32 PagesPerTable() const {
    return h_->pages_per_block_;
  }
  CTP_GPU_FUN clio::run::u64 NumPages() const {
    return (h_->num_elems_ + h_->elems_per_page_ - 1) / h_->elems_per_page_;
  }

  /** Ranges a single BeginFetch may name. */
  static constexpr clio::run::u32 kMaxFetchRanges = 8;

  /**
   * Require every page of the given element ranges to become resident, at the
   * expense of other pages. MANDATORY: HoldPage no longer faults, so a page
   * that was not fetched is a caller error.
   *
   * Takes (offset, count) pairs: BeginFetch(off1, n1, off2, n2). Returns
   * once the fetch has been SUBMITTED; pair it with AwaitFetch before
   * holding.
   *
   * Only the named bytes become valid in the frame, and only those bytes may
   * be read back: a hold over elements this fetch did not name traps.
   */
  template <typename... Args>
  __device__ clio::run::gpu::YCoroTask BeginFetch(clio::run::u64 generation,
                                                  Args... args) {
    if (threadIdx.x == 0) Tasks()->fetch_generation = generation;
    __syncthreads();
    clio::run::u64 lo[kMaxFetchRanges], hi[kMaxFetchRanges];
    clio::run::u32 nr = 0;
    GatherRanges(lo, hi, nr, args...);
    // NO ROUNDING. A fetch transfers exactly the elements named. Partial
    // pages are safe because residency is not validity: each frame records
    // the interval it actually holds, and SubmitFetch widens a fetch to the
    // union of that interval and the new request, so a later fetch of a
    // different slice of the same page transfers the gap rather than
    // trusting bytes nobody read.
    // One fetch in flight per block: staging into a task the runtime is still
    // reading would overwrite records mid-transfer.
    if (FetchBusy()) {
      co_await AwaitFetch();
    }
    // Pin what is already resident BEFORE making room, so the eviction that
    // frees frames for the missing pages cannot take a page of this very
    // range and turn one fetch into two.
    // COUNT ON EVERY THREAD. The scan is read-only, so every thread reaches
    // the same number and no broadcast is needed. Voting it with
    // __syncthreads_or would collapse the COUNT to 0/1 and free one frame for
    // a range missing several pages -- which is how "no free frame after
    // EvictPages" fired.
    if (Shared()) {
      // SHARED: no separate count/evict/pin pass. SubmitFetch claims each page
      // in its home set under that set's lock, evicting there if it has to,
      // and leaves it pinned. Those pins persist until the caller's
      // UnpinRange -- that is the whole difference from the private path,
      // which releases them below because there the guard owns the pin.
      if (threadIdx.x == 0) SubmitFetch(lo, hi, nr);
      __syncthreads();
      co_return;
    }
    const clio::run::u32 missing = CountMissing(lo, hi, nr);
    // Pinning mutates, so only thread 0 does it.
    if (threadIdx.x == 0) PinResident(lo, hi, nr);
    __syncthreads();
    // EVICTION IS SIZED BY MISSING PAGES; THE SUBMIT IS NOT.
    //
    // CountMissing counts pages with no frame, which is the right number of
    // frames to free -- but a page can be RESIDENT and still need a
    // transfer: holding a different slice of itself, or an older generation.
    // Gating the submit on `missing` meant those never reached SubmitFetch
    // at all, so a generational fetch of a resident-but-stale page silently
    // returned the stale bytes. SubmitFetch itself decides what to send and
    // sends nothing when everything is current.
    if (missing != 0) {
      if (threadIdx.x == 0 && !EvictPages(missing, missing)) __trap();
      __syncthreads();
    }
    if (threadIdx.x == 0) SubmitFetch(lo, hi, nr);
    if (threadIdx.x == 0) UnpinRanges(lo, hi, nr);
    __syncthreads();
    co_return;
  }

  /**
   * Release the pins a shared-cache Fetch took over [off, off+count).
   *
   * In shared mode FetchRange is the pinner, because a page fetched by this
   * CUDA block can be evicted by ANY other one before the hold happens. The
   * pin closes that window, and it lasts until this call -- so a range you
   * fetched and never unpin is a frame nobody can reclaim, and eventually a
   * set that is full of pinned frames.
   *
   * A no-op in private mode, where the guard owns the pin and dies on its own.
   * Callers can therefore issue it unconditionally, which is what lets one
   * kernel run under both modes for the A/B.
   *
   * Block-collective: call it with the whole block, like Fetch.
   */
  CTP_GPU_FUN void UnpinRange(clio::run::u64 off, clio::run::u64 count) {
    if (!Shared() || count == 0) return;
    __syncthreads();
    if (threadIdx.x == 0) {
      const clio::run::u64 lo = off;
      const clio::run::u64 hi = off + count;
      UnpinRanges(&lo, &hi, 1u);
    }
    __syncthreads();
  }

  /** Wait for the outstanding BeginFetch and publish its pages. */
  __device__ clio::run::gpu::YCoroTask AwaitFetch() {
    CLIO_CO_YIELD_WHEN(;, FetchBusy() && !FetchDone(), FetchTag());
    if (threadIdx.x == 0 && FetchBusy()) PublishFetch();
    __syncthreads();
    co_return;
  }

  /**
   * BeginFetch then AwaitFetch. Same ranges, same rules -- this is the whole
   * of it, for callers with nothing to overlap the transfer with. Splitting
   * the two is what lets a caller compute over one range while the next is
   * in flight; when there is no such work, the split is only noise.
   */
  template <typename... Args>
  __device__ clio::run::gpu::YCoroTask Fetch(clio::run::u64 generation,
                                             Args... args) {
    co_await BeginFetch(generation, args...);
    co_await AwaitFetch();
    co_return;
  }

  /**
   * The page holding `off`. Does NOT fault: BeginFetch/AwaitFetch put it
   * there. A page that is still absent is a caller error and the kernel
   * stops -- proceeding would read another page's bytes.
   */
  __device__ clio::run::gpu::YCoroTaskT<Held<T>> HoldPage(
      clio::run::u64 off, clio::run::u64 count, bool write = false) {
    const clio::run::u64 pn = PageOf(off);
    Page *p = Find(pn);
    if (p == nullptr) {
      co_await AwaitFetch();      // it may simply not have landed yet
      p = Find(pn);
    }
    if (p == nullptr) {
      if (threadIdx.x == 0) {
        Page *tbl = Pages();
        clio::run::u32 have = 0, fetching = 0, free_n = 0;
        for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
          if (tbl[i].page_num == pn) ++have;
          if (tbl[i].fetching) ++fetching;
          if (tbl[i].page_num == kNoPage) ++free_n;
        }
        printf("[gpu_vector] FATAL table=%u page=%llu not resident "
               "(frames holding it=%u fetching=%u free=%u of %u | "
               "fetch_busy=%u fetch_n=%u). HoldPage does not fetch -- name "
               "it in a BeginFetch first.\n",
               Table(), (unsigned long long) pn, have, fetching, free_n,
               h_->pages_per_block_, Tasks()->fetch_busy, Tasks()->fetch_n);
      }
      __trap();
    }
    // RESIDENT IS NOT VALID. The frame may hold a different slice of this
    // page -- a fetch transfers only the range it was given -- and reading
    // the rest would hand back whatever the frame held before.
    if (!write) {
      const clio::run::u64 in = off % h_->elems_per_page_;
      clio::run::u64 run = h_->elems_per_page_ - in;
      if (run > count) run = count;
      if (!Covers(p, static_cast<clio::run::u32>(in),
                  static_cast<clio::run::u32>(in + run))) {
        if (threadIdx.x == 0) {
          printf("[gpu_vector] FATAL table=%u page=%llu: elements [%u,%u) of "
                 "this page were never fetched -- the frame holds [%u,%u). "
                 "Name the range in a Fetch first.\n",
                 Table(), (unsigned long long) PageOf(off),
                 (unsigned) in, (unsigned) (in + run), p->valid_lo,
                 p->valid_hi);
        }
        __trap();
      }
    }
    co_return Pin(p, off, count, write);
  }

  /**
   * Write back only the named element ranges, as (offset, count) pairs. Use
   * this when several blocks write different parts of one page: a full-page
   * flush would send the frame's other bytes too and clobber whatever
   * another block put there.
   */
  template <typename... Rest>
  __device__ clio::run::gpu::YCoroTask BeginFlush(clio::run::u64 generation,
                                                  clio::run::u64 off,
                                                  clio::run::u64 count,
                                                  Rest... rest) {
    if (threadIdx.x == 0) Tasks()->flush_generation = generation;
    __syncthreads();
    clio::run::u64 rlo[kMaxFetchRanges], rhi[kMaxFetchRanges];
    clio::run::u32 nr = 0;
    GatherRanges(rlo, rhi, nr, off, count, rest...);
    CLIO_CO_YIELD_WHEN(;, FlushBusy() && !FlushDone(), FlushTag());
    if (threadIdx.x == 0) {
      if (FlushBusy()) RetireFlush();
      SubmitFlushRanges(rlo, rhi, nr);
    }
    __syncthreads();
    co_return;
  }

  /**
   * Make this block's copy of a range current, sourcing it from a peer's
   * cache when possible instead of from the tier stack.
   *
   * Already resident here -> hold it, no transfer. Otherwise, if the block
   * named by `from` has the page, COPY it frame-to-frame -- at most one page
   * (256 KB in the MD bench), device-to-device, a few hundred nanoseconds
   * against the microseconds a fault through the runtime costs. Otherwise
   * fall back to a fetch, so a wrong `from` costs latency, never
   * correctness.
   *
   * `from` is an argument, never discovered. Working out an owner at runtime
   * needs a page directory, and a directory needs publication, pinning and
   * eviction coordination between blocks -- the coupling this design exists
   * to avoid. NVSHMEM makes the same call: nvshmemx_getmem_block takes the
   * peer as a parameter and the application supplies it from its own
   * decomposition.
   *
   * THE RESULT IS THIS BLOCK'S OWN FRAME, so it is writable like any other
   * hold and flush works normally. Copying is what buys that: a pointer INTO
   * the owner's frame could not be written without clobbering the owner when
   * it publishes.
   *
   * It does NOT make a page safe for two writers. The copy is a snapshot;
   * if two blocks write the same page and both flush, one loses. Writes
   * still need a single owner per range.
   */
  __device__ clio::run::gpu::YCoroTaskT<Held<T>> UpdateRange(
      clio::run::u64 off, clio::run::u64 count, clio::run::u32 from,
      bool write = false, clio::run::u64 generation = 0) {
    const clio::run::u64 pn = PageOf(off);
    // Already here: this is FetchRange semantics -- do not re-transfer a
    // version we hold.
    Page *mine = Find(pn);
    if (mine != nullptr && mine->generation >= generation) {
      co_return Pin(mine, off, count, write);
    }
    // Does the peer have it? Decide ONCE and block-uniformly: a concurrent
    // eviction in the owner can make one thread's probe succeed and
    // another's fail, and divergent threads would hang on the barrier in the
    // fallback's co_await.
    __shared__ Page *s_src;
    if (threadIdx.x == 0) {
      s_src = nullptr;
      if (from != Table() && from < h_->nblocks_) {
        Page *p = FindIn(from, pn);
        if (p != nullptr) {
          // PIN, THEN VERIFY -- never probe-then-pin. Release() clears
          // page_num before a frame is reused and eviction refuses a pinned
          // frame, so a pin that survives the re-read cannot be pulled out
          // from under the copy. The reverse order loses the frame in the
          // window between the two, which is the shape of two bugs already
          // fixed in this tree.
          atomicAdd(&p->pins, 1u);
          __threadfence();
          clio::run::u32 a = 0, b = 0;
          const bool need = PageNeed(pn, off, off + count, &a, &b);
          // The peer having the PAGE is not the peer having the RANGE.
          // The peer having the RANGE is still not the peer having the
          // GENERATION: a copy of last round's bytes is exactly the stale
          // read this is meant to prevent.
          if (static_cast<volatile Page *>(p)->page_num == pn && need &&
              Covers(p, a, b) && p->generation >= generation) {
            s_src = p;
          } else {
            atomicSub(&p->pins, 1u);
          }
        }
      }
    }
    __syncthreads();
    // OUT OF SHARED BEFORE ANYTHING CAN SUSPEND. EvictPages parks, and a park
    // is a kernel exit that destroys shared memory; a Page* left in __shared__
    // across it comes back garbage. A local lives in the coroutine frame,
    // which is in global memory and does survive.
    Page *src = s_src;
    if (src != nullptr) {
      // Room for the copy. The source stays pinned across the eviction, so
      // it cannot be reclaimed while we make space for its copy.
      __shared__ Page *s_dst;
      if (threadIdx.x == 0) s_dst = FindFree();
      __syncthreads();
      Page *dst = s_dst;
      if (dst == nullptr) {
        if (threadIdx.x == 0 && !EvictPages(1, 1)) __trap();
        __syncthreads();
        if (threadIdx.x == 0) s_dst = FindFree();
        __syncthreads();
        dst = s_dst;
      }
      if (dst != nullptr) {
        CopyFrame(dst->data, src->data, h_->page_bytes_);
        __syncthreads();
        if (threadIdx.x == 0) {
          // Clean: the bytes match what the peer had, and this block has
          // written nothing of its own yet.
          dst->page_num = pn;
          dst->valid_lo = src->valid_lo;
          dst->valid_hi = src->valid_hi;
          dst->generation = src->generation;
          dst->fetching = 0u;
          dst->flushing = 0u;
          dst->score = kDefaultScore;
          atomicSub(&src->pins, 1u);
        }
        __syncthreads();
        co_return Pin(dst, off, count, write);
      }
      if (threadIdx.x == 0) atomicSub(&src->pins, 1u);
      __syncthreads();
    }
    co_await Fetch(generation, PageLo(off), PageSpan(off, count));
    Held<T> h = co_await HoldPage(off, count, write);
    co_return static_cast<Held<T> &&>(h);
  }

  /** Wait for the writeback started by BeginFlush. */
  __device__ clio::run::gpu::YCoroTask EndFlush() {
    CLIO_CO_YIELD_WHEN(;, FlushBusy() && !FlushDone(), FlushTag());
    if (threadIdx.x == 0 && FlushBusy()) RetireFlush();
    __syncthreads();
    co_return;
  }

  /**
   * BeginFlush then EndFlush. The put has landed when this returns, so the
   * range is readable by any other block. Prefer the split form inside a
   * loop, where the put of one page overlaps the next page's compute.
   */
  template <typename... Rest>
  __device__ clio::run::gpu::YCoroTask Flush(clio::run::u64 generation,
                                             clio::run::u64 off,
                                             clio::run::u64 count,
                                             Rest... rest) {
    co_await BeginFlush(generation, off, count, rest...);
    co_await EndFlush();
    co_return;
  }

 private:
  // ======================= 4. Private functions ======================

  /** This block's table. Unbound is a caller error, never blockIdx.x: that
   *  fallback silently gave a block a different cache on every relaunch. */
  CTP_GPU_FUN clio::run::u32 Table() const {
    if (table_ == kUnbound) {
      if (threadIdx.x == 0) {
        printf("[gpu_vector] FATAL: vector used before Init(). Call "
               "v.Init(yv.Block()) at the top of the kernel.\n");
      }
      __trap();
    }
    return table_;
  }
  CTP_GPU_FUN Page *Pages() const {
    return h_->pages_ + static_cast<size_t>(Table()) * h_->pages_per_block_;
  }

  /** Ints between adjacent set locks, so two sets never share a cache line. */
  static constexpr clio::run::u32 kLockStride = 32;

  /** True when one cache serves the whole grid. */
  CTP_GPU_FUN bool Shared() const { return h_->shared_ != 0u; }

  /**
   * A page's home set.
   *
   * SHARED: derived from the PAGE, so every CUDA block agrees on where a page
   * lives and stores it once. Mixed, because page_num alone sends a strided
   * access pattern into one set -- the multiply is Fibonacci hashing.
   * PRIVATE: this block's own table, i.e. today's behaviour.
   */
  CTP_GPU_FUN clio::run::u32 SetOf(clio::run::u64 pn) const {
    if (!Shared()) return Table();
    const clio::run::u64 mixed = (pn ^ (pn >> 29)) * 0x9E3779B97F4A7C15ull;
    return static_cast<clio::run::u32>((mixed >> 32) % h_->nblocks_);
  }
  CTP_GPU_FUN Page *SetPages(clio::run::u32 set) const {
    return h_->pages_ + static_cast<size_t>(set) * h_->pages_per_block_;
  }

  /**
   * Take a set's lock. SPINS, and that is correct: the critical section is a
   * bounded scan of one set with no transfer and no suspend in it. The
   * never-spin rule is about waiting on I/O.
   *
   * Thread 0 only -- every caller is already inside a `threadIdx.x == 0`
   * region. Nothing that can suspend may run while this is held.
   */
  CTP_GPU_FUN void LockSet(clio::run::u32 set) {
    if (h_->set_locks_ == nullptr) return;      // private: single writer
    int *w = &h_->set_locks_[static_cast<size_t>(set) * kLockStride];
    while (atomicCAS(w, 0, 1) != 0) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(32);
#endif
    }
    __threadfence();
  }
  CTP_GPU_FUN void UnlockSet(clio::run::u32 set) {
    if (h_->set_locks_ == nullptr) return;
    __threadfence();
    atomicExch(&h_->set_locks_[static_cast<size_t>(set) * kLockStride], 0);
  }

  /** Charge one frame against the byte budget; true if we are now over. */
  CTP_GPU_FUN bool ChargeFrame() {
    if (h_->cache_frames_ == nullptr) return false;
    const unsigned long long prev = atomicAdd(h_->cache_frames_, 1ull);
    return prev + 1ull > h_->max_frames_;
  }
  CTP_GPU_FUN void CreditFrame() {
    // No atomicSub for unsigned long long; adding -1 wraps to the same thing.
    if (h_->cache_frames_ != nullptr) {
      atomicAdd(h_->cache_frames_, ~0ull);
    }
  }
  CTP_GPU_FUN BlockTasks *Tasks() const { return &h_->tasks_[Table()]; }

  /** The frame holding `pn` and readable, or null. */
  /**
   * Copy one frame's bytes, the whole block cooperating.
   *
   * 16 bytes per thread per step. There is no hardware path for this:
   * cp.async / memcpy_async and Hopper's cp.async.bulk all target SHARED
   * memory, and both ends here are page frames in GLOBAL memory, so a
   * vectorized strided loop IS the fast path. Device memcpy() would be a
   * per-thread BYTE loop -- every thread copying every byte.
   *
   * Frames are page-sized slices of one allocation, so src and dst share
   * alignment; the scalar tail covers page sizes that are not a multiple
   * of 16.
   */
  CTP_GPU_FUN void CopyFrame(void *dst, const void *src,
                             clio::run::u64 bytes) const {
    const clio::run::u64 addr =
        reinterpret_cast<clio::run::u64>(dst) |
        reinterpret_cast<clio::run::u64>(src);
    clio::run::u64 done = 0;
    if ((addr & 15u) == 0u && bytes >= 16u) {
      const clio::run::u64 n16 = bytes >> 4;
      auto *d4 = static_cast<int4 *>(dst);
      const auto *s4 = static_cast<const int4 *>(src);
      for (clio::run::u64 i = threadIdx.x; i < n16; i += blockDim.x) {
        d4[i] = s4[i];
      }
      done = n16 << 4;
    }
    auto *d1 = static_cast<char *>(dst);
    const auto *s1 = static_cast<const char *>(src);
    for (clio::run::u64 i = done + threadIdx.x; i < bytes; i += blockDim.x) {
      d1[i] = s1[i];
    }
  }

  /**
   * Probe ANOTHER block's table. Volatile because the owner writes these
   * fields from a different block, and a cached load would happily reuse a
   * slot that has since been released.
   */
  CTP_GPU_FUN Page *FindIn(clio::run::u32 table, clio::run::u64 pn) const {
    Page *tbl = h_->pages_ + static_cast<size_t>(table) * h_->pages_per_block_;
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      volatile Page *c = &tbl[i];
      if (c->page_num == pn && c->fetching == 0u && c->flushing == 0u) {
        return &tbl[i];
      }
    }
    return nullptr;
  }
  /** The frame holding `pn` and readable, searched in its home set. */
  CTP_GPU_FUN Page *Find(clio::run::u64 pn) const {
    Page *tbl = SetPages(SetOf(pn));
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      // volatile: in shared mode another CUDA block writes these fields, and
      // a cached load would reuse a frame that has since been re-tagged.
      volatile Page *c = &tbl[i];
      if (c->page_num == pn && !c->fetching) return &tbl[i];
    }
    return nullptr;
  }

  /** Frame holding `pn` even if a fetch into it is still in flight. */
  CTP_GPU_FUN Page *FindClaimed(clio::run::u64 pn) const {
    Page *tbl = SetPages(SetOf(pn));
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      volatile Page *c = &tbl[i];
      if (c->page_num == pn) return &tbl[i];
    }
    return nullptr;
  }

  /** A frame with no page in this set, or null. */
  CTP_GPU_FUN Page *FindFreeIn(clio::run::u32 set) const {
    Page *tbl = SetPages(set);
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].page_num == kNoPage) return &tbl[i];
    }
    return nullptr;
  }
  CTP_GPU_FUN Page *FindFree() const { return FindFreeIn(Table()); }

  /** Drop a resident, unpinned, clean page. */
  CTP_GPU_FUN void Release(Page *p) {
    p->page_num = kNoPage;
    p->valid_lo = 0u;
    p->valid_hi = 0u;
    p->generation = 0;
    p->score = kDefaultScore;
    if (h_->stat_evicts_ != nullptr) atomicAdd(h_->stat_evicts_, 1ull);
  }

  /**
   * Guarantee at least `min` free frames, freeing up to `max`.
   *
   * EVICTION PERFORMS NO I/O. It selects victims and drops them: no transfer,
   * no task submission, no suspend. A frame that is mid-flush or mid-fetch is
   * simply not a candidate; an unpinned, quiet frame is dropped on the spot.
   *
   * This used to `co_await EndFlush()` and retry, which put I/O on the
   * allocation path. Everything else depends on it not doing that -- a
   * bounded, non-suspending eviction is what lets the caller hold a lock
   * across this call.
   */
  CTP_GPU_FUN bool EvictPages(clio::run::u32 min, clio::run::u32 max) {
    const bool ok = FreeSome(min, max);
    if (!ok) ReportStuck(min);
    return ok;
  }

  /**
   * Reclaim frames ANYWHERE to get back under the byte budget.
   *
   * BUDGET EVICTION, not slot eviction -- do not confuse the two. This one
   * answers "the cache is over its size"; any set will do, which is why it
   * walks them. Slot eviction answers "this page's home set has no free
   * frame", and only that set can help.
   */
  CTP_GPU_FUN void EvictForBudget(clio::run::u32 want, clio::run::u32 home) {
    if (h_->cache_frames_ == nullptr) return;
    for (clio::run::u32 n = 0; n < h_->nblocks_ && want != 0; ++n) {
      // Start away from the home set: its frames are the ones we just
      // pinned, and it is the set most likely to be contended.
      const clio::run::u32 set = (home + 1u + n) % h_->nblocks_;
      LockSet(set);
      const clio::run::u32 got = FreeSomeIn(set, 0, want);
      UnlockSet(set);
      want = (got >= want) ? 0u : (want - got);
    }
  }

  /**
   * Find `pn` in its home set, or claim a frame for it. Thread 0 only.
   *
   * THE WHOLE POINT OF THE LOCK. "Is it here, and if not it is mine to fill"
   * has to be ONE decision: two CUDA blocks fetching the same page must not
   * both miss and both claim a frame for it, or the page is stored twice in
   * one set -- the exact duplication a shared cache exists to remove -- with
   * two transfers racing into two frames.
   *
   * Returns the frame with a pin already taken, or null if the set is full of
   * pinned frames. `is_new` says whether the caller must fill it.
   */
  CTP_GPU_FUN Page *AllocatePage(clio::run::u64 pn, bool *is_new) {
    const clio::run::u32 set = SetOf(pn);
    LockSet(set);
    Page *p = FindClaimed(pn);
    if (p != nullptr) {
      atomicAdd(&p->pins, 1u);
      UnlockSet(set);
      *is_new = false;
      return p;
    }
    p = FindFreeIn(set);
    if (p == nullptr) {
      FreeSomeIn(set, 1, 1);          // slot eviction: only this set can help
      p = FindFreeIn(set);
    }
    if (p == nullptr) {
      UnlockSet(set);
      *is_new = false;
      return nullptr;                 // caller reports; see ReportSetFull
    }
    p->page_num = pn;
    p->valid_lo = 0u;
    p->valid_hi = 0u;
    p->generation = 0;
    // MARK IT IN FLIGHT BEFORE PUBLISHING THE TAG. Find() skips fetching
    // frames, so until the transfer lands no other CUDA block can mistake
    // this claimed-but-empty frame for a resident page -- which it otherwise
    // does, and then traps on the hold because the valid range is [0,0).
    p->fetching = 1u;
    atomicAdd(&p->pins, 1u);
    UnlockSet(set);
    // Charge AFTER releasing the lock: paying the budget back may need other
    // sets' locks, and taking a second lock while holding one is how lock
    // ordering deadlocks start.
    if (ChargeFrame()) EvictForBudget(1u, set);
    *is_new = true;
    return p;
  }

  /** Say which set overflowed and what is holding it. */
  CTP_GPU_FUN void ReportSetFull(clio::run::u64 pn) const {
    const clio::run::u32 set = SetOf(pn);
    Page *tbl = SetPages(set);
    clio::run::u32 pinned = 0, busy = 0;
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].pins != 0u) ++pinned;
      if (tbl[i].fetching || tbl[i].flushing) ++busy;
    }
    printf("[gpu_vector] FATAL set=%u full for page %llu: %u of %u frames "
           "pinned, %u in flight.\n"
           "  Freeing space in another set cannot help -- this page's home is "
           "fixed by the hash. Fetch fewer pages at once, UnpinRange sooner, "
           "or raise the cache size.\n",
           set, (unsigned long long) pn, pinned, h_->pages_per_block_, busy);
  }

  /** One eviction pass over one set; returns how many frames it dropped. */
  CTP_GPU_FUN clio::run::u32 FreeSomeIn(clio::run::u32 set, clio::run::u32 min,
                                        clio::run::u32 max) {
    Page *tbl = SetPages(set);
    clio::run::u32 dropped = 0;
    while (dropped < max) {
      Page *victim = nullptr;
      for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
        Page *c = &tbl[i];
        if (c->page_num == kNoPage) continue;
        if (c->pins != 0u || c->fetching || c->flushing) continue;
        if (victim == nullptr || c->score < victim->score ||
            (c->score == victim->score &&
             c->last_access < victim->last_access)) {
          victim = c;
        }
      }
      if (victim == nullptr) break;
      Release(victim);
      CreditFrame();
      ++dropped;
    }
    (void)min;
    return dropped;
  }

  /** One eviction pass; true when at least `min` frames are free. */
  CTP_GPU_FUN bool FreeSome(clio::run::u32 min, clio::run::u32 max) {
    Page *tbl = Pages();
    clio::run::u32 freed = 0;
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].page_num == kNoPage) ++freed;
    }
    while (freed < max) {
      Page *victim = nullptr;
      for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
        Page *c = &tbl[i];
        if (c->page_num == kNoPage) continue;
        if (c->pins != 0u || c->fetching || c->flushing) continue;
        if (victim == nullptr || c->score < victim->score ||
            (c->score == victim->score &&
             c->last_access < victim->last_access)) {
          victim = c;
        }
      }
      if (victim == nullptr) break;
      Release(victim);
      ++freed;
    }
    return freed >= min;
  }

  /** Say what is holding the table before the trap. */
  CTP_GPU_FUN void ReportStuck(clio::run::u32 min) const {
    Page *tbl = Pages();
    clio::run::u32 pinned = 0, busy = 0;
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].pins != 0u) ++pinned;
      if (tbl[i].fetching || tbl[i].flushing) ++busy;
    }
    // NAME WHAT IS HOLDING IT. "no free frame" without the counts cost a day
    // to decode once already.
    printf("[gpu_vector] FATAL table=%u: need %u free of %u frames, "
           "%u pinned, %u in flight.\n"
           "  Hold fewer pages at once. Eviction performs no I/O, so it "
           "cannot wait for the in-flight ones.\n",
           Table(), min, h_->pages_per_block_, pinned, busy);
  }

  /** Pin a resident frame and build the guard. */
  CTP_GPU_FUN Held<T> Pin(Page *p, clio::run::u64 off, clio::run::u64 count,
                          bool write) {
    if (p == nullptr) return Held<T>();
    // SHARED: the fetch already pinned this frame and UnpinRange releases it,
    // so the guard neither takes nor drops a reference -- it only resolves
    // one. PRIVATE: the guard IS the pin, as before.
    const bool owns = !Shared();
    if (owns) atomicAdd(&p->pins, 1u);
    if (threadIdx.x == 0) {
      p->last_access = clock64();
    }
    const clio::run::u64 in = off % h_->elems_per_page_;
    clio::run::u64 run = h_->elems_per_page_ - in;
    if (run > count) run = count;
    if (write && threadIdx.x == 0) {
      // Writing defines these elements whether or not they were ever
      // fetched, so a first-touch page becomes valid without a transfer.
      // Conservative: the interval is unioned, so writing two disjoint
      // slices of one page claims the gap between them as valid too.
      const clio::run::u32 a = static_cast<clio::run::u32>(in);
      const clio::run::u32 b = static_cast<clio::run::u32>(in + run);
      if (p->valid_hi <= p->valid_lo) {
        p->valid_lo = a;
        p->valid_hi = b;
      } else {
        if (a < p->valid_lo) p->valid_lo = a;
        if (b > p->valid_hi) p->valid_hi = b;
      }
    }
    return Held<T>(p, static_cast<T *>(p->data) + in, off, run, owns);
  }

  // ------------------------------ fetch ------------------------------

  /** Ranges arrive as (offset, count) pairs; stored as [lo, hi). */
  CTP_GPU_FUN void GatherRanges(clio::run::u64 *, clio::run::u64 *,
                                clio::run::u32 &) const {}
  template <typename... Rest>
  CTP_GPU_FUN void GatherRanges(clio::run::u64 *lo, clio::run::u64 *hi,
                                clio::run::u32 &n, clio::run::u64 off,
                                clio::run::u64 count, Rest... rest) const {
    if (n < kMaxFetchRanges && count != 0) {
      lo[n] = off;
      hi[n] = off + count;
      ++n;
    }
    GatherRanges(lo, hi, n, rest...);
  }

  /**
   * The byte extent of element range [lo, hi) that lies inside page `pn`.
   * @return false when the range does not touch the page.
   *
   * This is what makes fetch and flush PARTIAL: a caller that names half a
   * page moves half a page, and two blocks writing different parts of one
   * page no longer overwrite each other.
   */
  /** Element interval of [lo,hi) that falls inside page `pn`, as offsets
   *  WITHIN the page. Returns false when the range misses the page. */
  CTP_GPU_FUN bool PageNeed(clio::run::u64 pn, clio::run::u64 lo,
                            clio::run::u64 hi, clio::run::u32 *a,
                            clio::run::u32 *b) const {
    const clio::run::u64 epp = h_->elems_per_page_;
    const clio::run::u64 pstart = pn * epp;
    const clio::run::u64 x = lo > pstart ? lo : pstart;
    const clio::run::u64 y = hi < pstart + epp ? hi : pstart + epp;
    if (y <= x) return false;
    *a = static_cast<clio::run::u32>(x - pstart);
    *b = static_cast<clio::run::u32>(y - pstart);
    return true;
  }

  /** Does this frame actually hold elements [a,b) of its page? */
  CTP_GPU_FUN bool Covers(const Page *p, clio::run::u32 a,
                          clio::run::u32 b) const {
    return p != nullptr && p->valid_hi > p->valid_lo && p->valid_lo <= a &&
           p->valid_hi >= b;
  }

  CTP_GPU_FUN bool ClipToPage(clio::run::u64 pn, clio::run::u64 lo,
                              clio::run::u64 hi, clio::run::u64 *byte_off,
                              clio::run::u64 *byte_len) const {
    const clio::run::u64 epp = h_->elems_per_page_;
    const clio::run::u64 pstart = pn * epp;
    const clio::run::u64 a = lo > pstart ? lo : pstart;
    const clio::run::u64 b = hi < pstart + epp ? hi : pstart + epp;
    if (b <= a) return false;
    *byte_off = (a - pstart) * sizeof(T);
    *byte_len = (b - a) * sizeof(T);
    return true;
  }

  /** How many pages of the ranges are not resident. Read-only, so every
   *  thread computes the same answer. */
  CTP_GPU_FUN clio::run::u32 CountMissing(const clio::run::u64 *lo,
                                          const clio::run::u64 *hi,
                                          clio::run::u32 nr) const {
    clio::run::u32 missing = 0;
    for (clio::run::u32 r = 0; r < nr; ++r) {
      if (hi[r] <= lo[r]) continue;
      const clio::run::u64 p0 = PageOf(lo[r]);
      const clio::run::u64 p1 = PageOf(hi[r] - 1);
      for (clio::run::u64 pn = p0; pn <= p1; ++pn) {
        if (Find(pn) == nullptr) ++missing;
      }
    }
    return missing;
  }

  /** Pin every resident page of the ranges, so the eviction that makes room
   *  for the missing ones cannot take a page of this very range. */
  CTP_GPU_FUN void PinResident(const clio::run::u64 *lo,
                               const clio::run::u64 *hi, clio::run::u32 nr) {
    for (clio::run::u32 r = 0; r < nr; ++r) {
      if (hi[r] <= lo[r]) continue;
      const clio::run::u64 p0 = PageOf(lo[r]);
      const clio::run::u64 p1 = PageOf(hi[r] - 1);
      for (clio::run::u64 pn = p0; pn <= p1; ++pn) {
        Page *p = Find(pn);
        if (p != nullptr) atomicAdd(&p->pins, 1u);
      }
    }
  }

  /** Undo PinResidentAndCountMissing's pins. */
  CTP_GPU_FUN void UnpinRanges(const clio::run::u64 *lo,
                               const clio::run::u64 *hi, clio::run::u32 nr) {
    for (clio::run::u32 r = 0; r < nr; ++r) {
      if (hi[r] <= lo[r]) continue;
      const clio::run::u64 p0 = PageOf(lo[r]);
      const clio::run::u64 p1 = PageOf(hi[r] - 1);
      for (clio::run::u64 pn = p0; pn <= p1; ++pn) {
        Page *p = Find(pn);
        if (p != nullptr && p->pins != 0u) atomicSub(&p->pins, 1u);
      }
    }
  }

  /** Claim a frame per missing page, fill the bulk get, send it. Thread 0. */
  CTP_GPU_FUN void SubmitFetch(const clio::run::u64 *lo,
                               const clio::run::u64 *hi, clio::run::u32 nr) {
    BlockTasks *bt = Tasks();
    auto *t = bt->fetch;
    t->count_ = 0;
    clio::run::u32 n = 0;
    for (clio::run::u32 r = 0; r < nr; ++r) {
      if (hi[r] <= lo[r]) continue;
      const clio::run::u64 p0 = PageOf(lo[r]);
      const clio::run::u64 p1 = PageOf(hi[r] - 1);
      for (clio::run::u64 pn = p0;
           pn <= p1 && n < clio::cte::core::kPodMultiMax; ++pn) {
        clio::run::u32 a = 0, b = 0;
        if (!PageNeed(pn, lo[r], hi[r], &a, &b)) continue;
        bool is_new = false;
        Page *p = nullptr;
        if (Shared()) {
          // Find-or-claim is ONE decision, taken under the set lock, and it
          // leaves the frame PINNED -- the pin is what stops another CUDA
          // block evicting this page between here and the caller's HoldPage.
          p = AllocatePage(pn, &is_new);
          if (p == nullptr) {
            ReportSetFull(pn);
            __trap();
          }
          (void)is_new;
        } else {
          p = Find(pn);
        }
        // RESIDENT IS NOT THE SAME AS VALID, and VALID IS NOT THE SAME AS
        // CURRENT. A frame holding a different slice of this page needs a
        // transfer; so does one holding an older generation, or a
        // generational fetch would be satisfied by exactly the stale copy it
        // exists to refuse. Widen to the union so the valid range stays one
        // interval.
        const bool current =
            (bt->fetch_generation == 0) || (p != nullptr &&
                                            p->generation >= bt->fetch_generation);
        if (Covers(p, a, b) && current) continue;
        if (p != nullptr && !current) {
          // Stale: refetch the whole page rather than patch an interval whose
          // two halves would then be from different generations.
          p->valid_lo = 0u;
          p->valid_hi = 0u;
        }
        clio::run::u32 flo = a, fhi = b;
        if (p != nullptr) {
          if (p->valid_hi > p->valid_lo) {
            flo = p->valid_lo < a ? p->valid_lo : a;
            fhi = p->valid_hi > b ? p->valid_hi : b;
          }
        } else {
          p = FindFree();
          if (p == nullptr) {
            printf("[gpu_vector] FATAL table=%u: no free frame for page %llu "
                   "after EvictPages\n", Table(), (unsigned long long) pn);
            __trap();
          }
          p->valid_lo = 0u;
          p->valid_hi = 0u;
          if (ChargeFrame()) EvictForBudget(1u, SetOf(pn));
        }
        const clio::run::u64 boff =
            static_cast<clio::run::u64>(flo) * sizeof(T);
        const clio::run::u64 blen =
            static_cast<clio::run::u64>(fhi - flo) * sizeof(T);
        p->fetching = 1u;
        p->page_num = pn;
        const clio::run::u32 pn32 = static_cast<clio::run::u32>(pn);
        // Only the bytes the caller named. The rest of the frame keeps
        // whatever it held; the caller owns what it asked for.
        t->Add("", boff, blen,
               RawPtr(static_cast<char *>(p->data) + boff), 0.5f);
        t->reqs_[n].blob_name_.assign(reinterpret_cast<const char *>(&pn32),
                                      sizeof(pn32));
        bt->fetch_vlo[n] = flo;
        bt->fetch_vhi[n] = fhi;
        bt->fetch_slot[n++] = static_cast<clio::run::u32>(p - h_->pages_);
      }
    }
    bt->fetch_n = n;
    if (n == 0) return;
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = DeviceTaskId(Table(), kKindFetch, bt->seq++);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodMultiGetBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    t->context_ = clio::cte::core::Context();
    t->context_.compress_lib_ = h_->compress_lib_;
    t->context_.compress_preset_ = h_->compress_preset_;
    // OR, NEVER ASSIGN. op_flags_ carries more than the name encoding now,
    // and a plain assignment here wiped the generational bit that the flush
    // had just set -- so every generational put went out as an ordinary one.
    t->context_.op_flags_ |= clio::cte::core::Context::kBlobNameRawInt32;
    // A page nobody has written yet has no blob; the CTE creates it and
    // returns success, so the vector needs no first-touch case of its own.
    t->context_.create_on_get_ = true;
    // GENERATIONAL FETCH: when a generation was named, the runtime does not
    // serve the get until the writer has published it. create_on_get_ stays
    // on for the ordinary case, where a never-written page legitimately reads
    // as empty -- but a generational get is exactly the case where "empty"
    // must not be mistaken for "ready", and the gate runs before
    // create_on_get_ gets to answer.
    if (bt->fetch_generation != 0) {
      t->context_.op_flags_ |= clio::cte::core::Context::kGenerational;
      t->context_.generation_ = bt->fetch_generation;
    }
    ClearRunCtx(t);
    if (h_->stat_faults_ != nullptr) {
      atomicAdd(h_->stat_faults_, static_cast<unsigned long long>(n));
    }
    bt->fetch_busy = 1u;
    t->fut_.is_complete_.store(0);   // reused task: clear the last completion
    bt->fetch_fut =
        clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(t));
  }

  CTP_GPU_FUN bool FetchBusy() const { return Tasks()->fetch_busy != 0u; }
  CTP_GPU_FUN bool FetchDone() const {
    BlockTasks *bt = Tasks();
    return bt->fetch_fut.IsNull() ||
           (bt->fetch->fut_.is_complete_.load() & 1u) != 0u;
  }
  CTP_GPU_FUN clio::run::u64 FetchTag() const {
    return reinterpret_cast<clio::run::u64>(
        &Tasks()->fetch->fut_.is_complete_.x);
  }

  /** Make a landed fetch's pages readable. Thread 0 only. */
  CTP_GPU_FUN void PublishFetch() {
    BlockTasks *bt = Tasks();
    Page *tbl = Pages();
    if (bt->fetch->GetReturnCode() != 0 && h_->stat_get_errors_ != nullptr) {
      atomicAdd(h_->stat_get_errors_, 1ull);
    }
    for (clio::run::u32 i = 0; i < bt->fetch_n; ++i) {
      Page *p = &h_->pages_[bt->fetch_slot[i]];
      // UNDER THE SET LOCK, because this is a read-modify-write on a frame
      // several CUDA blocks are publishing into at once. Unioning without it
      // is a lost update: with eight blocks filling eight slices of one page,
      // ALL EIGHT read valid=[0,0) and the last writer's slice is the only
      // one that survives -- measured, every block saw pre=[0,0).
      const bool locked = Shared();
      if (locked) LockSet(SetOf(p->page_num));
      if (Shared() && p->valid_hi > p->valid_lo) {
        // UNION, DO NOT OVERWRITE. In a shared cache several CUDA blocks fill
        // the SAME frame with different slices of one page. Assigning made
        // the frame claim only whichever landed last, and every other block
        // then failed Covers on a range it had itself just fetched:
        //   "elements [896,1024) were never fetched -- the frame holds
        //    [640,768)"
        // where [640,768) is another block's slice. Private tables never see
        // this: one writer per frame.
        if (bt->fetch_vlo[i] < p->valid_lo) p->valid_lo = bt->fetch_vlo[i];
        if (bt->fetch_vhi[i] > p->valid_hi) p->valid_hi = bt->fetch_vhi[i];
      } else {
        p->valid_lo = bt->fetch_vlo[i];
        p->valid_hi = bt->fetch_vhi[i];
      }
      p->generation = bt->fetch_generation;
      p->score = kDefaultScore;
      p->last_access = clock64();
      __threadfence();
      p->fetching = 0u;      // published last: Find skips a fetching frame
      if (locked) UnlockSet(SetOf(p->page_num));
    }
    bt->fetch_n = 0;
    bt->fetch_busy = 0u;
  }

  // ------------------------------ flush ------------------------------

  CTP_GPU_FUN bool FlushBusy() const { return Tasks()->flush_busy != 0u; }
  CTP_GPU_FUN bool FlushDone() const {
    BlockTasks *bt = Tasks();
    return bt->flush_fut.IsNull() ||
           (bt->flush->fut_.is_complete_.load() & 1u) != 0u;
  }
  CTP_GPU_FUN clio::run::u64 FlushTag() const {
    return reinterpret_cast<clio::run::u64>(
        &Tasks()->flush->fut_.is_complete_.x);
  }

  /** Stage only the named ranges. Thread 0 only. */
  CTP_GPU_FUN void SubmitFlushRanges(const clio::run::u64 *lo,
                                     const clio::run::u64 *hi,
                                     clio::run::u32 nr) {
    BlockTasks *bt = Tasks();
    Page *tbl = Pages();
    auto *t = bt->flush;
    t->count_ = 0;
    clio::run::u32 n = 0, dropped = 0;
    for (clio::run::u32 r = 0; r < nr; ++r) {
      if (hi[r] <= lo[r]) continue;
      const clio::run::u64 p0 = PageOf(lo[r]);
      const clio::run::u64 p1 = PageOf(hi[r] - 1);
      for (clio::run::u64 pn = p0; pn <= p1; ++pn) {
        Page *p = Find(pn);
        if (p == nullptr) continue;
        // `flushing` IS A COUNT, NOT A FLAG. In a shared cache several CUDA
        // blocks flush DISJOINT slices of the same frame at the same time;
        // skipping because someone else is mid-flush silently threw this
        // block's bytes away. Measured: eight blocks writing eight slices of
        // one page, and only the first one's slice reached the store.
        // Private tables never saw it -- one writer per frame.
        if (!Shared() && p->flushing) continue;
        if (n >= clio::cte::core::kPodMultiMax) { ++dropped; continue; }
        clio::run::u64 boff = 0, blen = 0;
        if (!ClipToPage(pn, lo[r], hi[r], &boff, &blen)) continue;
        const clio::run::u32 pn32 = static_cast<clio::run::u32>(pn);
        t->Add("", boff, blen,
               RawPtr(static_cast<char *>(p->data) + boff), 0.5f);
        t->reqs_[n].blob_name_.assign(reinterpret_cast<const char *>(&pn32),
                                      sizeof(pn32));
        bt->flush_slot[n++] = static_cast<clio::run::u32>(p - h_->pages_);
        atomicAdd(&p->flushing, 1u);
      }
    }
    bt->flush_n = n;
    if (dropped != 0) {
      printf("[gpu_vector] FATAL table=%u: flush range needs %u records, one "
             "task carries %u. Split the range.\n",
             Table(), n + dropped, clio::cte::core::kPodMultiMax);
      __trap();
    }
    if (n == 0) return;
    FinishFlushSubmit(t, n);
  }

  /** The task fields every flush submission shares. */
  CTP_GPU_FUN void FinishFlushSubmit(MultiPutSlot *t, clio::run::u32 n) {
    BlockTasks *bt = Tasks();
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = DeviceTaskId(Table(), kKindFlush, bt->seq++);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodMultiPutBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    t->context_ = clio::cte::core::Context();
    t->context_.compress_lib_ = h_->compress_lib_;
    if (bt->flush_generation != 0) {
      // GENERATIONAL PUT: stamp what this writer is publishing, so a reader
      // waiting on this generation is released only once the bytes are in.
      t->context_.op_flags_ |= clio::cte::core::Context::kGenerational;
      t->context_.generation_ = bt->flush_generation;
    }
    t->context_.compress_preset_ = h_->compress_preset_;
    // OR, NEVER ASSIGN. op_flags_ carries more than the name encoding now,
    // and a plain assignment here wiped the generational bit that the flush
    // had just set -- so every generational put went out as an ordinary one.
    t->context_.op_flags_ |= clio::cte::core::Context::kBlobNameRawInt32;
    ClearRunCtx(t);
    if (h_->stat_puts_ != nullptr) {
      atomicAdd(h_->stat_puts_, static_cast<unsigned long long>(n));
    }
    bt->flush_busy = 1u;
    t->fut_.is_complete_.store(0);
    bt->flush_fut =
        clio::run::gpu::IpcManager::GetBlockIpcManager()->Send(SlotPtr(t));
  }

  /** Clear the flags of a landed flush. Thread 0 only. */
  CTP_GPU_FUN void RetireFlush() {
    BlockTasks *bt = Tasks();
    if (bt->flush_n != 0 && bt->flush->GetReturnCode() != 0 &&
        h_->stat_put_errors_ != nullptr) {
      atomicAdd(h_->stat_put_errors_, 1ull);
    }
    Page *tbl = Pages();
    for (clio::run::u32 i = 0; i < bt->flush_n; ++i) {
      Page *p = &h_->pages_[bt->flush_slot[i]];
      atomicSub(&p->flushing, 1u);   // count, so concurrent flushes nest
    }
    bt->flush_n = 0;
    bt->flush_busy = 0u;
  }

  // ----------------------------- plumbing ----------------------------

  /** A task must arrive with a zeroed RunContext; the runtime builds it. */
  template <typename TaskT>
  CTP_GPU_FUN void ClearRunCtx(TaskT *t) const {
    char *raw = reinterpret_cast<char *>(&t->run_ctx_);
    for (unsigned i = 0; i < sizeof(t->run_ctx_); ++i) raw[i] = 0;
  }

  template <typename TaskT>
  CTP_GPU_FUN ctp::ipc::FullPtr<TaskT> SlotPtr(TaskT *task) const {
    ctp::ipc::FullPtr<TaskT> fp;
    fp.shm_.alloc_id_ = h_->task_alloc_id_;
    fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task);
    fp.ptr_ = task;
    return fp;
  }
  CTP_GPU_FUN ctp::ipc::ShmPtr<> RawPtr(void *addr) const {
    ctp::ipc::ShmPtr<> p;
    p.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
    p.off_ = reinterpret_cast<clio::run::u64>(addr);
    return p;
  }
};

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_DEVICE_VECTOR_H_
