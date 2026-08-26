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
  Page *pages_;              // nsets_ * set_size_
  /** ONE TASK SET PER CUDA BLOCK, indexed by the LOGICAL block id. A block
   *  owns its MultiGet and MultiPut outright: two blocks staging into one
   *  slot would overwrite each other's records mid-transfer. Dimensioned by
   *  the LAUNCH, which is why it is not nsets_ -- the cache's geometry and
   *  the grid's width are unrelated numbers that happened to be equal. */
  BlockTasks *tasks_;        // nblocks_
  clio::run::u64 page_bytes_;
  clio::run::u64 elems_per_page_;
  clio::run::u64 num_elems_;
  /** CUDA blocks in the launch this vector was built for = task sets. */
  clio::run::u32 nblocks_;
  /** Associative sets in the cache, and frames in each. */
  clio::run::u32 nsets_;
  clio::run::u32 set_size_;
  /** One spin lock per set, kLockStride ints apart so two sets never share a
   *  cache line. Null in private mode: a private table has exactly one writer.
   */
  int *set_locks_;
  /**
   * THE PAGE ALLOCATOR. Storage is no longer welded to a slot.
   *
   * A slot used to own a page-sized region for its whole life, so a set wide
   * enough to avoid collisions cost a page per spare slot -- 580 spare list
   * slots at 512 KB was 290 MB of emptiness, and that was most of the gap to
   * a resident MPI run. Now `regions_` holds EXACTLY the capacity the caller
   * asked for, a slot is a 64-byte tag that points at a region only while it
   * holds a page, and associativity costs tags rather than pages.
   *
   * Regions are owned PER BLOCK: block b owns
   * [b * regions_per_block_, (b+1) * regions_per_block_), and a free region
   * goes back to its owner rather than to a global pool, so the common path
   * touches only this block's list. Ownership is recomputed from the address
   * (RegionOwner), never stored.
   */
  char *regions_;
  clio::run::u32 nregions_;          // exactly the requested capacity
  clio::run::u32 regions_per_block_; // ownership stride
  /** Per-block free lists: `free_q_` is nregions_ region indices, block b
   *  using [b*regions_per_block_, ...), with head/tail counters per block. */
  clio::run::u32 *free_q_;
  clio::run::u32 *free_head_;
  clio::run::u32 *free_tail_;
  int *free_lock_;                   // one per block, kLockStride apart
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
  /** THE FATAL CHANNEL: 8 slots of PINNED HOST memory the device writes just
   *  before it traps. Neither of the obvious channels survives a trap --
   *  device printf is buffered and dies with the context, and device memory
   *  dies with it too -- so a kernel that traps normally tells you nothing but
   *  "CUDA Error 715: an illegal instruction". Host memory outlives the
   *  context, so a poller sees the reason. Slot 0 is the code (0 = nothing
   *  happened), 1..4 are its arguments, and it is written once and latched. */
  unsigned long long *fatal_;
};

/** Reasons a gpu_vector kernel traps. Slot 0 of the fatal channel. */
enum FatalCode : unsigned long long {
  kFatalNone = 0,
  kFatalInitBlock = 1,      // Init(block) beyond the task table
  kFatalNotResident = 2,    // HoldPage: page absent (args: page, set, have)
  kFatalNotCovered = 3,     // HoldPage: range never fetched
  kFatalUnbound = 4,        // vector used before Init()
  kFatalSetFull = 5,        // AllocatePage: no frame (args: page, set, pinned)
  kFatalFlushSplit = 6,     // flush range needs more records than one task
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
        printf("[gpu_vector] FATAL Init(%u): the vector was built for %u CUDA "
               "blocks, so there is no task set for this one.\n", block,
               h_->nblocks_);
      }
      if (h_->fatal_ != nullptr && threadIdx.x == 0 &&
          atomicCAS(h_->fatal_, kFatalNone, kFatalInitBlock) == kFatalNone) {
        h_->fatal_[1] = block;
        h_->fatal_[2] = h_->nblocks_;
        __threadfence_system();
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

  /** Associative sets in the one cache this vector has. */
  CTP_GPU_FUN clio::run::u32 NumSets() const { return h_->nsets_; }
  /** Every frame, for diagnostics that dump residency. Walk NumFrames() of
   *  them: the array is the WHOLE cache, not one block's share, because
   *  blocks do not have shares. */
  CTP_GPU_FUN const Page *TableForDebug() const { return h_->pages_; }
  CTP_GPU_FUN clio::run::u64 NumFrames() const {
    return static_cast<clio::run::u64>(h_->nsets_) * h_->set_size_;
  }
  /** The device header, so a host diagnostic can copy the table out. */
  CTP_CROSS_FUN const VecHeader *HeaderForDebug() const { return h_; }
  /** Frames in one associative set. */
  CTP_GPU_FUN clio::run::u32 SetSize() const { return h_->set_size_; }
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
    // One fetch in flight per block: staging into a task the runtime is still
    // reading would overwrite records mid-transfer.
    if (FetchBusy()) {
      co_await AwaitFetch();
    }
    // NO SEPARATE COUNT/EVICT/PIN PASS. SubmitFetch claims each page in its
    // home set under that set's lock, evicting there if it has to, and leaves
    // it pinned. Those pins persist until the caller's UnpinRange.
    if (threadIdx.x == 0) SubmitFetch(lo, hi, nr);
    __syncthreads();
    co_return;
  }

  /**
   * Release the pins a Fetch took over [off, off+count).
   *
   * FETCH IS THE PINNER, because a page fetched by this CUDA block can be
   * evicted by ANY other one before the hold happens -- there is one cache
   * and every block is in it. The pin closes that window, and it lasts until
   * this call, so a range you fetched and never unpin is a frame nobody can
   * reclaim and eventually a set that is full of pinned frames.
   *
   * Block-collective: call it with the whole block, like Fetch.
   */
  CTP_GPU_FUN void UnpinRange(clio::run::u64 off, clio::run::u64 count) {
    if (count == 0) return;
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
    // EVERY BRANCH TO A BARRIER MUST BE VOTED, because `p` is derived from
    // state OTHER BLOCKS MUTATE. Two threads of this block can read the same
    // frame a microsecond apart and disagree about whether it is resident,
    // and both paths below contain __syncthreads -- so a plain
    // `if (p == nullptr)` puts part of the block into a barrier the rest never
    // reaches. That is undefined behaviour, and it does not announce itself:
    // it came back as "CUDA Error 715: an illegal instruction" from a kernel
    // with no trap in it, roughly one run in ten, and vanished under
    // CUDA_LAUNCH_BLOCKING and compute-sanitizer alike.
    //
    // Private tables never had this problem: nobody else touched the table,
    // so every thread of the block always agreed.
    if (__syncthreads_or(p == nullptr ? 1 : 0)) {
      co_await AwaitFetch();      // it may simply not have landed yet
      p = Find(pn);
    }
    // WAIT FOR THE PEER THAT CLAIMED IT. In a shared cache the block that
    // claims a frame is the only one that fills it -- a second transfer into
    // a live frame overwrites whatever the other blocks have written into it.
    // So a block that finds the frame claimed but not yet published has
    // nothing to fetch and everything to wait for. The claimer always reaches
    // its own AwaitFetch, so this makes progress.
    //
    // Voted for the same reason, and re-voted every pass: the loop condition
    // is a read of the same peer-mutated state.
    for (;;) {
      if (!__syncthreads_or(p == nullptr ? 1 : 0)) break;
      if (!__syncthreads_or(FindClaimed(pn) != nullptr ? 1 : 0)) break;
      // TAG 0, DELIBERATELY: "no condition, always resume". A wait tag names
      // a COMPLETION WORD the host polls, and what this block waits for --
      // another block publishing a frame -- has no such word. Passing the
      // frame's address instead made the driver test the low bit of
      // `page_num`, which flips for no reason connected to the wait: the
      // round cap hit with 7 of 8 blocks suspended forever. Resuming every
      // round and re-checking costs relaunches; it terminates.
      int once = 0;
      CLIO_CO_YIELD_WHEN(;, once++ == 0, 0ull);
      p = Find(pn);
    }
    if (__syncthreads_or(p == nullptr ? 1 : 0)) {
      if (threadIdx.x == 0) {
        Page *tbl = SetPages(SetOf(pn));
        clio::run::u32 have = 0, fetching = 0, free_n = 0;
        for (clio::run::u32 i = 0; i < h_->set_size_; ++i) {
          if (tbl[i].page_num == pn) ++have;
          if (tbl[i].fetching) ++fetching;
          if (tbl[i].page_num == kNoPage) ++free_n;
        }
        printf("[gpu_vector] FATAL set=%u page=%llu not resident "
               "(frames holding it=%u fetching=%u free=%u of %u | "
               "fetch_busy=%u fetch_n=%u). HoldPage does not fetch -- name "
               "it in a BeginFetch first.\n",
               SetOf(pn), (unsigned long long) pn, have, fetching, free_n,
               h_->set_size_, Tasks()->fetch_busy, Tasks()->fetch_n);
      }
      FatalNote(kFatalNotResident, pn, SetOf(pn), h_->set_size_);
      __trap();
    }
    // RESIDENT IS NOT VALID. The frame may hold a different slice of this
    // page -- a fetch transfers only the range it was given -- and reading
    // the rest would hand back whatever the frame held before.
    if (!write) {
      const clio::run::u64 in = off % h_->elems_per_page_;
      clio::run::u64 run = h_->elems_per_page_ - in;
      if (run > count) run = count;
      // Voted like the residency check above: `valid_lo/hi` is peer-mutated,
      // so threads can disagree, and the report inside is thread-0 only --
      // an unvoted take by some other thread would trap with no message.
      if (__syncthreads_or(!Covers(p, static_cast<clio::run::u32>(in),
                                   static_cast<clio::run::u32>(in + run))
                               ? 1 : 0)) {
        if (threadIdx.x == 0) {
          printf("[gpu_vector] FATAL table=%u page=%llu: elements [%u,%u) of "
                 "this page were never fetched -- the frame holds [%u,%u). "
                 "Name the range in a Fetch first.\n",
                 Table(), (unsigned long long) PageOf(off),
                 (unsigned) in, (unsigned) (in + run), p->valid_lo,
                 p->valid_hi);
        }
        FatalNote(kFatalNotCovered, PageOf(off), in, p->valid_hi);
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
   * Make a range current and hold it.
   *
   * `from` NAMED A PEER'S PAGE TABLE, and there are no peers any more: one
   * associative cache serves the whole grid, so a page another block has IS
   * the page this block has, found in the same set at the same address. The
   * frame-to-frame copy this verb used to perform -- probe the owner's table,
   * pin, re-verify, copy a page device-to-device -- was the private cache
   * paying to undo its own replication. The argument is kept so callers that
   * expressed an ownership hint still compile, and it is ignored.
   *
   * What remains is Fetch + HoldPage, which is what the peer-miss path always
   * fell back to.
   */
  __device__ clio::run::gpu::YCoroTaskT<Held<T>> UpdateRange(
      clio::run::u64 off, clio::run::u64 count, clio::run::u32 from,
      bool write = false, clio::run::u64 generation = 0) {
    (void) from;
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

  /**
   * Latch the reason for a trap where the host can still read it.
   *
   * Thread 0 only, and FIRST WRITER WINS -- a trap is taken by every thread of
   * the block and often by several blocks, and the first one explains the
   * rest. __threadfence_system() is what makes the write cross to the host
   * before the trap kills the context.
   */
  CTP_GPU_FUN void FatalNote(unsigned long long code, unsigned long long a1,
                             unsigned long long a2,
                             unsigned long long a3) const {
    if (h_->fatal_ == nullptr || threadIdx.x != 0) return;
    if (atomicCAS(h_->fatal_, kFatalNone, code) != kFatalNone) return;
    h_->fatal_[1] = a1;
    h_->fatal_[2] = a2;
    h_->fatal_[3] = a3;
    h_->fatal_[4] = static_cast<unsigned long long>(table_);
    __threadfence_system();
  }

  /** This block's table. Unbound is a caller error, never blockIdx.x: that
   *  fallback silently gave a block a different cache on every relaunch. */
  CTP_GPU_FUN clio::run::u32 Table() const {
    if (table_ == kUnbound) {
      if (threadIdx.x == 0) {
        printf("[gpu_vector] FATAL: vector used before Init(). Call "
               "v.Init(yv.Block()) at the top of the kernel.\n");
      }
      if (h_->fatal_ != nullptr && threadIdx.x == 0) {
        atomicCAS(h_->fatal_, kFatalNone, kFatalUnbound);
        __threadfence_system();
      }
      __trap();
    }
    return table_;
  }
  /** Ints between adjacent set locks, so two sets never share a cache line. */
  static constexpr clio::run::u32 kLockStride = 32;

  /** True when one cache serves the whole grid. */

  /**
   * A page's home set.
   *
   * SHARED: derived from the PAGE, so every CUDA block agrees on where a page
   * lives and stores it once. Mixed, because page_num alone sends a strided
   * access pattern into one set -- the multiply is Fibonacci hashing.
   * PRIVATE: this block's own table, i.e. today's behaviour.
   */
  CTP_GPU_FUN clio::run::u32 SetOf(clio::run::u64 pn) const {
    const clio::run::u64 mixed = (pn ^ (pn >> 29)) * 0x9E3779B97F4A7C15ull;
    return static_cast<clio::run::u32>((mixed >> 32) % h_->nsets_);
  }
  CTP_GPU_FUN Page *SetPages(clio::run::u32 set) const {
    return h_->pages_ + static_cast<size_t>(set) * h_->set_size_;
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

  /** Which block owns a region, recomputed from its address.
   *
   * Never stored: the layout IS the mapping. Regions are page-sized and laid
   * out contiguously, so the index is the byte offset over the page size, and
   * ownership is that index over the per-block stride. (A byte offset divided
   * by a block count would not be an index -- the page size has to come out
   * first.) */
  CTP_GPU_FUN clio::run::u32 RegionOwner(const void *region) const {
    const clio::run::u64 off =
        static_cast<clio::run::u64>(static_cast<const char *>(region) -
                                    h_->regions_);
    const clio::run::u64 idx = off / h_->page_bytes_;
    const clio::run::u32 stride =
        h_->regions_per_block_ != 0 ? h_->regions_per_block_ : 1u;
    clio::run::u32 owner = static_cast<clio::run::u32>(idx / stride);
    return owner < h_->nblocks_ ? owner : (h_->nblocks_ - 1u);
  }

  CTP_GPU_FUN void LockFree(clio::run::u32 b) {
    int *w = &h_->free_lock_[static_cast<size_t>(b) * kLockStride];
    while (atomicCAS(w, 0, 1) != 0) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(32);
#endif
    }
    __threadfence();
  }
  CTP_GPU_FUN void UnlockFree(clio::run::u32 b) {
    __threadfence();
    atomicExch(&h_->free_lock_[static_cast<size_t>(b) * kLockStride], 0);
  }

  /**
   * Take a region from block `b`'s free list, or null.
   *
   * LOCKED, NOT LOCK-FREE, and that is a deliberate call. A per-block SPSC
   * ring would let the owner pop without a lock -- but the whole point of the
   * per-block split is that a block whose own list is empty STEALS from
   * another, and a stealer is a second consumer, which is exactly what single
   * consumer rules out. Rather than a ring that is SPSC only until it matters,
   * every list op takes that list's lock: uncontended on the owner's own
   * list, a handful of instructions, and next to a page transfer it does not
   * register.
   */
  CTP_GPU_FUN char *PopRegion(clio::run::u32 b) {
    if (h_->free_q_ == nullptr) return nullptr;
    char *r = nullptr;
    LockFree(b);
    if (h_->free_head_[b] != h_->free_tail_[b]) {
      const clio::run::u32 stride = h_->regions_per_block_;
      const clio::run::u32 pos = h_->free_head_[b] % stride;
      const clio::run::u32 idx = h_->free_q_[b * stride + pos];
      ++h_->free_head_[b];
      r = h_->regions_ + static_cast<clio::run::u64>(idx) * h_->page_bytes_;
    }
    UnlockFree(b);
    return r;
  }

  /** Give a region back to the block that owns it. */
  CTP_GPU_FUN void PushRegion(char *region) {
    if (h_->free_q_ == nullptr || region == nullptr) return;
    const clio::run::u32 b = RegionOwner(region);
    const clio::run::u64 idx =
        static_cast<clio::run::u64>(region - h_->regions_) / h_->page_bytes_;
    const clio::run::u32 stride = h_->regions_per_block_;
    LockFree(b);
    const clio::run::u32 pos = h_->free_tail_[b] % stride;
    h_->free_q_[b * stride + pos] = static_cast<clio::run::u32>(idx);
    ++h_->free_tail_[b];
    UnlockFree(b);
  }

  /** This block's list first, then anyone else's. Null when the cache is
   *  genuinely full and only eviction can help. */
  CTP_GPU_FUN char *AllocRegion() {
    char *r = PopRegion(Table());
    if (r != nullptr) return r;
    for (clio::run::u32 n = 1; n < h_->nblocks_; ++n) {
      r = PopRegion((Table() + n) % h_->nblocks_);
      if (r != nullptr) return r;
    }
    return nullptr;
  }

  CTP_GPU_FUN BlockTasks *Tasks() const { return &h_->tasks_[Table()]; }


  /**
   * The frame holding `pn` and readable, searched in its home set.
   *
   * IN FLIGHT IS NOT THE SAME AS UNREADABLE. `fetching` exists to hide a
   * frame that has been CLAIMED but never filled -- its valid interval is
   * empty and a hold on it would read whatever the frame held before. It must
   * not hide a frame that is already published, because in a shared cache a
   * peer block re-fetching a different slice of the same page raises
   * `fetching` again on a frame this block has already had published to it.
   * Rejecting that frame is how a page that IS resident came back "not
   * resident" (page 9, set 1, have=1, fetching=1, 6 frames free) and trapped.
   *
   * So the test is emptiness, not flight: an empty frame is invisible until
   * someone publishes it; a published frame stays visible while a peer
   * extends it, and HoldPage's Covers check decides whether THIS block's
   * range is among the bytes that landed.
   */
  CTP_GPU_FUN Page *Find(clio::run::u64 pn) const {
    Page *tbl = SetPages(SetOf(pn));
    for (clio::run::u32 i = 0; i < h_->set_size_; ++i) {
      // volatile: in shared mode another CUDA block writes these fields, and
      // a cached load would reuse a frame that has since been re-tagged.
      volatile Page *c = &tbl[i];
      if (c->page_num != pn) continue;
      if (c->fetching != 0u && c->valid_hi <= c->valid_lo) continue;
      return &tbl[i];
    }
    return nullptr;
  }

  /** Frame holding `pn` even if a fetch into it is still in flight. */
  CTP_GPU_FUN Page *FindClaimed(clio::run::u64 pn) const {
    Page *tbl = SetPages(SetOf(pn));
    for (clio::run::u32 i = 0; i < h_->set_size_; ++i) {
      volatile Page *c = &tbl[i];
      if (c->page_num == pn) return &tbl[i];
    }
    return nullptr;
  }

  /** A frame with no page in this set, or null. */
  CTP_GPU_FUN Page *FindFreeIn(clio::run::u32 set) const {
    Page *tbl = SetPages(set);
    for (clio::run::u32 i = 0; i < h_->set_size_; ++i) {
      if (tbl[i].page_num == kNoPage) return &tbl[i];
    }
    return nullptr;
  }

  /** Drop a resident, unpinned, clean page. */
  CTP_GPU_FUN void Release(Page *p) {
    // THE REGION GOES BACK TO ITS OWNER, not to the slot. A slot is a tag
    // now; the storage it pointed at is the scarce thing.
    if (p->data != nullptr) {
      PushRegion(static_cast<char *>(p->data));
      p->data = nullptr;
    }
    p->page_num = kNoPage;
    p->valid_lo = 0u;
    p->valid_hi = 0u;
    p->generation = 0;
    p->score = kDefaultScore;
    if (h_->stat_evicts_ != nullptr) atomicAdd(h_->stat_evicts_, 1ull);
  }


  /**
   * Reclaim REGIONS from anywhere. The caller wants storage, not a slot.
   *
   * Two evictions, still deliberately distinct: this one answers "there is no
   * page-sized region left in the whole cache", and any set can supply one,
   * which is why it walks them. Slot eviction answers "this page's home set
   * has no free TAG", and only that set can help.
   */
  CTP_GPU_FUN void EvictForRegions(clio::run::u32 want, clio::run::u32 home) {
    if (h_->free_q_ == nullptr) return;
    for (clio::run::u32 n = 0; n < h_->nsets_ && want != 0; ++n) {
      // Start away from the home set: its frames are the ones we just
      // pinned, and it is the set most likely to be contended.
      const clio::run::u32 set = (home + 1u + n) % h_->nsets_;
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
    // MARK IT EMPTY AND IN FLIGHT BEFORE PUBLISHING THE TAG, and fence between
    // the two. Find() reads page_num first and rejects a frame that is empty
    // and in flight, so a reader that sees the new tag is guaranteed to see
    // the state that goes with it; publishing the tag first lets a peer match
    // it against the PREVIOUS occupant's valid range and hold a frame whose
    // contents are about to be overwritten.
    // A FREE SLOT IS NOT YET A PLACE TO PUT A PAGE. It needs a region, and
    // the cache being full means the REGIONS are gone, not the tags -- so a
    // failure here is answered by evicting somewhere, anywhere, and trying
    // again, not by widening the set.
    char *region = AllocRegion();
    if (region == nullptr) {
      // DROP THE SET LOCK BEFORE EVICTING ELSEWHERE. EvictForRegions takes
      // OTHER sets' locks, and taking a second set lock while holding one is
      // a cycle: two blocks with different home sets deadlock holding each
      // other's. Free-list locks are fine to nest inside a set lock -- the
      // order is always set then free, never the reverse -- but set-then-set
      // is not, so the window is reopened instead.
      UnlockSet(set);
      EvictForRegions(1u, set);
      LockSet(set);
      // RE-DECIDE FROM SCRATCH. The set was unlocked, so a peer may have
      // claimed this very page, taken the slot, or freed another.
      Page *again = FindClaimed(pn);
      if (again != nullptr) {
        atomicAdd(&again->pins, 1u);
        UnlockSet(set);
        *is_new = false;
        return again;
      }
      p = FindFreeIn(set);
      if (p == nullptr) {
        FreeSomeIn(set, 1, 1);
        p = FindFreeIn(set);
      }
      region = (p != nullptr) ? AllocRegion() : nullptr;
    }
    if (p == nullptr || region == nullptr) {
      if (region != nullptr) PushRegion(region);
      UnlockSet(set);
      *is_new = false;
      return nullptr;                 // caller reports; see ReportSetFull
    }
    p->data = region;
    p->valid_lo = 0u;
    p->valid_hi = 0u;
    p->generation = 0;
    // A COUNT, NOT A FLAG -- see PublishFetch. Several blocks fetch into one
    // frame; whoever publishes first must not clear the others' marks.
    atomicAdd(&p->fetching, 1u);
    __threadfence();
    p->page_num = pn;
    atomicAdd(&p->pins, 1u);
    UnlockSet(set);
    *is_new = true;
    return p;
  }


  /** How many frames of a set are pinned. For the fatal channel. */
  CTP_GPU_FUN clio::run::u32 PinnedInSet(clio::run::u32 set) const {
    Page *tbl = SetPages(set);
    clio::run::u32 n = 0;
    for (clio::run::u32 i = 0; i < h_->set_size_; ++i) {
      if (tbl[i].pins != 0u) ++n;
    }
    return n;
  }

  /** Say which set overflowed and what is holding it. */
  CTP_GPU_FUN void ReportSetFull(clio::run::u64 pn) const {
    const clio::run::u32 set = SetOf(pn);
    Page *tbl = SetPages(set);
    clio::run::u32 pinned = 0, busy = 0;
    for (clio::run::u32 i = 0; i < h_->set_size_; ++i) {
      if (tbl[i].pins != 0u) ++pinned;
      if (tbl[i].fetching || tbl[i].flushing) ++busy;
    }
    printf("[gpu_vector] FATAL set=%u full for page %llu: %u of %u frames "
           "pinned, %u in flight.\n"
           "  Freeing space in another set cannot help -- this page's home is "
           "fixed by the hash. Fetch fewer pages at once, UnpinRange sooner, "
           "or raise the cache size.\n",
           set, (unsigned long long) pn, pinned, h_->set_size_, busy);
  }

  /** One eviction pass over one set; returns how many frames it dropped. */
  CTP_GPU_FUN clio::run::u32 FreeSomeIn(clio::run::u32 set, clio::run::u32 min,
                                        clio::run::u32 max) {
    Page *tbl = SetPages(set);
    clio::run::u32 dropped = 0;
    while (dropped < max) {
      Page *victim = nullptr;
      for (clio::run::u32 i = 0; i < h_->set_size_; ++i) {
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
      Release(victim);   // returns the region to its owner's free list
      ++dropped;
    }
    (void)min;
    return dropped;
  }



  /** Pin a resident frame and build the guard. */
  CTP_GPU_FUN Held<T> Pin(Page *p, clio::run::u64 off, clio::run::u64 count,
                          bool write) {
    if (p == nullptr) return Held<T>();
    // SHARED: the fetch already pinned this frame and UnpinRange releases it,
    // so the guard neither takes nor drops a reference -- it only resolves
    // one. PRIVATE: the guard IS the pin, as before.
    // THE HOLD IS NOT THE PIN. Fetch pinned this frame and UnpinRange
    // releases it; the guard only resolves a pointer into it.
    const bool owns = false;
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



  /** Give back one pin per page of each range. */
  CTP_GPU_FUN void UnpinRanges(const clio::run::u64 *lo,
                               const clio::run::u64 *hi, clio::run::u32 nr) {
    for (clio::run::u32 r = 0; r < nr; ++r) {
      if (hi[r] <= lo[r]) continue;
      const clio::run::u64 p0 = PageOf(lo[r]);
      const clio::run::u64 p1 = PageOf(hi[r] - 1);
      for (clio::run::u64 pn = p0; pn <= p1; ++pn) {
        // FindClaimed, NOT Find. Unpinning is about the frame's IDENTITY, not
        // its readability: a caller may release a range it began fetching and
        // never awaited (a prefetch window it decided not to wait for), and
        // those frames are claimed-but-empty, which is exactly what Find
        // hides. Looking them up with Find silently skipped the unpin and
        // leaked one pin per prefetched page -- the overlap bench filled a
        // 16-frame cache in 16 pages, every frame pinned, nothing in flight.
        Page *p = FindClaimed(pn);
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
        // A SHARED FRAME IS FETCHED WHOLE, ALWAYS.
        //
        // One valid interval per frame cannot describe what several CUDA
        // blocks fetch into one frame. Two blocks asking for disjoint slices
        // of a page BOTH read valid=[0,0), so each transfers only its own
        // slice -- and PublishFetch then unions the two, publishing the GAP
        // BETWEEN THEM as valid although nobody transferred it. A later hold
        // over that gap passes Covers and reads whatever the frame held
        // before: no trap, no error, just wrong numbers. Measured on the MD
        // resort, which lost no atoms at all and still moved the potential
        // energy by 1.6% at 4 blocks and 0.5% at 2 (exact at 1).
        //
        // Fetching the page whole makes the interval degenerate -- always
        // [0, page) -- so there is no hole to misdescribe. It moves more
        // bytes per miss than the caller asked for; it also moves them ONCE
        // for the whole grid instead of once per block, which is the point of
        // sharing. Private tables keep the exact-range behaviour, where one
        // writer per frame makes the interval sound.
        {
          const clio::run::u64 pstart = pn * h_->elems_per_page_;
          const clio::run::u64 avail =
              h_->num_elems_ > pstart ? h_->num_elems_ - pstart : 0;
          const clio::run::u64 whole =
              avail < h_->elems_per_page_ ? avail : h_->elems_per_page_;
          a = 0;
          b = static_cast<clio::run::u32>(whole > b ? whole : b);
        }
        bool is_new = false;
        Page *p = nullptr;
        {
          // Find-or-claim is ONE decision, taken under the set lock, and it
          // leaves the frame PINNED -- the pin is what stops another CUDA
          // block evicting this page between here and the caller's HoldPage.
          p = AllocatePage(pn, &is_new);
          if (p == nullptr) {
            ReportSetFull(pn);
            FatalNote(kFatalSetFull, pn, SetOf(pn), PinnedInSet(SetOf(pn)));
            __trap();
          }
          if (!is_new) {
            const bool stale = bt->fetch_generation != 0 &&
                               p->generation < bt->fetch_generation;
            if (!stale) continue;              // pinned; peer fills it
            if (atomicCAS(&p->fetching, 0u, 1u) != 0u) continue;
            p->valid_lo = 0u;                  // ours to refill, whole
            p->valid_hi = 0u;
            is_new = true;
          }
        }
        // Everything below runs for a frame THIS block just claimed and must
        // fill: a frame a peer owns took the `continue` above, and a stale one
        // was reset to empty when the CAS won it. So there is no residency to
        // re-check, no interval to widen, and no free-frame search -- the
        // claim already did all three.
        const clio::run::u32 flo = a, fhi = b;
        const clio::run::u64 boff =
            static_cast<clio::run::u64>(flo) * sizeof(T);
        const clio::run::u64 blen =
            static_cast<clio::run::u64>(fhi - flo) * sizeof(T);
        // AllocatePage (or the staleness CAS) already marked this frame in
        // flight, and PublishFetch takes that mark back. One mark per record.
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
      LockSet(SetOf(p->page_num));
      if (p->valid_hi > p->valid_lo) {
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
      // A COUNT, LIKE `flushing`. Assigning zero here is one block clearing
      // every other block's mark: with several blocks fetching slices of one
      // frame, the first to publish declared the frame settled while the rest
      // were still in flight.
      atomicSub(&p->fetching, 1u);
      UnlockSet(SetOf(p->page_num));
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
      FatalNote(kFatalFlushSplit, n + dropped, clio::cte::core::kPodMultiMax, 0);
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
