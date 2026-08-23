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

/** Shared state, one per vector per GPU. */
struct VecHeader {
  Page *pages_;              // nblocks_ * pages_per_block_
  BlockTasks *tasks_;        // nblocks_
  clio::run::u64 page_bytes_;
  clio::run::u64 elems_per_page_;
  clio::run::u64 num_elems_;
  clio::run::u32 nblocks_;
  clio::run::u32 pages_per_block_;
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
                   clio::run::u64 run)
      : page_(page), data_(data), begin_(begin), run_(run) {}
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
    if (page_ != nullptr) atomicSub(&page_->pins, 1u);
    page_ = nullptr;
  }
  CTP_GPU_FUN void Steal(Held &o) {
    page_ = o.page_;
    data_ = o.data_;
    begin_ = o.begin_;
    run_ = o.run_;
    o.page_ = nullptr;
  }

  Page *page_ = nullptr;
  T *data_ = nullptr;
  clio::run::u64 begin_ = 0;
  clio::run::u64 run_ = 0;
};

template <typename T>
class DeviceVector {
 public:
  // ======================= 1. Public variables =======================

  /** Page table this block uses; ~0u means blockIdx.x. */
  clio::run::u32 block_override_ = ~0u;

 private:
  // ======================= 2. Private variables ======================

  VecHeader *h_ = nullptr;

 public:
  // ======================= 3. Public functions =======================

  CTP_CROSS_FUN DeviceVector() = default;
  CTP_CROSS_FUN explicit DeviceVector(VecHeader *h) : h_(h) {}

  CTP_GPU_FUN clio::run::u64 size() const { return h_->num_elems_; }
  CTP_GPU_FUN clio::run::u64 PageBytes() const { return h_->page_bytes_; }
  CTP_GPU_FUN clio::run::u64 ElemsPerPage() const { return h_->elems_per_page_; }
  CTP_GPU_FUN clio::run::u64 PageOf(clio::run::u64 off) const {
    return off / h_->elems_per_page_;
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
   * Takes (lo, hi) pairs: BeginFetch(a0, a1, b0, b1). Returns once the fetch
   * has been SUBMITTED; pair it with AwaitFetch before holding.
   */
  template <typename... Args>
  __device__ clio::run::gpu::YCoroTask BeginFetch(Args... args) {
    clio::run::u64 lo[kMaxFetchRanges], hi[kMaxFetchRanges];
    clio::run::u32 nr = 0;
    GatherRanges(lo, hi, nr, args...);
    // One fetch in flight per block: staging into a task the runtime is still
    // reading would overwrite records mid-transfer.
    if (FetchBusy()) {
      co_await AwaitFetch();
    }
    // Pin what is already resident BEFORE making room, so the eviction that
    // frees frames for the missing pages cannot take a page of this very
    // range and turn one fetch into two.
    clio::run::u32 missing = 0;
    if (threadIdx.x == 0) {
      missing = PinResidentAndCountMissing(lo, hi, nr);
    }
    missing = __syncthreads_or(static_cast<int>(missing));
    if (missing != 0) {
      co_await EvictPages(missing, missing);
      if (threadIdx.x == 0) SubmitFetch(lo, hi, nr);
    }
    if (threadIdx.x == 0) UnpinRanges(lo, hi, nr);
    __syncthreads();
    co_return;
  }

  /** Wait for the outstanding BeginFetch and publish its pages. */
  __device__ clio::run::gpu::YCoroTask AwaitFetch() {
    CLIO_CO_YIELD_WHEN(;, FetchBusy() && !FetchDone(), FetchTag());
    if (threadIdx.x == 0 && FetchBusy()) PublishFetch();
    __syncthreads();
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
    co_return Pin(p, off, count, write);
  }

  /** Start writing back every dirty page of this block's table. */
  __device__ clio::run::gpu::YCoroTask BeginFlush() {
    // Wait only while the previous put is still moving. Callers flush inside
    // a loop to overlap the next page's compute, so a landed flush is retired
    // here rather than requiring an EndFlush between every pair.
    CLIO_CO_YIELD_WHEN(;, FlushBusy() && !FlushDone(), FlushTag());
    if (threadIdx.x == 0) {
      if (FlushBusy()) RetireFlush();
      SubmitFlush();
    }
    __syncthreads();
    co_return;
  }

  /** Wait for the writeback started by BeginFlush. */
  __device__ clio::run::gpu::YCoroTask EndFlush() {
    CLIO_CO_YIELD_WHEN(;, FlushBusy() && !FlushDone(), FlushTag());
    if (threadIdx.x == 0) RetireFlush();
    __syncthreads();
    co_return;
  }

 private:
  // ======================= 4. Private functions ======================

  CTP_GPU_FUN clio::run::u32 Table() const {
    return block_override_ == ~0u ? blockIdx.x : block_override_;
  }
  CTP_GPU_FUN Page *Pages() const {
    return h_->pages_ + static_cast<size_t>(Table()) * h_->pages_per_block_;
  }
  CTP_GPU_FUN BlockTasks *Tasks() const { return &h_->tasks_[Table()]; }

  /** The frame holding `pn` and readable, or null. */
  CTP_GPU_FUN Page *Find(clio::run::u64 pn) const {
    Page *tbl = Pages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].page_num == pn && !tbl[i].fetching) return &tbl[i];
    }
    return nullptr;
  }

  /** A frame with no page, or null. */
  CTP_GPU_FUN Page *FindFree() const {
    Page *tbl = Pages();
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].page_num == kNoPage) return &tbl[i];
    }
    return nullptr;
  }

  /** Drop a resident, unpinned, clean page. */
  CTP_GPU_FUN void Release(Page *p) {
    p->page_num = kNoPage;
    p->dirty = 0u;
    p->score = kDefaultScore;
    if (h_->stat_evicts_ != nullptr) atomicAdd(h_->stat_evicts_, 1ull);
  }

  /**
   * Guarantee at least `min` free frames, freeing up to `max`. If there are
   * not enough, wait for this block's outstanding flush to land and retry.
   */
  __device__ clio::run::gpu::YCoroTask EvictPages(clio::run::u32 min,
                                                  clio::run::u32 max) {
    CLIO_CO_YIELD_WHEN(;, (threadIdx.x == 0) ? !FreeSome(min, max) : false,
                       FlushTag());
    co_return;
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
        if (c->pins != 0u || c->dirty || c->fetching || c->flushing) continue;
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
    if (freed >= min) return true;
    // Nothing evictable. If this block has a flush in flight, waiting for it
    // is progress; otherwise the caller is asking for more pages at once than
    // its table holds, and no amount of waiting fixes that.
    if (FlushBusy()) return false;
    ReportStuck(min);
    __trap();
    return false;
  }

  /** Say what is holding the table before the trap. */
  CTP_GPU_FUN void ReportStuck(clio::run::u32 min) const {
    Page *tbl = Pages();
    clio::run::u32 pinned = 0, dirty = 0;
    for (clio::run::u32 i = 0; i < h_->pages_per_block_; ++i) {
      if (tbl[i].pins != 0u) ++pinned;
      if (tbl[i].dirty) ++dirty;
    }
    printf("[gpu_vector] FATAL table=%u: need %u free of %u frames, "
           "%u pinned %u dirty, no flush in flight.\n"
           "  Hold fewer pages at once, or flush before the dirty set fills "
           "the table.\n",
           Table(), min, h_->pages_per_block_, pinned, dirty);
  }

  /** Pin a resident frame and build the guard. */
  CTP_GPU_FUN Held<T> Pin(Page *p, clio::run::u64 off, clio::run::u64 count,
                          bool write) {
    if (p == nullptr) return Held<T>();
    atomicAdd(&p->pins, 1u);
    if (threadIdx.x == 0) {
      if (write) p->dirty = 1u;
      p->last_access = clock64();
    }
    const clio::run::u64 in = off % h_->elems_per_page_;
    clio::run::u64 run = h_->elems_per_page_ - in;
    if (run > count) run = count;
    return Held<T>(p, static_cast<T *>(p->data) + in, off, run);
  }

  // ------------------------------ fetch ------------------------------

  CTP_GPU_FUN void GatherRanges(clio::run::u64 *, clio::run::u64 *,
                                clio::run::u32 &) const {}
  template <typename... Rest>
  CTP_GPU_FUN void GatherRanges(clio::run::u64 *lo, clio::run::u64 *hi,
                                clio::run::u32 &n, clio::run::u64 a,
                                clio::run::u64 b, Rest... rest) const {
    if (n < kMaxFetchRanges) {
      lo[n] = a;
      hi[n] = b;
      ++n;
    }
    GatherRanges(lo, hi, n, rest...);
  }

  /** Pin every resident page of the ranges; @return how many are missing. */
  CTP_GPU_FUN clio::run::u32 PinResidentAndCountMissing(
      const clio::run::u64 *lo, const clio::run::u64 *hi,
      clio::run::u32 nr) {
    clio::run::u32 missing = 0;
    for (clio::run::u32 r = 0; r < nr; ++r) {
      if (hi[r] <= lo[r]) continue;
      const clio::run::u64 p0 = PageOf(lo[r]);
      const clio::run::u64 p1 = PageOf(hi[r] - 1);
      for (clio::run::u64 pn = p0; pn <= p1; ++pn) {
        Page *p = Find(pn);
        if (p != nullptr) {
          ++p->pins;
        } else {
          ++missing;
        }
      }
    }
    return missing;
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
        if (p != nullptr && p->pins != 0u) --p->pins;
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
        if (Find(pn) != nullptr) continue;     // already here
        Page *p = FindFree();
        if (p == nullptr) {
          printf("[gpu_vector] FATAL table=%u: no free frame for page %llu "
                 "after EvictPages\n", Table(), (unsigned long long) pn);
          __trap();
        }
        p->fetching = 1u;
        p->page_num = pn;
        const clio::run::u32 pn32 = static_cast<clio::run::u32>(pn);
        t->Add("", 0, h_->page_bytes_, RawPtr(p->data), 0.5f);
        t->reqs_[n].blob_name_.assign(reinterpret_cast<const char *>(&pn32),
                                      sizeof(pn32));
        bt->fetch_slot[n++] = static_cast<clio::run::u32>(p - Pages());
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
    t->context_.blob_name_flags_ =
        clio::cte::core::Context::kBlobNameRawInt32;
    // A page nobody has written yet has no blob; the CTE creates it and
    // returns success, so the vector needs no first-touch case of its own.
    t->context_.create_on_get_ = true;
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
      Page *p = &tbl[bt->fetch_slot[i]];
      p->dirty = 0u;
      p->score = kDefaultScore;
      p->last_access = clock64();
      __threadfence();
      p->fetching = 0u;      // published last: Find skips a fetching frame
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

  /** Stage every dirty page into the flush task and send it. Thread 0 only. */
  CTP_GPU_FUN void SubmitFlush() {
    BlockTasks *bt = Tasks();
    Page *tbl = Pages();
    auto *t = bt->flush;
    t->count_ = 0;
    clio::run::u32 n = 0;
    for (clio::run::u32 i = 0;
         i < h_->pages_per_block_ && n < clio::cte::core::kPodMultiMax; ++i) {
      Page *p = &tbl[i];
      if (!p->dirty || p->flushing || p->fetching) continue;
      const clio::run::u32 pn32 = static_cast<clio::run::u32>(p->page_num);
      t->Add("", 0, h_->page_bytes_, RawPtr(p->data), 0.5f);
      // Overwrite the record's name with the raw page number; Add() takes a
      // C string and would stop at the first zero byte.
      t->reqs_[n].blob_name_.assign(reinterpret_cast<const char *>(&pn32),
                                    sizeof(pn32));
      bt->flush_slot[n++] = i;
      p->flushing = 1u;
    }
    bt->flush_n = n;
    if (n == 0) return;
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = DeviceTaskId(Table(), kKindFlush, bt->seq++);
    t->pool_id_ = h_->pool_id_;
    t->method_ = clio::cte::core::Method::kPodMultiPutBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = h_->tag_id_;
    t->context_ = clio::cte::core::Context();
    t->context_.compress_lib_ = h_->compress_lib_;
    t->context_.compress_preset_ = h_->compress_preset_;
    t->context_.blob_name_flags_ =
        clio::cte::core::Context::kBlobNameRawInt32;
    ClearRunCtx(t);
    if (h_->stat_puts_ != nullptr) {
      atomicAdd(h_->stat_puts_, static_cast<unsigned long long>(n));
    }
    bt->flush_busy = 1u;
    t->fut_.is_complete_.store(0);   // reused task: clear the last completion
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
      Page *p = &tbl[bt->flush_slot[i]];
      p->flushing = 0u;
      p->dirty = 0u;
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
