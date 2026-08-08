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
template <typename T>
class DeviceVector {
 public:
  DeviceVector() = default;

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
    Page *p = Resolve(PageOf(off));
    const clio::run::u64 within = IndexIn(off, p);
    const clio::run::u64 left = elems_per_page_ - within;
    return (count < left) ? count : left;
  }

  /**
   * Element access through the HELD page -- no resolution, no checks.
   *
   * Assumes HoldPage() covers `off`: it indexes last_page_ directly, which is
   * what makes it cost about the same as a pointer dereference. Accessing
   * outside the held run reads the wrong page's bytes, so the run length
   * HoldPage returned is a contract, not a hint. Use RefFault/AtFault when the
   * access pattern is not known ahead of time and demand faulting is wanted.
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

  /** Demand-faulting write access: resolves the page on every call. */
  CTP_GPU_FUN T &RefFault(clio::run::u64 off) {
    Page *p = Resolve(PageOf(off));
    p->dirty = 1u;
    return static_cast<T *>(p->data)[IndexIn(off, p)];
  }

  /** Demand-faulting read access: resolves the page on every call. */
  CTP_GPU_FUN const T &AtFault(clio::run::u64 off) {
    Page *p = Resolve(PageOf(off));
    return static_cast<const T *>(p->data)[IndexIn(off, p)];
  }

  /**
   * Which page an offset falls in.
   *
   * A GPU has NO hardware 64-bit integer divide, so `off / elems_per_page_`
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
    return page_shift_ ? (off >> page_shift_) : (off / elems_per_page_);
  }

  /** Offset of an element within its page. */
  CTP_GPU_FUN clio::run::u64 IndexIn(clio::run::u64 off, const Page *p) const {
    return page_shift_ ? (off & page_mask_)
                       : (off - p->page_num * elems_per_page_);
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
    for (clio::run::u32 i = 0; i < pages_per_block_; ++i) {
      tbl[i].score = 0.0f;
    }
    EvictLocked(pages_per_block_);
    for (clio::run::u32 i = 0; i < pages_per_block_; ++i) {
      tbl[i].last_access = 0;
    }
    UnlockBlock();
    last_page_ = nullptr;   // the per-thread cache now points at a free slot
  }

  /** EvictPages' body, for callers that already hold the block lock. */
  CTP_GPU_FUN void EvictLocked(clio::run::u32 num_pages) {
    Page *tbl = BlockPages();
    for (clio::run::u32 k = 0; k < num_pages; ++k) {
      clio::run::u32 victim = pages_per_block_;
      for (clio::run::u32 i = 0; i < pages_per_block_; ++i) {
        if (tbl[i].page_num == kNoPage) continue;
        // A page with a transfer in flight is not a candidate: its slot is
        // already promised to that transfer, and evicting it would let the
        // copy land in a slot that now belongs to a different page.
        if (tbl[i].fetching) continue;
        if (victim == pages_per_block_) {
          victim = i;
          continue;
        }
        if (tbl[i].score < tbl[victim].score ||
            (tbl[i].score == tbl[victim].score &&
             tbl[i].last_access < tbl[victim].last_access)) {
          victim = i;
        }
      }
      if (victim == pages_per_block_) return;   // nothing resident
      Page *p = &tbl[victim];
      if (p->dirty || p->flushing) {
        SubmitPut(p);
        AwaitPut(p);
      }
      p->page_num = kNoPage;
      p->dirty = 0u;
      p->flushing = 0u;
      Bump(stat_evicts_);
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
    Page *slot = (p != nullptr) ? p : &BlockPages()[page_id % pages_per_block_];
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
    t->pool_id_ = pool_id_;
    t->method_ = clio::cte::core::Method::kPodReorganizeBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = tag_id_;
    char name[32];
    PageBlobName(page_id, name);
    t->blob_name_ = name;
    t->new_score_ = score;
    t->replica_ = 0;
    // Fire and forget: a hint that arrives late is still useful, and
    // waiting here would serialise the kernel behind placement.
    ClearRunCtx(t);
    Bump(stat_rescores_);
    slot->rescore_fut = ipc_->Send(SlotPtr(slot->rescore));
    slot->rescoring = 1u;
  }

 public:
  /** Number of elements in the vector. */
  CTP_GPU_FUN clio::run::u64 size() const { return size_; }

  /** Set by the kernel prologue (see CLIO_GPU_INIT). */
  clio::run::gpu::IpcManager *ipc_ = nullptr;

 private:
  /** Per-thread cache of the last page touched. NOT __shared__. */
  Page *last_page_ = nullptr;

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
    if (block_locks_ == nullptr) return;
    int *lk = block_locks_ + (blockIdx.x % nblocks_);
    while (atomicCAS(lk, 0, 1) != 0) {
      __nanosleep(32);
    }
    __threadfence();
  }

  CTP_GPU_FUN void UnlockBlock() {
    if (block_locks_ == nullptr) return;
    // Publish every page-table write before the lock is observed free.
    __threadfence();
    atomicExch(block_locks_ + (blockIdx.x % nblocks_), 0);
  }

  /** Increment an instrumentation counter if the test enabled it. */
  CTP_GPU_FUN void Bump(unsigned long long *c) const {
    if (c != nullptr) atomicAdd(c, 1ull);
  }

  CTP_GPU_FUN clio::run::u64 Now() const {
    return static_cast<clio::run::u64>(clock64());
  }

  /** This block's slice of the page table. */
  CTP_GPU_FUN Page *BlockPages() const {
    const clio::run::u32 b = blockIdx.x % nblocks_;
    return pages_ + static_cast<clio::run::u64>(b) * pages_per_block_;
  }

  /** A globally unique id for the next task this vector submits. */
  CTP_GPU_FUN clio::run::TaskId NextTaskId(clio::run::u32 kind) const {
    unsigned long long n =
        (task_seq_ != nullptr) ? atomicAdd(task_seq_, 1ull) : 0ull;
    return DeviceTaskId(n, kind, static_cast<clio::run::u32>(n));
  }

  /** Index of `p` in the whole page table -- the basis of its task ids. */
  CTP_GPU_FUN clio::run::u64 SlotOf(const Page *p) const {
    return static_cast<clio::run::u64>(p - pages_);
  }

  CTP_GPU_FUN Page *Find(clio::run::u64 page_num) const {
    Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < pages_per_block_; ++i) {
      if (tbl[i].page_num == page_num) return &tbl[i];
    }
    return nullptr;
  }

  template <typename TaskT>
  CTP_GPU_FUN ctp::ipc::FullPtr<TaskT> SlotPtr(TaskT *task) const {
    ctp::ipc::FullPtr<TaskT> fp;
    fp.shm_.alloc_id_ = task_alloc_id_;
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
    t->pool_id_ = pool_id_;
    t->method_ = clio::cte::core::Method::kPodPutBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = tag_id_;
    char name[32];
    PageBlobName(p->page_num, name);
    t->blob_name_ = name;
    t->offset_ = 0;
    t->size_ = page_bytes_;
    t->blob_data_ = RawPtr(p->data);
    t->score_ = p->score;
    t->flags_ = 0;
    t->context_ = clio::cte::core::Context();
    t->context_.compress_lib_ = compress_lib_;
    ClearRunCtx(t);
    Bump(stat_puts_);
    p->put_fut = ipc_->Send(SlotPtr(p->put));
    // Clean as of THIS put: the bytes it carries are what the page held
    // when it was submitted. A later write dirties it again for the next.
    p->dirty = 0u;
    p->flushing = 1u;
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
    clio::run::u32 free_slot = pages_per_block_;
    for (clio::run::u32 i = 0; i < pages_per_block_; ++i) {
      if (tbl[i].page_num == kNoPage) {
        free_slot = i;
        break;
      }
    }
    if (free_slot == pages_per_block_) {
      EvictLocked(1);
      for (clio::run::u32 i = 0; i < pages_per_block_; ++i) {
        if (tbl[i].page_num == kNoPage) {
          free_slot = i;
          break;
        }
      }
      // Everything resident is mid-transfer: report failure rather than
      // stalling, so the caller simply falls back to a demand fault later.
      if (free_slot == pages_per_block_) return false;
    }
    Page *p = &tbl[free_slot];
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
    Bump(stat_faults_);
    Bump(stat_prefetches_);
    p->get_fut = ipc_->Send(SlotPtr(p->get));
    p->fetching = 1u;
  }

  /** Wait for an in-flight asynchronous get on `p`. */
  CTP_GPU_FUN void AwaitFetch(Page *p) {
    if (!p->fetching) return;
    p->get_fut.Wait();
    p->fetching = 0u;
  }

  /** Fill in the get task's fields for `page_num`. */
  CTP_GPU_FUN void PrepareGet(Page *p, clio::run::u64 page_num) {
    auto *t = p->get;
    t->task_flags_.Clear();
    t->return_code_.store(0);
    t->task_id_ = NextTaskId(kKindGet);
    t->pool_id_ = pool_id_;
    t->method_ = clio::cte::core::Method::kPodGetBlob;
    t->pool_query_ = clio::run::PoolQuery::ToLocalCpu();
    t->tag_id_ = tag_id_;
    char name[32];
    PageBlobName(page_num, name);
    t->blob_name_ = name;
    t->offset_ = 0;
    t->size_ = page_bytes_;
    t->blob_data_ = RawPtr(p->data);
    t->flags_ = 0;
    ClearRunCtx(t);
  }

  /** Fault `page_num` into `p` with a SYNCHRONOUS get. */
  CTP_GPU_FUN void SubmitGet(Page *p, clio::run::u64 page_num) {
    PrepareGet(p, page_num);
    Bump(stat_faults_);
    auto fut = ipc_->Send(SlotPtr(p->get));
    fut.Wait();                           // demand faults block by definition
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
    // Everything past the per-thread fast path touches the shared page table
    // (residency search, slot claim, eviction, the fault itself), so it is all
    // under the block's lock. Without it, concurrent lanes claim the same free
    // slot and submit competing gets into one task slot.
    LockBlock();
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
          Bump(stat_prefetch_hits_);
        } else {
          Bump(stat_prefetch_late_);
        }
        AwaitFetch(p);
      }
    }
    if (p == nullptr) {
      Page *tbl = BlockPages();
      for (;;) {
        clio::run::u32 free_slot = pages_per_block_;
        for (clio::run::u32 i = 0; i < pages_per_block_; ++i) {
          if (tbl[i].page_num == kNoPage) {
            free_slot = i;
            break;
          }
        }
        if (free_slot != pages_per_block_) {
          p = &tbl[free_slot];
          p->page_num = page_num;
          p->dirty = 0u;
          p->flushing = 0u;
          p->score = 0.0f;
          SubmitGet(p, page_num);
          break;
        }
        EvictLocked(1);   // already holding the lock
      }
    }
    if (p != nullptr) p->last_access = Now();
    UnlockBlock();
    last_page_ = p;
    return p;
  }

  /** Apply `fn` to each RESIDENT page overlapping the element range. */
  template <typename Fn>
  CTP_GPU_FUN void ForEachResident(clio::run::u64 off, clio::run::u64 count,
                                   Fn fn) {
    if (count == 0) return;
    const clio::run::u64 first = off / elems_per_page_;
    const clio::run::u64 last = (off + count - 1) / elems_per_page_;
    Page *tbl = BlockPages();
    for (clio::run::u32 i = 0; i < pages_per_block_; ++i) {
      if (tbl[i].page_num == kNoPage) continue;
      if (tbl[i].page_num < first || tbl[i].page_num > last) continue;
      fn(&tbl[i]);
    }
  }
#endif  // CTP_IS_GPU_COMPILER
};

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_DEVICE_VECTOR_H_
