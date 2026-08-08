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

#if CTP_IS_GPU_COMPILER
  // ---- device API -----------------------------------------------------

  /**
   * Element access. Resolves the page (faulting it in if needed), records
   * the access, and returns a reference into the page's bytes.
   *
   * The non-const form marks the page dirty: a reference handed out for
   * writing cannot be observed later, so the write must be assumed.
   */
  CTP_GPU_FUN T &operator[](clio::run::u64 off) {
    Page *p = Resolve(off / elems_per_page_);
    p->dirty = 1u;
    p->last_access = Now();
    return static_cast<T *>(p->data)[off - p->page_num * elems_per_page_];
  }

  /** Read-only access: does NOT dirty the page. */
  CTP_GPU_FUN const T &at(clio::run::u64 off) {
    Page *p = Resolve(off / elems_per_page_);
    p->last_access = Now();
    return static_cast<const T *>(p->data)[off - p->page_num * elems_per_page_];
  }

  /**
   * Evict `num_pages` resident pages, lowest score first and oldest access
   * breaking ties. A clean page is simply dropped; a dirty one is flushed
   * whole and waited on, because its bytes are the only copy.
   */
  CTP_GPU_FUN void EvictPages(clio::run::u32 num_pages) {
    Page *tbl = BlockPages();
    for (clio::run::u32 k = 0; k < num_pages; ++k) {
      clio::run::u32 victim = pages_per_block_;
      for (clio::run::u32 i = 0; i < pages_per_block_; ++i) {
        if (tbl[i].page_num == kNoPage) continue;
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
    }
  }

  /**
   * Start writing back every page overlapping [off, off+count) elements.
   * Pages are marked clean immediately -- the put carries the bytes as they
   * are now, and a later write re-dirties the page for the next flush.
   */
  CTP_GPU_FUN void BeginFlush(clio::run::u64 off, clio::run::u64 count) {
    ForEachResident(off, count, [this](Page *p) {
      if (!p->dirty) return;
      SubmitPut(p);
    });
  }

  /** Wait for the puts started by BeginFlush over the same range. */
  CTP_GPU_FUN void WaitFlush(clio::run::u64 off, clio::run::u64 count) {
    ForEachResident(off, count, [this](Page *p) {
      if (p->flushing) AwaitPut(p);
    });
  }

  /**
   * Set a page's score. If the page is resident the score is applied
   * locally (it steers eviction immediately, with no round trip); a
   * PodReorganizeBlobTask is also sent so the CTE's placement follows,
   * which is what makes this usable as a prefetch hint.
   */
  CTP_GPU_FUN void RescorePage(clio::run::u64 page_id, float score) {
    Page *p = Find(page_id);
    if (p != nullptr) {
      p->score = score;
    }
    Page *slot = (p != nullptr) ? p : &BlockPages()[page_id % pages_per_block_];
    if (slot->rescore == nullptr) return;
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
    (void) ipc_->Send(SlotPtr(slot->rescore));
  }

  /** Number of elements in the vector. */
  CTP_GPU_FUN clio::run::u64 size() const { return size_; }

  /** Set by the kernel prologue (see CLIO_GPU_INIT). */
  clio::run::gpu::IpcManager *ipc_ = nullptr;

 private:
  /** Per-thread cache of the last page touched. NOT __shared__. */
  Page *last_page_ = nullptr;

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

  /** Fault `page_num` into `p` with a synchronous PodGetBlobTask. */
  CTP_GPU_FUN void SubmitGet(Page *p, clio::run::u64 page_num) {
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
    auto fut = ipc_->Send(SlotPtr(p->get));
    fut.Wait();                           // page faults are synchronous
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
    if (p != nullptr && p->page_num == page_num) return p;
    p = Find(page_num);
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
        EvictPages(1);
      }
    }
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
