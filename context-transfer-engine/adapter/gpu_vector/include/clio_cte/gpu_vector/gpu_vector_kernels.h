/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_GPU_VECTOR_KERNELS_H_
#define CLIO_CTE_GPU_VECTOR_KERNELS_H_

#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/types.h>
#include <clio_cte/gpu_vector/gpu_vector_page.h>
#include <clio_cte/gpu_vector/gpu_vector_view.h>

#if CTP_IS_GPU_COMPILER

namespace clio::cte::gpu_vector {

namespace detail {

/** Per-warp last-page cache. Lane 0 reads / writes; other lanes read after
 *  __syncwarp. The user kernel must provide a __shared__ array via
 *  CLIO_GPU_VECTOR_KERNEL_INIT — this function just resolves the lane slot. */
CTP_GPU_FUN Page *&LaneLastPage(Page **last_page_array) {
  return last_page_array[threadIdx.x & 31];
}

/** Broadcast a Page* from lane 0 to the whole warp.
 *
 *  Replaces the old __shared__ broadcast slot. That slot was ONE per
 *  block, so every warp wrote index 0 and the per-lane cache aliased on
 *  (threadIdx.x & 31) -- lane 5 of warp 0 and lane 5 of warp 1 were the
 *  same entry. Only one warp per block could therefore drive a read, and
 *  a second warp corrupted the first's resolved page (illegal access).
 *  A shuffle needs no shared memory at all, so the vector handle can be
 *  a plain per-thread value and any number of warps can run concurrently.
 */
CTP_GPU_FUN Page *WarpBroadcastPage(Page *p) {
  unsigned long long v = static_cast<unsigned long long>(
      reinterpret_cast<uintptr_t>(p));
  v = __shfl_sync(0xffffffffu, v, 0);
  return reinterpret_cast<Page *>(static_cast<uintptr_t>(v));
}

/** Atomically swap a 32-bit field to `new_val` and return the old value. */
CTP_GPU_FUN int32_t AtomicExchI32(int32_t *p, int32_t new_val) {
  return atomicExch(reinterpret_cast<int *>(p), static_cast<int>(new_val));
}

/** Atomic CAS for u32. */
CTP_GPU_FUN clio::run::u32 AtomicCasU32(clio::run::u32 *p, clio::run::u32 expected,
                                          clio::run::u32 desired) {
  return atomicCAS(reinterpret_cast<unsigned int *>(p),
                   static_cast<unsigned int>(expected),
                   static_cast<unsigned int>(desired));
}

/** Atomic OR on u32 — returns the old value. */
CTP_GPU_FUN clio::run::u32 AtomicOrU32(clio::run::u32 *p, clio::run::u32 mask) {
  return atomicOr(reinterpret_cast<unsigned int *>(p),
                  static_cast<unsigned int>(mask));
}

/** Atomic AND-NOT on u32 (clears bits). */
CTP_GPU_FUN clio::run::u32 AtomicClearBitsU32(clio::run::u32 *p, clio::run::u32 mask) {
  return atomicAnd(reinterpret_cast<unsigned int *>(p),
                   static_cast<unsigned int>(~mask));
}

/** Atomic min for i32. */
CTP_GPU_FUN void AtomicMinI32(int32_t *p, int32_t v) {
  atomicMin(reinterpret_cast<int *>(p), static_cast<int>(v));
}

/** Atomic max for i32. */
CTP_GPU_FUN void AtomicMaxI32(int32_t *p, int32_t v) {
  atomicMax(reinterpret_cast<int *>(p), static_cast<int>(v));
}

/** Atomic increment for u32. */
CTP_GPU_FUN clio::run::u32 AtomicIncU32(clio::run::u32 *p) {
  return atomicAdd(reinterpret_cast<unsigned int *>(p), 1u);
}

/** Atomic decrement for u32. */
CTP_GPU_FUN clio::run::u32 AtomicDecU32(clio::run::u32 *p) {
  return atomicSub(reinterpret_cast<unsigned int *>(p), 1u);
}

/** Attempt to acquire kPageBusy. Returns true on success. */
CTP_GPU_FUN bool TryAcquireBusy(Page *p) {
  clio::run::u32 prev = AtomicOrU32(&p->flags, kPageBusy);
  return (prev & kPageBusy) == 0;
}

/** Release kPageBusy. */
CTP_GPU_FUN void ReleaseBusy(Page *p) {
  AtomicClearBitsU32(&p->flags, kPageBusy);
}

/**
 * Build a blob_data_ ShmPtr that ToFullPtr resolves directly to a raw
 * pointer via its null-alloc_id branch (the canonical pattern used by
 * the kernel-side CTE benchmarks: see workload_cte_client_overhead.cc).
 *
 * For kDeviceMem-backed pages, off_ holds a CUDA/HIP device address.
 * For kPinnedHost-backed pages, off_ holds the mapped host address (it
 * is GPU-addressable via the pinned mapping). DeviceAwareMemcpy's
 * registered hook (cudaMemcpyDefault / hipMemcpyDefault) auto-detects
 * the memory kind via pointer attributes and copies in the right
 * direction without staging.
 */
CTP_GPU_FUN ctp::ipc::ShmPtr<> MakeBlobShmPtr(void *device_addr,
                                                  ctp::ipc::AllocatorId alloc_id) {
  (void)alloc_id;
  ctp::ipc::ShmPtr<> p;
  p.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  p.off_ = reinterpret_cast<clio::run::u64>(device_addr);
  return p;
}

/**
 * Push a hint into a block's per-block rescore queue. Producer-side
 * MPSC: bump tail atomically; drop if full. Returns true if the entry
 * was enqueued.
 */
CTP_GPU_FUN bool RescorePush(RescoreQueue *q, clio::run::u32 page_idx,
                              float score) {
  clio::run::u32 cur_tail = AtomicIncU32(&q->tail);
  clio::run::u32 head = q->head;
  if (cur_tail - head >= kRescoreQueueCap) {
    // Roll back. Not perfectly atomic, but a dropped hint is benign.
    AtomicDecU32(&q->tail);
    return false;
  }
  q->slots[cur_tail & (kRescoreQueueCap - 1)] =
      RescoreEntry{page_idx, score};
  return true;
}

/** Remaining capacity in the rescore queue (lower bound — may be stale). */
CTP_GPU_FUN clio::run::u32 RescoreRemaining(const RescoreQueue *q) {
  clio::run::u32 used = q->tail - q->head;
  if (used > kRescoreQueueCap) used = kRescoreQueueCap;
  return kRescoreQueueCap - used;
}

}  // namespace detail

/** T-agnostic FlushPage: takes element size in bytes explicitly so the
 *  cache-management / drain kernels can compile as non-template
 *  __global__ functions (template __global__'s aren't reliably
 *  registered by nvcc's launch glue).
 *
 *  Caller must pass the kernel-scope `g_ipc_manager_ptr` from
 *  CLIO_GPU_INIT — going through CLIO_IPC in this device function
 *  trips the host-pass typing check (CLIO_IPC expands to clio::run::IpcManager*
 *  on host pass, which returns clio::run::Future, not gpu::Future). */
CTP_GPU_FUN void FlushPageBase(::clio::run::gpu::IpcManager *ipc,
                                 const DeviceViewBase &v, clio::run::u32 block_idx,
                                 Page *page, clio::run::u32 slot) {
  if (page->page_idx < 0) return;

  int32_t mn = detail::AtomicExchI32(&page->modify_min, -1);
  int32_t mx = detail::AtomicExchI32(&page->modify_max, -1);
  if (mn < 0 || mx < 0 || mx < mn) return;

  detail::AtomicDecU32(&GetBlock(v, block_idx)->num_modified);

  auto *task = GetPutTask(v, block_idx, slot);
  // Reset lifecycle flags + fresh task_id for slot reuse.
  task->task_flags_.Clear();
  task->return_code_.store(0);
  task->task_id_ = clio::run::CreateTaskId();
  // T-agnostic path: flush the whole page. The page-keyed blob name is
  // composed runtime-side from blob_name_ + "_pi" + gpu_page_idx_.
  task->offset_ = 0;
  task->size_ = v.page_size_bytes;
  task->gpu_page_idx_ = static_cast<clio::run::u32>(page->page_idx);
  task->gpu_family_idx_ = FamilyOf(v, page->page_idx);
  ctp::ipc::AllocatorId alloc =
      (page->tier == 0) ? v.pages_alloc_id : v.host_pages_alloc_id;
  task->blob_data_ = detail::MakeBlobShmPtr(page->device_ptr, alloc);

  ctp::ipc::FullPtr<clio::cte::core::PodPutBlobTask> fp;
  fp.shm_.alloc_id_ = v.put_pool_alloc_id;
  fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task);
  fp.ptr_ = task;
  page->active_put = ipc->Send(fp);
}

/**
 * Submit a PutBlob for `page` covering its current modify range. Caller
 * already CAS-holds the kPagePutInFlight bit. Returns once the queue
 * push completes; does not wait for the runtime to ack. `ipc` is the
 * kernel-scope `g_ipc_manager_ptr` from CLIO_GPU_INIT.
 *
 * `slot` is the index in `Block::pages[]` (covering BOTH tiers), used
 * to pick which task slot in the put pool we use.
 */
template <typename T>
CTP_GPU_FUN void FlushPage(::clio::run::gpu::IpcManager *ipc,
                            const DeviceView<T> &v, clio::run::u32 block_idx,
                            Page *page, clio::run::u32 slot) {
  if (page->page_idx < 0) return;

  // Atomically reset the dirty range so the next concurrent writer
  // observes a fresh window. Whatever range we capture here is what we
  // promise the runtime; later writes form a new range that the next
  // tick picks up.
  int32_t mn = detail::AtomicExchI32(&page->modify_min, -1);
  int32_t mx = detail::AtomicExchI32(&page->modify_max, -1);
  if (mn < 0 || mx < 0 || mx < mn) return;

  // Bookkeeping: this page is no longer in the dirty count.
  detail::AtomicDecU32(&GetBlock(v.base, block_idx)->num_modified);

  auto *task = GetPutTask(v.base, block_idx, slot);
  // Clear lifecycle flags carried over from the previous put through
  // this slot (TASK_ROUTED in particular). Mint a fresh task_id_.
  task->task_flags_.Clear();
  task->return_code_.store(0);
  task->task_id_ = clio::run::CreateTaskId();
  clio::run::u64 t_size = sizeof(T);
  clio::run::u64 mn_b = static_cast<clio::run::u64>(mn) * t_size;
  clio::run::u64 mx_b = (static_cast<clio::run::u64>(mx) + 1) * t_size;
  task->offset_ = mn_b;            // offset within the per-page blob
  task->size_ = mx_b - mn_b;
  task->gpu_page_idx_ = static_cast<clio::run::u32>(page->page_idx);
  task->gpu_family_idx_ = FamilyOf(v.base, page->page_idx);
  ctp::ipc::AllocatorId alloc =
      (page->tier == 0) ? v.base.pages_alloc_id : v.base.host_pages_alloc_id;
  task->blob_data_ = detail::MakeBlobShmPtr(
      reinterpret_cast<char *>(page->device_ptr) + mn_b, alloc);

  ctp::ipc::FullPtr<clio::cte::core::PodPutBlobTask> fp;
  fp.shm_.alloc_id_ = v.base.put_pool_alloc_id;
  fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task);
  fp.ptr_ = task;

  page->active_put = ipc->Send(fp);
}

/** "Compute the family from the view's fam_ppb policy" sentinel. */
static constexpr clio::run::u32 kFamAuto = ~static_cast<clio::run::u32>(0);

/** Submit a GetBlob to fault `target_page_idx` into `page->device_ptr`. */
template <typename T>
CTP_GPU_FUN void FaultPage(::clio::run::gpu::IpcManager *ipc,
                            const DeviceView<T> &v, clio::run::u32 block_idx,
                            Page *page, clio::run::u32 slot,
                            int32_t target_page_idx,
                            clio::run::u32 fam = kFamAuto) {
  auto *task = GetGetTask(v.base, block_idx, slot);
  // Reset lifecycle flags + mint a fresh task_id for slot reuse.
  task->task_flags_.Clear();
  task->return_code_.store(0);
  task->task_id_ = clio::run::CreateTaskId();
  task->size_ = v.base.page_size_bytes;
  if (v.base.flat_layout) {
    // Flat store: one blob "w", page addressed by OFFSET. Sentinels keep the
    // runtime from appending any _b/_pi suffix to the slot's stem.
    task->offset_ = static_cast<clio::run::u64>(target_page_idx) *
                    v.base.page_size_bytes;
    task->gpu_page_idx_   = clio::cte::core::PodGetBlobTask::kNoPageIdx;
    task->gpu_family_idx_ = clio::cte::core::PodGetBlobTask::kNoFamilyIdx;
    (void) fam;
  } else {
    task->offset_ = 0;
    task->gpu_page_idx_ = static_cast<clio::run::u32>(target_page_idx);
    task->gpu_family_idx_ =
        (fam == kFamAuto) ? FamilyOf(v.base, target_page_idx) : fam;
  }
  ctp::ipc::AllocatorId alloc =
      (page->tier == 0) ? v.base.pages_alloc_id : v.base.host_pages_alloc_id;
  task->blob_data_ = detail::MakeBlobShmPtr(page->device_ptr, alloc);
  ctp::ipc::FullPtr<clio::cte::core::PodGetBlobTask> fp;
  fp.shm_.alloc_id_ = v.base.get_pool_alloc_id;
  fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task);
  fp.ptr_ = task;
  page->active_get = ipc->Send(fp);
}


/** Wait on any in-flight put for this page, then clear the slot. */
CTP_GPU_FUN void DrainPut(Page *page) {
  if (!page->active_put.IsNull()) {
    page->active_put.Wait();
    page->active_put = clio::run::gpu::Future<clio::cte::core::PodPutBlobTask>();
    detail::AtomicClearBitsU32(&page->flags, kPagePutInFlight);
  }
}


/** Wait on any in-flight get for this page, then clear the slot. */
CTP_GPU_FUN void DrainGet(Page *page, DeviceViewBase *vb = nullptr) {
  if (!page->active_get.IsNull()) {
    const long long t0 = clock64();
    page->active_get.Wait();
    if (vb != nullptr && vb->stats != nullptr) {
      atomicAdd_system(&vb->stats->fault_wait_ticks,
                       (unsigned long long)(clock64() - t0));
      atomicAdd_system(&vb->stats->fault_wait_count, 1ULL);
    }
    if (vb != nullptr && vb->stats != nullptr) {
      const int rc = (int) page->active_get->return_code_.load();
      atomicAdd_system(rc == 0 ? &vb->stats->fault_get_ok
                               : &vb->stats->fault_get_fail, 1ULL);
      if (rc != 0) {
        vb->stats->fault_get_last_rc = (unsigned long long)(long long) rc;
        vb->stats->fault_get_last_page =
            (unsigned long long)(long long) page->page_idx;
      }
    }
    page->active_get = clio::run::gpu::Future<clio::cte::core::PodGetBlobTask>();
    detail::AtomicClearBitsU32(&page->flags, kPageGetInFlight);
  }
}

/** After a batch member's completion has been OBSERVED (its shared future
 *  waited), clear the in-flight flag and stale future copies on EVERY member
 *  of that batch. Without this, the head slot's task could be reinitialised
 *  for a later fault while unaccessed members still poll its old memory --
 *  a spin that never ends. modify_max carries the head slot. */
CTP_GPU_FUN void BatchSweep(const DeviceViewBase &vb, clio::run::u32 block_idx,
                            Page *observed) {
  const int32_t head = observed->modify_max;
  if (head < 0) return;
  Block *b = GetBlock(vb, block_idx);
  const clio::run::u32 total = vb.gpu_pages_per_block;
  for (clio::run::u32 s = 0; s < total; ++s) {
    Page *p = &b->pages[s];
    if (p->modify_max != head) continue;
    p->modify_max = -1;
    detail::AtomicClearBitsU32(&p->flags, kPageGetInFlight);
    p->active_get = clio::run::gpu::Future<clio::cte::core::PodGetBlobTask>();
  }
}

/** Claim k CONSECUTIVE slots (kPageBusy held on each), or return false.
 *  Consecutive slots are contiguous device memory within a block, which is
 *  what lets k pages arrive with ONE flat-offset read. */
template <typename T>
CTP_GPU_FUN bool EvictRun(::clio::run::gpu::IpcManager *ipc,
                          const DeviceView<T> &v, clio::run::u32 block_idx,
                          clio::run::u32 k, clio::run::u32 *out_s0) {
  Block *b = GetBlock(v.base, block_idx);
  const clio::run::u32 total = v.base.gpu_pages_per_block;
  if (k == 0 || k > total) return false;
  for (clio::run::u32 s0 = 0; s0 + k <= total; ++s0) {
    clio::run::u32 got = 0;
    for (; got < k; ++got) {
      Page *p = &b->pages[s0 + got];
      if (p->flags & (kPageBusy | kPageGetInFlight)) break;
      if (!detail::TryAcquireBusy(p)) break;
      // claimed; drain any residue so the slot is quiescent
      DrainPut(p);
      DrainGet(p, (DeviceViewBase *) &v.base);
      // Completion observed here counts too: without the sweep, reusing a
      // batch's HEAD slot leaves unaccessed members polling its
      // reinitialised task forever (hang under hoisted mass prefault).
      if (p->modify_max >= 0) BatchSweep(v.base, block_idx, p);
    }
    if (got == k) { *out_s0 = s0; return true; }
    for (clio::run::u32 r = 0; r < got; ++r)
      detail::ReleaseBusy(&b->pages[s0 + r]);
  }
  return false;
}

/** Issue ONE flat-offset read covering pages [p0, p0+k) into the k claimed
 *  consecutive slots starting at s0. Uses the HEAD slot's Pod task; all k
 *  pages share its future and are marked with modify_max = head slot so the
 *  completion SWEEP (BatchSweep) can clear the whole run at once. Waiting
 *  happens at first ACCESS, not here. */
template <typename T>
CTP_GPU_FUN void FaultRun(::clio::run::gpu::IpcManager *ipc,
                          const DeviceView<T> &v, clio::run::u32 block_idx,
                          clio::run::u32 s0, int32_t p0, clio::run::u32 k) {
  Block *b = GetBlock(v.base, block_idx);
  auto *task = GetGetTask(v.base, block_idx, s0);
  task->task_flags_.Clear();
  task->return_code_.store(0);
  task->task_id_ = clio::run::CreateTaskId();
  task->offset_  = static_cast<clio::run::u64>(p0) * v.base.page_size_bytes;
  task->size_    = static_cast<clio::run::u64>(k) * v.base.page_size_bytes;
  task->gpu_page_idx_   = clio::cte::core::PodGetBlobTask::kNoPageIdx;
  task->gpu_family_idx_ = clio::cte::core::PodGetBlobTask::kNoFamilyIdx;
  task->blob_data_ =
      detail::MakeBlobShmPtr(b->pages[s0].device_ptr, v.base.pages_alloc_id);
  ctp::ipc::FullPtr<clio::cte::core::PodGetBlobTask> fp;
  fp.shm_.alloc_id_ = v.base.get_pool_alloc_id;
  fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task);
  fp.ptr_ = task;
  auto fut = ipc->Send(fp);
  for (clio::run::u32 i = 0; i < k; ++i) {
    Page *p = &b->pages[s0 + i];
    p->page_idx   = p0 + static_cast<int32_t>(i);
    p->lru_clock  = clock64();
    p->modify_min = -1;
    p->modify_max = static_cast<int32_t>(s0);   // batch membership marker
    p->active_get = fut;
    detail::AtomicOrU32(&p->flags, kPageGetInFlight);
    detail::ReleaseBusy(p);
  }
}


/** Flush every dirty page in the calling block across BOTH tiers. */
template <typename T>
CTP_GPU_FUN void FlushAllInBlock(::clio::run::gpu::IpcManager *ipc,
                                  const DeviceView<T> &v,
                                  clio::run::u32 block_idx) {
  Block *b = GetBlock(v.base, block_idx);
  clio::run::u32 total = TotalPagesPerBlock(v.base);
  for (clio::run::u32 s = 0; s < total; ++s) {
    Page *p = &b->pages[s];
    if (p->modify_min < 0) continue;
    if (detail::AtomicOrU32(&p->flags, kPagePutInFlight) & kPagePutInFlight) {
      continue;
    }
    FlushPage(ipc, v, block_idx, p, s);
  }
}

/** Returned by EvictSlot* when no slot could be claimed; the caller must
 *  re-check the cache and retry rather than treat it as a slot index. */
constexpr clio::run::u32 kNoSlot = 0xFFFFFFFFu;

/**
 * Pick a victim slot (free first, else LRU) within a single tier.
 * Drains any in-flight ops on it before returning so the caller can
 * reuse the slot freely. `tier_lo`/`tier_hi` bound the search range.
 */
template <typename T>
CTP_GPU_FUN clio::run::u32 EvictSlotInRange(::clio::run::gpu::IpcManager *ipc,
                                       const DeviceView<T> &v,
                                       clio::run::u32 block_idx,
                                       clio::run::u32 tier_lo, clio::run::u32 tier_hi) {
  Block *b = GetBlock(v.base, block_idx);
  // Flush every dirty page first (kicks Sends in flight; we'll Wait on
  // the chosen victim's active_put below).
  for (clio::run::u32 s = tier_lo; s < tier_hi; ++s) {
    Page *p = &b->pages[s];
    if (p->modify_min < 0) continue;
    if (detail::AtomicOrU32(&p->flags, kPagePutInFlight) & kPagePutInFlight) {
      continue;
    }
    FlushPage(ipc, v, block_idx, p, s);
  }
  // Free slot? CLAIM before draining. DrainGet WAITS on the victim's future,
  // so two CUDA blocks sharing this cache block could otherwise both drain the
  // SAME in-flight fault and one would consume the other's completion. The
  // slot is returned with kPageBusy held; Resolve releases it after faulting.
  for (clio::run::u32 s = tier_lo; s < tier_hi; ++s) {
    if (b->pages[s].page_idx >= 0) continue;
    if (!detail::TryAcquireBusy(&b->pages[s])) continue;
    DrainPut(&b->pages[s]);
    DrainGet(&b->pages[s], (DeviceViewBase *) &v.base);
    return s;
  }
  // LRU within range, SKIPPING slots another CUDA block is already binding or
  // fetching. BlockIdx() wraps blockIdx.x % nblocks, so several CUDA blocks
  // share this cache block; handing two of them the same victim let both write
  // the same per-(block, slot) task struct.
  clio::run::u32 lru = tier_hi;
  clio::run::u64 lru_clock = 0;
  for (clio::run::u32 s = tier_lo; s < tier_hi; ++s) {
    if (b->pages[s].flags & (kPageBusy | kPageGetInFlight)) continue;
    if (lru == tier_hi || b->pages[s].lru_clock < lru_clock) {
      lru_clock = b->pages[s].lru_clock;
      lru = s;
    }
  }
  // Claim SOME victim -- NEVER return a slot we do not own. The old fallback
  // could give up after one scan and return an unowned slot, racing an
  // in-flight fetch's completion (bind + task reuse against a Send still on
  // the wire: the qwen MoE illegal-memory-access). An IN-FLIGHT slot is a
  // legitimate victim here: claiming it and DrainGet-ing below waits out its
  // fetch before reuse, which also reclaims prefetched pages that were never
  // accessed and would otherwise hold their slots forever.
  // BOUNDED. This loop used to spin forever when EVERY slot carried
  // kPageBusy, which is reachable whenever a block has few slots: the
  // async cache manager can hold them for its prefetch, and it only
  // clears them from a kernel that may not be scheduled while this one is
  // spinning. That is a hard hang, and it is what made the in-kernel
  // benchmark fail non-deterministically above 16 blocks (2 slots per
  // block: 32 blocks passed twice, then hung).
  //
  // Give the holders a bounded window to finish, then report failure so
  // the CALLER can re-check the cache and retry -- the page it wants may
  // well have arrived in the meantime. A wrong answer is not on the table
  // either way; the choice is between progress and a wedge.
  constexpr int kClaimRounds = 4096;
  bool claimed = false;
  for (int round = 0; round < kClaimRounds; ++round) {
    if (lru != tier_hi && detail::TryAcquireBusy(&b->pages[lru])) {
      claimed = true;
      break;
    }
    lru = tier_hi;
    for (clio::run::u32 s = tier_lo; s < tier_hi; ++s) {
      Page *p = &b->pages[s];
      if (p->flags & kPageBusy) continue;      // owned by someone else
      if (detail::TryAcquireBusy(p)) { lru = s; claimed = true; break; }
    }
    if (claimed) break;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(64);
#endif
  }
  if (!claimed) return kNoSlot;
  DrainPut(&b->pages[lru]);
  DrainGet(&b->pages[lru], (DeviceViewBase *) &v.base);
  if (b->pages[lru].modify_max >= 0)
    BatchSweep(v.base, block_idx, &b->pages[lru]);
  return lru;
}

/** Pick a victim slot anywhere in the block, preferring HBM (tier 0). */
template <typename T>
CTP_GPU_FUN clio::run::u32 EvictSlot(::clio::run::gpu::IpcManager *ipc,
                                const DeviceView<T> &v,
                                clio::run::u32 block_idx) {
  return EvictSlotInRange(ipc, v, block_idx, 0, v.base.gpu_pages_per_block);
}

/**
 * BLOCK-collective page fault: bring `target_page` into this cache block.
 *
 * The whole block participates. Independent warps faulting into the same
 * block's page table is what deadlocks -- two cold-missing warps pick the
 * same victim slot to evict, or one holds a slot busy while another spins
 * for it. Making the fault a block-wide operation removes the race by
 * construction: exactly one thread mutates the page table, everyone waits
 * on the same barrier, and every lane then finds the page for itself (so
 * the resulting pointer is PER-LANE, with no shared broadcast slot).
 *
 * MUST be called uniformly by every thread of the block.
 */
template <typename T, typename FamFn>
CTP_GPU_FUN Page *BlockFaultPage(::clio::run::gpu::IpcManager *ipc,
                                 const DeviceView<T> &v,
                                 clio::run::u32 block_idx,
                                 int32_t target_page, FamFn fam_fn) {
  Block *b = GetBlock(v.base, block_idx);
  const clio::run::u32 total = TotalPagesPerBlock(v.base);
  // One thread mutates the page table for the entire block.
  if (threadIdx.x == 0) {
    // Retry across the whole resolve: a failed claim means every slot was
    // held, and the page we want may arrive (or a slot free up) while we
    // wait. Re-checking the cache each round is what turns a wedge into
    // progress.
    for (int attempt = 0; attempt < 64; ++attempt) {
    Page *hit = nullptr;
    for (clio::run::u32 s = 0; s < total; ++s) {
      Page *p = &b->pages[s];
      if (p->page_idx == target_page && !(p->flags & kPageBusy)) {
        hit = p;
        break;
      }
    }
    if (hit == nullptr) {
      const clio::run::u32 slot = EvictSlot(ipc, v, block_idx);
      if (slot == kNoSlot) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(256);
#endif
        continue;                       // re-check, then try again
      }
      Page *p = &b->pages[slot];
      p->page_idx = target_page;
      if (v.base.stats) {
        // A real synchronous fault get, same accounting the warp path
        // reported (the oversubscribe suite asserts one per page).
        atomicAdd_system(&v.base.stats->resolve_fault_get, 1ULL);
      }
      FaultPage(ipc, v, block_idx, p, slot, target_page,
                fam_fn(v.base, static_cast<clio::run::u64>(target_page)));
      DrainGet(p, const_cast<DeviceViewBase *>(&v.base));
      detail::ReleaseBusy(p);
    } else {
      DrainGet(hit, const_cast<DeviceViewBase *>(&v.base));
    }
    break;                              // page is resident
    }
  }
  __syncthreads();
  // Every lane locates the page itself: the handle keeps a PER-LANE
  // pointer, so nothing is broadcast.
  Page *found = nullptr;
  for (clio::run::u32 s = 0; s < total; ++s) {
    Page *p = &b->pages[s];
    if (p->page_idx == target_page) {
      found = p;
      break;
    }
  }
  return found;
}

/**
 * Warp-cooperative copy of a 4-byte-aligned region between two page
 * pointers. All 32 lanes participate. The page-size bytes are split
 * across lanes as uint4 stores (16 bytes per thread per step). This is
 * the same pattern as Phase 2 swap but exposed as a building block for
 * eviction.
 */
CTP_GPU_FUN void WarpCopyUint4(void *dst, const void *src,
                                 clio::run::u64 bytes, clio::run::u32 lane) {
  clio::run::u64 n = bytes / sizeof(uint4);
  uint4 *d = static_cast<uint4 *>(dst);
  const uint4 *s = static_cast<const uint4 *>(src);
  for (clio::run::u64 i = lane; i < n; i += 32) d[i] = s[i];
}

/**
 * WARP-COOPERATIVE allocate-slot-for-write. All 32 lanes must call.
 * Strategy, in order of preference:
 *   1. Any FREE slot (HBM-first, then DRAM) — no copy required.
 *   2. Else: pick LRU HBM victim. If DRAM has any free slot, copy
 *      HBM-victim → free-DRAM-slot warp-coop (1 step, ~80 µs at PCIe).
 *      Then return the freed HBM slot.
 *   3. Else (DRAM also full): pick LRU HBM victim AND LRU DRAM victim.
 *      If DRAM victim is dirty, kick a Send. Copy HBM-victim → DRAM-
 *      victim. Return HBM slot.
 *
 * In all cases the returned slot has page_idx == target_page, kPageBusy
 * held by the caller, ready for the warp-coop write that follows.
 */
template <typename T>
CTP_GPU_FUN Page *WarpCoopAllocSlotForWrite(
    ::clio::run::gpu::IpcManager *ipc, const DeviceView<T> &v,
    clio::run::u32 block_idx, int32_t target_page, clio::run::u32 lane) {
  __shared__ Page *s_dst;
  __shared__ Page *s_evict_src;   // populated only if we need to copy
  __shared__ Page *s_evict_dst;
  Block *b = GetBlock(v.base, block_idx);
  if (lane == 0) {
    s_dst = nullptr;
    s_evict_src = nullptr;
    s_evict_dst = nullptr;
    clio::run::u32 total = TotalPagesPerBlock(v.base);
    // Path 1: free HBM?
    for (clio::run::u32 s = 0; s < v.base.gpu_pages_per_block; ++s) {
      Page *p = &b->pages[s];
      if (p->flags & (kPageBusy | kPageGetInFlight)) continue;
      if (p->page_idx < 0) {
        if (detail::TryAcquireBusy(p)) {
          if (p->page_idx < 0) { s_dst = p; break; }
          detail::ReleaseBusy(p);
        }
      }
    }
    // Path 1b: free DRAM?
    if (s_dst == nullptr && v.base.host_pages_per_block > 0) {
      for (clio::run::u32 s = v.base.gpu_pages_per_block; s < total; ++s) {
        Page *p = &b->pages[s];
        if (p->flags & (kPageBusy | kPageGetInFlight)) continue;
        if (p->page_idx < 0) {
          if (detail::TryAcquireBusy(p)) {
            if (p->page_idx < 0) { s_dst = p; break; }
            detail::ReleaseBusy(p);
          }
        }
      }
    }
    // Path 2/3: need eviction. Pick LRU HBM victim.
    if (s_dst == nullptr) {
      clio::run::u32 hv = ~0u; clio::run::u64 hclk = ~0ULL;
      for (clio::run::u32 s = 0; s < v.base.gpu_pages_per_block; ++s) {
        Page *p = &b->pages[s];
        if (p->flags & (kPageBusy | kPageGetInFlight)) continue;
        if (p->lru_clock < hclk) { hclk = p->lru_clock; hv = s; }
      }
      if (hv != ~0u) {
        Page *vp = &b->pages[hv];
        while (!detail::TryAcquireBusy(vp)) {}
        DrainPut(vp); DrainGet(vp);
        s_evict_src = vp;
        // Find a DRAM slot — free first, else LRU.
        if (v.base.host_pages_per_block > 0 && vp->page_idx >= 0) {
          clio::run::u32 dv = ~0u; clio::run::u64 dclk = ~0ULL;
          for (clio::run::u32 s = v.base.gpu_pages_per_block; s < total; ++s) {
            Page *p = &b->pages[s];
            if (p->flags & (kPageBusy | kPageGetInFlight)) continue;
            if (p->page_idx < 0) { dv = s; break; }
            if (p->lru_clock < dclk) { dclk = p->lru_clock; dv = s; }
          }
          if (dv != ~0u) {
            Page *dp = &b->pages[dv];
            while (!detail::TryAcquireBusy(dp)) {}
            DrainPut(dp); DrainGet(dp);
            if (dp->page_idx >= 0 && dp->modify_min >= 0) {
              if (!(detail::AtomicOrU32(&dp->flags, kPagePutInFlight) &
                    kPagePutInFlight)) {
                FlushPage(ipc, v, block_idx, dp, dv);
              }
            }
            s_evict_dst = dp;
          }
        }
        s_dst = vp;  // HBM victim becomes the destination after eviction
      }
    }
  }
  __syncwarp();
  Page *dst = s_dst;
  Page *evict_src = s_evict_src;
  Page *evict_dst = s_evict_dst;
  // Warp-coop copy of HBM victim → DRAM (1 step, ~80 µs at PCIe).
  if (evict_src != nullptr && evict_dst != nullptr) {
    WarpCopyUint4(evict_dst->device_ptr, evict_src->device_ptr,
                  v.base.page_size_bytes, lane);
    __syncwarp();
    if (lane == 0) {
      evict_dst->page_idx = evict_src->page_idx;
      evict_dst->modify_min = evict_src->modify_min;
      evict_dst->modify_max = evict_src->modify_max;
      evict_dst->lru_clock = evict_src->lru_clock;
      evict_dst->score = evict_src->score;
      __threadfence();
      detail::ReleaseBusy(evict_dst);
    }
  }
  __syncwarp();
  if (lane == 0 && dst != nullptr) {
    dst->page_idx = target_page;
    dst->modify_min = -1;
    dst->modify_max = -1;
    dst->lru_clock = clock64();
    dst->score = 1.0f;
  }
  __syncwarp();
  return dst;
}

// Back-compat alias for the previous name.
template <typename T>
CTP_GPU_FUN Page *WarpCoopEvictHbmToDram(
    ::clio::run::gpu::IpcManager *ipc, const DeviceView<T> &v,
    clio::run::u32 block_idx, int32_t target_page, clio::run::u32 lane) {
  return WarpCoopAllocSlotForWrite(ipc, v, block_idx, target_page, lane);
}

/** Resolve `i` to (page,offset) and return a pointer to the byte slot.
 *  Used by both read and write paths. `is_write` controls FaultPage vs
 *  not — writes don't bother faulting because they overwrite.
 *
 *  Two-tier lookup: scan HBM first, then DRAM. On a cold miss the
 *  fallback evicts in the HBM tier (the hottest target) and faults
 *  from CTE. In kAsync mode the caller also pushes a high-score hint
 *  into the rescore queue so neighboring pages get pulled in next tick. */
/** Default FamFn for Resolve/dev::vector: the view's fam_ppb policy. */
struct ViewFamily {
  CTP_GPU_FUN clio::run::u32 operator()(const DeviceViewBase &v,
                                        clio::run::u64 pg) const {
    return FamilyOf(v, pg);
  }
};

template <typename T, typename FamFn = ViewFamily>
CTP_GPU_FUN T *Resolve(::clio::run::gpu::IpcManager *ipc, DeviceView<T> v,
                        Page *&last, clio::run::u64 i, bool is_write,
                        FamFn fam_fn = FamFn{}) {
  // Wrap: the compute grid is sized by the caller and is routinely wider than
  // the cache's block count, so an unwrapped index runs off the block array.
  const clio::run::u32 nblk = v.base.nblocks ? v.base.nblocks : 1u;
  clio::run::u32 block_idx = blockIdx.x % nblk;
  Block *b = GetBlock(v.base, block_idx);
  int32_t target_page = static_cast<int32_t>(i / v.page_capacity_t);
  clio::run::u64 off_t = i - static_cast<clio::run::u64>(target_page) * v.page_capacity_t;
  // Instrument: bump resolve_total. Cheap relative to the IPC work below.
  if (v.base.stats) {
    atomicAdd_system(&v.base.stats->resolve_total, 1ULL);
  }

  // (1) Per-THREAD fast path (was per-lane into a shared array).
  Page *hit = nullptr;
  if (last && last->page_idx == target_page &&
      !(last->flags & (kPageBusy | kPageGetInFlight))) {
    hit = last;
  } else {
    // (2) Block-local linear scan across BOTH tiers (HBM first).
    //     Skip slots that are locked (manager prefetch in flight) —
    //     treat them as not-cached so we fault into a different slot
    //     rather than spin-waiting on TryAcquireBusy below.
    clio::run::u32 total = TotalPagesPerBlock(v.base);
    for (clio::run::u32 s = 0; s < total; ++s) {
      Page *p = &b->pages[s];
      if (p->page_idx != target_page) continue;
      if (p->flags & kPageBusy) continue;      // being re-bound: not ours yet
      // kPageGetInFlight is NOT a reason to skip. The page IS the one we want,
      // it is simply still arriving, and the DrainGet at the end of Resolve
      // waits for exactly that. Skipping it here forced a re-fault of a page
      // already on the wire, which is what made bulk faulting impossible:
      // every miss had to be issued AND awaited before the next could start.
      hit = p;
      last = hit;
      break;
    }
  }
  if (!hit) {
    if (v.base.stats) {
      atomicAdd_system(&v.base.stats->resolve_cold_miss, 1ULL);
    }
    if (v.base.allow_cold_miss_fault) {
      // (3a) Synchronous fault path: evict + bind + (read) fault.
      if (v.base.stats && !is_write) {
        atomicAdd_system(&v.base.stats->resolve_fault_get, 1ULL);
      }
      // CLAIM-OR-RETRY, not a spinlock. A GPU spinlock across blocks
      // deadlocks: with many CUDA blocks per cache block the spinners occupy
      // the SMs and starve the holder. Here a failed claim just picks a
      // DIFFERENT victim next round, so every block makes progress.
      // EvictSlot returns the victim with kPageBusy ALREADY held, claimed
      // BEFORE it drained the slot's outstanding futures. Claiming afterwards
      // (what this did) was too late -- the drain had already raced.
      clio::run::u32 slot = EvictSlot(ipc, v, block_idx);
      Page *p = &b->pages[slot];
      p->page_idx = target_page;
      p->lru_clock = clock64();
      p->modify_min = -1;
      p->modify_max = -1;
      p->flags = kPageBusy;   // hold the claim across bind + fault
      if (!is_write) {
        FaultPage(ipc, v, block_idx, p, slot, target_page,
                  fam_fn(v.base, target_page));
        // WAIT for the fetch. This is the "synchronous" fault path but it
        // returned the page with the GetBlob still in flight: the caller's
        // TryAcquireBusy tests kPageBusy, not kPageGetInFlight, so it
        // acquired immediately and read a page the fetch had not filled yet.
        // Small models won the race and looked correct; a 17 GB model lost
        // it on every span, yielding all-zero weights with no error anywhere
        // (77 faults issued, 0 completions ever observed).
        DrainGet(p, (DeviceViewBase *) &v.base);
      }
      detail::ReleaseBusy(p);
      hit = p;
      last = hit;
      detail::RescorePush(&b->rescore_q,
                          static_cast<clio::run::u32>(target_page), 1.0f);
    } else {
      // (3b) Async-only path: push a high-priority hint and spin-wait
      // for the manager to populate the page. Score sign distinguishes
      // alloc-only (writes; negative) from prefetch (reads; positive).
      float hint_score = is_write ? -1.0f : 1.0f;
      detail::RescorePush(&b->rescore_q,
                          static_cast<clio::run::u32>(target_page), hint_score);
      clio::run::u32 total = TotalPagesPerBlock(v.base);
      while (!hit) {
        // __threadfence() ensures we observe the manager kernel's
        // writes to page_idx and flags. Without it, the compiler
        // may hoist the load out of the loop.
        __threadfence();
        for (clio::run::u32 s = 0; s < total; ++s) {
          Page *p = &b->pages[s];
          // Use volatile reads to defeat caching across iterations.
          int32_t pi = *reinterpret_cast<volatile int32_t *>(&p->page_idx);
          if (pi != target_page) continue;
          clio::run::u32 fl = *reinterpret_cast<volatile clio::run::u32 *>(&p->flags);
          if (fl & (kPageBusy | kPageGetInFlight)) continue;
          hit = p;
          last = hit;
          break;
        }
        if (!hit) {
          if (v.base.stats) {
            atomicAdd_system(&v.base.stats->resolve_spin_iters, 1ULL);
          }
          __nanosleep(1000);
          // Re-push in case the manager dropped it (queue overflow).
          detail::RescorePush(&b->rescore_q,
                              static_cast<clio::run::u32>(target_page),
                              hint_score);
        }
      }
    }
  } else {
    if (v.base.stats) atomicAdd_system(&v.base.stats->resolve_hits, 1ULL);
  }

  // Wait on outstanding fault before returning the byte (read path).
  if (!is_write) {
    DrainGet(hit);
    // A batch member's completion clears the WHOLE run (stale-future guard).
    if (v.base.flat_layout && hit->modify_max >= 0)
      BatchSweep(v.base, block_idx, hit);
  }

  // LRU bookkeeping: cheap monotonic block-local counter would be ideal,
  // but lru_clock is read by the manager rescore phase. For now leave
  // it at 0; the rescore queue carries the hot/cold signal.

  if (is_write) {
    int32_t off_i = static_cast<int32_t>(off_t);
    // Plain stores everywhere on the per-page dirty range.
    //
    // Concurrency assumption in kAsync mode: the caller holds kPageBusy
    // on `hit` for the duration of the in-page mutation, so the manager
    // cannot atomicExch modify_min/max underneath us. In kLegacy mode
    // (host_pages_per_block == 0, manager kernel runs after user
    // kernel) there is no overlap so the bit isn't required.
    if (hit->modify_min == -1) {
      hit->modify_min = off_i;
      hit->modify_max = off_i;
      ++b->num_modified;
    } else {
      if (off_i < hit->modify_min) hit->modify_min = off_i;
      if (off_i > hit->modify_max) hit->modify_max = off_i;
    }
  }
  return reinterpret_cast<T *>(hit->device_ptr) + off_t;
}

}  // namespace clio::cte::gpu_vector

namespace cte::gpu::dev {

template <typename T,
          typename FamFn = ::clio::cte::gpu_vector::ViewFamily>
class vector;

/**
 * Per-block in-kernel handle to a host-side Vector<T>. Constructed once
 * at the top of a user kernel after CLIO_GPU_INIT. The ctor allocates
 * the per-warp last-page cache in __shared__ memory and zero-initializes
 * it via the first warp.
 *
 * Usage:
 *   __global__ void K(clio::run::IpcManagerGpuInfo info,
 *                     clio::cte::gpu_vector::DeviceView<int> view) {
 *     CLIO_GPU_INIT(info, nullptr);
 *     cte::gpu::dev::vector<int> v(view, g_ipc_manager_ptr);
 *     v.write_range(lo, hi, [] (clio::run::u64 i) { return T_for(i); });
 *     v.read_range (lo, hi, [](clio::run::u64 i, T val) { use(i, val); });
 *   }
 *
 * Only thread 0 of each warp issues `Send`s under the hood (matches the
 * IpcGpu2Cpu::ClientSend threadIdx.x==0 contract). All threads in the
 * block must construct the handle so the ctor's __syncthreads is
 * balanced. ElementRef / operator[] is intentionally NOT exposed — the
 * bulk APIs are the only race-free hot path under kAsync mode.
 */
template <typename T, typename FamFn>
class vector {

  /** Index of the cache block this CUDA block uses.
   *
   * The kernel grid is chosen by the CALLER for its arithmetic and is routinely
   * wider than the vector's block count -- a GEMV wants hundreds of blocks
   * while the cache may have one. Every internal use of blockIdx.x indexes the
   * per-block array, so an unwrapped index runs off the end (observed as a
   * segfault the moment the grid exceeded nblocks). Wrapping makes a wide grid
   * safe: CUDA blocks share cache sets instead of overrunning, and when the
   * resident window already covers a span no cache block is touched at all.
   */
  __device__ clio::run::u32 BlockIdx() const {
    const clio::run::u32 n = view_.base.nblocks ? view_.base.nblocks : 1u;
    return blockIdx.x % n;
  }
 public:
  using DeviceView = ::clio::cte::gpu_vector::DeviceView<T>;

  /**
   * @param view DeviceView<T> from `Vector<T>::Device()` (POD, captured
   *             by the kernel by value).
   * @param ipc  Kernel-scope `g_ipc_manager_ptr` declared by
   *             CLIO_GPU_INIT.
   */
  CTP_GPU_FUN vector(const DeviceView &view,
                      ::clio::run::gpu::IpcManager *ipc) noexcept
      : view_(view), ipc_(ipc), fam_fn_() {
    // No shared storage and no block-wide barrier: the handle is a plain
    // per-thread value, so a kernel may construct it anywhere (including
    // in divergent code) and every warp reads independently.
    last_page_ = nullptr;
  }

  /**
   * Same, with an explicit page->blob-family mapping. `fam_fn` is any
   * callable `(const DeviceViewBase &, u64 global_page) -> u32 family`
   * (a __device__-compatible lambda works); it is consulted on every
   * cold-miss fault this handle fires, so the composed blob name
   * "<tag>_b<family>_pi<page>" can follow whatever sharding the writer
   * used. The default (ViewFamily) applies the view's fam_ppb policy —
   * which the host Vector sets to the CAE assimilator's sharding.
   */
  CTP_GPU_FUN vector(const DeviceView &view,
                      ::clio::run::gpu::IpcManager *ipc,
                      FamFn fam_fn) noexcept
      : view_(view), ipc_(ipc), fam_fn_(fam_fn) {
    // No shared storage and no block-wide barrier: the handle is a plain
    // per-thread value, so a kernel may construct it anywhere (including
    // in divergent code) and every warp reads independently.
    last_page_ = nullptr;
  }

  /**
   * Stride-1 write fast path. Equivalent to:
   *   for (i = lo; i < hi; ++i) (*this)[i] = value_at(i);
   * but resolves the page once per page-spanning sub-range and runs a
   * **warp-cooperative** inner loop where all 32 lanes issue coalesced
   * stores in parallel.
   */
  template <typename F>
  CTP_GPU_FUN void write_range(clio::run::u64 lo, clio::run::u64 hi, F &&value_at) {
    if (lo >= hi) return;
    clio::run::u32 lane = threadIdx.x & 31;
    clio::run::u64 cap = view_.page_capacity_t;
    while (lo < hi) {
      int32_t target_page = static_cast<int32_t>(lo / cap);
      // 1. Cache lookup (lane 0, broadcast). Acquire kPageBusy if hit.
      ::clio::cte::gpu_vector::Page *hit_bcast = nullptr;
      if (lane == 0) {
        ::clio::cte::gpu_vector::Page *hit = nullptr;
        ::clio::cte::gpu_vector::Page *&last = last_page_;
        const clio::run::u32 busy_mask =
            ::clio::cte::gpu_vector::kPageBusy |
            ::clio::cte::gpu_vector::kPageGetInFlight;
        if (last && last->page_idx == target_page &&
            !(last->flags & busy_mask)) {
          hit = last;
        } else {
          ::clio::cte::gpu_vector::Block *bx =
              ::clio::cte::gpu_vector::GetBlock(view_.base, BlockIdx());
          clio::run::u32 total =
              ::clio::cte::gpu_vector::TotalPagesPerBlock(view_.base);
          for (clio::run::u32 s = 0; s < total; ++s) {
            ::clio::cte::gpu_vector::Page *p = &bx->pages[s];
            if (p->page_idx != target_page) continue;
            if (p->flags & busy_mask) continue;
            hit = p;
            last = hit;
            break;
          }
        }
        if (hit) {
          while (!::clio::cte::gpu_vector::detail::TryAcquireBusy(hit)) {}
          if (view_.base.stats) {
            atomicAdd_system(&view_.base.stats->resolve_hits, 1ULL);
          }
        } else if (view_.base.stats) {
          atomicAdd_system(&view_.base.stats->resolve_cold_miss, 1ULL);
        }
        if (view_.base.stats) {
          atomicAdd_system(&view_.base.stats->resolve_total, 1ULL);
        }
        hit_bcast = hit;
      }
      __syncwarp();
      ::clio::cte::gpu_vector::Page *p =
          ::clio::cte::gpu_vector::detail::WarpBroadcastPage(
              (lane == 0) ? hit_bcast : nullptr);
      // 2. Cold miss → warp-coop evict HBM→DRAM, bind target_page in HBM.
      if (p == nullptr) {
        p = ::clio::cte::gpu_vector::WarpCoopEvictHbmToDram(
            ipc_, view_, BlockIdx(), target_page, lane);
        if (lane == 0) {
          last_page_ = p;
        }
        __syncwarp();
      }
      // 3. Warp-coop write.
      T *page_base = static_cast<T *>(p->device_ptr);
      clio::run::u64 page_start_i =
          static_cast<clio::run::u64>(p->page_idx) * cap;
      clio::run::u64 page_end_i = page_start_i + cap;
      clio::run::u64 stop = (hi < page_end_i) ? hi : page_end_i;
      clio::run::u64 page_off_lo = lo - page_start_i;
      if (lane == 0) {
        int32_t hi_off = static_cast<int32_t>(stop - 1 - page_start_i);
        if (p->modify_min < 0) {
          p->modify_min = static_cast<int32_t>(page_off_lo);
          ::clio::cte::gpu_vector::detail::AtomicIncU32(
              &::clio::cte::gpu_vector::GetBlock(view_.base, BlockIdx())
                   ->num_modified);
        } else if (static_cast<int32_t>(page_off_lo) < p->modify_min) {
          p->modify_min = static_cast<int32_t>(page_off_lo);
        }
        if (hi_off > p->modify_max) p->modify_max = hi_off;
      }
      __syncwarp();
      clio::run::u64 nelem = stop - lo;
      for (clio::run::u64 j = lane; j < nelem; j += 32) {
        page_base[page_off_lo + j] = value_at(lo + j);
      }
      __syncwarp();
      if (lane == 0) {
        ::clio::cte::gpu_vector::detail::ReleaseBusy(p);
      }
      __syncwarp();
      lo = stop;
    }
  }

  /**
   * Pointer to a span lying entirely within ONE page, or nullptr.
   *
   * Staging exists only because read_range hands bytes back through a
   * per-value callback and never exposes where they live; the bytes are
   * already in device memory and that copy measured 15x the compute it
   * feeds. This returns the location instead, with the page pinned.
   *
   * WARP-COLLECTIVE: every lane of one warp must enter (lane 0 resolves and
   * acquires, all lanes read the broadcast slot). On success the page is
   * PINNED (kPageBusy) and the caller must release_span() after reading.
   * Returns nullptr -- pinning nothing -- when the span straddles a page
   * boundary, or when the slot was rebound between resolve and acquire; the
   * caller stages that (rare) span instead.
   */
  CTP_GPU_FUN const uint8_t *span_ptr(clio::run::u64 lo, clio::run::u64 hi,
                                      ::clio::cte::gpu_vector::Page **held) {
    *held = nullptr;
    if (lo >= hi) return nullptr;
    const clio::run::u64 cap = view_.page_capacity_t;
    const clio::run::u64 pg = lo / cap;
    if (pg != ((hi - 1) / cap)) return nullptr;   // straddles: caller stages
    const clio::run::u32 lane = threadIdx.x & 31;
    if (lane == 0) {
      (void)::clio::cte::gpu_vector::Resolve(
          ipc_, view_, last_page_, lo, /*is_write=*/false, fam_fn_);
      ::clio::cte::gpu_vector::Page *p0 =
          ::clio::cte::gpu_vector::detail::WarpBroadcastPage(last_page_);
      while (!::clio::cte::gpu_vector::detail::TryAcquireBusy(p0)) {}
      // ISSUE-AHEAD (async prefetch): start the next pages' fetches, wait
      // only at first access (hit-in-waiting + DrainGet provide the wait).
      {
        ::clio::cte::gpu_vector::Block *bx =
            ::clio::cte::gpu_vector::GetBlock(view_.base, BlockIdx());
        const clio::run::u32 total =
            ::clio::cte::gpu_vector::TotalPagesPerBlock(view_.base);
        for (clio::run::u64 la = 1; la <= 4; ++la) {
          const clio::run::u64 np = pg + la;
          if (view_.base.fam_ppb != 0 && np >= view_.base.fam_ppb) break;
          bool have = false;
          clio::run::u32 nfl = 0;
          for (clio::run::u32 s2 = 0; s2 < total; ++s2) {
            if (bx->pages[s2].flags &
                ::clio::cte::gpu_vector::kPageGetInFlight) ++nfl;
            if (bx->pages[s2].page_idx == static_cast<int32_t>(np)) {
              have = true;
            }
          }
          if (have) continue;
          // Cap outstanding prefetches: a prefetched page's in-flight flag is
          // only cleared at ACCESS, so uncapped issue leaks slots until the
          // block runs out of them entirely.
          if (nfl >= 8) break;
          const clio::run::u32 sl =
              ::clio::cte::gpu_vector::EvictSlot(ipc_, view_, BlockIdx());
          ::clio::cte::gpu_vector::Page *p2 = &bx->pages[sl];
          p2->page_idx   = static_cast<int32_t>(np);
          p2->lru_clock  = clock64();
          p2->modify_min = -1;
          p2->modify_max = -1;
          ::clio::cte::gpu_vector::FaultPage(ipc_, view_, BlockIdx(), p2, sl,
                                             static_cast<int32_t>(np),
                                             fam_fn_(view_.base, np));
          ::clio::cte::gpu_vector::detail::AtomicOrU32(
              &p2->flags, ::clio::cte::gpu_vector::kPageGetInFlight);
          ::clio::cte::gpu_vector::detail::ReleaseBusy(p2);
        }
      }
    }
    __syncwarp();
    ::clio::cte::gpu_vector::Page *p =
        ::clio::cte::gpu_vector::detail::WarpBroadcastPage(last_page_);
    // Validate under the pin: the slot can be rebound between resolve and
    // acquire. A stale slot must not be dereferenced.
    if (p == nullptr || p->device_ptr == nullptr ||
        static_cast<clio::run::u64>(p->page_idx) != pg) {
      __syncwarp();
      if (lane == 0 && p != nullptr)
        ::clio::cte::gpu_vector::detail::ReleaseBusy(p);
      return nullptr;
    }
    *held = p;
    return static_cast<const uint8_t *>(p->device_ptr) + (lo - pg * cap);
  }

  CTP_GPU_FUN void release_span(::clio::cte::gpu_vector::Page *held) {
    if (held) ::clio::cte::gpu_vector::detail::ReleaseBusy(held);
  }

  /**
   * BULK read of a contiguous span into `dst`, using 16-byte accesses.
   *
   * The callback form above moves ONE ELEMENT per lane per iteration, and the
   * vector is instantiated with T = uint8_t, so it moves one BYTE per lane:
   * 32 B per warp-iteration, 768 iterations for a 24 KiB span, each a
   * dependent global->shared round trip. Measured at ~68 MB/s per block, which
   * made staging 15x the cost of the dequant it feeds. The per-byte
   * `consume(index, value)` signature is what forces that granularity, so the
   * fix is a separate entry point rather than a change to it.
   *
   * Falls back to bytes when either side is not 16-byte aligned.
   */
  CTP_GPU_FUN void read_range_to(clio::run::u64 lo, clio::run::u64 hi,
                                 uint8_t *dst) {
    if (lo >= hi) return;
    const clio::run::u64 lo0 = lo;
    clio::run::u32 lane = threadIdx.x & 31;
    clio::run::u64 cap = view_.page_capacity_t;
    while (lo < hi) {
      if (lane == 0) {
        (void)::clio::cte::gpu_vector::Resolve(
            ipc_, view_, last_page_, lo, /*is_write=*/false, fam_fn_);
        ::clio::cte::gpu_vector::Page *p0 =
          ::clio::cte::gpu_vector::detail::WarpBroadcastPage(last_page_);
        while (!::clio::cte::gpu_vector::detail::TryAcquireBusy(p0)) {}
      }
      __syncwarp();
      ::clio::cte::gpu_vector::Page *p =
        ::clio::cte::gpu_vector::detail::WarpBroadcastPage(last_page_);
      const uint8_t *page_base = static_cast<const uint8_t *>(p->device_ptr);
      clio::run::u64 page_start_i =
          static_cast<clio::run::u64>(p->page_idx) * cap;
      clio::run::u64 page_end_i = page_start_i + cap;
      clio::run::u64 stop = (hi < page_end_i) ? hi : page_end_i;
      const uint8_t *src = page_base + (lo - page_start_i);
      uint8_t *d = dst + (lo - lo0);
      clio::run::u64 n = stop - lo;
      const bool wide = ((reinterpret_cast<uintptr_t>(src) & 15u) == 0) &&
                        ((reinterpret_cast<uintptr_t>(d) & 15u) == 0);
      if (wide) {
        const clio::run::u64 n16 = n >> 4;
        const uint4 *s4 = reinterpret_cast<const uint4 *>(src);
        uint4 *d4 = reinterpret_cast<uint4 *>(d);
        for (clio::run::u64 j = lane; j < n16; j += 32) d4[j] = s4[j];
        for (clio::run::u64 j = (n16 << 4) + lane; j < n; j += 32) d[j] = src[j];
      } else {
        for (clio::run::u64 j = lane; j < n; j += 32) d[j] = src[j];
      }
      __syncwarp();
      if (lane == 0) ::clio::cte::gpu_vector::detail::ReleaseBusy(p);
      __syncwarp();
      lo = stop;
    }
  }

  /**
   * Stride-1 read fast path — warp-cooperative twin of write_range.
   * consume is called from each lane as `void consume(clio::run::u64 i, T v)`.
   */
  template <typename F>
  CTP_GPU_FUN void read_range(clio::run::u64 lo, clio::run::u64 hi, F &&consume) {
    if (lo >= hi) return;
    const clio::run::u64 cap = view_.page_capacity_t;
    // BLOCK-COLLECTIVE. Every thread of the block walks the same range;
    // the per-lane cached page satisfies the common case with no
    // coordination at all, and when ANY thread lacks the page the whole
    // block faults it in together (see BlockFaultPage). Must therefore be
    // called uniformly by the block.
    while (lo < hi) {
      const int32_t target = static_cast<int32_t>(lo / cap);
      ::clio::cte::gpu_vector::Page *p = last_page_;
      const bool have = (p != nullptr) && p->page_idx == target &&
                        !(p->flags & (::clio::cte::gpu_vector::kPageBusy |
                                      ::clio::cte::gpu_vector::kPageGetInFlight));
      if (!__syncthreads_and(have ? 1 : 0)) {
        // One resolve per PAGE TRANSITION for the whole block -- the same
        // accounting the warp path reported, so "a sequential read
        // resolves exactly once per page" still holds.
        if (threadIdx.x == 0 && view_.base.stats) {
          atomicAdd_system(&view_.base.stats->resolve_total, 1ULL);
        }
        const bool was_cached = (p != nullptr && p->page_idx == target);
        p = ::clio::cte::gpu_vector::BlockFaultPage(
            ipc_, view_, BlockIdx(), target, fam_fn_);
        if (threadIdx.x == 0 && view_.base.stats) {
          if (was_cached) {
            atomicAdd_system(&view_.base.stats->resolve_hits, 1ULL);
          } else {
            atomicAdd_system(&view_.base.stats->resolve_cold_miss, 1ULL);
          }
        }
        last_page_ = p;          // per-lane
      }
      if (p == nullptr) return;
      const clio::run::u64 page_start =
          static_cast<clio::run::u64>(p->page_idx) * cap;
      const clio::run::u64 stop =
          (hi < page_start + cap) ? hi : (page_start + cap);
      const T *base = static_cast<const T *>(p->device_ptr);
      for (clio::run::u64 j = lo + threadIdx.x; j < stop; j += blockDim.x) {
        consume(j, base[j - page_start]);
      }
      lo = stop;
      __syncthreads();   // page may change on the next iteration
    }
  }

  /**
   * Iterator + rescore write form. `next_i(k)` returns the k-th index
   * (`k` in [0, n)). `value_at(i)` returns the T to store at element i.
   * `rescore(cur_k, clipped_lookahead, clipped_lookbehind, queue_ref)`
   * is called by lane 0 once per page transition; the framework clips
   * `lookahead` to the rescore queue's remaining capacity so the
   * lambda cannot overflow.
   *
   * Per-element re-resolve makes this slower than the contiguous
   * (lo, hi) form for arbitrary index streams. For contiguous streams
   * the lane-fast-path makes consecutive same-page hits free.
   */
  template <typename NextI, typename V, typename R>
  CTP_GPU_FUN void write_range(NextI next_i, clio::run::u64 n,
                                 V value_at, R rescore,
                                 clio::run::u32 lookahead, clio::run::u32 lookbehind) {
    if (n == 0) return;
    clio::run::u32 lane = threadIdx.x & 31;
    // Iterator form is lane-0 only. Lanes != 0 just return; control
    // reconverges naturally at the function epilogue (no __syncwarp
    // here — that would deadlock against the early-returned lanes).
    if (lane != 0) return;
    ::clio::cte::gpu_vector::Block *b =
        ::clio::cte::gpu_vector::GetBlock(view_.base, BlockIdx());
    clio::run::u64 cap = view_.page_capacity_t;
    ::clio::cte::gpu_vector::Page *held = nullptr;
    int32_t held_page = -1;
    for (clio::run::u64 k = 0; k < n; ++k) {
      clio::run::u64 idx = next_i(k);
      int32_t pg = static_cast<int32_t>(idx / cap);
      if (pg != held_page) {
        if (held) ::clio::cte::gpu_vector::detail::ReleaseBusy(held);
        clio::run::u32 cap_left =
            ::clio::cte::gpu_vector::detail::RescoreRemaining(&b->rescore_q);
        clio::run::u32 clipped_la = (lookahead < cap_left) ? lookahead : cap_left;
        rescore(k, clipped_la, lookbehind, b->rescore_q);
        (void)::clio::cte::gpu_vector::Resolve(
            ipc_, view_, last_page_, idx, /*is_write=*/true, fam_fn_);
        held = ::clio::cte::gpu_vector::detail::WarpBroadcastPage(last_page_);
        while (!::clio::cte::gpu_vector::detail::TryAcquireBusy(held)) {}
        held_page = pg;
      }
      clio::run::u64 off_t = idx - static_cast<clio::run::u64>(pg) * cap;
      T *page_base = static_cast<T *>(held->device_ptr);
      page_base[off_t] = value_at(idx);
      int32_t off_i = static_cast<int32_t>(off_t);
      if (off_i < held->modify_min || held->modify_min < 0) {
        held->modify_min = off_i;
      }
      if (off_i > held->modify_max) held->modify_max = off_i;
    }
    if (held) ::clio::cte::gpu_vector::detail::ReleaseBusy(held);
  }

  /** Iterator + rescore read form (lane-0 only). */
  template <typename NextI, typename C, typename R>
  CTP_GPU_FUN void read_range(NextI next_i, clio::run::u64 n,
                                C consume, R rescore,
                                clio::run::u32 lookahead, clio::run::u32 lookbehind) {
    if (n == 0) return;
    clio::run::u32 lane = threadIdx.x & 31;
    if (lane != 0) return;
    ::clio::cte::gpu_vector::Block *b =
        ::clio::cte::gpu_vector::GetBlock(view_.base, BlockIdx());
    clio::run::u64 cap = view_.page_capacity_t;
    ::clio::cte::gpu_vector::Page *held = nullptr;
    int32_t held_page = -1;
    for (clio::run::u64 k = 0; k < n; ++k) {
      clio::run::u64 idx = next_i(k);
      int32_t pg = static_cast<int32_t>(idx / cap);
      if (pg != held_page) {
        if (held) ::clio::cte::gpu_vector::detail::ReleaseBusy(held);
        clio::run::u32 cap_left =
            ::clio::cte::gpu_vector::detail::RescoreRemaining(&b->rescore_q);
        clio::run::u32 clipped_la = (lookahead < cap_left) ? lookahead : cap_left;
        rescore(k, clipped_la, lookbehind, b->rescore_q);
        (void)::clio::cte::gpu_vector::Resolve(
            ipc_, view_, last_page_, idx, /*is_write=*/false, fam_fn_);
        held = ::clio::cte::gpu_vector::detail::WarpBroadcastPage(last_page_);
        while (!::clio::cte::gpu_vector::detail::TryAcquireBusy(held)) {}
        held_page = pg;
      }
      clio::run::u64 off_t = idx - static_cast<clio::run::u64>(pg) * cap;
      T *page_base = static_cast<T *>(held->device_ptr);
      consume(idx, page_base[off_t]);
    }
    if (held) ::clio::cte::gpu_vector::detail::ReleaseBusy(held);
  }

  /** Flush every dirty page in the calling block. Lane-0 only: the
   *  gpu2cpu Send producer contract requires threadIdx.x == 0. */
  CTP_GPU_FUN void FlushAll() {
    if ((threadIdx.x & 31) != 0) return;
    if (threadIdx.x != 0) return;
    ::clio::cte::gpu_vector::FlushAllInBlock(ipc_, view_, BlockIdx());
  }

  CTP_GPU_FUN const DeviceView &view() const { return view_; }

 private:
  DeviceView view_;
  ::clio::run::gpu::IpcManager *ipc_;
  /** Per-THREAD cache of the last page this handle resolved. Formerly a
   *  pointer into a per-block __shared__ array, which is what limited a
   *  block to one reading warp; see WarpBroadcastPage. */
  ::clio::cte::gpu_vector::Page *last_page_;
  /** Page -> blob-family mapping consulted on every fault (see ctor). */
  FamFn fam_fn_;
};

}  // namespace cte::gpu::dev

#endif  // CTP_IS_GPU_COMPILER

#endif  // CLIO_CTE_GPU_VECTOR_KERNELS_H_
