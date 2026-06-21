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
  ctp::ipc::AllocatorId alloc =
      (page->tier == 0) ? v.pages_alloc_id : v.host_pages_alloc_id;
  task->blob_data_ = detail::MakeBlobShmPtr(page->device_ptr, alloc);

  ctp::ipc::FullPtr<clio::cte::core::PutBlobTask> fp;
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
  ctp::ipc::AllocatorId alloc =
      (page->tier == 0) ? v.base.pages_alloc_id : v.base.host_pages_alloc_id;
  task->blob_data_ = detail::MakeBlobShmPtr(
      reinterpret_cast<char *>(page->device_ptr) + mn_b, alloc);

  ctp::ipc::FullPtr<clio::cte::core::PutBlobTask> fp;
  fp.shm_.alloc_id_ = v.base.put_pool_alloc_id;
  fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task);
  fp.ptr_ = task;

  page->active_put = ipc->Send(fp);
}

/** Submit a GetBlob to fault `target_page_idx` into `page->device_ptr`. */
template <typename T>
CTP_GPU_FUN void FaultPage(::clio::run::gpu::IpcManager *ipc,
                            const DeviceView<T> &v, clio::run::u32 block_idx,
                            Page *page, clio::run::u32 slot,
                            int32_t target_page_idx) {
  auto *task = GetGetTask(v.base, block_idx, slot);
  // Reset lifecycle flags + mint a fresh task_id for slot reuse.
  task->task_flags_.Clear();
  task->return_code_.store(0);
  task->task_id_ = clio::run::CreateTaskId();
  task->offset_ = 0;
  task->size_ = v.base.page_size_bytes;
  task->gpu_page_idx_ = static_cast<clio::run::u32>(target_page_idx);
  ctp::ipc::AllocatorId alloc =
      (page->tier == 0) ? v.base.pages_alloc_id : v.base.host_pages_alloc_id;
  task->blob_data_ = detail::MakeBlobShmPtr(page->device_ptr, alloc);
  ctp::ipc::FullPtr<clio::cte::core::GetBlobTask> fp;
  fp.shm_.alloc_id_ = v.base.get_pool_alloc_id;
  fp.shm_.off_ = reinterpret_cast<clio::run::u64>(task);
  fp.ptr_ = task;
  page->active_get = ipc->Send(fp);
}

/** Wait on any in-flight put for this page, then clear the slot. */
CTP_GPU_FUN void DrainPut(Page *page) {
  if (!page->active_put.IsNull()) {
    page->active_put.Wait();
    page->active_put = clio::run::gpu::Future<clio::cte::core::PutBlobTask>();
    detail::AtomicClearBitsU32(&page->flags, kPagePutInFlight);
  }
}

/** Wait on any in-flight get for this page, then clear the slot. */
CTP_GPU_FUN void DrainGet(Page *page) {
  if (!page->active_get.IsNull()) {
    page->active_get.Wait();
    page->active_get = clio::run::gpu::Future<clio::cte::core::GetBlobTask>();
    detail::AtomicClearBitsU32(&page->flags, kPageGetInFlight);
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
  // Free slot?
  for (clio::run::u32 s = tier_lo; s < tier_hi; ++s) {
    if (b->pages[s].page_idx < 0) {
      DrainPut(&b->pages[s]);
      DrainGet(&b->pages[s]);
      return s;
    }
  }
  // LRU within range.
  clio::run::u32 lru = tier_lo;
  clio::run::u64 lru_clock = b->pages[tier_lo].lru_clock;
  for (clio::run::u32 s = tier_lo + 1; s < tier_hi; ++s) {
    if (b->pages[s].lru_clock < lru_clock) {
      lru_clock = b->pages[s].lru_clock;
      lru = s;
    }
  }
  DrainPut(&b->pages[lru]);
  DrainGet(&b->pages[lru]);
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
template <typename T>
CTP_GPU_FUN T *Resolve(::clio::run::gpu::IpcManager *ipc, DeviceView<T> v,
                        Page **last_page_array, clio::run::u64 i, bool is_write) {
  clio::run::u32 block_idx = blockIdx.x;
  Block *b = GetBlock(v.base, block_idx);
  int32_t target_page = static_cast<int32_t>(i / v.page_capacity_t);
  clio::run::u64 off_t = i - static_cast<clio::run::u64>(target_page) * v.page_capacity_t;
  // Instrument: bump resolve_total. Cheap relative to the IPC work below.
  if (v.base.stats) {
    atomicAdd_system(&v.base.stats->resolve_total, 1ULL);
  }

  // (1) Per-lane fast path.
  Page *&last = detail::LaneLastPage(last_page_array);
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
      if (p->flags & (kPageBusy | kPageGetInFlight)) continue;
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
      clio::run::u32 slot = EvictSlot(ipc, v, block_idx);
      Page *p = &b->pages[slot];
      p->page_idx = target_page;
      p->lru_clock = clock64();
      p->modify_min = -1;
      p->modify_max = -1;
      p->flags = 0;
      if (!is_write) FaultPage(ipc, v, block_idx, p, slot, target_page);
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
  if (!is_write) DrainGet(hit);

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

template <typename T>
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
template <typename T>
class vector {
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
      : view_(view), ipc_(ipc) {
    __shared__ ::clio::cte::gpu_vector::Page *last_page_storage[32];
    last_page_array_ = last_page_storage;
    if (threadIdx.x < 32) last_page_array_[threadIdx.x] = nullptr;
    __syncthreads();
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
      if (lane == 0) {
        ::clio::cte::gpu_vector::Page *hit = nullptr;
        ::clio::cte::gpu_vector::Page *&last =
            ::clio::cte::gpu_vector::detail::LaneLastPage(last_page_array_);
        const clio::run::u32 busy_mask =
            ::clio::cte::gpu_vector::kPageBusy |
            ::clio::cte::gpu_vector::kPageGetInFlight;
        if (last && last->page_idx == target_page &&
            !(last->flags & busy_mask)) {
          hit = last;
        } else {
          ::clio::cte::gpu_vector::Block *bx =
              ::clio::cte::gpu_vector::GetBlock(view_.base, blockIdx.x);
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
        last_page_array_[0] = hit;
      }
      __syncwarp();
      ::clio::cte::gpu_vector::Page *p = last_page_array_[0];
      // 2. Cold miss → warp-coop evict HBM→DRAM, bind target_page in HBM.
      if (p == nullptr) {
        p = ::clio::cte::gpu_vector::WarpCoopEvictHbmToDram(
            ipc_, view_, blockIdx.x, target_page, lane);
        if (lane == 0) {
          ::clio::cte::gpu_vector::detail::LaneLastPage(last_page_array_) = p;
          last_page_array_[0] = p;
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
              &::clio::cte::gpu_vector::GetBlock(view_.base, blockIdx.x)
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
   * Stride-1 read fast path — warp-cooperative twin of write_range.
   * consume is called from each lane as `void consume(clio::run::u64 i, T v)`.
   */
  template <typename F>
  CTP_GPU_FUN void read_range(clio::run::u64 lo, clio::run::u64 hi, F &&consume) {
    if (lo >= hi) return;
    clio::run::u32 lane = threadIdx.x & 31;
    clio::run::u64 cap = view_.page_capacity_t;
    const clio::run::u32 first_page = static_cast<clio::run::u32>(lo / cap);
    const clio::run::u32 last_page = static_cast<clio::run::u32>((hi - 1) / cap);
    // Lookahead = ~HBM tier size. Larger values just thrash the
    // rescore_q (it's size 256 with kRescoreQueueCap; each page
    // transition pushes kLookahead entries, so at lookahead=32 with
    // 40 pages per block × 4 blocks we push 5120 entries, most
    // dropped via atomic rollback — pure contention).
    constexpr int kLookahead = 8;
    constexpr int kLookbehind = 2;
    while (lo < hi) {
      if (lane == 0) {
        clio::run::u32 cur_page = static_cast<clio::run::u32>(lo / cap);
        ::clio::cte::gpu_vector::Block *bx =
            ::clio::cte::gpu_vector::GetBlock(view_.base, blockIdx.x);
        for (int la = 1; la <= kLookahead; ++la) {
          clio::run::u32 hint = cur_page + static_cast<clio::run::u32>(la);
          if (hint > last_page) break;
          (void)::clio::cte::gpu_vector::detail::RescorePush(
              &bx->rescore_q, hint,
              1.0f - static_cast<float>(la) / (kLookahead + 1));
        }
        for (int lb = 1; lb <= kLookbehind; ++lb) {
          if (cur_page < first_page + static_cast<clio::run::u32>(lb)) break;
          clio::run::u32 hint = cur_page - static_cast<clio::run::u32>(lb);
          (void)::clio::cte::gpu_vector::detail::RescorePush(
              &bx->rescore_q, hint, 0.1f);
        }
        (void)::clio::cte::gpu_vector::Resolve(
            ipc_, view_, last_page_array_, lo, /*is_write=*/false);
        ::clio::cte::gpu_vector::Page *p = last_page_array_[0];
        while (!::clio::cte::gpu_vector::detail::TryAcquireBusy(p)) {}
      }
      __syncwarp();
      ::clio::cte::gpu_vector::Page *p = last_page_array_[0];
      T *page_base = static_cast<T *>(p->device_ptr);
      clio::run::u64 page_start_i =
          static_cast<clio::run::u64>(p->page_idx) * cap;
      clio::run::u64 page_end_i = page_start_i + cap;
      clio::run::u64 stop = (hi < page_end_i) ? hi : page_end_i;
      clio::run::u64 page_off_lo = lo - page_start_i;
      clio::run::u64 nelem = stop - lo;
      for (clio::run::u64 j = lane; j < nelem; j += 32) {
        consume(lo + j, page_base[page_off_lo + j]);
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
        ::clio::cte::gpu_vector::GetBlock(view_.base, blockIdx.x);
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
            ipc_, view_, last_page_array_, idx, /*is_write=*/true);
        held = last_page_array_[0];
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
        ::clio::cte::gpu_vector::GetBlock(view_.base, blockIdx.x);
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
            ipc_, view_, last_page_array_, idx, /*is_write=*/false);
        held = last_page_array_[0];
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
    ::clio::cte::gpu_vector::FlushAllInBlock(ipc_, view_, blockIdx.x);
  }

  CTP_GPU_FUN const DeviceView &view() const { return view_; }

 private:
  DeviceView view_;
  ::clio::run::gpu::IpcManager *ipc_;
  ::clio::cte::gpu_vector::Page **last_page_array_;
};

}  // namespace cte::gpu::dev

#endif  // CTP_IS_GPU_COMPILER

#endif  // CLIO_CTE_GPU_VECTOR_KERNELS_H_
