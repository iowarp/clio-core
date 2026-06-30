/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_GPU_VECTOR_H_
#define CLIO_CTE_GPU_VECTOR_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/gpu/gpu_ipc_manager.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>
#include <clio_ctp/util/gpu_api.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector_kernels.h>
#include <clio_cte/gpu_vector/gpu_vector_page.h>
#include <clio_cte/gpu_vector/gpu_vector_view.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

#if CTP_IS_GPU_COMPILER

namespace clio::cte::gpu_vector {

/**
 * Cache management strategy. kLegacy reproduces the original
 * pre-async-thread behavior (single HBM tier, periodic flush kernel,
 * no rescore queue). kAsync wires in the new four-phase
 * CacheManagerKernel, two-tier cache, and per-block rescore queues.
 *
 * `host_pages_per_block > 0` auto-promotes the mode to kAsync. Existing
 * callers that pass only the legacy parameters get kLegacy by default.
 */
enum class CacheMode {
  kLegacy = 0,
  kAsync = 1,
};

/**
 * Producer-only GPU-resident vector backed by CTE blob storage. See
 * the file-level comment in gpu_vector_page.h for the data layout and
 * AGENTS.md "GPU Producer-Only Model" for the CLIO_IPC->Send model.
 *
 * Lifecycle:
 *   - Caller must have already initialized the CLIO Runtime runtime and
 *     the CTE core pool (see clio::cte::core::CLIO_CTE_CLIENT_INIT plus
 *     AsyncCreate(...)).
 *   - ctor: allocates the backends (HBM pages [kDeviceMem], optional
 *           DRAM pages [kPinnedHost], swap scratch [kDeviceMem], meta
 *           [kDeviceMem], task pools [kPinnedHost]); registers them
 *           with the runtime; resolves/creates the tag; placement-news
 *           every PutBlob/GetBlob slot; runs an InitMeta kernel; spawns
 *           the cache-management thread.
 *   - dtor: stops the cache thread, FlushAllSync, unregisters + frees
 *           backends.
 */
template <typename T>
class Vector {
 public:
  /**
   * @param tag_name                  CTE tag name (per-page blobs land here).
   * @param nblocks                   Number of independent block streams.
   * @param gpu_id                    GPU device id.
   * @param gpu_pages_per_block       HBM cache slots per block.
   * @param host_pages_per_block      DRAM cache slots per block.
   *                                  0 ⇒ legacy single-tier mode.
   * @param page_size_bytes           Per-page byte size.
   * @param cache_period_us           Cache-thread tick in MICROSECONDS.
   *                                  0 disables the thread; FlushAllSync
   *                                  still works. Default 50000 us = 50 ms.
   * @param mode                      Force CacheMode. Default: kLegacy
   *                                  when host_pages_per_block == 0,
   *                                  else kAsync (auto-promoted in ctor).
   * @param manager_threads_per_block Threads per CacheManagerKernel
   *                                  block. Must be a multiple of 32
   *                                  (default 32, i.e. one warp).
   */
  Vector(const std::string &tag_name, clio::run::u32 nblocks,
         clio::run::u32 gpu_id = 0,
         clio::run::u32 gpu_pages_per_block = 4,
         clio::run::u32 host_pages_per_block = 0,
         clio::run::u64 page_size_bytes = 1ULL << 20,
         clio::run::u32 cache_period_us = 50000,
         CacheMode mode = CacheMode::kLegacy,
         clio::run::u32 manager_threads_per_block = 32,
         bool allow_cold_miss_fault = false);
  ~Vector();

  Vector(const Vector &) = delete;
  Vector &operator=(const Vector &) = delete;

  /** Synchronously drain every dirty page. Launches the cache-mgmt
   *  kernel once and a drain kernel that calls Wait() on all in-flight
   *  put / get futures. */
  void FlushAllSync();

  /** Cache-thread-only: drains the per-block host_prefetch_q, issues
   *  AsyncGetBlob via the CPU-side CTE client for each directive,
   *  waits for completion, then launches a tiny kernel to clear
   *  kPageBusy + kPageGetInFlight on the prefetched slot. The
   *  stream parameter is the non-blocking cache-thread stream so
   *  the clear-flags kernel doesn't block behind any default-stream
   *  user kernel (which would deadlock against the slot lock the
   *  user kernel is spin-waiting on). */
  void DrainHostPrefetchQueue(void *cuda_stream);

  /** POD captured by user kernels — pass by value into a __global__. */
  DeviceView<T> Device() const { return view_; }

  const clio::cte::core::TagId &TagId() const { return view_.base.tag_id; }

  /** Read a coherent snapshot of the pinned-host counters. Caller
   *  should cudaDeviceSync first so the device's atomicAdd writes
   *  are visible to the host through PCIe cache snooping. */
  ::clio::cte::gpu_vector::VectorStats StatsSnapshot() const {
    ::clio::cte::gpu_vector::VectorStats out{};
#if !CTP_IS_DEVICE_PASS
    if (impl_ && impl_->stats) out = *impl_->stats;
#endif
    return out;
  }
  /** Zero out all counters so the caller can measure a single phase.
   *  Caller should cudaDeviceSync afterward so any in-flight manager
   *  writes don't race with the host memset. */
  void StatsReset() {
#if !CTP_IS_DEVICE_PASS
    if (impl_ && impl_->stats) {
      std::memset(impl_->stats, 0,
                  sizeof(::clio::cte::gpu_vector::VectorStats));
    }
#endif
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  DeviceView<T> view_{};
};

namespace detail {

/** One-shot init kernel — non-template so nvcc reliably registers it
 *  through the standard <<<>>> launch glue. Operates on DeviceViewBase
 *  so it doesn't need T. Initializes BOTH tiers' Page slots, plus the
 *  per-block rescore/prefetch queue headers and swap_scratch slot.
 *
 *  Grid contract: <<<nblocks, threads_per_block>>>. Each block owns its
 *  Block struct and walks all its slots in stride. */
__global__ void InitMetaKernel(DeviceViewBase v, char *pages_base,
                                char *host_pages_base, char *swap_base) {
  if (blockIdx.x >= v.nblocks) return;
  Block *b = GetBlock(v, blockIdx.x);
  clio::run::u32 total = TotalPagesPerBlock(v);
  if (threadIdx.x == 0) {
    b->block_idx = blockIdx.x;
    b->num_modified = 0;
    b->gpu_pages_per_block = v.gpu_pages_per_block;
    b->host_pages_per_block = v.host_pages_per_block;
    b->swap_scratch = swap_base
        ? swap_base + static_cast<clio::run::u64>(blockIdx.x) * v.page_size_bytes
        : nullptr;
    b->flush_cursor = 0;
    b->rescore_q.head = 0;
    b->rescore_q.tail = 0;
    b->prefetch_q.head = 0;
    b->prefetch_q.tail = 0;
  }
  __syncthreads();
  for (clio::run::u32 s = threadIdx.x; s < total; s += blockDim.x) {
    Page *p = &b->pages[s];
    if (s < v.gpu_pages_per_block) {
      p->tier = 0;
      p->device_ptr = pages_base +
          (static_cast<clio::run::u64>(blockIdx.x) * v.gpu_pages_per_block + s) *
           v.page_size_bytes;
    } else {
      p->tier = 1;
      clio::run::u32 host_slot = s - v.gpu_pages_per_block;
      p->device_ptr = host_pages_base +
          (static_cast<clio::run::u64>(blockIdx.x) * v.host_pages_per_block +
           host_slot) * v.page_size_bytes;
    }
    p->page_idx = -1;
    p->modify_min = -1;
    p->modify_max = -1;
    p->flags = 0;
    p->lru_clock = 0;
    p->score = 0.0f;
    new (&p->active_put) clio::run::gpu::Future<clio::cte::core::PodPutBlobTask>();
    new (&p->active_get) clio::run::gpu::Future<clio::cte::core::PodGetBlobTask>();
  }
}

/** Legacy cache-management kernel — preserved verbatim from the
 *  pre-async-thread design for callers that opt into kLegacy mode.
 *  Walks every (block, slot) pair and submits PutBlob for any with a
 *  non-empty modify range. */
__global__ void LegacyFlushKernel(::clio::run::IpcManagerGpuInfo info,
                                   DeviceViewBase v) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  clio::run::u32 total_per_block = TotalPagesPerBlock(v);
  clio::run::u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  clio::run::u32 total = v.nblocks * total_per_block;
  if (idx >= total) return;
  clio::run::u32 b_idx = idx / total_per_block;
  clio::run::u32 slot = idx - b_idx * total_per_block;
  Block *b = GetBlock(v, b_idx);
  Page *p = &b->pages[slot];
  if (p->modify_min < 0) return;
  if (detail::AtomicOrU32(&p->flags, kPagePutInFlight) & kPagePutInFlight) {
    return;
  }
  FlushPageBase(g_ipc_manager_ptr, v, b_idx, p, slot);
  (void)g_ipc_manager;
}

/** Atomic-clear kPageBusy + kPageGetInFlight on a single (block, slot).
 *  Used by the cache thread after a host-driven prefetch's AsyncGetBlob
 *  completes (the data has landed in the page; we just need to release
 *  the slot lock so the user kernel can use it). */
__global__ void ClearHostPrefetchFlagsKernel(DeviceViewBase v,
                                              clio::run::u32 block_idx,
                                              clio::run::u32 slot_idx) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  Block *b = GetBlock(v, block_idx);
  Page *p = &b->pages[slot_idx];
  AtomicClearBitsU32(&p->flags, kPageBusy | kPageGetInFlight);
}

/** Batched clear: takes a fixed-size list of (block, slot) pairs and
 *  clears kPageBusy+kPageGetInFlight on each. Launched as a single
 *  kernel per drain pass so we don't hit the multi-launch hang
 *  observed when many ClearHostPrefetchFlagsKernel calls are queued
 *  in sequence on the cache-thread stream while the user kernel is
 *  spin-waiting on FaultPage futures. */
inline constexpr clio::run::u32 kClearBatchCap = 256;
__global__ void ClearHostPrefetchFlagsBatchKernel(DeviceViewBase v,
                                                   clio::run::u32 *block_arr,
                                                   clio::run::u32 *slot_arr,
                                                   clio::run::u32 n) {
  clio::run::u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  Block *b = GetBlock(v, block_arr[idx]);
  Page *p = &b->pages[slot_arr[idx]];
  AtomicClearBitsU32(&p->flags, kPageBusy | kPageGetInFlight);
}

/** Drain kernel — non-template. Waits on every page's active_put /
 *  active_get and clears the slots. */
__global__ void DrainKernel(::clio::run::IpcManagerGpuInfo info, DeviceViewBase v) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  clio::run::u32 total_per_block = TotalPagesPerBlock(v);
  clio::run::u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  clio::run::u32 total = v.nblocks * total_per_block;
  if (idx >= total) return;
  clio::run::u32 b_idx = idx / total_per_block;
  clio::run::u32 slot = idx - b_idx * total_per_block;
  Page *p = &GetBlock(v, b_idx)->pages[slot];
  DrainPut(p);
  DrainGet(p);
  (void)g_ipc_manager;
}

/** Warp-parallel reduce: each thread feeds a (value, index) pair.
 *  Returns the lane with the minimum value via shuffles. Result is
 *  broadcast to all lanes. */
CTP_GPU_FUN void WarpReduceMinScore(float &val, int &idx) {
  for (int off = 16; off > 0; off >>= 1) {
    float other_val = __shfl_xor_sync(0xffffffff, val, off);
    int other_idx = __shfl_xor_sync(0xffffffff, idx, off);
    if (other_val < val || (other_val == val && other_idx < idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

CTP_GPU_FUN void WarpReduceMaxScore(float &val, int &idx) {
  for (int off = 16; off > 0; off >>= 1) {
    float other_val = __shfl_xor_sync(0xffffffff, val, off);
    int other_idx = __shfl_xor_sync(0xffffffff, idx, off);
    if (other_val > val || (other_val == val && other_idx < idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

CTP_GPU_FUN clio::run::u64 WarpReduceMinU64(clio::run::u64 val) {
  for (int off = 16; off > 0; off >>= 1) {
    clio::run::u64 hi = __shfl_xor_sync(0xffffffff, (unsigned int)(val >> 32), off);
    clio::run::u64 lo = __shfl_xor_sync(0xffffffff, (unsigned int)(val & 0xffffffffu), off);
    clio::run::u64 other = (hi << 32) | lo;
    if (other < val) val = other;
  }
  return val;
}

CTP_GPU_FUN clio::run::u64 WarpReduceMaxU64(clio::run::u64 val) {
  for (int off = 16; off > 0; off >>= 1) {
    clio::run::u64 hi = __shfl_xor_sync(0xffffffff, (unsigned int)(val >> 32), off);
    clio::run::u64 lo = __shfl_xor_sync(0xffffffff, (unsigned int)(val & 0xffffffffu), off);
    clio::run::u64 other = (hi << 32) | lo;
    if (other > val) val = other;
  }
  return val;
}

/** Scan one tier of a Block's pages[] and report the slot with the
 *  smallest score (skipping kPageBusy, empty slots, and in-flight). */
CTP_GPU_FUN void TierMinScore(Block *b, clio::run::u32 lo, clio::run::u32 hi,
                                clio::run::u32 lane,
                                float &out_min, int &out_slot) {
  float best_v = INFINITY;
  int best_s = -1;
  for (clio::run::u32 s = lo + lane; s < hi; s += 32) {
    Page *p = &b->pages[s];
    if (p->page_idx < 0) continue;
    if (p->flags & (kPageBusy | kPageGetInFlight)) continue;
    if (p->score < best_v) { best_v = p->score; best_s = (int)s; }
  }
  WarpReduceMinScore(best_v, best_s);
  out_min = best_v;
  out_slot = best_s;
}

CTP_GPU_FUN void TierMaxScore(Block *b, clio::run::u32 lo, clio::run::u32 hi,
                                clio::run::u32 lane,
                                float &out_max, int &out_slot) {
  float best_v = -INFINITY;
  int best_s = -1;
  for (clio::run::u32 s = lo + lane; s < hi; s += 32) {
    Page *p = &b->pages[s];
    if (p->page_idx < 0) continue;
    if (p->flags & (kPageBusy | kPageGetInFlight)) continue;
    if (p->score > best_v) { best_v = p->score; best_s = (int)s; }
  }
  WarpReduceMaxScore(best_v, best_s);
  out_max = best_v;
  out_slot = best_s;
}

/** Warp-cooperative byte copy: copy `bytes` from src to dst via 4-byte
 *  loads/stores spread across 32 lanes. Both pointers must be 4-byte
 *  aligned and `bytes` must be a multiple of 4. */
CTP_GPU_FUN void WarpCopy4(void *dst, const void *src, clio::run::u64 bytes,
                             clio::run::u32 lane) {
  clio::run::u64 n = bytes >> 2;
  unsigned int *d = static_cast<unsigned int *>(dst);
  const unsigned int *s = static_cast<const unsigned int *>(src);
  for (clio::run::u64 i = lane; i < n; i += 32) d[i] = s[i];
}

/**
 * Four-phase async cache manager. Launched as <<<nblocks, warpsize>>>
 * where `nblocks` matches the user kernel grid. Each manager block
 * services one user Block.
 *
 * Phase 1 — Rescore:
 *   1a. warp-reduce min/max(lru_clock) across non-empty slots.
 *   1b. p->score = (lru_clock - min_c) / max(1, max_c - min_c);
 *       empty slots -> score = -INF (won't beat anything in min/max).
 *   1c. lane 0 drains rescore_q; matching cached pages get their score
 *       overwritten; non-matching entries forward to prefetch_q (drop if
 *       full).
 *   1d. (computed lazily in Phase 4 via warp-parallel rescan.)
 * Phase 2 — Reorganize:
 *   If min(DRAM) > min(HBM) AND there exists an HBM slot below min(DRAM)
 *   AND a DRAM slot above min(HBM), CAS-acquire kPageBusy on both, drain
 *   their puts, swap their contents via swap_scratch in a 3-step
 *   warp-cooperative copy, swap metadata (page_idx, modify_min/max,
 *   score, lru_clock, active_put, active_get) — tier stays slot-bound.
 *   Release kPageBusy.
 * Phase 3 — Flush:
 *   For every page in both tiers, if dirty and not busy/in-flight,
 *   CAS-acquire kPagePutInFlight and call FlushPageBase. No Wait.
 * Phase 4 — Prefetch:
 *   Pop prefetch_q. For each entry: warp-parallel rescan to find the
 *   current min in each tier. If e.score >= min_hbm evict that HBM
 *   slot and FaultPage. Else if e.score >= min_dram evict that DRAM
 *   slot and FaultPage. Else drop. Bounded to a few pops per tick.
 */
/**
 * Run a single Phase 1→2→3→4 pass on one block. Returns the number of
 * "units of work" performed (rescore drained + flushes kicked + prefetches
 * claimed + swaps). 0 means the block was idle this tick. Used by the
 * persistent kernel to decide whether to back off / self-terminate.
 */
CTP_GPU_FUN clio::run::u32 RunCachePass(::clio::run::gpu::IpcManager *ipc,
                                    DeviceViewBase v, Block *b,
                                    clio::run::u32 lane, clio::run::u32 total) {
  clio::run::u32 work = 0;
  (void)ipc;
  if (v.stats && lane == 0) atomicAdd_system(&v.stats->manager_iters, 1ULL);

  // ── Phase 1a: warp-reduce min/max clock ─────────────────────────
  clio::run::u64 my_min = ~0ULL;
  clio::run::u64 my_max = 0ULL;
  for (clio::run::u32 s = lane; s < total; s += 32) {
    Page *p = &b->pages[s];
    if (p->page_idx < 0) continue;
    if (p->lru_clock < my_min) my_min = p->lru_clock;
    if (p->lru_clock > my_max) my_max = p->lru_clock;
  }
  clio::run::u64 min_c = WarpReduceMinU64(my_min);
  clio::run::u64 max_c = WarpReduceMaxU64(my_max);
  float range = (max_c > min_c) ? (float)(max_c - min_c) : 1.0f;

  // ── Phase 1b: normalize ─────────────────────────────────────────
  for (clio::run::u32 s = lane; s < total; s += 32) {
    Page *p = &b->pages[s];
    if (p->page_idx < 0) { p->score = -INFINITY; continue; }
    p->score = (float)(p->lru_clock - min_c) / range;
  }
  __syncwarp();

  // ── Phase 1c: drain rescore_q ───────────────────────────────────
  // CRITICAL: snapshot tail at start. The user kernel produces into
  // tail concurrently; without snapshotting, a fast producer can keep
  // tail ahead of head forever and this loop never terminates.
  clio::run::u32 phase1c_work = 0;
  if (lane == 0) {
    clio::run::u32 snap_tail = b->rescore_q.tail;
    while (b->rescore_q.head != snap_tail) {
      RescoreEntry e =
          b->rescore_q.slots[b->rescore_q.head & (kRescoreQueueCap - 1)];
      ++b->rescore_q.head;
      ++phase1c_work;
      bool matched = false;
      for (clio::run::u32 s = 0; s < total; ++s) {
        Page *p = &b->pages[s];
        if (p->page_idx == (int32_t)e.page_idx) {
          // Score sign is only used to signal alloc-vs-prefetch on
          // unmatched entries (Phase 4). For already-cached pages we
          // just update the cache score using the magnitude.
          if (!(p->flags & kPageBusy)) {
            p->score = (e.score < 0.0f) ? -e.score : e.score;
          }
          matched = true;
          break;
        }
      }
      if (!matched) {
        clio::run::u32 used = b->prefetch_q.tail - b->prefetch_q.head;
        if (used < kPrefetchQueueCap) {
          b->prefetch_q.slots[b->prefetch_q.tail & (kPrefetchQueueCap - 1)] = e;
          ++b->prefetch_q.tail;
        }
      }
    }
  }
  phase1c_work = __shfl_sync(0xffffffff, phase1c_work, 0);
  work += phase1c_work;
  if (v.stats && lane == 0 && phase1c_work > 0)
    atomicAdd_system(&v.stats->phase1c_drained, (unsigned long long)phase1c_work);
  __syncwarp();

  // ── Phase 2: Reorganize — promote hot DRAM pages to HBM ────────
  // Disabled by default (CLIO_CTE_PHASE2=1 to re-enable). Phase 2's
  // 3-step swap holds kPageBusy on both an HBM and a DRAM slot for
  // ~240 µs; the user kernel's read_range spin-waits on those flags
  // and falls into the cold-miss path, dwarfing whatever benefit
  // promotion provides. For streaming workloads, leave pages where
  // Phase 4 placed them.
#if defined(CLIO_CTE_PHASE2)
  if (v.host_pages_per_block > 0) {
    clio::run::u32 max_swaps_this_tick = v.gpu_pages_per_block;
    for (clio::run::u32 si = 0; si < max_swaps_this_tick; ++si) {
      float min_hbm; int min_hbm_slot;
      TierMinScore(b, 0, v.gpu_pages_per_block, lane, min_hbm, min_hbm_slot);
      float max_dram; int max_dram_slot;
      TierMaxScore(b, v.gpu_pages_per_block, total, lane,
                   max_dram, max_dram_slot);
      int do_swap = 0;
      if (lane == 0 && min_hbm_slot >= 0 && max_dram_slot >= 0 &&
          max_dram > min_hbm) {
        Page *ph = &b->pages[min_hbm_slot];
        Page *pd = &b->pages[max_dram_slot];
        if (TryAcquireBusy(ph)) {
          if (TryAcquireBusy(pd)) {
            DrainPut(ph); DrainPut(pd);
            DrainGet(ph); DrainGet(pd);
            do_swap = 1;
          } else {
            ReleaseBusy(ph);
          }
        }
      }
      do_swap = __shfl_sync(0xffffffff, do_swap, 0);
      if (!do_swap) break;
      Page *ph = &b->pages[min_hbm_slot];
      Page *pd = &b->pages[max_dram_slot];
      WarpCopy4(b->swap_scratch, ph->device_ptr, v.page_size_bytes, lane);
      __syncwarp();
      WarpCopy4(ph->device_ptr, pd->device_ptr, v.page_size_bytes, lane);
      __syncwarp();
      WarpCopy4(pd->device_ptr, b->swap_scratch, v.page_size_bytes, lane);
      __syncwarp();
      if (lane == 0) {
        int32_t pi = ph->page_idx;   ph->page_idx   = pd->page_idx;   pd->page_idx   = pi;
        int32_t mn = ph->modify_min; ph->modify_min = pd->modify_min; pd->modify_min = mn;
        int32_t mx = ph->modify_max; ph->modify_max = pd->modify_max; pd->modify_max = mx;
        clio::run::u64 lc = ph->lru_clock; ph->lru_clock  = pd->lru_clock;  pd->lru_clock  = lc;
        float sc = ph->score;        ph->score      = pd->score;      pd->score      = sc;
        __threadfence();
        ReleaseBusy(ph);
        ReleaseBusy(pd);
      }
      ++work;
      if (v.stats && lane == 0) atomicAdd_system(&v.stats->phase2_swaps, 1ULL);
      __syncwarp();
    }
  }
#endif  // CLIO_CTE_PHASE2

  // ── Phase 3: Flush dirty unlocked pages (bounded) ──────────────
  // Bound work per tick so we don't saturate the gpu2cpu queue
  // while the user kernel is also pushing PutBlob/GetBlob entries
  // (the queue is system-scoped 64-bit-atomic so producer Sends
  // contend on the tail). vec.flush() handles the rest synchronously.
  // Phase 3: flush dirty unlocked pages, capped to keep the gpu2cpu
  // queue from saturating (the manager's Sends contend with the user
  // kernel's). Cap = host_pages_per_block so we can flush the full
  // DRAM tier in one tick if needed.
  clio::run::u32 phase3_work = 0;
  if (lane == 0) {
    const clio::run::u32 max_flush_per_tick = v.host_pages_per_block > 0
                                            ? v.host_pages_per_block
                                            : v.gpu_pages_per_block;
    clio::run::u32 s = b->flush_cursor;
    for (clio::run::u32 i = 0; i < total && phase3_work < max_flush_per_tick; ++i) {
      Page *p = &b->pages[s];
      if (p->page_idx >= 0 && p->modify_min >= 0 &&
          !(p->flags & kPageBusy) &&
          !(AtomicOrU32(&p->flags, kPagePutInFlight) & kPagePutInFlight)) {
        FlushPageBase(ipc, v, blockIdx.x, p, s);
        ++phase3_work;
      }
      s = (s + 1u) % total;
    }
    b->flush_cursor = s;
  }
  phase3_work = __shfl_sync(0xffffffff, phase3_work, 0);
  work += phase3_work;
  if (v.stats && lane == 0 && phase3_work > 0)
    atomicAdd_system(&v.stats->phase3_flushes, (unsigned long long)phase3_work);
  __syncwarp();

  // ── Phase 4: Prefetch into DRAM tier ONLY ──────────────────────
  // GPU-side Send from the manager kernel races with user kernel's
  // Sends on the gpu2cpu queue tail (system-scoped 64-bit atomic
  // contention crashes the GPU). So Phase 4 is host-driven:
  //   1. Kernel picks a free/lowest-score DRAM slot (NOT HBM —
  //      prefetching into HBM would steal a slot the user kernel
  //      may be actively using). User kernel can read DRAM pages
  //      directly via Resolve's two-tier scan; they're slower but
  //      free up HBM for hot pages.
  //   2. Kernel CAS-acquires kPageBusy on the DRAM victim, sets
  //      page_idx + kPageGetInFlight, pushes a directive into the
  //      pinned-host host_prefetch_q. Does NOT release kPageBusy.
  //   3. Cache thread (post-kernel) issues AsyncGetBlob via the
  //      CPU client (separate cpu2cpu path, no system-atomic
  //      contention), waits, then launches a tiny clear-flags
  //      kernel on the cache-thread's non-blocking stream to
  //      release kPageBusy + kPageGetInFlight.
  // Phase 4: device-side prefetch claim. Kernel acquires kPageBusy on a
  // free/low-score DRAM slot, sets page_idx + kPageGetInFlight on the
  // device. Does NOT write to pinned host memory (that empirically
  // hangs the user kernel — see git log for the iter-9 debug trace).
  // The cache thread will later cudaMemcpyAsync the meta region to a
  // host scratch, scan for kPageGetInFlight slots, issue AsyncGetBlob
  // per slot, and launch a clear-flags kernel to release the slot
  // when each fault completes.
  // Phase 4: DRAM-tier prefetch. The kernel acquires kPageBusy on a
  // DRAM slot and sets kPageGetInFlight; the cache thread (CPU)
  // releases both flags after the AsyncGetBlob completes via
  // DrainHostPrefetchQueue. Bounded per tick to avoid claiming all
  // DRAM slots and starving the user kernel.
  if (v.host_pages_per_block > 0 && lane == 0) {
    // Prefetch depth scales with DRAM tier capacity so the manager can
    // populate the entire DRAM tier in a single tick when there's a
    // backlog. Bounded by kPrefetchQueueCap (32) naturally.
    const int kMaxPrefetchPerTick =
        static_cast<int>(v.host_pages_per_block);
    for (int i = 0; i < kMaxPrefetchPerTick; ++i) {
      clio::run::u32 pq_used = b->prefetch_q.tail - b->prefetch_q.head;
      if (pq_used == 0) break;
      RescoreEntry e =
          b->prefetch_q.slots[b->prefetch_q.head & (kPrefetchQueueCap - 1)];
      ++b->prefetch_q.head;
      if (v.stats) atomicAdd_system(&v.stats->phase4_pops, 1ULL);
      // Already cached? Skip.
      bool already_cached = false;
      for (clio::run::u32 s = 0; s < total; ++s) {
        if (b->pages[s].page_idx == (int32_t)e.page_idx) {
          already_cached = true; break;
        }
      }
      if (already_cached) {
        if (v.stats) atomicAdd_system(&v.stats->phase4_skip_cached, 1ULL);
        continue;
      }
      // Target HBM first when EvictSlot is disabled (no race against
      // Resolve's `p->flags = 0` after EvictSlot). Falls back to DRAM
      // if no HBM slot is acquirable. When EvictSlot is on, we have
      // to limit to DRAM to avoid the slot-clobber race.
      clio::run::u32 tier_lo = v.allow_cold_miss_fault
                            ? v.gpu_pages_per_block
                            : 0;
      int target = -1;
      float lo_score = INFINITY;
      for (clio::run::u32 s = tier_lo; s < total; ++s) {
        Page *p = &b->pages[s];
        if (p->flags & kPageBusy) continue;
        if (p->page_idx < 0) {
          target = (int)s;
          break;
        }
        if (p->score < lo_score) {
          lo_score = p->score;
          target = (int)s;
        }
      }
      if (target < 0) {
        if (v.stats) atomicAdd_system(&v.stats->phase4_skip_nofree, 1ULL);
        continue;
      }
      Page *p = &b->pages[target];
      if (!TryAcquireBusy(p)) {
        if (v.stats) atomicAdd_system(&v.stats->phase4_skip_nofree, 1ULL);
        continue;
      }
      DrainPut(p);
      DrainGet(p);
      p->page_idx = (int32_t)e.page_idx;
      p->modify_min = -1;
      p->modify_max = -1;
      p->lru_clock = clock64();
      // Negative score = "alloc-only" (write hint): set up the slot
      // but skip the AsyncGetBlob (no data exists yet). Release
      // kPageBusy so the user kernel can claim the slot immediately.
      // Positive score = read prefetch: leave kPageBusy + set
      // kPageGetInFlight; cache thread issues AsyncGetBlob and clears.
      if (e.score < 0.0f) {
        p->score = -e.score;            // store positive score in slot
        // Fence so the user kernel (concurrent on a different SM)
        // sees the page_idx / modify_min/max stores before the
        // ReleaseBusy clears kPageBusy.
        __threadfence();
        ReleaseBusy(p);
      } else {
        p->score = e.score;
        __threadfence();
        AtomicOrU32(&p->flags, kPageGetInFlight);
      }
      if (v.stats) atomicAdd_system(&v.stats->phase4_prefetches, 1ULL);
      ++work;
    }
  }
  if (v.stats && lane == 0 && work > 0)
    atomicAdd_system(&v.stats->manager_work_iters, 1ULL);
  (void)g_ipc_manager;
  return work;
}

/**
 * Persistent cache-manager kernel. Launched ONCE per Vector lifetime on
 * a non-blocking stream. Each block loops, running RunCachePass; when a
 * pass finds no work the block backs off exponentially (base→max
 * nanoseconds via `__nanosleep`); after a contiguous idle period of
 * `idle_exit_ns`, the block self-terminates. The kernel as a whole
 * exits when all blocks have terminated OR when the host sets the
 * `stop_flag` (pinned host u32, polled via volatile read each iter).
 *
 * The host does NOT relaunch this kernel each tick — a single launch
 * powers an entire Vector's lifetime, plus optional resurrection on
 * Vector dtor or if the host detects new rescore_q activity post-exit.
 */
__global__ void PersistentCacheManagerKernel(
    ::clio::run::IpcManagerGpuInfo info,
    DeviceViewBase v,
    clio::run::u32 *stop_flag_pinned,
    clio::run::u32 base_sleep_ns,
    clio::run::u32 max_sleep_ns,
    clio::run::u64 idle_exit_ns) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (blockIdx.x >= v.nblocks) return;
  Block *b = GetBlock(v, blockIdx.x);
  clio::run::u32 total = TotalPagesPerBlock(v);
  clio::run::u32 lane = threadIdx.x & 31;
  clio::run::u32 cur_sleep_ns = base_sleep_ns;
  clio::run::u64 idle_ns_total = 0;
  volatile clio::run::u32 *stop_flag = stop_flag_pinned;

  while (true) {
    clio::run::u32 stop = 0;
    if (lane == 0) stop = *stop_flag;
    stop = __shfl_sync(0xffffffff, stop, 0);
    if (stop) break;

    clio::run::u32 work = RunCachePass(g_ipc_manager_ptr, v, b, lane, total);

    if (work > 0) {
      cur_sleep_ns = base_sleep_ns;
      idle_ns_total = 0;
    } else {
      idle_ns_total += cur_sleep_ns;
      if (idle_ns_total >= idle_exit_ns) break;
      clio::run::u32 doubled = cur_sleep_ns * 2u;
      cur_sleep_ns = (doubled > max_sleep_ns) ? max_sleep_ns : doubled;
    }
    __nanosleep(cur_sleep_ns);
    __syncwarp();
  }
  (void)g_ipc_manager;
}

/**
 * Original one-shot cache-manager kernel, kept for FlushAllSync's drain
 * path (LegacyFlushKernel covers that already) and as a fallback if the
 * persistent kernel can't run (compute_cap < sm_70). New code should
 * launch PersistentCacheManagerKernel.
 */
__global__ void CacheManagerKernel(::clio::run::IpcManagerGpuInfo info,
                                    DeviceViewBase v) {
  CLIO_GPU_INIT(info, /*ipc_ptr=*/nullptr);
  if (blockIdx.x >= v.nblocks) return;
  Block *b = GetBlock(v, blockIdx.x);
  clio::run::u32 total = TotalPagesPerBlock(v);
  clio::run::u32 lane = threadIdx.x & 31;
  (void)RunCachePass(g_ipc_manager_ptr, v, b, lane, total);
  (void)g_ipc_manager;
}

/** Compute the byte stride between adjacent Block structs in the meta
 *  backend, accounting for the flexible Page array and total tier
 *  count. Aligned to 16. */
inline clio::run::u32 BlockStrideBytes(clio::run::u32 total_pages_per_block) {
  size_t raw = sizeof(Block) + sizeof(Page) * total_pages_per_block;
  return static_cast<clio::run::u32>((raw + 15) & ~static_cast<size_t>(15));
}

/** sizeof(TaskT), 16-byte aligned. The Task is self-contained (its fut_ holds
 *  the completion record), so there is no co-located gpu::FutureShm. */
template <typename TaskT>
inline clio::run::u32 TaskSlotStride() {
  size_t raw = sizeof(TaskT);
  return static_cast<clio::run::u32>((raw + 15) & ~static_cast<size_t>(15));
}

}  // namespace detail

template <typename T>
struct Vector<T>::Impl {
  clio::run::u32 gpu_id = 0;
  clio::run::u32 cache_period_us = 50;
  clio::run::u32 manager_threads_per_block = 32;
  CacheMode mode = CacheMode::kLegacy;
  ctp::ipc::AllocatorId pages_alloc_id;
  ctp::ipc::AllocatorId host_pages_alloc_id;
  ctp::ipc::AllocatorId meta_alloc_id;
  ctp::ipc::AllocatorId swap_alloc_id;
  ctp::ipc::AllocatorId put_alloc_id;
  ctp::ipc::AllocatorId get_alloc_id;
  char *pages_base = nullptr;
  char *host_pages_base = nullptr;
  char *meta_base = nullptr;
  char *swap_base = nullptr;
  char *put_base = nullptr;
  char *get_base = nullptr;
  /** Pinned host scratch (block_stride * nblocks bytes). The cache
   *  thread cudaMemcpyAsync's the device meta_base here every tick so
   *  it can scan for kPageGetInFlight slots and issue AsyncGetBlobs
   *  on the host side (the manager kernel only marks slots and
   *  doesn't issue Sends or pinned-host writes). */
  char *host_meta_scratch = nullptr;
  clio::run::u32 block_stride_cached = 0;
  /** Pinned-host index arrays sized kClearBatchCap; cache thread fills
   *  them with (block, slot) pairs of slots whose prefetch completed,
   *  then launches ClearHostPrefetchFlagsBatchKernel once per tick to
   *  release the locks. Launching one batched kernel avoids the
   *  multi-launch hang observed with per-slot launches. */
  clio::run::u32 *clear_block_arr = nullptr;
  clio::run::u32 *clear_slot_arr = nullptr;
  /** Pinned-host instrumentation counters. The user kernel and the
   *  manager kernel atomicAdd_system into these so the host can read
   *  a coherent snapshot via Vector::GetStatsSnapshot(). */
  ::clio::cte::gpu_vector::VectorStats *stats = nullptr;
  /** Number of host-driven AsyncGetBlobs currently in flight. Cache
   *  thread bumps before dispatching, decrements after the .Wait()
   *  returns. Vector dtor waits for this to reach 0 before freeing
   *  backends — without this gate, the runtime's cudaMemcpyAsync may
   *  reference a freed device_ptr and trip CUDA error 700. */
  std::atomic<clio::run::u32> async_inflight{0};
  /** Pinned-host u32 read by the persistent CacheManagerKernel each
   *  iteration. Host sets to 1 to signal the kernel to exit. The
   *  kernel can also self-terminate after the idle window elapses
   *  (in which case the host detects via cudaStreamQuery). */
  clio::run::u32 *kernel_stop_flag = nullptr;
  /** CUDA stream the persistent kernel runs on. Non-blocking so it
   *  doesn't depend on the main thread's default-stream work. */
  cudaStream_t persistent_stream = nullptr;
  /** Separate non-blocking stream for DrainHostPrefetchQueue's
   *  cudaMemcpyAsync + clear-flags kernel. Distinct from
   *  persistent_stream so meta snapshots don't queue behind the
   *  persistent kernel (which never returns until idle-exit). */
  cudaStream_t drain_stream = nullptr;
  /** True iff PersistentCacheManagerKernel is currently in flight on
   *  persistent_stream. Cache thread (re)launches when this is false
   *  and there is work that needs handling. */
  bool persistent_kernel_running = false;
  std::string tag_name;
  clio::run::PoolId cte_pool_id = clio::run::PoolId(0, 0);
  std::thread cache_thread;
  std::atomic<bool> cache_thread_run{false};
  clio::run::u32 nblocks_cached = 0;
  clio::run::u64 page_size_cached = 0;
  clio::run::u32 gpu_ppb_cached = 0;
  clio::run::u32 host_ppb_cached = 0;
};

template <typename T>
inline Vector<T>::Vector(const std::string &tag_name, clio::run::u32 nblocks,
                          clio::run::u32 gpu_id, clio::run::u32 gpu_pages_per_block,
                          clio::run::u32 host_pages_per_block,
                          clio::run::u64 page_size_bytes,
                          clio::run::u32 cache_period_us,
                          CacheMode mode,
                          clio::run::u32 manager_threads_per_block,
                          bool allow_cold_miss_fault) {
#if !CTP_IS_DEVICE_PASS
  // Body gated for the host pass only.
  if (nblocks == 0 || gpu_pages_per_block == 0 || page_size_bytes == 0) {
    throw std::invalid_argument(
        "clio::cte::gpu_vector::Vector: nblocks/gpu_pages_per_block/page_size "
        "must all be > 0");
  }
  if (page_size_bytes % sizeof(T) != 0) {
    throw std::invalid_argument(
        "clio::cte::gpu_vector::Vector: page_size_bytes must be a multiple "
        "of sizeof(T)");
  }
  if (manager_threads_per_block == 0 ||
      (manager_threads_per_block % 32) != 0) {
    throw std::invalid_argument(
        "clio::cte::gpu_vector::Vector: manager_threads_per_block must be a "
        "positive multiple of 32");
  }
  // Auto-promote mode when DRAM tier is requested.
  if (host_pages_per_block > 0 && mode == CacheMode::kLegacy) {
    mode = CacheMode::kAsync;
  }
  impl_ = std::make_unique<Impl>();
  impl_->cache_period_us = cache_period_us;
  impl_->manager_threads_per_block = manager_threads_per_block;
  impl_->mode = mode;
  impl_->gpu_id = gpu_id;

  auto *cpu_ipc = CLIO_CPU_IPC;

  clio::run::u32 total_ppb = gpu_pages_per_block + host_pages_per_block;

  // 1. Allocate the HBM page backend.
  clio::run::u64 hbm_bytes = static_cast<clio::run::u64>(nblocks) * gpu_pages_per_block *
                       page_size_bytes;
  impl_->pages_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem, hbm_bytes,
      &impl_->pages_base);
  if (impl_->pages_alloc_id.IsNull()) {
    throw std::runtime_error("gpu_vector: HBM pages backend allocation failed");
  }

  // 1b. Allocate the DRAM (pinned host) page backend if requested.
  if (host_pages_per_block > 0) {
    clio::run::u64 dram_bytes = static_cast<clio::run::u64>(nblocks) *
                          host_pages_per_block * page_size_bytes;
    impl_->host_pages_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost, dram_bytes,
        &impl_->host_pages_base);
    if (impl_->host_pages_alloc_id.IsNull()) {
      throw std::runtime_error(
          "gpu_vector: DRAM (pinned-host) pages backend allocation failed");
    }
  } else {
    impl_->host_pages_alloc_id = ctp::ipc::AllocatorId::GetNull();
    impl_->host_pages_base = nullptr;
  }

  // 1c. Allocate the swap scratch (one HBM page per block) for in-kernel
  //     reorganize. Only needed when DRAM tier is active.
  if (host_pages_per_block > 0) {
    clio::run::u64 swap_bytes =
        static_cast<clio::run::u64>(nblocks) * page_size_bytes;
    impl_->swap_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
        gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem, swap_bytes,
        &impl_->swap_base);
    if (impl_->swap_alloc_id.IsNull()) {
      throw std::runtime_error(
          "gpu_vector: swap scratch backend allocation failed");
    }
  } else {
    impl_->swap_alloc_id = ctp::ipc::AllocatorId::GetNull();
    impl_->swap_base = nullptr;
  }

  // 2. Meta backend (per-block Block struct with full Page array).
  clio::run::u32 block_stride = detail::BlockStrideBytes(total_ppb);
  clio::run::u64 meta_bytes = static_cast<clio::run::u64>(block_stride) * nblocks;
  impl_->meta_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kDeviceMem, meta_bytes,
      &impl_->meta_base);
  if (impl_->meta_alloc_id.IsNull()) {
    throw std::runtime_error("gpu_vector: meta_backend allocation failed");
  }

  // 3. Task pools cover the FULL tier slot count.
  clio::run::u32 put_stride =
      detail::TaskSlotStride<clio::cte::core::PodPutBlobTask>();
  clio::run::u32 get_stride =
      detail::TaskSlotStride<clio::cte::core::PodGetBlobTask>();
  clio::run::u64 task_count = static_cast<clio::run::u64>(nblocks) * total_ppb;
  impl_->put_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost,
      task_count * put_stride, &impl_->put_base);
  if (impl_->put_alloc_id.IsNull()) {
    throw std::runtime_error("gpu_vector: put_pool allocation failed");
  }
  impl_->get_alloc_id = cpu_ipc->AllocateAndRegisterGpuBackend(
      gpu_id, clio::run::gpu::IpcManager::MemKind::kPinnedHost,
      task_count * get_stride, &impl_->get_base);
  if (impl_->get_alloc_id.IsNull()) {
    throw std::runtime_error("gpu_vector: get_pool allocation failed");
  }
  std::memset(impl_->put_base, 0, task_count * put_stride);
  std::memset(impl_->get_base, 0, task_count * get_stride);

  // 3b. Allocate the host meta scratch. Pinned host memory so
  //     cudaMemcpyAsync can use DMA. The cache thread copies the
  //     device meta_base here every tick. Reads of pinned host from
  //     device (the DMA copy direction) are safe — the hang only
  //     happens when GPU kernels WRITE to pinned host concurrently
  //     with spin-wait kernels.
  if (host_pages_per_block > 0) {
    clio::run::u64 meta_scratch_bytes = static_cast<clio::run::u64>(block_stride) * nblocks;
    impl_->host_meta_scratch = ctp::GpuApi::MallocHost<char>(meta_scratch_bytes);
    if (!impl_->host_meta_scratch) {
      throw std::runtime_error(
          "gpu_vector: host_meta_scratch allocation failed");
    }
    std::memset(impl_->host_meta_scratch, 0, meta_scratch_bytes);
    impl_->block_stride_cached = block_stride;
    impl_->clear_block_arr =
        ctp::GpuApi::MallocHost<clio::run::u32>(
            sizeof(clio::run::u32) * detail::kClearBatchCap);
    impl_->clear_slot_arr =
        ctp::GpuApi::MallocHost<clio::run::u32>(
            sizeof(clio::run::u32) * detail::kClearBatchCap);
    if (!impl_->clear_block_arr || !impl_->clear_slot_arr) {
      throw std::runtime_error(
          "gpu_vector: clear-batch index array allocation failed");
    }
    // Pinned-host stop flag the persistent kernel polls each iteration.
    impl_->kernel_stop_flag = ctp::GpuApi::MallocHost<clio::run::u32>(1);
    if (!impl_->kernel_stop_flag) {
      throw std::runtime_error(
          "gpu_vector: kernel_stop_flag allocation failed");
    }
    *impl_->kernel_stop_flag = 0;
  }

  // 4. Create the CTE tag.
  clio::cte::core::Client cte_client(clio::cte::core::kCtePoolId);
  auto tag_fut = cte_client.AsyncGetOrCreateTag(tag_name);
  tag_fut.Wait();
  if (tag_fut->GetReturnCode() != 0) {
    throw std::runtime_error(
        "gpu_vector: GetOrCreateTag failed for tag '" + tag_name + "'");
  }
  view_.base.tag_id = tag_fut->tag_id_;
  impl_->tag_name = tag_name;
  impl_->cte_pool_id = clio::cte::core::kCtePoolId;

  // 5. Populate DeviceView.
  view_.base.blocks = reinterpret_cast<Block *>(impl_->meta_base);
  view_.base.block_stride_bytes = block_stride;
  view_.base.pages_base = impl_->pages_base;
  view_.base.host_pages_base = impl_->host_pages_base;
  view_.base.put_pool_base = impl_->put_base;
  view_.base.get_pool_base = impl_->get_base;
  view_.base.put_slot_stride = put_stride;
  view_.base.get_slot_stride = get_stride;
  view_.base.pages_alloc_id = impl_->pages_alloc_id;
  view_.base.host_pages_alloc_id = impl_->host_pages_alloc_id;
  view_.base.put_pool_alloc_id = impl_->put_alloc_id;
  view_.base.get_pool_alloc_id = impl_->get_alloc_id;
  view_.base.nblocks = nblocks;
  view_.base.gpu_pages_per_block = gpu_pages_per_block;
  view_.base.host_pages_per_block = host_pages_per_block;
  view_.base.page_size_bytes = page_size_bytes;
  view_.base.allow_cold_miss_fault = allow_cold_miss_fault;
  // Pinned-host instrumentation counters. Mapped so the device can
  // atomicAdd_system into them and the host can read them directly
  // (no cudaMemcpy round-trip). cudaHostAllocMapped + UVA means the
  // same pointer works on both sides.
  {
    void *p = nullptr;
    cudaHostAlloc(&p, sizeof(::clio::cte::gpu_vector::VectorStats),
                  cudaHostAllocMapped | cudaHostAllocPortable);
    impl_->stats = static_cast<::clio::cte::gpu_vector::VectorStats *>(p);
    if (impl_->stats) {
      std::memset(impl_->stats, 0,
                  sizeof(::clio::cte::gpu_vector::VectorStats));
    }
  }
  view_.base.stats = impl_->stats;
  view_.page_capacity_t = page_size_bytes / sizeof(T);
  impl_->nblocks_cached = nblocks;
  impl_->page_size_cached = page_size_bytes;
  impl_->gpu_ppb_cached = gpu_pages_per_block;
  impl_->host_ppb_cached = host_pages_per_block;

  // 6. Placement-new every PutBlob/GetBlob slot.
  for (clio::run::u32 b = 0; b < nblocks; ++b) {
    for (clio::run::u32 s = 0; s < total_ppb; ++s) {
      clio::run::u64 slot_idx =
          static_cast<clio::run::u64>(b) * total_ppb + s;
      char *put_addr = impl_->put_base + slot_idx * put_stride;
      char *get_addr = impl_->get_base + slot_idx * get_stride;
      std::string blob_name = tag_name + "_b" + std::to_string(b);
      auto put_task = new (put_addr) clio::cte::core::PodPutBlobTask(
          clio::run::CreateTaskId(), impl_->cte_pool_id,
          clio::run::PoolQuery::ToLocalCpu(), view_.base.tag_id,
          blob_name.c_str(), /*offset=*/0, /*size=*/0,
          ctp::ipc::ShmPtr<>::GetNull(), /*score=*/-1.0f,
          clio::cte::core::Context(), /*flags=*/0);
      put_task->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(*put_task));

      auto get_task = new (get_addr) clio::cte::core::PodGetBlobTask(
          clio::run::CreateTaskId(), impl_->cte_pool_id,
          clio::run::PoolQuery::ToLocalCpu(), view_.base.tag_id,
          blob_name.c_str(), /*offset=*/0, /*size=*/0,
          /*flags=*/0, ctp::ipc::ShmPtr<>::GetNull());
      get_task->fut_.task_size_ = static_cast<clio::run::u32>(sizeof(*get_task));
    }
  }

  // 7. Initialize the meta backend on-device. One block per Block; threads
  //    walk slots in stride.
  clio::run::u32 init_threads = (total_ppb < 32u) ? total_ppb : 32u;
  if (init_threads == 0) init_threads = 1;
  detail::InitMetaKernel<<<nblocks, init_threads>>>(
      view_.base, static_cast<char *>(impl_->pages_base),
      static_cast<char *>(impl_->host_pages_base),
      static_cast<char *>(impl_->swap_base));
  ctp::GpuApi::Synchronize();

  // 7b. Pre-launch the manager kernel ONCE from the main thread so the
  //     CUDA module is fully resident on the device before the cache
  //     thread (a separate host thread) tries to launch it concurrently
  //     with active user kernels. Without this warmup, the first
  //     cross-thread launch can deadlock against a long-running user
  //     kernel that uses spin-wait primitives (gpu::Future::Wait).
  //     Also pre-launches the host-prefetch flag-clear batch kernel so
  //     IT doesn't hit the same first-launch deadlock from the cache
  //     thread.
  if (mode == CacheMode::kAsync) {
    auto *gpu_ipc_mgr0 = CLIO_CPU_IPC->GetGpuIpcManager();
    clio::run::IpcManagerGpuInfo info0 = gpu_ipc_mgr0->GetGpuInfo(gpu_id);
    detail::CacheManagerKernel<<<nblocks, manager_threads_per_block>>>(
        info0, view_.base);
    detail::ClearHostPrefetchFlagsBatchKernel<<<1, 1>>>(
        view_.base, impl_->clear_block_arr, impl_->clear_slot_arr, 0);
    ctp::GpuApi::Synchronize();
  }

  // 8. Spawn the cache-management thread. It periodically launches a
  // short-lived CacheManagerKernel that does rescore / reorganize /
  // flush / prefetch in one pass, then exits. The thread relaunches
  // every cache_period_us. This is the only pattern that actually
  // schedules concurrently with user kernels on consumer GPUs —
  // a true "persistent" __nanosleep-loop kernel blocks the scheduler.
  //
  // Cache management is the Vector's responsibility, NOT the user's:
  // the user never calls flush, the user kernel's cudaStreamSync only
  // waits for its own stream, and the cache thread runs on its own
  // non-blocking stream.
  if (mode != CacheMode::kAsync || cache_period_us == 0) {
    return;
  }
  impl_->cache_thread_run.store(true);
  Vector<T> *self = this;
  impl_->cache_thread = std::thread([self]() {
    auto *gpu_ipc_mgr = CLIO_CPU_IPC->GetGpuIpcManager();
    clio::run::IpcManagerGpuInfo info =
        gpu_ipc_mgr->GetGpuInfo(self->impl_->gpu_id);
    cudaFree(0);
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    self->impl_->persistent_stream = stream;
    cudaStream_t drain_stream = nullptr;
    cudaStreamCreateWithFlags(&drain_stream, cudaStreamNonBlocking);
    self->impl_->drain_stream = drain_stream;
    while (self->impl_->cache_thread_run.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(self->impl_->cache_period_us));
      if (!self->impl_->cache_thread_run.load()) break;
      const auto &v = self->view_.base;
      // One-shot pass: launches, runs all four phases, exits. The
      // kernel returns the SM scheduler queue slot so user kernels
      // on other streams can run between ticks.
      detail::CacheManagerKernel<<<v.nblocks,
                                    self->impl_->manager_threads_per_block,
                                    0, stream>>>(info, v);
      cudaStreamSynchronize(stream);
      if (self->impl_->host_meta_scratch != nullptr) {
        self->DrainHostPrefetchQueue(drain_stream);
      }
    }
    if (drain_stream) cudaStreamSynchronize(drain_stream);
    if (stream) cudaStreamDestroy(stream);
    if (drain_stream) cudaStreamDestroy(drain_stream);
    self->impl_->persistent_stream = nullptr;
    self->impl_->drain_stream = nullptr;
  });
#else
  (void)tag_name; (void)nblocks; (void)gpu_id;
  (void)gpu_pages_per_block; (void)host_pages_per_block;
  (void)page_size_bytes; (void)cache_period_us; (void)mode;
  (void)manager_threads_per_block;
#endif  // !CTP_IS_DEVICE_PASS
}

template <typename T>
inline Vector<T>::~Vector() {
#if !CTP_IS_DEVICE_PASS
  if (!impl_) return;
  // Stop the cache thread. It owns persistent_stream / drain_stream,
  // syncs them, and destroys them as it exits.
  impl_->cache_thread_run.store(false, std::memory_order_release);
  if (impl_->cache_thread.joinable()) impl_->cache_thread.join();
  // Wait for any host-driven AsyncGetBlobs to settle.
  {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::seconds(5);
    while (impl_->async_inflight.load(std::memory_order_acquire) > 0 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  auto *cpu_ipc = CLIO_CPU_IPC;
  if (impl_->pages_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->pages_alloc_id);
  if (impl_->host_pages_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->host_pages_alloc_id);
  if (impl_->meta_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->meta_alloc_id);
  if (impl_->swap_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->swap_alloc_id);
  if (impl_->put_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->put_alloc_id);
  if (impl_->get_base)
    cpu_ipc->FreeGpuBackend(impl_->gpu_id, impl_->get_alloc_id);
  if (impl_->host_meta_scratch)
    ctp::GpuApi::FreeHost(impl_->host_meta_scratch);
  if (impl_->clear_block_arr)
    ctp::GpuApi::FreeHost(impl_->clear_block_arr);
  if (impl_->clear_slot_arr)
    ctp::GpuApi::FreeHost(impl_->clear_slot_arr);
  if (impl_->kernel_stop_flag)
    ctp::GpuApi::FreeHost(impl_->kernel_stop_flag);
  if (impl_->stats)
    cudaFreeHost(impl_->stats);
#endif  // !CTP_IS_DEVICE_PASS
}

template <typename T>
inline void Vector<T>::DrainHostPrefetchQueue(void *cuda_stream) {
#if !CTP_IS_DEVICE_PASS
  if (!impl_ || !impl_->host_meta_scratch) return;
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  clio::cte::core::Client cte_client(impl_->cte_pool_id);
  const char *dbg = std::getenv("GPU_VECTOR_DEBUG_CACHE");
  clio::run::u32 nb = impl_->nblocks_cached;
  clio::run::u32 gpu_ppb = impl_->gpu_ppb_cached;
  clio::run::u32 host_ppb = impl_->host_ppb_cached;
  clio::run::u64 psz = impl_->page_size_cached;
  clio::run::u32 block_stride = impl_->block_stride_cached;
  clio::run::u32 total_ppb = gpu_ppb + host_ppb;
  // Snapshot the device meta region into the host scratch via the
  // dedicated cache-thread stream.
  clio::run::u64 meta_bytes = static_cast<clio::run::u64>(block_stride) * nb;
  cudaMemcpyAsync(impl_->host_meta_scratch, view_.base.blocks,
                  meta_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  clio::run::u32 n_to_clear = 0;
  // Two-pass pipeline so prefetches actually overlap:
  //   Pass 1: scan meta, dispatch AsyncGetBlob for every slot with
  //           kPageGetInFlight set. Collect futures + clear info.
  //   Pass 2: Wait on each future (they run in parallel through the
  //           runtime). Bump/decrement async_inflight bookkeeping.
  //   Pass 3 (after the loop): one batched clear-flags kernel.
  std::vector<clio::run::Future<clio::cte::core::PodGetBlobTask>> futs;
  futs.reserve(static_cast<size_t>(detail::kClearBatchCap));
  for (clio::run::u32 b = 0; b < nb; ++b) {
    Block *bp = reinterpret_cast<Block *>(impl_->host_meta_scratch +
                  static_cast<clio::run::u64>(b) * block_stride);
    for (clio::run::u32 s = 0; s < total_ppb; ++s) {
      Page *p = &bp->pages[s];
      if (!(p->flags & kPageGetInFlight)) continue;
      if (p->page_idx < 0) continue;
      if (n_to_clear >= detail::kClearBatchCap) break;
      char *device_ptr = nullptr;
      if (s < gpu_ppb) {
        device_ptr = impl_->pages_base +
            (static_cast<clio::run::u64>(b) * gpu_ppb + s) * psz;
      } else {
        clio::run::u32 host_slot = s - gpu_ppb;
        device_ptr = impl_->host_pages_base +
            (static_cast<clio::run::u64>(b) * host_ppb + host_slot) * psz;
      }
      std::string blob_name = impl_->tag_name + "_b" + std::to_string(b) +
                               "_pi" + std::to_string(p->page_idx);
      ctp::ipc::ShmPtr<> blob_data;
      blob_data.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
      blob_data.off_ = reinterpret_cast<clio::run::u64>(device_ptr);
      if (dbg && dbg[0] == '1') {
        std::fprintf(stderr, "[DRAIN] block %u slot %u page %d dispatch\n",
                     b, s, p->page_idx);
      }
      impl_->async_inflight.fetch_add(1, std::memory_order_acq_rel);
      futs.push_back(cte_client.AsyncPodGetBlob(view_.base.tag_id, blob_name,
                                                 /*offset=*/0, psz,
                                                 /*flags=*/0, blob_data,
                                                 clio::run::PoolQuery::ToLocalCpu()));
      impl_->clear_block_arr[n_to_clear] = b;
      impl_->clear_slot_arr[n_to_clear] = s;
      ++n_to_clear;
    }
  }
  // Now wait — futures complete in parallel through the runtime.
  for (auto &fut : futs) {
    fut.Wait();
    impl_->async_inflight.fetch_sub(1, std::memory_order_acq_rel);
  }
  if (n_to_clear > 0) {
    clio::run::u32 threads = (n_to_clear < 32) ? n_to_clear : 32;
    clio::run::u32 blocks = (n_to_clear + threads - 1) / threads;
    detail::ClearHostPrefetchFlagsBatchKernel<<<blocks, threads, 0, stream>>>(
        view_.base, impl_->clear_block_arr, impl_->clear_slot_arr,
        n_to_clear);
  }
  cudaStreamSynchronize(stream);
#endif
}

template <typename T>
inline void Vector<T>::FlushAllSync() {
#if !CTP_IS_DEVICE_PASS
  // No-op. The persistent manager kernel flushes dirty pages
  // continuously. The Vector is automatically coherent — the user
  // never needs to flush. This method is retained as a no-op so
  // legacy callers compile, but it does NOT do anything.
#endif
}

}  // namespace clio::cte::gpu_vector

#endif  // CTP_IS_GPU_COMPILER

#endif  // CLIO_CTE_GPU_VECTOR_H_
