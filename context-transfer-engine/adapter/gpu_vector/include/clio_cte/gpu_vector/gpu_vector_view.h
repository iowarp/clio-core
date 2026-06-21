/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_GPU_VECTOR_VIEW_H_
#define CLIO_CTE_GPU_VECTOR_VIEW_H_

#include <clio_runtime/types.h>
#include <clio_runtime/gpu/future.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/gpu_vector/gpu_vector_page.h>

namespace clio::cte::gpu_vector {

/**
 * Stride between adjacent Block structs in the meta backend (the
 * Page[total_pages_per_block] flexible array makes sizeof(Block) too
 * small). Computed once at ctor time and stored on DeviceView so
 * kernels can index `((char*)blocks) + block_stride_bytes * block_idx`.
 *
 * Two-tier layout: `pages_base` is kDeviceMem (HBM, the fast tier);
 * `host_pages_base` is kPinnedHost (DRAM, larger fall-back tier that
 * the GPU still loads/stores directly). Per-Block slots
 * `[0, gpu_pages_per_block)` reference HBM pages; slots
 * `[gpu_pages_per_block, total_pages_per_block)` reference DRAM pages.
 *
 * `host_pages_per_block == 0` ⇒ legacy single-tier mode. The cache
 * thread runs `LegacyFlushKernel` (the original CacheMgmtKernel), the
 * rescore_q is dormant, and `kPageBusy` is unused.
 */
/**
 * Pinned-host instrumentation counters. The user kernel and the
 * cache-manager kernel both atomicAdd_system into these so the host
 * can read a coherent snapshot at any time without a sync. All zero
 * by default — set the pointer to nullptr (via SetStatsPtr) and the
 * counters are skipped at runtime (one branch per call site).
 */
struct VectorStats {
  /** User-kernel side. */
  unsigned long long resolve_total;       /**< page lookups attempted */
  unsigned long long resolve_hits;        /**< page found in cache */
  unsigned long long resolve_cold_miss;   /**< page not cached → EvictSlot path */
  unsigned long long resolve_fault_get;   /**< cold miss on read → FaultPage */
  /** Cache-manager kernel side. */
  unsigned long long manager_iters;       /**< RunCachePass calls */
  unsigned long long manager_work_iters;  /**< RunCachePass returned work>0 */
  unsigned long long phase1c_drained;     /**< rescore entries consumed */
  unsigned long long phase2_swaps;        /**< actual HBM↔DRAM swaps */
  unsigned long long phase3_flushes;      /**< Sends from manager */
  unsigned long long phase4_pops;         /**< prefetch_q entries popped */
  unsigned long long phase4_prefetches;   /**< prefetch slot claims succeeded */
  unsigned long long phase4_skip_cached;  /**< already cached when popped */
  unsigned long long phase4_skip_nofree;  /**< no DRAM slot acquirable */
  unsigned long long resolve_spin_iters;  /**< nanosleep spins while waiting
                                            *  for the manager to populate
                                            *  a page (cold-miss fault off) */
};

struct DeviceViewBase {
  Block *blocks;                  /**< device pointer (kDeviceMem). */
  clio::run::u32 block_stride_bytes;
  void *pages_base;               /**< HBM page backend (kDeviceMem). */
  void *host_pages_base;          /**< DRAM page backend (kPinnedHost). nullptr in legacy. */
  /**
   * Pre-allocated PutBlob/GetBlob task slots in pinned host memory.
   * One Task+FutureShm pair per (block, slot) where slot covers BOTH
   * tiers. Total slot count per block is
   * `gpu_pages_per_block + host_pages_per_block`.
   */
  char *put_pool_base;            /**< pinned host. */
  char *get_pool_base;            /**< pinned host. */
  clio::run::u32 put_slot_stride;       /**< sizeof(PutBlobTask)+sizeof(gpu::FutureShm). */
  clio::run::u32 get_slot_stride;
  /**
   * Allocator ids of the page backends. The bdev runtime resolves
   * blob_data_ ShmPtrs via ctp::DeviceAwareMemcpy; pinned-host
   * pages decode the same way (cudaMemcpyDefault auto-detects).
   */
  ctp::ipc::AllocatorId pages_alloc_id;
  ctp::ipc::AllocatorId host_pages_alloc_id;  /**< null in legacy. */
  ctp::ipc::AllocatorId put_pool_alloc_id;
  ctp::ipc::AllocatorId get_pool_alloc_id;
  /** Tag the kernel writes blobs against. Set once at Vector ctor. */
  clio::cte::core::TagId tag_id;
  clio::run::u32 nblocks;
  clio::run::u32 gpu_pages_per_block;   /**< HBM cache slots per block. */
  clio::run::u32 host_pages_per_block;  /**< DRAM cache slots per block. 0 ⇒ legacy. */
  clio::run::u64 page_size_bytes;
  /** When false (default), Resolve's cold-miss path does NOT do the
   *  synchronous EvictSlot+FaultPage. Instead the user kernel pushes
   *  a high-priority hint and spin-waits for the cache-manager kernel
   *  to populate the page. Use this for workloads that have prefetch
   *  hints in flight ahead of access; never use it without hints or
   *  the user kernel will deadlock. */
  bool allow_cold_miss_fault;
  /** Optional pinned-host counters block; nullptr means disabled. */
  VectorStats *stats;
};

/** Total slots per block across both tiers. */
CTP_INLINE_CROSS_FUN clio::run::u32 TotalPagesPerBlock(const DeviceViewBase &v) {
  return v.gpu_pages_per_block + v.host_pages_per_block;
}

/**
 * Strongly-typed view captured by user kernels. Trivially copyable POD.
 * The kernel side macro CLIO_GPU_VECTOR_KERNEL_INIT(view) sets up the
 * per-warp last-page cache so the operator[]() fast path works.
 */
template <typename T>
struct DeviceView {
  DeviceViewBase base;
  /**
   * Number of T-elements per page. Computed as
   * `page_size_bytes / sizeof(T)` at ctor time.
   */
  clio::run::u64 page_capacity_t;
};

/** Per-block resolution helper — handles the variable-size Page array. */
CTP_INLINE_CROSS_FUN Block *GetBlock(const DeviceViewBase &v,
                                             clio::run::u32 block_idx) {
  return reinterpret_cast<Block *>(
      reinterpret_cast<char *>(v.blocks) +
      static_cast<clio::run::u64>(v.block_stride_bytes) * block_idx);
}

/** Resolve the i-th task in the put pool. `slot` indexes BOTH tiers
 *  (0..total_pages_per_block - 1). */
CTP_INLINE_CROSS_FUN clio::cte::core::PutBlobTask *GetPutTask(
    const DeviceViewBase &v, clio::run::u32 block_idx, clio::run::u32 slot) {
  clio::run::u64 off =
      (static_cast<clio::run::u64>(block_idx) * TotalPagesPerBlock(v) + slot) *
      v.put_slot_stride;
  return reinterpret_cast<clio::cte::core::PutBlobTask *>(v.put_pool_base + off);
}

/** Resolve the i-th task in the get pool. */
CTP_INLINE_CROSS_FUN clio::cte::core::GetBlobTask *GetGetTask(
    const DeviceViewBase &v, clio::run::u32 block_idx, clio::run::u32 slot) {
  clio::run::u64 off =
      (static_cast<clio::run::u64>(block_idx) * TotalPagesPerBlock(v) + slot) *
      v.get_slot_stride;
  return reinterpret_cast<clio::cte::core::GetBlobTask *>(v.get_pool_base + off);
}

/** Co-located gpu::FutureShm for a put task. */
CTP_INLINE_CROSS_FUN clio::run::gpu::FutureShm *GetPutFutureShm(
    const DeviceViewBase &v, clio::run::u32 block_idx, clio::run::u32 slot) {
  return reinterpret_cast<clio::run::gpu::FutureShm *>(
      reinterpret_cast<char *>(GetPutTask(v, block_idx, slot)) +
      sizeof(clio::cte::core::PutBlobTask));
}

/** Co-located gpu::FutureShm for a get task. */
CTP_INLINE_CROSS_FUN clio::run::gpu::FutureShm *GetGetFutureShm(
    const DeviceViewBase &v, clio::run::u32 block_idx, clio::run::u32 slot) {
  return reinterpret_cast<clio::run::gpu::FutureShm *>(
      reinterpret_cast<char *>(GetGetTask(v, block_idx, slot)) +
      sizeof(clio::cte::core::GetBlobTask));
}

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_VIEW_H_
