/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_CTE_GPU_VECTOR_PAGE_H_
#define CLIO_CTE_GPU_VECTOR_PAGE_H_

#include <clio_runtime/types.h>
#include <clio_runtime/gpu/future.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cte::gpu_vector {

/**
 * One entry in the per-block rescore ring buffer. Producers (user
 * kernels' rescore lambda) push `{page_idx, score}` pairs. The cache
 * manager kernel drains the queue, either updating the score on a
 * matching cached Page or forwarding the entry to its internal
 * `prefetch_q` for the prefetch phase.
 */
struct RescoreEntry {
  clio::run::u32 page_idx; /**< Logical blob page index. */
  float score;       /**< Caller-provided priority (higher = hotter). */
};

/** Power-of-two capacities so producer ring indexing is `tail & (cap-1)`. */
inline constexpr clio::run::u32 kRescoreQueueCap = 256;
inline constexpr clio::run::u32 kPrefetchQueueCap = 256;

/**
 * Per-block MPSC ring fed by user kernels. `tail` is the producer
 * cursor (atomicAdd'd by writers); `head` is the consumer cursor (only
 * touched by the cache manager kernel, so no atomic needed there).
 * `remaining()` is used by `write_range`/`read_range` to clip the
 * rescore lambda's lookahead to what fits.
 */
struct RescoreQueue {
  clio::run::u32 head;
  clio::run::u32 tail;
  RescoreEntry slots[kRescoreQueueCap];
};

/**
 * Cache-manager-internal SPSC queue: rescore phase produces entries
 * for blob pages not currently cached; prefetch phase consumes them.
 * No cross-thread access — only the manager warp/kernel touches this.
 */
struct PrefetchQueue {
  clio::run::u32 head;
  clio::run::u32 tail;
  RescoreEntry slots[kPrefetchQueueCap];
};

/**
 * Per-page metadata. Lives in device memory only — the CPU never reads or
 * writes Page fields directly. Bookkeeping changes flow through the
 * cache-management kernel and the user kernel via atomic ops.
 *
 * `modify_min` / `modify_max` are element offsets within the page (in T
 * units). They are atomically swapped to -1 by the kernel that picks up
 * the page for flush, so the user kernel and the cache-management kernel
 * never disagree about which range is in flight.
 *
 * `active_put` / `active_get` are the in-flight Send futures. Empty
 * (`IsNull()`) means the slot is idle. The next access boundary that
 * needs the page calls `Wait()` and then clears the future.
 *
 * `score` is set by the CacheManagerKernel's rescore phase from the
 * normalized lru_clock and overridden by entries from the per-block
 * rescore_q. `tier` is bound to the slot's backing memory (0 = HBM,
 * 1 = pinned-host DRAM) and never changes for the lifetime of the
 * Vector; reorganize swaps *contents* between slots, not the tier.
 */
struct Page {
  void *device_ptr;        /**< Base of this page inside its tier backend. */
  int32_t page_idx;       /**< -1 if slot is empty. */
  int32_t modify_min;     /**< -1 = clean. */
  int32_t modify_max;     /**< -1 = clean. */
  clio::run::u32 flags;          /**< See kPage* bits below. */
  clio::run::u64 lru_clock;      /**< clock64() at last access (for LRU). */
  float score;             /**< Normalized priority, set by manager. */
  clio::run::u32 tier;           /**< 0 = HBM (kDeviceMem), 1 = DRAM (kPinnedHost). */
  clio::run::gpu::Future<clio::cte::core::PodPutBlobTask> active_put;
  clio::run::gpu::Future<clio::cte::core::PodGetBlobTask> active_get;
};

/**
 * Per-block control structure: a fixed-size Page table sized
 * `gpu_pages_per_block + host_pages_per_block`, where the first
 * `gpu_pages_per_block` entries are tier=0 (HBM) and the rest are
 * tier=1 (DRAM). Reorganize swaps page contents between tiers via
 * `swap_scratch` (one page of HBM per block).
 *
 * Blob naming scheme is `<tag_name>_b<block>` (the per-page suffix
 * `_pi<page_idx>` is appended runtime-side from `gpu_page_idx_`).
 */
struct Block {
  clio::run::u32 block_idx;
  clio::run::u32 num_modified;          /**< atomic counter, bumped by writers. */
  clio::run::u32 gpu_pages_per_block;
  clio::run::u32 host_pages_per_block;
  void *swap_scratch;             /**< page-sized HBM scratch for swap. */
  clio::run::u32 flush_cursor;          /**< Round-robin Phase 3 starting slot. */
  clio::run::u32 _pad0;                 /**< keep pages[] 16-byte aligned. */
  RescoreQueue rescore_q;
  PrefetchQueue prefetch_q;
  /** `pages[i]` is the i-th cached slot for this block. Slots
   *  `[0, gpu_pages_per_block)` are HBM, `[gpu_pages_per_block,
   *  gpu_pages_per_block + host_pages_per_block)` are DRAM. */
  Page pages[];
};

/** Page::flags bits. */
/** A PutBlob is in flight for this page (FlushPage will not double-submit). */
inline constexpr clio::run::u32 kPagePutInFlight = 1u << 0;
/** Layout mutex — CAS-acquired by whoever mutates page_idx / modify_min/max /
 *  tier-contents. Both the user kernel (write_range/read_range) and the
 *  CacheManagerKernel (reorganize/prefetch) honor this bit. */
inline constexpr clio::run::u32 kPageBusy = 1u << 1;
/** A GetBlob is in flight for this page (FaultPage path). */
inline constexpr clio::run::u32 kPageGetInFlight = 1u << 2;

}  // namespace clio::cte::gpu_vector

#endif  // CLIO_CTE_GPU_VECTOR_PAGE_H_
