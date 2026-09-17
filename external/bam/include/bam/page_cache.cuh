/**
 * bam/page_cache.cuh -- GPU-side page cache for BaM
 *
 * Implements a software-managed page cache in GPU HBM (VRAM).
 *
 * Two access modes:
 *   1. Per-thread (original): each thread independently acquires pages.
 *   2. Warp-cooperative (inspired by GIDS): lane 0 acquires, all 32 lanes
 *      cooperate on page loading. ~32x faster page fills.
 *
 * Concurrency: lock-free with atomicCAS on page tags.
 */
#ifndef BAM_PAGE_CACHE_CUH
#define BAM_PAGE_CACHE_CUH

#include <bam/types.h>
#include <cuda_runtime.h>

namespace bam {

/* ================================================================== */
/* Per-thread page cache operations (original API)                     */
/* ================================================================== */

__device__ inline uint8_t *page_cache_acquire(
    PageCacheDeviceState &state,
    uint64_t offset,
    bool *needs_load) {
  uint32_t slot = (uint32_t)((offset >> state.page_shift) % state.num_pages);
  uint8_t *page = state.cache_mem + (uint64_t)slot * state.page_size;

  unsigned long long desired = (unsigned long long)offset;
  unsigned long long *tag_ptr = (unsigned long long *)&state.page_tags[slot];
  unsigned long long prev = atomicCAS(tag_ptr, desired, desired);

  if (prev == desired) {
    uint32_t st = atomicAdd(&state.page_states[slot], 0);
    if (st == static_cast<uint32_t>(PageState::kValid) ||
        st == static_cast<uint32_t>(PageState::kDirty)) {
      *needs_load = false;
      return page;
    }
    while (true) {
      __threadfence();
      st = atomicAdd(&state.page_states[slot], 0);
      if (st == static_cast<uint32_t>(PageState::kValid) ||
          st == static_cast<uint32_t>(PageState::kDirty)) {
        *needs_load = false;
        return page;
      }
      unsigned long long cur = atomicAdd(tag_ptr, 0ULL);
      if (cur != desired) break;
    }
  }

  atomicExch(tag_ptr, desired);
  atomicExch(&state.page_states[slot],
             static_cast<uint32_t>(PageState::kLoading));
  __threadfence();
  *needs_load = true;
  return page;
}

__device__ inline void page_cache_finish_load(
    PageCacheDeviceState &state,
    uint64_t offset) {
  uint32_t slot = (uint32_t)((offset >> state.page_shift) % state.num_pages);
  __threadfence();
  atomicExch(&state.page_states[slot],
             static_cast<uint32_t>(PageState::kValid));
  __threadfence();
}

/**
 * Acquire a page AND PIN IT. Upstream BaM's acquire_page/release_page pair
 * is a reference count, not just a lookup: it is what stops a slot being
 * re-tagged while another block is mid-read of it. Without the pin this
 * cache was documented as unusable above 2 blocks out of core.
 *
 * A takeover waits for the slot's pin count to drain. See
 * PageCacheDeviceState::page_refs for the one-pin-per-thread contract that
 * keeps that wait acyclic on a direct-mapped cache.
 */
__device__ inline uint8_t *page_cache_acquire_pinned(
    PageCacheDeviceState &state,
    uint64_t offset,
    bool *needs_load) {
  const uint32_t slot =
      (uint32_t)((offset >> state.page_shift) % state.num_pages);
  uint8_t *page = state.cache_mem + (uint64_t)slot * state.page_size;
  unsigned long long desired = (unsigned long long)offset;
  unsigned long long *tag_ptr = (unsigned long long *)&state.page_tags[slot];

  for (;;) {
    // Pin FIRST, then check the tag. Reversed, the slot could be taken over
    // between the check and the pin -- the same publish-then-verify order a
    // page cache always needs.
    atomicAdd(&state.page_refs[slot], 1u);
    __threadfence();
    if (atomicAdd(tag_ptr, 0ULL) == desired) {
      const uint32_t st = atomicAdd(&state.page_states[slot], 0);
      if (st == static_cast<uint32_t>(PageState::kValid) ||
          st == static_cast<uint32_t>(PageState::kDirty)) {
        *needs_load = false;
        return page;                      // hit, pinned
      }
      if (st == static_cast<uint32_t>(PageState::kLoading)) {
        atomicSub(&state.page_refs[slot], 1u);   // someone else is filling it
        __nanosleep(64);
        continue;
      }
    }
    // Miss: claim the slot. Drop our pin while waiting for the others to
    // drain, or we would be waiting on ourselves.
    atomicSub(&state.page_refs[slot], 1u);
    while (atomicAdd(&state.page_refs[slot], 0u) != 0u) {
      __nanosleep(64);
    }
    if (atomicCAS(&state.page_states[slot],
                  static_cast<uint32_t>(PageState::kValid),
                  static_cast<uint32_t>(PageState::kLoading)) ==
            static_cast<uint32_t>(PageState::kValid) ||
        atomicCAS(&state.page_states[slot],
                  static_cast<uint32_t>(PageState::kInvalid),
                  static_cast<uint32_t>(PageState::kLoading)) ==
            static_cast<uint32_t>(PageState::kInvalid)) {
      atomicExch(tag_ptr, desired);
      __threadfence();
      atomicAdd(&state.page_refs[slot], 1u);   // pin the page we will fill
      *needs_load = true;
      return page;
    }
    __nanosleep(64);                     // lost the claim; re-examine
  }
}

/** Drop one pin taken by page_cache_acquire_pinned. */
__device__ inline void page_cache_release(
    PageCacheDeviceState &state,
    uint64_t offset) {
  const uint32_t slot =
      (uint32_t)((offset >> state.page_shift) % state.num_pages);
  __threadfence();
  atomicSub(&state.page_refs[slot], 1u);
}

__device__ inline void page_cache_mark_dirty(
    PageCacheDeviceState &state,
    uint64_t offset) {
  uint32_t slot = (uint32_t)((offset >> state.page_shift) % state.num_pages);
  atomicExch(&state.page_states[slot],
             static_cast<uint32_t>(PageState::kDirty));
}

/* ================================================================== */
/* Warp-cooperative page cache (inspired by GIDS)                      */
/*                                                                     */
/* All 32 lanes in a warp cooperate:                                   */
/*   - Lane 0 does the atomicCAS and broadcasts result                 */
/*   - All lanes participate in page copy (32x faster)                 */
/*   - Eliminates 31 redundant atomics per warp                        */
/* ================================================================== */

/**
 * Warp-cooperative page acquire.
 * Only lane 0 does the atomic tag check. Result is broadcast to all lanes.
 * Returns pointer to the cache page and whether loading is needed.
 */
__device__ inline uint8_t *warp_page_cache_acquire(
    PageCacheDeviceState &state,
    uint64_t offset,
    bool *needs_load) {
  uint32_t lane_id = threadIdx.x & 31;
  uint32_t slot = (uint32_t)((offset >> state.page_shift) % state.num_pages);
  uint8_t *page = state.cache_mem + (uint64_t)slot * state.page_size;

  uint32_t load_flag = 0;

  if (lane_id == 0) {
    unsigned long long desired = (unsigned long long)offset;
    unsigned long long *tag_ptr = (unsigned long long *)&state.page_tags[slot];
    unsigned long long prev = atomicCAS(tag_ptr, desired, desired);

    if (prev == desired) {
      // Tag matches — check if already loaded
      uint32_t st = atomicAdd(&state.page_states[slot], 0);
      if (st == static_cast<uint32_t>(PageState::kValid) ||
          st == static_cast<uint32_t>(PageState::kDirty)) {
        load_flag = 0;  // Cache hit
      } else {
        // Another warp is loading — just reload (idempotent, no deadlock)
        load_flag = 1;
      }
    } else {
      // Cache miss — install our tag and load
      atomicExch(tag_ptr, desired);
      atomicExch(&state.page_states[slot],
                 static_cast<uint32_t>(PageState::kLoading));
      __threadfence();
      load_flag = 1;
    }
  }

  // Broadcast load_flag from lane 0 to all lanes
  load_flag = __shfl_sync(0xFFFFFFFF, load_flag, 0);
  *needs_load = (load_flag != 0);
  return page;
}

/**
 * Warp-cooperative page load from pinned DRAM to HBM.
 * All 32 lanes participate — 32x faster than single-thread copy.
 * Inspired by GIDS: `for (; tid < dim; tid += 32)`
 */
__device__ inline void warp_host_read_page(
    uint8_t *dst,
    const uint8_t *host_base,
    uint64_t offset,
    uint32_t page_size) {
  uint32_t lane_id = threadIdx.x & 31;
  const uint8_t *src = host_base + offset;

  // 32 lanes x 16 bytes = 512 bytes per iteration
  uint32_t n_uint4 = page_size / sizeof(uint4);
  for (uint32_t i = lane_id; i < n_uint4; i += 32) {
    reinterpret_cast<uint4 *>(dst)[i] =
        reinterpret_cast<const uint4 *>(src)[i];
  }
  __syncwarp();
  __threadfence();
}

/**
 * Warp-cooperative page write from HBM back to pinned DRAM.
 */
__device__ inline void warp_host_write_page(
    const uint8_t *src,
    uint8_t *host_base,
    uint64_t offset,
    uint32_t page_size) {
  uint32_t lane_id = threadIdx.x & 31;
  uint8_t *dst = host_base + offset;

  uint32_t n_uint4 = page_size / sizeof(uint4);
  for (uint32_t i = lane_id; i < n_uint4; i += 32) {
    reinterpret_cast<uint4 *>(dst)[i] =
        reinterpret_cast<const uint4 *>(src)[i];
  }
  __syncwarp();
  __threadfence_system();
}

/**
 * Warp-cooperative finish load. Only lane 0 updates state.
 */
__device__ inline void warp_page_cache_finish_load(
    PageCacheDeviceState &state,
    uint64_t offset) {
  uint32_t lane_id = threadIdx.x & 31;
  if (lane_id == 0) {
    uint32_t slot = (uint32_t)((offset >> state.page_shift) % state.num_pages);
    __threadfence();
    atomicExch(&state.page_states[slot],
               static_cast<uint32_t>(PageState::kValid));
    __threadfence();
  }
  __syncwarp();
}

/* ================================================================== */
/* Single-thread DRAM I/O (kept for backward compat)                   */
/* ================================================================== */

__device__ inline void host_read_page(
    uint8_t *dst,
    const uint8_t *host_base,
    uint64_t offset,
    uint32_t page_size) {
  const uint8_t *src = host_base + offset;
  for (uint32_t i = 0; i < page_size; i += sizeof(uint4)) {
    *reinterpret_cast<uint4 *>(dst + i) =
        *reinterpret_cast<const uint4 *>(src + i);
  }
  __threadfence();
}

__device__ inline void host_write_page(
    const uint8_t *src,
    uint8_t *host_base,
    uint64_t offset,
    uint32_t page_size) {
  uint8_t *dst = host_base + offset;
  for (uint32_t i = 0; i < page_size; i += sizeof(uint4)) {
    *reinterpret_cast<uint4 *>(dst + i) =
        *reinterpret_cast<const uint4 *>(src + i);
  }
  __threadfence_system();
}

/* ================================================================== */
/* NVMe I/O stubs                                                      */
/* ================================================================== */

/**
 * Fill a cache page from the "device".
 *
 * NOT a queue-pair emulation. A full submission/doorbell/completion path was
 * built and removed: it needs device atomics on mapped host memory, which
 * this GPU cannot do (cudaDevAttrHostNativeAtomicSupported = 0), and the
 * host-side controller it requires serialises against the very kernel
 * waiting on it. The protocol shape was not worth that.
 *
 * What a miss actually costs on real BaM is a bulk asynchronous transfer
 * from storage into the cache page, and CUDA already gives that: the
 * warp-cooperative copy below moves a page from pinned host memory over
 * PCIe with all 32 lanes in flight, GPU-initiated, no host thread. That is
 * the transfer being modelled, so it IS the emulation -- the medium is
 * DRAM rather than flash, which understates a real NVMe miss and is worth
 * saying out loud whenever a miss-heavy number is quoted.
 */
__device__ inline int nvme_read_page(
    QueuePairDevice &qp, uint64_t bus_addr,
    uint64_t offset, uint32_t page_size) {
  (void)qp; (void)bus_addr; (void)offset; (void)page_size;
  return -1;   // wired only when real gnb hardware is present
}

__device__ inline int nvme_write_page(
    QueuePairDevice &qp, uint64_t bus_addr,
    uint64_t offset, uint32_t page_size) {
  (void)qp; (void)bus_addr; (void)offset; (void)page_size;
  return -1;
}

}  // namespace bam

#endif  // BAM_PAGE_CACHE_CUH
