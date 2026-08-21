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

/** Device-side mirrors of bam::EmuSqe / bam::EmuCqe. Kept here so the
 *  kernel path does not have to include the host header. */
struct EmuSqeDev {
  uint64_t offset; uint64_t bus_addr;
  uint32_t nbytes; uint32_t opcode; uint32_t cid; uint32_t pad;
};
struct EmuCqeDev {
  uint32_t cid; uint32_t status; uint32_t phase; uint32_t pad;
};

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
 * Submit one page-granular command and wait for its completion.
 *
 * This is the real sequence a GPU issues to an NVMe queue pair -- claim an
 * SQ slot, write the SQE, ring the doorbell, poll the CQ for the phase this
 * command is due -- against an emulated controller whose medium is pinned
 * host memory (see bam/nvme_emu.h). It replaces stubs that ignored their
 * arguments and returned -1, which is why a "miss" used to cost a memcpy
 * and nothing else.
 *
 * Called by ONE thread per page fill; the rest of the warp waits at the
 * barrier its caller already holds.
 */
__device__ inline int nvme_submit_page(
    QueuePairDevice &qp, uint64_t bus_addr,
    uint64_t offset, uint32_t page_size, uint32_t opcode) {
  if (qp.sq == nullptr || qp.cq == nullptr) return -1;

  // CLAIM ON THE DEVICE, PUBLISH INTO HOST MEMORY.
  //
  // The claim is an atomicAdd on device memory because this GPU cannot do
  // atomics on mapped host memory at all (cudaDevAttrHostNativeAtomicSupported
  // = 0) -- a doorbell register in host memory would be illegal, not just
  // slow. Publication is therefore a per-slot STAMP: write the entry, fence
  // system-wide, then store the stamp. A controller that observes the stamp
  // is guaranteed to observe the entry beside it.
  const unsigned int mine = atomicAdd(qp.sq_alloc, 1u);

  const uint32_t idx = mine & (qp.sq_depth - 1u);
  volatile EmuSqeDev *sqe = &((volatile EmuSqeDev *)qp.sq)[idx];
  sqe->offset = offset;
  sqe->bus_addr = bus_addr;
  sqe->nbytes = page_size;
  sqe->opcode = opcode;
  sqe->cid = mine;
  __threadfence_system();
  qp.sq_ready[idx] = (mine / qp.sq_depth) + 1u;   // publish
  __threadfence_system();

  // Poll for our completion. The phase distinguishes this generation's CQE
  // from the previous one occupying the same slot.
  const uint32_t cidx = mine & (qp.cq_depth - 1u);
  volatile EmuCqeDev *cqe = &((volatile EmuCqeDev *)qp.cq)[cidx];
  const unsigned int want = (mine / qp.cq_depth) + 1u;
  for (;;) {
    const unsigned int ph = cqe->phase;
    if (ph == want) {
      __threadfence_system();
      return (int)cqe->status;
    }
    __nanosleep(32);
  }
}

__device__ inline int nvme_read_page(
    QueuePairDevice &qp, uint64_t bus_addr,
    uint64_t offset, uint32_t page_size) {
  return nvme_submit_page(qp, bus_addr, offset, page_size, /*opcode=*/0u);
}

__device__ inline int nvme_write_page(
    QueuePairDevice &qp, uint64_t bus_addr,
    uint64_t offset, uint32_t page_size) {
  return nvme_submit_page(qp, bus_addr, offset, page_size, /*opcode=*/1u);
}

}  // namespace bam

#endif  // BAM_PAGE_CACHE_CUH
