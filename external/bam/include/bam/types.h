/**
 * bam/types.h -- Core type definitions for BaM (Big Accelerator Memory)
 *
 * Defines shared types used across the BaM GPU page cache and
 * storage backend abstraction.
 */
#ifndef BAM_TYPES_H
#define BAM_TYPES_H

#include <cstdint>
#include <cstddef>

#ifdef __CUDACC__
#define BAM_HOST_DEVICE __host__ __device__
#define BAM_DEVICE      __device__
#define BAM_HOST        __host__
#else
#define BAM_HOST_DEVICE
#define BAM_DEVICE
#define BAM_HOST
#endif

namespace bam {

/** Page states in the GPU page cache. */
enum class PageState : uint32_t {
  kInvalid = 0,
  kLoading = 1,
  kValid   = 2,
  kDirty   = 3,
  kEvicting = 4,
};

/** Storage backend type. */
enum class BackendType : uint32_t {
  kHostMemory = 0,   // Fallback: data pre-loaded in pinned host memory
  kNvme       = 1,   // GPU-direct NVMe via gpu-nvme-bridge
};

/** Configuration for the BaM page cache. */
struct PageCacheConfig {
  size_t   page_size;        // Bytes per cache page (e.g., 4096 or 65536)
  size_t   num_pages;        // Number of cache pages in GPU memory
  uint32_t num_queues;       // Number of NVMe I/O queue pairs (NVMe mode)
  uint32_t queue_depth;      // Depth of each queue pair
  BackendType backend;       // Which storage backend to use
  const char *nvme_dev;      // NVMe device path (e.g., "/dev/nvme0"), or NULL
};

/** Device-side descriptor for the page cache. Copied to GPU constant/global memory. */
struct PageCacheDeviceState {
  uint8_t  *cache_mem;       // GPU VRAM: page cache buffer
  uint32_t *page_states;     // GPU VRAM: per-page state (atomics)
  uint64_t *page_tags;       // GPU VRAM: per-page storage offset tag
  uint32_t *page_locks;      // GPU VRAM: per-page spinlocks
  /**
   * GPU VRAM: per-page reference count -- the PIN behind acquire_page.
   *
   * Without it `release_page` was a no-op and a slot could be re-tagged
   * while another block was mid-read of it, which is why out-of-core runs
   * above 2 blocks were documented as "not a result". A takeover now waits
   * for the count to drain.
   *
   * CONTRACT: a thread may hold at most ONE pin at a time. The cache is
   * direct-mapped, so pinning page A and then missing on page B that maps to
   * the same slot would wait on yourself. bam_ptr::update_page releases
   * before it acquires for exactly this reason.
   */
  uint32_t *page_refs;
  size_t    page_size;
  uint32_t  num_pages;
  uint32_t  page_shift;      // log2(page_size)
};

/** Device-side descriptor for a single NVMe queue pair. */
struct QueuePairDevice {
  volatile void *sq;         // SQ entries
  volatile void *cq;         // CQ entries
  volatile uint32_t *sq_doorbell;
  volatile uint32_t *cq_doorbell;
  uint32_t sq_depth;
  uint32_t cq_depth;
  uint32_t nsid;
  uint32_t lba_shift;
};

}  // namespace bam

#endif  // BAM_TYPES_H
