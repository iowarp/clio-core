/**
 * bam/page_cache_host.h -- Host-side page cache manager
 *
 * Allocates GPU VRAM for the page cache and manages the NVMe controller
 * (or host-memory fallback). This is the main entry point for setting up
 * BaM on the host side.
 */
#ifndef BAM_PAGE_CACHE_HOST_H
#define BAM_PAGE_CACHE_HOST_H

#include <bam/types.h>
#include <bam/controller.h>
#include <memory>
#include <vector>

namespace bam {


class PageCache {
 public:
  /**
   * Create and initialize the page cache.
   *
   * Allocates GPU memory for cache pages, metadata arrays, and
   * optionally connects to NVMe via gpu-nvme-bridge.
   */
  explicit PageCache(const PageCacheConfig &config);
  ~PageCache();

  /** Get the device-side state descriptor (for passing to GPU kernels). */
  PageCacheDeviceState device_state() const { return dev_state_; }

  /** Get a queue pair device descriptor for the given queue index. */
  QueuePairDevice queue_pair_device(uint32_t idx) const;

  /** Get the backend type. */
  BackendType backend() const { return config_.backend; }

  /** Get the NVMe controller (may be null if host-memory backend). */
  NvmeController *controller() { return ctrl_.get(); }

  /**
   * Get the pinned host memory base pointer (for host-memory fallback).
   * NULL if using NVMe backend.
   */
  uint8_t *host_buffer() const { return host_buf_; }

  /**
   * Allocate and pin host memory for the backing store (host-memory mode).
   * @param total_bytes  Total size of the backing store
   * @return 0 on success
   */
  int alloc_host_backing(size_t total_bytes);

  /**
   * Get per-cache-page bus addresses for NVMe mode.
   * Returns device pointer to the bus address array.
   */
  uint64_t *device_bus_addrs() const { return d_bus_addrs_; }

  const PageCacheConfig &config() const { return config_; }

 private:
  PageCacheConfig config_;
  PageCacheDeviceState dev_state_;

  // GPU allocations
  uint8_t  *d_cache_mem_;     // GPU VRAM for cached pages
  uint32_t *d_page_states_;   // Per-page state
  uint64_t *d_page_tags_;     // Per-page tag (storage offset)
  uint32_t *d_page_locks_;    // Per-page spinlock
  uint32_t *d_page_refs_;     // Per-page pin count (acquire_page/release_page)
 private:
  uint64_t *d_bus_addrs_;     // Per-cache-page bus addresses (NVMe mode)

  // NVMe controller (optional)
  std::unique_ptr<NvmeController> ctrl_;

  // Host-memory fallback
  uint8_t *host_buf_;         // Pinned host memory
  size_t   host_buf_size_;

  int init_gpu_memory();
  int init_nvme_backend();
  int init_host_backend();
};

}  // namespace bam

#endif  // BAM_PAGE_CACHE_HOST_H
