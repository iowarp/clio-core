/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CLIO_CTE_UVM_GPU_VMM_H_
#define CLIO_CTE_UVM_GPU_VMM_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace clio::cte::core {
class Tag;  // clio_cte/core/core_client.h; kept out of this header so CUDA
            // consumers (nvcc) need not parse the CTE client
}  // namespace clio::cte::core

namespace clio::cte::uvm {

/** Configuration for the GPU Virtual Memory Manager */
struct GpuVmmConfig {
  size_t va_size_bytes = 512ULL * 1024 * 1024 * 1024;  // 512 GB (safe for most GPUs)
  size_t page_size = 2ULL * 1024 * 1024;               // 2 MB (GPU granularity)
  int fill_value = 5;                                   // Default fill value
  int device = 0;                                       // CUDA device ordinal
  size_t prefetch_window = 4;                           // Pages to prefetch ahead on touch
  bool use_cte = false;                                  // Use CTE blob store instead of host RAM
  std::string cte_tag_name = "gpu_vmm_pages";            // CTE tag name for page blobs
};

/**
 * Software-managed demand paging for GPU virtual memory.
 *
 * Reserves a large virtual address range using CUDA driver APIs
 * (cuMemAddressReserve). Physical memory is NOT allocated upfront. Instead,
 * pages are backed on-demand when explicitly accessed through touchPage().
 *
 * Evicted pages are saved to pinned host RAM (default) or to CTE blob
 * store (when use_cte=true) and restored on re-touch, preserving data
 * across eviction cycles. Async variants and prefetching allow overlapping
 * data transfer with GPU compute.
 *
 * This runs entirely in userspace with no root privileges required.
 */
class GpuVirtualMemoryManager {
 public:
  GpuVirtualMemoryManager();
  ~GpuVirtualMemoryManager();

  GpuVirtualMemoryManager(const GpuVirtualMemoryManager &) = delete;
  GpuVirtualMemoryManager &operator=(const GpuVirtualMemoryManager &) = delete;

  /** Initialize the VMM: reserve VA space, create streams, prepare page table */
  CUresult init(const GpuVmmConfig &config = GpuVmmConfig());

  /** Destroy the VMM: unmap all pages, free backing store, release VA */
  void destroy();

  /** Get the base device pointer for the entire VA range */
  CUdeviceptr getBasePtr() const { return va_base_; }

  /** Get the configured page size in bytes */
  size_t getPageSize() const { return page_size_; }

  /** Get total number of pages in the VA range */
  size_t getTotalPages() const { return total_pages_; }

  /** Get number of currently backed (physically mapped) pages */
  size_t getMappedPageCount() const;

  /** Get number of pages currently held in host backing store */
  size_t getEvictedPageCount() const;

  /**
   * Ensure a page is backed by physical memory (synchronous).
   * If previously evicted, restores from host RAM. Otherwise fills with fill_value.
   * Triggers prefetchAhead for subsequent pages.
   */
  CUresult touchPage(size_t page_index);

  /**
   * Ensure a page is backed by physical memory (async on transfer stream).
   * Does NOT trigger prefetch. Caller must syncTransfer() before accessing.
   */
  CUresult touchPageAsync(size_t page_index);

  /**
   * Touch all pages covering the given byte range [offset, offset+size).
   */
  CUresult touchRange(size_t offset, size_t size);

  /** Check whether a page is currently backed by physical memory */
  bool isMapped(size_t page_index) const;

  /** Check whether a page has data saved in the host backing store */
  bool isEvictedToHost(size_t page_index) const;

  /**
   * Evict a page (synchronous): save to host RAM, then unmap and release.
   * The VA slot remains reserved. Data is preserved for future touchPage.
   */
  CUresult evictPage(size_t page_index);

  /**
   * Evict a page (async copy on transfer stream, sync before unmap).
   * Compute stream continues unblocked during the D2H copy.
   */
  CUresult evictPageAsync(size_t page_index);

  /** Prefetch pages [page_index+1 .. page_index+window] asynchronously */
  void prefetchAhead(size_t page_index);

  /** Get the device pointer for a specific page */
  CUdeviceptr getPagePtr(size_t page_index) const;

  /** Get the transfer stream (for caller synchronization) */
  cudaStream_t getTransferStream() const { return transfer_stream_; }

  /** Get the compute stream (for caller kernel launches) */
  cudaStream_t getComputeStream() const { return compute_stream_; }

  /** Synchronize the transfer stream, then free the staging buffers of the
   *  async CTE restores it carried */
  void syncTransfer();

  /** Synchronize the compute stream */
  void syncCompute();

 private:
  CUdeviceptr va_base_ = 0;
  size_t va_size_ = 0;
  size_t page_size_ = 0;
  size_t total_pages_ = 0;
  int fill_value_ = 5;
  CUdevice device_ = 0;
  size_t prefetch_window_ = 4;

  struct PageEntry {
    CUmemGenericAllocationHandle alloc_handle = 0;
    bool mapped = false;
    bool evicted_to_host = false;  ///< saved off the GPU (host RAM or CTE)
    /** A page_<i> blob exists. It is the saved copy unless host RAM also
     *  holds one, which is then newer and wins the restore. */
    bool in_cte = false;
  };

  /** CTE staging buffers still being read by async H2D copies (defined in
   *  gpu_vmm.cc). */
  struct PendingFrees;

  std::vector<PageEntry> page_table_;
  mutable std::mutex mutex_;

  // Host RAM backing store: page_index -> pinned host buffer
  std::unordered_map<size_t, char *> host_backing_store_;

  // CUDA streams for async overlap
  cudaStream_t transfer_stream_ = nullptr;
  cudaStream_t compute_stream_ = nullptr;

  // CTE backing store (GpuVmmConfig::use_cte). Unconditional members: the
  // class layout must not depend on a macro the consumer may not define.
  bool use_cte_ = false;
  std::unique_ptr<clio::cte::core::Tag> cte_tag_;
  // Staging buffers of async CTE restores, freed once their copy completed
  // (freeing them at once was a use-after-free, #1190). Guarded by mutex_.
  std::unique_ptr<PendingFrees> pending_frees_;

  /** Allocate physical memory, map into VA, set access (no fill) */
  CUresult mapAndBackPage_(size_t page_index);

  /** Unmap a page and release its physical memory. Caller holds mutex_. */
  void unmapPage_(size_t page_index);

  /**
   * Fill a freshly mapped page: restore its saved copy (host RAM or CTE) or
   * write the fill value. On failure the page is unmapped again and its
   * saved copy kept. Caller holds mutex_.
   * @param page_index the page
   * @param async true: copies/fills go on transfer_stream_
   * @return CUDA_SUCCESS or the failing call's error
   */
  CUresult populatePage_(size_t page_index, bool async);

  /**
   * Restore a page from its page_<i> blob. Caller holds mutex_.
   * @param page_index the page
   * @param async true: H2D on transfer_stream_, staging freed later
   * @return CUDA_SUCCESS, or an error (blob read or copy failed)
   */
  CUresult restoreFromCte_(size_t page_index, bool async);

  /**
   * Save an evicted page's bytes as its page_<i> blob. Caller holds mutex_.
   * @param page_index the page
   * @param host_buf the page's bytes (page_size_)
   * @return true when the blob was stored
   */
  bool saveToCte_(size_t page_index, const char *host_buf);

  /** Keep an evicted page's bytes: as a CTE blob when use_cte_ (host RAM if
   *  that fails), else in host RAM. Takes ownership of host_buf. Caller
   *  holds mutex_. */
  void storeEvicted_(size_t page_index, char *host_buf);

  /**
   * Free pending staging buffers whose copy has completed.
   * @param wait true: wait for every copy first (teardown)
   */
  void drainPendingFrees_(bool wait);

  /** Blob name of a page in the CTE tag. */
  static std::string pageBlobName_(size_t page_index);

  /** Free all pinned host backing store buffers */
  void freeHostBackingStore_();
};

}  // namespace clio::cte::uvm

#endif  // CLIO_CTE_UVM_GPU_VMM_H_
