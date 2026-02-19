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

#ifndef WRP_CTE_UVM_GPU_VMM_H_
#define WRP_CTE_UVM_GPU_VMM_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef WRP_CTE_AVAILABLE
#include <wrp_cte/core/core_client.h>
#endif

namespace wrp_cte::uvm {

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

  /** Synchronize the transfer stream */
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
    bool evicted_to_host = false;
  };

  std::vector<PageEntry> page_table_;
  mutable std::mutex mutex_;

  // Host RAM backing store: page_index -> pinned host buffer
  std::unordered_map<size_t, char *> host_backing_store_;

  // CUDA streams for async overlap
  cudaStream_t transfer_stream_ = nullptr;
  cudaStream_t compute_stream_ = nullptr;

  // CTE backing store (optional, compile-time gated)
  bool use_cte_ = false;
#ifdef WRP_CTE_AVAILABLE
  std::unique_ptr<wrp_cte::core::Tag> cte_tag_;
#endif

  /** Allocate physical memory, map into VA, set access (no fill) */
  CUresult mapAndBackPage_(size_t page_index);

  /** Free all pinned host backing store buffers */
  void freeHostBackingStore_();
};

}  // namespace wrp_cte::uvm

#endif  // WRP_CTE_UVM_GPU_VMM_H_
