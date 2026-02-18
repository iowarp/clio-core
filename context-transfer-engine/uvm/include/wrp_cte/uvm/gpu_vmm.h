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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace wrp_cte::uvm {

/** Configuration for the GPU Virtual Memory Manager */
struct GpuVmmConfig {
  size_t va_size_bytes = 512ULL * 1024 * 1024 * 1024;  // 512 GB (safe for most GPUs)
  size_t page_size = 2ULL * 1024 * 1024;                      // 2 MB (GPU granularity)
  int fill_value = 5;                                          // Default fill value
  int device = 0;                                              // CUDA device ordinal
};

/**
 * Software-managed demand paging for GPU virtual memory.
 *
 * Reserves a large virtual address range (up to 16TB) using CUDA driver APIs
 * (cuMemAddressReserve). Physical memory is NOT allocated upfront. Instead,
 * pages are backed on-demand when explicitly accessed through touchPage().
 *
 * Since true GPU page fault interception is impossible in userspace without
 * UVM, this class provides a software-managed page table that tracks which
 * virtual pages have physical backing. The caller must check isMapped()
 * before accessing memory, or use touchPage() to ensure backing exists.
 *
 * This runs entirely in userspace with no root privileges required.
 */
class GpuVirtualMemoryManager {
 public:
  GpuVirtualMemoryManager();
  ~GpuVirtualMemoryManager();

  GpuVirtualMemoryManager(const GpuVirtualMemoryManager &) = delete;
  GpuVirtualMemoryManager &operator=(const GpuVirtualMemoryManager &) = delete;

  /** Initialize the VMM: reserve VA space and prepare physical allocation pool */
  CUresult init(const GpuVmmConfig &config = GpuVmmConfig());

  /** Destroy the VMM: unmap all pages, free physical memory, release VA */
  void destroy();

  /** Get the base device pointer for the entire VA range */
  CUdeviceptr getBasePtr() const { return va_base_; }

  /** Get the configured page size in bytes */
  size_t getPageSize() const { return page_size_; }

  /** Get total number of pages in the VA range */
  size_t getTotalPages() const { return total_pages_; }

  /** Get number of currently backed (physically mapped) pages */
  size_t getMappedPageCount() const;

  /**
   * Ensure a page is backed by physical memory.
   * If the page at the given index is already mapped, this is a no-op.
   * Otherwise, allocates physical memory, maps it into the VA range,
   * sets access permissions, and fills with the configured fill_value.
   *
   * @param page_index Zero-based page index within the VA range
   * @return CUDA_SUCCESS on success, error code otherwise
   */
  CUresult touchPage(size_t page_index);

  /**
   * Touch all pages covering the given byte range [offset, offset+size).
   *
   * @param offset Byte offset from VA base
   * @param size Number of bytes in the range
   * @return CUDA_SUCCESS on success, error code otherwise
   */
  CUresult touchRange(size_t offset, size_t size);

  /** Check whether a page is currently backed by physical memory */
  bool isMapped(size_t page_index) const;

  /**
   * Evict a page: unmap physical memory and free it.
   * The VA range slot remains reserved but becomes unbacked.
   *
   * @param page_index Zero-based page index
   * @return CUDA_SUCCESS on success, error code otherwise
   */
  CUresult evictPage(size_t page_index);

  /** Get the device pointer for a specific page */
  CUdeviceptr getPagePtr(size_t page_index) const;

 private:
  CUdeviceptr va_base_ = 0;              // Start of reserved VA range
  size_t va_size_ = 0;                   // Total VA size in bytes
  size_t page_size_ = 0;                 // Page size in bytes (GPU allocation granularity)
  size_t total_pages_ = 0;              // Total pages in VA range
  int fill_value_ = 5;                  // Value to fill new pages with
  CUdevice device_ = 0;                 // CUDA device handle

  struct PageEntry {
    CUmemGenericAllocationHandle alloc_handle = 0;
    bool mapped = false;
  };

  std::vector<PageEntry> page_table_;    // Software page table
  mutable std::mutex mutex_;             // Thread safety for page table ops
};

}  // namespace wrp_cte::uvm

#endif  // WRP_CTE_UVM_GPU_VMM_H_
