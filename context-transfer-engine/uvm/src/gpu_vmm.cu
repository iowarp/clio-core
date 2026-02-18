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

#include "wrp_cte/uvm/gpu_vmm.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace wrp_cte::uvm {

/** Kernel to fill a GPU memory region with a repeated int value */
__global__ void fillKernel(int *ptr, int value, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    ptr[idx] = value;
  }
}

GpuVirtualMemoryManager::GpuVirtualMemoryManager() = default;

GpuVirtualMemoryManager::~GpuVirtualMemoryManager() { destroy(); }

CUresult GpuVirtualMemoryManager::init(const GpuVmmConfig &config) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Initialize CUDA driver API
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuInit failed: %d\n", res);
    return res;
  }

  // Get the device handle
  res = cuDeviceGet(&device_, config.device);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuDeviceGet failed: %d\n", res);
    return res;
  }

  // Query the allocation granularity for the device
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = config.device;

  size_t granularity = 0;
  res = cuMemGetAllocationGranularity(&granularity, &prop,
                                       CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuMemGetAllocationGranularity failed: %d\n", res);
    return res;
  }

  // Align page size up to hardware granularity
  page_size_ = config.page_size;
  if (page_size_ < granularity) {
    page_size_ = granularity;
  }
  // Round up to multiple of granularity
  page_size_ = ((page_size_ + granularity - 1) / granularity) * granularity;

  // Align total VA size to page_size
  va_size_ = config.va_size_bytes;
  va_size_ = ((va_size_ + page_size_ - 1) / page_size_) * page_size_;

  fill_value_ = config.fill_value;
  total_pages_ = va_size_ / page_size_;

  // Reserve virtual address range -- no physical memory is consumed here
  res = cuMemAddressReserve(&va_base_, va_size_, page_size_, 0, 0);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr,
            "GpuVmm: cuMemAddressReserve failed for %zu bytes: %d\n"
            "  This GPU may not support a %zu-byte VA reservation.\n"
            "  Try a smaller va_size_bytes.\n",
            va_size_, res, va_size_);
    return res;
  }

  // Initialize the software page table (all pages start unmapped)
  page_table_.resize(total_pages_);

  fprintf(stdout,
          "GpuVmm: Initialized\n"
          "  VA base:       0x%llx\n"
          "  VA size:       %zu bytes (%.2f TB)\n"
          "  Page size:     %zu bytes (%.2f MB)\n"
          "  Total pages:   %zu\n"
          "  HW granularity: %zu bytes\n",
          (unsigned long long)va_base_, va_size_,
          (double)va_size_ / (1024.0 * 1024 * 1024 * 1024), page_size_,
          (double)page_size_ / (1024.0 * 1024), total_pages_, granularity);

  return CUDA_SUCCESS;
}

void GpuVirtualMemoryManager::destroy() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (va_base_ == 0) return;

  // Unmap and free all backed pages
  for (size_t i = 0; i < total_pages_; ++i) {
    PageEntry &entry = page_table_[i];
    if (entry.mapped) {
      CUdeviceptr page_addr = va_base_ + i * page_size_;
      cuMemUnmap(page_addr, page_size_);
      cuMemRelease(entry.alloc_handle);
      entry.mapped = false;
    }
  }

  // Release the VA reservation
  cuMemAddressFree(va_base_, va_size_);
  va_base_ = 0;
  va_size_ = 0;
  total_pages_ = 0;
  page_table_.clear();
}

size_t GpuVirtualMemoryManager::getMappedPageCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = 0;
  for (const auto &entry : page_table_) {
    if (entry.mapped) ++count;
  }
  return count;
}

CUresult GpuVirtualMemoryManager::touchPage(size_t page_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (page_index >= total_pages_) {
    fprintf(stderr, "GpuVmm: touchPage: page_index %zu out of range [0, %zu)\n",
            page_index, total_pages_);
    return CUDA_ERROR_INVALID_VALUE;
  }

  PageEntry &entry = page_table_[page_index];
  if (entry.mapped) {
    return CUDA_SUCCESS;  // Already backed
  }

  // 1. Allocate physical memory (a "handle" -- not yet mapped anywhere)
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_;

  CUresult res = cuMemCreate(&entry.alloc_handle, page_size_, &prop, 0);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuMemCreate failed for page %zu: %d\n",
            page_index, res);
    return res;
  }

  // 2. Map the physical allocation into the reserved VA slot
  CUdeviceptr page_addr = va_base_ + page_index * page_size_;
  res = cuMemMap(page_addr, page_size_, 0, entry.alloc_handle, 0);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuMemMap failed for page %zu: %d\n",
            page_index, res);
    cuMemRelease(entry.alloc_handle);
    return res;
  }

  // 3. Set access permissions (read+write for this device)
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device_;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  res = cuMemSetAccess(page_addr, page_size_, &access, 1);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuMemSetAccess failed for page %zu: %d\n",
            page_index, res);
    cuMemUnmap(page_addr, page_size_);
    cuMemRelease(entry.alloc_handle);
    return res;
  }

  entry.mapped = true;

  // 4. Fill the page with the configured value using a GPU kernel
  size_t num_ints = page_size_ / sizeof(int);
  int threads = 256;
  int blocks = (int)((num_ints + threads - 1) / threads);
  fillKernel<<<blocks, threads>>>((int *)page_addr, fill_value_, num_ints);
  cudaDeviceSynchronize();

  return CUDA_SUCCESS;
}

CUresult GpuVirtualMemoryManager::touchRange(size_t offset, size_t size) {
  if (size == 0) return CUDA_SUCCESS;

  size_t first_page = offset / page_size_;
  size_t last_page = (offset + size - 1) / page_size_;

  for (size_t i = first_page; i <= last_page; ++i) {
    CUresult res = touchPage(i);
    if (res != CUDA_SUCCESS) return res;
  }
  return CUDA_SUCCESS;
}

bool GpuVirtualMemoryManager::isMapped(size_t page_index) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_index >= total_pages_) return false;
  return page_table_[page_index].mapped;
}

CUresult GpuVirtualMemoryManager::evictPage(size_t page_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (page_index >= total_pages_) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  PageEntry &entry = page_table_[page_index];
  if (!entry.mapped) {
    return CUDA_SUCCESS;  // Nothing to evict
  }

  CUdeviceptr page_addr = va_base_ + page_index * page_size_;

  CUresult res = cuMemUnmap(page_addr, page_size_);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuMemUnmap failed for page %zu: %d\n",
            page_index, res);
    return res;
  }

  res = cuMemRelease(entry.alloc_handle);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuMemRelease failed for page %zu: %d\n",
            page_index, res);
    return res;
  }

  entry.mapped = false;
  entry.alloc_handle = 0;
  return CUDA_SUCCESS;
}

CUdeviceptr GpuVirtualMemoryManager::getPagePtr(size_t page_index) const {
  if (page_index >= total_pages_) return 0;
  return va_base_ + page_index * page_size_;
}

}  // namespace wrp_cte::uvm
