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
#include <string>

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
  page_size_ = ((page_size_ + granularity - 1) / granularity) * granularity;

  // Align total VA size to page_size
  va_size_ = config.va_size_bytes;
  va_size_ = ((va_size_ + page_size_ - 1) / page_size_) * page_size_;

  fill_value_ = config.fill_value;
  prefetch_window_ = config.prefetch_window;
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

  // Create CUDA streams for async overlap
  cudaStreamCreate(&transfer_stream_);
  cudaStreamCreate(&compute_stream_);

  // Initialize CTE backing store if requested
#ifdef WRP_CTE_AVAILABLE
  use_cte_ = config.use_cte;
  if (use_cte_) {
    cte_tag_ = std::make_unique<wrp_cte::core::Tag>(config.cte_tag_name);
    fprintf(stdout, "GpuVmm: CTE backing store enabled (tag: %s)\n",
            config.cte_tag_name.c_str());
  }
#endif

  fprintf(stdout,
          "GpuVmm: Initialized\n"
          "  VA base:       0x%llx\n"
          "  VA size:       %zu bytes (%.2f TB)\n"
          "  Page size:     %zu bytes (%.2f MB)\n"
          "  Total pages:   %zu\n"
          "  HW granularity: %zu bytes\n"
          "  Prefetch window: %zu pages\n",
          (unsigned long long)va_base_, va_size_,
          (double)va_size_ / (1024.0 * 1024 * 1024 * 1024), page_size_,
          (double)page_size_ / (1024.0 * 1024), total_pages_, granularity,
          prefetch_window_);

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

  // Free all host backing store buffers
  freeHostBackingStore_();

  // Release CTE tag
#ifdef WRP_CTE_AVAILABLE
  cte_tag_.reset();
#endif

  // Destroy CUDA streams
  if (transfer_stream_) {
    cudaStreamDestroy(transfer_stream_);
    transfer_stream_ = nullptr;
  }
  if (compute_stream_) {
    cudaStreamDestroy(compute_stream_);
    compute_stream_ = nullptr;
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

size_t GpuVirtualMemoryManager::getEvictedPageCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = 0;
  for (const auto &entry : page_table_) {
    if (entry.evicted_to_host) ++count;
  }
  return count;
}

CUresult GpuVirtualMemoryManager::mapAndBackPage_(size_t page_index) {
  // Caller must hold mutex_
  PageEntry &entry = page_table_[page_index];

  // Allocate physical memory
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

  // Map into VA slot
  CUdeviceptr page_addr = va_base_ + page_index * page_size_;
  res = cuMemMap(page_addr, page_size_, 0, entry.alloc_handle, 0);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuMemMap failed for page %zu: %d\n",
            page_index, res);
    cuMemRelease(entry.alloc_handle);
    return res;
  }

  // Set access permissions
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
  return CUDA_SUCCESS;
}

CUresult GpuVirtualMemoryManager::touchPage(size_t page_index) {
  {
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

    // Allocate + map + set access
    CUresult res = mapAndBackPage_(page_index);
    if (res != CUDA_SUCCESS) return res;

    CUdeviceptr page_addr = va_base_ + page_index * page_size_;

    // Restore from backing store or fill with default value
    bool restored = false;

#ifdef WRP_CTE_AVAILABLE
    if (use_cte_ && entry.evicted_to_host) {
      // Restore from CTE: AsyncGetBlob → SHM → cudaMemcpy → GPU
      std::string blob_name = "page_" + std::to_string(page_index);
      hipc::FullPtr<char> shm = CHI_IPC->AllocateBuffer(page_size_);
      auto future = WRP_CTE_CLIENT->AsyncGetBlob(
          cte_tag_->GetTagId(), blob_name, 0, page_size_, 0, shm.shm_);
      future.Wait();
      cudaMemcpy((void *)page_addr, shm.ptr_, page_size_,
                 cudaMemcpyHostToDevice);
      CHI_IPC->FreeBuffer(shm);
      entry.evicted_to_host = false;
      restored = true;
    }
#endif

    if (!restored) {
      auto it = host_backing_store_.find(page_index);
      if (entry.evicted_to_host && it != host_backing_store_.end()) {
        // Restore saved data from host RAM
        cudaMemcpy((void *)page_addr, it->second, page_size_,
                   cudaMemcpyHostToDevice);
        cudaFreeHost(it->second);
        host_backing_store_.erase(it);
        entry.evicted_to_host = false;
      } else {
        // Fresh page: fill with configured value
        size_t num_ints = page_size_ / sizeof(int);
        int threads = 256;
        int blocks = (int)((num_ints + threads - 1) / threads);
        fillKernel<<<blocks, threads>>>((int *)page_addr, fill_value_, num_ints);
        cudaDeviceSynchronize();
        entry.evicted_to_host = false;
      }
    }
  }

  // Prefetch ahead (outside mutex)
  prefetchAhead(page_index);

  return CUDA_SUCCESS;
}

CUresult GpuVirtualMemoryManager::touchPageAsync(size_t page_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (page_index >= total_pages_) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  PageEntry &entry = page_table_[page_index];
  if (entry.mapped) {
    return CUDA_SUCCESS;
  }

  CUresult res = mapAndBackPage_(page_index);
  if (res != CUDA_SUCCESS) return res;

  CUdeviceptr page_addr = va_base_ + page_index * page_size_;

  bool restored = false;

#ifdef WRP_CTE_AVAILABLE
  if (use_cte_ && entry.evicted_to_host) {
    // Restore from CTE (sync get, then async H2D)
    std::string blob_name = "page_" + std::to_string(page_index);
    hipc::FullPtr<char> shm = CHI_IPC->AllocateBuffer(page_size_);
    auto future = WRP_CTE_CLIENT->AsyncGetBlob(
        cte_tag_->GetTagId(), blob_name, 0, page_size_, 0, shm.shm_);
    future.Wait();
    cudaMemcpyAsync((void *)page_addr, shm.ptr_, page_size_,
                    cudaMemcpyHostToDevice, transfer_stream_);
    // SHM freed after transfer completes (caller must syncTransfer)
    CHI_IPC->FreeBuffer(shm);
    entry.evicted_to_host = false;
    restored = true;
  }
#endif

  if (!restored) {
    auto it = host_backing_store_.find(page_index);
    if (entry.evicted_to_host && it != host_backing_store_.end()) {
      // Async restore from host
      cudaMemcpyAsync((void *)page_addr, it->second, page_size_,
                      cudaMemcpyHostToDevice, transfer_stream_);
      // Note: host buffer freed after sync (kept alive for async safety)
      entry.evicted_to_host = false;
    } else {
      // Async fill
      size_t num_ints = page_size_ / sizeof(int);
      int threads = 256;
      int blocks = (int)((num_ints + threads - 1) / threads);
      fillKernel<<<blocks, threads, 0, transfer_stream_>>>(
          (int *)page_addr, fill_value_, num_ints);
      entry.evicted_to_host = false;
    }
  }

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

bool GpuVirtualMemoryManager::isEvictedToHost(size_t page_index) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_index >= total_pages_) return false;
  return page_table_[page_index].evicted_to_host;
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

  // Save page contents to pinned host RAM
  char *host_buf = nullptr;
  auto it = host_backing_store_.find(page_index);
  if (it != host_backing_store_.end()) {
    host_buf = it->second;  // Reuse existing buffer
  } else {
    cudaError_t err = cudaMallocHost(&host_buf, page_size_);
    if (err != cudaSuccess) {
      fprintf(stderr, "GpuVmm: cudaMallocHost failed for page %zu: %d\n",
              page_index, err);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }

  cudaMemcpy(host_buf, (void *)page_addr, page_size_, cudaMemcpyDeviceToHost);

  // Store to CTE or keep in host RAM
#ifdef WRP_CTE_AVAILABLE
  if (use_cte_) {
    // Copy pinned host → SHM → AsyncPutBlob → Wait → free both
    std::string blob_name = "page_" + std::to_string(page_index);
    hipc::FullPtr<char> shm = CHI_IPC->AllocateBuffer(page_size_);
    memcpy(shm.ptr_, host_buf, page_size_);
    auto future = cte_tag_->AsyncPutBlob(blob_name, shm.shm_, page_size_);
    future.Wait();
    CHI_IPC->FreeBuffer(shm);
    cudaFreeHost(host_buf);
  } else
#endif
  {
    host_backing_store_[page_index] = host_buf;
  }

  // Unmap and release GPU physical memory
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
  entry.evicted_to_host = true;

  return CUDA_SUCCESS;
}

CUresult GpuVirtualMemoryManager::evictPageAsync(size_t page_index) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (page_index >= total_pages_) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  PageEntry &entry = page_table_[page_index];
  if (!entry.mapped) {
    return CUDA_SUCCESS;
  }

  CUdeviceptr page_addr = va_base_ + page_index * page_size_;

  // Allocate or reuse pinned host buffer
  char *host_buf = nullptr;
  auto it = host_backing_store_.find(page_index);
  if (it != host_backing_store_.end()) {
    host_buf = it->second;
  } else {
    cudaError_t err = cudaMallocHost(&host_buf, page_size_);
    if (err != cudaSuccess) {
      fprintf(stderr, "GpuVmm: cudaMallocHost failed for page %zu: %d\n",
              page_index, err);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }

  // Async copy GPU -> host on transfer stream
  cudaMemcpyAsync(host_buf, (void *)page_addr, page_size_,
                  cudaMemcpyDeviceToHost, transfer_stream_);

  // Must sync transfer stream before cuMemUnmap (driver API, not stream-able)
  cudaStreamSynchronize(transfer_stream_);

  // Store to CTE or keep in host RAM
#ifdef WRP_CTE_AVAILABLE
  if (use_cte_) {
    std::string blob_name = "page_" + std::to_string(page_index);
    hipc::FullPtr<char> shm = CHI_IPC->AllocateBuffer(page_size_);
    memcpy(shm.ptr_, host_buf, page_size_);
    auto future = cte_tag_->AsyncPutBlob(blob_name, shm.shm_, page_size_);
    future.Wait();
    CHI_IPC->FreeBuffer(shm);
    cudaFreeHost(host_buf);
  } else
#endif
  {
    host_backing_store_[page_index] = host_buf;
  }

  // Unmap and release
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
  entry.evicted_to_host = true;

  return CUDA_SUCCESS;
}

void GpuVirtualMemoryManager::prefetchAhead(size_t page_index) {
  for (size_t i = 1; i <= prefetch_window_; ++i) {
    size_t target = page_index + i;
    if (target >= total_pages_) break;
    if (isMapped(target)) continue;
    touchPageAsync(target);
  }
}

CUdeviceptr GpuVirtualMemoryManager::getPagePtr(size_t page_index) const {
  if (page_index >= total_pages_) return 0;
  return va_base_ + page_index * page_size_;
}

void GpuVirtualMemoryManager::syncTransfer() {
  cudaStreamSynchronize(transfer_stream_);
}

void GpuVirtualMemoryManager::syncCompute() {
  cudaStreamSynchronize(compute_stream_);
}

void GpuVirtualMemoryManager::freeHostBackingStore_() {
  for (auto &pair : host_backing_store_) {
    cudaFreeHost(pair.second);
  }
  host_backing_store_.clear();
}

}  // namespace wrp_cte::uvm
