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

#include "clio_cte/uvm/gpu_vmm.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <clio_cte/core/core_client.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace clio::cte::uvm {

/** One CTE staging buffer still read by an async H2D copy. */
struct PendingFree {
  ctp::ipc::FullPtr<char> shm;  ///< the staging buffer
  cudaEvent_t done = nullptr;   ///< recorded after the copy on the stream
};

struct GpuVirtualMemoryManager::PendingFrees {
  std::vector<PendingFree> list;
};

GpuVirtualMemoryManager::GpuVirtualMemoryManager()
    : pending_frees_(std::make_unique<PendingFrees>()) {}

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

  // Hold our own reference on the device's primary context: destroy()
  // releases one. Without this retain, that release dropped a reference
  // GpuVmm never took -- the CUDA runtime's -- and could destroy the primary
  // context, freeing every pinned allocation in it (the clio runtime's
  // gpu2cpu queue arena included, which its GPU worker then segfaulted
  // reading, #1216).
  CUcontext primary = nullptr;
  res = cuDevicePrimaryCtxRetain(&primary, device_);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "GpuVmm: cuDevicePrimaryCtxRetain failed: %d\n", res);
    return res;
  }
  ctx_retained_ = true;
  cuCtxSetCurrent(primary);
  // Every failure from here on gives that reference back.
  auto fail = [this](CUresult r) {
    cuDevicePrimaryCtxRelease(device_);
    ctx_retained_ = false;
    return r;
  };

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
    return fail(res);
  }

  // CTE backing store, when requested. It used to sit behind a macro nothing
  // defined, so use_cte=true was silently ignored (#1190); a requested CTE
  // that cannot be reached is now an init failure, not a quiet fallback.
  use_cte_ = config.use_cte;
  if (use_cte_) {
    if (!clio::cte::core::CLIO_CTE_CLIENT_INIT("",
                                               clio::run::PoolQuery::Local())) {
      fprintf(stderr, "GpuVmm: use_cte=true but the CTE client could not be "
              "initialized (is the runtime up?)\n");
      return fail(CUDA_ERROR_NOT_INITIALIZED);
    }
    cte_tag_ = std::make_unique<clio::cte::core::Tag>(config.cte_tag_name);
    if (cte_tag_->GetTagId().IsNull()) {
      fprintf(stderr, "GpuVmm: could not create CTE tag %s\n",
              config.cte_tag_name.c_str());
      cte_tag_.reset();
      return fail(CUDA_ERROR_NOT_INITIALIZED);
    }
    fprintf(stdout, "GpuVmm: CTE backing store enabled (tag: %s)\n",
            config.cte_tag_name.c_str());
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
    cte_tag_.reset();
    return fail(res);
  }

  // Initialize the software page table (all pages start unmapped)
  page_table_.resize(total_pages_);

  // Create CUDA streams for async overlap
  cudaStreamCreate(&transfer_stream_);
  cudaStreamCreate(&compute_stream_);

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

  // In-flight async restores still read their staging buffers.
  drainPendingFrees_(/*wait=*/true);

  // Unmap and free all backed pages; drop the page blobs this run stored
  // (they used to outlive the manager, #1190).
  for (size_t i = 0; i < total_pages_; ++i) {
    PageEntry &entry = page_table_[i];
    if (entry.mapped) {
      unmapPage_(i);
    }
    if (entry.in_cte && cte_tag_) {
      auto *cte = CLIO_CTE_CLIENT;
      auto del = cte->AsyncDelBlob(cte_tag_->GetTagId(), pageBlobName_(i));
      del.Wait();
      entry.in_cte = false;
    }
  }

  // Free all host backing store buffers
  freeHostBackingStore_();

  cte_tag_.reset();
  use_cte_ = false;

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

  // Release the primary-context reference init() took (and only that
  // one).
  if (ctx_retained_) {
    cuDevicePrimaryCtxRelease(device_);
    ctx_retained_ = false;
  }
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

void GpuVirtualMemoryManager::unmapPage_(size_t page_index) {
  PageEntry &entry = page_table_[page_index];
  CUdeviceptr page_addr = va_base_ + page_index * page_size_;
  cuMemUnmap(page_addr, page_size_);
  cuMemRelease(entry.alloc_handle);
  entry.alloc_handle = 0;
  entry.mapped = false;
}

std::string GpuVirtualMemoryManager::pageBlobName_(size_t page_index) {
  return "page_" + std::to_string(page_index);
}

CUresult GpuVirtualMemoryManager::restoreFromCte_(size_t page_index,
                                                  bool async) {
  CUdeviceptr page_addr = va_base_ + page_index * page_size_;
  auto *ipc = CLIO_CPU_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(page_size_);
  if (shm.IsNull()) {
    fprintf(stderr, "GpuVmm: no staging buffer to restore page %zu\n",
            page_index);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  auto *cte = CLIO_CTE_CLIENT;
  auto get = cte->AsyncGetBlob(cte_tag_->GetTagId(), pageBlobName_(page_index),
                               0, page_size_, 0,
                               shm.shm_.template Cast<void>());
  get.Wait();
  if (get->GetReturnCode() != 0) {
    fprintf(stderr, "GpuVmm: reading page %zu from CTE failed (rc %u)\n",
            page_index, get->GetReturnCode());
    ipc->FreeBuffer(shm);
    return CUDA_ERROR_UNKNOWN;
  }
  if (!async) {
    cudaError_t err = cudaMemcpy((void *)page_addr, shm.ptr_, page_size_,
                                 cudaMemcpyHostToDevice);
    ipc->FreeBuffer(shm);
    return err == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  cudaError_t err = cudaMemcpyAsync((void *)page_addr, shm.ptr_, page_size_,
                                    cudaMemcpyHostToDevice, transfer_stream_);
  if (err != cudaSuccess) {
    ipc->FreeBuffer(shm);
    return CUDA_ERROR_UNKNOWN;
  }
  // The copy reads shm until it runs: free it once an event recorded after
  // the copy has fired, not now (#1190).
  PendingFree pf;
  pf.shm = shm;
  if (cudaEventCreateWithFlags(&pf.done, cudaEventDisableTiming) !=
          cudaSuccess ||
      cudaEventRecord(pf.done, transfer_stream_) != cudaSuccess) {
    // No event to wait on: wait for the stream itself.
    if (pf.done != nullptr) cudaEventDestroy(pf.done);
    cudaStreamSynchronize(transfer_stream_);
    ipc->FreeBuffer(shm);
    return CUDA_SUCCESS;
  }
  pending_frees_->list.push_back(pf);
  return CUDA_SUCCESS;
}

void GpuVirtualMemoryManager::drainPendingFrees_(bool wait) {
  auto *ipc = CLIO_CPU_IPC;
  auto &list = pending_frees_->list;
  size_t kept = 0;
  for (auto &pf : list) {
    cudaError_t st =
        wait ? cudaEventSynchronize(pf.done) : cudaEventQuery(pf.done);
    if (st == cudaErrorNotReady) {
      list[kept++] = pf;
      continue;
    }
    cudaEventDestroy(pf.done);
    ipc->FreeBuffer(pf.shm);
  }
  list.resize(kept);
}

CUresult GpuVirtualMemoryManager::populatePage_(size_t page_index,
                                                bool async) {
  PageEntry &entry = page_table_[page_index];
  CUdeviceptr page_addr = va_base_ + page_index * page_size_;
  CUresult res = CUDA_SUCCESS;
  auto it = host_backing_store_.find(page_index);
  if (entry.evicted_to_host && it != host_backing_store_.end()) {
    // Restore saved data from host RAM. The sync path frees the buffer; the
    // async path keeps it alive (reused by the next eviction).
    cudaError_t err =
        async ? cudaMemcpyAsync((void *)page_addr, it->second, page_size_,
                                cudaMemcpyHostToDevice, transfer_stream_)
              : cudaMemcpy((void *)page_addr, it->second, page_size_,
                           cudaMemcpyHostToDevice);
    res = (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    if (res == CUDA_SUCCESS && !async) {
      cudaFreeHost(it->second);
      host_backing_store_.erase(it);
    }
  } else if (entry.evicted_to_host && entry.in_cte) {
    res = restoreFromCte_(page_index, async);
  } else {
    // Fresh page: fill with configured value using driver API memset.
    // cuMemsetD32 uses the same driver API context as cuMemMap/cuMemSetAccess,
    // avoiding the runtime/driver context mismatch that causes fillKernel
    // writes to appear as 0 when read back (A100 / Polaris).
    size_t num_ints = page_size_ / sizeof(int);
    res = async ? cuMemsetD32Async(page_addr, (unsigned int)fill_value_,
                                   num_ints, transfer_stream_)
                : cuMemsetD32(page_addr, (unsigned int)fill_value_, num_ints);
  }
  if (res != CUDA_SUCCESS) {
    // Report it (it used to be logged and then answered with success) and
    // leave the page unmapped with its saved copy intact for a retry.
    fprintf(stderr, "GpuVmm: populating page %zu failed: %d\n", page_index,
            res);
    unmapPage_(page_index);
    return res;
  }
  entry.evicted_to_host = false;
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

    // Allocate + map + set access, then restore or fill
    CUresult res = mapAndBackPage_(page_index);
    if (res != CUDA_SUCCESS) return res;
    res = populatePage_(page_index, /*async=*/false);
    if (res != CUDA_SUCCESS) return res;
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
  drainPendingFrees_(/*wait=*/false);

  PageEntry &entry = page_table_[page_index];
  if (entry.mapped) {
    return CUDA_SUCCESS;
  }

  CUresult res = mapAndBackPage_(page_index);
  if (res != CUDA_SUCCESS) return res;
  return populatePage_(page_index, /*async=*/true);
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

bool GpuVirtualMemoryManager::saveToCte_(size_t page_index,
                                         const char *host_buf) {
  auto *ipc = CLIO_CPU_IPC;
  ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(page_size_);
  if (shm.IsNull()) return false;
  memcpy(shm.ptr_, host_buf, page_size_);
  auto put = cte_tag_->AsyncPutBlob(pageBlobName_(page_index),
                                    shm.shm_.template Cast<void>(),
                                    page_size_);
  put.Wait();
  const bool ok = put->GetReturnCode() == 0;
  ipc->FreeBuffer(shm);
  if (!ok) {
    fprintf(stderr, "GpuVmm: storing page %zu in CTE failed (rc %u); "
            "keeping it in host RAM\n", page_index, put->GetReturnCode());
  }
  return ok;
}

void GpuVirtualMemoryManager::storeEvicted_(size_t page_index,
                                            char *host_buf) {
  PageEntry &entry = page_table_[page_index];
  if (use_cte_ && saveToCte_(page_index, host_buf)) {
    entry.in_cte = true;
    host_backing_store_.erase(page_index);
    cudaFreeHost(host_buf);
    return;
  }
  // Host RAM holds the newest copy and wins the restore (populatePage_
  // checks it first); an older blob stays flagged so destroy() removes it.
  host_backing_store_[page_index] = host_buf;
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
  storeEvicted_(page_index, host_buf);

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
  drainPendingFrees_(/*wait=*/false);

  // Store to CTE or keep in host RAM
  storeEvicted_(page_index, host_buf);

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
  std::lock_guard<std::mutex> lock(mutex_);
  drainPendingFrees_(/*wait=*/false);
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

}  // namespace clio::cte::uvm
