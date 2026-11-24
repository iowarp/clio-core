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

#ifndef CTP_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H
#define CTP_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/errors.h"
#include "clio_ctp/util/gpu_api.h"
#include "clio_ctp/util/logging.h"
#include "memory_backend.h"

namespace ctp::ipc {

/**
 * GPU+CPU coherent shared memory backend using CUDA Unified Virtual Memory.
 *
 * Uses cudaMallocManaged (UVM) so both CPU and GPU can access the same memory
 * with hardware-managed cache coherence.  This avoids the need for clflush,
 * write-combining tricks, or system-scope atomics that require
 * cudaDevAttrHostNativeAtomicSupported (not available on all discrete GPUs).
 *
 * On GPUs with ConcurrentManagedAccess (SM 7.0+), the hardware migration
 * engine handles coherence transparently so std::atomic on CPU and device-
 * scope atomics on GPU both see each other's writes immediately.
 *
 * Not shareable across processes — single-process use only.
 *
 * Memory layout:
 *   region_ -> [4KB MemoryBackendHeader | Data]
 */
class GpuShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  std::string url_;

 public:
  /** Constructor */
  CTP_CROSS_FUN
  GpuShmMmap() {}

  /** Destructor */
  CTP_CROSS_FUN
  ~GpuShmMmap() {
#if CTP_IS_HOST
    _Destroy();
#endif
  }

  /**
   * Initialize UVM backend accessible from both CPU and GPU.
   *
   * @param backend_id Unique identifier for this backend
   * @param backend_size Total size in bytes (headers + data)
   * @param url Identifier string (informational only)
   * @param gpu_id GPU device ID (informational only)
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size,
                const std::string &url, int gpu_id = 0) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }
    url_ = url;
    (void)gpu_id;

    // Allocate pinned host memory accessible from both CPU and GPU.
    //
    // Why cudaMallocHost instead of cudaMallocManaged:
    //   - cudaMallocManaged with SetPreferredLocation=CPU does NOT provide
    //     cache-coherent GPU→CPU visibility on PCIe-only systems (e.g.,
    //     Polaris: AMD EPYC + A100 without NVLink to host). PCIe writes from
    //     GPU bypass the CPU cache, so atomicExch_system writes are invisible
    //     to CPU loads until an explicit page migration occurs.
    //   - cudaMallocHost allocates pages locked in CPU DRAM. With CUDA Unified
    //     Virtual Addressing (UVA), the same pointer is valid in both host code
    //     and GPU kernels at the same virtual address. GPU PCIe DMA writes to
    //     pinned pages DO participate in cache snooping on x86, making
    //     system-scope atomics (atomicExch_system, atomicAdd_system) immediately
    //     visible to CPU reads after cudaDeviceSynchronize() or via polling.
    // Route through GpuApi so each backend (CUDA, HIP, SYCL) uses its
    // native pinned-host allocator: cudaMallocHost / hipHostMalloc /
    // sycl::malloc_host respectively.
    void *pinned_ptr = GpuApi::MallocHost<char>(backend_size);
    if (!pinned_ptr) {
      HLOG(kError, "GpuApi::MallocHost failed (backend_size={})",
           backend_size);
      return false;
    }

    // Zero-initialize so offset-based allocators start from a clean state.
    memset(pinned_ptr, 0, backend_size);

    // Layout: [kBackendHeaderSize header] [data]
    region_ = reinterpret_cast<char*>(pinned_ptr);
    header_ = reinterpret_cast<MemoryBackendHeader *>(region_);
    data_ = region_ + kBackendHeaderSize;

    // Initialize backend header fields
    id_ = backend_id;
    backend_size_ = backend_size;
    data_capacity_ = backend_size - kBackendHeaderSize;
    data_id_ = gpu_id;
    flags_.Clear();

    // Copy header fields into the managed memory header region
    new (header_) MemoryBackendHeader();
    (*header_) = (const MemoryBackendHeader &)*this;

    // Mark as initialized and owned
    flags_.SetBits(MEMORY_BACKEND_INITIALIZED);
    SetOwner();
    return true;
  }

  /**
   * Attach — not supported (single-process UVM, no cross-process sharing).
   * Returns false. Kept for API compatibility.
   */
  bool shm_attach(const std::string &url) {
    HLOG(kError, "GpuShmMmap::shm_attach not supported (UVM is process-local)");
    return false;
  }

  /** Detach / free */
  void shm_detach() { _Destroy(); }
  void shm_destroy() { _Destroy(); }

 protected:
  /**
   * Unregister memory region from GPU
   */
  void _UnregisterFromGpu(void *ptr) {
    GpuApi::UnregisterHostMemory(ptr);
  }

  /** Detach from (and free) pinned host memory */
  void _Detach() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }

    // Free pinned host memory via the same backend used at shm_init.
    if (region_) {
      GpuApi::FreeHost(region_);
      region_ = nullptr;
      header_ = nullptr;
      data_ = nullptr;
    }

    flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED);
  }

  /** Destroy pinned memory (same as detach for cudaMallocHost) */
  void _Destroy() { _Detach(); }
};

}  // namespace ctp::ipc

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL

#endif  // CTP_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H
