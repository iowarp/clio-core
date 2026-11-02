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

#ifndef GPU_MALLOC_H
#define GPU_MALLOC_H

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM

#include <string>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/errors.h"
#include "clio_ctp/util/gpu_api.h"
#include "clio_ctp/util/logging.h"
#include "memory_backend.h"

namespace ctp::ipc {

/**
 * GPU-only memory backend using cudaMalloc/hipMalloc
 *
 * Memory layout (all in GPU memory):
 *   region_ -> [4KB MemoryBackendHeader | Data]
 *
 * All memory is allocated with cudaMalloc/hipMalloc on GPU.
 * IPC handle is obtained on-demand via GpuApi::GetIpcMemHandle.
 */
class GpuMalloc : public MemoryBackend, public UrlMemoryBackend {
 protected:
  std::string url_;  // Identifier for this backend (not used for shm)

 public:
  /** Constructor */
  CTP_CROSS_FUN
  GpuMalloc() = default;

  /** Destructor */
  ~GpuMalloc() {
    if (IsOwner()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /**
   * Initialize backend with GPU memory
   *
   * @param backend_id Unique identifier for this backend
   * @param data_size Size of GPU data buffer
   * @param url Identifier for this backend (not used for shared memory)
   * @param gpu_id GPU device ID
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t data_size,
                const std::string &url, int gpu_id = 0) {
    // Enforce minimum data size of 1MB
    constexpr size_t kMinDataSize = 1024 * 1024;  // 1MB
    if (data_size < kMinDataSize) {
      data_size = kMinDataSize;
    }

    // Initialize flags before calling methods that use it
    flags_.Clear();
    url_ = url;

    // Allocate GPU memory
    region_ = GpuApi::Malloc<char>(data_size);
    if (!region_) {
      HLOG(kError, "Failed to allocate GPU memory");
      return false;
    }

    header_ = reinterpret_cast<MemoryBackendHeader *>(region_);
    data_ = region_ + kBackendHeaderSize;

    id_ = backend_id;
    backend_size_ = data_size;
    data_capacity_ = data_size - kBackendHeaderSize;
    data_id_ = gpu_id;

    // Write header into GPU memory
    MemoryBackendHeader hdr;
    hdr.id_ = id_;
    hdr.flags_ = flags_;
    hdr.backend_size_ = backend_size_;
    hdr.data_capacity_ = data_capacity_;
    hdr.data_id_ = data_id_;
    GpuApi::Memcpy(header_, &hdr, sizeof(MemoryBackendHeader));

    // Mark this process as the owner of the backend
    SetOwner();

    return true;
  }

  /**
   * Attach to existing GPU memory backend via IPC handle
   *
   * Opens a CUDA/HIP IPC memory handle from another process and
   * reads the backend header to recover metadata.
   *
   * @param ipc_handle IPC handle obtained from the owning process
   * @return true on success, false on failure
   */
  bool shm_attach_ipc(const GpuIpcMemHandle &ipc_handle) {
    flags_.Clear();

    // Open IPC handle to get mapped pointer in this process
    GpuApi::OpenIpcMemHandle(const_cast<GpuIpcMemHandle &>(ipc_handle),
                             &region_);
    if (!region_) {
      HLOG(kError, "GpuMalloc::shm_attach_ipc: Failed to open IPC handle");
      return false;
    }

    // Read header from GPU memory
    header_ = reinterpret_cast<MemoryBackendHeader *>(region_);
    data_ = region_ + kBackendHeaderSize;

    MemoryBackendHeader hdr;
    GpuApi::Memcpy(&hdr, header_, sizeof(MemoryBackendHeader));
    id_ = hdr.id_;
    backend_size_ = hdr.backend_size_;
    data_capacity_ = hdr.data_capacity_;
    data_id_ = hdr.data_id_;

    UnsetOwner();
    flags_.SetBits(MEMORY_BACKEND_INITIALIZED);
    return true;
  }

  /**
   * Attach to existing GPU memory backend (URL-based, not implemented)
   */
  bool shm_attach(const std::string &url) {
    flags_.Clear();
    url_ = url;
    HLOG(kError, "GpuMalloc::shm_attach requires IPC handle (use shm_attach_ipc instead)");
    return false;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Detach from memory (closes IPC handle for non-owner) */
  void _Detach() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }

    // Close IPC handle if we mapped via shm_attach_ipc
    if (region_) {
      GpuApi::CloseIpcMemHandle(region_);
    }

    region_ = nullptr;
    header_ = nullptr;
    data_ = nullptr;

    flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED);
  }

  /** Destroy memory */
  void _Destroy() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }

    // Free entire GPU region (includes headers and data)
    if (region_) {
      GpuApi::Free(region_);
      region_ = nullptr;
      header_ = nullptr;
      data_ = nullptr;
    }

    flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED);
  }
};

}  // namespace ctp::ipc

#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM

#endif  // GPU_MALLOC_H
