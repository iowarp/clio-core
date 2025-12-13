/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

/**
 * GPU-accessible shared memory backend using POSIX shared memory with CUDA/ROCm IPC
 *
 * Similar to PosixShmMmap but registers memory with GPU for IPC access.
 * Memory layout in file:
 *   [4KB backend header] [4KB shared header] [data]
 *
 * Memory layout in virtual memory (region_):
 *   [4KB private header] [4KB shared header] [data]
 */
class GpuShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  std::string url_;
  GpuIpcMemHandle ipc_handle_;  // IPC handle for GPU memory sharing

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  GpuShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~GpuShmMmap() {
#if HSHM_IS_HOST
    if (IsOwner()) {
      _Destroy();
    } else {
      _Detach();
    }
#endif
  }

  /**
   * Initialize backend with GPU-accessible shared memory
   *
   * @param backend_id Unique identifier for this backend
   * @param backend_size Total size of the region (including headers and data)
   * @param url POSIX shared memory object name (e.g., "/my_gpu_shm")
   * @param gpu_id GPU device ID to register memory with
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size,
                const std::string &url, int gpu_id = 0) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }

    // File layout: [4KB backend header] [4KB shared header] [data]
    size_t shared_size = backend_size - kBackendHeaderSize;

    // Create shared memory object with entire backend size
    SystemInfo::DestroySharedMemory(url);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url, backend_size)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Map the first 4KB (backend header) using MapShared
    header_ = reinterpret_cast<MemoryBackendHeader *>(
        SystemInfo::MapSharedMemory(fd_, kBackendHeaderSize, 0));
    if (!header_) {
      HILOG(kError, "Failed to map header");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Map the entire region using MapMixed
    // Private portion: 4KB (private header)
    // Shared portion: backend_size - 4KB (shared header + data)
    // Offset into file: 0 (maps entire file starting from offset 0)
    region_ = reinterpret_cast<char *>(
        SystemInfo::MapMixedMemory(fd_, kBackendHeaderSize, shared_size, 0));
    if (!region_) {
      HILOG(kError, "Failed to create mixed mapping");
      SystemInfo::UnmapMemory(header_, kBackendHeaderSize);
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Calculate pointers
    char *shared_header_ptr = region_ + kBackendHeaderSize;
    data_ = shared_header_ptr + kBackendHeaderSize;

    // Initialize backend header fields
    id_ = backend_id;
    backend_size_ = backend_size;
    data_capacity_ = backend_size - 2 * kBackendHeaderSize;
    data_id_ = gpu_id;
    priv_header_off_ = static_cast<size_t>(data_ - region_);
    flags_.Clear();

    // Set GPU-only flag since this backend requires GPU IPC
    SetGpuOnly();

    // Copy all header fields to shared header
    new (header_) MemoryBackendHeader();
    (*header_) = (const MemoryBackendHeader&)*this;

    // Register memory with GPU for IPC
    _RegisterWithGpu(region_, backend_size);

    // Get IPC handle for this memory region
    GpuApi::GetIpcMemHandle(ipc_handle_, (void *)data_);

    // Mark this process as the owner
    SetOwner();

    return true;
  }

  /**
   * Attach to existing GPU-accessible shared memory
   *
   * @param url POSIX shared memory object name
   * @return true on success, false on failure
   */
  bool shm_attach(const std::string &url) {
    flags_.Clear();

    if (!SystemInfo::OpenSharedMemory(fd_, url)) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Map the first 4KB (backend header) to read size information
    header_ = reinterpret_cast<MemoryBackendHeader *>(
        SystemInfo::MapSharedMemory(fd_, kBackendHeaderSize, 0));
    if (!header_) {
      HILOG(kError, "Failed to map header");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Get backend size from header
    size_t backend_size = header_->backend_size_;
    size_t shared_size = backend_size - kBackendHeaderSize;

    // Map the entire region using MapMixed
    region_ = reinterpret_cast<char *>(
        SystemInfo::MapMixedMemory(fd_, kBackendHeaderSize, shared_size, 0));
    if (!region_) {
      HILOG(kError, "Failed to create mixed mapping");
      SystemInfo::UnmapMemory(header_, kBackendHeaderSize);
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Calculate pointers
    char *shared_header_ptr = region_ + kBackendHeaderSize;
    data_ = shared_header_ptr + kBackendHeaderSize;

    // Copy header fields to local object
    id_ = header_->id_;
    backend_size_ = header_->backend_size_;
    data_capacity_ = header_->data_capacity_;
    data_id_ = header_->data_id_;
    priv_header_off_ = header_->priv_header_off_;
    flags_ = header_->flags_;

    // Register memory with GPU for IPC
    _RegisterWithGpu(region_, backend_size);

    // Mark this process as NOT the owner
    UnsetOwner();

    return true;
  }

  /** Detach from shared memory */
  void shm_detach() { _Detach(); }

  /** Destroy shared memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /**
   * Register memory region with GPU for IPC access
   */
  void _RegisterWithGpu(void *ptr, size_t size) {
    GpuApi::RegisterHostMemory(ptr, size);
  }

  /**
   * Unregister memory region from GPU
   */
  void _UnregisterFromGpu(void *ptr) {
    GpuApi::UnregisterHostMemory(ptr);
  }

  /** Detach from shared memory */
  void _Detach() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }

    // Unregister from GPU
    if (region_) {
      _UnregisterFromGpu(region_);
    }

    // Unmap memory
    if (region_) {
      SystemInfo::UnmapMemory(region_, backend_size_);
      region_ = nullptr;
      data_ = nullptr;
    }
    if (header_) {
      SystemInfo::UnmapMemory(header_, kBackendHeaderSize);
      header_ = nullptr;
    }

    // Close file descriptor
    SystemInfo::CloseSharedMemory(fd_);

    flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED);
  }

  /** Destroy shared memory */
  void _Destroy() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }

    _Detach();
    SystemInfo::DestroySharedMemory(url_);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H
