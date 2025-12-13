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

#ifndef GPU_MALLOC_H
#define GPU_MALLOC_H

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

/**
 * Header extension for GpuMalloc backend
 * Stores IPC handle for GPU memory sharing
 */
struct GpuMallocPrivateHeader {
  GpuIpcMemHandle ipc_handle_;  // IPC handle for data_ buffer
};

/**
 * GPU-only memory backend using cudaMalloc/hipMalloc
 *
 * Memory layout:
 *   - Header (MemoryBackendHeader): Allocated with regular malloc on host
 *   - Private header (GpuMallocPrivateHeader): Also allocated with malloc, stores IPC handle
 *   - Data: Allocated with cudaMalloc/hipMalloc on GPU
 *
 * The header and private header are stored in regular host memory for metadata access,
 * while the actual data buffer is pure GPU memory accessible via IPC.
 */
class GpuMalloc : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  std::string url_;
  size_t metadata_size_;  // Size of shared memory for metadata only

 public:
  /** Constructor */
  HSHM_CROSS_FUN
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
   * @param url POSIX shared memory name for metadata sharing
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

    // Calculate metadata size: backend header + private header
    metadata_size_ = 2 * kBackendHeaderSize;

    // Create shared memory for metadata only
    SystemInfo::DestroySharedMemory(url);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url, metadata_size_)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Map shared memory for metadata
    region_ = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, metadata_size_, 0));
    if (!region_) {
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }

    // Layout in region_: [MemoryBackendHeader | GpuMallocPrivateHeader]
    header_ = reinterpret_cast<MemoryBackendHeader *>(region_);
    GpuMallocPrivateHeader *priv_header =
        reinterpret_cast<GpuMallocPrivateHeader *>(region_ + kBackendHeaderSize);

    // Initialize headers
    new (header_) MemoryBackendHeader();
    new (priv_header) GpuMallocPrivateHeader();

    // Set backend header fields
    header_->id_ = backend_id;
    header_->backend_size_ = metadata_size_ + data_size;  // Total logical size
    header_->data_capacity_ = data_size;
    header_->data_id_ = gpu_id;
    header_->priv_header_off_ = kBackendHeaderSize;
    header_->flags_.Clear();

    // Copy to local object
    id_ = backend_id;
    backend_size_ = header_->backend_size_;
    data_capacity_ = data_size;
    data_id_ = gpu_id;
    priv_header_off_ = kBackendHeaderSize;

    // Allocate GPU memory separately using cudaMalloc/hipMalloc
    data_ = GpuApi::Malloc<char>(data_size);
    if (!data_) {
      HILOG(kError, "Failed to allocate GPU memory");
      SystemInfo::UnmapMemory(region_, metadata_size_);
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Get IPC handle for GPU memory and store in private header
    GpuApi::GetIpcMemHandle(priv_header->ipc_handle_, (void *)data_);

    // Set GPU-only flag
    SetGpuOnly();
    header_->flags_ = flags_;

    // Mark this process as the owner of the backend
    SetOwner();

    return true;
  }

  /**
   * Attach to existing GPU memory backend
   *
   * @param url POSIX shared memory name for metadata
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

    // Map metadata region
    metadata_size_ = 2 * kBackendHeaderSize;
    region_ = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, metadata_size_, 0));
    if (!region_) {
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Get pointers to headers
    header_ = reinterpret_cast<MemoryBackendHeader *>(region_);
    GpuMallocPrivateHeader *priv_header =
        reinterpret_cast<GpuMallocPrivateHeader *>(region_ + kBackendHeaderSize);

    // Copy header fields to local object
    id_ = header_->id_;
    backend_size_ = header_->backend_size_;
    data_capacity_ = header_->data_capacity_;
    data_id_ = header_->data_id_;
    priv_header_off_ = header_->priv_header_off_;
    flags_ = header_->flags_;

    // Open GPU memory from IPC handle
    // Open IPC handle for GPU memory
    GpuApi::OpenIpcMemHandle(priv_header->ipc_handle_, (void **)&data_);

    // Mark this process as NOT the owner (attaching to existing backend)
    UnsetOwner();

    return true;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Detach from memory */
  void _Detach() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }

    // Clear GPU memory pointer (don't free, we're not the owner)
    data_ = nullptr;

    // Unmap the metadata region
    if (region_) {
      SystemInfo::UnmapMemory(region_, metadata_size_);
      region_ = nullptr;
      header_ = nullptr;
    }

    SystemInfo::CloseSharedMemory(fd_);
    flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED);
  }

  /** Destroy memory */
  void _Destroy() {
    if (!flags_.Any(MEMORY_BACKEND_INITIALIZED)) {
      return;
    }

    // Free GPU memory (owner only)
    if (data_) {
      GpuApi::Free(data_);
      data_ = nullptr;
    }

    // Unmap and destroy shared metadata
    if (region_) {
      SystemInfo::UnmapMemory(region_, metadata_size_);
      region_ = nullptr;
      header_ = nullptr;
    }

    SystemInfo::CloseSharedMemory(fd_);
    SystemInfo::DestroySharedMemory(url_);
    flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#endif  // GPU_MALLOC_H
