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

class GpuShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  std::string url_;
  size_t total_size_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  GpuShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~GpuShmMmap() {
#if HSHM_IS_HOST
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
#endif
  }

  /** Initialize shared memory */
  bool shm_init(const MemoryBackendId& backend_id, size_t size,
                const std::string& url, int gpu_id = 0) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (size < kMinBackendSize) {
      size = kMinBackendSize;
    }

    // Initialize flags before calling methods that use it
    flags_.Clear();
    Own();

    // Calculate sizes: header + md section + alignment + data section
    constexpr size_t kAlignment = 4096;  // 4KB alignment
    size_t header_size = sizeof(MemoryBackendHeader);
    size_t md_size = header_size;  // md section stores the header
    size_t aligned_md_size = ((md_size + kAlignment - 1) / kAlignment) * kAlignment;
    total_size_ = aligned_md_size + size;

    // Create shared memory
    SystemInfo::DestroySharedMemory(url);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url, total_size_)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Map the entire shared memory region as one contiguous block
    char *ptr = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, total_size_, 0));
    if (!ptr) {
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }

    // Layout: [MemoryBackendHeader | padding to 4KB] [accel_data]
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    new (header_) MemoryBackendHeader();
    header_->id_ = backend_id;
    header_->md_size_ = md_size;
    header_->accel_data_size_ = size;
    header_->accel_id_ = gpu_id;
    header_->flags_.Clear();

    // md_ points to the header itself (metadata for process connection)
    md_ = ptr;
    md_size_ = md_size;

    // accel_data_ starts at 4KB aligned boundary after md section
    accel_data_ = ptr + aligned_md_size;
    accel_data_size_ = size;
    accel_id_ = gpu_id;

    // Register both metadata and data sections with GPU
    Register(md_, aligned_md_size);
    Register(accel_data_, size);

    return true;
  }

  /** SHM deserialize */
  bool shm_attach(const std::string& url) {
    flags_.Clear();
    Disown();

    if (!SystemInfo::OpenSharedMemory(fd_, url)) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // First, map just the header to get the size information
    constexpr size_t kAlignment = 4096;
    char *ptr = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, kAlignment, 0));
    if (!ptr) {
      return false;
    }

    // Calculate total size based on header information
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    size_t md_size = header_->md_size_;
    size_t aligned_md_size = ((md_size + kAlignment - 1) / kAlignment) * kAlignment;
    total_size_ = aligned_md_size + header_->accel_data_size_;

    // Unmap the header
    SystemInfo::UnmapMemory(ptr, kAlignment);

    // Map the entire region
    ptr = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, total_size_, 0));
    if (!ptr) {
      return false;
    }

    // Set up pointers
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    md_ = ptr;
    md_size_ = header_->md_size_;
    accel_data_ = ptr + aligned_md_size;
    accel_data_size_ = header_->accel_data_size_;
    accel_id_ = header_->accel_id_;

    // Register both metadata and data sections with GPU
    Register(md_, aligned_md_size);
    Register(accel_data_, accel_data_size_);

    return true;
  }

  /** Map shared memory */
  template <typename T>
  void Register(T* ptr, size_t size) {
    GpuApi::RegisterHostMemory(ptr, size);
  }

  /** Detach shared memory */
  void _Detach() {
      return;
    }
    // Unregister GPU memory
    if (md_) {
      GpuApi::UnregisterHostMemory(md_);
    }
    if (accel_data_ && accel_data_ != md_) {
      GpuApi::UnregisterHostMemory(accel_data_);
    }
    // Unmap the entire contiguous region
    if (header_) {
      SystemInfo::UnmapMemory(reinterpret_cast<void *>(header_), total_size_);
    }
    SystemInfo::CloseSharedMemory(fd_);
  }

  /** Destroy shared memory */
  void _Destroy() {
      return;
    }
    _Detach();
    SystemInfo::DestroySharedMemory(url_);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_GPU_SHM_MMAP_H
