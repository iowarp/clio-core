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

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#ifndef _WIN32
#include <fcntl.h>
#endif
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

class PosixShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  std::string url_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  PosixShmMmap() : fd_{} {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~PosixShmMmap() = default;

  /**
   * Initialize backend with mixed private/shared mapping
   *
   * @param backend_id Unique identifier for this backend
   * @param backend_size Total size of the region (including all headers and
   * data)
   * @param url POSIX shared memory object name (e.g., "/my_shm")
   * @return true on success, false on failure
   *
   * Memory layout in file:
   *   [4KB backend header] [4KB private header] [4KB shared header] [data]
   *
   * Memory layout in virtual memory:
   *   [4KB backend header MAP_SHARED] [4KB private header MAP_PRIVATE] [4KB
   * shared header MAP_SHARED] [data MAP_SHARED]
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size,
                const std::string &url) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }

    // File layout: [4KB backend header] [4KB private header] [4KB shared
    // header] [data] Total file size includes all three headers plus data
    size_t total_file_size =
        backend_size + kBackendHeaderSize;  // backend_size already includes
                                            // private + shared headers + data

    // Create shared memory object with entire size
    SystemInfo::DestroySharedMemory(url);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url, total_file_size)) {
      char *err_buf = strerror(errno);
      HLOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Step 1: Map the backend header (4KB) as MAP_SHARED at offset 0
    header_ = reinterpret_cast<MemoryBackendHeader *>(
        SystemInfo::MapSharedMemory(fd_, kBackendHeaderSize, 0));
    if (!header_) {
      HLOG(kError, "Failed to map backend header");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Step 2: Map the rest using MapMixedMemory at offset kBackendHeaderSize
    // Private portion: 4KB (private header)
    // Shared portion: backend_size - kBackendHeaderSize (shared header + data)
    // Offset into file: kBackendHeaderSize (skip the backend header in file)
    size_t shared_portion_size = backend_size - kBackendHeaderSize;
    region_ = reinterpret_cast<char *>(SystemInfo::MapMixedMemory(
        fd_, kBackendHeaderSize, shared_portion_size, kBackendHeaderSize));
    if (!region_) {
      HLOG(kError, "Failed to create mixed mapping");
      SystemInfo::UnmapMemory(header_, kBackendHeaderSize);
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Memory layout after mapping:
    // header_ points to: [4KB backend header MAP_SHARED] at file offset 0
    // region_ points to: [4KB private header MAP_PRIVATE] [4KB shared header
    // MAP_SHARED] [data MAP_SHARED]
    //                     where shared portion maps to file offset
    //                     kBackendHeaderSize

    // Calculate data pointer
    // region_[0..4KB) = private header
    // region_[4KB..8KB) = shared header (maps to file offset 4KB..8KB)
    // region_[8KB...) = data (maps to file offset 8KB...)
    data_ = region_ + 2 * kBackendHeaderSize;

    // Initialize local fields
    id_ = backend_id;
    backend_size_ = backend_size;
    data_capacity_ = backend_size - 2 * kBackendHeaderSize;
    data_id_ = -1;
    priv_header_off_ = static_cast<size_t>(data_ - region_);
    flags_.Clear();

    // Copy all header fields to backend header (file offset 0)
    (*header_) = (const MemoryBackendHeader &)*this;

    HLOG(kDebug, "shm_init: Initialized with backend_size={}, data_capacity={}",
         backend_size_, data_capacity_);
    HLOG(kDebug, "shm_init: header_={}, region_={}, data_={}", (void *)header_,
         (void *)region_, (void *)data_);

    // Mark this process as the owner of the backend
    SetOwner();

    return true;
  }

  /**
   * Attach to existing backend with mixed private/shared mapping
   *
   * @param url POSIX shared memory object name
   * @return true on success, false on failure
   *
   * This method mirrors shm_init exactly:
   * - Maps backend header (4KB) as MAP_SHARED at offset 0
   * - Maps rest using MapMixedMemory at offset kBackendHeaderSize
   */
  bool shm_attach(const std::string &url) {
    if (!SystemInfo::OpenSharedMemory(fd_, url)) {
      const char *err_buf = strerror(errno);
      HLOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Step 1: Map the backend header (4KB) as MAP_SHARED at offset 0
    header_ = reinterpret_cast<MemoryBackendHeader *>(
        SystemInfo::MapSharedMemory(fd_, kBackendHeaderSize, 0));
    if (!header_) {
      HLOG(kError, "Failed to map backend header");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }
    (MemoryBackendHeader &)*this = (*header_);

    // Read backend_size from backend header
    size_t backend_size = header_->backend_size_;

    HLOG(kDebug, "shm_attach: Read backend_size={} from backend header at {}",
         backend_size, (void *)header_);
    HLOG(kDebug,
         "shm_attach: Header fields: id_=({},{}), data_capacity_={}, "
         "priv_header_off_={}",
         header_->id_.major_, header_->id_.minor_, header_->data_capacity_,
         header_->priv_header_off_);

    // Validate backend_size before subtraction to prevent underflow
    if (backend_size < kBackendHeaderSize) {
      HLOG(kError,
           "Invalid backend_size in header: {} bytes (must be >= {} bytes)",
           backend_size, kBackendHeaderSize);
      SystemInfo::UnmapMemory(header_, kBackendHeaderSize);
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Step 2: Map the rest using MapMixedMemory at offset kBackendHeaderSize
    // Private portion: 4KB (private header)
    // Shared portion: backend_size - kBackendHeaderSize (shared header + data)
    // Offset into file: kBackendHeaderSize (skip the backend header in file)
    size_t shared_portion_size = backend_size - kBackendHeaderSize;
    region_ = reinterpret_cast<char *>(SystemInfo::MapMixedMemory(
        fd_, kBackendHeaderSize, shared_portion_size, kBackendHeaderSize));
    if (!region_) {
      HLOG(kError, "Failed to create mixed mapping during attach");
      SystemInfo::UnmapMemory(header_, kBackendHeaderSize);
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Memory layout after mapping (same as shm_init):
    // header_ points to: [4KB backend header MAP_SHARED] at file offset 0
    // region_ points to: [4KB private header MAP_PRIVATE] [4KB shared header
    // MAP_SHARED] [data MAP_SHARED]
    //                     where shared portion maps to file offset
    //                     kBackendHeaderSize

    // Calculate data pointer (same layout as shm_init)
    // region_[0..4KB) = private header
    // region_[4KB..8KB) = shared header (maps to file offset 4KB..8KB)
    // region_[8KB...) = data (maps to file offset 8KB...)
    data_ = region_ + 2 * kBackendHeaderSize;

    // Mark this process as NOT the owner (attaching to existing backend)
    UnsetOwner();

    return true;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  char *_ShmMap(size_t size, i64 off) {
    char *ptr =
        reinterpret_cast<char *>(SystemInfo::MapSharedMemory(fd_, size, off));
    if (!ptr) {
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {
    if (header_ == nullptr) {
      return;
    }
    // Unmap the mixed mapping region (private header + shared header + data)
    if (region_ != nullptr) {
      // Total size is: private header (kBackendHeaderSize) + shared portion
      // (backend_size_ - kBackendHeaderSize)
      SystemInfo::UnmapMemory(region_, header_->backend_size_);
      region_ = nullptr;
    }
    // Unmap the backend header (separately mapped as MAP_SHARED)
    SystemInfo::UnmapMemory(header_, kBackendHeaderSize);
    SystemInfo::CloseSharedMemory(fd_);
    header_ = nullptr;
  }

  /** Destroy shared memory */
  void _Destroy() {
    _Detach();
    SystemInfo::DestroySharedMemory(url_);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
