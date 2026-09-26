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

#ifndef CTP_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
#define CTP_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "clio_ctp/constants/macros.h"
#ifndef _WIN32
#include <fcntl.h>
#endif
#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/errors.h"
#include "clio_ctp/util/logging.h"
#include "memory_backend.h"

namespace ctp::ipc {

class PosixShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  std::string url_;

 public:
  /** Constructor */
  CTP_CROSS_FUN
  PosixShmMmap() : fd_{} {}

  /** Destructor */
  CTP_CROSS_FUN
  ~PosixShmMmap() = default;

  /**
   * Initialize backend with shared mapping
   *
   * @param backend_id Unique identifier for this backend
   * @param backend_size Total size of the region (header + data)
   * @param url POSIX shared memory object name (e.g., "/my_shm")
   * @return true on success, false on failure
   *
   * File layout:
   *   [page-aligned header] [data]
   *
   * header_ is mapped separately (MAP_SHARED) at file offset 0.
   * region_/data_ are mapped at file offset kBackendHeaderSize.
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size,
                const std::string &url) {
    constexpr size_t kMinBackendSize = 1024 * 1024;
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }

    const size_t hdr_size = kBackendHeaderSize;

    SystemInfo::DestroySharedMemory(url);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url, backend_size)) {
      // NOT strerror(errno): none of the three platform implementations is
      // shm_open(), and the Win32 one does not set errno at all, so this used
      // to report a commit-limit failure as "Resource temporarily
      // unavailable". Ask the platform what actually went wrong.
      //
      // Read it into a local BEFORE logging, and not as an argument to HLOG.
      // GetLastError() is thread-global, and HLOG's first use in a process
      // constructs the Logger singleton, which looks up CTP_LOG_LEVEL and
      // CTP_LOG_OUT -- leaving ERROR_ENVVAR_NOT_FOUND behind when they are
      // unset, as they normally are. The order of the two is unspecified, and
      // in practice the singleton wins, so the FIRST shm failure in a process
      // was reported as "could not find the environment option that was
      // entered (Win32 error 203)" -- a code nothing in the create path can
      // produce -- while later ones reported the truth. That is exactly what
      // the two failures in CDash build 4029381 look like: 203 from the first,
      // the real ERROR_USER_MAPPED_FILE from the second.
      std::string shm_err = SystemInfo::GetLastSharedMemoryError();
      HLOG(kError, "could not create shared memory segment {} of {} bytes: {}",
           url, backend_size, shm_err);
      return false;
    }
    url_ = url;

    // Map the backend header at file offset 0
    header_ = reinterpret_cast<MemoryBackendHeader *>(
        SystemInfo::MapSharedMemory(fd_, hdr_size, 0));
    if (!header_) {
      HLOG(kError, "Failed to map backend header");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Map the data region at file offset hdr_size
    size_t data_size = backend_size - hdr_size;
    region_ = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, data_size, hdr_size));
    if (!region_) {
      HLOG(kError, "Failed to map data region");
      SystemInfo::UnmapMemory(header_, hdr_size);
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }
    data_ = region_;

    id_ = backend_id;
    backend_size_ = backend_size;
    data_capacity_ = data_size;
    data_id_ = -1;
    flags_.Clear();

    // Persist header fields into the shared mapping
    (*header_) = (const MemoryBackendHeader &)*this;

    SetOwner();
    return true;
  }

  /**
   * Attach to existing backend with shared mapping
   *
   * @param url POSIX shared memory object name
   * @return true on success, false on failure
   */
  bool shm_attach(const std::string &url) {
    if (!SystemInfo::OpenSharedMemory(fd_, url)) {
      const char *err_buf = strerror(errno);
      HLOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    const size_t hdr_size = kBackendHeaderSize;

    // Map the backend header at file offset 0
    header_ = reinterpret_cast<MemoryBackendHeader *>(
        SystemInfo::MapSharedMemory(fd_, hdr_size, 0));
    if (!header_) {
      HLOG(kError, "Failed to map backend header");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }
    (MemoryBackendHeader &)*this = (*header_);

    size_t backend_size = header_->backend_size_;
    if (backend_size < hdr_size) {
      HLOG(kError,
           "Invalid backend_size in header: {} bytes (must be >= {} bytes)",
           backend_size, hdr_size);
      SystemInfo::UnmapMemory(header_, hdr_size);
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Map data region at file offset hdr_size
    size_t data_size = backend_size - hdr_size;
    region_ = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, data_size, hdr_size));
    if (!region_) {
      HLOG(kError, "Failed to map data region during attach");
      SystemInfo::UnmapMemory(header_, hdr_size);
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }
    data_ = region_;

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
      CTP_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {
    if (header_ == nullptr) {
      return;
    }
    const size_t hdr_size = kBackendHeaderSize;
    // Unmap the data region
    if (region_ != nullptr) {
      SystemInfo::UnmapMemory(region_, header_->backend_size_ - hdr_size);
      region_ = nullptr;
    }
    // Unmap the backend header (separately mapped as MAP_SHARED)
    SystemInfo::UnmapMemory(header_, hdr_size);
    SystemInfo::CloseSharedMemory(fd_);
    header_ = nullptr;
  }

  /** Destroy shared memory */
  void _Destroy() {
    _Detach();
    SystemInfo::DestroySharedMemory(url_);
  }
};

}  // namespace ctp::ipc

#endif  // CTP_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
