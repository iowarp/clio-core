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

#ifndef CTP_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H
#define CTP_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "clio_ctp/constants/macros.h"
#if CTP_ENABLE_PROCFS_SYSINFO
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/errors.h"
#include "memory_backend.h"

namespace ctp::ipc {

class PosixMmap : public MemoryBackend {
 public:
  /** Constructor */
  CTP_CROSS_FUN
  PosixMmap() = default;

  /** Destructor */
  ~PosixMmap() = default;

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }

    char *ptr = _Map(backend_size);
    if (!ptr) {
      return false;
    }

    region_ = ptr;
    header_ = reinterpret_cast<MemoryBackendHeader *>(region_);
    data_ = region_ + kBackendHeaderSize;

    id_ = backend_id;
    backend_size_ = backend_size;
    data_capacity_ = backend_size - kBackendHeaderSize;
    data_id_ = -1;
    flags_.Clear();

    new (header_) MemoryBackendHeader();
    (*header_) = (const MemoryBackendHeader &)*this;

    SetOwner();

    return true;
  }

  /** Deserialize the backend */
  bool shm_attach(const std::string &url) {
    (void)url;
    CTP_THROW_ERROR(SHMEM_NOT_SUPPORTED);
    return false;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  template <typename T = char>
  T *_Map(size_t size) {
    T *ptr = reinterpret_cast<T *>(
        SystemInfo::MapPrivateMemory(MemoryAlignment::AlignToPageSize(size)));
    if (!ptr) {
      CTP_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {
    if (region_) {
      SystemInfo::UnmapMemory(region_, backend_size_);
      region_ = nullptr;
    }
  }

  /** Destroy shared memory */
  void _Destroy() {
    _Detach();
  }
};

}  // namespace ctp::ipc

#endif  // CTP_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H
