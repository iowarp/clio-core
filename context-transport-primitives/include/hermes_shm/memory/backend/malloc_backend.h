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

#ifndef HSHM_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H
#define HSHM_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "memory_backend.h"

namespace hshm::ipc {

class MallocBackend : public MemoryBackend {
 public:
  HSHM_CROSS_FUN
  MallocBackend() = default;

  ~MallocBackend() { _Destroy(); }

  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackendId &backend_id, size_t backend_size) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (backend_size < kMinBackendSize) {
      backend_size = kMinBackendSize;
    }

    // Allocate total memory
    char *ptr = (char *)malloc(backend_size);
    if (!ptr) {
      return false;
    }

    region_ = ptr;  // Save allocation start for cleanup

    // Layout: [header (4KB)] [priv_header (4KB)] [shared_header (4KB)] [data...]
    // header_ is the first part of the buffer
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    char *priv_header_ptr = ptr + kBackendHeaderSize;
    char *shared_header_ptr = priv_header_ptr + kBackendHeaderSize;

    id_ = backend_id;
    backend_size_ = backend_size;

    // data_ starts after shared header
    data_ = shared_header_ptr + kBackendHeaderSize;
    data_capacity_ = backend_size - 3 * kBackendHeaderSize;
    data_id_ = -1;
    priv_header_off_ = static_cast<size_t>(data_ - priv_header_ptr);
    flags_.Clear();

    // Copy all header fields to shared header
    new (header_) MemoryBackendHeader();
    (*header_) = (const MemoryBackendHeader&)*this;

    // Mark this process as the owner of the backend
    SetOwner();

    return true;
  }

  bool shm_attach(const std::string &url) {
    (void)url;
    HSHM_THROW_ERROR(SHMEM_NOT_SUPPORTED);
    return false;
  }

  void shm_detach() { _Detach(); }

  void shm_destroy() { _Destroy(); }

 protected:
  void _Detach() {
    if (region_) {
      free(region_);  // Free from allocation start (includes private region)
      region_ = nullptr;
    }
  }

  void _Destroy() {
    if (region_) {
      free(region_);  // Free from allocation start (includes private region)
      region_ = nullptr;
    }
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H
