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

#ifndef CTP_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H
#define CTP_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/errors.h"
#include "memory_backend.h"

namespace ctp::ipc {

class MallocBackend : public MemoryBackend {
 public:
  CTP_CROSS_FUN
  MallocBackend() = default;

  ~MallocBackend() { _Destroy(); }

  CTP_CROSS_FUN
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

  bool shm_attach(const std::string &url) {
    (void)url;
    CTP_THROW_ERROR(SHMEM_NOT_SUPPORTED);
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

}  // namespace ctp::ipc

#endif  // CTP_INCLUDE_HSHM_MEMORY_BACKEND_MALLOC_H
