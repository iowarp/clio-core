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

#ifndef CTP_INCLUDE_HSHM_MEMORY_BACKEND_ARRAY_BACKEND_H_
#define CTP_INCLUDE_HSHM_MEMORY_BACKEND_ARRAY_BACKEND_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/util/errors.h"
#include "memory_backend.h"

namespace ctp::ipc {

class ArrayBackend : public MemoryBackend {
 public:
  CTP_CROSS_FUN
  ArrayBackend() = default;

  ~ArrayBackend() = default;

  /**
   * Initialize ArrayBackend with external array
   *
   * @param backend_id Backend identifier
   * @param size Size of the ENTIRE array
   * @param region Pointer to the BEGINNING of the array
   * @return true on success
   *
   * The entire region is available as data.
   */
  CTP_CROSS_FUN
  bool shm_init(const MemoryBackendId &backend_id, size_t size, char *region) {
    region_ = region;
    header_ = reinterpret_cast<MemoryBackendHeader *>(region_);
    data_ = region_ + kBackendHeaderSize;

    id_ = backend_id;
    backend_size_ = size;
    data_capacity_ = size - kBackendHeaderSize;
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

  void shm_detach() {}

  void shm_destroy() {}
};

}  // namespace ctp::ipc

#endif  // CTP_INCLUDE_HSHM_MEMORY_BACKEND_ARRAY_BACKEND_H_
