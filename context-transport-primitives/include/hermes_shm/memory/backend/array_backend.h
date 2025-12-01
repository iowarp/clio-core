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

#ifndef HSHM_INCLUDE_HSHM_MEMORY_BACKEND_ARRAY_BACKEND_H_
#define HSHM_INCLUDE_HSHM_MEMORY_BACKEND_ARRAY_BACKEND_H_

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "memory_backend.h"

namespace hshm::ipc {

class ArrayBackend : public MemoryBackend {
 public:
  HSHM_CROSS_FUN
  ArrayBackend() = default;

  ~ArrayBackend() {
    if (IsOwned() && header_ != nullptr) {
      header_->~MemoryBackendHeader();
      free(header_);
      header_ = nullptr;
    }
  }

  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackendId &backend_id, size_t size, char *region, u64 offset = 0) {
    SetInitialized();
    Own();

    // Allocate metadata using malloc and construct with placement new
    void *header_mem = malloc(sizeof(MemoryBackendHeader));
    header_ = new (header_mem) MemoryBackendHeader();
    md_ = reinterpret_cast<char*>(header_);
    md_size_ = sizeof(MemoryBackendHeader);

    // Initialize header
    header_->id_ = backend_id;
    header_->md_size_ = md_size_;
    header_->data_size_ = size;
    header_->data_id_ = -1;
    header_->flags_.Clear();

    // Data segment from region
    data_size_ = size;
    data_ = region;
    data_id_ = -1;
    data_offset_ = offset;

    return true;
  }

  bool shm_attach(const std::string &url) {
    (void)url;
    HSHM_THROW_ERROR(SHMEM_NOT_SUPPORTED);
    return false;
  }

  void shm_detach() {}

  void shm_destroy() {}
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_HSHM_MEMORY_BACKEND_ARRAY_BACKEND_H_
