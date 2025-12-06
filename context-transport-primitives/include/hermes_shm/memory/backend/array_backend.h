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

  /**
   * Initialize ArrayBackend with external array
   *
   * @param backend_id Backend identifier
   * @param size Size of the ENTIRE array
   * @param region Pointer to the BEGINNING of the array
   * @param offset Offset within the array
   * @return true on success
   *
   * Memory layout in the region:
   * - Bytes 0 to kBackendHeaderSize-1: Private header
   * - Bytes kBackendHeaderSize to 2*kBackendHeaderSize-1: Shared header
   * - Bytes 2*kBackendHeaderSize onwards: Backend header (header_)
   * - Then data_ follows after backend header
   */
  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackendId &backend_id, size_t size, char *region, u64 offset = 0) {

    // Headers are at the beginning of the region
    char *shared_header = region + kBackendHeaderSize;
    char *backend_header_ptr = region + 2 * kBackendHeaderSize;

    // Store backend metadata header
    header_ = reinterpret_cast<MemoryBackendHeader*>(backend_header_ptr);
    md_ = backend_header_ptr;
    md_size_ = sizeof(MemoryBackendHeader);

    // Initialize header
    header_->id_ = backend_id;
    header_->md_size_ = md_size_;
    header_->data_size_ = size - 2 * kBackendHeaderSize - sizeof(MemoryBackendHeader);
    header_->data_id_ = -1;
    header_->flags_.Clear();

    // data_ points to start of data region (after backend header)
    data_ = backend_header_ptr + sizeof(MemoryBackendHeader);
    data_size_ = size - 2 * kBackendHeaderSize - sizeof(MemoryBackendHeader);
    data_capacity_ = size - 2 * kBackendHeaderSize - sizeof(MemoryBackendHeader);
    data_id_ = -1;
    data_offset_ = offset;

    // Set priv_header_off_: distance from data_ back to region start
    // priv_header_off_ = data_ - region
    priv_header_off_ = static_cast<size_t>(data_ - region);

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
