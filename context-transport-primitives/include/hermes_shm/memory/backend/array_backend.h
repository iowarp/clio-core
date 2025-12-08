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

  ~ArrayBackend() = default;

  /**
   * Initialize ArrayBackend with external array
   *
   * @param backend_id Backend identifier
   * @param size Size of the ENTIRE array
   * @param region Pointer to the BEGINNING of the array
   * @return true on success
   *
   * Memory layout in the region:
   * - Bytes 0 to kBackendHeaderSize-1: Private header
   * - Bytes kBackendHeaderSize to 2*kBackendHeaderSize-1: Shared header
   * - Bytes 2*kBackendHeaderSize onwards: Custom header and data
   */
  HSHM_CROSS_FUN
  bool shm_init(const MemoryBackendId &backend_id, size_t size, char *region) {
    // Headers are at the beginning of the region
    region_ = region;
    char *priv_header_ptr = region + kBackendHeaderSize;
    char *shared_header_ptr = priv_header_ptr + kBackendHeaderSize;

    // Store backend metadata header
    header_ = reinterpret_cast<MemoryBackendHeader *>(shared_header_ptr +
                                                      kBackendHeaderSize);
    data_capacity_ = size - 3 * kBackendHeaderSize;
    id_ = backend_id;
    backend_size_ = size;
    data_id_ = -1;
    priv_header_off_ = static_cast<size_t>(priv_header_ptr - region);
    flags_.Clear();

    // data_ points to start of data region (after shared header)
    data_ = shared_header_ptr + kBackendHeaderSize;

    // Copy all header fields to shared header
    new (header_) MemoryBackendHeader();
    (*header_) = (const MemoryBackendHeader&)*this;

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
