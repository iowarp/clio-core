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

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#if HSHM_ENABLE_PROCFS_SYSINFO
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "memory_backend.h"

namespace hshm::ipc {

class PosixMmap : public MemoryBackend {
 private:
  size_t total_size_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  PosixMmap() = default;

  /** Destructor */
  ~PosixMmap() {
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t size) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (size < kMinBackendSize) {
      size = kMinBackendSize;
    }

    // Initialize flags before calling methods that use it
    flags_.Clear();
    SetInitialized();
    Own();

    // Calculate sizes: header + md section + alignment + data section
    constexpr size_t kAlignment = 4096;  // 4KB alignment
    size_t header_size = sizeof(MemoryBackendHeader);
    size_t md_size = header_size;  // md section stores the header
    size_t aligned_md_size = ((md_size + kAlignment - 1) / kAlignment) * kAlignment;
    total_size_ = aligned_md_size + size;

    // Map memory
    char *ptr = _Map(total_size_);
    if (!ptr) {
      return false;
    }

    // Layout: [MemoryBackendHeader | padding to 4KB] [data]
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    header_->id_ = backend_id;
    header_->md_size_ = md_size;
    header_->data_size_ = size;
    header_->data_id_ = -1;
    header_->flags_.Clear();

    // md_ points to the header itself (metadata for process connection)
    md_ = ptr;
    md_size_ = md_size;

    // data_ starts at 4KB aligned boundary after md section
    data_ = ptr + aligned_md_size;
    data_size_ = size;
    data_id_ = -1;
    data_offset_ = 0;

    return true;
  }

  /** Deserialize the backend */
  bool shm_attach(const std::string &url) {
    (void)url;
    HSHM_THROW_ERROR(SHMEM_NOT_SUPPORTED);
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
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }
    return ptr;
  }

  /** Unmap shared memory */
  void _Detach() {
    if (!IsInitialized()) {
      return;
    }
    SystemInfo::UnmapMemory(reinterpret_cast<void *>(header_), total_size_);
    UnsetInitialized();
  }

  /** Destroy shared memory */
  void _Destroy() {
    if (!IsInitialized()) {
      return;
    }
    _Detach();
    UnsetInitialized();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_POSIX_MMAP_H
