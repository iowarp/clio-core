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

#ifndef HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
#define HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

class PosixShmMmap : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  std::string url_;
  size_t total_size_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  PosixShmMmap() {}

  /** Destructor */
  HSHM_CROSS_FUN
  ~PosixShmMmap() {
#if HSHM_IS_HOST
    _Detach();
#endif
  }

  /**
   * Initialize backend with mixed private/shared mapping
   *
   * Creates a contiguous virtual memory region:
   * - First kBackendHeaderSize (4KB): SHARED header (shared between processes)
   * - Second kBackendHeaderSize (4KB): PRIVATE header (process-local, not shared)
   * - Remaining: SHARED mapping (inter-process shared memory)
   *
   * The private header can be used for process-local metadata, TLS pointers,
   * or other data that should not be shared between processes.
   *
   * @param backend_id Unique identifier for this backend
   * @param size Size of the shared data region (excluding headers)
   * @param url POSIX shared memory object name (e.g., "/my_shm")
   * @return true on success, false on failure
   *
   * Memory layout: [kBackendHeaderSize shared header] [kBackendHeaderSize private header] [MemoryBackendHeader] [data]
   *                 ^ptr (shared start)                 ^private start                     ^header_             ^data_
   */
  bool shm_init(const MemoryBackendId &backend_id, size_t size,
                const std::string &url) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (size < kMinBackendSize) {
      size = kMinBackendSize;
    }

    // Initialize flags before calling methods that use it
    flags_.Clear();

    // Calculate sizes: header + md section + alignment + data section
    constexpr size_t kAlignment = 4096;  // 4KB alignment
    size_t header_size = sizeof(MemoryBackendHeader);
    size_t md_size = header_size;
    size_t aligned_md_size = ((md_size + kAlignment - 1) / kAlignment) * kAlignment;

    // Total layout: [kBackendHeaderSize shared header | kBackendHeaderSize private header | aligned_md_size | data]
    // The shared memory object contains: shared header + aligned_md_size + data
    size_t shared_size = kBackendHeaderSize + aligned_md_size + size;
    total_size_ = kBackendHeaderSize + shared_size;

    // Create shared memory object (includes shared header + md + data)
    SystemInfo::DestroySharedMemory(url);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url, shared_size)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Create mixed mapping: [kBackendHeaderSize private header | shared_size shared]
    char *ptr = reinterpret_cast<char *>(
        SystemInfo::MapMixedMemory(fd_, kBackendHeaderSize, shared_size, 0));
    if (!ptr) {
      HILOG(kError, "Failed to create mixed mapping");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // ptr points to start of private header
    // ptr + kBackendHeaderSize points to start of shared region (which begins with shared header)
    char *shared_region_start = ptr + kBackendHeaderSize;
    char *shared_ptr = shared_region_start + kBackendHeaderSize;

    // Now we have: [kBackendHeaderSize private header | kBackendHeaderSize shared header | aligned_md_size | data]
    // The first kBackendHeaderSize is private (process-local), the second is shared

    // Layout: [kBackendHeaderSize shared header] [kBackendHeaderSize private header] [MemoryBackendHeader | padding to 4KB] [data]
    header_ = reinterpret_cast<MemoryBackendHeader *>(shared_ptr);
    new (header_) MemoryBackendHeader();
    header_->id_ = backend_id;
    header_->md_size_ = md_size;
    header_->data_size_ = size;
    header_->data_id_ = -1;
    header_->flags_.Clear();

    // md_ points to the header itself (metadata for process connection)
    md_ = shared_ptr;
    md_size_ = md_size;

    // data_ starts at 4KB aligned boundary after md section
    data_ = shared_ptr + aligned_md_size;
    data_size_ = size;
    data_capacity_ = size;  // Full capacity equals data size for root backend
    data_id_ = -1;
    data_offset_ = 0;

    // Set priv_header_off_: distance from data_ back to start of private header
    // private header is at ptr, data_ is at shared_ptr + aligned_md_size
    // distance = (shared_ptr + aligned_md_size) - ptr
    priv_header_off_ = static_cast<size_t>(data_ - ptr);

    return true;
  }

  /**
   * Attach to existing backend with mixed private/shared mapping
   *
   * Recreates the same memory layout as the initializing process:
   * - First kBackendHeaderSize (4KB): PRIVATE header (process-local, independent per process)
   * - Remaining: SHARED mapping (attached to shared memory object, starts with shared header)
   *
   * @param url POSIX shared memory object name
   * @return true on success, false on failure
   */
  bool shm_attach(const std::string &url) {
    flags_.Clear();

    if (!SystemInfo::OpenSharedMemory(fd_, url)) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Map just past the shared header temporarily to read MemoryBackendHeader
    constexpr size_t kAlignment = 4096;
    char *temp_header = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, kAlignment, kBackendHeaderSize));
    if (!temp_header) {
      HILOG(kError, "Failed to map header");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    // Read header information
    MemoryBackendHeader *hdr = reinterpret_cast<MemoryBackendHeader *>(temp_header);
    size_t md_size = hdr->md_size_;
    size_t data_size = hdr->data_size_;

    // Calculate sizes
    size_t aligned_md_size = ((md_size + kAlignment - 1) / kAlignment) * kAlignment;
    size_t shared_size = kBackendHeaderSize + aligned_md_size + data_size;
    total_size_ = kBackendHeaderSize + shared_size;

    // Unmap the temporary header
    SystemInfo::UnmapMemory(temp_header, kAlignment);

    // Create mixed mapping with correct sizes
    char *ptr = reinterpret_cast<char *>(
        SystemInfo::MapMixedMemory(fd_, kBackendHeaderSize, shared_size, 0));
    if (!ptr) {
      HILOG(kError, "Failed to create mixed mapping during attach");
      SystemInfo::CloseSharedMemory(fd_);
      return false;
    }

    char *shared_region_start = ptr + kBackendHeaderSize;
    char *shared_ptr = shared_region_start + kBackendHeaderSize;

    // Set up pointers (same layout as shm_init)
    header_ = reinterpret_cast<MemoryBackendHeader *>(shared_ptr);
    md_ = shared_ptr;
    md_size_ = md_size;
    data_ = shared_ptr + aligned_md_size;
    data_size_ = data_size;
    data_capacity_ = data_size;  // Full capacity equals data size for root backend
    data_id_ = header_->data_id_;
    data_offset_ = 0;

    // Set priv_header_off_: distance from data_ back to start of private header
    // private header is at ptr, data_ is at shared_ptr + aligned_md_size
    // distance = (shared_ptr + aligned_md_size) - ptr
    priv_header_off_ = static_cast<size_t>(data_ - ptr);

    return true;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

  // GetPrivateHeader() and GetSharedHeader() are inherited from MemoryBackend base class

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
    if (!IsInitialized()) {
      return;
    }
    // Unmap the entire contiguous region (starts 2*kBackendHeaderSize bytes before header_)
    char *mapping_start = reinterpret_cast<char*>(header_) - 2 * kBackendHeaderSize;
    SystemInfo::UnmapMemory(mapping_start, total_size_);
    SystemInfo::CloseSharedMemory(fd_);
    UnsetInitialized();
  }

  /** Destroy shared memory */
  void _Destroy() {
    if (!IsInitialized()) {
      return;
    }
    _Detach();
    SystemInfo::DestroySharedMemory(url_);
    UnsetInitialized();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_MEMORY_BACKEND_POSIX_SHM_MMAP_H
