//
// Created by llogan on 25/10/24.
//

#ifndef GPU_MALLOC_H
#define GPU_MALLOC_H

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#include <string>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/util/errors.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/util/logging.h"
#include "memory_backend.h"

namespace hshm::ipc {

struct GpuMallocHeader : public MemoryBackendHeader {
  GpuIpcMemHandle ipc_;
};

class GpuMalloc : public MemoryBackend, public UrlMemoryBackend {
 protected:
  File fd_;
  std::string url_;
  size_t total_size_;

 public:
  /** Constructor */
  HSHM_CROSS_FUN
  GpuMalloc() = default;

  /** Destructor */
  ~GpuMalloc() {
    if (IsOwned()) {
      _Destroy();
    } else {
      _Detach();
    }
  }

  /** Initialize backend */
  bool shm_init(const MemoryBackendId &backend_id, size_t accel_data_size,
                const std::string &url, int gpu_id = 0) {
    // Enforce minimum backend size of 1MB
    constexpr size_t kMinBackendSize = 1024 * 1024;  // 1MB
    if (accel_data_size < kMinBackendSize) {
      accel_data_size = kMinBackendSize;
    }

    // Initialize flags before calling methods that use it
    flags_.Clear();
    Own();

    // Calculate sizes: header + md section + alignment
    constexpr size_t kAlignment = 4096;  // 4KB alignment
    size_t header_size = sizeof(GpuMallocHeader);
    size_t md_size = header_size;  // md section stores the header
    size_t aligned_md_size = ((md_size + kAlignment - 1) / kAlignment) * kAlignment;
    total_size_ = aligned_md_size;  // GPU data is separate, not in shared memory

    // Create shared memory for metadata
    SystemInfo::DestroySharedMemory(url);
    if (!SystemInfo::CreateNewSharedMemory(fd_, url, total_size_)) {
      char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Map shared memory for metadata
    char *ptr = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, total_size_, 0));
    if (!ptr) {
      HSHM_THROW_ERROR(SHMEM_CREATE_FAILED);
    }

    // Layout: [GpuMallocHeader | padding to 4KB]
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    GpuMallocHeader *gpu_header = reinterpret_cast<GpuMallocHeader *>(ptr);
    new (gpu_header) GpuMallocHeader();
    gpu_header->id_ = backend_id;
    gpu_header->md_size_ = md_size;
    gpu_header->accel_data_size_ = accel_data_size;
    gpu_header->accel_id_ = gpu_id;
    gpu_header->flags_.Clear();

    // md_ points to the header itself (metadata for process connection)
    md_ = ptr;
    md_size_ = md_size;

    // Allocate GPU memory separately
    accel_data_ = _Map(accel_data_size);
    accel_data_size_ = accel_data_size;
    accel_id_ = gpu_id;

    // Store IPC handle for GPU memory in shared metadata
    GpuApi::GetIpcMemHandle(gpu_header->ipc_, (void *)accel_data_);

    return true;
  }

  /** Deserialize the backend */
  bool shm_attach(const std::string &url) {
    flags_.Clear();
    Disown();

    if (!SystemInfo::OpenSharedMemory(fd_, url)) {
      const char *err_buf = strerror(errno);
      HILOG(kError, "shm_open failed: {}", err_buf);
      return false;
    }
    url_ = url;

    // Map just the header to get size information
    constexpr size_t kAlignment = 4096;
    char *ptr = reinterpret_cast<char *>(
        SystemInfo::MapSharedMemory(fd_, kAlignment, 0));
    if (!ptr) {
      return false;
    }

    // Calculate total size for metadata
    GpuMallocHeader *gpu_header = reinterpret_cast<GpuMallocHeader *>(ptr);
    size_t md_size = gpu_header->md_size_;
    size_t aligned_md_size = ((md_size + kAlignment - 1) / kAlignment) * kAlignment;
    total_size_ = aligned_md_size;

    // Set up pointers
    header_ = reinterpret_cast<MemoryBackendHeader *>(ptr);
    md_ = ptr;
    md_size_ = md_size;

    // Open GPU memory from IPC handle
    accel_data_size_ = gpu_header->accel_data_size_;
    accel_id_ = gpu_header->accel_id_;
    GpuApi::OpenIpcMemHandle(gpu_header->ipc_, &accel_data_);

    return true;
  }

  /** Detach the mapped memory */
  void shm_detach() { _Detach(); }

  /** Destroy the mapped memory */
  void shm_destroy() { _Destroy(); }

 protected:
  /** Map shared memory */
  template <typename T = char>
  T *_Map(size_t size) {
    return GpuApi::Malloc<T>(size);
  }

  /** Unmap shared memory */
  void _Detach() {
      return;
    }
    // Unmap the metadata region
    if (md_) {
      SystemInfo::UnmapMemory(reinterpret_cast<void *>(md_), total_size_);
    }
    // Close IPC handle for GPU memory
    if (accel_data_) {
      GpuApi::CloseIpcMemHandle(accel_data_);
    }
    SystemInfo::CloseSharedMemory(fd_);
  }

  /** Destroy shared memory */
  void _Destroy() {
      return;
    }
    // Free GPU memory
    if (accel_data_) {
      GpuApi::Free(accel_data_);
    }
    // Unmap and destroy shared metadata
    if (md_) {
      SystemInfo::UnmapMemory(reinterpret_cast<void *>(md_), total_size_);
    }
    SystemInfo::CloseSharedMemory(fd_);
    SystemInfo::DestroySharedMemory(url_);
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM

#endif  // GPU_MALLOC_H
