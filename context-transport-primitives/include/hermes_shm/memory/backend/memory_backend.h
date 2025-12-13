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

#ifndef HSHM_MEMORY_H
#define HSHM_MEMORY_H

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "hermes_shm/constants/macros.h"
// #include "hermes_shm/data_structures/ipc/chararr.h"  // Deleted during hard refactoring
#include "hermes_shm/memory/allocator/allocator.h"

namespace hshm::ipc {

/** Forward declaration for FullPtr (defined in allocator.h after this header) */
template <typename T, bool ATOMIC>
struct FullPtr;

/** ID for memory backend */
class MemoryBackendId {
 public:
  u32 major_;  // Major ID (e.g., PID)
  u32 minor_;  // Minor ID (relative to major)

  HSHM_CROSS_FUN
  MemoryBackendId() : major_(0), minor_(0) {}

  HSHM_CROSS_FUN
  MemoryBackendId(u32 major, u32 minor) : major_(major), minor_(minor) {}

  HSHM_CROSS_FUN
  MemoryBackendId(const MemoryBackendId &other) : major_(other.major_), minor_(other.minor_) {}

  HSHM_CROSS_FUN
  MemoryBackendId(MemoryBackendId &&other) noexcept : major_(other.major_), minor_(other.minor_) {}

  HSHM_CROSS_FUN
  MemoryBackendId &operator=(const MemoryBackendId &other) {
    major_ = other.major_;
    minor_ = other.minor_;
    return *this;
  }

  HSHM_CROSS_FUN
  MemoryBackendId &operator=(MemoryBackendId &&other) noexcept {
    major_ = other.major_;
    minor_ = other.minor_;
    return *this;
  }

  HSHM_CROSS_FUN
  static MemoryBackendId GetRoot() { return {0, 0}; }

  HSHM_CROSS_FUN
  static MemoryBackendId Get(u32 major, u32 minor) { return {major, minor}; }

  HSHM_CROSS_FUN
  bool operator==(const MemoryBackendId &other) const {
    return major_ == other.major_ && minor_ == other.minor_;
  }

  HSHM_CROSS_FUN
  bool operator!=(const MemoryBackendId &other) const {
    return major_ != other.major_ || minor_ != other.minor_;
  }

  /** Get the null backend ID */
  HSHM_CROSS_FUN
  static MemoryBackendId GetNull() {
    return MemoryBackendId(UINT32_MAX, UINT32_MAX);
  }

  /** Set this backend ID to null */
  HSHM_CROSS_FUN
  void SetNull() { *this = GetNull(); }

  /** Check if this is the null backend ID */
  HSHM_CROSS_FUN
  bool IsNull() const { return *this == GetNull(); }

  /** To index */
  HSHM_CROSS_FUN
  uint32_t ToIndex() const {
    return major_ * 2 + minor_;
  }

  /** Serialize */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    ar & major_;
    ar & minor_;
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() const {
    printf("(%s) Memory Backend ID: (%u,%u)\n", kCurrentDevice, major_, minor_);
  }

  friend std::ostream &operator<<(std::ostream &os, const MemoryBackendId &id) {
    os << "(" << id.major_ << "," << id.minor_ << ")";
    return os;
  }
};
typedef MemoryBackendId memory_backend_id_t;

struct MemoryBackendHeader {
  MemoryBackendId id_;
  bitfield64_t flags_;
  size_t backend_size_;      // Total size of region_
  size_t data_capacity_;     // Capacity available for data allocation
  int data_id_;              // Device ID for the data buffer (GPU ID, etc.)
  size_t priv_header_off_;   // Offset from data_ back to start of private header

  HSHM_CROSS_FUN void Print() const {
    printf("(%s) MemoryBackendHeader: id: (%u, %u), backend_size: %lu, data_capacity: %lu, priv_header_off: %lu\n",
           kCurrentDevice, id_.major_, id_.minor_, (long unsigned)backend_size_, (long unsigned)data_capacity_, (long unsigned)priv_header_off_);
  }
};

#define MEMORY_BACKEND_INITIALIZED BIT_OPT(u64, 0)
#define MEMORY_BACKEND_OWNED BIT_OPT(u64, 1)
#define MEMORY_BACKEND_GPU_ONLY BIT_OPT(u64, 2)

class UrlMemoryBackend {};

/**
 * Global constant for backend header sizes
 * Each header (shared and private) is 4KB
 */
static constexpr size_t kBackendHeaderSize = 4 * 1024;  // 4KB per header (shared + private = 8KB total)

/** Forward declaration for kernel functions */
class MemoryBackend;

/**
 * GPU kernel for MakeAlloc - constructs and initializes allocator on GPU
 */
template<typename AllocT, typename... Args>
HSHM_GPU_KERNEL
void MakeAllocKernel(MemoryBackend* backend, AllocT** result, Args... args);

/**
 * GPU kernel for AttachAlloc - attaches to allocator on GPU
 */
template<typename AllocT>
HSHM_GPU_KERNEL
void AttachAllocKernel(MemoryBackend* backend, AllocT** result);

class MemoryBackend : public MemoryBackendHeader {
 public:
  MemoryBackendHeader *header_;  // Pointer to the MemoryBackendHeader in shared/mapped memory
  char *region_;    // The entire region: [private header] [shared header] [data]
  char *data_;      // Data buffer for allocators (points to start of data section)

 public:
  HSHM_CROSS_FUN
  MemoryBackend() : header_(nullptr), region_(nullptr), data_(nullptr) {
    // Initialize inherited MemoryBackendHeader fields
    id_ = MemoryBackendId();
    flags_.Clear();
    backend_size_ = 0;
    data_capacity_ = 0;
    data_id_ = -1;
    priv_header_off_ = 0;
  }

  ~MemoryBackend() = default;

  /** Get the ID of this backend */
  HSHM_CROSS_FUN
  MemoryBackendId &GetId() { return id_; }

  /** Get the ID of this backend */
  HSHM_CROSS_FUN
  const MemoryBackendId &GetId() const { return id_; }

  /**
   * Set the MEMORY_BACKEND_OWNED flag
   * Called during shm_init to indicate this process owns the backend
   */
  HSHM_CROSS_FUN
  void SetOwner() {
    flags_.SetBits(MEMORY_BACKEND_OWNED);
  }

  /**
   * Check if this process owns the backend
   * @return true if MEMORY_BACKEND_OWNED flag is set
   */
  HSHM_CROSS_FUN
  bool IsOwner() const {
    return flags_.Any(MEMORY_BACKEND_OWNED) != 0;
  }

  /**
   * Unset the MEMORY_BACKEND_OWNED flag
   * Called during shm_attach to indicate this process is attaching to
   * a backend created by another process
   */
  HSHM_CROSS_FUN
  void UnsetOwner() {
    flags_.UnsetBits(MEMORY_BACKEND_OWNED);
  }

  /**
   * Set the MEMORY_BACKEND_GPU_ONLY flag
   * Indicates this backend is GPU-only and requires accelerator path for host access
   */
  HSHM_CROSS_FUN
  void SetGpuOnly() {
    flags_.SetBits(MEMORY_BACKEND_GPU_ONLY);
  }

  /**
   * Check if this backend is GPU-only
   * @return true if MEMORY_BACKEND_GPU_ONLY flag is set
   */
  HSHM_CROSS_FUN
  bool IsGpuOnly() const {
    return flags_.Any(MEMORY_BACKEND_GPU_ONLY) != 0;
  }

  /**
   * Unset the MEMORY_BACKEND_GPU_ONLY flag
   */
  HSHM_CROSS_FUN
  void UnsetGpuOnly() {
    flags_.UnsetBits(MEMORY_BACKEND_GPU_ONLY);
  }

  /**
   * Determine if we need to use the accelerator path
   * @return true if this backend is GPU-only AND we're executing on the host
   */
  HSHM_CROSS_FUN
  bool DoAccelPath() const {
#if HSHM_IS_HOST
    return IsGpuOnly();
#else
    return false;
#endif
  }

  /**
   * Get pointer to the private header given a data pointer.
   * Private header is kBackendHeaderSize bytes before the custom header.
   *
   * @tparam T Type to cast the private header to (default: char)
   * @param data The data pointer to calculate offset from
   * @return Pointer to the kBackendHeaderSize-byte private header, or nullptr if data is null
   */
  template<typename T = char>
  HSHM_CROSS_FUN
  T *GetPrivateHeader(char *data) {
    if (data == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<T*>(data - priv_header_off_);
  }

  /**
   * Get pointer to the private header given a data pointer (const version).
   *
   * @tparam T Type to cast the private header to (default: char)
   * @param data The data pointer to calculate offset from
   * @return Const pointer to the kBackendHeaderSize-byte private header, or nullptr if data is null
   */
  template<typename T = char>
  HSHM_CROSS_FUN
  const T *GetPrivateHeader(const char *data) const {
    if (data == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<const T*>(data - priv_header_off_);
  }

  /**
   * Get pointer to the shared header given a data pointer.
   * Shared header is located kBackendHeaderSize bytes after the private header.
   *
   * @tparam T Type to cast the shared header to (default: char)
   * @param data The data pointer to calculate offset from
   * @return Pointer to the kBackendHeaderSize-byte shared header, or nullptr if data is null
   */
  template<typename T = char>
  HSHM_CROSS_FUN
  T *GetSharedHeader(char *data) {
    if (data == nullptr) {
      return nullptr;
    }
    char *priv = GetPrivateHeader<char>(data);
    return reinterpret_cast<T*>(priv + kBackendHeaderSize);
  }

  /**
   * Get pointer to the shared header given a data pointer (const version).
   * Shared header is located kBackendHeaderSize bytes after the private header.
   *
   * @tparam T Type to cast the shared header to (default: char)
   * @param data The data pointer to calculate offset from
   * @return Const pointer to the kBackendHeaderSize-byte shared header, or nullptr if data is null
   */
  template<typename T = char>
  HSHM_CROSS_FUN
  const T *GetSharedHeader(const char *data) const {
    if (data == nullptr) {
      return nullptr;
    }
    const char *priv = GetPrivateHeader<char>(data);
    return reinterpret_cast<const T*>(priv + kBackendHeaderSize);
  }
  
  /**
   * Get pointer to the shared header (4KB before data_, at data_ - kBackendHeaderSize)
   *
   * This region is shared between processes and typically used for
   * allocator-level shared metadata (e.g., custom allocator headers).
   * Located AFTER the private header (closer to data_).
   *
   * @tparam T Type to cast the shared header to (default: char)
   * @return Pointer to the kBackendHeaderSize-byte shared header, or nullptr if data_ is null
   */
  template<typename T = char>
  HSHM_CROSS_FUN
  T *GetSharedHeader() {
    return GetSharedHeader<T>(data_);
  }

  /**
   * Get pointer to the shared header (const version)
   *
   * @tparam T Type to cast the shared header to (default: char)
   * @return Const pointer to the kBackendHeaderSize-byte shared header, or nullptr if data_ is null
   */
  template<typename T = char>
  HSHM_CROSS_FUN
  const T *GetSharedHeader() const {
    return GetSharedHeader<T>(data_);
  }

  /**
   * Get pointer to the private header (process-local storage)
   *
   * This region is process-local and not shared between processes.
   * Each process that attaches gets its own independent copy.
   * Located BEFORE the shared header.
   * Useful for thread-local storage and process-specific metadata.
   *
   * @tparam T Type to cast the private header to (default: char)
   * @return Pointer to the kBackendHeaderSize-byte private header, or nullptr if data_ is null
   */
  template<typename T = char>
  HSHM_CROSS_FUN
  T *GetPrivateHeader() {
    return GetPrivateHeader<T>(data_);
  }

  /**
   * Get pointer to the private header (const version)
   *
   * @tparam T Type to cast the private header to (default: char)
   * @return Const pointer to the kBackendHeaderSize-byte private header, or nullptr if data_ is null
   */
  template<typename T = char>
  HSHM_CROSS_FUN
  const T *GetPrivateHeader() const {
    return GetPrivateHeader<T>(data_);
  }

  /**
   * Get size of the private header
   * @return Size of private header (always kBackendHeaderSize = 4KB)
   */
  HSHM_CROSS_FUN
  static constexpr size_t GetPrivateHeaderSize() {
    return kBackendHeaderSize;
  }

  /**
   * Get size of the shared header
   * @return Size of shared header (always kBackendHeaderSize = 4KB)
   */
  HSHM_CROSS_FUN
  static constexpr size_t GetSharedHeaderSize() {
    return kBackendHeaderSize;
  }

  /**
   * Cast data_ pointer to an Allocator type
   *
   * This allows treating the backend's data region as an allocator.
   * The allocator should be initialized in-place at the start of data_.
   * Note: data_offset_ indicates where the allocator's MANAGED region starts,
   * not where the allocator object itself is located.
   *
   * @return Pointer to allocator at the start of data_
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  AllocT* Cast() {
    return reinterpret_cast<AllocT*>(data_);
  }

  /**
   * Cast data_ pointer to an Allocator type (const version)
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  AllocT* Cast() const {
    return reinterpret_cast<AllocT*>(data_);
  }

  /** Friend declarations for kernel functions */
  template<typename AllocT, typename... Args>
  friend HSHM_GPU_KERNEL void MakeAllocKernel(MemoryBackend* backend, AllocT** result, Args... args);

  template<typename AllocT>
  friend HSHM_GPU_KERNEL void AttachAllocKernel(MemoryBackend* backend, AllocT** result);

  /**
   * Create and initialize an allocator in one line
   *
   * This method casts the data_ pointer to the allocator type,
   * constructs the allocator using placement new, and calls shm_init
   * with this backend as the first argument, followed by any additional arguments.
   *
   * If DoAccelPath() is true, executes via GPU kernel.
   *
   * @tparam AllocT The allocator type to create
   * @tparam Args Variadic template for additional shm_init arguments (after backend)
   * @param args Additional arguments to pass to shm_init (after the backend parameter)
   * @return Pointer to the constructed and initialized allocator
   */
  template<typename AllocT, typename... Args>
  HSHM_CROSS_FUN
  AllocT* MakeAlloc(Args&&... args) {
    if (!DoAccelPath()) {
      // Direct path: execute on current device
      AllocT* alloc = Cast<AllocT>();
      new (alloc) AllocT();
      alloc->shm_init(*this, std::forward<Args>(args)...);
      return alloc;
    } else {
#if HSHM_ENABLE_CUDA
      // Accelerator path: launch GPU kernel from host
      AllocT* result_host;
      AllocT** result_dev;
      cudaMalloc(&result_dev, sizeof(AllocT*));
      MakeAllocKernel<AllocT><<<1, 1>>>(this, result_dev, args...);
      cudaDeviceSynchronize();
      cudaMemcpy(&result_host, result_dev, sizeof(AllocT*), cudaMemcpyDeviceToHost);
      cudaFree(result_dev);
      return result_host;
#elif HSHM_ENABLE_ROCM
      // Accelerator path: launch GPU kernel from host (ROCm)
      AllocT* result_host;
      AllocT** result_dev;
      hipMalloc(&result_dev, sizeof(AllocT*));
      hipLaunchKernelGGL(MakeAllocKernel<AllocT>, dim3(1), dim3(1), 0, 0, this, result_dev, args...);
      hipDeviceSynchronize();
      hipMemcpy(&result_host, result_dev, sizeof(AllocT*), hipMemcpyDeviceToHost);
      hipFree(result_dev);
      return result_host;
#else
      // No GPU support compiled in - should not reach here
      return nullptr;
#endif
    }
  }

  /**
   * Attach to an existing allocator in one line
   *
   * This method casts the data_ pointer to the allocator type and
   * calls shm_attach to connect to the existing shared memory allocator.
   *
   * If DoAccelPath() is true, executes via GPU kernel.
   *
   * @tparam AllocT The allocator type to attach to
   * @return Pointer to the attached allocator
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  AllocT* AttachAlloc() {
    if (!DoAccelPath()) {
      // Direct path: execute on current device
      AllocT* alloc = Cast<AllocT>();
      alloc->shm_attach(*this);
      return alloc;
    } else {
#if HSHM_ENABLE_CUDA
      // Accelerator path: launch GPU kernel from host
      AllocT* result_host;
      AllocT** result_dev;
      cudaMalloc(&result_dev, sizeof(AllocT*));
      AttachAllocKernel<AllocT><<<1, 1>>>(this, result_dev);
      cudaDeviceSynchronize();
      cudaMemcpy(&result_host, result_dev, sizeof(AllocT*), cudaMemcpyDeviceToHost);
      cudaFree(result_dev);
      return result_host;
#elif HSHM_ENABLE_ROCM
      // Accelerator path: launch GPU kernel from host (ROCm)
      AllocT* result_host;
      AllocT** result_dev;
      hipMalloc(&result_dev, sizeof(AllocT*));
      hipLaunchKernelGGL(AttachAllocKernel<AllocT>, dim3(1), dim3(1), 0, 0, this, result_dev);
      hipDeviceSynchronize();
      hipMemcpy(&result_host, result_dev, sizeof(AllocT*), hipMemcpyDeviceToHost);
      hipFree(result_dev);
      return result_host;
#else
      // No GPU support compiled in - should not reach here
      return nullptr;
#endif
    }
  }

  HSHM_CROSS_FUN
  void Print() const {
    printf("(%s) MemoryBackend: region: %p, data: %p, data_capacity: %lu\n",
           kCurrentDevice, region_, data_, (long unsigned)data_capacity_);
  }

  /// Each allocator must define its own shm_init.
  // virtual bool shm_init(size_t size, ...) = 0;
  // virtual bool shm_attach(const hshm::chararr &url) = 0;
  // virtual void shm_detach() = 0;
  // virtual void shm_destroy() = 0;
};

/**
 * GPU kernel for MakeAlloc - constructs and initializes allocator on GPU
 */
template<typename AllocT, typename... Args>
HSHM_GPU_KERNEL
void MakeAllocKernel(MemoryBackend* backend, AllocT** result, Args... args) {
  AllocT* alloc = backend->template Cast<AllocT>();
  new (alloc) AllocT();
  alloc->shm_init(*backend, args...);
  *result = alloc;
}

/**
 * GPU kernel for AttachAlloc - attaches to allocator on GPU
 */
template<typename AllocT>
HSHM_GPU_KERNEL
void AttachAllocKernel(MemoryBackend* backend, AllocT** result) {
  AllocT* alloc = backend->template Cast<AllocT>();
  alloc->shm_attach(*backend);
  *result = alloc;
}

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_H
