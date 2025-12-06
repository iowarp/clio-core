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
template<typename T, typename PointerT>
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
  size_t md_size_;    // Metadata size for process connection
  MemoryBackendId id_;
  bitfield64_t flags_;
  size_t data_size_;  // Actual data buffer size for allocators
  int data_id_;       // Device ID for the data buffer (GPU ID, etc.)

  HSHM_CROSS_FUN void Print() const {
    printf("(%s) MemoryBackendHeader: id: (%u, %u), md_size: %lu, data_size: %lu\n",
           kCurrentDevice, id_.major_, id_.minor_, (long unsigned)md_size_, (long unsigned)data_size_);
  }
};

#define MEMORY_BACKEND_INITIALIZED BIT_OPT(u64, 0)
#define MEMORY_BACKEND_OWNED BIT_OPT(u64, 1)
#define MEMORY_BACKEND_HAS_ALLOC BIT_OPT(u64, 2)
#define MEMORY_BACKEND_HAS_GPU_ALLOC BIT_OPT(u64, 3)
#define MEMORY_BACKEND_IS_SCANNED BIT_OPT(u64, 4)

class UrlMemoryBackend {};

/**
 * Global constant for backend header sizes
 * Each header (shared and private) is 4KB
 */
static constexpr size_t kBackendHeaderSize = 4 * 1024;  // 4KB per header (shared + private = 8KB total)

class MemoryBackend {
 public:
  MemoryBackendHeader *header_;
  char *md_;       // Metadata for how processes (on CPU) connect to this backend. Not required for allocators.
  size_t md_size_; // Metadata size. Not required for allocators.
  bitfield64_t flags_;
  char *data_;      // Data buffer for allocators (points to the SHARED part of the region)
  size_t data_size_;// Data buffer size for allocators (size of SHARED region only)
  size_t data_capacity_; // Full size of backend (doesn't change with shift)
  int data_id_;     // Device ID for the data buffer (GPU ID, etc.)
  u64 data_offset_; // Offset from root backend (0 if this is root, non-zero for sub-allocators)
  size_t priv_header_off_; // Offset from buffer start to private header (dynamic tracking)

 public:
  HSHM_CROSS_FUN
  MemoryBackend() : header_(nullptr), md_(nullptr), md_size_(0), data_(nullptr), data_size_(0), data_capacity_(0), data_id_(-1), data_offset_(0), priv_header_off_(0) {}

  ~MemoryBackend() = default;

  /** Get the ID of this backend */
  HSHM_CROSS_FUN
  MemoryBackendId &GetId() { return header_->id_; }

  /** Get the ID of this backend */
  HSHM_CROSS_FUN
  const MemoryBackendId &GetId() const { return header_->id_; }

  /**
   * Create a shifted copy of this memory backend
   * Updates data_, data_size_, and data_offset_ fields in the returned copy
   *
   * @param offset The amount to shift the data pointer
   * @return A new MemoryBackend with shifted data pointer and updated offset
   */
  HSHM_CROSS_FUN
  MemoryBackend Shift(size_t offset) const {
    MemoryBackend shifted = *this;
    shifted.data_ = data_ + offset;
    shifted.data_size_ -= offset;
    shifted.data_offset_ += offset;
    return shifted;
  }

  /**
   * Create a shifted backend positioned at a specific offset
   *
   * @param off The offset to shift to
   * @return A new MemoryBackend positioned at the offset
   */
  HSHM_CROSS_FUN
  MemoryBackend ShiftTo(size_t off) const {
    MemoryBackend shifted = *this;
    size_t shift_amount = off - data_offset_;
    shifted.data_size_ -= shift_amount;
    shifted.data_offset_ = off;
    return shifted;
  }

  /**
   * Create a shifted backend positioned at a raw pointer location
   *
   * Updates only the data_offset to track where in the root backend this pointer is located.
   * data_ remains unchanged (still points to root data). The data_offset is calculated as
   * the distance from the root data_ to the given pointer.
   *
   * @param ptr Raw pointer to position the backend at
   * @param region_size Total size of the managed region
   * @return A new MemoryBackend with updated data_offset
   */
  HSHM_CROSS_FUN
  MemoryBackend ShiftTo(char *ptr, size_t region_size) const {
    MemoryBackend shifted = *this;
    shifted.data_offset_ = static_cast<size_t>(ptr - data_);
    shifted.data_size_ = region_size;
    return shifted;
  }

  /**
   * Create a shifted backend positioned at a specific FullPtr location
   *
   * @tparam T The type pointed to by the FullPtr
   * @tparam PointerT The pointer type used in FullPtr
   * @param ptr The FullPtr indicating where to position the backend
   * @param size The size of the new backend region
   * @return A new MemoryBackend positioned at ptr's offset with the given size
   */
  template<typename T, typename PointerT>
  HSHM_CROSS_FUN
  MemoryBackend ShiftTo(FullPtr<T, PointerT> ptr, size_t size) const;

  /**
   * Get pointer to the private header given a data pointer.
   * Uses the dynamic priv_header_off_ offset for correct calculation.
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
   * Uses the dynamic priv_header_off_ offset for correct calculation.
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
  const AllocT* Cast() const {
    return reinterpret_cast<const AllocT*>(data_);
  }

  /**
   * Create and initialize an allocator in one line
   *
   * This method casts the data_ pointer to the allocator type,
   * constructs the allocator using placement new, and calls shm_init
   * with this backend as the first argument, followed by any additional arguments.
   *
   * @tparam AllocT The allocator type to create
   * @tparam Args Variadic template for additional shm_init arguments (after backend)
   * @param args Additional arguments to pass to shm_init (after the backend parameter)
   * @return Pointer to the constructed and initialized allocator
   */
  template<typename AllocT, typename... Args>
  HSHM_CROSS_FUN
  AllocT* MakeAlloc(Args&&... args) {
    AllocT* alloc = Cast<AllocT>();
    new (alloc) AllocT();
    alloc->shm_init(*this, std::forward<Args>(args)...);
    return alloc;
  }

  /**
   * Attach to an existing allocator in one line
   *
   * This method casts the data_ pointer to the allocator type and
   * calls shm_attach to connect to the existing shared memory allocator.
   *
   * @tparam AllocT The allocator type to attach to
   * @return Pointer to the attached allocator
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  AllocT* AttachAlloc() {
    AllocT* alloc = Cast<AllocT>();
    alloc->shm_attach(*this);
    return alloc;
  }

  HSHM_CROSS_FUN
  void Print() const {
    header_->Print();
    printf("(%s) MemoryBackend: md: %p, md_size: %lu, data: %p, data_size: %lu, data_offset: %lu\n",
           kCurrentDevice, md_, (long unsigned)md_size_, data_, (long unsigned)data_size_, (long unsigned)data_offset_);
  }

  /// Each allocator must define its own shm_init.
  // virtual bool shm_init(size_t size, ...) = 0;
  // virtual bool shm_attach(const hshm::chararr &url) = 0;
  // virtual void shm_detach() = 0;
  // virtual void shm_destroy() = 0;
};

// Implementation of ShiftTo template method (defined after FullPtr is fully declared in allocator.h)
template<typename T, typename PointerT>
HSHM_CROSS_FUN
MemoryBackend MemoryBackend::ShiftTo(FullPtr<T, PointerT> ptr, size_t size) const {
  MemoryBackend shifted = *this;
  shifted.data_offset_ = ptr.shm_.off_.load();
  shifted.data_size_ = size;
  // NOTE: Do NOT update data_ - it must remain pointing to the root backend data
  // The allocator may be located at a different address than data_ + data_offset_
  return shifted;
}

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_H
