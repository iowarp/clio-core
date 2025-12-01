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
#include <limits>
#include <string>
#include <vector>

#include "hermes_shm/constants/macros.h"
// #include "hermes_shm/data_structures/ipc/chararr.h"  // Deleted during hard refactoring
#include "hermes_shm/memory/allocator/allocator.h"

namespace hshm::ipc {

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

class MemoryBackend {
 public:
  MemoryBackendHeader *header_;
  char *md_;       // Metadata for how processes (on CPU) connect to this backend. Not required for allocators.
  size_t md_size_; // Metadata size. Not required for allocators.
  bitfield64_t flags_;
  char *data_;      // Data buffer for allocators
  size_t data_size_;// Data buffer size for allocators
  int data_id_;     // Device ID for the data buffer (GPU ID, etc.)
  u64 data_offset_; // Offset from root backend (0 if this is root, non-zero for sub-allocators)

 public:
  HSHM_CROSS_FUN
  MemoryBackend() : header_(nullptr), md_(nullptr), md_size_(0), data_(nullptr), data_size_(0), data_id_(-1), data_offset_(0) {}

  ~MemoryBackend() = default;

  /** Mark data as valid */
  HSHM_CROSS_FUN
  void SetInitialized() { flags_.SetBits(MEMORY_BACKEND_INITIALIZED); }

  /** Check if data is valid */
  HSHM_CROSS_FUN
  bool IsInitialized() { return flags_.Any(MEMORY_BACKEND_INITIALIZED); }

  /** Mark data as invalid */
  HSHM_CROSS_FUN
  void UnsetInitialized() { flags_.UnsetBits(MEMORY_BACKEND_INITIALIZED); }


  /** Mark data as having an allocation */
  HSHM_CROSS_FUN
  void SetHasAlloc() { header_->flags_.SetBits(MEMORY_BACKEND_HAS_ALLOC); }

  /** Check if data has an allocation */
  HSHM_CROSS_FUN
  bool IsHasAlloc() { return header_->flags_.Any(MEMORY_BACKEND_HAS_ALLOC); }

  /** Unmark data as having an allocation */
  HSHM_CROSS_FUN
  void UnsetHasAlloc() { header_->flags_.UnsetBits(MEMORY_BACKEND_HAS_ALLOC); }

  /** This is the process which destroys the backend */
  HSHM_CROSS_FUN
  void Own() { flags_.SetBits(MEMORY_BACKEND_OWNED); }

  /** This is owned */
  HSHM_CROSS_FUN
  bool IsOwned() { return flags_.Any(MEMORY_BACKEND_OWNED); }

  /** This is not the process which destroys the backend */
  HSHM_CROSS_FUN
  void Disown() { flags_.UnsetBits(MEMORY_BACKEND_OWNED); }

  /** Get the ID of this backend */
  HSHM_CROSS_FUN
  MemoryBackendId &GetId() { return header_->id_; }

  /** Get the ID of this backend */
  HSHM_CROSS_FUN
  const MemoryBackendId &GetId() const { return header_->id_; }

  /**
   * Shift the data pointer by an offset amount
   * Updates both data_size_ and data_offset_ fields
   *
   * @param offset The amount to shift the data pointer
   */
  HSHM_CROSS_FUN
  void Shift(size_t offset) {
    data_ += offset;
    data_size_ -= offset;
    data_offset_ += offset;
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

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_H
