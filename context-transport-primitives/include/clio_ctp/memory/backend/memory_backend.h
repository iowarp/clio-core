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

#ifndef CTP_MEMORY_H
#define CTP_MEMORY_H

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "clio_ctp/constants/macros.h"
// #include "clio_ctp/data_structures/ipc/chararr.h"  // Deleted during hard refactoring
#include "clio_ctp/memory/allocator/allocator.h"

namespace ctp::ipc {

/** Forward declaration for FullPtr (defined in allocator.h after this header) */
template <typename T, bool ATOMIC>
struct FullPtr;

/** ID for memory backend */
class MemoryBackendId {
 public:
  u32 major_;  // Major ID (e.g., PID)
  u32 minor_;  // Minor ID (relative to major)

  CTP_CROSS_FUN
  MemoryBackendId() : major_(0), minor_(0) {}

  CTP_CROSS_FUN
  MemoryBackendId(u32 major, u32 minor) : major_(major), minor_(minor) {}

  CTP_CROSS_FUN
  MemoryBackendId(const MemoryBackendId &other) : major_(other.major_), minor_(other.minor_) {}

  CTP_CROSS_FUN
  MemoryBackendId(MemoryBackendId &&other) noexcept : major_(other.major_), minor_(other.minor_) {}

  CTP_CROSS_FUN
  MemoryBackendId &operator=(const MemoryBackendId &other) {
    major_ = other.major_;
    minor_ = other.minor_;
    return *this;
  }

  CTP_CROSS_FUN
  MemoryBackendId &operator=(MemoryBackendId &&other) noexcept {
    major_ = other.major_;
    minor_ = other.minor_;
    return *this;
  }

  CTP_CROSS_FUN
  static MemoryBackendId GetRoot() { return {0, 0}; }

  CTP_CROSS_FUN
  static MemoryBackendId Get(u32 major, u32 minor) { return {major, minor}; }

  CTP_CROSS_FUN
  bool operator==(const MemoryBackendId &other) const {
    return major_ == other.major_ && minor_ == other.minor_;
  }

  CTP_CROSS_FUN
  bool operator!=(const MemoryBackendId &other) const {
    return major_ != other.major_ || minor_ != other.minor_;
  }

  /** Get the null backend ID */
  CTP_CROSS_FUN
  static MemoryBackendId GetNull() {
    return MemoryBackendId(UINT32_MAX, UINT32_MAX);
  }

  /**
   * THE OFFSET IS A RAW GPU DEVICE ADDRESS.
   *
   * Distinct from GetNull(), which also means "the offset is the address" but
   * refers to HOST memory the receiving process can simply dereference. A
   * device address can only be dereferenced in the owning GPU context, and it
   * has to be moved with a device-aware copy -- so the two cases cannot share
   * a tag. They did: the GPU vector built its page pointers with GetNull(),
   * which made a device address indistinguishable from a host one and left
   * "run this task where the pointer is valid" as an unwritten rule enforced
   * by pinning the task to the local node. Labelling the pointer is what lets
   * a holder ASK, and bounce it when it has to cross a node.
   */
  CTP_CROSS_FUN
  static MemoryBackendId GetGpuPointer() {
    return MemoryBackendId(UINT32_MAX, UINT32_MAX - 1);
  }

  /** Is this the GPU-device-address tag? See GetGpuPointer. */
  CTP_CROSS_FUN
  bool IsGpuPointer() const { return *this == GetGpuPointer(); }

  /** Does `off_` hold a raw address rather than an allocator offset? True for
   *  both the host (null) and device (GPU pointer) forms. */
  CTP_CROSS_FUN
  bool IsRawAddress() const { return IsNull() || IsGpuPointer(); }

  /** Set this backend ID to null */
  CTP_CROSS_FUN
  void SetNull() { *this = GetNull(); }

  /** Check if this is the null backend ID */
  CTP_CROSS_FUN
  bool IsNull() const { return *this == GetNull(); }

  /** To index */
  CTP_CROSS_FUN
  uint32_t ToIndex() const {
    return major_ * 2 + minor_;
  }

  /** Serialize */
  template <typename Ar>
  CTP_CROSS_FUN void serialize(Ar &ar) {
    ar & major_;
    ar & minor_;
  }

  /** Print */
  CTP_CROSS_FUN
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

  CTP_CROSS_FUN void Print() const {
    printf("(%s) MemoryBackendHeader: id: (%u, %u), backend_size: %lu, data_capacity: %lu\n",
           kCurrentDevice, id_.major_, id_.minor_, (long unsigned)backend_size_, (long unsigned)data_capacity_);
  }
};

#define MEMORY_BACKEND_INITIALIZED BIT_OPT(u64, 0)
#define MEMORY_BACKEND_OWNED BIT_OPT(u64, 1)

class UrlMemoryBackend {};

/** Size of the backend header region.
 *  Sized to 64 KiB so the data region's file offset (= kBackendHeaderSize)
 *  satisfies Windows' MapViewOfFile alignment rule: the offset must be a
 *  multiple of SYSTEM_INFO::dwAllocationGranularity, which is 64 KiB on
 *  x64/ARM64 Windows (NOT the 4 KiB page size). The same constant is used
 *  on POSIX, where the only constraint is page alignment, so the extra
 *  60 KiB of overhead per backend is harmless (a typical segment is >=1 GiB).
 *  TODO: revisit per GH issue — a single mmap split into header+data slices
 *  would avoid the constant bump entirely. */
static const size_t kBackendHeaderSize = 65536;

class MemoryBackend : public MemoryBackendHeader {
 public:
  MemoryBackendHeader *header_;  // Pointer to persisted header in region_
  char *region_;    // The entire allocated/mapped region
  char *data_;      // Data buffer (region_ + kBackendHeaderSize)

 public:
  CTP_CROSS_FUN
  MemoryBackend() : header_(nullptr), region_(nullptr), data_(nullptr) {
    // Initialize inherited MemoryBackendHeader fields
    id_ = MemoryBackendId();
    flags_.Clear();
    backend_size_ = 0;
    data_capacity_ = 0;
    data_id_ = -1;
  }

  ~MemoryBackend() = default;

  /** Get the ID of this backend */
  CTP_CROSS_FUN
  MemoryBackendId &GetId() { return id_; }

  /** Get the ID of this backend */
  CTP_CROSS_FUN
  const MemoryBackendId &GetId() const { return id_; }

  /**
   * Set the MEMORY_BACKEND_OWNED flag
   * Called during shm_init to indicate this process owns the backend
   */
  CTP_CROSS_FUN
  void SetOwner() {
    flags_.SetBits(MEMORY_BACKEND_OWNED);
  }

  /**
   * Check if this process owns the backend
   * @return true if MEMORY_BACKEND_OWNED flag is set
   */
  CTP_CROSS_FUN
  bool IsOwner() const {
    return flags_.Any(MEMORY_BACKEND_OWNED) != 0;
  }

  /**
   * Unset the MEMORY_BACKEND_OWNED flag
   * Called during shm_attach to indicate this process is attaching to
   * a backend created by another process
   */
  CTP_CROSS_FUN
  void UnsetOwner() {
    flags_.UnsetBits(MEMORY_BACKEND_OWNED);
  }

  /**
   * Create a sub-backend view over a portion of this backend's data region.
   *
   * The returned backend shares the same id_ but has data_ and data_capacity_
   * adjusted to cover [data_ + offset, data_ + offset + size).
   *
   * @param offset Byte offset from data_ where the clip starts
   * @param size   Size of the clipped region in bytes
   * @return A new MemoryBackend describing the sub-region
   */
  CTP_CROSS_FUN
  MemoryBackend Clip(size_t offset, size_t size) const {
    MemoryBackend sub;
    sub.id_ = id_;
    sub.data_ = data_ + offset;
    sub.data_capacity_ = size;
    return sub;
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
  CTP_CROSS_FUN
  AllocT* Cast() {
    return reinterpret_cast<AllocT*>(data_);
  }

  /**
   * Cast data_ pointer to an Allocator type (const version)
   */
  template<typename AllocT>
  CTP_CROSS_FUN
  AllocT* Cast() const {
    return reinterpret_cast<AllocT*>(data_);
  }

  /**
   * Create and initialize an allocator in one line
   *
   * This method casts the data_ pointer to the allocator type,
   * constructs the allocator using placement new, and calls shm_init
   * with this backend as the first argument, followed by any additional arguments.
   *
   * The allocator supports execution on both GPU and CPU using pinned host memory
   * and mallocManaged, so no special GPU kernel path is needed.
   *
   * @tparam AllocT The allocator type to create
   * @tparam Args Variadic template for additional shm_init arguments (after backend)
   * @param args Additional arguments to pass to shm_init (after the backend parameter)
   * @return Pointer to the constructed and initialized allocator
   */
  template<typename AllocT, typename... Args>
  CTP_CROSS_FUN
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
   * The allocator supports execution on both GPU and CPU using pinned host memory
   * and mallocManaged, so no special GPU kernel path is needed.
   *
   * @tparam AllocT The allocator type to attach to
   * @return Pointer to the attached allocator
   */
  template<typename AllocT>
  CTP_CROSS_FUN
  AllocT* AttachAlloc() {
    AllocT* alloc = Cast<AllocT>();
    alloc->shm_attach(*this);
    return alloc;
  }

  CTP_CROSS_FUN
  void Print() const {
    printf("(%s) MemoryBackend: region: %p, data: %p, data_capacity: %lu\n",
           kCurrentDevice, region_, data_, (long unsigned)data_capacity_);
  }

  /// Each allocator must define its own shm_init.
  // virtual bool shm_init(size_t size, ...) = 0;
  // virtual bool shm_attach(const ctp::chararr &url) = 0;
  // virtual void shm_detach() = 0;
  // virtual void shm_destroy() = 0;
};

}  // namespace ctp::ipc

#endif  // CTP_MEMORY_H
