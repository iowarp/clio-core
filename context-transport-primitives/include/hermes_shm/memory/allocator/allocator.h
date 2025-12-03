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

#ifndef HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/introspect/system_info.h"
#include "hermes_shm/memory/backend/memory_backend.h"
#include "hermes_shm/memory/backend/array_backend.h"
#include "hermes_shm/thread/thread_model/thread_model.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/types/bitfield.h"
#include "hermes_shm/types/hash.h"
#include "hermes_shm/types/numbers.h"
#include "hermes_shm/types/real_number.h"
#include "hermes_shm/util/errors.h"

namespace hshm::ipc {

/**
 * AllocatorId is now just an alias for MemoryBackendId
 * Kept for backward compatibility
 * */
using AllocatorId = MemoryBackendId;

class Allocator;

/** ShmPtr type base */
class ShmPointer {};

/** Forward declarations for pointer types */
template<typename T = void, bool ATOMIC = false>
struct OffsetPtrBase;

template<typename T = void, bool ATOMIC = false>
struct ShmPtrBase;

/**
 * The basic shared-memory allocator header.
 * Allocators inherit from this.
 * */
struct AllocatorHeader {
  size_t custom_header_size_;
  hipc::atomic<hshm::size_t> total_alloc_;

  HSHM_CROSS_FUN
  AllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(size_t custom_header_size) {
    custom_header_size_ = custom_header_size;
    total_alloc_ = 0;
  }

  HSHM_INLINE_CROSS_FUN
  void AddSize(hshm::size_t size) {
#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ += size;
#endif
  }

  HSHM_INLINE_CROSS_FUN
  void SubSize(hshm::size_t size) {
#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ -= size;
#endif
  }

  HSHM_INLINE_CROSS_FUN
  hshm::size_t GetCurrentlyAllocatedSize() { return total_alloc_.load(); }
};

/** The allocator information struct */
class Allocator {
 public:
  size_t alloc_header_size_;  /**< Size of the allocator object (sizeof derived class) */
  size_t custom_header_size_;  /**< Size of custom header after allocator object */

 private:
  MemoryBackend backend_;  /**< Memory backend (not fully compatible with shared memory) */

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Allocator() : alloc_header_size_(0), custom_header_size_(0) {}

  /** Get the allocator identifier from backend */
  HSHM_INLINE_CROSS_FUN
  MemoryBackendId GetId() const { return backend_.GetId(); }

  /** Get the allocator identifier (non-const version) */
  HSHM_INLINE_CROSS_FUN
  MemoryBackendId GetId() { return backend_.GetId(); }

  /**
   * Get backend data pointer (reconstructs from allocator position)
   */
  HSHM_INLINE_CROSS_FUN
  char* GetBackendData() const {
    return reinterpret_cast<char*>(const_cast<Allocator*>(this)) - backend_.data_offset_;
  }

  /**
   * Get backend shift (data_offset_)
   * @return The offset from backend data pointer to the allocator's region
   */
  HSHM_INLINE_CROSS_FUN
  size_t GetBackendShift() const {
    return backend_.data_offset_;
  }

  /**
   * Get typed private header (process-local storage before backend.data_)
   * This region is kBackendPrivate bytes and is NOT shared between processes
   * @return Pointer to private header of type HEADER_T
   */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T* GetPrivateHeader() const {
    return reinterpret_cast<HEADER_T*>(GetBackendData() - kBackendPrivate);
  }

  /**
   * Get a copy of the memory backend
   * Reconstructs backend.data_ to point to this allocator
   */
  HSHM_INLINE_CROSS_FUN
  MemoryBackend GetBackend() const {
    MemoryBackend backend = backend_;
    backend.data_ = GetBackendData();
    return backend;
  }

  /**
   * Get a shifted backend where data_ points to data_ + data_offset_
   * This is useful for sub-allocators that need to operate on a region
   * within the parent allocator's space.
   *
   * @return Backend with data_ = GetBackendData() + data_offset_
   */
  HSHM_INLINE_CROSS_FUN
  MemoryBackend GetShiftedBackend() const {
    MemoryBackend backend = backend_;
    backend.data_ = GetBackendData() + backend_.data_offset_;
    return backend;
  }

  /**
   * Get the custom header of the shared-memory allocator
   * The custom header is located immediately after the allocator object
   *
   * @return Custom header pointer
   */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetCustomHeader() {
    return reinterpret_cast<HEADER_T*>(reinterpret_cast<char*>(this) + alloc_header_size_);
  }

  /**
   * Get the custom header of the shared-memory allocator (const)
   *
   * @return Custom header pointer
   */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN const HEADER_T *GetCustomHeader() const {
    return reinterpret_cast<const HEADER_T*>(reinterpret_cast<const char*>(this) + alloc_header_size_);
  }

  /**
   * Construct custom header
   */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *ConstructHeader(void *buffer) {
    new ((HEADER_T *)buffer) HEADER_T();
    return reinterpret_cast<HEADER_T *>(buffer);
  }

  /**
   * Get the size of the backend data region
   * This is the total size available to the allocator
   *
   * @return Size of backend data region in bytes
   */
  HSHM_INLINE_CROSS_FUN
  size_t GetBackendDataSize() const {
    return backend_.data_size_;
  }

  /**
   * Get the start of the allocator's data region (after allocator object and custom header)
   * This is where the allocator's managed heap begins
   *
   * @return Pointer to the start of allocator data
   */
  HSHM_INLINE_CROSS_FUN
  char* GetAllocatorDataStart() const {
    return const_cast<char*>(GetCustomHeader<char>() + custom_header_size_);
  }

  /**
   * Get the offset of the allocator's data region (relative to backend data_)
   * This is the offset where the allocator's managed heap begins
   *
   * @return Offset from backend data_ to allocator data start
   */
  HSHM_INLINE_CROSS_FUN
  size_t GetAllocatorDataOff() const {
    return GetAllocatorDataStart() - GetBackendData();
  }

  /**
   * Get the size of the allocator's data region (excluding allocator object and custom header)
   * This is the size available for the allocator's managed heap
   *
   * @return Size of allocator data region in bytes
   */
  HSHM_INLINE_CROSS_FUN
  size_t GetAllocatorDataSize() const {
    return GetBackendDataSize() - (GetAllocatorDataStart() - reinterpret_cast<const char*>(this));
  }

  /**
   * Determine whether or not this allocator contains a process-specific
   * pointer
   *
   * @param ptr process-specific pointer
   * @return True or false
   * */
  template <typename T = void>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const T *ptr) const {
    MemoryBackend backend = GetBackend();
    size_t offset = reinterpret_cast<const char*>(ptr) - backend.data_;
    return offset >= backend.data_offset_ && offset < (backend.data_offset_ + backend.data_size_);
  }

  /**
   * Check if an OffsetPtr is within this allocator's region
   */
  template<typename T, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const OffsetPtrBase<T, ATOMIC> &ptr) const {
    MemoryBackend backend = GetBackend();
    size_t off = ptr.off_.load();
    return off >= backend.data_offset_ && off < (backend.data_offset_ + backend.data_size_);
  }

  /**
   * Check if a ShmPtr is within this allocator's region
   */
  template<typename T, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const ShmPtrBase<T, ATOMIC> &ptr) const {
    MemoryBackend backend = GetBackend();
    size_t off = ptr.off_.load();
    return off >= backend.data_offset_ && off < (backend.data_offset_ + backend.data_size_);
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() {
    MemoryBackend backend = GetBackend();
    printf("(%s) Allocator: id: (%u,%u), size: %lu\n",
           kCurrentDevice, GetId().major_,
           GetId().minor_, (unsigned long)backend.data_size_);
  }

 protected:
  /** Set the backend (for use by derived classes during initialization) */
  HSHM_INLINE_CROSS_FUN
  void SetBackend(const MemoryBackend &backend) {
    backend_ = backend;
  }

  /**====================================
   * Object Constructors
   * ===================================*/

  /**
   * Construct each object in an array of objects.
   *
   * @param ptr the array of objects (potentially archived)
   * @param old_count the original size of the ptr
   * @param new_count the new size of the ptr
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObjs(T *ptr, size_t old_count,
                                                  size_t new_count,
                                                  Args &&...args) {
    if (ptr == nullptr) {
      return;
    }
    for (size_t i = old_count; i < new_count; ++i) {
      ConstructObj<T>(*(ptr + i), std::forward<Args>(args)...);
    }
  }

  /**
   * Construct an object.
   *
   * @param ptr the object to construct (potentially archived)
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObj(T &obj, Args &&...args) {
    new (&obj) T(std::forward<Args>(args)...);
  }

  /**
   * Destruct an array of objects
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObjs(T *ptr, size_t count) {
    if (ptr == nullptr) {
      return;
    }
    for (size_t i = 0; i < count; ++i) {
      DestructObj<T>(*(ptr + i));
    }
  }

  /**
   * Destruct an object
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObj(T &obj) {
    obj.~T();
  }
};

/**
 * Stores an offset into a memory region. Assumes the developer knows
 * which allocator the pointer comes from.
 *
 * @tparam T The type being pointed to (void by default)
 * @tparam ATOMIC Whether the offset should be atomic
 * */
template <typename T, bool ATOMIC>
struct OffsetPtrBase : public ShmPointer {
  hipc::opt_atomic<hshm::size_t, ATOMIC>
      off_; /**< Offset within the allocator's slot */

  /** Serialize an hipc::OffsetPtrBase */
  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar & off_;
  }

  /** ostream operator */
  friend std::ostream &operator<<(std::ostream &os,
                                  const OffsetPtrBase &ptr) {
    os << ptr.off_.load();
    return os;
  }

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPtrBase(size_t off) : off_(off) {}

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPtrBase(
      hipc::opt_atomic<hshm::size_t, ATOMIC> off)
      : off_(off.load()) {}

  /** Copy constructor from different type */
  template<typename U>
  HSHM_INLINE_CROSS_FUN explicit OffsetPtrBase(const OffsetPtrBase<U, ATOMIC> &other)
      : off_(other.off_.load()) {}

  /** ShmPtr constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPtrBase(AllocatorId alloc_id,
                                                   size_t off)
      : off_(off) {
    (void)alloc_id;
  }

  /** ShmPtr constructor (alloc + atomic offset)*/
  HSHM_INLINE_CROSS_FUN explicit OffsetPtrBase(AllocatorId id,
                                                   OffsetPtrBase<T, true> off)
      : off_(off.load()) {
    (void)id;
  }

  /** ShmPtr constructor (alloc + non-offeset) */
  HSHM_INLINE_CROSS_FUN explicit OffsetPtrBase(AllocatorId id,
                                                   OffsetPtrBase<T, false> off)
      : off_(off.load()) {
    (void)id;
  }

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase(const OffsetPtrBase &other)
      : off_(other.off_.load()) {}

  /** Other copy constructor */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase(
      const OffsetPtrBase<T, !ATOMIC> &other)
      : off_(other.off_.load()) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase(OffsetPtrBase &&other) noexcept
      : off_(other.off_.load()) {
    other.SetNull();
  }

  /** Get the offset pointer */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase<T, false> ToOffsetPtr() const {
    return OffsetPtrBase<T, false>(off_.load());
  }

  /** Set to null (offsets can be 0, so not 0) */
  HSHM_INLINE_CROSS_FUN void SetNull() { off_ = (size_t)-1; }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const {
    return off_.load() == (size_t)-1;
  }

  /** Get the null pointer */
  HSHM_INLINE_CROSS_FUN static OffsetPtrBase GetNull() {
    return OffsetPtrBase((size_t)-1);
  }

  /** Atomic load wrapper */
  HSHM_INLINE_CROSS_FUN size_t
  load(std::memory_order order = std::memory_order_seq_cst) const {
    return off_.load(order);
  }

  /** Atomic exchange wrapper */
  HSHM_INLINE_CROSS_FUN void exchange(
      size_t count, std::memory_order order = std::memory_order_seq_cst) {
    off_.exchange(count, order);
  }

  /** Atomic compare exchange weak wrapper */
  HSHM_INLINE_CROSS_FUN bool compare_exchange_weak(
      size_t &expected, size_t desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return off_.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic compare exchange strong wrapper */
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong(
      size_t &expected, size_t desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return off_.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic add operator */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase operator+(size_t count) const {
    return OffsetPtrBase(off_ + count);
  }

  /** Atomic subtract operator */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase operator-(size_t count) const {
    return OffsetPtrBase(off_ - count);
  }

  /** Atomic add assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase &operator+=(size_t count) {
    off_ += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase &operator-=(size_t count) {
    off_ -= count;
    return *this;
  }

  /** Atomic increment (post) */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase operator++(int) {
    return OffsetPtrBase(off_++);
  }

  /** Atomic increment (pre) */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase &operator++() {
    ++off_;
    return *this;
  }

  /** Atomic decrement (post) */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase operator--(int) {
    return OffsetPtrBase(off_--);
  }

  /** Atomic decrement (pre) */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase &operator--() {
    --off_;
    return *this;
  }

  /** Atomic assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase &operator=(size_t count) {
    off_ = count;
    return *this;
  }

  /** Atomic copy assign operator */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase &operator=(
      const OffsetPtrBase &count) {
    off_ = count.load();
    return *this;
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const OffsetPtrBase &other) const {
    return off_ == other.off_;
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const OffsetPtrBase &other) const {
    return off_ != other.off_;
  }

  /** Mark first bit */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase Mark() const {
    return OffsetPtrBase(MARK_FIRST_BIT(size_t, off_.load()));
  }

  /** Check if first bit is marked */
  HSHM_INLINE_CROSS_FUN bool IsMarked() const {
    return IS_FIRST_BIT_MARKED(size_t, off_.load());
  }

  /** Unmark first bit */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase Unmark() const {
    return OffsetPtrBase(UNMARK_FIRST_BIT(size_t, off_.load()));
  }

  /** Set to 0 */
  HSHM_INLINE_CROSS_FUN void SetZero() { off_ = 0; }
};

/** Non-atomic offset pointer (defaults to void*) */
template <typename T = void>
using OffsetPtr = OffsetPtrBase<T, false>;

/** Atomic offset pointer (defaults to void*) */
template <typename T = void>
using AtomicOffsetPtr = OffsetPtrBase<T, true>;

/**
 * A process-independent pointer, which stores both the allocator's
 * information and the offset within the allocator's region
 *
 * @tparam T The type being pointed to (void by default)
 * @tparam ATOMIC Whether the offset should be atomic
 * */
template <typename T, bool ATOMIC>
struct ShmPtrBase : public ShmPointer {
  AllocatorId alloc_id_;           /// Allocator the pointer comes from
  OffsetPtrBase<T, ATOMIC> off_;  /// Offset within the allocator's slot

  /** Serialize a pointer */
  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar & alloc_id_;
    ar & off_;
  }

  /** Ostream operator */
  friend std::ostream &operator<<(std::ostream &os, const ShmPtrBase &ptr) {
    os << ptr.alloc_id_ << "::" << ptr.off_;
    return os;
  }

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN ShmPtrBase() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN explicit ShmPtrBase(AllocatorId id, size_t off)
      : alloc_id_(id), off_(off) {}

  /** Full constructor using offset pointer */
  HSHM_INLINE_CROSS_FUN explicit ShmPtrBase(AllocatorId id, OffsetPtrBase<T, ATOMIC> off)
      : alloc_id_(id), off_(off) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN ShmPtrBase(const ShmPtrBase &other)
      : alloc_id_(other.alloc_id_), off_(other.off_) {}

  /** Other copy constructor (different atomicity) */
  HSHM_INLINE_CROSS_FUN ShmPtrBase(const ShmPtrBase<T, !ATOMIC> &other)
      : alloc_id_(other.alloc_id_), off_(other.off_.load()) {}

  /** Copy constructor from different type */
  template<typename U>
  HSHM_INLINE_CROSS_FUN explicit ShmPtrBase(const ShmPtrBase<U, ATOMIC> &other)
      : alloc_id_(other.alloc_id_), off_(other.off_.load()) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN ShmPtrBase(ShmPtrBase &&other) noexcept
      : alloc_id_(other.alloc_id_), off_(other.off_) {
    other.SetNull();
  }

  /** Get the offset pointer */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase<T, false> ToOffsetPtr() const {
    return OffsetPtrBase<T, false>(off_.load());
  }

  /** Set to null */
  HSHM_INLINE_CROSS_FUN void SetNull() { alloc_id_.SetNull(); }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return alloc_id_.IsNull(); }

  /** Get the null pointer */
  HSHM_INLINE_CROSS_FUN static ShmPtrBase GetNull() {
    return ShmPtrBase{AllocatorId::GetNull(), OffsetPtrBase<T, ATOMIC>::GetNull()};
  }

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN ShmPtrBase &operator=(const ShmPtrBase &other) {
    if (this != &other) {
      alloc_id_ = other.alloc_id_;
      off_ = other.off_;
    }
    return *this;
  }

  /** Move assignment operator */
  HSHM_INLINE_CROSS_FUN ShmPtrBase &operator=(ShmPtrBase &&other) {
    if (this != &other) {
      alloc_id_ = other.alloc_id_;
      off_.exchange(other.off_.load());
      other.SetNull();
    }
    return *this;
  }

  /** Addition operator */
  HSHM_INLINE_CROSS_FUN ShmPtrBase operator+(size_t size) const {
    ShmPtrBase p;
    p.alloc_id_ = alloc_id_;
    p.off_ = off_ + size;
    return p;
  }

  /** Subtraction operator */
  HSHM_INLINE_CROSS_FUN ShmPtrBase operator-(size_t size) const {
    ShmPtrBase p;
    p.alloc_id_ = alloc_id_;
    p.off_ = off_ - size;
    return p;
  }

  /** Addition assignment operator */
  HSHM_INLINE_CROSS_FUN ShmPtrBase &operator+=(size_t size) {
    off_ += size;
    return *this;
  }

  /** Subtraction assignment operator */
  HSHM_INLINE_CROSS_FUN ShmPtrBase &operator-=(size_t size) {
    off_ -= size;
    return *this;
  }

  /** Increment operator (pre) */
  HSHM_INLINE_CROSS_FUN ShmPtrBase &operator++() {
    off_++;
    return *this;
  }

  /** Decrement operator (pre) */
  HSHM_INLINE_CROSS_FUN ShmPtrBase &operator--() {
    off_--;
    return *this;
  }

  /** Increment operator (post) */
  HSHM_INLINE_CROSS_FUN ShmPtrBase operator++(int) {
    ShmPtrBase tmp(*this);
    operator++();
    return tmp;
  }

  /** Decrement operator (post) */
  HSHM_INLINE_CROSS_FUN ShmPtrBase operator--(int) {
    ShmPtrBase tmp(*this);
    operator--();
    return tmp;
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const ShmPtrBase &other) const {
    return (other.alloc_id_ == alloc_id_ && other.off_ == off_);
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const ShmPtrBase &other) const {
    return (other.alloc_id_ != alloc_id_ || other.off_ != off_);
  }

  /** Mark first bit */
  HSHM_INLINE_CROSS_FUN ShmPtrBase Mark() const {
    return ShmPtrBase(alloc_id_, off_.Mark());
  }

  /** Check if first bit is marked */
  HSHM_INLINE_CROSS_FUN bool IsMarked() const { return off_.IsMarked(); }

  /** Unmark first bit */
  HSHM_INLINE_CROSS_FUN ShmPtrBase Unmark() const {
    return ShmPtrBase(alloc_id_, off_.Unmark());
  }

  /** Set to 0 */
  HSHM_INLINE_CROSS_FUN void SetZero() { off_.SetZero(); }
};

/** Non-atomic shared memory pointer (defaults to void*) */
template <typename T = void>
using ShmPtr = ShmPtrBase<T, false>;

/** Atomic shared memory pointer (defaults to void*) */
template <typename T = void>
using AtomicShmPtr = ShmPtrBase<T, true>;


/** Struct containing both private and shared pointer */
template <typename T = char, typename PointerT = ShmPtr<>>
struct FullPtr : public ShmPointer {
  T *ptr_;
  PointerT shm_;

  /** Serialize hipc::FullPtr */
  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar & shm_;
  }

  // /** Serialize an hipc::FullPtr */
  // template <typename Ar>
  // HSHM_INLINE_CROSS_FUN void save(Ar &ar) const {
  //   ar & shm_;
  // }

  // /** Deserialize an hipc::FullPtr */
  // template <typename Ar>
  // HSHM_INLINE_CROSS_FUN void load(Ar &ar) {
  //   ar & shm_;
  //   ptr_ = FullPtr<T>(shm_).ptr_;
  // }

  /** Ostream operator */
  friend std::ostream &operator<<(std::ostream &os, const FullPtr &ptr) {
    os << (void *)ptr.ptr_ << " " << ptr.shm_;
    return os;
  }

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN FullPtr() = default;

  /** Full constructor */
  HSHM_INLINE_CROSS_FUN FullPtr(const T *ptr, const PointerT &shm)
      : ptr_(const_cast<T *>(ptr)), shm_(shm) {}

  /** Private half + alloc constructor */
  template<typename AllocT>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(AllocT *alloc, const T *ptr) {
    if (alloc->ContainsPtr(ptr)) {
      shm_.off_ = (size_t)(reinterpret_cast<const char*>(ptr) - alloc->GetBackendData());
      shm_.alloc_id_ = alloc->GetId();
      ptr_ = const_cast<T*>(ptr);
    } else {
        HSHM_THROW_ERROR(INVALID_FREE);
    }
  }

  /** Shared half + alloc constructor for OffsetPtr */
  template<typename AllocT, bool ATOMIC,
           typename = typename std::enable_if<!std::is_same<T, AllocT>::value>::type>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(AllocT *alloc,
                                         const OffsetPtrBase<T, ATOMIC> &shm) {
    if (alloc->ContainsPtr(shm)) {
      shm_.off_ = shm.load();
      shm_.alloc_id_ = alloc->GetId();
      ptr_ = reinterpret_cast<T*>(alloc->GetBackendData() + shm.load());
    } else {
        HSHM_THROW_ERROR(INVALID_FREE);
    }
  }

  /** Shared half + alloc constructor for ShmPtr */
  template<typename AllocT, bool ATOMIC,
           typename = typename std::enable_if<!std::is_same<T, AllocT>::value>::type>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(AllocT *alloc,
                                         const ShmPtrBase<T, ATOMIC> &shm) {
    if (alloc->ContainsPtr(shm)) {
      shm_.off_ = shm.off_.load();
      shm_.alloc_id_ = shm.alloc_id_;
      ptr_ = reinterpret_cast<T*>(alloc->GetBackendData() + shm.off_.load());
    } else{
        HSHM_THROW_ERROR(INVALID_FREE);
    }
  }

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN FullPtr(const FullPtr &other)
      : ptr_(other.ptr_), shm_(other.shm_) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN FullPtr(FullPtr &&other) noexcept
      : ptr_(other.ptr_), shm_(other.shm_) {
    other.SetNull();
  }

  /** Copy assignment operator */
  HSHM_INLINE_CROSS_FUN FullPtr &operator=(const FullPtr &other) {
    if (this != &other) {
      ptr_ = other.ptr_;
      shm_ = other.shm_;
    }
    return *this;
  }

  /** Move assignment operator */
  HSHM_INLINE_CROSS_FUN FullPtr &operator=(FullPtr &&other) {
    if (this != &other) {
      ptr_ = other.ptr_;
      shm_ = other.shm_;
      other.SetNull();
    }
    return *this;
  }

  /** Overload arrow */
  template<typename U = T>
  HSHM_INLINE_CROSS_FUN typename std::enable_if<!std::is_void<U>::value, U*>::type
  operator->() const { return ptr_; }

  /** Overload dereference */
  template<typename U = T>
  HSHM_INLINE_CROSS_FUN typename std::enable_if<!std::is_void<U>::value, U&>::type
  operator*() const { return *ptr_; }

  /** Equality operator */
  HSHM_INLINE_CROSS_FUN bool operator==(const FullPtr &other) const {
    return ptr_ == other.ptr_ && shm_ == other.shm_;
  }

  /** Inequality operator */
  HSHM_INLINE_CROSS_FUN bool operator!=(const FullPtr &other) const {
    return ptr_ != other.ptr_ || shm_ != other.shm_;
  }

  /** Addition operator */
  HSHM_INLINE_CROSS_FUN FullPtr operator+(size_t size) const {
    return FullPtr(ptr_ + size, shm_ + size);
  }

  /** Subtraction operator */
  HSHM_INLINE_CROSS_FUN FullPtr operator-(size_t size) const {
    return FullPtr(ptr_ - size, shm_ - size);
  }

  /** Addition assignment operator */
  HSHM_INLINE_CROSS_FUN FullPtr &operator+=(size_t size) {
    ptr_ += size;
    shm_ += size;
    return *this;
  }

  /** Subtraction assignment operator */
  HSHM_INLINE_CROSS_FUN FullPtr &operator-=(size_t size) {
    ptr_ -= size;
    shm_ -= size;
    return *this;
  }

  /** Increment operator (pre) */
  HSHM_INLINE_CROSS_FUN FullPtr &operator++() {
    ptr_++;
    shm_++;
    return *this;
  }

  /** Decrement operator (pre) */
  HSHM_INLINE_CROSS_FUN FullPtr &operator--() {
    ptr_--;
    shm_--;
    return *this;
  }

  /** Increment operator (post) */
  HSHM_INLINE_CROSS_FUN FullPtr operator++(int) {
    FullPtr tmp(*this);
    operator++();
    return tmp;
  }

  /** Decrement operator (post) */
  HSHM_INLINE_CROSS_FUN FullPtr operator--(int) {
    FullPtr tmp(*this);
    operator--();
    return tmp;
  }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return ptr_ == nullptr || shm_.IsNull(); }

  /** Get null */
  HSHM_INLINE_CROSS_FUN static FullPtr GetNull() {
    return FullPtr(nullptr, ShmPtr<>::GetNull());
  }

  /** Set to null */
  HSHM_INLINE_CROSS_FUN void SetNull() { ptr_ = nullptr; }

  /** Reintrepret cast to other internal type */
  template <typename U>
  HSHM_INLINE_CROSS_FUN FullPtr<U, PointerT> &Cast() {
    return DeepCast<FullPtr<U, PointerT>>();
  }

  /** Reintrepret cast to other internal type (const) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN const FullPtr<U, PointerT> &Cast() const {
    return DeepCast<FullPtr<U, PointerT>>();
  }

  /** Reintrepret cast to another FullPtr */
  template <typename FullPtrT>
  HSHM_INLINE_CROSS_FUN FullPtrT &DeepCast() {
    return *((FullPtrT *)this);
  }

  /** Reintrepret cast to another FullPtr (const) */
  template <typename FullPtrT>
  HSHM_INLINE_CROSS_FUN const FullPtrT &DeepCast() const {
    return *((FullPtrT *)this);
  }

  /** Mark first bit */
  HSHM_INLINE_CROSS_FUN FullPtr Mark() const {
    return FullPtr(ptr_, shm_.Mark());
  }

  /** Check if first bit is marked */
  HSHM_INLINE_CROSS_FUN bool IsMarked() const { return shm_.IsMarked(); }

  /** Unmark first bit */
  HSHM_INLINE_CROSS_FUN FullPtr Unmark() const {
    return FullPtr(ptr_, shm_.Unmark());
  }

  /** Set to 0 */
  HSHM_INLINE_CROSS_FUN void SetZero() { shm_.SetZero(); }
};

/**
 * The allocator base class.
 * */
template <typename CoreAllocT>
class BaseAllocator : public CoreAllocT {
 public:
  /**====================================
   * Constructors
   * ===================================*/
  /**
   * Create the shared-memory allocator with \a id unique allocator id over
   * the particular slot of a memory backend.
   *
   * The shm_init function is required, but cannot be marked virtual as
   * each allocator has its own arguments to this method.
   * */
  template <typename... Args>
  HSHM_CROSS_FUN void shm_init(Args&&... args) {
    static_cast<CoreAllocT*>(this)->shm_init(std::forward<Args>(args)...);
  }

  /**
   * Deserialize allocator from a buffer.
   * */
  HSHM_CROSS_FUN
  void shm_attach(const MemoryBackend &backend) {
    static_cast<CoreAllocT*>(this)->shm_attach(backend);
  }

  /**====================================
   * Core Allocator API
   * ===================================*/
 public:
  /**
   * Allocate a region of memory of \a size size
   * */
  HSHM_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size) {
    return CoreAllocT::AllocateOffset(size);
  }

  /**
   * Allocate a region of memory of \a size size with \a alignment alignment
   * */
  HSHM_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size, size_t alignment) {
    return CoreAllocT::AllocateOffset(size, alignment);
  }

  /**
   * Reallocate \a pointer to \a new_size new size.
   * Assumes that p is not kNulFullPtr.
   *
   * @return true if p was modified.
   * */
  HSHM_CROSS_FUN
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    return CoreAllocT::ReallocateOffsetNoNullCheck(p, new_size);
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(OffsetPtr<> p) {
    CoreAllocT::FreeOffsetNoNullCheck(p);
  }

  /**
   * Create a thread-local storage segment. This storage
   * is unique even across processes.
   * */
  HSHM_CROSS_FUN
  void CreateTls() { CoreAllocT::CreateTls(); }

  /**
   * Free a thread-local storage segment.
   * */
  HSHM_CROSS_FUN
  void FreeTls() { CoreAllocT::FreeTls(); }

  /** Get the allocator identifier */
  HSHM_INLINE_CROSS_FUN
  AllocatorId GetId() { return CoreAllocT::GetId(); }

  /** Get the allocator identifier (const) */
  HSHM_INLINE_CROSS_FUN
  AllocatorId GetId() const { return CoreAllocT::GetId(); }

  /**
   * Check if pointer is contained in this allocator
   */
  template <typename T = void>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const T *ptr) const {
    return CoreAllocT::template ContainsPtr<T>(ptr);
  }

  /**
   * Check if OffsetPtr is contained in this allocator
   */
  template<typename T, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const OffsetPtrBase<T, ATOMIC> &ptr) const {
    return CoreAllocT::template ContainsPtr<T>(ptr);
  }

  /**
   * Check if ShmPtr is contained in this allocator
   */
  template<typename T, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const ShmPtrBase<T, ATOMIC> &ptr) const {
    return CoreAllocT::template ContainsPtr<T>(ptr);
  }

  /**
   * Get the amount of memory that was allocated, but not yet freed.
   * Useful for memory leak checks.
   * */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() {
    return CoreAllocT::GetCurrentlyAllocatedSize();
  }

  /**====================================
   * SHM ShmPtr Allocator
   * ===================================*/
 public:
  /**
   * Allocate a region of memory to a specific pointer type
   * */
  template <typename T = void, typename PointerT = ShmPtr<>>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Allocate(size_t size) {
    FullPtr<T, PointerT> result;
    CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
    result.shm_ = PointerT(GetId(), AllocateOffset(size).load());
    result.ptr_ = reinterpret_cast<T*>(reinterpret_cast<char*>(core_this) + result.shm_.off_.load());
    return result;
  }

  /**
   * Allocate a region of \a size size and \a alignment
   * alignment. Will fall back to regular Allocate if
   * alignment is 0.
   * */
  template <typename T = void, typename PointerT = ShmPtr<>>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Allocate(size_t size,
                                          size_t alignment) {
    if (alignment == 0) {
      return Allocate<T, PointerT>(size);
    } else {
      FullPtr<T, PointerT> result;
      CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
      result.shm_ = PointerT(GetId(),
                      CoreAllocT::AllocateOffset(size, alignment).load());
      result.ptr_ = reinterpret_cast<T*>(reinterpret_cast<char*>(core_this) + result.shm_.off_.load());
      return result;
    }
  }

  /**
   * Reallocate \a pointer to \a new_size new size
   * If p is kNulFullPtr, will internally call Allocate.
   *
   * @return the reallocated FullPtr.
   * */
  template <typename T = void, typename PointerT = ShmPtr<>>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Reallocate(const FullPtr<T, PointerT> &p,
                                        size_t new_size) {
    if (p.IsNull()) {
      return Allocate<T, PointerT>(new_size);
    }
    auto new_off =
        ReallocateOffsetNoNullCheck(p.shm_.ToOffsetPtr(), new_size);
    FullPtr<T, PointerT> result;
    CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
    result.shm_ = PointerT(GetId(), new_off.load());
    result.ptr_ = reinterpret_cast<T*>(reinterpret_cast<char*>(core_this) + result.shm_.off_.load());
    return result;
  }

  /**
   * Free the memory pointed to by \a p Pointer
   * */
  template <typename T = void, typename PointerT = ShmPtr<>>
  HSHM_INLINE_CROSS_FUN void Free(const FullPtr<T, PointerT> &p) {
    if (p.IsNull()) {
      HSHM_THROW_ERROR(INVALID_FREE);
    }
    FreeOffsetNoNullCheck(OffsetPtr<>(p.shm_.off_.load()));
  }





  /**====================================
   * Private Object Allocators
   * ===================================*/

  /**
   * Allocate an array of objects (but don't construct).
   *
   * @return A FullPtr to the allocated memory
   * */
  template <typename T, typename PointerT = ShmPtr<>>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> AllocateObjs(size_t count) {
    return Allocate<T, PointerT>(count * sizeof(T));
  }

  /** Allocate + construct an array of objects */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN FullPtr<T> NewObjs(size_t count,
                                   Args &&...args) {
    auto alloc_result = AllocateObjs<T, ShmPtr>(count);
    ConstructObjs<T>(alloc_result.ptr_, 0, count, std::forward<Args>(args)...);
    return alloc_result;
  }

  /** Allocate + construct a single object */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN FullPtr<T> NewObj(Args &&...args) {
    return NewObjs<T>(1, std::forward<Args>(args)...);
  }

  /**
   * Reallocate a pointer of objects to a new size.
   *
   * @param p FullPtr to reallocate (input & output)
   * @param new_count the new number of objects
   *
   * @return A FullPtr to the reallocated objects
   * */
  template <typename T, typename PointerT = ShmPtr<>>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> ReallocateObjs(FullPtr<T, PointerT> &p,
                                                             size_t new_count) {
    FullPtr<void, PointerT> old_full_ptr(reinterpret_cast<void*>(p.ptr_), p.shm_);
    auto new_full_ptr = Reallocate<void, PointerT>(old_full_ptr, new_count * sizeof(T));
    p.shm_ = new_full_ptr.shm_;
    p.ptr_ = reinterpret_cast<T*>(new_full_ptr.ptr_);
    return p;
  }

  /**
   * Free + destruct objects
   * */
  template <typename T, typename PointerT = ShmPtr<>>
  HSHM_INLINE_CROSS_FUN void DelObjs(FullPtr<T, PointerT> &p,
                                     size_t count) {
    DestructObjs<T>(p.ptr_, count);
    FullPtr<void, PointerT> void_ptr(reinterpret_cast<void*>(p.ptr_), p.shm_);
    Free<void, PointerT>(void_ptr);
  }

  /**
   * Free + destruct an object
   * */
  template <typename T, typename PointerT = ShmPtr<>>
  HSHM_INLINE_CROSS_FUN void DelObj(FullPtr<T, PointerT> &p) {
    DelObjs<T, PointerT>(p, 1);
  }


  /**====================================
   * Object Constructors
   * ===================================*/

  /**
   * Construct each object in an array of objects.
   *
   * @param ptr the array of objects (potentially archived)
   * @param old_count the original size of the ptr
   * @param new_count the new size of the ptr
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObjs(T *ptr, size_t old_count,
                                                  size_t new_count,
                                                  Args &&...args) {
    CoreAllocT::template ConstructObjs<T>(ptr, old_count, new_count,
                                          std::forward<Args>(args)...);
  }

  /**
   * Construct an object.
   *
   * @param ptr the object to construct (potentially archived)
   * @param args parameters to construct object of type T
   * @return None
   * */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN static void ConstructObj(T &obj, Args &&...args) {
    CoreAllocT::template ConstructObj<T>(obj, std::forward<Args>(args)...);
  }

  /**
   * Destruct an array of objects
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObjs(T *ptr, size_t count) {
    CoreAllocT::template DestructObjs<T>(ptr, count);
  }

  /**
   * Destruct an object
   *
   * @param ptr the object to destruct (potentially archived)
   * @param count the length of the object array
   * @return None
   * */
  template <typename T>
  HSHM_INLINE_CROSS_FUN static void DestructObj(T &obj) {
    CoreAllocT::template DestructObj<T>(obj);
  }

  /**====================================
   * Helpers
   * ===================================*/

  /**
   * Get the custom header of the shared-memory allocator
   *
   * @return Custom header pointer
   * */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetCustomHeader() {
    return CoreAllocT::template GetCustomHeader<HEADER_T>();
  }

  /**
   * Determine whether or not this allocator contains a process-specific
   * pointer
   *
   * @param ptr process-specific pointer
   * @return True or false
   * */
  template <typename T = void>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const T *ptr) {
    return CoreAllocT::template ContainsPtr<T>(ptr);
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() { CoreAllocT::Print(); }

  /**====================================
   * Sub-Allocator Management
   * ===================================*/

  /**
   * Create a sub-allocator within this allocator
   *
   * @tparam SubAllocCoreT The sub-allocator core type to create
   * @param sub_id The unique sub-allocator ID (deprecated, not used)
   * @param size Size of the region for the sub-allocator
   * @param args Additional arguments for the sub-allocator initialization
   * @return Pointer to the created sub-allocator
   */
  template<typename SubAllocCoreT, typename ...Args>
  HSHM_CROSS_FUN BaseAllocator<SubAllocCoreT> *CreateSubAllocator(u64 sub_id,
                                                                   size_t size,
                                                                   Args&& ...args) {
    (void)sub_id;  // Unused parameter

    // Calculate total size needed:
    // - kBackendPrivate for private region
    // - sizeof(SubAllocCoreT) for the allocator object itself
    // - size for the allocator's managed region
    using SubAllocT = BaseAllocator<SubAllocCoreT>;
    size_t alloc_size = sizeof(SubAllocCoreT);
    size_t total_size = hipc::kBackendPrivate + alloc_size + size;

    // Allocate region from this allocator
    FullPtr<char> region = Allocate<char>(total_size);
    if (region.IsNull()) {
      return nullptr;
    }

    // Get the backend ID from this allocator
    CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
    MemoryBackendId backend_id = core_this->GetId();

    // Create ArrayBackend for the sub-allocator
    // The ArrayBackend points to: [kBackendPrivate][SubAllocCoreT object][managed region]
    hipc::ArrayBackend backend;
    char *shared_ptr = region.ptr_ + hipc::kBackendPrivate;
    u64 shared_offset = region.shm_.off_.load() + hipc::kBackendPrivate;
    backend.shm_init(backend_id, alloc_size + size, shared_ptr, shared_offset);

    // Use MakeAlloc to construct and initialize the sub-allocator
    // MakeAlloc automatically passes backend as first parameter to shm_init
    // The allocator's shm_init will use GetAllocatorDataOff() to handle offset calculations
    SubAllocCoreT *sub_alloc_core = backend.MakeAlloc<SubAllocCoreT>(std::forward<Args>(args)...);

    // Disown the local backend so its destructor doesn't free the region
    backend.Disown();

    // Return as wrapped type
    return reinterpret_cast<SubAllocT*>(sub_alloc_core);
  }

  /**
   * Free a sub-allocator
   *
   * @tparam AllocT The sub-allocator type
   * @param sub_alloc ShmPtr to the sub-allocator to free
   */
  template<typename AllocT>
  HSHM_CROSS_FUN void FreeSubAllocator(AllocT *sub_alloc) {
    if (sub_alloc == nullptr) {
      return;
    }

    // Get the offset of the sub-allocator's data region
    CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
    OffsetPtr<> offset_ptr(reinterpret_cast<char*>(sub_alloc) - core_this->GetBackendData());

    // Free the data region
    FreeOffsetNoNullCheck(offset_ptr);

    // Destroy and free the allocator object
    sub_alloc->~AllocT();
    free(sub_alloc);
  }
};


/** Thread-local storage manager */
template <typename AllocT>
class TlsAllocatorInfo : public thread::ThreadLocalData {
 public:
  AllocT *alloc_;

 public:
  HSHM_CROSS_FUN
  TlsAllocatorInfo() : alloc_(nullptr) {}

  HSHM_CROSS_FUN
  void destroy() { alloc_->FreeTls(); }
};

class MemoryAlignment {
 public:
  /**
   * Round up to the nearest multiple of the alignment
   * @param alignment the alignment value (e.g., 4096)
   * @param size the size to make a multiple of alignment (e.g., 4097)
   * @return the new size  (e.g., 8192)
   * */
  static size_t AlignTo(size_t alignment, size_t size) {
    auto page_size = HSHM_SYSTEM_INFO->page_size_;
    size_t new_size = size;
    size_t page_off = size % alignment;
    if (page_off) {
      new_size = size + page_size - page_off;
    }
    return new_size;
  }

  /**
   * Round up to the nearest multiple of page size
   * @param size the size to align to the PAGE_SIZE
   * */
  static size_t AlignToPageSize(size_t size) {
    auto page_size = HSHM_SYSTEM_INFO->page_size_;
    size_t new_size = AlignTo(page_size, size);
    return new_size;
  }
};

}  // namespace hshm::ipc

namespace std {

/** Memory Backend ID hash (AllocatorId is now an alias) */
template <>
struct hash<hshm::ipc::MemoryBackendId> {
  std::size_t operator()(const hshm::ipc::MemoryBackendId &key) const {
    return hshm::hash<uint64_t>{}((uint64_t)key.major_ << 32 | key.minor_);
  }
};

}  // namespace std

namespace hshm {

/** Memory Backend ID hash (AllocatorId is now an alias) */
template <>
struct hash<hshm::ipc::MemoryBackendId> {
  HSHM_INLINE_CROSS_FUN std::size_t operator()(
      const hshm::ipc::MemoryBackendId &key) const {
    return hshm::hash<uint64_t>{}((uint64_t)key.major_ << 32 | key.minor_);
  }
};

}  // namespace hshm

#define IS_SHM_POINTER(T) std::is_base_of_v<hipc::ShmPointer, T>

#endif  // HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_
