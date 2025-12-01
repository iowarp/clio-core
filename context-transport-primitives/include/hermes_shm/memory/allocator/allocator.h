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
 * The identifier for an allocator
 * */
struct AllocatorId {
  MemoryBackendId backend_id_;  // The backend this allocator is attached to
  u32 sub_id_;                  // The unique ID of allocator on this backend (0 for main allocator)

  HSHM_INLINE_CROSS_FUN AllocatorId() : backend_id_(), sub_id_(0) {}

  /**
   * Constructor which sets backend_id and sub_id
   * */
  HSHM_INLINE_CROSS_FUN explicit AllocatorId(MemoryBackendId backend_id, u32 sub_id = 0)
    : backend_id_(backend_id), sub_id_(sub_id) {}

  /**
   * Legacy constructor which sets major & minor (treats as backend_id with sub_id=0)
   * */
  HSHM_INLINE_CROSS_FUN explicit AllocatorId(i32 major, i32 minor)
    : backend_id_(major, minor), sub_id_(0) {}

  /**
   * Set this allocator to null
   * */
  HSHM_INLINE_CROSS_FUN void SetNull() { (*this) = GetNull(); }

  /**
   * Check if this is the null allocator
   * */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return (*this) == GetNull(); }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const AllocatorId &other) const {
    return backend_id_ == other.backend_id_ && sub_id_ == other.sub_id_;
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const AllocatorId &other) const {
    return backend_id_ != other.backend_id_ || sub_id_ != other.sub_id_;
  }

  /** Get the null allocator */
  HSHM_INLINE_CROSS_FUN static AllocatorId GetNull() {
    return AllocatorId(MemoryBackendId(UINT32_MAX, UINT32_MAX), UINT32_MAX);
  }

  /** To index */
  HSHM_INLINE_CROSS_FUN uint32_t ToIndex() const {
    return backend_id_.major_ * 2 + backend_id_.minor_;
  }

  /** Serialize an hipc::allocator_id */
  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar & backend_id_.major_;
    ar & backend_id_.minor_;
    ar & sub_id_;
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() const {
    printf("(%s) Allocator ID: (%u,%u).%u\n", kCurrentDevice,
           backend_id_.major_, backend_id_.minor_, sub_id_);
  }

  friend std::ostream &operator<<(std::ostream &os, const AllocatorId &id) {
    os << "(" << id.backend_id_.major_ << "," << id.backend_id_.minor_ << ")." << id.sub_id_;
    return os;
  }
};

class Allocator;

/** ShmPtr type base */
class ShmPointer {};

/** Forward declarations for pointer types */
template<bool ATOMIC = false>
struct OffsetPtrBase;

template<bool ATOMIC = false>
struct ShmPtrBase;

/**
 * The basic shared-memory allocator header.
 * Allocators inherit from this.
 * */
struct AllocatorHeader {
  AllocatorId alloc_id_;
  size_t custom_header_size_;
  hipc::atomic<hshm::size_t> total_alloc_;

  HSHM_CROSS_FUN
  AllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId allocator_id, size_t custom_header_size) {
    alloc_id_ = allocator_id;
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

/** Memory context */
class MemContext {
 public:
  ThreadId tid_ = ThreadId::GetNull();

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  MemContext() = default;

  /** Constructor */
  HSHM_INLINE_CROSS_FUN
  MemContext(const ThreadId &tid) : tid_(tid) {}
};

/** The allocator information struct */
class Allocator {
 public:
  AllocatorId id_;
  MemoryBackend backend_;
  int accel_id_;
  char *custom_header_;

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Allocator() : custom_header_(nullptr) {}

  /** Get the allocator identifier */
  HSHM_INLINE_CROSS_FUN
  AllocatorId &GetId() { return id_; }

  /** Get the allocator identifier (const) */
  HSHM_INLINE_CROSS_FUN
  const AllocatorId &GetId() const { return id_; }

  /**
   * Construct custom header
   */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *ConstructHeader(void *buffer) {
    new ((HEADER_T *)buffer) HEADER_T();
    return reinterpret_cast<HEADER_T *>(buffer);
  }

  /**
   * Get the custom header of the shared-memory allocator
   *
   * @return Custom header pointer
   * */
  template <typename HEADER_T>
  HSHM_INLINE_CROSS_FUN HEADER_T *GetCustomHeader() {
    return reinterpret_cast<HEADER_T *>(custom_header_);
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
    return reinterpret_cast<size_t>(backend_.data_) <= reinterpret_cast<size_t>(ptr) &&
           reinterpret_cast<size_t>(ptr) <
               reinterpret_cast<size_t>(backend_.data_) + backend_.data_size_;
  }

  template<bool ATOMIC>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const OffsetPtrBase<ATOMIC> &ptr) const {
    return ptr.off_.load() < backend_.data_size_;
  }

  template<bool ATOMIC>
  HSHM_INLINE_CROSS_FUN bool ContainsPtr(const ShmPtrBase<ATOMIC> &ptr) const {
    return ptr.off_.load() < backend_.data_size_;
  }

  /** Print */
  HSHM_CROSS_FUN
  void Print() {
    printf("(%s) Allocator: id: (%u,%u).%u, custom_header: %p\n",
           kCurrentDevice, GetId().backend_id_.major_,
           GetId().backend_id_.minor_, GetId().sub_id_, custom_header_);
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
 * */
template <bool ATOMIC>
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

  /** ShmPtr constructor */
  HSHM_INLINE_CROSS_FUN explicit OffsetPtrBase(AllocatorId alloc_id,
                                                   size_t off)
      : off_(off) {
    (void)alloc_id;
  }

  /** ShmPtr constructor (alloc + atomic offset)*/
  HSHM_INLINE_CROSS_FUN explicit OffsetPtrBase(AllocatorId id,
                                                   OffsetPtrBase<true> off)
      : off_(off.load()) {
    (void)id;
  }

  /** ShmPtr constructor (alloc + non-offeset) */
  HSHM_INLINE_CROSS_FUN explicit OffsetPtrBase(AllocatorId id,
                                                   OffsetPtrBase<false> off)
      : off_(off.load()) {
    (void)id;
  }

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase(const OffsetPtrBase &other)
      : off_(other.off_.load()) {}

  /** Other copy constructor */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase(
      const OffsetPtrBase<!ATOMIC> &other)
      : off_(other.off_.load()) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase(OffsetPtrBase &&other) noexcept
      : off_(other.off_.load()) {
    other.SetNull();
  }

  /** Get the offset pointer */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase<false> ToOffsetPtr() const {
    return OffsetPtrBase<false>(off_.load());
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

/** Non-atomic offset */
typedef OffsetPtrBase<false> OffsetPtr;

/** Atomic offset */
typedef OffsetPtrBase<true> AtomicOffsetPtr;

/** Typed offset pointer */
template <typename T = void>
using TypedOffsetPtr = OffsetPtr;

/** Typed atomic pointer */
template <typename T = void>
using TypedAtomicOffsetPtr = AtomicOffsetPtr;

/**
 * A process-independent pointer, which stores both the allocator's
 * information and the offset within the allocator's region
 * */
template <bool ATOMIC>
struct ShmPtrBase : public ShmPointer {
  AllocatorId alloc_id_;           /// Allocator the pointer comes from
  OffsetPtrBase<ATOMIC> off_;  /// Offset within the allocator's slot

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
  HSHM_INLINE_CROSS_FUN explicit ShmPtrBase(AllocatorId id, OffsetPtr off)
      : alloc_id_(id), off_(off) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN ShmPtrBase(const ShmPtrBase &other)
      : alloc_id_(other.alloc_id_), off_(other.off_) {}

  /** Other copy constructor */
  HSHM_INLINE_CROSS_FUN ShmPtrBase(const ShmPtrBase<!ATOMIC> &other)
      : alloc_id_(other.alloc_id_), off_(other.off_.load()) {}

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN ShmPtrBase(ShmPtrBase &&other) noexcept
      : alloc_id_(other.alloc_id_), off_(other.off_) {
    other.SetNull();
  }

  /** Get the offset pointer */
  HSHM_INLINE_CROSS_FUN OffsetPtrBase<false> ToOffsetPtr() const {
    return OffsetPtrBase<false>(off_.load());
  }

  /** Set to null */
  HSHM_INLINE_CROSS_FUN void SetNull() { alloc_id_.SetNull(); }

  /** Check if null */
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return alloc_id_.IsNull(); }

  /** Get the null pointer */
  HSHM_INLINE_CROSS_FUN static ShmPtrBase GetNull() {
    return ShmPtrBase{AllocatorId::GetNull(), OffsetPtr::GetNull()};
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

/** Non-atomic pointer */
typedef ShmPtrBase<false> ShmPtr;

/** Atomic pointer */
typedef ShmPtrBase<true> AtomicShmPtr;

/** Typed pointer */
template <typename T = void>
using TypedShmPtr = ShmPtr;

/** Typed atomic pointer */
template <typename T = void>
using TypedAtomicShmPtr = AtomicShmPtr;

/**
 * Allocator with thread-local storage identifier
 * */
template <typename AllocT>
struct CtxAllocator {
  MemContext ctx_;
  AllocT *alloc_;

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator() = default;

  /** Allocator-only constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(AllocT *alloc) : alloc_(alloc), ctx_() {}

  /** Allocator and thread identifier constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(AllocT *alloc, const ThreadId &tid) : alloc_(alloc), ctx_(tid) {}

  /** Allocator and thread identifier constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(const ThreadId &tid, AllocT *alloc) : alloc_(alloc), ctx_(tid) {}

  /** Allocator and ctx constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(const MemContext &ctx, AllocT *alloc)
      : alloc_(alloc), ctx_(ctx) {}

  /** ctx and Allocator constructor */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator(AllocT *alloc, const MemContext &ctx)
      : alloc_(alloc), ctx_(ctx) {}

  /** Arrow operator */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator->() { return alloc_; }

  /** Arrow operator (const) */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator->() const { return alloc_; }

  /** Star operator */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator*() { return alloc_; }

  /** Star operator (const) */
  HSHM_INLINE_CROSS_FUN
  AllocT *operator*() const { return alloc_; }

  /** Equality operator */
  HSHM_INLINE_CROSS_FUN
  bool operator==(const CtxAllocator &rhs) const {
    return alloc_ == rhs.alloc_;
  }

  /** Inequality operator */
  HSHM_INLINE_CROSS_FUN
  bool operator!=(const CtxAllocator &rhs) const {
    return alloc_ != rhs.alloc_;
  }
};

/** Struct containing both private and shared pointer */
template <typename T = char, typename PointerT = ShmPtr>
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
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const hipc::CtxAllocator<AllocT> &ctx_alloc, const T *ptr) {
    if (ctx_alloc->ContainsPtr(ptr)) {
      shm_.off_ = (size_t)(reinterpret_cast<const char*>(ptr) - ctx_alloc->backend_.data_);
      shm_.alloc_id_ = ctx_alloc->id_;
      ptr_ = const_cast<T*>(ptr);
    } else {
        HSHM_THROW_ERROR(INVALID_FREE);
    }
  }

  /** Shared half + alloc constructor for OffsetPtr */
  template<typename AllocT, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const hipc::CtxAllocator<AllocT> &ctx_alloc,
                                         const OffsetPtrBase<ATOMIC> &shm) {
    if (ctx_alloc->ContainsPtr(shm)) {
      shm_.off_ = shm.load();
      shm_.alloc_id_ = ctx_alloc->id_;
      ptr_ = reinterpret_cast<T*>(ctx_alloc->backend_.data_ + shm.load());
    } else {
        HSHM_THROW_ERROR(INVALID_FREE);
    }
 }

 /** Shared half + alloc constructor for ShmPtr */
  template<typename AllocT, bool ATOMIC>
  HSHM_INLINE_CROSS_FUN explicit FullPtr(const hipc::CtxAllocator<AllocT> &ctx_alloc,
                                         const ShmPtrBase<ATOMIC> &shm) {
    if (ctx_alloc->ContainsPtr(shm)) {
      shm_.off_ = shm.off_.load();
      shm_.alloc_id_ = shm.alloc_id_;
      ptr_ = reinterpret_cast<T*>(ctx_alloc->backend_.data_ + shm.off_.load());
    } else{
        HSHM_THROW_ERROR(INVALID_FREE);
    }
 }

  /** Direct allocator + OffsetPtr constructor */
  template<typename AllocT, bool ATOMIC,
           typename = typename std::enable_if<!std::is_same<T, AllocT>::value>::type>
  HSHM_INLINE_CROSS_FUN FullPtr(AllocT *alloc, const OffsetPtrBase<ATOMIC> &shm) {
    shm_.off_ = shm.load();
    shm_.alloc_id_ = alloc->GetId();
    ptr_ = reinterpret_cast<T*>(alloc->backend_.data_ + shm.load());
  }

  /** Direct allocator + ShmPtr constructor */
  template<typename AllocT, bool ATOMIC,
           typename = typename std::enable_if<!std::is_same<T, AllocT>::value>::type>
  HSHM_INLINE_CROSS_FUN FullPtr(AllocT *alloc, const ShmPtrBase<ATOMIC> &shm) {
    shm_.off_ = shm.off_.load();
    shm_.alloc_id_ = shm.alloc_id_;
    ptr_ = reinterpret_cast<T*>(alloc->backend_.data_ + shm.off_.load());
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
  HSHM_INLINE_CROSS_FUN bool IsNull() const { return ptr_ == nullptr; }

  /** Get null */
  HSHM_INLINE_CROSS_FUN static FullPtr GetNull() {
    return FullPtr(nullptr, ShmPtr::GetNull());
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
   * each allocator has its own arguments to this method. Though each
   * allocator must have "id" as its first argument.
   * */
  template <typename... Args>
  HSHM_CROSS_FUN void shm_init(AllocatorId id, Args... args) {
    CoreAllocT::shm_init(id, std::forward<Args>(args)...);
  }

  /**
   * Deserialize allocator from a buffer.
   * */
  HSHM_CROSS_FUN
  void shm_attach(const MemoryBackend &backend) {
    CoreAllocT::shm_attach(backend);
  }

  /**====================================
   * Core Allocator API
   * ===================================*/
 public:
  /**
   * Allocate a region of memory of \a size size
   * */
  HSHM_CROSS_FUN
  OffsetPtr AllocateOffset(const MemContext &ctx, size_t size) {
    return CoreAllocT::AllocateOffset(ctx, size);
  }

  /**
   * Allocate a region of memory of \a size size with \a alignment alignment
   * */
  HSHM_CROSS_FUN
  OffsetPtr AllocateOffset(const MemContext &ctx, size_t size, size_t alignment) {
    return CoreAllocT::AllocateOffset(ctx, size, alignment);
  }

  /**
   * Reallocate \a pointer to \a new_size new size.
   * Assumes that p is not kNulFullPtr.
   *
   * @return true if p was modified.
   * */
  HSHM_CROSS_FUN
  OffsetPtr ReallocateOffsetNoNullCheck(const MemContext &ctx,
                                            OffsetPtr p, size_t new_size) {
    return CoreAllocT::ReallocateOffsetNoNullCheck(ctx, p, new_size);
  }

  /**
   * Free the memory pointed to by \a ptr Pointer
   * */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const MemContext &ctx, OffsetPtr p) {
    CoreAllocT::FreeOffsetNoNullCheck(ctx, p);
  }

  /**
   * Create a thread-local storage segment. This storage
   * is unique even across processes.
   * */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) { CoreAllocT::CreateTls(ctx); }

  /**
   * Free a thread-local storage segment.
   * */
  HSHM_CROSS_FUN
  void FreeTls(const MemContext &ctx) { CoreAllocT::FreeTls(ctx); }

  /** Get the allocator identifier */
  HSHM_INLINE_CROSS_FUN
  AllocatorId &GetId() { return CoreAllocT::GetId(); }

  /** Get the allocator identifier (const) */
  HSHM_INLINE_CROSS_FUN
  const AllocatorId &GetId() const { return CoreAllocT::GetId(); }

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
  template <typename T = void, typename PointerT = ShmPtr>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Allocate(const MemContext &ctx, size_t size) {
    FullPtr<T, PointerT> result;
    CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
    result.shm_ = PointerT(GetId(), AllocateOffset(ctx, size).load());
    result.ptr_ = reinterpret_cast<T*>(core_this->backend_.data_ + result.shm_.off_.load());
    return result;
  }

  /**
   * Allocate a region of memory to a specific pointer type
   * */
  template <typename T = void, typename PointerT = ShmPtr>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> AlignedAllocate(const MemContext &ctx,
                                                 size_t size,
                                                 size_t alignment) {
    FullPtr<T, PointerT> result;
    CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
    result.shm_ = PointerT(GetId(),
                    CoreAllocT::AllocateOffset(ctx, size, alignment).load());
    result.ptr_ = reinterpret_cast<T*>(core_this->backend_.data_ + result.shm_.off_.load());
    return result;
  }

  /**
   * Allocate a region of \a size size and \a alignment
   * alignment. Will fall back to regular Allocate if
   * alignmnet is 0.
   * */
  template <typename T = void, typename PointerT = ShmPtr>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Allocate(const MemContext &ctx, size_t size,
                                          size_t alignment) {
    if (alignment == 0) {
      return Allocate<T, PointerT>(ctx, size);
    } else {
      return AlignedAllocate<T, PointerT>(ctx, size, alignment);
    }
  }

  /**
   * Reallocate \a pointer to \a new_size new size
   * If p is kNulFullPtr, will internally call Allocate.
   *
   * @return the reallocated FullPtr.
   * */
  template <typename T = void, typename PointerT = ShmPtr>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> Reallocate(const MemContext &ctx, const FullPtr<T, PointerT> &p,
                                        size_t new_size) {
    if (p.IsNull()) {
      return Allocate<T, PointerT>(ctx, new_size);
    }
    auto new_off =
        ReallocateOffsetNoNullCheck(ctx, p.shm_.ToOffsetPtr(), new_size);
    FullPtr<T, PointerT> result;
    CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
    result.shm_ = PointerT(GetId(), new_off.load());
    result.ptr_ = reinterpret_cast<T*>(core_this->backend_.data_ + result.shm_.off_.load());
    return result;
  }

  /**
   * Free the memory pointed to by \a p Pointer
   * */
  template <typename T = void, typename PointerT = ShmPtr>
  HSHM_INLINE_CROSS_FUN void Free(const MemContext &ctx, const FullPtr<T, PointerT> &p) {
    if (p.IsNull()) {
      HSHM_THROW_ERROR(INVALID_FREE);
    }
    FreeOffsetNoNullCheck(ctx, OffsetPtr(p.shm_.off_.load()));
  }





  /**====================================
   * Private Object Allocators
   * ===================================*/

  /**
   * Allocate an array of objects (but don't construct).
   *
   * @return A FullPtr to the allocated memory
   * */
  template <typename T, typename PointerT = ShmPtr>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> AllocateObjs(const MemContext &ctx, size_t count) {
    return Allocate<T, PointerT>(ctx, count * sizeof(T));
  }

  /** Allocate + construct an array of objects */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN FullPtr<T> NewObjs(const MemContext &ctx, size_t count,
                                   Args &&...args) {
    auto alloc_result = AllocateObjs<T, ShmPtr>(ctx, count);
    ConstructObjs<T>(alloc_result.ptr_, 0, count, std::forward<Args>(args)...);
    return alloc_result;
  }

  /** Allocate + construct a single object */
  template <typename T, typename... Args>
  HSHM_INLINE_CROSS_FUN FullPtr<T> NewObj(const MemContext &ctx, Args &&...args) {
    return NewObjs<T>(ctx, 1, std::forward<Args>(args)...);
  }

  /**
   * Reallocate a pointer of objects to a new size.
   *
   * @param p FullPtr to reallocate (input & output)
   * @param new_count the new number of objects
   *
   * @return A FullPtr to the reallocated objects
   * */
  template <typename T, typename PointerT = ShmPtr>
  HSHM_INLINE_CROSS_FUN FullPtr<T, PointerT> ReallocateObjs(const MemContext &ctx, 
                                                             FullPtr<T, PointerT> &p,
                                                             size_t new_count) {
    FullPtr<void, PointerT> old_full_ptr(reinterpret_cast<void*>(p.ptr_), p.shm_);
    auto new_full_ptr = Reallocate<void, PointerT>(ctx, old_full_ptr, new_count * sizeof(T));
    p.shm_ = new_full_ptr.shm_;
    p.ptr_ = reinterpret_cast<T*>(new_full_ptr.ptr_);
    return p;
  }

  /**
   * Free + destruct objects
   * */
  template <typename T, typename PointerT = ShmPtr>
  HSHM_INLINE_CROSS_FUN void DelObjs(const MemContext &ctx, 
                                     FullPtr<T, PointerT> &p,
                                     size_t count) {
    DestructObjs<T>(p.ptr_, count);
    FullPtr<void, PointerT> void_ptr(reinterpret_cast<void*>(p.ptr_), p.shm_);
    Free<void, PointerT>(ctx, void_ptr);
  }

  /**
   * Free + destruct an object
   * */
  template <typename T, typename PointerT = ShmPtr>
  HSHM_INLINE_CROSS_FUN void DelObj(const MemContext &ctx, 
                                    FullPtr<T, PointerT> &p) {
    DelObjs<T, PointerT>(ctx, p, 1);
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
   * @tparam AllocT The sub-allocator type to create
   * @param ctx Memory context
   * @param sub_id The unique sub-allocator ID
   * @param size Size of the region for the sub-allocator
   * @param args Additional arguments for the sub-allocator initialization
   * @return ShmPtr to the created sub-allocator
   */
  template<typename SubAllocCoreT, typename ...Args>
  HSHM_CROSS_FUN BaseAllocator<SubAllocCoreT> *CreateSubAllocator(const MemContext &ctx,
                                                                   u64 sub_id,
                                                                   size_t size,
                                                                   Args&& ...args) {
    // Allocate region for the sub-allocator
    FullPtr<char> region = Allocate<char>(ctx, size);

    // Get the backend ID from this allocator through the core allocator type
    CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
    MemoryBackendId backend_id = core_this->backend_.GetId();

    // Create ArrayBackend for the sub-allocator
    hipc::ArrayBackend backend;
    backend.shm_init(backend_id, size, region.ptr_, region.shm_.off_.load());

    // Create allocator ID for sub-allocator
    AllocatorId sub_alloc_id(backend_id, sub_id);

    // Allocate and initialize the sub-allocator (wrapped type)
    using SubAllocT = BaseAllocator<SubAllocCoreT>;
    SubAllocT *sub_alloc = reinterpret_cast<SubAllocT*>(malloc(sizeof(SubAllocT)));
    new (sub_alloc) SubAllocT();
    sub_alloc->shm_init(sub_alloc_id, std::forward<Args>(args)..., backend);

    // Disown the local backend so its destructor doesn't free the header
    // The sub-allocator now owns the backend
    backend.Disown();

    return sub_alloc;
  }

  /**
   * Free a sub-allocator
   *
   * @tparam AllocT The sub-allocator type
   * @param ctx Memory context
   * @param sub_alloc ShmPtr to the sub-allocator to free
   */
  template<typename AllocT>
  HSHM_CROSS_FUN void FreeSubAllocator(const MemContext &ctx,
                                       AllocT *sub_alloc) {
    if (sub_alloc == nullptr) {
      return;
    }

    // Get the offset of the sub-allocator's data region
    CoreAllocT *core_this = static_cast<CoreAllocT*>(this);
    OffsetPtr offset_ptr(sub_alloc->backend_.data_ - core_this->backend_.data_);

    // Free the data region
    FreeOffsetNoNullCheck(ctx, offset_ptr);

    // Destroy and free the allocator object
    sub_alloc->~AllocT();
    free(sub_alloc);
  }
};

/**
 * Scoped Allocator (thread-local)
 * */
template <typename AllocT>
class ScopedTlsAllocator {
 public:
  CtxAllocator<AllocT> alloc_;

 public:
  HSHM_INLINE_CROSS_FUN
  ScopedTlsAllocator(const MemContext &ctx, AllocT *alloc)
      : alloc_(ctx, alloc) {
    alloc_->CreateTls(alloc_.ctx_);
  }

  HSHM_INLINE_CROSS_FUN
  ScopedTlsAllocator(const CtxAllocator<AllocT> &alloc) : alloc_(alloc) {
    alloc_->CreateTls(alloc_.ctx_);
  }

  HSHM_INLINE_CROSS_FUN
  ~ScopedTlsAllocator() { alloc_->FreeTls(alloc_.ctx_); }

  /** Arrow operator */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator<AllocT> &operator->() { return alloc_; }

  /** Arrow operator (const) */
  HSHM_INLINE_CROSS_FUN
  const CtxAllocator<AllocT> &operator->() const { return alloc_; }

  /** Star operator */
  HSHM_INLINE_CROSS_FUN
  CtxAllocator<AllocT> &operator*() { return alloc_; }

  /** Star operator (const) */
  HSHM_INLINE_CROSS_FUN
  const CtxAllocator<AllocT> &operator*() const { return alloc_; }
};

/** Thread-local storage manager */
template <typename AllocT>
class TlsAllocatorInfo : public thread::ThreadLocalData {
 public:
  AllocT *alloc_;
  ThreadId tid_;

 public:
  HSHM_CROSS_FUN
  TlsAllocatorInfo() : alloc_(nullptr), tid_(ThreadId::GetNull()) {}

  HSHM_CROSS_FUN
  void destroy() { alloc_->FreeTls(tid_); }
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

/** Allocator ID hash */
template <>
struct hash<hshm::ipc::AllocatorId> {
  std::size_t operator()(const hshm::ipc::AllocatorId &key) const {
    // Combine backend_id and sub_id into a single hash
    size_t h1 = hshm::hash<uint64_t>{}((uint64_t)key.backend_id_.major_ << 32 | key.backend_id_.minor_);
    size_t h2 = hshm::hash<uint32_t>{}(key.sub_id_);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

}  // namespace std

namespace hshm {

/** Allocator ID hash */
template <>
struct hash<hshm::ipc::AllocatorId> {
  HSHM_INLINE_CROSS_FUN std::size_t operator()(
      const hshm::ipc::AllocatorId &key) const {
    // Combine backend_id and sub_id into a single hash
    size_t h1 = hshm::hash<uint64_t>{}((uint64_t)key.backend_id_.major_ << 32 | key.backend_id_.minor_);
    size_t h2 = hshm::hash<uint32_t>{}(key.sub_id_);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

}  // namespace hshm

#define IS_SHM_POINTER(T) std::is_base_of_v<hipc::ShmPointer, T>

#endif  // HSHM_MEMORY_ALLOCATOR_ALLOCATOR_H_
