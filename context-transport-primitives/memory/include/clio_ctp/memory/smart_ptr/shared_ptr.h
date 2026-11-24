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

#ifndef CTP_MEMORY_SMART_PTR_SHARED_PTR_H_
#define CTP_MEMORY_SMART_PTR_SHARED_PTR_H_

#include <cstddef>
#include <utility>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/types/atomic.h"
#include "clio_ctp/types/numbers.h"

namespace ctp {

/**
 * Control block for ctp::shared_ptr.
 *
 * Co-located immediately in front of the managed object: make_shared performs
 * a single allocation holding [SharedPtrHeader | padding | T] so the header and
 * data share one lifetime and one free. Only the reference count lives here --
 * the owning shared_ptr knows T statically (it destructs data_ as T directly),
 * so no type-erased deleter is needed.
 */
struct SharedPtrHeader {
  /** Strong reference count. Starts at 1 (set by make_shared). */
  ctp::ipc::atomic<ctp::u32> ref_count_;

  /**
   * Type-erased destructor. make_shared installs &DestroyAs<T> for the concrete
   * T, so the last owner destructs the object as its REAL type even when the
   * surviving shared_ptr is a base view (e.g. shared_ptr<Task> owning a
   * CreateTask). This is what lets Task keep a non-virtual destructor while
   * tasks are allocated by runtime method id. nullptr => trivially destructible
   * / no destruction needed.
   */
  void (*destroy_)(void *obj);

  CTP_CROSS_FUN SharedPtrHeader() : ref_count_(1), destroy_(nullptr) {}
};

/** Type-erased destructor body: destructs *obj as a concrete T. */
template <typename T>
CTP_CROSS_FUN void SharedPtrDestroyAs(void *obj) {
  static_cast<T *>(obj)->~T();
}

// Forward declaration so make_shared can be befriended below.
template <typename T, typename AllocT>
class shared_ptr;

template <typename T, typename AllocT, typename... Args>
CTP_CROSS_FUN shared_ptr<T, AllocT> make_shared(AllocT *alloc, Args &&...args);

/**
 * An intrusively-allocated, reference-counted smart pointer.
 *
 * Unlike std::shared_ptr, the control block (SharedPtrHeader) and the managed
 * object are carved from a single allocation served by an allocator (AllocT),
 * and the shared_ptr remembers that allocator so the last owner can free the
 * block. Construct one ONLY via ctp::make_shared -- there is deliberately no
 * constructor that adopts a raw pointer.
 *
 * Layout (must stay {alloc_, header_, data_} for Cast() to be a valid in-place
 * reinterpret across instantiations):
 *   1. AllocT *alloc_   -- allocator the block was carved from (for Free)
 *   2. Header *header_  -- control block (reference count)
 *   3. T      *data_    -- the managed object
 *
 * @tparam T      The managed object type
 * @tparam AllocT The allocator type (e.g. ctp::ipc::MallocAllocator)
 */
template <typename T, typename AllocT>
class shared_ptr {
 public:
  using element_type = T;
  using allocator_type = AllocT;
  using Header = SharedPtrHeader;

  // All instantiations are friends so Cast<U>() and the copy/move from a
  // related type can touch each other's members.
  template <typename U, typename A>
  friend class shared_ptr;

  template <typename U, typename A, typename... Args>
  friend CTP_CROSS_FUN shared_ptr<U, A> make_shared(A *alloc, Args &&...args);

  AllocT *alloc_;   /**< Allocator the block was carved from */
  Header *header_;  /**< Control block (reference count) */
  T *data_;         /**< The managed object */

 private:
  /**
   * Internal constructor used by make_shared. Adopts an already-constructed
   * block whose header ref_count_ is 1. Private so callers cannot smuggle in
   * raw pointers.
   */
  CTP_CROSS_FUN shared_ptr(AllocT *alloc, Header *header, T *data)
      : alloc_(alloc), header_(header), data_(data) {}

  /** Decrement the refcount; destruct + free the block when it hits zero. */
  CTP_CROSS_FUN void Release() {
    if (header_ != nullptr) {
      // fetch_sub returns the PRIOR value, so ==1 means we dropped the last
      // reference.
      if (header_->ref_count_.fetch_sub(1) == 1) {
        if (data_ != nullptr && header_->destroy_ != nullptr) {
          // Destruct as the CONCRETE type recorded by make_shared, not as the
          // static T of this (possibly base) view.
          header_->destroy_(data_);
        }
        header_->~Header();
        if (alloc_ != nullptr) {
          // Reconstruct a FullPtr to the block start (the header) and free it.
          // The (alloc, raw-ptr) FullPtr ctor recomputes the offset, which is
          // correct for both the MallocAllocator (backend data == 0, offset ==
          // address) and offset-based allocators.
          ctp::ipc::FullPtr<void> block(
              alloc_, reinterpret_cast<void *>(header_));
          alloc_->Free(block);
        }
      }
    }
    alloc_ = nullptr;
    header_ = nullptr;
    data_ = nullptr;
  }

 public:
  /** Default constructor - null pointer. */
  CTP_CROSS_FUN shared_ptr() : alloc_(nullptr), header_(nullptr),
                               data_(nullptr) {}

  /**
   * Build a NON-OWNING view over an existing object. The result has a null
   * control header, so it participates in no reference counting and frees
   * nothing on destruction — it is purely a typed handle. Used for interop at
   * boundaries where an object's storage is owned elsewhere (e.g. a reused
   * scratch buffer or a device-resident task copied to host). Prefer make_shared
   * for ordinary ownership; this is the deliberate, named exception to the
   * "no raw pointers" rule.
   */
  CTP_CROSS_FUN static shared_ptr WrapNonOwning(T *ptr) {
    shared_ptr p;
    p.alloc_ = nullptr;
    p.header_ = nullptr;
    p.data_ = ptr;
    return p;
  }

  /** Copy constructor - shares ownership (increments refcount). */
  CTP_CROSS_FUN shared_ptr(const shared_ptr &other)
      : alloc_(other.alloc_), header_(other.header_), data_(other.data_) {
    if (header_ != nullptr) {
      header_->ref_count_.fetch_add(1);
    }
  }

  /** Move constructor - transfers ownership (no refcount change). */
  CTP_CROSS_FUN shared_ptr(shared_ptr &&other) noexcept
      : alloc_(other.alloc_), header_(other.header_), data_(other.data_) {
    other.alloc_ = nullptr;
    other.header_ = nullptr;
    other.data_ = nullptr;
  }

  /** Copy assignment - shares ownership. */
  CTP_CROSS_FUN shared_ptr &operator=(const shared_ptr &other) {
    if (this != &other) {
      Release();
      alloc_ = other.alloc_;
      header_ = other.header_;
      data_ = other.data_;
      if (header_ != nullptr) {
        header_->ref_count_.fetch_add(1);
      }
    }
    return *this;
  }

  /** Move assignment - transfers ownership. */
  CTP_CROSS_FUN shared_ptr &operator=(shared_ptr &&other) noexcept {
    if (this != &other) {
      Release();
      alloc_ = other.alloc_;
      header_ = other.header_;
      data_ = other.data_;
      other.alloc_ = nullptr;
      other.header_ = nullptr;
      other.data_ = nullptr;
    }
    return *this;
  }

  /** Destructor - drops this reference. */
  CTP_CROSS_FUN ~shared_ptr() { Release(); }

  /**
   * Reinterpret-cast in place to a shared_ptr of a different element type.
   *
   * Returns a REFERENCE to this same object viewed as shared_ptr<U, AllocT>;
   * it does NOT copy and does NOT change the reference count. Safe because all
   * shared_ptr<*, AllocT> share the {alloc_, header_, data_} layout. Used to
   * up/down-cast between a concrete task type and the Task base.
   */
  template <typename U>
  CTP_CROSS_FUN shared_ptr<U, AllocT> &Cast() {
    return *reinterpret_cast<shared_ptr<U, AllocT> *>(this);
  }

  /** Reinterpret-cast in place (const overload). */
  template <typename U>
  CTP_CROSS_FUN const shared_ptr<U, AllocT> &Cast() const {
    return *reinterpret_cast<const shared_ptr<U, AllocT> *>(this);
  }

  /** Release this reference and become null. */
  CTP_CROSS_FUN void reset() { Release(); }

  /** Raw pointer to the managed object. */
  CTP_CROSS_FUN T *get() const { return data_; }

  /**
   * Implicit NON-OWNING view as a ctp::ipc::FullPtr<T> (null allocator), for
   * interop with the FullPtr-based transport/serialization/per-task APIs
   * (e.g. Task::Copy / AggregateOut / serialize which still take a
   * `const FullPtr<T>&`). The returned FullPtr does not own the object — the
   * shared_ptr must outlive it. Do NOT use this to take ownership (e.g.
   * `FullPtr<T> x = sp;` is fine only while sp lives).
   */
  CTP_CROSS_FUN operator ctp::ipc::FullPtr<T>() const {
    return ctp::ipc::FullPtr<T>(data_);
  }

  /** Member access. */
  CTP_CROSS_FUN T *operator->() const { return data_; }

  /** Dereference. */
  CTP_CROSS_FUN T &operator*() const { return *data_; }

  /** True when this pointer owns nothing. */
  CTP_CROSS_FUN bool IsNull() const { return data_ == nullptr; }

  /** Boolean test (true when non-null). */
  CTP_CROSS_FUN explicit operator bool() const { return data_ != nullptr; }

  /** Current reference count (0 when null). */
  CTP_CROSS_FUN ctp::u32 use_count() const {
    return header_ != nullptr ? header_->ref_count_.load() : 0;
  }

  /** Equality (identity of the managed object). */
  CTP_CROSS_FUN bool operator==(const shared_ptr &other) const {
    return data_ == other.data_;
  }

  /** Inequality. */
  CTP_CROSS_FUN bool operator!=(const shared_ptr &other) const {
    return data_ != other.data_;
  }
};

/**
 * Allocate a single block holding a SharedPtrHeader and a T, construct both,
 * and return a shared_ptr owning it with a reference count of 1.
 *
 * The header and the object always come from ONE allocation. The data sub-
 * region is aligned to alignof(T). Returns a null shared_ptr if the allocation
 * fails.
 *
 * @tparam T      Object type to construct
 * @tparam AllocT Allocator type
 * @param alloc   Allocator to carve the block from
 * @param args    Constructor arguments for T
 */
template <typename T, typename AllocT, typename... Args>
CTP_CROSS_FUN shared_ptr<T, AllocT> make_shared(AllocT *alloc, Args &&...args) {
  // Place T after the header, aligned to alignof(T).
  constexpr size_t kAlign = alignof(T) > alignof(SharedPtrHeader)
                                ? alignof(T)
                                : alignof(SharedPtrHeader);
  constexpr size_t kDataOff =
      (sizeof(SharedPtrHeader) + kAlign - 1) / kAlign * kAlign;
  const size_t total = kDataOff + sizeof(T);

  ctp::ipc::FullPtr<char> block = alloc->template Allocate<char>(total);
  if (block.IsNull()) {
    return shared_ptr<T, AllocT>();
  }

  auto *header = reinterpret_cast<SharedPtrHeader *>(block.ptr_);
  new (header) SharedPtrHeader();  // ref_count_ = 1
  header->destroy_ = &SharedPtrDestroyAs<T>;  // destruct as concrete T
  T *data = reinterpret_cast<T *>(block.ptr_ + kDataOff);
  new (data) T(std::forward<Args>(args)...);

  return shared_ptr<T, AllocT>(alloc, header, data);
}

}  // namespace ctp

#endif  // CTP_MEMORY_SMART_PTR_SHARED_PTR_H_
