/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CTP_MEMORY_SMART_PTR_UNIQUE_PTR_H_
#define CTP_MEMORY_SMART_PTR_UNIQUE_PTR_H_

#include <cstddef>
#include <utility>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/memory/allocator/allocator.h"

namespace ctp {

/** Type-erased destructor body for unique_ptr: destructs *obj as a concrete T.
 *  Captured by make_unique where T is complete, so ~unique_ptr can run with an
 *  incomplete T (e.g. a unique_ptr<RunContext> member destroyed inside an inline
 *  Task destructor where RunContext is only forward-declared). */
template <typename T>
CTP_CROSS_FUN void UniquePtrDestroyAs(void *obj) {
  static_cast<T *>(obj)->~T();
}

template <typename T, typename AllocT>
class unique_ptr;

template <typename T, typename AllocT, typename... Args>
CTP_CROSS_FUN unique_ptr<T, AllocT> make_unique(AllocT *alloc, Args &&...args);

/**
 * A move-only, single-owner smart pointer carved from an allocator (AllocT).
 *
 * Mirrors ctp::shared_ptr but without the reference count: the sole owner frees
 * the object on destruction. Construct ONLY via ctp::make_unique (no raw-pointer
 * adoption). The methods are CTP_CROSS_FUN, but allocation only happens on the
 * host (make_unique is a no-op on device); a default-constructed unique_ptr is
 * inert on device, which is all device code ever sees (e.g. a Task's RunContext
 * handle is host-only).
 *
 * Stores {alloc_, data_, destroy_}: the allocator, the object, and a type-erased
 * deleter (so destruction needs no complete type at the use site).
 */
template <typename T, typename AllocT>
class unique_ptr {
 public:
  using element_type = T;

  template <typename U, typename A, typename... Args>
  friend CTP_CROSS_FUN unique_ptr<U, A> make_unique(A *alloc, Args &&...args);

  AllocT *alloc_;          /**< Allocator the object was carved from */
  T *data_;                /**< The managed object */
  void (*destroy_)(void *);/**< Type-erased destructor (concrete T) */

 private:
  CTP_CROSS_FUN unique_ptr(AllocT *alloc, T *data, void (*destroy)(void *))
      : alloc_(alloc), data_(data), destroy_(destroy) {}

  CTP_CROSS_FUN void DoReset() {
    if (data_ != nullptr) {
      if (destroy_ != nullptr) {
        destroy_(data_);
      }
      if (alloc_ != nullptr) {
        ctp::ipc::FullPtr<void> block(alloc_, reinterpret_cast<void *>(data_));
        alloc_->Free(block);
      }
    }
    alloc_ = nullptr;
    data_ = nullptr;
    destroy_ = nullptr;
  }

 public:
  /** Default constructor - null. */
  CTP_CROSS_FUN unique_ptr() : alloc_(nullptr), data_(nullptr),
                               destroy_(nullptr) {}

  /** Move constructor - transfers ownership. */
  CTP_CROSS_FUN unique_ptr(unique_ptr &&other) noexcept
      : alloc_(other.alloc_), data_(other.data_), destroy_(other.destroy_) {
    other.alloc_ = nullptr;
    other.data_ = nullptr;
    other.destroy_ = nullptr;
  }

  /** Move assignment - transfers ownership. */
  CTP_CROSS_FUN unique_ptr &operator=(unique_ptr &&other) noexcept {
    if (this != &other) {
      DoReset();
      alloc_ = other.alloc_;
      data_ = other.data_;
      destroy_ = other.destroy_;
      other.alloc_ = nullptr;
      other.data_ = nullptr;
      other.destroy_ = nullptr;
    }
    return *this;
  }

  /** Non-copyable. */
  unique_ptr(const unique_ptr &) = delete;
  unique_ptr &operator=(const unique_ptr &) = delete;

  /** Destructor - frees the owned object. */
  CTP_CROSS_FUN ~unique_ptr() { DoReset(); }

  /** Destroy the owned object and become null. */
  CTP_CROSS_FUN void reset() { DoReset(); }

  /** Relinquish ownership without freeing; returns the raw pointer. */
  CTP_CROSS_FUN T *release() {
    T *d = data_;
    alloc_ = nullptr;
    data_ = nullptr;
    destroy_ = nullptr;
    return d;
  }

  CTP_CROSS_FUN T *get() const { return data_; }
  CTP_CROSS_FUN T *operator->() const { return data_; }
  CTP_CROSS_FUN T &operator*() const { return *data_; }
  CTP_CROSS_FUN bool IsNull() const { return data_ == nullptr; }
  CTP_CROSS_FUN explicit operator bool() const { return data_ != nullptr; }
};

/**
 * Allocate and construct a single T from `alloc`, returning a unique_ptr that
 * owns it. Host-only: on a device pass it returns a null unique_ptr (device code
 * never allocates these). Returns null on allocation failure.
 */
template <typename T, typename AllocT, typename... Args>
CTP_CROSS_FUN unique_ptr<T, AllocT> make_unique(AllocT *alloc, Args &&...args) {
#if CTP_IS_HOST
  ctp::ipc::FullPtr<char> block = alloc->template Allocate<char>(sizeof(T));
  if (block.IsNull()) {
    return unique_ptr<T, AllocT>();
  }
  T *data = reinterpret_cast<T *>(block.ptr_);
  new (data) T(std::forward<Args>(args)...);
  return unique_ptr<T, AllocT>(alloc, data, &UniquePtrDestroyAs<T>);
#else
  (void)alloc;
  return unique_ptr<T, AllocT>();
#endif
}

}  // namespace ctp

#endif  // CTP_MEMORY_SMART_PTR_UNIQUE_PTR_H_
