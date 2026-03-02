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

#ifndef HSHM_INCLUDE_HSHM_TYPES_ATOMIC_H_
#define HSHM_INCLUDE_HSHM_TYPES_ATOMIC_H_

#include <atomic>
#include <type_traits>

#include "hermes_shm/constants/macros.h"
#include "numbers.h"
#if HSHM_IS_CUDA_COMPILER
#include <cuda/atomic>
#endif
#if HSHM_IS_ROCM_COMPILER
#include <hip/hip_runtime.h>
#endif

namespace hshm::ipc {

/** Provides the API of an atomic, without being atomic */
template <typename T>
struct nonatomic {
  T x;

  /** Serialization */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    ar(x);
  }

  /** Integer convertion */
  HSHM_INLINE_CROSS_FUN operator T() const { return x; }

  /** Constructor */
  HSHM_INLINE_CROSS_FUN nonatomic() = default;

  /** Full constructor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic(U def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN nonatomic(const nonatomic &other) : x(other.x) {}

  /* Move constructor */
  HSHM_INLINE_CROSS_FUN nonatomic(nonatomic &&other) : x(std::move(other.x)) {}

  /** Copy assign operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator=(const nonatomic &other) {
    x = other.x;
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator=(nonatomic &&other) {
    x = std::move(other.x);
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_add(U count, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    T orig_x = x;
    x += (T)count;
    return orig_x;
  }

  /** Atomic fetch_sub wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_sub(U count, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    T orig_x = x;
    x -= (T)count;
    return orig_x;
  }

  /** Atomic load wrapper */
  HSHM_INLINE_CROSS_FUN T
  load(std::memory_order order = std::memory_order_seq_cst) const {
    (void)order;
    return x;
  }

  /** Atomic store wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void
  store(U val, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    x = (T)val;
  }

  /** System-scope store (same as store for nonatomic) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void store_system(U val) { x = (T)val; }

  /** System-scope load (same as load for nonatomic) */
  HSHM_INLINE_CROSS_FUN T load_system() const { return x; }

  /** Get reference to x */
  HSHM_INLINE_CROSS_FUN T &ref() { return x; }

  /** Get const reference to x */
  HSHM_INLINE_CROSS_FUN const T &ref() const { return x; }

  /** Atomic exchange wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void exchange(
      U count, std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    x = count;
  }

  /** Atomic compare exchange weak wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_weak(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    if (x == expected) {
      x = (T)desired;
      return true;
    } else {
      expected = x;
      return false;
    }
  }

  /** Atomic compare exchange strong wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    (void)order;
    if (x == expected) {
      x = (T)desired;
      return true;
    } else {
      expected = x;
      return false;
    }
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator++() {
    ++x;
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE_CROSS_FUN nonatomic operator++(int) { return atomic(x + 1); }

  /** Atomic pre-decrement operator */
  HSHM_INLINE_CROSS_FUN nonatomic &operator--() {
    --x;
    return *this;
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE_CROSS_FUN nonatomic operator--(int) {
    nonatomic orig_x(x);
    --x;
    return orig_x;
  }

  /** Atomic add operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator+(U count) const {
    return nonatomic(x + count);
  }

  /** Atomic subtract operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator-(U count) const {
    return nonatomic(x - count);
  }

  /** Atomic add assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator+=(U count) {
    x += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator-=(U count) {
    x -= count;
    return *this;
  }

  /** Atomic assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator=(U count) {
    x = count;
    return *this;
  }

  /** Equality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator==(U other) const {
    return (static_cast<T>(other) == x);
  }

  /** Inequality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator!=(U other) const {
    return (static_cast<T>(other) != x);
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const nonatomic &other) const {
    return (other.x == x);
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const nonatomic &other) const {
    return (other.x != x);
  }

  /** Bitwise and */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator&(U other) const {
    return nonatomic(x & other);
  }

  /** Bitwise or */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator|(U other) const {
    return nonatomic(x | other);
  }

  /** Bitwise xor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic operator^(U other) const {
    return nonatomic(x ^ other);
  }

  /** Bitwise and assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator&=(U other) {
    x &= other;
    return *this;
  }

  /** Bitwise or assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator|=(U other) {
    x |= other;
    return *this;
  }

  /** System-scope bitwise or assign (same as |= for nonatomic) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &or_system(U other) {
    x |= other;
    return *this;
  }

  /** Bitwise xor assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN nonatomic &operator^=(U other) {
    x ^= other;
    return *this;
  }
};

/** A wrapper for CUDA atomic operations */
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
template <typename T>
struct rocm_atomic {
  T x;

  /** Integer convertion */
  HSHM_INLINE_CROSS_FUN operator T() const { return x; }

  /** Constructor */
  HSHM_INLINE_CROSS_FUN rocm_atomic() = default;

  /** Full constructor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic(U def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN rocm_atomic(const rocm_atomic &other) : x(other.x) {}

  /* Move constructor */
  HSHM_INLINE_CROSS_FUN rocm_atomic(rocm_atomic &&other)
      : x(std::move(other.x)) {}

  /** Copy assign operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator=(const rocm_atomic &other) {
    x = other.x;
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator=(rocm_atomic &&other) {
    x = std::move(other.x);
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_add(U count, std::memory_order order = std::memory_order_seq_cst) {
    if constexpr (sizeof(T) == 8) {
      return atomicAdd(reinterpret_cast<unsigned long long*>(&x), static_cast<unsigned long long>(count));
    } else {
      return atomicAdd(&x, count);
    }
  }

  /** Atomic fetch_sub wrapper*/
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  fetch_sub(U count, std::memory_order order = std::memory_order_seq_cst) {
    if constexpr (sizeof(T) == 8) {
      return atomicAdd(reinterpret_cast<unsigned long long*>(&x), static_cast<unsigned long long>(-count));
    } else {
      return atomicAdd(&x, -count);
    }
  }

  /** Atomic load wrapper */
  HSHM_INLINE_CROSS_FUN T
  load(std::memory_order order = std::memory_order_seq_cst) const {
    return x;
  }

  /** Atomic store wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void store(
      U count, std::memory_order order = std::memory_order_seq_cst) {
    exchange(count);
  }

  /** Atomic exchange wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN T
  exchange(U count, std::memory_order order = std::memory_order_seq_cst) {
    if constexpr (sizeof(T) == 8) {
      return atomicExch(reinterpret_cast<unsigned long long*>(&x), static_cast<unsigned long long>(count));
    } else {
      return atomicExch(&x, count);
    }
  }

  /** System-scope atomic store (visible to CPU from GPU) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN void store_system(U count) {
#if HSHM_IS_GPU
    if constexpr (sizeof(T) == 8) {
      atomicExch_system(reinterpret_cast<unsigned long long*>(&x),
                        static_cast<unsigned long long>(count));
    } else {
      atomicExch_system(reinterpret_cast<unsigned int*>(&x),
                        static_cast<unsigned int>(count));
    }
#else
    exchange(count);
#endif
  }

  /** System-scope atomic load (visible across GPU/CPU boundary).
   * Uses volatile read since HostNativeAtomicSupported=0 on many GPUs
   * means atomicAdd_system(ptr,0) won't see CPU-side writes. */
  HSHM_INLINE_CROSS_FUN T load_system() const {
#if HSHM_IS_GPU
    return *reinterpret_cast<const volatile T*>(&x);
#else
    return x;
#endif
  }

  /** Atomic compare exchange weak wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_weak(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return atomicCAS(const_cast<T *>(&x), expected, desired);
  }

  /** Atomic compare exchange strong wrapper */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool compare_exchange_strong(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return atomicCAS(const_cast<T *>(&x), expected, desired);
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator++() {
    atomicAdd(&x, 1);
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic operator++(int) { return atomic(x + 1); }

  /** Atomic pre-decrement operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator--() {
    atomicAdd(&x, (T)(-1));
    return (*this);
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE_CROSS_FUN rocm_atomic operator--(int) { return atomic(x - 1); }

  /** Atomic add operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator+(U count) const {
    return atomicAdd(&x, count);
  }

  /** Atomic subtract operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator-(U count) const {
    return atomicAdd(&x, (T)(-count));
  }

  /** Atomic add assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator+=(U count) {
    atomicAdd(&x, count);
    return *this;
  }

  /** Atomic subtract assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator-=(U count) {
    atomicAdd(&x, -count);
    return *this;
  }

  /** Atomic assign operator */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator=(U count) {
    store(count);
    return *this;
  }

  /** Equality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator==(U other) const {
    return atomicCAS(const_cast<T *>(&x), other, other);
  }

  /** Inequality check (number) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN bool operator!=(U other) const {
    return !atomicCAS(const_cast<T *>(&x), other, other);
  }

  /** Equality check */
  HSHM_INLINE_CROSS_FUN bool operator==(const rocm_atomic &other) const {
    return atomicCAS(const_cast<T *>(&x), other.x, other.x);
  }

  /** Inequality check */
  HSHM_INLINE_CROSS_FUN bool operator!=(const rocm_atomic &other) const {
    return !atomicCAS(const_cast<T *>(&x), other.x, other.x);
  }

  /** Bitwise and */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator&(U other) const {
    T *addr = const_cast<T *>(&x);
    return atomicAnd(addr, other);
  }

  /** Bitwise or */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator|(U other) const {
    T *addr = const_cast<T *>(&x);
    return atomicOr(addr, other);
  }

  /** Bitwise xor */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic operator^(U other) const {
    T *addr = const_cast<T *>(&x);
    return atomicXor(addr, other);
  }

  /** Bitwise and assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator&=(U other) {
    atomicAnd(&x, other);
    return *this;
  }

  /** Bitwise or assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator|=(U other) {
    atomicOr(&x, other);
    return *this;
  }

  /** System-scope bitwise or assign (visible to CPU from GPU) */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &or_system(U other) {
#if HSHM_IS_GPU
    atomicOr_system(reinterpret_cast<unsigned int*>(&x),
                    static_cast<unsigned int>(other));
#else
    atomicOr(&x, other);
#endif
    return *this;
  }

  /** Bitwise xor assign */
  template <typename U>
  HSHM_INLINE_CROSS_FUN rocm_atomic &operator^=(U other) {
    atomicXor(&x, other);
    return *this;
  }

  /** Serialization */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    ar(x);
  }
};
#endif

/** A wrapper around std::atomic */
template <typename T>
struct std_atomic {
  std::atomic<T> x;

  /** Serialization - properly handles std::atomic by loading/storing value */
  template <typename Ar>
  void save(Ar &ar) const {
    T val = x.load(std::memory_order_relaxed);
    ar(val);
  }

  /** Deserialization - properly handles std::atomic by loading/storing value */
  template <typename Ar>
  void load(Ar &ar) {
    T val;
    ar(val);
    x.store(val, std::memory_order_relaxed);
  }

  /** Integer convertion */
  HSHM_INLINE_CROSS_FUN operator T() const { return x; }

  /** Constructor */
  HSHM_INLINE std_atomic() = default;

  /** Full constructor */
  template <typename U>
  HSHM_INLINE std_atomic(U def) : x(def) {}

  /** Copy constructor */
  HSHM_INLINE std_atomic(const std_atomic &other) : x(other.x.load()) {}

  /* Move constructor */
  HSHM_INLINE std_atomic(std_atomic &&other) : x(other.x.load()) {}

  /** Copy assign operator */
  HSHM_INLINE std_atomic &operator=(const std_atomic &other) {
    x = other.x.load();
    return *this;
  }

  /** Move assign operator */
  HSHM_INLINE std_atomic &operator=(std_atomic &&other) {
    x = other.x.load();
    return *this;
  }

  /** Atomic fetch_add wrapper*/
  template <typename U>
  HSHM_INLINE T fetch_add(U count,
                          std::memory_order order = std::memory_order_seq_cst) {
    return x.fetch_add(count, order);
  }

  /** Atomic fetch_sub wrapper*/
  template <typename U>
  HSHM_INLINE T fetch_sub(U count,
                          std::memory_order order = std::memory_order_seq_cst) {
    return x.fetch_sub(count, order);
  }

  /** Atomic load wrapper */
  HSHM_INLINE T
  load(std::memory_order order = std::memory_order_seq_cst) const {
    return x.load(order);
  }

  /** Atomic store wrapper */
  template <typename U>
  HSHM_INLINE void store(U count,
                         std::memory_order order = std::memory_order_seq_cst) {
    x.store(count, order);
  }

  /** System-scope store (same as store for std_atomic) */
  template <typename U>
  HSHM_INLINE void store_system(U count) {
    x.store(count, std::memory_order_seq_cst);
  }

  /** System-scope load (same as load for std_atomic) */
  HSHM_INLINE T load_system() const {
    return x.load(std::memory_order_seq_cst);
  }

  /** Atomic exchange wrapper */
  template <typename U>
  HSHM_INLINE void exchange(
      U count, std::memory_order order = std::memory_order_seq_cst) {
    x.exchange(count, order);
  }

  /** Atomic compare exchange weak wrapper */
  template <typename U>
  HSHM_INLINE bool compare_exchange_weak(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return x.compare_exchange_weak(expected, desired, order);
  }

  /** Atomic compare exchange strong wrapper */
  template <typename U>
  HSHM_INLINE bool compare_exchange_strong(
      T &expected, U desired,
      std::memory_order order = std::memory_order_seq_cst) {
    return x.compare_exchange_strong(expected, desired, order);
  }

  /** Atomic pre-increment operator */
  HSHM_INLINE std_atomic &operator++() {
    ++x;
    return *this;
  }

  /** Atomic post-increment operator */
  HSHM_INLINE std_atomic operator++(int) { return atomic(x + 1); }

  /** Atomic pre-decrement operator */
  HSHM_INLINE std_atomic &operator--() {
    --x;
    return *this;
  }

  /** Atomic post-decrement operator */
  HSHM_INLINE std_atomic operator--(int) { return atomic(x - 1); }

  /** Atomic add operator */
  template <typename U>
  HSHM_INLINE std_atomic operator+(U count) const {
    return x + count;
  }

  /** Atomic subtract operator */
  template <typename U>
  HSHM_INLINE std_atomic operator-(U count) const {
    return x - count;
  }

  /** Atomic add assign operator */
  template <typename U>
  HSHM_INLINE std_atomic &operator+=(U count) {
    x += count;
    return *this;
  }

  /** Atomic subtract assign operator */
  template <typename U>
  HSHM_INLINE std_atomic &operator-=(U count) {
    x -= count;
    return *this;
  }

  /** Atomic assign operator */
  template <typename U>
  HSHM_INLINE std_atomic &operator=(U count) {
    x.exchange(count);
    return *this;
  }

  /** Equality check (number) */
  template <typename U>
  HSHM_INLINE bool operator==(U other) const {
    return (other == x);
  }

  /** Inequality check (number) */
  template <typename U>
  HSHM_INLINE bool operator!=(U other) const {
    return (other != x);
  }

  /** Equality check */
  HSHM_INLINE bool operator==(const std_atomic &other) const {
    return (other.x == x);
  }

  /** Inequality check */
  HSHM_INLINE bool operator!=(const std_atomic &other) const {
    return (other.x != x);
  }

  /** Bitwise and */
  template <typename U>
  HSHM_INLINE std_atomic operator&(U other) const {
    return x & other;
  }

  /** Bitwise or */
  template <typename U>
  HSHM_INLINE std_atomic operator|(U other) const {
    return x | other;
  }

  /** Bitwise xor */
  template <typename U>
  HSHM_INLINE std_atomic operator^(U other) const {
    return x ^ other;
  }

  /** Bitwise and assign */
  template <typename U>
  HSHM_INLINE std_atomic &operator&=(U other) {
    x &= other;
    return *this;
  }

  /** Bitwise or assign */
  template <typename U>
  HSHM_INLINE std_atomic &operator|=(U other) {
    x |= other;
    return *this;
  }

  /** System-scope bitwise or assign (same as |= for std_atomic) */
  template <typename U>
  HSHM_INLINE std_atomic &or_system(U other) {
    x |= other;
    return *this;
  }

  /** Bitwise xor assign */
  template <typename U>
  HSHM_INLINE std_atomic &operator^=(U other) {
    x ^= other;
    return *this;
  }
};

#if HSHM_IS_HOST
template <typename T>
using atomic = std_atomic<T>;
#endif

#if HSHM_IS_GPU && HSHM_ENABLE_CUDA_OR_ROCM
template <typename T>
using atomic = rocm_atomic<T>;
#endif

#if HSHM_IS_GPU && !HSHM_ENABLE_CUDA_OR_ROCM
// Fallback for nvcc's device-compilation pass when HSHM_ENABLE_CUDA=0.
// HSHM_IS_GPU=1 (via __CUDA_ARCH__) but no GPU atomic backend is configured.
// nonatomic<T> is HSHM_CROSS_FUN-safe and prevents "atomic is not a template"
// cascade errors in downstream lock/allocator headers.
template <typename T>
using atomic = nonatomic<T>;
#endif

template <typename T, bool is_atomic>
using opt_atomic =
    typename std::conditional<is_atomic, atomic<T>, nonatomic<T>>::type;

/** Device-scope memory fence */
HSHM_INLINE_CROSS_FUN static void threadfence() {
#if HSHM_IS_GPU
  __threadfence();
#else
  std::atomic_thread_fence(std::memory_order_release);
#endif
}

/** System-scope memory fence (ensures GPU writes are visible to CPU) */
HSHM_INLINE_CROSS_FUN static void threadfence_system() {
#if HSHM_IS_GPU
  __threadfence_system();
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace hshm::ipc

#endif  // HSHM_INCLUDE_HSHM_TYPES_ATOMIC_H_
