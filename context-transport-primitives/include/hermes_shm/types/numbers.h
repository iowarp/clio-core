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

#ifndef HSHM_SHM_INCLUDE_HSHM_SHM_TYPES_NUMBERS_H_
#define HSHM_SHM_INCLUDE_HSHM_SHM_TYPES_NUMBERS_H_

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

#include "hermes_shm/constants/macros.h"

namespace hshm {

typedef uint8_t u8;   /**< 8-bit unsigned integer */
typedef uint16_t u16; /**< 16-bit unsigned integer */
typedef uint32_t u32; /**< 32-bit unsigned integer */
typedef uint64_t u64; /**< 64-bit unsigned integer */
typedef int8_t i8;    /**< 8-bit signed integer */
typedef int16_t i16;  /**< 16-bit signed integer */
typedef int32_t i32;  /**< 32-bit signed integer */
typedef int64_t i64;  /**< 64-bit signed integer */
typedef float f32;    /**< 32-bit float */
typedef double f64;   /**< 64-bit float */

typedef char byte;                   /**< Signed char */
typedef unsigned char ubyte;         /**< Unsigned char */
typedef short short_int;             /**< Signed int */
typedef unsigned short short_uint;   /**< Unsigned int */
typedef int reg_int;                 /**< Signed int */
typedef unsigned reg_uint;           /**< Unsigned int */
typedef long long big_int;           /**< Long long */
typedef unsigned long long big_uint; /**< Unsigned long long */

struct ThreadId {
  hshm::u64 tid_;

  HSHM_INLINE_CROSS_FUN
  ThreadId() = default;

  HSHM_INLINE_CROSS_FUN
  explicit ThreadId(hshm::u64 tid) : tid_(tid) {}

  HSHM_INLINE_CROSS_FUN
  static ThreadId GetNull() { return ThreadId{(hshm::u64)-1}; }

  HSHM_INLINE_CROSS_FUN
  bool IsNull() const { return tid_ == (hshm::u64)-1; }

  HSHM_INLINE_CROSS_FUN
  void SetNull() { tid_ = (hshm::u64)-1; }

  HSHM_INLINE_CROSS_FUN
  bool operator==(const ThreadId &other) const { return tid_ == other.tid_; }

  HSHM_INLINE_CROSS_FUN
  bool operator!=(const ThreadId &other) const { return tid_ != other.tid_; }

  HSHM_INLINE_CROSS_FUN
  bool operator<(const ThreadId &other) const { return tid_ < other.tid_; }

  HSHM_INLINE_CROSS_FUN
  bool operator>(const ThreadId &other) const { return tid_ > other.tid_; }

  HSHM_INLINE_CROSS_FUN
  bool operator<=(const ThreadId &other) const { return tid_ <= other.tid_; }

  HSHM_INLINE_CROSS_FUN
  bool operator>=(const ThreadId &other) const { return tid_ >= other.tid_; }

  HSHM_INLINE_CROSS_FUN
  friend std::ostream &operator<<(std::ostream &os, const ThreadId &tid) {
    os << tid.tid_;
    return os;
  }
};

#if HSHM_ENABLE_CUDA or HSHM_ENABLE_ROCM
typedef reg_int min_i16;
typedef reg_int min_i32;
typedef big_uint min_i64;

typedef reg_uint min_u16;
typedef reg_uint min_u32;
typedef big_uint min_u64;
#else
typedef i16 min_i16;
typedef i32 min_i32;
typedef i64 min_i64;

typedef u16 min_u16;
typedef u32 min_u32;
typedef u64 min_u64;
#endif


template <typename T>
class Unit {
 public:
  template <typename U>
  CLS_CONST T Bytes(U n) {
    return (T)((n) * (((T)1) << 0));
  }
  template <typename U>
  CLS_CONST T Kilobytes(U n) {
    return (T)((n) * (((T)1) << 10));
  }
  template <typename U>
  CLS_CONST T Megabytes(U n) {
    return (T)((n) * (((T)1) << 20));
  }
  template <typename U>
  CLS_CONST T Gigabytes(U n) {
    return (T)((n) * (((T)1) << 30));
  }
  template <typename U>
  CLS_CONST T Terabytes(U n) {
    return (T)((n) * (((T)1) << 40));
  }
  template <typename U>
  CLS_CONST T Petabytes(U n) {
    return (T)((n) * (((T)1) << 50));
  }
  template <typename U>
  CLS_CONST T Seconds(U n) {
    return (T)(T(n) * 1000000000);
  }
  template <typename U>
  CLS_CONST T Milliseconds(U n) {
    return (T)((n) * 1000000);
  }
  template <typename U>
  CLS_CONST T Microseconds(U n) {
    return (T)((n) * 1000);
  }
  template <typename U>
  CLS_CONST T Nanoseconds(U n) {
    return (T)((n));
  }
};

/** DWORD type for windows compatability */
typedef u32 DWORD;

/** HANDLE type for windows compatability */
typedef void *HANDLE;

}  // namespace hshm

/** Bytes -> Bytes */
#ifndef BYTES
#define BYTES(n) (hshm::u64)((n) * (((hshm::u64)1) << 0))
#endif

/** KILOBYTES -> Bytes */
#ifndef KILOBYTES
#define KILOBYTES(n) (hshm::u64)((n) * (((hshm::u64)1) << 10))
#endif

/** MEGABYTES -> Bytes */
#ifndef MEGABYTES
#define MEGABYTES(n) (hshm::u64)((n) * (((hshm::u64)1) << 20))
#endif

/** GIGABYTES -> Bytes */
#ifndef GIGABYTES
#define GIGABYTES(n) (hshm::u64)((n) * (((hshm::u64)1) << 30))
#endif

/** TERABYTES -> Bytes */
#ifndef TERABYTES
#define TERABYTES(n) (hshm::u64)((n) * (((hshm::u64)1) << 40))
#endif

/** PETABYTES -> Bytes */
#ifndef PETABYTES
#define PETABYTES(n) (hshm::u64)((n) * (((hshm::u64)1) << 50))
#endif

/** Seconds to nanoseconds */
#ifndef SECONDS
#define SECONDS(n) (hshm::u64)((n) * 1000000000)
#endif

/** Milliseconds to nanoseconds */
#ifndef MILLISECONDS
#define MILLISECONDS(n) (hshm::u64)((n) * 1000000)
#endif

/** Microseconds to nanoseconds */
#ifndef MICROSECONDS
#define MICROSECONDS(n) (hshm::u64)((n) * 1000)
#endif

/** Nanoseconds to nanoseconds */
#ifndef NANOSECONDS
#define NANOSECONDS(n) (hshm::u64)(n)
#endif

#endif  // HSHM_SHM_INCLUDE_HSHM_SHM_TYPES_NUMBERS_H_
