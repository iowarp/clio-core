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

#ifndef HSHM_INCLUDE_HSHM_TYPES_BITFIELD_H_
#define HSHM_INCLUDE_HSHM_TYPES_BITFIELD_H_

#include <cstdint>

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/atomic.h"

namespace hshm {

#define BIT_OPT(T, n) (((T)1) << n)
#define ALL_BITS(T) (~((T)0))

/**
 * A generic bitfield template
 * */
template <typename T = u32, bool ATOMIC = false>
struct bitfield {
  hipc::opt_atomic<T, ATOMIC> bits_;

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN bitfield() : bits_(0) {}

  /** Emplace constructor */
  HSHM_INLINE_CROSS_FUN explicit bitfield(T mask) : bits_(mask) {}

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN bitfield(const bitfield &other) : bits_(other.bits_) {}

  /** Copy assignment */
  HSHM_INLINE_CROSS_FUN bitfield &operator=(const bitfield &other) {
    bits_ = other.bits_;
    return *this;
  }

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN bitfield(bitfield &&other) noexcept
      : bits_(other.bits_) {}

  /** Move assignment */
  HSHM_INLINE_CROSS_FUN bitfield &operator=(bitfield &&other) noexcept {
    bits_ = other.bits_;
    return *this;
  }

  /** Copy from any bitfield */
  template <bool ATOMIC2>
  HSHM_INLINE_CROSS_FUN bitfield(const bitfield<T, ATOMIC2> &other)
      : bits_(other.bits_) {}

  /** Copy assignment from any bitfield */
  template <bool ATOMIC2>
  HSHM_INLINE_CROSS_FUN bitfield &operator=(const bitfield<T, ATOMIC2> &other) {
    bits_ = other.bits_;
    return *this;
  }

  /** Set bits using mask */
  HSHM_INLINE_CROSS_FUN void SetBits(T mask) { bits_ |= mask; }

  /** Set bits using system-scope atomic (visible to CPU from GPU) */
  HSHM_INLINE_CROSS_FUN void SetBitsSystem(T mask) {
    T cur = bits_.load();
    bits_.store_system(cur | mask);
  }

  /** Unset bits in mask */
  HSHM_INLINE_CROSS_FUN void UnsetBits(T mask) { bits_ &= ~mask; }

  /** Check if any bits are set */
  HSHM_INLINE_CROSS_FUN T Any(T mask) const { return (bits_ & mask).load(); }

  /** Check if all bits are set */
  HSHM_INLINE_CROSS_FUN T All(T mask) const { return Any(mask) == mask; }

  /** Copy bits from another bitfield */
  HSHM_INLINE_CROSS_FUN void CopyBits(bitfield field, T mask) {
    bits_ &= (field.bits_ & mask);
  }

  /** Clear all bits */
  HSHM_INLINE_CROSS_FUN void Clear() { bits_ = 0; }

  /** Make a mask */
  HSHM_INLINE_CROSS_FUN static T MakeMask(int start, int length) {
    return ((((T)1) << length) - 1) << start;
  }

  /** Serialization */
  template <typename Ar>
  HSHM_CROSS_FUN void serialize(Ar &ar) {
    ar & bits_;
  }
};
typedef bitfield<u8> bitfield8_t;
typedef bitfield<u16> bitfield16_t;
typedef bitfield<u32> bitfield32_t;
typedef bitfield<u64> bitfield64_t;
typedef bitfield<int> ibitfield;

template <typename T>
using abitfield = bitfield<T, true>;
typedef abitfield<u8> abitfield8_t;
typedef abitfield<u16> abitfield16_t;
typedef abitfield<u32> abitfield32_t;
typedef abitfield<int> aibitfield;

/**
 * A helper type needed for std::conditional
 * */
template <size_t LEN>
struct len_bits {
  static constexpr size_t value = LEN;
};

/**
 * A generic bitfield template
 * */
template <size_t NUM_BITS,
          typename LEN = typename std::conditional<
              ((NUM_BITS % 32 == 0) && (NUM_BITS > 0)),
              len_bits<(NUM_BITS / 32)>, len_bits<(NUM_BITS / 32) + 1>>::type>
struct big_bitfield {
  bitfield32_t bits_[LEN::value];

  HSHM_INLINE_CROSS_FUN big_bitfield() : bits_() {}

  HSHM_INLINE_CROSS_FUN size_t size() const { return LEN::value; }

  HSHM_INLINE_CROSS_FUN void SetBits(int start, int length) {
    int bf_idx = start / 32;
    int bf_idx_count = 32 - bf_idx;
    int rem = length;
    while (rem) {
      bits_[bf_idx].SetBits(bitfield32_t::MakeMask(start, bf_idx_count));
      rem -= bf_idx_count;
      bf_idx += 1;
      if (rem >= 32) {
        bf_idx_count = 32;
      } else {
        bf_idx_count = rem;
      }
    }
  }

  HSHM_INLINE_CROSS_FUN void UnsetBits(int start, int length) {
    int bf_idx = start / 32;
    int bf_idx_count = 32 - bf_idx;
    int rem = length;
    while (rem) {
      bits_[bf_idx].SetBits(bitfield32_t::MakeMask(start, bf_idx_count));
      rem -= bf_idx_count;
      bf_idx += 1;
      if (rem >= 32) {
        bf_idx_count = 32;
      } else {
        bf_idx_count = rem;
      }
    }
  }

  HSHM_INLINE_CROSS_FUN bool Any(int start, int length) const {
    int bf_idx = start / 32;
    int bf_idx_count = 32 - bf_idx;
    int rem = length;
    while (rem) {
      if (bits_[bf_idx].Any(bitfield32_t::MakeMask(start, bf_idx_count))) {
        return true;
      }
      rem -= bf_idx_count;
      bf_idx += 1;
      if (rem >= 32) {
        bf_idx_count = 32;
      } else {
        bf_idx_count = rem;
      }
    }
    return false;
  }

  HSHM_INLINE_CROSS_FUN bool All(int start, int length) const {
    int bf_idx = start / 32;
    int bf_idx_count = 32 - bf_idx;
    int rem = length;
    while (rem) {
      if (!bits_[bf_idx].All(bitfield32_t::MakeMask(start, bf_idx_count))) {
        return false;
      }
      rem -= bf_idx_count;
      bf_idx += 1;
      if (rem >= 32) {
        bf_idx_count = 32;
      } else {
        bf_idx_count = rem;
      }
    }
    return true;
  }

  HSHM_INLINE_CROSS_FUN void Clear() {
    memset((void *)bits_, 0, sizeof(bitfield32_t) * LEN::value);
  }
};

}  // namespace hshm

#endif  // HSHM_INCLUDE_HSHM_TYPES_BITFIELD_H_
