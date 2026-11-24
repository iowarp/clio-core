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

#ifndef CTP_TYPES_HASH_H_
#define CTP_TYPES_HASH_H_

#include "clio_ctp/constants/macros.h"
#include <cstddef>
#include <cstdint>

namespace ctp {

/**
 * GPU-compatible hash template
 * Uses std::hash on CPU, custom implementation on GPU
 */
template <typename T>
struct hash {
  CTP_INLINE_CROSS_FUN std::size_t operator()(const T &key) const {
#if CTP_IS_GPU
    // GPU-compatible FNV-1a hash for basic types
    return fnv1a_hash(reinterpret_cast<const unsigned char*>(&key), sizeof(T));
#else
    // Use std::hash on CPU
    return std::hash<T>{}(key);
#endif
  }

 private:
  // FNV-1a hash constants
  static constexpr std::size_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
  static constexpr std::size_t FNV_PRIME = 1099511628211ULL;

  CTP_INLINE_CROSS_FUN
  static std::size_t fnv1a_hash(const unsigned char* data, std::size_t size) {
    std::size_t hash = FNV_OFFSET_BASIS;
    for (std::size_t i = 0; i < size; ++i) {
      hash ^= static_cast<std::size_t>(data[i]);
      hash *= FNV_PRIME;
    }
    return hash;
  }
};

// Specializations for common types to ensure std::hash compatibility on CPU
template <>
struct hash<uint8_t> {
  CTP_INLINE_CROSS_FUN std::size_t operator()(uint8_t key) const {
#if CTP_IS_GPU
    return static_cast<std::size_t>(key);
#else
    return std::hash<uint8_t>{}(key);
#endif
  }
};

template <>
struct hash<uint16_t> {
  CTP_INLINE_CROSS_FUN std::size_t operator()(uint16_t key) const {
#if CTP_IS_GPU
    return static_cast<std::size_t>(key);
#else
    return std::hash<uint16_t>{}(key);
#endif
  }
};

template <>
struct hash<uint32_t> {
  CTP_INLINE_CROSS_FUN std::size_t operator()(uint32_t key) const {
#if CTP_IS_GPU
    return static_cast<std::size_t>(key);
#else
    return std::hash<uint32_t>{}(key);
#endif
  }
};

template <>
struct hash<uint64_t> {
  CTP_INLINE_CROSS_FUN std::size_t operator()(uint64_t key) const {
#if CTP_IS_GPU
    return static_cast<std::size_t>(key);
#else
    return std::hash<uint64_t>{}(key);
#endif
  }
};

template <>
struct hash<int8_t> {
  CTP_INLINE_CROSS_FUN std::size_t operator()(int8_t key) const {
#if CTP_IS_GPU
    return static_cast<std::size_t>(key);
#else
    return std::hash<int8_t>{}(key);
#endif
  }
};

template <>
struct hash<int16_t> {
  CTP_INLINE_CROSS_FUN std::size_t operator()(int16_t key) const {
#if CTP_IS_GPU
    return static_cast<std::size_t>(key);
#else
    return std::hash<int16_t>{}(key);
#endif
  }
};

template <>
struct hash<int32_t> {
  CTP_INLINE_CROSS_FUN std::size_t operator()(int32_t key) const {
#if CTP_IS_GPU
    return static_cast<std::size_t>(key);
#else
    return std::hash<int32_t>{}(key);
#endif
  }
};

template <>
struct hash<int64_t> {
  CTP_INLINE_CROSS_FUN std::size_t operator()(int64_t key) const {
#if CTP_IS_GPU
    return static_cast<std::size_t>(key);
#else
    return std::hash<int64_t>{}(key);
#endif
  }
};

/**
 * GPU-compatible equal_to functor
 */
template <typename T>
struct equal_to {
  CTP_INLINE_CROSS_FUN bool operator()(const T &a, const T &b) const {
    return a == b;
  }
};

}  // namespace ctp

#endif  // CTP_TYPES_HASH_H_
