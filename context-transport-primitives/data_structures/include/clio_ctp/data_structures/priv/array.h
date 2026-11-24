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

#ifndef CTP_DATA_STRUCTURES_PRIV_ARRAY_H_
#define CTP_DATA_STRUCTURES_PRIV_ARRAY_H_

#include "clio_ctp/constants/macros.h"
#include <cstring>

namespace ctp::ipc {

/**
 * Fixed-size stack-allocated array.
 *
 * No allocator, no heap. Compatible with LocalSerialize/LocalDeserialize
 * as a drop-in replacement for vector<T, AllocT> when the maximum size
 * is known at compile time.
 *
 * @tparam T Element type
 * @tparam N Maximum number of elements
 */
template <typename T, size_t N>
class array {
 private:
  T data_[N];
  size_t size_;

 public:
  CTP_INLINE_CROSS_FUN array() : size_(0) {}

  CTP_INLINE_CROSS_FUN T *data() { return data_; }
  CTP_INLINE_CROSS_FUN const T *data() const { return data_; }
  CTP_INLINE_CROSS_FUN size_t size() const { return size_; }
  CTP_INLINE_CROSS_FUN static constexpr size_t capacity() { return N; }

  CTP_INLINE_CROSS_FUN void resize(size_t new_size) { size_ = new_size; }
  CTP_INLINE_CROSS_FUN void reserve(size_t) {}

  CTP_INLINE_CROSS_FUN T &operator[](size_t i) { return data_[i]; }
  CTP_INLINE_CROSS_FUN const T &operator[](size_t i) const { return data_[i]; }
};

}  // namespace ctp::ipc

#endif  // CTP_DATA_STRUCTURES_PRIV_ARRAY_H_
