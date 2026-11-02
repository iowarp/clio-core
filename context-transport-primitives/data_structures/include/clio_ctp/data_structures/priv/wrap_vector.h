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

#ifndef CTP_DATA_STRUCTURES_PRIV_WRAP_VECTOR_H_
#define CTP_DATA_STRUCTURES_PRIV_WRAP_VECTOR_H_

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/memory/allocator/allocator.h"

namespace ctp::priv {

/**
 * Non-owning char vector that wraps an existing FullPtr<char> buffer.
 * Provides the same API surface as array_vector for use with
 * LocalSerialize / LocalDeserialize, but does not own or allocate memory.
 *
 * Typical usage: wrap a pre-allocated device-memory buffer so that
 * ShmTransport::SendDevice / RecvDevice can serialize into it.
 */
class wrap_vector {
 public:
  ctp::ipc::FullPtr<char> data_;
  size_t size_ = 0;
  size_t capacity_ = 0;

  CTP_CROSS_FUN wrap_vector() = default;

  CTP_CROSS_FUN wrap_vector(ctp::ipc::FullPtr<char> data, size_t capacity)
      : data_(data), size_(0), capacity_(capacity) {}

  CTP_CROSS_FUN void set(ctp::ipc::FullPtr<char> data, size_t capacity) {
    data_ = data;
    size_ = 0;
    capacity_ = capacity;
  }

  CTP_CROSS_FUN char *data() { return data_.ptr_; }
  CTP_CROSS_FUN const char *data() const { return data_.ptr_; }
  CTP_CROSS_FUN size_t size() const { return size_; }
  CTP_CROSS_FUN size_t capacity() const { return capacity_; }
  CTP_CROSS_FUN bool empty() const { return size_ == 0; }

  CTP_CROSS_FUN bool reserve(size_t n) { return n <= capacity_; }
  CTP_CROSS_FUN bool resize(size_t n) { size_ = n; return n <= capacity_; }
  CTP_CROSS_FUN bool resize_no_init(size_t n) { size_ = n; return n <= capacity_; }
  CTP_CROSS_FUN void clear() { size_ = 0; }

  CTP_CROSS_FUN void push_back(char c) { data_.ptr_[size_++] = c; }
  CTP_CROSS_FUN char &operator[](size_t i) { return data_.ptr_[i]; }
  CTP_CROSS_FUN const char &operator[](size_t i) const { return data_.ptr_[i]; }

  CTP_CROSS_FUN char *begin() { return data_.ptr_; }
  CTP_CROSS_FUN char *end() { return data_.ptr_ + size_; }
  CTP_CROSS_FUN const char *begin() const { return data_.ptr_; }
  CTP_CROSS_FUN const char *end() const { return data_.ptr_ + size_; }

  CTP_CROSS_FUN ctp::ipc::FullPtr<char> &GetFullPtr() { return data_; }
  CTP_CROSS_FUN const ctp::ipc::FullPtr<char> &GetFullPtr() const { return data_; }

  /** Serialize (save) — write size + data bytes */
  template <class Archive>
  CTP_CROSS_FUN void save(Archive &ar) const {
    ar << size_;
    if (size_ > 0) {
      ar.write_binary(data_.ptr_, size_);
    }
  }

  /** Deserialize (load) — read size + data bytes into wrapped buffer */
  template <class Archive>
  CTP_CROSS_FUN void load(Archive &ar) {
    size_t sz = 0;
    ar >> sz;
    resize(sz);
    if (sz > 0) {
      ar.read_binary(data_.ptr_, sz);
    }
  }
};

}  // namespace ctp::priv

#endif  // CTP_DATA_STRUCTURES_PRIV_WRAP_VECTOR_H_
