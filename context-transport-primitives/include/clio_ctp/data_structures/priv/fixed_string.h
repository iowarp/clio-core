/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CTP_DATA_STRUCTURES_PRIV_FIXED_STRING_H_
#define CTP_DATA_STRUCTURES_PRIV_FIXED_STRING_H_

#include <cstring>
#include <type_traits>
#include <utility>

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/data_structures/priv/array_vector.h"
#include "clio_ctp/data_structures/serialization/serialize_common.h"

#if CTP_IS_HOST
#include <functional>
#include <string>
#include <string_view>
#endif

namespace ctp::priv {

/**
 * Fixed-capacity, inline (POD) string.
 *
 * Inherits the inline buffer + size_ from array_vector<N> and adds NO data
 * members of its own, so it is **bitwise-relocatable**: no heap, no SSO, no
 * self-referential pointer, and an identical layout on host and device. A raw
 * memcpy/cudaMemcpy of a struct embedding a fixed_string is correct with ZERO
 * post-copy fixup — which is exactly why GPU tasks can carry it without the
 * priv::string FixupSsoPointer dance. The copy/move/dtor are defaulted so the
 * type remains trivially copyable (the converting constructors below do not
 * affect that).
 *
 * The buffer is N bytes; the usable string length is at most N-1, since one
 * byte is reserved for a NUL terminator so c_str() is always valid. Inputs
 * longer than N-1 are truncated.
 *
 * @tparam N Total buffer capacity in bytes (default 32).
 */
template <size_t N = 32>
class fixed_string : public array_vector<N> {
 public:
  using Base = array_vector<N>;
  using Base::data_;
  using Base::size_;

  /** Max usable string length (one byte reserved for the NUL terminator). */
  static constexpr size_t kMaxLen = N - 1;

  /** strlen that does not depend on a device libc. */
  CTP_CROSS_FUN static size_t CStrLen(const char *s) {
    size_t n = 0;
    if (s != nullptr) {
      while (s[n] != '\0') {
        ++n;
      }
    }
    return n;
  }

  CTP_CROSS_FUN fixed_string() {
    size_ = 0;
    data_[0] = '\0';
  }

  // Trivial special members keep fixed_string bitwise-relocatable / POD-copyable.
  fixed_string(const fixed_string &) = default;
  fixed_string(fixed_string &&) = default;
  fixed_string &operator=(const fixed_string &) = default;
  fixed_string &operator=(fixed_string &&) = default;
  ~fixed_string() = default;

  // --- Converting constructors / assignment (do not affect trivial-copyable) ---
  CTP_CROSS_FUN fixed_string(const char *s) { assign(s, CStrLen(s)); }
  CTP_CROSS_FUN fixed_string(const char *s, size_t len) { assign(s, len); }
  CTP_CROSS_FUN fixed_string &operator=(const char *s) {
    assign(s, CStrLen(s));
    return *this;
  }

  /** Copy min(len, kMaxLen) bytes from s and NUL-terminate. */
  CTP_CROSS_FUN void assign(const char *s, size_t len) {
    if (len > kMaxLen) {
      len = kMaxLen;
    }
    if (s != nullptr && len > 0) {
      memcpy(data_, s, len);
    }
    size_ = len;
    data_[size_] = '\0';
  }

  // --- Accessors ---
  CTP_CROSS_FUN const char *c_str() const { return data_; }
  CTP_CROSS_FUN char *data() { return data_; }
  CTP_CROSS_FUN const char *data() const { return data_; }
  CTP_CROSS_FUN size_t size() const { return size_; }
  CTP_CROSS_FUN size_t length() const { return size_; }
  CTP_CROSS_FUN bool empty() const { return size_ == 0; }
  CTP_CROSS_FUN static constexpr size_t max_size() { return kMaxLen; }

  CTP_CROSS_FUN char &operator[](size_t i) { return data_[i]; }
  CTP_CROSS_FUN const char &operator[](size_t i) const { return data_[i]; }

  // --- Mutators that maintain the NUL terminator invariant ---
  CTP_CROSS_FUN void clear() {
    size_ = 0;
    data_[0] = '\0';
  }

  CTP_CROSS_FUN void push_back(char c) {
    if (size_ < kMaxLen) {
      data_[size_++] = c;
      data_[size_] = '\0';
    }
  }

  // --- Comparison ---
  CTP_CROSS_FUN bool operator==(const fixed_string &o) const {
    if (size_ != o.size_) {
      return false;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (data_[i] != o.data_[i]) {
        return false;
      }
    }
    return true;
  }
  CTP_CROSS_FUN bool operator!=(const fixed_string &o) const {
    return !(*this == o);
  }
  CTP_CROSS_FUN bool operator==(const char *s) const {
    size_t n = CStrLen(s);
    if (n != size_) {
      return false;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (data_[i] != s[i]) {
        return false;
      }
    }
    return true;
  }
  CTP_CROSS_FUN bool operator!=(const char *s) const { return !(*this == s); }

  // --- Serialization (size-prefixed, like priv::string; works with every
  //     archive via the has_load_save_cls dispatch). load NUL-terminates so
  //     c_str() stays valid after deserialization. ---
  template <typename Archive>
  CTP_INLINE_CROSS_FUN void save(Archive &ar) const {
    ctp::ipc::save_string(ar, *this);
  }
  template <typename Archive>
  CTP_INLINE_CROSS_FUN void load(Archive &ar) {
    ctp::ipc::load_string(ar, *this);
    if (size_ > kMaxLen) {
      size_ = kMaxLen;
    }
    data_[size_] = '\0';
  }

#if CTP_IS_HOST
  // --- Host-only std interop (device tasks use the const char* paths) ---
  fixed_string(const std::string &s) { assign(s.data(), s.size()); }
  fixed_string(std::string_view s) { assign(s.data(), s.size()); }
  fixed_string &operator=(const std::string &s) {
    assign(s.data(), s.size());
    return *this;
  }
  fixed_string &operator=(std::string_view s) {
    assign(s.data(), s.size());
    return *this;
  }
  std::string str() const { return std::string(data_, size_); }
  std::string_view view() const { return std::string_view(data_, size_); }
  operator std::string() const { return std::string(data_, size_); }
  bool operator==(const std::string &s) const {
    return view() == std::string_view(s);
  }
  bool operator!=(const std::string &s) const { return !(*this == s); }
#endif
};

}  // namespace ctp::priv

#if CTP_IS_HOST
namespace std {
template <size_t N>
struct hash<ctp::priv::fixed_string<N>> {
  size_t operator()(const ctp::priv::fixed_string<N> &s) const {
    return std::hash<std::string_view>()(s.view());
  }
};
}  // namespace std
#endif

#endif  // CTP_DATA_STRUCTURES_PRIV_FIXED_STRING_H_
