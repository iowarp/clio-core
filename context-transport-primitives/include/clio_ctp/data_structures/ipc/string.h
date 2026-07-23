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

#ifndef CTP_DATA_STRUCTURES_IPC_STRING_H_
#define CTP_DATA_STRUCTURES_IPC_STRING_H_

#include "clio_ctp/data_structures/ipc/shm_container.h"
#include "clio_ctp/constants/macros.h"

#if CTP_IS_HOST
#include <string>
#endif

namespace ctp::ipc {

/**
 * Shared-memory string (issue #783).
 *
 * Counterpart to ctp::priv::string, but every internal reference is a
 * SEGMENT-RELATIVE OFFSET rather than a raw pointer, so the same bytes are
 * readable from any process that maps the segment at any base address.
 *
 * SMALL-STRING OPTIMIZATION. Strings up to kSsoCapacity bytes live inline in
 * the object; longer ones spill to an allocation in the same segment. This is
 * not just an allocation-count win:
 *   - The metadata cache is dominated by short keys (tag and blob names), so
 *     most strings never touch the allocator at all -- meaningful when the
 *     whole point is a sub-5us read.
 *   - A lock-free reader that must follow an offset pointer into the arena is
 *     at its most exposed while doing so. Inline data removes the pointer
 *     chase entirely for the common case.
 *
 * NOT NULL-SAFE ACROSS ALLOCATORS. Like every ShmContainer, the allocator is
 * recorded as an offset from `this`, so a string MUST live in the same segment
 * as its allocator, and copying one requires naming the destination allocator
 * (see the (alloc, other) constructor). The implicit copy constructor is
 * therefore deleted -- a bitwise copy would carry an offset that is meaningless
 * at the destination address.
 *
 * SHM-SAFE: no virtual functions, no raw pointers, trivially relocatable.
 */
template <typename AllocT>
class string : public ShmContainer<AllocT> {
 public:
  using allocator_type = AllocT;

  /** Bytes storable inline (excluding the NUL terminator). */
  static constexpr size_t kSsoCapacity = 23;

 private:
  size_t size_;           /**< Length in bytes, excluding NUL */
  size_t capacity_;       /**< Usable capacity, excluding NUL */
  OffsetPtr<char> data_;  /**< Offset to heap buffer; null when inline */
  char sso_[kSsoCapacity + 1];  /**< Inline buffer, always NUL-terminated */

 public:
  /** Default constructor -- empty, no allocator, no allocation. */
  CTP_CROSS_FUN
  string()
      : ShmContainer<AllocT>(),
        size_(0),
        capacity_(kSsoCapacity),
        data_(OffsetPtr<char>::GetNull()) {
    sso_[0] = '\0';
  }

  /** Construct empty with an allocator. */
  CTP_INLINE_CROSS_FUN
  explicit string(AllocT *alloc)
      : ShmContainer<AllocT>(alloc),
        size_(0),
        capacity_(kSsoCapacity),
        data_(OffsetPtr<char>::GetNull()) {
    sso_[0] = '\0';
  }

  /** Construct from a C string. */
  CTP_INLINE_CROSS_FUN
  string(AllocT *alloc, const char *str)
      : ShmContainer<AllocT>(alloc),
        size_(0),
        capacity_(kSsoCapacity),
        data_(OffsetPtr<char>::GetNull()) {
    sso_[0] = '\0';
    assign(str, StrLen(str));
  }

  /** Construct from a buffer + length (may contain embedded NULs). */
  CTP_INLINE_CROSS_FUN
  string(AllocT *alloc, const char *str, size_t len)
      : ShmContainer<AllocT>(alloc),
        size_(0),
        capacity_(kSsoCapacity),
        data_(OffsetPtr<char>::GetNull()) {
    sso_[0] = '\0';
    assign(str, len);
  }

#if CTP_IS_HOST
  /** Construct from a std::string. */
  string(AllocT *alloc, const std::string &str)
      : ShmContainer<AllocT>(alloc),
        size_(0),
        capacity_(kSsoCapacity),
        data_(OffsetPtr<char>::GetNull()) {
    sso_[0] = '\0';
    assign(str.data(), str.size());
  }
#endif

  /**
   * Copy into a (possibly different) allocator. Required instead of a plain
   * copy constructor because ShmContainer stores the allocator as an offset
   * from `this`; a bitwise copy would inherit an offset that does not resolve
   * at the destination's address.
   */
  CTP_INLINE_CROSS_FUN
  string(AllocT *alloc, const string &other)
      : ShmContainer<AllocT>(alloc),
        size_(0),
        capacity_(kSsoCapacity),
        data_(OffsetPtr<char>::GetNull()) {
    sso_[0] = '\0';
    assign(other.c_str(), other.size());
  }

  /** Deleted: see the (alloc, other) constructor above. */
  string(const string &other) = delete;
  string &operator=(const string &other) = delete;

  CTP_INLINE_CROSS_FUN
  ~string() { FreeHeap(); }

  /** Replace contents. Returns false if a needed allocation failed, in which
   *  case the string is left EMPTY rather than partially written. */
  CTP_INLINE_CROSS_FUN
  bool assign(const char *str, size_t len) {
    if (!reserve(len)) {
      clear();
      return false;
    }
    char *dst = mutable_data();
    if (dst == nullptr) {
      clear();
      return false;
    }
    for (size_t i = 0; i < len; ++i) {
      dst[i] = str[i];
    }
    dst[len] = '\0';
    size_ = len;
    return true;
  }

  CTP_INLINE_CROSS_FUN
  bool assign(const char *str) { return assign(str, StrLen(str)); }

  /**
   * Ensure capacity for `n` bytes (excluding NUL).
   *
   * Growth is one-way: once spilled to the heap a string never shrinks back
   * inline, so a reader that has resolved the heap offset cannot have it
   * yanked back under it by a later shrink.
   */
  CTP_INLINE_CROSS_FUN
  bool reserve(size_t n) {
    if (n <= capacity_) {
      return true;
    }
    AllocT *alloc = this->GetAllocator();
    if (alloc == nullptr) {
      return false;
    }
    // Grow geometrically to avoid O(n^2) on repeated appends.
    size_t new_cap = capacity_ * 2;
    if (new_cap < n) {
      new_cap = n;
    }
    // Allocator exhaustion must be a RECOVERABLE condition here, not an
    // exception escaping into a caller. AllocateOffset throws OUT_OF_MEMORY
    // (arena_allocator.h) rather than returning null, and the whole
    // metadata-cache design rests on "cache full -> fall back to RPC": a
    // throw propagating out of a client read path, or out of a runtime task
    // that merely wanted to populate a cache entry, would turn a benign
    // capacity limit into a failure. So catch it and report false.
    OffsetPtr<> off = OffsetPtr<>::GetNull();
#if !CTP_IS_DEVICE_PASS
    try {
      off = alloc->AllocateOffset(new_cap + 1);
    } catch (...) {
      return false;
    }
#else
    // Device passes compile CTP_THROW to a no-op, so the null check below is
    // the only signal available there.
    off = alloc->AllocateOffset(new_cap + 1);
#endif
    if (off.IsNull()) {
      return false;
    }
    char *new_buf = FullPtr<char>(alloc, OffsetPtr<char>(off.load())).ptr_;
    const char *old = c_str();
    for (size_t i = 0; i < size_; ++i) {
      new_buf[i] = old[i];
    }
    new_buf[size_] = '\0';
    FreeHeap();
    data_ = OffsetPtr<char>(off.load());
    capacity_ = new_cap;
    return true;
  }

  CTP_INLINE_CROSS_FUN
  void clear() {
    size_ = 0;
    char *d = mutable_data();
    if (d != nullptr) {
      d[0] = '\0';
    }
  }

  /** Read-only pointer to the bytes. Always NUL-terminated. */
  CTP_INLINE_CROSS_FUN
  const char *c_str() const {
    if (data_.IsNull()) {
      return sso_;
    }
    AllocT *alloc = this->GetAllocator();
    if (alloc == nullptr) {
      return sso_;
    }
    return FullPtr<char>(alloc, data_).ptr_;
  }

  CTP_INLINE_CROSS_FUN
  const char *data() const { return c_str(); }

  CTP_INLINE_CROSS_FUN
  size_t size() const { return size_; }

  CTP_INLINE_CROSS_FUN
  size_t length() const { return size_; }

  CTP_INLINE_CROSS_FUN
  size_t capacity() const { return capacity_; }

  CTP_INLINE_CROSS_FUN
  bool empty() const { return size_ == 0; }

  /** True if the bytes live inline (no pointer chase for readers). */
  CTP_INLINE_CROSS_FUN
  bool is_inline() const { return data_.IsNull(); }

  CTP_INLINE_CROSS_FUN
  char operator[](size_t i) const { return c_str()[i]; }

  CTP_INLINE_CROSS_FUN
  bool operator==(const string &other) const {
    return Equals(other.c_str(), other.size());
  }

  CTP_INLINE_CROSS_FUN
  bool operator!=(const string &other) const { return !(*this == other); }

  CTP_INLINE_CROSS_FUN
  bool operator==(const char *str) const { return Equals(str, StrLen(str)); }

  CTP_INLINE_CROSS_FUN
  bool operator!=(const char *str) const { return !(*this == str); }

#if CTP_IS_HOST
  bool operator==(const std::string &str) const {
    return Equals(str.data(), str.size());
  }
  bool operator!=(const std::string &str) const { return !(*this == str); }

  /** Materialize as a std::string (copies out of shared memory). */
  std::string str() const { return std::string(c_str(), size_); }
#endif

  /** FNV-1a over the bytes. Defined here so the hash is identical in every
   *  process regardless of std::hash's implementation -- a std::hash-based
   *  bucket index would not be stable across processes or libstdc++ versions,
   *  which would corrupt a shared hash table. */
  CTP_INLINE_CROSS_FUN
  size_t hash() const { return HashBytes(c_str(), size_); }

  /** FNV-1a, exposed so callers can hash a raw key without building a
   *  string (e.g. probing the map with a const char*). */
  CTP_INLINE_CROSS_FUN
  static size_t HashBytes(const char *p, size_t n) {
    size_t h = 1469598103934665603ULL;  // FNV offset basis
    for (size_t i = 0; i < n; ++i) {
      h ^= static_cast<size_t>(static_cast<unsigned char>(p[i]));
      h *= 1099511628211ULL;  // FNV prime
    }
    return h;
  }

  /**
   * Compare against raw bytes without constructing a string.
   *
   * Required by the SHM hash map's client-side lookup: clients must never
   * allocate inside the metadata segment (they are readers, and allocating
   * would mutate runtime-owned structures), so probing the table with a
   * temporary ipc::string key is not an option.
   */
  CTP_INLINE_CROSS_FUN
  bool EqualsBytes(const char *p, size_t n) const { return Equals(p, n); }

  CTP_INLINE_CROSS_FUN
  static size_t StrLen(const char *s) {
    size_t n = 0;
    if (s == nullptr) {
      return 0;
    }
    while (s[n] != '\0') {
      ++n;
    }
    return n;
  }

 private:
  CTP_INLINE_CROSS_FUN
  bool Equals(const char *p, size_t n) const {
    if (n != size_) {
      return false;
    }
    const char *self = c_str();
    for (size_t i = 0; i < n; ++i) {
      if (self[i] != p[i]) {
        return false;
      }
    }
    return true;
  }

  CTP_INLINE_CROSS_FUN
  char *mutable_data() {
    if (data_.IsNull()) {
      return sso_;
    }
    AllocT *alloc = this->GetAllocator();
    if (alloc == nullptr) {
      return nullptr;
    }
    return FullPtr<char>(alloc, data_).ptr_;
  }

  CTP_INLINE_CROSS_FUN
  void FreeHeap() {
    if (data_.IsNull()) {
      return;
    }
    AllocT *alloc = this->GetAllocator();
    if (alloc != nullptr) {
      alloc->FreeOffsetNoNullCheck(OffsetPtr<>(data_.load()));
    }
    data_ = OffsetPtr<char>::GetNull();
  }
};

}  // namespace ctp::ipc

#endif  // CTP_DATA_STRUCTURES_IPC_STRING_H_
