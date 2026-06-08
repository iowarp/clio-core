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

#ifndef CTP_DATA_STRUCTURES_PRIV_STRING_H_
#define CTP_DATA_STRUCTURES_PRIV_STRING_H_

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/types/numbers.h"
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/data_structures/serialization/serialize_common.h"
#include "vector.h"
#include <cstring>
#include <iterator>
#include <type_traits>
#include <stdexcept>
#include <algorithm>

namespace ctp::priv {

/** GPU-compatible ctp_memcmp: character-by-character on GPU, memcmp on CPU */
CTP_INLINE_CROSS_FUN int ctp_memcmp(const void *a, const void *b, size_t n) {
#if CTP_IS_HOST
  return ::memcmp(a, b, n);
#else
  const unsigned char *pa = static_cast<const unsigned char*>(a);
  const unsigned char *pb = static_cast<const unsigned char*>(b);
  for (size_t i = 0; i < n; ++i) {
    if (pa[i] != pb[i]) return (pa[i] < pb[i]) ? -1 : 1;
  }
  return 0;
#endif
}

/**
 * Private-memory string container with Short String Optimization (SSO)
 *
 * This string class provides std::string-like functionality for private memory,
 * using the library's allocator API with FullPtr for proper memory management.
 * It implements Short String Optimization (SSO) to avoid allocation for small strings.
 * The string uses ctp::priv::vector internally for large string storage,
 * minimizing code duplication.
 *
 * @tparam T The character type (default: char)
 * @tparam AllocT The allocator type (must have Allocate/AllocateObjs/Free methods)
 * @tparam SSOSize The SSO buffer size in bytes (default: 32)
 */
template<typename T, typename AllocT, size_t SSOSize = 32>
class basic_string {
 public:
  using allocator_type = AllocT;
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  static constexpr size_t npos = static_cast<size_t>(-1);

  /**
   * Random access iterator for string
   */
  class iterator {
   private:
    T *ptr_;  /**< Current character pointer */

   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using reference = T&;

    /**
     * Default constructor
     */
    CTP_INLINE_CROSS_FUN
    iterator() : ptr_(nullptr) {}

    /**
     * Construct from pointer
     *
     * @param ptr The character pointer
     */
    CTP_INLINE_CROSS_FUN
    explicit iterator(T *ptr) : ptr_(ptr) {}

    /**
     * Dereference operator
     *
     * @return Reference to the current character
     */
    CTP_INLINE_CROSS_FUN
    T& operator*() const { return *ptr_; }

    /**
     * Arrow operator
     *
     * @return Pointer to the current character
     */
    CTP_INLINE_CROSS_FUN
    T* operator->() const { return ptr_; }

    /**
     * Pre-increment operator
     *
     * @return Reference to this iterator
     */
    CTP_INLINE_CROSS_FUN
    iterator& operator++() {
      ++ptr_;
      return *this;
    }

    /**
     * Post-increment operator
     *
     * @return Copy of this iterator before incrementing
     */
    CTP_INLINE_CROSS_FUN
    iterator operator++(int) {
      iterator temp = *this;
      ++ptr_;
      return temp;
    }

    /**
     * Pre-decrement operator
     *
     * @return Reference to this iterator
     */
    CTP_INLINE_CROSS_FUN
    iterator& operator--() {
      --ptr_;
      return *this;
    }

    /**
     * Post-decrement operator
     *
     * @return Copy of this iterator before decrementing
     */
    CTP_INLINE_CROSS_FUN
    iterator operator--(int) {
      iterator temp = *this;
      --ptr_;
      return temp;
    }

    /**
     * Addition operator
     *
     * @param n Number of characters to advance
     * @return New iterator advanced by n positions
     */
    CTP_INLINE_CROSS_FUN
    iterator operator+(difference_type n) const {
      return iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n Number of characters to go back
     * @return New iterator moved back by n positions
     */
    CTP_INLINE_CROSS_FUN
    iterator operator-(difference_type n) const {
      return iterator(ptr_ - n);
    }

    /**
     * Addition assignment operator
     *
     * @param n Number of characters to advance
     * @return Reference to this iterator
     */
    CTP_INLINE_CROSS_FUN
    iterator& operator+=(difference_type n) {
      ptr_ += n;
      return *this;
    }

    /**
     * Subtraction assignment operator
     *
     * @param n Number of characters to go back
     * @return Reference to this iterator
     */
    CTP_INLINE_CROSS_FUN
    iterator& operator-=(difference_type n) {
      ptr_ -= n;
      return *this;
    }

    /**
     * Difference operator
     *
     * @param other Another iterator
     * @return The number of characters between this and other
     */
    CTP_INLINE_CROSS_FUN
    difference_type operator-(const iterator& other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n Index offset from current position
     * @return Reference to the character at offset n
     */
    CTP_INLINE_CROSS_FUN
    T& operator[](difference_type n) const {
      return ptr_[n];
    }

    /**
     * Equality comparison operator
     *
     * @param other Another iterator
     * @return True if both iterators point to the same character
     */
    CTP_INLINE_CROSS_FUN
    bool operator==(const iterator& other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality comparison operator
     *
     * @param other Another iterator
     * @return True if iterators point to different characters
     */
    CTP_INLINE_CROSS_FUN
    bool operator!=(const iterator& other) const {
      return ptr_ != other.ptr_;
    }

    /**
     * Less than comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes before other
     */
    CTP_INLINE_CROSS_FUN
    bool operator<(const iterator& other) const {
      return ptr_ < other.ptr_;
    }

    /**
     * Less than or equal comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes before or equals other
     */
    CTP_INLINE_CROSS_FUN
    bool operator<=(const iterator& other) const {
      return ptr_ <= other.ptr_;
    }

    /**
     * Greater than comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes after other
     */
    CTP_INLINE_CROSS_FUN
    bool operator>(const iterator& other) const {
      return ptr_ > other.ptr_;
    }

    /**
     * Greater than or equal comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes after or equals other
     */
    CTP_INLINE_CROSS_FUN
    bool operator>=(const iterator& other) const {
      return ptr_ >= other.ptr_;
    }

    /**
     * Get the raw pointer
     *
     * @return The underlying pointer
     */
    CTP_INLINE_CROSS_FUN
    T* get() const { return ptr_; }
  };

  /**
   * Const iterator for string
   */
  class const_iterator {
   private:
    const T *ptr_;  /**< Current character pointer */

   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;

    /**
     * Default constructor
     */
    CTP_INLINE_CROSS_FUN
    const_iterator() : ptr_(nullptr) {}

    /**
     * Construct from pointer
     *
     * @param ptr The character pointer
     */
    CTP_INLINE_CROSS_FUN
    explicit const_iterator(const T *ptr) : ptr_(ptr) {}

    /**
     * Construct from non-const iterator
     *
     * @param it The iterator to convert
     */
    CTP_INLINE_CROSS_FUN
    const_iterator(const iterator& it) : ptr_(it.get()) {}

    /**
     * Dereference operator
     *
     * @return Const reference to the current character
     */
    CTP_INLINE_CROSS_FUN
    const T& operator*() const { return *ptr_; }

    /**
     * Arrow operator
     *
     * @return Const pointer to the current character
     */
    CTP_INLINE_CROSS_FUN
    const T* operator->() const { return ptr_; }

    /**
     * Pre-increment operator
     *
     * @return Reference to this iterator
     */
    CTP_INLINE_CROSS_FUN
    const_iterator& operator++() {
      ++ptr_;
      return *this;
    }

    /**
     * Post-increment operator
     *
     * @return Copy of this iterator before incrementing
     */
    CTP_INLINE_CROSS_FUN
    const_iterator operator++(int) {
      const_iterator temp = *this;
      ++ptr_;
      return temp;
    }

    /**
     * Pre-decrement operator
     *
     * @return Reference to this iterator
     */
    CTP_INLINE_CROSS_FUN
    const_iterator& operator--() {
      --ptr_;
      return *this;
    }

    /**
     * Post-decrement operator
     *
     * @return Copy of this iterator before decrementing
     */
    CTP_INLINE_CROSS_FUN
    const_iterator operator--(int) {
      const_iterator temp = *this;
      --ptr_;
      return temp;
    }

    /**
     * Addition operator
     *
     * @param n Number of characters to advance
     * @return New iterator advanced by n positions
     */
    CTP_INLINE_CROSS_FUN
    const_iterator operator+(difference_type n) const {
      return const_iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n Number of characters to go back
     * @return New iterator moved back by n positions
     */
    CTP_INLINE_CROSS_FUN
    const_iterator operator-(difference_type n) const {
      return const_iterator(ptr_ - n);
    }

    /**
     * Addition assignment operator
     *
     * @param n Number of characters to advance
     * @return Reference to this iterator
     */
    CTP_INLINE_CROSS_FUN
    const_iterator& operator+=(difference_type n) {
      ptr_ += n;
      return *this;
    }

    /**
     * Subtraction assignment operator
     *
     * @param n Number of characters to go back
     * @return Reference to this iterator
     */
    CTP_INLINE_CROSS_FUN
    const_iterator& operator-=(difference_type n) {
      ptr_ -= n;
      return *this;
    }

    /**
     * Difference operator
     *
     * @param other Another iterator
     * @return The number of characters between this and other
     */
    CTP_INLINE_CROSS_FUN
    difference_type operator-(const const_iterator& other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n Index offset from current position
     * @return Const reference to the character at offset n
     */
    CTP_INLINE_CROSS_FUN
    const T& operator[](difference_type n) const {
      return ptr_[n];
    }

    /**
     * Equality comparison operator
     *
     * @param other Another iterator
     * @return True if both iterators point to the same character
     */
    CTP_INLINE_CROSS_FUN
    bool operator==(const const_iterator& other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality comparison operator
     *
     * @param other Another iterator
     * @return True if iterators point to different characters
     */
    CTP_INLINE_CROSS_FUN
    bool operator!=(const const_iterator& other) const {
      return ptr_ != other.ptr_;
    }

    /**
     * Less than comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes before other
     */
    CTP_INLINE_CROSS_FUN
    bool operator<(const const_iterator& other) const {
      return ptr_ < other.ptr_;
    }

    /**
     * Less than or equal comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes before or equals other
     */
    CTP_INLINE_CROSS_FUN
    bool operator<=(const const_iterator& other) const {
      return ptr_ <= other.ptr_;
    }

    /**
     * Greater than comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes after other
     */
    CTP_INLINE_CROSS_FUN
    bool operator>(const const_iterator& other) const {
      return ptr_ > other.ptr_;
    }

    /**
     * Greater than or equal comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes after or equals other
     */
    CTP_INLINE_CROSS_FUN
    bool operator>=(const const_iterator& other) const {
      return ptr_ >= other.ptr_;
    }

    /**
     * Get the raw pointer
     *
     * @return The underlying pointer
     */
    CTP_INLINE_CROSS_FUN
    const T* get() const { return ptr_; }
  };

  /**
   * Reverse iterator for string
   */
  using reverse_iterator = std::reverse_iterator<iterator>;

  /**
   * Const reverse iterator for string
   */
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  /**
   * Heap metadata stored in the union when not using SSO.
   * Replaces the vector pointer to avoid GPU device malloc.
   */
  struct HeapInfo {
    size_type capacity_;  /**< Allocated capacity (in elements, including null) */
    size_type off_;       /**< Allocator offset for freeing */
  };

  /**
   * SSO storage union containing either inline buffer or heap metadata
   */
  union SsoStorage {
    T buffer_[SSOSize];  /**< Inline SSO buffer */
    HeapInfo heap_;       /**< Heap metadata for large strings */
  };

  SsoStorage storage_;  /**< Storage union for SSO */
  size_type size_;      /**< Current string length (not including null terminator) */
  bool using_sso_;      /**< Flag to track if using SSO or vector */
  AllocT* alloc_;       /**< Pointer to allocator for memory management */
  T* data_;             /**< Cached data pointer (avoids branch + pointer chase) */

 public:
  /**
   * Check if current string is using SSO buffer
   *
   * @return True if string data is in SSO buffer
   */
  CTP_INLINE_CROSS_FUN
  bool UsingSso() const {
    return using_sso_;
  }

  /**
   * Fix up the data_ pointer after a bulk memcpy.
   * After memcpy, data_ still points to the source object's buffer.
   * This repoints it to this object's own SSO buffer.
   * Only valid when UsingSso() is true.
   */
  CTP_INLINE_CROSS_FUN
  void FixupSsoPointer() {
    data_ = storage_.buffer_;
  }

  /**
   * Initialize this string by copying SSO data from another string.
   * No allocator needed — copies the inline buffer directly.
   * Only valid when the source is using SSO (size <= SSOSize).
   */
  CTP_INLINE_CROSS_FUN
  void InitFromSso(const basic_string &other) {
    size_ = other.size_;
    using_sso_ = true;
    alloc_ = nullptr;
    for (size_type i = 0; i <= size_ && i < SSOSize; ++i) {
      storage_.buffer_[i] = other.storage_.buffer_[i];
    }
    data_ = storage_.buffer_;
  }

  /**
   * POD-safe state for SSO strings (no pointers).
   * Serialized as a flat memcpy-able struct via ar.range().
   */
  struct SsoState {
    T buffer_[SSOSize];
    size_type size_;
  };

  /** Export SSO state for bulk serialization. */
  CTP_INLINE_CROSS_FUN SsoState GetSsoState() const {
    SsoState s;
    memcpy(s.buffer_, storage_.buffer_, SSOSize * sizeof(T));
    s.size_ = size_;
    return s;
  }

  /** Import SSO state after bulk deserialization. */
  CTP_INLINE_CROSS_FUN void SetSsoState(const SsoState &s) {
    memcpy(storage_.buffer_, s.buffer_, SSOSize * sizeof(T));
    size_ = s.size_;
    using_sso_ = true;
    alloc_ = nullptr;
    data_ = storage_.buffer_;
  }

 private:

  /**
   * Get pointer to current string data (SSO or heap)
   *
   * @return Pointer to string data
   */
  CTP_INLINE_CROSS_FUN
  T* GetData() {
    return data_;
  }

  /**
   * Get const pointer to current string data (SSO or heap)
   *
   * @return Const pointer to string data
   */
  CTP_INLINE_CROSS_FUN
  const T* GetData() const {
    return data_;
  }

  /**
   * Allocate a heap buffer of the given capacity (in elements) from the
   * custom allocator. Sets data_, storage_.heap_.capacity_, and
   * storage_.heap_.off_. Does NOT set using_sso_ or size_.
   */
  CTP_CROSS_FUN
  void HeapAlloc(size_type capacity) {
    auto p = alloc_->template AllocateObjs<T>(capacity);
    data_ = p.ptr_;
    storage_.heap_.capacity_ = capacity;
    storage_.heap_.off_ = p.shm_.off_.load();
  }

  /**
   * Build a FullPtr from the current heap state for freeing.
   */
  CTP_INLINE_CROSS_FUN
  ctp::ipc::FullPtr<T> HeapFullPtr() const {
    ctp::ipc::FullPtr<T> p;
    p.ptr_ = data_;
    p.shm_.off_ = storage_.heap_.off_;
    return p;
  }

  /**
   * Free the current heap buffer via the custom allocator.
   * Caller must ensure using_sso_ == false.
   */
  CTP_CROSS_FUN
  void HeapFree() {
    auto p = HeapFullPtr();
    alloc_->Free(p);
  }

  /**
   * Ensure the heap buffer has at least new_cap elements of capacity.
   * If already large enough, does nothing. Otherwise reallocates and
   * copies existing data.
   */
  CTP_CROSS_FUN
  void HeapReserve(size_type new_cap) {
    if (new_cap <= storage_.heap_.capacity_) {
      return;
    }
    // Grow by at least 2x to amortize
    size_type grow_cap = storage_.heap_.capacity_ * 2;
    if (new_cap > grow_cap) {
      grow_cap = new_cap;
    }
    auto old_p = HeapFullPtr();
    HeapAlloc(grow_cap);
    memcpy(data_, old_p.ptr_, size_ * sizeof(T));
    alloc_->Free(old_p);
  }

  /**
   * Transition from SSO to heap, copying existing SSO data and
   * allocating for total_capacity elements.
   */
  CTP_CROSS_FUN
  void TransitionToHeap(size_type total_capacity) {
    T sso_copy[SSOSize];
    memcpy(sso_copy, storage_.buffer_, size_ * sizeof(T));
    HeapAlloc(total_capacity);
    memcpy(data_, sso_copy, size_ * sizeof(T));
    using_sso_ = false;
  }

  /**
   * Append from C-style string (bulk memcpy path).
   */
  void AppendCStr(const T* s, size_type len) {
    if (len == 0) {
      return;
    }

    if (UsingSso() && size_ + len <= SSOSize - 1) {
      // Still fits in SSO
      memcpy(&storage_.buffer_[size_], s, len * sizeof(T));
      size_ += len;
      storage_.buffer_[size_] = T();
    } else {
      if (UsingSso()) {
        TransitionToHeap(size_ + len + 1);
      } else {
        HeapReserve(size_ + len + 1);
      }
      memcpy(data_ + size_, s, len * sizeof(T));
      size_ += len;
    }
  }

 public:
  /**
   * Default constructor (no allocator).
   * Creates an empty SSO string. Allocator is set via copy/move assignment.
   */
  CTP_CROSS_FUN basic_string()
    : size_(0), using_sso_(true), alloc_(nullptr), data_(storage_.buffer_) {
    storage_.buffer_[0] = T();
  }

  /**
   * Constructor with allocator.
   * Creates an empty string with SSO buffer initialized.
   *
   * @param alloc Pointer to allocator instance for memory management
   */
  CTP_CROSS_FUN explicit basic_string(AllocT* alloc)
    : size_(0), using_sso_(true), alloc_(alloc), data_(storage_.buffer_) {
    storage_.buffer_[0] = T();
  }

  /**
   * Destructor.
   * Frees heap memory if not using SSO. Does not deallocate the allocator.
   */
  CTP_CROSS_FUN
  ~basic_string() {
    if (!UsingSso()) {
      HeapFree();
    }
  }

  /**
   * Constructor from C-style string (allocator first - preferred).
   * Creates a string from a null-terminated C-style string.
   *
   * @param alloc Pointer to allocator instance
   * @param s The C-style string (null-terminated)
   */
  CTP_CROSS_FUN basic_string(AllocT* alloc, const T* s)
    : size_(0), using_sso_(true), alloc_(alloc), data_(storage_.buffer_) {
    if (s != nullptr) {
      size_type len = 0;
      while (s[len] != T()) ++len;
      if (len < SSOSize - 1) {
        memcpy(storage_.buffer_, s, len * sizeof(T));
        storage_.buffer_[len] = T();
        size_ = len;
      } else {
        HeapAlloc(len + 1);
        memcpy(data_, s, len * sizeof(T));
        size_ = len;
        using_sso_ = false;
      }
    } else {
      storage_.buffer_[0] = T();
    }
  }

  /**
   * Constructor from C-style string (legacy order - deprecated).
   * Creates a string from a null-terminated C-style string.
   *
   * @param s The C-style string (null-terminated)
   * @param alloc Pointer to allocator instance
   */
  basic_string(const T* s, AllocT* alloc)
    : basic_string(alloc, s) {}

  /**
   * Constructor from character count.
   * Creates a string with count copies of character c.
   *
   * @param count Number of characters
   * @param c The character to repeat
   * @param alloc Pointer to allocator instance
   */
  basic_string(size_type count, T c, AllocT* alloc)
    : size_(0), using_sso_(true), alloc_(alloc), data_(storage_.buffer_) {
    if (count < SSOSize - 1) {
      memset(storage_.buffer_, c, count * sizeof(T));
      storage_.buffer_[count] = T();
      size_ = count;
    } else {
      HeapAlloc(count + 1);
      memset(data_, c, count * sizeof(T));
      size_ = count;
      using_sso_ = false;
    }
  }

  /**
   * Constructor from substring.
   * Creates a string from a substring of another string.
   *
   * @param other String to copy from
   * @param pos Starting position
   * @param count Number of characters (or npos for remainder)
   * @param alloc Pointer to allocator instance
   */
  basic_string(const basic_string& other, size_type pos, size_type count,
               AllocT* alloc)
    : size_(0), using_sso_(true), alloc_(alloc), data_(storage_.buffer_) {
    if (pos > other.size_) {
      CTP_THROW(std::out_of_range("Substring position out of range"));
    }
    size_type copy_count = (count == npos) ? (other.size_ - pos) : count;
    copy_count = std::min(copy_count, other.size_ - pos);

    const T* other_data = other.GetData();
    if (copy_count < SSOSize - 1) {
      memcpy(storage_.buffer_, &other_data[pos], copy_count * sizeof(T));
      storage_.buffer_[copy_count] = T();
      size_ = copy_count;
    } else {
      HeapAlloc(copy_count + 1);
      memcpy(data_, &other_data[pos], copy_count * sizeof(T));
      size_ = copy_count;
      using_sso_ = false;
    }
  }

  /**
   * Copy constructor.
   * Creates a copy of another string.
   *
   * @param other String to copy from
   */
  CTP_CROSS_FUN basic_string(const basic_string& other)
    : size_(0), using_sso_(true), alloc_(other.alloc_), data_(storage_.buffer_) {
    const T* other_data = other.GetData();
    if (other.size_ < SSOSize - 1) {
      memcpy(storage_.buffer_, other_data, other.size_ * sizeof(T));
      storage_.buffer_[other.size_] = T();
      size_ = other.size_;
    } else {
#if CTP_IS_HOST
      HeapAlloc(other.size_ + 1);
      memcpy(data_, other_data, other.size_ * sizeof(T));
      size_ = other.size_;
      using_sso_ = false;
#else
      // GPU: only SSO strings (< SSOSize chars) supported for copy
      // Long strings are truncated to SSO capacity
      size_type copy_count = SSOSize - 1;
      memcpy(storage_.buffer_, other_data, copy_count * sizeof(T));
      storage_.buffer_[copy_count] = T();
      size_ = copy_count;
#endif
    }
  }

  /**
   * Move constructor.
   * Transfers ownership of data from other string to this one.
   *
   * @param other String to move from
   */
  CTP_CROSS_FUN basic_string(basic_string&& other) noexcept
    : size_(other.size_), using_sso_(other.using_sso_), alloc_(other.alloc_) {
    if (other.UsingSso()) {
      memcpy(storage_.buffer_, other.storage_.buffer_,
                  other.size_ * sizeof(T));
      storage_.buffer_[other.size_] = T();
      data_ = storage_.buffer_;
    } else {
      storage_.heap_ = other.storage_.heap_;
      data_ = other.data_;
    }
    other.size_ = 0;
    other.using_sso_ = true;
    other.data_ = other.storage_.buffer_;
    other.alloc_ = nullptr;
  }

  /**
   * Constructor from initializer list.
   * Creates a string from an initializer list of characters.
   *
   * @param init Initializer list of characters
   * @param alloc Pointer to allocator instance
   */
  basic_string(std::initializer_list<T> init, AllocT* alloc)
    : size_(0), using_sso_(true), alloc_(alloc), data_(storage_.buffer_) {
    if (init.size() < SSOSize - 1) {
      size_type i = 0;
      for (const T& c : init) {
        storage_.buffer_[i++] = c;
      }
      storage_.buffer_[init.size()] = T();
      size_ = init.size();
    } else {
      HeapAlloc(init.size() + 1);
      size_type i = 0;
      for (const T& c : init) {
        data_[i++] = c;
      }
      size_ = init.size();
      using_sso_ = false;
    }
  }

  /**
   * Constructor from std::basic_string.
   * Creates a string by copying the contents of a std::basic_string.
   * Parameters are ordered (allocator, string) to match common initialization patterns.
   *
   * @param alloc Pointer to allocator instance
   * @param str The std::basic_string to copy from
   */
  template<typename U>
  basic_string(AllocT* alloc, const std::basic_string<T, U>& str)
    : size_(0), using_sso_(true), alloc_(alloc), data_(storage_.buffer_) {
    size_type len = str.size();
    if (len < SSOSize - 1) {
      memcpy(storage_.buffer_, str.data(), len * sizeof(T));
      storage_.buffer_[len] = T();
      size_ = len;
    } else {
      HeapAlloc(len + 1);
      memcpy(data_, str.data(), len * sizeof(T));
      size_ = len;
      using_sso_ = false;
    }
  }

  /**
   * Copy assignment operator.
   * Replaces this string's contents with a copy of other's contents.
   *
   * @param other String to copy from
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& operator=(const basic_string& other) {
    if (this != &other) {
      if (!UsingSso()) {
        HeapFree();
      }
      size_ = 0;
      using_sso_ = true;
      alloc_ = other.alloc_;

      const T* other_data = other.GetData();
      if (other.size_ < SSOSize - 1) {
        memcpy(storage_.buffer_, other_data, other.size_ * sizeof(T));
        storage_.buffer_[other.size_] = T();
        size_ = other.size_;
        data_ = storage_.buffer_;
      } else {
        HeapAlloc(other.size_ + 1);
        memcpy(data_, other_data, other.size_ * sizeof(T));
        size_ = other.size_;
        using_sso_ = false;
      }
    }
    return *this;
  }

  /**
   * Move assignment operator.
   * Transfers ownership of data from other string to this one.
   *
   * @param other String to move from
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& operator=(basic_string&& other) noexcept {
    if (this != &other) {
      if (!UsingSso()) {
        HeapFree();
      }
      size_ = other.size_;
      using_sso_ = other.using_sso_;
      alloc_ = other.alloc_;

      if (other.UsingSso()) {
        memcpy(storage_.buffer_, other.storage_.buffer_,
                    other.size_ * sizeof(T));
        storage_.buffer_[other.size_] = T();
        data_ = storage_.buffer_;
      } else {
        storage_.heap_ = other.storage_.heap_;
        data_ = other.data_;
      }

      other.size_ = 0;
      other.using_sso_ = true;
      other.data_ = other.storage_.buffer_;
      other.alloc_ = nullptr;
    }
    return *this;
  }

  /**
   * Assignment operator from C-style string.
   *
   * @param s Null-terminated C-style string
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& operator=(const T* s) {
    if (s == nullptr) {
      clear();
      return *this;
    }

    if (!UsingSso()) {
      HeapFree();
    }
    size_ = 0;
    using_sso_ = true;

    size_type len = 0;
    while (s[len] != T()) ++len;

    if (len < SSOSize - 1) {
      memcpy(storage_.buffer_, s, len * sizeof(T));
      storage_.buffer_[len] = T();
      size_ = len;
      data_ = storage_.buffer_;
    } else {
      HeapAlloc(len + 1);
      memcpy(data_, s, len * sizeof(T));
      size_ = len;
      using_sso_ = false;
    }
    return *this;
  }

  /**
   * Assignment operator from initializer list.
   *
   * @param init Initializer list of characters
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& operator=(std::initializer_list<T> init) {
    if (!UsingSso()) {
      HeapFree();
    }
    size_ = 0;
    using_sso_ = true;

    if (init.size() < SSOSize - 1) {
      size_type i = 0;
      for (const T& c : init) {
        storage_.buffer_[i++] = c;
      }
      storage_.buffer_[init.size()] = T();
      size_ = init.size();
      data_ = storage_.buffer_;
    } else {
      HeapAlloc(init.size() + 1);
      size_type i = 0;
      for (const T& c : init) {
        data_[i++] = c;
      }
      size_ = init.size();
      using_sso_ = false;
    }
    return *this;
  }

  /**
   * Assignment operator from std::basic_string.
   * Assigns the contents of a std::basic_string to this string.
   *
   * @param str The std::basic_string to assign from
   * @return Reference to this string
   */
  template<typename U>
  CTP_CROSS_FUN
  basic_string& operator=(const std::basic_string<T, U>& str) {
    if (!UsingSso()) {
      HeapFree();
    }
    size_ = 0;
    using_sso_ = true;

    size_type len = str.size();
    if (len < SSOSize - 1) {
      memcpy(storage_.buffer_, str.data(), len * sizeof(T));
      storage_.buffer_[len] = T();
      size_ = len;
      data_ = storage_.buffer_;
    } else {
      HeapAlloc(len + 1);
      memcpy(data_, str.data(), len * sizeof(T));
      size_ = len;
      using_sso_ = false;
    }
    return *this;
  }

  /**
   * Get element at position with bounds checking.
   * Throws std::out_of_range if position is out of bounds.
   *
   * @param pos Position to access
   * @return Reference to character at position
   * @throws std::out_of_range if position is out of bounds
   */
  CTP_CROSS_FUN
  T& at(size_type pos) {
    if (pos >= size_) {
      CTP_THROW(std::out_of_range("String index out of bounds"));
    }
    return GetData()[pos];
  }

  /**
   * Get const element at position with bounds checking.
   *
   * @param pos Position to access
   * @return Const reference to character at position
   * @throws std::out_of_range if position is out of bounds
   */
  CTP_CROSS_FUN
  const T& at(size_type pos) const {
    if (pos >= size_) {
      CTP_THROW(std::out_of_range("String index out of bounds"));
    }
    return GetData()[pos];
  }

  /**
   * Subscript operator without bounds checking.
   * Provides fast unchecked access to characters.
   *
   * @param pos Position to access
   * @return Reference to character at position
   */
  CTP_INLINE_CROSS_FUN
  T& operator[](size_type pos) {
    return GetData()[pos];
  }

  /**
   * Const subscript operator without bounds checking.
   *
   * @param pos Position to access
   * @return Const reference to character at position
   */
  CTP_INLINE_CROSS_FUN
  const T& operator[](size_type pos) const {
    return GetData()[pos];
  }

  /**
   * Get reference to first character.
   * Behavior is undefined if string is empty.
   *
   * @return Reference to first character
   */
  CTP_INLINE_CROSS_FUN
  T& front() {
    return GetData()[0];
  }

  /**
   * Get const reference to first character.
   *
   * @return Const reference to first character
   */
  CTP_INLINE_CROSS_FUN
  const T& front() const {
    return GetData()[0];
  }

  /**
   * Get reference to last character.
   * Behavior is undefined if string is empty.
   *
   * @return Reference to last character
   */
  CTP_INLINE_CROSS_FUN
  T& back() {
    return GetData()[size_ - 1];
  }

  /**
   * Get const reference to last character.
   *
   * @return Const reference to last character
   */
  CTP_INLINE_CROSS_FUN
  const T& back() const {
    return GetData()[size_ - 1];
  }

  /**
   * Get pointer to data.
   * Returns the character array pointer (SSO or vector).
   *
   * @return Pointer to underlying data array
   */
  CTP_INLINE_CROSS_FUN
  T* data() {
    return GetData();
  }

  /**
   * Get const pointer to data.
   *
   * @return Const pointer to underlying data array
   */
  CTP_INLINE_CROSS_FUN
  const T* data() const {
    return GetData();
  }

  /**
   * Get C-style null-terminated string.
   * Ensures string is null-terminated.
   *
   * @return Pointer to null-terminated string
   */
  CTP_CROSS_FUN
  const T* c_str() const {
    if (!UsingSso() && storage_.heap_.capacity_ <= size_) {
      const_cast<basic_string*>(this)->HeapReserve(size_ + 1);
    }
    T* ptr = const_cast<T*>(GetData());
    ptr[size_] = T();
    return ptr;
  }

  /**
   * Get iterator to beginning.
   *
   * @return Iterator to first character
   */
  CTP_INLINE_CROSS_FUN
  iterator begin() {
    return iterator(GetData());
  }

  /**
   * Get const iterator to beginning.
   *
   * @return Const iterator to first character
   */
  CTP_INLINE_CROSS_FUN
  const_iterator begin() const {
    return const_iterator(GetData());
  }

  /**
   * Get const iterator to beginning.
   *
   * @return Const iterator to first character
   */
  CTP_INLINE_CROSS_FUN
  const_iterator cbegin() const {
    return const_iterator(GetData());
  }

  /**
   * Get iterator to end.
   *
   * @return Iterator to one past last character
   */
  CTP_INLINE_CROSS_FUN
  iterator end() {
    return iterator(GetData() + size_);
  }

  /**
   * Get const iterator to end.
   *
   * @return Const iterator to one past last character
   */
  CTP_INLINE_CROSS_FUN
  const_iterator end() const {
    return const_iterator(GetData() + size_);
  }

  /**
   * Get const iterator to end.
   *
   * @return Const iterator to one past last character
   */
  CTP_INLINE_CROSS_FUN
  const_iterator cend() const {
    return const_iterator(GetData() + size_);
  }

  /**
   * Get reverse iterator to beginning
   *
   * @return Reverse iterator to last character
   */
  CTP_INLINE_CROSS_FUN
  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }

  /**
   * Get const reverse iterator to beginning
   *
   * @return Const reverse iterator to last character
   */
  CTP_INLINE_CROSS_FUN
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  /**
   * Get const reverse iterator to beginning
   *
   * @return Const reverse iterator to last character
   */
  CTP_INLINE_CROSS_FUN
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(end());
  }

  /**
   * Get reverse iterator to end
   *
   * @return Reverse iterator to one before first character
   */
  CTP_INLINE_CROSS_FUN
  reverse_iterator rend() {
    return reverse_iterator(begin());
  }

  /**
   * Get const reverse iterator to end
   *
   * @return Const reverse iterator to one before first character
   */
  CTP_INLINE_CROSS_FUN
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  /**
   * Get const reverse iterator to end
   *
   * @return Const reverse iterator to one before first character
   */
  CTP_INLINE_CROSS_FUN
  const_reverse_iterator crend() const {
    return const_reverse_iterator(begin());
  }

  /**
   * Check if string is empty
   *
   * @return True if size is zero
   */
  CTP_INLINE_CROSS_FUN
  bool empty() const {
    return size_ == 0;
  }

  /**
   * Get number of characters in string
   *
   * @return Current size (not including null terminator)
   */
  CTP_INLINE_CROSS_FUN
  size_type size() const {
    return size_;
  }

  /**
   * Get number of characters in string (alias for size)
   *
   * @return Current length
   */
  CTP_INLINE_CROSS_FUN
  size_type length() const {
    return size_;
  }

  /**
   * Get allocated capacity
   *
   * @return Current capacity
   */
  CTP_INLINE_CROSS_FUN
  size_type capacity() const {
    if (UsingSso()) {
      return SSOSize;
    } else {
      return storage_.heap_.capacity_;
    }
  }

  /**
   * Reserve capacity for characters.
   * Allocates new memory if new_capacity exceeds current capacity.
   *
   * @param new_capacity Desired capacity
   */
  CTP_CROSS_FUN
  void reserve(size_type new_capacity) {
    if (new_capacity <= capacity()) {
      return;
    }

    if (UsingSso()) {
      TransitionToHeap(new_capacity);
    } else {
      HeapReserve(new_capacity);
    }
  }

  /**
   * Shrink capacity to match size.
   * Reduces capacity to match current size, freeing unused memory.
   */
  CTP_CROSS_FUN
  void shrink_to_fit() {
    if (!UsingSso() && size_ < SSOSize - 1) {
      // Can transition back to SSO
      T heap_copy[SSOSize];
      memcpy(heap_copy, data_, size_ * sizeof(T));
      HeapFree();
      memcpy(storage_.buffer_, heap_copy, size_ * sizeof(T));
      storage_.buffer_[size_] = T();
      using_sso_ = true;
      data_ = storage_.buffer_;
    } else if (!UsingSso() && storage_.heap_.capacity_ > size_ + 1) {
      // Reallocate to exact size
      auto old_p = HeapFullPtr();
      HeapAlloc(size_ + 1);
      memcpy(data_, old_p.ptr_, size_ * sizeof(T));
      alloc_->Free(old_p);
    }
  }

  /**
   * Add character to end of string
   *
   * @param c Character to add
   */
  CTP_CROSS_FUN
  void push_back(T c) {
    if (UsingSso()) {
      if (size_ + 1 <= SSOSize - 1) {
        storage_.buffer_[size_] = c;
        storage_.buffer_[size_ + 1] = T();
        ++size_;
      } else {
        TransitionToHeap(size_ + 2);
        data_[size_] = c;
        ++size_;
      }
    } else {
      HeapReserve(size_ + 2);
      data_[size_] = c;
      ++size_;
    }
  }

  /**
   * Remove last character from string
   */
  CTP_CROSS_FUN
  void pop_back() {
    if (size_ > 0) {
      --size_;
      if (UsingSso()) {
        storage_.buffer_[size_] = T();
      }
    }
  }

  /**
   * Clear all characters from string
   */
  CTP_CROSS_FUN
  void clear() {
    if (!UsingSso()) {
      HeapFree();
    }
    size_ = 0;
    using_sso_ = true;
    data_ = storage_.buffer_;
    storage_.buffer_[0] = T();
  }

  /**
   * Resize the string to contain count characters.
   * If count is smaller than current size, the string is truncated.
   * If count is larger, the string is extended with default-initialized characters.
   *
   * @param count New size of the string
   */
  CTP_CROSS_FUN
  void resize(size_type count) {
    if (count < size_) {
      // Truncate
      size_ = count;
      GetData()[size_] = T();  // Null terminator
    } else if (count > size_) {
      // Extend
      reserve(count + 1);
      T* d = GetData();
      memset(&d[size_], 0, (count - size_) * sizeof(T));
      size_ = count;
    }
    // If count == size_, do nothing
  }

  /**
   * Resize without initializing new elements.
   * Caller must immediately overwrite the new region.
   *
   * @param count New size of the string
   */
  CTP_CROSS_FUN
  void resize_no_init(size_type count) {
    reserve(count + 1);
    size_ = count;
  }

  /**
   * Append string from another basic_string
   *
   * @param str String to append
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& append(const basic_string& str) {
    AppendCStr(str.GetData(), str.size_);
    return *this;
  }

  /**
   * Append substring from another basic_string
   *
   * @param str String to append from
   * @param pos Starting position in str
   * @param count Number of characters to append (or npos for remainder)
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& append(const basic_string& str, size_type pos,
                       size_type count = npos) {
    if (pos > str.size_) {
      CTP_THROW(std::out_of_range("Append position out of range"));
    }
    size_type append_count = (count == npos) ? (str.size_ - pos) : count;
    append_count = std::min(append_count, str.size_ - pos);
    AppendCStr(&str.GetData()[pos], append_count);
    return *this;
  }

  /**
   * Append C-style string
   *
   * @param s Null-terminated C-style string
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& append(const T* s) {
    if (s != nullptr) {
      size_type len = 0;
      while (s[len] != T()) ++len;
      AppendCStr(s, len);
    }
    return *this;
  }

  /**
   * Append C-style string with specified length
   *
   * @param s Character array
   * @param count Number of characters to append
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& append(const T* s, size_type count) {
    AppendCStr(s, count);
    return *this;
  }

  /**
   * Append count copies of a character
   *
   * @param count Number of times to repeat character
   * @param c Character to append
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& append(size_type count, T c) {
    if (count == 0) return *this;
    size_type new_size = size_ + count;
    if (UsingSso() && new_size <= SSOSize - 1) {
      memset(&storage_.buffer_[size_], (int)c, count * sizeof(T));
      size_ = new_size;
      storage_.buffer_[size_] = T();
    } else {
      if (UsingSso()) {
        TransitionToHeap(new_size + 1);
      } else {
        HeapReserve(new_size + 1);
      }
      memset(data_ + size_, (int)c, count * sizeof(T));
      size_ = new_size;
    }
    return *this;
  }

  /**
   * Append from initializer list
   *
   * @param init Initializer list of characters
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& append(std::initializer_list<T> init) {
    AppendCStr(init.begin(), init.size());
    return *this;
  }

  /**
   * Addition assignment operator (append another string)
   *
   * @param str String to append
   * @return Reference to this string
   */
  CTP_INLINE_CROSS_FUN
  basic_string& operator+=(const basic_string& str) {
    return append(str);
  }

  /**
   * Addition assignment operator (append C-style string)
   *
   * @param s C-style string to append
   * @return Reference to this string
   */
  CTP_INLINE_CROSS_FUN
  basic_string& operator+=(const T* s) {
    return append(s);
  }

  /**
   * Addition assignment operator (append character)
   *
   * @param c Character to append
   * @return Reference to this string
   */
  CTP_INLINE_CROSS_FUN
  basic_string& operator+=(T c) {
    push_back(c);
    return *this;
  }

  /**
   * Addition assignment operator (append initializer list)
   *
   * @param init Initializer list of characters
   * @return Reference to this string
   */
  CTP_INLINE_CROSS_FUN
  basic_string& operator+=(std::initializer_list<T> init) {
    return append(init);
  }

  /**
   * Compare strings for equality
   *
   * @param str String to compare with
   * @return True if strings are equal
   */
  CTP_CROSS_FUN
  bool operator==(const basic_string& str) const {
    if (size_ != str.size_) {
      return false;
    }
    return ctp_memcmp(GetData(), str.GetData(), size_ * sizeof(T)) == 0;
  }

  /**
   * Compare string with C-style string for equality
   *
   * @param s C-style string to compare with
   * @return True if strings are equal
   */
  CTP_CROSS_FUN
  bool operator==(const T* s) const {
    if (s == nullptr) {
      return size_ == 0;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;
    if (size_ != s_len) {
      return false;
    }
    return ctp_memcmp(GetData(), s, size_ * sizeof(T)) == 0;
  }

  /**
   * Compare strings for inequality
   *
   * @param str String to compare with
   * @return True if strings are not equal
   */
  CTP_INLINE_CROSS_FUN
  bool operator!=(const basic_string& str) const {
    return !operator==(str);
  }

  /**
   * Compare string with C-style string for inequality
   *
   * @param s C-style string to compare with
   * @return True if strings are not equal
   */
  CTP_INLINE_CROSS_FUN
  bool operator!=(const T* s) const {
    return !operator==(s);
  }

  /**
   * Compare strings lexicographically
   *
   * @param str String to compare with
   * @return 0 if equal, <0 if this < str, >0 if this > str
   */
  CTP_CROSS_FUN
  int compare(const basic_string& str) const {
    size_type cmp_len = std::min(size_, str.size_);
    int result = ctp_memcmp(GetData(), str.GetData(), cmp_len * sizeof(T));
    if (result != 0) {
      return result;
    }
    if (size_ < str.size_) {
      return -1;
    } else if (size_ > str.size_) {
      return 1;
    }
    return 0;
  }

  /**
   * Compare string with C-style string lexicographically
   *
   * @param s C-style string to compare with
   * @return 0 if equal, <0 if this < s, >0 if this > s
   */
  CTP_CROSS_FUN
  int compare(const T* s) const {
    if (s == nullptr) {
      return size_ > 0 ? 1 : 0;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;

    size_type cmp_len = std::min(size_, s_len);
    int result = ctp_memcmp(GetData(), s, cmp_len * sizeof(T));
    if (result != 0) {
      return result;
    }
    if (size_ < s_len) {
      return -1;
    } else if (size_ > s_len) {
      return 1;
    }
    return 0;
  }

  /**
   * Check if string starts with a substring
   *
   * @param str String to check
   * @return True if string starts with str
   */
  CTP_CROSS_FUN
  bool starts_with(const basic_string& str) const {
    if (str.size_ > size_) {
      return false;
    }
    return ctp_memcmp(GetData(), str.GetData(), str.size_ * sizeof(T)) == 0;
  }

  /**
   * Check if string starts with a character
   *
   * @param c Character to check
   * @return True if string starts with c
   */
  CTP_INLINE_CROSS_FUN
  bool starts_with(T c) const {
    return size_ > 0 && GetData()[0] == c;
  }

  /**
   * Check if string starts with C-style string
   *
   * @param s C-style string to check
   * @return True if string starts with s
   */
  CTP_CROSS_FUN
  bool starts_with(const T* s) const {
    if (s == nullptr) {
      return true;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;
    if (s_len > size_) {
      return false;
    }
    return ctp_memcmp(GetData(), s, s_len * sizeof(T)) == 0;
  }

  /**
   * Check if string ends with a substring
   *
   * @param str String to check
   * @return True if string ends with str
   */
  CTP_CROSS_FUN
  bool ends_with(const basic_string& str) const {
    if (str.size_ > size_) {
      return false;
    }
    return ctp_memcmp(&GetData()[size_ - str.size_], str.GetData(),
                      str.size_ * sizeof(T)) == 0;
  }

  /**
   * Check if string ends with a character
   *
   * @param c Character to check
   * @return True if string ends with c
   */
  CTP_INLINE_CROSS_FUN
  bool ends_with(T c) const {
    return size_ > 0 && GetData()[size_ - 1] == c;
  }

  /**
   * Check if string ends with C-style string
   *
   * @param s C-style string to check
   * @return True if string ends with s
   */
  CTP_CROSS_FUN
  bool ends_with(const T* s) const {
    if (s == nullptr) {
      return true;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;
    if (s_len > size_) {
      return false;
    }
    return ctp_memcmp(&GetData()[size_ - s_len], s, s_len * sizeof(T)) == 0;
  }

  /**
   * Find substring in string
   *
   * @param str Substring to find
   * @param pos Starting position (default: 0)
   * @return Position of substring, or npos if not found
   */
  CTP_CROSS_FUN
  size_type find(const basic_string& str, size_type pos = 0) const {
    if (str.size_ == 0) {
      return pos <= size_ ? pos : npos;
    }
    if (pos + str.size_ > size_) {
      return npos;
    }

    const T* data = GetData();
    for (size_type i = pos; i <= size_ - str.size_; ++i) {
      if (ctp_memcmp(&data[i], str.GetData(), str.size_ * sizeof(T)) == 0) {
        return i;
      }
    }
    return npos;
  }

  /**
   * Find character in string
   *
   * @param c Character to find
   * @param pos Starting position (default: 0)
   * @return Position of character, or npos if not found
   */
  CTP_CROSS_FUN
  size_type find(T c, size_type pos = 0) const {
    if (pos >= size_) {
      return npos;
    }

    const T* data = GetData();
    for (size_type i = pos; i < size_; ++i) {
      if (data[i] == c) {
        return i;
      }
    }
    return npos;
  }

  /**
   * Find C-style string in string
   *
   * @param s C-style string to find
   * @param pos Starting position (default: 0)
   * @return Position of substring, or npos if not found
   */
  CTP_CROSS_FUN
  size_type find(const T* s, size_type pos = 0) const {
    if (s == nullptr) {
      return pos <= size_ ? pos : npos;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;
    if (s_len == 0) {
      return pos <= size_ ? pos : npos;
    }
    if (pos + s_len > size_) {
      return npos;
    }

    const T* data = GetData();
    for (size_type i = pos; i <= size_ - s_len; ++i) {
      if (ctp_memcmp(&data[i], s, s_len * sizeof(T)) == 0) {
        return i;
      }
    }
    return npos;
  }

  /**
   * Extract substring
   *
   * @param pos Starting position
   * @param count Number of characters (or npos for remainder)
   * @return New string with substring
   */
  CTP_CROSS_FUN
  basic_string substr(size_type pos = 0, size_type count = npos) const {
    if (pos > size_) {
      CTP_THROW(std::out_of_range("Substring position out of range"));
    }
    size_type sub_count = (count == npos) ? (size_ - pos) : count;
    sub_count = std::min(sub_count, size_ - pos);
    return basic_string(*this, pos, sub_count, alloc_);
  }

  /**
   * Replace substring with string
   *
   * @param pos Starting position
   * @param count Number of characters to replace
   * @param str String to replace with
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& replace(size_type pos, size_type count,
                       const basic_string& str) {
    if (pos > size_) {
      CTP_THROW(std::out_of_range("Replace position out of range"));
    }
    size_type rep_count = std::min(count, size_ - pos);
    size_type new_size = size_ - rep_count + str.size_;

    if (UsingSso() && new_size <= SSOSize - 1) {
      // Can do in-place with SSO
      memmove(&storage_.buffer_[pos + str.size_],
                   &storage_.buffer_[pos + rep_count],
                   (size_ - pos - rep_count) * sizeof(T));
      memcpy(&storage_.buffer_[pos], str.GetData(),
                  str.size_ * sizeof(T));
      size_ = new_size;
      storage_.buffer_[size_] = T();
    } else {
      if (UsingSso()) {
        TransitionToHeap(new_size + 1);
      } else {
        HeapReserve(new_size + 1);
      }
      // Shift tail, then insert replacement
      memmove(data_ + pos + str.size_,
              data_ + pos + rep_count,
              (size_ - pos - rep_count) * sizeof(T));
      memcpy(data_ + pos, str.GetData(), str.size_ * sizeof(T));
      size_ = new_size;
    }

    return *this;
  }

  /**
   * Replace substring with C-style string
   *
   * @param pos Starting position
   * @param count Number of characters to replace
   * @param s C-style string to replace with
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& replace(size_type pos, size_type count, const T* s) {
    if (s == nullptr) {
      erase(pos, count);
      return *this;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;

    basic_string temp(s, alloc_);
    return replace(pos, count, temp);
  }

  /**
   * Erase characters from string
   *
   * @param pos Starting position
   * @param count Number of characters to erase (or npos for remainder)
   * @return Reference to this string
   */
  CTP_CROSS_FUN
  basic_string& erase(size_type pos = 0, size_type count = npos) {
    if (pos > size_) {
      CTP_THROW(std::out_of_range("Erase position out of range"));
    }
    size_type erase_count = (count == npos) ? (size_ - pos) : count;
    erase_count = std::min(erase_count, size_ - pos);

    if (UsingSso()) {
      memmove(&storage_.buffer_[pos],
                   &storage_.buffer_[pos + erase_count],
                   (size_ - pos - erase_count) * sizeof(T));
      size_ -= erase_count;
      storage_.buffer_[size_] = T();
    } else {
      T* data = GetData();
      memmove(&data[pos], &data[pos + erase_count],
                   (size_ - pos - erase_count) * sizeof(T));
      size_ -= erase_count;
    }

    return *this;
  }

  /**
   * Swap contents with another string
   *
   * @param other String to swap with
   */
  CTP_CROSS_FUN
  void swap(basic_string& other) noexcept {
    std::swap(storage_, other.storage_);
    std::swap(size_, other.size_);
    std::swap(using_sso_, other.using_sso_);
    std::swap(alloc_, other.alloc_);
    std::swap(data_, other.data_);
    // Fix data_ for SSO strings (buffer address changed)
    if (using_sso_) data_ = storage_.buffer_;
    if (other.using_sso_) other.data_ = other.storage_.buffer_;
  }

  /**
   * Convert to std::basic_string for testing/interoperability
   *
   * @return std::basic_string with same content
   */
  CTP_CROSS_FUN
  operator std::basic_string<T>() const {
    return std::basic_string<T>(GetData(), size_);
  }

  /**
   * Convert to std::string (specialization for char)
   *
   * @return std::string with same content
   */
  template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, char>>>
  CTP_CROSS_FUN
  operator std::string() const {
    return std::string(GetData(), size_);
  }

  /**
   * Convert to std::string.
   * Returns an std::string with same content as this string.
   *
   * @return std::string with same content
   */
  CTP_CROSS_FUN
  std::string str() const {
    return std::string(GetData(), size_);
  }

  /**
   * Serialize string to archive.
   * Uses cereal serialization framework to save string data.
   *
   * @tparam Archive Cereal archive type
   * @param ar Archive to save to
   */
  template<class Archive>
  CTP_INLINE_CROSS_FUN void save(Archive& ar) const {
    ctp::ipc::save_string(ar, *this);
  }

  /**
   * Deserialize string from archive.
   * Uses cereal serialization framework to load string data.
   *
   * @tparam Archive Cereal archive type
   * @param ar Archive to load from
   */
  template<class Archive>
  CTP_INLINE_CROSS_FUN void load(Archive& ar) {
    ctp::ipc::load_string(ar, *this);
  }
  /**
   * Convert an integer to a string.
   *
   * @param alloc Allocator to use
   * @param val Integer value to convert
   * @return basic_string containing the decimal representation
   */
  template <typename IntT,
            typename = std::enable_if_t<std::is_integral_v<IntT>>>
  CTP_CROSS_FUN
  static basic_string FromNumber(AllocT *alloc, IntT val) {
    T buf[32];
    size_type len = NumberToStr(buf, sizeof(buf), val);
    return basic_string(alloc, buf, len);
  }

  /**
   * Convert a floating-point number to a string.
   *
   * @param alloc Allocator to use
   * @param val Float/double value to convert
   * @param precision Number of decimal digits (default: 6)
   * @return basic_string containing the decimal representation
   */
  template <typename FloatT,
            typename = std::enable_if_t<std::is_floating_point_v<FloatT>>,
            typename = void>
  CTP_CROSS_FUN
  static basic_string FromNumber(AllocT *alloc, FloatT val,
                                 int precision = 6) {
    T buf[64];
    size_type len = NumberToStr(buf, sizeof(buf), val, precision);
    return basic_string(alloc, buf, len);
  }

 private:
  /**
   * Convert an unsigned integer to a char buffer (GPU-compatible).
   * Returns the number of characters written (excluding null terminator).
   */
  template <typename UIntT>
  CTP_CROSS_FUN
  static size_type UintToStr(T *buf, size_type buf_size, UIntT val) {
    if (buf_size == 0) return 0;
    if (val == 0) {
      buf[0] = '0';
      if (buf_size > 1) buf[1] = T();
      return 1;
    }
    // Write digits in reverse
    T tmp[32];
    size_type nd = 0;
    while (val > 0 && nd < 32) {
      tmp[nd++] = '0' + static_cast<T>(val % 10);
      val /= 10;
    }
    size_type len = (nd < buf_size) ? nd : buf_size - 1;
    for (size_type i = 0; i < len; ++i) {
      buf[i] = tmp[nd - 1 - i];
    }
    if (len < buf_size) buf[len] = T();
    return len;
  }

 public:
  /**
   * Convert an integer to a char buffer (GPU-compatible).
   * Handles signed and unsigned types.
   *
   * @param buf Output buffer
   * @param buf_size Size of output buffer
   * @param val Integer value
   * @return Number of characters written (excluding null terminator)
   */
  template <typename IntT,
            typename = std::enable_if_t<std::is_integral_v<IntT>>>
  CTP_CROSS_FUN
  static size_type NumberToStr(T *buf, size_type buf_size, IntT val) {
    if (buf_size == 0) return 0;
    if constexpr (std::is_signed_v<IntT>) {
      if (val < 0) {
        buf[0] = '-';
        using UIntT = std::make_unsigned_t<IntT>;
        // Handle INT_MIN: cast to unsigned first to avoid UB
        UIntT uval = static_cast<UIntT>(0) - static_cast<UIntT>(val);
        size_type len = UintToStr(buf + 1, buf_size - 1, uval);
        return 1 + len;
      }
    }
    using UIntT = std::make_unsigned_t<IntT>;
    return UintToStr(buf, buf_size, static_cast<UIntT>(val));
  }

  /**
   * Convert a floating-point number to a char buffer (GPU-compatible).
   *
   * @param buf Output buffer
   * @param buf_size Size of output buffer
   * @param val Float/double value
   * @param precision Number of decimal digits (default: 6)
   * @return Number of characters written (excluding null terminator)
   */
  template <typename FloatT,
            typename = std::enable_if_t<std::is_floating_point_v<FloatT>>,
            typename = void>
  CTP_CROSS_FUN
  static size_type NumberToStr(T *buf, size_type buf_size, FloatT val,
                               int precision = 6) {
    if (buf_size == 0) return 0;
    size_type pos = 0;

    // Handle negative
    if (val < 0) {
      buf[pos++] = '-';
      val = -val;
    }

    // Integer part
    auto int_part = static_cast<unsigned long long>(val);
    size_type int_len = UintToStr(buf + pos, buf_size - pos, int_part);
    pos += int_len;

    if (precision > 0 && pos + 1 < buf_size) {
      buf[pos++] = '.';

      // Fractional part
      FloatT frac = val - static_cast<FloatT>(int_part);
      for (int d = 0; d < precision && pos < buf_size - 1; ++d) {
        frac *= 10;
        int digit = static_cast<int>(frac);
        buf[pos++] = '0' + static_cast<T>(digit);
        frac -= digit;
      }
    }

    if (pos < buf_size) buf[pos] = T();
    return pos;
  }
};

/**
 * Convenience typedef for char strings
 */
template<typename AllocT, size_t SSOSize = 32>
using string = basic_string<char, AllocT, SSOSize>;

}  // namespace ctp::priv

#include "clio_ctp/types/hash.h"

namespace ctp {

/** FNV-1a hash specialization for priv::string */
template <typename AllocT, size_t SSOSize>
struct hash<ctp::priv::basic_string<char, AllocT, SSOSize>> {
  CTP_INLINE_CROSS_FUN
  std::size_t operator()(
      const ctp::priv::basic_string<char, AllocT, SSOSize> &s) const {
    std::size_t h = 14695981039346656037ULL;
    const char *data = s.data();
    for (std::size_t i = 0; i < s.size(); ++i) {
      h ^= static_cast<std::size_t>(static_cast<unsigned char>(data[i]));
      h *= 1099511628211ULL;
    }
    return h;
  }
};

}  // namespace ctp

#endif  // CTP_DATA_STRUCTURES_PRIV_STRING_H_
