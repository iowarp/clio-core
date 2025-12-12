/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HSHM_DATA_STRUCTURES_PRIV_STRING_H_
#define HSHM_DATA_STRUCTURES_PRIV_STRING_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/numbers.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include "vector.h"
#include <cstring>
#include <iterator>
#include <type_traits>
#include <stdexcept>
#include <algorithm>

namespace hshm::priv {

/**
 * Private-memory string container with Short String Optimization (SSO)
 *
 * This string class provides std::string-like functionality for private memory,
 * using the library's allocator API with FullPtr for proper memory management.
 * It implements Short String Optimization (SSO) to avoid allocation for small strings.
 * The string uses hshm::priv::vector internally for large string storage,
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
    HSHM_INLINE_CROSS_FUN
    iterator() : ptr_(nullptr) {}

    /**
     * Construct from pointer
     *
     * @param ptr The character pointer
     */
    HSHM_INLINE_CROSS_FUN
    explicit iterator(T *ptr) : ptr_(ptr) {}

    /**
     * Dereference operator
     *
     * @return Reference to the current character
     */
    HSHM_INLINE_CROSS_FUN
    T& operator*() const { return *ptr_; }

    /**
     * Arrow operator
     *
     * @return Pointer to the current character
     */
    HSHM_INLINE_CROSS_FUN
    T* operator->() const { return ptr_; }

    /**
     * Pre-increment operator
     *
     * @return Reference to this iterator
     */
    HSHM_INLINE_CROSS_FUN
    iterator& operator++() {
      ++ptr_;
      return *this;
    }

    /**
     * Post-increment operator
     *
     * @return Copy of this iterator before incrementing
     */
    HSHM_INLINE_CROSS_FUN
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
    HSHM_INLINE_CROSS_FUN
    iterator& operator--() {
      --ptr_;
      return *this;
    }

    /**
     * Post-decrement operator
     *
     * @return Copy of this iterator before decrementing
     */
    HSHM_INLINE_CROSS_FUN
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
    HSHM_INLINE_CROSS_FUN
    iterator operator+(difference_type n) const {
      return iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n Number of characters to go back
     * @return New iterator moved back by n positions
     */
    HSHM_INLINE_CROSS_FUN
    iterator operator-(difference_type n) const {
      return iterator(ptr_ - n);
    }

    /**
     * Addition assignment operator
     *
     * @param n Number of characters to advance
     * @return Reference to this iterator
     */
    HSHM_INLINE_CROSS_FUN
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
    HSHM_INLINE_CROSS_FUN
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
    HSHM_INLINE_CROSS_FUN
    difference_type operator-(const iterator& other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n Index offset from current position
     * @return Reference to the character at offset n
     */
    HSHM_INLINE_CROSS_FUN
    T& operator[](difference_type n) const {
      return ptr_[n];
    }

    /**
     * Equality comparison operator
     *
     * @param other Another iterator
     * @return True if both iterators point to the same character
     */
    HSHM_INLINE_CROSS_FUN
    bool operator==(const iterator& other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality comparison operator
     *
     * @param other Another iterator
     * @return True if iterators point to different characters
     */
    HSHM_INLINE_CROSS_FUN
    bool operator!=(const iterator& other) const {
      return ptr_ != other.ptr_;
    }

    /**
     * Less than comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes before other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator<(const iterator& other) const {
      return ptr_ < other.ptr_;
    }

    /**
     * Less than or equal comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes before or equals other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator<=(const iterator& other) const {
      return ptr_ <= other.ptr_;
    }

    /**
     * Greater than comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes after other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator>(const iterator& other) const {
      return ptr_ > other.ptr_;
    }

    /**
     * Greater than or equal comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes after or equals other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator>=(const iterator& other) const {
      return ptr_ >= other.ptr_;
    }

    /**
     * Get the raw pointer
     *
     * @return The underlying pointer
     */
    HSHM_INLINE_CROSS_FUN
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
    HSHM_INLINE_CROSS_FUN
    const_iterator() : ptr_(nullptr) {}

    /**
     * Construct from pointer
     *
     * @param ptr The character pointer
     */
    HSHM_INLINE_CROSS_FUN
    explicit const_iterator(const T *ptr) : ptr_(ptr) {}

    /**
     * Construct from non-const iterator
     *
     * @param it The iterator to convert
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator(const iterator& it) : ptr_(it.get()) {}

    /**
     * Dereference operator
     *
     * @return Const reference to the current character
     */
    HSHM_INLINE_CROSS_FUN
    const T& operator*() const { return *ptr_; }

    /**
     * Arrow operator
     *
     * @return Const pointer to the current character
     */
    HSHM_INLINE_CROSS_FUN
    const T* operator->() const { return ptr_; }

    /**
     * Pre-increment operator
     *
     * @return Reference to this iterator
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator& operator++() {
      ++ptr_;
      return *this;
    }

    /**
     * Post-increment operator
     *
     * @return Copy of this iterator before incrementing
     */
    HSHM_INLINE_CROSS_FUN
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
    HSHM_INLINE_CROSS_FUN
    const_iterator& operator--() {
      --ptr_;
      return *this;
    }

    /**
     * Post-decrement operator
     *
     * @return Copy of this iterator before decrementing
     */
    HSHM_INLINE_CROSS_FUN
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
    HSHM_INLINE_CROSS_FUN
    const_iterator operator+(difference_type n) const {
      return const_iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n Number of characters to go back
     * @return New iterator moved back by n positions
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator operator-(difference_type n) const {
      return const_iterator(ptr_ - n);
    }

    /**
     * Addition assignment operator
     *
     * @param n Number of characters to advance
     * @return Reference to this iterator
     */
    HSHM_INLINE_CROSS_FUN
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
    HSHM_INLINE_CROSS_FUN
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
    HSHM_INLINE_CROSS_FUN
    difference_type operator-(const const_iterator& other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n Index offset from current position
     * @return Const reference to the character at offset n
     */
    HSHM_INLINE_CROSS_FUN
    const T& operator[](difference_type n) const {
      return ptr_[n];
    }

    /**
     * Equality comparison operator
     *
     * @param other Another iterator
     * @return True if both iterators point to the same character
     */
    HSHM_INLINE_CROSS_FUN
    bool operator==(const const_iterator& other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality comparison operator
     *
     * @param other Another iterator
     * @return True if iterators point to different characters
     */
    HSHM_INLINE_CROSS_FUN
    bool operator!=(const const_iterator& other) const {
      return ptr_ != other.ptr_;
    }

    /**
     * Less than comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes before other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator<(const const_iterator& other) const {
      return ptr_ < other.ptr_;
    }

    /**
     * Less than or equal comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes before or equals other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator<=(const const_iterator& other) const {
      return ptr_ <= other.ptr_;
    }

    /**
     * Greater than comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes after other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator>(const const_iterator& other) const {
      return ptr_ > other.ptr_;
    }

    /**
     * Greater than or equal comparison operator
     *
     * @param other Another iterator
     * @return True if this iterator comes after or equals other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator>=(const const_iterator& other) const {
      return ptr_ >= other.ptr_;
    }

    /**
     * Get the raw pointer
     *
     * @return The underlying pointer
     */
    HSHM_INLINE_CROSS_FUN
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
   * SSO storage union containing either inline buffer or vector pointer
   */
  union SsoStorage {
    T buffer_[SSOSize];  /**< Inline SSO buffer */
    vector<T, AllocT>* vec_;  /**< Pointer to vector for large strings */
  };

  SsoStorage storage_;  /**< Storage union for SSO */
  size_type size_;      /**< Current string length (not including null terminator) */
  bool using_sso_;      /**< Flag to track if using SSO or vector */
  AllocT* alloc_;       /**< Pointer to allocator for memory management */

  /**
   * Check if current string is using SSO buffer
   *
   * @return True if string data is in SSO buffer
   */
  HSHM_INLINE_CROSS_FUN
  bool UsingSso() const {
    return using_sso_;
  }

  /**
   * Get pointer to current string data (SSO or vector)
   *
   * @return Pointer to string data
   */
  HSHM_INLINE_CROSS_FUN
  T* GetData() {
    if (UsingSso()) {
      return storage_.buffer_;
    } else {
      return storage_.vec_->data();
    }
  }

  /**
   * Get const pointer to current string data (SSO or vector)
   *
   * @return Const pointer to string data
   */
  HSHM_INLINE_CROSS_FUN
  const T* GetData() const {
    if (UsingSso()) {
      return storage_.buffer_;
    } else {
      return storage_.vec_->data();
    }
  }


  /**
   * Helper to append from C-style string
   *
   * @param s The C-style string pointer
   * @param len The length of the string
   */
  void AppendCStr(const T* s, size_type len) {
    if (len == 0) {
      return;
    }

    if (UsingSso() && size_ + len <= SSOSize - 1) {
      // Still fits in SSO
      std::memcpy(&storage_.buffer_[size_], s, len * sizeof(T));
      size_ += len;
      storage_.buffer_[size_] = T();
    } else {
      // Need to transition to vector
      if (UsingSso()) {
        storage_.vec_ = new vector<T, AllocT>(alloc_);
        // Reserve enough space upfront to avoid re-allocations
        storage_.vec_->reserve(size_ + len + 1);
        for (size_type i = 0; i < size_; ++i) {
          storage_.vec_->push_back(storage_.buffer_[i]);
        }
        using_sso_ = false;
      }
      for (size_type i = 0; i < len; ++i) {
        storage_.vec_->push_back(s[i]);
      }
      size_ += len;
    }
  }

 public:
  /**
   * Default constructor.
   * Creates an empty string with SSO buffer initialized.
   *
   * @param alloc Pointer to allocator instance for memory management
   */
  explicit basic_string(AllocT* alloc)
    : size_(0), using_sso_(true), alloc_(alloc) {
    storage_.buffer_[0] = T();
  }

  /**
   * Destructor.
   * Frees vector memory if using vector storage. Does not deallocate the allocator.
   */
  HSHM_CROSS_FUN
  ~basic_string() {
    if (!UsingSso()) {
      delete storage_.vec_;
    }
  }

  /**
   * Constructor from C-style string.
   * Creates a string from a null-terminated C-style string.
   *
   * @param s The C-style string (null-terminated)
   * @param alloc Pointer to allocator instance
   */
  basic_string(const T* s, AllocT* alloc)
    : size_(0), using_sso_(true), alloc_(alloc) {
    if (s != nullptr) {
      size_type len = 0;
      while (s[len] != T()) ++len;
      if (len < SSOSize - 1) {
        std::memcpy(storage_.buffer_, s, len * sizeof(T));
        storage_.buffer_[len] = T();
        size_ = len;
      } else {
        storage_.vec_ = new vector<T, AllocT>(alloc_);
        for (size_type i = 0; i < len; ++i) {
          storage_.vec_->push_back(s[i]);
        }
        size_ = len;
        using_sso_ = false;
      }
    } else {
      storage_.buffer_[0] = T();
    }
  }

  /**
   * Constructor from character count.
   * Creates a string with count copies of character c.
   *
   * @param count Number of characters
   * @param c The character to repeat
   * @param alloc Pointer to allocator instance
   */
  basic_string(size_type count, T c, AllocT* alloc)
    : size_(0), using_sso_(true), alloc_(alloc) {
    if (count < SSOSize - 1) {
      for (size_type i = 0; i < count; ++i) {
        storage_.buffer_[i] = c;
      }
      storage_.buffer_[count] = T();
      size_ = count;
    } else {
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      for (size_type i = 0; i < count; ++i) {
        storage_.vec_->push_back(c);
      }
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
    : size_(0), using_sso_(true), alloc_(alloc) {
    if (pos > other.size_) {
      throw std::out_of_range("Substring position out of range");
    }
    size_type copy_count = (count == npos) ? (other.size_ - pos) : count;
    copy_count = std::min(copy_count, other.size_ - pos);

    const T* other_data = other.GetData();
    if (copy_count < SSOSize - 1) {
      std::memcpy(storage_.buffer_, &other_data[pos], copy_count * sizeof(T));
      storage_.buffer_[copy_count] = T();
      size_ = copy_count;
    } else {
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      for (size_type i = 0; i < copy_count; ++i) {
        storage_.vec_->push_back(other_data[pos + i]);
      }
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
  basic_string(const basic_string& other)
    : size_(0), using_sso_(true), alloc_(other.alloc_) {
    const T* other_data = other.GetData();
    if (other.size_ < SSOSize - 1) {
      std::memcpy(storage_.buffer_, other_data, other.size_ * sizeof(T));
      storage_.buffer_[other.size_] = T();
      size_ = other.size_;
    } else {
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      for (size_type i = 0; i < other.size_; ++i) {
        storage_.vec_->push_back(other_data[i]);
      }
      size_ = other.size_;
      using_sso_ = false;
    }
  }

  /**
   * Move constructor.
   * Transfers ownership of data from other string to this one.
   *
   * @param other String to move from
   */
  basic_string(basic_string&& other) noexcept
    : size_(other.size_), using_sso_(other.using_sso_), alloc_(other.alloc_) {
    if (other.UsingSso()) {
      std::memcpy(storage_.buffer_, other.storage_.buffer_,
                  other.size_ * sizeof(T));
      storage_.buffer_[other.size_] = T();
    } else {
      storage_.vec_ = other.storage_.vec_;
    }
    other.size_ = 0;
    other.using_sso_ = true;
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
    : size_(0), using_sso_(true), alloc_(alloc) {
    if (init.size() < SSOSize - 1) {
      size_type i = 0;
      for (const T& c : init) {
        storage_.buffer_[i++] = c;
      }
      storage_.buffer_[init.size()] = T();
      size_ = init.size();
    } else {
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      for (const T& c : init) {
        storage_.vec_->push_back(c);
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
    : size_(0), using_sso_(true), alloc_(alloc) {
    size_type len = str.size();
    if (len < SSOSize - 1) {
      std::memcpy(storage_.buffer_, str.data(), len * sizeof(T));
      storage_.buffer_[len] = T();
      size_ = len;
    } else {
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      for (size_type i = 0; i < len; ++i) {
        storage_.vec_->push_back(str[i]);
      }
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
  HSHM_CROSS_FUN
  basic_string& operator=(const basic_string& other) {
    if (this != &other) {
      if (!UsingSso()) {
        delete storage_.vec_;
      }
      size_ = 0;
      using_sso_ = true;
      alloc_ = other.alloc_;

      const T* other_data = other.GetData();
      if (other.size_ < SSOSize - 1) {
        std::memcpy(storage_.buffer_, other_data, other.size_ * sizeof(T));
        storage_.buffer_[other.size_] = T();
        size_ = other.size_;
      } else {
        storage_.vec_ = new vector<T, AllocT>(alloc_);
        for (size_type i = 0; i < other.size_; ++i) {
          storage_.vec_->push_back(other_data[i]);
        }
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
  HSHM_CROSS_FUN
  basic_string& operator=(basic_string&& other) noexcept {
    if (this != &other) {
      if (!UsingSso()) {
        delete storage_.vec_;
      }
      size_ = other.size_;
      using_sso_ = other.using_sso_;
      alloc_ = other.alloc_;

      if (other.UsingSso()) {
        std::memcpy(storage_.buffer_, other.storage_.buffer_,
                    other.size_ * sizeof(T));
        storage_.buffer_[other.size_] = T();
      } else {
        storage_.vec_ = other.storage_.vec_;
      }

      other.size_ = 0;
      other.using_sso_ = true;
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
  HSHM_CROSS_FUN
  basic_string& operator=(const T* s) {
    if (s == nullptr) {
      clear();
      return *this;
    }

    if (!UsingSso()) {
      delete storage_.vec_;
    }
    size_ = 0;
    using_sso_ = true;

    size_type len = 0;
    while (s[len] != T()) ++len;

    if (len < SSOSize - 1) {
      std::memcpy(storage_.buffer_, s, len * sizeof(T));
      storage_.buffer_[len] = T();
      size_ = len;
    } else {
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      for (size_type i = 0; i < len; ++i) {
        storage_.vec_->push_back(s[i]);
      }
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
  HSHM_CROSS_FUN
  basic_string& operator=(std::initializer_list<T> init) {
    if (!UsingSso()) {
      delete storage_.vec_;
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
    } else {
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      for (const T& c : init) {
        storage_.vec_->push_back(c);
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
  HSHM_CROSS_FUN
  basic_string& operator=(const std::basic_string<T, U>& str) {
    if (!UsingSso()) {
      delete storage_.vec_;
    }
    size_ = 0;
    using_sso_ = true;

    size_type len = str.size();
    if (len < SSOSize - 1) {
      std::memcpy(storage_.buffer_, str.data(), len * sizeof(T));
      storage_.buffer_[len] = T();
      size_ = len;
    } else {
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      for (size_type i = 0; i < len; ++i) {
        storage_.vec_->push_back(str[i]);
      }
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
  HSHM_CROSS_FUN
  T& at(size_type pos) {
    if (pos >= size_) {
      throw std::out_of_range("String index out of bounds");
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
  HSHM_CROSS_FUN
  const T& at(size_type pos) const {
    if (pos >= size_) {
      throw std::out_of_range("String index out of bounds");
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
  HSHM_INLINE_CROSS_FUN
  T& operator[](size_type pos) {
    return GetData()[pos];
  }

  /**
   * Const subscript operator without bounds checking.
   *
   * @param pos Position to access
   * @return Const reference to character at position
   */
  HSHM_INLINE_CROSS_FUN
  const T& operator[](size_type pos) const {
    return GetData()[pos];
  }

  /**
   * Get reference to first character.
   * Behavior is undefined if string is empty.
   *
   * @return Reference to first character
   */
  HSHM_INLINE_CROSS_FUN
  T& front() {
    return GetData()[0];
  }

  /**
   * Get const reference to first character.
   *
   * @return Const reference to first character
   */
  HSHM_INLINE_CROSS_FUN
  const T& front() const {
    return GetData()[0];
  }

  /**
   * Get reference to last character.
   * Behavior is undefined if string is empty.
   *
   * @return Reference to last character
   */
  HSHM_INLINE_CROSS_FUN
  T& back() {
    return GetData()[size_ - 1];
  }

  /**
   * Get const reference to last character.
   *
   * @return Const reference to last character
   */
  HSHM_INLINE_CROSS_FUN
  const T& back() const {
    return GetData()[size_ - 1];
  }

  /**
   * Get pointer to data.
   * Returns the character array pointer (SSO or vector).
   *
   * @return Pointer to underlying data array
   */
  HSHM_INLINE_CROSS_FUN
  T* data() {
    return GetData();
  }

  /**
   * Get const pointer to data.
   *
   * @return Const pointer to underlying data array
   */
  HSHM_INLINE_CROSS_FUN
  const T* data() const {
    return GetData();
  }

  /**
   * Get C-style null-terminated string.
   * Ensures string is null-terminated.
   *
   * @return Pointer to null-terminated string
   */
  HSHM_CROSS_FUN
  const T* c_str() const {
    T* ptr = const_cast<T*>(GetData());
    ptr[size_] = T();
    return ptr;
  }

  /**
   * Get iterator to beginning.
   *
   * @return Iterator to first character
   */
  HSHM_INLINE_CROSS_FUN
  iterator begin() {
    return iterator(GetData());
  }

  /**
   * Get const iterator to beginning.
   *
   * @return Const iterator to first character
   */
  HSHM_INLINE_CROSS_FUN
  const_iterator begin() const {
    return const_iterator(GetData());
  }

  /**
   * Get const iterator to beginning.
   *
   * @return Const iterator to first character
   */
  HSHM_INLINE_CROSS_FUN
  const_iterator cbegin() const {
    return const_iterator(GetData());
  }

  /**
   * Get iterator to end.
   *
   * @return Iterator to one past last character
   */
  HSHM_INLINE_CROSS_FUN
  iterator end() {
    return iterator(GetData() + size_);
  }

  /**
   * Get const iterator to end.
   *
   * @return Const iterator to one past last character
   */
  HSHM_INLINE_CROSS_FUN
  const_iterator end() const {
    return const_iterator(GetData() + size_);
  }

  /**
   * Get const iterator to end.
   *
   * @return Const iterator to one past last character
   */
  HSHM_INLINE_CROSS_FUN
  const_iterator cend() const {
    return const_iterator(GetData() + size_);
  }

  /**
   * Get reverse iterator to beginning
   *
   * @return Reverse iterator to last character
   */
  HSHM_INLINE_CROSS_FUN
  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }

  /**
   * Get const reverse iterator to beginning
   *
   * @return Const reverse iterator to last character
   */
  HSHM_INLINE_CROSS_FUN
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  /**
   * Get const reverse iterator to beginning
   *
   * @return Const reverse iterator to last character
   */
  HSHM_INLINE_CROSS_FUN
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(end());
  }

  /**
   * Get reverse iterator to end
   *
   * @return Reverse iterator to one before first character
   */
  HSHM_INLINE_CROSS_FUN
  reverse_iterator rend() {
    return reverse_iterator(begin());
  }

  /**
   * Get const reverse iterator to end
   *
   * @return Const reverse iterator to one before first character
   */
  HSHM_INLINE_CROSS_FUN
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  /**
   * Get const reverse iterator to end
   *
   * @return Const reverse iterator to one before first character
   */
  HSHM_INLINE_CROSS_FUN
  const_reverse_iterator crend() const {
    return const_reverse_iterator(begin());
  }

  /**
   * Check if string is empty
   *
   * @return True if size is zero
   */
  HSHM_INLINE_CROSS_FUN
  bool empty() const {
    return size_ == 0;
  }

  /**
   * Get number of characters in string
   *
   * @return Current size (not including null terminator)
   */
  HSHM_INLINE_CROSS_FUN
  size_type size() const {
    return size_;
  }

  /**
   * Get number of characters in string (alias for size)
   *
   * @return Current length
   */
  HSHM_INLINE_CROSS_FUN
  size_type length() const {
    return size_;
  }

  /**
   * Get allocated capacity
   *
   * @return Current capacity
   */
  HSHM_INLINE_CROSS_FUN
  size_type capacity() const {
    if (UsingSso()) {
      return SSOSize;
    } else {
      return storage_.vec_->capacity();
    }
  }

  /**
   * Reserve capacity for characters.
   * Allocates new memory if new_capacity exceeds current capacity.
   *
   * @param new_capacity Desired capacity
   */
  HSHM_CROSS_FUN
  void reserve(size_type new_capacity) {
    if (new_capacity <= capacity()) {
      return;
    }

    if (UsingSso()) {
      // Need to transition to vector
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      storage_.vec_->reserve(new_capacity);

      // Copy SSO data to vector
      for (size_type i = 0; i < size_; ++i) {
        storage_.vec_->push_back(storage_.buffer_[i]);
      }
      using_sso_ = false;
    } else {
      storage_.vec_->reserve(new_capacity);
    }
  }

  /**
   * Shrink capacity to match size.
   * Reduces capacity to match current size, freeing unused memory.
   */
  HSHM_CROSS_FUN
  void shrink_to_fit() {
    if (!UsingSso() && size_ < SSOSize - 1) {
      // Can transition back to SSO
      vector<T, AllocT>* old_vec = storage_.vec_;

      for (size_type i = 0; i < size_; ++i) {
        storage_.buffer_[i] = (*old_vec)[i];
      }
      storage_.buffer_[size_] = T();
      delete old_vec;
      using_sso_ = true;
    } else if (!UsingSso()) {
      storage_.vec_->shrink_to_fit();
    }
  }

  /**
   * Add character to end of string
   *
   * @param c Character to add
   */
  HSHM_CROSS_FUN
  void push_back(T c) {
    if (UsingSso()) {
      if (size_ + 1 <= SSOSize - 1) {
        storage_.buffer_[size_] = c;
        storage_.buffer_[size_ + 1] = T();
        ++size_;
      } else {
        // Transition to vector
        storage_.vec_ = new vector<T, AllocT>(alloc_);
        for (size_type i = 0; i < size_; ++i) {
          storage_.vec_->push_back(storage_.buffer_[i]);
        }
        storage_.vec_->push_back(c);
        ++size_;
        using_sso_ = false;
      }
    } else {
      storage_.vec_->push_back(c);
      ++size_;
    }
  }

  /**
   * Remove last character from string
   */
  HSHM_CROSS_FUN
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
  HSHM_CROSS_FUN
  void clear() {
    if (!UsingSso()) {
      delete storage_.vec_;
    }
    size_ = 0;
    using_sso_ = true;
    storage_.buffer_[0] = T();
  }

  /**
   * Resize the string to contain count characters.
   * If count is smaller than current size, the string is truncated.
   * If count is larger, the string is extended with default-initialized characters.
   *
   * @param count New size of the string
   */
  HSHM_CROSS_FUN
  void resize(size_type count) {
    if (count < size_) {
      // Truncate
      size_ = count;
      GetData()[size_] = T();  // Null terminator
    } else if (count > size_) {
      // Extend
      reserve(count + 1);
      // Fill with default-initialized characters
      for (size_type i = size_; i < count; ++i) {
        push_back(T());
      }
    }
    // If count == size_, do nothing
  }

  /**
   * Append string from another basic_string
   *
   * @param str String to append
   * @return Reference to this string
   */
  HSHM_CROSS_FUN
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
  HSHM_CROSS_FUN
  basic_string& append(const basic_string& str, size_type pos,
                       size_type count = npos) {
    if (pos > str.size_) {
      throw std::out_of_range("Append position out of range");
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
  HSHM_CROSS_FUN
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
  HSHM_CROSS_FUN
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
  HSHM_CROSS_FUN
  basic_string& append(size_type count, T c) {
    for (size_type i = 0; i < count; ++i) {
      push_back(c);
    }
    return *this;
  }

  /**
   * Append from initializer list
   *
   * @param init Initializer list of characters
   * @return Reference to this string
   */
  HSHM_CROSS_FUN
  basic_string& append(std::initializer_list<T> init) {
    for (const T& c : init) {
      push_back(c);
    }
    return *this;
  }

  /**
   * Addition assignment operator (append another string)
   *
   * @param str String to append
   * @return Reference to this string
   */
  HSHM_INLINE_CROSS_FUN
  basic_string& operator+=(const basic_string& str) {
    return append(str);
  }

  /**
   * Addition assignment operator (append C-style string)
   *
   * @param s C-style string to append
   * @return Reference to this string
   */
  HSHM_INLINE_CROSS_FUN
  basic_string& operator+=(const T* s) {
    return append(s);
  }

  /**
   * Addition assignment operator (append character)
   *
   * @param c Character to append
   * @return Reference to this string
   */
  HSHM_INLINE_CROSS_FUN
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
  HSHM_INLINE_CROSS_FUN
  basic_string& operator+=(std::initializer_list<T> init) {
    return append(init);
  }

  /**
   * Compare strings for equality
   *
   * @param str String to compare with
   * @return True if strings are equal
   */
  HSHM_CROSS_FUN
  bool operator==(const basic_string& str) const {
    if (size_ != str.size_) {
      return false;
    }
    return std::memcmp(GetData(), str.GetData(), size_ * sizeof(T)) == 0;
  }

  /**
   * Compare string with C-style string for equality
   *
   * @param s C-style string to compare with
   * @return True if strings are equal
   */
  HSHM_CROSS_FUN
  bool operator==(const T* s) const {
    if (s == nullptr) {
      return size_ == 0;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;
    if (size_ != s_len) {
      return false;
    }
    return std::memcmp(GetData(), s, size_ * sizeof(T)) == 0;
  }

  /**
   * Compare strings for inequality
   *
   * @param str String to compare with
   * @return True if strings are not equal
   */
  HSHM_INLINE_CROSS_FUN
  bool operator!=(const basic_string& str) const {
    return !operator==(str);
  }

  /**
   * Compare string with C-style string for inequality
   *
   * @param s C-style string to compare with
   * @return True if strings are not equal
   */
  HSHM_INLINE_CROSS_FUN
  bool operator!=(const T* s) const {
    return !operator==(s);
  }

  /**
   * Compare strings lexicographically
   *
   * @param str String to compare with
   * @return 0 if equal, <0 if this < str, >0 if this > str
   */
  HSHM_CROSS_FUN
  int compare(const basic_string& str) const {
    size_type cmp_len = std::min(size_, str.size_);
    int result = std::memcmp(GetData(), str.GetData(), cmp_len * sizeof(T));
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
  HSHM_CROSS_FUN
  int compare(const T* s) const {
    if (s == nullptr) {
      return size_ > 0 ? 1 : 0;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;

    size_type cmp_len = std::min(size_, s_len);
    int result = std::memcmp(GetData(), s, cmp_len * sizeof(T));
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
  HSHM_CROSS_FUN
  bool starts_with(const basic_string& str) const {
    if (str.size_ > size_) {
      return false;
    }
    return std::memcmp(GetData(), str.GetData(), str.size_ * sizeof(T)) == 0;
  }

  /**
   * Check if string starts with a character
   *
   * @param c Character to check
   * @return True if string starts with c
   */
  HSHM_INLINE_CROSS_FUN
  bool starts_with(T c) const {
    return size_ > 0 && GetData()[0] == c;
  }

  /**
   * Check if string starts with C-style string
   *
   * @param s C-style string to check
   * @return True if string starts with s
   */
  HSHM_CROSS_FUN
  bool starts_with(const T* s) const {
    if (s == nullptr) {
      return true;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;
    if (s_len > size_) {
      return false;
    }
    return std::memcmp(GetData(), s, s_len * sizeof(T)) == 0;
  }

  /**
   * Check if string ends with a substring
   *
   * @param str String to check
   * @return True if string ends with str
   */
  HSHM_CROSS_FUN
  bool ends_with(const basic_string& str) const {
    if (str.size_ > size_) {
      return false;
    }
    return std::memcmp(&GetData()[size_ - str.size_], str.GetData(),
                      str.size_ * sizeof(T)) == 0;
  }

  /**
   * Check if string ends with a character
   *
   * @param c Character to check
   * @return True if string ends with c
   */
  HSHM_INLINE_CROSS_FUN
  bool ends_with(T c) const {
    return size_ > 0 && GetData()[size_ - 1] == c;
  }

  /**
   * Check if string ends with C-style string
   *
   * @param s C-style string to check
   * @return True if string ends with s
   */
  HSHM_CROSS_FUN
  bool ends_with(const T* s) const {
    if (s == nullptr) {
      return true;
    }
    size_type s_len = 0;
    while (s[s_len] != T()) ++s_len;
    if (s_len > size_) {
      return false;
    }
    return std::memcmp(&GetData()[size_ - s_len], s, s_len * sizeof(T)) == 0;
  }

  /**
   * Find substring in string
   *
   * @param str Substring to find
   * @param pos Starting position (default: 0)
   * @return Position of substring, or npos if not found
   */
  HSHM_CROSS_FUN
  size_type find(const basic_string& str, size_type pos = 0) const {
    if (str.size_ == 0) {
      return pos <= size_ ? pos : npos;
    }
    if (pos + str.size_ > size_) {
      return npos;
    }

    const T* data = GetData();
    for (size_type i = pos; i <= size_ - str.size_; ++i) {
      if (std::memcmp(&data[i], str.GetData(), str.size_ * sizeof(T)) == 0) {
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
  HSHM_CROSS_FUN
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
  HSHM_CROSS_FUN
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
      if (std::memcmp(&data[i], s, s_len * sizeof(T)) == 0) {
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
  HSHM_CROSS_FUN
  basic_string substr(size_type pos = 0, size_type count = npos) const {
    if (pos > size_) {
      throw std::out_of_range("Substring position out of range");
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
  HSHM_CROSS_FUN
  basic_string& replace(size_type pos, size_type count,
                       const basic_string& str) {
    if (pos > size_) {
      throw std::out_of_range("Replace position out of range");
    }
    size_type rep_count = std::min(count, size_ - pos);
    size_type new_size = size_ - rep_count + str.size_;

    if (UsingSso() && new_size <= SSOSize - 1) {
      // Can do in-place with SSO
      std::memmove(&storage_.buffer_[pos + str.size_],
                   &storage_.buffer_[pos + rep_count],
                   (size_ - pos - rep_count) * sizeof(T));
      std::memcpy(&storage_.buffer_[pos], str.GetData(),
                  str.size_ * sizeof(T));
      size_ = new_size;
      storage_.buffer_[size_] = T();
    } else {
      // Use vector for large replacement
      if (UsingSso()) {
        storage_.vec_ = new vector<T, AllocT>(alloc_);
        storage_.vec_->reserve(new_size + 1);
        for (size_type i = 0; i < size_; ++i) {
          storage_.vec_->push_back(storage_.buffer_[i]);
        }
        using_sso_ = false;
      }

      // Create new vector with replaced content
      vector<T, AllocT> new_vec(alloc_);
      T* data = GetData();

      // Copy before replacement
      for (size_type i = 0; i < pos; ++i) {
        new_vec.push_back(data[i]);
      }

      // Copy replacement
      for (size_type i = 0; i < str.size_; ++i) {
        new_vec.push_back(str.GetData()[i]);
      }

      // Copy after replacement
      for (size_type i = pos + rep_count; i < size_; ++i) {
        new_vec.push_back(data[i]);
      }

      // Swap with old vector
      storage_.vec_->clear();
      delete storage_.vec_;
      storage_.vec_ = new vector<T, AllocT>(alloc_);
      for (size_type i = 0; i < new_vec.size(); ++i) {
        storage_.vec_->push_back(new_vec[i]);
      }
      size_ = new_vec.size();
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
  HSHM_CROSS_FUN
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
  HSHM_CROSS_FUN
  basic_string& erase(size_type pos = 0, size_type count = npos) {
    if (pos > size_) {
      throw std::out_of_range("Erase position out of range");
    }
    size_type erase_count = (count == npos) ? (size_ - pos) : count;
    erase_count = std::min(erase_count, size_ - pos);

    if (UsingSso()) {
      std::memmove(&storage_.buffer_[pos],
                   &storage_.buffer_[pos + erase_count],
                   (size_ - pos - erase_count) * sizeof(T));
      size_ -= erase_count;
      storage_.buffer_[size_] = T();
    } else {
      T* data = GetData();
      std::memmove(&data[pos], &data[pos + erase_count],
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
  HSHM_CROSS_FUN
  void swap(basic_string& other) noexcept {
    std::swap(storage_, other.storage_);
    std::swap(size_, other.size_);
    std::swap(alloc_, other.alloc_);
  }

  /**
   * Convert to std::basic_string for testing/interoperability
   *
   * @return std::basic_string with same content
   */
  HSHM_CROSS_FUN
  operator std::basic_string<T>() const {
    return std::basic_string<T>(GetData(), size_);
  }

  /**
   * Convert to std::string (specialization for char)
   *
   * @return std::string with same content
   */
  template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, char>>>
  HSHM_CROSS_FUN
  operator std::string() const {
    return std::string(GetData(), size_);
  }

  /**
   * Convert to std::string.
   * Returns an std::string with same content as this string.
   *
   * @return std::string with same content
   */
  HSHM_CROSS_FUN
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
  void save(Archive& ar) const {
    hshm::ipc::save_string(ar, *this);
  }

  /**
   * Deserialize string from archive.
   * Uses cereal serialization framework to load string data.
   *
   * @tparam Archive Cereal archive type
   * @param ar Archive to load from
   */
  template<class Archive>
  void load(Archive& ar) {
    hshm::ipc::load_string(ar, *this);
  }
};

/**
 * Convenience typedef for char strings
 */
template<typename AllocT, size_t SSOSize = 32>
using string = basic_string<char, AllocT, SSOSize>;

}  // namespace hshm::priv

#endif  // HSHM_DATA_STRUCTURES_PRIV_STRING_H_
