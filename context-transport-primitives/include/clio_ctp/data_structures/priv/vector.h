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

#ifndef CTP_DATA_STRUCTURES_PRIV_VECTOR_H_
#define CTP_DATA_STRUCTURES_PRIV_VECTOR_H_

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/types/numbers.h"
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/data_structures/serialization/serialize_common.h"
#include <cstring>
#include <iterator>
#include <type_traits>
#include <stdexcept>
#include <algorithm>

namespace ctp::priv {

/**
 * Private-memory vector container with allocator integration and SVO
 *
 * This vector class provides dynamic array functionality for private memory,
 * using the library's allocator API with FullPtr for proper memory management.
 * It supports both POD (Plain Old Data) types and complex class types.
 * POD types use optimized memcpy/memset operations for better performance.
 *
 * Simple Vector Optimization (SVO): small vectors (up to SVO_SIZE elements)
 * are stored inline in a stack-local buffer, avoiding allocator calls entirely.
 * When the vector grows beyond SVO_SIZE, it transitions to heap allocation.
 *
 * @tparam T The element type
 * @tparam AllocT The allocator type (must have Allocate/AllocateObjs/Free methods)
 * @tparam SVO_SIZE Number of elements stored inline before heap allocation (default 8)
 */
template<typename T, typename AllocT, size_t SVO_SIZE = 8>
class vector {
 public:
  using allocator_type = AllocT;
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  /**
   * Random access iterator for vector
   */
  class iterator {
   private:
    T *ptr_;  /**< Current element pointer */

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
     * @param ptr The element pointer
     */
    CTP_INLINE_CROSS_FUN
    explicit iterator(T *ptr) : ptr_(ptr) {}

    /**
     * Dereference operator
     *
     * @return Reference to the current element
     */
    CTP_INLINE_CROSS_FUN
    T& operator*() const { return *ptr_; }

    /**
     * Arrow operator
     *
     * @return Pointer to the current element
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
     * @param n Number of elements to advance
     * @return New iterator advanced by n positions
     */
    CTP_INLINE_CROSS_FUN
    iterator operator+(difference_type n) const {
      return iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n Number of elements to go back
     * @return New iterator moved back by n positions
     */
    CTP_INLINE_CROSS_FUN
    iterator operator-(difference_type n) const {
      return iterator(ptr_ - n);
    }

    /**
     * Addition assignment operator
     *
     * @param n Number of elements to advance
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
     * @param n Number of elements to go back
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
     * @return The number of elements between this and other
     */
    CTP_INLINE_CROSS_FUN
    difference_type operator-(const iterator& other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n Index offset from current position
     * @return Reference to the element at offset n
     */
    CTP_INLINE_CROSS_FUN
    T& operator[](difference_type n) const {
      return ptr_[n];
    }

    /**
     * Equality comparison operator
     *
     * @param other Another iterator
     * @return True if both iterators point to the same element
     */
    CTP_INLINE_CROSS_FUN
    bool operator==(const iterator& other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality comparison operator
     *
     * @param other Another iterator
     * @return True if iterators point to different elements
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
   * Const iterator for vector
   */
  class const_iterator {
   private:
    const T *ptr_;  /**< Current element pointer */

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
     * @param ptr The element pointer
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
     * @return Const reference to the current element
     */
    CTP_INLINE_CROSS_FUN
    const T& operator*() const { return *ptr_; }

    /**
     * Arrow operator
     *
     * @return Const pointer to the current element
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
     * @param n Number of elements to advance
     * @return New iterator advanced by n positions
     */
    CTP_INLINE_CROSS_FUN
    const_iterator operator+(difference_type n) const {
      return const_iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n Number of elements to go back
     * @return New iterator moved back by n positions
     */
    CTP_INLINE_CROSS_FUN
    const_iterator operator-(difference_type n) const {
      return const_iterator(ptr_ - n);
    }

    /**
     * Addition assignment operator
     *
     * @param n Number of elements to advance
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
     * @param n Number of elements to go back
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
     * @return The number of elements between this and other
     */
    CTP_INLINE_CROSS_FUN
    difference_type operator-(const const_iterator& other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n Index offset from current position
     * @return Const reference to the element at offset n
     */
    CTP_INLINE_CROSS_FUN
    const T& operator[](difference_type n) const {
      return ptr_[n];
    }

    /**
     * Equality comparison operator
     *
     * @param other Another iterator
     * @return True if both iterators point to the same element
     */
    CTP_INLINE_CROSS_FUN
    bool operator==(const const_iterator& other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality comparison operator
     *
     * @param other Another iterator
     * @return True if iterators point to different elements
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
   * Reverse iterator for vector
   */
  using reverse_iterator = std::reverse_iterator<iterator>;

  /**
   * Const reverse iterator for vector
   */
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  ctp::ipc::FullPtr<T> data_;  /**< FullPtr to the data array (both private and shared) */
  size_type size_;         /**< Current number of elements */
  size_type capacity_;     /**< Allocated capacity */
  AllocT* alloc_;          /**< Pointer to allocator for memory management */
  alignas(alignof(T)) char svo_[SVO_SIZE * sizeof(T)];  /**< Inline SVO buffer */

  /**
   * Helper to determine if type is POD
   */
  static constexpr bool kIsPod = std::is_trivial_v<T>;

  /**
   * Get typed pointer to the SVO buffer
   */
  CTP_INLINE_CROSS_FUN
  T* svo_data() { return reinterpret_cast<T*>(svo_); }

  /**
   * Get const typed pointer to the SVO buffer
   */
  CTP_INLINE_CROSS_FUN
  const T* svo_data() const { return reinterpret_cast<const T*>(svo_); }

  /**
   * Check if vector is currently using the inline SVO buffer
   */
  CTP_INLINE_CROSS_FUN
  bool IsUsingSvo() const { return data_.ptr_ == svo_data(); }

  /**
   * Initialize data_ to point to the SVO buffer with null shm
   */
  CTP_INLINE_CROSS_FUN
  void InitSvo() {
    data_ = ctp::ipc::FullPtr<T>::GetNull();
    data_.ptr_ = svo_data();
    capacity_ = SVO_SIZE;
  }

  /**
   * Grow the capacity to accommodate new elements.
   * Doubles capacity until it's large enough for min_capacity.
   *
   * @param min_capacity Minimum capacity needed
   */
  CTP_INLINE_CROSS_FUN
  void Grow(size_type min_capacity) {
    size_type new_capacity = (capacity_ == 0) ? 1 : capacity_ * 2;
    while (new_capacity < min_capacity) {
      new_capacity *= 2;
    }
    reserve(new_capacity);
  }

  /**
   * Construct element at position with move semantics.
   * Uses placement new for non-POD types, direct assignment for POD types.
   *
   * @param pos Position to construct at
   * @param val Value to move construct
   */
  CTP_INLINE_CROSS_FUN
  void ConstructMove(size_type pos, T&& val) {
    if constexpr (kIsPod) {
      data_.ptr_[pos] = static_cast<T&&>(val);
    } else {
      new (&data_.ptr_[pos]) T(static_cast<T&&>(val));
    }
  }

  /**
   * Construct element at position with copy semantics.
   * Uses placement new for non-POD types, direct assignment for POD types.
   *
   * @param pos Position to construct at
   * @param val Value to copy construct
   */
  CTP_INLINE_CROSS_FUN
  void ConstructCopy(size_type pos, const T& val) {
    if constexpr (kIsPod) {
      data_.ptr_[pos] = val;
    } else {
      new (&data_.ptr_[pos]) T(val);
    }
  }

  /**
   * Destroy element at position.
   * Calls destructor for non-POD types. No-op for POD types.
   *
   * @param pos Position to destroy
   */
  CTP_INLINE_CROSS_FUN
  void Destroy(size_type pos) {
    if constexpr (!kIsPod) {
      data_.ptr_[pos].~T();
    }
  }

  /**
   * Destroy range of elements.
   * Calls destructors for all elements in range [first, last).
   *
   * @param first First position to destroy
   * @param last Last position (exclusive)
   */
  CTP_INLINE_CROSS_FUN
  void DestroyRange(size_type first, size_type last) {
    if constexpr (!kIsPod) {
      for (size_type i = first; i < last; ++i) {
        Destroy(i);
      }
    }
  }

 public:
  /**
   * Re-seat data_.ptr_ to this object's inline SVO buffer after a memcpy.
   * Call after cudaMemcpy/memcpy when the vector was using SVO (size <= SVO_SIZE).
   * Preserves size_ and the data already in svo_[].
   */
  CTP_INLINE_CROSS_FUN
  void FixupSvoPtr() {
    data_.ptr_ = svo_data();
  }

  /**
   * Default constructor (no allocator).
   * Creates an empty vector with SVO buffer available.
   */
  CTP_INLINE_CROSS_FUN
  vector()
    : size_(0), alloc_(nullptr) {
    InitSvo();
  }

  /**
   * Constructor with allocator.
   * Creates an empty vector with SVO buffer available.
   *
   * @param alloc Pointer to allocator instance for memory management
   */
  CTP_INLINE_CROSS_FUN
  explicit vector(AllocT* alloc)
    : size_(0), alloc_(alloc) {
    InitSvo();
  }

  /**
   * Destructor.
   * Destroys all elements and deallocates heap memory if not using SVO.
   */
  CTP_INLINE_CROSS_FUN
  ~vector() {
    clear();
    if (!IsUsingSvo() && !data_.IsNull() && alloc_ != nullptr) {
      alloc_->Free(data_);
    }
  }

  /**
   * Constructor with default initialization (single size_type parameter).
   * When called with one size_type argument, creates vector with specified
   * number of default-initialized elements.
   *
   * @param count Number of elements to create with default values
   * @param alloc Pointer to allocator instance for memory management
   */
  CTP_INLINE_CROSS_FUN
  explicit vector(size_type count, AllocT* alloc)
      : size_(0), alloc_(alloc) {
    InitSvo();
    reserve(count);
    if constexpr (!kIsPod) {
      for (size_type i = 0; i < count; ++i) {
        new (&data_.ptr_[size_]) T();
        ++size_;
      }
    } else {
      memset(data_.ptr_, 0, count * sizeof(T));
      size_ = count;
    }
  }

  /**
   * Constructor with fill initialization.
   * Creates a vector with count elements, all initialized to value.
   *
   * @param count Number of elements to create
   * @param value Value to initialize elements with
   * @param alloc Pointer to allocator instance for memory management
   */
  CTP_INLINE_CROSS_FUN
  vector(size_type count, const T& value, AllocT* alloc)
      : size_(0), alloc_(alloc) {
    InitSvo();
    reserve(count);
    for (size_type i = 0; i < count; ++i) {
      push_back(value);
    }
  }

  /**
   * Constructor with initializer list.
   * Creates a vector from an initializer list.
   *
   * @param init Initializer list
   * @param alloc Pointer to allocator instance for memory management
   */
  CTP_INLINE_CROSS_FUN
  vector(std::initializer_list<T> init, AllocT* alloc)
      : size_(0), alloc_(alloc) {
    InitSvo();
    reserve(init.size());
    for (const auto& val : init) {
      push_back(val);
    }
  }

  /**
   * Copy constructor.
   * Creates a copy of another vector, sharing the same allocator.
   *
   * @param other Vector to copy from
   */
  CTP_INLINE_CROSS_FUN
  vector(const vector& other)
      : size_(0), alloc_(other.alloc_) {
    InitSvo();
    if (alloc_ != nullptr || other.size_ <= SVO_SIZE) {
      reserve(other.size_);
      for (const auto& val : other) {
        push_back(val);
      }
    }
  }

  /**
   * Move constructor.
   * Transfers ownership of data from other vector to this one.
   * If source uses SVO, elements are moved to this vector's SVO buffer.
   *
   * @param other Vector to move from
   */
  CTP_INLINE_CROSS_FUN
  vector(vector&& other) noexcept
      : size_(0), alloc_(other.alloc_) {
    if (other.IsUsingSvo()) {
      InitSvo();
      if constexpr (kIsPod) {
        memcpy(svo_, other.svo_, other.size_ * sizeof(T));
      } else {
        for (size_type i = 0; i < other.size_; ++i) {
          new (&data_.ptr_[i]) T(std::move(other.data_.ptr_[i]));
          other.Destroy(i);
        }
      }
      size_ = other.size_;
      other.size_ = 0;
    } else {
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.InitSvo();
      other.size_ = 0;
      // Keep other.alloc_ intact: InitSvo() already left `other` in a
      // valid empty state whose destructor is a no-op (IsUsingSvo()), so
      // clearing alloc_ is unnecessary for safety -- and harmful, since
      // reserve() treats alloc_==nullptr as "no-op success", which would
      // make a moved-from vector silently unable to grow (its next
      // resize() would write past the SVO buffer). Retaining the
      // allocator keeps the moved-from vector reusable, matching
      // std::vector semantics and the move-then-reuse pattern in
      // unordered_map_ll::rehash_no_lock().
    }
  }

  /**
   * Copy assignment operator.
   * Replaces this vector's contents with a copy of other's contents.
   *
   * @param other Vector to copy from
   * @return Reference to this vector
   */
  CTP_INLINE_CROSS_FUN
  vector& operator=(const vector& other) {
    if (this != &other) {
      clear();
      if (!IsUsingSvo() && !data_.IsNull() && alloc_ != nullptr) {
        alloc_->Free(data_);
      }
      alloc_ = other.alloc_;
      InitSvo();
      if (alloc_ != nullptr || other.size_ <= SVO_SIZE) {
        reserve(other.size_);
        for (const auto& val : other) {
          push_back(val);
        }
      }
    }
    return *this;
  }

  /**
   * Move assignment operator.
   * Transfers ownership of data from other vector to this one.
   *
   * @param other Vector to move from
   * @return Reference to this vector
   */
  CTP_INLINE_CROSS_FUN
  vector& operator=(vector&& other) noexcept {
    if (this != &other) {
      clear();
      if (!IsUsingSvo() && !data_.IsNull() && alloc_ != nullptr) {
        alloc_->Free(data_);
      }
      alloc_ = other.alloc_;
      if (other.IsUsingSvo()) {
        InitSvo();
        if constexpr (kIsPod) {
          memcpy(svo_, other.svo_, other.size_ * sizeof(T));
        } else {
          for (size_type i = 0; i < other.size_; ++i) {
            new (&data_.ptr_[i]) T(std::move(other.data_.ptr_[i]));
            other.Destroy(i);
          }
        }
        size_ = other.size_;
        other.size_ = 0;
      } else {
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.InitSvo();
        other.size_ = 0;
        // Keep other.alloc_ intact (see move-ctor rationale): clearing it
        // would make the moved-from vector silently unable to grow,
        // overflowing its SVO buffer on the next resize().
      }
    }
    return *this;
  }

  /**
   * Initializer list assignment operator
   *
   * @param init Initializer list
   * @return Reference to this vector
   */
  CTP_INLINE_CROSS_FUN
  vector& operator=(std::initializer_list<T> init) {
    clear();
    reserve(init.size());
    for (const auto& val : init) {
      push_back(val);
    }
    return *this;
  }

  /**
   * Get element at position with bounds checking.
   * Throws std::out_of_range if position is out of bounds.
   *
   * @param pos Position to access
   * @return Reference to element at position
   * @throws std::out_of_range if position is out of bounds
   */
  CTP_INLINE_CROSS_FUN
  T& at(size_type pos) {
    if (pos >= size_) {
      CTP_THROW(std::out_of_range("Vector index out of bounds"));
    }
    return data_.ptr_[pos];
  }

  /**
   * Get const element at position with bounds checking.
   * Throws std::out_of_range if position is out of bounds.
   *
   * @param pos Position to access
   * @return Const reference to element at position
   * @throws std::out_of_range if position is out of bounds
   */
  CTP_INLINE_CROSS_FUN
  const T& at(size_type pos) const {
    if (pos >= size_) {
      CTP_THROW(std::out_of_range("Vector index out of bounds"));
    }
    return data_.ptr_[pos];
  }

  /**
   * Subscript operator without bounds checking.
   * Provides fast unchecked access to elements.
   *
   * @param pos Position to access
   * @return Reference to element at position
   */
  CTP_INLINE_CROSS_FUN
  T& operator[](size_type pos) {
    return data_.ptr_[pos];
  }

  /**
   * Const subscript operator without bounds checking.
   * Provides fast unchecked access to const elements.
   *
   * @param pos Position to access
   * @return Const reference to element at position
   */
  CTP_INLINE_CROSS_FUN
  const T& operator[](size_type pos) const {
    return data_.ptr_[pos];
  }

  /**
   * Get reference to first element.
   * Behavior is undefined if vector is empty.
   *
   * @return Reference to first element
   */
  CTP_INLINE_CROSS_FUN
  T& front() {
    return data_.ptr_[0];
  }

  /**
   * Get const reference to first element.
   * Behavior is undefined if vector is empty.
   *
   * @return Const reference to first element
   */
  CTP_INLINE_CROSS_FUN
  const T& front() const {
    return data_.ptr_[0];
  }

  /**
   * Get reference to last element.
   * Behavior is undefined if vector is empty.
   *
   * @return Reference to last element
   */
  CTP_INLINE_CROSS_FUN
  T& back() {
    return data_.ptr_[size_ - 1];
  }

  /**
   * Get const reference to last element.
   * Behavior is undefined if vector is empty.
   *
   * @return Const reference to last element
   */
  CTP_INLINE_CROSS_FUN
  const T& back() const {
    return data_.ptr_[size_ - 1];
  }

  /**
   * Get pointer to data.
   * Returns the private memory pointer for direct element access.
   *
   * @return Pointer to underlying data array
   */
  CTP_INLINE_CROSS_FUN
  T* data() {
    return data_.ptr_;
  }

  /**
   * Get const pointer to data.
   * Returns the private memory pointer for direct const element access.
   *
   * @return Const pointer to underlying data array
   */
  CTP_INLINE_CROSS_FUN
  const T* data() const {
    return data_.ptr_;
  }

  /**
   * Get iterator to beginning.
   * Returns an iterator pointing to the first element.
   *
   * @return Iterator to first element
   */
  CTP_INLINE_CROSS_FUN
  iterator begin() {
    return iterator(data_.ptr_);
  }

  /**
   * Get const iterator to beginning.
   * Returns a const iterator pointing to the first element.
   *
   * @return Const iterator to first element
   */
  CTP_INLINE_CROSS_FUN
  const_iterator begin() const {
    return const_iterator(data_.ptr_);
  }

  /**
   * Get const iterator to beginning.
   * Returns a const iterator pointing to the first element.
   *
   * @return Const iterator to first element
   */
  CTP_INLINE_CROSS_FUN
  const_iterator cbegin() const {
    return const_iterator(data_.ptr_);
  }

  /**
   * Get iterator to end.
   * Returns an iterator pointing to one past the last element.
   *
   * @return Iterator to one past last element
   */
  CTP_INLINE_CROSS_FUN
  iterator end() {
    return iterator(data_.ptr_ + size_);
  }

  /**
   * Get const iterator to end.
   * Returns a const iterator pointing to one past the last element.
   *
   * @return Const iterator to one past last element
   */
  CTP_INLINE_CROSS_FUN
  const_iterator end() const {
    return const_iterator(data_.ptr_ + size_);
  }

  /**
   * Get const iterator to end.
   * Returns a const iterator pointing to one past the last element.
   *
   * @return Const iterator to one past last element
   */
  CTP_INLINE_CROSS_FUN
  const_iterator cend() const {
    return const_iterator(data_.ptr_ + size_);
  }

  /**
   * Get reverse iterator to beginning
   *
   * @return Reverse iterator to last element
   */
  CTP_INLINE_CROSS_FUN
  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }

  /**
   * Get const reverse iterator to beginning
   *
   * @return Const reverse iterator to last element
   */
  CTP_INLINE_CROSS_FUN
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  /**
   * Get const reverse iterator to beginning
   *
   * @return Const reverse iterator to last element
   */
  CTP_INLINE_CROSS_FUN
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(end());
  }

  /**
   * Get reverse iterator to end
   *
   * @return Reverse iterator to one before first element
   */
  CTP_INLINE_CROSS_FUN
  reverse_iterator rend() {
    return reverse_iterator(begin());
  }

  /**
   * Get const reverse iterator to end
   *
   * @return Const reverse iterator to one before first element
   */
  CTP_INLINE_CROSS_FUN
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  /**
   * Get const reverse iterator to end
   *
   * @return Const reverse iterator to one before first element
   */
  CTP_INLINE_CROSS_FUN
  const_reverse_iterator crend() const {
    return const_reverse_iterator(begin());
  }

  /**
   * Check if vector is empty
   *
   * @return True if size is zero
   */
  CTP_INLINE_CROSS_FUN
  bool empty() const {
    return size_ == 0;
  }

  /**
   * Get number of elements in vector
   *
   * @return Current size
   */
  CTP_INLINE_CROSS_FUN
  size_type size() const {
    return size_;
  }

  /**
   * Get allocated capacity
   *
   * @return Current capacity
   */
  CTP_INLINE_CROSS_FUN
  size_type capacity() const {
    return capacity_;
  }

  /**
   * Reserve capacity for elements.
   * If new_capacity fits within SVO, no allocation occurs.
   * Otherwise allocates new memory using the allocator, copying
   * existing elements from SVO or old heap allocation.
   *
   * @param new_capacity Desired capacity
   */
  CTP_INLINE_CROSS_FUN
  bool reserve(size_type new_capacity) {
    if (new_capacity <= capacity_ || alloc_ == nullptr) {
      return true;
    }

    auto new_data = alloc_->template AllocateObjs<T>(new_capacity);
    if (new_data.IsNull()) {
      return false;
    }

    // Copy existing elements from current buffer (SVO or heap)
    if (size_ > 0) {
      if constexpr (kIsPod) {
        memcpy(new_data.ptr_, data_.ptr_, size_ * sizeof(T));
      } else {
        for (size_type i = 0; i < size_; ++i) {
          new (&new_data.ptr_[i]) T(std::move(data_.ptr_[i]));
          Destroy(i);
        }
      }
    }

    // Free old heap allocation (SVO has IsNull()==true, so this is skipped)
    if (!data_.IsNull()) {
      alloc_->Free(data_);
    }

    data_ = new_data;
    capacity_ = new_capacity;
    return true;
  }

  /**
   * Shrink capacity to match size.
   * If on heap, reduces capacity to match current size.
   * If size fits in SVO, moves data back to SVO buffer.
   */
  CTP_INLINE_CROSS_FUN
  void shrink_to_fit() {
    if (size_ <= capacity_ && alloc_ != nullptr) {
      if (size_ == 0 && !IsUsingSvo()) {
        if (!data_.IsNull()) {
          alloc_->Free(data_);
        }
        InitSvo();
      } else if (!IsUsingSvo() && size_ <= SVO_SIZE) {
        // Move from heap back to SVO
        if constexpr (kIsPod) {
          memcpy(svo_, data_.ptr_, size_ * sizeof(T));
        } else {
          for (size_type i = 0; i < size_; ++i) {
            new (svo_data() + i) T(std::move(data_.ptr_[i]));
            Destroy(i);
          }
        }
        alloc_->Free(data_);
        data_ = ctp::ipc::FullPtr<T>::GetNull();
        data_.ptr_ = svo_data();
        capacity_ = SVO_SIZE;
      } else if (!IsUsingSvo() && size_ < capacity_) {
        auto new_data = alloc_->template AllocateObjs<T>(size_);
        if constexpr (kIsPod) {
          memcpy(new_data.ptr_, data_.ptr_, size_ * sizeof(T));
        } else {
          for (size_type i = 0; i < size_; ++i) {
            new (&new_data.ptr_[i]) T(std::move(data_.ptr_[i]));
            Destroy(i);
          }
        }
        alloc_->Free(data_);
        data_ = new_data;
        capacity_ = size_;
      }
    }
  }

  /**
   * Add element to end of vector
   *
   * @param val Element to add
   */
  CTP_INLINE_CROSS_FUN
  void push_back(const T& val) {
    if (size_ >= capacity_) {
      Grow(size_ + 1);
    }
    ConstructCopy(size_, val);
    ++size_;
  }

  /**
   * Add element to end of vector with move semantics
   *
   * @param val Element to add
   */
  CTP_INLINE_CROSS_FUN
  void push_back(T&& val) {
    if (size_ >= capacity_) {
      Grow(size_ + 1);
    }
    ConstructMove(size_, std::move(val));
    ++size_;
  }

  /**
   * Remove last element from vector
   */
  CTP_INLINE_CROSS_FUN
  void pop_back() {
    if (size_ > 0) {
      Destroy(size_ - 1);
      --size_;
    }
  }

  /**
   * Clear all elements from vector
   */
  CTP_INLINE_CROSS_FUN
  void clear() {
    DestroyRange(0, size_);
    size_ = 0;
  }

  /**
   * Insert element at position.
   * Inserts a single element, shifting existing elements as needed.
   *
   * @param pos Iterator to insertion position
   * @param val Value to insert
   * @return Iterator to inserted element
   */
  CTP_INLINE_CROSS_FUN
  iterator insert(const_iterator pos, const T& val) {
    size_type idx = pos.get() - data_.ptr_;
    if (size_ >= capacity_) {
      Grow(size_ + 1);
    }

    if constexpr (kIsPod) {
      memmove(&data_.ptr_[idx + 1], &data_.ptr_[idx], (size_ - idx) * sizeof(T));
      data_.ptr_[idx] = val;
    } else {
      for (size_type i = size_; i > idx; --i) {
        new (&data_.ptr_[i]) T(std::move(data_.ptr_[i - 1]));
        Destroy(i - 1);
      }
      new (&data_.ptr_[idx]) T(val);
    }

    ++size_;
    return iterator(&data_.ptr_[idx]);
  }

  /**
   * Insert element at position with move semantics.
   * Inserts a single element using move semantics, shifting existing elements.
   *
   * @param pos Iterator to insertion position
   * @param val Value to insert
   * @return Iterator to inserted element
   */
  CTP_INLINE_CROSS_FUN
  iterator insert(const_iterator pos, T&& val) {
    size_type idx = pos.get() - data_.ptr_;
    if (size_ >= capacity_) {
      Grow(size_ + 1);
    }

    if constexpr (kIsPod) {
      memmove(&data_.ptr_[idx + 1], &data_.ptr_[idx], (size_ - idx) * sizeof(T));
      data_.ptr_[idx] = std::move(val);
    } else {
      for (size_type i = size_; i > idx; --i) {
        new (&data_.ptr_[i]) T(std::move(data_.ptr_[i - 1]));
        Destroy(i - 1);
      }
      new (&data_.ptr_[idx]) T(std::move(val));
    }

    ++size_;
    return iterator(&data_.ptr_[idx]);
  }

  /**
   * Insert range of elements at position.
   * Inserts multiple elements from a range, shifting existing elements.
   *
   * @param pos Iterator to insertion position
   * @param first Iterator to first element to insert
   * @param last Iterator to one past last element to insert
   * @return Iterator to first inserted element
   */
  CTP_INLINE_CROSS_FUN
  iterator insert(const_iterator pos, const_iterator first,
                  const_iterator last) {
    size_type idx = pos.get() - data_.ptr_;
    size_type count = last.get() - first.get();

    if (size_ + count > capacity_) {
      Grow(size_ + count);
    }

    if constexpr (kIsPod) {
      memmove(&data_.ptr_[idx + count], &data_.ptr_[idx], (size_ - idx) * sizeof(T));
      memcpy(&data_.ptr_[idx], first.get(), count * sizeof(T));
    } else {
      for (size_type i = size_ + count - 1; i >= idx + count; --i) {
        new (&data_.ptr_[i]) T(std::move(data_.ptr_[i - count]));
        Destroy(i - count);
      }
      size_type src_idx = 0;
      for (size_type i = idx; i < idx + count; ++i, ++src_idx) {
        new (&data_.ptr_[i]) T(*(first.get() + src_idx));
      }
    }

    size_ += count;
    return iterator(&data_.ptr_[idx]);
  }

  /**
   * Erase element at position.
   * Removes the element at pos and shifts remaining elements back.
   *
   * @param pos Iterator to element to erase
   * @return Iterator to element following erased element
   */
  CTP_INLINE_CROSS_FUN
  iterator erase(const_iterator pos) {
    size_type idx = pos.get() - data_.ptr_;

    if constexpr (kIsPod) {
      memmove(&data_.ptr_[idx], &data_.ptr_[idx + 1], (size_ - idx - 1) * sizeof(T));
    } else {
      Destroy(idx);
      for (size_type i = idx; i < size_ - 1; ++i) {
        new (&data_.ptr_[i]) T(std::move(data_.ptr_[i + 1]));
        Destroy(i + 1);
      }
    }

    --size_;
    return iterator(&data_.ptr_[idx]);
  }

  /**
   * Erase range of elements.
   * Removes all elements in range [first, last) and shifts remaining elements.
   *
   * @param first Iterator to first element to erase
   * @param last Iterator to one past last element to erase
   * @return Iterator to element following erased elements
   */
  CTP_INLINE_CROSS_FUN
  iterator erase(const_iterator first, const_iterator last) {
    size_type first_idx = first.get() - data_.ptr_;
    size_type last_idx = last.get() - data_.ptr_;
    size_type count = last_idx - first_idx;

    if constexpr (kIsPod) {
      memmove(&data_.ptr_[first_idx], &data_.ptr_[last_idx],
                   (size_ - last_idx) * sizeof(T));
    } else {
      DestroyRange(first_idx, last_idx);
      for (size_type i = first_idx; i < size_ - count; ++i) {
        new (&data_.ptr_[i]) T(std::move(data_.ptr_[i + count]));
        Destroy(i + count);
      }
    }

    size_ -= count;
    return iterator(&data_.ptr_[first_idx]);
  }

  /**
   * Resize vector without initializing new elements.
   * Use only when caller will immediately overwrite the new region.
   *
   * @param new_size New size
   */
  CTP_INLINE_CROSS_FUN
  bool resize_no_init(size_type new_size) {
    if (new_size > capacity_) {
      if (!reserve(new_size)) return false;
    }
    size_ = new_size;
    return true;
  }

  /**
   * Resize vector to specified size.
   * Grows or shrinks the vector to the new size, default-initializing new elements.
   *
   * @param new_size New size
   * @return true on success, false if allocation failed
   */
  CTP_INLINE_CROSS_FUN
  bool resize(size_type new_size) {
    if (new_size > size_) {
      if (new_size > capacity_) {
        if (!reserve(new_size)) return false;
      }
      if constexpr (!kIsPod) {
        for (size_type i = size_; i < new_size; ++i) {
          new (&data_.ptr_[i]) T();
        }
      } else {
        memset(data_.ptr_ + size_, 0, (new_size - size_) * sizeof(T));
      }
      size_ = new_size;
    } else if (new_size < size_) {
      DestroyRange(new_size, size_);
      size_ = new_size;
    }
    return true;
  }

  /**
   * Resize vector to specified size with fill value.
   * Grows or shrinks the vector to the new size, initializing new elements to value.
   *
   * @param new_size New size
   * @param value Value to fill new elements with
   */
  CTP_INLINE_CROSS_FUN
  bool resize(size_type new_size, const T& value) {
    if (new_size > size_) {
      if (new_size > capacity_) {
        if (!reserve(new_size)) return false;
      }
      for (size_type i = size_; i < new_size; ++i) {
        ConstructCopy(i, value);
      }
      size_ = new_size;
    } else if (new_size < size_) {
      DestroyRange(new_size, size_);
      size_ = new_size;
    }
    return true;
  }

  /**
   * Swap contents with another vector.
   * Handles SVO and heap combinations correctly.
   *
   * @param other Vector to swap with
   */
  CTP_INLINE_CROSS_FUN
  void swap(vector& other) noexcept {
    if (!IsUsingSvo() && !other.IsUsingSvo()) {
      // Both on heap — simple pointer swap
      std::swap(data_, other.data_);
      std::swap(size_, other.size_);
      std::swap(capacity_, other.capacity_);
      std::swap(alloc_, other.alloc_);
    } else {
      // At least one is SVO — use move through temporary
      vector temp(std::move(*this));
      *this = std::move(other);
      other = std::move(temp);
    }
  }

  /**
   * Serialize vector to archive.
   * Uses cereal serialization framework to save vector data.
   *
   * @tparam Archive Cereal archive type
   * @param ar Archive to save to
   */
  template<class Archive>
  CTP_INLINE_CROSS_FUN
  void save(Archive& ar) const {
    ctp::ipc::save_vec<Archive, vector<T, AllocT, SVO_SIZE>, T>(ar, *this);
  }

  /**
   * Deserialize vector from archive.
   * Uses cereal serialization framework to load vector data.
   *
   * @tparam Archive Cereal archive type
   * @param ar Archive to load from
   */
  template<class Archive>
  CTP_INLINE_CROSS_FUN
  void load(Archive& ar) {
    ctp::ipc::load_vec<Archive, vector<T, AllocT, SVO_SIZE>, T>(ar, *this);
  }
};

}  // namespace ctp::priv

#endif  // CTP_DATA_STRUCTURES_PRIV_VECTOR_H_
