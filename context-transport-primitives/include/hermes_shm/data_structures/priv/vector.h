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

#ifndef HSHM_DATA_STRUCTURES_PRIV_VECTOR_H_
#define HSHM_DATA_STRUCTURES_PRIV_VECTOR_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/numbers.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/data_structures/serialization/serialize_common.h"
#include <cstring>
#include <iterator>
#include <type_traits>
#include <stdexcept>
#include <algorithm>

namespace hshm::priv {

/**
 * Private-memory vector container with allocator integration
 *
 * This vector class provides dynamic array functionality for private memory,
 * using the library's allocator API with FullPtr for proper memory management.
 * It supports both POD (Plain Old Data) types and complex class types.
 * POD types use optimized memcpy/memset operations for better performance.
 * The vector integrates with the library's memory allocation system.
 *
 * @tparam T The element type
 * @tparam AllocT The allocator type (must have Allocate/AllocateObjs/Free methods)
 */
template<typename T, typename AllocT>
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
    HSHM_INLINE_CROSS_FUN
    iterator() : ptr_(nullptr) {}

    /**
     * Construct from pointer
     *
     * @param ptr The element pointer
     */
    HSHM_INLINE_CROSS_FUN
    explicit iterator(T *ptr) : ptr_(ptr) {}

    /**
     * Dereference operator
     *
     * @return Reference to the current element
     */
    HSHM_INLINE_CROSS_FUN
    T& operator*() const { return *ptr_; }

    /**
     * Arrow operator
     *
     * @return Pointer to the current element
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
     * @param n Number of elements to advance
     * @return New iterator advanced by n positions
     */
    HSHM_INLINE_CROSS_FUN
    iterator operator+(difference_type n) const {
      return iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n Number of elements to go back
     * @return New iterator moved back by n positions
     */
    HSHM_INLINE_CROSS_FUN
    iterator operator-(difference_type n) const {
      return iterator(ptr_ - n);
    }

    /**
     * Addition assignment operator
     *
     * @param n Number of elements to advance
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
     * @param n Number of elements to go back
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
     * @return The number of elements between this and other
     */
    HSHM_INLINE_CROSS_FUN
    difference_type operator-(const iterator& other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n Index offset from current position
     * @return Reference to the element at offset n
     */
    HSHM_INLINE_CROSS_FUN
    T& operator[](difference_type n) const {
      return ptr_[n];
    }

    /**
     * Equality comparison operator
     *
     * @param other Another iterator
     * @return True if both iterators point to the same element
     */
    HSHM_INLINE_CROSS_FUN
    bool operator==(const iterator& other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality comparison operator
     *
     * @param other Another iterator
     * @return True if iterators point to different elements
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
    HSHM_INLINE_CROSS_FUN
    const_iterator() : ptr_(nullptr) {}

    /**
     * Construct from pointer
     *
     * @param ptr The element pointer
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
     * @return Const reference to the current element
     */
    HSHM_INLINE_CROSS_FUN
    const T& operator*() const { return *ptr_; }

    /**
     * Arrow operator
     *
     * @return Const pointer to the current element
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
     * @param n Number of elements to advance
     * @return New iterator advanced by n positions
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator operator+(difference_type n) const {
      return const_iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n Number of elements to go back
     * @return New iterator moved back by n positions
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator operator-(difference_type n) const {
      return const_iterator(ptr_ - n);
    }

    /**
     * Addition assignment operator
     *
     * @param n Number of elements to advance
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
     * @param n Number of elements to go back
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
     * @return The number of elements between this and other
     */
    HSHM_INLINE_CROSS_FUN
    difference_type operator-(const const_iterator& other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n Index offset from current position
     * @return Const reference to the element at offset n
     */
    HSHM_INLINE_CROSS_FUN
    const T& operator[](difference_type n) const {
      return ptr_[n];
    }

    /**
     * Equality comparison operator
     *
     * @param other Another iterator
     * @return True if both iterators point to the same element
     */
    HSHM_INLINE_CROSS_FUN
    bool operator==(const const_iterator& other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality comparison operator
     *
     * @param other Another iterator
     * @return True if iterators point to different elements
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
   * Reverse iterator for vector
   */
  using reverse_iterator = std::reverse_iterator<iterator>;

  /**
   * Const reverse iterator for vector
   */
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  hipc::FullPtr<T> data_;  /**< FullPtr to the data array (both private and shared) */
  size_type size_;         /**< Current number of elements */
  size_type capacity_;     /**< Allocated capacity */
  AllocT* alloc_;          /**< Pointer to allocator for memory management */

  /**
   * Helper to determine if type is POD
   */
  static constexpr bool kIsPod = std::is_trivial_v<T>;

  /**
   * Grow the capacity to accommodate new elements.
   * Doubles capacity until it's large enough for min_capacity.
   *
   * @param min_capacity Minimum capacity needed
   */
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
  void DestroyRange(size_type first, size_type last) {
    if constexpr (!kIsPod) {
      for (size_type i = first; i < last; ++i) {
        Destroy(i);
      }
    }
  }

 public:
  /**
   * Default constructor.
   * Creates an empty vector with zero capacity.
   *
   * @param alloc Pointer to allocator instance for memory management
   */
  explicit vector(AllocT* alloc)
    : data_(hipc::FullPtr<T>::GetNull()), size_(0), capacity_(0), alloc_(alloc) {}

  /**
   * Destructor.
   * Destroys all elements and deallocates memory using allocator.
   * Note: Does not deallocate the allocator itself (managed externally).
   */
  ~vector() {
    clear();
    if (!data_.IsNull() && alloc_ != nullptr) {
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
  explicit vector(size_type count, AllocT* alloc)
      : data_(hipc::FullPtr<T>::GetNull()), size_(0), capacity_(0), alloc_(alloc) {
    reserve(count);
    for (size_type i = 0; i < count; ++i) {
      if constexpr (kIsPod) {
        data_.ptr_[size_] = T();
      } else {
        new (&data_.ptr_[size_]) T();
      }
      ++size_;
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
  vector(size_type count, const T& value, AllocT* alloc)
      : data_(hipc::FullPtr<T>::GetNull()), size_(0), capacity_(0), alloc_(alloc) {
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
  vector(std::initializer_list<T> init, AllocT* alloc)
      : data_(hipc::FullPtr<T>::GetNull()), size_(0), capacity_(0), alloc_(alloc) {
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
  vector(const vector& other)
      : data_(hipc::FullPtr<T>::GetNull()), size_(0), capacity_(0), alloc_(other.alloc_) {
    if (alloc_ != nullptr) {
      reserve(other.size_);
      for (const auto& val : other) {
        push_back(val);
      }
    }
  }

  /**
   * Move constructor.
   * Transfers ownership of data from other vector to this one.
   *
   * @param other Vector to move from
   */
  vector(vector&& other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_),
        alloc_(other.alloc_) {
    other.data_ = hipc::FullPtr<T>::GetNull();
    other.size_ = 0;
    other.capacity_ = 0;
    other.alloc_ = nullptr;
  }

  /**
   * Copy assignment operator.
   * Replaces this vector's contents with a copy of other's contents.
   *
   * @param other Vector to copy from
   * @return Reference to this vector
   */
  vector& operator=(const vector& other) {
    if (this != &other) {
      clear();
      alloc_ = other.alloc_;
      if (alloc_ != nullptr) {
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
  vector& operator=(vector&& other) noexcept {
    if (this != &other) {
      clear();
      if (!data_.IsNull() && alloc_ != nullptr) {
        alloc_->Free(data_);
      }
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      alloc_ = other.alloc_;
      other.data_ = hipc::FullPtr<T>::GetNull();
      other.size_ = 0;
      other.capacity_ = 0;
      other.alloc_ = nullptr;
    }
    return *this;
  }

  /**
   * Initializer list assignment operator
   *
   * @param init Initializer list
   * @return Reference to this vector
   */
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
  T& at(size_type pos) {
    if (pos >= size_) {
      throw std::out_of_range("Vector index out of bounds");
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
  const T& at(size_type pos) const {
    if (pos >= size_) {
      throw std::out_of_range("Vector index out of bounds");
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
  const T& operator[](size_type pos) const {
    return data_.ptr_[pos];
  }

  /**
   * Get reference to first element.
   * Behavior is undefined if vector is empty.
   *
   * @return Reference to first element
   */
  T& front() {
    return data_.ptr_[0];
  }

  /**
   * Get const reference to first element.
   * Behavior is undefined if vector is empty.
   *
   * @return Const reference to first element
   */
  const T& front() const {
    return data_.ptr_[0];
  }

  /**
   * Get reference to last element.
   * Behavior is undefined if vector is empty.
   *
   * @return Reference to last element
   */
  T& back() {
    return data_.ptr_[size_ - 1];
  }

  /**
   * Get const reference to last element.
   * Behavior is undefined if vector is empty.
   *
   * @return Const reference to last element
   */
  const T& back() const {
    return data_.ptr_[size_ - 1];
  }

  /**
   * Get pointer to data.
   * Returns the private memory pointer for direct element access.
   *
   * @return Pointer to underlying data array
   */
  T* data() {
    return data_.ptr_;
  }

  /**
   * Get const pointer to data.
   * Returns the private memory pointer for direct const element access.
   *
   * @return Const pointer to underlying data array
   */
  const T* data() const {
    return data_.ptr_;
  }

  /**
   * Get iterator to beginning.
   * Returns an iterator pointing to the first element.
   *
   * @return Iterator to first element
   */
  iterator begin() {
    return iterator(data_.ptr_);
  }

  /**
   * Get const iterator to beginning.
   * Returns a const iterator pointing to the first element.
   *
   * @return Const iterator to first element
   */
  const_iterator begin() const {
    return const_iterator(data_.ptr_);
  }

  /**
   * Get const iterator to beginning.
   * Returns a const iterator pointing to the first element.
   *
   * @return Const iterator to first element
   */
  const_iterator cbegin() const {
    return const_iterator(data_.ptr_);
  }

  /**
   * Get iterator to end.
   * Returns an iterator pointing to one past the last element.
   *
   * @return Iterator to one past last element
   */
  iterator end() {
    return iterator(data_.ptr_ + size_);
  }

  /**
   * Get const iterator to end.
   * Returns a const iterator pointing to one past the last element.
   *
   * @return Const iterator to one past last element
   */
  const_iterator end() const {
    return const_iterator(data_.ptr_ + size_);
  }

  /**
   * Get const iterator to end.
   * Returns a const iterator pointing to one past the last element.
   *
   * @return Const iterator to one past last element
   */
  const_iterator cend() const {
    return const_iterator(data_.ptr_ + size_);
  }

  /**
   * Get reverse iterator to beginning
   *
   * @return Reverse iterator to last element
   */
  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }

  /**
   * Get const reverse iterator to beginning
   *
   * @return Const reverse iterator to last element
   */
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  /**
   * Get const reverse iterator to beginning
   *
   * @return Const reverse iterator to last element
   */
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(end());
  }

  /**
   * Get reverse iterator to end
   *
   * @return Reverse iterator to one before first element
   */
  reverse_iterator rend() {
    return reverse_iterator(begin());
  }

  /**
   * Get const reverse iterator to end
   *
   * @return Const reverse iterator to one before first element
   */
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  /**
   * Get const reverse iterator to end
   *
   * @return Const reverse iterator to one before first element
   */
  const_reverse_iterator crend() const {
    return const_reverse_iterator(begin());
  }

  /**
   * Check if vector is empty
   *
   * @return True if size is zero
   */
  bool empty() const {
    return size_ == 0;
  }

  /**
   * Get number of elements in vector
   *
   * @return Current size
   */
  size_type size() const {
    return size_;
  }

  /**
   * Get allocated capacity
   *
   * @return Current capacity
   */
  size_type capacity() const {
    return capacity_;
  }

  /**
   * Reserve capacity for elements.
   * Allocates new memory if new_capacity exceeds current capacity.
   * Uses allocator's AllocateObjs method for proper memory management.
   *
   * @param new_capacity Desired capacity
   */
  void reserve(size_type new_capacity) {
    if (new_capacity <= capacity_ || alloc_ == nullptr) {
      return;
    }

    auto new_data = alloc_->template AllocateObjs<T>(new_capacity);

    if (!data_.IsNull()) {
      if constexpr (kIsPod) {
        std::memcpy(new_data.ptr_, data_.ptr_, size_ * sizeof(T));
      } else {
        for (size_type i = 0; i < size_; ++i) {
          new (&new_data.ptr_[i]) T(std::move(data_.ptr_[i]));
          Destroy(i);
        }
      }
      alloc_->Free(data_);
    }

    data_ = new_data;
    capacity_ = new_capacity;
  }

  /**
   * Shrink capacity to match size.
   * Reduces capacity to match current size, freeing unused memory.
   * Uses allocator's AllocateObjs and Free methods for proper management.
   */
  void shrink_to_fit() {
    if (size_ < capacity_ && alloc_ != nullptr) {
      if (size_ == 0) {
        if (!data_.IsNull()) {
          alloc_->Free(data_);
          data_ = hipc::FullPtr<T>::GetNull();
        }
        capacity_ = 0;
      } else {
        auto new_data = alloc_->template AllocateObjs<T>(size_);

        if constexpr (kIsPod) {
          std::memcpy(new_data.ptr_, data_.ptr_, size_ * sizeof(T));
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
  void pop_back() {
    if (size_ > 0) {
      Destroy(size_ - 1);
      --size_;
    }
  }

  /**
   * Clear all elements from vector
   */
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
  iterator insert(const_iterator pos, const T& val) {
    size_type idx = pos.get() - data_.ptr_;
    if (size_ >= capacity_) {
      Grow(size_ + 1);
    }

    if constexpr (kIsPod) {
      std::memmove(&data_.ptr_[idx + 1], &data_.ptr_[idx], (size_ - idx) * sizeof(T));
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
  iterator insert(const_iterator pos, T&& val) {
    size_type idx = pos.get() - data_.ptr_;
    if (size_ >= capacity_) {
      Grow(size_ + 1);
    }

    if constexpr (kIsPod) {
      std::memmove(&data_.ptr_[idx + 1], &data_.ptr_[idx], (size_ - idx) * sizeof(T));
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
  iterator insert(const_iterator pos, const_iterator first,
                  const_iterator last) {
    size_type idx = pos.get() - data_.ptr_;
    size_type count = last.get() - first.get();

    if (size_ + count > capacity_) {
      Grow(size_ + count);
    }

    if constexpr (kIsPod) {
      std::memmove(&data_.ptr_[idx + count], &data_.ptr_[idx], (size_ - idx) * sizeof(T));
      std::memcpy(&data_.ptr_[idx], first.get(), count * sizeof(T));
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
  iterator erase(const_iterator pos) {
    size_type idx = pos.get() - data_.ptr_;

    if constexpr (kIsPod) {
      std::memmove(&data_.ptr_[idx], &data_.ptr_[idx + 1], (size_ - idx - 1) * sizeof(T));
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
  iterator erase(const_iterator first, const_iterator last) {
    size_type first_idx = first.get() - data_.ptr_;
    size_type last_idx = last.get() - data_.ptr_;
    size_type count = last_idx - first_idx;

    if constexpr (kIsPod) {
      std::memmove(&data_.ptr_[first_idx], &data_.ptr_[last_idx],
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
   * Resize vector to specified size.
   * Grows or shrinks the vector to the new size, default-initializing new elements.
   *
   * @param new_size New size
   */
  void resize(size_type new_size) {
    if (new_size > size_) {
      if (new_size > capacity_) {
        reserve(new_size);
      }
      for (size_type i = size_; i < new_size; ++i) {
        if constexpr (kIsPod) {
          data_.ptr_[i] = T();
        } else {
          new (&data_.ptr_[i]) T();
        }
      }
      size_ = new_size;
    } else if (new_size < size_) {
      DestroyRange(new_size, size_);
      size_ = new_size;
    }
  }

  /**
   * Resize vector to specified size with fill value.
   * Grows or shrinks the vector to the new size, initializing new elements to value.
   *
   * @param new_size New size
   * @param value Value to fill new elements with
   */
  void resize(size_type new_size, const T& value) {
    if (new_size > size_) {
      if (new_size > capacity_) {
        reserve(new_size);
      }
      for (size_type i = size_; i < new_size; ++i) {
        ConstructCopy(i, value);
      }
      size_ = new_size;
    } else if (new_size < size_) {
      DestroyRange(new_size, size_);
      size_ = new_size;
    }
  }

  /**
   * Swap contents with another vector
   *
   * @param other Vector to swap with
   */
  void swap(vector& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }

  /**
   * Serialize vector to archive.
   * Uses cereal serialization framework to save vector data.
   *
   * @tparam Archive Cereal archive type
   * @param ar Archive to save to
   */
  template<class Archive>
  void save(Archive& ar) const {
    hshm::ipc::save_vec<Archive, vector<T, AllocT>, T>(ar, *this);
  }

  /**
   * Deserialize vector from archive.
   * Uses cereal serialization framework to load vector data.
   *
   * @tparam Archive Cereal archive type
   * @param ar Archive to load from
   */
  template<class Archive>
  void load(Archive& ar) {
    hshm::ipc::load_vec<Archive, vector<T, AllocT>, T>(ar, *this);
  }
};

}  // namespace hshm::priv

#endif  // HSHM_DATA_STRUCTURES_PRIV_VECTOR_H_
