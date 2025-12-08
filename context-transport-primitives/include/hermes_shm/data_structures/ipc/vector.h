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

#ifndef HSHM_DATA_STRUCTURES_IPC_VECTOR_H_
#define HSHM_DATA_STRUCTURES_IPC_VECTOR_H_

#include "hermes_shm/data_structures/ipc/shm_container.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/numbers.h"
#include <cstring>
#include <iterator>
#include <type_traits>

namespace hshm::ipc {

/**
 * Shared-memory compatible vector container
 *
 * This vector class provides dynamic array functionality for shared memory.
 * It stores data using offset pointers to maintain compatibility across
 * process boundaries. The allocator pointer is stored as an offset for
 * process-independent access.
 *
 * @tparam T The element type
 * @tparam AllocT The allocator type
 */
template<typename T, typename AllocT>
class vector : public ShmContainer<AllocT> {
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
   * Forward iterator for vector
   *
   * Provides standard iterator interface for traversing vector elements.
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
     * @return Copy of this iterator before increment
     */
    HSHM_INLINE_CROSS_FUN
    iterator operator++(int) {
      iterator tmp = *this;
      ++ptr_;
      return tmp;
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
     * @return Copy of this iterator before decrement
     */
    HSHM_INLINE_CROSS_FUN
    iterator operator--(int) {
      iterator tmp = *this;
      --ptr_;
      return tmp;
    }

    /**
     * Equality operator
     *
     * @param other The iterator to compare with
     * @return True if pointing to same element
     */
    HSHM_INLINE_CROSS_FUN
    bool operator==(const iterator &other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality operator
     *
     * @param other The iterator to compare with
     * @return True if pointing to different elements
     */
    HSHM_INLINE_CROSS_FUN
    bool operator!=(const iterator &other) const {
      return ptr_ != other.ptr_;
    }

    /**
     * Less than operator
     *
     * @param other The iterator to compare with
     * @return True if this points before other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator<(const iterator &other) const {
      return ptr_ < other.ptr_;
    }

    /**
     * Less or equal operator
     *
     * @param other The iterator to compare with
     * @return True if this points before or at other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator<=(const iterator &other) const {
      return ptr_ <= other.ptr_;
    }

    /**
     * Greater than operator
     *
     * @param other The iterator to compare with
     * @return True if this points after other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator>(const iterator &other) const {
      return ptr_ > other.ptr_;
    }

    /**
     * Greater or equal operator
     *
     * @param other The iterator to compare with
     * @return True if this points after or at other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator>=(const iterator &other) const {
      return ptr_ >= other.ptr_;
    }

    /**
     * Addition operator
     *
     * @param n The number of elements to advance
     * @return New iterator advanced by n positions
     */
    HSHM_INLINE_CROSS_FUN
    iterator operator+(ptrdiff_t n) const {
      return iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n The number of elements to retreat
     * @return New iterator retreated by n positions
     */
    HSHM_INLINE_CROSS_FUN
    iterator operator-(ptrdiff_t n) const {
      return iterator(ptr_ - n);
    }

    /**
     * Difference operator
     *
     * @param other The iterator to compare with
     * @return Number of elements between iterators
     */
    HSHM_INLINE_CROSS_FUN
    ptrdiff_t operator-(const iterator &other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n The index to access
     * @return Reference to the element at offset n
     */
    HSHM_INLINE_CROSS_FUN
    T& operator[](ptrdiff_t n) const {
      return *(ptr_ + n);
    }

    /**
     * Addition assignment operator
     *
     * @param n The number of elements to advance
     * @return Reference to this iterator
     */
    HSHM_INLINE_CROSS_FUN
    iterator& operator+=(ptrdiff_t n) {
      ptr_ += n;
      return *this;
    }

    /**
     * Subtraction assignment operator
     *
     * @param n The number of elements to retreat
     * @return Reference to this iterator
     */
    HSHM_INLINE_CROSS_FUN
    iterator& operator-=(ptrdiff_t n) {
      ptr_ -= n;
      return *this;
    }
  };

  /**
   * Const iterator for vector
   *
   * Provides constant iterator interface for traversing vector elements.
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
     * Convert from iterator
     *
     * @param iter The iterator to convert from
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator(const iterator &iter) : ptr_(&(*iter)) {}

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
     * @return Copy of this iterator before increment
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++ptr_;
      return tmp;
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
     * @return Copy of this iterator before decrement
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --ptr_;
      return tmp;
    }

    /**
     * Equality operator
     *
     * @param other The iterator to compare with
     * @return True if pointing to same element
     */
    HSHM_INLINE_CROSS_FUN
    bool operator==(const const_iterator &other) const {
      return ptr_ == other.ptr_;
    }

    /**
     * Inequality operator
     *
     * @param other The iterator to compare with
     * @return True if pointing to different elements
     */
    HSHM_INLINE_CROSS_FUN
    bool operator!=(const const_iterator &other) const {
      return ptr_ != other.ptr_;
    }

    /**
     * Less than operator
     *
     * @param other The iterator to compare with
     * @return True if this points before other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator<(const const_iterator &other) const {
      return ptr_ < other.ptr_;
    }

    /**
     * Less or equal operator
     *
     * @param other The iterator to compare with
     * @return True if this points before or at other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator<=(const const_iterator &other) const {
      return ptr_ <= other.ptr_;
    }

    /**
     * Greater than operator
     *
     * @param other The iterator to compare with
     * @return True if this points after other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator>(const const_iterator &other) const {
      return ptr_ > other.ptr_;
    }

    /**
     * Greater or equal operator
     *
     * @param other The iterator to compare with
     * @return True if this points after or at other
     */
    HSHM_INLINE_CROSS_FUN
    bool operator>=(const const_iterator &other) const {
      return ptr_ >= other.ptr_;
    }

    /**
     * Addition operator
     *
     * @param n The number of elements to advance
     * @return New iterator advanced by n positions
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator operator+(ptrdiff_t n) const {
      return const_iterator(ptr_ + n);
    }

    /**
     * Subtraction operator
     *
     * @param n The number of elements to retreat
     * @return New iterator retreated by n positions
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator operator-(ptrdiff_t n) const {
      return const_iterator(ptr_ - n);
    }

    /**
     * Difference operator
     *
     * @param other The iterator to compare with
     * @return Number of elements between iterators
     */
    HSHM_INLINE_CROSS_FUN
    ptrdiff_t operator-(const const_iterator &other) const {
      return ptr_ - other.ptr_;
    }

    /**
     * Subscript operator
     *
     * @param n The index to access
     * @return Const reference to the element at offset n
     */
    HSHM_INLINE_CROSS_FUN
    const T& operator[](ptrdiff_t n) const {
      return *(ptr_ + n);
    }

    /**
     * Addition assignment operator
     *
     * @param n The number of elements to advance
     * @return Reference to this iterator
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator& operator+=(ptrdiff_t n) {
      ptr_ += n;
      return *this;
    }

    /**
     * Subtraction assignment operator
     *
     * @param n The number of elements to retreat
     * @return Reference to this iterator
     */
    HSHM_INLINE_CROSS_FUN
    const_iterator& operator-=(ptrdiff_t n) {
      ptr_ -= n;
      return *this;
    }
  };

 private:
  size_t size_;      /**< Current number of elements */
  size_t capacity_;  /**< Allocated capacity */
  OffsetPtr<T> data_;  /**< Offset pointer to element array */

 public:
  /**
   * Default constructor
   *
   * Creates an empty vector with no allocated storage.
   */
  HSHM_CROSS_FUN
  vector()
      : ShmContainer<AllocT>(nullptr),
        size_(0),
        capacity_(0),
        data_(OffsetPtr<T>::GetNull()) {}

  /**
   * Constructor with allocator
   *
   * Creates an empty vector associated with an allocator.
   *
   * @param alloc The allocator to use
   */
  HSHM_INLINE_CROSS_FUN
  explicit vector(AllocT *alloc)
      : ShmContainer<AllocT>(alloc),
        size_(0),
        capacity_(0),
        data_(OffsetPtr<T>::GetNull()) {}

  /**
   * Constructor with explicit size and default initialization
   *
   * Creates a vector with the specified size, with elements default-initialized.
   *
   * @param alloc The allocator to use
   * @param size The initial size of the vector
   */
  HSHM_CROSS_FUN
  vector(AllocT *alloc, size_t size);

  /**
   * Constructor with explicit size and fill value
   *
   * Creates a vector with the specified size, with all elements initialized
   * to the provided value.
   *
   * @param alloc The allocator to use
   * @param size The initial size of the vector
   * @param value The value to fill with
   */
  HSHM_CROSS_FUN
  vector(AllocT *alloc, size_t size, const T &value);

  /**
   * Copy constructor
   *
   * Creates a new vector as a copy of an existing vector.
   *
   * @param other The vector to copy from
   */
  HSHM_CROSS_FUN
  vector(const vector &other);

  /**
   * Move constructor
   *
   * Creates a new vector by moving from an existing vector.
   *
   * @param other The vector to move from
   */
  HSHM_CROSS_FUN
  vector(vector &&other) noexcept;

  /**
   * Range constructor from iterators
   *
   * Creates a vector from a range of elements.
   *
   * @tparam IterT Iterator type
   * @param alloc The allocator to use
   * @param first Iterator to the first element
   * @param last Iterator past the last element
   */
  template<typename IterT,
           typename = typename std::enable_if<!std::is_integral<IterT>::value>::type>
  HSHM_CROSS_FUN
  vector(AllocT *alloc, IterT first, IterT last);

  /**
   * Initializer list constructor
   *
   * Creates a vector from an initializer list.
   *
   * @param alloc The allocator to use
   * @param init The initializer list
   */
  HSHM_CROSS_FUN
  vector(AllocT *alloc, std::initializer_list<T> init);

  /**
   * Destructor
   *
   * Destroys all elements and deallocates storage.
   */
  HSHM_CROSS_FUN
  ~vector();

  /**
   * Copy assignment operator
   *
   * @param other The vector to copy from
   * @return Reference to this vector
   */
  HSHM_CROSS_FUN
  vector& operator=(const vector &other);

  /**
   * Move assignment operator
   *
   * @param other The vector to move from
   * @return Reference to this vector
   */
  HSHM_CROSS_FUN
  vector& operator=(vector &&other) noexcept;

  /**
   * Equality comparison operator
   *
   * @param other The vector to compare with
   * @return True if vectors have equal elements
   */
  HSHM_CROSS_FUN
  bool operator==(const vector &other) const;

  /**
   * Inequality comparison operator
   *
   * @param other The vector to compare with
   * @return True if vectors have different elements
   */
  HSHM_CROSS_FUN
  bool operator!=(const vector &other) const;

  /**
   * Get element at index with bounds checking
   *
   * @param idx The index to access
   * @return Reference to the element at idx
   * @throws Bounds checking is asserted but not thrown
   */
  HSHM_INLINE_CROSS_FUN
  T& at(size_t idx) {
    auto fp = FullPtr(this->GetAllocator(), data_); T *ptr = fp.ptr_;
    return ptr[idx];
  }

  /**
   * Get const element at index with bounds checking
   *
   * @param idx The index to access
   * @return Const reference to the element at idx
   * @throws Bounds checking is asserted but not thrown
   */
  HSHM_INLINE_CROSS_FUN
  const T& at(size_t idx) const {
    const auto fp = FullPtr(this->GetAllocator(), data_); T *ptr = fp.ptr_;
    return ptr[idx];
  }

  /**
   * Get element at index without bounds checking
   *
   * @param idx The index to access
   * @return Reference to the element at idx
   */
  HSHM_INLINE_CROSS_FUN
  T& operator[](size_t idx) {
    auto fp = FullPtr(this->GetAllocator(), data_); T *ptr = fp.ptr_;
    return ptr[idx];
  }

  /**
   * Get const element at index without bounds checking
   *
   * @param idx The index to access
   * @return Const reference to the element at idx
   */
  HSHM_INLINE_CROSS_FUN
  const T& operator[](size_t idx) const {
    const auto fp = FullPtr(this->GetAllocator(), data_); T *ptr = fp.ptr_;
    return ptr[idx];
  }

  /**
   * Get reference to first element
   *
   * @return Reference to the first element
   */
  HSHM_INLINE_CROSS_FUN
  T& front() {
    auto fp = FullPtr(this->GetAllocator(), data_); T *ptr = fp.ptr_;
    return *ptr;
  }

  /**
   * Get const reference to first element
   *
   * @return Const reference to the first element
   */
  HSHM_INLINE_CROSS_FUN
  const T& front() const {
    const auto fp = FullPtr(this->GetAllocator(), data_); T *ptr = fp.ptr_;
    return *ptr;
  }

  /**
   * Get reference to last element
   *
   * @return Reference to the last element
   */
  HSHM_INLINE_CROSS_FUN
  T& back() {
    auto fp = FullPtr(this->GetAllocator(), data_); T *ptr = fp.ptr_;
    return ptr[size_ - 1];
  }

  /**
   * Get const reference to last element
   *
   * @return Const reference to the last element
   */
  HSHM_INLINE_CROSS_FUN
  const T& back() const {
    const auto fp = FullPtr(this->GetAllocator(), data_); T *ptr = fp.ptr_;
    return ptr[size_ - 1];
  }

  /**
   * Get raw pointer to data
   *
   * @return Pointer to the first element
   */
  HSHM_INLINE_CROSS_FUN
  T* data() {
    if (data_.IsNull()) return nullptr;
    return FullPtr(this->GetAllocator(), data_).ptr_;
  }

  /**
   * Get const raw pointer to data
   *
   * @return Const pointer to the first element
   */
  HSHM_INLINE_CROSS_FUN
  const T* data() const {
    if (data_.IsNull()) return nullptr;
    return FullPtr(this->GetAllocator(), data_).ptr_;
  }

  /**
   * Get iterator to beginning
   *
   * @return Iterator to the first element
   */
  HSHM_INLINE_CROSS_FUN
  iterator begin() {
    T *ptr = data_.IsNull() ? nullptr : FullPtr(this->GetAllocator(), data_).ptr_;
    return iterator(ptr);
  }

  /**
   * Get const iterator to beginning
   *
   * @return Const iterator to the first element
   */
  HSHM_INLINE_CROSS_FUN
  const_iterator begin() const {
    const T *ptr = data_.IsNull() ? nullptr : FullPtr(this->GetAllocator(), data_).ptr_;
    return const_iterator(ptr);
  }

  /**
   * Get iterator to end
   *
   * @return Iterator past the last element
   */
  HSHM_INLINE_CROSS_FUN
  iterator end() {
    T *ptr = data_.IsNull() ? nullptr : FullPtr(this->GetAllocator(), data_).ptr_;
    return iterator(ptr + size_);
  }

  /**
   * Get const iterator to end
   *
   * @return Const iterator past the last element
   */
  HSHM_INLINE_CROSS_FUN
  const_iterator end() const {
    const T *ptr = data_.IsNull() ? nullptr : FullPtr(this->GetAllocator(), data_).ptr_;
    return const_iterator(ptr + size_);
  }

  /**
   * Get const iterator to beginning
   *
   * @return Const iterator to the first element
   */
  HSHM_INLINE_CROSS_FUN
  const_iterator cbegin() const {
    const T *ptr = data_.IsNull() ? nullptr : FullPtr(this->GetAllocator(), data_).ptr_;
    return const_iterator(ptr);
  }

  /**
   * Get const iterator to end
   *
   * @return Const iterator past the last element
   */
  HSHM_INLINE_CROSS_FUN
  const_iterator cend() const {
    const T *ptr = data_.IsNull() ? nullptr : FullPtr(this->GetAllocator(), data_).ptr_;
    return const_iterator(ptr + size_);
  }

  /**
   * Add element to the end
   *
   * @param value The value to add
   */
  HSHM_CROSS_FUN
  void push_back(const T &value);

  /**
   * Add element to the end with move semantics
   *
   * @param value The value to add (will be moved)
   */
  HSHM_CROSS_FUN
  void push_back(T &&value);

  /**
   * Add element to the end with in-place construction
   *
   * @tparam Args The argument types
   * @param args The arguments to forward to the constructor
   */
  template<typename... Args>
  HSHM_CROSS_FUN
  void emplace_back(Args&&... args);

  /**
   * Insert element at position
   *
   * @param pos The position to insert at
   * @param value The value to insert
   * @return Iterator to the inserted element
   */
  HSHM_CROSS_FUN
  iterator insert(const_iterator pos, const T &value);

  /**
   * Insert element at position with move semantics
   *
   * @param pos The position to insert at
   * @param value The value to insert (will be moved)
   * @return Iterator to the inserted element
   */
  HSHM_CROSS_FUN
  iterator insert(const_iterator pos, T &&value);

  /**
   * Insert element at position with in-place construction
   *
   * @tparam Args The argument types
   * @param pos The position to insert at
   * @param args The arguments to forward to the constructor
   * @return Iterator to the inserted element
   */
  template<typename... Args>
  HSHM_CROSS_FUN
  iterator emplace(const_iterator pos, Args&&... args);

  /**
   * Erase element at position
   *
   * @param pos The position to erase at
   * @return Iterator to the element following the erase
   */
  HSHM_CROSS_FUN
  iterator erase(const_iterator pos);

  /**
   * Erase range of elements
   *
   * @param first Iterator to the first element to erase
   * @param last Iterator past the last element to erase
   * @return Iterator to the element following the erase
   */
  HSHM_CROSS_FUN
  iterator erase(const_iterator first, const_iterator last);

  /**
   * Remove all elements
   *
   * Destroys all elements but does not deallocate storage.
   */
  HSHM_CROSS_FUN
  void clear();

  /**
   * Get the number of elements
   *
   * @return The size of the vector
   */
  HSHM_INLINE_CROSS_FUN
  size_t size() const {
    return size_;
  }

  /**
   * Get the allocated capacity
   *
   * @return The capacity of the vector
   */
  HSHM_INLINE_CROSS_FUN
  size_t capacity() const {
    return capacity_;
  }

  /**
   * Check if vector is empty
   *
   * @return True if size is 0
   */
  HSHM_INLINE_CROSS_FUN
  bool empty() const {
    return size_ == 0;
  }

  /**
   * Reserve space for at least new_capacity elements
   *
   * @param new_capacity The new minimum capacity
   */
  HSHM_CROSS_FUN
  void reserve(size_t new_capacity);

  /**
   * Reduce capacity to match size
   *
   * Reallocates storage to exactly fit the current size.
   */
  HSHM_CROSS_FUN
  void shrink_to_fit();

  /**
   * Resize vector to new size with default-initialized elements
   *
   * @param new_size The new size of the vector
   */
  HSHM_CROSS_FUN
  void resize(size_t new_size);

  /**
   * Resize vector to new size with fill value
   *
   * @param new_size The new size of the vector
   * @param value The value to fill new elements with
   */
  HSHM_CROSS_FUN
  void resize(size_t new_size, const T &value);

 private:
  /**
   * Helper to allocate and initialize storage
   *
   * @param capacity The new capacity
   */
  HSHM_CROSS_FUN
  void AllocateStorage(size_t capacity);

  /**
   * Helper to deallocate storage
   */
  HSHM_CROSS_FUN
  void DeallocateStorage();

  /**
   * Helper to destroy all elements
   */
  HSHM_CROSS_FUN
  void DestroyElements();

  /**
   * Helper to copy elements
   *
   * @param src Source pointer
   * @param count Number of elements
   */
  HSHM_CROSS_FUN
  void CopyElements(const T *src, size_t count);

  /**
   * Helper to move elements
   *
   * @param src Source pointer
   * @param count Number of elements
   */
  HSHM_CROSS_FUN
  void MoveElements(T *src, size_t count);
};

/**
 * Constructor with explicit size and default initialization
 *
 * Creates a vector with the specified size, with elements default-initialized.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param alloc The allocator to use
 * @param size The initial size of the vector
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN vector<T, AllocT>::vector(AllocT *alloc, size_t size)
    : ShmContainer<AllocT>(alloc),
      size_(0),
      capacity_(0),
      data_(OffsetPtr<T>::GetNull()) {
  if (size > 0) {
    AllocateStorage(size);
    // Default-initialize elements
    auto fp = FullPtr(this->GetAllocator(), data_);
    if (fp.ptr_) {
      for (size_t i = 0; i < size; ++i) {
        new (fp.ptr_ + i) T();
      }
      size_ = size;
    }
  }
}

/**
 * Constructor with explicit size and fill value
 *
 * Creates a vector with the specified size, with all elements initialized
 * to the provided value.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param alloc The allocator to use
 * @param size The initial size of the vector
 * @param value The value to fill with
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN vector<T, AllocT>::vector(AllocT *alloc, size_t size, const T &value)
    : ShmContainer<AllocT>(alloc),
      size_(0),
      capacity_(0),
      data_(OffsetPtr<T>::GetNull()) {
  if (size > 0) {
    AllocateStorage(size);
    // Fill with value
    auto fp = FullPtr(this->GetAllocator(), data_);
    if (fp.ptr_) {
      for (size_t i = 0; i < size; ++i) {
        new (fp.ptr_ + i) T(value);
      }
      size_ = size;
    }
  }
}

/**
 * Copy constructor
 *
 * Creates a new vector as a copy of an existing vector.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param other The vector to copy from
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN vector<T, AllocT>::vector(const vector &other)
    : ShmContainer<AllocT>(other.GetAllocator()),
      size_(0),
      capacity_(0),
      data_(OffsetPtr<T>::GetNull()) {
  if (other.size_ > 0) {
    AllocateStorage(other.size_);
    CopyElements(other.data(), other.size_);
  }
}

/**
 * Move constructor
 *
 * Creates a new vector by moving from an existing vector.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param other The vector to move from
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN vector<T, AllocT>::vector(vector &&other) noexcept
    : ShmContainer<AllocT>(other.GetAllocator()),
      size_(other.size_),
      capacity_(other.capacity_),
      data_(other.data_) {
  // Clear other vector
  other.size_ = 0;
  other.capacity_ = 0;
  other.data_ = OffsetPtr<T>::GetNull();
}

/**
 * Range constructor from iterators
 *
 * Creates a vector from a range of elements.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @tparam IterT Iterator type
 * @param alloc The allocator to use
 * @param first Iterator to the first element
 * @param last Iterator past the last element
 */
template<typename T, typename AllocT>
template<typename IterT, typename>
HSHM_CROSS_FUN vector<T, AllocT>::vector(AllocT *alloc, IterT first, IterT last)
    : ShmContainer<AllocT>(alloc),
      size_(0),
      capacity_(0),
      data_(OffsetPtr<T>::GetNull()) {
  // Calculate size from iterators
  size_t count = std::distance(first, last);
  if (count > 0) {
    AllocateStorage(count);
    auto fp = FullPtr(this->GetAllocator(), data_);
    if (fp.ptr_) {
      size_t idx = 0;
      for (IterT it = first; it != last; ++it, ++idx) {
        new (fp.ptr_ + idx) T(*it);
      }
      size_ = count;
    }
  }
}

/**
 * Initializer list constructor
 *
 * Creates a vector from an initializer list.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param alloc The allocator to use
 * @param init The initializer list
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN vector<T, AllocT>::vector(AllocT *alloc, std::initializer_list<T> init)
    : ShmContainer<AllocT>(alloc),
      size_(0),
      capacity_(0),
      data_(OffsetPtr<T>::GetNull()) {
  size_t count = init.size();
  if (count > 0) {
    AllocateStorage(count);
    CopyElements(init.begin(), count);
  }
}

/**
 * Destructor
 *
 * Destroys all elements and deallocates storage.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN vector<T, AllocT>::~vector() {
  DestroyElements();
  DeallocateStorage();
}

/**
 * AllocateStorage - Allocate memory for elements
 *
 * Allocates memory for the specified capacity and stores as an offset pointer.
 * Does nothing if capacity is 0.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param capacity The number of elements to allocate space for
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::AllocateStorage(size_t capacity) {
  if (capacity == 0) {
    capacity_ = 0;
    data_ = OffsetPtr<T>::GetNull();
    return;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    return;
  }

  // Allocate memory for capacity elements
  size_t alloc_size = capacity * sizeof(T);
  OffsetPtr<> offset = alloc->AllocateOffset(alloc_size);

  if (offset.IsNull()) {
    // Allocation failed
    capacity_ = 0;
    data_ = OffsetPtr<T>::GetNull();
    return;
  }

  // Store offset pointer
  data_ = OffsetPtr<T>(offset.load());
  capacity_ = capacity;
}

/**
 * DeallocateStorage - Deallocate vector storage
 *
 * Deallocates the memory used by the vector's data array.
 * Does nothing if data is null.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::DeallocateStorage() {
  if (data_.IsNull()) {
    return;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    data_ = OffsetPtr<T>::GetNull();
    capacity_ = 0;
    size_ = 0;
    return;
  }

  // Deallocate the memory
  size_t offset = data_.load();
  size_t alloc_size = capacity_ * sizeof(T);
  alloc->FreeOffsetNoNullCheck(OffsetPtr<>(offset));

  // Reset state
  data_ = OffsetPtr<T>::GetNull();
  capacity_ = 0;
  size_ = 0;
}

/**
 * DestroyElements - Destroy all vector elements
 *
 * Calls destructors on all elements if type is non-trivially destructible.
 * For POD types, this is a no-op.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::DestroyElements() {
  // Only call destructors for non-trivially destructible types
  if (std::is_trivially_destructible<T>::value || size_ == 0) {
    return;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    return;
  }

  auto fp = FullPtr(alloc, data_);
  if (!fp.ptr_) {
    return;
  }

  // Call destructor on each element
  for (size_t i = 0; i < size_; ++i) {
    fp.ptr_[i].~T();
  }
}

/**
 * CopyElements - Copy elements from source
 *
 * Copies count elements from source into allocated storage.
 * Uses memcpy for POD types, placement new for non-POD types.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param src Source pointer to copy from
 * @param count Number of elements to copy
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::CopyElements(const T *src, size_t count) {
  if (!src || count == 0) {
    size_ = 0;
    return;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    size_ = 0;
    return;
  }

  T *dest = FullPtr(alloc, data_).ptr_;
  if (!dest) {
    size_ = 0;
    return;
  }

  if (std::is_trivially_copyable<T>::value) {
    // Use memcpy for POD types
    std::memcpy(dest, src, count * sizeof(T));
  } else {
    // Use copy constructor for non-POD types
    for (size_t i = 0; i < count; ++i) {
      new (dest + i) T(src[i]);
    }
  }

  size_ = count;
}

/**
 * MoveElements - Move elements from source
 *
 * Moves count elements from source into allocated storage.
 * Uses memcpy for POD types, move constructor for non-POD types.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param src Source pointer to move from
 * @param count Number of elements to move
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::MoveElements(T *src, size_t count) {
  if (!src || count == 0) {
    size_ = 0;
    return;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    size_ = 0;
    return;
  }

  T *dest = FullPtr(alloc, data_).ptr_;
  if (!dest) {
    size_ = 0;
    return;
  }

  if (std::is_trivially_copyable<T>::value) {
    // Use memcpy for POD types
    std::memcpy(dest, src, count * sizeof(T));
  } else {
    // Use move constructor for non-POD types
    for (size_t i = 0; i < count; ++i) {
      new (dest + i) T(std::move(src[i]));
    }
  }

  size_ = count;
}

/**
 * emplace_back - Add element with in-place construction
 *
 * Constructs an element in-place at the end of the vector.
 * Reallocates storage if needed.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @tparam Args Argument types for constructor
 * @param args Arguments to forward to the constructor
 */
template<typename T, typename AllocT>
template<typename... Args>
HSHM_CROSS_FUN void vector<T, AllocT>::emplace_back(Args&&... args) {
  if (size_ >= capacity_) {
    // Need to grow
    size_t new_capacity = (capacity_ == 0) ? 1 : (capacity_ * 2);
    reserve(new_capacity);
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    return;
  }

  auto fp = FullPtr(alloc, data_);
  if (fp.ptr_) {
    // Construct element in-place at current size position
    new (fp.ptr_ + size_) T(std::forward<Args>(args)...);
    size_++;
  }
}

/**
 * push_back - Add element with copy semantics
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param value The value to add
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::push_back(const T &value) {
  emplace_back(value);
}

/**
 * push_back - Add element with move semantics
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param value The value to add (will be moved)
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::push_back(T &&value) {
  emplace_back(std::move(value));
}

/**
 * emplace - Insert element at position with in-place construction
 *
 * Constructs an element in-place at the specified position.
 * Elements after the position are shifted right.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @tparam Args Argument types for constructor
 * @param pos Position to insert at
 * @param args Arguments to forward to the constructor
 * @return Iterator to the inserted element
 */
template<typename T, typename AllocT>
template<typename... Args>
HSHM_CROSS_FUN typename vector<T, AllocT>::iterator
vector<T, AllocT>::emplace(const_iterator pos, Args&&... args) {
  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    return iterator(nullptr);
  }

  auto fp = FullPtr(alloc, data_);
  if (!fp.ptr_) {
    return iterator(nullptr);
  }

  // Calculate index from iterator
  size_t idx = &(*pos) - fp.ptr_;

  // Check if we need to grow
  if (size_ >= capacity_) {
    size_t new_capacity = (capacity_ == 0) ? 1 : (capacity_ * 2);
    reserve(new_capacity);
    fp = FullPtr(alloc, data_);
  }

  // Shift elements to the right
  for (size_t i = size_; i > idx; --i) {
    new (fp.ptr_ + i) T(std::move(fp.ptr_[i - 1]));
    fp.ptr_[i - 1].~T();
  }

  // Construct new element
  new (fp.ptr_ + idx) T(std::forward<Args>(args)...);
  size_++;

  return iterator(fp.ptr_ + idx);
}

/**
 * insert - Insert element at position with copy semantics
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param pos Position to insert at
 * @param value The value to insert
 * @return Iterator to the inserted element
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN typename vector<T, AllocT>::iterator
vector<T, AllocT>::insert(const_iterator pos, const T &value) {
  return emplace(pos, value);
}

/**
 * insert - Insert element at position with move semantics
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param pos Position to insert at
 * @param value The value to insert (will be moved)
 * @return Iterator to the inserted element
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN typename vector<T, AllocT>::iterator
vector<T, AllocT>::insert(const_iterator pos, T &&value) {
  return emplace(pos, std::move(value));
}

/**
 * erase - Erase element at position
 *
 * Removes the element at the specified position.
 * Elements after the position are shifted left.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param pos Position to erase at
 * @return Iterator to the element following the erase
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN typename vector<T, AllocT>::iterator
vector<T, AllocT>::erase(const_iterator pos) {
  AllocT *alloc = this->GetAllocator();
  if (!alloc || size_ == 0) {
    return iterator(nullptr);
  }

  auto fp = FullPtr(alloc, data_);
  if (!fp.ptr_) {
    return iterator(nullptr);
  }

  // Calculate index from iterator
  size_t idx = &(*pos) - fp.ptr_;

  // Call destructor on the element
  fp.ptr_[idx].~T();

  // Shift elements to the left
  for (size_t i = idx; i < size_ - 1; ++i) {
    new (fp.ptr_ + i) T(std::move(fp.ptr_[i + 1]));
    fp.ptr_[i + 1].~T();
  }

  size_--;

  return iterator(fp.ptr_ + idx);
}

/**
 * erase - Erase range of elements
 *
 * Removes elements in the range [first, last).
 * Elements after the range are shifted left.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param first Iterator to the first element to erase
 * @param last Iterator past the last element to erase
 * @return Iterator to the element following the erase
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN typename vector<T, AllocT>::iterator
vector<T, AllocT>::erase(const_iterator first, const_iterator last) {
  AllocT *alloc = this->GetAllocator();
  if (!alloc || size_ == 0) {
    return iterator(nullptr);
  }

  auto fp = FullPtr(alloc, data_);
  if (!fp.ptr_) {
    return iterator(nullptr);
  }

  // Calculate indices from iterators
  size_t first_idx = &(*first) - fp.ptr_;
  size_t last_idx = &(*last) - fp.ptr_;
  size_t count = last_idx - first_idx;

  if (count == 0) {
    return iterator(fp.ptr_ + first_idx);
  }

  // Destroy elements in range
  for (size_t i = first_idx; i < last_idx; ++i) {
    fp.ptr_[i].~T();
  }

  // Shift remaining elements to the left
  for (size_t i = first_idx; i < size_ - count; ++i) {
    new (fp.ptr_ + i) T(std::move(fp.ptr_[i + count]));
    fp.ptr_[i + count].~T();
  }

  size_ -= count;

  return iterator(fp.ptr_ + first_idx);
}

/**
 * clear - Remove all elements
 *
 * Destroys all elements but does not deallocate storage.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::clear() {
  DestroyElements();
  size_ = 0;
}

/**
 * reserve - Reserve capacity for elements
 *
 * Allocates storage for at least new_capacity elements.
 * If new_capacity <= capacity_, does nothing.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param new_capacity The new minimum capacity
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::reserve(size_t new_capacity) {
  if (new_capacity <= capacity_) {
    return;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    return;
  }

  // Save old data
  OffsetPtr<T> old_data = data_;
  size_t old_capacity = capacity_;
  size_t old_size = size_;

  // Allocate new storage
  capacity_ = 0;
  size_ = 0;
  data_ = OffsetPtr<T>::GetNull();
  AllocateStorage(new_capacity);

  // Copy elements to new storage
  if (old_size > 0 && !old_data.IsNull()) {
    T *old_ptr = FullPtr(alloc, old_data).ptr_;
    CopyElements(old_ptr, old_size);

    // Deallocate old storage
    size_t offset = old_data.load();
    size_t alloc_size = old_capacity * sizeof(T);
    alloc->FreeOffsetNoNullCheck(OffsetPtr<>(offset));
  }
}

/**
 * shrink_to_fit - Reduce capacity to match size
 *
 * Reallocates storage to exactly fit the current size.
 * Does nothing if capacity == size.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::shrink_to_fit() {
  if (capacity_ == size_ || size_ == 0) {
    if (size_ == 0) {
      DeallocateStorage();
    }
    return;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    return;
  }

  // Save old data
  OffsetPtr<T> old_data = data_;
  size_t old_capacity = capacity_;
  size_t current_size = size_;

  // Allocate new storage with exact size
  capacity_ = 0;
  size_ = 0;
  data_ = OffsetPtr<T>::GetNull();

  if (current_size > 0) {
    AllocateStorage(current_size);

    // Copy elements to new storage
    if (!old_data.IsNull()) {
      T *old_ptr = FullPtr(alloc, old_data).ptr_;
      CopyElements(old_ptr, current_size);

      // Deallocate old storage
      size_t offset = old_data.load();
      size_t alloc_size = old_capacity * sizeof(T);
      alloc->FreeOffsetNoNullCheck(OffsetPtr<>(offset));
    }
  } else {
    // Just deallocate old storage
    size_t offset = old_data.load();
    size_t alloc_size = old_capacity * sizeof(T);
    alloc->FreeOffsetNoNullCheck(OffsetPtr<>(offset));
  }
}

/**
 * resize - Resize vector to new size with default-initialized elements
 *
 * If new_size > size_, default-initializes new elements.
 * If new_size < size_, destroys extra elements.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param new_size The new size of the vector
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::resize(size_t new_size) {
  if (new_size == size_) {
    return;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    return;
  }

  if (new_size > size_) {
    // Need to grow
    if (new_size > capacity_) {
      reserve(new_size);
    }

    // Default-initialize new elements
    auto fp = FullPtr(alloc, data_);
    if (fp.ptr_) {
      for (size_t i = size_; i < new_size; ++i) {
        new (fp.ptr_ + i) T();
      }
    }
    size_ = new_size;
  } else {
    // Need to shrink
    auto fp = FullPtr(alloc, data_);
    if (fp.ptr_ && !std::is_trivially_destructible<T>::value) {
      for (size_t i = new_size; i < size_; ++i) {
        fp.ptr_[i].~T();
      }
    }
    size_ = new_size;
  }
}

/**
 * resize - Resize vector to new size with fill value
 *
 * If new_size > size_, fills new elements with value.
 * If new_size < size_, destroys extra elements.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param new_size The new size of the vector
 * @param value The value to fill new elements with
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN void vector<T, AllocT>::resize(size_t new_size, const T &value) {
  if (new_size == size_) {
    return;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    return;
  }

  if (new_size > size_) {
    // Need to grow
    if (new_size > capacity_) {
      reserve(new_size);
    }

    // Fill new elements with value
    auto fp = FullPtr(alloc, data_);
    if (fp.ptr_) {
      for (size_t i = size_; i < new_size; ++i) {
        new (fp.ptr_ + i) T(value);
      }
    }
    size_ = new_size;
  } else {
    // Need to shrink
    auto fp = FullPtr(alloc, data_);
    if (fp.ptr_ && !std::is_trivially_destructible<T>::value) {
      for (size_t i = new_size; i < size_; ++i) {
        fp.ptr_[i].~T();
      }
    }
    size_ = new_size;
  }
}

/**
 * Copy assignment operator
 *
 * Assigns a copy of another vector to this vector.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param other The vector to copy from
 * @return Reference to this vector
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN vector<T, AllocT>&
vector<T, AllocT>::operator=(const vector &other) {
  if (this == &other) {
    return *this;
  }

  // Clear current contents
  clear();

  // Set allocator from other
  this->this_ = other.this_;

  // Copy elements from other
  if (other.size_ > 0) {
    if (capacity_ < other.size_) {
      AllocateStorage(other.size_);
    }
    CopyElements(other.data(), other.size_);
  }

  return *this;
}

/**
 * Move assignment operator
 *
 * Moves another vector's data to this vector.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param other The vector to move from
 * @return Reference to this vector
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN vector<T, AllocT>&
vector<T, AllocT>::operator=(vector &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  // Clean up current contents
  DestroyElements();
  DeallocateStorage();

  // Move data from other
  this->this_ = other.this_;
  size_ = other.size_;
  capacity_ = other.capacity_;
  data_ = other.data_;

  // Clear other
  other.size_ = 0;
  other.capacity_ = 0;
  other.data_ = OffsetPtr<T>::GetNull();

  return *this;
}

/**
 * Equality comparison operator
 *
 * Compares two vectors for equality.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param other The vector to compare with
 * @return True if vectors have equal size and elements
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN bool
vector<T, AllocT>::operator==(const vector &other) const {
  if (size_ != other.size_) {
    return false;
  }

  AllocT *alloc = this->GetAllocator();
  if (!alloc) {
    return size_ == 0;
  }

  auto fp = FullPtr(alloc, data_);
  auto other_fp = FullPtr(other.GetAllocator(), other.data_);

  if (!fp.ptr_ && !other_fp.ptr_) {
    return true;
  }

  if (!fp.ptr_ || !other_fp.ptr_) {
    return false;
  }

  // Compare elements
  for (size_t i = 0; i < size_; ++i) {
    if (!(fp.ptr_[i] == other_fp.ptr_[i])) {
      return false;
    }
  }

  return true;
}

/**
 * Inequality comparison operator
 *
 * Compares two vectors for inequality.
 *
 * @tparam T Element type
 * @tparam AllocT Allocator type
 * @param other The vector to compare with
 * @return True if vectors have unequal size or elements
 */
template<typename T, typename AllocT>
HSHM_CROSS_FUN bool
vector<T, AllocT>::operator!=(const vector &other) const {
  return !(*this == other);
}

}  // namespace hshm::ipc

#endif  // HSHM_DATA_STRUCTURES_IPC_VECTOR_H_
