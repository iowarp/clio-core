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

#ifndef HSHM_MEMORY_ALLOCATOR_HEAP_H_
#define HSHM_MEMORY_ALLOCATOR_HEAP_H_

#include "hermes_shm/constants/macros.h"
#include "hermes_shm/types/atomic.h"
#include "hermes_shm/util/errors.h"

namespace hshm::ipc {

/**
 * Heap helper class for simple bump-pointer allocation
 *
 * This is not an allocator itself, but a utility for implementing
 * allocators that need monotonically increasing offset allocation.
 *
 * @tparam ATOMIC Whether the heap pointer should be atomic
 */
template<bool ATOMIC>
class Heap {
 private:
  hipc::opt_atomic<size_t, ATOMIC> heap_;  /// Current heap offset
  size_t max_offset_;                      /// Maximum heap offset (initial_offset + max_size)

 public:
  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  Heap() : heap_(0), max_offset_(0) {}

  /**
   * Constructor with initial offset and max offset
   *
   * @param initial_offset Initial heap offset
   * @param max_offset Maximum offset the heap can reach (initial_offset + max_size)
   */
  HSHM_CROSS_FUN
  Heap(size_t initial_offset, size_t max_offset)
      : heap_(initial_offset), max_offset_(max_offset) {}

  /**
   * Initialize the heap
   *
   * @param initial_offset Initial heap offset
   * @param max_offset Maximum offset the heap can reach (initial_offset + max_size)
   */
  HSHM_CROSS_FUN
  void Init(size_t initial_offset, size_t max_offset) {
    heap_.store(initial_offset);
    max_offset_ = max_offset;
  }

  /**
   * Allocate space from the heap
   *
   * @param size Number of bytes to allocate
   * @return Offset of the allocated region, or 0 on failure (out of memory)
   */
  HSHM_CROSS_FUN
  size_t Allocate(size_t size) {
    // Check if heap may have enough space
    if (heap_.load() + size > max_offset_) {
      return 0;
    }
    
    // Atomically fetch current offset and advance heap by size
    size_t off = heap_.fetch_add(size);

    // Calculate actual end offset after this allocation
    size_t end_off = off + size;

    // Check if allocation would exceed maximum offset
    if (end_off > max_offset_) {
      max_offset_ = end_off;
      return 0;  // Return 0 to indicate failure (out of memory)
    }

    return off;
  }

  /**
   * Get the current heap offset
   *
   * @return Current offset at the top of the heap
   */
  HSHM_CROSS_FUN
  size_t GetOffset() const {
    return heap_.load();
  }

  /**
   * Get the maximum heap offset
   *
   * @return Maximum offset the heap can reach
   */
  HSHM_CROSS_FUN
  size_t GetMaxOffset() const {
    return max_offset_;
  }

  /**
   * Get the maximum heap size (for backward compatibility)
   *
   * @return Maximum size the heap can grow to
   */
  HSHM_CROSS_FUN
  size_t GetMaxSize() const {
    return max_offset_;
  }

  /**
   * Get the remaining space in the heap
   *
   * @return Number of bytes remaining
   */
  HSHM_CROSS_FUN
  size_t GetRemainingSize() const {
    size_t current = heap_.load();
    return (current < max_offset_) ? (max_offset_ - current) : 0;
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_HEAP_H_
