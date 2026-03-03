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
      heap_.fetch_sub(size);
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
