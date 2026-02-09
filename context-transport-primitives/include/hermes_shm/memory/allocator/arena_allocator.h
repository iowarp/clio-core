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

#ifndef HSHM_MEMORY_ALLOCATOR_ARENA_ALLOCATOR_H_
#define HSHM_MEMORY_ALLOCATOR_ARENA_ALLOCATOR_H_

#include <cstdint>
#include <limits>

#include "allocator.h"
#include "heap.h"
#include "hermes_shm/thread/lock.h"

namespace hshm::ipc {

/**
 * Forward declarations
 */
template<bool ATOMIC>
class _ArenaAllocator;

template<bool ATOMIC = false>
using ArenaAllocator = BaseAllocator<_ArenaAllocator<ATOMIC>>;

/**
 * Arena allocator implementation
 *
 * Simple bump-pointer allocator that grows upwards. Does not support
 * freeing individual allocations - only bulk reset.
 *
 * @tparam ATOMIC Whether the allocator should be thread-safe using atomics
 */
template<bool ATOMIC>
class _ArenaAllocator : public Allocator {
 private:
  Heap<ATOMIC> heap_;  /**< Heap for bump-pointer allocation */
  hipc::atomic<hshm::size_t> total_alloc_;  /**< Total allocated size tracking */
  size_t heap_begin_;  /**< Initial heap offset (for reset) */
  size_t heap_max_;    /**< Maximum heap size (for reset) */

 public:
  /**
   * Allocator constructor
   */
  HSHM_CROSS_FUN
  _ArenaAllocator() : total_alloc_(0), heap_begin_(0), heap_max_(0) {}

  /**
   * Initialize the allocator in shared memory
   *
   * @param backend Memory backend
   * @param region_size Size of the region in bytes. If 0, defaults to backend.data_capacity_
   */
  HSHM_CROSS_FUN
  void shm_init(const MemoryBackend &backend, size_t region_size = 0) {
    SetBackend(backend);
    alloc_header_size_ = sizeof(_ArenaAllocator<ATOMIC>);

    // Default region_size to data_capacity_ if not specified
    if (region_size == 0) {
      region_size = backend.data_capacity_;
    }

    // Store region_size for use in GetAllocatorDataSize()
    region_size_ = region_size;

    // Calculate data_start_ - where the allocator's managed region begins
    // For ArenaAllocator, data starts immediately after the allocator object
    data_start_ = sizeof(_ArenaAllocator<ATOMIC>);

    total_alloc_ = 0;

    // Calculate and store the offset of this allocator object within the backend data
    // This must be calculated BEFORE any GetBackendData() calls
    this_ = reinterpret_cast<char*>(this) - reinterpret_cast<char*>(backend.data_);

    // Store initial heap parameters for reset using helper functions
    heap_begin_ = GetAllocatorDataOff();
    heap_max_ = GetAllocatorDataOff() + region_size - data_start_;

    // Initialize heap using helper functions
    // GetAllocatorDataOff() returns the offset where managed heap begins
    // heap_max_ is calculated from region_size
    heap_.Init(heap_begin_, heap_max_);
  }

  /**
   * Attach an existing allocator from shared memory
   *
   * ArenaAllocator state (heap_, total_alloc_, heap_begin_, heap_max_) is
   * fully in shared memory. The base class GetBackendData() reconstructs
   * pointers from the this_ offset, so no per-process setup is needed.
   */
  HSHM_CROSS_FUN
  void shm_attach(MemoryBackend backend) {
    (void)backend;
  }

  /**
   * Allocate memory of specified size
   *
   * @param size Size to allocate
   * @param alignment Optional alignment (ignored)
   * @return Offset pointer to allocated memory
   * @throws OUT_OF_MEMORY if allocation fails
   */
  HSHM_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size, size_t alignment = 1) {
    (void)alignment;  // Alignment is not supported in arena allocator

    if (heap_.GetRemainingSize() < size) {
      HSHM_THROW_ERROR(OUT_OF_MEMORY);
    }

    // Allocate from heap
    size_t off = heap_.Allocate(size);
    if (off == 0 && heap_.GetOffset() != 0) {
      // Allocation failed (out of memory)
      HSHM_THROW_ERROR(OUT_OF_MEMORY);
    }

#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ += size;
#endif
    return OffsetPtr<>(off);
  }

  /**
   * Reallocate memory (NOT IMPLEMENTED)
   *
   * Arena allocators do not support reallocation.
   */
  HSHM_CROSS_FUN
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    (void)p;
    (void)new_size;
    HSHM_THROW_ERROR(NOT_IMPLEMENTED,
                     "ArenaAllocator does not support reallocation");
    return OffsetPtr<>(0);
  }

  /**
   * Free memory (NOT IMPLEMENTED)
   *
   * Arena allocators do not support freeing individual allocations.
   * Memory is freed in bulk when the arena is reset or destroyed.
   */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(OffsetPtr<> p) {
    (void)p;
    // Arena allocator does not support individual frees
    // This is intentionally a no-op (not an error)
  }

  /**
   * Get the current amount of data allocated
   *
   * @return Total bytes allocated
   */
  HSHM_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() {
    return (size_t)total_alloc_.load();
  }

  /**
   * Create thread-local storage (NOT IMPLEMENTED)
   *
   * Arena allocators do not require TLS.
   */
  HSHM_CROSS_FUN
  void CreateTls() {
    // No TLS needed for arena allocator
  }

  /**
   * Free thread-local storage (NOT IMPLEMENTED)
   *
   * Arena allocators do not require TLS.
   */
  HSHM_CROSS_FUN
  void FreeTls() {
    // No TLS needed for arena allocator
  }

  /**
   * Reset the arena to empty state
   *
   * Resets the heap offset to the initial position, effectively freeing all allocations.
   */
  HSHM_CROSS_FUN
  void Reset() {
#ifdef HSHM_ALLOC_TRACK_SIZE
    total_alloc_ = 0;
#endif
    heap_.Init(heap_begin_, heap_max_);
  }

  /**
   * Get the heap offset (for debugging/inspection)
   *
   * @return Current heap offset
   */
  HSHM_CROSS_FUN
  size_t GetHeapOffset() const {
    return heap_.GetOffset();
  }

  /**
   * Get remaining space in the arena
   *
   * @return Bytes remaining
   */
  HSHM_CROSS_FUN
  size_t GetRemainingSize() const {
    return heap_.GetRemainingSize();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_ARENA_ALLOCATOR_H_
