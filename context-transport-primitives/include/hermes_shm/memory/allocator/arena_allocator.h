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
 * Arena allocator header stored in shared memory
 */
template<bool ATOMIC>
struct _ArenaAllocatorHeader : public AllocatorHeader {
  Heap<ATOMIC> heap_;  /// Heap for bump-pointer allocation

  HSHM_CROSS_FUN
  _ArenaAllocatorHeader() = default;

  HSHM_CROSS_FUN
  void Configure(AllocatorId alloc_id, size_t custom_header_size, size_t arena_size) {
    AllocatorHeader::Configure(alloc_id, custom_header_size);
    heap_.Init(0, arena_size);
  }
};

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
  _ArenaAllocatorHeader<ATOMIC> *header_;

 public:
  /**
   * Allocator constructor
   */
  HSHM_CROSS_FUN
  _ArenaAllocator() : header_(nullptr) {}

  /**
   * Initialize the allocator in shared memory
   *
   * @param id Allocator ID
   * @param custom_header_size Size of custom header extension
   * @param arena_size Maximum size of the arena
   * @param backend Memory backend
   */
  HSHM_CROSS_FUN
  void shm_init(AllocatorId id, size_t custom_header_size, size_t arena_size,
                MemoryBackend backend) {
    id_ = id;
    backend_ = backend;
    accel_id_ = backend_.data_id_;

    // Allocate and construct header
    header_ = ConstructHeader<_ArenaAllocatorHeader<ATOMIC>>(
        malloc(sizeof(_ArenaAllocatorHeader<ATOMIC>) + custom_header_size));
    custom_header_ = reinterpret_cast<char *>(header_ + 1);
    header_->Configure(id, custom_header_size, arena_size);
  }

  /**
   * Attach an existing allocator from shared memory
   */
  HSHM_CROSS_FUN
  void shm_attach(MemoryBackend backend) {
    HSHM_THROW_ERROR(NOT_IMPLEMENTED, "_ArenaAllocator::shm_attach");
  }

  /**
   * Allocate memory of specified size
   *
   * @param ctx Memory context
   * @param size Size to allocate
   * @param alignment Optional alignment (default: 1)
   * @return Offset pointer to allocated memory
   */
  HSHM_CROSS_FUN
  OffsetPtr AllocateOffset(const hipc::MemContext &ctx, size_t size, size_t alignment = 1) {
    size_t off = header_->heap_.Allocate(size, alignment);
    header_->AddSize(size);
    return OffsetPtr(off);
  }

  /**
   * Reallocate memory (NOT IMPLEMENTED)
   *
   * Arena allocators do not support reallocation.
   */
  HSHM_CROSS_FUN
  OffsetPtr ReallocateOffsetNoNullCheck(const hipc::MemContext &ctx,
                                            OffsetPtr p, size_t new_size) {
    HSHM_THROW_ERROR(NOT_IMPLEMENTED,
                     "ArenaAllocator does not support reallocation");
    return OffsetPtr(0);
  }

  /**
   * Free memory (NOT IMPLEMENTED)
   *
   * Arena allocators do not support freeing individual allocations.
   * Memory is freed in bulk when the arena is reset or destroyed.
   */
  HSHM_CROSS_FUN
  void FreeOffsetNoNullCheck(const hipc::MemContext &ctx, OffsetPtr p) {
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
    return (size_t)header_->GetCurrentlyAllocatedSize();
  }

  /**
   * Create thread-local storage (NOT IMPLEMENTED)
   *
   * Arena allocators do not require TLS.
   */
  HSHM_CROSS_FUN
  void CreateTls(MemContext &ctx) {
    // No TLS needed for arena allocator
  }

  /**
   * Free thread-local storage (NOT IMPLEMENTED)
   *
   * Arena allocators do not require TLS.
   */
  HSHM_CROSS_FUN
  void FreeTls(const hipc::MemContext &ctx) {
    // No TLS needed for arena allocator
  }

  /**
   * Reset the arena to empty state
   *
   * Resets the heap offset to 0, effectively freeing all allocations.
   */
  HSHM_CROSS_FUN
  void Reset() {
    size_t current_size = header_->GetCurrentlyAllocatedSize();
    header_->SubSize(current_size);
    header_->heap_.Init(0, header_->heap_.GetMaxSize());
  }

  /**
   * Get the heap offset (for debugging/inspection)
   *
   * @return Current heap offset
   */
  HSHM_CROSS_FUN
  size_t GetHeapOffset() const {
    return header_->heap_.GetOffset();
  }

  /**
   * Get remaining space in the arena
   *
   * @return Bytes remaining
   */
  HSHM_CROSS_FUN
  size_t GetRemainingSize() const {
    return header_->heap_.GetRemainingSize();
  }
};

}  // namespace hshm::ipc

#endif  // HSHM_MEMORY_ALLOCATOR_ARENA_ALLOCATOR_H_
