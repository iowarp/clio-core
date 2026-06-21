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

#ifndef CTP_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_
#define CTP_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_

#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/memory/allocator/buddy_allocator.h"
#include "clio_ctp/data_structures/ipc/slist_pre.h"
#include "clio_ctp/thread/lock/mutex.h"
#include "clio_ctp/thread/thread_model_manager.h"
#ifndef _WIN32
#include <sys/types.h>
#include <unistd.h>
#endif

namespace ctp::ipc {

/**
 * Forward declarations
 */
class PcThreadBlock;
class _ProducerConsumerAllocator;

/**
 * Typedef for the complete ProducerConsumerAllocator with BaseAllocator wrapper
 */
typedef BaseAllocator<_ProducerConsumerAllocator> ProducerConsumerAllocator;

/** Backward compatibility */
typedef ProducerConsumerAllocator MultiProcessAllocator;

/**
 * Per-thread allocator block providing lock-free fast-path allocation.
 *
 * Each thread has its own PcThreadBlock with a private BuddyAllocator,
 * enabling concurrent allocations without contention. When a PcThreadBlock
 * runs out of memory, it requests expansion from the global allocator.
 */
class PcThreadBlock : public pre::slist_node {
 public:
  int tid_;                    /**< Thread ID */
  BuddyAllocator alloc_;       /**< Private buddy allocator (MUST BE LAST) */

  /**
   * Default constructor
   */
  PcThreadBlock() : tid_(-1) {}

  /**
   * Initialize the thread block.
   *
   * @param backend Memory backend from the allocator
   * @param region_size Size of the memory region in bytes
   * @param tid Thread ID for this block
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, size_t region_size, int tid) {
    tid_ = tid;
    size_t alloc_region_size = region_size - sizeof(PcThreadBlock);
    alloc_.shm_init(backend, alloc_region_size);
    return true;
  }

  /**
   * Expand this PcThreadBlock with a new memory region.
   *
   * @param region Offset pointer to new memory region
   * @param region_size Size of the new region in bytes
   */
  void Expand(OffsetPtr<> region, size_t region_size) {
    alloc_.Expand(region, region_size);
  }
};

/**
 * Producer-consumer allocator for shared memory.
 *
 * Designed for a single-producer model where one process (the producer)
 * allocates and frees memory, while other processes (consumers) only read.
 * The runtime never allocates or frees from the client's memory segment.
 *
 * Architecture:
 * - Global BuddyAllocator manages the entire shared memory region
 * - PcThreadBlocks provide per-thread lock-free allocation
 *
 * Allocation Strategy (2-tier fallback):
 * 1. Fast path: Allocate from thread-local PcThreadBlock (no locks)
 * 2. Slow path: Expand PcThreadBlock from global allocator (global lock)
 *
 * Memory Layout:
 * The allocator itself is placed at the beginning of shared memory.
 * The BuddyAllocator follows immediately after the allocator header.
 */
class _ProducerConsumerAllocator : public Allocator {
 public:
  int tid_count_;               /**< Number of thread blocks allocated */
  ctp::Mutex lock_;            /**< Mutex protecting global allocator */
  ThreadLocalKey tblock_key_;   /**< TLS key for PcThreadBlock* */
  size_t thread_unit_;          /**< Default PcThreadBlock expansion size */
  BuddyAllocator alloc_;        /**< Global buddy allocator */

 public:
  /**
   * Default constructor
   */
  _ProducerConsumerAllocator() : tid_count_(0), thread_unit_(0) {}

  /**
   * Initialize the allocator with a new memory region.
   *
   * @param backend Memory backend (allocator placed at backend.data_)
   * @param region_size Size of the region in bytes (0 = entire backend)
   * @return true on success, false on failure
   */
  bool shm_init(const MemoryBackend &backend, size_t region_size = 0) {
    if (region_size == 0) {
      region_size = backend.data_capacity_;
    }

    SetBackend(backend);
    alloc_header_size_ = sizeof(_ProducerConsumerAllocator);
    data_start_ = sizeof(_ProducerConsumerAllocator);
    region_size_ = region_size;

    tid_count_ = 0;
    lock_.Init();

    // Set up TLS key
    void *null_ptr = nullptr;
    if (!CTP_THREAD_MODEL->CreateTls<void>(tblock_key_, null_ptr)) {
      return false;
    }

    // Initialize global buddy allocator
    size_t available_size = region_size - sizeof(_ProducerConsumerAllocator);
    alloc_.shm_init(backend, available_size);

    // Default thread expansion unit: 2MB
    thread_unit_ = 2 * 1024 * 1024;

    return true;
  }

  /**
   * Attach to an existing allocator (consumer process).
   *
   * Consumers only read from shared memory; they do not allocate or free.
   * Do NOT call CreateTls here: tblock_key_ lives in the shared region and
   * was set by the producer in shm_init. pthread keys are process-private,
   * so calling pthread_key_create here would clobber the producer's key
   * with one valid only in this process. The producer's later
   * pthread_setspecific calls would then silently fail (EINVAL) and
   * EnsureTls would allocate a new PcThreadBlock on every alloc.
   *
   * Consumers do not need TLS — they only resolve ShmPtrs, never allocate.
   *
   * @param backend The memory backend (unused, for API compatibility)
   * @return true on success
   */
  bool shm_attach(const MemoryBackend &backend) {
    (void)backend;
    return true;
  }

  /**
   * Detach from the allocator (cleanup TLS).
   */
  void shm_detach() {
    // TLS cleanup handled automatically
  }

 public:
  /**
   * Ensure thread-local storage is set up for the current thread.
   *
   * Checks TLS for a PcThreadBlock, and if not found, allocates one
   * from the global allocator.
   *
   * @return Pointer to the thread's PcThreadBlock, or nullptr on failure
   */
  PcThreadBlock* EnsureTls() {
    // Check if we already have a PcThreadBlock in TLS
    void *tblock_data = CTP_THREAD_MODEL->GetTls<void>(tblock_key_);
    if (tblock_data != nullptr) {
      return reinterpret_cast<PcThreadBlock*>(tblock_data);
    }

    // Allocate from global allocator
    FullPtr<PcThreadBlock> tblock_ptr;
    {
      ScopedMutex scoped_lock(lock_, 0);
      tblock_ptr = alloc_.AllocateRegion<PcThreadBlock>(thread_unit_);
    }
    if (tblock_ptr.IsNull()) {
      return nullptr;
    }

    // Initialize PcThreadBlock and store in TLS
    int tid = tid_count_++;
    tblock_ptr.ptr_->shm_init(GetBackend(), thread_unit_, tid);
    CTP_THREAD_MODEL->SetTls<void>(
        tblock_key_, reinterpret_cast<void*>(tblock_ptr.ptr_));
    return tblock_ptr.ptr_;
  }

  /**
   * Allocate memory from the producer-consumer allocator.
   *
   * Implements a 2-tier fallback strategy:
   * 1. Fast path: Thread-local PcThreadBlock (lock-free)
   * 2. Slow path: Expand PcThreadBlock from global allocator (global lock)
   *
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null on failure
   */
  OffsetPtr<> AllocateOffset(size_t size) {
    // Per-thread PcThreadBlock fast path is unsafe for cross-process
    // alloc/free: producer (client) allocates from its tblock; consumer
    // (daemon) frees on a different thread whose EnsureTls() returns the
    // daemon's tblock — Free then corrupts the daemon's free lists with
    // phantom entries pointing into producer memory. Symptom: 4n / 256m
    // FPP stalls after a few hundred ops with clio workers idle and
    // FUSE adapter threads parked in Future.Wait.
    // Route all alloc/free through the global buddy allocator under
    // lock_ so alloc and free hit the same data structure.
    ScopedMutex scoped_lock(lock_, 0);
    return alloc_.AllocateOffset(size);
  }

  /**
   * Reallocate memory (NOT IMPLEMENTED).
   *
   * @param p The original offset pointer
   * @param new_size The new size in bytes
   * @return Null offset pointer (reallocation not supported)
   */
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    (void)p;
    (void)new_size;
    CTP_THROW_ERROR(NOT_IMPLEMENTED,
                     "ProducerConsumerAllocator does not support reallocation");
    return OffsetPtr<>();
  }

  /**
   * Free memory allocated from the allocator.
   *
   * Frees to the current thread's PcThreadBlock allocator.
   *
   * @param p The offset pointer to free
   */
  void FreeOffsetNoNullCheck(OffsetPtr<> p) {
    // Route to the global allocator under lock_ — see AllocateOffset for
    // the cross-process per-tblock corruption that this avoids.
    ScopedMutex scoped_lock(lock_, 0);
    alloc_.FreeOffset(p);
  }

  /**
   * Reallocate memory to a new size.
   *
   * @param offset Original offset pointer
   * @param new_size New size in bytes
   * @return New offset pointer, or null on failure
   */
  OffsetPtr<> ReallocateOffset(OffsetPtr<> offset, size_t new_size) {
    if (offset.IsNull()) {
      return AllocateOffset(new_size);
    }

    // Per-tblock realloc disabled — same cross-process corruption.
    OffsetPtr<> new_offset = AllocateOffset(new_size);
    if (new_offset.IsNull()) {
      return new_offset;
    }

    // Read old size from BuddyPage header
    size_t page_offset = offset.load() - sizeof(BuddyPage<>);
    auto *page = reinterpret_cast<BuddyPage<>*>(
        GetBackendData() + page_offset);
    size_t old_size = page->size_;

    char *old_data = GetBackendData() + offset.load();
    char *new_data = GetBackendData() + new_offset.load();
    size_t copy_size = (old_size < new_size) ? old_size : new_size;
    memcpy(new_data, old_data, copy_size);

    FreeOffset(offset);
    return new_offset;
  }

  /**
   * Free allocated memory.
   *
   * @param offset Offset pointer to memory to free
   */
  void FreeOffset(OffsetPtr<> offset) {
    if (offset.IsNull()) {
      return;
    }
    FreeOffsetNoNullCheck(offset);
  }

  /** No-op TLS management (handled by EnsureTls). */
  void CreateTls() {}
  void FreeTls() {}

  /** Arena support not implemented for this allocator. */
  bool PushArenaState(ArenaState &prior, OffsetPtr<> &block, size_t size) {
    (void)prior; (void)block; (void)size;
    return false;
  }

  void PopArenaState(const ArenaState &prior, OffsetPtr<> block) {
    (void)prior; (void)block;
  }

 private:
  /**
   * Expand PcThreadBlock from global allocator and retry allocation.
   *
   * @param tblock The thread's PcThreadBlock to expand
   * @param size Size in bytes to allocate
   * @return Offset pointer to allocated memory, or null on failure
   */
  OffsetPtr<> ExpandAndAllocate(PcThreadBlock *tblock, size_t size) {
    // Calculate expansion size with 25% overhead for metadata
    size_t required_size = size + (size / 4) + sizeof(BuddyPage<>);
    size_t expand_size = (required_size > thread_unit_)
                           ? required_size : thread_unit_;

    OffsetPtr<> expand_ptr;
    {
      ScopedMutex scoped_lock(lock_, 0);
      expand_ptr = alloc_.AllocateOffset(expand_size);
    }
    if (expand_ptr.IsNull()) {
      return OffsetPtr<>::GetNull();
    }

    tblock->Expand(expand_ptr, expand_size);
    return tblock->alloc_.AllocateOffset(size);
  }
};

}  // namespace ctp::ipc

#endif  // CTP_MEMORY_ALLOCATOR_MP_ALLOCATOR_H_
