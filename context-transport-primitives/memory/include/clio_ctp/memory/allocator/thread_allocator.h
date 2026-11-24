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

#ifndef CTP_MEMORY_ALLOCATOR_THREAD_ALLOCATOR_H_
#define CTP_MEMORY_ALLOCATOR_THREAD_ALLOCATOR_H_

#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/memory/allocator/buddy_allocator.h"

namespace ctp::ipc {

class _PartitionedAllocator;
typedef BaseAllocator<_PartitionedAllocator> PartitionedAllocator;
// Backward-compatibility typedef
// ThreadAllocator is a legacy name; use PartitionedAllocator
typedef PartitionedAllocator ThreadAllocator;

/**
 * Per-thread allocation partition.
 *
 * Each thread (CPU thread or GPU block) gets its own TaThreadBlock with
 * a private BuddyAllocator. The BuddyAllocator MUST BE LAST because it
 * is followed by its variable-size managed region.
 */
struct TaThreadBlock {
  ctp::ipc::atomic<int> initialized_;  /**< 0=uninitialized, 1=ready */
  PrivateBuddyAllocator alloc_;   /**< Private buddy allocator (MUST BE LAST) */

  CTP_CROSS_FUN
  TaThreadBlock() : initialized_(0) {}

  CTP_CROSS_FUN
  bool shm_init(const MemoryBackend &backend, size_t region_size) {
    size_t alloc_region_size = region_size - sizeof(TaThreadBlock);
    alloc_.shm_init(backend, alloc_region_size);
    initialized_.store(1);
#if !CTP_IS_HOST
    __threadfence_system();
#endif
    return true;
  }
};

/**
 * Partitioned allocator with per-thread/block BuddyAllocator partitions.
 *
 * Designed for single-process, multi-threaded (or multi-GPU-block) use.
 * Each thread/block gets its own BuddyAllocator partition, lazily
 * initialized on first use. Partitions are pre-allocated as a contiguous
 * fixed-size table during shm_init — no dynamic allocation is needed
 * at runtime.
 *
 * Thread ID mapping:
 *   CPU: caller-provided tid (e.g., thread_local counter)
 *   GPU: blockIdx.x % max_threads_
 *
 * Memory layout (within the backend region):
 *   [_PartitionedAllocator header]
 *   [TaThreadBlock partition 0 (thread_unit_ bytes)]
 *   [TaThreadBlock partition 1 (thread_unit_ bytes)]
 *   ...
 *   [TaThreadBlock partition N-1 (thread_unit_ bytes)]
 */
class _PartitionedAllocator : public Allocator {
 public:
  ctp::ipc::atomic<int> heap_ready_;  /**< 0=not ready, 1=ready (grid-level sync) */
  volatile int max_threads_;     /**< Fixed thread count (set at init).
                                      Volatile: cross-block reads must bypass L1
                                      after WaitReady to see block 0's shm_init. */
  volatile size_t thread_unit_;  /**< Bytes per thread partition (volatile: same) */
  char * volatile base_;         /**< Cached base pointer (volatile: same) */

 public:
  CTP_CROSS_FUN
  _PartitionedAllocator()
      : heap_ready_(0), max_threads_(0), thread_unit_(0), base_(nullptr) {}

  /**
   * Initialize the thread allocator.
   *
   * Divides the region after the header into max_threads fixed-size
   * partitions of thread_unit bytes each. No allocation happens here —
   * each partition's BuddyAllocator is lazily initialized on first use
   * by the owning thread.
   *
   * @param backend Memory backend
   * @param region_size Size of region (0 = entire backend)
   * @param max_threads Maximum number of threads/GPU blocks
   * @param thread_unit Bytes per thread partition
   * @return true on success
   */
  CTP_CROSS_FUN
  bool shm_init(const MemoryBackend &backend,
                size_t region_size = 0,
                int max_threads = 32,
                size_t thread_unit = 1024 * 1024) {
    if (region_size == 0) {
      region_size = backend.data_capacity_;
    }

    SetBackend(backend);
    this_ = reinterpret_cast<char *>(this) -
            reinterpret_cast<char *>(backend.data_);
    base_ = reinterpret_cast<char *>(backend.data_);
    alloc_header_size_ = sizeof(_PartitionedAllocator);
    data_start_ = sizeof(_PartitionedAllocator);
    region_size_ = region_size;
    max_threads_ = max_threads;
    // Align thread_unit down to 16 bytes so partition starts stay aligned
    // for GPU atomics (atomicExch requires 4-byte alignment minimum)
    thread_unit_ = thread_unit & ~(size_t)15;

    // Zero-init all partition headers so initialized_ starts at 0
    char *base = base_;
    for (int i = 0; i < max_threads_; ++i) {
      char *part = base + sizeof(_PartitionedAllocator) +
                   static_cast<size_t>(i) * thread_unit_;
      auto *block = reinterpret_cast<TaThreadBlock *>(part);
      block->initialized_.store(0);
    }

    return true;
  }

  /**
   * Get a TaThreadBlock by tid.
   * O(1) — computed directly from the fixed partition layout.
   */
  CTP_INLINE_CROSS_FUN
  TaThreadBlock* GetThreadBlock(int tid) {
    char *b = base_;  // volatile read (bypasses L1)
    size_t tu = thread_unit_;  // volatile read
    char *part = b + sizeof(_PartitionedAllocator) +
                 static_cast<size_t>(tid) * tu;
    return reinterpret_cast<TaThreadBlock *>(part);
  }

  /**
   * Lazily initialize the thread partition for the given tid.
   * No mutex needed — each tid is only initialized by its owning thread.
   */
  CTP_CROSS_FUN
  bool LazyInitThread(int tid) {
    if (tid < 0 || tid >= max_threads_) {
      return false;
    }

    TaThreadBlock *block = GetThreadBlock(tid);
    // Use load_device() to bypass L1 cache — the initialized_ flag may have
    // been set by a different SM (block 0's shm_init) or be stale from a
    // previous kernel launch on this SM.
    if (block->initialized_.load_device() == 1) {
      return true;
    }

    // Initialize in-place
    new (block) TaThreadBlock();
    MemoryBackend backend = GetBackend();
    block->shm_init(backend, thread_unit_);
    return true;
  }

  /**
   * Get the auto-detected partition ID (global warp index).
   * GPU: global warp ID, or -1 if beyond max_threads_
   * CPU: 0 (caller should provide explicit tid for multi-threaded use)
   */
  CTP_INLINE_CROSS_FUN
  int GetAutoTid() {
#if CTP_IS_GPU
    int tid = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
    if (tid >= max_threads_) return -1;
    return tid;
#else
    return 0;
#endif
  }

  /**
   * Allocate memory with auto-detected tid.
   */
  CTP_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size) {
    return AllocateOffset(size, GetAutoTid());
  }

  /**
   * Allocate memory with explicit tid.
   */
  CTP_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size, int tid) {
    if (!LazyInitThread(tid)) {
      return OffsetPtr<>::GetNull();
    }
    return GetThreadBlock(tid)->alloc_.AllocateOffset(size);
  }

  /**
   * Reallocate memory (not supported — allocate new + copy manually).
   */
  CTP_CROSS_FUN
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    (void)p;
    (void)new_size;
    return OffsetPtr<>::GetNull();
  }

  /**
   * Free memory.
   *
   * Computes the owning thread partition in O(1) by address arithmetic,
   * then frees to that partition's SlabAllocator.
   */
  CTP_CROSS_FUN
  void FreeOffsetNoNullCheck(OffsetPtr<> p) {
    char *ptr_addr = base_ + p.load();
    char *partitions_base = base_ + sizeof(_PartitionedAllocator);
    size_t offset_in_partitions =
        static_cast<size_t>(ptr_addr - partitions_base);
    int owner = static_cast<int>(offset_in_partitions / thread_unit_);
    if (owner >= 0 && owner < max_threads_) {
      GetThreadBlock(owner)->alloc_.FreeOffsetNoNullCheck(p);
    }
  }

  /**
   * Free memory (null-safe wrapper).
   */
  CTP_CROSS_FUN
  void FreeOffset(OffsetPtr<> p) {
    if (p.IsNull()) return;
    FreeOffsetNoNullCheck(p);
  }

#ifdef CTP_BUDDY_ALLOC_DEBUG
  /** Get the BuddyAllocator for a given thread/partition (debug only) */
  CTP_CROSS_FUN PrivateBuddyAllocator *DbgGetPartition(int tid) {
    if (tid >= 0 && tid < max_threads_) {
      auto *block = GetThreadBlock(tid);
      if (block->initialized_.load_device() == 1) {
        return &block->alloc_;
      }
    }
    return nullptr;
  }
  CTP_CROSS_FUN int DbgMaxThreads() const { return max_threads_; }
#endif

  /** Push arena on the current thread's BuddyAllocator */
  CTP_CROSS_FUN bool PushArenaState(ArenaState &prior, OffsetPtr<> &block, size_t size) {
    int tid = GetAutoTid();
    if (!LazyInitThread(tid)) return false;
    return GetThreadBlock(tid)->alloc_.PushArenaState(prior, block, size);
  }

  /** Pop arena on the owning thread's BuddyAllocator */
  CTP_CROSS_FUN void PopArenaState(const ArenaState &prior, OffsetPtr<> block) {
    if (block.IsNull()) return;
    char *ptr_addr = base_ + block.load();
    char *partitions_base = base_ + sizeof(_PartitionedAllocator);
    size_t offset_in_partitions =
        static_cast<size_t>(ptr_addr - partitions_base);
    int owner = static_cast<int>(offset_in_partitions / thread_unit_);
    if (owner >= 0 && owner < max_threads_) {
      GetThreadBlock(owner)->alloc_.PopArenaState(prior, block);
    }
  }

  /**
   * Get the PrivateBuddyAllocator for the current warp.
   * Calls GetAutoTid + LazyInitThread, then returns the partition's allocator.
   * @return Pointer to the warp's BuddyAllocator, or nullptr on failure
   */
  CTP_INLINE_CROSS_FUN
  PrivateBuddyAllocator* GetWarpAllocator() {
    int tid = GetAutoTid();
    if (!LazyInitThread(tid)) return nullptr;
    return &GetThreadBlock(tid)->alloc_;
  }

  /** Get the PrivateBuddyAllocator pointer for a specific tid (no init). */
  CTP_INLINE_CROSS_FUN
  PrivateBuddyAllocator* GetWarpAllocatorByTid(int tid) {
    if (tid < 0 || tid >= max_threads_) return nullptr;
    return &GetThreadBlock(tid)->alloc_;
  }

  /** No-op TLS management (thread IDs are caller-provided) */
  CTP_CROSS_FUN void CreateTls() {}
  CTP_CROSS_FUN void FreeTls() {}

  /**
   * Mark the allocator as ready (grid-level sync).
   * Call after shm_init completes on the initializing thread.
   */
  CTP_CROSS_FUN void MarkReady() {
    heap_ready_.store(1);
#if !CTP_IS_HOST
    __threadfence_system();
#endif
  }

  /**
   * Spin-wait until the allocator is marked ready.
   * Used by non-initializing GPU blocks to wait for block 0.
   */
  CTP_CROSS_FUN void WaitReady() {
#if !CTP_IS_HOST
    // Use load_device() (atomicAdd-based) to bypass per-SM L1 cache.
    while (heap_ready_.load_device() != 1) {
      ctp::ipc::threadfence();
    }
    ctp::ipc::threadfence_system();
#endif
  }
};

}  // namespace ctp::ipc

#endif  // CTP_MEMORY_ALLOCATOR_THREAD_ALLOCATOR_H_
