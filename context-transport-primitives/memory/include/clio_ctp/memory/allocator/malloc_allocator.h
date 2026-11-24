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

#ifndef HERMES_SHM_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_
#define HERMES_SHM_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_

#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/memory/allocator/leak_checker.h"
#include <memory>

namespace ctp::ipc {

/**
 * Page header for malloc allocations
 * Stored before each allocated block
 */
struct MallocPage {
  static constexpr uint64_t MAGIC = 0xDEADBEEFCAFEBABEULL;
  uint64_t magic_;        // Magic number to change alignment
  size_t page_size_;      // Size of this allocation including header

  CTP_CROSS_FUN MallocPage() : magic_(MAGIC), page_size_(0) {}
  CTP_CROSS_FUN explicit MallocPage(size_t size) : magic_(MAGIC), page_size_(size) {}
};

/**
 * Allocator that wraps standard malloc/free
 * Uses null allocator ID since memory is not shared
 */
template <bool ATOMIC>
class _MallocAllocator : public Allocator {
 public:
  /** Running total of bytes allocated but not yet freed. Only mutated when
   *  CTP_ALLOC_TRACK_SIZE is defined (see CLIO_CORE_ENABLE_LEAK_CHECK); stays
   *  zero otherwise so GetCurrentlyAllocatedSize() is a cheap, valid no-op.
   *  Mirrors _ArenaAllocator::total_alloc_. */
  ctp::ipc::atomic<ctp::big_uint> total_alloc_;

  /**
   * Default constructor - initializes with null backend
   */
  CTP_CROSS_FUN
  _MallocAllocator() {
    // Create a null backend with max capacity so ContainsPtr always returns true
    MemoryBackend null_backend;
    null_backend.id_ = MemoryBackendId::GetNull();
    null_backend.backend_size_ = 0;
    null_backend.data_capacity_ = SIZE_MAX;  // Accept all pointer offsets
    null_backend.header_ = nullptr;
    null_backend.region_ = nullptr;
    null_backend.data_ = nullptr;
    SetBackend(null_backend);

    // Minimal header size
    alloc_header_size_ = sizeof(_MallocAllocator);

    // No fixed region size (allocates on demand via malloc)
    region_size_ = 0;
    data_start_ = 0;
    // Set this_ so that GetBackendData() returns 0
    // Formula: GetBackendData() = this - this_ = 0
    // Then: 0 + raw_pointer_as_offset = raw_pointer
    this_ = reinterpret_cast<size_t>(this);

    // Always zero-initialize the tracking counter (atomic's default ctor does
    // not), so GetCurrentlyAllocatedSize() reads 0 even when tracking is off.
    total_alloc_ = 0;
  }

  /**
   * Destructor. In leak-tracking builds (CTP_ALLOC_TRACK_SIZE) report to the
   * process-global AllocatorLeakChecker if this allocator is being torn down
   * with a non-empty heap — e.g. CTP_MALLOC (MallocAllocatorSingleton) at
   * static teardown after a task/buffer/NewObj was never freed. A no-op in
   * normal builds (total_alloc_ is a constant 0) and on the device (no
   * process-lifetime teardown to observe).
   */
  CTP_CROSS_FUN
  ~_MallocAllocator() {
#if CTP_IS_HOST && defined(CTP_ALLOC_TRACK_SIZE)
    size_t outstanding = static_cast<size_t>(total_alloc_.load());
    if (outstanding != 0) {
      ::ctp::ipc::AllocatorLeakChecker::Get().Report("MallocAllocator",
                                                     outstanding);
    }
#endif
  }

  /**
   * Initialize allocator (for compatibility with backend pattern)
   */
  CTP_CROSS_FUN
  void shm_init(const MemoryBackend &backend, size_t region_size = 0) {
    // Ignore backend - we use malloc directly
    (void)backend;
    (void)region_size;
  }

  /**
   * Attach to existing allocator
   * Not supported for malloc allocator (no-op on GPU where exceptions are unavailable)
   */
  CTP_CROSS_FUN
  void shm_attach(const MemoryBackend &backend) {
    (void)backend;
#if CTP_IS_HOST
    throw SHMEM_NOT_SUPPORTED.format();
#endif
  }

  /**
   * Allocate memory via malloc
   */
  CTP_CROSS_FUN
  OffsetPtr<> AllocateOffset(size_t size) {
    // Allocate space for header + user data
    size_t total_size = sizeof(MallocPage) + size;
    void *ptr = malloc(total_size);

    if (!ptr) {
      return OffsetPtr<>();  // Null pointer
    }

    // Initialize page header
    MallocPage *page = reinterpret_cast<MallocPage*>(ptr);
    page->page_size_ = total_size;

    // User data starts after header
    void *user_ptr = reinterpret_cast<char*>(ptr) + sizeof(MallocPage);

#ifdef CTP_ALLOC_TRACK_SIZE
    total_alloc_ += size;
#endif

    // Return offset (cast pointer to offset for interface compatibility)
    return OffsetPtr<>(reinterpret_cast<size_t>(user_ptr));
  }

  /**
   * Reallocate memory.
   * On host: uses realloc for efficiency.
   * On GPU: realloc is unavailable; falls back to malloc+memcpy+free.
   */
  CTP_CROSS_FUN
  OffsetPtr<> ReallocateOffsetNoNullCheck(OffsetPtr<> p, size_t new_size) {
    void *user_ptr = reinterpret_cast<void*>(p.load());
    MallocPage *old_page = reinterpret_cast<MallocPage*>(
        reinterpret_cast<char*>(user_ptr) - sizeof(MallocPage));
    size_t old_size = old_page->page_size_ - sizeof(MallocPage);
    size_t new_total_size = sizeof(MallocPage) + new_size;

#if CTP_IS_GPU
    // CUDA device code: no realloc, use malloc+memcpy+free
    void *new_raw = malloc(new_total_size);
    if (!new_raw) return OffsetPtr<>();
    size_t copy_size = old_size < new_size ? old_size : new_size;
    memcpy(reinterpret_cast<char*>(new_raw) + sizeof(MallocPage),
           user_ptr, copy_size);
    free(old_page);
    MallocPage *new_page = reinterpret_cast<MallocPage*>(new_raw);
    new_page->page_size_ = new_total_size;
    void *new_user = reinterpret_cast<char*>(new_raw) + sizeof(MallocPage);
#else
    void *new_ptr = realloc(old_page, new_total_size);
    if (!new_ptr) return OffsetPtr<>();
    MallocPage *new_page = reinterpret_cast<MallocPage*>(new_ptr);
    new_page->page_size_ = new_total_size;
    void *new_user = reinterpret_cast<char*>(new_ptr) + sizeof(MallocPage);
#endif

#ifdef CTP_ALLOC_TRACK_SIZE
    total_alloc_ += (new_size - old_size);
#endif
    return OffsetPtr<>(reinterpret_cast<size_t>(new_user));
  }

  /**
   * Free memory via free
   */
  CTP_CROSS_FUN
  void FreeOffsetNoNullCheck(OffsetPtr<> p) {
    // Get page header (before user data)
    void *user_ptr = reinterpret_cast<void*>(p.load());
    MallocPage *page = reinterpret_cast<MallocPage*>(
        reinterpret_cast<char*>(user_ptr) - sizeof(MallocPage));

#ifdef CTP_ALLOC_TRACK_SIZE
    size_t size = page->page_size_ - sizeof(MallocPage);
    total_alloc_ -= size;
#endif

    // Free entire allocation (header + data)
    free(page);
  }

  /**
   * Get currently allocated size
   */
  CTP_CROSS_FUN
  size_t GetCurrentlyAllocatedSize() {
    // total_alloc_ is only incremented/decremented when CTP_ALLOC_TRACK_SIZE
    // is defined; otherwise it stays 0, so this is a valid no-op either way.
    return static_cast<size_t>(total_alloc_.load());
  }

  /**
   * Thread-local storage (no-op for malloc)
   */
  CTP_CROSS_FUN void CreateTls() {}
  CTP_CROSS_FUN void FreeTls() {}

  /** Not supported for malloc allocator */
  CTP_CROSS_FUN bool PushArenaState(ArenaState &prior, OffsetPtr<> &block, size_t size) {
    (void)prior; (void)block; (void)size;
    return false;
  }
  CTP_CROSS_FUN void PopArenaState(const ArenaState &prior, OffsetPtr<> block) {
    (void)prior; (void)block;
  }

  /**
   * Reset: no-op for malloc allocator (individual frees handle reclamation)
   */
  CTP_CROSS_FUN void Reset() {}
};

// Type alias
typedef BaseAllocator<_MallocAllocator<false>> MallocAllocator;

/**
 * Singleton wrapper for MallocAllocator
 * Self-initializing and globally accessible
 */
class MallocAllocatorSingleton {
 public:
  /**
   * Get the singleton instance
   * Initializes lazily on first access
   */
  static MallocAllocator* Get() {
    static MallocAllocatorSingleton instance;
    return &instance.allocator_;
  }

 private:
  MallocAllocatorSingleton() = default;
  ~MallocAllocatorSingleton() = default;

  // Non-copyable, non-movable
  MallocAllocatorSingleton(const MallocAllocatorSingleton&) = delete;
  MallocAllocatorSingleton& operator=(const MallocAllocatorSingleton&) = delete;

  MallocAllocator allocator_;
};

}  // namespace ctp::ipc

// Global accessor macro
#define CTP_MALLOC ::ctp::ipc::MallocAllocatorSingleton::Get()

#endif  // HERMES_SHM_MEMORY_ALLOCATOR_MALLOC_ALLOCATOR_H_
