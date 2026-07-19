/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CTP_MEMORY_ALLOCATOR_SLAB_CACHE_ALLOCATOR_H_
#define CTP_MEMORY_ALLOCATOR_SLAB_CACHE_ALLOCATOR_H_

#include "clio_ctp/data_structures/ipc/ring_buffer.h"
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/memory/allocator/malloc_allocator.h"
#include "clio_ctp/thread/thread_model/thread_model.h"

namespace ctp::ipc {

/**
 * Per-thread cache of freed slab regions: an extensible SPSC queue of raw
 * region pointers, owned by SlabAllocator's TLS (one per thread). The cache's
 * own storage always comes from the process MallocAllocator.
 */
struct SlabTls : public thread::ThreadLocalData {
  ext_ring_buffer<void *, MallocAllocator> regions_;

  explicit SlabTls(MallocAllocator *cache_alloc, size_t cap)
      : regions_(cache_alloc, cap) {}
};

/**
 * Private-memory caching slab allocator.
 *
 * Caches freed fixed-size regions in a thread-local extensible SPSC queue and
 * reuses them on the next allocation, falling back to the backing allocator
 * (AllocT — defaults to the process MallocAllocator) only on a cache miss. Free
 * returns the region to the calling thread's cache (or to the backing allocator
 * if the cache is full).
 *
 * Host-only (relies on the thread model's TLS); intended for hot, uniform-size
 * per-thread allocations such as Boost fiber stacks. AllocT must expose the
 * MallocAllocator-style `FullPtr<void> Allocate<void>(size)` / `Free(FullPtr)`
 * surface (BoostStackAllocator mirrors it).
 *
 * @tparam T      Default slab element type (sizeof(T) is the default size)
 * @tparam AllocT Backing allocator type
 */
template <typename T, typename AllocT = MallocAllocator>
class SlabAllocator {
 public:
  AllocT *alloc_;              /**< Backing allocator (cache miss / overflow) */
  size_t slab_size_;           /**< Default region size */
  size_t cap_;                 /**< Per-thread cache capacity */
  thread::ThreadLocalKey key_; /**< TLS key for the per-thread region cache */

  explicit SlabAllocator(AllocT *alloc = CTP_MALLOC,
                         size_t slab_size = sizeof(T), size_t cap = 1024)
      : alloc_(alloc), slab_size_(slab_size), cap_(cap) {
    CTP_THREAD_MODEL->template CreateTls<SlabTls>(key_, nullptr);
  }

  /** Get (lazily creating) this thread's region cache. */
  SlabTls *Tls() {
    SlabTls *tls = CTP_THREAD_MODEL->template GetTls<SlabTls>(key_);
    if (tls == nullptr) {
      tls = new SlabTls(CTP_MALLOC, cap_);
      CTP_THREAD_MODEL->template SetTls<SlabTls>(key_, tls);
    }
    return tls;
  }

  /** Allocate a region: reuse a cached one, else allocate from the backing
   *  allocator. Returns a FullPtr<void> over the region. */
  FullPtr<void> Allocate(size_t size = 0) {
    if (size == 0) {
      size = slab_size_;
    }
    SlabTls *tls = Tls();
    void *region = nullptr;
    if (tls->regions_.Pop(region)) {
      return FullPtr<void>(region);
    }
    return alloc_->template Allocate<void>(size);
  }

  /** Return a region to this thread's cache (or to the backing allocator if the
   *  cache is full). */
  void Free(const FullPtr<void> &region) {
    if (region.ptr_ == nullptr) {
      return;
    }
    SlabTls *tls = Tls();
    if (!tls->regions_.Emplace(region.ptr_)) {
      alloc_->Free(region);
    }
  }
};

}  // namespace ctp::ipc

#endif  // CTP_MEMORY_ALLOCATOR_SLAB_CACHE_ALLOCATOR_H_
