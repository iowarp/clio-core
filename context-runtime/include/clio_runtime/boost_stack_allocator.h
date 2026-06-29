/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_BOOST_STACK_ALLOCATOR_H_
#define CLIO_RUNTIME_INCLUDE_BOOST_STACK_ALLOCATOR_H_

#if defined(CLIO_ENABLE_BOOST_COROUTINES)

#include <boost/context/fixedsize_stack.hpp>

#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/memory/allocator/slab_cache_allocator.h"
#include "clio_runtime/task.h"  // clio::run::detail::boost_stack_size()

namespace clio::run {

/**
 * Allocator that hands out Boost.Context fixedsize_stack regions.
 *
 * Mirrors the MallocAllocator surface (Allocate<T>(size) -> FullPtr<T>,
 * Free(FullPtr)) so it can serve as the backing allocator for a
 * ctp::ipc::SlabAllocator. Stacks are a fixed size, so only the stack pointer
 * (sp) needs to be tracked; Free reconstructs the boost stack_context {sp,size}
 * required by fixedsize_stack::deallocate.
 */
class BoostStackAllocator {
 public:
  size_t stack_size_;

  explicit BoostStackAllocator(
      size_t stack_size = clio::run::detail::boost_stack_size())
      : stack_size_(stack_size) {}

  template <typename T = void>
  ctp::ipc::FullPtr<T> Allocate(size_t size = 0) {
    (void)size;  // fixed-size stacks
    boost::context::fixedsize_stack maker(stack_size_);
    boost::context::stack_context sctx = maker.allocate();
    return ctp::ipc::FullPtr<T>(reinterpret_cast<T *>(sctx.sp));
  }

  template <typename T = void>
  void Free(const ctp::ipc::FullPtr<T> &region) {
    if (region.ptr_ == nullptr) {
      return;
    }
    boost::context::stack_context sctx;
    sctx.size = stack_size_;
    sctx.sp = const_cast<void *>(reinterpret_cast<const void *>(region.ptr_));
    boost::context::fixedsize_stack(stack_size_).deallocate(sctx);
  }
};

/**
 * Process-wide, per-thread-cached pool of Boost fiber stacks. The SlabAllocator
 * gives each worker thread its own reuse cache (replacing the former per-worker
 * free-stack ring buffer); a cache miss allocates a fresh stack via
 * BoostStackAllocator, and Free returns the stack to the calling thread's cache.
 */
inline ctp::ipc::SlabAllocator<char, BoostStackAllocator> &BoostStackPool() {
  static BoostStackAllocator backing;
  static ctp::ipc::SlabAllocator<char, BoostStackAllocator> pool(
      &backing, clio::run::detail::boost_stack_size());
  return pool;
}

/**
 * Boost.Context StackAllocator interface (allocate()/deallocate()) backed by the
 * shared, per-thread-cached BoostStackPool(). Pass directly to a Boost fiber via
 * std::allocator_arg — no per-worker WorkerStackAllocator wrapper needed. Empty
 * and trivially copyable, as Boost requires (it copies the allocator into the
 * fiber). Stacks are fixed-size, so the stack_context is reconstructed from the
 * pooled region pointer.
 */
struct BoostStackPoolAllocator {
  boost::context::stack_context allocate() {
    ctp::ipc::FullPtr<void> region = BoostStackPool().Allocate();
    boost::context::stack_context sctx;
    sctx.sp = region.ptr_;
    sctx.size = clio::run::detail::boost_stack_size();
    return sctx;
  }

  void deallocate(boost::context::stack_context &sctx) noexcept {
    BoostStackPool().Free(ctp::ipc::FullPtr<void>(sctx.sp));
  }
};

}  // namespace clio::run

#endif  // CLIO_ENABLE_BOOST_COROUTINES

#endif  // CLIO_RUNTIME_INCLUDE_BOOST_STACK_ALLOCATOR_H_
