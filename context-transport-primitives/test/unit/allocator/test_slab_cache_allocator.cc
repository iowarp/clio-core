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

/**
 * SlabAllocator: the per-thread region cache must be BOUNDED.
 *
 * The overflow path was dead code for as long as it existed. regions_ is an
 * ext_ring_buffer (RING_BUFFER_DYNAMIC_SIZE), whose Emplace GROWS a full ring
 * and returns true unconditionally, so `if (!Emplace(...)) alloc_->Free(...)`
 * never fired and cap_ was only the ring's INITIAL capacity: every region a
 * thread freed stayed cached for the life of the process. The real user is the
 * Boost fiber-stack pool, where that is a 256 KiB stack retained per finished
 * fiber, without bound.
 *
 * These tests pin the contract the cache is supposed to have — reuse up to
 * cap_, hand everything past it back to the backing allocator — by counting
 * what the backing allocator is asked to do.
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <vector>

#include "clio_ctp/memory/allocator/slab_cache_allocator.h"

namespace {

/**
 * Backing allocator with the MallocAllocator surface SlabAllocator requires,
 * counting the calls so a test can tell a cache hit from a real allocation.
 */
class CountingAllocator {
 public:
  size_t allocs_ = 0;
  size_t frees_ = 0;

  template <typename T = void>
  ctp::ipc::FullPtr<T> Allocate(size_t size) {
    ++allocs_;
    return ctp::ipc::FullPtr<T>(reinterpret_cast<T *>(std::malloc(size)));
  }

  template <typename T = void>
  void Free(const ctp::ipc::FullPtr<T> &region) {
    ++frees_;
    std::free(const_cast<void *>(reinterpret_cast<const void *>(region.ptr_)));
  }
};

constexpr size_t kSlabSize = 64;
constexpr size_t kCap = 4;

}  // namespace

TEST_CASE("SlabAllocator caches up to cap_ regions and no more",
          "[slab][cache]") {
  CountingAllocator backing;
  ctp::ipc::SlabAllocator<char, CountingAllocator> slab(&backing, kSlabSize,
                                                       kCap);

  // Every region here is fresh: the cache starts empty.
  std::vector<ctp::ipc::FullPtr<void>> regions;
  const size_t kOverflow = 3;
  for (size_t i = 0; i < kCap + kOverflow; ++i) {
    regions.push_back(slab.Allocate());
  }
  REQUIRE(backing.allocs_ == kCap + kOverflow);
  REQUIRE(backing.frees_ == 0);

  // Freeing more than cap_ of them must return the excess to the backing
  // allocator instead of growing the cache. This is the assertion the old code
  // could not satisfy: it cached all seven and called Free zero times.
  for (auto &r : regions) {
    slab.Free(r);
  }
  REQUIRE(backing.frees_ == kOverflow);

  // The cap_ that stayed cached are handed back out without touching the
  // backing allocator...
  const size_t allocs_before = backing.allocs_;
  for (size_t i = 0; i < kCap; ++i) {
    ctp::ipc::FullPtr<void> reused = slab.Allocate();
    REQUIRE(reused.ptr_ != nullptr);
    REQUIRE(backing.allocs_ == allocs_before);
    regions[i] = reused;
  }

  // ...and the next one, with the cache now empty, is a real allocation.
  ctp::ipc::FullPtr<void> fresh = slab.Allocate();
  REQUIRE(backing.allocs_ == allocs_before + 1);

  slab.Free(fresh);
  for (size_t i = 0; i < kCap; ++i) {
    slab.Free(regions[i]);
  }
}

TEST_CASE("SlabAllocator stays bounded under sustained churn",
          "[slab][cache]") {
  CountingAllocator backing;
  ctp::ipc::SlabAllocator<char, CountingAllocator> slab(&backing, kSlabSize,
                                                       kCap);

  // One region at a time, over and over: the cache holds at most one, so this
  // allocates once and then hits the cache forever.
  ctp::ipc::FullPtr<void> r = slab.Allocate();
  for (int i = 0; i < 1000; ++i) {
    slab.Free(r);
    r = slab.Allocate();
  }
  REQUIRE(backing.allocs_ == 1);
  REQUIRE(backing.frees_ == 0);
  slab.Free(r);

  // Now the shape that leaked: hold many regions live, then free them all at
  // once. Only cap_ may be retained, whatever the burst size.
  std::vector<ctp::ipc::FullPtr<void>> burst;
  const size_t kBurst = 256;
  for (size_t i = 0; i < kBurst; ++i) {
    burst.push_back(slab.Allocate());
  }
  const size_t frees_before = backing.frees_;
  for (auto &b : burst) {
    slab.Free(b);
  }
  // The burst's first Allocate drained the one region the churn loop left
  // cached, so the cache was empty going into these frees and absorbs a full
  // kCap of them; everything past that goes back to the backing allocator.
  REQUIRE(backing.frees_ == frees_before + kBurst - kCap);
}
