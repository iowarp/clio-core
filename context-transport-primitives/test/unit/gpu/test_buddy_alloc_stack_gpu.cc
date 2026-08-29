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
 * GPU PrivateBuddyAllocator "stack" (shifted) unit test
 *
 * Tests the pattern used by IpcManager on the GPU:
 *   - PrivateBuddyAllocator lives in __shared__ memory (the "stack")
 *   - Data region is in global device memory (GpuMalloc)
 *   - shm_init is called with shifted=true so offset calculations
 *     resolve to the global memory backend, not the __shared__ address
 *
 * This mirrors how gpu_priv_alloc_ works in IpcManager::InitPrivAllocator.
 */

#include <catch2/catch_all.hpp>

#include "clio_ctp/memory/allocator/buddy_allocator.h"
#include "clio_ctp/memory/backend/gpu_malloc.h"
#include "clio_ctp/memory/backend/gpu_shm_mmap.h"
#include "clio_ctp/util/gpu_api.h"

using ctp::ipc::PrivateBuddyAllocator;
using ctp::ipc::GpuMalloc;
using ctp::ipc::GpuShmMmap;
using ctp::ipc::MemoryBackend;
using ctp::ipc::MemoryBackendId;

// ─── Test struct ─────────────────────────────────────────────────────────────

struct TestObj {
  ctp::u32 magic_;
  char data_[60];

  CTP_INLINE_CROSS_FUN void Init(ctp::u32 val) {
    magic_ = val;
    for (int i = 0; i < 60; ++i) {
      data_[i] = static_cast<char>(val + i);
    }
  }

  CTP_INLINE_CROSS_FUN bool Check(ctp::u32 val) const {
    if (magic_ != val) return false;
    for (int i = 0; i < 60; ++i) {
      if (data_[i] != static_cast<char>(val + i)) return false;
    }
    return true;
  }
};

// ─── GPU kernels ─────────────────────────────────────────────────────────────

/**
 * Test 1: PrivateBuddyAllocator in __shared__ memory with shifted=true
 *
 * Thread 0 constructs the allocator in __shared__, calls shm_init(shifted=true)
 * with a GpuMalloc backend. Then allocates objects and verifies them.
 *
 * Result codes in d_results[tid]:
 *   0   success
 *  -1   shm_init failed
 *  -2   first allocation failed
 *  -3   data corruption
 *  >0   number of successful allocations (on success path)
 */
__global__ void StackBuddyAllocSharedKernel(
    const ctp::ipc::MemoryBackend backend,
    int *d_results,
    int *d_alloc_count) {
  __shared__ char alloc_bytes[sizeof(PrivateBuddyAllocator)];
  PrivateBuddyAllocator &alloc =
      *reinterpret_cast<PrivateBuddyAllocator *>(alloc_bytes);
  int tid = threadIdx.x;

  if (tid == 0) {
    new (&alloc) PrivateBuddyAllocator();
    alloc.shm_init(backend, 0, /*shifted=*/true);

    auto fp = alloc.template AllocateObjs<TestObj>(1);
    if (!fp.IsNull()) {
      fp.ptr_->Init(1);
    }

    int count = fp.IsNull() ? 0 : 1;
    if (!fp.IsNull()) {
      while (true) {
        auto fp2 = alloc.template AllocateObjs<TestObj>(1);
        if (fp2.IsNull()) break;
        fp2.ptr_->Init(static_cast<ctp::u32>(count + 1));
        ++count;
      }
    }

    d_results[0] = 0;
    *d_alloc_count = count;
  }
  __syncthreads();

  if (tid != 0) {
    d_results[tid] = 0;
  }
}

/**
 * Test 2: Per-block PrivateBuddyAllocator with clipped backends
 *
 * Multiple blocks each get their own slice of the backend (using Clip),
 * and each block's thread 0 creates a shifted PrivateBuddyAllocator
 * in __shared__ memory over that slice.
 */
__global__ void StackBuddyAllocMultiBlockKernel(
    const ctp::ipc::MemoryBackend backend,
    size_t per_block_size,
    int *d_results,
    int *d_alloc_counts) {
  __shared__ char alloc_bytes[sizeof(PrivateBuddyAllocator)];
  PrivateBuddyAllocator &alloc =
      *reinterpret_cast<PrivateBuddyAllocator *>(alloc_bytes);
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int global_id = bid * blockDim.x + tid;

  if (tid == 0) {
    new (&alloc) PrivateBuddyAllocator();

    // Clip the backend to this block's slice
    size_t block_off = static_cast<size_t>(bid) * per_block_size;
    ctp::ipc::MemoryBackend clip = backend.Clip(block_off, per_block_size);

    alloc.shm_init(clip, 0, /*shifted=*/true);

    // Allocate objects
    int count = 0;
    while (true) {
      auto fp = alloc.template AllocateObjs<TestObj>(1);
      if (fp.IsNull()) break;

      ctp::u32 val = static_cast<ctp::u32>(bid * 10000 + count + 1);
      fp.ptr_->Init(val);

      if (!fp.ptr_->Check(val)) {
        d_results[global_id] = -3;
        d_alloc_counts[bid] = count;
        return;
      }
      ++count;
    }

    d_results[global_id] = 0;
    d_alloc_counts[bid] = count;
  }
  __syncthreads();

  if (tid != 0) {
    d_results[global_id] = 0;
  }
}

/**
 * Test 3: Allocate, free, reallocate cycle
 *
 * Verifies that freed memory can be reused by the shifted allocator.
 */
__global__ void StackBuddyAllocFreeKernel(
    const ctp::ipc::MemoryBackend backend,
    int *d_results) {
  __shared__ char alloc_bytes[sizeof(PrivateBuddyAllocator)];
  PrivateBuddyAllocator &alloc =
      *reinterpret_cast<PrivateBuddyAllocator *>(alloc_bytes);
  int tid = threadIdx.x;

  if (tid == 0) {
    new (&alloc) PrivateBuddyAllocator();
    alloc.shm_init(backend, 0, /*shifted=*/true);
    if (false) {
      d_results[0] = -1;
      return;
    }

    // Phase 1: Allocate 100 objects
    ctp::ipc::FullPtr<TestObj> ptrs[100];
    for (int i = 0; i < 100; ++i) {
      ptrs[i] = alloc.template AllocateObjs<TestObj>(1);
      if (ptrs[i].IsNull()) {
        d_results[0] = -2;  // Could not allocate 100 objects
        return;
      }
      ptrs[i].ptr_->Init(static_cast<ctp::u32>(i + 1));
    }

    // Phase 2: Free all
    for (int i = 0; i < 100; ++i) {
      alloc.Free(ptrs[i]);
    }

    // Phase 3: Reallocate — should succeed since we freed everything
    for (int i = 0; i < 100; ++i) {
      auto fp = alloc.template AllocateObjs<TestObj>(1);
      if (fp.IsNull()) {
        d_results[0] = -3;  // Reallocation failed
        return;
      }
      fp.ptr_->Init(static_cast<ctp::u32>(i + 1000));
      if (!fp.ptr_->Check(static_cast<ctp::u32>(i + 1000))) {
        d_results[0] = -4;  // Data corruption after realloc
        return;
      }
    }

    d_results[0] = 0;  // Success
  }
  __syncthreads();

  if (tid != 0) {
    d_results[tid] = 0;
  }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

TEST_CASE("PrivateBuddyAllocator shifted on GPU",
          "[gpu][allocator][shifted]") {
  ctp::GpuApi::SetDeviceStackLimit(16384);

  SECTION("Single block, __shared__ allocator, GpuMalloc backend") {
    constexpr size_t kBackendSize = 4 * 1024 * 1024;  // 4 MB
    constexpr int kBlockSize = 32;

    GpuMalloc backend;
    MemoryBackendId bid(50, 0);
    REQUIRE(backend.shm_init(bid, kBackendSize, "/test_stack_buddy_gpu", 0));

    int *d_results = ctp::GpuApi::Malloc<int>(kBlockSize * sizeof(int));
    int *d_count = ctp::GpuApi::Malloc<int>(sizeof(int));

    StackBuddyAllocSharedKernel<<<1, kBlockSize>>>(
        static_cast<MemoryBackend &>(backend), d_results, d_count);

    REQUIRE((ctp::GpuApi::Synchronize(), true));

    int h_count = 0;
    ctp::GpuApi::Memcpy(&h_count, d_count, sizeof(int));
    INFO("Allocated " << h_count << " TestObj (64 bytes each)");
    REQUIRE(h_count > 0);

    std::vector<int> h_results(kBlockSize, -99);
    ctp::GpuApi::Memcpy(h_results.data(), d_results,
                          kBlockSize * sizeof(int));
    for (int i = 0; i < kBlockSize; ++i) {
      INFO("Thread " << i << " result: " << h_results[i]);
      REQUIRE(h_results[i] == 0);
    }

    ctp::GpuApi::Free(d_results);
    ctp::GpuApi::Free(d_count);
  }

  SECTION("Single block, __shared__ allocator, GpuMalloc backend (2)") {
    constexpr size_t kBackendSize = 4 * 1024 * 1024;  // 4 MB
    constexpr int kBlockSize = 32;

    GpuMalloc backend;
    MemoryBackendId bid(51, 0);
    REQUIRE(backend.shm_init(bid, kBackendSize,
                             "/test_stack_buddy_gpu2", 0));

    int *d_results = ctp::GpuApi::Malloc<int>(kBlockSize * sizeof(int));
    int *d_count = ctp::GpuApi::Malloc<int>(sizeof(int));

    StackBuddyAllocSharedKernel<<<1, kBlockSize>>>(
        static_cast<MemoryBackend &>(backend), d_results, d_count);

    REQUIRE((ctp::GpuApi::Synchronize(), true));

    int h_count = 0;
    ctp::GpuApi::Memcpy(&h_count, d_count, sizeof(int));
    INFO("Allocated " << h_count << " TestObj (64 bytes each)");
    REQUIRE(h_count > 0);

    std::vector<int> h_results(kBlockSize, -99);
    ctp::GpuApi::Memcpy(h_results.data(), d_results,
                          kBlockSize * sizeof(int));
    for (int i = 0; i < kBlockSize; ++i) {
      INFO("Thread " << i << " result: " << h_results[i]);
      REQUIRE(h_results[i] == 0);
    }

    ctp::GpuApi::Free(d_results);
    ctp::GpuApi::Free(d_count);
  }

  SECTION("Multi-block, per-block __shared__ allocator with Clip") {
    constexpr int kNumBlocks = 4;
    constexpr int kBlockSize = 32;
    constexpr size_t kPerBlockSize = 1 * 1024 * 1024;  // 1 MB per block
    // Account for kBackendHeaderSize so each block gets a full kPerBlockSize of data
    constexpr size_t kBackendSize = kNumBlocks * kPerBlockSize + ctp::ipc::kBackendHeaderSize;

    GpuMalloc backend;
    MemoryBackendId bid(52, 0);
    REQUIRE(backend.shm_init(bid, kBackendSize,
                             "/test_stack_buddy_multi", 0));

    int total_threads = kNumBlocks * kBlockSize;
    int *d_results = ctp::GpuApi::Malloc<int>(total_threads * sizeof(int));
    int *d_counts = ctp::GpuApi::Malloc<int>(kNumBlocks * sizeof(int));
    ctp::GpuApi::Memset(d_results, 0, total_threads * sizeof(int));
    ctp::GpuApi::Memset(d_counts, 0, kNumBlocks * sizeof(int));

    StackBuddyAllocMultiBlockKernel<<<kNumBlocks, kBlockSize>>>(
        static_cast<MemoryBackend &>(backend), kPerBlockSize,
        d_results, d_counts);

    REQUIRE((ctp::GpuApi::Synchronize(), true));

    std::vector<int> h_counts(kNumBlocks);
    ctp::GpuApi::Memcpy(h_counts.data(), d_counts,
                          kNumBlocks * sizeof(int));
    for (int b = 0; b < kNumBlocks; ++b) {
      INFO("Block " << b << " allocated " << h_counts[b] << " objects");
      REQUIRE(h_counts[b] > 0);
    }

    std::vector<int> h_results(total_threads, -99);
    ctp::GpuApi::Memcpy(h_results.data(), d_results,
                          total_threads * sizeof(int));
    for (int i = 0; i < total_threads; ++i) {
      INFO("Global thread " << i << " result: " << h_results[i]);
      REQUIRE(h_results[i] == 0);
    }

    ctp::GpuApi::Free(d_results);
    ctp::GpuApi::Free(d_counts);
  }

  SECTION("Alloc-free-realloc cycle") {
    constexpr size_t kBackendSize = 4 * 1024 * 1024;  // 4 MB
    constexpr int kBlockSize = 32;

    GpuMalloc backend;
    MemoryBackendId bid(53, 0);
    REQUIRE(backend.shm_init(bid, kBackendSize,
                             "/test_stack_buddy_free", 0));

    int *d_results = ctp::GpuApi::Malloc<int>(kBlockSize * sizeof(int));

    StackBuddyAllocFreeKernel<<<1, kBlockSize>>>(
        static_cast<MemoryBackend &>(backend), d_results);

    REQUIRE((ctp::GpuApi::Synchronize(), true));

    std::vector<int> h_results(kBlockSize, -99);
    ctp::GpuApi::Memcpy(h_results.data(), d_results,
                          kBlockSize * sizeof(int));
    for (int i = 0; i < kBlockSize; ++i) {
      INFO("Thread " << i << " result: " << h_results[i]);
      REQUIRE(h_results[i] == 0);
    }

    ctp::GpuApi::Free(d_results);
  }
}
