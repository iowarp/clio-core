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
 * GPU ThreadAllocator unit test
 *
 * Tests that ThreadAllocator can be initialized on CPU and then used from
 * multiple GPU blocks, each using blockIdx.x as its tid. Each block
 * allocates/frees small objects and verifies no corruption.
 */

#include <type_traits>
#include <catch2/catch_all.hpp>

#include "clio_ctp/memory/allocator/thread_allocator.h"
#include "clio_ctp/memory/backend/gpu_shm_mmap.h"
#include "clio_ctp/util/gpu_api.h"

using ctp::ipc::PartitionedAllocator;
using ctp::ipc::GpuShmMmap;
using ctp::ipc::MemoryBackendId;

/**
 * Init kernel: initialize the ThreadAllocator on GPU-accessible memory.
 * Run with 1 thread, 1 block.
 */
__global__ void ThreadAllocInitKernel(
    char *backend_base,
    size_t data_capacity,
    ctp::ipc::MemoryBackendId backend_id,
    int max_threads) {

  auto *alloc = reinterpret_cast<ctp::ipc::_PartitionedAllocator*>(backend_base);
  new (alloc) ctp::ipc::_PartitionedAllocator();

  ctp::ipc::MemoryBackend sub_backend;
  sub_backend.data_ = backend_base;
  sub_backend.data_capacity_ = data_capacity;
  sub_backend.id_ = backend_id;

  size_t thread_unit = (data_capacity - sizeof(ctp::ipc::_PartitionedAllocator)) /
                       max_threads;
  alloc->shm_init(sub_backend, 0, max_threads, thread_unit);
}

/**
 * 128-byte POD struct for testing.
 */
struct TestObj128 {
  ctp::u32 magic0_;
  char body_[120];
  ctp::u32 magic1_;

  CTP_INLINE_CROSS_FUN void Init(ctp::u32 tid, ctp::u32 idx) {
    ctp::u32 m = (tid << 16) | (idx & 0xFFFFu);
    magic0_ = m;
    magic1_ = ~m;
    for (int i = 0; i < 120; ++i) {
      body_[i] = static_cast<char>((m >> ((i % 4) * 8)) & 0xFF);
    }
  }

  CTP_INLINE_CROSS_FUN bool Check(ctp::u32 tid, ctp::u32 idx) const {
    ctp::u32 m = (tid << 16) | (idx & 0xFFFFu);
    if (magic0_ != m) return false;
    if (magic1_ != ~m) return false;
    for (int i = 0; i < 120; ++i) {
      if (body_[i] != static_cast<char>((m >> ((i % 4) * 8)) & 0xFF)) {
        return false;
      }
    }
    return true;
  }
};

/**
 * Work kernel: each block allocates/frees TestObj128 using blockIdx.x as tid.
 *
 * Result codes in d_results[blockIdx.x]:
 *   -1  lazy init failed
 *   -2  pattern check failed (data corruption)
 *   >= 0  number of objects successfully allocated and verified
 */
__global__ void ThreadAllocWorkKernel(
    char *backend_base,
    int *d_results) {

  int tid = static_cast<int>(blockIdx.x);
  auto *alloc = reinterpret_cast<ctp::ipc::_PartitionedAllocator*>(backend_base);

  // Lazily initialize this thread's partition
  if (!alloc->LazyInitThread(tid)) {
    d_results[tid] = -1;
    return;
  }

  auto *block = alloc->GetThreadBlock(tid);

  // Allocate from this thread block's BuddyAllocator
  int count = 0;
  while (count < 100) {
    ctp::ipc::OffsetPtr<> off = block->alloc_.AllocateOffset(sizeof(TestObj128));
    if (off.IsNull()) break;

    auto *obj = reinterpret_cast<TestObj128*>(
        backend_base + off.load());
    obj->Init(static_cast<ctp::u32>(tid),
              static_cast<ctp::u32>(count));

    if (!obj->Check(static_cast<ctp::u32>(tid),
                    static_cast<ctp::u32>(count))) {
      d_results[tid] = -2;
      return;
    }

    // Free immediately to test alloc/free cycling
    block->alloc_.FreeOffsetNoNullCheck(off);
    ++count;
  }

  d_results[tid] = count;
}

/**
 * Cross-block free test: 2 blocks, each with 1 thread.
 * Block 0 allocates kNumObjs objects and stores offsets in shared array.
 * Block 1 frees them all via the ThreadAllocator's FreeOffsetNoNullCheck,
 * which should route each free back to block 0's buddy allocator.
 * Then block 0 re-allocates to prove its partition reclaimed the memory.
 *
 * d_offsets: pinned host array for passing offsets between kernels.
 * d_results[0]: block 0 alloc count (phase 1)
 * d_results[1]: block 1 free count
 * d_results[2]: block 0 re-alloc count (phase 2)
 * d_results[3]: data integrity check (0=ok, -2=corruption)
 */
static constexpr int kCrossBlockObjs = 32;

/** Phase 1: block 0 allocates objects and stores offsets. */
__global__ void CrossBlockAllocKernel(
    char *backend_base,
    unsigned long long *d_offsets,
    int *d_results) {

  int tid = static_cast<int>(blockIdx.x);
  if (tid != 0) return;

  auto *alloc = reinterpret_cast<ctp::ipc::_PartitionedAllocator*>(backend_base);
  if (!alloc->LazyInitThread(0)) {
    d_results[0] = -1;
    return;
  }
  auto *block = alloc->GetThreadBlock(0);

  int count = 0;
  for (int i = 0; i < kCrossBlockObjs; ++i) {
    ctp::ipc::OffsetPtr<> off = block->alloc_.AllocateOffset(sizeof(TestObj128));
    if (off.IsNull()) break;

    auto *obj = reinterpret_cast<TestObj128*>(backend_base + off.load());
    obj->Init(0, static_cast<ctp::u32>(i));
    d_offsets[i] = off.load();
    ++count;
  }
  d_results[0] = count;
}

/** Phase 2: block 1 frees block 0's allocations via ThreadAllocator. */
__global__ void CrossBlockFreeKernel(
    char *backend_base,
    unsigned long long *d_offsets,
    int num_to_free,
    int *d_results) {

  int tid = static_cast<int>(blockIdx.x);
  if (tid != 1) return;

  auto *alloc = reinterpret_cast<ctp::ipc::_PartitionedAllocator*>(backend_base);

  int count = 0;
  for (int i = 0; i < num_to_free; ++i) {
    ctp::ipc::OffsetPtr<> off(d_offsets[i]);
    // Verify data before freeing
    auto *obj = reinterpret_cast<TestObj128*>(backend_base + off.load());
    if (!obj->Check(0, static_cast<ctp::u32>(i))) {
      d_results[3] = -2;
      return;
    }
    alloc->FreeOffsetNoNullCheck(off);
    ++count;
  }
  d_results[1] = count;
}

/** Phase 3: block 0 re-allocates to prove memory was reclaimed. */
__global__ void CrossBlockReallocKernel(
    char *backend_base,
    int *d_results) {

  int tid = static_cast<int>(blockIdx.x);
  if (tid != 0) return;

  auto *alloc = reinterpret_cast<ctp::ipc::_PartitionedAllocator*>(backend_base);
  if (!alloc->LazyInitThread(0)) {
    d_results[2] = -1;
    return;
  }
  auto *block = alloc->GetThreadBlock(0);

  int count = 0;
  for (int i = 0; i < kCrossBlockObjs; ++i) {
    ctp::ipc::OffsetPtr<> off = block->alloc_.AllocateOffset(sizeof(TestObj128));
    if (off.IsNull()) break;

    auto *obj = reinterpret_cast<TestObj128*>(backend_base + off.load());
    obj->Init(99, static_cast<ctp::u32>(i));

    if (!obj->Check(99, static_cast<ctp::u32>(i))) {
      d_results[3] = -2;
      return;
    }

    block->alloc_.FreeOffsetNoNullCheck(off);
    ++count;
  }
  d_results[2] = count;
}

TEST_CASE("ThreadAllocatorGpu", "[gpu][allocator]") {
  SECTION("MultiBlockAllocFree") {
    constexpr int kNumBlocks = 16;
    constexpr size_t kBackendSize = 64u * 1024u * 1024u;  // 64 MB

    ctp::GpuApi::SetDeviceStackLimit(16384);

    GpuShmMmap backend;
    MemoryBackendId backend_id(99, 0);
    REQUIRE(backend.shm_init(backend_id, kBackendSize,
                             "/test_thread_alloc_gpu", 0));

    // Init kernel: set up the allocator (uses backend.data_capacity_)
    ThreadAllocInitKernel<<<1, 1>>>(
        backend.data_,
        backend.data_capacity_,
        backend_id,
        kNumBlocks);
    REQUIRE((ctp::GpuApi::Synchronize(), true));
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // Per-block result array
    int *d_results = nullptr;
    d_results = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(d_results)>>(kNumBlocks * sizeof(int));
    REQUIRE(d_results != nullptr);
    memset(d_results, 0, kNumBlocks * sizeof(int));

    // Work kernel: each block uses its own tid
    ThreadAllocWorkKernel<<<kNumBlocks, 1>>>(
        backend.data_,
        d_results);
    REQUIRE((ctp::GpuApi::Synchronize(), true));
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    for (int i = 0; i < kNumBlocks; ++i) {
      INFO("Block " << i << " result: " << d_results[i]);
      REQUIRE(d_results[i] >= 0);
      // Each block has 512KB; with alloc/free cycling, should manage many objects
      REQUIRE(d_results[i] >= 10);
    }

    ctp::GpuApi::FreeHost(d_results);
  }

  SECTION("CrossBlockFree") {
    constexpr int kNumBlocks = 4;
    constexpr size_t kBackendSize = 64u * 1024u * 1024u;

    ctp::GpuApi::SetDeviceStackLimit(16384);

    GpuShmMmap backend;
    MemoryBackendId backend_id(100, 0);
    REQUIRE(backend.shm_init(backend_id, kBackendSize,
                             "/test_thread_alloc_crossfree", 0));

    // Init kernel: set up allocator (uses backend.data_capacity_)
    ThreadAllocInitKernel<<<1, 1>>>(
        backend.data_, backend.data_capacity_, backend_id, kNumBlocks);
    REQUIRE((ctp::GpuApi::Synchronize(), true));
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    // Allocate result + offset arrays (pinned host for GPU access)
    int *d_results = nullptr;
    d_results = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(d_results)>>(4 * sizeof(int));
    REQUIRE(d_results != nullptr);
    memset(d_results, 0, 4 * sizeof(int));

    unsigned long long *d_offsets = nullptr;
    d_offsets = ctp::GpuApi::MallocHost<std::remove_pointer_t<decltype(d_offsets)>>(kCrossBlockObjs * sizeof(unsigned long long));
    REQUIRE(d_offsets != nullptr);
    memset(d_offsets, 0, kCrossBlockObjs * sizeof(unsigned long long));

    // Phase 1: block 0 allocates objects
    CrossBlockAllocKernel<<<1, 1>>>(backend.data_, d_offsets, d_results);
    REQUIRE((ctp::GpuApi::Synchronize(), true));
    REQUIRE(ctp::GpuApi::LastError() == nullptr);
    INFO("Phase 1 alloc count: " << d_results[0]);
    REQUIRE(d_results[0] == kCrossBlockObjs);

    // Phase 2: block 1 frees block 0's allocations
    CrossBlockFreeKernel<<<2, 1>>>(
        backend.data_, d_offsets, d_results[0], d_results);
    REQUIRE((ctp::GpuApi::Synchronize(), true));
    REQUIRE(ctp::GpuApi::LastError() == nullptr);
    INFO("Phase 2 free count: " << d_results[1]);
    INFO("Data integrity: " << d_results[3]);
    REQUIRE(d_results[3] == 0);  // no corruption
    REQUIRE(d_results[1] == kCrossBlockObjs);

    // Phase 3: block 0 re-allocates — memory should be reclaimed
    CrossBlockReallocKernel<<<1, 1>>>(backend.data_, d_results);
    REQUIRE((ctp::GpuApi::Synchronize(), true));
    REQUIRE(ctp::GpuApi::LastError() == nullptr);
    INFO("Phase 3 realloc count: " << d_results[2]);
    INFO("Data integrity: " << d_results[3]);
    REQUIRE(d_results[3] == 0);  // no corruption
    REQUIRE(d_results[2] == kCrossBlockObjs);

    ctp::GpuApi::FreeHost(d_offsets);
    ctp::GpuApi::FreeHost(d_results);
  }
}
