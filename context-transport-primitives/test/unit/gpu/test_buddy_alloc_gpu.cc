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
 * GPU BuddyAllocator unit test
 *
 * Tests that BuddyAllocator can be initialized and used entirely from within
 * a GPU kernel.  Each GPU thread gets an independent 1 MB region of
 * GpuShmMmap (pinned, device-accessible) and uses it to back a private
 * BuddyAllocator.  The thread then allocates as many 408-byte FakeBig408
 * structs as the allocator will serve, writes a per-(thread,index) magic
 * pattern into each, reads it back immediately, and reports the count.
 *
 * 408 bytes is chosen because it equals sizeof(BuddyAllocator) itself —
 * a realistic "large object" that exercises the small-allocation path
 * (< 16 KB threshold) and forces multiple RepopulateSmallArena() calls.
 *
 * Stack-size requirement (measured with ptxas -v):
 *   When BuddyAllocator is inlined into the clio_gpu_orchestrator kernel
 *   it adds 80 B to the stack frame (752 B vs 672 B for ArenaAllocator),
 *   well within the 4 096 B limit used by the orchestrator.  In isolation
 *   (this test) the kernel compiles to 0 bytes stack frame — everything fits
 *   in registers — so even 2 048 B passes.  4 096 B is used here to match
 *   the orchestrator's runtime setting.
 */

#include <catch2/catch_all.hpp>

#include "clio_ctp/memory/allocator/buddy_allocator.h"
#include "clio_ctp/memory/backend/gpu_shm_mmap.h"
#include "clio_ctp/util/gpu_api.h"

using ctp::ipc::BuddyAllocator;
using ctp::ipc::GpuShmMmap;
using ctp::ipc::MemoryBackendId;

// ─── Test struct ─────────────────────────────────────────────────────────────

/**
 * 408-byte POD struct — matches sizeof(BuddyAllocator).
 * magic0_ and magic1_ bracket a 400-byte body so both header and
 * tail corruption are detected.  Init/Check are callable from GPU.
 */
struct FakeBig408 {
  ctp::u32 magic0_;  ///< = (tid << 16) | idx
  char      body_[400];
  ctp::u32 magic1_;  ///< = ~magic0_

  CTP_INLINE_CROSS_FUN void Init(ctp::u32 tid, ctp::u32 idx) {
    ctp::u32 m = (tid << 16) | (idx & 0xFFFFu);
    magic0_ = m;
    magic1_ = ~m;
    for (int i = 0; i < 400; ++i) {
      body_[i] = static_cast<char>((m >> ((i % 4) * 8)) & 0xFF);
    }
  }

  CTP_INLINE_CROSS_FUN bool Check(ctp::u32 tid, ctp::u32 idx) const {
    ctp::u32 m = (tid << 16) | (idx & 0xFFFFu);
    if (magic0_ != m) return false;
    if (magic1_ != ~m) return false;
    for (int i = 0; i < 400; ++i) {
      if (body_[i] != static_cast<char>((m >> ((i % 4) * 8)) & 0xFF)) {
        return false;
      }
    }
    return true;
  }
};

static_assert(sizeof(FakeBig408) == 408,
              "FakeBig408 must be exactly 408 bytes");

// ─── GPU kernel ──────────────────────────────────────────────────────────────

/**
 * Each thread partitions the GpuShmMmap backend into per-thread slices of
 * @p per_thread_bytes.  It placement-news a BuddyAllocator at the start of
 * its slice, calls shm_init with a sub-backend whose data_ equals the full
 * backend base (so ShmPtr offsets are globally valid), then allocates
 * FakeBig408 objects until the allocator is exhausted.
 *
 * Result codes written to d_results[tid]:
 *   -1   shm_init failed
 *   -2   pattern check failed (data corruption)
 *   ≥ 0  number of FakeBig408 successfully allocated and verified
 */
__global__ void BuddyAllocKernel(
    char                   *backend_base,
    size_t                  total_capacity,
    size_t                  per_thread_bytes,
    ctp::ipc::MemoryBackendId   backend_id,
    int                    *d_results) {

  int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);

  // ── Build a per-thread BuddyAllocator on its 1 MB slice ──────────────────
  // The allocator object lives at the very start of the slice; shm_init
  // tells it to manage [this_, this_ + per_thread_bytes).
  char *slice = backend_base + static_cast<size_t>(tid) * per_thread_bytes;
  auto *alloc = reinterpret_cast<BuddyAllocator *>(slice);
  new (alloc) BuddyAllocator();

  // Sub-backend: data_ = full backend base so every OffsetPtr offset is
  // a valid index into the entire 32 MB region (matches ClientInitGpu pattern).
  ctp::ipc::MemoryBackend sub_backend;
  sub_backend.data_          = backend_base;
  sub_backend.data_capacity_ = total_capacity;
  sub_backend.id_            = backend_id;

  alloc->shm_init(sub_backend, per_thread_bytes);
  // Validate init succeeded: the allocator must be able to serve at least one
  // allocation (IsNull check on the first alloc is the real gate below).

  // ── Allocate FakeBig408 structs until the allocator is full ───────────────
  int count = 0;
  while (true) {
    auto fp = alloc->template AllocateObjs<FakeBig408>(1);
    if (fp.IsNull()) break;

    fp.ptr_->Init(static_cast<ctp::u32>(tid),
                  static_cast<ctp::u32>(count));

    if (!fp.ptr_->Check(static_cast<ctp::u32>(tid),
                        static_cast<ctp::u32>(count))) {
      d_results[tid] = -2;  // corruption
      return;
    }
    ++count;
  }

  d_results[tid] = count;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

TEST_CASE("BuddyAllocatorGpu", "[gpu][allocator]") {
  /**
   * 32 GPU threads, each backed by 1 MB of GpuShmMmap.
   *
   * Expected outcome:
   *   - Every thread initialises its BuddyAllocator without error.
   *   - Every thread allocates at least 1 FakeBig408 struct.
   *   - No pattern-check failures are reported.
   *
   * The stack limit is set to 16 384 B before the launch; without this the
   * deep BuddyAllocator call chain overflows the default 1 024 B stack and
   * causes cudaErrorMisalignedAddress (error 716).
   */
  SECTION("Alloc408ByteStructs1MBPerThread") {
    constexpr int    kNumThreads     = 32;
    constexpr size_t kPerThreadBytes = 1u * 1024u * 1024u;   // 1 MB
    constexpr size_t kBackendSize    = kNumThreads * kPerThreadBytes;  // 32 MB

    // ptxas -v shows this kernel compiles to 0 bytes stack frame (fully
    // register-allocated).  4 096 B matches the CLIO Runtime orchestrator setting
    // and is generous for this test; even 2 048 B passes in practice.
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    // Allocate 32 MB of pinned, device-accessible memory.
    GpuShmMmap   backend;
    MemoryBackendId backend_id(42, 0);
    REQUIRE(backend.shm_init(backend_id, kBackendSize,
                             "/test_buddy_alloc_gpu", 0));

    // Per-thread result array in pinned host memory (readable after sync).
    int *d_results = nullptr;
    cudaMallocHost(&d_results, kNumThreads * sizeof(int));
    REQUIRE(d_results != nullptr);
    memset(d_results, 0, kNumThreads * sizeof(int));

    BuddyAllocKernel<<<1, kNumThreads>>>(
        backend.data_,
        kBackendSize,
        kPerThreadBytes,
        backend_id,
        d_results);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    for (int i = 0; i < kNumThreads; ++i) {
      INFO("Thread " << i << " result: " << d_results[i]);
      // -1 = shm_init failed, -2 = data corruption, <0 = any error
      REQUIRE(d_results[i] >= 0);
      // Each thread has 1 MB; after BuddyAllocator overhead and the first
      // RepopulateSmallArena (64 KB arena → 128 × 512-byte slots for 408-byte
      // objects) at least 100 allocations must succeed.
      REQUIRE(d_results[i] >= 100);
    }

    cudaFreeHost(d_results);
    // backend destructor cleans up GpuShmMmap
  }
}
