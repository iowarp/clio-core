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
 * RoundRobinAllocator GPU unit test: concurrent device-side alloc/free.
 *
 * N child blocks run CONCURRENTLY in one grid; each claims a partition,
 * allocates, writes a unique pattern, reads it back, and frees. Cross-block
 * corruption is what the test exists to catch.
 *
 * HOST-launched children, not CDP. This used a CDP parent (fire-and-forget
 * children + tail launch), but the concurrency under test is the same either
 * way -- what matters is that the child blocks contend for the allocator at
 * the same time -- and nothing in production launches device-side (no
 * cudaStreamFireAndForget outside this file). CDP was also unbuildable under
 * clang-CUDA: CMake's clang device-link step is broken against CUDA 13's
 * fatbinary, and even a hand-driven RDC link produced a binary whose
 * device-side launches were silently dropped. A plain grid needs none of it.
 */

#include <catch2/catch_all.hpp>

#include "clio_ctp/memory/allocator/round_robin_allocator.h"
#include "clio_ctp/util/gpu_api.h"

using ctp::ipc::RoundRobinAllocator;

// ============================================================================
// CDP child kernel: allocate, write pattern, read back, verify, free
// ============================================================================

/**
 * Each child kernel block claims a partition, allocates a buffer,
 * writes a unique pattern, reads it back, and reports success/failure.
 *
 * @param alloc Pointer to the RoundRobinAllocator (in device global memory)
 * @param alloc_size Bytes to allocate per child
 * @param results Per-child result: 1=pass, negative=error code
 * @param child_id Unique ID for this child (for pattern generation)
 */
__global__ void RrChildKernel(
    RoundRobinAllocator *alloc,
    size_t alloc_size,
    int *results) {
  const int child_id = blockIdx.x;
  // Claim a partition (thread 0 only, broadcast via __shared__)
  __shared__ int s_partition;
  if (threadIdx.x == 0) {
    s_partition = alloc->ClaimPartition();
  }
  __syncthreads();
  int part = s_partition;

  // Lazy-init the partition (thread 0 only)
  if (threadIdx.x == 0) {
    if (!alloc->LazyInitPartition(part)) {
      printf("[CHILD %d] LazyInitPartition(%d) FAILED\n", child_id, part);
      results[child_id] = -1;
      return;
    }
  }
  __syncthreads();

  // Allocate a buffer from this partition (thread 0, locked)
  __shared__ char *s_buf;
  __shared__ size_t s_off;
  if (threadIdx.x == 0) {
    auto *pblock = alloc->GetPartitionBlock(part);
    auto off = pblock->LockedAllocate(alloc_size);
    if (off.IsNull()) {
      printf("[CHILD %d] LockedAllocate(%zu, part=%d) FAILED\n",
             child_id, alloc_size, part);
      results[child_id] = -2;
      s_buf = nullptr;
    } else {
      s_off = off.load();
      // Resolve offset: the allocator's base is the backend data start
      char *base = reinterpret_cast<char *>(alloc);
      s_buf = base + s_off;
    }
  }
  __syncthreads();

  if (s_buf == nullptr) return;

  // Write pattern: byte[i] = (child_id + i) & 0xFF
  // All threads participate for speed
  for (size_t i = threadIdx.x; i < alloc_size; i += blockDim.x) {
    s_buf[i] = static_cast<char>((child_id + i) & 0xFF);
  }
  __syncthreads();

  // Verify pattern
  if (threadIdx.x == 0) {
    int mismatches = 0;
    for (size_t i = 0; i < alloc_size; ++i) {
      char expected = static_cast<char>((child_id + i) & 0xFF);
      if (s_buf[i] != expected) {
        if (mismatches == 0) {
          printf("[CHILD %d] MISMATCH at byte %zu: expected 0x%02x got 0x%02x\n",
                 child_id, i, (unsigned char)expected, (unsigned char)s_buf[i]);
        }
        ++mismatches;
      }
    }
    if (mismatches > 0) {
      printf("[CHILD %d] %d / %zu bytes mismatched\n",
             child_id, mismatches, alloc_size);
      results[child_id] = -3;
    } else {
      results[child_id] = 1;  // PASS
    }
  }

  // Free the buffer
  if (threadIdx.x == 0) {
    ctp::ipc::OffsetPtr<> off;
    off = s_off;
    alloc->FreeOffset(off);
  }
}

// ============================================================================
// Init kernel: construct the allocator in device memory
// ============================================================================

__global__ void RrInitKernel(char *alloc_base, size_t alloc_capacity,
                             int num_partitions) {
  if (threadIdx.x != 0) return;
  auto *alloc = reinterpret_cast<RoundRobinAllocator *>(alloc_base);
  new (alloc) RoundRobinAllocator();

  ctp::ipc::MemoryBackend backend;
  backend.data_ = alloc_base;
  backend.data_capacity_ = alloc_capacity;
  backend.id_ = ctp::ipc::MemoryBackendId(999, 0);

  alloc->shm_init(backend, 0, num_partitions, 0);
  alloc->MarkReady();
  printf("[INIT] RoundRobinAllocator ready: %d partitions, %llu bytes\n",
         num_partitions, (unsigned long long)alloc_capacity);
}

namespace {

/** Init the allocator, run `num_children` concurrent child blocks, verify. */
void RunConcurrentChildren(int num_partitions, int num_children,
                           size_t alloc_size, size_t alloc_capacity) {
  char *d_alloc = ctp::GpuApi::Malloc<char>(alloc_capacity);
  REQUIRE(d_alloc != nullptr);

  int *h_results = ctp::GpuApi::MallocHost<int>(
      static_cast<size_t>(num_children) * sizeof(int));
  REQUIRE(h_results != nullptr);
  memset(h_results, 0, static_cast<size_t>(num_children) * sizeof(int));

  RrInitKernel<<<1, 1>>>(d_alloc, alloc_capacity, num_partitions);
  ctp::GpuApi::Synchronize();

  // One grid, one block per child: every child contends for the allocator
  // at the same time, which is the concurrency under test.
  RrChildKernel<<<num_children, 32>>>(
      reinterpret_cast<RoundRobinAllocator *>(d_alloc), alloc_size,
      h_results);
  REQUIRE(ctp::GpuApi::LastError() == nullptr);
  ctp::GpuApi::Synchronize();

  int pass_count = 0;
  int fail_count = 0;
  for (int i = 0; i < num_children; ++i) {
    if (h_results[i] == 1) {
      ++pass_count;
    } else {
      ++fail_count;
      WARN("Child " << i << " failed with code " << h_results[i]);
    }
  }
  INFO("Passed: " << pass_count << " / " << num_children);
  REQUIRE(fail_count == 0);

  ctp::GpuApi::Free(d_alloc);
  ctp::GpuApi::FreeHost(h_results);
}

}  // namespace

// ============================================================================
// Host test
// ============================================================================

TEST_CASE("RoundRobinAllocator - concurrent alloc/free",
          "[gpu][allocator]") {
  RunConcurrentChildren(/*num_partitions=*/16, /*num_children=*/8,
                        /*alloc_size=*/256, /*alloc_capacity=*/4 * 1024 * 1024);
}

TEST_CASE("RoundRobinAllocator - many children stress",
          "[gpu][allocator][stress]") {
  RunConcurrentChildren(/*num_partitions=*/32, /*num_children=*/64,
                        /*alloc_size=*/512,
                        /*alloc_capacity=*/16 * 1024 * 1024);
}
