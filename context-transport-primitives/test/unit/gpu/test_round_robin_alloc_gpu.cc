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
 * RoundRobinAllocator GPU unit test with CDP child kernels.
 *
 * Tests the exact pattern used by the GPU orchestrator:
 *   1. Parent kernel initializes the RoundRobinAllocator
 *   2. Parent launches N CDP child kernels concurrently
 *   3. Each child claims a partition, allocates, writes, reads, frees
 *   4. Verify no corruption across concurrent child kernels
 *
 * CDP flavor: CUDA 12 removed the device-side cudaDeviceSynchronize() that CDP1
 * used to join child grids (it is not even declared for __CUDA_ARCH__ >= 900,
 * so forcing CDP1 cannot bring it back on Hopper). The parent therefore uses
 * CDP2: children go to the fire-and-forget stream and the completion work is
 * tail-launched, which the runtime schedules only after the parent grid and all
 * of its fire-and-forget children have finished. Pre-CDP2 toolkits (CUDA < 12,
 * where cudaStreamFireAndForget is not defined) keep the old CDP1 path.
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
    int *results,
    int child_id) {
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
// Parent kernel: init allocator, launch N CDP children, wait
// ============================================================================

#if defined(cudaStreamFireAndForget)
#define CTP_TEST_CDP2 1
#endif

#ifdef CTP_TEST_CDP2
/**
 * Completion kernel, tail-launched by the parent. A tail launch runs only once
 * the parent grid AND every one of its fire-and-forget children have completed,
 * so this is the CDP2 replacement for the parent's old cudaDeviceSynchronize().
 *
 * @param num_children Number of children that were launched (for the log line)
 * @param done Completion flag (pinned host memory)
 */
__global__ void RrDoneKernel(int num_children, int *done) {
  if (threadIdx.x != 0) return;
  printf("[PARENT] All %d children completed\n", num_children);
  *done = 1;
  __threadfence_system();
}
#endif

/**
 * Parent persistent kernel that:
 *   1. Initializes the RoundRobinAllocator
 *   2. Launches num_children CDP child kernels (fire-and-forget)
 *   3. Joins them — tail launch under CDP2, cudaDeviceSynchronize under CDP1
 *   4. Writes completion flag
 *
 * @param alloc_base Device memory for the allocator
 * @param alloc_capacity Total bytes for the allocator
 * @param num_partitions Number of RR partitions
 * @param num_children Number of CDP child kernels to launch
 * @param alloc_size Bytes each child allocates
 * @param results Per-child results (pinned host memory)
 * @param done Completion flag (pinned host memory)
 */
__global__ void RrParentKernel(
    char *alloc_base,
    size_t alloc_capacity,
    int num_partitions,
    int num_children,
    size_t alloc_size,
    int *results,
    int *done) {
  if (threadIdx.x != 0) return;

  // Initialize the RoundRobinAllocator at the base of device memory
  auto *alloc = reinterpret_cast<RoundRobinAllocator *>(alloc_base);
  new (alloc) RoundRobinAllocator();

  ctp::ipc::MemoryBackend backend;
  backend.data_ = alloc_base;
  backend.data_capacity_ = alloc_capacity;
  backend.id_ = ctp::ipc::MemoryBackendId(999, 0);

  alloc->shm_init(backend, 0, num_partitions, 0);
  alloc->MarkReady();

  printf("[PARENT] RoundRobinAllocator initialized: %d partitions, %zu bytes\n",
         num_partitions, alloc_capacity);

#ifdef CTP_TEST_CDP2
  // Launch CDP child kernels (fire-and-forget)
  for (int i = 0; i < num_children; ++i) {
    RrChildKernel<<<1, 32, 0, cudaStreamFireAndForget>>>(
        alloc, alloc_size, results, i);
  }

  // Join: the tail launch is deferred until every child above has completed.
  RrDoneKernel<<<1, 1, 0, cudaStreamTailLaunch>>>(num_children, done);
#else
  // Launch CDP child kernels (fire-and-forget)
  for (int i = 0; i < num_children; ++i) {
    RrChildKernel<<<1, 32>>>(alloc, alloc_size, results, i);
  }

  // Wait for all children to complete
  cudaDeviceSynchronize();

  printf("[PARENT] All %d children completed\n", num_children);
  *done = 1;
  __threadfence_system();
#endif
}

// ============================================================================
// Host test
// ============================================================================

TEST_CASE("RoundRobinAllocator - CDP concurrent alloc/free", "[gpu][allocator]") {
  constexpr int kNumPartitions = 16;
  constexpr int kNumChildren = 8;
  constexpr size_t kAllocSize = 256;  // bytes per child allocation
  constexpr size_t kAllocatorCapacity = 4 * 1024 * 1024;  // 4 MB

  // Set CDP limits
  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 256);
  cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  // CDP2 does not use the pending-launch-count limit and rejects it with
  // cudaErrorUnsupportedLimit. That is harmless here, but it would otherwise be
  // picked up by the cudaGetLastError() check on the kernel launch below.
  cudaGetLastError();

  // Allocate device memory for the allocator
  char *d_alloc = ctp::GpuApi::Malloc<char>(kAllocatorCapacity);
  REQUIRE(d_alloc != nullptr);

  // Allocate pinned host memory for results + done flag
  int *h_results;
  int *h_done;
  cudaMallocHost(&h_results, kNumChildren * sizeof(int));
  cudaMallocHost(&h_done, sizeof(int));
  memset(h_results, 0, kNumChildren * sizeof(int));
  *h_done = 0;

  // Launch parent kernel
  void *stream = ctp::GpuApi::CreateStream();
  RrParentKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      d_alloc, kAllocatorCapacity, kNumPartitions, kNumChildren,
      kAllocSize, h_results, h_done);

  cudaError_t launch_err = cudaGetLastError();
  REQUIRE(launch_err == cudaSuccess);

  // Wait for the parent to complete. A grid is not complete until all of its
  // descendants are, so this covers the children and the tail launch too.
  ctp::GpuApi::Synchronize(stream);
  ctp::GpuApi::DestroyStream(stream);

  REQUIRE(*h_done == 1);

  // Check all children passed
  int pass_count = 0;
  int fail_count = 0;
  for (int i = 0; i < kNumChildren; ++i) {
    if (h_results[i] == 1) {
      ++pass_count;
    } else {
      ++fail_count;
      WARN("Child " << i << " failed with code " << h_results[i]);
    }
  }
  INFO("Passed: " << pass_count << " / " << kNumChildren);
  REQUIRE(fail_count == 0);

  // Cleanup
  ctp::GpuApi::Free(d_alloc);
  cudaFreeHost(h_results);
  cudaFreeHost(h_done);
}

TEST_CASE("RoundRobinAllocator - CDP many children stress", "[gpu][allocator][stress]") {
  constexpr int kNumPartitions = 32;
  constexpr int kNumChildren = 64;
  constexpr size_t kAllocSize = 512;
  constexpr size_t kAllocatorCapacity = 16 * 1024 * 1024;  // 16 MB

  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 256);
  cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  // CDP2 does not use the pending-launch-count limit and rejects it with
  // cudaErrorUnsupportedLimit. That is harmless here, but it would otherwise be
  // picked up by the cudaGetLastError() check on the kernel launch below.
  cudaGetLastError();

  char *d_alloc = ctp::GpuApi::Malloc<char>(kAllocatorCapacity);
  REQUIRE(d_alloc != nullptr);

  int *h_results;
  int *h_done;
  cudaMallocHost(&h_results, kNumChildren * sizeof(int));
  cudaMallocHost(&h_done, sizeof(int));
  memset(h_results, 0, kNumChildren * sizeof(int));
  *h_done = 0;

  void *stream = ctp::GpuApi::CreateStream();
  RrParentKernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      d_alloc, kAllocatorCapacity, kNumPartitions, kNumChildren,
      kAllocSize, h_results, h_done);

  REQUIRE(cudaGetLastError() == cudaSuccess);
  ctp::GpuApi::Synchronize(stream);
  ctp::GpuApi::DestroyStream(stream);

  REQUIRE(*h_done == 1);

  int pass_count = 0;
  for (int i = 0; i < kNumChildren; ++i) {
    if (h_results[i] == 1) ++pass_count;
    else WARN("Child " << i << " failed: " << h_results[i]);
  }
  INFO("Passed: " << pass_count << " / " << kNumChildren);
  REQUIRE(pass_count == kNumChildren);

  ctp::GpuApi::Free(d_alloc);
  cudaFreeHost(h_results);
  cudaFreeHost(h_done);
}
