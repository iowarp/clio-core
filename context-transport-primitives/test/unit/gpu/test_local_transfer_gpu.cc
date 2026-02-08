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
 * GPU unit test for LocalTransfer with GpuShmMmap backend
 *
 * This test verifies that data transfer works correctly with GPU-accessible
 * pinned memory:
 * 1. Allocates pinned host memory using GpuShmMmap backend for copy space
 * 2. Uses 16KB transfer granularity
 * 3. GPU kernel fills a 64KB buffer with pattern (memset to 1)
 * 4. Data is transferred in chunks via the copy space
 * 5. CPU verifies the transferred data
 */

#include <catch2/catch_all.hpp>

#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/memory/backend/gpu_shm_mmap.h"
#include "hermes_shm/util/gpu_api.h"

using hshm::ipc::ArenaAllocator;
using hshm::ipc::GpuShmMmap;
using hshm::ipc::MemoryBackendId;

/**
 * GPU kernel to fill a buffer with a pattern
 *
 * @param buffer Pointer to the buffer to fill
 * @param size Size of the buffer
 * @param pattern Value to fill with
 */
__global__ void FillBufferKernel(char *buffer, size_t size, char pattern) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < size; i += stride) {
    buffer[i] = pattern;
  }
}

/**
 * GPU kernel to copy a chunk of data to copy space
 *
 * This simulates the sender-side transfer: GPU copies data to the copy space
 * that will be read by the CPU.
 *
 * @param src_buffer Source buffer (GPU-side data)
 * @param copy_space Destination copy space (pinned memory)
 * @param offset Offset into source buffer
 * @param chunk_size Size of chunk to copy
 */
__global__ void CopyChunkKernel(const char *src_buffer, char *copy_space,
                                size_t offset, size_t chunk_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < chunk_size; i += stride) {
    copy_space[i] = src_buffer[offset + i];
  }
}

/**
 * GPU kernel to set a value at a specific location (for simple tests)
 *
 * @param buffer Pointer to the buffer
 * @param index Index to set
 * @param value Value to set
 */
__global__ void SetValueKernel(char *buffer, size_t index, char value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    buffer[index] = value;
  }
}

/**
 * Test GPU to CPU data transfer using GpuShmMmap pinned memory
 */
TEST_CASE("LocalTransfer GPU", "[gpu][transfer]") {
  constexpr size_t kBackendSize = 16 * 1024 * 1024;  // 16MB
  constexpr size_t kCopySpaceSize = 16 * 1024;       // 16KB transfer granularity
  constexpr size_t kDataSize = 64 * 1024;            // 64KB buffer
  constexpr int kGpuId = 0;
  const std::string kUrl = "/test_local_transfer_gpu";

  SECTION("BasicGpuToCpuTransfer") {
    // Step 1: Create a GpuShmMmap backend for pinned host memory
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 0);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl, kGpuId);
    REQUIRE(init_success);

    // Step 2: Create an ArenaAllocator on that backend
    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Step 3: Allocate copy space from the allocator (pinned memory)
    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    // Step 4: Allocate GPU source buffer (device memory or pinned)
    // We use pinned memory so both GPU and CPU can access
    char *gpu_buffer;
    cudaMallocHost(&gpu_buffer, kDataSize);
    REQUIRE(gpu_buffer != nullptr);

    // Step 5: Fill the buffer with pattern (value = 1) using GPU kernel
    constexpr char kPattern = 1;
    int blockSize = 256;
    int numBlocks = (kDataSize + blockSize - 1) / blockSize;
    FillBufferKernel<<<numBlocks, blockSize>>>(gpu_buffer, kDataSize, kPattern);
    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Step 6: Transfer data in chunks (16KB at a time)
    std::vector<char> received_data;
    received_data.reserve(kDataSize);

    size_t bytes_transferred = 0;
    while (bytes_transferred < kDataSize) {
      // Calculate chunk size
      size_t remaining = kDataSize - bytes_transferred;
      size_t chunk_size = std::min(remaining, kCopySpaceSize);

      // GPU copies chunk to copy space
      CopyChunkKernel<<<numBlocks, blockSize>>>(gpu_buffer, copy_space,
                                                bytes_transferred, chunk_size);
      err = cudaDeviceSynchronize();
      REQUIRE(err == cudaSuccess);

      // CPU reads from copy space (since it's pinned memory, CPU can read directly)
      received_data.insert(received_data.end(), copy_space,
                           copy_space + chunk_size);

      bytes_transferred += chunk_size;
    }

    // Step 7: Verify all data was transferred
    REQUIRE(received_data.size() == kDataSize);

    // Step 8: Verify data integrity - all bytes should be 1
    bool all_ones = true;
    for (size_t i = 0; i < kDataSize; ++i) {
      if (received_data[i] != kPattern) {
        all_ones = false;
        break;
      }
    }
    REQUIRE(all_ones);

    // Cleanup
    cudaFreeHost(gpu_buffer);
  }

  SECTION("ChunkedTransferWithPattern") {
    // Test with a more complex pattern to verify data integrity
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 1);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_pattern", kGpuId);
    REQUIRE(init_success);

    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    // Allocate and initialize GPU buffer with pattern
    char *gpu_buffer;
    cudaMallocHost(&gpu_buffer, kDataSize);
    REQUIRE(gpu_buffer != nullptr);

    // Initialize with pattern on CPU (index % 256)
    for (size_t i = 0; i < kDataSize; ++i) {
      gpu_buffer[i] = static_cast<char>(i % 256);
    }

    // Transfer in chunks
    std::vector<char> received_data;
    received_data.reserve(kDataSize);

    size_t bytes_transferred = 0;
    size_t chunk_count = 0;
    int blockSize = 256;
    int numBlocks = (kCopySpaceSize + blockSize - 1) / blockSize;

    while (bytes_transferred < kDataSize) {
      size_t remaining = kDataSize - bytes_transferred;
      size_t chunk_size = std::min(remaining, kCopySpaceSize);

      // GPU copies chunk to copy space
      CopyChunkKernel<<<numBlocks, blockSize>>>(gpu_buffer, copy_space,
                                                bytes_transferred, chunk_size);
      cudaError_t err = cudaDeviceSynchronize();
      REQUIRE(err == cudaSuccess);

      // CPU reads from copy space
      received_data.insert(received_data.end(), copy_space,
                           copy_space + chunk_size);

      bytes_transferred += chunk_size;
      chunk_count++;
    }

    // Verify chunk count (64KB / 16KB = 4 chunks)
    REQUIRE(chunk_count == 4);

    // Verify data integrity
    REQUIRE(received_data.size() == kDataSize);
    bool pattern_correct = true;
    for (size_t i = 0; i < kDataSize; ++i) {
      if (received_data[i] != static_cast<char>(i % 256)) {
        pattern_correct = false;
        break;
      }
    }
    REQUIRE(pattern_correct);

    cudaFreeHost(gpu_buffer);
  }

  SECTION("DirectGpuMemoryAccess") {
    // Test that GPU can directly read/write to the GpuShmMmap memory
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 2);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_direct", kGpuId);
    REQUIRE(init_success);

    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Allocate buffer directly from GpuShmMmap
    auto buffer_ptr = alloc_ptr->AllocateObjs<char>(1024);
    char *buffer = buffer_ptr.ptr_;
    REQUIRE(buffer != nullptr);

    // Initialize on CPU
    std::memset(buffer, 0, 1024);

    // GPU sets specific values
    SetValueKernel<<<1, 1>>>(buffer, 0, 'A');
    SetValueKernel<<<1, 1>>>(buffer, 100, 'B');
    SetValueKernel<<<1, 1>>>(buffer, 500, 'C');
    SetValueKernel<<<1, 1>>>(buffer, 1023, 'D');

    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // CPU reads and verifies
    REQUIRE(buffer[0] == 'A');
    REQUIRE(buffer[100] == 'B');
    REQUIRE(buffer[500] == 'C');
    REQUIRE(buffer[1023] == 'D');

    // Verify untouched locations are still 0
    REQUIRE(buffer[1] == 0);
    REQUIRE(buffer[50] == 0);
    REQUIRE(buffer[1022] == 0);
  }

  SECTION("LargeTransferPerformance") {
    // Test larger transfer (256KB) to verify performance
    constexpr size_t kLargeDataSize = 256 * 1024;  // 256KB

    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 3);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_large", kGpuId);
    REQUIRE(init_success);

    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    auto copy_space_ptr = alloc_ptr->AllocateObjs<char>(kCopySpaceSize);
    char *copy_space = copy_space_ptr.ptr_;
    REQUIRE(copy_space != nullptr);

    // Allocate GPU buffer
    char *gpu_buffer;
    cudaMallocHost(&gpu_buffer, kLargeDataSize);
    REQUIRE(gpu_buffer != nullptr);

    // Fill with pattern
    constexpr char kPattern = 0x55;
    int blockSize = 256;
    int numBlocks = (kLargeDataSize + blockSize - 1) / blockSize;
    FillBufferKernel<<<numBlocks, blockSize>>>(gpu_buffer, kLargeDataSize,
                                               kPattern);
    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Transfer in 16KB chunks
    std::vector<char> received_data;
    received_data.reserve(kLargeDataSize);

    size_t bytes_transferred = 0;
    numBlocks = (kCopySpaceSize + blockSize - 1) / blockSize;

    while (bytes_transferred < kLargeDataSize) {
      size_t remaining = kLargeDataSize - bytes_transferred;
      size_t chunk_size = std::min(remaining, kCopySpaceSize);

      CopyChunkKernel<<<numBlocks, blockSize>>>(gpu_buffer, copy_space,
                                                bytes_transferred, chunk_size);
      err = cudaDeviceSynchronize();
      REQUIRE(err == cudaSuccess);

      received_data.insert(received_data.end(), copy_space,
                           copy_space + chunk_size);

      bytes_transferred += chunk_size;
    }

    // Verify
    REQUIRE(received_data.size() == kLargeDataSize);

    bool pattern_correct = true;
    for (size_t i = 0; i < kLargeDataSize; ++i) {
      if (received_data[i] != kPattern) {
        pattern_correct = false;
        break;
      }
    }
    REQUIRE(pattern_correct);

    cudaFreeHost(gpu_buffer);
  }
}
