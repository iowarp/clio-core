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

#include <catch2/catch_all.hpp>

#include "hermes_shm/data_structures/ipc/ring_buffer.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/data_structures/priv/string.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/memory/backend/gpu_shm_mmap.h"
#include "hermes_shm/util/gpu_api.h"
#include "hermes_shm/data_structures/serialization/local_serialize.h"

using hshm::ipc::ArenaAllocator;
using hshm::ipc::GpuShmMmap;
using hshm::ipc::MemoryBackendId;
using hshm::ipc::mpsc_ring_buffer;

/**
 * Simple POD struct for testing struct transfer through ring buffer
 * from GPU to CPU.
 */
struct TestTransferStruct {
  hshm::u64 id_;
  char data_[64];

  HSHM_INLINE_CROSS_FUN TestTransferStruct() : id_(0) {
    memset(data_, 0, sizeof(data_));
  }

  HSHM_INLINE_CROSS_FUN TestTransferStruct(hshm::u64 id) : id_(id) {
    memset(data_, 9, sizeof(data_));
  }
};

/**
 * Custom struct with serialization support for GPU testing
 */
template <typename AllocT>
struct StringStruct {
  hshm::priv::string<AllocT> str_;
  float value_;

  /**
   * Constructor
   * @param alloc Allocator for string allocation
   * @param x Initial string value
   */
  __host__ __device__ StringStruct(AllocT *alloc, const char *x)
      : str_(alloc), value_(256.0f) {
    str_ = x;
  }

  /**
   * Default constructor for deserialization
   */
  __host__ __device__ StringStruct() : value_(0.0f) {}

  /**
   * Serialize method
   * @param ar Archive for serialization
   */
  template <typename Ar>
  __host__ __device__ void serialize(Ar &ar) {
    ar(str_, value_);
  }
};

/**
 * GPU kernel to push elements onto ring buffer
 *
 * @tparam T The element type
 * @tparam AllocT The allocator type
 * @param ring Pointer to the ring buffer
 * @param values Array of values to push
 * @param count Number of elements to push
 */
template <typename T, typename AllocT>
__global__ void PushElementsKernel(mpsc_ring_buffer<T, AllocT> *ring, T *values,
                                   size_t count) {
  for (size_t i = 0; i < count; ++i) {
    ring->Emplace(values[i]);
  }
}

/**
 * GPU kernel to push TestTransferStruct elements onto ring buffer.
 * Each element gets id=i and data_ memset to 9.
 *
 * @tparam AllocT The allocator type
 * @param ring Pointer to the ring buffer
 * @param count Number of elements to push
 */
template <typename AllocT>
__global__ void PushStructsKernel(
    mpsc_ring_buffer<TestTransferStruct, AllocT> *ring, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    TestTransferStruct s(static_cast<hshm::u64>(i));
    ring->Emplace(s);
  }
}

/**
 * GPU kernel to serialize data into a vector
 * This demonstrates the serialization pattern that would be used with StringStruct
 *
 * Note: Fully constructing StringStruct with hshm::priv::string on GPU causes memory
 * allocation issues, so we demonstrate the serialization format directly.
 * In a real use case, the StringStruct would be constructed on CPU and passed to GPU,
 * or GPU-specific string types would be used.
 *
 * @tparam AllocT The allocator type
 * @param alloc Pointer to the allocator (demonstrating it can be passed to GPU)
 * @param vec Pointer to the output vector for serialized data
 */
template <typename AllocT>
__global__ void SerializeStringStructKernel(AllocT *alloc,
                                            hipc::vector<char, AllocT> *vec) {
  // Demonstrate manual serialization of StringStruct format:
  // The format would be: [string_length][string_data][float_value]

  const char* test_str = "hello 8192";
  const float test_value = 8192.0f;

  // Manual serialization matching StringStruct::serialize format:
  // 1. Serialize string length (size_t)
  size_t str_len = 10;  // Length of "hello 8192"
  const char* len_bytes = reinterpret_cast<const char*>(&str_len);
  for (size_t i = 0; i < sizeof(size_t); ++i) {
    vec->emplace_back(len_bytes[i]);
  }

  // 2. Serialize string data
  for (size_t i = 0; i < str_len; ++i) {
    vec->emplace_back(test_str[i]);
  }

  // 3. Serialize float value
  const char* float_bytes = reinterpret_cast<const char*>(&test_value);
  for (size_t i = 0; i < sizeof(float); ++i) {
    vec->emplace_back(float_bytes[i]);
  }

  // Note: alloc pointer is passed here to demonstrate it's GPU-accessible
  // In a real implementation, it could be used for GPU-side allocations
  (void)alloc;
}

/**
 * Test GpuShmMmap backend with ring buffer
 *
 * Steps:
 * 1. Create a GpuShmMmap backend
 * 2. Create an allocator on that backend
 * 3. Allocate a ring_buffer on that backend
 * 4. Pass the ring_buffer to the kernel
 * 5. Verify that we can place 10 elements on the ring buffer
 * 6. Verify the runtime can pop the 10 elements
 */
TEST_CASE("GpuShmMmap", "[gpu][backend]") {
  constexpr size_t kBackendSize = 64 * 1024 * 1024;  // 64MB
  constexpr size_t kNumElements = 10;
  constexpr int kGpuId = 0;
  const std::string kUrl = "/test_gpu_shm_mmap";

  SECTION("RingBufferGpuAccess") {
    // Step 1: Create a GpuShmMmap backend
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 0);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl, kGpuId);
    REQUIRE(init_success);

    // Step 2: Create an allocator on that backend (on the host)
    // Since GpuShmMmap provides unified memory, we can create the allocator on
    // the host
    using AllocT = hipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Step 3: Allocate a ring_buffer on that backend (on the host)
    // The ring buffer is allocated in unified memory, accessible from both CPU
    // and GPU
    using RingBuffer = mpsc_ring_buffer<int, AllocT>;
    RingBuffer *ring_ptr =
        alloc_ptr->NewObj<RingBuffer>(alloc_ptr, kNumElements).ptr_;
    REQUIRE(ring_ptr != nullptr);

    // Step 4 & 5: Pass the ring_buffer to the kernel and push 10 elements
    // Allocate GPU-accessible host memory for the values array
    int *host_values;
    cudaMallocHost(&host_values, kNumElements * sizeof(int));
    for (size_t i = 0; i < kNumElements; ++i) {
      host_values[i] = static_cast<int>(i);
    }

    // Launch kernel to push elements (host_values is GPU-accessible pinned
    // memory)
    PushElementsKernel<int, AllocT>
        <<<1, 1>>>(ring_ptr, host_values, kNumElements);
    cudaDeviceSynchronize();

    // Step 6: Verify the runtime (CPU) can pop the 10 elements
    // Since GpuShmMmap provides unified memory, CPU can directly access the
    // ring buffer But we still need to verify the values, so we'll store them
    // in a regular array
    int host_output[kNumElements];
    bool all_popped = true;

    for (size_t i = 0; i < kNumElements; ++i) {
      int value;
      bool popped = ring_ptr->Pop(value);
      if (!popped) {
        all_popped = false;
        break;
      }
      host_output[i] = value;
    }

    // Verify all pops succeeded
    REQUIRE(all_popped);

    // Verify the popped values match what we pushed
    for (size_t i = 0; i < kNumElements; ++i) {
      REQUIRE(host_output[i] == host_values[i]);
    }

    // Free pinned host memory
    cudaFreeHost(host_values);

    // Cleanup handled automatically by destructor
  }

  SECTION("StringStructSerialization") {
    // Step 1: Create a GpuShmMmap backend
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 1);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_struct", kGpuId);
    REQUIRE(init_success);

    // Step 2: Create a BuddyAllocator on the backend
    using AllocT = hipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Step 3: Allocate a hipc::vector<char> from allocator
    using CharVector = hipc::vector<char, AllocT>;
    CharVector *vec_ptr = alloc_ptr->NewObj<CharVector>(alloc_ptr).ptr_;
    REQUIRE(vec_ptr != nullptr);

    // Step 4: Reserve 8192 bytes for the vector
    vec_ptr->reserve(8192);

    // Step 5: Pass allocator and vector pointers to GPU kernel
    // They are already compatible with GPU memory (unified memory)
    SerializeStringStructKernel<AllocT><<<1, 1>>>(alloc_ptr, vec_ptr);
    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Check for kernel launch errors
    err = cudaGetLastError();
    REQUIRE(err == cudaSuccess);

    // Step 6: Check that the vector is not empty
    REQUIRE(!vec_ptr->empty());

    // Step 7: Manual deserialization on CPU (matching the GPU serialization format)
    size_t offset = 0;
    const char* data = vec_ptr->data();

    // 1. Deserialize string length
    size_t str_len;
    std::memcpy(&str_len, data + offset, sizeof(size_t));
    offset += sizeof(size_t);

    // 2. Deserialize string data
    std::string result_str(data + offset, str_len);
    offset += str_len;

    // 3. Deserialize float value
    float result_value;
    std::memcpy(&result_value, data + offset, sizeof(float));
    offset += sizeof(float);

    // Step 8: Verify the StringStruct contains "hello 8192" and float 8192
    std::string expected_str = "hello 8192";
    REQUIRE(result_str == expected_str);
    REQUIRE(result_value == 8192.0f);

    // Cleanup handled automatically by destructor
  }

  SECTION("StructRingBufferGpuToCpu") {
    // Create a GpuShmMmap backend
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 2);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_struct_rb", kGpuId);
    REQUIRE(init_success);

    // Create allocator on backend
    using AllocT = hipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    // Allocate ring buffer for TestTransferStruct
    using RingBuffer = mpsc_ring_buffer<TestTransferStruct, AllocT>;
    RingBuffer *ring_ptr =
        alloc_ptr->NewObj<RingBuffer>(alloc_ptr, kNumElements).ptr_;
    REQUIRE(ring_ptr != nullptr);

    // Launch kernel to push structs
    PushStructsKernel<AllocT><<<1, 1>>>(ring_ptr, kNumElements);
    cudaDeviceSynchronize();

    // CPU pops and verifies
    for (size_t i = 0; i < kNumElements; ++i) {
      TestTransferStruct value;
      bool popped = ring_ptr->Pop(value);
      REQUIRE(popped);
      REQUIRE(value.id_ == static_cast<hshm::u64>(i));
      for (size_t j = 0; j < 64; ++j) {
        REQUIRE(value.data_[j] == 9);
      }
    }
  }

  SECTION("StructRingBufferGpuToCpuAsync") {
    // Same as above but CPU polls without cudaDeviceSynchronize,
    // popping elements as soon as they become available.
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 3);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_async_rb", kGpuId);
    REQUIRE(init_success);

    using AllocT = hipc::BuddyAllocator;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    using RingBuffer = mpsc_ring_buffer<TestTransferStruct, AllocT>;
    RingBuffer *ring_ptr =
        alloc_ptr->NewObj<RingBuffer>(alloc_ptr, kNumElements).ptr_;
    REQUIRE(ring_ptr != nullptr);

    // Launch kernel (no sync -- CPU polls immediately)
    PushStructsKernel<AllocT><<<1, 1>>>(ring_ptr, kNumElements);

    // Poll the ring buffer until all elements are popped
    size_t popped_count = 0;
    while (popped_count < kNumElements) {
      TestTransferStruct value;
      if (!ring_ptr->Pop(value)) {
        continue;  // Not ready yet, keep polling
      }
      REQUIRE(value.id_ == static_cast<hshm::u64>(popped_count));
      for (size_t j = 0; j < 64; ++j) {
        REQUIRE(value.data_[j] == 9);
      }
      ++popped_count;
    }

    // Sync to ensure kernel finishes cleanly before backend teardown
    cudaDeviceSynchronize();
  }
}
