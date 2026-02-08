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
 * GPU unit test for serialization with hshm::priv::vector
 *
 * This test verifies that serialization works correctly between GPU and CPU:
 * 1. Allocates pinned host memory using GpuShmMmap backend
 * 2. GPU kernel serializes integers and floats into hshm::priv::vector
 *    (using byte-by-byte push_back, the proven GPU-compatible pattern)
 * 3. CPU deserializes using LocalDeserialize and verifies the data
 *
 * Note: Direct LocalSerialize usage on GPU has issues with memcpy, so we use
 * manual byte-by-byte serialization on GPU (matching test_gpu_shm_mmap.cc pattern)
 * and LocalDeserialize on CPU for deserialization.
 */

#include <catch2/catch_all.hpp>

#include "hermes_shm/data_structures/priv/vector.h"
#include "hermes_shm/data_structures/serialization/local_serialize.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/memory/backend/gpu_shm_mmap.h"
#include "hermes_shm/util/gpu_api.h"

using hshm::ipc::ArenaAllocator;
using hshm::ipc::GpuShmMmap;
using hshm::ipc::MemoryBackendId;

/**
 * Helper to serialize a value byte-by-byte into a vector on GPU
 *
 * Note: We use manual byte-by-byte serialization because LocalSerialize
 * uses memcpy which may not work correctly on all GPU architectures.
 * This matches the pattern used in test_gpu_shm_mmap.cc.
 *
 * @tparam T The type to serialize
 * @tparam VecT The vector type
 * @param vec Pointer to the vector
 * @param value The value to serialize
 */
template <typename T, typename VecT>
__device__ void GpuSerializeValue(VecT *vec, const T &value) {
  const char *bytes = reinterpret_cast<const char *>(&value);
  for (size_t i = 0; i < sizeof(T); ++i) {
    vec->push_back(bytes[i]);
  }
}

/**
 * GPU kernel to serialize integers and floats into a vector
 *
 * This kernel demonstrates serialization working on GPU with hshm::priv::vector
 * using byte-by-byte push_back (the pattern proven to work in test_gpu_shm_mmap.cc).
 *
 * @tparam AllocT The allocator type
 * @param alloc Pointer to the allocator
 * @param vec Pointer to the output vector for serialized data
 * @param int_vals Array of integers to serialize
 * @param float_vals Array of floats to serialize
 * @param num_ints Number of integers
 * @param num_floats Number of floats
 */
template <typename AllocT>
__global__ void SerializeKernel(AllocT *alloc,
                                hshm::priv::vector<char, AllocT> *vec,
                                int *int_vals, float *float_vals,
                                size_t num_ints, size_t num_floats) {
  // Use byte-by-byte serialization (matches test_gpu_shm_mmap.cc pattern)
  // This avoids memcpy issues on GPU

  // Serialize the count of integers
  GpuSerializeValue(vec, num_ints);

  // Serialize each integer
  for (size_t i = 0; i < num_ints; ++i) {
    GpuSerializeValue(vec, int_vals[i]);
  }

  // Serialize the count of floats
  GpuSerializeValue(vec, num_floats);

  // Serialize each float
  for (size_t i = 0; i < num_floats; ++i) {
    GpuSerializeValue(vec, float_vals[i]);
  }

  // Mark alloc as used (it's passed to demonstrate GPU accessibility)
  (void)alloc;
}

/**
 * GPU kernel to append a value to an existing vector
 *
 * @tparam AllocT The allocator type
 * @param vec Pointer to the vector
 * @param value Value to append
 */
template <typename AllocT>
__global__ void SerializeAppendKernel(hshm::priv::vector<char, AllocT> *vec,
                                      int value) {
  // Use byte-by-byte serialization to append
  GpuSerializeValue(vec, value);
}

/**
 * Test LocalSerialize with GPU kernel serialization and CPU deserialization
 */
TEST_CASE("LocalSerialize GPU", "[gpu][serialize]") {
  constexpr size_t kBackendSize = 16 * 1024 * 1024;  // 16MB
  constexpr int kGpuId = 0;
  const std::string kUrl = "/test_local_serialize_gpu";

  SECTION("BasicIntFloatSerialization") {
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

    // Step 3: Allocate a priv::vector<char> from allocator
    using CharVector = hshm::priv::vector<char, AllocT>;
    CharVector *vec_ptr = alloc_ptr->NewObj<CharVector>(alloc_ptr).ptr_;
    REQUIRE(vec_ptr != nullptr);

    // Reserve space for serialized data
    vec_ptr->reserve(4096);

    // Step 4: Prepare test data on GPU-accessible pinned memory
    constexpr size_t kNumInts = 5;
    constexpr size_t kNumFloats = 3;

    int *host_ints;
    float *host_floats;
    cudaMallocHost(&host_ints, kNumInts * sizeof(int));
    cudaMallocHost(&host_floats, kNumFloats * sizeof(float));

    // Initialize test values
    int expected_ints[kNumInts] = {10, 20, 30, 40, 50};
    float expected_floats[kNumFloats] = {1.5f, 2.5f, 3.5f};

    for (size_t i = 0; i < kNumInts; ++i) {
      host_ints[i] = expected_ints[i];
    }
    for (size_t i = 0; i < kNumFloats; ++i) {
      host_floats[i] = expected_floats[i];
    }

    // Step 5: Launch kernel to serialize data on GPU
    SerializeKernel<AllocT><<<1, 1>>>(alloc_ptr, vec_ptr, host_ints,
                                      host_floats, kNumInts, kNumFloats);
    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Check for kernel launch errors
    err = cudaGetLastError();
    REQUIRE(err == cudaSuccess);

    // Step 6: Verify the vector is not empty
    REQUIRE(!vec_ptr->empty());

    // Step 7: Deserialize on CPU
    hshm::ipc::LocalDeserialize<CharVector> deserializer(*vec_ptr);

    // Deserialize integer count
    size_t num_ints;
    deserializer >> num_ints;
    REQUIRE(num_ints == kNumInts);

    // Deserialize integers
    for (size_t i = 0; i < num_ints; ++i) {
      int val;
      deserializer >> val;
      REQUIRE(val == expected_ints[i]);
    }

    // Deserialize float count
    size_t num_floats;
    deserializer >> num_floats;
    REQUIRE(num_floats == kNumFloats);

    // Deserialize floats
    for (size_t i = 0; i < num_floats; ++i) {
      float val;
      deserializer >> val;
      REQUIRE(val == expected_floats[i]);
    }

    // Cleanup
    cudaFreeHost(host_ints);
    cudaFreeHost(host_floats);
  }

  SECTION("LargeDataSerialization") {
    // Test with larger data to verify chunked operations work
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 1);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_large", kGpuId);
    REQUIRE(init_success);

    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    using CharVector = hshm::priv::vector<char, AllocT>;
    CharVector *vec_ptr = alloc_ptr->NewObj<CharVector>(alloc_ptr).ptr_;
    REQUIRE(vec_ptr != nullptr);

    // Reserve space for larger data
    vec_ptr->reserve(64 * 1024);  // 64KB

    constexpr size_t kNumInts = 1000;
    constexpr size_t kNumFloats = 500;

    int *host_ints;
    float *host_floats;
    cudaMallocHost(&host_ints, kNumInts * sizeof(int));
    cudaMallocHost(&host_floats, kNumFloats * sizeof(float));

    // Initialize with pattern
    for (size_t i = 0; i < kNumInts; ++i) {
      host_ints[i] = static_cast<int>(i * 7);  // Pattern: 0, 7, 14, ...
    }
    for (size_t i = 0; i < kNumFloats; ++i) {
      host_floats[i] = static_cast<float>(i) * 0.5f;  // Pattern: 0.0, 0.5, 1.0, ...
    }

    // Launch kernel
    SerializeKernel<AllocT><<<1, 1>>>(alloc_ptr, vec_ptr, host_ints,
                                      host_floats, kNumInts, kNumFloats);
    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    err = cudaGetLastError();
    REQUIRE(err == cudaSuccess);

    // Verify serialized data
    REQUIRE(!vec_ptr->empty());

    // Deserialize and verify
    hshm::ipc::LocalDeserialize<CharVector> deserializer(*vec_ptr);

    size_t num_ints;
    deserializer >> num_ints;
    REQUIRE(num_ints == kNumInts);

    for (size_t i = 0; i < num_ints; ++i) {
      int val;
      deserializer >> val;
      REQUIRE(val == static_cast<int>(i * 7));
    }

    size_t num_floats;
    deserializer >> num_floats;
    REQUIRE(num_floats == kNumFloats);

    for (size_t i = 0; i < num_floats; ++i) {
      float val;
      deserializer >> val;
      REQUIRE(val == static_cast<float>(i) * 0.5f);
    }

    cudaFreeHost(host_ints);
    cudaFreeHost(host_floats);
  }

  SECTION("MixedTypeSerialization") {
    // Test with mixed types: int, float, double, size_t
    GpuShmMmap backend;
    MemoryBackendId backend_id(0, 2);
    bool init_success =
        backend.shm_init(backend_id, kBackendSize, kUrl + "_mixed", kGpuId);
    REQUIRE(init_success);

    using AllocT = hipc::ArenaAllocator<false>;
    AllocT *alloc_ptr = backend.MakeAlloc<AllocT>();
    REQUIRE(alloc_ptr != nullptr);

    using CharVector = hshm::priv::vector<char, AllocT>;
    CharVector *vec_ptr = alloc_ptr->NewObj<CharVector>(alloc_ptr).ptr_;
    REQUIRE(vec_ptr != nullptr);
    vec_ptr->reserve(4096);

    // For this test, we'll manually serialize different types
    // by writing bytes directly to the vector from GPU

    // Use the existing serialize kernel with just ints and floats
    // but verify the binary format is correct
    constexpr size_t kNumInts = 2;
    constexpr size_t kNumFloats = 2;

    int *host_ints;
    float *host_floats;
    cudaMallocHost(&host_ints, kNumInts * sizeof(int));
    cudaMallocHost(&host_floats, kNumFloats * sizeof(float));

    host_ints[0] = 12345;
    host_ints[1] = -9876;
    host_floats[0] = 3.14159f;
    host_floats[1] = 2.71828f;

    SerializeKernel<AllocT><<<1, 1>>>(alloc_ptr, vec_ptr, host_ints,
                                      host_floats, kNumInts, kNumFloats);
    cudaError_t err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    // Deserialize
    hshm::ipc::LocalDeserialize<CharVector> deserializer(*vec_ptr);

    size_t num_ints;
    deserializer >> num_ints;
    REQUIRE(num_ints == 2);

    int val1, val2;
    deserializer >> val1 >> val2;
    REQUIRE(val1 == 12345);
    REQUIRE(val2 == -9876);

    size_t num_floats;
    deserializer >> num_floats;
    REQUIRE(num_floats == 2);

    float fval1, fval2;
    deserializer >> fval1 >> fval2;
    REQUIRE(fval1 == Catch::Approx(3.14159f));
    REQUIRE(fval2 == Catch::Approx(2.71828f));

    cudaFreeHost(host_ints);
    cudaFreeHost(host_floats);
  }
}
