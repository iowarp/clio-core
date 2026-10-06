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

#include <type_traits>
#include <catch2/catch_all.hpp>

#include "clio_ctp/data_structures/ipc/ring_buffer.h"
#include "clio_ctp/memory/allocator/arena_allocator.h"
#include "clio_ctp/memory/allocator/buddy_allocator.h"
#include "clio_ctp/memory/backend/gpu_malloc.h"
#include "clio_ctp/util/gpu_api.h"

using ctp::ipc::ArenaAllocator;
using ctp::ipc::GpuMalloc;
using ctp::ipc::MemoryBackend;
using ctp::ipc::MemoryBackendId;
using ctp::ipc::mpsc_ring_buffer;

/**
 * GPU kernel to create and initialize an allocator
 *
 * @tparam AllocT The allocator type to create
 * @param backend Pointer to the memory backend (must be in GPU-accessible
 * memory)
 * @param result Output pointer to store the created allocator pointer
 */
template <typename AllocT>
__global__ void MakeAllocKernel(MemoryBackend *backend, AllocT **result) {
  // Use MakeAlloc to create and initialize the allocator
  *result = backend->MakeAlloc<AllocT>();
  printf("MakeAllocKernel: Allocator created: %p\n", *result);
}

/**
 * GPU kernel to allocate and construct a ring buffer from the allocator
 *
 * @tparam AllocT The allocator type
 * @tparam T The ring buffer element type
 * @param alloc Pointer to the allocator
 * @param depth Depth (capacity + 1) of the ring buffer
 * @param result Output pointer to store the allocated ring buffer pointer
 */
template <typename AllocT, typename T>
__global__ void AllocateRingBufferKernel(AllocT *alloc, size_t capacity,
                                         mpsc_ring_buffer<T, AllocT> **result) {
  // Use NewObj to allocate and construct the ring buffer
  auto ring_ptr =
      alloc->template NewObj<mpsc_ring_buffer<T, AllocT>>(alloc, capacity);
  printf("AllocateRingBufferKernel: Ring buffer created: %p %p\n", alloc,
         ring_ptr.ptr_);
  // Return the ring buffer pointer
  *result = ring_ptr.ptr_;
}

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
 * GPU kernel to pop elements from ring buffer and verify values
 *
 * @tparam T The element type
 * @tparam AllocT The allocator type
 * @param ring Pointer to the ring buffer
 * @param output Array to store popped values
 * @param count Number of elements to pop
 * @param success Output flag - set to 1 if all pops succeeded, 0 otherwise
 */
template <typename T, typename AllocT>
__global__ void PopElementsKernel(mpsc_ring_buffer<T, AllocT> *ring, T *output,
                                  size_t count, int *success) {
  *success = 1;  // Assume success
  for (size_t i = 0; i < count; ++i) {
    T value;
    bool popped = ring->Pop(value);
    if (!popped) {
      *success = 0;  // Pop failed
      return;
    }
    output[i] = value;
  }
}

/**
 * Test GpuMalloc backend with ring buffer
 *
 * Steps:
 * 1. Create a GpuMalloc backend
 * 2. Create an allocator on that backend
 * 3. Allocate a ring_buffer on that backend
 * 4. Pass the ring_buffer to the kernel
 * 5. Verify that we can place 10 elements on the ring buffer
 * 6. Verify the runtime can pop the 10 elements
 */
TEST_CASE("GpuMalloc", "[gpu][backend]") {
  constexpr size_t kDataSize = 64 * 1024 * 1024;  // 64MB
  constexpr size_t kNumElements = 10;
  constexpr int kGpuId = 0;
  const std::string kUrl = "/test_gpu_malloc";

  SECTION("RingBufferGpuAccess") {
    // Step 1: Create a GpuMalloc backend
    GpuMalloc backend;
    MemoryBackendId backend_id(0, 1);
    bool init_success = backend.shm_init(backend_id, kDataSize, kUrl, kGpuId);
    REQUIRE(init_success);

    // Step 2: Create an allocator on that backend (using GPU kernel)
    using AllocT = ctp::ipc::BuddyAllocator;
    AllocT **alloc_result_dev;
    alloc_result_dev = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(alloc_result_dev)>>(sizeof(AllocT *));

    MemoryBackend *backend_dev;
    backend_dev = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(backend_dev)>>(sizeof(GpuMalloc));
    ctp::GpuApi::Memcpy(backend_dev, &backend, sizeof(GpuMalloc));

    MakeAllocKernel<AllocT><<<1, 1>>>(backend_dev, alloc_result_dev);
    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    AllocT *alloc_ptr;
    ctp::GpuApi::Memcpy(&alloc_ptr, alloc_result_dev, sizeof(AllocT *));
    REQUIRE(alloc_ptr != nullptr);

    // Step 3: Allocate a ring_buffer on that backend (using GPU kernel)
    using RingBuffer = mpsc_ring_buffer<int, AllocT>;
    RingBuffer **ring_result_dev;
    ring_result_dev = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(ring_result_dev)>>(sizeof(RingBuffer *));

    AllocateRingBufferKernel<AllocT, int>
        <<<1, 1>>>(alloc_ptr, kNumElements, ring_result_dev);
    ctp::GpuApi::Synchronize();
    REQUIRE(ctp::GpuApi::LastError() == nullptr);

    RingBuffer *ring_ptr;
    ctp::GpuApi::Memcpy(&ring_ptr, ring_result_dev, sizeof(RingBuffer *));
    REQUIRE(ring_ptr != nullptr);

    // Step 4 & 5: Pass the ring_buffer to the kernel and push 10 elements
    // Prepare values to push on the host
    int host_values[kNumElements];
    for (size_t i = 0; i < kNumElements; ++i) {
      host_values[i] = static_cast<int>(i);
    }

    // Copy values to GPU
    int *dev_values;
    dev_values = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(dev_values)>>(kNumElements * sizeof(int));
    ctp::GpuApi::Memcpy(dev_values, host_values, kNumElements * sizeof(int));

    // Launch kernel to push elements
    PushElementsKernel<int, AllocT>
        <<<1, 1>>>(ring_ptr, dev_values, kNumElements);
    ctp::GpuApi::Synchronize();

    // Step 6: Verify the runtime can pop the 10 elements
    // Allocate device memory for output values and success flag
    int *dev_output;
    dev_output = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(dev_output)>>(kNumElements * sizeof(int));

    int *dev_success;
    dev_success = ctp::GpuApi::Malloc<std::remove_pointer_t<decltype(dev_success)>>(sizeof(int));

    // Launch kernel to pop elements from ring buffer
    PopElementsKernel<int, AllocT>
        <<<1, 1>>>(ring_ptr, dev_output, kNumElements, dev_success);
    ctp::GpuApi::Synchronize();

    // Copy results back to host
    int host_output[kNumElements];
    int success_flag;
    ctp::GpuApi::Memcpy(host_output, dev_output, kNumElements * sizeof(int));
    ctp::GpuApi::Memcpy(&success_flag, dev_success, sizeof(int));

    // Verify all pops succeeded
    REQUIRE(success_flag == 1);

    // Verify the popped values match what we pushed
    for (size_t i = 0; i < kNumElements; ++i) {
      REQUIRE(host_output[i] == host_values[i]);
    }

    // Cleanup GPU temporary allocations
    ctp::GpuApi::Free(dev_success);
    ctp::GpuApi::Free(dev_output);
    ctp::GpuApi::Free(dev_values);
    ctp::GpuApi::Free(ring_result_dev);
    ctp::GpuApi::Free(alloc_result_dev);
    ctp::GpuApi::Free(backend_dev);

    // Backend cleanup handled automatically by destructor
  }
}
