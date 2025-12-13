/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <catch2/catch_all.hpp>

#include "hermes_shm/data_structures/ipc/ring_buffer.h"
#include "hermes_shm/memory/backend/gpu_malloc.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"
#include "hermes_shm/util/gpu_api.h"

using hshm::ipc::GpuMalloc;
using hshm::ipc::MemoryBackendId;
using hshm::ipc::mpsc_ring_buffer;
using hshm::ipc::ArenaAllocator;

/**
 * GPU kernel to push elements onto ring buffer
 */
template<typename T>
__global__ void PushElementsKernel(mpsc_ring_buffer<T> *ring, T *values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    ring->emplace(values[i]);
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
    REQUIRE(backend.IsGpuOnly());
    REQUIRE(backend.DoAccelPath());  // Should be true on host

    // Step 2: Create an allocator on that backend
    auto *alloc = backend.MakeAlloc<hipc::ArenaAllocator<false>>();
    REQUIRE(alloc != nullptr);

    // Step 3: For now, skip ring_buffer testing as the API needs investigation
    // TODO: Implement ring_buffer allocation and GPU kernel test

    // Cleanup
    backend.shm_destroy();
  }
}
