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

#include <catch2/catch_test_macros.hpp>
#include "allocator_test.h"
#include "clio_ctp/memory/backend/posix_mmap.h"
#include "clio_ctp/memory/allocator/mp_allocator.h"

using ctp::testing::AllocatorTest;

TEST_CASE("ProducerConsumerAllocator - Allocate and Free Immediate", "[ProducerConsumerAllocator]") {
  ctp::ipc::PosixMmap backend;
  size_t heap_size = 512 * 1024 * 1024;  // 512 MB heap
  size_t alloc_size = sizeof(ctp::ipc::ProducerConsumerAllocator);
  backend.shm_init(ctp::ipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<ctp::ipc::ProducerConsumerAllocator>();

  AllocatorTest<ctp::ipc::ProducerConsumerAllocator> tester(alloc);

  SECTION("Small allocations (1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(10000, 1024));
  }

  SECTION("Medium allocations (64KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(1000, 64 * 1024));
  }

  SECTION("Large allocations (1MB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(100, 1024 * 1024));
  }

  alloc->shm_detach();
}

TEST_CASE("ProducerConsumerAllocator - Batch Allocate and Free", "[ProducerConsumerAllocator]") {
  ctp::ipc::PosixMmap backend;
  size_t heap_size = 512 * 1024 * 1024;  // 512 MB heap
  size_t alloc_size = sizeof(ctp::ipc::ProducerConsumerAllocator);
  backend.shm_init(ctp::ipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<ctp::ipc::ProducerConsumerAllocator>();

  AllocatorTest<ctp::ipc::ProducerConsumerAllocator> tester(alloc);

  SECTION("Small batches (10 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(1000, 10, 4096));
  }

  SECTION("Medium batches (100 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(100, 100, 4096));
  }

  SECTION("Large batches (1000 allocations of 1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(10, 1000, 1024));
  }

  alloc->shm_detach();
}

TEST_CASE("ProducerConsumerAllocator - Random Allocation", "[ProducerConsumerAllocator]") {
  ctp::ipc::PosixMmap backend;
  size_t heap_size = 512 * 1024 * 1024;  // 512 MB heap
  size_t alloc_size = sizeof(ctp::ipc::ProducerConsumerAllocator);
  backend.shm_init(ctp::ipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<ctp::ipc::ProducerConsumerAllocator>();

  AllocatorTest<ctp::ipc::ProducerConsumerAllocator> tester(alloc);

  SECTION("256 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(256));
  }

  SECTION("32 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(32));
  }

  alloc->shm_detach();
}

TEST_CASE("ProducerConsumerAllocator - Multi-threaded Random", "[ProducerConsumerAllocator][multithread]") {
  ctp::ipc::PosixMmap backend;
  size_t heap_size = 512 * 1024 * 1024;  // 512 MB heap
  size_t alloc_size = sizeof(ctp::ipc::ProducerConsumerAllocator);
  backend.shm_init(ctp::ipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  auto *alloc = backend.MakeAlloc<ctp::ipc::ProducerConsumerAllocator>();

  AllocatorTest<ctp::ipc::ProducerConsumerAllocator> tester(alloc);

  SECTION("8 threads, 16 iterations each") {
    REQUIRE_NOTHROW(tester.TestMultiThreadedRandom(8, 16));
  }

  SECTION("4 threads, 256 iterations each") {
    REQUIRE_NOTHROW(tester.TestMultiThreadedRandom(4, 256));
  }

  alloc->shm_detach();
}
