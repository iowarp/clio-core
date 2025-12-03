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

#include <catch2/catch_test_macros.hpp>
#include "allocator_test.h"
#include "hermes_shm/memory/backend/malloc_backend.h"
#include "hermes_shm/memory/allocator/arena_allocator.h"

using hshm::testing::AllocatorTest;

/**
 * Helper function to create a MallocBackend and ArenaAllocator<false>
 * Returns the allocator pointer (caller must manage backend lifetime)
 */
hipc::BaseAllocator<hipc::_ArenaAllocator<false>>* CreateArenaAllocator(hipc::MallocBackend &backend) {
  // Initialize backend with space for allocator + heap
  size_t heap_size = 128 * 1024 * 1024;  // 128 MB heap
  size_t alloc_size = sizeof(hipc::BaseAllocator<hipc::_ArenaAllocator<false>>);
  backend.shm_init(hipc::MemoryBackendId(0, 0), alloc_size + heap_size);

  // Construct allocator at beginning of backend
  auto *alloc = backend.Cast<hipc::BaseAllocator<hipc::_ArenaAllocator<false>>>();
  new (alloc) hipc::BaseAllocator<hipc::_ArenaAllocator<false>>();

  // Create heap backend view (starts after allocator object)
  hipc::MemoryBackend heap_backend = backend;
  heap_backend.data_offset_ = alloc_size;
  heap_backend.data_size_ = heap_size;

  // Initialize allocator with heap backend
  alloc->shm_init(heap_backend, 0);

  return alloc;
}

TEST_CASE("ArenaAllocator<false> - Allocate and Free Immediate", "[ArenaAllocator<false>]") {
  hipc::MallocBackend backend;
  auto *alloc = CreateArenaAllocator(backend);

  AllocatorTest<hipc::BaseAllocator<hipc::_ArenaAllocator<false>>> tester(alloc);

  SECTION("Small allocations (1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(10000, 1024));
  }

  SECTION("Medium allocations (64KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(1000, 64 * 1024));
  }

  SECTION("Large allocations (1MB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeImmediate(100, 1024 * 1024));
  }

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator<false> - Batch Allocate and Free", "[ArenaAllocator<false>]") {
  hipc::MallocBackend backend;
  auto *alloc = CreateArenaAllocator(backend);

  AllocatorTest<hipc::BaseAllocator<hipc::_ArenaAllocator<false>>> tester(alloc);

  SECTION("Small batches (10 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(1000, 10, 4096));
  }

  SECTION("Medium batches (100 allocations of 4KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(100, 100, 4096));
  }

  SECTION("Large batches (1000 allocations of 1KB)") {
    REQUIRE_NOTHROW(tester.TestAllocFreeBatch(10, 1000, 1024));
  }

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator<false> - Random Allocation", "[ArenaAllocator<false>]") {
  hipc::MallocBackend backend;
  auto *alloc = CreateArenaAllocator(backend);

  AllocatorTest<hipc::BaseAllocator<hipc::_ArenaAllocator<false>>> tester(alloc);

  SECTION("16 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(16));
  }

  SECTION("32 iterations of random allocations") {
    REQUIRE_NOTHROW(tester.TestRandomAllocation(32));
  }

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator<false> - Multi-threaded Random", "[ArenaAllocator<false>][multithread]") {
  hipc::MallocBackend backend;
  auto *alloc = CreateArenaAllocator(backend);

  AllocatorTest<hipc::BaseAllocator<hipc::_ArenaAllocator<false>>> tester(alloc);

  SECTION("8 threads, 2 iterations each") {
    REQUIRE_NOTHROW(tester.TestMultiThreadedRandom(8, 2));
  }

  SECTION("4 threads, 4 iterations each") {
    REQUIRE_NOTHROW(tester.TestMultiThreadedRandom(4, 4));
  }

  backend.shm_destroy();
}

TEST_CASE("ArenaAllocator<false> - Run All Tests", "[ArenaAllocator<false>][all]") {
  hipc::MallocBackend backend;
  auto *alloc = CreateArenaAllocator(backend);

  AllocatorTest<hipc::BaseAllocator<hipc::_ArenaAllocator<false>>> tester(alloc);

  REQUIRE_NOTHROW(tester.RunAllTests());

  backend.shm_destroy();
}
