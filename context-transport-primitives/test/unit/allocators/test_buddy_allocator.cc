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
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#include "hermes_shm/memory/backend/malloc_backend.h"
#include <vector>
#include <algorithm>

using namespace hshm::ipc;

TEST_CASE("BuddyAllocator - Initialization", "[buddy_allocator]") {
  MallocBackend backend;
  size_t heap_size = 1024 * 1024;  // 1MB
  backend.shm_init(MemoryBackendId(0, 0), heap_size);

  BuddyAllocator allocator;
  bool success = allocator.shm_init(backend.data_, heap_size);

  REQUIRE(success);

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Small Allocations", "[buddy_allocator]") {
  MallocBackend backend;
  size_t heap_size = 1024 * 1024;  // 1MB
  backend.shm_init(MemoryBackendId(0, 0), heap_size);

  BuddyAllocator allocator;
  allocator.shm_init(backend.data_, heap_size);

  SECTION("Single small allocation") {
    auto offset = allocator.AllocateOffset(64);
    REQUIRE_FALSE(offset.IsNull());
    REQUIRE(offset.load() > 0);
  }

  SECTION("Multiple small allocations") {
    std::vector<OffsetPtr> allocations;

    // Allocate 10 small blocks
    for (int i = 0; i < 10; ++i) {
      auto offset = allocator.AllocateOffset(64);
      REQUIRE_FALSE(offset.IsNull());
      allocations.push_back(offset);
    }

    // Verify all allocations are unique
    for (size_t i = 0; i < allocations.size(); ++i) {
      for (size_t j = i + 1; j < allocations.size(); ++j) {
        REQUIRE(allocations[i].load() != allocations[j].load());
      }
    }
  }

  SECTION("Various small sizes") {
    std::vector<size_t> sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

    for (size_t size : sizes) {
      auto offset = allocator.AllocateOffset(size);
      REQUIRE_FALSE(offset.IsNull());
    }
  }

  SECTION("Minimum size allocation") {
    auto offset = allocator.AllocateOffset(1);
    REQUIRE_FALSE(offset.IsNull());
  }

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Large Allocations", "[buddy_allocator]") {
  MallocBackend backend;
  size_t heap_size = 10 * 1024 * 1024;  // 10MB
  backend.shm_init(MemoryBackendId(0, 0), heap_size);

  BuddyAllocator allocator;
  allocator.shm_init(backend.data_, heap_size);

  SECTION("Single large allocation") {
    auto offset = allocator.AllocateOffset(32 * 1024);  // 32KB
    REQUIRE_FALSE(offset.IsNull());
  }

  SECTION("Multiple large allocations") {
    std::vector<OffsetPtr> allocations;

    // Allocate 5 large blocks
    for (int i = 0; i < 5; ++i) {
      auto offset = allocator.AllocateOffset(64 * 1024);  // 64KB each
      REQUIRE_FALSE(offset.IsNull());
      allocations.push_back(offset);
    }

    // Verify all allocations are unique
    for (size_t i = 0; i < allocations.size(); ++i) {
      for (size_t j = i + 1; j < allocations.size(); ++j) {
        REQUIRE(allocations[i].load() != allocations[j].load());
      }
    }
  }

  SECTION("Various large sizes") {
    std::vector<size_t> sizes = {
      16 * 1024,   // 16KB
      32 * 1024,   // 32KB
      64 * 1024,   // 64KB
      128 * 1024,  // 128KB
      256 * 1024,  // 256KB
      512 * 1024   // 512KB
    };

    for (size_t size : sizes) {
      auto offset = allocator.AllocateOffset(size);
      REQUIRE_FALSE(offset.IsNull());
    }
  }

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Free and Reallocation", "[buddy_allocator]") {
  MallocBackend backend;
  size_t heap_size = 1024 * 1024;  // 1MB
  backend.shm_init(MemoryBackendId(0, 0), heap_size);

  BuddyAllocator allocator;
  allocator.shm_init(backend.data_, heap_size);

  SECTION("Free and reallocate small") {
    auto offset1 = allocator.AllocateOffset(128);
    REQUIRE_FALSE(offset1.IsNull());

    allocator.FreeOffset(offset1);

    auto offset2 = allocator.AllocateOffset(128);
    REQUIRE_FALSE(offset2.IsNull());
    // Should reuse the freed block
    REQUIRE(offset1.load() == offset2.load());
  }

  SECTION("Free and reallocate large") {
    auto offset1 = allocator.AllocateOffset(64 * 1024);
    REQUIRE_FALSE(offset1.IsNull());

    allocator.FreeOffset(offset1);

    auto offset2 = allocator.AllocateOffset(64 * 1024);
    REQUIRE_FALSE(offset2.IsNull());
    // Should reuse the freed block
    REQUIRE(offset1.load() == offset2.load());
  }

  SECTION("Multiple free and reallocate") {
    std::vector<OffsetPtr> allocations;

    // Allocate 10 blocks
    for (int i = 0; i < 10; ++i) {
      allocations.push_back(allocator.AllocateOffset(256));
    }

    // Free all blocks
    for (auto &offset : allocations) {
      allocator.FreeOffset(offset);
    }

    // Reallocate and verify we can allocate successfully
    for (int i = 0; i < 10; ++i) {
      auto offset = allocator.AllocateOffset(256);
      REQUIRE_FALSE(offset.IsNull());
    }
  }

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Coalescing", "[buddy_allocator]") {
  MallocBackend backend;
  size_t heap_size = 2 * 1024 * 1024;  // 2MB
  backend.shm_init(MemoryBackendId(0, 0), heap_size);

  BuddyAllocator allocator;
  allocator.shm_init(backend.data_, heap_size);

  SECTION("Coalesce adjacent small blocks") {
    // Allocate several small blocks
    std::vector<OffsetPtr> allocations;
    for (int i = 0; i < 8; ++i) {
      auto offset = allocator.AllocateOffset(1024);
      REQUIRE_FALSE(offset.IsNull());
      allocations.push_back(offset);
    }

    // Free all blocks (they should be adjacent or near each other)
    for (auto &offset : allocations) {
      allocator.FreeOffset(offset);
    }

    // Now try to allocate a large block
    // If coalescing works, this should succeed by merging the freed blocks
    auto large_offset = allocator.AllocateOffset(8 * 1024);
    REQUIRE_FALSE(large_offset.IsNull());
  }

  SECTION("Fragmentation and coalescing") {
    // Create a fragmented heap
    std::vector<OffsetPtr> keep;
    std::vector<OffsetPtr> free_later;

    // Allocate alternating blocks
    for (int i = 0; i < 16; ++i) {
      auto offset = allocator.AllocateOffset(512);
      if (i % 2 == 0) {
        keep.push_back(offset);
      } else {
        free_later.push_back(offset);
      }
    }

    // Free every other block
    for (auto &offset : free_later) {
      allocator.FreeOffset(offset);
    }

    // Try to allocate - should still work with coalescing
    auto offset = allocator.AllocateOffset(1024);
    REQUIRE_FALSE(offset.IsNull());

    // Clean up
    for (auto &off : keep) {
      allocator.FreeOffset(off);
    }
  }

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Mixed Small and Large", "[buddy_allocator]") {
  MallocBackend backend;
  size_t heap_size = 5 * 1024 * 1024;  // 5MB
  backend.shm_init(MemoryBackendId(0, 0), heap_size);

  BuddyAllocator allocator;
  allocator.shm_init(backend.data_, heap_size);

  SECTION("Interleaved small and large allocations") {
    std::vector<OffsetPtr> allocations;

    // Mix of small and large
    allocations.push_back(allocator.AllocateOffset(128));
    allocations.push_back(allocator.AllocateOffset(32 * 1024));
    allocations.push_back(allocator.AllocateOffset(512));
    allocations.push_back(allocator.AllocateOffset(64 * 1024));
    allocations.push_back(allocator.AllocateOffset(2048));
    allocations.push_back(allocator.AllocateOffset(128 * 1024));

    // All should succeed
    for (auto &offset : allocations) {
      REQUIRE_FALSE(offset.IsNull());
    }

    // Free all
    for (auto &offset : allocations) {
      allocator.FreeOffset(offset);
    }
  }

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Stress Test", "[buddy_allocator]") {
  MallocBackend backend;
  size_t heap_size = 10 * 1024 * 1024;  // 10MB
  backend.shm_init(MemoryBackendId(0, 0), heap_size);

  BuddyAllocator allocator;
  allocator.shm_init(backend.data_, heap_size);

  SECTION("Many allocations and frees") {
    std::vector<OffsetPtr> active;

    // Allocate 100 blocks of varying sizes
    for (int i = 0; i < 100; ++i) {
      size_t size = 32 * (1 + (i % 32));  // Vary from 64 to 1024
      auto offset = allocator.AllocateOffset(size);
      if (!offset.IsNull()) {
        active.push_back(offset);
      }
    }

    // Should have successfully allocated many blocks
    REQUIRE(active.size() > 50);

    // Free half of them
    size_t half = active.size() / 2;
    for (size_t i = 0; i < half; ++i) {
      allocator.FreeOffset(active[i]);
    }

    // Allocate more
    for (int i = 0; i < 50; ++i) {
      size_t size = 64 * (1 + (i % 16));
      auto offset = allocator.AllocateOffset(size);
      if (!offset.IsNull()) {
        active.push_back(offset);
      }
    }

    // Clean up
    for (auto &offset : active) {
      if (!offset.IsNull()) {
        allocator.FreeOffset(offset);
      }
    }
  }

  backend.shm_destroy();
}

TEST_CASE("BuddyAllocator - Out of Memory", "[buddy_allocator]") {
  MallocBackend backend;
  size_t heap_size = 64 * 1024;  // Small 64KB heap
  backend.shm_init(MemoryBackendId(0, 0), heap_size);

  BuddyAllocator allocator;
  allocator.shm_init(backend.data_, heap_size);

  SECTION("Exhaust heap") {
    std::vector<OffsetPtr> allocations;

    // Keep allocating until we fail
    for (int i = 0; i < 1000; ++i) {
      auto offset = allocator.AllocateOffset(1024);
      if (offset.IsNull()) {
        break;
      }
      allocations.push_back(offset);
    }

    // Should eventually fail
    auto failed = allocator.AllocateOffset(8192);
    REQUIRE(failed.IsNull());

    // Clean up
    for (auto &offset : allocations) {
      allocator.FreeOffset(offset);
    }
  }

  backend.shm_destroy();
}
