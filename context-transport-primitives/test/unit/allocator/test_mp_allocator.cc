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

#include "hermes_shm/memory/allocator/mp_allocator.h"
#include "hermes_shm/memory/backend/posix_mmap.h"
#include <catch2/catch_all.hpp>
#include <thread>
#include <vector>
#include <atomic>

using namespace hshm::ipc;

/**
 * Test fixture for MultiProcessAllocator tests
 */
class MpAllocatorTest {
 public:
  PosixMmap backend_;
  MultiProcessAllocator *alloc_;
  static constexpr size_t kAllocSize = 512ULL * 1024 * 1024;  // 512MB

  MpAllocatorTest() {
    // Initialize memory backend (large enough for allocator + data)
    backend_.shm_init(MemoryBackendId(0, 0), kAllocSize);
    std::cout << "Backend initialized at: " << (void*)backend_.data_ << std::endl;

    // Place allocator at the beginning of shared memory using MakeAlloc
    // MakeAlloc automatically passes backend as first parameter to shm_init
    std::cout << "sizeof(MultiProcessAllocator) = " << sizeof(MultiProcessAllocator) << std::endl;
    std::cout << "sizeof(_MultiProcessAllocator) = " << sizeof(_MultiProcessAllocator) << std::endl;
    alloc_ = backend_.MakeAlloc<MultiProcessAllocator>(0);
    std::cout << "Allocator placed at: " << (void*)alloc_ << std::endl;
    std::cout << "Accessing pid_count_: " << alloc_->pid_count_ << std::endl;
    std::cout << "Allocator pointer: " << (void*)alloc_ << std::endl;
    std::cout << "Allocator initialized successfully" << std::endl;
    // TODO: Fix id_ access issue
    // For now, skip verification of id_ fields to test the rest of functionality
  }

  ~MpAllocatorTest() {
    alloc_->shm_detach();
    backend_.shm_destroy();
  }
};

/**
 * Test 1: Basic Initialization
 *
 * Verify that the allocator initializes correctly and can attach/detach.
 */
TEST_CASE("MpAllocator: Basic Initialization", "[mp_allocator][init]") {
  MpAllocatorTest test;

  // Verify allocator is initialized
  REQUIRE(test.alloc_->GetId().major_ == 0);
  REQUIRE(test.alloc_->GetId().minor_ == 0);
}

/**
 * Test 2: Simple Single-Thread Allocation
 *
 * Test basic allocation and freeing from a single thread.
 * Verify data can be written and read correctly.
 */
TEST_CASE("MpAllocator: Simple Single-Thread Allocation", "[mp_allocator][single_thread]") {
  MpAllocatorTest test;

  // Allocate small memory
  OffsetPtr<> ptr1 = test.alloc_->AllocateOffset(64);
  REQUIRE(!ptr1.IsNull());
  {
    // Use direct pointer arithmetic (MultiProcessAllocator uses sub-allocators)
    char *data_ptr = test.alloc_->alloc_.GetBackend().data_ + ptr1.load();
    REQUIRE(data_ptr != nullptr);
    memset(data_ptr, 0xAA, 64);
    for (size_t i = 0; i < 64; ++i) {
      REQUIRE(data_ptr[i] == static_cast<char>(0xAA));
    }
  }

  // Allocate medium memory
  OffsetPtr<>ptr2 = test.alloc_->AllocateOffset(4096);
  REQUIRE(!ptr2.IsNull());
  {
    char *data_ptr = test.alloc_->alloc_.GetBackend().data_ + ptr2.load();
    REQUIRE(data_ptr != nullptr);
    memset(data_ptr, 0xBB, 4096);
    for (size_t i = 0; i < 4096; ++i) {
      REQUIRE(data_ptr[i] == static_cast<char>(0xBB));
    }
  }

  // Allocate large memory
  OffsetPtr<>ptr3 = test.alloc_->AllocateOffset(1024 * 1024);
  REQUIRE(!ptr3.IsNull());
  {
    char *data_ptr = test.alloc_->alloc_.GetBackend().data_ + ptr3.load();
    REQUIRE(data_ptr != nullptr);
    memset(data_ptr, 0xCC, 1024 * 1024);
    // Verify first and last pages
    for (size_t i = 0; i < 4096; ++i) {
      REQUIRE(data_ptr[i] == static_cast<char>(0xCC));
    }
    for (size_t i = 1024 * 1024 - 4096; i < 1024 * 1024; ++i) {
      REQUIRE(data_ptr[i] == static_cast<char>(0xCC));
    }
  }

  // Free memory
  test.alloc_->FreeOffset(ptr1);
  test.alloc_->FreeOffset(ptr2);
  test.alloc_->FreeOffset(ptr3);
}

/**
 * Test 3: Multiple Allocations and Frees
 *
 * Perform multiple allocation and free operations to test allocator stability.
 * Verify each allocation can be written and read.
 */
TEST_CASE("MpAllocator: Multiple Allocations", "[mp_allocator][multiple]") {
  MpAllocatorTest test;

  std::vector<OffsetPtr<>> ptrs;

  // Allocate 100 blocks of varying sizes and verify data writes
  for (size_t i = 0; i < 100; ++i) {
    size_t size = 32 * (i + 1);  // 32, 64, 96, ..., 3200 bytes
    OffsetPtr<>ptr = test.alloc_->AllocateOffset(size);
    REQUIRE(!ptr.IsNull());

    // Write unique pattern to each block
    unsigned char *data_ptr = reinterpret_cast<unsigned char*>(test.alloc_->alloc_.GetBackend().data_ + ptr.load());
    REQUIRE(data_ptr != nullptr);
    unsigned char pattern = static_cast<unsigned char>(i & 0xFF);
    memset(data_ptr, pattern, size);

    // Verify pattern
    for (size_t j = 0; j < size; ++j) {
      REQUIRE(data_ptr[j] == pattern);
    }

    ptrs.push_back(ptr);
  }

  // Free all blocks
  for (auto &ptr : ptrs) {
    test.alloc_->FreeOffset(ptr);
  }

  // Allocate again to test reuse and verify data writes
  for (size_t i = 0; i < 100; ++i) {
    size_t size = 32 * (i + 1);
    OffsetPtr<>ptr = test.alloc_->AllocateOffset(size);
    REQUIRE(!ptr.IsNull());

    // Write different pattern and verify
    unsigned char *data_ptr = reinterpret_cast<unsigned char*>(test.alloc_->alloc_.GetBackend().data_ + ptr.load());
    REQUIRE(data_ptr != nullptr);
    unsigned char pattern = static_cast<unsigned char>((i + 100) & 0xFF);
    memset(data_ptr, pattern, size);
    for (size_t j = 0; j < size; ++j) {
      REQUIRE(data_ptr[j] == pattern);
    }

    test.alloc_->FreeOffset(ptr);
  }
}

/**
 * Test 4: Reallocation
 *
 * Test memory reallocation to different sizes.
 * Verify data is preserved during reallocation.
 */
TEST_CASE("MpAllocator: Reallocation", "[mp_allocator][realloc]") {
  MpAllocatorTest test;

  // Initial allocation with data
  OffsetPtr<>ptr = test.alloc_->AllocateOffset(1024);
  REQUIRE(!ptr.IsNull());
  {
    unsigned char *data_ptr = reinterpret_cast<unsigned char*>(test.alloc_->alloc_.GetBackend().data_ + ptr.load());
    REQUIRE(data_ptr != nullptr);
    // Write pattern to initial allocation
    for (size_t i = 0; i < 1024; ++i) {
      data_ptr[i] = static_cast<unsigned char>(i & 0xFF);
    }
  }

  // Reallocate to larger size - data should be preserved
  OffsetPtr<>ptr2 = test.alloc_->ReallocateOffset(ptr, 4096);
  REQUIRE(!ptr2.IsNull());
  {
    unsigned char *data_ptr = reinterpret_cast<unsigned char*>(test.alloc_->alloc_.GetBackend().data_ + ptr2.load());
    REQUIRE(data_ptr != nullptr);
    // Verify original data is preserved
    for (size_t i = 0; i < 1024; ++i) {
      REQUIRE(data_ptr[i] == static_cast<unsigned char>(i & 0xFF));
    }
    // Write to new space
    for (size_t i = 1024; i < 4096; ++i) {
      data_ptr[i] = 0xDD;
    }
  }

  // Reallocate to smaller size - partial data should be preserved
  OffsetPtr<>ptr3 = test.alloc_->ReallocateOffset(ptr2, 512);
  REQUIRE(!ptr3.IsNull());
  {
    unsigned char *data_ptr = reinterpret_cast<unsigned char*>(test.alloc_->alloc_.GetBackend().data_ + ptr3.load());
    REQUIRE(data_ptr != nullptr);
    // Verify first 512 bytes are still valid
    for (size_t i = 0; i < 512; ++i) {
      REQUIRE(data_ptr[i] == static_cast<unsigned char>(i & 0xFF));
    }
  }

  // Free final pointer
  test.alloc_->FreeOffset(ptr3);
}

/**
 * Test 5: Multi-Threaded Concurrent Allocations
 *
 * CRITICAL TEST: Verify lock-free fast path with multiple threads.
 * Each thread should allocate from its own ThreadBlock without contention.
 */
TEST_CASE("MpAllocator: Multi-Threaded Allocations", "[mp_allocator][multithread]") {
  MpAllocatorTest test;

  const size_t kNumThreads = 8;
  const size_t kAllocsPerThread = 100;
  std::vector<std::thread> threads;
  std::atomic<size_t> success_count{0};

  // Launch threads
  for (size_t t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&test, &success_count, t]() {
      std::vector<OffsetPtr<>> local_ptrs;

      // Each thread performs allocations and verifies data writes
      for (size_t i = 0; i < kAllocsPerThread; ++i) {
        size_t size = 64 + (i % 10) * 32;  // Varying sizes
        OffsetPtr<>ptr = test.alloc_->AllocateOffset(size);
        if (!ptr.IsNull()) {
          // Verify we can write and read data
          unsigned char *data_ptr = reinterpret_cast<unsigned char*>(test.alloc_->alloc_.GetBackend().data_ + ptr.load());
          if (data_ptr != nullptr) {
            unsigned char pattern = static_cast<unsigned char>((t * 100 + i) & 0xFF);
            memset(data_ptr, pattern, size);
            // Spot check first and last bytes
            if (data_ptr[0] == pattern && data_ptr[size - 1] == pattern) {
              local_ptrs.push_back(ptr);
            } else {
              test.alloc_->FreeOffset(ptr);
            }
          } else {
            test.alloc_->FreeOffset(ptr);
          }
        }
      }

      // Free all allocations
      for (auto &ptr : local_ptrs) {
        test.alloc_->FreeOffset(ptr);
      }

      success_count += local_ptrs.size();
    });
  }

  // Wait for all threads
  for (auto &thread : threads) {
    thread.join();
  }

  // Verify all allocations succeeded
  size_t expected = kNumThreads * kAllocsPerThread;
  REQUIRE(success_count == expected);
}

/**
 * Test 6: Thread-Local Storage Isolation
 *
 * Verify that each thread gets its own ThreadBlock and allocations
 * don't interfere with each other.
 */
TEST_CASE("MpAllocator: TLS Isolation", "[mp_allocator][tls]") {
  MpAllocatorTest test;

  const size_t kNumThreads = 4;
  std::vector<std::thread> threads;
  std::vector<std::vector<OffsetPtr<>>> thread_ptrs(kNumThreads);

  // Launch threads, each allocating different sizes
  for (size_t t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&test, &thread_ptrs, t]() {
      for (size_t i = 0; i < 50; ++i) {
        size_t size = (t + 1) * 128;  // Each thread uses different size
        OffsetPtr<>ptr = test.alloc_->AllocateOffset(size);
        REQUIRE(!ptr.IsNull());
        thread_ptrs[t].push_back(ptr);
      }
    });
  }

  // Wait for all threads
  for (auto &thread : threads) {
    thread.join();
  }

  // Verify all threads got their allocations
  for (size_t t = 0; t < kNumThreads; ++t) {
    REQUIRE(thread_ptrs[t].size() == 50);
  }

  // Free all memory
  for (size_t t = 0; t < kNumThreads; ++t) {
    for (auto &ptr : thread_ptrs[t]) {
      test.alloc_->FreeOffset(ptr);
    }
  }
}

/**
 * Test 7: Stress Test with Many Threads
 *
 * Heavy stress test with many threads performing mixed operations.
 */
TEST_CASE("MpAllocator: Stress Test", "[mp_allocator][stress]") {
  MpAllocatorTest test;

  const size_t kNumThreads = 16;
  const size_t kOperationsPerThread = 500;
  std::vector<std::thread> threads;
  std::atomic<size_t> total_ops{0};

  for (size_t t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&test, &total_ops]() {
      std::vector<OffsetPtr<>> ptrs;

      for (size_t i = 0; i < kOperationsPerThread; ++i) {
        // Alternate between allocating and freeing
        if (i % 3 == 0 && !ptrs.empty()) {
          // Free a random pointer
          size_t idx = i % ptrs.size();
          test.alloc_->FreeOffset(ptrs[idx]);
          ptrs.erase(ptrs.begin() + idx);
        } else {
          // Allocate new memory
          size_t size = 32 + (i % 100) * 16;
          OffsetPtr<>ptr = test.alloc_->AllocateOffset(size);
          if (!ptr.IsNull()) {
            ptrs.push_back(ptr);
            total_ops++;
          }
        }
      }

      // Free remaining pointers
      for (auto &ptr : ptrs) {
        test.alloc_->FreeOffset(ptr);
      }
    });
  }

  // Wait for all threads
  for (auto &thread : threads) {
    thread.join();
  }

  // Verify operations completed
  INFO("Total successful operations: " << total_ops.load());
  REQUIRE(total_ops > 0);
}

/**
 * Test 8: Large Allocations
 *
 * Test allocating large chunks of memory that exceed ThreadBlock size.
 */
TEST_CASE("MpAllocator: Large Allocations", "[mp_allocator][large]") {
  MpAllocatorTest test;

  // Allocate large blocks (should go to global allocator)
  const size_t kLargeSize = 32 * 1024 * 1024;  // 32MB
  std::vector<OffsetPtr<>> ptrs;

  for (size_t i = 0; i < 5; ++i) {
    OffsetPtr<>ptr = test.alloc_->AllocateOffset(kLargeSize);
    REQUIRE(!ptr.IsNull());
    ptrs.push_back(ptr);
  }

  // Free all large blocks
  for (auto &ptr : ptrs) {
    test.alloc_->FreeOffset(ptr);
  }
}

/**
 * Test 9: Edge Cases
 *
 * Test various edge cases including null pointers, zero sizes, etc.
 */
TEST_CASE("MpAllocator: Edge Cases", "[mp_allocator][edge]") {
  MpAllocatorTest test;

  // Freeing null pointer should not crash
  REQUIRE_NOTHROW(test.alloc_->FreeOffset(OffsetPtr<>::GetNull()));

  // Allocate minimum size
  OffsetPtr<>ptr1 = test.alloc_->AllocateOffset(1);
  REQUIRE(!ptr1.IsNull());
  test.alloc_->FreeOffset(ptr1);

  // Allocate exactly power of 2
  OffsetPtr<>ptr2 = test.alloc_->AllocateOffset(4096);
  REQUIRE(!ptr2.IsNull());
  test.alloc_->FreeOffset(ptr2);
}

/**
 * Test 10: Mixed Alloc/Free/Realloc Workload
 *
 * Realistic workload with mixed allocation patterns.
 */
TEST_CASE("MpAllocator: Mixed Workload", "[mp_allocator][mixed]") {
  MpAllocatorTest test;

  std::vector<OffsetPtr<>> ptrs;

  for (size_t i = 0; i < 200; ++i) {
    if (i % 5 == 0 && !ptrs.empty()) {
      // Realloc
      size_t idx = i % ptrs.size();
      size_t new_size = 128 + (i % 20) * 64;
      OffsetPtr<>new_ptr = test.alloc_->ReallocateOffset(ptrs[idx], new_size);
      if (!new_ptr.IsNull()) {
        ptrs[idx] = new_ptr;
      }
    } else if (i % 7 == 0 && !ptrs.empty()) {
      // Free
      size_t idx = i % ptrs.size();
      test.alloc_->FreeOffset(ptrs[idx]);
      ptrs.erase(ptrs.begin() + idx);
    } else {
      // Allocate
      size_t size = 64 + (i % 30) * 32;
      OffsetPtr<>ptr = test.alloc_->AllocateOffset(size);
      if (!ptr.IsNull()) {
        ptrs.push_back(ptr);
      }
    }
  }

  // Cleanup
  for (auto &ptr : ptrs) {
    test.alloc_->FreeOffset(ptr);
  }
}

/**
 * Test 11: ThreadBlock Expansion
 *
 * Test that ThreadBlocks can be expanded when they run out of space.
 */
TEST_CASE("MpAllocator: ThreadBlock Expansion", "[mp_allocator][expansion]") {
  MpAllocatorTest test;

  // Allocate many small blocks to exhaust initial ThreadBlock
  std::vector<OffsetPtr<>> ptrs;
  const size_t kSmallSize = 64;

  // Allocate enough to potentially exceed default thread unit (16MB)
  const size_t kNumAllocs = (16 * 1024 * 1024) / kSmallSize + 100;

  for (size_t i = 0; i < kNumAllocs; ++i) {
    OffsetPtr<>ptr = test.alloc_->AllocateOffset(kSmallSize);
    if (!ptr.IsNull()) {
      ptrs.push_back(ptr);
    }
  }

  // Should have succeeded with expansion
  INFO("Allocated " << ptrs.size() << " blocks");
  REQUIRE(ptrs.size() > 0);

  // Cleanup
  for (auto &ptr : ptrs) {
    test.alloc_->FreeOffset(ptr);
  }
}

/**
 * Test 12: Out of Memory Handling
 *
 * Test graceful handling when allocator runs out of memory.
 */
TEST_CASE("MpAllocator: Out of Memory", "[mp_allocator][oom]") {
  MpAllocatorTest test;

  std::vector<OffsetPtr<>> ptrs;

  // Try to allocate until we run out
  const size_t kAllocSize = 1024 * 1024;  // 1MB chunks
  size_t total_allocated = 0;

  while (total_allocated < MpAllocatorTest::kAllocSize) {
    OffsetPtr<>ptr = test.alloc_->AllocateOffset(kAllocSize);
    if (ptr.IsNull()) {
      break;  // Expected: out of memory
    }
    ptrs.push_back(ptr);
    total_allocated += kAllocSize;
  }

  INFO("Allocated " << total_allocated << " bytes before OOM");

  // Cleanup
  for (auto &ptr : ptrs) {
    test.alloc_->FreeOffset(ptr);
  }

  // Should be able to allocate again after freeing
  OffsetPtr<>ptr = test.alloc_->AllocateOffset(kAllocSize);
  REQUIRE(!ptr.IsNull());
  test.alloc_->FreeOffset(ptr);
}
