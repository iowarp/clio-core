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

#ifndef HSHM_TEST_UNIT_ALLOCATOR_ALLOCATOR_TEST_H_
#define HSHM_TEST_UNIT_ALLOCATOR_ALLOCATOR_TEST_H_

#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <cstring>
#include "hermes_shm/memory/allocator/allocator.h"

namespace hshm::testing {

/**
 * Templated allocator test class
 * Tests all allocator APIs for a given allocator type
 */
template<typename AllocT>
class AllocatorTest {
 private:
  AllocT *alloc_;
  std::mt19937 rng_;

 public:
  /**
   * Constructor
   * @param alloc The allocator to test
   */
  explicit AllocatorTest(AllocT *alloc)
    : alloc_(alloc), rng_(std::random_device{}()) {}

  /**
   * Test 1: Allocate and free immediately in a loop
   * Same memory size for each allocation
   *
   * @param iterations Number of iterations
   * @param alloc_size Size of each allocation
   */
  void TestAllocFreeImmediate(size_t iterations, size_t alloc_size) {
    for (size_t i = 0; i < iterations; ++i) {
      auto ptr = alloc_->template Allocate<char>(alloc_size);
      if (ptr.IsNull()) {
        throw std::runtime_error("Allocation failed in TestAllocFreeImmediate");
      }
      // Verify allocator correctness by writing to all allocated memory
      std::memset(ptr.ptr_, static_cast<unsigned char>(i & 0xFF), alloc_size);
      alloc_->Free(ptr);
    }
  }

  /**
   * Test 2: Allocate a bunch, then free the bunch
   * Iteratively in a loop. Same memory size per alloc
   *
   * @param iterations Number of iterations
   * @param batch_size Number of allocations per batch
   * @param alloc_size Size of each allocation
   */
  void TestAllocFreeBatch(size_t iterations, size_t batch_size, size_t alloc_size) {
    std::vector<hipc::FullPtr<char>> ptrs;
    ptrs.reserve(batch_size);

    for (size_t iter = 0; iter < iterations; ++iter) {
      // Allocate batch
      for (size_t i = 0; i < batch_size; ++i) {
        auto ptr = alloc_->template Allocate<char>(alloc_size);
        if (ptr.IsNull()) {
          // Clean up already allocated pointers
          for (auto &p : ptrs) {
            alloc_->Free(p);
          }
          throw std::runtime_error("Allocation failed in TestAllocFreeBatch");
        }
        // Verify allocator correctness by writing to all allocated memory
        std::memset(ptr.ptr_, static_cast<unsigned char>((iter * 100 + i) & 0xFF), alloc_size);
        ptrs.push_back(ptr);
      }

      // Free batch
      for (auto &ptr : ptrs) {
        alloc_->Free(ptr);
      }
      ptrs.clear();
    }
  }

  /**
   * Test 3: Random allocation with random sizes
   * Random sizes between 0 and 1MB
   * Up to a total of 64MB or 5000 allocations
   * After all allocations, free. Do this iteratively.
   *
   * @param iterations Number of iterations
   */
  void TestRandomAllocation(size_t iterations) {
    const size_t kMaxAllocSize = 16 * 1024 * 1024;  // 16 MB
    const size_t kMaxTotalSize = 32 * 1024 * 1024;  // 32 MB
    const size_t kMaxAllocations = 5000;

    std::uniform_int_distribution<size_t> size_dist(1, kMaxAllocSize);
    std::vector<std::pair<hipc::FullPtr<char>, size_t>> ptrs;
    ptrs.reserve(kMaxAllocations);

    for (size_t iter = 0; iter < iterations; ++iter) {
      size_t total_allocated = 0;

      // Random allocations
      while (total_allocated < kMaxTotalSize && ptrs.size() < kMaxAllocations) {
        size_t alloc_size = size_dist(rng_);

        // Stop if this allocation would exceed the limit
        if (total_allocated + alloc_size > kMaxTotalSize) {
          break;
        }

        // printf("Allocating size: %lu\n", alloc_size);
        auto ptr = alloc_->template Allocate<char>(alloc_size);
        if (ptr.IsNull()) {
          // Allocation failed - clean up and break
          break;
        }

        // Verify allocator correctness by writing to all allocated memory
        if (!ptr.Validate(alloc_)) {
          throw std::runtime_error("Allocation failed in TestRandomAllocation");
        }
        std::memset(ptr.ptr_, static_cast<unsigned char>((iter + ptrs.size()) & 0xFF), alloc_size);
        ptrs.push_back({ptr, alloc_size});
        total_allocated += alloc_size;
      }

      // Free all allocations
      for (auto &p : ptrs) {
        alloc_->Free(p.first);
      }
      ptrs.clear();
    }
  }

  /**
   * Helper: Random allocation with explicit RNG and size constraints
   * Same as TestRandomAllocation but uses provided RNG and allows specifying min/max sizes
   *
   * @param iterations Number of iterations
   * @param rng Random number generator to use
   * @param min_alloc_size Minimum allocation size (default: 1 byte)
   * @param max_alloc_size Maximum allocation size (default: 1 MB)
   */
  void TestRandomAllocationWithRNG(size_t iterations, std::mt19937 &rng,
                                   size_t min_alloc_size = 1,
                                   size_t max_alloc_size = 1024 * 1024) {
    const size_t kMaxTotalSize = 64 * 1024 * 1024;  // 64 MB
    const size_t kMaxAllocations = 5000;

    std::uniform_int_distribution<size_t> size_dist(min_alloc_size, max_alloc_size);
    std::vector<std::pair<hipc::FullPtr<char>, size_t>> ptrs;
    ptrs.reserve(kMaxAllocations);

    for (size_t iter = 0; iter < iterations; ++iter) {
      size_t total_allocated = 0;

      // Random allocations
      while (total_allocated < kMaxTotalSize && ptrs.size() < kMaxAllocations) {
        size_t alloc_size = size_dist(rng);

        // Stop if this allocation would exceed the limit
        if (total_allocated + alloc_size > kMaxTotalSize) {
          break;
        }

        auto ptr = alloc_->template Allocate<char>(alloc_size);
        if (ptr.IsNull()) {
          // Allocation failed - clean up and break
          break;
        }

        // Verify allocator correctness by writing to all allocated memory
        std::memset(ptr.ptr_,
                    static_cast<unsigned char>((iter + ptrs.size()) & 0xFF),
                    alloc_size);
        
        ptrs.push_back({ptr, alloc_size});
        total_allocated += alloc_size;
      }

      // Free all allocations
      for (auto &p : ptrs) {
        alloc_->Free(p.first);
      }
      ptrs.clear();
    }
  }

  /**
   * Test: Timed random allocation
   * Calls TestRandomAllocation(1) in a loop until time runs out
   *
   * @param duration_sec Duration to run in seconds
   * @param min_alloc_size Minimum allocation size (default: 1 byte)
   * @param max_alloc_size Maximum allocation size (default: 1 MB)
   */
  void TestRandomAllocationTimed(int duration_sec,
                                 size_t min_alloc_size = 1,
                                 size_t max_alloc_size = 1024 * 1024) {
    // Create thread-local RNG
    static thread_local std::mt19937 thread_rng(std::random_device{}());

    auto start = std::chrono::steady_clock::now();
    auto end = start + std::chrono::seconds(duration_sec);

    while (std::chrono::steady_clock::now() < end) {
      try {
        TestRandomAllocationWithRNG(1, thread_rng, min_alloc_size, max_alloc_size);
      } catch (const std::exception &e) {
        // Continue on allocation failure
        continue;
      }
    }
  }

  /**
   * Test 4: Multi-threaded random allocation test
   * Multiple threads calling the random allocation test
   *
   * @param num_threads Number of threads to spawn
   * @param iterations_per_thread Number of iterations per thread
   */
  void TestMultiThreadedRandom(size_t num_threads, size_t iterations_per_thread) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Launch threads
    for (size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back([this, iterations_per_thread]() {
        TestRandomAllocation(iterations_per_thread);
      });
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
      thread.join();
    }
  }

  /**
   * Test 5: Large-then-small allocation pattern
   * First allocate many large blocks, free them, then allocate many small blocks.
   * This tests allocator behavior with different size classes and fragmentation.
   *
   * @param iterations Number of iterations
   * @param num_large_allocs Number of large allocations per iteration
   * @param large_size Size of each large allocation
   * @param num_small_allocs Number of small allocations per iteration
   * @param small_size Size of each small allocation
   */
  void TestLargeThenSmall(size_t iterations, size_t num_large_allocs, size_t large_size,
                          size_t num_small_allocs, size_t small_size) {
    std::vector<hipc::FullPtr<char>> large_ptrs;
    std::vector<hipc::FullPtr<char>> small_ptrs;
    large_ptrs.reserve(num_large_allocs);
    small_ptrs.reserve(num_small_allocs);

    for (size_t iter = 0; iter < iterations; ++iter) {
      // Phase 1: Allocate large blocks
      for (size_t i = 0; i < num_large_allocs; ++i) {
        auto ptr = alloc_->template Allocate<char>(large_size);
        if (ptr.IsNull()) {
          // Clean up and throw
          for (auto &p : large_ptrs) {
            alloc_->Free(p);
          }
          throw std::runtime_error("Large allocation failed in TestLargeThenSmall");
        }
        // Verify allocator correctness by writing to all allocated memory
        std::memset(ptr.ptr_, static_cast<unsigned char>((iter + i) & 0xFF), large_size);
        large_ptrs.push_back(ptr);
      }

      // Phase 2: Free all large blocks
      for (auto &ptr : large_ptrs) {
        alloc_->Free(ptr);
      }
      large_ptrs.clear();

      // Phase 3: Allocate small blocks
      for (size_t i = 0; i < num_small_allocs; ++i) {
        auto ptr = alloc_->template Allocate<char>(small_size);
        if (ptr.IsNull()) {
          // Clean up and throw
          for (auto &p : small_ptrs) {
            alloc_->Free(p);
          }
          throw std::runtime_error("Small allocation failed in TestLargeThenSmall");
        }
        // Verify allocator correctness by writing to all allocated memory
        std::memset(ptr.ptr_, static_cast<unsigned char>((iter + i + 128) & 0xFF), small_size);
        small_ptrs.push_back(ptr);
      }

      // Phase 4: Free all small blocks
      for (auto &ptr : small_ptrs) {
        alloc_->Free(ptr);
      }
      small_ptrs.clear();
    }
  }

  /**
   * Test 6: Timed multi-threaded random workload generator
   * Each thread performs random allocations for a specified duration.
   *
   * @param num_threads Number of threads to spawn
   * @param duration_sec Duration to run the test in seconds
   * @param min_alloc_size Minimum allocation size (default: 1 byte)
   * @param max_alloc_size Maximum allocation size (default: 1 MB)
   */
  void TestTimedMultiThreadedWorkload(size_t num_threads, int duration_sec,
                                      size_t min_alloc_size = 1,
                                      size_t max_alloc_size = 1024 * 1024) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Launch threads that run timed random allocation
    for (size_t i = 0; i < num_threads; ++i) {
      threads.emplace_back([this, duration_sec, min_alloc_size, max_alloc_size]() {
        TestRandomAllocationTimed(duration_sec, min_alloc_size, max_alloc_size);
      });
    }

    // Wait for all threads to complete
    for (auto &thread : threads) {
      thread.join();
    }
  }

  /**
   * Run all tests with default parameters
   */
  void RunAllTests() {
    // Test 1: Allocate and free immediately
    TestAllocFreeImmediate(10000, 1024);

    // Test 2: Batch allocations
    TestAllocFreeBatch(100, 100, 4096);

    // Test 3: Random allocations
    TestRandomAllocation(16);

    // Test 4: Multi-threaded
    TestMultiThreadedRandom(8, 2);

    // Test 5: Large-then-small pattern
    TestLargeThenSmall(10, 100, 1024 * 1024, 1000, 128);
  }
};

}  // namespace hshm::testing

#endif  // HSHM_TEST_UNIT_ALLOCATOR_ALLOCATOR_TEST_H_
