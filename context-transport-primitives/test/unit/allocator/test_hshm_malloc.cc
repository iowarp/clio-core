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

#include "../../../context-runtime/test/simple_test.h"
#include "hermes_shm/memory/allocator/malloc_allocator.h"
#include <iostream>

using namespace hshm::ipc;

/**
 * Test basic allocation and deallocation with HSHM_MALLOC
 */
TEST_CASE("HSHM_MALLOC: basic allocate and free", "[hshm_malloc][basic]") {
  std::cout << "\n=== Test 1: Basic Allocate and Free ===" << std::endl;

  // Allocate a buffer
  size_t size = 1024;
  auto buffer = HSHM_MALLOC->AllocateObjs<char>(size);

  std::cout << "Allocated " << size << " bytes" << std::endl;
  std::cout << "  ptr_ = " << (void*)buffer.ptr_ << std::endl;
  std::cout << "  off_ = " << buffer.shm_.off_.load() << std::endl;
  std::cout << "  alloc_id = (" << buffer.shm_.alloc_id_.major_
            << "." << buffer.shm_.alloc_id_.minor_ << ")" << std::endl;

  REQUIRE(!buffer.IsNull());
  REQUIRE(buffer.ptr_ != nullptr);

  // Write to the buffer to ensure it's valid
  for (size_t i = 0; i < size; i++) {
    buffer.ptr_[i] = static_cast<char>(i % 256);
  }

  // Verify the data
  for (size_t i = 0; i < size; i++) {
    REQUIRE(buffer.ptr_[i] == static_cast<char>(i % 256));
  }

  std::cout << "Buffer is valid and writable" << std::endl;

  // Free the buffer
  std::cout << "Calling HSHM_MALLOC->Free()..." << std::endl;
  HSHM_MALLOC->Free(buffer);
  std::cout << "Free completed successfully" << std::endl;
}

/**
 * Test multiple allocations and deallocations
 */
TEST_CASE("HSHM_MALLOC: multiple allocations", "[hshm_malloc][multiple]") {
  std::cout << "\n=== Test 2: Multiple Allocations ===" << std::endl;

  const int num_buffers = 10;
  std::vector<FullPtr<char>> buffers;

  // Allocate multiple buffers
  for (int i = 0; i < num_buffers; i++) {
    size_t size = 512 + i * 128;
    auto buffer = HSHM_MALLOC->AllocateObjs<char>(size);

    std::cout << "Allocated buffer " << i << ": " << size << " bytes at ptr_="
              << (void*)buffer.ptr_ << std::endl;

    REQUIRE(!buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    // Write unique pattern
    for (size_t j = 0; j < size; j++) {
      buffer.ptr_[j] = static_cast<char>((i * 100 + j) % 256);
    }

    buffers.push_back(buffer);
  }

  // Verify all buffers
  for (int i = 0; i < num_buffers; i++) {
    size_t size = 512 + i * 128;
    for (size_t j = 0; j < size; j++) {
      REQUIRE(buffers[i].ptr_[j] == static_cast<char>((i * 100 + j) % 256));
    }
  }

  std::cout << "All buffers verified successfully" << std::endl;

  // Free all buffers
  for (int i = 0; i < num_buffers; i++) {
    std::cout << "Freeing buffer " << i << " at ptr_=" << (void*)buffers[i].ptr_ << std::endl;
    HSHM_MALLOC->Free(buffers[i]);
  }

  std::cout << "All buffers freed successfully" << std::endl;
}

/**
 * Test allocation with NULL allocator ID
 */
TEST_CASE("HSHM_MALLOC: allocator ID check", "[hshm_malloc][allocator_id]") {
  std::cout << "\n=== Test 3: Allocator ID Check ===" << std::endl;

  auto buffer = HSHM_MALLOC->AllocateObjs<char>(2048);

  REQUIRE(!buffer.IsNull());

  // Check that HSHM_MALLOC uses NULL allocator ID
  AllocatorId null_id = AllocatorId::GetNull();
  std::cout << "Expected NULL allocator ID: (" << null_id.major_ << "." << null_id.minor_ << ")" << std::endl;
  std::cout << "Actual allocator ID: (" << buffer.shm_.alloc_id_.major_
            << "." << buffer.shm_.alloc_id_.minor_ << ")" << std::endl;

  REQUIRE(buffer.shm_.alloc_id_ == null_id);

  std::cout << "Allocator ID is correctly NULL" << std::endl;

  HSHM_MALLOC->Free(buffer);
  std::cout << "Buffer freed successfully" << std::endl;
}

/**
 * Test the relationship between ptr_ and off_ in FullPtr
 */
TEST_CASE("HSHM_MALLOC: ptr and offset relationship", "[hshm_malloc][ptr_offset]") {
  std::cout << "\n=== Test 4: Ptr and Offset Relationship ===" << std::endl;

  auto buffer = HSHM_MALLOC->AllocateObjs<char>(1024);

  REQUIRE(!buffer.IsNull());

  std::cout << "ptr_ = " << (void*)buffer.ptr_ << std::endl;
  std::cout << "off_ = " << buffer.shm_.off_.load() << std::endl;

  // For HSHM_MALLOC with MallocPage header:
  // - The off_ should point to the data (after MallocPage header)
  // - The ptr_ should also point to the data
  // - They should be the same value

  uintptr_t ptr_value = reinterpret_cast<uintptr_t>(buffer.ptr_);
  size_t off_value = buffer.shm_.off_.load();

  std::cout << "ptr_ as uintptr_t = " << ptr_value << std::endl;
  std::cout << "off_ as size_t = " << off_value << std::endl;

  REQUIRE(ptr_value == off_value);
  std::cout << "ptr_ and off_ are equal (correct for HSHM_MALLOC)" << std::endl;

  HSHM_MALLOC->Free(buffer);
  std::cout << "Buffer freed successfully" << std::endl;
}

/**
 * Test allocating a FullPtr similar to how IpcManager does it
 */
TEST_CASE("HSHM_MALLOC: simulate IpcManager allocation", "[hshm_malloc][ipc_manager]") {
  std::cout << "\n=== Test 5: Simulate IpcManager Allocation Pattern ===" << std::endl;

  // Simulate the pattern used in IpcManager::AllocateBuffer (RUNTIME mode)
  size_t size = sizeof(int) + 4096;  // Simulate FutureShm + copy_space

  std::cout << "Allocating " << size << " bytes (simulating FutureShm)" << std::endl;
  FullPtr<char> buffer = HSHM_MALLOC->AllocateObjs<char>(size);

  REQUIRE(!buffer.IsNull());
  REQUIRE(buffer.ptr_ != nullptr);
  REQUIRE(buffer.shm_.alloc_id_ == AllocatorId::GetNull());

  std::cout << "Allocation successful:" << std::endl;
  std::cout << "  ptr_ = " << (void*)buffer.ptr_ << std::endl;
  std::cout << "  off_ = " << buffer.shm_.off_.load() << std::endl;
  std::cout << "  alloc_id = NULL (as expected)" << std::endl;

  // Write some data
  std::memset(buffer.ptr_, 0xAB, size);

  // Verify
  for (size_t i = 0; i < size; i++) {
    REQUIRE(buffer.ptr_[i] == static_cast<char>(0xAB));
  }

  std::cout << "Data written and verified" << std::endl;

  // Now free it the same way IpcManager::FreeBuffer does
  std::cout << "Freeing buffer with HSHM_MALLOC->Free()..." << std::endl;

  // This is the exact call that IpcManager makes
  HSHM_MALLOC->Free(buffer);

  std::cout << "Free completed successfully!" << std::endl;
}

/**
 * Test double-free detection (should fail if attempted)
 */
TEST_CASE("HSHM_MALLOC: verify single free only", "[hshm_malloc][single_free]") {
  std::cout << "\n=== Test 6: Verify Single Free Only ===" << std::endl;

  auto buffer = HSHM_MALLOC->AllocateObjs<char>(512);
  REQUIRE(!buffer.IsNull());

  std::cout << "Allocated buffer at ptr_=" << (void*)buffer.ptr_ << std::endl;

  // Free once (should work)
  std::cout << "Freeing buffer (first time)..." << std::endl;
  HSHM_MALLOC->Free(buffer);
  std::cout << "First free completed" << std::endl;

  // Note: We do NOT attempt a second free here because that would crash the test
  // The point is to verify that a single free works correctly

  std::cout << "Single free verified - not attempting double free" << std::endl;
}

/**
 * Test reallocation functionality
 */
TEST_CASE("HSHM_MALLOC: reallocation", "[hshm_malloc][realloc]") {
  std::cout << "\n=== Test 7: Reallocation ===" << std::endl;

  // Initial allocation
  size_t initial_size = 512;
  auto buffer = HSHM_MALLOC->AllocateObjs<char>(initial_size);
  REQUIRE(!buffer.IsNull());

  std::cout << "Initial allocation: " << initial_size << " bytes at ptr_=" << (void*)buffer.ptr_ << std::endl;

  // Write unique pattern
  for (size_t i = 0; i < initial_size; i++) {
    buffer.ptr_[i] = static_cast<char>(i % 256);
  }

  // Reallocate to larger size
  size_t new_size = 2048;
  auto new_buffer = HSHM_MALLOC->ReallocateObjs<char>(buffer, new_size);
  REQUIRE(!new_buffer.IsNull());

  std::cout << "Reallocated to " << new_size << " bytes at ptr_=" << (void*)new_buffer.ptr_ << std::endl;

  // Verify original data is preserved
  for (size_t i = 0; i < initial_size; i++) {
    REQUIRE(new_buffer.ptr_[i] == static_cast<char>(i % 256));
  }
  std::cout << "Original data preserved after reallocation" << std::endl;

  // Write to the new space
  for (size_t i = initial_size; i < new_size; i++) {
    new_buffer.ptr_[i] = static_cast<char>((i + 100) % 256);
  }

  // Verify all data
  for (size_t i = initial_size; i < new_size; i++) {
    REQUIRE(new_buffer.ptr_[i] == static_cast<char>((i + 100) % 256));
  }
  std::cout << "New space is accessible" << std::endl;

  // Free
  HSHM_MALLOC->Free(new_buffer);
  std::cout << "Reallocation test completed successfully" << std::endl;
}

/**
 * Test reallocation to smaller size
 */
TEST_CASE("HSHM_MALLOC: realloc smaller", "[hshm_malloc][realloc_small]") {
  std::cout << "\n=== Test 8: Reallocation to Smaller Size ===" << std::endl;

  // Initial large allocation
  size_t initial_size = 4096;
  auto buffer = HSHM_MALLOC->AllocateObjs<char>(initial_size);
  REQUIRE(!buffer.IsNull());

  std::cout << "Initial allocation: " << initial_size << " bytes" << std::endl;

  // Write pattern
  for (size_t i = 0; i < initial_size; i++) {
    buffer.ptr_[i] = static_cast<char>(i % 256);
  }

  // Reallocate to smaller size
  size_t new_size = 256;
  auto new_buffer = HSHM_MALLOC->ReallocateObjs<char>(buffer, new_size);
  REQUIRE(!new_buffer.IsNull());

  std::cout << "Reallocated to " << new_size << " bytes" << std::endl;

  // Verify data in smaller region is preserved
  for (size_t i = 0; i < new_size; i++) {
    REQUIRE(new_buffer.ptr_[i] == static_cast<char>(i % 256));
  }
  std::cout << "Data preserved in smaller buffer" << std::endl;

  HSHM_MALLOC->Free(new_buffer);
  std::cout << "Smaller reallocation test completed" << std::endl;
}

/**
 * Test TLS functions (no-op but should not crash)
 */
TEST_CASE("HSHM_MALLOC: TLS functions", "[hshm_malloc][tls]") {
  std::cout << "\n=== Test 9: TLS Functions ===" << std::endl;

  // These are no-ops for MallocAllocator but should not crash
  HSHM_MALLOC->CreateTls();
  std::cout << "CreateTls() completed (no-op)" << std::endl;

  HSHM_MALLOC->FreeTls();
  std::cout << "FreeTls() completed (no-op)" << std::endl;

  std::cout << "TLS functions test passed" << std::endl;
}

/**
 * Test GetCurrentlyAllocatedSize
 */
TEST_CASE("HSHM_MALLOC: allocated size tracking", "[hshm_malloc][size_tracking]") {
  std::cout << "\n=== Test 10: Allocated Size Tracking ===" << std::endl;

  // This will return 0 unless HSHM_ALLOC_TRACK_SIZE is defined
  size_t initial = HSHM_MALLOC->GetCurrentlyAllocatedSize();
  std::cout << "Initial allocated size: " << initial << std::endl;

  // The result depends on compile-time flag, so just verify it doesn't crash
  std::cout << "GetCurrentlyAllocatedSize() test passed" << std::endl;
}

/**
 * Test various allocation sizes
 */
TEST_CASE("HSHM_MALLOC: various sizes", "[hshm_malloc][sizes]") {
  std::cout << "\n=== Test 11: Various Allocation Sizes ===" << std::endl;

  std::vector<size_t> sizes = {1, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144};

  for (size_t size : sizes) {
    auto buffer = HSHM_MALLOC->AllocateObjs<char>(size);
    REQUIRE(!buffer.IsNull());

    // Write and verify
    std::memset(buffer.ptr_, 0xAB, size);
    for (size_t i = 0; i < size; i++) {
      REQUIRE(buffer.ptr_[i] == static_cast<char>(0xAB));
    }

    HSHM_MALLOC->Free(buffer);
    std::cout << "  Size " << size << " bytes: OK" << std::endl;
  }

  std::cout << "Various sizes test passed" << std::endl;
}

/**
 * Stress test with rapid alloc/free cycles
 */
TEST_CASE("HSHM_MALLOC: stress test", "[hshm_malloc][stress]") {
  std::cout << "\n=== Test 12: Stress Test ===" << std::endl;

  const int iterations = 10000;
  const size_t size = 1024;

  for (int i = 0; i < iterations; i++) {
    auto buffer = HSHM_MALLOC->AllocateObjs<char>(size);
    REQUIRE(!buffer.IsNull());

    // Quick write
    buffer.ptr_[0] = static_cast<char>(i % 256);
    buffer.ptr_[size - 1] = static_cast<char>(i % 256);

    HSHM_MALLOC->Free(buffer);
  }

  std::cout << "Completed " << iterations << " alloc/free cycles" << std::endl;
  std::cout << "Stress test passed" << std::endl;
}

SIMPLE_TEST_MAIN()
