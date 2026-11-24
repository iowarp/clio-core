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
 * Single-threaded multi-process unit test for BuddyAllocator with Ownership Tracking
 *
 * Usage: test_buddy_allocator_multiprocess <rank> <duration_sec>
 *
 * rank 0: Initializes shared memory (owner), optionally runs for duration_sec
 *         then calls UnsetOwner() to indicate another process is taking over
 * rank 1+: Attaches to shared memory (non-owner), calls SetOwner() to indicate
 *          it will manage cleanup, and runs for duration_sec
 *
 * This test validates BuddyAllocator in a single-threaded environment across
 * multiple processes using small allocations (1 byte to 16KB), and tests
 * ownership tracking between processes.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>

#include "clio_ctp/memory/allocator/buddy_allocator.h"
#include "clio_ctp/memory/backend/posix_shm_mmap.h"
#include "allocator_test.h"

using namespace ctp::ipc;
using namespace ctp::testing;

// Shared memory configuration
constexpr size_t kShmSize = 512UL * 1024UL * 1024UL;  // 512 MB
const std::string kShmUrl = "/buddy_allocator_multiprocess_test";

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <rank> <duration_sec>" << std::endl;
    return 1;
  }

  int rank = std::atoi(argv[1]);
  int duration_sec = std::atoi(argv[2]);

  std::cout << "Rank " << rank << ": Starting single-threaded test for "
            << duration_sec << " seconds" << std::endl;

  // Create or attach to shared memory
  PosixShmMmap backend;
  bool success = false;

  if (rank == 0) {
    // Rank 0 initializes
    std::cout << "Rank 0: Initializing shared memory" << std::endl;
    success = backend.shm_init(MemoryBackendId(0, 0), kShmSize, kShmUrl);
    if (!success) {
      std::cerr << "Rank 0: Failed to initialize shared memory" << std::endl;
      return 1;
    }
    std::cout << "Rank 0: Shared memory initialized successfully" << std::endl;
    std::cout << "  Shared memory size: " << kShmSize << " bytes ("
              << (kShmSize / (1024UL * 1024UL)) << " MB)" << std::endl;
    std::cout << "Rank 0: Backend owner flag set (IsOwner = "
              << (backend.IsOwner() ? "true" : "false") << ")" << std::endl;

    backend.UnsetOwner();
  } else {
    // Other ranks attach to existing shared memory
    std::cout << "Rank " << rank << ": Attaching to shared memory" << std::endl;

    // Give rank 0 time to fully initialize before we try to attach
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    success = backend.shm_attach(kShmUrl);
    if (!success) {
      std::cerr << "Rank " << rank << ": Failed to attach to shared memory"
                << std::endl;
      return 1;
    }
    std::cout << "Rank " << rank << ": Attached to shared memory successfully"
              << std::endl;
    std::cout << "Rank " << rank << ": Backend owner flag set (IsOwner = "
              << (backend.IsOwner() ? "true" : "false") << ")" << std::endl;

    // Rank 1+ takes ownership of the backend
    backend.SetOwner();
    std::cout << "Rank " << rank << ": Called SetOwner() (IsOwner = "
              << (backend.IsOwner() ? "true" : "false") << ")" << std::endl;
  }

  // Initialize or attach allocator
  BuddyAllocator *allocator = nullptr;

  if (rank == 0) {
    std::cout << "Rank 0: Initializing BuddyAllocator" << std::endl;
    std::cout << "  Backend data capacity: " << backend.data_capacity_
              << " bytes" << std::endl;

    allocator = backend.MakeAlloc<BuddyAllocator>();
    if (allocator == nullptr) {
      std::cerr << "Rank 0: Failed to initialize BuddyAllocator" << std::endl;
      return 1;
    }

    std::cout << "Rank 0: BuddyAllocator initialized successfully" << std::endl;
    std::cout << "  Allocator size: " << sizeof(BuddyAllocator) << " bytes"
              << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Attaching to BuddyAllocator"
              << std::endl;

    // Attach to existing allocator without reinitializing
    allocator = backend.AttachAlloc<BuddyAllocator>();
    if (allocator == nullptr) {
      std::cerr << "Rank " << rank << ": Failed to attach to BuddyAllocator"
                << std::endl;
      return 1;
    }

    std::cout << "Rank " << rank
              << ": Attached to BuddyAllocator successfully" << std::endl;
  }

  // Run test if duration > 0
  if (duration_sec > 0) {
    std::cout << "Rank " << rank
              << ": Starting single-threaded timed workload test for "
              << duration_sec << " seconds" << std::endl;
    std::cout << "Rank " << rank
              << ": Testing SMALL allocations only (1 byte to 16KB)"
              << std::endl;

    // Create allocator tester and run timed workload with SMALL allocations
    AllocatorTest<BuddyAllocator> tester(allocator);
    constexpr size_t kSmallMin = 1UL;          // 1 byte
    constexpr size_t kSmallMax = 16UL * 1024UL;  // 16 KB

    // Run the timed random allocation test (single-threaded)
    tester.TestRandomAllocationTimed(duration_sec, kSmallMin, kSmallMax);

    std::cout << "Rank " << rank << ": TEST PASSED" << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Initialization complete, exiting"
              << std::endl;
  }

  // Rank 0 releases ownership after test completes
  if (rank == 0) {
    backend.UnsetOwner();
    std::cout << "Rank 0: Called UnsetOwner() (IsOwner = "
              << (backend.IsOwner() ? "true" : "false") << ")" << std::endl;
  }

  return 0;
}
