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
 * Multi-process unit test for ProducerConsumerAllocator
 *
 * Usage: test_mp_allocator_multiprocess <rank> <duration_sec> <nthreads>
 *
 * rank 0: Initializes shared memory (producer), runs allocations for
 *         duration_sec seconds
 * rank 1+: Attaches to shared memory (consumer), reads only
 */

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "clio_ctp/memory/allocator/mp_allocator.h"
#include "clio_ctp/memory/backend/posix_shm_mmap.h"
#include "allocator_test.h"

using namespace ctp::ipc;
using namespace ctp::testing;

// Shared memory configuration
constexpr size_t kShmSize = 512 * 1024 * 1024;  // 512 MB
const std::string kShmUrl = "/mp_allocator_multiprocess_test";

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <rank> <duration_sec> <nthreads>" << std::endl;
    return 1;
  }

  int rank = std::atoi(argv[1]);
  int duration_sec = std::atoi(argv[2]);
  int nthreads = std::atoi(argv[3]);

  std::cout << "Rank " << rank << ": Starting test with " << nthreads
            << " threads for " << duration_sec << " seconds" << std::endl;

  // Create or attach to shared memory
  PosixShmMmap backend;
  bool success = false;

  if (rank == 0) {
    // Rank 0 is the producer — initializes shared memory
    std::cout << "Rank 0: Initializing shared memory" << std::endl;
    success = backend.shm_init(MemoryBackendId(0, 0), kShmSize, kShmUrl);
    if (!success) {
      std::cerr << "Rank 0: Failed to initialize shared memory" << std::endl;
      return 1;
    }
    std::cout << "Rank 0: Backend owner flag set (IsOwner = "
              << (backend.IsOwner() ? "true" : "false") << ")" << std::endl;
    // Memset backend.data_ to 11 before allocator construction
    std::memset(backend.data_, 11, backend.data_capacity_);
    backend.UnsetOwner();
  } else {
    // Other ranks are consumers — attach to existing shared memory
    std::cout << "Rank " << rank << ": Attaching to shared memory" << std::endl;
    success = backend.shm_attach(kShmUrl);
    if (!success) {
      std::cerr << "Rank " << rank << ": Failed to attach to shared memory" << std::endl;
      return 1;
    }
    std::cout << "Rank " << rank << ": Backend owner flag set (IsOwner = "
              << (backend.IsOwner() ? "true" : "false") << ")" << std::endl;

    // Consumer takes ownership for cleanup purposes
    backend.SetOwner();
    std::cout << "Rank " << rank << ": Called SetOwner() (IsOwner = "
              << (backend.IsOwner() ? "true" : "false") << ")" << std::endl;
  }

  // Initialize or attach allocator
  ProducerConsumerAllocator *allocator = nullptr;

  if (rank == 0) {
    std::cout << "Rank 0: Initializing allocator" << std::endl;
    std::cout << "  Backend data capacity: " << backend.data_capacity_ << std::endl;
    allocator = backend.MakeAlloc<ProducerConsumerAllocator>();
    if (allocator == nullptr) {
      std::cerr << "Rank 0: Failed to initialize allocator" << std::endl;
      return 1;
    }
    std::cout << "Rank 0: Allocator initialized successfully" << std::endl;
    std::cout << "  Allocator size: " << sizeof(ProducerConsumerAllocator) << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Attaching to allocator" << std::endl;

    // Give rank 0 time to fully initialize before we try to attach
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Attach to existing allocator without reinitializing
    allocator = backend.AttachAlloc<ProducerConsumerAllocator>();
    if (allocator == nullptr) {
      std::cerr << "Rank " << rank << ": Failed to attach to allocator" << std::endl;
      return 1;
    }
    std::cout << "Rank " << rank << ": Attached to allocator successfully" << std::endl;
  }

  // Run test if duration > 0
  if (duration_sec > 0) {
    std::cout << "Rank " << rank << ": Starting timed workload test with " << nthreads
              << " threads for " << duration_sec << " seconds" << std::endl;

    // Create allocator tester and run timed workload with SMALL allocations only
    AllocatorTest<ProducerConsumerAllocator> tester(allocator);
    constexpr size_t kAllocMin = 1;           // 1 byte
    constexpr size_t kAllocMax = 16 * 1024;   // 16 KB
    tester.TestTimedMultiThreadedWorkload(nthreads, duration_sec, kAllocMin,
                                          kAllocMax);
    std::cout << "Rank " << rank << ": TEST PASSED" << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Initialization complete, exiting" << std::endl;
  }

  // Rank 0 releases ownership after test completes
  if (rank == 0) {
    backend.UnsetOwner();
    std::cout << "Rank 0: Called UnsetOwner() (IsOwner = "
              << (backend.IsOwner() ? "true" : "false") << ")" << std::endl;
  }

  return 0;
}
