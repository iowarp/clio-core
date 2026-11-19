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

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "clio_ctp/memory/allocator/mp_allocator.h"
#include "clio_ctp/memory/backend/posix_shm_mmap.h"
#include "allocator_test.h"

using namespace ctp::ipc;
using namespace ctp::testing;

// Shared memory configuration
constexpr size_t kShmSize = 512 * 1024 * 1024;  // 512 MB
const std::string kShmUrl = "/mp_allocator_multiprocess_test";
/** How long a consumer waits for rank 0's ready flag. A hang beyond this is
 *  a real failure (rank 0 died or never initialized), not a slow runner. */
constexpr int kReadyWaitMs = 30000;

/**
 * Path of the flag rank 0 publishes once its allocator is constructed.
 * The driver script passes a per-run path in CTP_MP_ALLOC_READY_FILE.
 * @return the flag path
 */
std::filesystem::path ReadyFlagPath() {
  const char *e = std::getenv("CTP_MP_ALLOC_READY_FILE");
  if (e != nullptr && *e != '\0') return std::filesystem::path(e);
  return std::filesystem::temp_directory_path() /
         "mp_allocator_multiprocess_test.ready";
}

/**
 * Publish the ready flag atomically (write a temp file, then rename), so a
 * consumer never sees a half-created flag.
 * @return true if the flag now exists
 */
bool PublishReadyFlag() {
  const std::filesystem::path flag = ReadyFlagPath();
  const std::filesystem::path tmp = flag.string() + ".tmp";
  { std::ofstream(tmp) << "ready\n"; }
  std::error_code ec;
  std::filesystem::rename(tmp, flag, ec);
  return !ec;
}

/**
 * Block until rank 0 has published the ready flag.
 * @param timeout_ms give up after this long
 * @return true once the flag exists, false on timeout
 */
bool WaitForReadyFlag(int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  const std::filesystem::path flag = ReadyFlagPath();
  while (std::chrono::steady_clock::now() < deadline) {
    std::error_code ec;
    if (std::filesystem::exists(flag, ec)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

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
    std::error_code rm_ec;
    std::filesystem::remove(ReadyFlagPath(), rm_ec);  // a previous run's flag
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
    // Other ranks are consumers. Rank 0 publishes a flag only after both the
    // segment and the allocator in it are constructed; wait for that rather
    // than guessing a delay.
    if (!WaitForReadyFlag(kReadyWaitMs)) {
      std::cerr << "Rank " << rank << ": rank 0 not ready after "
                << kReadyWaitMs << " ms (flag " << ReadyFlagPath() << ")"
                << std::endl;
      return 1;
    }
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
    if (!PublishReadyFlag()) {
      std::cerr << "Rank 0: Failed to publish ready flag " << ReadyFlagPath()
                << std::endl;
      return 1;
    }
    std::cout << "  Allocator size: " << sizeof(ProducerConsumerAllocator) << std::endl;
  } else {
    std::cout << "Rank " << rank << ": Attaching to allocator" << std::endl;

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
