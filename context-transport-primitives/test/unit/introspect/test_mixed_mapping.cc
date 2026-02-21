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
 * Unit test for SystemInfo::MapMixedMemory functionality
 *
 * Tests that:
 * 1. The first 4KB is private (process-local, not shared)
 * 2. The remaining region is shared (visible across processes)
 * 3. The regions are contiguous in virtual memory
 */

#include <iostream>
#include <cstring>
#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#endif
#include "hermes_shm/introspect/system_info.h"

using namespace hshm;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <rank>\n";
    std::cerr << "  rank 0: Initialize and write to memory\n";
    std::cerr << "  rank 1: Attach and verify memory\n";
    return 1;
  }

  int rank = std::atoi(argv[1]);
  const char *shm_name = "/test_mixed_mapping";
  constexpr size_t kPrivateSize = 4 * 1024;  // kBackendHeaderSize (4KB)
  constexpr size_t kSharedSize = 1024 * 1024;  // 1MB shared region
  constexpr size_t kTotalSize = kPrivateSize + kSharedSize;
  constexpr uint8_t kTestValue = 5;

  if (rank == 0) {
    std::cout << "=== Rank 0: Initializing shared memory ===\n";

    // Create shared memory object
    shm_unlink(shm_name);  // Clean up any existing
    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
      perror("shm_open");
      return 1;
    }

    // Set size of shared memory
    if (ftruncate(fd, kSharedSize) == -1) {
      perror("ftruncate");
      close(fd);
      return 1;
    }

    // Create mixed mapping
    File file_handle;
    file_handle.posix_fd_ = fd;
    char *ptr = reinterpret_cast<char *>(
        SystemInfo::MapMixedMemory(file_handle, kPrivateSize, kSharedSize, 0));
    if (!ptr) {
      std::cerr << "ERROR: MapMixedMemory failed\n";
      close(fd);
      return 1;
    }

    char *private_region = ptr;
    char *shared_region = ptr + kPrivateSize;

    std::cout << "Rank 0: Created mixed mapping\n";
    std::cout << "  Private region: " << (void*)private_region << "\n";
    std::cout << "  Shared region:  " << (void*)shared_region << "\n";
    std::cout << "  Total size:     " << kTotalSize << " bytes\n";

    // Verify regions are contiguous
    size_t offset = shared_region - private_region;
    if (offset != kPrivateSize) {
      std::cerr << "ERROR: Regions are not contiguous!\n";
      std::cerr << "  Expected offset: " << kPrivateSize << " bytes\n";
      std::cerr << "  Actual offset:   " << offset << " bytes\n";
      munmap(ptr, kTotalSize);
      close(fd);
      return 1;
    }
    std::cout << "✓ Regions are contiguous (offset: " << offset << " bytes)\n";

    // Memset the ENTIRE region (private + shared) to kTestValue
    std::cout << "Rank 0: Setting entire region to value " << (int)kTestValue << "\n";
    memset(private_region, kTestValue, kTotalSize);

    // Verify the write
    std::cout << "Rank 0: Verifying write...\n";
    std::cout << "  Private region first byte: " << (int)(uint8_t)private_region[0] << "\n";
    std::cout << "  Shared region first byte: " << (int)(uint8_t)shared_region[0] << "\n";

    std::cout << "Rank 0: Complete. Keeping memory alive...\n";
    std::cout << "Press Enter to cleanup...";
    std::cin.get();

    // Cleanup
    munmap(ptr, kTotalSize);
    close(fd);
    shm_unlink(shm_name);
    std::cout << "Rank 0: Cleaned up\n";

  } else if (rank == 1) {
    std::cout << "=== Rank 1: Attaching to shared memory ===\n";

    // Give rank 0 time to initialize
    sleep(1);

    // Open existing shared memory
    int fd = shm_open(shm_name, O_RDWR, 0666);
    if (fd == -1) {
      perror("shm_open");
      return 1;
    }

    // Create mixed mapping
    File file_handle;
    file_handle.posix_fd_ = fd;
    char *ptr = reinterpret_cast<char *>(
        SystemInfo::MapMixedMemory(file_handle, kPrivateSize, kSharedSize, 0));
    if (!ptr) {
      std::cerr << "ERROR: MapMixedMemory failed\n";
      close(fd);
      return 1;
    }

    char *private_region = ptr;
    char *shared_region = ptr + kPrivateSize;

    std::cout << "Rank 1: Attached to mixed mapping\n";
    std::cout << "  Private region: " << (void*)private_region << "\n";
    std::cout << "  Shared region:  " << (void*)shared_region << "\n";

    // Verify regions are contiguous
    size_t offset = shared_region - private_region;
    if (offset != kPrivateSize) {
      std::cerr << "ERROR: Regions are not contiguous!\n";
      std::cerr << "  Expected offset: " << kPrivateSize << " bytes\n";
      std::cerr << "  Actual offset:   " << offset << " bytes\n";
      munmap(ptr, kTotalSize);
      close(fd);
      return 1;
    }
    std::cout << "✓ Regions are contiguous (offset: " << offset << " bytes)\n";

    // Check the private region (should NOT be kTestValue since it's process-local)
    std::cout << "\nRank 1: Checking private region...\n";
    bool private_has_test_value = true;
    for (size_t i = 0; i < kPrivateSize; i++) {
      if ((uint8_t)private_region[i] != kTestValue) {
        private_has_test_value = false;
        break;
      }
    }

    if (private_has_test_value) {
      std::cerr << "ERROR: Private region contains test value (should be independent!)\n";
      std::cerr << "  First byte: " << (int)(uint8_t)private_region[0] << "\n";
      munmap(ptr, kTotalSize);
      close(fd);
      return 1;
    }
    std::cout << "✓ Private region is NOT shared (correct)\n";
    std::cout << "  First byte: " << (int)(uint8_t)private_region[0] << " (expected: not " << (int)kTestValue << ")\n";

    // Check the shared region (SHOULD be kTestValue)
    std::cout << "\nRank 1: Checking shared region...\n";
    bool shared_has_test_value = true;
    size_t check_size = std::min((size_t)4096, kSharedSize);  // Check first 4KB
    for (size_t i = 0; i < check_size; i++) {
      if ((uint8_t)shared_region[i] != kTestValue) {
        std::cerr << "ERROR: Shared region byte " << i << " is "
                  << (int)(uint8_t)shared_region[i] << ", expected " << (int)kTestValue << "\n";
        shared_has_test_value = false;
        break;
      }
    }

    if (!shared_has_test_value) {
      std::cerr << "ERROR: Shared region does not contain test value!\n";
      munmap(ptr, kTotalSize);
      close(fd);
      return 1;
    }
    std::cout << "✓ Shared region IS shared (correct)\n";
    std::cout << "  First byte: " << (int)(uint8_t)shared_region[0] << " (expected: " << (int)kTestValue << ")\n";

    std::cout << "\n===========================================\n";
    std::cout << "ALL TESTS PASSED\n";
    std::cout << "===========================================\n";

    // Cleanup
    munmap(ptr, kTotalSize);
    close(fd);

  } else {
    std::cerr << "ERROR: Invalid rank " << rank << " (must be 0 or 1)\n";
    return 1;
  }

  return 0;
}
