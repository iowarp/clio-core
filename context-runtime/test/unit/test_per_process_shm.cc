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
 * Unit tests for per-process shared memory functionality
 *
 * Tests the IpcManager's per-process shared memory allocation with:
 * - AllocateBuffer() with allocations larger than 1GB to trigger IncreaseClientShm
 * - Multiple segment creation and allocation fallback strategies
 */

#include <chimaera/chimaera.h>

#include <cstring>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>
#ifndef _WIN32
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#endif

#include "../simple_test.h"

namespace {
// Test setup helper - same pattern as other tests
bool initialize_chimaera() {
  return chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
}

/**
 * Start a Chimaera server in a forked child process
 * @return Server process PID
 */
pid_t StartServerProcess() {
  pid_t server_pid = fork();
  if (server_pid == 0) {
    // Redirect child output to prevent log flooding
    (void)freopen("/dev/null", "w", stdout);
    (void)freopen("/dev/null", "w", stderr);
    setenv("CHI_WITH_RUNTIME", "1", 1);
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer, true);
    if (!success) {
      _exit(1);
    }
    sleep(300);
    _exit(0);
  }
  return server_pid;
}

/**
 * Wait for the server's shared memory segment to become available
 * @param max_attempts Maximum polling attempts
 * @return True if server is ready
 */
bool WaitForServer(int max_attempts = 50) {
  const char *user = std::getenv("USER");
  std::string memfd_path =
      std::string("/tmp/chimaera_memfd/chi_main_segment_") +
      (user ? user : "");
  for (int i = 0; i < max_attempts; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    int fd = open(memfd_path.c_str(), O_RDONLY);
    if (fd >= 0) {
      close(fd);
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      return true;
    }
  }
  return false;
}

/**
 * Kill the server process and clean up shared memory
 * @param server_pid PID of the server process
 */
void CleanupServer(pid_t server_pid) {
  if (server_pid > 0) {
    kill(server_pid, SIGTERM);
    int status;
    waitpid(server_pid, &status, 0);
    const char *user = std::getenv("USER");
    std::string memfd_path =
        std::string("/tmp/chimaera_memfd/chi_main_segment_") +
        (user ? user : "");
    unlink(memfd_path.c_str());
  }
}

// Constants for testing
constexpr size_t k1MB = 1ULL * 1024 * 1024;
constexpr size_t k100MB = 100ULL * 1024 * 1024;
constexpr size_t k500MB = 500ULL * 1024 * 1024;
constexpr size_t k1GB = 1ULL * 1024 * 1024 * 1024;
constexpr size_t k1_5GB = 1536ULL * 1024 * 1024;  // 1.5 GB
}  // namespace

// This test MUST be first: it forks server+client processes and requires
// that no runtime has been initialized in the parent yet.
TEST_CASE("Per-process shared memory GetClientShmInfo",
          "[ipc][per_process_shm][shm_info][fork]") {
  // Fork a server, then fork a client child to test GetClientShmInfo.
  // Both children start with clean process state (no prior CHIMAERA_INIT).
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);
  REQUIRE(WaitForServer());

  // Fork a client child to test GetClientShmInfo
  pid_t client_pid = fork();
  if (client_pid == 0) {
    (void)freopen("/dev/null", "w", stdout);
    (void)freopen("/dev/null", "w", stderr);
    setenv("CHI_WITH_RUNTIME", "0", 1);
    setenv("CHI_IPC_MODE", "SHM", 1);
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
      _exit(1);
    }
    auto *client_ipc = CHI_IPC;
    if (!client_ipc) _exit(2);

    auto buffer = client_ipc->AllocateBuffer(k1MB);
    if (buffer.IsNull()) _exit(3);

    chi::ClientShmInfo info = client_ipc->GetClientShmInfo(0);
    if (info.owner_pid != getpid()) _exit(4);
    if (info.shm_index != 0) _exit(5);
    if (info.size == 0) _exit(6);

    std::string expected_prefix =
        "chimaera_" + std::to_string(getpid()) + "_";
    if (info.shm_name.find(expected_prefix) != 0) _exit(7);

    _exit(0);  // Success
  }

  // Parent: wait for client child
  int status = 0;
  waitpid(client_pid, &status, 0);
  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  INFO("Client child exit code: " << exit_code);
  REQUIRE(exit_code == 0);

  CleanupServer(server_pid);
}

TEST_CASE("Per-process shared memory AllocateBuffer medium sizes",
          "[ipc][per_process_shm][allocate][medium]") {
  REQUIRE(initialize_chimaera());

  auto* ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Allocate 100MB buffer") {
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(k100MB);

    REQUIRE_FALSE(buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    // Test memory access at boundaries
    buffer.ptr_[0] = 0x01;
    buffer.ptr_[k100MB - 1] = static_cast<char>(0xFF);

    REQUIRE(buffer.ptr_[0] == 0x01);
    REQUIRE(static_cast<unsigned char>(buffer.ptr_[k100MB - 1]) == 0xFF);

    INFO("Successfully allocated 100MB buffer");
  }

  SECTION("Allocate 500MB buffer") {
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(k500MB);

    REQUIRE_FALSE(buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    // Test memory access at boundaries
    buffer.ptr_[0] = static_cast<char>(0xAA);
    buffer.ptr_[k500MB - 1] = static_cast<char>(0xBB);

    REQUIRE(static_cast<unsigned char>(buffer.ptr_[0]) == 0xAA);
    REQUIRE(static_cast<unsigned char>(buffer.ptr_[k500MB - 1]) == 0xBB);

    INFO("Successfully allocated 500MB buffer");
  }
}

TEST_CASE("Per-process shared memory AllocateBuffer exceeding 1GB",
          "[ipc][per_process_shm][allocate][large]") {
  REQUIRE(initialize_chimaera());

  auto* ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Allocate 1.5GB buffer triggers IncreaseMemory") {
    // This allocation is larger than the initial 1GB segment
    // Should trigger IncreaseMemory() to create a new segment

    INFO("Attempting to allocate 1.5GB buffer (should trigger IncreaseMemory)");

    auto start_time = std::chrono::high_resolution_clock::now();

    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(k1_5GB);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    INFO("Allocation took " << duration.count() << " ms");

    REQUIRE_FALSE(buffer.IsNull());
    REQUIRE(buffer.ptr_ != nullptr);

    // Test memory access at multiple points in the buffer
    buffer.ptr_[0] = 'A';
    buffer.ptr_[k1GB] = 'B';  // Past the 1GB mark
    buffer.ptr_[k1_5GB - 1] = 'Z';

    REQUIRE(buffer.ptr_[0] == 'A');
    REQUIRE(buffer.ptr_[k1GB] == 'B');
    REQUIRE(buffer.ptr_[k1_5GB - 1] == 'Z');

    INFO("Successfully allocated and accessed 1.5GB buffer");
  }
}

TEST_CASE("Per-process shared memory multiple large allocations",
          "[ipc][per_process_shm][allocate][multiple_large]") {
  REQUIRE(initialize_chimaera());

  auto* ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Multiple allocations spanning segments") {
    // Allocate multiple buffers that together exceed the initial segment
    std::vector<hipc::FullPtr<char>> buffers;

    // First allocation: 600MB
    INFO("Allocating first 600MB buffer");
    auto buffer1 = ipc_manager->AllocateBuffer(600ULL * k1MB);
    REQUIRE_FALSE(buffer1.IsNull());
    buffers.push_back(buffer1);

    // Second allocation: 600MB (total 1.2GB, should trigger new segment)
    INFO("Allocating second 600MB buffer (should trigger IncreaseMemory)");
    auto buffer2 = ipc_manager->AllocateBuffer(600ULL * k1MB);
    REQUIRE_FALSE(buffer2.IsNull());
    buffers.push_back(buffer2);

    // Verify all buffers are usable and have different addresses
    for (size_t i = 0; i < buffers.size(); ++i) {
      buffers[i].ptr_[0] = static_cast<char>(i);
      buffers[i].ptr_[600ULL * k1MB - 1] = static_cast<char>(i + 100);
    }

    // Verify data integrity
    for (size_t i = 0; i < buffers.size(); ++i) {
      REQUIRE(buffers[i].ptr_[0] == static_cast<char>(i));
      REQUIRE(buffers[i].ptr_[600ULL * k1MB - 1] == static_cast<char>(i + 100));
    }

    // Verify all buffers have different addresses
    for (size_t i = 0; i < buffers.size(); ++i) {
      for (size_t j = i + 1; j < buffers.size(); ++j) {
        REQUIRE(buffers[i].ptr_ != buffers[j].ptr_);
      }
    }

    INFO("Successfully allocated multiple large buffers spanning segments");
  }
}

TEST_CASE("Per-process shared memory allocation patterns",
          "[ipc][per_process_shm][allocate][patterns]") {
  REQUIRE(initialize_chimaera());

  auto* ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Mixed small and large allocations") {
    std::vector<hipc::FullPtr<char>> small_buffers;
    std::vector<hipc::FullPtr<char>> large_buffers;

    // Allocate some small buffers first
    for (int i = 0; i < 5; ++i) {
      auto buffer = ipc_manager->AllocateBuffer(k1MB);
      REQUIRE_FALSE(buffer.IsNull());
      small_buffers.push_back(buffer);
    }

    // Now allocate a large buffer
    auto large1 = ipc_manager->AllocateBuffer(k500MB);
    REQUIRE_FALSE(large1.IsNull());
    large_buffers.push_back(large1);

    // More small buffers
    for (int i = 0; i < 3; ++i) {
      auto buffer = ipc_manager->AllocateBuffer(k1MB);
      REQUIRE_FALSE(buffer.IsNull());
      small_buffers.push_back(buffer);
    }

    // Another large buffer
    auto large2 = ipc_manager->AllocateBuffer(k500MB);
    REQUIRE_FALSE(large2.IsNull());
    large_buffers.push_back(large2);

    // Verify all small buffers
    for (size_t i = 0; i < small_buffers.size(); ++i) {
      small_buffers[i].ptr_[0] = static_cast<char>(i);
      REQUIRE(small_buffers[i].ptr_[0] == static_cast<char>(i));
    }

    // Verify large buffers
    large_buffers[0].ptr_[0] = 'X';
    large_buffers[0].ptr_[k500MB - 1] = 'Y';
    large_buffers[1].ptr_[0] = 'A';
    large_buffers[1].ptr_[k500MB - 1] = 'B';

    REQUIRE(large_buffers[0].ptr_[0] == 'X');
    REQUIRE(large_buffers[0].ptr_[k500MB - 1] == 'Y');
    REQUIRE(large_buffers[1].ptr_[0] == 'A');
    REQUIRE(large_buffers[1].ptr_[k500MB - 1] == 'B');

    INFO("Mixed allocation pattern test passed");
  }
}

TEST_CASE("Per-process shared memory FreeBuffer",
          "[ipc][per_process_shm][free]") {
  REQUIRE(initialize_chimaera());

  auto* ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Free allocated buffer") {
    // Allocate a buffer
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(k100MB);
    REQUIRE_FALSE(buffer.IsNull());

    // Write some data
    buffer.ptr_[0] = 'T';
    buffer.ptr_[k100MB - 1] = 'E';

    // Free the buffer
    ipc_manager->FreeBuffer(buffer);

    // Note: After freeing, the pointer is no longer valid
    // We just verify that FreeBuffer didn't crash
    INFO("FreeBuffer completed without error");
  }

  SECTION("Allocate-free-allocate cycle") {
    // Allocate
    hipc::FullPtr<char> buffer1 = ipc_manager->AllocateBuffer(k100MB);
    REQUIRE_FALSE(buffer1.IsNull());

    // Free
    ipc_manager->FreeBuffer(buffer1);

    // Allocate again - should work
    hipc::FullPtr<char> buffer2 = ipc_manager->AllocateBuffer(k100MB);
    REQUIRE_FALSE(buffer2.IsNull());

    // Verify new buffer is usable
    buffer2.ptr_[0] = 'R';
    REQUIRE(buffer2.ptr_[0] == 'R');

    INFO("Allocate-free-allocate cycle passed");
  }
}

TEST_CASE("Per-process shared memory ToFullPtr conversion",
          "[ipc][per_process_shm][tofullptr]") {
  REQUIRE(initialize_chimaera());

  auto* ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("ToFullPtr from raw pointer") {
    // Allocate a buffer
    hipc::FullPtr<char> original = ipc_manager->AllocateBuffer(k1MB);
    REQUIRE_FALSE(original.IsNull());

    // Get the raw pointer
    char* raw_ptr = original.ptr_;

    // Convert back to FullPtr
    hipc::FullPtr<char> converted = ipc_manager->ToFullPtr(raw_ptr);

    // Should get the same pointer back
    REQUIRE_FALSE(converted.IsNull());
    REQUIRE(converted.ptr_ == raw_ptr);

    // Write via original, read via converted
    original.ptr_[0] = 'X';
    REQUIRE(converted.ptr_[0] == 'X');

    // Write via converted, read via original
    converted.ptr_[100] = 'Y';
    REQUIRE(original.ptr_[100] == 'Y');

    INFO("ToFullPtr conversion test passed");
  }
}

TEST_CASE("Per-process shared memory stress test",
          "[ipc][per_process_shm][stress]") {
  REQUIRE(initialize_chimaera());

  auto* ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("Many small allocations") {
    const size_t num_allocs = 100;
    const size_t alloc_size = 10 * k1MB;  // 10MB each = 1GB total
    std::vector<hipc::FullPtr<char>> buffers;

    INFO("Allocating " << num_allocs << " buffers of " << (alloc_size / k1MB) << "MB each");

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_allocs; ++i) {
      auto buffer = ipc_manager->AllocateBuffer(alloc_size);
      REQUIRE_FALSE(buffer.IsNull());

      // Touch the buffer to ensure it's really allocated
      buffer.ptr_[0] = static_cast<char>(i & 0xFF);
      buffer.ptr_[alloc_size - 1] = static_cast<char>((i + 1) & 0xFF);

      buffers.push_back(buffer);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    INFO("Allocated " << num_allocs << " buffers in " << duration.count() << " ms");

    // Verify all data is intact
    for (size_t i = 0; i < num_allocs; ++i) {
      REQUIRE(static_cast<unsigned char>(buffers[i].ptr_[0]) == (i & 0xFF));
      REQUIRE(static_cast<unsigned char>(buffers[i].ptr_[alloc_size - 1]) == ((i + 1) & 0xFF));
    }

    INFO("Stress test with " << num_allocs << " allocations passed");
  }
}

TEST_CASE("Per-process shared memory ClientShmInfo",
          "[ipc][per_process_shm][shm_info]") {
  REQUIRE(initialize_chimaera());

  auto* ipc_manager = CHI_IPC;
  REQUIRE(ipc_manager != nullptr);

  SECTION("ClientShmInfo struct creation") {
    // Create a ClientShmInfo manually
    chi::ClientShmInfo info;
    info.shm_name = "test_shm";
    info.owner_pid = getpid();
    info.shm_index = 0;
    info.size = k100MB;
    info.alloc_id = hipc::AllocatorId(static_cast<chi::u32>(getpid()), 0);

    REQUIRE(info.shm_name == "test_shm");
    REQUIRE(info.owner_pid == getpid());
    REQUIRE(info.shm_index == 0);
    REQUIRE(info.size == k100MB);

    INFO("ClientShmInfo struct test passed");
  }
}

// Main function to run all tests
SIMPLE_TEST_MAIN()
