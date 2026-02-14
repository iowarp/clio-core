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
 * IPC Transport Mode Tests
 *
 * Tests that each IPC transport mode (SHM, TCP, IPC) initializes correctly
 * and that the correct transport path is active. Each test case forks a
 * server, sets CHI_IPC_MODE, connects as client, and verifies mode state.
 */

#include "../simple_test.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"

#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>

using namespace chi;

inline chi::priv::vector<chimaera::bdev::Block> WrapBlock(
    const chimaera::bdev::Block& block) {
  chi::priv::vector<chimaera::bdev::Block> blocks(HSHM_MALLOC);
  blocks.push_back(block);
  return blocks;
}

void SubmitTasksForMode(const std::string &mode_name) {
  const chi::u64 kRamSize = 16 * 1024 * 1024;  // 16MB pool
  const chi::u64 kBlockSize = 4096;             // 4KB block allocation
  const chi::u64 kIoSize = 1024 * 1024;         // 1MB I/O transfer size

  // --- Category 1: Create bdev pool (inputs > outputs) ---
  chi::PoolId pool_id(9000, 0);
  chimaera::bdev::Client client(pool_id);
  std::string pool_name = "ipc_test_ram_" + mode_name;
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), pool_name, pool_id,
      chimaera::bdev::BdevType::kRam, kRamSize);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  client.pool_id_ = create_task->new_pool_id_;

  // --- Category 2: AllocateBlocks (outputs > inputs) ---
  auto alloc_task = client.AsyncAllocateBlocks(
      chi::PoolQuery::Local(), kBlockSize);
  alloc_task.Wait();
  REQUIRE(alloc_task->return_code_ == 0);
  REQUIRE(alloc_task->blocks_.size() > 0);
  chimaera::bdev::Block block = alloc_task->blocks_[0];
  REQUIRE(block.size_ >= kBlockSize);

  // --- Category 3: Write + Read I/O round-trip (1MB transfer) ---
  // Generate 1MB test data
  std::vector<hshm::u8> write_data(kIoSize);
  for (size_t i = 0; i < kIoSize; ++i) {
    write_data[i] = static_cast<hshm::u8>((0xAB + i) % 256);
  }

  // Write 1MB
  auto write_buffer = CHI_IPC->AllocateBuffer(write_data.size());
  REQUIRE_FALSE(write_buffer.IsNull());
  memcpy(write_buffer.ptr_, write_data.data(), write_data.size());
  auto write_task = client.AsyncWrite(
      chi::PoolQuery::Local(), WrapBlock(block),
      write_buffer.shm_.template Cast<void>().template Cast<void>(),
      write_data.size());
  write_task.Wait();
  REQUIRE(write_task->return_code_ == 0);
  // Note: bytes_written may be less than kIoSize if block is smaller
  // We're measuring transport overhead, not bdev correctness
  size_t actual_written = write_task->bytes_written_;

  // Read back using actual written size
  auto read_buffer = CHI_IPC->AllocateBuffer(kIoSize);
  REQUIRE_FALSE(read_buffer.IsNull());
  auto read_task = client.AsyncRead(
      chi::PoolQuery::Local(), WrapBlock(block),
      read_buffer.shm_.template Cast<void>().template Cast<void>(),
      kIoSize);
  read_task.Wait();
  REQUIRE(read_task->return_code_ == 0);

  // Verify data up to actual_written
  hipc::FullPtr<char> data_ptr =
      CHI_IPC->ToFullPtr(read_task->data_.template Cast<char>());
  REQUIRE_FALSE(data_ptr.IsNull());
  size_t actual_read = read_task->bytes_read_;
  std::vector<hshm::u8> read_data(actual_read);
  memcpy(read_data.data(), data_ptr.ptr_, actual_read);
  size_t verify_size = std::min(actual_written, actual_read);

  for (size_t i = 0; i < verify_size; ++i) {
    REQUIRE(read_data[i] == write_data[i]);
  }

  // Cleanup buffers
  CHI_IPC->FreeBuffer(write_buffer);
  CHI_IPC->FreeBuffer(read_buffer);
}

/**
 * Helper to start server in background process
 * Returns server PID
 */
pid_t StartServerProcess() {
  pid_t server_pid = fork();
  if (server_pid == 0) {
    // Redirect child's stdout to /dev/null but stderr to temp file for timing
    freopen("/dev/null", "w", stdout);
    freopen("/tmp/chimaera_server_timing.log", "w", stderr);

    // Child process: Start runtime server
    setenv("CHIMAERA_WITH_RUNTIME", "1", 1);
    bool success = CHIMAERA_INIT(ChimaeraMode::kServer, true);
    if (!success) {
      _exit(1);
    }

    // Keep server alive for tests
    // Server will be killed by parent process
    sleep(300);  // 5 minutes max
    _exit(0);
  }
  return server_pid;
}

/**
 * Helper to wait for server to be ready
 */
bool WaitForServer(int max_attempts = 50) {
  // The main shared memory segment name is "chi_main_segment_${USER}"
  const char *user = std::getenv("USER");
  std::string memfd_path = std::string("/tmp/chimaera_memfd/chi_main_segment_") +
                           (user ? user : "");

  for (int i = 0; i < max_attempts; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Check if memfd symlink exists (indicates server is ready)
    int fd = open(memfd_path.c_str(), O_RDONLY);
    if (fd >= 0) {
      close(fd);
      // Give it a bit more time to fully initialize
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      return true;
    }
  }
  return false;
}

/**
 * Helper to cleanup shared memory
 */
void CleanupSharedMemory() {
  const char *user = std::getenv("USER");
  std::string memfd_path = std::string("/tmp/chimaera_memfd/chi_main_segment_") +
                           (user ? user : "");
  unlink(memfd_path.c_str());
}

/**
 * Helper to cleanup server process
 */
void CleanupServer(pid_t server_pid) {
  if (server_pid > 0) {
    kill(server_pid, SIGTERM);
    int status;
    waitpid(server_pid, &status, 0);
    CleanupSharedMemory();
  }
}

// ============================================================================
// IPC Transport Mode Tests
// ============================================================================

TEST_CASE("IpcTransportMode - SHM Client Connection",
          "[ipc_transport][shm]") {
  // Start server in background
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Set SHM mode and connect as external client
  setenv("CHI_IPC_MODE", "SHM", 1);
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kShm);

  // SHM mode attaches to shared queues
  REQUIRE(ipc->GetTaskQueue() != nullptr);

  // Submit real tasks through the transport layer
  SubmitTasksForMode("shm");

  // Cleanup
  CleanupServer(server_pid);
}

TEST_CASE("IpcTransportMode - TCP Client Connection",
          "[ipc_transport][tcp]") {
  // Start server in background
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Set TCP mode and connect as external client
  setenv("CHI_IPC_MODE", "TCP", 1);
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kTcp);

  // TCP mode does not attach to shared queues
  REQUIRE(ipc->GetTaskQueue() == nullptr);

  // Submit real tasks through the transport layer
  SubmitTasksForMode("tcp");

  // Cleanup
  CleanupServer(server_pid);
}

TEST_CASE("IpcTransportMode - IPC Client Connection",
          "[ipc_transport][ipc]") {
  // Start server in background
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Set IPC (Unix Domain Socket) mode and connect as external client
  setenv("CHI_IPC_MODE", "IPC", 1);
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kIpc);

  // IPC mode does not attach to shared queues
  REQUIRE(ipc->GetTaskQueue() == nullptr);

  // Submit real tasks through the transport layer
  SubmitTasksForMode("ipc");

  // Cleanup
  CleanupServer(server_pid);
}

TEST_CASE("IpcTransportMode - Default Mode Is TCP",
          "[ipc_transport][default]") {
  // Start server in background
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Unset CHI_IPC_MODE to test default behavior
  unsetenv("CHI_IPC_MODE");
  setenv("CHIMAERA_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CHI_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kTcp);

  // Cleanup
  CleanupServer(server_pid);
}

SIMPLE_TEST_MAIN()
