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
 * and that the correct transport path is active. Each test case launches the
 * runtime daemon out-of-process (clio::run::test::RuntimeServer -> clio_run start),
 * sets CLIO_IPC_MODE, connects as client, and verifies mode state.
 */

#include "../simple_test.h"
#include "../runtime_server.h"

#include <cstdlib>
#include <string>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc_manager.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>

using namespace clio::run;

inline clio::run::priv::vector<clio::run::bdev::Block> WrapBlock(
    const clio::run::bdev::Block& block) {
  clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
  blocks.push_back(block);
  return blocks;
}

void SubmitTasksForMode(const std::string &mode_name) {
  const clio::run::u64 kRamSize = 16 * 1024 * 1024;  // 16MB pool
  const clio::run::u64 kBlockSize = 4096;             // 4KB block allocation
  const clio::run::u64 kIoSize = 1024 * 1024;         // 1MB I/O transfer size

  // --- Category 1: Create bdev pool (inputs > outputs) ---
  clio::run::PoolId pool_id(9000, 0);
  clio::run::bdev::Client client(pool_id);
  std::string pool_name = "ipc_test_ram_" + mode_name;
  auto create_task = client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), pool_name, pool_id,
      clio::run::bdev::BdevType::kRam, kRamSize);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  client.pool_id_ = create_task->new_pool_id_;

  // --- Category 2: AllocateBlocks (outputs > inputs) ---
  auto alloc_task = client.AsyncAllocateBlocks(
      clio::run::PoolQuery::Local(), kBlockSize);
  alloc_task.Wait();
  REQUIRE(alloc_task->return_code_ == 0);
  REQUIRE(alloc_task->blocks_.size() > 0);
  clio::run::bdev::Block block = alloc_task->blocks_[0];
  REQUIRE(block.size_ >= kBlockSize);

  // --- Category 3: Write + Read I/O round-trip (1MB transfer) ---
  // Generate 1MB test data
  std::vector<ctp::u8> write_data(kIoSize);
  for (size_t i = 0; i < kIoSize; ++i) {
    write_data[i] = static_cast<ctp::u8>((0xAB + i) % 256);
  }

  // Write 1MB
  auto write_buffer = CLIO_IPC->AllocateBuffer(write_data.size());
  REQUIRE_FALSE(write_buffer.IsNull());
  memcpy(write_buffer.ptr_, write_data.data(), write_data.size());
  auto write_task = client.AsyncWrite(
      clio::run::PoolQuery::Local(), WrapBlock(block),
      write_buffer.shm_.template Cast<void>().template Cast<void>(),
      write_data.size());
  write_task.Wait();
  REQUIRE(write_task->return_code_ == 0);
  // Note: bytes_written may be less than kIoSize if block is smaller
  // We're measuring transport overhead, not bdev correctness
  size_t actual_written = write_task->bytes_written_;

  // Read back using actual written size
  auto read_buffer = CLIO_IPC->AllocateBuffer(kIoSize);
  REQUIRE_FALSE(read_buffer.IsNull());
  auto read_task = client.AsyncRead(
      clio::run::PoolQuery::Local(), WrapBlock(block),
      read_buffer.shm_.template Cast<void>().template Cast<void>(),
      kIoSize);
  read_task.Wait();
  REQUIRE(read_task->return_code_ == 0);

  // Verify data up to actual_written
  ctp::ipc::FullPtr<char> data_ptr =
      CLIO_IPC->ToFullPtr(read_task->data_.template Cast<char>());
  REQUIRE_FALSE(data_ptr.IsNull());
  size_t actual_read = read_task->bytes_read_;
  std::vector<ctp::u8> read_data(actual_read);
  memcpy(read_data.data(), data_ptr.ptr_, actual_read);
  size_t verify_size = std::min(actual_written, actual_read);

  for (size_t i = 0; i < verify_size; ++i) {
    REQUIRE(read_data[i] == write_data[i]);
  }

  // Cleanup buffers
  CLIO_IPC->FreeBuffer(write_buffer);
  CLIO_IPC->FreeBuffer(read_buffer);
}

// The runtime server is launched out-of-process via clio::run::test::RuntimeServer
// (clio_run start), which is portable across Linux, macOS and Windows. The old
// fork()+CLIO_INIT(kServer) helpers were removed: fork-without-exec is
// unsupported on macOS (post-fork dlopen of ChiMods deadlocks) and impossible
// on Windows. See context-runtime/test/runtime_server.h.

// ============================================================================
// IPC Transport Mode Tests
// ============================================================================

TEST_CASE("IpcTransportMode - SHM Client Connection",
          "[ipc_transport][shm]") {
  // Start the runtime daemon out-of-process
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  // Set SHM mode and connect as external client
  clio::run::test::SetEnvVar("CLIO_IPC_MODE", "SHM");
  clio::run::test::SetEnvVar("CLIO_WITH_RUNTIME", "0");
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kShm);

  // SHM mode attaches to shared queues
  REQUIRE(ipc->GetTaskQueue() != nullptr);

  // Submit real tasks through the transport layer
  SubmitTasksForMode("shm");
}

TEST_CASE("IpcTransportMode - TCP Client Connection",
          "[ipc_transport][tcp]") {
  // Start the runtime daemon out-of-process
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  // Set TCP mode and connect as external client
  clio::run::test::SetEnvVar("CLIO_IPC_MODE", "TCP");
  clio::run::test::SetEnvVar("CLIO_WITH_RUNTIME", "0");
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kTcp);

  // TCP mode does not attach to shared queues
  REQUIRE(ipc->GetTaskQueue() == nullptr);

  // Submit real tasks through the transport layer
  SubmitTasksForMode("tcp");
}

TEST_CASE("IpcTransportMode - IPC Client Connection",
          "[ipc_transport][ipc]") {
  // Start the runtime daemon out-of-process
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  // Set IPC (Unix Domain Socket) mode and connect as external client
  clio::run::test::SetEnvVar("CLIO_IPC_MODE", "IPC");
  clio::run::test::SetEnvVar("CLIO_WITH_RUNTIME", "0");
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kIpc);

  // IPC mode does not attach to shared queues
  REQUIRE(ipc->GetTaskQueue() == nullptr);

  // Submit real tasks through the transport layer
  SubmitTasksForMode("ipc");
}

TEST_CASE("IpcTransportMode - Default Mode Is TCP",
          "[ipc_transport][default]") {
  // Start the runtime daemon out-of-process
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start());
  REQUIRE(server.WaitForReady());

  // Unset CLIO_IPC_MODE to test default behavior
  clio::run::test::UnsetEnvVar("CLIO_IPC_MODE");
  clio::run::test::SetEnvVar("CLIO_WITH_RUNTIME", "0");
  bool success = CLIO_INIT(RuntimeMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());
  REQUIRE(ipc->GetIpcMode() == IpcMode::kTcp);
}

SIMPLE_TEST_MAIN()
