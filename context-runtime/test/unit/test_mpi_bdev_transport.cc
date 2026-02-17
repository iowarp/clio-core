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
 * MPI BDev Transport Integration Test
 *
 * Tests concurrent I/O across multiple transport modes using MPI.
 *
 * Architecture:
 *   Rank 0: Chimaera server
 *   Rank 1: Client with SHM mode
 *   Rank 2: Client with TCP mode
 *   Rank 3: Client with IPC mode
 *
 * Falls back to single TCP client if only 2 ranks available.
 */

#include <mpi.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <hermes_shm/util/logging.h>

#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"

#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>

using namespace chi;

// --- Helpers ---

inline chi::priv::vector<chimaera::bdev::Block> WrapBlock(
    const chimaera::bdev::Block& block) {
  chi::priv::vector<chimaera::bdev::Block> blocks(HSHM_MALLOC);
  blocks.push_back(block);
  return blocks;
}

bool WaitForServer(int max_attempts = 100) {
  const char* user = std::getenv("USER");
  std::string memfd_path = std::string("/tmp/chimaera_memfd/chi_main_segment_") +
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

void CleanupSharedMemory() {
  const char* user = std::getenv("USER");
  std::string memfd_path = std::string("/tmp/chimaera_memfd/chi_main_segment_") +
                           (user ? user : "");
  unlink(memfd_path.c_str());
}

/**
 * Run a BDev I/O cycle: alloc, write, read, verify, free.
 * Returns true on success, false on failure.
 */
bool RunBdevIoTest(int rank, const std::string& mode_name, size_t io_size) {
  const chi::u64 kRamSize = 16 * 1024 * 1024;  // 16MB pool
  const chi::u64 kBlockSize = io_size;

  // Rank-specific pool to avoid conflicts
  chi::PoolId pool_id(9000 + rank, 0);
  chimaera::bdev::Client client(pool_id);
  std::string pool_name = "mpi_bdev_" + mode_name + "_rank" + std::to_string(rank);

  // Create pool
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), pool_name, pool_id,
      chimaera::bdev::BdevType::kRam, kRamSize);
  create_task.Wait();
  if (create_task->return_code_ != 0) {
    HLOG(kError, "[Rank {}] Create pool failed: {}", rank, create_task->return_code_);
    return false;
  }
  client.pool_id_ = create_task->new_pool_id_;

  // Allocate blocks
  auto alloc_task = client.AsyncAllocateBlocks(
      chi::PoolQuery::Local(), kBlockSize);
  alloc_task.Wait();
  if (alloc_task->return_code_ != 0 || alloc_task->blocks_.size() == 0) {
    HLOG(kError, "[Rank {}] AllocateBlocks failed", rank);
    return false;
  }
  chimaera::bdev::Block block = alloc_task->blocks_[0];

  // Generate rank-specific test data
  std::vector<hshm::u8> write_data(io_size);
  for (size_t i = 0; i < io_size; ++i) {
    write_data[i] = static_cast<hshm::u8>((0xAB + rank * 37 + i) % 256);
  }

  // Write
  auto write_buffer = CHI_IPC->AllocateBuffer(write_data.size());
  if (write_buffer.IsNull()) {
    HLOG(kError, "[Rank {}] AllocateBuffer for write failed", rank);
    return false;
  }
  memcpy(write_buffer.ptr_, write_data.data(), write_data.size());
  auto write_task = client.AsyncWrite(
      chi::PoolQuery::Local(), WrapBlock(block),
      write_buffer.shm_.template Cast<void>().template Cast<void>(),
      write_data.size());
  write_task.Wait();
  if (write_task->return_code_ != 0) {
    HLOG(kError, "[Rank {}] Write failed: {}", rank, write_task->return_code_);
    CHI_IPC->FreeBuffer(write_buffer);
    return false;
  }
  size_t actual_written = write_task->bytes_written_;

  // Read
  auto read_buffer = CHI_IPC->AllocateBuffer(io_size);
  if (read_buffer.IsNull()) {
    HLOG(kError, "[Rank {}] AllocateBuffer for read failed", rank);
    CHI_IPC->FreeBuffer(write_buffer);
    return false;
  }
  auto read_task = client.AsyncRead(
      chi::PoolQuery::Local(), WrapBlock(block),
      read_buffer.shm_.template Cast<void>().template Cast<void>(),
      io_size);
  read_task.Wait();
  if (read_task->return_code_ != 0) {
    HLOG(kError, "[Rank {}] Read failed: {}", rank, read_task->return_code_);
    CHI_IPC->FreeBuffer(write_buffer);
    CHI_IPC->FreeBuffer(read_buffer);
    return false;
  }

  // Verify data
  hipc::FullPtr<char> data_ptr =
      CHI_IPC->ToFullPtr(read_task->data_.template Cast<char>());
  if (data_ptr.IsNull()) {
    HLOG(kError, "[Rank {}] Read data pointer is null", rank);
    CHI_IPC->FreeBuffer(write_buffer);
    CHI_IPC->FreeBuffer(read_buffer);
    return false;
  }
  size_t actual_read = read_task->bytes_read_;
  size_t verify_size = std::min(actual_written, actual_read);

  int mismatches = 0;
  for (size_t i = 0; i < verify_size; ++i) {
    if (static_cast<hshm::u8>(data_ptr.ptr_[i]) != write_data[i]) {
      mismatches++;
      if (mismatches <= 3) {
        HLOG(kError, "[Rank {}] Mismatch at byte {}: got {} expected {}", rank, i,
             (int)(hshm::u8)data_ptr.ptr_[i], (int)write_data[i]);
      }
    }
  }

  CHI_IPC->FreeBuffer(write_buffer);
  CHI_IPC->FreeBuffer(read_buffer);

  if (mismatches > 0) {
    HLOG(kError, "[Rank {}] {} mismatches in {} bytes", rank, mismatches, verify_size);
    return false;
  }

  HLOG(kInfo, "[Rank {}] {} I/O test passed ({} bytes)", rank, mode_name, io_size);
  return true;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    if (rank == 0) {
      HLOG(kError, "Need at least 2 MPI ranks (got {})", size);
    }
    MPI_Finalize();
    return 1;
  }

  // Determine transport mode for each client rank
  // 4 ranks: rank0=server, rank1=SHM, rank2=TCP, rank3=IPC
  // 2 ranks: rank0=server, rank1=TCP
  std::string mode_name;
  if (size >= 4) {
    switch (rank) {
      case 1: mode_name = "SHM"; break;
      case 2: mode_name = "TCP"; break;
      case 3: mode_name = "IPC"; break;
      default: break;  // rank 0 is server, ranks > 3 also TCP
    }
    if (rank > 3) mode_name = "TCP";
  } else {
    // Minimal mode: only TCP clients
    if (rank > 0) mode_name = "TCP";
  }

  int local_pass = 1;  // 1 = pass, 0 = fail

  if (rank == 0) {
    // --- Server rank ---
    HLOG(kInfo, "[Rank 0] Starting Chimaera server...");

    // Cleanup stale shared memory
    CleanupSharedMemory();

    setenv("CHIMAERA_WITH_RUNTIME", "1", 1);
    bool success = CHIMAERA_INIT(ChimaeraMode::kServer, true);
    if (!success) {
      HLOG(kError, "[Rank 0] CHIMAERA_INIT(kServer) failed!");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
    HLOG(kInfo, "[Rank 0] Server started.");

    // Signal clients that server is ready
    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for all clients to connect
    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for all I/O to complete
    MPI_Barrier(MPI_COMM_WORLD);

    // Collect results
    int global_pass = 0;
    MPI_Reduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN,
               0, MPI_COMM_WORLD);

    if (global_pass == 1) {
      HLOG(kSuccess, "\n=== ALL MPI BDev TRANSPORT TESTS PASSED ===");
    } else {
      HLOG(kError, "\n=== SOME MPI BDev TRANSPORT TESTS FAILED ===");
    }

    // Cleanup
    CleanupSharedMemory();
    MPI_Finalize();
    return (global_pass == 1) ? 0 : 1;
  } else {
    // --- Client ranks ---

    // Wait for server to be ready
    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for server to fully start
    bool server_ready = WaitForServer();
    if (!server_ready) {
      HLOG(kError, "[Rank {}] Server not ready!", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }

    // Set transport mode
    setenv("CHI_IPC_MODE", mode_name.c_str(), 1);
    setenv("CHIMAERA_WITH_RUNTIME", "0", 1);

    HLOG(kInfo, "[Rank {}] Connecting as {} client...", rank, mode_name);

    bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
    if (!success) {
      HLOG(kError, "[Rank {}] CHIMAERA_INIT(kClient) failed!", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
    HLOG(kInfo, "[Rank {}] Connected as {}.", rank, mode_name);

    // Signal all clients connected
    MPI_Barrier(MPI_COMM_WORLD);

    // Run I/O tests
    bool pass_4k = RunBdevIoTest(rank, mode_name, 4096);
    bool pass_1m = RunBdevIoTest(rank, mode_name, 1024 * 1024);

    local_pass = (pass_4k && pass_1m) ? 1 : 0;
    if (!pass_4k) HLOG(kError, "[Rank {}] 4KB I/O FAILED", rank);
    if (!pass_1m) HLOG(kError, "[Rank {}] 1MB I/O FAILED", rank);

    // Signal I/O complete
    MPI_Barrier(MPI_COMM_WORLD);

    // Reduce results to rank 0
    int global_pass = 0;
    MPI_Reduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN,
               0, MPI_COMM_WORLD);

    MPI_Finalize();
    return (local_pass == 1) ? 0 : 1;
  }
}
