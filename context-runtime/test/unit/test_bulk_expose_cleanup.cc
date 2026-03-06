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
 * Regression test: BULK_EXPOSE buffer allocated by the server during
 * ReadTask deserialization must be freed when the task is deleted.
 *
 * Before the fix, SendOut() cleared TASK_DATA_OWNER before deferred
 * deletion, so ~ReadTask() never freed the AllocateBuffer'd data.
 * Each AsyncRead leaked the server-side BULK_EXPOSE buffer.
 *
 * Strategy
 * --------
 * Uses TASK_FORCE_NET to force the network serialization path even in
 * embedded mode. This ensures the task goes through:
 *   Send -> ZMQ -> RecvIn (sets TASK_DATA_OWNER, BULK_EXPOSE alloc)
 *   -> Run -> SendOut (deferred delete with destructor freeing buffer)
 *
 * Because TASK_FORCE_NET with embedded mode creates a FutureShm with
 * origin=SHM (rather than TCP), we must patch the origin to TCP so
 * ~Future properly calls CleanupResponseArchive for ZMQ recv buffers.
 *
 * Measures glibc heap growth via mallinfo() to detect the leak.
 */

#include "../simple_test.h"
#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/bdev/bdev_client.h"
#include "chimaera/bdev/bdev_tasks.h"

#include <malloc.h>
#include <cstring>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

static constexpr chi::PoolId kTestPoolId = chi::PoolId(888, 0);
static constexpr chi::u64 kRamSize = 16 * 1024 * 1024;  // 16MB
static constexpr chi::u64 kBlockSize = 4096;             // 4KB
static constexpr chi::u64 kIoSize = 64 * 1024;           // 64KB read size
static constexpr int kBatchSize = 200;
static constexpr long kMaxBytesPerRead = 512;  // generous ceiling

static bool g_initialized = false;

/** Wrap a single block into the vector form ReadTask expects. */
static inline chi::priv::vector<chimaera::bdev::Block> WrapBlock(
    const chimaera::bdev::Block &block) {
  chi::priv::vector<chimaera::bdev::Block> blocks(HSHM_MALLOC);
  blocks.push_back(block);
  return blocks;
}

/** Returns mallinfo.uordblks - heap bytes currently in active use. */
static long heap_in_use() {
  struct mallinfo mi = mallinfo();
  return (long)(unsigned int)mi.uordblks;
}

/** Send a ReadTask with TASK_FORCE_NET so it goes through the network
 *  serialization path (RecvIn -> BULK_EXPOSE alloc -> SendOut).
 *  Patches origin to TCP so ~Future calls CleanupResponseArchive. */
static chi::Future<chimaera::bdev::ReadTask> SendForceNetRead(
    chimaera::bdev::Client &client,
    const chimaera::bdev::Block &block,
    hipc::ShmPtr<> data,
    size_t size) {
  auto *ipc = CHI_IPC;
  auto task_ptr = ipc->NewTask<chimaera::bdev::ReadTask>(
      chi::CreateTaskId(), client.pool_id_, chi::PoolQuery::Local(),
      WrapBlock(block), data, size);
  task_ptr->SetFlags(TASK_FORCE_NET);
  auto future = ipc->Send(task_ptr);
  // Patch origin from SHM to TCP so ~Future calls CleanupResponseArchive
  // for the ZMQ recv buffers created by TASK_FORCE_NET loopback.
  auto fs = future.GetFutureShm();
  if (!fs.IsNull()) {
    fs->origin_ = chi::FutureShm::FUTURE_CLIENT_TCP;
  }
  return future;
}

class BulkExposeFixture {
 public:
  BulkExposeFixture() {
    if (!g_initialized) {
      bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (ok) {
        g_initialized = true;
        SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
        std::this_thread::sleep_for(500ms);
      }
    }
  }
};

TEST_CASE("BULK_EXPOSE buffer freed after ReadTask (TASK_FORCE_NET)",
          "[bdev][memory][regression]") {
  BulkExposeFixture fixture;
  REQUIRE(g_initialized);

  HLOG(kInfo, "Creating RAM bdev pool...");

  // Create RAM bdev pool
  chimaera::bdev::Client client(kTestPoolId);
  auto create_task = client.AsyncCreate(
      chi::PoolQuery::Dynamic(), "bulk_expose_test", kTestPoolId,
      chimaera::bdev::BdevType::kRam, kRamSize);
  create_task.Wait();
  HLOG(kInfo, "create_task done, return_code={}", create_task->return_code_);
  REQUIRE(create_task->return_code_ == 0);
  client.pool_id_ = create_task->new_pool_id_;

  // Allocate a block
  auto alloc_task = client.AsyncAllocateBlocks(
      chi::PoolQuery::Local(), kBlockSize);
  alloc_task.Wait();
  REQUIRE(alloc_task->return_code_ == 0);
  REQUIRE(alloc_task->blocks_.size() > 0);
  chimaera::bdev::Block block = alloc_task->blocks_[0];

  // Write test data so reads return real data (use TASK_FORCE_NET too)
  {
    auto write_buffer = CHI_IPC->AllocateBuffer(kIoSize);
    REQUIRE_FALSE(write_buffer.IsNull());
    memset(write_buffer.ptr_, 0xAB, kIoSize);
    auto *ipc = CHI_IPC;
    auto write_task_ptr = ipc->NewTask<chimaera::bdev::WriteTask>(
        chi::CreateTaskId(), client.pool_id_, chi::PoolQuery::Local(),
        WrapBlock(block), write_buffer.shm_.template Cast<void>(), kIoSize);
    write_task_ptr->SetFlags(TASK_FORCE_NET);
    auto write_task = ipc->Send(write_task_ptr);
    write_task.Wait();
    REQUIRE(write_task->return_code_ == 0);
    CHI_IPC->FreeBuffer(write_buffer);
  }

  HLOG(kInfo, "Warm-up reads...");

  // Warm-up: let allocator reach steady state
  for (int i = 0; i < 20; ++i) {
    auto read_buffer = CHI_IPC->AllocateBuffer(kIoSize);
    REQUIRE_FALSE(read_buffer.IsNull());
    auto read_task = SendForceNetRead(
        client, block, read_buffer.shm_.template Cast<void>(), kIoSize);
    read_task.Wait();
    CHI_IPC->FreeBuffer(read_buffer);
  }
  malloc_trim(0);

  HLOG(kInfo, "Starting heap measurement...");

  SECTION("Heap growth per read is below threshold") {
    long before = heap_in_use();
    INFO("Heap in-use before batch: " << before << " bytes");

    for (int i = 0; i < kBatchSize; ++i) {
      auto read_buffer = CHI_IPC->AllocateBuffer(kIoSize);
      REQUIRE_FALSE(read_buffer.IsNull());
      auto read_task = SendForceNetRead(
          client, block, read_buffer.shm_.template Cast<void>(), kIoSize);
      read_task.Wait();
      REQUIRE(read_task->return_code_ == 0);
      CHI_IPC->FreeBuffer(read_buffer);
    }

    malloc_trim(0);
    long after = heap_in_use();
    INFO("Heap in-use after batch:  " << after << " bytes");

    long growth = after - before;
    long per_read = growth / kBatchSize;
    INFO("Total heap growth: " << growth << " bytes  (" << per_read
                               << " bytes/read)");
    REQUIRE(per_read < kMaxBytesPerRead);
  }

  SECTION("Heap stable across two read batches (no monotonic leak)") {
    for (int i = 0; i < kBatchSize; ++i) {
      auto read_buffer = CHI_IPC->AllocateBuffer(kIoSize);
      auto read_task = SendForceNetRead(
          client, block, read_buffer.shm_.template Cast<void>(), kIoSize);
      read_task.Wait();
      CHI_IPC->FreeBuffer(read_buffer);
    }
    malloc_trim(0);
    long after_a = heap_in_use();
    INFO("Heap after batch A: " << after_a << " bytes");

    for (int i = 0; i < kBatchSize; ++i) {
      auto read_buffer = CHI_IPC->AllocateBuffer(kIoSize);
      auto read_task = SendForceNetRead(
          client, block, read_buffer.shm_.template Cast<void>(), kIoSize);
      read_task.Wait();
      CHI_IPC->FreeBuffer(read_buffer);
    }
    malloc_trim(0);
    long after_b = heap_in_use();
    INFO("Heap after batch B: " << after_b << " bytes");

    long delta = after_b - after_a;
    long per_read = delta / kBatchSize;
    INFO("Heap delta A->B: " << delta << " bytes  (" << per_read
                             << " bytes/read)");
    REQUIRE(per_read < kMaxBytesPerRead);
  }
}

SIMPLE_TEST_MAIN()
