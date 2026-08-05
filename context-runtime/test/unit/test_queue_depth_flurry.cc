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
 * issue #822 regression test: a deliberately tiny queue_depth (16) plus a
 * flurry of far more concurrent tasks than the depth. With the old
 * WAIT_FOR_SPACE MPSC worker lanes this configuration deadlocks (a full lane
 * busy-spins its producer forever; when the producer is the lane's own
 * consumer nothing can ever drain it). With the ext_spsc_queue lanes a full
 * lane grows instead, so every push completes and the flurry must finish.
 * A hang here is caught by the CTest TIMEOUT.
 */

#include "../simple_test.h"
#include "../runtime_server.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc_manager.h"

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>

using namespace clio::run;

namespace {

inline clio::run::priv::vector<clio::run::bdev::Block> WrapBlock(
    const clio::run::bdev::Block &block) {
  clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
  blocks.push_back(block);
  return blocks;
}

/** Write a runtime config with a tiny queue depth and return its path. */
std::string WriteTinyDepthConfig() {
  std::error_code ec;
  std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
  if (ec) dir = ".";
  std::filesystem::path path = dir / "clio_flurry_depth16.yaml";
  FILE *f = fopen(path.string().c_str(), "w");
  REQUIRE(f != nullptr);
  fputs("runtime:\n  queue_depth: 16\n", f);
  fclose(f);
  return path.string();
}

}  // namespace

TEST_CASE("QueueDepthFlurry - depth 16 survives a 4096-task flurry",
          "[queue_depth][flurry]") {
  // Point the daemon (spawned child inherits env) and this client at a
  // config whose queue_depth (16) is far below the task flurry size.
  const std::string conf = WriteTinyDepthConfig();
  clio::run::test::SetEnvVar("CLIO_SERVER_CONF", conf);

  // Dedicated port: segment names are port-keyed, so this test can never
  // collide with another daemon (or another test) on the default 10500.
  clio::run::test::RuntimeServer server;
  REQUIRE(server.Start(10622));
  REQUIRE(server.WaitForReady());

  // SHM client: tasks flow through the shared-memory worker lanes, the
  // topology where the old producer==consumer WAIT_FOR_SPACE hang lived.
  clio::run::test::SetEnvVar("CLIO_IPC_MODE", "SHM");
  clio::run::test::SetEnvVar("CLIO_WITH_RUNTIME", "0");
  REQUIRE(CLIO_INIT(RuntimeMode::kClient, false));
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());

  // A RAM bdev pool to aim the flurry at.
  const clio::run::u64 kRamSize = 64 * 1024 * 1024;
  const clio::run::u64 kBlockSize = 4096;
  clio::run::PoolId pool_id(9822, 0);
  clio::run::bdev::Client client(pool_id);
  auto create_task = client.AsyncCreate(
      clio::run::PoolQuery::Dynamic(), "flurry_ram_822", pool_id,
      clio::run::bdev::BdevType::kRam, kRamSize);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  client.pool_id_ = create_task->new_pool_id_;

  auto alloc_task =
      client.AsyncAllocateBlocks(clio::run::PoolQuery::Local(), kBlockSize);
  alloc_task.Wait();
  REQUIRE(alloc_task->return_code_ == 0);
  REQUIRE(alloc_task->blocks_.size() > 0);
  clio::run::bdev::Block block = alloc_task->blocks_[0];

  // One shared source buffer; write tasks only read it.
  auto write_buffer = CLIO_IPC->AllocateBuffer(kBlockSize);
  REQUIRE_FALSE(write_buffer.IsNull());
  memset(write_buffer.ptr_, 0x22, kBlockSize);

  // The flurry: issue every task asynchronously BEFORE waiting on any of
  // them, so in-flight tasks exceed queue_depth by ~256x. Under the old
  // lanes this wedges; under growable lanes it must complete.
  constexpr int kFlurry = 4096;
  std::vector<clio::run::Future<clio::run::bdev::WriteTask>> futures;
  futures.reserve(kFlurry);
  for (int i = 0; i < kFlurry; ++i) {
    futures.emplace_back(client.AsyncWrite(
        clio::run::PoolQuery::Local(), WrapBlock(block),
        write_buffer.shm_.template Cast<void>(), kBlockSize));
  }

  for (int i = 0; i < kFlurry; ++i) {
    futures[i].Wait();
    REQUIRE(futures[i]->return_code_ == 0);
  }

  CLIO_IPC->FreeBuffer(write_buffer);
  clio::run::test::UnsetEnvVar("CLIO_SERVER_CONF");
}

SIMPLE_TEST_MAIN()
