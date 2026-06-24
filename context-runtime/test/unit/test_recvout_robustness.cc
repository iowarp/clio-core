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
 * RecvOut stale-replica robustness tests (issue #628)
 *
 * Exercises the defensive "skip the bad item, keep the batch" behavior added
 * to the inbound output path:
 *   - IpcManagerRun2Run::RecvOut / RecvOutDeserialize must tolerate a replica
 *     response whose origin task is no longer in send_map_ (a stale/duplicate
 *     straggler) instead of aborting the whole message.
 *   - Worker::EndTask must guard against a null RunContext rather than
 *     dereferencing it and crashing.
 */

#include "../simple_test.h"

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/ipc/ipc_run2run.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/task_archives.h>
#include <clio_runtime/types.h>
#include <clio_runtime/worker.h>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>

using namespace chi;

// ============================================================================
// Global Setup - Initialize once for all tests
// ============================================================================
static bool InitializeRuntime() {
  static bool initialized = false;
  if (!initialized) {
    bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
    initialized = success;
    if (success) SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
  }
  return initialized;
}

// Build a kSerializeOut message carrying `count` replica responses. Each
// carries a distinct net_key so RecvOut treats them as independent items.
static void AppendReplicaResponses(chi::LoadTaskArchive &archive, chi::u32 count,
                                   size_t base_net_key) {
  for (chi::u32 i = 0; i < count; ++i) {
    chi::TaskInfo info{};
    info.task_id_ = chi::TaskId(/*pid=*/1, /*tid=*/1, /*major=*/1,
                                /*replica_id=*/i, /*unique=*/0,
                                /*node_id=*/0,
                                /*net_key=*/base_net_key + i);
    info.pool_id_ = chi::kAdminPoolId;
    info.method_id_ = 0;
    archive.task_infos_.push_back(info);
  }
}

// ============================================================================
// RecvOut: stale / duplicate replica responses
// ============================================================================

TEST_CASE("RecvOutRobustness - stale replica responses are skipped, not aborted",
          "[ipc][recvout][robustness]") {
  REQUIRE(InitializeRuntime());

  // A fresh manager has an empty send_map_, which models the case where every
  // origin task has already completed and been erased (a batch of late
  // stragglers / duplicate 3rd-replica responses, or responses racing
  // dead-node recovery). Before the fix the first such item returned an error
  // code and aborted deserialization for the whole message; now each is
  // skipped and RecvOut still reports success.
  clio::run::IpcManagerRun2Run mgr;
  REQUIRE(mgr.GetSendMapSize() == 0);

  chi::LoadTaskArchive archive;
  archive.msg_type_ = chi::MsgType::kSerializeOut;
  AppendReplicaResponses(archive, /*count=*/3, /*base_net_key=*/0xBEEF0000u);

  int rc = -1;
  REQUIRE_NOTHROW(rc = mgr.RecvOut(archive, /*lbm_transport=*/nullptr));
  REQUIRE(rc == 0);

  // Nothing was completed or inserted; the stale responses were simply dropped.
  REQUIRE(mgr.GetSendMapSize() == 0);
}

TEST_CASE("RecvOutRobustness - empty output message is a no-op",
          "[ipc][recvout][robustness]") {
  REQUIRE(InitializeRuntime());

  clio::run::IpcManagerRun2Run mgr;
  chi::LoadTaskArchive archive;
  archive.msg_type_ = chi::MsgType::kSerializeOut;

  int rc = -1;
  REQUIRE_NOTHROW(rc = mgr.RecvOut(archive, /*lbm_transport=*/nullptr));
  REQUIRE(rc == 0);
  REQUIRE(mgr.GetSendMapSize() == 0);
}

// ============================================================================
// Worker::EndTask: null RunContext guard
// ============================================================================

TEST_CASE("RecvOutRobustness - EndTask tolerates a null RunContext",
          "[worker][recvout][robustness]") {
  REQUIRE(InitializeRuntime());

  auto *ipc_manager = CLIO_IPC;
  REQUIRE(ipc_manager != nullptr);

  // A straggler / recovery response whose origin was already torn down can
  // reach EndTask with a null RunContext. The worker must log and bail rather
  // than dereferencing it and SIGSEGVing.
  auto task = ipc_manager->NewTask<clio::run::admin::CreateTask>(
      chi::TaskId(1, 1, 1, 0, 1), chi::kAdminPoolId, chi::PoolQuery::Local(),
      "robustness_mod", "robustness_pool", chi::PoolId(9000, 0),
      nullptr);
  REQUIRE(!task.IsNull());

  // A bare (non-Init'd) worker is enough: EndTask returns at the null-ctx guard
  // before touching any worker state.
  clio::run::Worker worker(/*worker_id=*/65000);
  ctp::ipc::FullPtr<chi::Task> task_ptr = task.template Cast<chi::Task>();
  REQUIRE_NOTHROW(worker.EndTask(task_ptr, /*run_ctx=*/nullptr,
                                 /*can_resched=*/false));
}

// ============================================================================
// Global Cleanup - Finalize once at the end
// ============================================================================

TEST_CASE("RecvOutRobustness - ZZZ Final Cleanup", "[recvout][cleanup]") {
  // Runs last (ZZZ prefix). Force exit to avoid hanging on worker-thread joins
  // / ZMQ teardown during finalization, mirroring the other runtime tests.
  SIMPLE_TEST_PROCESS_EXIT(0);
}

SIMPLE_TEST_MAIN()
