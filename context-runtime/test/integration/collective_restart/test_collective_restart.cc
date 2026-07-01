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
 * Reproducer: a cross-node collective must not hang forever when a
 * participant node is restarted mid-flight.
 *
 * Orchestrated by run_tests.sh:
 *   1. Start a 4-node cluster (SWIM failure detection deliberately SLOW so a
 *      fast `docker restart` brings the node back BEFORE it is ever declared
 *      dead -- this isolates the "live node, dropped task" case that dead-node
 *      timeouts cannot catch, issue #628).
 *   2. [setup] Create a MOD_NAME pool with one container per node.
 *   3. [hang]  Broadcast a long-hold CoMutexTest to all 4 nodes and Wait().
 *      While the replicas are holding, run_tests.sh restarts node 4. Node 4
 *      rejoins alive but its in-memory replica task is gone, so its response
 *      never comes back.
 *
 * Expected:
 *   - Pre-fix:  the origin waits on the missing replica forever -> the [hang]
 *     case never returns and run_tests.sh reports a reproduced hang.
 *   - Post-fix: the periodic QueryTaskProgress validity check sees node 4's
 *     replica is Gone and completes the origin (partial + timeout RC), so the
 *     broadcast returns.
 *
 * The pool is created in a separate [setup] invocation and reused by [hang]
 * (fixed pool id) so pool creation finishes before the restart window opens.
 */

#include "simple_test.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>

using namespace std::chrono_literals;

namespace {
bool g_initialized = false;

// Fixed pool id shared between the [setup] and [hang] invocations.
const clio::run::PoolId kReproPoolId(50100, 0);

// Long hold so run_tests.sh has a wide window to restart a node while every
// replica is still executing. Overridable via REPRO_HOLD_MS.
clio::run::u32 HoldMs() {
  const char *env = std::getenv("REPRO_HOLD_MS");
  return env ? static_cast<clio::run::u32>(std::atoi(env)) : 15000u;
}

class ReproFixture {
 public:
  ReproFixture() {
    if (!g_initialized) {
      bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      REQUIRE(ok);
      g_initialized = true;
      SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
      std::this_thread::sleep_for(500ms);
      REQUIRE(CLIO_IPC != nullptr);
      REQUIRE(CLIO_IPC->IsInitialized());
    }
  }
};
}  // namespace

// Phase 1: create the pool (containers on all 4 nodes). Runs to completion
// before any node is restarted.
TEST_CASE("Collective restart: create pool", "[collective_restart][setup]") {
  ReproFixture fixture;
  clio::run::MOD_NAME::Client client(kReproPoolId);
  auto create_task = client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                        "collective_restart_pool", kReproPoolId);
  create_task.Wait();
  REQUIRE(create_task->return_code_ == 0);
  std::cout << "[REPRO] pool created" << std::endl;
}

// Phase 2: broadcast a long-hold task to all nodes. run_tests.sh restarts a
// participant during the hold. Pre-fix this Wait() never returns.
TEST_CASE("Collective survives a mid-flight node restart",
          "[collective_restart][hang]") {
  ReproFixture fixture;
  clio::run::MOD_NAME::Client client(kReproPoolId);
  client.pool_id_ = kReproPoolId;

  const clio::run::u32 hold_ms = HoldMs();
  auto task = client.AsyncCoMutexTest(clio::run::PoolQuery::Broadcast(),
                                      /*test_id=*/42, hold_ms);
  // Marker: run_tests.sh waits for this line, then restarts a node.
  std::cout << "[REPRO] broadcast dispatched (hold_ms=" << hold_ms << ")"
            << std::endl;

  task.Wait();

  std::cout << "[REPRO] broadcast returned rc=" << task->return_code_
            << std::endl;
  // Post-fix the collective completes (rc 0 if every reachable replica
  // answered, or a timeout RC when the restarted node's replica was dropped).
  // The point of the reproducer is that it RETURNS rather than hanging.
  INFO("broadcast returned without hanging");
  REQUIRE(true);
}

SIMPLE_TEST_MAIN()
