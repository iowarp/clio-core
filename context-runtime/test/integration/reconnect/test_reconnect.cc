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
 * Integration test for client failover to new hosts
 *
 * Validates WaitForServerAndReconnect Phase 2 (CHI_CLIENT_TRY_NEW_SERVERS).
 *
 * Test flow (single phase, orchestrated by run_tests.sh):
 *   1. Start 4 runtimes (via docker-compose)
 *   2. Client on node 1 initializes and connects to local runtime
 *   3. Client creates a MOD_NAME pool and runs a task to verify connectivity
 *   4. Client sends AsyncStopRuntime to shut down node 1's runtime
 *   5. Client waits for the runtime to die
 *   6. Client sends another task — Recv() detects the server is dead
 *   7. WaitForServerAndReconnect Phase 1 times out (original server gone)
 *   8. Phase 2 picks random hosts from hostfile, reconnects to a survivor
 *   9. Task completes successfully on the new host
 *
 * Environment (set by docker-compose):
 *   CHI_CLIENT_RETRY_TIMEOUT=5    (Phase 1 gives up after 5s)
 *   CHI_CLIENT_TRY_NEW_SERVERS=16 (Phase 2 tries up to 16 random hosts)
 */

#include "simple_test.h"

#include <chrono>
#include <thread>

#include <chimaera/chimaera.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>

#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>

#include <chimaera/admin/admin_client.h>
#include <chimaera/admin/admin_tasks.h>

using namespace std::chrono_literals;

namespace {
bool g_initialized = false;
const chi::PoolId kReconnectPoolId(60000, 0);
constexpr chi::u32 kHoldMs = 100;
}  // namespace

class ReconnectTestFixture {
 public:
  ReconnectTestFixture() {
    if (!g_initialized) {
      INFO("Initializing Chimaera client for Reconnect tests...");
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (success) {
        g_initialized = true;
        // NOTE: Do NOT set g_test_finalize here because the test
        // intentionally kills the local runtime.  Calling CHIMAERA_FINALIZE
        // after the server is dead would hang or crash.
        std::this_thread::sleep_for(500ms);
        REQUIRE(CHI_CHIMAERA_MANAGER != nullptr);
        REQUIRE(CHI_IPC != nullptr);
        REQUIRE(CHI_POOL_MANAGER != nullptr);
        REQUIRE(CHI_IPC->IsInitialized());
        INFO("Chimaera initialization successful");
      } else {
        FAIL("Failed to initialize Chimaera");
      }
    }
  }
};

TEST_CASE("Failover to new host after server shutdown",
          "[reconnect]") {
  ReconnectTestFixture fixture;

  SECTION("reconnect_failover") {
    REQUIRE(g_initialized);

    // ------------------------------------------------------------------
    // Step 1: Create a MOD_NAME pool and verify a task completes locally
    // ------------------------------------------------------------------
    INFO("Step 1: Creating MOD_NAME pool and running pre-shutdown task");
    chimaera::MOD_NAME::Client mod_name_client(kReconnectPoolId);
    {
      auto create_task = mod_name_client.AsyncCreate(
          chi::PoolQuery::Dynamic(), "reconnect_test_pool", kReconnectPoolId);
      create_task.Wait();
      mod_name_client.pool_id_ = create_task->new_pool_id_;
      mod_name_client.return_code_ = create_task->return_code_;
      REQUIRE(create_task->return_code_ == 0);
      INFO("Pool created: " << kReconnectPoolId.ToU64());
    }

    {
      auto task = mod_name_client.AsyncCoMutexTest(
          chi::PoolQuery::Local(), 1, kHoldMs);
      task.Wait();
      REQUIRE(task->return_code_ == 0);
      INFO("Step 1: Pre-shutdown task completed on local node");
    }

    // ------------------------------------------------------------------
    // Step 2: Shut down the local runtime via AsyncStopRuntime
    // ------------------------------------------------------------------
    INFO("Step 2: Sending AsyncStopRuntime to local runtime");
    {
      chimaera::admin::Client admin_client(chi::kAdminPoolId);
      // Fire-and-forget: send stop with short grace period.
      // Do NOT call Wait() — the server may die before responding.
      admin_client.AsyncStopRuntime(
          chi::PoolQuery::Local(), 0, 1000);
    }

    // Wait for the runtime process to actually exit
    INFO("Step 2: Waiting for local runtime to die...");
    std::this_thread::sleep_for(5s);
    INFO("Step 2: Local runtime should be dead now");

    // ------------------------------------------------------------------
    // Step 3: Send a new task — this triggers reconnection
    // ------------------------------------------------------------------
    // The Recv() call will detect the server is dead, enter
    // WaitForServerAndReconnect, time out on the original server
    // (Phase 1, ~5s via CHI_CLIENT_RETRY_TIMEOUT), then try random
    // hosts from the hostfile (Phase 2, CHI_CLIENT_TRY_NEW_SERVERS=16).
    // One of nodes 2/3/4 should accept the connection.
    //
    // After reconnection the client is on a new node where the pool
    // does not exist.  We create a fresh pool on the new host and
    // run a task to verify end-to-end connectivity.
    // ------------------------------------------------------------------
    INFO("Step 3: Creating pool on new host after reconnection");
    {
      chimaera::MOD_NAME::Client new_client(kReconnectPoolId);
      auto create_task = new_client.AsyncCreate(
          chi::PoolQuery::Dynamic(), "reconnect_post_failover_pool",
          kReconnectPoolId);
      create_task.Wait();
      new_client.pool_id_ = create_task->new_pool_id_;
      new_client.return_code_ = create_task->return_code_;
      REQUIRE(create_task->return_code_ == 0);
      INFO("Step 3: Pool created on new host");

      auto task = new_client.AsyncCoMutexTest(
          chi::PoolQuery::Local(), 1, kHoldMs);
      task.Wait();
      REQUIRE(task->return_code_ == 0);
      INFO("Step 3: Post-failover task completed successfully");
    }

    INFO("Reconnect failover test PASSED");
  }
}

SIMPLE_TEST_MAIN()
