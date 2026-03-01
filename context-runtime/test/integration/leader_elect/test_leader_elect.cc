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
 * Integration test for leader election / failover / restart
 *
 * Validates the full cycle:
 *   Phase 1 ([leader_elect][failover]):
 *     1. Client on node 1 (leader) initializes and connects
 *     2. Client creates a MOD_NAME pool and runs a task
 *     3. Client sends AsyncStopRuntime to shut down node 1's runtime
 *     4. Client waits for the runtime to die
 *     5. Client sends another task — triggers reconnection to a survivor
 *     6. Task completes on new host
 *
 *   External step (run_tests.sh restarts node 1's runtime)
 *
 *   Phase 2 ([leader_elect][post_restart]):
 *     1. Fresh client init after leader restart
 *     2. Create pool and run task
 *     3. Verify completion — system healthy after leader restart
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
const chi::PoolId kLeaderElectPoolId(70000, 0);
constexpr chi::u32 kHoldMs = 100;
}  // namespace

class LeaderElectFixture {
 public:
  LeaderElectFixture() {
    if (!g_initialized) {
      INFO("Initializing Chimaera client for leader election tests...");
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (success) {
        g_initialized = true;
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

// =========================================================================
// Phase 1: Shut down the leader and fail over to another node
// =========================================================================
TEST_CASE("Leader shutdown and failover to new host",
          "[leader_elect][failover]") {
  LeaderElectFixture fixture;

  SECTION("leader_failover") {
    REQUIRE(g_initialized);

    // ------------------------------------------------------------------
    // Step 1: Create a MOD_NAME pool and verify a task completes locally
    // ------------------------------------------------------------------
    INFO("Step 1: Creating MOD_NAME pool and running pre-shutdown task");
    chimaera::MOD_NAME::Client mod_name_client(kLeaderElectPoolId);
    {
      auto create_task = mod_name_client.AsyncCreate(
          chi::PoolQuery::Dynamic(), "leader_elect_test_pool",
          kLeaderElectPoolId);
      create_task.Wait();
      mod_name_client.pool_id_ = create_task->new_pool_id_;
      mod_name_client.return_code_ = create_task->return_code_;
      REQUIRE(create_task->return_code_ == 0);
      INFO("Pool created: " << kLeaderElectPoolId.ToU64());
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
      admin_client.AsyncStopRuntime(
          chi::PoolQuery::Local(), 0, 1000);
    }

    INFO("Step 2: Waiting for local runtime to die...");
    std::this_thread::sleep_for(5s);
    INFO("Step 2: Local runtime should be dead now");

    // ------------------------------------------------------------------
    // Step 3: Send a new task — triggers reconnection to a survivor
    // ------------------------------------------------------------------
    INFO("Step 3: Creating pool on new host after reconnection");
    {
      chimaera::MOD_NAME::Client new_client(kLeaderElectPoolId);
      auto create_task = new_client.AsyncCreate(
          chi::PoolQuery::Dynamic(), "leader_elect_post_failover_pool",
          kLeaderElectPoolId);
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

    INFO("Leader failover test PASSED");
  }
}

// =========================================================================
// Phase 2: After the leader has been restarted, verify the system is healthy
// =========================================================================
TEST_CASE("System healthy after leader restart",
          "[leader_elect][post_restart]") {
  // Fresh initialization — the previous client state is stale because
  // run_tests.sh launches this as a separate process invocation.
  INFO("Initializing fresh Chimaera client after leader restart...");
  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  REQUIRE(success);
  std::this_thread::sleep_for(500ms);
  REQUIRE(CHI_CHIMAERA_MANAGER != nullptr);
  REQUIRE(CHI_IPC != nullptr);
  REQUIRE(CHI_POOL_MANAGER != nullptr);
  REQUIRE(CHI_IPC->IsInitialized());
  INFO("Fresh Chimaera initialization successful");

  SECTION("post_restart_task") {
    // ------------------------------------------------------------------
    // Create a pool and run a task on the restarted leader
    // ------------------------------------------------------------------
    INFO("Creating MOD_NAME pool on restarted leader");
    chimaera::MOD_NAME::Client mod_name_client(kLeaderElectPoolId);
    {
      auto create_task = mod_name_client.AsyncCreate(
          chi::PoolQuery::Dynamic(), "leader_elect_post_restart_pool",
          kLeaderElectPoolId);
      create_task.Wait();
      mod_name_client.pool_id_ = create_task->new_pool_id_;
      mod_name_client.return_code_ = create_task->return_code_;
      REQUIRE(create_task->return_code_ == 0);
      INFO("Pool created on restarted leader");
    }

    {
      auto task = mod_name_client.AsyncCoMutexTest(
          chi::PoolQuery::Local(), 1, kHoldMs);
      task.Wait();
      REQUIRE(task->return_code_ == 0);
      INFO("Post-restart task completed successfully");
    }

    INFO("Post-restart test PASSED");
  }
}

SIMPLE_TEST_MAIN()
