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
 * Integration test for node failure recovery
 *
 * Test flow (orchestrated by run_tests.sh):
 * Phase 1 (pre-failure):
 *   1. Start 4 runtimes (via docker-compose)
 *   2. Create MOD_NAME pool with 4 containers (one per node)
 *   3. Submit CoMutexTest with DirectHash(3) targeting container 3 on node 4
 *   4. Submit CoMutexTest with Broadcast() across all 4 nodes
 *
 * (run_tests.sh kills node 4, waits 30s for SWIM detection + recovery)
 *
 * Phase 2 (post-failure):
 *   1. Connect to surviving 3-node cluster
 *   2. Reuse pool from Phase 1 (container 3 should have been recovered)
 *   3. Submit CoMutexTest with DirectHash(3) — should route to recovered container
 *   4. Submit CoMutexTest with Broadcast() — should succeed across surviving nodes
 */

#include "simple_test.h"

#include <chrono>
#include <thread>
#include <vector>

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

// Fixed pool ID shared between Phase 1 and Phase 2
// Phase 2 reuses the pool created by Phase 1 (no new pool creation
// since Dynamic() would try to reach the dead node)
const chi::PoolId kRecoveryPoolId(50000, 0);

constexpr chi::u32 kHoldMs = 100;
}  // namespace

class RecoveryTestFixture {
 public:
  RecoveryTestFixture() {
    if (!g_initialized) {
      INFO("Initializing Chimaera for Recovery tests...");
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

  chi::PoolId getTestPoolId() const { return kRecoveryPoolId; }

  bool createModNamePool(const std::string &pool_name) {
    try {
      chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
      chimaera::MOD_NAME::Client mod_name_client(kRecoveryPoolId);
      auto create_task =
          mod_name_client.AsyncCreate(pool_query, pool_name, kRecoveryPoolId);
      create_task.Wait();
      mod_name_client.pool_id_ = create_task->new_pool_id_;
      mod_name_client.return_code_ = create_task->return_code_;
      bool success = (create_task->return_code_ == 0);
      REQUIRE(success);
      INFO("MOD_NAME pool created with ID: " << kRecoveryPoolId.ToU64());
      return true;
    } catch (const std::exception &e) {
      FAIL("Exception creating MOD_NAME pool: " << e.what());
      return false;
    }
  }
};

TEST_CASE("Pre-failure: verify tasks work on all nodes",
          "[recovery][pre_failure]") {
  RecoveryTestFixture fixture;

  SECTION("pre_failure_direct_and_broadcast") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool("test_recovery_pool"));

    chimaera::MOD_NAME::Client mod_name_client(kRecoveryPoolId);
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_recovery_pool";
    auto create_task = mod_name_client.AsyncCreate(
        create_query, pool_name, kRecoveryPoolId);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    REQUIRE(create_task->return_code_ == 0);

    chi::PoolId pool_id = mod_name_client.pool_id_;
    INFO("Pool created: " << pool_id.ToU64());

    // Step 1: Submit CoMutexTest targeting container 3 (on node 4)
    INFO("Step 1: Submitting CoMutexTest with DirectHash(3) pre-failure");
    auto direct_task = mod_name_client.AsyncCoMutexTest(
        chi::PoolQuery::DirectHash(3), 1, kHoldMs);
    direct_task.Wait();
    REQUIRE(direct_task->return_code_ == 0);
    INFO("Step 1: DirectHash(3) task completed successfully");

    // Step 2: Submit broadcast CoMutexTest across all 4 nodes
    INFO("Step 2: Submitting CoMutexTest with Broadcast() pre-failure");
    auto broadcast_task = mod_name_client.AsyncCoMutexTest(
        chi::PoolQuery::Broadcast(), 1, kHoldMs);
    broadcast_task.Wait();
    REQUIRE(broadcast_task->return_code_ == 0);
    INFO("Step 2: Broadcast task completed successfully");

    INFO("Pre-failure tests PASSED");
  }
}

TEST_CASE("Post-failure: verify recovery re-routes tasks",
          "[recovery][post_failure]") {
  RecoveryTestFixture fixture;

  SECTION("post_failure_direct_and_broadcast") {
    REQUIRE(g_initialized);

    // Reuse the pool created by Phase 1 — do NOT create a new pool
    // because Dynamic() would try to reach the dead node and hang.
    // The recovery system should have re-created container 3 on a
    // surviving node after SWIM detected node 4's death.
    chimaera::MOD_NAME::Client mod_name_client(kRecoveryPoolId);
    mod_name_client.pool_id_ = kRecoveryPoolId;
    INFO("Reusing pool from Phase 1: " << kRecoveryPoolId.ToU64());

    // Step 1: Submit CoMutexTest targeting container 3
    // Originally on node 4 (now dead), should route to recovery destination
    INFO("Step 1: Submitting CoMutexTest with DirectHash(3) post-failure");
    auto direct_task = mod_name_client.AsyncCoMutexTest(
        chi::PoolQuery::DirectHash(3), 1, kHoldMs);
    direct_task.Wait();
    REQUIRE(direct_task->return_code_ == 0);
    INFO("Step 1: DirectHash(3) post-failure task completed — recovery works");

    // Step 2: Submit broadcast CoMutexTest
    // Should succeed across 3 surviving nodes + recovered container
    INFO("Step 2: Submitting CoMutexTest with Broadcast() post-failure");
    auto broadcast_task = mod_name_client.AsyncCoMutexTest(
        chi::PoolQuery::Broadcast(), 1, kHoldMs);
    broadcast_task.Wait();
    REQUIRE(broadcast_task->return_code_ == 0);
    INFO("Step 2: Broadcast post-failure task completed successfully");

    INFO("Post-failure recovery tests PASSED");
  }
}

SIMPLE_TEST_MAIN()
