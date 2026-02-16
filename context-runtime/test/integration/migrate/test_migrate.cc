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
 * Integration test for container migration with retry queue
 *
 * Test flow:
 * 1. Start 4 runtimes (via docker-compose)
 * 2. Create MOD_NAME pool with 4 containers (one per node)
 * 3. Submit CoMutexTest with DirectHash(0) targeting container 0 on node 0
 * 4. Migrate container 0 from node 0 to node 1
 * 5. Submit another CoMutexTest with DirectHash(0) — should route to node 1
 * 6. Verify both tasks completed successfully
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

chi::PoolId generateTestPoolId() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  return chi::PoolId(static_cast<chi::u32>(microseconds & 0xFFFFFFFF) + 1000,
                     0);
}

constexpr chi::u32 kHoldMs = 100;
}  // namespace

class MigrateTestFixture {
 public:
  MigrateTestFixture() : test_pool_id_(generateTestPoolId()) {
    if (!g_initialized) {
      INFO("Initializing Chimaera for Migrate tests...");
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

  chi::PoolId getTestPoolId() const { return test_pool_id_; }

  bool createModNamePool() {
    try {
      chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
      chimaera::MOD_NAME::Client mod_name_client(test_pool_id_);
      std::string pool_name = "test_migrate_pool";
      auto create_task =
          mod_name_client.AsyncCreate(pool_query, pool_name, test_pool_id_);
      create_task.Wait();
      mod_name_client.pool_id_ = create_task->new_pool_id_;
      mod_name_client.return_code_ = create_task->return_code_;
      bool success = (create_task->return_code_ == 0);
      REQUIRE(success);
      INFO("MOD_NAME pool created with ID: " << test_pool_id_.ToU64());
      return true;
    } catch (const std::exception &e) {
      FAIL("Exception creating MOD_NAME pool: " << e.what());
      return false;
    }
  }

 private:
  chi::PoolId test_pool_id_;
};

TEST_CASE("Migrate container and verify task re-routing",
          "[migrate][reroute]") {
  MigrateTestFixture fixture;

  SECTION("migrate_reroute") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_migrate_pool";
    auto create_task = mod_name_client.AsyncCreate(
        create_query, pool_name, fixture.getTestPoolId());
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    REQUIRE(create_task->return_code_ == 0);

    chi::PoolId pool_id = mod_name_client.pool_id_;
    INFO("Pool created: " << pool_id.ToU64());

    // Step 1: Submit CoMutexTest targeting container 0 (on node 0)
    INFO("Step 1: Submitting CoMutexTest with DirectHash(0) pre-migration");
    auto pre_task = mod_name_client.AsyncCoMutexTest(
        chi::PoolQuery::DirectHash(0), 1, kHoldMs);
    pre_task.Wait();
    REQUIRE(pre_task->return_code_ == 0);
    INFO("Step 1: Pre-migration task completed successfully");

    // Step 2: Migrate container 0 from node 0 to node 1
    // Node IDs are 1-indexed in the hostfile (node 0 -> node_id 1, etc.)
    // Container 0 is on node_id 1 (first node), migrate to node_id 2 (second)
    INFO("Step 2: Migrating container 0 from node 1 to node 2");
    auto *admin_client = CHI_ADMIN;
    REQUIRE(admin_client != nullptr);

    std::vector<chi::MigrateInfo> migrations;
    migrations.emplace_back(pool_id, chi::ContainerId(0), 2);

    auto migrate_task = admin_client->AsyncMigrateContainers(
        chi::PoolQuery::Local(), migrations);
    migrate_task.Wait();
    REQUIRE(migrate_task->GetReturnCode() == 0);
    REQUIRE(migrate_task->num_migrated_ == 1);
    INFO("Step 2: Migration completed, " << migrate_task->num_migrated_
                                          << " container(s) migrated");

    // Step 3: Submit another CoMutexTest targeting container 0
    // After migration, DirectHash(0) should resolve to node 2 via address_map_
    // If source node still has the task, the retry queue should re-route it
    INFO("Step 3: Submitting CoMutexTest with DirectHash(0) post-migration");
    auto post_task = mod_name_client.AsyncCoMutexTest(
        chi::PoolQuery::DirectHash(0), 2, kHoldMs);
    post_task.Wait();
    REQUIRE(post_task->return_code_ == 0);
    INFO("Step 3: Post-migration task completed successfully");

    INFO("Migration + retry queue test PASSED");
  }
}

TEST_CASE("Migrate container during broadcast event",
          "[migrate][broadcast]") {
  MigrateTestFixture fixture;

  SECTION("migrate_broadcast") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_migrate_broadcast_pool";
    auto create_task = mod_name_client.AsyncCreate(
        create_query, pool_name, fixture.getTestPoolId());
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    REQUIRE(create_task->return_code_ == 0);

    chi::PoolId pool_id = mod_name_client.pool_id_;
    INFO("Pool created: " << pool_id.ToU64());

    // Step 1: Submit broadcast CoMutexTest pre-migration
    INFO("Step 1: Submitting CoMutexTest with Broadcast() pre-migration");
    auto pre_task = mod_name_client.AsyncCoMutexTest(
        chi::PoolQuery::Broadcast(), 1, kHoldMs);
    pre_task.Wait();
    REQUIRE(pre_task->return_code_ == 0);
    INFO("Step 1: Pre-migration broadcast task completed successfully");

    // Step 2: Migrate container 0 from node 1 to node 2
    INFO("Step 2: Migrating container 0 from node 1 to node 2");
    auto *admin_client = CHI_ADMIN;
    REQUIRE(admin_client != nullptr);

    std::vector<chi::MigrateInfo> migrations;
    migrations.emplace_back(pool_id, chi::ContainerId(0), 2);

    auto migrate_task = admin_client->AsyncMigrateContainers(
        chi::PoolQuery::Local(), migrations);
    migrate_task.Wait();
    REQUIRE(migrate_task->GetReturnCode() == 0);
    REQUIRE(migrate_task->num_migrated_ == 1);
    INFO("Step 2: Migration completed, " << migrate_task->num_migrated_
                                          << " container(s) migrated");

    // Step 3: Submit broadcast CoMutexTest post-migration
    // Broadcast should reach all 4 containers including the migrated one
    // on its new node (node 2 now has containers 0 and 1)
    INFO("Step 3: Submitting CoMutexTest with Broadcast() post-migration");
    auto post_task = mod_name_client.AsyncCoMutexTest(
        chi::PoolQuery::Broadcast(), 2, kHoldMs);
    post_task.Wait();
    REQUIRE(post_task->return_code_ == 0);
    INFO("Step 3: Post-migration broadcast task completed successfully");

    INFO("Migration + broadcast test PASSED");
  }
}

SIMPLE_TEST_MAIN()
