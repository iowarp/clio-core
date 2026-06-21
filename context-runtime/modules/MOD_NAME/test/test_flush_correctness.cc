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
 * Unit tests for FlushTask correctness
 *
 * Tests the flush functionality with admin chimod pool initialization
 * to verify proper runtime setup and flush operations.
 */

#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>
#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/pool_manager.h>
#include <clio_runtime/work_orchestrator.h>
#include <simple_test.h>

#include <chrono>
#include <thread>

namespace {

// Test helper to initialize CLIO Runtime system
class ClioTestFixture {
 public:
  ClioTestFixture() {
    // Use the unified CLIO Runtime initialization
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  }

  ~ClioTestFixture() {
    // Cleanup handled by runtime
  }
};

}  // anonymous namespace

TEST_CASE("FlushTask Basic Functionality", "[flush][admin]") {
  ClioTestFixture fixture;

  SECTION("Flush with no work remaining returns success immediately") {
    // Create admin client
    clio::run::admin::Client admin_client(clio::run::kAdminPoolId);

    // Create flush task
    auto pool_query = clio::run::PoolQuery();
    auto flush_task = admin_client.AsyncFlush(pool_query);

    // Wait for completion
    flush_task.Wait();

    // Verify results
    REQUIRE(flush_task->return_code_ == 0);
    REQUIRE(flush_task->total_work_done_ == 0);
  }
}

TEST_CASE("FlushTask with MOD_NAME Container and Async Tasks",
          "[flush][mod_name]") {
  ClioTestFixture fixture;

  SECTION("Flush waits for MOD_NAME async Custom tasks to complete") {
    // Create MOD_NAME client and container - CreateTask will auto-create pool
    const clio::run::PoolId mod_name_pool_id = clio::run::PoolId(4000, 0);
    clio::run::MOD_NAME::Client mod_name_client(mod_name_pool_id);

    // Create the MOD_NAME container with local pool query - this will create
    // pool if needed
    auto pool_query = clio::run::PoolQuery::Local();
    std::string pool_name = "flush_test_mod_name_pool";
    auto create_task = mod_name_client.AsyncCreate(pool_query, pool_name, mod_name_pool_id);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    mod_name_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE(success);

    // Send multiple async Custom tasks to the MOD_NAME runtime
    const int num_async_tasks = 5;
    std::vector<clio::run::Future<clio::run::MOD_NAME::CustomTask>> async_tasks;

    for (int i = 0; i < num_async_tasks; i++) {
      std::string input_data = "test_data_" + std::to_string(i);
      clio::run::u32 operation_id = static_cast<clio::run::u32>(i + 1);

      // Create async custom task
      auto async_task = mod_name_client.AsyncCustom(clio::run::PoolQuery::Local(), input_data, operation_id);

      async_tasks.push_back(async_task);
    }

    // Start flush operation in background thread
    clio::run::admin::Client flush_admin_client(clio::run::kAdminPoolId);
    std::atomic<bool> flush_completed{false};
    std::atomic<clio::run::u32> flush_result_code{999};

    std::thread flush_thread([&]() {
      auto flush_task =
          flush_admin_client.AsyncFlush(clio::run::PoolQuery::Local());
      flush_task.Wait();

      flush_result_code.store(flush_task->return_code_);
      flush_completed.store(true);
    });

    // Give the flush a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Wait for all async operations to complete
    for (auto& async_task : async_tasks) {
      async_task.Wait();
      REQUIRE(async_task->return_code_ == 0);
    }

    // Wait for flush to complete
    flush_thread.join();
    REQUIRE(flush_completed.load());
    REQUIRE(flush_result_code.load() == 0);

    INFO("MOD_NAME flush test completed - flush works with async Custom tasks");
  }
}

SIMPLE_TEST_MAIN()