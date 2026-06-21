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
 * Unit tests for SubmitBatch functionality
 *
 * Tests the TaskBatch and SubmitBatchTask functionality for efficient
 * batch task submission.
 */

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

/**
 * Test helper to initialize CLIO Runtime system
 */
class ClioTestFixture {
 public:
  ClioTestFixture() {
    // Use the unified CLIO Runtime initialization
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;

    // Wait for runtime to fully initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Verify initialization
    REQUIRE(CLIO_IPC != nullptr);
    REQUIRE(CLIO_IPC->IsInitialized());
  }

  ~ClioTestFixture() {
    // Cleanup handled by runtime
  }
};

}  // anonymous namespace

TEST_CASE("TaskBatch Basic Functionality", "[submit_batch][admin]") {
  ClioTestFixture fixture;

  SECTION("TaskBatch starts empty") {
    clio::run::admin::TaskBatch batch;
    REQUIRE(batch.GetTaskCount() == 0);
    REQUIRE(batch.GetTaskInfos().empty());
    REQUIRE(batch.GetSerializedData().empty());
  }

  SECTION("TaskBatch Add increments task count") {
    clio::run::admin::TaskBatch batch;

    // Add a FlushTask to the batch
    batch.Add<clio::run::admin::FlushTask>(
        clio::run::CreateTaskId(),
        clio::run::kAdminPoolId,
        clio::run::PoolQuery::Local());

    REQUIRE(batch.GetTaskCount() == 1);
    REQUIRE(batch.GetTaskInfos().size() == 1);
    REQUIRE(!batch.GetSerializedData().empty());
  }

  SECTION("TaskBatch can add multiple tasks") {
    clio::run::admin::TaskBatch batch;

    // Add multiple FlushTasks to the batch
    for (int i = 0; i < 5; ++i) {
      batch.Add<clio::run::admin::FlushTask>(
          clio::run::CreateTaskId(),
          clio::run::kAdminPoolId,
          clio::run::PoolQuery::Local());
    }

    REQUIRE(batch.GetTaskCount() == 5);
    REQUIRE(batch.GetTaskInfos().size() == 5);
  }
}

TEST_CASE("SubmitBatch Empty Batch", "[submit_batch][admin]") {
  ClioTestFixture fixture;

  SECTION("SubmitBatch with empty batch returns success immediately") {
    // Create admin client
    clio::run::admin::Client admin_client(clio::run::kAdminPoolId);

    // Create empty batch
    clio::run::admin::TaskBatch batch;

    // Submit empty batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        clio::run::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results - empty batch should succeed with 0 tasks completed
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == 0);
  }
}

TEST_CASE("SubmitBatch Single Task", "[submit_batch][admin]") {
  ClioTestFixture fixture;

  SECTION("SubmitBatch with single FlushTask succeeds") {
    // Create admin client
    clio::run::admin::Client admin_client(clio::run::kAdminPoolId);

    // Create batch with one task
    clio::run::admin::TaskBatch batch;
    batch.Add<clio::run::admin::FlushTask>(
        clio::run::CreateTaskId(),
        clio::run::kAdminPoolId,
        clio::run::PoolQuery::Local());

    INFO("TaskBatch has " << batch.GetTaskCount() << " tasks");

    // Submit batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        clio::run::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == 1);
  }
}

TEST_CASE("SubmitBatch Multiple Tasks", "[submit_batch][admin]") {
  ClioTestFixture fixture;

  SECTION("SubmitBatch with multiple FlushTasks succeeds") {
    // Create admin client
    clio::run::admin::Client admin_client(clio::run::kAdminPoolId);

    // Create batch with multiple tasks
    const size_t num_tasks = 10;
    clio::run::admin::TaskBatch batch;

    for (size_t i = 0; i < num_tasks; ++i) {
      batch.Add<clio::run::admin::FlushTask>(
          clio::run::CreateTaskId(),
          clio::run::kAdminPoolId,
          clio::run::PoolQuery::Local());
    }

    INFO("TaskBatch has " << batch.GetTaskCount() << " tasks");
    REQUIRE(batch.GetTaskCount() == num_tasks);

    // Submit batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        clio::run::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == num_tasks);
  }
}

TEST_CASE("SubmitBatch Large Batch", "[submit_batch][admin]") {
  ClioTestFixture fixture;

  SECTION("SubmitBatch with batch larger than parallel limit (32) succeeds") {
    // Create admin client
    clio::run::admin::Client admin_client(clio::run::kAdminPoolId);

    // Create batch with more tasks than parallel limit (32)
    const size_t num_tasks = 50;
    clio::run::admin::TaskBatch batch;

    for (size_t i = 0; i < num_tasks; ++i) {
      batch.Add<clio::run::admin::FlushTask>(
          clio::run::CreateTaskId(),
          clio::run::kAdminPoolId,
          clio::run::PoolQuery::Local());
    }

    INFO("TaskBatch has " << batch.GetTaskCount() << " tasks");
    REQUIRE(batch.GetTaskCount() == num_tasks);

    // Submit batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        clio::run::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == num_tasks);
  }
}

TEST_CASE("SubmitBatch with MonitorTask", "[submit_batch][admin]") {
  ClioTestFixture fixture;

  SECTION("SubmitBatch with MonitorTasks succeeds") {
    // Create admin client
    clio::run::admin::Client admin_client(clio::run::kAdminPoolId);

    // Create batch with multiple MonitorTasks
    const size_t num_tasks = 5;
    clio::run::admin::TaskBatch batch;

    for (size_t i = 0; i < num_tasks; ++i) {
      batch.Add<clio::run::admin::MonitorTask>(
          clio::run::CreateTaskId(),
          clio::run::kAdminPoolId,
          clio::run::PoolQuery::Local(),
          std::string("status"));
    }

    INFO("TaskBatch has " << batch.GetTaskCount() << " MonitorTasks");
    REQUIRE(batch.GetTaskCount() == num_tasks);

    // Submit batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        clio::run::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == num_tasks);
  }
}

SIMPLE_TEST_MAIN()
