/**
 * Unit tests for SubmitBatch functionality
 *
 * Tests the TaskBatch and SubmitBatchTask functionality for efficient
 * batch task submission.
 */

#include <chimaera/admin/admin_client.h>
#include <chimaera/admin/admin_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/ipc_manager.h>
#include <chimaera/pool_manager.h>
#include <chimaera/work_orchestrator.h>
#include <simple_test.h>

#include <chrono>
#include <thread>

namespace {

/**
 * Test helper to initialize Chimaera system
 */
class ChimaeraTestFixture {
 public:
  ChimaeraTestFixture() {
    // Use the unified Chimaera initialization
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);

    // Wait for runtime to fully initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Verify initialization
    REQUIRE(CHI_IPC != nullptr);
    REQUIRE(CHI_IPC->IsInitialized());
  }

  ~ChimaeraTestFixture() {
    // Cleanup handled by runtime
  }
};

}  // anonymous namespace

TEST_CASE("TaskBatch Basic Functionality", "[submit_batch][admin]") {
  ChimaeraTestFixture fixture;

  SECTION("TaskBatch starts empty") {
    chimaera::admin::TaskBatch batch;
    REQUIRE(batch.GetTaskCount() == 0);
    REQUIRE(batch.GetTaskInfos().empty());
    REQUIRE(batch.GetSerializedData().empty());
  }

  SECTION("TaskBatch Add increments task count") {
    chimaera::admin::TaskBatch batch;

    // Add a FlushTask to the batch
    batch.Add<chimaera::admin::FlushTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,
        chi::PoolQuery::Local());

    REQUIRE(batch.GetTaskCount() == 1);
    REQUIRE(batch.GetTaskInfos().size() == 1);
    REQUIRE(!batch.GetSerializedData().empty());
  }

  SECTION("TaskBatch can add multiple tasks") {
    chimaera::admin::TaskBatch batch;

    // Add multiple FlushTasks to the batch
    for (int i = 0; i < 5; ++i) {
      batch.Add<chimaera::admin::FlushTask>(
          chi::CreateTaskId(),
          chi::kAdminPoolId,
          chi::PoolQuery::Local());
    }

    REQUIRE(batch.GetTaskCount() == 5);
    REQUIRE(batch.GetTaskInfos().size() == 5);
  }
}

TEST_CASE("SubmitBatch Empty Batch", "[submit_batch][admin]") {
  ChimaeraTestFixture fixture;

  SECTION("SubmitBatch with empty batch returns success immediately") {
    // Create admin client
    chimaera::admin::Client admin_client(chi::kAdminPoolId);

    // Create empty batch
    chimaera::admin::TaskBatch batch;

    // Submit empty batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        chi::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results - empty batch should succeed with 0 tasks completed
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == 0);
  }
}

TEST_CASE("SubmitBatch Single Task", "[submit_batch][admin]") {
  ChimaeraTestFixture fixture;

  SECTION("SubmitBatch with single FlushTask succeeds") {
    // Create admin client
    chimaera::admin::Client admin_client(chi::kAdminPoolId);

    // Create batch with one task
    chimaera::admin::TaskBatch batch;
    batch.Add<chimaera::admin::FlushTask>(
        chi::CreateTaskId(),
        chi::kAdminPoolId,
        chi::PoolQuery::Local());

    INFO("TaskBatch has " << batch.GetTaskCount() << " tasks");

    // Submit batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        chi::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == 1);
  }
}

TEST_CASE("SubmitBatch Multiple Tasks", "[submit_batch][admin]") {
  ChimaeraTestFixture fixture;

  SECTION("SubmitBatch with multiple FlushTasks succeeds") {
    // Create admin client
    chimaera::admin::Client admin_client(chi::kAdminPoolId);

    // Create batch with multiple tasks
    const size_t num_tasks = 10;
    chimaera::admin::TaskBatch batch;

    for (size_t i = 0; i < num_tasks; ++i) {
      batch.Add<chimaera::admin::FlushTask>(
          chi::CreateTaskId(),
          chi::kAdminPoolId,
          chi::PoolQuery::Local());
    }

    INFO("TaskBatch has " << batch.GetTaskCount() << " tasks");
    REQUIRE(batch.GetTaskCount() == num_tasks);

    // Submit batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        chi::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == num_tasks);
  }
}

TEST_CASE("SubmitBatch Large Batch", "[submit_batch][admin]") {
  ChimaeraTestFixture fixture;

  SECTION("SubmitBatch with batch larger than parallel limit (32) succeeds") {
    // Create admin client
    chimaera::admin::Client admin_client(chi::kAdminPoolId);

    // Create batch with more tasks than parallel limit (32)
    const size_t num_tasks = 50;
    chimaera::admin::TaskBatch batch;

    for (size_t i = 0; i < num_tasks; ++i) {
      batch.Add<chimaera::admin::FlushTask>(
          chi::CreateTaskId(),
          chi::kAdminPoolId,
          chi::PoolQuery::Local());
    }

    INFO("TaskBatch has " << batch.GetTaskCount() << " tasks");
    REQUIRE(batch.GetTaskCount() == num_tasks);

    // Submit batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        chi::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == num_tasks);
  }
}

TEST_CASE("SubmitBatch with MonitorTask", "[submit_batch][admin]") {
  ChimaeraTestFixture fixture;

  SECTION("SubmitBatch with MonitorTasks succeeds") {
    // Create admin client
    chimaera::admin::Client admin_client(chi::kAdminPoolId);

    // Create batch with multiple MonitorTasks
    const size_t num_tasks = 5;
    chimaera::admin::TaskBatch batch;

    for (size_t i = 0; i < num_tasks; ++i) {
      batch.Add<chimaera::admin::MonitorTask>(
          chi::CreateTaskId(),
          chi::kAdminPoolId,
          chi::PoolQuery::Local());
    }

    INFO("TaskBatch has " << batch.GetTaskCount() << " MonitorTasks");
    REQUIRE(batch.GetTaskCount() == num_tasks);

    // Submit batch
    auto submit_task = admin_client.AsyncSubmitBatch(
        chi::PoolQuery::Local(), batch);

    // Wait for completion
    submit_task.Wait();

    // Verify results
    REQUIRE(submit_task->GetReturnCode() == 0);
    REQUIRE(submit_task->tasks_completed_ == num_tasks);
  }
}

SIMPLE_TEST_MAIN()
