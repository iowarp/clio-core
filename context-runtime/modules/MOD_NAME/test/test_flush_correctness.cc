/**
 * Unit tests for FlushTask correctness
 *
 * Tests the flush functionality with admin chimod pool initialization
 * to verify proper runtime setup and flush operations.
 */

#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>
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

// Test helper to initialize Chimaera system
class ChimaeraTestFixture {
 public:
  ChimaeraTestFixture() {
    // Use the unified Chimaera initialization
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);
  }

  ~ChimaeraTestFixture() {
    // Cleanup handled by runtime
  }
};

}  // anonymous namespace

TEST_CASE("FlushTask Basic Functionality", "[flush][admin]") {
  ChimaeraTestFixture fixture;

  SECTION("Flush with no work remaining returns success immediately") {
    // Create admin client
    chimaera::admin::Client admin_client(chi::kAdminPoolId);

    // Create flush task
    auto pool_query = chi::PoolQuery();
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
  ChimaeraTestFixture fixture;

  SECTION("Flush waits for MOD_NAME async Custom tasks to complete") {
    // Create MOD_NAME client and container - CreateTask will auto-create pool
    const chi::PoolId mod_name_pool_id = chi::PoolId(4000, 0);
    chimaera::MOD_NAME::Client mod_name_client(mod_name_pool_id);

    // Create the MOD_NAME container with local pool query - this will create
    // pool if needed
    auto pool_query = chi::PoolQuery::Local();
    std::string pool_name = "flush_test_mod_name_pool";
    auto create_task = mod_name_client.AsyncCreate(pool_query, pool_name, mod_name_pool_id);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    mod_name_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE(success);

    // Send multiple async Custom tasks to the MOD_NAME runtime
    const int num_async_tasks = 5;
    std::vector<chi::Future<chimaera::MOD_NAME::CustomTask>> async_tasks;

    for (int i = 0; i < num_async_tasks; i++) {
      std::string input_data = "test_data_" + std::to_string(i);
      chi::u32 operation_id = static_cast<chi::u32>(i + 1);

      // Create async custom task
      auto async_task = mod_name_client.AsyncCustom(chi::PoolQuery::Local(), input_data, operation_id);

      async_tasks.push_back(async_task);
    }

    // Start flush operation in background thread
    chimaera::admin::Client flush_admin_client(chi::kAdminPoolId);
    std::atomic<bool> flush_completed{false};
    std::atomic<chi::u32> flush_result_code{999};

    std::thread flush_thread([&]() {
      auto flush_task =
          flush_admin_client.AsyncFlush(chi::PoolQuery::Local());
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