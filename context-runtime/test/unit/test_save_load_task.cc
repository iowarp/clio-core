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
 * Unit tests for Container SaveTask and LoadTask methods
 *
 * Tests the complete SaveIn -> LoadIn -> SaveOut -> LoadOut flow:
 * 1. Create task with input parameters
 * 2. SaveIn (serialize IN/INOUT parameters)
 * 3. LoadIn (deserialize IN/INOUT parameters into new task)
 * 4. Verify loaded task has same inputs (IN fields, task IDs, etc.)
 * 5. Modify output parameters in loaded task
 * 6. SaveOut (serialize OUT/INOUT parameters)
 * 7. LoadOut (deserialize OUT/INOUT parameters)
 * 8. Verify final task has correct inputs and outputs
 */

#include "../simple_test.h"
#include <memory>
#include <string>
#include <vector>

// Include CLIO Runtime headers
#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/memory/allocator/malloc_allocator.h>
#include <clio_runtime/container.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/module_manager.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/task.h>
#include <clio_runtime/task_archives.h>
#include <clio_runtime/types.h>

// Include admin tasks for testing
#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_runtime.h>
#include <clio_runtime/admin/admin_tasks.h>

// Include bdev tasks for testing
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>

using namespace clio::run;

namespace {
// Helper allocator for tests - uses CTP_MALLOC for non-IPC allocations
ctp::ipc::MallocAllocator* GetTestAllocator() {
  return CTP_MALLOC;
}

// Initialize CLIO Runtime runtime for tests
class ClioTestFixture {
public:
  ClioTestFixture() {
    // Initialize CLIO Runtime (client with embedded runtime)
    clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;

    // Create admin pool
    admin_client_ =
        std::make_unique<clio::run::admin::Client>(clio::run::kAdminPoolId);
    auto create_task = admin_client_->AsyncCreate(clio::run::PoolQuery::Local(), "admin",
                                                   clio::run::kAdminPoolId);
    create_task.Wait();

    // Update client pool_id_ with the actual pool ID from the task
    admin_client_->pool_id_ = create_task->new_pool_id_;
    admin_client_->return_code_ = create_task->return_code_;

    // Verify admin creation succeeded
    if (create_task->GetReturnCode() != 0) {
      throw std::runtime_error("Failed to create admin pool");
    }
    // Task automatically freed when create_task goes out of scope
  }

  ~ClioTestFixture() {
    // Cleanup
    admin_client_.reset();
  }

  clio::run::admin::Client *GetAdminClient() { return admin_client_.get(); }

private:
  std::unique_ptr<clio::run::admin::Client> admin_client_;
};
} // namespace

TEST_CASE("SaveTask and LoadTask - Admin CreateTask full flow",
          "[save_load_task][admin][create]") {
  ClioTestFixture fixture;

  auto *ipc_manager = CLIO_IPC;
  auto alloc = GetTestAllocator();

  // Get container for SaveTask/LoadTask
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto container = pool_manager->GetStaticContainer(clio::run::kAdminPoolId).get();
  REQUIRE(container != nullptr);

  // Step 1: Create original task with input parameters
  auto orig_task = ipc_manager->NewTask<clio::run::admin::CreateTask>(
      clio::run::TaskId(100, 200, 300, 0, 400), // Specific task ID
      clio::run::kAdminPoolId, clio::run::PoolQuery::Local(), "test_chimod_lib",
      "test_pool_name", clio::run::PoolId(5000, 0),
      nullptr);  // No client for test task

  REQUIRE(!orig_task.IsNull());

  // Record original input values
  clio::run::TaskId orig_task_id = orig_task->task_id_;
  clio::run::PoolId orig_pool_id = orig_task->pool_id_;
  clio::run::MethodId orig_method = orig_task->method_;
  std::string orig_chimod_name = orig_task->chimod_name_.str();
  std::string orig_pool_name = orig_task->pool_name_.str();
  clio::run::PoolId orig_new_pool_id = orig_task->new_pool_id_;

  // Step 2: SaveIn - serialize IN/INOUT parameters
  clio::run::SaveTaskArchive save_in_archive(clio::run::MsgType::kSerializeIn);
  clio::run::shared_ptr<clio::run::Task> orig_task_ptr = orig_task.template Cast<clio::run::Task>();
  container->SaveTask(clio::run::admin::Method::kCreate, save_in_archive,
                      orig_task_ptr);

  // Step 3: LoadIn - deserialize IN/INOUT parameters into new task
  std::string save_in_data = save_in_archive.GetData();
  REQUIRE(!save_in_data.empty());

  clio::run::LoadTaskArchive load_in_archive(save_in_data);
  load_in_archive.msg_type_ = clio::run::MsgType::kSerializeIn;  // Explicitly set msg_type

  // Create task manually (matching pattern from test_task_archive.cc)
  auto loaded_in_task = ipc_manager->NewTask<clio::run::admin::CreateTask>();
  loaded_in_task->chimod_name_ = clio::run::priv::string(CTP_MALLOC, std::string(""));
  loaded_in_task->pool_name_ = clio::run::priv::string(CTP_MALLOC, std::string(""));
  loaded_in_task->chimod_params_ = clio::run::priv::string(CTP_MALLOC, std::string(""));
  loaded_in_task->error_message_ = clio::run::priv::string(CTP_MALLOC, std::string(""));
  load_in_archive >> *loaded_in_task;

  clio::run::shared_ptr<clio::run::Task> loaded_in_task_ptr = loaded_in_task.template Cast<clio::run::Task>();

  REQUIRE(!loaded_in_task_ptr.IsNull());

  // Step 4: Verify loaded task has same inputs (IN fields and task metadata)
  SECTION("Verify IN parameters after LoadIn") {
    // Verify base Task IN fields
    REQUIRE(loaded_in_task->task_id_ == orig_task_id);
    REQUIRE(loaded_in_task->pool_id_ == orig_pool_id);
    REQUIRE(loaded_in_task->method_ == orig_method);

    // Verify CreateTask IN parameters
    REQUIRE(loaded_in_task->chimod_name_.str() == orig_chimod_name);
    REQUIRE(loaded_in_task->pool_name_.str() == orig_pool_name);

    // Verify CreateTask INOUT parameters
    REQUIRE(loaded_in_task->new_pool_id_ == orig_new_pool_id);
  }

  // Step 5: Modify output parameters in loaded task
  loaded_in_task->new_pool_id_ = clio::run::PoolId(7000, 0);
  loaded_in_task->error_message_ =
      clio::run::priv::string(CTP_MALLOC, std::string("test error message from server"));
  loaded_in_task->SetReturnCode(42);

  // Step 6: SaveOut - serialize OUT/INOUT parameters
  clio::run::SaveTaskArchive save_out_archive(clio::run::MsgType::kSerializeOut);
  clio::run::shared_ptr<clio::run::Task> loaded_in_task_ptr2 = loaded_in_task.template Cast<clio::run::Task>();
  container->SaveTask(clio::run::admin::Method::kCreate, save_out_archive,
                      loaded_in_task_ptr2);

  // Step 7: LoadOut - deserialize OUT/INOUT parameters
  std::string save_out_data = save_out_archive.GetData();
  REQUIRE(!save_out_data.empty());

  clio::run::LoadTaskArchive load_out_archive(save_out_data);
  load_out_archive.msg_type_ = clio::run::MsgType::kSerializeOut;  // Explicitly set msg_type

  // Create task manually (matching pattern from test_task_archive.cc)
  auto loaded_out_task = ipc_manager->NewTask<clio::run::admin::CreateTask>();
  loaded_out_task->chimod_name_ = clio::run::priv::string(CTP_MALLOC, std::string(""));
  loaded_out_task->pool_name_ = clio::run::priv::string(CTP_MALLOC, std::string(""));
  loaded_out_task->chimod_params_ = clio::run::priv::string(CTP_MALLOC, std::string(""));
  loaded_out_task->error_message_ = clio::run::priv::string(CTP_MALLOC, std::string(""));
  load_out_archive >> *loaded_out_task;

  REQUIRE(!loaded_out_task.IsNull());

  // Step 8: Verify final task has correct outputs and preserved INOUT
  // parameters
  SECTION("Verify OUT parameters after LoadOut") {
    // Verify INOUT parameter (should be preserved from loaded_in_task)
    REQUIRE(loaded_out_task->new_pool_id_ == clio::run::PoolId(7000, 0));

    // Verify OUT parameters
    REQUIRE(loaded_out_task->error_message_.str() ==
            "test error message from server");

    // Note: return_code_ is in base Task and not serialized by SerializeOut
    // so it won't be in loaded_out_task
  }

  // Cleanup
  orig_task.reset();
  loaded_in_task.reset();
  loaded_out_task.reset();
}

TEST_CASE("SaveTask and LoadTask - Admin FlushTask full flow",
          "[save_load_task][admin][flush]") {
  ClioTestFixture fixture;

  auto *ipc_manager = CLIO_IPC;
  auto alloc = GetTestAllocator();
  (void)alloc;  // Suppress unused variable warning

  // Get container
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto container = pool_manager->GetStaticContainer(clio::run::kAdminPoolId).get();
  REQUIRE(container != nullptr);

  // Step 1: Create original task
  auto orig_task = ipc_manager->NewTask<clio::run::admin::FlushTask>(
      clio::run::TaskId(111, 222, 333, 0, 444), clio::run::kAdminPoolId,
      clio::run::PoolQuery::Local());

  REQUIRE(!orig_task.IsNull());

  // Record original values
  clio::run::TaskId orig_task_id = orig_task->task_id_;
  clio::run::PoolId orig_pool_id = orig_task->pool_id_;
  clio::run::MethodId orig_method = orig_task->method_;

  // Step 2: SaveIn - FlushTask has no IN parameters beyond base Task
  clio::run::SaveTaskArchive save_in_archive(clio::run::MsgType::kSerializeIn);
  clio::run::shared_ptr<clio::run::Task> orig_task_ptr = orig_task.template Cast<clio::run::Task>();
  container->SaveTask(clio::run::admin::Method::kFlush, save_in_archive,
                      orig_task_ptr);

  // Step 3: LoadIn
  std::string save_in_data = save_in_archive.GetData();
  clio::run::LoadTaskArchive load_in_archive(save_in_data);
  load_in_archive.msg_type_ = clio::run::MsgType::kSerializeIn;  // Explicitly set msg_type

  // Create task manually
  auto loaded_in_task = ipc_manager->NewTask<clio::run::admin::FlushTask>();
  load_in_archive >> *loaded_in_task;

  REQUIRE(!loaded_in_task.IsNull());

  // Step 4: Verify base Task fields
  SECTION("Verify base Task IN parameters") {
    REQUIRE(loaded_in_task->task_id_ == orig_task_id);
    REQUIRE(loaded_in_task->pool_id_ == orig_pool_id);
    REQUIRE(loaded_in_task->method_ == orig_method);
  }

  // Step 5: Modify output parameters
  loaded_in_task->total_work_done_ = 12345;

  // Step 6: SaveOut
  clio::run::SaveTaskArchive save_out_archive(clio::run::MsgType::kSerializeOut);
  clio::run::shared_ptr<clio::run::Task> loaded_in_task_ptr2 = loaded_in_task.template Cast<clio::run::Task>();
  container->SaveTask(clio::run::admin::Method::kFlush, save_out_archive,
                      loaded_in_task_ptr2);

  // Step 7: LoadOut
  std::string save_out_data = save_out_archive.GetData();
  clio::run::LoadTaskArchive load_out_archive(save_out_data);
  load_out_archive.msg_type_ = clio::run::MsgType::kSerializeOut;  // Explicitly set msg_type

  // Create task manually
  auto loaded_out_task = ipc_manager->NewTask<clio::run::admin::FlushTask>();
  load_out_archive >> *loaded_out_task;

  REQUIRE(!loaded_out_task.IsNull());

  // Step 8: Verify output parameters
  SECTION("Verify OUT parameters") {
    REQUIRE(loaded_out_task->total_work_done_ == 12345);
  }

  // Cleanup
  orig_task.reset();
  loaded_in_task.reset();
  loaded_out_task.reset();
}

TEST_CASE("SaveTask and LoadTask - Admin SendTask full flow",
          "[save_load_task][admin][send]") {
  ClioTestFixture fixture;

  auto *ipc_manager = CLIO_IPC;
  auto alloc = GetTestAllocator();

  // Get container
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto container = pool_manager->GetStaticContainer(clio::run::kAdminPoolId).get();
  REQUIRE(container != nullptr);

  // Step 1: Create original SendTask (simplified - only transfer_flags parameter)
  auto orig_task = ipc_manager->NewTask<clio::run::admin::SendTask>(
      clio::run::TaskId(555, 666, 777, 0, 888), clio::run::kAdminPoolId,
      clio::run::PoolQuery::Local(),
      123); // transfer_flags IN parameter

  REQUIRE(!orig_task.IsNull());

  // Record original values
  clio::run::TaskId orig_task_id = orig_task->task_id_;
  clio::run::u32 orig_transfer_flags = orig_task->transfer_flags_;

  // Step 2: SaveIn
  clio::run::SaveTaskArchive save_in_archive(clio::run::MsgType::kSerializeIn);
  clio::run::shared_ptr<clio::run::Task> orig_task_ptr = orig_task.template Cast<clio::run::Task>();
  container->SaveTask(clio::run::admin::Method::kSend, save_in_archive,
                      orig_task_ptr);

  // Step 3: LoadIn
  std::string save_in_data = save_in_archive.GetData();
  clio::run::LoadTaskArchive load_in_archive(save_in_data);
  load_in_archive.msg_type_ = clio::run::MsgType::kSerializeIn;  // Explicitly set msg_type

  // Create task manually (default constructor properly initializes error_message_)
  auto loaded_in_task = ipc_manager->NewTask<clio::run::admin::SendTask>();
  load_in_archive >> *loaded_in_task;

  REQUIRE(!loaded_in_task.IsNull());

  // Step 4: Verify IN parameters
  SECTION("Verify IN parameters after LoadIn") {
    REQUIRE(loaded_in_task->task_id_ == orig_task_id);
    REQUIRE(loaded_in_task->transfer_flags_ == orig_transfer_flags);
  }

  // Step 5: Modify output parameters
  loaded_in_task->error_message_ =
      clio::run::priv::string(CTP_MALLOC, std::string("send completed successfully"));

  // Step 6: SaveOut
  clio::run::SaveTaskArchive save_out_archive(clio::run::MsgType::kSerializeOut);
  clio::run::shared_ptr<clio::run::Task> loaded_in_task_ptr2 = loaded_in_task.template Cast<clio::run::Task>();
  container->SaveTask(clio::run::admin::Method::kSend, save_out_archive,
                      loaded_in_task_ptr2);

  // Step 7: LoadOut
  std::string save_out_data = save_out_archive.GetData();
  clio::run::LoadTaskArchive load_out_archive(save_out_data);
  load_out_archive.msg_type_ = clio::run::MsgType::kSerializeOut;  // Explicitly set msg_type

  // Create task manually (default constructor properly initializes error_message_)
  auto loaded_out_task = ipc_manager->NewTask<clio::run::admin::SendTask>();
  load_out_archive >> *loaded_out_task;

  REQUIRE(!loaded_out_task.IsNull());

  // Step 8: Verify OUT parameters
  SECTION("Verify OUT parameters after LoadOut") {
    // Verify OUT parameters
    REQUIRE(loaded_out_task->error_message_.str() ==
            "send completed successfully");
  }

  // Cleanup
  orig_task.reset();
  loaded_in_task.reset();
  loaded_out_task.reset();
}

TEST_CASE("SaveTask and LoadTask - Admin DestroyPoolTask full flow",
          "[save_load_task][admin][destroy]") {
  ClioTestFixture fixture;

  auto *ipc_manager = CLIO_IPC;
  auto alloc = GetTestAllocator();

  // Get container
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto container = pool_manager->GetStaticContainer(clio::run::kAdminPoolId).get();
  REQUIRE(container != nullptr);

  // Step 1: Create original task with IN parameters
  auto orig_task = ipc_manager->NewTask<clio::run::admin::DestroyPoolTask>(
      clio::run::TaskId(11, 22, 33, 0, 44), clio::run::kAdminPoolId,
      clio::run::PoolQuery::Local(),
      clio::run::PoolId(9000, 0), // target_pool_id IN parameter
      456);                 // destruction_flags IN parameter

  REQUIRE(!orig_task.IsNull());

  // Record original values
  clio::run::TaskId orig_task_id = orig_task->task_id_;
  clio::run::PoolId orig_target_pool_id = orig_task->target_pool_id_;
  clio::run::u32 orig_destruction_flags = orig_task->destruction_flags_;

  // Step 2: SaveIn
  clio::run::SaveTaskArchive save_in_archive(clio::run::MsgType::kSerializeIn);
  clio::run::shared_ptr<clio::run::Task> orig_task_ptr = orig_task.template Cast<clio::run::Task>();
  container->SaveTask(clio::run::admin::Method::kDestroyPool, save_in_archive,
                      orig_task_ptr);

  // Step 3: LoadIn
  std::string save_in_data = save_in_archive.GetData();
  clio::run::LoadTaskArchive load_in_archive(save_in_data);
  load_in_archive.msg_type_ = clio::run::MsgType::kSerializeIn;  // Explicitly set msg_type

  // Create task manually (default constructor properly initializes error_message_)
  auto loaded_in_task = ipc_manager->NewTask<clio::run::admin::DestroyPoolTask>();
  load_in_archive >> *loaded_in_task;

  REQUIRE(!loaded_in_task.IsNull());

  // Step 4: Verify IN parameters
  SECTION("Verify IN parameters after LoadIn") {
    REQUIRE(loaded_in_task->task_id_ == orig_task_id);
    REQUIRE(loaded_in_task->target_pool_id_ == orig_target_pool_id);
    REQUIRE(loaded_in_task->destruction_flags_ == orig_destruction_flags);
  }

  // Step 5: Modify output parameters
  loaded_in_task->error_message_ = clio::run::priv::string(CTP_MALLOC, std::string("pool destroyed"));

  // Step 6: SaveOut
  clio::run::SaveTaskArchive save_out_archive(clio::run::MsgType::kSerializeOut);
  clio::run::shared_ptr<clio::run::Task> loaded_in_task_ptr2 = loaded_in_task.template Cast<clio::run::Task>();
  container->SaveTask(clio::run::admin::Method::kDestroyPool, save_out_archive,
                      loaded_in_task_ptr2);

  // Step 7: LoadOut
  std::string save_out_data = save_out_archive.GetData();
  clio::run::LoadTaskArchive load_out_archive(save_out_data);
  load_out_archive.msg_type_ = clio::run::MsgType::kSerializeOut;  // Explicitly set msg_type

  // Create task manually (default constructor properly initializes error_message_)
  auto loaded_out_task = ipc_manager->NewTask<clio::run::admin::DestroyPoolTask>();
  load_out_archive >> *loaded_out_task;

  REQUIRE(!loaded_out_task.IsNull());

  // Step 8: Verify OUT parameters
  SECTION("Verify OUT parameters after LoadOut") {
    REQUIRE(loaded_out_task->error_message_.str() == "pool destroyed");
  }

  // Cleanup
  orig_task.reset();
  loaded_in_task.reset();
  loaded_out_task.reset();
}

// Define main function for test executable
SIMPLE_TEST_MAIN()
