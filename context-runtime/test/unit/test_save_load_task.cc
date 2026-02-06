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

// Include Chimaera headers
#include <chimaera/chimaera.h>
#include <hermes_shm/memory/allocator/malloc_allocator.h>
#include <chimaera/container.h>
#include <chimaera/ipc_manager.h>
#include <chimaera/module_manager.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/task.h>
#include <chimaera/task_archives.h>
#include <chimaera/types.h>

// Include admin tasks for testing
#include <chimaera/admin/admin_client.h>
#include <chimaera/admin/admin_runtime.h>
#include <chimaera/admin/admin_tasks.h>

// Include bdev tasks for testing
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>

using namespace chi;

namespace {
// Helper allocator for tests - uses HSHM_MALLOC for non-IPC allocations
hipc::MallocAllocator* GetTestAllocator() {
  return HSHM_MALLOC;
}

// Initialize Chimaera runtime for tests
class ChimaeraTestFixture {
public:
  ChimaeraTestFixture() {
    // Initialize Chimaera (client with embedded runtime)
    chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

    // Create admin pool
    admin_client_ =
        std::make_unique<chimaera::admin::Client>(chi::kAdminPoolId);
    auto create_task = admin_client_->AsyncCreate(chi::PoolQuery::Local(), "admin",
                                                   chi::kAdminPoolId);
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

  ~ChimaeraTestFixture() {
    // Cleanup
    admin_client_.reset();
  }

  chimaera::admin::Client *GetAdminClient() { return admin_client_.get(); }

private:
  std::unique_ptr<chimaera::admin::Client> admin_client_;
};
} // namespace

TEST_CASE("SaveTask and LoadTask - Admin CreateTask full flow",
          "[save_load_task][admin][create]") {
  ChimaeraTestFixture fixture;

  auto *ipc_manager = CHI_IPC;
  auto alloc = GetTestAllocator();

  // Get container for SaveTask/LoadTask
  auto *pool_manager = CHI_POOL_MANAGER;
  auto *container = pool_manager->GetContainer(chi::kAdminPoolId);
  REQUIRE(container != nullptr);

  // Step 1: Create original task with input parameters
  auto orig_task = ipc_manager->NewTask<chimaera::admin::CreateTask>(
      chi::TaskId(100, 200, 300, 0, 400), // Specific task ID
      chi::kAdminPoolId, chi::PoolQuery::Local(), "test_chimod_lib",
      "test_pool_name", chi::PoolId(5000, 0),
      nullptr);  // No client for test task

  REQUIRE(!orig_task.IsNull());

  // Record original input values
  chi::TaskId orig_task_id = orig_task->task_id_;
  chi::PoolId orig_pool_id = orig_task->pool_id_;
  chi::MethodId orig_method = orig_task->method_;
  std::string orig_chimod_name = orig_task->chimod_name_.str();
  std::string orig_pool_name = orig_task->pool_name_.str();
  chi::PoolId orig_new_pool_id = orig_task->new_pool_id_;

  // Step 2: SaveIn - serialize IN/INOUT parameters
  chi::SaveTaskArchive save_in_archive(chi::MsgType::kSerializeIn);
  hipc::FullPtr<chi::Task> orig_task_ptr = orig_task.template Cast<chi::Task>();
  container->SaveTask(chimaera::admin::Method::kCreate, save_in_archive,
                      orig_task_ptr);

  // Step 3: LoadIn - deserialize IN/INOUT parameters into new task
  std::string save_in_data = save_in_archive.GetData();
  REQUIRE(!save_in_data.empty());

  chi::LoadTaskArchive load_in_archive(save_in_data);
  load_in_archive.msg_type_ = chi::MsgType::kSerializeIn;  // Explicitly set msg_type

  // Create task manually (matching pattern from test_task_archive.cc)
  auto loaded_in_task = ipc_manager->NewTask<chimaera::admin::CreateTask>();
  loaded_in_task->chimod_name_ = chi::priv::string(HSHM_MALLOC, std::string(""));
  loaded_in_task->pool_name_ = chi::priv::string(HSHM_MALLOC, std::string(""));
  loaded_in_task->chimod_params_ = chi::priv::string(HSHM_MALLOC, std::string(""));
  loaded_in_task->error_message_ = chi::priv::string(HSHM_MALLOC, std::string(""));
  load_in_archive >> *loaded_in_task;

  hipc::FullPtr<chi::Task> loaded_in_task_ptr = loaded_in_task.template Cast<chi::Task>();

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
  loaded_in_task->new_pool_id_ = chi::PoolId(7000, 0);
  loaded_in_task->error_message_ =
      chi::priv::string(HSHM_MALLOC, std::string("test error message from server"));
  loaded_in_task->SetReturnCode(42);

  // Step 6: SaveOut - serialize OUT/INOUT parameters
  chi::SaveTaskArchive save_out_archive(chi::MsgType::kSerializeOut);
  hipc::FullPtr<chi::Task> loaded_in_task_ptr2 = loaded_in_task.template Cast<chi::Task>();
  container->SaveTask(chimaera::admin::Method::kCreate, save_out_archive,
                      loaded_in_task_ptr2);

  // Step 7: LoadOut - deserialize OUT/INOUT parameters
  std::string save_out_data = save_out_archive.GetData();
  REQUIRE(!save_out_data.empty());

  chi::LoadTaskArchive load_out_archive(save_out_data);
  load_out_archive.msg_type_ = chi::MsgType::kSerializeOut;  // Explicitly set msg_type

  // Create task manually (matching pattern from test_task_archive.cc)
  auto loaded_out_task = ipc_manager->NewTask<chimaera::admin::CreateTask>();
  loaded_out_task->chimod_name_ = chi::priv::string(HSHM_MALLOC, std::string(""));
  loaded_out_task->pool_name_ = chi::priv::string(HSHM_MALLOC, std::string(""));
  loaded_out_task->chimod_params_ = chi::priv::string(HSHM_MALLOC, std::string(""));
  loaded_out_task->error_message_ = chi::priv::string(HSHM_MALLOC, std::string(""));
  load_out_archive >> *loaded_out_task;

  REQUIRE(!loaded_out_task.IsNull());

  // Step 8: Verify final task has correct outputs and preserved INOUT
  // parameters
  SECTION("Verify OUT parameters after LoadOut") {
    // Verify INOUT parameter (should be preserved from loaded_in_task)
    REQUIRE(loaded_out_task->new_pool_id_ == chi::PoolId(7000, 0));

    // Verify OUT parameters
    REQUIRE(loaded_out_task->error_message_.str() ==
            "test error message from server");

    // Note: return_code_ is in base Task and not serialized by SerializeOut
    // so it won't be in loaded_out_task
  }

  // Cleanup
  ipc_manager->DelTask(orig_task);
  ipc_manager->DelTask(loaded_in_task);
  ipc_manager->DelTask(loaded_out_task);
}

TEST_CASE("SaveTask and LoadTask - Admin FlushTask full flow",
          "[save_load_task][admin][flush]") {
  ChimaeraTestFixture fixture;

  auto *ipc_manager = CHI_IPC;
  auto alloc = GetTestAllocator();
  (void)alloc;  // Suppress unused variable warning

  // Get container
  auto *pool_manager = CHI_POOL_MANAGER;
  auto *container = pool_manager->GetContainer(chi::kAdminPoolId);
  REQUIRE(container != nullptr);

  // Step 1: Create original task
  auto orig_task = ipc_manager->NewTask<chimaera::admin::FlushTask>(
      chi::TaskId(111, 222, 333, 0, 444), chi::kAdminPoolId,
      chi::PoolQuery::Local());

  REQUIRE(!orig_task.IsNull());

  // Record original values
  chi::TaskId orig_task_id = orig_task->task_id_;
  chi::PoolId orig_pool_id = orig_task->pool_id_;
  chi::MethodId orig_method = orig_task->method_;

  // Step 2: SaveIn - FlushTask has no IN parameters beyond base Task
  chi::SaveTaskArchive save_in_archive(chi::MsgType::kSerializeIn);
  hipc::FullPtr<chi::Task> orig_task_ptr = orig_task.template Cast<chi::Task>();
  container->SaveTask(chimaera::admin::Method::kFlush, save_in_archive,
                      orig_task_ptr);

  // Step 3: LoadIn
  std::string save_in_data = save_in_archive.GetData();
  chi::LoadTaskArchive load_in_archive(save_in_data);
  load_in_archive.msg_type_ = chi::MsgType::kSerializeIn;  // Explicitly set msg_type

  // Create task manually
  auto loaded_in_task = ipc_manager->NewTask<chimaera::admin::FlushTask>();
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
  chi::SaveTaskArchive save_out_archive(chi::MsgType::kSerializeOut);
  hipc::FullPtr<chi::Task> loaded_in_task_ptr2 = loaded_in_task.template Cast<chi::Task>();
  container->SaveTask(chimaera::admin::Method::kFlush, save_out_archive,
                      loaded_in_task_ptr2);

  // Step 7: LoadOut
  std::string save_out_data = save_out_archive.GetData();
  chi::LoadTaskArchive load_out_archive(save_out_data);
  load_out_archive.msg_type_ = chi::MsgType::kSerializeOut;  // Explicitly set msg_type

  // Create task manually
  auto loaded_out_task = ipc_manager->NewTask<chimaera::admin::FlushTask>();
  load_out_archive >> *loaded_out_task;

  REQUIRE(!loaded_out_task.IsNull());

  // Step 8: Verify output parameters
  SECTION("Verify OUT parameters") {
    REQUIRE(loaded_out_task->total_work_done_ == 12345);
  }

  // Cleanup
  ipc_manager->DelTask(orig_task);
  ipc_manager->DelTask(loaded_in_task);
  ipc_manager->DelTask(loaded_out_task);
}

TEST_CASE("SaveTask and LoadTask - Admin SendTask full flow",
          "[save_load_task][admin][send]") {
  ChimaeraTestFixture fixture;

  auto *ipc_manager = CHI_IPC;
  auto alloc = GetTestAllocator();

  // Get container
  auto *pool_manager = CHI_POOL_MANAGER;
  auto *container = pool_manager->GetContainer(chi::kAdminPoolId);
  REQUIRE(container != nullptr);

  // Step 1: Create original SendTask (simplified - only transfer_flags parameter)
  auto orig_task = ipc_manager->NewTask<chimaera::admin::SendTask>(
      chi::TaskId(555, 666, 777, 0, 888), chi::kAdminPoolId,
      chi::PoolQuery::Local(),
      123); // transfer_flags IN parameter

  REQUIRE(!orig_task.IsNull());

  // Record original values
  chi::TaskId orig_task_id = orig_task->task_id_;
  chi::u32 orig_transfer_flags = orig_task->transfer_flags_;

  // Step 2: SaveIn
  chi::SaveTaskArchive save_in_archive(chi::MsgType::kSerializeIn);
  hipc::FullPtr<chi::Task> orig_task_ptr = orig_task.template Cast<chi::Task>();
  container->SaveTask(chimaera::admin::Method::kSend, save_in_archive,
                      orig_task_ptr);

  // Step 3: LoadIn
  std::string save_in_data = save_in_archive.GetData();
  chi::LoadTaskArchive load_in_archive(save_in_data);
  load_in_archive.msg_type_ = chi::MsgType::kSerializeIn;  // Explicitly set msg_type

  // Create task manually (default constructor properly initializes error_message_)
  auto loaded_in_task = ipc_manager->NewTask<chimaera::admin::SendTask>();
  load_in_archive >> *loaded_in_task;

  REQUIRE(!loaded_in_task.IsNull());

  // Step 4: Verify IN parameters
  SECTION("Verify IN parameters after LoadIn") {
    REQUIRE(loaded_in_task->task_id_ == orig_task_id);
    REQUIRE(loaded_in_task->transfer_flags_ == orig_transfer_flags);
  }

  // Step 5: Modify output parameters
  loaded_in_task->error_message_ =
      chi::priv::string(HSHM_MALLOC, std::string("send completed successfully"));

  // Step 6: SaveOut
  chi::SaveTaskArchive save_out_archive(chi::MsgType::kSerializeOut);
  hipc::FullPtr<chi::Task> loaded_in_task_ptr2 = loaded_in_task.template Cast<chi::Task>();
  container->SaveTask(chimaera::admin::Method::kSend, save_out_archive,
                      loaded_in_task_ptr2);

  // Step 7: LoadOut
  std::string save_out_data = save_out_archive.GetData();
  chi::LoadTaskArchive load_out_archive(save_out_data);
  load_out_archive.msg_type_ = chi::MsgType::kSerializeOut;  // Explicitly set msg_type

  // Create task manually (default constructor properly initializes error_message_)
  auto loaded_out_task = ipc_manager->NewTask<chimaera::admin::SendTask>();
  load_out_archive >> *loaded_out_task;

  REQUIRE(!loaded_out_task.IsNull());

  // Step 8: Verify OUT parameters
  SECTION("Verify OUT parameters after LoadOut") {
    // Verify OUT parameters
    REQUIRE(loaded_out_task->error_message_.str() ==
            "send completed successfully");
  }

  // Cleanup
  ipc_manager->DelTask(orig_task);
  ipc_manager->DelTask(loaded_in_task);
  ipc_manager->DelTask(loaded_out_task);
}

TEST_CASE("SaveTask and LoadTask - Admin DestroyPoolTask full flow",
          "[save_load_task][admin][destroy]") {
  ChimaeraTestFixture fixture;

  auto *ipc_manager = CHI_IPC;
  auto alloc = GetTestAllocator();

  // Get container
  auto *pool_manager = CHI_POOL_MANAGER;
  auto *container = pool_manager->GetContainer(chi::kAdminPoolId);
  REQUIRE(container != nullptr);

  // Step 1: Create original task with IN parameters
  auto orig_task = ipc_manager->NewTask<chimaera::admin::DestroyPoolTask>(
      chi::TaskId(11, 22, 33, 0, 44), chi::kAdminPoolId,
      chi::PoolQuery::Local(),
      chi::PoolId(9000, 0), // target_pool_id IN parameter
      456);                 // destruction_flags IN parameter

  REQUIRE(!orig_task.IsNull());

  // Record original values
  chi::TaskId orig_task_id = orig_task->task_id_;
  chi::PoolId orig_target_pool_id = orig_task->target_pool_id_;
  chi::u32 orig_destruction_flags = orig_task->destruction_flags_;

  // Step 2: SaveIn
  chi::SaveTaskArchive save_in_archive(chi::MsgType::kSerializeIn);
  hipc::FullPtr<chi::Task> orig_task_ptr = orig_task.template Cast<chi::Task>();
  container->SaveTask(chimaera::admin::Method::kDestroyPool, save_in_archive,
                      orig_task_ptr);

  // Step 3: LoadIn
  std::string save_in_data = save_in_archive.GetData();
  chi::LoadTaskArchive load_in_archive(save_in_data);
  load_in_archive.msg_type_ = chi::MsgType::kSerializeIn;  // Explicitly set msg_type

  // Create task manually (default constructor properly initializes error_message_)
  auto loaded_in_task = ipc_manager->NewTask<chimaera::admin::DestroyPoolTask>();
  load_in_archive >> *loaded_in_task;

  REQUIRE(!loaded_in_task.IsNull());

  // Step 4: Verify IN parameters
  SECTION("Verify IN parameters after LoadIn") {
    REQUIRE(loaded_in_task->task_id_ == orig_task_id);
    REQUIRE(loaded_in_task->target_pool_id_ == orig_target_pool_id);
    REQUIRE(loaded_in_task->destruction_flags_ == orig_destruction_flags);
  }

  // Step 5: Modify output parameters
  loaded_in_task->error_message_ = chi::priv::string(HSHM_MALLOC, std::string("pool destroyed"));

  // Step 6: SaveOut
  chi::SaveTaskArchive save_out_archive(chi::MsgType::kSerializeOut);
  hipc::FullPtr<chi::Task> loaded_in_task_ptr2 = loaded_in_task.template Cast<chi::Task>();
  container->SaveTask(chimaera::admin::Method::kDestroyPool, save_out_archive,
                      loaded_in_task_ptr2);

  // Step 7: LoadOut
  std::string save_out_data = save_out_archive.GetData();
  chi::LoadTaskArchive load_out_archive(save_out_data);
  load_out_archive.msg_type_ = chi::MsgType::kSerializeOut;  // Explicitly set msg_type

  // Create task manually (default constructor properly initializes error_message_)
  auto loaded_out_task = ipc_manager->NewTask<chimaera::admin::DestroyPoolTask>();
  load_out_archive >> *loaded_out_task;

  REQUIRE(!loaded_out_task.IsNull());

  // Step 8: Verify OUT parameters
  SECTION("Verify OUT parameters after LoadOut") {
    REQUIRE(loaded_out_task->error_message_.str() == "pool destroyed");
  }

  // Cleanup
  ipc_manager->DelTask(orig_task);
  ipc_manager->DelTask(loaded_in_task);
  ipc_manager->DelTask(loaded_out_task);
}

// Define main function for test executable
SIMPLE_TEST_MAIN()
