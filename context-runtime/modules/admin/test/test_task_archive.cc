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
 * Comprehensive unit tests for Task Archive Serialization System
 *
 * Tests the complete task archive functionality including:
 * - SaveTaskArchive and LoadTaskArchive (unified archives)
 * - NetTaskArchive base class
 * - Task serialization/deserialization with BaseSerialize methods
 * - Container Save/Load methods
 * - Bulk transfer support
 * - Various task types from admin module
 */

#include "simple_test.h"
#include <memory>
#include <string>
#include <vector>
#include <clio_ctp/util/logging.h>

// Include CLIO Runtime headers
#include <clio_runtime/clio_runtime.h>
#include <clio_ctp/memory/allocator/malloc_allocator.h>
#include <clio_runtime/container.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/task.h>
#include <clio_runtime/task_archives.h>
#include <clio_runtime/types.h>

// Include admin tasks for testing concrete task types
#include <clio_runtime/admin/admin_tasks.h>

// Include GlobalSerialize for serialization
#include "clio_ctp/data_structures/serialization/global_serialize.h"

using namespace clio::run;

// Test constants
constexpr clio::run::u32 kTestWriteFlag = 0x1;  // BULK_XFER
constexpr clio::run::u32 kTestExposeFlag = 0x2; // BULK_EXPOSE

// Helper allocator for tests - uses CTP_MALLOC for non-IPC allocations
ctp::ipc::MallocAllocator* GetTestAllocator() {
  return CTP_MALLOC;
}

// Helper to create test task with sample data
std::unique_ptr<clio::run::Task> CreateTestTask() {
  auto task = std::make_unique<clio::run::Task>(clio::run::TaskId(1, 1, 1, 0, 1),
                                          clio::run::PoolId(100, 0), clio::run::PoolQuery(),
                                          clio::run::MethodId(42));
  task->period_ns_ = 1000000.0; // 1ms
  task->task_flags_.SetBits(0x10);
  return task;
}

// Helper to create test admin task with sample data
std::unique_ptr<clio::run::admin::CreateTask> CreateTestAdminTask() {
  auto alloc = GetTestAllocator();
  auto task = std::make_unique<clio::run::admin::CreateTask>(
      clio::run::TaskId(2, 2, 2, 0, 2), clio::run::PoolId(200, 0), clio::run::PoolQuery(),
      "test_chimod", "test_pool", clio::run::PoolId(300, 0),
      nullptr);  // No client for test task
  task->return_code_ = 42;
  task->error_message_ = clio::run::priv::string(CTP_MALLOC, "test error message");
  return task;
}

// Test data structure for non-Task serialization
struct TestData {
  int value;
  std::string text;
  std::vector<double> numbers;

  template <class Archive> void serialize(Archive &ar) {
    ar(value, text, numbers);
  }

  bool operator==(const TestData &other) const {
    return value == other.value && text == other.text &&
           numbers == other.numbers;
  }
};

// Create test data
TestData CreateTestData() {
  return TestData{42, "hello world", {1.5, 2.7, 3.14159}};
}

//==============================================================================
// NetTaskArchive Base Class Tests
//==============================================================================

TEST_CASE("NetTaskArchive - Base Class Functionality",
          "[task_archive][net_task_archive]") {
  SECTION("SaveTaskArchive inherits from NetTaskArchive") {
    clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeIn);

    // Test NetTaskArchive methods
    REQUIRE(archive.GetMsgType() == clio::run::MsgType::kSerializeIn);
    REQUIRE(archive.GetTaskInfos().empty());
    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);
  }

  SECTION("LoadTaskArchive inherits from NetTaskArchive") {
    clio::run::LoadTaskArchive archive;

    // Test NetTaskArchive methods
    REQUIRE(archive.GetMsgType() == clio::run::MsgType::kSerializeIn);
    REQUIRE(archive.GetTaskInfos().empty());
    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);
  }

  SECTION("SaveTaskArchive with different message types") {
    clio::run::SaveTaskArchive archive_in(clio::run::MsgType::kSerializeIn);
    clio::run::SaveTaskArchive archive_out(clio::run::MsgType::kSerializeOut);
    clio::run::SaveTaskArchive archive_hb(clio::run::MsgType::kHeartbeat);

    REQUIRE(archive_in.GetMsgType() == clio::run::MsgType::kSerializeIn);
    REQUIRE(archive_out.GetMsgType() == clio::run::MsgType::kSerializeOut);
    REQUIRE(archive_hb.GetMsgType() == clio::run::MsgType::kHeartbeat);
  }
}

//==============================================================================
// SaveTaskArchive Tests
//==============================================================================

TEST_CASE("SaveTaskArchive - Basic Construction and Data Retrieval",
          "[task_archive][save_archive]") {
  SECTION("Construction with message type") {
    clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeIn);

    // Should start with empty bulk transfers
    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);

    // Data should be empty initially
    std::string data = archive.GetData();
    REQUIRE(data.empty());
  }

  SECTION("Serializing simple data") {
    clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeIn);
    int test_value = 42;

    REQUIRE_NOTHROW(archive << test_value);

    std::string data = archive.GetData();
    REQUIRE_FALSE(data.empty());
  }
}

//==============================================================================
// LoadTaskArchive Tests
//==============================================================================

TEST_CASE("LoadTaskArchive - Basic Construction",
          "[task_archive][load_archive]") {
  SECTION("Default construction") {
    clio::run::LoadTaskArchive archive;

    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);
  }

  SECTION("Construction from string") {
    std::string test_data = "test serialized data";
    clio::run::LoadTaskArchive archive(test_data);

    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);
  }

  SECTION("Construction from const char* and size") {
    const char *test_data = "test data";
    size_t size = strlen(test_data);
    clio::run::LoadTaskArchive archive(test_data, size);

    REQUIRE(archive.GetSendBulkCount() == 0);
  }
}

//==============================================================================
// Bulk Transfer Tests
//==============================================================================

TEST_CASE("Bulk Transfer Recording", "[task_archive][bulk_transfer]") {
  SECTION("SaveTaskArchive bulk transfer recording") {
    clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeIn);

    ctp::ipc::ShmPtr<> test_ptr = ctp::ipc::ShmPtr<>::GetNull();
    size_t test_size = 1024;
    uint32_t test_flags = kTestWriteFlag;

    REQUIRE_NOTHROW(archive.bulk(test_ptr, test_size, test_flags));

    // Bulk should be added to send vector
    REQUIRE(archive.GetSendBulkCount() == 1);
    REQUIRE(archive.send[0].size == test_size);
    REQUIRE(archive.send[0].flags.bits_ == test_flags);
  }

  SECTION("Multiple bulk transfers") {
    clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeIn);

    // Add multiple bulk transfers
    archive.bulk(ctp::ipc::ShmPtr<>::GetNull(), 100, kTestWriteFlag);
    archive.bulk(ctp::ipc::ShmPtr<>::GetNull(), 200, kTestExposeFlag);
    archive.bulk(ctp::ipc::ShmPtr<>::GetNull(), 300, kTestWriteFlag | kTestExposeFlag);

    REQUIRE(archive.GetSendBulkCount() == 3);
    REQUIRE(archive.send[0].size == 100);
    REQUIRE(archive.send[1].size == 200);
    REQUIRE(archive.send[2].size == 300);
    REQUIRE(archive.send[2].flags.bits_ == (kTestWriteFlag | kTestExposeFlag));
  }
}

//==============================================================================
// Non-Task Object Serialization Tests
//==============================================================================

TEST_CASE("Non-Task Object Serialization", "[task_archive][non_task]") {
  SECTION("Round-trip serialization of custom struct") {
    TestData original = CreateTestData();

    // Serialize using SaveTaskArchive
    clio::run::SaveTaskArchive out_archive(clio::run::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive << original);
    std::string serialized_data = out_archive.GetData();

    // Deserialize using LoadTaskArchive
    clio::run::LoadTaskArchive in_archive(serialized_data);
    TestData deserialized;
    REQUIRE_NOTHROW(in_archive >> deserialized);

    // Verify data integrity
    REQUIRE(deserialized == original);
  }

  SECTION("Bidirectional operator() for multiple values") {
    std::string str1 = "hello";
    int int1 = 42;
    double double1 = 3.14159;

    // Serialize using operator()
    clio::run::SaveTaskArchive out_archive(clio::run::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive(str1, int1, double1));
    std::string serialized_data = out_archive.GetData();

    // Deserialize using operator()
    clio::run::LoadTaskArchive in_archive(serialized_data);
    std::string str2;
    int int2;
    double double2;
    REQUIRE_NOTHROW(in_archive(str2, int2, double2));

    // Verify data
    REQUIRE(str1 == str2);
    REQUIRE(int1 == int2);
    REQUIRE(double1 == double2);
  }
}

//==============================================================================
// Task Base Class Serialization Tests
//==============================================================================

TEST_CASE("Task Base Class Serialization", "[task_archive][task_base]") {
  SECTION("Task SerializeIn round-trip") {
    auto original_task = CreateTestTask();

    // Serialize using SaveTaskArchive with kSerializeIn mode
    clio::run::SaveTaskArchive out_archive(clio::run::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive << *original_task);
    std::string serialized_data = out_archive.GetData();

    // Verify task info was recorded
    REQUIRE(out_archive.GetTaskInfos().size() == 1);
    REQUIRE(out_archive.GetTaskInfos()[0].task_id_ == original_task->task_id_);
    REQUIRE(out_archive.GetTaskInfos()[0].pool_id_ == original_task->pool_id_);
    REQUIRE(out_archive.GetTaskInfos()[0].method_id_ == original_task->method_);

    // Deserialize using LoadTaskArchive with kSerializeIn mode
    clio::run::LoadTaskArchive in_archive(serialized_data);
    in_archive.msg_type_ = clio::run::MsgType::kSerializeIn;
    auto new_task = CreateTestTask();
    new_task->SetNull(); // Clear data to ensure deserialization works
    REQUIRE_NOTHROW(in_archive >> *new_task);

    // Verify base task fields were preserved
    REQUIRE(new_task->pool_id_ == original_task->pool_id_);
    REQUIRE(new_task->task_id_ == original_task->task_id_);
    REQUIRE(new_task->method_ == original_task->method_);
    REQUIRE(new_task->period_ns_ == original_task->period_ns_);
    REQUIRE(new_task->task_flags_.bits_.load() ==
            original_task->task_flags_.bits_.load());
  }

  SECTION("Task SerializeOut round-trip") {
    auto original_task = CreateTestTask();
    original_task->return_code_ = 42; // Set a return code (OUT parameter)

    // Serialize using SaveTaskArchive with kSerializeOut mode
    clio::run::SaveTaskArchive out_archive(clio::run::MsgType::kSerializeOut);
    REQUIRE_NOTHROW(out_archive << *original_task);
    std::string serialized_data = out_archive.GetData();

    // Deserialize using LoadTaskArchive with kSerializeOut mode
    clio::run::LoadTaskArchive in_archive(serialized_data);
    in_archive.msg_type_ = clio::run::MsgType::kSerializeOut;
    auto new_task = CreateTestTask();
    new_task->return_code_ = 0; // Clear return code
    REQUIRE_NOTHROW(in_archive >> *new_task);

    // Verify OUT parameters were preserved
    REQUIRE(new_task->return_code_ == original_task->return_code_);
  }
}

//==============================================================================
// Admin Task Serialization Tests
//==============================================================================

TEST_CASE("Admin Task Serialization", "[task_archive][admin_tasks]") {
  SECTION("CreateTask SerializeIn/SerializeOut") {
    auto original_task = CreateTestAdminTask();

    // Test IN parameter serialization
    clio::run::SaveTaskArchive out_archive_in(clio::run::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive_in << *original_task);
    std::string in_data = out_archive_in.GetData();

    clio::run::LoadTaskArchive in_archive_in(in_data);
    in_archive_in.msg_type_ = clio::run::MsgType::kSerializeIn;
    auto new_task_in = CreateTestAdminTask();
    new_task_in->chimod_name_ = clio::run::priv::string(CTP_MALLOC, ""); // Clear
    new_task_in->pool_name_ = clio::run::priv::string(CTP_MALLOC, "");
    REQUIRE_NOTHROW(in_archive_in >> *new_task_in);

    // Verify IN/INOUT parameters were preserved
    REQUIRE(new_task_in->chimod_name_.str() ==
            original_task->chimod_name_.str());
    REQUIRE(new_task_in->pool_name_.str() == original_task->pool_name_.str());
    REQUIRE(new_task_in->pool_id_ == original_task->pool_id_);

    // Test OUT parameter serialization
    clio::run::SaveTaskArchive out_archive_out(clio::run::MsgType::kSerializeOut);
    REQUIRE_NOTHROW(out_archive_out << *original_task);
    std::string out_data = out_archive_out.GetData();

    clio::run::LoadTaskArchive in_archive_out(out_data);
    in_archive_out.msg_type_ = clio::run::MsgType::kSerializeOut;
    auto new_task_out = CreateTestAdminTask();
    new_task_out->return_code_ = 0; // Clear
    new_task_out->error_message_ = clio::run::priv::string(CTP_MALLOC, "");
    REQUIRE_NOTHROW(in_archive_out >> *new_task_out);

    // Verify OUT/INOUT parameters were preserved
    REQUIRE(new_task_out->return_code_ == original_task->return_code_);
    REQUIRE(new_task_out->error_message_.str() ==
            original_task->error_message_.str());
    REQUIRE(new_task_out->pool_id_ ==
            original_task->pool_id_); // INOUT parameter
  }

  SECTION("DestroyPoolTask serialization") {
    auto alloc = GetTestAllocator();
    clio::run::admin::DestroyPoolTask original_task(
        clio::run::TaskId(3, 3, 3, 0, 3), clio::run::PoolId(400, 0),
        clio::run::PoolQuery(), clio::run::PoolId(500, 0), 0x123);
    original_task.return_code_ = 99;
    original_task.error_message_ = clio::run::priv::string(CTP_MALLOC, "destroy error");

    // Test round-trip IN parameters
    clio::run::SaveTaskArchive out_archive_in(clio::run::MsgType::kSerializeIn);
    out_archive_in << original_task;

    clio::run::LoadTaskArchive in_archive_in(out_archive_in.GetData());
    in_archive_in.msg_type_ = clio::run::MsgType::kSerializeIn;
    clio::run::admin::DestroyPoolTask new_task_in;
    in_archive_in >> new_task_in;

    REQUIRE(new_task_in.target_pool_id_ == original_task.target_pool_id_);
    REQUIRE(new_task_in.destruction_flags_ == original_task.destruction_flags_);

    // Test round-trip OUT parameters
    clio::run::SaveTaskArchive out_archive_out(clio::run::MsgType::kSerializeOut);
    out_archive_out << original_task;

    clio::run::LoadTaskArchive in_archive_out(out_archive_out.GetData());
    in_archive_out.msg_type_ = clio::run::MsgType::kSerializeOut;
    clio::run::admin::DestroyPoolTask new_task_out;
    in_archive_out >> new_task_out;

    REQUIRE(new_task_out.return_code_ == original_task.return_code_);
    REQUIRE(new_task_out.error_message_.str() ==
            original_task.error_message_.str());
  }

  SECTION("StopRuntimeTask serialization") {
    auto alloc = GetTestAllocator();
    clio::run::admin::StopRuntimeTask original_task(
        clio::run::TaskId(4, 4, 4, 0, 4), clio::run::PoolId(600, 0),
        clio::run::PoolQuery(), 0x456, 10000);
    original_task.return_code_ = 777;
    original_task.error_message_ = clio::run::priv::string(CTP_MALLOC, "shutdown error");

    // Test IN parameters
    clio::run::SaveTaskArchive out_archive_in(clio::run::MsgType::kSerializeIn);
    out_archive_in << original_task;

    clio::run::LoadTaskArchive in_archive_in(out_archive_in.GetData());
    in_archive_in.msg_type_ = clio::run::MsgType::kSerializeIn;
    clio::run::admin::StopRuntimeTask new_task_in;
    in_archive_in >> new_task_in;

    REQUIRE(new_task_in.shutdown_flags_ == original_task.shutdown_flags_);
    REQUIRE(new_task_in.grace_period_ms_ == original_task.grace_period_ms_);

    // Test OUT parameters
    clio::run::SaveTaskArchive out_archive_out(clio::run::MsgType::kSerializeOut);
    out_archive_out << original_task;

    clio::run::LoadTaskArchive in_archive_out(out_archive_out.GetData());
    in_archive_out.msg_type_ = clio::run::MsgType::kSerializeOut;
    clio::run::admin::StopRuntimeTask new_task_out;
    in_archive_out >> new_task_out;

    REQUIRE(new_task_out.return_code_ == original_task.return_code_);
    REQUIRE(new_task_out.error_message_.str() ==
            original_task.error_message_.str());
  }
}

//==============================================================================
// Archive Bidirectional Operator Tests
//==============================================================================

TEST_CASE("Archive Operator() Bidirectional Functionality",
          "[task_archive][bidirectional]") {
  SECTION("LoadTaskArchive operator() acts as input") {
    // Create test data
    int value1 = 42;
    std::string value2 = "test string";
    double value3 = 3.14159;

    // Serialize with GlobalSerialize
    std::vector<char> buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> out_archive(buf);
      out_archive(value1, value2, value3);
      out_archive.Finalize();
    }

    // Deserialize with LoadTaskArchive using operator()
    clio::run::LoadTaskArchive in_archive(std::string(buf.begin(), buf.end()));
    int result1;
    std::string result2;
    double result3;
    REQUIRE_NOTHROW(in_archive(result1, result2, result3));

    REQUIRE(result1 == value1);
    REQUIRE(result2 == value2);
    REQUIRE(result3 == value3);
  }

  SECTION("SaveTaskArchive operator() acts as output") {
    int value1 = 123;
    std::string value2 = "output test";
    double value3 = 2.71828;

    // Serialize with SaveTaskArchive using operator()
    clio::run::SaveTaskArchive out_archive(clio::run::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive(value1, value2, value3));

    // Deserialize using LoadTaskArchive
    clio::run::LoadTaskArchive in_archive(out_archive.GetData());
    int result1;
    std::string result2;
    double result3;
    in_archive(result1, result2, result3);

    REQUIRE(result1 == value1);
    REQUIRE(result2 == value2);
    REQUIRE(result3 == value3);
  }
}

//==============================================================================
// Container Serialization Method Tests
//==============================================================================

// Test container class that implements all pure virtual methods
class TestContainer : public clio::run::Container {
public:
  clio::run::u64 GetWorkRemaining() const override {
    return 0; // Test implementation returns no work
  }

  clio::run::TaskResume Run(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task_ptr) override {
    // Test implementation - do nothing
    (void)method;
    (void)task_ptr;
    CLIO_TASK_BODY_BEGIN
    CLIO_CO_RETURN;
    CLIO_TASK_BODY_END
  }

  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override {
    // Test implementation - just call task serialization
    (void)method;
    archive << *task_ptr;
  }

  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override {
    // Test implementation - just call task deserialization
    (void)method;
    archive >> *task_ptr;
  }

  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive) override {
    // Test implementation - allocate task and call LoadTask
    clio::run::shared_ptr<clio::run::Task> task_ptr = NewTask(method);
    if (!task_ptr.IsNull()) {
      LoadTask(method, archive, task_ptr);
    }
    return task_ptr;
  }

  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
                                        bool deep) override {
    // Test implementation - create new task and copy
    (void)method;
    (void)deep;
    auto *ipc_manager = CLIO_IPC;
    if (ipc_manager) {
      auto new_task_ptr = ipc_manager->NewTask<clio::run::Task>();
      if (!new_task_ptr.IsNull()) {
        new_task_ptr->Copy(orig_task_ptr);
      }
      return new_task_ptr;
    }
    return clio::run::shared_ptr<clio::run::Task>();
  }

  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override {
    // Test implementation - create a basic Task
    (void)method;
    auto *ipc_manager = CLIO_IPC;
    if (ipc_manager) {
      return ipc_manager->NewTask<clio::run::Task>();
    }
    return clio::run::shared_ptr<clio::run::Task>();
  }

  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override {
    // Test implementation - do nothing for now
    (void)method;
    (void)archive;
    (void)task_ptr;
  }

  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override {
    // Test implementation - allocate task and call LocalLoadTask
    clio::run::shared_ptr<clio::run::Task> task_ptr = NewTask(method);
    if (!task_ptr.IsNull()) {
      LocalLoadTask(method, archive, task_ptr);
    }
    return task_ptr;
  }

  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override {
    // Test implementation - do nothing
    (void)method;
    (void)archive;
    (void)task_ptr;
  }

  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                 const clio::run::shared_ptr<clio::run::Task>& replica_task) override {
    (void)method;
    orig_task->AggregateOut(replica_task);
  }

  // DelTask removed: tasks are clio::run::shared_ptr handles freed via RAII.
};

TEST_CASE("Container Serialization Methods", "[task_archive][container]") {
  SECTION("Container SaveTask/LoadTask with SerializeIn mode") {
    TestContainer container;
    auto original_task = CreateTestTask();
    auto task_ptr = clio::run::shared_ptr<clio::run::Task>::WrapNonOwning(original_task.get());

    // Test SaveTask with SerializeIn mode (inputs)
    clio::run::SaveTaskArchive save_archive(clio::run::MsgType::kSerializeIn);
    clio::run::u32 method = task_ptr->method_;
    REQUIRE_NOTHROW(container.SaveTask(method, save_archive, task_ptr));
    std::string serialized_data = save_archive.GetData();
    REQUIRE_FALSE(serialized_data.empty());

    // Verify task info was recorded
    REQUIRE(save_archive.GetTaskInfos().size() == 1);

    // Test AllocLoadTask with SerializeIn mode (inputs)
    clio::run::LoadTaskArchive load_archive(serialized_data);
    load_archive.msg_type_ = clio::run::MsgType::kSerializeIn;
    clio::run::shared_ptr<clio::run::Task> new_task_ptr;
    REQUIRE_NOTHROW(new_task_ptr = container.AllocLoadTask(method, load_archive));
  }

  SECTION("Container SaveTask/LoadTask with SerializeOut mode") {
    TestContainer container;
    auto original_task = CreateTestTask();
    auto task_ptr = clio::run::shared_ptr<clio::run::Task>::WrapNonOwning(original_task.get());

    // Test SaveTask with SerializeOut mode (outputs)
    clio::run::SaveTaskArchive save_archive(clio::run::MsgType::kSerializeOut);
    clio::run::u32 method = task_ptr->method_;
    REQUIRE_NOTHROW(container.SaveTask(method, save_archive, task_ptr));
    std::string serialized_data = save_archive.GetData();
    REQUIRE_FALSE(serialized_data.empty());

    // Test AllocLoadTask with SerializeOut mode (outputs)
    clio::run::LoadTaskArchive load_archive(serialized_data);
    load_archive.msg_type_ = clio::run::MsgType::kSerializeOut;
    clio::run::shared_ptr<clio::run::Task> new_task_ptr;
    REQUIRE_NOTHROW(new_task_ptr = container.AllocLoadTask(method, load_archive));
  }
}

//==============================================================================
// Error Handling and Edge Cases
//==============================================================================

TEST_CASE("Error Handling and Edge Cases", "[task_archive][error_handling]") {
  SECTION("Invalid serialization data") {
    std::string invalid_data = "this is not valid serialized data";
    clio::run::LoadTaskArchive archive(invalid_data);

    // Archive should be constructed without throwing
    REQUIRE_NOTHROW(archive.GetSendBulkCount());
  }

  SECTION("Empty serialization data") {
    std::string empty_data = "";
    clio::run::LoadTaskArchive archive(empty_data);

    REQUIRE(archive.GetSendBulkCount() == 0);
  }

  SECTION("Bulk transfer with null pointer") {
    clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeIn);
    ctp::ipc::ShmPtr<> null_ptr = ctp::ipc::ShmPtr<>::GetNull();

    REQUIRE_NOTHROW(archive.bulk(null_ptr, 0, 0));

    REQUIRE(archive.GetSendBulkCount() == 1);
    REQUIRE(archive.send[0].size == 0);
    REQUIRE(archive.send[0].flags.bits_ == 0);
  }
}

//==============================================================================
// Performance and Large Data Tests
//==============================================================================

TEST_CASE("Performance and Large Data", "[task_archive][performance]") {
  SECTION("Large string serialization") {
    std::string large_string(10000, 'X'); // 10KB string

    clio::run::SaveTaskArchive out_archive(clio::run::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive << large_string);

    std::string serialized_data = out_archive.GetData();
    REQUIRE(serialized_data.size() > large_string.size()); // Should include serialization overhead

    clio::run::LoadTaskArchive in_archive(serialized_data);
    std::string result_string;
    REQUIRE_NOTHROW(in_archive >> result_string);

    REQUIRE(result_string == large_string);
  }

  SECTION("Large vector serialization") {
    std::vector<double> large_vector(1000, 3.14159); // 1000 doubles

    clio::run::SaveTaskArchive out_archive(clio::run::MsgType::kSerializeIn);
    out_archive << large_vector;

    clio::run::LoadTaskArchive in_archive(out_archive.GetData());
    std::vector<double> result_vector;
    in_archive >> result_vector;

    REQUIRE(result_vector.size() == large_vector.size());
    REQUIRE(result_vector == large_vector);
  }

  SECTION("Multiple task serialization sequence") {
    // Test serializing multiple tasks in sequence
    std::vector<std::string> serialized_tasks;

    for (int i = 0; i < 10; ++i) {
      // Create tasks with unique TaskIds
      auto task = std::make_unique<clio::run::Task>(clio::run::TaskId(1, 1, 1, 0, static_cast<clio::run::u32>(i)),
                                              clio::run::PoolId(100, 0), clio::run::PoolQuery(),
                                              clio::run::MethodId(42));
      task->period_ns_ = 1000000.0 + i; // Vary the period to make tasks different

      clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeIn);
      archive << *task;
      serialized_tasks.push_back(archive.GetData());
    }

    // Verify all tasks were serialized uniquely
    REQUIRE(serialized_tasks.size() == 10);
    for (size_t i = 0; i < serialized_tasks.size(); ++i) {
      REQUIRE_FALSE(serialized_tasks[i].empty());
      // Each should be different due to different task_id_.unique_ and period_ns_
      for (size_t j = i + 1; j < serialized_tasks.size(); ++j) {
        REQUIRE(serialized_tasks[i] != serialized_tasks[j]);
      }
    }
  }
}

//==============================================================================
// Complete Serialization Flow Test
//==============================================================================

TEST_CASE("Complete Serialization Flow", "[task_archive][integration]") {
  SECTION("Complete round-trip flow for admin CreateTask") {
    auto original_task = CreateTestAdminTask();

    // Step 1: Serialize IN parameters (for sending task to remote node)
    clio::run::SaveTaskArchive in_archive(clio::run::MsgType::kSerializeIn);
    in_archive << *original_task;
    std::string in_data = in_archive.GetData();

    // Verify task info and bulk counts
    REQUIRE(in_archive.GetTaskInfos().size() == 1);

    // Step 2: Simulate remote node receiving and deserializing IN parameters
    clio::run::LoadTaskArchive recv_in_archive(in_data);
    recv_in_archive.msg_type_ = clio::run::MsgType::kSerializeIn;
    auto remote_task = CreateTestAdminTask();
    remote_task->chimod_name_ = clio::run::priv::string(CTP_MALLOC, ""); // Clear
    remote_task->pool_name_ = clio::run::priv::string(CTP_MALLOC, "");
    recv_in_archive >> *remote_task;

    // Verify IN parameters were transferred
    REQUIRE(remote_task->chimod_name_.str() ==
            original_task->chimod_name_.str());
    REQUIRE(remote_task->pool_name_.str() == original_task->pool_name_.str());
    REQUIRE(remote_task->pool_id_ == original_task->pool_id_);

    // Step 3: Simulate task execution and result generation on remote node
    remote_task->return_code_ = 123;
    remote_task->error_message_ =
        clio::run::priv::string(CTP_MALLOC, "remote execution result");

    // Step 4: Serialize OUT parameters (for sending results back)
    clio::run::SaveTaskArchive out_archive(clio::run::MsgType::kSerializeOut);
    out_archive << *remote_task;
    std::string out_data = out_archive.GetData();

    // Step 5: Simulate client receiving and deserializing OUT parameters
    clio::run::LoadTaskArchive recv_out_archive(out_data);
    recv_out_archive.msg_type_ = clio::run::MsgType::kSerializeOut;
    auto final_task = CreateTestAdminTask();
    final_task->return_code_ = 0; // Clear
    final_task->error_message_ = clio::run::priv::string(CTP_MALLOC, "");
    recv_out_archive >> *final_task;

    // Verify OUT parameters were transferred back
    REQUIRE(final_task->return_code_ == 123);
    REQUIRE(final_task->error_message_.str() == "remote execution result");
    REQUIRE(final_task->pool_id_ ==
            original_task->pool_id_); // INOUT parameter preserved

    INFO("Complete serialization flow completed successfully");
  }

  SECTION("Network Transport Round-Trip Simulation") {
    // This test simulates the exact serialization path used by ZeroMQ transport:
    // 1. SaveTaskArchive is serialized with GlobalSerialize
    // 2. LoadTaskArchive is deserialized with GlobalDeserialize
    // This mimics zmq_transport.h Send() and RecvMetadata() behavior
    INFO("Testing network transport round-trip serialization");

    // Step 1: Create a SaveTaskArchive with test data
    clio::run::SaveTaskArchive save_archive(clio::run::MsgType::kSerializeIn);
    auto original_task = CreateTestAdminTask();
    original_task->chimod_name_ = clio::run::priv::string(CTP_MALLOC, "test_module");
    original_task->pool_name_ = clio::run::priv::string(CTP_MALLOC, "test_pool");
    original_task->pool_id_ = clio::run::PoolId(100, 200);
    save_archive << *original_task;

    // Step 2: Serialize SaveTaskArchive using GlobalSerialize (mimics ZMQ Send)
    std::vector<char> net_buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(net_buf);
      ar(save_archive);  // Calls SaveTaskArchive::serialize()
      ar.Finalize();
    }
    std::string network_data(net_buf.begin(), net_buf.end());
    INFO("Network data size: " << network_data.size() << " bytes");
    REQUIRE(network_data.size() > 0);

    // Step 3: Deserialize into LoadTaskArchive (mimics ZMQ RecvMetadata)
    clio::run::LoadTaskArchive load_archive;
    {
      std::vector<char> recv_buf(network_data.begin(), network_data.end());
      ctp::ipc::GlobalDeserialize<std::vector<char>> ar(recv_buf);
      ar(load_archive);  // Calls LoadTaskArchive::serialize()
    }

    // Step 4: Verify metadata was transferred correctly
    REQUIRE(load_archive.GetMsgType() == clio::run::MsgType::kSerializeIn);
    REQUIRE(load_archive.GetTaskInfos().size() == 1);
    const auto& task_info = load_archive.GetTaskInfos()[0];
    REQUIRE(task_info.pool_id_ == clio::run::PoolId(100, 200));

    // Step 5: Deserialize the actual task from LoadTaskArchive
    auto recv_task = CreateTestAdminTask();
    recv_task->chimod_name_ = clio::run::priv::string(CTP_MALLOC, "");
    recv_task->pool_name_ = clio::run::priv::string(CTP_MALLOC, "");
    load_archive >> *recv_task;

    // Step 6: Verify task data was transferred correctly
    REQUIRE(recv_task->chimod_name_.str() == "test_module");
    REQUIRE(recv_task->pool_name_.str() == "test_pool");
    REQUIRE(recv_task->pool_id_ == clio::run::PoolId(100, 200));

    INFO("Network transport round-trip simulation completed successfully");
  }

  SECTION("Network Transport Round-Trip with Bulk") {
    // Test network round-trip with bulk data
    INFO("Testing network transport round-trip with bulk data");

    // Step 1: Create a SaveTaskArchive with bulk data
    clio::run::SaveTaskArchive save_archive(clio::run::MsgType::kSerializeIn);
    auto original_task = CreateTestAdminTask();
    original_task->chimod_name_ = clio::run::priv::string(CTP_MALLOC, "bulk_module");
    original_task->pool_name_ = clio::run::priv::string(CTP_MALLOC, "bulk_pool");
    save_archive << *original_task;

    // Add a bulk descriptor to the send vector
    ctp::lbm::Bulk test_bulk;
    test_bulk.size = 1024;
    test_bulk.flags.SetBits(BULK_XFER);
    save_archive.send.push_back(test_bulk);

    // Step 2: Serialize using GlobalSerialize
    std::vector<char> net_buf;
    {
      ctp::ipc::GlobalSerialize<std::vector<char>> ar(net_buf);
      ar(save_archive);
      ar.Finalize();
    }
    std::string network_data(net_buf.begin(), net_buf.end());
    INFO("Network data size with bulk: " << network_data.size() << " bytes");
    REQUIRE(network_data.size() > 0);

    // Step 3: Deserialize into LoadTaskArchive
    clio::run::LoadTaskArchive load_archive;
    {
      std::vector<char> recv_buf(network_data.begin(), network_data.end());
      ctp::ipc::GlobalDeserialize<std::vector<char>> ar(recv_buf);
      ar(load_archive);
    }

    // Step 4: Verify bulk descriptors were transferred
    REQUIRE(load_archive.send.size() == 1);
    REQUIRE(load_archive.send[0].size == 1024);
    REQUIRE(load_archive.send[0].flags.Any(BULK_XFER));

    // Step 5: Verify task infos and message type
    REQUIRE(load_archive.GetMsgType() == clio::run::MsgType::kSerializeIn);
    REQUIRE(load_archive.GetTaskInfos().size() == 1);

    INFO("Network transport round-trip with bulk completed successfully");
  }
}

// Main function to run all tests with CLIO Runtime runtime initialization
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  // Initialize CLIO Runtime runtime for memory management
  bool runtime_success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (!runtime_success) {
    HLOG(kError, "Failed to initialize Clio runtime");
    return 1;
  }

  // Run all tests
  int result = SimpleTest::run_all_tests();

  // Runtime will be cleaned up automatically. SIMPLE_TEST_PROCESS_EXIT is
  // TerminateProcess() on Windows (to dodge a libzmq static-destructor
  // abort that fires after all tests pass) and a plain return elsewhere.
  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
