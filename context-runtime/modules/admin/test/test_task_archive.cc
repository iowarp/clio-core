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
#include <sstream>
#include <string>
#include <vector>
#include <hermes_shm/util/logging.h>

// Include Chimaera headers
#include <chimaera/chimaera.h>
#include <hermes_shm/memory/allocator/malloc_allocator.h>
#include <chimaera/container.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/task.h>
#include <chimaera/task_archives.h>
#include <chimaera/types.h>

// Include admin tasks for testing concrete task types
#include <chimaera/admin/admin_tasks.h>

// Include cereal for comparison tests
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

using namespace chi;

// Test constants
constexpr chi::u32 kTestWriteFlag = 0x1;  // BULK_XFER
constexpr chi::u32 kTestExposeFlag = 0x2; // BULK_EXPOSE

// Helper allocator for tests - uses HSHM_MALLOC for non-IPC allocations
hipc::MallocAllocator* GetTestAllocator() {
  return HSHM_MALLOC;
}

// Helper to create test task with sample data
std::unique_ptr<chi::Task> CreateTestTask() {
  auto task = std::make_unique<chi::Task>(chi::TaskId(1, 1, 1, 0, 1),
                                          chi::PoolId(100, 0), chi::PoolQuery(),
                                          chi::MethodId(42));
  task->period_ns_ = 1000000.0; // 1ms
  task->task_flags_.SetBits(0x10);
  return task;
}

// Helper to create test admin task with sample data
std::unique_ptr<chimaera::admin::CreateTask> CreateTestAdminTask() {
  auto alloc = GetTestAllocator();
  auto task = std::make_unique<chimaera::admin::CreateTask>(
      chi::TaskId(2, 2, 2, 0, 2), chi::PoolId(200, 0), chi::PoolQuery(),
      "test_chimod", "test_pool", chi::PoolId(300, 0),
      nullptr);  // No client for test task
  task->return_code_ = 42;
  task->error_message_ = chi::priv::string(HSHM_MALLOC, "test error message");
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
    chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn);

    // Test NetTaskArchive methods
    REQUIRE(archive.GetMsgType() == chi::MsgType::kSerializeIn);
    REQUIRE(archive.GetTaskInfos().empty());
    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);
  }

  SECTION("LoadTaskArchive inherits from NetTaskArchive") {
    chi::LoadTaskArchive archive;

    // Test NetTaskArchive methods
    REQUIRE(archive.GetMsgType() == chi::MsgType::kSerializeIn);
    REQUIRE(archive.GetTaskInfos().empty());
    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);
  }

  SECTION("SaveTaskArchive with different message types") {
    chi::SaveTaskArchive archive_in(chi::MsgType::kSerializeIn);
    chi::SaveTaskArchive archive_out(chi::MsgType::kSerializeOut);
    chi::SaveTaskArchive archive_hb(chi::MsgType::kHeartbeat);

    REQUIRE(archive_in.GetMsgType() == chi::MsgType::kSerializeIn);
    REQUIRE(archive_out.GetMsgType() == chi::MsgType::kSerializeOut);
    REQUIRE(archive_hb.GetMsgType() == chi::MsgType::kHeartbeat);
  }
}

//==============================================================================
// SaveTaskArchive Tests
//==============================================================================

TEST_CASE("SaveTaskArchive - Basic Construction and Data Retrieval",
          "[task_archive][save_archive]") {
  SECTION("Construction with message type") {
    chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn);

    // Should start with empty bulk transfers
    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);

    // Data should be empty initially
    std::string data = archive.GetData();
    REQUIRE(data.empty());
  }

  SECTION("Serializing simple data") {
    chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn);
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
    chi::LoadTaskArchive archive;

    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);
  }

  SECTION("Construction from string") {
    std::string test_data = "test serialized data";
    chi::LoadTaskArchive archive(test_data);

    REQUIRE(archive.GetSendBulkCount() == 0);
    REQUIRE(archive.GetRecvBulkCount() == 0);
  }

  SECTION("Construction from const char* and size") {
    const char *test_data = "test data";
    size_t size = strlen(test_data);
    chi::LoadTaskArchive archive(test_data, size);

    REQUIRE(archive.GetSendBulkCount() == 0);
  }
}

//==============================================================================
// Bulk Transfer Tests
//==============================================================================

TEST_CASE("Bulk Transfer Recording", "[task_archive][bulk_transfer]") {
  SECTION("SaveTaskArchive bulk transfer recording") {
    chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn);

    hipc::ShmPtr<> test_ptr = hipc::ShmPtr<>::GetNull();
    size_t test_size = 1024;
    uint32_t test_flags = kTestWriteFlag;

    REQUIRE_NOTHROW(archive.bulk(test_ptr, test_size, test_flags));

    // Bulk should be added to send vector
    REQUIRE(archive.GetSendBulkCount() == 1);
    REQUIRE(archive.send[0].size == test_size);
    REQUIRE(archive.send[0].flags.bits_ == test_flags);
  }

  SECTION("Multiple bulk transfers") {
    chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn);

    // Add multiple bulk transfers
    archive.bulk(hipc::ShmPtr<>::GetNull(), 100, kTestWriteFlag);
    archive.bulk(hipc::ShmPtr<>::GetNull(), 200, kTestExposeFlag);
    archive.bulk(hipc::ShmPtr<>::GetNull(), 300, kTestWriteFlag | kTestExposeFlag);

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
    chi::SaveTaskArchive out_archive(chi::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive << original);
    std::string serialized_data = out_archive.GetData();

    // Deserialize using LoadTaskArchive
    chi::LoadTaskArchive in_archive(serialized_data);
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
    chi::SaveTaskArchive out_archive(chi::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive(str1, int1, double1));
    std::string serialized_data = out_archive.GetData();

    // Deserialize using operator()
    chi::LoadTaskArchive in_archive(serialized_data);
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
    chi::SaveTaskArchive out_archive(chi::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive << *original_task);
    std::string serialized_data = out_archive.GetData();

    // Verify task info was recorded
    REQUIRE(out_archive.GetTaskInfos().size() == 1);
    REQUIRE(out_archive.GetTaskInfos()[0].task_id_ == original_task->task_id_);
    REQUIRE(out_archive.GetTaskInfos()[0].pool_id_ == original_task->pool_id_);
    REQUIRE(out_archive.GetTaskInfos()[0].method_id_ == original_task->method_);

    // Deserialize using LoadTaskArchive with kSerializeIn mode
    chi::LoadTaskArchive in_archive(serialized_data);
    in_archive.msg_type_ = chi::MsgType::kSerializeIn;
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
    chi::SaveTaskArchive out_archive(chi::MsgType::kSerializeOut);
    REQUIRE_NOTHROW(out_archive << *original_task);
    std::string serialized_data = out_archive.GetData();

    // Deserialize using LoadTaskArchive with kSerializeOut mode
    chi::LoadTaskArchive in_archive(serialized_data);
    in_archive.msg_type_ = chi::MsgType::kSerializeOut;
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
    chi::SaveTaskArchive out_archive_in(chi::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive_in << *original_task);
    std::string in_data = out_archive_in.GetData();

    chi::LoadTaskArchive in_archive_in(in_data);
    in_archive_in.msg_type_ = chi::MsgType::kSerializeIn;
    auto new_task_in = CreateTestAdminTask();
    new_task_in->chimod_name_ = chi::priv::string(HSHM_MALLOC, ""); // Clear
    new_task_in->pool_name_ = chi::priv::string(HSHM_MALLOC, "");
    REQUIRE_NOTHROW(in_archive_in >> *new_task_in);

    // Verify IN/INOUT parameters were preserved
    REQUIRE(new_task_in->chimod_name_.str() ==
            original_task->chimod_name_.str());
    REQUIRE(new_task_in->pool_name_.str() == original_task->pool_name_.str());
    REQUIRE(new_task_in->pool_id_ == original_task->pool_id_);

    // Test OUT parameter serialization
    chi::SaveTaskArchive out_archive_out(chi::MsgType::kSerializeOut);
    REQUIRE_NOTHROW(out_archive_out << *original_task);
    std::string out_data = out_archive_out.GetData();

    chi::LoadTaskArchive in_archive_out(out_data);
    in_archive_out.msg_type_ = chi::MsgType::kSerializeOut;
    auto new_task_out = CreateTestAdminTask();
    new_task_out->return_code_ = 0; // Clear
    new_task_out->error_message_ = chi::priv::string(HSHM_MALLOC, "");
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
    chimaera::admin::DestroyPoolTask original_task(
        chi::TaskId(3, 3, 3, 0, 3), chi::PoolId(400, 0),
        chi::PoolQuery(), chi::PoolId(500, 0), 0x123);
    original_task.return_code_ = 99;
    original_task.error_message_ = chi::priv::string(HSHM_MALLOC, "destroy error");

    // Test round-trip IN parameters
    chi::SaveTaskArchive out_archive_in(chi::MsgType::kSerializeIn);
    out_archive_in << original_task;

    chi::LoadTaskArchive in_archive_in(out_archive_in.GetData());
    in_archive_in.msg_type_ = chi::MsgType::kSerializeIn;
    chimaera::admin::DestroyPoolTask new_task_in;
    in_archive_in >> new_task_in;

    REQUIRE(new_task_in.target_pool_id_ == original_task.target_pool_id_);
    REQUIRE(new_task_in.destruction_flags_ == original_task.destruction_flags_);

    // Test round-trip OUT parameters
    chi::SaveTaskArchive out_archive_out(chi::MsgType::kSerializeOut);
    out_archive_out << original_task;

    chi::LoadTaskArchive in_archive_out(out_archive_out.GetData());
    in_archive_out.msg_type_ = chi::MsgType::kSerializeOut;
    chimaera::admin::DestroyPoolTask new_task_out;
    in_archive_out >> new_task_out;

    REQUIRE(new_task_out.return_code_ == original_task.return_code_);
    REQUIRE(new_task_out.error_message_.str() ==
            original_task.error_message_.str());
  }

  SECTION("StopRuntimeTask serialization") {
    auto alloc = GetTestAllocator();
    chimaera::admin::StopRuntimeTask original_task(
        chi::TaskId(4, 4, 4, 0, 4), chi::PoolId(600, 0),
        chi::PoolQuery(), 0x456, 10000);
    original_task.return_code_ = 777;
    original_task.error_message_ = chi::priv::string(HSHM_MALLOC, "shutdown error");

    // Test IN parameters
    chi::SaveTaskArchive out_archive_in(chi::MsgType::kSerializeIn);
    out_archive_in << original_task;

    chi::LoadTaskArchive in_archive_in(out_archive_in.GetData());
    in_archive_in.msg_type_ = chi::MsgType::kSerializeIn;
    chimaera::admin::StopRuntimeTask new_task_in;
    in_archive_in >> new_task_in;

    REQUIRE(new_task_in.shutdown_flags_ == original_task.shutdown_flags_);
    REQUIRE(new_task_in.grace_period_ms_ == original_task.grace_period_ms_);

    // Test OUT parameters
    chi::SaveTaskArchive out_archive_out(chi::MsgType::kSerializeOut);
    out_archive_out << original_task;

    chi::LoadTaskArchive in_archive_out(out_archive_out.GetData());
    in_archive_out.msg_type_ = chi::MsgType::kSerializeOut;
    chimaera::admin::StopRuntimeTask new_task_out;
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

    // Serialize with standard cereal
    std::ostringstream oss;
    cereal::BinaryOutputArchive out_archive(oss);
    out_archive(value1, value2, value3);

    // Deserialize with LoadTaskArchive using operator()
    chi::LoadTaskArchive in_archive(oss.str());
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
    chi::SaveTaskArchive out_archive(chi::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive(value1, value2, value3));

    // Deserialize using LoadTaskArchive
    chi::LoadTaskArchive in_archive(out_archive.GetData());
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
class TestContainer : public chi::Container {
public:
  chi::u64 GetWorkRemaining() const override {
    return 0; // Test implementation returns no work
  }

  chi::TaskResume Run(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr,
                      chi::RunContext &rctx) override {
    // Test implementation - do nothing
    (void)method;
    (void)task_ptr;
    (void)rctx;
    co_return;
  }

  void DelTask(chi::u32 method, hipc::FullPtr<chi::Task> task_ptr) override {
    // Test implementation - do nothing
    (void)method;
    (void)task_ptr;
  }

  void SaveTask(chi::u32 method, chi::SaveTaskArchive &archive,
                hipc::FullPtr<chi::Task> task_ptr) override {
    // Test implementation - just call task serialization
    (void)method;
    archive << *task_ptr;
  }

  void LoadTask(chi::u32 method, chi::LoadTaskArchive &archive,
                hipc::FullPtr<chi::Task> task_ptr) override {
    // Test implementation - just call task deserialization
    (void)method;
    archive >> *task_ptr;
  }

  hipc::FullPtr<chi::Task> AllocLoadTask(chi::u32 method, chi::LoadTaskArchive &archive) override {
    // Test implementation - allocate task and call LoadTask
    hipc::FullPtr<chi::Task> task_ptr = NewTask(method);
    if (!task_ptr.IsNull()) {
      LoadTask(method, archive, task_ptr);
    }
    return task_ptr;
  }

  hipc::FullPtr<chi::Task> NewCopyTask(chi::u32 method, hipc::FullPtr<chi::Task> orig_task_ptr,
                                        bool deep) override {
    // Test implementation - create new task and copy
    (void)method;
    (void)deep;
    auto *ipc_manager = CHI_IPC;
    if (ipc_manager) {
      auto new_task_ptr = ipc_manager->NewTask<chi::Task>();
      if (!new_task_ptr.IsNull()) {
        new_task_ptr->Copy(orig_task_ptr);
      }
      return new_task_ptr;
    }
    return hipc::FullPtr<chi::Task>();
  }

  hipc::FullPtr<chi::Task> NewTask(chi::u32 method) override {
    // Test implementation - create a basic Task
    (void)method;
    auto *ipc_manager = CHI_IPC;
    if (ipc_manager) {
      return ipc_manager->NewTask<chi::Task>();
    }
    return hipc::FullPtr<chi::Task>();
  }

  void Aggregate(chi::u32 method, hipc::FullPtr<chi::Task> origin_task_ptr,
                 hipc::FullPtr<chi::Task> replica_task_ptr) override {
    // Test implementation - do nothing
    (void)method;
    (void)origin_task_ptr;
    (void)replica_task_ptr;
  }

  void LocalLoadTask(chi::u32 method, chi::LocalLoadTaskArchive &archive,
                     hipc::FullPtr<chi::Task> task_ptr) override {
    // Test implementation - do nothing for now
    (void)method;
    (void)archive;
    (void)task_ptr;
  }

  hipc::FullPtr<chi::Task> LocalAllocLoadTask(chi::u32 method, chi::LocalLoadTaskArchive &archive) override {
    // Test implementation - allocate task and call LocalLoadTask
    hipc::FullPtr<chi::Task> task_ptr = NewTask(method);
    if (!task_ptr.IsNull()) {
      LocalLoadTask(method, archive, task_ptr);
    }
    return task_ptr;
  }

  void LocalSaveTask(chi::u32 method, chi::LocalSaveTaskArchive &archive,
                     hipc::FullPtr<chi::Task> task_ptr) override {
    // Test implementation - do nothing
    (void)method;
    (void)archive;
    (void)task_ptr;
  }
};

TEST_CASE("Container Serialization Methods", "[task_archive][container]") {
  SECTION("Container SaveTask/LoadTask with SerializeIn mode") {
    TestContainer container;
    auto original_task = CreateTestTask();
    hipc::FullPtr<chi::Task> task_ptr(original_task.get());

    // Test SaveTask with SerializeIn mode (inputs)
    chi::SaveTaskArchive save_archive(chi::MsgType::kSerializeIn);
    chi::u32 method = task_ptr->method_;
    REQUIRE_NOTHROW(container.SaveTask(method, save_archive, task_ptr));
    std::string serialized_data = save_archive.GetData();
    REQUIRE_FALSE(serialized_data.empty());

    // Verify task info was recorded
    REQUIRE(save_archive.GetTaskInfos().size() == 1);

    // Test AllocLoadTask with SerializeIn mode (inputs)
    chi::LoadTaskArchive load_archive(serialized_data);
    load_archive.msg_type_ = chi::MsgType::kSerializeIn;
    hipc::FullPtr<chi::Task> new_task_ptr;
    REQUIRE_NOTHROW(new_task_ptr = container.AllocLoadTask(method, load_archive));
  }

  SECTION("Container SaveTask/LoadTask with SerializeOut mode") {
    TestContainer container;
    auto original_task = CreateTestTask();
    hipc::FullPtr<chi::Task> task_ptr(original_task.get());

    // Test SaveTask with SerializeOut mode (outputs)
    chi::SaveTaskArchive save_archive(chi::MsgType::kSerializeOut);
    chi::u32 method = task_ptr->method_;
    REQUIRE_NOTHROW(container.SaveTask(method, save_archive, task_ptr));
    std::string serialized_data = save_archive.GetData();
    REQUIRE_FALSE(serialized_data.empty());

    // Test AllocLoadTask with SerializeOut mode (outputs)
    chi::LoadTaskArchive load_archive(serialized_data);
    load_archive.msg_type_ = chi::MsgType::kSerializeOut;
    hipc::FullPtr<chi::Task> new_task_ptr;
    REQUIRE_NOTHROW(new_task_ptr = container.AllocLoadTask(method, load_archive));
  }
}

//==============================================================================
// Error Handling and Edge Cases
//==============================================================================

TEST_CASE("Error Handling and Edge Cases", "[task_archive][error_handling]") {
  SECTION("Invalid serialization data") {
    std::string invalid_data = "this is not valid cereal data";
    chi::LoadTaskArchive archive(invalid_data);

    // Archive should be constructed without throwing
    REQUIRE_NOTHROW(archive.GetSendBulkCount());
  }

  SECTION("Empty serialization data") {
    std::string empty_data = "";
    chi::LoadTaskArchive archive(empty_data);

    REQUIRE(archive.GetSendBulkCount() == 0);
  }

  SECTION("Bulk transfer with null pointer") {
    chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn);
    hipc::ShmPtr<> null_ptr = hipc::ShmPtr<>::GetNull();

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

    chi::SaveTaskArchive out_archive(chi::MsgType::kSerializeIn);
    REQUIRE_NOTHROW(out_archive << large_string);

    std::string serialized_data = out_archive.GetData();
    REQUIRE(serialized_data.size() > large_string.size()); // Should include cereal overhead

    chi::LoadTaskArchive in_archive(serialized_data);
    std::string result_string;
    REQUIRE_NOTHROW(in_archive >> result_string);

    REQUIRE(result_string == large_string);
  }

  SECTION("Large vector serialization") {
    std::vector<double> large_vector(1000, 3.14159); // 1000 doubles

    chi::SaveTaskArchive out_archive(chi::MsgType::kSerializeIn);
    out_archive << large_vector;

    chi::LoadTaskArchive in_archive(out_archive.GetData());
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
      auto task = std::make_unique<chi::Task>(chi::TaskId(1, 1, 1, 0, static_cast<chi::u32>(i)),
                                              chi::PoolId(100, 0), chi::PoolQuery(),
                                              chi::MethodId(42));
      task->period_ns_ = 1000000.0 + i; // Vary the period to make tasks different

      chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn);
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
    chi::SaveTaskArchive in_archive(chi::MsgType::kSerializeIn);
    in_archive << *original_task;
    std::string in_data = in_archive.GetData();

    // Verify task info and bulk counts
    REQUIRE(in_archive.GetTaskInfos().size() == 1);

    // Step 2: Simulate remote node receiving and deserializing IN parameters
    chi::LoadTaskArchive recv_in_archive(in_data);
    recv_in_archive.msg_type_ = chi::MsgType::kSerializeIn;
    auto remote_task = CreateTestAdminTask();
    remote_task->chimod_name_ = chi::priv::string(HSHM_MALLOC, ""); // Clear
    remote_task->pool_name_ = chi::priv::string(HSHM_MALLOC, "");
    recv_in_archive >> *remote_task;

    // Verify IN parameters were transferred
    REQUIRE(remote_task->chimod_name_.str() ==
            original_task->chimod_name_.str());
    REQUIRE(remote_task->pool_name_.str() == original_task->pool_name_.str());
    REQUIRE(remote_task->pool_id_ == original_task->pool_id_);

    // Step 3: Simulate task execution and result generation on remote node
    remote_task->return_code_ = 123;
    remote_task->error_message_ =
        chi::priv::string(HSHM_MALLOC, "remote execution result");

    // Step 4: Serialize OUT parameters (for sending results back)
    chi::SaveTaskArchive out_archive(chi::MsgType::kSerializeOut);
    out_archive << *remote_task;
    std::string out_data = out_archive.GetData();

    // Step 5: Simulate client receiving and deserializing OUT parameters
    chi::LoadTaskArchive recv_out_archive(out_data);
    recv_out_archive.msg_type_ = chi::MsgType::kSerializeOut;
    auto final_task = CreateTestAdminTask();
    final_task->return_code_ = 0; // Clear
    final_task->error_message_ = chi::priv::string(HSHM_MALLOC, "");
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
    // 1. SaveTaskArchive is serialized with cereal::BinaryOutputArchive
    // 2. LoadTaskArchive is deserialized with cereal::BinaryInputArchive
    // This mimics zmq_transport.h Send() and RecvMetadata() behavior
    INFO("Testing network transport round-trip serialization");

    // Step 1: Create a SaveTaskArchive with test data
    chi::SaveTaskArchive save_archive(chi::MsgType::kSerializeIn);
    auto original_task = CreateTestAdminTask();
    original_task->chimod_name_ = chi::priv::string(HSHM_MALLOC, "test_module");
    original_task->pool_name_ = chi::priv::string(HSHM_MALLOC, "test_pool");
    original_task->pool_id_ = chi::PoolId(100, 200);
    save_archive << *original_task;

    // Step 2: Serialize SaveTaskArchive using cereal (mimics ZMQ Send)
    std::ostringstream oss(std::ios::binary);
    {
      cereal::BinaryOutputArchive ar(oss);
      ar(save_archive);  // Calls SaveTaskArchive::save()
    }
    std::string network_data = oss.str();
    INFO("Network data size: " << network_data.size() << " bytes");
    REQUIRE(network_data.size() > 0);

    // Step 3: Deserialize into LoadTaskArchive (mimics ZMQ RecvMetadata)
    chi::LoadTaskArchive load_archive;
    {
      std::istringstream iss(network_data, std::ios::binary);
      cereal::BinaryInputArchive ar(iss);
      ar(load_archive);  // Calls LoadTaskArchive::load()
    }

    // Step 4: Verify metadata was transferred correctly
    REQUIRE(load_archive.GetMsgType() == chi::MsgType::kSerializeIn);
    REQUIRE(load_archive.GetTaskInfos().size() == 1);
    const auto& task_info = load_archive.GetTaskInfos()[0];
    REQUIRE(task_info.pool_id_ == chi::PoolId(100, 200));

    // Step 5: Deserialize the actual task from LoadTaskArchive
    auto recv_task = CreateTestAdminTask();
    recv_task->chimod_name_ = chi::priv::string(HSHM_MALLOC, "");
    recv_task->pool_name_ = chi::priv::string(HSHM_MALLOC, "");
    load_archive >> *recv_task;

    // Step 6: Verify task data was transferred correctly
    REQUIRE(recv_task->chimod_name_.str() == "test_module");
    REQUIRE(recv_task->pool_name_.str() == "test_pool");
    REQUIRE(recv_task->pool_id_ == chi::PoolId(100, 200));

    INFO("Network transport round-trip simulation completed successfully");
  }

  SECTION("Network Transport Round-Trip with Bulk") {
    // Test network round-trip with bulk data
    INFO("Testing network transport round-trip with bulk data");

    // Step 1: Create a SaveTaskArchive with bulk data
    chi::SaveTaskArchive save_archive(chi::MsgType::kSerializeIn);
    auto original_task = CreateTestAdminTask();
    original_task->chimod_name_ = chi::priv::string(HSHM_MALLOC, "bulk_module");
    original_task->pool_name_ = chi::priv::string(HSHM_MALLOC, "bulk_pool");
    save_archive << *original_task;

    // Add a bulk descriptor to the send vector
    hshm::lbm::Bulk test_bulk;
    test_bulk.size = 1024;
    test_bulk.flags.SetBits(BULK_XFER);
    save_archive.send.push_back(test_bulk);

    // Step 2: Serialize using cereal
    std::ostringstream oss(std::ios::binary);
    {
      cereal::BinaryOutputArchive ar(oss);
      ar(save_archive);
    }
    std::string network_data = oss.str();
    INFO("Network data size with bulk: " << network_data.size() << " bytes");
    REQUIRE(network_data.size() > 0);

    // Step 3: Deserialize into LoadTaskArchive
    chi::LoadTaskArchive load_archive;
    {
      std::istringstream iss(network_data, std::ios::binary);
      cereal::BinaryInputArchive ar(iss);
      ar(load_archive);
    }

    // Step 4: Verify bulk descriptors were transferred
    REQUIRE(load_archive.send.size() == 1);
    REQUIRE(load_archive.send[0].size == 1024);
    REQUIRE(load_archive.send[0].flags.Any(BULK_XFER));

    // Step 5: Verify task infos and message type
    REQUIRE(load_archive.GetMsgType() == chi::MsgType::kSerializeIn);
    REQUIRE(load_archive.GetTaskInfos().size() == 1);

    INFO("Network transport round-trip with bulk completed successfully");
  }
}

// Main function to run all tests with Chimaera runtime initialization
int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  // Initialize Chimaera runtime for memory management
  bool runtime_success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!runtime_success) {
    HLOG(kError, "Failed to initialize Chimaera runtime");
    return 1;
  }

  // Run all tests
  int result = SimpleTest::run_all_tests();

  // Runtime will be cleaned up automatically
  return result;
}
