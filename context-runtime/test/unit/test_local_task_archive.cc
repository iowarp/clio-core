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
 * Unit test for LocalTaskArchive serialization and deserialization
 */

#include "simple_test.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/local_task_archives.h"
#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/bdev/bdev_client.h"

using namespace clio::run;

/**
 * Test fixture for LocalTaskArchive tests
 */
class LocalTaskArchiveTest {
public:
  LocalTaskArchiveTest() {
    // Initialize CLIO Runtime with client mode and runtime
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;

    // Give runtime time to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Verify initialization
    REQUIRE(CLIO_IPC != nullptr);
    REQUIRE(CLIO_IPC->IsInitialized());
  }

  ~LocalTaskArchiveTest() {
    // Cleanup handled automatically
  }
};

TEST_CASE("LocalTaskArchive: Serialize and deserialize basic types", "[local_task_archive]") {
  // Test serializing and deserializing basic types
  int int_val = 42;
  double double_val = 3.14159;
  std::string str_val = "hello world";

  // Serialize
  clio::run::priv::vector<char> save_buf;
  DefaultSaveArchive save_archive(LocalMsgType::kSerializeIn, save_buf);
  save_archive << int_val << double_val << str_val;

  // Deserialize
  DefaultLoadArchive load_archive(save_buf);
  int restored_int = 0;
  double restored_double = 0.0;
  std::string restored_str;
  load_archive >> restored_int >> restored_double >> restored_str;

  // Verify
  REQUIRE(restored_int == 42);
  REQUIRE(restored_double == 3.14159);
  REQUIRE(restored_str == "hello world");
}

TEST_CASE("LocalTaskArchive: Serialize ShmPtr with bulk()", "[local_task_archive][bulk]") {
  // Create a ShmPtr
  ctp::ipc::ShmPtr<char> shm_ptr;
  shm_ptr.off_ = 12345;
  shm_ptr.alloc_id_ = ctp::ipc::AllocatorId(1, 2);

  size_t data_size = 1024;
  uint32_t flags = 0;

  // Serialize with bulk
  clio::run::priv::vector<char> save_buf;
  DefaultSaveArchive save_archive(LocalMsgType::kSerializeIn, save_buf);
  save_archive.bulk(shm_ptr, data_size, flags);

  // Deserialize with bulk
  DefaultLoadArchive load_archive(save_buf);
  ctp::ipc::ShmPtr<char> restored_ptr;
  load_archive.bulk(restored_ptr, data_size, flags);

  // Verify the ShmPtr was correctly serialized
  REQUIRE(restored_ptr.off_.load() == shm_ptr.off_.load());
  REQUIRE(restored_ptr.alloc_id_.major_ == shm_ptr.alloc_id_.major_);
  REQUIRE(restored_ptr.alloc_id_.minor_ == shm_ptr.alloc_id_.minor_);
}

TEST_CASE("LocalTaskArchive: Serialize FullPtr with bulk()", "[local_task_archive][bulk]") {
  // Create a FullPtr
  ctp::ipc::FullPtr<char> full_ptr;
  full_ptr.shm_.off_ = 54321;
  full_ptr.shm_.alloc_id_ = ctp::ipc::AllocatorId(3, 4);
  full_ptr.ptr_ = reinterpret_cast<char *>(0xDEADBEEF);

  size_t data_size = 2048;
  uint32_t flags = 1;

  // Serialize with bulk
  clio::run::priv::vector<char> save_buf;
  DefaultSaveArchive save_archive(LocalMsgType::kSerializeIn, save_buf);
  save_archive.bulk(full_ptr, data_size, flags);

  // Deserialize with bulk (only shm_ part is serialized)
  DefaultLoadArchive load_archive(save_buf);
  ctp::ipc::FullPtr<char> restored_ptr;
  load_archive.bulk(restored_ptr, data_size, flags);

  // Verify only the ShmPtr part was serialized (ptr_ is not serialized)
  REQUIRE(restored_ptr.shm_.off_.load() == full_ptr.shm_.off_.load());
  REQUIRE(restored_ptr.shm_.alloc_id_.major_ == full_ptr.shm_.alloc_id_.major_);
  REQUIRE(restored_ptr.shm_.alloc_id_.minor_ == full_ptr.shm_.alloc_id_.minor_);
}

TEST_CASE("LocalTaskArchive: Serialize raw pointer with bulk()", "[local_task_archive][bulk]") {
  // Create raw data
  const size_t data_size = 16;
  char original_data[data_size];
  for (size_t i = 0; i < data_size; ++i) {
    original_data[i] = static_cast<char>('A' + i);
  }

  uint32_t flags = 2;

  // Serialize with bulk (full memory copy)
  clio::run::priv::vector<char> save_buf;
  DefaultSaveArchive save_archive(LocalMsgType::kSerializeIn, save_buf);
  save_archive.bulk(original_data, data_size, flags);

  // Deserialize with bulk (full memory copy)
  DefaultLoadArchive load_archive(save_buf);
  char restored_data[data_size];
  load_archive.bulk(restored_data, data_size, flags);

  // Verify the data was correctly copied
  for (size_t i = 0; i < data_size; ++i) {
    REQUIRE(restored_data[i] == original_data[i]);
  }
}

TEST_CASE("LocalTaskArchive: Mixed serialization with operator()", "[local_task_archive]") {
  int val1 = 100;
  std::string val2 = "test";
  double val3 = 2.71828;

  // Serialize using operator()
  clio::run::priv::vector<char> save_buf;
  DefaultSaveArchive save_archive(LocalMsgType::kSerializeIn, save_buf);
  save_archive(val1, val2, val3);

  // Deserialize using operator()
  DefaultLoadArchive load_archive(save_buf);
  int restored1 = 0;
  std::string restored2;
  double restored3 = 0.0;
  load_archive(restored1, restored2, restored3);

  // Verify
  REQUIRE(restored1 == 100);
  REQUIRE(restored2 == "test");
  REQUIRE(restored3 == 2.71828);
}

TEST_CASE("LocalTaskArchive: Serialize and deserialize multiple values", "[local_task_archive]") {
  std::vector<int> vec = {1, 2, 3, 4, 5};
  std::string str1 = "first";
  std::string str2 = "second";
  int num = 999;

  // Serialize
  clio::run::priv::vector<char> save_buf;
  DefaultSaveArchive save_archive(LocalMsgType::kSerializeIn, save_buf);
  save_archive << vec << str1 << str2 << num;

  // Deserialize
  DefaultLoadArchive load_archive(save_buf);
  std::vector<int> restored_vec;
  std::string restored_str1;
  std::string restored_str2;
  int restored_num = 0;
  load_archive >> restored_vec >> restored_str1 >> restored_str2 >> restored_num;

  // Verify
  REQUIRE(restored_vec.size() == 5);
  for (size_t i = 0; i < vec.size(); ++i) {
    REQUIRE(restored_vec[i] == vec[i]);
  }
  REQUIRE(restored_str1 == "first");
  REQUIRE(restored_str2 == "second");
  REQUIRE(restored_num == 999);
}

TEST_CASE("LocalTaskArchive: Message types", "[local_task_archive]") {
  // Test kSerializeIn
  clio::run::priv::vector<char> buf_in;
  DefaultSaveArchive save_in(LocalMsgType::kSerializeIn, buf_in);
  REQUIRE(save_in.GetMsgType() == LocalMsgType::kSerializeIn);

  // Test kSerializeOut
  clio::run::priv::vector<char> buf_out;
  DefaultSaveArchive save_out(LocalMsgType::kSerializeOut, buf_out);
  REQUIRE(save_out.GetMsgType() == LocalMsgType::kSerializeOut);
}

TEST_CASE("LocalTaskArchive: Multiple bulk transfers", "[local_task_archive][bulk]") {
  // Create multiple different pointer types
  ctp::ipc::ShmPtr<char> shm_ptr1;
  shm_ptr1.off_ = 100;
  shm_ptr1.alloc_id_ = ctp::ipc::AllocatorId(1, 1);

  ctp::ipc::FullPtr<char> full_ptr;
  full_ptr.shm_.off_ = 200;
  full_ptr.shm_.alloc_id_ = ctp::ipc::AllocatorId(2, 2);

  char raw_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  ctp::ipc::ShmPtr<char> shm_ptr2;
  shm_ptr2.off_ = 300;
  shm_ptr2.alloc_id_ = ctp::ipc::AllocatorId(3, 3);

  // Serialize all bulk transfers
  clio::run::priv::vector<char> save_buf;
  DefaultSaveArchive save_archive(LocalMsgType::kSerializeIn, save_buf);
  save_archive.bulk(shm_ptr1, 10, 0);
  save_archive.bulk(full_ptr, 20, 1);
  save_archive.bulk(raw_data, 8, 2);
  save_archive.bulk(shm_ptr2, 30, 3);

  // Deserialize all bulk transfers
  DefaultLoadArchive load_archive(save_buf);
  ctp::ipc::ShmPtr<char> restored_shm1;
  ctp::ipc::FullPtr<char> restored_full;
  char restored_raw[8];
  ctp::ipc::ShmPtr<char> restored_shm2;

  load_archive.bulk(restored_shm1, 10, 0);
  load_archive.bulk(restored_full, 20, 1);
  load_archive.bulk(restored_raw, 8, 2);
  load_archive.bulk(restored_shm2, 30, 3);

  // Verify all values
  REQUIRE(restored_shm1.off_.load() == 100);
  REQUIRE(restored_full.shm_.off_.load() == 200);
  for (int i = 0; i < 8; ++i) {
    REQUIRE(restored_raw[i] == raw_data[i]);
  }
  REQUIRE(restored_shm2.off_.load() == 300);
}

SIMPLE_TEST_MAIN()
