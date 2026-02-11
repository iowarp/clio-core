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
 * Unit tests for LocalTransfer class
 *
 * Tests the chunked data transfer mechanism without requiring
 * the full Chimaera runtime.
 */

#include "../simple_test.h"
#include "chimaera/local_transfer.h"
#include "chimaera/task.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

using namespace chi;

/**
 * Helper to create a FutureShm with a specified copy_space size
 * Returns raw pointer - caller is responsible for cleanup
 */
static FutureShm* CreateFutureShm(size_t copy_space_size) {
  size_t total_size = sizeof(FutureShm) + copy_space_size;
  char* buffer = new char[total_size];
  std::memset(buffer, 0, total_size);

  // Construct FutureShm in-place
  FutureShm* future_shm = new (buffer) FutureShm();
  future_shm->input_.copy_space_size_ = copy_space_size;
  future_shm->output_.copy_space_size_ = copy_space_size;

  return future_shm;
}

/**
 * Helper to create a FullPtr<FutureShm> from raw pointer
 */
static hipc::FullPtr<FutureShm> MakeFullPtr(FutureShm* ptr) {
  return hipc::FullPtr<FutureShm>(ptr);
}

/**
 * Helper to clean up FutureShm
 */
static void DestroyFutureShm(FutureShm* ptr) {
  if (ptr) {
    ptr->~FutureShm();
    delete[] reinterpret_cast<char*>(ptr);
  }
}

/**
 * Generate test data with a predictable pattern
 */
static std::vector<char> GenerateTestData(size_t size) {
  std::vector<char> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  return data;
}

/**
 * Verify data matches the expected pattern
 */
static bool VerifyTestData(const std::vector<char>& data, size_t expected_size) {
  if (data.size() != expected_size) {
    return false;
  }
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] != static_cast<char>(i % 256)) {
      return false;
    }
  }
  return true;
}

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST_CASE("LocalTransfer - Sender Construction", "[local_transfer][construct]") {
  FutureShm* future_shm = CreateFutureShm(4096);
  auto future_ptr = MakeFullPtr(future_shm);

  std::vector<char> data = GenerateTestData(1000);
  LocalTransfer transfer(std::move(data), future_ptr);

  REQUIRE(transfer.IsSender());
  REQUIRE_FALSE(transfer.IsComplete());
  REQUIRE(transfer.GetTotalSize() == 1000);
  REQUIRE(transfer.GetBytesTransferred() == 0);

  // Verify output_size was set in FutureShm
  REQUIRE(future_shm->output_.total_written_.load() == 1000);

  DestroyFutureShm(future_shm);
  INFO("Sender construction test passed");
}

TEST_CASE("LocalTransfer - Receiver Construction", "[local_transfer][construct]") {
  FutureShm* future_shm = CreateFutureShm(4096);
  auto future_ptr = MakeFullPtr(future_shm);

  LocalTransfer transfer(future_ptr, 1000);

  REQUIRE_FALSE(transfer.IsSender());
  REQUIRE_FALSE(transfer.IsComplete());
  REQUIRE(transfer.GetTotalSize() == 1000);
  REQUIRE(transfer.GetBytesTransferred() == 0);

  DestroyFutureShm(future_shm);
  INFO("Receiver construction test passed");
}

TEST_CASE("LocalTransfer - Move Constructor", "[local_transfer][construct]") {
  FutureShm* future_shm = CreateFutureShm(4096);
  auto future_ptr = MakeFullPtr(future_shm);

  std::vector<char> data = GenerateTestData(500);
  LocalTransfer original(std::move(data), future_ptr);

  REQUIRE(original.IsSender());
  REQUIRE(original.GetTotalSize() == 500);

  // Move construct
  LocalTransfer moved(std::move(original));

  // Verify moved object has the state
  REQUIRE(moved.IsSender());
  REQUIRE(moved.GetTotalSize() == 500);
  REQUIRE_FALSE(moved.IsComplete());

  DestroyFutureShm(future_shm);
  INFO("Move constructor test passed");
}

TEST_CASE("LocalTransfer - Move Assignment", "[local_transfer][construct]") {
  FutureShm* future_shm1 = CreateFutureShm(4096);
  FutureShm* future_shm2 = CreateFutureShm(4096);
  auto future_ptr1 = MakeFullPtr(future_shm1);
  auto future_ptr2 = MakeFullPtr(future_shm2);

  std::vector<char> data1 = GenerateTestData(500);
  std::vector<char> data2 = GenerateTestData(1000);

  LocalTransfer transfer1(std::move(data1), future_ptr1);
  LocalTransfer transfer2(std::move(data2), future_ptr2);

  // Move assign
  transfer1 = std::move(transfer2);

  // Verify transfer1 now has transfer2's state
  REQUIRE(transfer1.IsSender());
  REQUIRE(transfer1.GetTotalSize() == 1000);

  DestroyFutureShm(future_shm1);
  DestroyFutureShm(future_shm2);
  INFO("Move assignment test passed");
}

// ============================================================================
// Single-Chunk Transfer Tests (data fits in copy_space)
// ============================================================================

TEST_CASE("LocalTransfer - Single Chunk Send/Recv", "[local_transfer][single]") {
  const size_t COPY_SPACE_SIZE = 4096;
  const size_t DATA_SIZE = 1000;  // Fits in single chunk

  FutureShm* future_shm = CreateFutureShm(COPY_SPACE_SIZE);
  auto future_ptr = MakeFullPtr(future_shm);

  // Create sender
  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  LocalTransfer sender(std::move(send_data), future_ptr);

  // Send all data (should complete in one call)
  bool send_complete = sender.Send(100000);  // 100ms budget
  REQUIRE(send_complete);
  REQUIRE(sender.IsComplete());
  REQUIRE(sender.GetBytesTransferred() == DATA_SIZE);

  // Verify FUTURE_NEW_DATA is set
  REQUIRE(future_shm->flags_.Any(FutureShm::FUTURE_NEW_DATA));

  // Create receiver with same FutureShm
  LocalTransfer receiver(future_ptr, DATA_SIZE);

  // Receive all data (blocks until complete)
  bool recv_complete = receiver.Recv();
  REQUIRE(recv_complete);
  REQUIRE(receiver.IsComplete());
  REQUIRE(receiver.GetBytesTransferred() == DATA_SIZE);

  // Verify FUTURE_NEW_DATA is cleared
  REQUIRE_FALSE(future_shm->flags_.Any(FutureShm::FUTURE_NEW_DATA));

  // Verify FUTURE_COMPLETE is set
  REQUIRE(future_shm->flags_.Any(FutureShm::FUTURE_COMPLETE));

  // Verify data integrity
  REQUIRE(VerifyTestData(receiver.GetData(), DATA_SIZE));

  DestroyFutureShm(future_shm);
  INFO("Single chunk transfer test passed");
}

// ============================================================================
// Multi-Chunk Transfer Tests (data larger than copy_space) - Threaded
// ============================================================================

TEST_CASE("LocalTransfer - Multi Chunk Transfer (Threaded)", "[local_transfer][multi]") {
  const size_t COPY_SPACE_SIZE = 256;  // Small copy_space to force multiple chunks
  const size_t DATA_SIZE = 1000;       // ~4 chunks needed

  FutureShm* future_shm = CreateFutureShm(COPY_SPACE_SIZE);
  auto future_ptr = MakeFullPtr(future_shm);

  // Prepare sender data
  std::vector<char> send_data = GenerateTestData(DATA_SIZE);

  bool send_complete = false;
  bool recv_complete = false;
  std::vector<char> received_data;

  // Sender thread
  std::thread sender_thread([&]() {
    LocalTransfer sender(std::move(send_data), future_ptr);
    while (!sender.IsComplete()) {
      sender.Send(1000);  // 1ms per call
    }
    send_complete = true;
  });

  // Receiver thread
  std::thread receiver_thread([&]() {
    LocalTransfer receiver(future_ptr, DATA_SIZE);
    receiver.Recv();  // Blocks until complete
    received_data = std::move(receiver.GetData());
    recv_complete = true;
  });

  // Wait for both threads
  sender_thread.join();
  receiver_thread.join();

  REQUIRE(send_complete);
  REQUIRE(recv_complete);
  REQUIRE(received_data.size() == DATA_SIZE);
  REQUIRE(VerifyTestData(received_data, DATA_SIZE));

  DestroyFutureShm(future_shm);
  INFO("Multi-chunk threaded transfer test passed");
}

// ============================================================================
// Large Data Transfer Tests
// ============================================================================

TEST_CASE("LocalTransfer - Large Data (64KB) Threaded", "[local_transfer][large]") {
  const size_t COPY_SPACE_SIZE = 4096;
  const size_t DATA_SIZE = 64 * 1024;  // 64KB

  FutureShm* future_shm = CreateFutureShm(COPY_SPACE_SIZE);
  auto future_ptr = MakeFullPtr(future_shm);

  // Prepare sender data
  std::vector<char> send_data = GenerateTestData(DATA_SIZE);

  bool send_complete = false;
  bool recv_complete = false;
  std::vector<char> received_data;

  // Sender thread
  std::thread sender_thread([&]() {
    LocalTransfer sender(std::move(send_data), future_ptr);
    while (!sender.IsComplete()) {
      sender.Send(10000);  // 10ms per call
    }
    send_complete = true;
  });

  // Receiver thread
  std::thread receiver_thread([&]() {
    LocalTransfer receiver(future_ptr, DATA_SIZE);
    receiver.Recv();  // Blocks until complete
    received_data = std::move(receiver.GetData());
    recv_complete = true;
  });

  // Wait for both threads
  sender_thread.join();
  receiver_thread.join();

  REQUIRE(send_complete);
  REQUIRE(recv_complete);
  REQUIRE(received_data.size() == DATA_SIZE);
  REQUIRE(VerifyTestData(received_data, DATA_SIZE));

  DestroyFutureShm(future_shm);
  INFO("64KB threaded transfer test passed");
}

TEST_CASE("LocalTransfer - Very Large Data (1MB) Threaded", "[local_transfer][verylarge]") {
  const size_t COPY_SPACE_SIZE = 4096;
  const size_t DATA_SIZE = 1024 * 1024;  // 1MB

  FutureShm* future_shm = CreateFutureShm(COPY_SPACE_SIZE);
  auto future_ptr = MakeFullPtr(future_shm);

  // Prepare sender data
  std::vector<char> send_data = GenerateTestData(DATA_SIZE);

  bool send_complete = false;
  bool recv_complete = false;
  std::vector<char> received_data;

  // Sender thread - runs concurrently with receiver
  std::thread sender_thread([&]() {
    LocalTransfer sender(std::move(send_data), future_ptr);
    while (!sender.IsComplete()) {
      sender.Send(50000);  // 50ms per call
    }
    send_complete = true;
  });

  // Receiver thread - runs concurrently with sender
  std::thread receiver_thread([&]() {
    LocalTransfer receiver(future_ptr, DATA_SIZE);
    receiver.Recv();  // Blocks until complete
    received_data = std::move(receiver.GetData());
    recv_complete = true;
  });

  // Wait for both threads
  sender_thread.join();
  receiver_thread.join();

  REQUIRE(send_complete);
  REQUIRE(recv_complete);
  REQUIRE(received_data.size() == DATA_SIZE);
  REQUIRE(VerifyTestData(received_data, DATA_SIZE));

  DestroyFutureShm(future_shm);
  INFO("1MB threaded transfer test passed");
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_CASE("LocalTransfer - Empty Data", "[local_transfer][edge]") {
  const size_t COPY_SPACE_SIZE = 4096;

  FutureShm* future_shm = CreateFutureShm(COPY_SPACE_SIZE);
  auto future_ptr = MakeFullPtr(future_shm);

  // Create sender with empty data
  std::vector<char> send_data;
  LocalTransfer sender(std::move(send_data), future_ptr);

  // Should be immediately complete
  bool send_complete = sender.Send(1000);
  REQUIRE(send_complete);
  REQUIRE(sender.IsComplete());
  REQUIRE(sender.GetBytesTransferred() == 0);

  // Create receiver expecting empty data
  LocalTransfer receiver(future_ptr, 0);

  // Should be immediately complete
  bool recv_complete = receiver.Recv();
  REQUIRE(recv_complete);
  REQUIRE(receiver.IsComplete());
  REQUIRE(receiver.GetData().empty());

  DestroyFutureShm(future_shm);
  INFO("Empty data transfer test passed");
}

TEST_CASE("LocalTransfer - Exact Copy Space Size", "[local_transfer][edge]") {
  const size_t COPY_SPACE_SIZE = 1024;
  const size_t DATA_SIZE = COPY_SPACE_SIZE;  // Exactly matches

  FutureShm* future_shm = CreateFutureShm(COPY_SPACE_SIZE);
  auto future_ptr = MakeFullPtr(future_shm);

  // Create sender
  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  LocalTransfer sender(std::move(send_data), future_ptr);

  // Should complete in one chunk
  bool send_complete = sender.Send(100000);
  REQUIRE(send_complete);
  REQUIRE(sender.GetBytesTransferred() == DATA_SIZE);

  // Create receiver
  LocalTransfer receiver(future_ptr, DATA_SIZE);

  bool recv_complete = receiver.Recv();
  REQUIRE(recv_complete);
  REQUIRE(VerifyTestData(receiver.GetData(), DATA_SIZE));

  DestroyFutureShm(future_shm);
  INFO("Exact copy space size test passed");
}

TEST_CASE("LocalTransfer - One Byte Over Copy Space (Threaded)", "[local_transfer][edge]") {
  const size_t COPY_SPACE_SIZE = 1024;
  const size_t DATA_SIZE = COPY_SPACE_SIZE + 1;  // One byte over

  FutureShm* future_shm = CreateFutureShm(COPY_SPACE_SIZE);
  auto future_ptr = MakeFullPtr(future_shm);

  // Prepare sender data
  std::vector<char> send_data = GenerateTestData(DATA_SIZE);

  bool send_complete = false;
  bool recv_complete = false;
  std::vector<char> received_data;

  // Sender thread
  std::thread sender_thread([&]() {
    LocalTransfer sender(std::move(send_data), future_ptr);
    while (!sender.IsComplete()) {
      sender.Send(10000);
    }
    send_complete = true;
  });

  // Receiver thread
  std::thread receiver_thread([&]() {
    LocalTransfer receiver(future_ptr, DATA_SIZE);
    receiver.Recv();
    received_data = std::move(receiver.GetData());
    recv_complete = true;
  });

  sender_thread.join();
  receiver_thread.join();

  REQUIRE(send_complete);
  REQUIRE(recv_complete);
  REQUIRE(VerifyTestData(received_data, DATA_SIZE));

  DestroyFutureShm(future_shm);
  INFO("One byte over copy space test passed");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_CASE("LocalTransfer - Invalid State Handling", "[local_transfer][error]") {
  // Test default-constructed LocalTransfer
  LocalTransfer invalid_transfer;

  // Send should fail gracefully
  bool send_result = invalid_transfer.Send(1000);
  REQUIRE_FALSE(send_result);

  // Recv should fail gracefully
  bool recv_result = invalid_transfer.Recv();
  REQUIRE_FALSE(recv_result);

  INFO("Invalid state handling test passed");
}

TEST_CASE("LocalTransfer - Zero Capacity Copy Space", "[local_transfer][error]") {
  // Create FutureShm with zero capacity
  FutureShm* future_shm = CreateFutureShm(0);
  auto future_ptr = MakeFullPtr(future_shm);

  // Create sender
  std::vector<char> send_data = GenerateTestData(100);
  LocalTransfer sender(std::move(send_data), future_ptr);

  // Send should fail due to zero capacity
  bool send_result = sender.Send(1000);
  REQUIRE_FALSE(send_result);

  // Create receiver
  LocalTransfer receiver(future_ptr, 100);

  // Recv should fail due to zero capacity
  bool recv_result = receiver.Recv();
  REQUIRE_FALSE(recv_result);

  DestroyFutureShm(future_shm);
  INFO("Zero capacity handling test passed");
}

// ============================================================================
// Concurrent Thread Test
// ============================================================================

TEST_CASE("LocalTransfer - Threaded Send/Recv", "[local_transfer][threaded]") {
  const size_t COPY_SPACE_SIZE = 1024;
  const size_t DATA_SIZE = 10000;  // 10KB

  FutureShm* future_shm = CreateFutureShm(COPY_SPACE_SIZE);
  auto future_ptr = MakeFullPtr(future_shm);

  // Prepare sender data
  std::vector<char> send_data = GenerateTestData(DATA_SIZE);

  bool send_complete = false;
  bool recv_complete = false;
  std::vector<char> received_data;

  // Sender thread
  std::thread sender_thread([&]() {
    LocalTransfer sender(std::move(send_data), future_ptr);
    while (!sender.IsComplete()) {
      sender.Send(5000);  // 5ms per call
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    send_complete = true;
  });

  // Receiver thread
  std::thread receiver_thread([&]() {
    LocalTransfer receiver(future_ptr, DATA_SIZE);
    receiver.Recv();  // Blocks until complete
    received_data = std::move(receiver.GetData());
    recv_complete = true;
  });

  // Wait for both threads
  sender_thread.join();
  receiver_thread.join();

  REQUIRE(send_complete);
  REQUIRE(recv_complete);
  REQUIRE(VerifyTestData(received_data, DATA_SIZE));

  DestroyFutureShm(future_shm);
  INFO("Threaded send/recv test passed");
}

SIMPLE_TEST_MAIN()
