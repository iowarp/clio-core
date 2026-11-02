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
 * Unit tests for ShmTransport Send/Recv (SPSC ring buffer)
 *
 * Tests the chunked data transfer mechanism using ShmTransferInfo and
 * copy_space directly, without requiring the full CLIO Runtime runtime.
 */

#include <catch2/catch_all.hpp>

#include <clio_ctp/lightbeam/shm_transport.h>

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

using namespace ctp::lbm;

// Helper context for tests
template <size_t N>
struct TestTransferContext {
  char copy_space[N];
  ShmTransferInfo shm_info;

  TestTransferContext() {
    std::memset(copy_space, 0, sizeof(copy_space));
    shm_info.copy_space_size_ = N;
  }

  LbmContext MakeCtx() {
    LbmContext ctx;
    ctx.copy_space = copy_space;
    ctx.shm_info_ = &shm_info;
    return ctx;
  }
};

// Generate test data with a predictable pattern
static std::vector<char> GenerateTestData(size_t size) {
  std::vector<char> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  return data;
}

// Verify data matches the expected pattern
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

// Build an LbmMeta with one BULK_XFER descriptor pointing at data
static LbmMeta<> MakeSendMeta(std::vector<char>& data) {
  LbmMeta<> meta;
  Bulk bulk;
  bulk.data.ptr_ = data.data();
  bulk.data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  bulk.data.shm_.off_ = 0;
  bulk.size = data.size();
  bulk.flags = ctp::bitfield32_t(BULK_XFER);
  meta.send.push_back(bulk);
  meta.send_bulks = 1;
  return meta;
}

// Verify recv meta contains the expected pattern data and free it
static bool VerifyAndFreeRecvMeta(LbmMeta<>& recv_meta, size_t expected_size) {
  if (recv_meta.recv.size() != 1) return false;
  if (recv_meta.recv[0].size != expected_size) return false;
  if (!recv_meta.recv[0].data.ptr_) return false;
  std::vector<char> data(recv_meta.recv[0].data.ptr_,
                         recv_meta.recv[0].data.ptr_ + expected_size);
  std::free(recv_meta.recv[0].data.ptr_);
  recv_meta.recv[0].data.ptr_ = nullptr;
  return VerifyTestData(data, expected_size);
}

// ============================================================================
// Single-Chunk Transfer Tests (data fits in copy_space)
// ============================================================================

TEST_CASE("ShmTransfer - Single Chunk Transfer", "[shm_transfer][single]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;
  constexpr size_t DATA_SIZE = 1000;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;
  auto ctx = ctx_store.MakeCtx();

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  auto send_meta = MakeSendMeta(send_data);

  // Data fits in single pass since DATA_SIZE < COPY_SPACE_SIZE
  ShmTransport::Send(send_meta, ctx);

  LbmMeta<> recv_meta;
  ShmTransport::Recv(recv_meta, ctx);

  REQUIRE(VerifyAndFreeRecvMeta(recv_meta, DATA_SIZE));
}

// ============================================================================
// Multi-Chunk Threaded Transfer Tests (data larger than copy_space)
// ============================================================================

TEST_CASE("ShmTransfer - Multi Chunk Transfer (Threaded)", "[shm_transfer][multi]") {
  constexpr size_t COPY_SPACE_SIZE = 256;
  constexpr size_t DATA_SIZE = 1000;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  bool send_done = false;
  bool recv_done = false;
  LbmMeta<> recv_meta;

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    auto meta = MakeSendMeta(send_data);
    ShmTransport::Send(meta, ctx);
    send_done = true;
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::Recv(recv_meta, ctx);
    recv_done = true;
  });

  sender.join();
  receiver.join();

  REQUIRE(send_done);
  REQUIRE(recv_done);
  REQUIRE(VerifyAndFreeRecvMeta(recv_meta, DATA_SIZE));
}

// ============================================================================
// Large Data Transfer Tests
// ============================================================================

TEST_CASE("ShmTransfer - Large Data (64KB) Threaded", "[shm_transfer][large]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;
  constexpr size_t DATA_SIZE = 64 * 1024;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  LbmMeta<> recv_meta;

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    auto meta = MakeSendMeta(send_data);
    ShmTransport::Send(meta, ctx);
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::Recv(recv_meta, ctx);
  });

  sender.join();
  receiver.join();

  REQUIRE(VerifyAndFreeRecvMeta(recv_meta, DATA_SIZE));
}

TEST_CASE("ShmTransfer - Very Large Data (1MB) Threaded", "[shm_transfer][verylarge]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;
  constexpr size_t DATA_SIZE = 1024 * 1024;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  LbmMeta<> recv_meta;

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    auto meta = MakeSendMeta(send_data);
    ShmTransport::Send(meta, ctx);
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::Recv(recv_meta, ctx);
  });

  sender.join();
  receiver.join();

  REQUIRE(VerifyAndFreeRecvMeta(recv_meta, DATA_SIZE));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_CASE("ShmTransfer - Empty Data", "[shm_transfer][edge]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;
  auto ctx = ctx_store.MakeCtx();

  // Send/Recv with no bulk descriptors — only metadata frame is transferred
  LbmMeta<> send_meta;
  send_meta.send_bulks = 0;
  ShmTransport::Send(send_meta, ctx);

  LbmMeta<> recv_meta;
  auto info = ShmTransport::Recv(recv_meta, ctx);

  REQUIRE(info.rc == 0);
  REQUIRE(recv_meta.recv.empty());
}

TEST_CASE("ShmTransfer - Exact Copy Space Size", "[shm_transfer][edge]") {
  constexpr size_t COPY_SPACE_SIZE = 1024;
  constexpr size_t DATA_SIZE = COPY_SPACE_SIZE / 2;  // fits within copy space

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;
  auto ctx = ctx_store.MakeCtx();

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  auto send_meta = MakeSendMeta(send_data);

  ShmTransport::Send(send_meta, ctx);

  LbmMeta<> recv_meta;
  ShmTransport::Recv(recv_meta, ctx);

  REQUIRE(VerifyAndFreeRecvMeta(recv_meta, DATA_SIZE));
}

TEST_CASE("ShmTransfer - Data Larger Than Copy Space (Threaded)", "[shm_transfer][edge]") {
  constexpr size_t COPY_SPACE_SIZE = 1024;
  constexpr size_t DATA_SIZE = COPY_SPACE_SIZE * 4;  // multiple ring buffer passes

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  LbmMeta<> recv_meta;

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    auto meta = MakeSendMeta(send_data);
    ShmTransport::Send(meta, ctx);
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::Recv(recv_meta, ctx);
  });

  sender.join();
  receiver.join();

  REQUIRE(VerifyAndFreeRecvMeta(recv_meta, DATA_SIZE));
}

// ============================================================================
// Concurrent Transfer Tests
// ============================================================================

TEST_CASE("ShmTransfer - Concurrent with Delays", "[shm_transfer][threaded]") {
  constexpr size_t COPY_SPACE_SIZE = 1024;
  constexpr size_t DATA_SIZE = 10000;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  std::vector<char> send_data = GenerateTestData(DATA_SIZE);
  LbmMeta<> recv_meta;

  // DATA_SIZE > COPY_SPACE_SIZE, exercises SPSC backpressure
  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    auto meta = MakeSendMeta(send_data);
    ShmTransport::Send(meta, ctx);
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    ShmTransport::Recv(recv_meta, ctx);
  });

  sender.join();
  receiver.join();

  REQUIRE(VerifyAndFreeRecvMeta(recv_meta, DATA_SIZE));
}

// ============================================================================
// SendMsg/RecvMsg API Tests (full serialization path)
// ============================================================================

TEST_CASE("ShmTransfer - Send/Recv Basic", "[shm_transfer][sendrecv]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;
  constexpr size_t DATA_SIZE = 512;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  // Prepare send metadata with one bulk descriptor
  LbmMeta<> send_meta;
  std::vector<char> bulk_data = GenerateTestData(DATA_SIZE);
  Bulk bulk;
  bulk.data.ptr_ = bulk_data.data();
  bulk.data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  bulk.data.shm_.off_ = 0;
  bulk.size = DATA_SIZE;
  bulk.flags = ctp::bitfield32_t(BULK_XFER);
  send_meta.send.push_back(bulk);
  send_meta.send_bulks = 1;

  int send_rc = -1;
  LbmMeta<> recv_meta;
  int recv_rc = -1;

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    send_rc = ShmTransport::Send(send_meta, ctx);
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    auto info = ShmTransport::Recv(recv_meta, ctx);
    recv_rc = info.rc;
  });

  sender.join();
  receiver.join();

  REQUIRE(send_rc == 0);
  REQUIRE(recv_rc == 0);
  REQUIRE(recv_meta.send.size() == 1);
  REQUIRE(recv_meta.recv.size() == 1);
  REQUIRE(recv_meta.recv[0].size == DATA_SIZE);
  REQUIRE(recv_meta.recv[0].data.ptr_ != nullptr);

  // Verify received data matches
  bool data_correct = true;
  for (size_t i = 0; i < DATA_SIZE; ++i) {
    if (recv_meta.recv[0].data.ptr_[i] != static_cast<char>(i % 256)) {
      data_correct = false;
      break;
    }
  }
  REQUIRE(data_correct);

  std::free(recv_meta.recv[0].data.ptr_);
}

TEST_CASE("ShmTransfer - Send/Recv Large Multi-Chunk", "[shm_transfer][sendrecv]") {
  constexpr size_t COPY_SPACE_SIZE = 1024;
  constexpr size_t DATA_SIZE = 32 * 1024;  // 32KB through 1KB ring

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  LbmMeta<> send_meta;
  std::vector<char> bulk_data = GenerateTestData(DATA_SIZE);
  Bulk bulk;
  bulk.data.ptr_ = bulk_data.data();
  bulk.data.shm_.alloc_id_ = ctp::ipc::AllocatorId::GetNull();
  bulk.data.shm_.off_ = 0;
  bulk.size = DATA_SIZE;
  bulk.flags = ctp::bitfield32_t(BULK_XFER);
  send_meta.send.push_back(bulk);
  send_meta.send_bulks = 1;

  int send_rc = -1;
  LbmMeta<> recv_meta;
  int recv_rc = -1;

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    send_rc = ShmTransport::Send(send_meta, ctx);
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    auto info = ShmTransport::Recv(recv_meta, ctx);
    recv_rc = info.rc;
  });

  sender.join();
  receiver.join();

  REQUIRE(send_rc == 0);
  REQUIRE(recv_rc == 0);
  REQUIRE(recv_meta.recv.size() == 1);
  REQUIRE(recv_meta.recv[0].size == DATA_SIZE);

  bool data_correct = true;
  for (size_t i = 0; i < DATA_SIZE; ++i) {
    if (recv_meta.recv[0].data.ptr_[i] != static_cast<char>(i % 256)) {
      data_correct = false;
      break;
    }
  }
  REQUIRE(data_correct);

  std::free(recv_meta.recv[0].data.ptr_);
}

TEST_CASE("ShmTransfer - Send/Recv Metadata Only", "[shm_transfer][sendrecv]") {
  constexpr size_t COPY_SPACE_SIZE = 4096;

  TestTransferContext<COPY_SPACE_SIZE> ctx_store;

  // Send metadata with no bulk descriptors
  LbmMeta<> send_meta;
  send_meta.send_bulks = 0;

  int send_rc = -1;
  LbmMeta<> recv_meta;
  int recv_rc = -1;

  std::thread sender([&]() {
    auto ctx = ctx_store.MakeCtx();
    send_rc = ShmTransport::Send(send_meta, ctx);
  });

  std::thread receiver([&]() {
    auto ctx = ctx_store.MakeCtx();
    auto info = ShmTransport::Recv(recv_meta, ctx);
    recv_rc = info.rc;
  });

  sender.join();
  receiver.join();

  REQUIRE(send_rc == 0);
  REQUIRE(recv_rc == 0);
  REQUIRE(recv_meta.send.empty());
  REQUIRE(recv_meta.recv.empty());
}
