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

#include <hermes_shm/lightbeam/shm_transport.h>
#include <hermes_shm/lightbeam/transport_factory_impl.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using namespace hshm::lbm;

// Shared copy-space buffer and synchronization primitives
static constexpr size_t kCopySpaceSize = 256;

struct ShmTestContext {
  char copy_space[kCopySpaceSize];
  ShmTransferInfo shm_info;

  ShmTestContext() {
    std::memset(copy_space, 0, sizeof(copy_space));
    shm_info.copy_space_size_ = kCopySpaceSize;
  }
};

static LbmContext MakeCtx(ShmTestContext& shared) {
  LbmContext ctx;
  ctx.copy_space = shared.copy_space;
  ctx.shm_info_ = &shared.shm_info;
  return ctx;
}

// Custom metadata class that inherits from LbmMeta
class TestMeta : public LbmMeta {
 public:
  int request_id = 0;
  std::string operation;

  template <typename Ar>
  void serialize(Ar& ar) {
    LbmMeta::serialize(ar);
    ar(request_id, operation);
  }
};

void TestBasicShmTransfer() {
  std::cout << "\n==== Testing SHM Basic Transfer ====\n";

  ShmTestContext shared;
  ShmClient client;
  ShmServer server;
  LbmContext ctx = MakeCtx(shared);

  const char* data1 = "Hello, World!";
  const char* data2 = "Testing SHM Transport";
  size_t size1 = strlen(data1);
  size_t size2 = strlen(data2);

  TestMeta send_meta;
  send_meta.request_id = 42;
  send_meta.operation = "shm_test";

  Bulk bulk1 = client.Expose(hipc::FullPtr<char>(const_cast<char*>(data1)),
                             size1, BULK_XFER);
  Bulk bulk2 = client.Expose(hipc::FullPtr<char>(const_cast<char*>(data2)),
                             size2, BULK_XFER);
  send_meta.send.push_back(bulk1);
  send_meta.send.push_back(bulk2);
  send_meta.send_bulks = 2;

  // Client sends in one thread, server receives in another
  int send_rc = -1;
  std::thread sender([&]() {
    send_rc = client.Send(send_meta, ctx);
  });

  TestMeta recv_meta;
  int rc = server.RecvMetadata(recv_meta, ctx);
  assert(rc == 0);
  std::cout << "Server received metadata: request_id=" << recv_meta.request_id
            << ", operation=" << recv_meta.operation << "\n";
  assert(recv_meta.request_id == 42);
  assert(recv_meta.operation == "shm_test");
  assert(recv_meta.send.size() == 2);

  // Allocate receive buffers
  std::vector<char> recv_buf1(recv_meta.send[0].size);
  std::vector<char> recv_buf2(recv_meta.send[1].size);
  recv_meta.recv.push_back(server.Expose(
      hipc::FullPtr<char>(recv_buf1.data()), recv_buf1.size(),
      recv_meta.send[0].flags.bits_));
  recv_meta.recv.push_back(server.Expose(
      hipc::FullPtr<char>(recv_buf2.data()), recv_buf2.size(),
      recv_meta.send[1].flags.bits_));

  rc = server.RecvBulks(recv_meta, ctx);
  assert(rc == 0);

  sender.join();
  assert(send_rc == 0);

  std::string received1(recv_buf1.begin(), recv_buf1.end());
  std::string received2(recv_buf2.begin(), recv_buf2.end());
  std::cout << "Bulk 1: " << received1 << "\n";
  std::cout << "Bulk 2: " << received2 << "\n";
  assert(received1 == data1);
  assert(received2 == data2);

  std::cout << "[SHM Basic] Test passed!\n";
}

void TestMultipleBulks() {
  std::cout << "\n==== Testing SHM Multiple Bulks ====\n";

  ShmTestContext shared;
  ShmClient client;
  ShmServer server;
  LbmContext ctx = MakeCtx(shared);

  std::vector<std::string> data_chunks = {"Chunk 1", "Chunk 2 is longer",
                                          "Chunk 3", "Final chunk 4"};

  LbmMeta send_meta;
  for (const auto& chunk : data_chunks) {
    Bulk bulk = client.Expose(
        hipc::FullPtr<char>(const_cast<char*>(chunk.data())),
        chunk.size(), BULK_XFER);
    send_meta.send.push_back(bulk);
    send_meta.send_bulks++;
  }

  int send_rc = -1;
  std::thread sender([&]() {
    send_rc = client.Send(send_meta, ctx);
  });

  LbmMeta recv_meta;
  int rc = server.RecvMetadata(recv_meta, ctx);
  assert(rc == 0);
  assert(recv_meta.send.size() == data_chunks.size());

  std::vector<std::vector<char>> recv_buffers;
  for (size_t i = 0; i < recv_meta.send.size(); ++i) {
    recv_buffers.emplace_back(recv_meta.send[i].size);
    recv_meta.recv.push_back(server.Expose(
        hipc::FullPtr<char>(recv_buffers[i].data()),
        recv_buffers[i].size(),
        recv_meta.send[i].flags.bits_));
  }

  rc = server.RecvBulks(recv_meta, ctx);
  assert(rc == 0);

  sender.join();
  assert(send_rc == 0);

  for (size_t i = 0; i < data_chunks.size(); ++i) {
    std::string received(recv_buffers[i].begin(), recv_buffers[i].end());
    std::cout << "Chunk " << i << ": " << received << "\n";
    assert(received == data_chunks[i]);
  }

  std::cout << "[SHM Multiple Bulks] Test passed!\n";
}

void TestMetadataOnly() {
  std::cout << "\n==== Testing SHM Metadata Only (No Bulks) ====\n";

  ShmTestContext shared;
  ShmClient client;
  ShmServer server;
  LbmContext ctx = MakeCtx(shared);

  TestMeta send_meta;
  send_meta.request_id = 7;
  send_meta.operation = "ping";
  send_meta.send_bulks = 0;

  int send_rc = -1;
  std::thread sender([&]() {
    send_rc = client.Send(send_meta, ctx);
  });

  TestMeta recv_meta;
  int rc = server.RecvMetadata(recv_meta, ctx);
  assert(rc == 0);

  sender.join();
  assert(send_rc == 0);

  assert(recv_meta.request_id == 7);
  assert(recv_meta.operation == "ping");
  assert(recv_meta.send.empty());

  std::cout << "[SHM Metadata Only] Test passed!\n";
}

void TestLargeTransfer() {
  std::cout << "\n==== Testing SHM Large Transfer (multi-chunk) ====\n";

  ShmTestContext shared;
  ShmClient client;
  ShmServer server;
  LbmContext ctx = MakeCtx(shared);

  // Create data larger than copy_space_size to force chunking
  std::string large_data(kCopySpaceSize * 5 + 37, 'X');
  for (size_t i = 0; i < large_data.size(); ++i) {
    large_data[i] = static_cast<char>('A' + (i % 26));
  }

  LbmMeta send_meta;
  Bulk bulk = client.Expose(
      hipc::FullPtr<char>(const_cast<char*>(large_data.data())),
      large_data.size(), BULK_XFER);
  send_meta.send.push_back(bulk);
  send_meta.send_bulks = 1;

  int send_rc = -1;
  std::thread sender([&]() {
    send_rc = client.Send(send_meta, ctx);
  });

  LbmMeta recv_meta;
  int rc = server.RecvMetadata(recv_meta, ctx);
  assert(rc == 0);
  assert(recv_meta.send.size() == 1);

  // Use server-allocated buffer (nullptr -> malloc)
  recv_meta.recv.push_back(server.Expose(
      hipc::FullPtr<char>(nullptr), recv_meta.send[0].size,
      recv_meta.send[0].flags.bits_));

  rc = server.RecvBulks(recv_meta, ctx);
  assert(rc == 0);

  sender.join();
  assert(send_rc == 0);

  std::string received(recv_meta.recv[0].data.ptr_,
                       recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  assert(received == large_data);
  std::cout << "Transferred " << large_data.size()
            << " bytes through " << kCopySpaceSize
            << "-byte copy space (" << (large_data.size() / kCopySpaceSize + 1)
            << " chunks)\n";

  server.ClearRecvHandles(recv_meta);
  std::cout << "[SHM Large Transfer] Test passed!\n";
}

void TestShmPtrPassthrough() {
  std::cout << "\n==== Testing SHM Pointer Passthrough (no data copy) ====\n";

  ShmTestContext shared;
  ShmClient client;
  ShmServer server;
  LbmContext ctx = MakeCtx(shared);

  // Simulate a bulk whose data lives in shared memory (non-null alloc_id)
  hipc::FullPtr<char> shm_ptr;
  shm_ptr.ptr_ = reinterpret_cast<char*>(0xDEADBEEF);
  shm_ptr.shm_.alloc_id_ = hipc::AllocatorId(1, 2);
  shm_ptr.shm_.off_ = 0x1234;

  LbmMeta send_meta;
  Bulk bulk;
  bulk.data = shm_ptr;
  bulk.size = 4096;
  bulk.flags = hshm::bitfield32_t(BULK_XFER);
  send_meta.send.push_back(bulk);
  send_meta.send_bulks = 1;

  int send_rc = -1;
  std::thread sender([&]() {
    send_rc = client.Send(send_meta, ctx);
  });

  LbmMeta recv_meta;
  int rc = server.RecvMetadata(recv_meta, ctx);
  assert(rc == 0);

  // Provide a recv entry — ptr_ and shm_ will be overwritten by RecvBulks
  Bulk recv_bulk;
  recv_bulk.size = recv_meta.send[0].size;
  recv_bulk.flags = recv_meta.send[0].flags;
  recv_meta.recv.push_back(recv_bulk);

  rc = server.RecvBulks(recv_meta, ctx);
  assert(rc == 0);

  sender.join();
  assert(send_rc == 0);

  // Verify: ptr_ should be nullptr, shm_ should carry the original ShmPtr
  assert(recv_meta.recv[0].data.ptr_ == nullptr);
  assert(recv_meta.recv[0].data.shm_.alloc_id_ == hipc::AllocatorId(1, 2));
  assert(recv_meta.recv[0].data.shm_.off_.load() == 0x1234);

  std::cout << "ShmPtr passed through: alloc_id=("
            << recv_meta.recv[0].data.shm_.alloc_id_.major_ << ","
            << recv_meta.recv[0].data.shm_.alloc_id_.minor_ << ") off=0x"
            << std::hex << recv_meta.recv[0].data.shm_.off_.load()
            << std::dec << "\n";
  std::cout << "[SHM Pointer Passthrough] Test passed!\n";
}

void TestMixedBulks() {
  std::cout << "\n==== Testing SHM Mixed Bulks (data copy + ShmPtr) ====\n";

  ShmTestContext shared;
  ShmClient client;
  ShmServer server;
  LbmContext ctx = MakeCtx(shared);

  // Bulk 0: private memory (full copy)
  const char* private_data = "private heap data";
  size_t private_size = strlen(private_data);

  // Bulk 1: shared memory (ShmPtr passthrough)
  hipc::FullPtr<char> shm_ptr;
  shm_ptr.ptr_ = reinterpret_cast<char*>(0xCAFEBABE);
  shm_ptr.shm_.alloc_id_ = hipc::AllocatorId(3, 4);
  shm_ptr.shm_.off_ = 0x5678;

  LbmMeta send_meta;
  // Private bulk
  Bulk bulk0 = client.Expose(
      hipc::FullPtr<char>(const_cast<char*>(private_data)),
      private_size, BULK_XFER);
  send_meta.send.push_back(bulk0);
  // ShmPtr bulk
  Bulk bulk1;
  bulk1.data = shm_ptr;
  bulk1.size = 8192;
  bulk1.flags = hshm::bitfield32_t(BULK_XFER);
  send_meta.send.push_back(bulk1);
  send_meta.send_bulks = 2;

  int send_rc = -1;
  std::thread sender([&]() {
    send_rc = client.Send(send_meta, ctx);
  });

  LbmMeta recv_meta;
  int rc = server.RecvMetadata(recv_meta, ctx);
  assert(rc == 0);
  assert(recv_meta.send.size() == 2);

  // Recv bulk 0: pre-allocated buffer for data copy
  std::vector<char> recv_buf0(recv_meta.send[0].size);
  recv_meta.recv.push_back(server.Expose(
      hipc::FullPtr<char>(recv_buf0.data()), recv_buf0.size(),
      recv_meta.send[0].flags.bits_));
  // Recv bulk 1: empty entry for ShmPtr
  Bulk recv_bulk1;
  recv_bulk1.size = recv_meta.send[1].size;
  recv_bulk1.flags = recv_meta.send[1].flags;
  recv_meta.recv.push_back(recv_bulk1);

  rc = server.RecvBulks(recv_meta, ctx);
  assert(rc == 0);

  sender.join();
  assert(send_rc == 0);

  // Verify bulk 0: full data copy
  std::string received0(recv_buf0.begin(), recv_buf0.end());
  assert(received0 == private_data);
  std::cout << "Bulk 0 (data copy): " << received0 << "\n";

  // Verify bulk 1: ShmPtr passthrough
  assert(recv_meta.recv[1].data.ptr_ == nullptr);
  assert(recv_meta.recv[1].data.shm_.alloc_id_ == hipc::AllocatorId(3, 4));
  assert(recv_meta.recv[1].data.shm_.off_.load() == 0x5678);
  std::cout << "Bulk 1 (ShmPtr): alloc_id=("
            << recv_meta.recv[1].data.shm_.alloc_id_.major_ << ","
            << recv_meta.recv[1].data.shm_.alloc_id_.minor_ << ") off=0x"
            << std::hex << recv_meta.recv[1].data.shm_.off_.load()
            << std::dec << "\n";

  std::cout << "[SHM Mixed Bulks] Test passed!\n";
}

void TestFactory() {
  std::cout << "\n==== Testing SHM via TransportFactory ====\n";

  auto client = TransportFactory::GetClient("", Transport::kShm);
  auto server = TransportFactory::GetServer("", Transport::kShm);
  assert(client != nullptr);
  assert(server != nullptr);
  assert(server->GetAddress() == "shm");

  ShmTestContext shared;
  LbmContext ctx = MakeCtx(shared);

  const char* data = "Factory test";
  size_t size = strlen(data);

  TestMeta send_meta;
  send_meta.request_id = 100;
  send_meta.operation = "factory";
  Bulk bulk = client->Expose(hipc::FullPtr<char>(const_cast<char*>(data)),
                             size, BULK_XFER);
  send_meta.send.push_back(bulk);
  send_meta.send_bulks = 1;

  int send_rc = -1;
  std::thread sender([&]() {
    send_rc = client->Send(send_meta, ctx);
  });

  TestMeta recv_meta;
  int rc = server->RecvMetadata(recv_meta, ctx);
  assert(rc == 0);
  assert(recv_meta.request_id == 100);
  assert(recv_meta.operation == "factory");

  std::vector<char> recv_buf(recv_meta.send[0].size);
  recv_meta.recv.push_back(server->Expose(
      hipc::FullPtr<char>(recv_buf.data()), recv_buf.size(),
      recv_meta.send[0].flags.bits_));

  rc = server->RecvBulks(recv_meta, ctx);
  assert(rc == 0);

  sender.join();
  assert(send_rc == 0);

  std::string received(recv_buf.begin(), recv_buf.end());
  std::cout << "Received: " << received << "\n";
  assert(received == data);

  std::cout << "[SHM Factory] Test passed!\n";
}

int main() {
  TestBasicShmTransfer();
  TestMultipleBulks();
  TestMetadataOnly();
  TestLargeTransfer();
  TestShmPtrPassthrough();
  TestMixedBulks();
  TestFactory();
  std::cout << "\nAll SHM transport tests passed!" << std::endl;
  return 0;
}
