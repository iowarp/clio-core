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

#include <hermes_shm/lightbeam/zmq_transport.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using namespace hshm::lbm;

// Custom metadata class that inherits from LbmMeta
class TestMeta : public LbmMeta {
 public:
  int request_id;
  std::string operation;

  template <typename Ar>
  void serialize(Ar& ar) {
    LbmMeta::serialize(ar);
    ar(request_id, operation);
  }
};

void TestBasicTransfer() {
  std::cout << "\n==== Testing Basic Transfer with New API ====\n";

#ifdef HSHM_ENABLE_ZMQ
  // Create server
  std::string addr = "127.0.0.1";
  std::string protocol = "tcp";
  int port = 8195;

  auto server = std::make_unique<ZeroMqTransport>(TransportMode::kServer, addr, protocol, port);
  auto client = std::make_unique<ZeroMqTransport>(TransportMode::kClient, addr, protocol, port);

  // Give ZMQ time to connect
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Prepare data on client side
  const char* data1 = "Hello, World!";
  const char* data2 = "Testing Lightbeam";
  size_t size1 = strlen(data1);
  size_t size2 = strlen(data2);

  // Create metadata and expose bulks
  TestMeta send_meta;
  send_meta.request_id = 42;
  send_meta.operation = "test_op";

  Bulk bulk1 = client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data1)), size1, BULK_XFER);
  Bulk bulk2 = client->Expose(
      hipc::FullPtr<char>(const_cast<char*>(data2)), size2, BULK_XFER);

  send_meta.send.push_back(bulk1);
  send_meta.send.push_back(bulk2);

  // Send metadata and bulks
  int rc = client->Send(send_meta);
  assert(rc == 0);
  std::cout << "Client sent data successfully\n";

  // Recv with retry loop (does everything - metadata + bulks)
  TestMeta recv_meta;
  while (true) {
    auto info = server->Recv(recv_meta);
    rc = info.rc;
    if (rc == 0) break;
    if (rc != EAGAIN) {
      std::cerr << "Recv failed with error: " << rc << "\n";
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::cout << "Server received metadata: request_id=" << recv_meta.request_id
            << ", operation=" << recv_meta.operation << "\n";
  assert(recv_meta.request_id == 42);
  assert(recv_meta.operation == "test_op");
  assert(recv_meta.send.size() == 2);
  std::cout << "Server received bulk data successfully\n";

  // Verify received data from transport-allocated recv buffers
  std::string received1(recv_meta.recv[0].data.ptr_,
                         recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  std::string received2(recv_meta.recv[1].data.ptr_,
                         recv_meta.recv[1].data.ptr_ + recv_meta.recv[1].size);

  std::cout << "Bulk 1: " << received1 << "\n";
  std::cout << "Bulk 2: " << received2 << "\n";

  assert(received1 == data1);
  assert(received2 == data2);

  std::cout << "[New API] Test passed!\n";
#else
  std::cout << "ZMQ not enabled, skipping test\n";
#endif
}

void TestMultipleBulks() {
  std::cout << "\n==== Testing Multiple Bulks Transfer ====\n";

#ifdef HSHM_ENABLE_ZMQ
  std::string addr = "127.0.0.1";
  std::string protocol = "tcp";
  int port = 8196;

  auto server = std::make_unique<ZeroMqTransport>(TransportMode::kServer, addr, protocol, port);
  auto client = std::make_unique<ZeroMqTransport>(TransportMode::kClient, addr, protocol, port);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Prepare multiple data chunks
  std::vector<std::string> data_chunks = {"Chunk 1", "Chunk 2 is longer",
                                          "Chunk 3", "Final chunk 4"};

  // Create metadata
  LbmMeta send_meta;
  for (const auto& chunk : data_chunks) {
    Bulk bulk = client->Expose(
        hipc::FullPtr<char>(const_cast<char*>(chunk.data())),
        chunk.size(), BULK_XFER);
    send_meta.send.push_back(bulk);
  }

  // Send
  int rc = client->Send(send_meta);
  assert(rc == 0);

  // Recv with retry loop (does everything - metadata + bulks)
  LbmMeta recv_meta;
  while (true) {
    auto info = server->Recv(recv_meta);
    rc = info.rc;
    if (rc == 0) break;
    if (rc != EAGAIN) {
      std::cerr << "Recv failed with error: " << rc << "\n";
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  assert(recv_meta.send.size() == data_chunks.size());

  // Verify all chunks from transport-allocated recv buffers
  for (size_t i = 0; i < data_chunks.size(); ++i) {
    std::string received(recv_meta.recv[i].data.ptr_,
                         recv_meta.recv[i].data.ptr_ + recv_meta.recv[i].size);
    std::cout << "Chunk " << i << ": " << received << "\n";
    assert(received == data_chunks[i]);
  }

  std::cout << "[Multiple Bulks] Test passed!\n";
#else
  std::cout << "ZMQ not enabled, skipping test\n";
#endif
}

int main() {
  TestBasicTransfer();
  TestMultipleBulks();
  std::cout << "\nAll new API tests passed!" << std::endl;
  return 0;
}
