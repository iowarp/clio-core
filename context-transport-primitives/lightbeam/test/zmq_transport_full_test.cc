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

#include <clio_ctp/lightbeam/zmq_transport.h>
#include <clio_ctp/lightbeam/transport_factory_impl.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using namespace ctp::lbm;

class TestMeta : public LbmMeta<> {
 public:
  int request_id = 0;
  std::string operation;

  template <typename Ar>
  void serialize(Ar& ar) {
    LbmMeta<>::serialize(ar);
    ar(request_id, operation);
  }
};

// Helper: recv with retry loop
template <typename MetaT>
ClientInfo RecvWithRetry(Transport* server, MetaT& meta, int max_ms = 5000) {
  int attempts = 0;
  while (true) {
    auto info = server->Recv(meta);
    if (info.rc == 0) return info;
    if (info.rc != EAGAIN) return info;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (++attempts > max_ms) {
      std::cerr << "RecvWithRetry timed out\n";
      return info;
    }
  }
}

void TestBasicTransfer() {
  std::cout << "\n==== Testing ZMQ Basic Transfer ====\n";

  std::string addr = "127.0.0.1";
  int port = 8300;

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<ZeroMqTransport>(
      TransportMode::kClient, addr, "tcp", port);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  const char* data1 = "Hello, ZMQ World!";
  const char* data2 = "Testing ZMQ Transport";
  size_t size1 = strlen(data1);
  size_t size2 = strlen(data2);

  TestMeta send_meta;
  send_meta.request_id = 42;
  send_meta.operation = "zmq_test";
  send_meta.send.push_back(client->Expose(
      ctp::ipc::FullPtr<char>(const_cast<char*>(data1)), size1, BULK_XFER));
  send_meta.send.push_back(client->Expose(
      ctp::ipc::FullPtr<char>(const_cast<char*>(data2)), size2, BULK_XFER));

  int rc = client->Send(send_meta);
  assert(rc == 0);

  TestMeta recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.request_id == 42);
  assert(recv_meta.operation == "zmq_test");
  assert(recv_meta.send.size() == 2);

  std::string received1(recv_meta.recv[0].data.ptr_,
                         recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  std::string received2(recv_meta.recv[1].data.ptr_,
                         recv_meta.recv[1].data.ptr_ + recv_meta.recv[1].size);
  assert(received1 == data1);
  assert(received2 == data2);

  server->ClearRecvHandles(recv_meta);
  std::cout << "[ZMQ Basic Transfer] Test passed!\n";
}

void TestMultipleBulks() {
  std::cout << "\n==== Testing ZMQ Multiple Bulks ====\n";

  std::string addr = "127.0.0.1";
  int port = 8301;

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<ZeroMqTransport>(
      TransportMode::kClient, addr, "tcp", port);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::vector<std::string> data_chunks = {"Chunk 1", "Chunk 2 is longer",
                                          "Chunk 3", "Final chunk 4"};

  LbmMeta<> send_meta;
  for (const auto& chunk : data_chunks) {
    send_meta.send.push_back(client->Expose(
        ctp::ipc::FullPtr<char>(const_cast<char*>(chunk.data())),
        chunk.size(), BULK_XFER));
  }

  int rc = client->Send(send_meta);
  assert(rc == 0);

  LbmMeta<> recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.send.size() == data_chunks.size());

  for (size_t i = 0; i < data_chunks.size(); ++i) {
    std::string received(recv_meta.recv[i].data.ptr_,
                         recv_meta.recv[i].data.ptr_ + recv_meta.recv[i].size);
    assert(received == data_chunks[i]);
  }

  server->ClearRecvHandles(recv_meta);
  std::cout << "[ZMQ Multiple Bulks] Test passed!\n";
}

void TestMetadataOnly() {
  std::cout << "\n==== Testing ZMQ Metadata Only ====\n";

  std::string addr = "127.0.0.1";
  int port = 8302;

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<ZeroMqTransport>(
      TransportMode::kClient, addr, "tcp", port);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  TestMeta send_meta;
  send_meta.request_id = 7;
  send_meta.operation = "ping";
  send_meta.send_bulks = 0;

  int rc = client->Send(send_meta);
  assert(rc == 0);

  TestMeta recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.request_id == 7);
  assert(recv_meta.operation == "ping");
  assert(recv_meta.send.empty());

  std::cout << "[ZMQ Metadata Only] Test passed!\n";
}

void TestBulkExpose() {
  std::cout << "\n==== Testing ZMQ BULK_EXPOSE ====\n";

  std::string addr = "127.0.0.1";
  int port = 8303;

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<ZeroMqTransport>(
      TransportMode::kClient, addr, "tcp", port);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  const char* data = "expose_only_data";
  size_t size = strlen(data);

  LbmMeta<> send_meta;
  // BULK_EXPOSE: only metadata, no data transfer
  send_meta.send.push_back(client->Expose(
      ctp::ipc::FullPtr<char>(const_cast<char*>(data)), size, BULK_EXPOSE));
  send_meta.send_bulks = 0;  // No BULK_XFER entries

  int rc = client->Send(send_meta);
  assert(rc == 0);

  LbmMeta<> recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.recv.size() == 1);
  // BULK_EXPOSE entry: data pointer should be null (no data transferred)
  assert(recv_meta.recv[0].data.IsNull());
  assert(recv_meta.recv[0].size == size);

  std::cout << "[ZMQ BULK_EXPOSE] Test passed!\n";
}

void TestLargeTransfer() {
  std::cout << "\n==== Testing ZMQ Large Transfer (1MB) ====\n";

  std::string addr = "127.0.0.1";
  int port = 8304;

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<ZeroMqTransport>(
      TransportMode::kClient, addr, "tcp", port);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Generate 1MB test data
  size_t large_size = 1024 * 1024;
  std::vector<char> large_data(large_size);
  for (size_t i = 0; i < large_size; ++i) {
    large_data[i] = static_cast<char>('A' + (i % 26));
  }

  LbmMeta<> send_meta;
  send_meta.send.push_back(client->Expose(
      ctp::ipc::FullPtr<char>(large_data.data()), large_size, BULK_XFER));

  int rc = client->Send(send_meta);
  assert(rc == 0);

  LbmMeta<> recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.recv.size() == 1);
  assert(recv_meta.recv[0].size == large_size);

  // Verify data integrity
  int mismatches = 0;
  for (size_t i = 0; i < large_size; ++i) {
    if (recv_meta.recv[0].data.ptr_[i] != large_data[i]) {
      mismatches++;
    }
  }
  assert(mismatches == 0);

  server->ClearRecvHandles(recv_meta);
  std::cout << "[ZMQ Large Transfer] Test passed! (" << large_size << " bytes)\n";
}

void TestGetAddress() {
  std::cout << "\n==== Testing ZMQ GetAddress ====\n";

  std::string addr = "127.0.0.1";
  int port = 8306;

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", port);

  assert(server->GetAddress() == "127.0.0.1");

  std::cout << "[ZMQ GetAddress] Test passed!\n";
}

void TestClearRecvHandles() {
  std::cout << "\n==== Testing ZMQ ClearRecvHandles ====\n";

  std::string addr = "127.0.0.1";
  int port = 8307;

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<ZeroMqTransport>(
      TransportMode::kClient, addr, "tcp", port);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  const char* data = "clear_handles_test";
  size_t size = strlen(data);

  LbmMeta<> send_meta;
  send_meta.send.push_back(client->Expose(
      ctp::ipc::FullPtr<char>(const_cast<char*>(data)), size, BULK_XFER));

  int rc = client->Send(send_meta);
  assert(rc == 0);

  LbmMeta<> recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);

  // After recv, desc should be non-null (zmq_msg_t*)
  assert(recv_meta.recv[0].desc != nullptr);

  // Clear handles
  server->ClearRecvHandles(recv_meta);

  // After clear, desc should be null
  assert(recv_meta.recv[0].desc == nullptr);

  std::cout << "[ZMQ ClearRecvHandles] Test passed!\n";
}

void TestBidirectional() {
  std::cout << "\n==== Testing ZMQ Bidirectional ====\n";

  std::string addr = "127.0.0.1";
  int port = 8308;

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", port);
  auto client = std::make_unique<ZeroMqTransport>(
      TransportMode::kClient, addr, "tcp", port);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Client -> Server
  const char* request_data = "client_request";
  TestMeta send_meta;
  send_meta.request_id = 1;
  send_meta.operation = "request";
  send_meta.send.push_back(client->Expose(
      ctp::ipc::FullPtr<char>(const_cast<char*>(request_data)),
      strlen(request_data), BULK_XFER));

  int rc = client->Send(send_meta);
  assert(rc == 0);

  // Server receives
  TestMeta recv_meta;
  auto info = RecvWithRetry(server.get(), recv_meta);
  assert(info.rc == 0);
  assert(recv_meta.request_id == 1);
  assert(!info.identity_.empty());

  std::string received(recv_meta.recv[0].data.ptr_,
                       recv_meta.recv[0].data.ptr_ + recv_meta.recv[0].size);
  assert(received == request_data);
  server->ClearRecvHandles(recv_meta);

  // Server -> Client (using identity routing)
  const char* response_data = "server_response";
  TestMeta resp_meta;
  resp_meta.request_id = 2;
  resp_meta.operation = "response";
  resp_meta.client_info_.identity_ = info.identity_;
  resp_meta.send.push_back(server->Expose(
      ctp::ipc::FullPtr<char>(const_cast<char*>(response_data)),
      strlen(response_data), BULK_XFER));

  rc = server->Send(resp_meta);
  assert(rc == 0);

  // Client receives
  TestMeta client_recv;
  auto client_info = RecvWithRetry(client.get(), client_recv);
  assert(client_info.rc == 0);
  assert(client_recv.request_id == 2);
  assert(client_recv.operation == "response");

  std::string resp_received(client_recv.recv[0].data.ptr_,
                            client_recv.recv[0].data.ptr_ + client_recv.recv[0].size);
  assert(resp_received == response_data);
  client->ClearRecvHandles(client_recv);

  std::cout << "[ZMQ Bidirectional] Test passed!\n";
}

void TestFactoryTcp() {
  std::cout << "\n==== Testing ZMQ TransportFactory TCP ====\n";

  auto server = TransportFactory::Get(
      "127.0.0.1", TransportType::kZeroMq, TransportMode::kServer, "tcp", 8309);
  auto client = TransportFactory::Get(
      "127.0.0.1", TransportType::kZeroMq, TransportMode::kClient, "tcp", 8309);
  assert(server != nullptr);
  assert(client != nullptr);
  assert(server->IsServer());
  assert(client->IsClient());

  std::cout << "[ZMQ TransportFactory TCP] Test passed!\n";
}

void TestFactoryDefault() {
  std::cout << "\n==== Testing ZMQ TransportFactory Default ====\n";

  // Empty protocol and port=0 should use defaults: "tcp", 8192
  auto server = TransportFactory::Get(
      "127.0.0.1", TransportType::kZeroMq, TransportMode::kServer);
  assert(server != nullptr);
  assert(server->GetAddress() == "127.0.0.1");

  std::cout << "[ZMQ TransportFactory Default] Test passed!\n";
}

void TestFactoryWithDomain() {
  std::cout << "\n==== Testing ZMQ TransportFactory With Domain ====\n";

  auto server = TransportFactory::Get(
      "127.0.0.1", TransportType::kZeroMq, TransportMode::kServer,
      "tcp", 8311, "test_domain");
  assert(server != nullptr);
  assert(server->GetAddress() == "127.0.0.1");

  std::cout << "[ZMQ TransportFactory With Domain] Test passed!\n";
}

void TestIsServerIsClient() {
  std::cout << "\n==== Testing ZMQ IsServer/IsClient ====\n";

  std::string addr = "127.0.0.1";

  auto server = std::make_unique<ZeroMqTransport>(
      TransportMode::kServer, addr, "tcp", 8312);
  auto client = std::make_unique<ZeroMqTransport>(
      TransportMode::kClient, addr, "tcp", 8312);

  assert(server->IsServer());
  assert(!server->IsClient());
  assert(client->IsClient());
  assert(!client->IsServer());

  std::cout << "[ZMQ IsServer/IsClient] Test passed!\n";
}

int main() {
  TestBasicTransfer();
  TestMultipleBulks();
  TestMetadataOnly();
  TestBulkExpose();
  TestLargeTransfer();
  TestGetAddress();
  TestClearRecvHandles();
  TestBidirectional();
  TestFactoryTcp();
  TestFactoryDefault();
  TestFactoryWithDomain();
  TestIsServerIsClient();

  std::cout << "\nAll ZMQ transport full tests passed!" << std::endl;
  return 0;
}
